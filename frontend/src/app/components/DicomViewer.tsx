"use client";

import { useState, useRef, useCallback, useEffect, MouseEvent } from "react";
import { 
  RotateCcw, 
  Ruler as RulerIcon, 
  ZoomIn, 
  Maximize2,
  GitPullRequest
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";

// Import custom components
import AnnotationCanvas, { Box, LungZone } from "./AnnotationCanvas";

export interface Ruler {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

interface DicomViewerProps {
  imageBase64: string;
  heatmapBase64?: string;
  hasHeatmap?: boolean;
  label?: string;
  pixelSpacing?: number[] | null;

  viewMode: "original" | "heatmap" | "side-by-side" | "split" | "longitudinal";
  heatmapOpacity: number;
  priorImageSrc?: string;
  deltaHeatmapSrc?: string;

  boxes: Box[];
  setBoxes: React.Dispatch<React.SetStateAction<Box[]>>;
  activeZone: LungZone;
  annotateMode: boolean;
  annotationCanvasRef: React.RefObject<HTMLCanvasElement | null>;
  setViewMode?: (mode: "original" | "heatmap" | "side-by-side" | "split" | "longitudinal") => void;
  setAnnotateMode?: (active: boolean) => void;
  setHeatmapOpacity?: (opacity: number) => void;
  setActiveZone?: (zone: LungZone) => void;

  observationFocusRegion: {
    x1: number;
    y1: number;
    x2: number;
    y2: number;
    zoom: number;
    panX: number;
    panY: number;
  } | null;
}

const PRESETS = {
  default:    { brightness: 1.0, contrast: 1.0 },
  lung:       { brightness: 1.4, contrast: 1.8 },
  bone:       { brightness: 0.8, contrast: 2.5 },
  softTissue: { brightness: 1.15, contrast: 1.35 },
} as const;

type Preset = keyof typeof PRESETS;

export default function DicomViewer({
  imageBase64,
  heatmapBase64,
  label,
  pixelSpacing,
  viewMode,
  heatmapOpacity,
  boxes,
  setBoxes,
  activeZone,
  annotateMode,
  annotationCanvasRef,
  setViewMode,
  setAnnotateMode,
  observationFocusRegion,
  setHeatmapOpacity,
  setActiveZone,
  priorImageSrc,
  deltaHeatmapSrc,
  hasHeatmap = true
}: DicomViewerProps) {
  const [brightness, setBrightness] = useState(1.0);
  const [contrast, setContrast]     = useState(1.0);
  const [zoom, setZoom]             = useState(1.0);
  const [pan, setPan]               = useState({ x: 0, y: 0 });
  const [measureMode, setMeasureMode] = useState<"off" | "ruler" | "roi">("off");
  const [ruler, setRuler]           = useState<Ruler | null>(null);
  const [isDrawing, setIsDrawing]   = useState(false);
  
  const [isDragging, setIsDragging]   = useState(false);
  const [isWindowing, setIsWindowing] = useState(false);
  const [dragStart, setDragStart]     = useState({ x: 0, y: 0 });
  const [panStart, setPanStart]       = useState({ x: 0, y: 0 });
  const [windowBase, setWindowBase]   = useState({ brightness: 1.0, contrast: 1.0 });

  const [activePreset, setActivePreset] = useState<Preset>("default");
  const [invert, setInvert]         = useState(false);
  
  const [splitOffset, setSplitOffset] = useState(50); // percentage divider for split mode
  const [histogramData, setHistogramData] = useState<number[]>([]);

  const [zoomModeActive, setZoomModeActive] = useState(false);
  const [windowLevelModeActive, setWindowLevelModeActive] = useState(false);

  const canvasRef   = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const imgRef      = useRef<HTMLImageElement>(null);
  const histCanvasRef = useRef<HTMLCanvasElement>(null);

  const mmPerPx = pixelSpacing && pixelSpacing.length > 0 ? pixelSpacing[0] : 0.25;

  const applyPreset = (p: Preset) => {
    setBrightness(PRESETS[p].brightness);
    setContrast(PRESETS[p].contrast);
    setActivePreset(p);
  };

  const resetView = () => {
    setZoom(1.0); setPan({ x: 0, y: 0 }); setRuler(null);
    applyPreset("default"); setMeasureMode("off");
    setInvert(false);
    setZoomModeActive(false); setWindowLevelModeActive(false);
  };

  const adjustZoom = useCallback((mode: 'width' | 'height' | 'screen' | '1to1' | 'reset') => {
    const container = containerRef.current;
    const img = imgRef.current;
    if (!container || !img) return;

    const cW = container.clientWidth;
    const cH = container.clientHeight;
    const iW = img.naturalWidth || img.offsetWidth || 500;
    const iH = img.naturalHeight || img.offsetHeight || 500;

    let targetZoom = 1.0;
    if (mode === 'width') {
      targetZoom = (cW / iW) * 0.85;
    } else if (mode === 'height') {
      targetZoom = (cH / iH) * 0.85;
    } else if (mode === 'screen') {
      targetZoom = Math.min(cW / iW, cH / iH) * 0.85;
    } else if (mode === '1to1') {
      targetZoom = 1.0;
    } else if (mode === 'reset') {
      targetZoom = 1.0;
      setPan({ x: 0, y: 0 });
      return;
    }
    
    setZoom(targetZoom);
    setPan({ x: 0, y: 0 });
  }, []);

  // Sync zoom/pan when user clicks on a clinical observation hotspot
  useEffect(() => {
    if (observationFocusRegion) {
      setZoom(observationFocusRegion.zoom);
      setPan({ x: observationFocusRegion.panX, y: observationFocusRegion.panY });
    }
  }, [observationFocusRegion]);

  // Reset measure mode if annotation mode is activated
  useEffect(() => {
    if (annotateMode) {
      setMeasureMode("off");
    }
  }, [annotateMode]);

  const redrawCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    const img    = imgRef.current;
    if (!canvas || !img) return;
    canvas.width  = img.offsetWidth;
    canvas.height = img.offsetHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const w = canvas.width;
    const h = canvas.height;

    // 1. Draw active observation flashing indicator
    if (observationFocusRegion) {
      ctx.strokeStyle = "#22c55e"; // green-500
      ctx.lineWidth = 3;
      ctx.setLineDash([]);
      // Map percentage coords to offset size
      const fx1 = (observationFocusRegion.x1 / 224) * w;
      const fy1 = (observationFocusRegion.y1 / 224) * h;
      const fx2 = (observationFocusRegion.x2 / 224) * w;
      const fy2 = (observationFocusRegion.y2 / 224) * h;

      ctx.strokeRect(fx1, fy1, fx2 - fx1, fy2 - fy1);
      ctx.fillStyle = "rgba(34, 197, 94, 0.15)";
      ctx.fillRect(fx1, fy1, fx2 - fx1, fy2 - fy1);

      ctx.font = "bold 9px Inter, sans-serif";
      ctx.fillStyle = "#22c55e";
      ctx.fillText("ATTENTION AREA FOCUS", fx1 + 4, fy1 - 5);
    }

    // 3. Draw measurement graphics
    if (ruler) {
      if (measureMode === "ruler") {
        const dx = ruler.x2 - ruler.x1, dy = ruler.y2 - ruler.y1;
        const dist = Math.sqrt(dx * dx + dy * dy);
        const mm   = (dist * mmPerPx).toFixed(1);
        const angle = Math.atan2(dy, dx);
        const capLen = 8;

        ctx.beginPath(); ctx.moveTo(ruler.x1, ruler.y1); ctx.lineTo(ruler.x2, ruler.y2);
        ctx.strokeStyle = "#facc15"; ctx.lineWidth = 2; ctx.setLineDash([]); ctx.stroke();

        [[ruler.x1, ruler.y1], [ruler.x2, ruler.y2]].forEach(([cx, cy]) => {
          ctx.beginPath();
          ctx.moveTo(cx - capLen * Math.sin(angle), cy + capLen * Math.cos(angle));
          ctx.lineTo(cx + capLen * Math.sin(angle), cy - capLen * Math.cos(angle));
          ctx.strokeStyle = "#facc15"; ctx.lineWidth = 2; ctx.stroke();
        });

        const midX = (ruler.x1 + ruler.x2) / 2, midY = (ruler.y1 + ruler.y2) / 2 - 10;
        ctx.font = "bold 12px Inter, sans-serif";
        ctx.fillStyle = "#facc15"; ctx.textAlign = "center";
        ctx.fillText(`${mm} mm`, midX, midY);
      } else if (measureMode === "roi") {
        ctx.strokeStyle = "#94a3b860"; ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
        ctx.strokeRect(ruler.x1, ruler.y1, ruler.x2 - ruler.x1, ruler.y2 - ruler.y1);

        const cx = (ruler.x1 + ruler.x2) / 2;
        const cy = (ruler.y1 + ruler.y2) / 2;
        const rx = Math.abs(ruler.x2 - ruler.x1) / 2;
        const ry = Math.abs(ruler.y2 - ruler.y1) / 2;

        if (rx > 2 && ry > 2) {
          ctx.beginPath();
          ctx.ellipse(cx, cy, rx, ry, 0, 0, 2 * Math.PI);
          ctx.strokeStyle = "#facc15"; ctx.lineWidth = 2.5; ctx.setLineDash([]);
          ctx.stroke();
          ctx.fillStyle = "#facc151a"; ctx.fill();

          const wMM = (rx * 2 * mmPerPx).toFixed(1);
          const hMM = (ry * 2 * mmPerPx).toFixed(1);
          const area = (Math.PI * rx * mmPerPx * ry * mmPerPx).toFixed(1);

          ctx.font = "bold 11px Inter, sans-serif";
          ctx.fillStyle = "#facc15"; ctx.textAlign = "center";
          ctx.fillText(`W: ${wMM}mm, H: ${hMM}mm`, cx, cy - 6);
          ctx.fillText(`Area: ${area} mm²`, cx, cy + 8);
        }
      }
    }
  }, [ruler, measureMode, mmPerPx, observationFocusRegion]);

  const generateHistogram = useCallback(() => {
    const img = imgRef.current;
    if (!img) return;
    const canvas = document.createElement("canvas");
    canvas.width = 128;
    canvas.height = 128;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    try {
      ctx.drawImage(img, 0, 0, 128, 128);
      const imgData = ctx.getImageData(0, 0, 128, 128);
      const data = imgData.data;
      const hist = new Array(256).fill(0);
      for (let i = 0; i < data.length; i += 4) {
        const val = Math.round((data[i] + data[i + 1] + data[i + 2]) / 3);
        hist[val]++;
      }
      setHistogramData(hist);
    } catch (e) {
      console.warn("Histogram generation failed", e);
    }
  }, []);

  const drawHistogram = useCallback(() => {
    const canvas = histCanvasRef.current;
    if (!canvas || histogramData.length === 0) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const cw = canvas.width;
    const ch = canvas.height;
    ctx.clearRect(0, 0, cw, ch);

    const grad = ctx.createLinearGradient(0, 0, cw, 0);
    grad.addColorStop(0, "#0f172a");
    grad.addColorStop(1, "#1e293b");
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, cw, ch);

    ctx.strokeStyle = "#33415550"; ctx.lineWidth = 1;
    for (let x = cw / 4; x < cw; x += cw / 4) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, ch); ctx.stroke();
    }

    const adjustedHist = new Array(cw).fill(0);
    for (let i = 0; i < 256; i++) {
      let intensity = i / 255;
      intensity = (intensity - 0.5) * contrast + 0.5 + (brightness - 1.0);
      intensity = Math.max(0, Math.min(1, intensity));

      const col = Math.round(intensity * (cw - 1));
      adjustedHist[col] += histogramData[i];
    }

    const maxVal = Math.max(...adjustedHist) || 1;
    ctx.beginPath();
    ctx.moveTo(0, ch);
    for (let x = 0; x < cw; x++) {
      const h = (adjustedHist[x] / maxVal) * (ch - 6);
      ctx.lineTo(x, ch - h);
    }
    ctx.lineTo(cw, ch);
    ctx.fillStyle = "rgba(59, 130, 246, 0.25)";
    ctx.fill();

    ctx.strokeStyle = "#3b82f6"; ctx.lineWidth = 1.5;
    ctx.stroke();
  }, [histogramData, brightness, contrast]);

  useEffect(() => { redrawCanvas(); }, [ruler, observationFocusRegion, redrawCanvas]);
  useEffect(() => { drawHistogram(); }, [histogramData, brightness, contrast, drawHistogram]);

  const getCanvasCoords = (e: MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };
    const rect = canvas.getBoundingClientRect();
    return {
      x: rect.width > 0 ? (e.clientX - rect.left) * (canvas.width / rect.width) : 0,
      y: rect.height > 0 ? (e.clientY - rect.top) * (canvas.height / rect.height) : 0
    };
  };

  const handleMouseDown = (e: MouseEvent<HTMLDivElement>) => {
    if (measureMode !== "off" || annotateMode) return;
    
    if (e.button === 2 || e.shiftKey) {
      e.preventDefault();
      setIsWindowing(true);
      setDragStart({ x: e.clientX, y: e.clientY });
      setWindowBase({ brightness, contrast });
    } else {
      setIsDragging(true);
      setDragStart({ x: e.clientX, y: e.clientY });
      setPanStart({ ...pan });
    }
  };

  const handleMouseMove = (e: MouseEvent<HTMLDivElement>) => {
    if (measureMode !== "off" || annotateMode) return;

    if (isWindowing) {
      const dx = e.clientX - dragStart.x;
      const dy = e.clientY - dragStart.y;
      const nextBrightness = Math.max(0.1, Math.min(3.0, windowBase.brightness + dx * 0.005));
      const nextContrast = Math.max(0.1, Math.min(4.0, windowBase.contrast - dy * 0.005));
      setBrightness(nextBrightness);
      setContrast(nextContrast);
      setActivePreset("default");
    } else if (isDragging) {
      setPan({ x: panStart.x + e.clientX - dragStart.x, y: panStart.y + e.clientY - dragStart.y });
    }
  };

  const handleMouseUpOrLeave = () => {
    setIsDragging(false);
    setIsWindowing(false);
  };

  const handleMeasureToggle = (mode: "off" | "ruler" | "roi") => {
    setMeasureMode(mode);
    setRuler(null);
  };

  // Accept three encodings so locally created blob: previews, backend data:
  // URLs, and raw base64 strings all render correctly. Without the blob: check,
  // a blob URL becomes "data:image/png;base64,blob:http://..." and silently
  // fails to load, leaving the diagnostic viewport blank.
  const toImageSrc = (raw: string | undefined): string | undefined => {
    if (!raw) return undefined;
    if (raw.startsWith("data:") || raw.startsWith("blob:") || raw.startsWith("http")) {
      return raw;
    }
    return `data:image/png;base64,${raw}`;
  };

  const src = toImageSrc(imageBase64);
  const heatmapSrc = toImageSrc(heatmapBase64);

  const filterStyle = `brightness(${brightness}) contrast(${contrast}) ${invert ? "invert(1)" : ""}`;
  const sharpenFilter = "";

  return (
    <div className="space-y-3 flex-1 flex flex-col">

      {/* ── VIEWPORT TOOLBAR ── */}
      <div className="flex flex-col gap-3">
        {/* Row 1: Viewing Modes & Presets */}
        <div className="flex flex-wrap items-center justify-between gap-3 glass-panel p-3 rounded-xl text-xs">
          {/* Segmented Viewing Modes */}
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-[11px] font-bold text-muted-foreground uppercase tracking-wider mr-1">View Modes:</span>
            <div className="flex bg-black/20 dark:bg-black/40 p-0.5 rounded-full border border-white/5 backdrop-blur-md">
              {(["original", "heatmap", "side-by-side", "split", "longitudinal"] as const).map(mode => {
                const requiresHeatmap = ["heatmap", "side-by-side", "split", "longitudinal"].includes(mode);
                const isDisabled = requiresHeatmap && !hasHeatmap;
                
                let label = mode.replace("-", " ");
                if (mode === "original") label = "Radiograph";
                if (mode === "heatmap") label = "AI Overlay";
                if (mode === "split") label = "Split View";
                
                return (
                  <Button
                    key={mode}
                    size="sm"
                    disabled={isDisabled}
                    variant={viewMode === mode ? "default" : "ghost"}
                    className={`h-8 px-3.5 rounded-full text-xs capitalize cursor-pointer transition-all duration-300 ${
                      viewMode === mode 
                        ? "bg-primary text-primary-foreground shadow-lg shadow-primary/20 font-semibold" 
                        : isDisabled
                        ? "text-muted-foreground/30 hover:bg-transparent cursor-not-allowed opacity-40"
                        : "text-muted-foreground hover:text-foreground hover:bg-white/5"
                    }`}
                    onClick={() => !isDisabled && setViewMode?.(mode)}
                  >
                    {label}
                  </Button>
                );
              })}
            </div>

            {/* Contextual Heatmap Opacity Slider */}
            {viewMode === "heatmap" && setHeatmapOpacity && (
              <div className="flex items-center gap-2 pl-3 border-l border-white/10 animate-fadein">
                <span className="text-[10px] font-bold text-muted-foreground uppercase">Opacity:</span>
                <input
                  type="range"
                  min="0.1"
                  max="1.0"
                  step="0.05"
                  value={heatmapOpacity}
                  onChange={(e) => setHeatmapOpacity(parseFloat(e.target.value))}
                  className="w-20 h-1 bg-white/20 rounded-full appearance-none [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-2.5 [&::-webkit-slider-thumb]:h-2.5 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-[#5865F2] cursor-pointer"
                />
                <span className="text-[10px] font-mono text-muted-foreground">{Math.round(heatmapOpacity * 100)}%</span>
              </div>
            )}
          </div>

          {/* Reset All */}
          <Button size="sm" variant="ghost" className="h-8 px-3.5 rounded-full text-xs cursor-pointer text-muted-foreground hover:text-foreground hover:bg-white/5" onClick={resetView}>
            <RotateCcw className="w-3.5 h-3.5 mr-1" /> Reset All
          </Button>
        </div>
      </div>

      {/* Split screen partition position slider */}
      {viewMode === "split" && (
        <div className="flex items-center gap-3 border border-yellow-500/20 bg-yellow-500/5 px-3 py-1.5 rounded-lg text-[10px] font-semibold text-yellow-600 dark:text-yellow-400">
          <span>Split Screen partition position:</span>
          <input
            type="range"
            min="0"
            max="100"
            value={splitOffset}
            onChange={e => setSplitOffset(parseInt(e.target.value))}
            className="flex-1 h-1 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-yellow-500"
          />
          <span className="font-mono">{splitOffset}%</span>
        </div>
      )}

      {/* ── MAIN DIAGNOSTIC VIEWPORT ── */}
      <div
        ref={containerRef}
        onContextMenu={e => e.preventDefault()}
        className="relative border border-border/50 rounded-xl bg-black overflow-hidden flex items-center justify-center flex-1 p-2"
        style={{
          height: 600,
          cursor: annotateMode ? "default" : measureMode !== "off" ? "crosshair" : isWindowing ? "ns-resize" : isDragging ? "grabbing" : zoomModeActive ? "zoom-in" : windowLevelModeActive ? "ns-resize" : "grab",
        }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUpOrLeave}
        onMouseLeave={handleMouseUpOrLeave}
        onWheel={e => {
          if (e.ctrlKey || zoomModeActive) {
            e.preventDefault();
            setZoom(prev => Math.max(0.5, Math.min(4, prev - e.deltaY * 0.001)));
          }
        }}
      >
        <div style={{
          transform: `scale(${zoom}) translate(${pan.x / zoom}px, ${pan.y / zoom}px)`,
          transition: isDragging || isWindowing ? "none" : "transform 0.1s ease",
        }} className="relative w-fit mx-auto">
          
          {/* Side-by-Side Mode */}
          {viewMode === "side-by-side" ? (
            <div className="flex gap-4">
              <div className="relative">
                <img
                  ref={imgRef}
                  src={src}
                  alt="Original"
                  className="h-full object-contain rounded select-none block animate-fadein"
                  style={{ filter: filterStyle + sharpenFilter }}
                  draggable={false}
                  onLoad={() => {
                    adjustZoom('screen');
                    generateHistogram();
                  }}
                />
                <div className="absolute top-2 left-2 bg-black/60 px-1.5 py-0.5 rounded text-[8px] font-bold text-white border border-white/10 uppercase tracking-widest">
                  Original
                </div>
              </div>
              <div className="relative">
                <img
                  src={heatmapSrc || src}
                  alt="Heatmap"
                  className="h-full object-contain rounded select-none block animate-fadein"
                  style={{ filter: filterStyle }}
                  draggable={false}
                />
                <div className="absolute top-2 left-2 bg-black/60 px-1.5 py-0.5 rounded text-[8px] font-bold text-white border border-white/10 uppercase tracking-widest">
                  AI Attributions
                </div>
              </div>
            </div>
          ) : viewMode === "longitudinal" && priorImageSrc ? (
            <div className="flex gap-4">
              {/* Prior Image */}
              <div className="relative">
                <img
                  src={priorImageSrc.startsWith("data:") ? priorImageSrc : `data:image/png;base64,${priorImageSrc}`}
                  alt="Prior Scan"
                  className="h-full object-contain rounded select-none block animate-fadein"
                  style={{ filter: filterStyle + sharpenFilter }}
                  draggable={false}
                />
                <div className="absolute top-2 left-2 bg-black/60 px-1.5 py-0.5 rounded text-[8px] font-bold text-white border border-white/10 uppercase tracking-widest">
                  Prior Scan
                </div>
              </div>
              {/* Current Image with Delta Heatmap */}
              <div className="relative">
                <img
                  src={src}
                  alt="Current Scan"
                  className="h-full object-contain rounded select-none block animate-fadein"
                  style={{ filter: filterStyle + sharpenFilter }}
                  draggable={false}
                />
                {deltaHeatmapSrc && (
                  <img
                    src={deltaHeatmapSrc.startsWith("data:") ? deltaHeatmapSrc : `data:image/png;base64,${deltaHeatmapSrc}`}
                    alt="Delta Heatmap"
                    className="absolute top-0 left-0 w-full h-full object-contain rounded opacity-70 mix-blend-screen select-none pointer-events-none"
                  />
                )}
                <div className="absolute top-2 left-2 bg-black/60 px-1.5 py-0.5 rounded text-[8px] font-bold text-amber-400 border border-amber-400/30 uppercase tracking-widest">
                  Current (Delta Map)
                </div>
              </div>
            </div>
          ) : (
            // Single Viewport (Original, Heatmap, and Split Mode)
            <div className="relative">
              {/* Base Original Chest X-Ray */}
              <img
                ref={imgRef}
                src={src}
                alt="Radiograph"
                className="h-full object-contain rounded select-none block animate-fadein"
                style={{ filter: filterStyle + sharpenFilter }}
                draggable={false}
                onLoad={() => {
                  adjustZoom('screen');
                  redrawCanvas();
                  generateHistogram();
                }}
              />

              {/* Heatmap Overlay Mode */}
              {viewMode === "heatmap" && heatmapSrc && (
                <img
                  src={heatmapSrc}
                  alt="Heatmap Overlay"
                  className="h-full object-contain rounded select-none block absolute inset-0 w-full h-full animate-fadein"
                  style={{ 
                    filter: filterStyle,
                    opacity: heatmapOpacity 
                  }}
                  draggable={false}
                />
              )}

              {/* Split Screen Slider Mode */}
              {viewMode === "split" && heatmapSrc && (
                <div 
                  className="absolute inset-0 select-none block"
                  style={{ clipPath: `inset(0 ${100 - splitOffset}% 0 0)` }}
                >
                  <img
                    src={heatmapSrc}
                    alt="Heatmap Split"
                    className="h-full object-contain rounded select-none block w-full h-full"
                    style={{ filter: filterStyle }}
                    draggable={false}
                  />
                </div>
              )}

              {/* Headless vector drawing annotations overlay canvas */}
              <AnnotationCanvas
                boxes={boxes}
                setBoxes={setBoxes}
                activeZone={activeZone}
                annotateMode={annotateMode}
                imgRef={imgRef}
                canvasRef={annotationCanvasRef}
              />

              {/* Measurement and segmentations canvas layer */}
              <canvas
                ref={canvasRef}
                className="absolute inset-0"
                style={{ 
                  pointerEvents: measureMode !== "off" ? "all" : "none", 
                  cursor: measureMode !== "off" ? "crosshair" : "default",
                  zIndex: 21
                }}
                onMouseDown={e => { if (measureMode === "off") return; const p = getCanvasCoords(e); setIsDrawing(true); setRuler({ x1: p.x, y1: p.y, x2: p.x, y2: p.y }); }}
                onMouseMove={e => { if (measureMode === "off" || !isDrawing) return; const p = getCanvasCoords(e); setRuler(prev => prev ? { ...prev, x2: p.x, y2: p.y } : null); }}
                onMouseUp={() => setIsDrawing(false)}
              />
            </div>
          )}
        </div>

        {/* Viewport indicators */}
        <div className="absolute top-3 left-3 pointer-events-none flex flex-col gap-1.5 z-30 font-sans">
          <Badge variant="outline" className="bg-black/80 text-white border-white/10 text-[9px] px-2 py-0.5 font-sans font-bold shadow-md tracking-wide">
            Resolution: {mmPerPx.toFixed(3)} mm/px
          </Badge>
          <Badge variant="outline" className="bg-black/80 text-white border-white/10 text-[9px] px-2 py-0.5 font-sans font-bold shadow-md tracking-wide">
            💡 Drag: {windowLevelModeActive ? "Adjust W/L" : "Pan View"}
          </Badge>
        </div>

        <div className="absolute top-3 right-3 pointer-events-none flex flex-col gap-1 z-30">
          {viewMode === "split" && (
            <Badge variant="outline" className="bg-black/75 text-yellow-500 border-yellow-500/20 text-[9px] px-1.5 py-0.5 font-mono shadow-sm animate-pulse">
              SPLIT COMPARISON
            </Badge>
          )}
        </div>

        {/* Floating capsule toolbar overlay */}
        <div className="absolute bottom-4 left-1/2 -translate-x-1/2 flex items-center gap-1.5 px-3.5 py-2 bg-black/85 backdrop-blur-md border border-border/65 rounded-full shadow-lg z-30 pointer-events-auto">
          <Tooltip>
            <TooltipTrigger render={
              <Button
                size="sm"
                variant={zoomModeActive ? "default" : "ghost"}
                className={`h-8.5 w-8.5 p-0 rounded-full cursor-pointer transition-all ${zoomModeActive ? "bg-blue-600 hover:bg-blue-700 text-white" : "text-muted-foreground hover:text-foreground"}`}
                onClick={() => {
                  setZoomModeActive(!zoomModeActive);
                  setWindowLevelModeActive(false);
                  setAnnotateMode?.(false);
                  setMeasureMode("off");
                }}
              >
                <ZoomIn className="w-4 h-4" />
              </Button>
            }/>
            <TooltipContent>Zoom Mode</TooltipContent>
          </Tooltip>

          <Tooltip>
            <TooltipTrigger render={
              <Button
                size="sm"
                variant={annotateMode ? "default" : "ghost"}
                className={`h-8.5 w-8.5 p-0 rounded-full cursor-pointer transition-all ${annotateMode ? "bg-emerald-600 hover:bg-emerald-700 text-white" : "text-muted-foreground hover:text-foreground"}`}
                onClick={() => {
                  setAnnotateMode?.(!annotateMode);
                  setZoomModeActive(false);
                  setWindowLevelModeActive(false);
                  setMeasureMode("off");
                }}
              >
                <GitPullRequest className="w-4 h-4" />
              </Button>
            }/>
            <TooltipContent>Draw Annotations</TooltipContent>
          </Tooltip>

          <Tooltip>
            <TooltipTrigger render={
              <Button
                size="sm"
                variant={measureMode === "ruler" ? "default" : "ghost"}
                className={`h-8.5 w-8.5 p-0 rounded-full cursor-pointer transition-all ${measureMode === "ruler" ? "bg-yellow-500 text-black hover:bg-yellow-600" : "text-muted-foreground hover:text-foreground"}`}
                onClick={() => {
                  handleMeasureToggle(measureMode === "ruler" ? "off" : "ruler");
                  setZoomModeActive(false);
                  setWindowLevelModeActive(false);
                  setAnnotateMode?.(false);
                }}
              >
                <RulerIcon className="w-4 h-4" />
              </Button>
            }/>
            <TooltipContent>Calibration Measure</TooltipContent>
          </Tooltip>

          <div className="h-4 w-px bg-border/60 mx-1" />

          <Tooltip>
            <TooltipTrigger render={
              <Button
                size="sm"
                variant="ghost"
                className="h-8.5 w-8.5 p-0 rounded-full cursor-pointer text-muted-foreground hover:text-foreground"
                onClick={() => adjustZoom('screen')}
              >
                <Maximize2 className="w-4 h-4" />
              </Button>
            }/>
            <TooltipContent>Fit to Screen</TooltipContent>
          </Tooltip>
        </div>
      </div>
    </div>
  );
}
