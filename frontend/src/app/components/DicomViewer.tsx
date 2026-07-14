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
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { Badge } from "@/components/ui/badge";

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
  const [isDraggingSplit, setIsDraggingSplit] = useState(false);
  
  const [dragStart, setDragStart]     = useState({ x: 0, y: 0 });
  const [panStart, setPanStart]       = useState({ x: 0, y: 0 });
  const [windowBase, setWindowBase]   = useState({ brightness: 1.0, contrast: 1.0 });

  const [activePreset, setActivePreset] = useState<Preset>("default");
  const [invert, setInvert]         = useState(false);
  
  const [splitOffset, setSplitOffset] = useState(50); // percentage divider for split mode

  const [zoomModeActive, setZoomModeActive] = useState(false);
  const [windowLevelModeActive, setWindowLevelModeActive] = useState(false);

  const canvasRef   = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const imgRef      = useRef<HTMLImageElement>(null);

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

  useEffect(() => {
    if (observationFocusRegion) {
      setZoom(observationFocusRegion.zoom);
      setPan({ x: observationFocusRegion.panX, y: observationFocusRegion.panY });
    }
  }, [observationFocusRegion]);

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

    if (observationFocusRegion) {
      ctx.strokeStyle = "#22c55e"; 
      ctx.lineWidth = 3;
      ctx.setLineDash([]);
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

  useEffect(() => { redrawCanvas(); }, [ruler, observationFocusRegion, redrawCanvas]);

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
    
    // Check if we are clicking on the split slider handle
    if (viewMode === "split") {
      const container = containerRef.current;
      if (container) {
        const rect = container.getBoundingClientRect();
        const xPos = e.clientX - rect.left;
        const splitPixel = (splitOffset / 100) * rect.width;
        // if within 25 pixels of the handle
        if (Math.abs(xPos - splitPixel) < 25) {
          e.preventDefault();
          setIsDraggingSplit(true);
          return;
        }
      }
    }
    
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

    if (isDraggingSplit && viewMode === "split") {
      const container = containerRef.current;
      if (container) {
        const rect = container.getBoundingClientRect();
        const newOffset = ((e.clientX - rect.left) / rect.width) * 100;
        setSplitOffset(Math.max(0, Math.min(100, newOffset)));
      }
      return;
    }

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
    setIsDraggingSplit(false);
  };

  const handleMeasureToggle = (mode: "off" | "ruler" | "roi") => {
    setMeasureMode(mode);
    setRuler(null);
  };

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
    <div className="space-y-3 flex-1 flex flex-col min-h-0 h-full">

      {/* ── VIEWPORT TOOLBAR ── */}
      <div className="flex flex-col gap-3">
        {/* Row 1: Viewing Modes & Presets */}
        <div className="flex flex-wrap items-center justify-between gap-3 bg-[#141414] border border-[#262626] p-3 rounded-[30px] text-xs shadow-inner">
          {/* Segmented Viewing Modes */}
          <div className="flex items-center gap-3 flex-wrap">
            <span className="text-[10px] font-bold text-[#999999] uppercase tracking-tighter pl-1">View Modes</span>
            <div className="flex bg-[#141414] p-1 rounded-full border border-[#262626] shadow-inner">
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
                    variant="ghost"
                    className={`h-7 px-4 rounded-full text-[11px] capitalize cursor-pointer transition-all duration-300 ${
                      viewMode === mode 
                        ? "bg-[#ffffff] text-[#000000] shadow-md font-bold" 
                        : isDisabled
                        ? "text-[#999999]/30 hover:bg-transparent cursor-not-allowed opacity-40"
                        : "text-[#999999] hover:text-[#ffffff] hover:bg-[#262626] font-medium"
                    }`}
                    onClick={() => !isDisabled && setViewMode?.(mode)}
                  >
                    {label}
                  </Button>
                );
              })}
            </div>

            {/* Contextual Heatmap Opacity Presets */}
            {viewMode === "heatmap" && setHeatmapOpacity && (
              <div className="flex items-center gap-2 pl-3 animate-fadein border-l border-[#262626]">
                <span className="text-[10px] font-bold text-[#999999] uppercase tracking-tighter pl-2">Opacity</span>
                <div className="flex bg-[#141414] p-1 rounded-full border border-[#262626] shadow-inner">
                  {[0.25, 0.5, 0.75, 1.0].map(val => (
                    <button
                      key={val}
                      onClick={() => setHeatmapOpacity(val)}
                      className={`px-3 py-1 text-[10px] rounded-full font-mono transition-all duration-300 ${
                        heatmapOpacity === val 
                          ? 'bg-[#ffffff] text-[#000000] shadow-sm font-bold' 
                          : 'bg-transparent hover:bg-[#262626] text-[#999999] hover:text-[#ffffff]'
                      }`}
                    >
                      {val * 100}%
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Reset All */}
          <Button size="sm" variant="ghost" className="h-8 px-4 rounded-full text-[11px] font-semibold cursor-pointer text-[#999999] hover:text-[#ffffff] hover:bg-[#262626] border border-transparent hover:border-[#262626] transition-all" onClick={resetView}>
            <RotateCcw className="w-3.5 h-3.5 mr-1.5" /> Reset All
          </Button>
        </div>
      </div>

      {/* ── MAIN DIAGNOSTIC VIEWPORT ── */}
      <div
        ref={containerRef}
        onContextMenu={e => e.preventDefault()}
        className="relative border border-[#262626] rounded-[30px] bg-black/60 overflow-hidden flex items-center justify-center flex-1 p-0 shadow-inner min-h-0"
        style={{
          cursor: isDraggingSplit ? "col-resize" : annotateMode ? "default" : measureMode !== "off" ? "crosshair" : isWindowing ? "ns-resize" : isDragging ? "grabbing" : zoomModeActive ? "zoom-in" : windowLevelModeActive ? "ns-resize" : "grab",
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
        {/* Window/Level On-Screen Feedback Indicator */}
        <div 
          className={`absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 z-50 bg-black/70 backdrop-blur-md px-6 py-4 rounded-full border border-[#262626] shadow-2xl transition-opacity duration-300 pointer-events-none flex flex-col items-center gap-2 ${
            isWindowing ? 'opacity-100' : 'opacity-0'
          }`}
        >
          <span className="text-[10px] font-bold text-[#999999] uppercase tracking-tighter">Adjusting Display</span>
          <div className="flex items-center gap-6 text-sm font-mono font-bold text-white">
            <div className="flex flex-col items-center">
              <span className="text-blue-400 mb-1">Bright</span>
              <span>{(brightness * 100).toFixed(0)}%</span>
            </div>
            <div className="w-px h-8 bg-[#333333]"></div>
            <div className="flex flex-col items-center">
              <span className="text-emerald-400 mb-1">Contrast</span>
              <span>{(contrast * 100).toFixed(0)}%</span>
            </div>
          </div>
        </div>

        {/* Viewport indicators */}
        <div className="absolute top-4 left-4 pointer-events-none flex flex-col gap-2 z-30 font-sans">
          <Badge variant="outline" className="bg-[#141414] backdrop-blur-md text-white border-[#262626] text-[10px] px-2.5 py-1 font-sans font-bold shadow-md tracking-tight uppercase">
            Res: {mmPerPx.toFixed(3)} mm/px
          </Badge>
          <Badge variant="outline" className="bg-[#141414] backdrop-blur-md text-white border-[#262626] text-[10px] px-2.5 py-1 font-sans font-bold shadow-md tracking-tight uppercase">
            💡 {windowLevelModeActive ? "Drag: Adjust W/L" : "Drag: Pan / R-Click: W/L"}
          </Badge>
        </div>

        <div className="absolute top-4 right-4 pointer-events-none flex flex-col gap-2 z-30">
          {viewMode === "split" && (
            <Badge variant="outline" className="bg-[#141414] backdrop-blur-md text-yellow-500 border-yellow-500/20 text-[10px] px-2.5 py-1 font-sans font-bold tracking-tighter shadow-sm uppercase animate-pulse">
              SPLIT COMPARISON
            </Badge>
          )}
        </div>

        {/* Split View Slider Handle Overlay (absolute on top of images) */}
        {viewMode === "split" && heatmapSrc && (
          <div 
            className="absolute top-0 bottom-0 z-40 cursor-col-resize flex flex-col items-center justify-center group"
            style={{ left: `${splitOffset}%` }}
          >
            {/* The vertical divider line */}
            <div className={`absolute top-0 bottom-0 w-[2px] bg-white transition-all shadow-[0_0_10px_rgba(255,255,255,0.5)] ${isDraggingSplit ? 'opacity-100 shadow-[0_0_15px_rgba(59,130,246,0.8)] bg-blue-400 w-[3px]' : 'opacity-50 group-hover:opacity-100'}`}></div>
            {/* The pill grabber */}
            <div className={`absolute w-1.5 h-12 bg-white rounded-full shadow-md transition-all ${isDraggingSplit ? 'bg-blue-400 h-16 scale-110' : 'group-hover:scale-110'}`}></div>
          </div>
        )}

        <div style={{
          transform: `scale(${zoom}) translate(${pan.x / zoom}px, ${pan.y / zoom}px)`,
          transition: isDragging || isWindowing || isDraggingSplit ? "none" : "transform 0.1s ease",
          width: "100%",
          height: "100%",
          display: "flex",
          alignItems: "center",
          justifyContent: "center"
        }} className="relative">
          
          {/* Side-by-Side Mode */}
          {viewMode === "side-by-side" ? (
            <div className="flex gap-4 w-full h-full p-8 justify-center">
              <div className="relative h-full flex-1 flex justify-end">
                <div className="relative h-full">
                  <img
                    ref={imgRef}
                    src={src}
                    alt="Original"
                    className="h-full object-contain rounded-full select-none block animate-fadein"
                    style={{ filter: filterStyle + sharpenFilter }}
                    draggable={false}
                    onLoad={() => {
                      adjustZoom('screen');
                    }}
                  />
                  <div className="absolute top-2 left-2 bg-black/60 backdrop-blur px-2 py-1 rounded text-[9px] font-bold text-white border border-[#262626] uppercase tracking-tighter">
                    Original
                  </div>
                </div>
              </div>
              <div className="relative h-full flex-1 flex justify-start">
                <div className="relative h-full">
                  <img
                    src={heatmapSrc || src}
                    alt="Heatmap"
                    className="h-full object-contain rounded-full select-none block animate-fadein"
                    style={{ filter: filterStyle }}
                    draggable={false}
                  />
                  <div className="absolute top-2 left-2 bg-black/60 backdrop-blur px-2 py-1 rounded text-[9px] font-bold text-white border border-[#262626] uppercase tracking-tighter">
                    AI Attributions
                  </div>
                </div>
              </div>
            </div>
          ) : viewMode === "longitudinal" && priorImageSrc ? (
            <div className="flex gap-4 w-full h-full p-8 justify-center">
              {/* Prior Image */}
              <div className="relative h-full flex-1 flex justify-end">
                <div className="relative h-full">
                  <img
                    src={priorImageSrc.startsWith("data:") ? priorImageSrc : `data:image/png;base64,${priorImageSrc}`}
                    alt="Prior Scan"
                    className="h-full object-contain rounded-full select-none block animate-fadein"
                    style={{ filter: filterStyle + sharpenFilter }}
                    draggable={false}
                  />
                  <div className="absolute top-2 left-2 bg-black/60 backdrop-blur px-2 py-1 rounded text-[9px] font-bold text-white border border-[#262626] uppercase tracking-tighter">
                    Prior Scan
                  </div>
                </div>
              </div>
              {/* Current Image with Delta Heatmap */}
              <div className="relative h-full flex-1 flex justify-start">
                <div className="relative h-full">
                  <img
                    src={src}
                    alt="Current Scan"
                    className="h-full object-contain rounded-full select-none block animate-fadein"
                    style={{ filter: filterStyle + sharpenFilter }}
                    draggable={false}
                  />
                  {deltaHeatmapSrc && (
                    <img
                      src={deltaHeatmapSrc.startsWith("data:") ? deltaHeatmapSrc : `data:image/png;base64,${deltaHeatmapSrc}`}
                      alt="Delta Heatmap"
                      className="absolute top-0 left-0 w-full h-full object-contain rounded-full opacity-70 mix-blend-screen select-none pointer-events-none"
                    />
                  )}
                  <div className="absolute top-2 left-2 bg-black/60 backdrop-blur px-2 py-1 rounded text-[9px] font-bold text-amber-400 border border-amber-400/30 uppercase tracking-tighter">
                    Current (Delta Map)
                  </div>
                </div>
              </div>
            </div>
          ) : (
            // Single Viewport (Original, Heatmap, and Split Mode)
            <div className="relative h-full flex justify-center items-center">
              {/* Base Original Chest X-Ray */}
              <img
                ref={imgRef}
                src={src}
                alt="Radiograph"
                className="h-full object-contain rounded-lg select-none block animate-fadein max-h-full max-w-full"
                style={{ filter: filterStyle + sharpenFilter }}
                draggable={false}
                onLoad={() => {
                  adjustZoom('screen');
                  redrawCanvas();
                }}
              />

              {/* Heatmap Overlay Mode */}
              {viewMode === "heatmap" && heatmapSrc && (
                <img
                  src={heatmapSrc}
                  alt="Heatmap Overlay"
                  className="absolute inset-0 object-contain rounded-lg select-none block animate-fadein max-h-full max-w-full"
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
                  className="absolute inset-0 select-none block h-full w-full pointer-events-none"
                  style={{ clipPath: `inset(0 ${100 - splitOffset}% 0 0)` }}
                >
                  <img
                    src={heatmapSrc}
                    alt="Heatmap Split"
                    className="object-contain rounded-lg select-none block h-full w-full max-h-full max-w-full"
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

        {/* Floating capsule toolbar overlay */}
        <div className="absolute bottom-6 left-1/2 -translate-x-1/2 flex flex-wrap justify-center items-center gap-1.5 px-4 py-2.5 bg-[#141414] backdrop-blur-2xl border border-[#262626] rounded-full shadow-2xl z-30 pointer-events-auto max-w-[90vw]">
          <Tooltip>
            <TooltipTrigger render={
              <Button
                size="sm"
                variant={zoomModeActive ? "default" : "ghost"}
                className={`h-9 w-9 p-0 rounded-full cursor-pointer transition-all ${zoomModeActive ? "bg-blue-600 hover:bg-blue-700 text-white shadow-[0_0_15px_rgba(37,99,235,0.5)]" : "text-[#999999] hover:text-[#ffffff] hover:bg-[#262626]"}`}
                onClick={() => {
                  setZoomModeActive(!zoomModeActive);
                  setWindowLevelModeActive(false);
                  setAnnotateMode?.(false);
                  setMeasureMode("off");
                }}
              >
                <ZoomIn className="w-4 h-4" />
              </Button>
            } />
            <TooltipContent className="text-[10px] font-bold uppercase tracking-tighter bg-black/90 border-[#262626]">Zoom Mode</TooltipContent>
          </Tooltip>

          <Tooltip>
            <TooltipTrigger render={
              <Button
                size="sm"
                variant={annotateMode ? "default" : "ghost"}
                className={`h-9 w-9 p-0 rounded-full cursor-pointer transition-all ${annotateMode ? "bg-emerald-600 hover:bg-emerald-700 text-white shadow-[0_0_15px_rgba(5,150,105,0.5)]" : "text-[#999999] hover:text-[#ffffff] hover:bg-[#262626]"}`}
                onClick={() => {
                  setAnnotateMode?.(!annotateMode);
                  setZoomModeActive(false);
                  setWindowLevelModeActive(false);
                  setMeasureMode("off");
                }}
              >
                <GitPullRequest className="w-4 h-4" />
              </Button>
            } />
            <TooltipContent className="text-[10px] font-bold uppercase tracking-tighter bg-black/90 border-[#262626]">Draw Annotations</TooltipContent>
          </Tooltip>

          <Tooltip>
            <TooltipTrigger render={
              <Button
                size="sm"
                variant={measureMode === "ruler" ? "default" : "ghost"}
                className={`h-9 w-9 p-0 rounded-full cursor-pointer transition-all ${measureMode === "ruler" ? "bg-yellow-500 text-black hover:bg-yellow-600 shadow-[0_0_15px_rgba(234,179,8,0.5)]" : "text-[#999999] hover:text-[#ffffff] hover:bg-[#262626]"}`}
                onClick={() => {
                  handleMeasureToggle(measureMode === "ruler" ? "off" : "ruler");
                  setZoomModeActive(false);
                  setWindowLevelModeActive(false);
                  setAnnotateMode?.(false);
                }}
              >
                <RulerIcon className="w-4 h-4" />
              </Button>
            } />
            <TooltipContent className="text-[10px] font-bold uppercase tracking-tighter bg-black/90 border-[#262626]">Calibration Measure</TooltipContent>
          </Tooltip>

          <div className="h-5 w-px bg-[#333333] mx-2" />

          <Tooltip>
            <TooltipTrigger render={
              <Button
                size="sm"
                variant="ghost"
                className="h-9 w-9 p-0 rounded-full cursor-pointer text-[#999999] hover:text-[#ffffff] hover:bg-[#262626] transition-all"
                onClick={() => adjustZoom('screen')}
              >
                <Maximize2 className="w-4 h-4" />
              </Button>
            } />
            <TooltipContent className="text-[10px] font-bold uppercase tracking-tighter bg-black/90 border-[#262626]">Fit to Screen</TooltipContent>
          </Tooltip>
        </div>
      </div>
    </div>
  );
}
