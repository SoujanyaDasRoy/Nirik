"use client";

import { useState, useRef, useEffect } from "react";
import { Badge } from "@/components/ui/badge";
import { Slider } from "@/components/ui/slider";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { AnalysisResult } from "../hooks/useFileUpload";

interface XaiVisualizationProps {
  result: AnalysisResult;
  similarCases: { tb_similar: any[]; normal_similar: any[] } | null;
  loadingSimilar: boolean;
}

export default function XaiVisualization({ result, similarCases, loadingSimilar }: XaiVisualizationProps) {
  const [opacity, setOpacity] = useState<number>(55);
  const [showHeatmap, setShowHeatmap] = useState<boolean>(true);
  const [showBbox, setShowBbox] = useState<boolean>(true);
  const [showCircle, setShowCircle] = useState<boolean>(false);
  const [showContour, setShowContour] = useState<boolean>(true);
  const [heatmapMode, setHeatmapMode] = useState<"gradcam" | "gradcam_plusplus" | "attention" | "coverage" | "attribution">("gradcam_plusplus");
  const [hoveredRoiId, setHoveredRoiId] = useState<string | null>(null);
  const [zoomLevel, setZoomLevel] = useState<number>(1);
  const [isComparing, setIsComparing] = useState<boolean>(false);

  const imgRef = useRef<HTMLImageElement>(null);
  const [naturalSize, setNaturalSize] = useState({ w: 1, h: 1 });
  const [displaySize, setDisplaySize] = useState({ w: 1, h: 1 });

  const xai = result.xai_results;
  const rois = xai?.rois || [];
  const metrics = xai?.metrics;
  const summary = xai?.summary || "No explainability telemetry available for this study.";

  const handleImageLoad = () => {
    const img = imgRef.current;
    if (!img) return;
    setNaturalSize({ w: img.naturalWidth || 1, h: img.naturalHeight || 1 });
    setDisplaySize({ w: img.clientWidth || 1, h: img.clientHeight || 1 });
  };

  useEffect(() => {
    const handleResize = () => {
      const img = imgRef.current;
      if (!img) return;
      setDisplaySize({ w: img.clientWidth || 1, h: img.clientHeight || 1 });
    };
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  // Update size on change of heatmap mode or compare toggle
  useEffect(() => {
    setTimeout(handleImageLoad, 200);
  }, [heatmapMode, isComparing]);

  const scaleCoords = (coords: [number, number]) => {
    const [x, y] = coords;
    const dx = (x / naturalSize.w) * displaySize.w;
    const dy = (y / naturalSize.h) * displaySize.h;
    return [dx, dy];
  };

  const scaleBbox = (bbox: [number, number, number, number]) => {
    const [x, y, w, h] = bbox;
    const dx = (x / naturalSize.w) * displaySize.w;
    const dy = (y / naturalSize.h) * displaySize.h;
    const dw = (w / naturalSize.w) * displaySize.w;
    const dh = (h / naturalSize.h) * displaySize.h;
    return [dx, dy, dw, dh];
  };

  const scaleCircle = (circle: [number, number, number]) => {
    const [cx, cy, r] = circle;
    const dcx = (cx / naturalSize.w) * displaySize.w;
    const dcy = (cy / naturalSize.h) * displaySize.h;
    const dr = (r / Math.max(naturalSize.w, naturalSize.h)) * Math.max(displaySize.w, displaySize.h);
    return [dcx, dcy, dr];
  };

  // Get base64 heatmap image source based on selected mode
  const getHeatmapSrc = () => {
    if (!result.heatmaps) return result.heatmap_image;
    return result.heatmaps[heatmapMode] || result.heatmap_image;
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden p-6 text-slate-100 shadow-2xl space-y-6">
      {/* Header Bar */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center border-b border-slate-800 pb-4 gap-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-teal-500 animate-pulse"></span>
            <h2 className="text-lg font-bold tracking-tight text-teal-400">CLINICAL EXPLANATION WORKSPACE</h2>
          </div>
          <p className="text-xs text-slate-400 mt-0.5">
            Diagnostic Case Support System  ·  Study ID: <span className="font-semibold text-slate-200">{result.study_id || "N/A"}</span>
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button 
            variant="outline" 
            size="sm" 
            className="border-slate-700 text-slate-300 hover:bg-slate-800 text-xs"
            onClick={() => setIsComparing(!isComparing)}
          >
            {isComparing ? "Close Side-by-Side" : "Side-by-Side Compare"}
          </Button>
          <Button 
            variant="outline" 
            size="sm" 
            className="border-slate-700 text-slate-300 hover:bg-slate-800 text-xs"
            onClick={() => setZoomLevel(zoomLevel === 1.3 ? 1 : 1.3)}
          >
            🔍 Zoom {zoomLevel === 1.3 ? "100%" : "130%"}
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-12 gap-6">
        {/* Left Side: Image Visualizer (6 cols) */}
        <div className={`xl:col-span-7 ${isComparing ? "xl:col-span-8" : "xl:col-span-7"} space-y-4`}>
          <div className={`grid ${isComparing ? "grid-cols-1 md:grid-cols-2" : "grid-cols-1"} gap-4`}>
            {/* Main Visualizer Container */}
            <div className="relative border border-slate-800 bg-slate-950 rounded-lg overflow-hidden flex items-center justify-center min-h-[400px]">
              <div 
                className="relative transition-transform duration-200" 
                style={{ transform: `scale(${zoomLevel})` }}
              >
                {/* 1. Original Radiograph */}
                <img
                  ref={imgRef}
                  src={result.original_image}
                  alt="Original Chest X-Ray"
                  onLoad={handleImageLoad}
                  className="max-h-[500px] w-auto block object-contain"
                />

                {/* 2. Heatmap Overlay (layered) */}
                {showHeatmap && (
                  <img
                    src={getHeatmapSrc()}
                    alt="Grad-CAM Saliency Overlay"
                    className="absolute inset-0 w-full h-full object-contain pointer-events-none transition-opacity duration-150"
                    style={{ opacity: opacity / 100 }}
                  />
                )}

                {/* 3. SVG Interactive Annotation Layer */}
                <svg
                  className="absolute inset-0 w-full h-full pointer-events-none"
                  style={{ width: displaySize.w, height: displaySize.h }}
                >
                  {rois.map((roi: any) => {
                    const [bx, by, bw, bh] = scaleBbox(roi.bbox);
                    const [cx, cy, cr] = scaleCircle(roi.circle);
                    const isHovered = hoveredRoiId === roi.id;

                    return (
                      <g key={roi.id} className="pointer-events-auto cursor-pointer"
                         onMouseEnter={() => setHoveredRoiId(roi.id)}
                         onMouseLeave={() => setHoveredRoiId(null)}
                      >
                        {/* Contour Outline */}
                        {showContour && roi.contour && roi.contour.length > 0 && (
                          <polygon
                            points={roi.contour.map((pt: any) => scaleCoords(pt).join(",")).join(" ")}
                            fill="transparent"
                            stroke={isHovered ? "#ef4444" : "#eab308"}
                            strokeWidth={isHovered ? 2.5 : 1.5}
                            className="transition-all duration-100"
                          />
                        )}

                        {/* Circular Marker */}
                        {showCircle && roi.circle && (
                          <circle
                            cx={cx}
                            cy={cy}
                            r={cr}
                            fill="transparent"
                            stroke={isHovered ? "#f97316" : "#3b82f6"}
                            strokeWidth={1.5}
                            strokeDasharray="4,2"
                          />
                        )}

                        {/* Bounding Box */}
                        {showBbox && roi.bbox && (
                          <rect
                            x={bx}
                            y={by}
                            width={bw}
                            height={bh}
                            fill={isHovered ? "rgba(239, 68, 68, 0.08)" : "transparent"}
                            stroke={isHovered ? "#ef4444" : "rgba(13, 148, 136, 0.8)"}
                            strokeWidth={isHovered ? 2 : 1}
                            className="transition-all duration-100"
                          />
                        )}

                        {/* Text Label Tag */}
                        <g transform={`translate(${bx + 2}, ${by + 12})`}>
                          <rect
                            width={16}
                            height={12}
                            rx={2}
                            fill={isHovered ? "#ef4444" : "#0d9488"}
                          />
                          <text
                            x={8}
                            y={9}
                            fontSize="8"
                            fontWeight="bold"
                            fill="#ffffff"
                            textAnchor="middle"
                          >
                            {roi.id}
                          </text>
                        </g>
                      </g>
                    );
                  })}
                </svg>
              </div>

              {/* Top Layer Annotations Toggle Labels */}
              <div className="absolute top-2 left-2 flex gap-1.5 flex-wrap pointer-events-none z-10">
                <Badge className="bg-slate-900/80 border-slate-700 text-teal-400 text-[10px]">
                  Mode: {heatmapMode.toUpperCase().replace("_", "+")}
                </Badge>
                {hoveredRoiId && (
                  <Badge className="bg-red-500 border-none text-white text-[10px] animate-pulse">
                    Hovering Region {hoveredRoiId}
                  </Badge>
                )}
              </div>
            </div>

            {/* Side-by-Side Comparison Window */}
            {isComparing && (
              <div className="relative border border-slate-800 bg-slate-950 rounded-lg overflow-hidden flex items-center justify-center min-h-[400px]">
                <div 
                  className="relative transition-transform duration-200"
                  style={{ transform: `scale(${zoomLevel})` }}
                >
                  <img
                    src={result.original_image}
                    alt="Original Reference"
                    className="max-h-[500px] w-auto block object-contain"
                  />
                  <div className="absolute top-2 left-2 pointer-events-none z-10">
                    <Badge className="bg-slate-950/80 border-slate-700 text-slate-300 text-[10px]">
                      Original Reference View
                    </Badge>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Interactive Layer Visualizer Toolbars */}
          <div className="bg-slate-950 border border-slate-800 p-4 rounded-lg flex flex-col md:flex-row gap-4 items-center justify-between">
            <div className="flex items-center gap-6 w-full md:w-auto">
              <div className="flex items-center gap-2">
                <span className="text-xs text-slate-400 font-semibold">Heatmap:</span>
                <input
                  type="checkbox"
                  checked={showHeatmap}
                  onChange={(e) => setShowHeatmap(e.target.checked)}
                  className="rounded border-slate-800 bg-slate-900 text-teal-500 focus:ring-teal-500 h-4 w-4"
                />
              </div>
              <div className="flex items-center gap-2 flex-1 md:w-44">
                <span className="text-xs text-slate-400 font-semibold">Opacity:</span>
                <Slider
                  min={0}
                  max={100}
                  step={1}
                  value={[opacity]}
                  onValueChange={(val: any) => setOpacity(val[0])}
                  disabled={!showHeatmap}
                  className="w-24 md:w-32 bg-slate-800"
                />
                <span className="text-[10px] text-slate-300 font-mono w-6">{opacity}%</span>
              </div>
            </div>

            <div className="flex items-center gap-4 flex-wrap w-full md:w-auto justify-end">
              <div className="flex items-center gap-1 bg-slate-900 p-0.5 rounded border border-slate-800 text-[10px]">
                <button
                  onClick={() => setShowBbox(!showBbox)}
                  className={`px-2 py-1 rounded transition-colors ${showBbox ? "bg-teal-500/10 text-teal-400 border border-teal-500/20" : "text-slate-400"}`}
                >
                  Boxes
                </button>
                <button
                  onClick={() => setShowCircle(!showCircle)}
                  className={`px-2 py-1 rounded transition-colors ${showCircle ? "bg-teal-500/10 text-teal-400 border border-teal-500/20" : "text-slate-400"}`}
                >
                  Circles
                </button>
                <button
                  onClick={() => setShowContour(!showContour)}
                  className={`px-2 py-1 rounded transition-colors ${showContour ? "bg-teal-500/10 text-teal-400 border border-teal-500/20" : "text-slate-400"}`}
                >
                  Contours
                </button>
              </div>

              <select
                value={heatmapMode}
                onChange={(e) => setHeatmapMode(e.target.value as any)}
                className="bg-slate-900 border border-slate-800 rounded px-2 py-1 text-xs text-slate-300 focus:outline-none focus:border-slate-700"
              >
                <option value="gradcam_plusplus">Grad-CAM++</option>
                <option value="gradcam">Grad-CAM</option>
                <option value="attention">Saliency Attention</option>
                <option value="coverage">Spatial Coverage</option>
                <option value="attribution">Local Attribution</option>
              </select>
            </div>
          </div>
        </div>

        {/* Right Side: Diagnostics Dash (5 cols) */}
        <div className={`xl:col-span-5 ${isComparing ? "xl:col-span-4" : "xl:col-span-5"} space-y-4`}>
          {/* Panel 1: Confidence Diagnostics */}
          <div className="bg-slate-950 border border-slate-800 rounded-lg p-4 space-y-3">
            <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider">AI Confidence Metrics</h3>
            <div className="grid grid-cols-2 gap-3">
              <div className="bg-slate-900 p-2.5 rounded border border-slate-800/60">
                <span className="text-[10px] text-slate-400 block">Raw Probability</span>
                <span className="text-lg font-bold font-mono text-slate-200">
                  {metrics?.tb_probability !== undefined ? `${metrics.tb_probability}%` : `${(result.confidence! * 100).toFixed(1)}%`}
                </span>
              </div>
              <div className="bg-slate-900 p-2.5 rounded border border-slate-800/60">
                <span className="text-[10px] text-slate-400 block">Calibrated Certainty</span>
                <span className="text-lg font-bold font-mono text-teal-400">
                  {metrics?.calibrated_confidence !== undefined ? `${metrics.calibrated_confidence}%` : "50.0%"}
                </span>
              </div>
              <div className="bg-slate-900 p-2.5 rounded border border-slate-800/60">
                <span className="text-[10px] text-slate-400 block">System Reliability</span>
                <span className={`text-xs font-bold ${metrics?.reliability === "High" ? "text-emerald-400" : "text-amber-400"}`}>
                  {metrics?.reliability || "High"}
                </span>
              </div>
              <div className="bg-slate-900 p-2.5 rounded border border-slate-800/60">
                <span className="text-[10px] text-slate-400 block">Decision Uncertainty</span>
                <span className={`text-xs font-bold ${metrics?.uncertainty === "Low" ? "text-emerald-400" : metrics?.uncertainty === "Medium" ? "text-amber-400" : "text-red-400"}`}>
                  {metrics?.uncertainty || "Low"}
                </span>
              </div>
            </div>
          </div>

          {/* Panel 2: ROI Rankings list */}
          <div className="bg-slate-950 border border-slate-800 rounded-lg p-4 space-y-3">
            <div className="flex justify-between items-center">
              <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider">Attention Attribution Ranking</h3>
              <Badge className="bg-teal-500/10 text-teal-400 border border-teal-500/20 text-[9px]">
                {rois.length} Regions
              </Badge>
            </div>
            
            <div className="space-y-2 max-h-[170px] overflow-y-auto pr-1">
              {rois.map((roi: any) => {
                const isHovered = hoveredRoiId === roi.id;
                return (
                  <div
                    key={roi.id}
                    onMouseEnter={() => setHoveredRoiId(roi.id)}
                    onMouseLeave={() => setHoveredRoiId(null)}
                    className={`flex items-center justify-between p-2 rounded text-xs transition-all border ${
                      isHovered
                        ? "bg-red-500/10 border-red-500/30 text-white"
                        : "bg-slate-900 border-slate-800/50 text-slate-300 hover:border-slate-700"
                    }`}
                  >
                    <div className="flex items-center gap-2">
                      <span className={`w-5 h-5 rounded flex items-center justify-center font-bold text-[10px] ${
                        isHovered ? "bg-red-500 text-white" : "bg-teal-500/10 text-teal-400"
                      }`}>
                        {roi.id}
                      </span>
                      <div>
                        <div className="font-semibold text-slate-200">{roi.location}</div>
                        <div className="text-[10px] text-slate-500">Local Activation Intensity: {roi.activation_score}%</div>
                      </div>
                    </div>
                    <div className="text-right">
                      <span className="font-bold text-slate-200">{roi.contribution_pct}%</span>
                      <span className="text-[9px] text-slate-500 block">Contribution</span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Panel 3: Explainability Summary */}
          <div className="bg-slate-950 border border-slate-800 rounded-lg p-4 space-y-3">
            <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider">AI Explainability Summary</h3>
            <div className="bg-slate-900 border border-slate-800/60 p-3.5 rounded text-xs text-slate-300 leading-relaxed space-y-3">
              <p>{summary}</p>
              <div className="border-t border-slate-800 pt-2 flex items-start gap-1.5 text-[10px] text-slate-500 italic">
                <span>⚠️</span>
                <span>
                  Findings suggest feature patterns only and require correlation with clinical history, microbiology tests, and expert review.
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Bottom Section: Similar Case Matching */}
      <div className="bg-slate-950 border border-slate-800 rounded-lg p-4 space-y-3">
        <div className="flex justify-between items-center border-b border-slate-800 pb-2">
          <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider">
            Similar Reference Cases (K-Nearest Cohorts)
          </h3>
          <span className="text-[10px] text-slate-500 italic">Matched against Indian Patient Calibration sets</span>
        </div>

        {loadingSimilar ? (
          <div className="flex items-center justify-center py-6 text-xs text-slate-400">
            <span className="animate-spin mr-2">⏳</span> Loading similar reference profiles...
          </div>
        ) : !similarCases || (!similarCases.tb_similar?.length && !similarCases.normal_similar?.length) ? (
          <div className="text-center py-6 text-xs text-slate-500">
            No matching cohort records available for comparison in database.
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Similar TB cases */}
            <div className="space-y-2">
              <span className="text-[10px] font-bold text-red-400 block tracking-wide uppercase">
                Similar Confirmed Tuberculosis Cases
              </span>
              <div className="grid grid-cols-3 gap-2">
                {similarCases.tb_similar.map((cand, idx) => (
                  <div 
                    key={idx} 
                    className="bg-slate-900 border border-slate-800 hover:border-slate-700 rounded p-1.5 text-center transition-all cursor-pointer"
                  >
                    <div className="relative h-16 w-full bg-slate-950 rounded overflow-hidden flex items-center justify-center">
                      <img 
                        src={cand.original_image} 
                        alt="Cohort Case" 
                        className="h-full w-auto object-contain" 
                      />
                      <div className="absolute top-0.5 right-0.5">
                        <Badge className="bg-red-500 text-white text-[8px] px-1 py-0 border-none">
                          {cand.similarity_score}%
                        </Badge>
                      </div>
                    </div>
                    <div className="text-[9px] text-slate-300 font-semibold truncate mt-1">
                      {cand.patient_name || cand.patient_id}
                    </div>
                    <div className="text-[8px] text-slate-500">
                      {cand.age} yrs · {cand.sex}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Similar Normal cases */}
            <div className="space-y-2">
              <span className="text-[10px] font-bold text-emerald-400 block tracking-wide uppercase">
                Similar Confirmed Normal Cases
              </span>
              <div className="grid grid-cols-3 gap-2">
                {similarCases.normal_similar.map((cand, idx) => (
                  <div 
                    key={idx} 
                    className="bg-slate-900 border border-slate-800 hover:border-slate-700 rounded p-1.5 text-center transition-all cursor-pointer"
                  >
                    <div className="relative h-16 w-full bg-slate-950 rounded overflow-hidden flex items-center justify-center">
                      <img 
                        src={cand.original_image} 
                        alt="Cohort Case" 
                        className="h-full w-auto object-contain" 
                      />
                      <div className="absolute top-0.5 right-0.5">
                        <Badge className="bg-teal-500 text-white text-[8px] px-1 py-0 border-none">
                          {cand.similarity_score}%
                        </Badge>
                      </div>
                    </div>
                    <div className="text-[9px] text-slate-300 font-semibold truncate mt-1">
                      {cand.patient_name || cand.patient_id}
                    </div>
                    <div className="text-[8px] text-slate-500">
                      {cand.age} yrs · {cand.sex}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
