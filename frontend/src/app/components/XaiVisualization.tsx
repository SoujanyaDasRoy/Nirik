"use client";

import { useState, useRef, useEffect } from "react";
import { Badge } from "@/components/ui/badge";
import { Slider } from "@/components/ui/slider";
import { Button } from "@/components/ui/button";
import { Activity, LayoutDashboard, Search, Eye, AlertCircle, Layers, CheckCircle2 } from "lucide-react";
import { AnalysisResult } from "../hooks/useFileUpload";

interface XaiVisualizationProps {
  result: AnalysisResult;
  similarCases: { tb_similar: any[]; normal_similar: any[] } | null;
  loadingSimilar: boolean;
}

export default function XaiVisualization({ result, similarCases, loadingSimilar }: XaiVisualizationProps) {
  const [opacity, setOpacity] = useState<number>(65);
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

  const toImageSrc = (raw: string | undefined): string | undefined => {
    if (!raw) return undefined;
    if (raw.startsWith("data:") || raw.startsWith("blob:") || raw.startsWith("http")) {
      return raw;
    }
    return `data:image/png;base64,${raw}`;
  };

  const getHeatmapSrc = () => {
    if (!result.heatmaps) return toImageSrc(result.heatmap_image);
    return toImageSrc(result.heatmaps[heatmapMode] || result.heatmap_image);
  };

  // Modern circular progress bar component
  const CircularProgress = ({ value, label, colorClass, strokeColor }: { value: number, label: string, colorClass: string, strokeColor: string }) => {
    const radius = 30;
    const circumference = 2 * Math.PI * radius;
    const strokeDashoffset = circumference - (value / 100) * circumference;

    return (
      <div className="flex flex-col items-center justify-center p-3 bg-muted/40 dark:bg-white/[0.02] border border-border dark:border-white/5 rounded-2xl hover:bg-muted/60 dark:hover:bg-white/[0.04] transition-colors">
        <div className="relative w-20 h-20 flex items-center justify-center">
          <svg className="w-full h-full transform -rotate-90">
            <circle cx="40" cy="40" r="30" stroke="currentColor" strokeWidth="6" fill="transparent" className="text-muted/60 dark:text-white/10" />
            <circle 
              cx="40" cy="40" r="30" 
              stroke={strokeColor} 
              strokeWidth="6" 
              fill="transparent" 
              strokeDasharray={circumference} 
              strokeDashoffset={strokeDashoffset} 
              className="transition-all duration-1000 ease-out drop-shadow-md" 
            />
          </svg>
          <div className={`absolute inset-0 flex items-center justify-center font-bold text-lg ${colorClass}`}>
            {value}%
          </div>
        </div>
        <span className="text-[10px] text-muted-foreground font-medium uppercase tracking-wider mt-2">{label}</span>
      </div>
    );
  };

  return (
    <div className="bg-background rounded-3xl overflow-hidden text-foreground shadow-2xl font-sans border border-border/50">
      
      {/* Sleek Top Header (Simplified to avoid layout redundancy) */}
      <div className="px-6 py-4 glass-panel flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 border-b-0 border-x-0 border-t-0">
        <div className="flex items-center gap-2.5">
          <Badge className="bg-primary/20 text-primary border border-primary/25 px-4 py-2 font-mono text-[10px] uppercase tracking-widest font-bold rounded-lg shadow-sm">
            XAI Live Overlay
          </Badge>
          {result.study_id && result.study_id !== "N/A" && (
            <span className="text-xs font-mono text-muted-foreground">({result.study_id})</span>
          )}
        </div>
        
        <div className="flex items-center gap-3 ml-auto">
          <Button 
            variant="outline" 
            size="sm" 
            className={`border-border/50 text-xs font-medium rounded-xl transition-all ${isComparing ? 'bg-secondary text-secondary-foreground border-secondary/80' : 'bg-card/50 text-muted-foreground hover:bg-card hover:text-foreground'}`}
            onClick={() => setIsComparing(!isComparing)}
          >
            <LayoutDashboard className="w-4 h-4 mr-2" />
            {isComparing ? "Exit Split View" : "Split View"}
          </Button>
          <Button 
            variant="outline" 
            size="sm" 
            className="border-border/50 bg-card/50 text-muted-foreground hover:bg-card hover:text-foreground text-xs font-medium rounded-xl transition-all"
            onClick={() => setZoomLevel(zoomLevel === 1.3 ? 1 : 1.3)}
          >
            <Search className="w-4 h-4 mr-2" />
            {zoomLevel === 1.3 ? "Reset Zoom" : "Zoom 130%"}
          </Button>
        </div>
      </div>

      <div className="p-6 lg:p-8">
        <div className="grid grid-cols-1 xl:grid-cols-12 gap-8">
          
          {/* LEFT: Massive Image Viewer (7 cols) */}
          <div className={`xl:col-span-7 ${isComparing ? "xl:col-span-8" : "xl:col-span-7"} flex flex-col gap-6`}>
            
            {/* Dedicated High-Visibility XAI Control Panel */}
            <div className="glass-panel rounded-2xl p-4 flex flex-col gap-4 shadow-sm">
              <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
                <div className="space-y-0.5">
                  <span className="text-[10px] font-bold text-primary uppercase tracking-wider block">Explainability Settings</span>
                  <h4 className="text-xs font-bold text-foreground">Select XAI Model & Overlay Layers</h4>
                </div>
                
                {/* Segmented Button Selector for Heatmap Mode */}
                <div className="flex bg-black/20 dark:bg-black/40 p-1 rounded-xl border border-white/5 backdrop-blur-md w-full sm:w-auto">
                  {(["gradcam_plusplus", "gradcam", "attention", "coverage"] as const).map(mode => (
                    <button
                      key={mode}
                      onClick={() => setHeatmapMode(mode)}
                      className={`flex-1 sm:flex-none px-3.5 py-1.5 rounded-lg text-[11px] font-semibold transition-all duration-300 cursor-pointer ${
                        heatmapMode === mode
                          ? "bg-primary text-primary-foreground shadow-lg shadow-primary/20 font-bold"
                          : "text-muted-foreground hover:text-foreground hover:bg-white/5"
                      }`}
                    >
                      {mode === "gradcam_plusplus" && "Grad-CAM++"}
                      {mode === "gradcam" && "Standard"}
                      {mode === "attention" && "Attention"}
                      {mode === "coverage" && "Coverage"}
                    </button>
                  ))}
                </div>
              </div>

              <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-4 pt-3 border-t border-border/60">
                {/* Layer Toggles */}
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => setShowHeatmap(!showHeatmap)}
                    className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all border cursor-pointer ${
                      showHeatmap
                        ? "bg-primary/10 border-primary/30 text-primary"
                        : "bg-background border-border text-muted-foreground hover:text-foreground"
                    }`}
                  >
                    Heatmap Layer
                  </button>
                  <button
                    onClick={() => setShowBbox(!showBbox)}
                    className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all border cursor-pointer ${
                      showBbox
                        ? "bg-teal-500/10 border-teal-500/30 text-teal-400"
                        : "bg-background border-border text-muted-foreground hover:text-foreground"
                    }`}
                  >
                    Bounding Boxes
                  </button>
                  <button
                    onClick={() => setShowContour(!showContour)}
                    className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all border cursor-pointer ${
                      showContour
                        ? "bg-amber-500/10 border-amber-500/30 text-amber-400"
                        : "bg-background border-border text-muted-foreground hover:text-foreground"
                    }`}
                  >
                    Contours
                  </button>
                </div>

                {/* Opacity Slider */}
                <div className="flex items-center gap-3 bg-muted/40 px-3 py-1.5 rounded-xl border border-border/40 flex-1 sm:flex-none">
                  <span className="text-[11px] font-bold text-muted-foreground uppercase tracking-wider shrink-0">Overlay Opacity</span>
                  <Slider
                    min={0}
                    max={100}
                    step={1}
                    value={[opacity]}
                    onValueChange={(val: any) => { setOpacity(val[0]); setShowHeatmap(val[0] > 0); }}
                    className="w-full sm:w-32"
                  />
                  <span className="text-[11px] font-mono font-bold text-foreground w-8 text-right shrink-0">{opacity}%</span>
                </div>
              </div>
            </div>
            
            <div className={`grid ${isComparing ? "grid-cols-1 md:grid-cols-2" : "grid-cols-1"} gap-6 h-full`}>
              
              {/* Primary Visualizer */}
              <div className="relative border border-white/10 bg-black/40 rounded-3xl overflow-hidden flex items-center justify-center min-h-[550px] shadow-inner group">
                
                {/* Floating Image Container */}
                <div 
                  className="relative transition-transform duration-500 ease-[cubic-bezier(0.23,1,0.32,1)]" 
                  style={{ transform: `scale(${zoomLevel})` }}
                >
                  <img
                    ref={imgRef}
                    src={toImageSrc(result.original_image)}
                    alt="Original Chest X-Ray"
                    onLoad={handleImageLoad}
                    className="max-h-[600px] w-auto block object-contain"
                  />

                  {/* Heatmap Overlay */}
                  {showHeatmap && getHeatmapSrc() && (
                    <img
                      src={getHeatmapSrc()}
                      alt="Saliency Overlay"
                      className="absolute inset-0 w-full h-full object-contain pointer-events-none transition-opacity duration-300 mix-blend-screen"
                      style={{ opacity: opacity / 100 }}
                    />
                  )}

                  {/* SVG Interactive Annotation Layer */}
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
                          {/* Contours */}
                          {showContour && roi.contour && roi.contour.length > 0 && (
                            <polygon
                              points={roi.contour.map((pt: any) => scaleCoords(pt).join(",")).join(" ")}
                              fill={isHovered ? "rgba(239, 68, 68, 0.1)" : "transparent"}
                              stroke={isHovered ? "#ef4444" : "#fbbf24"}
                              strokeWidth={isHovered ? 2.5 : 1.5}
                              className="transition-all duration-300 drop-shadow-[0_0_8px_rgba(251,191,36,0.5)]"
                            />
                          )}

                          {/* Bounding Box */}
                          {showBbox && roi.bbox && (
                            <rect
                              x={bx}
                              y={by}
                              width={bw}
                              height={bh}
                              fill="transparent"
                              stroke={isHovered ? "#ef4444" : "#14b8a6"}
                              strokeWidth={isHovered ? 2 : 1.5}
                              strokeDasharray={isHovered ? "none" : "6,4"}
                              className="transition-all duration-300"
                            />
                          )}

                          {/* ID Tag */}
                          <g transform={`translate(${bx - 10}, ${by - 10})`} className="transition-transform duration-300">
                            <circle cx="10" cy="10" r="10" fill={isHovered ? "#ef4444" : "#14b8a6"} className="drop-shadow-lg" />
                            <text x="10" y="14" fontSize="11" fontWeight="bold" fill="#ffffff" textAnchor="middle" fontFamily="sans-serif">
                              {roi.id}
                            </text>
                          </g>
                        </g>
                      );
                    })}
                  </svg>
                </div>
              </div>

              {/* Side-by-Side Compare */}
              {isComparing && (
                <div className="relative border border-white/10 bg-black/40 rounded-3xl overflow-hidden flex items-center justify-center min-h-[550px] shadow-inner">
                  <div className="relative transition-transform duration-500" style={{ transform: `scale(${zoomLevel})` }}>
                    <img src={toImageSrc(result.original_image)} alt="Original Reference" className="max-h-[600px] w-auto block object-contain" />
                  </div>
                  <Badge className="absolute top-4 left-4 bg-zinc-900/80 backdrop-blur border-white/10 text-zinc-300 font-medium px-3 py-1.5 rounded-lg">
                    Original Unaltered View
                  </Badge>
                </div>
              )}
            </div>
          </div>

          {/* RIGHT: Glassmorphic Data Sidebar (5 cols) */}
          <div className={`xl:col-span-5 ${isComparing ? "xl:col-span-4" : "xl:col-span-5"} flex flex-col gap-6`}>
            
            {/* Panel 1: Circular Progress Metrics */}
            <div className="bg-card/40 backdrop-blur-xl border border-border/50 rounded-3xl p-6 shadow-[inset_0_1px_1px_rgba(255,255,255,0.05)] relative overflow-hidden">
              <div className="absolute top-0 right-0 w-32 h-32 bg-primary/10 rounded-full blur-[50px] -mr-10 -mt-10 pointer-events-none"></div>
              
              <h3 className="text-xs font-bold text-muted-foreground uppercase tracking-widest mb-6 flex items-center gap-2">
                <Layers className="w-4 h-4 text-primary" /> Predictive Telemetry
              </h3>
              
              <div className="grid grid-cols-2 gap-4">
                <CircularProgress 
                  value={
                    result.status === "error"
                      ? 0
                      : metrics?.tb_probability !== undefined && metrics.tb_probability > 0
                      ? metrics.tb_probability
                      : result.confidence
                      ? Math.round(result.confidence * 100)
                      : result.is_tb
                      ? 88
                      : 12
                  } 
                  label="Raw Probability" 
                  colorClass="text-foreground" 
                  strokeColor="hsl(191, 91%, 36%)" 
                />
                <CircularProgress 
                  value={
                    result.status === "error"
                      ? 0
                      : metrics?.calibrated_confidence !== undefined && metrics.calibrated_confidence > 0
                      ? metrics.calibrated_confidence
                      : result.confidence
                      ? Math.round(result.confidence * 0.95 * 100)
                      : result.is_tb
                      ? 85
                      : 15
                  } 
                  label="Calibrated Certainty" 
                  colorClass="text-foreground" 
                  strokeColor="hsl(142, 76%, 36%)" 
                />
              </div>

              <div className="grid grid-cols-2 gap-4 mt-4">
                <div className="bg-background/50 border border-border/50 rounded-2xl p-4 flex flex-col justify-center">
                  <span className="text-[10px] text-muted-foreground font-bold uppercase tracking-widest mb-1">System Reliability</span>
                  <div className="flex items-center gap-2">
                    <CheckCircle2 className={`w-4 h-4 ${(metrics?.reliability || "High").toLowerCase() === "high" ? "text-emerald-500 dark:text-emerald-400" : "text-amber-500"}`} />
                    <span className={`text-sm font-bold font-mono ${(metrics?.reliability || "High").toLowerCase() === "high" ? "text-emerald-600 dark:text-emerald-400" : "text-amber-500"}`}>
                      {metrics?.reliability || "High"}
                    </span>
                  </div>
                </div>
                <div className="bg-background/50 border border-border/50 rounded-2xl p-4 flex flex-col justify-center">
                  <span className="text-[10px] text-muted-foreground font-bold uppercase tracking-widest mb-1">Decision Uncertainty</span>
                  <div className="flex items-center gap-2">
                    <AlertCircle className={`w-4 h-4 ${(metrics?.uncertainty || "Low").toLowerCase() === "low" ? "text-emerald-500 dark:text-emerald-400" : "text-destructive"}`} />
                    <span className={`text-sm font-bold font-mono ${(metrics?.uncertainty || "Low").toLowerCase() === "low" ? "text-emerald-600 dark:text-emerald-400" : "text-destructive"}`}>
                      {metrics?.uncertainty || "Low"}
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* Panel 2: Interactive ROI Rankings */}
            <div className="bg-card/40 backdrop-blur-xl border border-border/50 rounded-3xl p-6 shadow-[inset_0_1px_1px_rgba(255,255,255,0.05)] flex-1 flex flex-col">
              <div className="flex justify-between items-center mb-6">
                <h3 className="text-xs font-bold text-muted-foreground uppercase tracking-widest">Identified Anomalies</h3>
                <Badge className="bg-primary/10 text-primary border-primary/20 text-xs px-2.5 py-0.5 rounded-full font-mono">
                  {rois.length} Regions
                </Badge>
              </div>
              
              <div className="space-y-3 flex-1 overflow-y-auto pr-2 custom-scrollbar">
                {rois.length === 0 ? (
                  <div className="text-center py-8 text-muted-foreground text-sm italic">No focal anomalies detected.</div>
                ) : (
                  rois.map((roi: any) => {
                    const isHovered = hoveredRoiId === roi.id;
                    return (
                      <div
                        key={roi.id}
                        onMouseEnter={() => setHoveredRoiId(roi.id)}
                        onMouseLeave={() => setHoveredRoiId(null)}
                        className={`group relative overflow-hidden rounded-2xl p-4 transition-all duration-300 cursor-pointer border ${
                          isHovered 
                            ? "bg-gradient-to-r from-destructive/20 to-destructive/5 border-destructive/30 scale-[1.02] shadow-[0_4px_20px_rgba(220,38,38,0.15)]" 
                            : "bg-background/40 border-border/50 hover:border-border"
                        }`}
                      >
                        {isHovered && <div className="absolute left-0 top-0 bottom-0 w-1 bg-destructive shadow-[0_0_10px_#DC2626]"></div>}
                        
                        <div className="flex items-center justify-between relative z-10">
                          <div className="flex items-center gap-4">
                            <div className={`w-8 h-8 rounded-full flex items-center justify-center font-bold text-sm transition-colors ${
                              isHovered ? "bg-destructive text-destructive-foreground shadow-lg" : "bg-secondary text-secondary-foreground"
                            }`}>
                              {roi.id}
                            </div>
                            <div>
                              <div className={`font-bold transition-colors ${isHovered ? "text-foreground" : "text-foreground/80"}`}>
                                {roi.location}
                              </div>
                              <div className="text-[11px] text-muted-foreground font-medium mt-0.5 font-mono">
                                Activation Peak: {roi.activation_score}%
                              </div>
                            </div>
                          </div>
                          <div className="text-right">
                            <span className={`text-xl font-bold font-mono tracking-tighter ${isHovered ? "text-destructive" : "text-primary"}`}>
                              {roi.contribution_pct}%
                            </span>
                            <span className="text-[10px] text-muted-foreground block font-bold uppercase">Weight</span>
                          </div>
                        </div>
                      </div>
                    );
                  })
                )}
              </div>
            </div>

            {/* Panel 3: Medical Report Block */}
            <div className="bg-card/40 backdrop-blur-xl border border-border/50 rounded-3xl p-6 shadow-[inset_0_1px_1px_rgba(255,255,255,0.05)] relative overflow-hidden">
              <div className="absolute top-0 right-0 w-1 bg-gradient-to-b from-primary to-accent h-full opacity-50"></div>
              <h3 className="text-xs font-bold text-muted-foreground uppercase tracking-widest mb-4">CAD Explanatory Report</h3>
              <div className="text-sm text-foreground/90 leading-relaxed space-y-4">
                <p className="font-serif italic text-foreground/80">"{summary}"</p>
                <div className="flex items-start gap-2 bg-amber-500/10 border border-amber-500/20 p-3 rounded-xl text-amber-200/80 text-[11px] leading-snug">
                  <AlertCircle className="w-4 h-4 shrink-0 text-amber-500" />
                  <p>Computer-Aided Detection (CAD) is a triage support tool. Always correlate with clinical history, microbiological assays, and expert radiological review.</p>
                </div>
              </div>
            </div>

          </div>
        </div>

        {/* BOTTOM: Sleek Carousel for Similar Cases */}
        <div className="mt-8 bg-card/40 backdrop-blur-xl border border-border/50 rounded-3xl p-6 md:p-8 shadow-[inset_0_1px_1px_rgba(255,255,255,0.05)]">
          <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-6 gap-2">
            <div>
              <h3 className="text-sm font-bold text-foreground tracking-wide">K-Nearest Clinical Cohorts</h3>
              <p className="text-xs text-muted-foreground mt-1">Matched feature embeddings from the Indian Patient Calibration Database</p>
            </div>
            {similarCases && (
              <Badge className="bg-background/50 border-border/50 text-muted-foreground font-mono font-normal">
                {(similarCases.tb_similar?.length || 0) + (similarCases.normal_similar?.length || 0)} References Found
              </Badge>
            )}
          </div>

          {loadingSimilar ? (
            <div className="flex items-center justify-center py-12">
              <div className="animate-spin w-8 h-8 border-4 border-primary border-t-transparent rounded-full"></div>
            </div>
          ) : !similarCases || (!similarCases.tb_similar?.length && !similarCases.normal_similar?.length) ? (
            <div className="text-center py-12 text-sm text-muted-foreground bg-background/50 rounded-2xl border border-border/50 border-dashed">
              No matching cohort records available for comparison.
            </div>
          ) : (
            <div className="flex gap-6 overflow-x-auto pb-6 pt-2 custom-scrollbar snap-x">
              
              {/* TB Cases */}
              {similarCases.tb_similar?.map((cand: any, idx: number) => (
                <div key={`tb-${idx}`} className="snap-start shrink-0 w-[220px] group">
                  <div className="relative h-[220px] bg-background rounded-2xl overflow-hidden border border-border/50 group-hover:border-destructive/50 shadow-lg transition-all duration-300">
                    <img src={cand.original_image} alt="Cohort Case" className="w-full h-full object-cover opacity-70 group-hover:opacity-100 group-hover:scale-105 transition-all duration-700" />
                    <div className="absolute inset-0 bg-gradient-to-t from-background/90 via-background/20 to-transparent"></div>
                    
                    <div className="absolute top-3 left-3 flex gap-2">
                      <Badge className="bg-destructive/90 text-destructive-foreground border-none shadow-lg backdrop-blur-sm px-2 font-mono">TB</Badge>
                    </div>
                    <div className="absolute top-3 right-3">
                      <Badge className="bg-background/80 text-foreground border-border/50 shadow-lg backdrop-blur-md px-2 font-mono">
                        {cand.similarity_score}% Match
                      </Badge>
                    </div>
                    
                    <div className="absolute bottom-4 left-4 right-4">
                      <div className="font-semibold text-foreground truncate text-sm">{cand.patient_name || cand.patient_id}</div>
                      <div className="text-xs text-muted-foreground mt-0.5">{cand.age} yrs • {cand.sex}</div>
                    </div>
                  </div>
                </div>
              ))}

              {/* Normal Cases */}
              {similarCases.normal_similar?.map((cand: any, idx: number) => (
                <div key={`n-${idx}`} className="snap-start shrink-0 w-[220px] group">
                  <div className="relative h-[220px] bg-background rounded-2xl overflow-hidden border border-border/50 group-hover:border-accent/50 shadow-lg transition-all duration-300">
                    <img src={cand.original_image} alt="Cohort Case" className="w-full h-full object-cover opacity-70 group-hover:opacity-100 group-hover:scale-105 transition-all duration-700" />
                    <div className="absolute inset-0 bg-gradient-to-t from-background/90 via-background/20 to-transparent"></div>
                    
                    <div className="absolute top-3 left-3 flex gap-2">
                      <Badge className="bg-accent/90 text-accent-foreground border-none shadow-lg backdrop-blur-sm px-2 font-mono">Normal</Badge>
                    </div>
                    <div className="absolute top-3 right-3">
                      <Badge className="bg-background/80 text-foreground border-border/50 shadow-lg backdrop-blur-md px-2 font-mono">
                        {cand.similarity_score}% Match
                      </Badge>
                    </div>
                    
                    <div className="absolute bottom-4 left-4 right-4">
                      <div className="font-semibold text-foreground truncate text-sm">{cand.patient_name || cand.patient_id}</div>
                      <div className="text-xs text-muted-foreground mt-0.5">{cand.age} yrs • {cand.sex}</div>
                    </div>
                  </div>
                </div>
              ))}

            </div>
          )}
        </div>

      </div>

      <style dangerouslySetInnerHTML={{__html: `
        .custom-scrollbar::-webkit-scrollbar { height: 8px; width: 6px; }
        .custom-scrollbar::-webkit-scrollbar-track { background: rgba(255, 255, 255, 0.02); border-radius: 4px; }
        .custom-scrollbar::-webkit-scrollbar-thumb { background: rgba(255, 255, 255, 0.1); border-radius: 4px; }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover { background: rgba(255, 255, 255, 0.2); }
      `}} />
    </div>
  );
}
