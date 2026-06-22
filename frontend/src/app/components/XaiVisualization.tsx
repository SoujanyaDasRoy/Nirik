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

  const getHeatmapSrc = () => {
    if (!result.heatmaps) return result.heatmap_image;
    return result.heatmaps[heatmapMode] || result.heatmap_image;
  };

  // Modern circular progress bar component
  const CircularProgress = ({ value, label, colorClass, strokeColor }: { value: number, label: string, colorClass: string, strokeColor: string }) => {
    const radius = 30;
    const circumference = 2 * Math.PI * radius;
    const strokeDashoffset = circumference - (value / 100) * circumference;

    return (
      <div className="flex flex-col items-center justify-center p-3 bg-white/[0.02] border border-white/5 rounded-2xl hover:bg-white/[0.04] transition-colors">
        <div className="relative w-20 h-20 flex items-center justify-center">
          <svg className="w-full h-full transform -rotate-90">
            <circle cx="40" cy="40" r="30" stroke="currentColor" strokeWidth="6" fill="transparent" className="text-white/10" />
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
        <span className="text-[10px] text-zinc-400 font-medium uppercase tracking-wider mt-2">{label}</span>
      </div>
    );
  };

  return (
    <div className="bg-[#09090b] rounded-3xl overflow-hidden text-zinc-100 shadow-2xl font-sans">
      
      {/* Sleek Top Header */}
      <div className="px-8 py-5 border-b border-white/10 bg-white/[0.02] backdrop-blur-xl flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
        <div className="flex items-center gap-4">
          <div className="w-10 h-10 rounded-xl bg-teal-500/20 border border-teal-500/30 flex items-center justify-center shadow-[0_0_15px_rgba(20,184,166,0.2)]">
            <Activity className="w-5 h-5 text-teal-400" />
          </div>
          <div>
            <h2 className="text-xl font-semibold tracking-tight text-white flex items-center gap-2">
              Diagnostic Workbench
              <Badge className="bg-teal-500/20 text-teal-300 border-none px-2 font-mono text-[10px] uppercase tracking-widest">
                XAI Live
              </Badge>
            </h2>
            <p className="text-sm text-zinc-400 mt-0.5">
              Study ID: <span className="font-mono text-zinc-300">{result.study_id || "N/A"}</span>
            </p>
          </div>
        </div>
        
        <div className="flex items-center gap-3">
          <Button 
            variant="outline" 
            size="sm" 
            className={`border-white/10 text-xs font-medium rounded-xl transition-all ${isComparing ? 'bg-indigo-500/20 text-indigo-300 border-indigo-500/30' : 'bg-white/5 text-zinc-300 hover:bg-white/10 hover:text-white'}`}
            onClick={() => setIsComparing(!isComparing)}
          >
            <LayoutDashboard className="w-4 h-4 mr-2" />
            {isComparing ? "Exit Split View" : "Split View"}
          </Button>
          <Button 
            variant="outline" 
            size="sm" 
            className="border-white/10 bg-white/5 text-zinc-300 hover:bg-white/10 hover:text-white text-xs font-medium rounded-xl transition-all"
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
                    src={result.original_image}
                    alt="Original Chest X-Ray"
                    onLoad={handleImageLoad}
                    className="max-h-[600px] w-auto block object-contain"
                  />

                  {/* Heatmap Overlay */}
                  {showHeatmap && (
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

                {/* Floating Dock Tool Bar (macOS style) */}
                <div className="absolute bottom-6 left-1/2 -translate-x-1/2 flex items-center gap-3 bg-zinc-900/80 backdrop-blur-2xl border border-white/10 p-2.5 rounded-2xl shadow-2xl opacity-90 group-hover:opacity-100 transition-opacity">
                  <div className="flex items-center gap-2 px-3 border-r border-white/10">
                    <Eye className={`w-4 h-4 ${showHeatmap ? 'text-teal-400' : 'text-zinc-500'}`} />
                    <Slider
                      min={0}
                      max={100}
                      step={1}
                      value={[opacity]}
                      onValueChange={(val: any) => { setOpacity(val[0]); setShowHeatmap(val[0] > 0); }}
                      className="w-24 bg-zinc-800"
                    />
                  </div>
                  
                  <div className="flex items-center gap-1.5 px-2">
                    <Button 
                      size="sm" variant="ghost" 
                      onClick={() => setShowBbox(!showBbox)}
                      className={`h-8 px-3 rounded-lg text-xs font-medium transition-all ${showBbox ? 'bg-teal-500/20 text-teal-300 hover:bg-teal-500/30' : 'text-zinc-400 hover:text-white'}`}
                    >
                      Boxes
                    </Button>
                    <Button 
                      size="sm" variant="ghost" 
                      onClick={() => setShowContour(!showContour)}
                      className={`h-8 px-3 rounded-lg text-xs font-medium transition-all ${showContour ? 'bg-amber-500/20 text-amber-300 hover:bg-amber-500/30' : 'text-zinc-400 hover:text-white'}`}
                    >
                      Contours
                    </Button>
                  </div>

                  <div className="border-l border-white/10 pl-2">
                    <select
                      value={heatmapMode}
                      onChange={(e) => setHeatmapMode(e.target.value as any)}
                      className="bg-black/50 border border-white/10 rounded-lg px-3 py-1.5 text-xs text-zinc-300 focus:outline-none focus:border-teal-500/50 appearance-none cursor-pointer hover:bg-black/70 transition-colors"
                    >
                      <option value="gradcam_plusplus">Grad-CAM++</option>
                      <option value="gradcam">Standard Grad-CAM</option>
                      <option value="attention">Attention Saliency</option>
                      <option value="coverage">Spatial Coverage</option>
                    </select>
                  </div>
                </div>

              </div>

              {/* Side-by-Side Compare */}
              {isComparing && (
                <div className="relative border border-white/10 bg-black/40 rounded-3xl overflow-hidden flex items-center justify-center min-h-[550px] shadow-inner">
                  <div className="relative transition-transform duration-500" style={{ transform: `scale(${zoomLevel})` }}>
                    <img src={result.original_image} alt="Original Reference" className="max-h-[600px] w-auto block object-contain" />
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
            <div className="bg-white/[0.02] backdrop-blur-xl border border-white/5 rounded-3xl p-6 shadow-xl relative overflow-hidden">
              <div className="absolute top-0 right-0 w-32 h-32 bg-teal-500/10 rounded-full blur-[50px] -mr-10 -mt-10 pointer-events-none"></div>
              
              <h3 className="text-xs font-bold text-zinc-400 uppercase tracking-widest mb-6 flex items-center gap-2">
                <Layers className="w-4 h-4 text-zinc-500" /> Predictive Telemetry
              </h3>
              
              <div className="grid grid-cols-2 gap-4">
                <CircularProgress 
                  value={metrics?.tb_probability !== undefined ? metrics.tb_probability : Math.round((result.confidence || 0) * 100)} 
                  label="Raw Probability" 
                  colorClass="text-white" 
                  strokeColor="#3b82f6" 
                />
                <CircularProgress 
                  value={metrics?.calibrated_confidence !== undefined ? metrics.calibrated_confidence : 50} 
                  label="Calibrated Certainty" 
                  colorClass="text-teal-400" 
                  strokeColor="#14b8a6" 
                />
              </div>

              <div className="grid grid-cols-2 gap-4 mt-4">
                <div className="bg-white/[0.02] border border-white/5 rounded-2xl p-4 flex flex-col justify-center">
                  <span className="text-[10px] text-zinc-500 font-bold uppercase tracking-widest mb-1">System Reliability</span>
                  <div className="flex items-center gap-2">
                    <CheckCircle2 className={`w-4 h-4 ${metrics?.reliability === "High" ? "text-emerald-500" : "text-amber-500"}`} />
                    <span className={`text-sm font-bold ${metrics?.reliability === "High" ? "text-emerald-400" : "text-amber-400"}`}>
                      {metrics?.reliability || "High"}
                    </span>
                  </div>
                </div>
                <div className="bg-white/[0.02] border border-white/5 rounded-2xl p-4 flex flex-col justify-center">
                  <span className="text-[10px] text-zinc-500 font-bold uppercase tracking-widest mb-1">Decision Uncertainty</span>
                  <div className="flex items-center gap-2">
                    <AlertCircle className={`w-4 h-4 ${metrics?.uncertainty === "Low" ? "text-emerald-500" : "text-red-500"}`} />
                    <span className={`text-sm font-bold ${metrics?.uncertainty === "Low" ? "text-emerald-400" : "text-red-400"}`}>
                      {metrics?.uncertainty || "Low"}
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* Panel 2: Interactive ROI Rankings */}
            <div className="bg-white/[0.02] backdrop-blur-xl border border-white/5 rounded-3xl p-6 shadow-xl flex-1 flex flex-col">
              <div className="flex justify-between items-center mb-6">
                <h3 className="text-xs font-bold text-zinc-400 uppercase tracking-widest">Identified Anomalies</h3>
                <Badge className="bg-teal-500/10 text-teal-300 border-teal-500/20 text-xs px-2.5 py-0.5 rounded-full">
                  {rois.length} Regions
                </Badge>
              </div>
              
              <div className="space-y-3 flex-1 overflow-y-auto pr-2 custom-scrollbar">
                {rois.length === 0 ? (
                  <div className="text-center py-8 text-zinc-500 text-sm italic">No focal anomalies detected.</div>
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
                            ? "bg-gradient-to-r from-red-500/20 to-rose-500/5 border-red-500/30 scale-[1.02] shadow-lg" 
                            : "bg-white/[0.02] border-white/5 hover:border-white/10"
                        }`}
                      >
                        {isHovered && <div className="absolute left-0 top-0 bottom-0 w-1 bg-red-500 shadow-[0_0_10px_#ef4444]"></div>}
                        
                        <div className="flex items-center justify-between relative z-10">
                          <div className="flex items-center gap-4">
                            <div className={`w-8 h-8 rounded-full flex items-center justify-center font-bold text-sm transition-colors ${
                              isHovered ? "bg-red-500 text-white shadow-lg" : "bg-white/10 text-zinc-300"
                            }`}>
                              {roi.id}
                            </div>
                            <div>
                              <div className={`font-bold transition-colors ${isHovered ? "text-white" : "text-zinc-200"}`}>
                                {roi.location}
                              </div>
                              <div className="text-[11px] text-zinc-500 font-medium mt-0.5">
                                Activation Peak: {roi.activation_score}%
                              </div>
                            </div>
                          </div>
                          <div className="text-right">
                            <span className={`text-xl font-bold font-mono tracking-tighter ${isHovered ? "text-red-400" : "text-teal-400"}`}>
                              {roi.contribution_pct}%
                            </span>
                            <span className="text-[10px] text-zinc-500 block font-bold uppercase">Weight</span>
                          </div>
                        </div>
                      </div>
                    );
                  })
                )}
              </div>
            </div>

            {/* Panel 3: Medical Report Block */}
            <div className="bg-white/[0.02] backdrop-blur-xl border border-white/5 rounded-3xl p-6 shadow-xl relative overflow-hidden">
              <div className="absolute top-0 right-0 w-1 bg-gradient-to-b from-teal-500 to-indigo-500 h-full opacity-50"></div>
              <h3 className="text-xs font-bold text-zinc-400 uppercase tracking-widest mb-4">CAD Explanatory Report</h3>
              <div className="text-sm text-zinc-300 leading-relaxed space-y-4">
                <p className="font-serif italic text-zinc-200">"{summary}"</p>
                <div className="flex items-start gap-2 bg-yellow-500/10 border border-yellow-500/20 p-3 rounded-xl text-yellow-200/80 text-[11px] leading-snug">
                  <AlertCircle className="w-4 h-4 shrink-0 text-yellow-500" />
                  <p>Computer-Aided Detection (CAD) is a triage support tool. Always correlate with clinical history, microbiological assays, and expert radiological review.</p>
                </div>
              </div>
            </div>

          </div>
        </div>

        {/* BOTTOM: Sleek Carousel for Similar Cases */}
        <div className="mt-8 bg-white/[0.02] backdrop-blur-xl border border-white/5 rounded-3xl p-6 md:p-8 shadow-xl">
          <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-6 gap-2">
            <div>
              <h3 className="text-sm font-bold text-zinc-200 tracking-wide">K-Nearest Clinical Cohorts</h3>
              <p className="text-xs text-zinc-500 mt-1">Matched feature embeddings from the Indian Patient Calibration Database</p>
            </div>
            {similarCases && (
              <Badge className="bg-white/5 border-white/10 text-zinc-400 font-normal">
                {(similarCases.tb_similar?.length || 0) + (similarCases.normal_similar?.length || 0)} References Found
              </Badge>
            )}
          </div>

          {loadingSimilar ? (
            <div className="flex items-center justify-center py-12">
              <div className="animate-spin w-8 h-8 border-4 border-teal-500 border-t-transparent rounded-full"></div>
            </div>
          ) : !similarCases || (!similarCases.tb_similar?.length && !similarCases.normal_similar?.length) ? (
            <div className="text-center py-12 text-sm text-zinc-500 bg-black/20 rounded-2xl border border-white/5 border-dashed">
              No matching cohort records available for comparison.
            </div>
          ) : (
            <div className="flex gap-6 overflow-x-auto pb-6 pt-2 custom-scrollbar snap-x">
              
              {/* TB Cases */}
              {similarCases.tb_similar?.map((cand: any, idx: number) => (
                <div key={`tb-${idx}`} className="snap-start shrink-0 w-[220px] group">
                  <div className="relative h-[220px] bg-black/60 rounded-2xl overflow-hidden border border-white/5 group-hover:border-red-500/30 transition-all duration-300">
                    <img src={cand.original_image} alt="Cohort Case" className="w-full h-full object-cover opacity-70 group-hover:opacity-100 group-hover:scale-105 transition-all duration-700" />
                    <div className="absolute inset-0 bg-gradient-to-t from-black/90 via-black/20 to-transparent"></div>
                    
                    <div className="absolute top-3 left-3 flex gap-2">
                      <Badge className="bg-red-500/90 text-white border-none shadow-lg backdrop-blur-sm px-2">TB</Badge>
                    </div>
                    <div className="absolute top-3 right-3">
                      <Badge className="bg-black/60 text-white border-white/10 shadow-lg backdrop-blur-md px-2 font-mono">
                        {cand.similarity_score}% Match
                      </Badge>
                    </div>
                    
                    <div className="absolute bottom-4 left-4 right-4">
                      <div className="font-semibold text-white truncate text-sm">{cand.patient_name || cand.patient_id}</div>
                      <div className="text-xs text-zinc-400 mt-0.5">{cand.age} yrs • {cand.sex}</div>
                    </div>
                  </div>
                </div>
              ))}

              {/* Normal Cases */}
              {similarCases.normal_similar?.map((cand: any, idx: number) => (
                <div key={`n-${idx}`} className="snap-start shrink-0 w-[220px] group">
                  <div className="relative h-[220px] bg-black/60 rounded-2xl overflow-hidden border border-white/5 group-hover:border-teal-500/30 transition-all duration-300">
                    <img src={cand.original_image} alt="Cohort Case" className="w-full h-full object-cover opacity-70 group-hover:opacity-100 group-hover:scale-105 transition-all duration-700" />
                    <div className="absolute inset-0 bg-gradient-to-t from-black/90 via-black/20 to-transparent"></div>
                    
                    <div className="absolute top-3 left-3 flex gap-2">
                      <Badge className="bg-teal-500/90 text-white border-none shadow-lg backdrop-blur-sm px-2">Normal</Badge>
                    </div>
                    <div className="absolute top-3 right-3">
                      <Badge className="bg-black/60 text-white border-white/10 shadow-lg backdrop-blur-md px-2 font-mono">
                        {cand.similarity_score}% Match
                      </Badge>
                    </div>
                    
                    <div className="absolute bottom-4 left-4 right-4">
                      <div className="font-semibold text-white truncate text-sm">{cand.patient_name || cand.patient_id}</div>
                      <div className="text-xs text-zinc-400 mt-0.5">{cand.age} yrs • {cand.sex}</div>
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
