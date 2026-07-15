"use client";

import { useState, useRef, useCallback, useEffect, MouseEvent as ReactMouseEvent } from "react";
import { ZoomIn, ZoomOut, Maximize2, Layers, RotateCcw } from "lucide-react";

export interface DicomViewerProps {
  imageBase64: string;
  heatmapBase64?: string;
  hasHeatmap?: boolean;
  label?: string;
  viewMode?: "original" | "heatmap" | string;
  heatmapOpacity?: number;
  setHeatmapOpacity?: (opacity: number) => void;
  [key: string]: any;
}

/** Normalises a base64 string or data-URL into a usable <img> src */
function toSrc(raw: string, fallbackMime = "image/png"): string {
  if (!raw) return "";
  if (raw.startsWith("data:")) return raw;
  if (raw.startsWith("blob:") || raw.startsWith("http")) return raw;
  return `data:${fallbackMime};base64,${raw}`;
}

export default function DicomViewer({
  imageBase64,
  heatmapBase64,
  hasHeatmap = false,
  label = "Nirikshon Viewport",
  viewMode = "original",
  heatmapOpacity = 0.55,
}: DicomViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  const [zoom, setZoom] = useState(1.0);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [panStart, setPanStart] = useState({ x: 0, y: 0 });
  const [showHeatmap, setShowHeatmap] = useState(viewMode === "heatmap");

  // Reset view whenever a new image arrives
  useEffect(() => {
    setZoom(1.0);
    setPan({ x: 0, y: 0 });
  }, [imageBase64]);

  useEffect(() => {
    setShowHeatmap(viewMode === "heatmap");
  }, [viewMode]);

  // Pan handlers
  const handleMouseDown = (e: ReactMouseEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
    setDragStart({ x: e.clientX, y: e.clientY });
    setPanStart({ ...pan });
  };
  const handleMouseMove = (e: ReactMouseEvent<HTMLDivElement>) => {
    if (!isDragging) return;
    setPan({ x: panStart.x + e.clientX - dragStart.x, y: panStart.y + e.clientY - dragStart.y });
  };
  const handleMouseUp = () => setIsDragging(false);

  // Scroll-to-zoom
  const handleWheel = useCallback((e: WheelEvent) => {
    e.preventDefault();
    setZoom((z) => Math.min(Math.max(0.2, z + (e.deltaY < 0 ? 0.1 : -0.1)), 8.0));
  }, []);
  useEffect(() => {
    const el = containerRef.current;
    if (el) {
      el.addEventListener("wheel", handleWheel, { passive: false });
      return () => el.removeEventListener("wheel", handleWheel);
    }
  }, [handleWheel]);

  const handleReset = () => { setZoom(1.0); setPan({ x: 0, y: 0 }); };
  const handleZoomIn = () => setZoom((z) => Math.min(8.0, z + 0.25));
  const handleZoomOut = () => setZoom((z) => Math.max(0.2, z - 0.25));

  const originalSrc = toSrc(imageBase64);
  const heatmapSrc  = toSrc(heatmapBase64 ?? "");

  return (
    <div className="relative w-full h-full flex flex-col bg-black rounded-[20px] overflow-hidden select-none">

      {/* Label badge */}
      <div className="absolute top-3 left-3 z-20 pointer-events-none">
        <span className="px-2.5 py-1 bg-black/70 backdrop-blur-md rounded-full border border-white/10 text-[11px] font-medium text-white/60">
          {label}
        </span>
      </div>

      {/* Zoom indicator */}
      {zoom !== 1.0 && (
        <div className="absolute top-3 right-3 z-20 pointer-events-none">
          <span className="px-2 py-1 bg-black/70 backdrop-blur-md rounded-full border border-white/10 text-[11px] font-mono text-white/50">
            {(zoom * 100).toFixed(0)}%
          </span>
        </div>
      )}

      {/* Image stage — fills the entire card */}
      <div
        ref={containerRef}
        className={`flex-1 w-full h-full relative overflow-hidden ${isDragging ? "cursor-grabbing" : "cursor-grab"}`}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      >
        {/* No image placeholder */}
        {!originalSrc && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 text-white/20">
            <svg className="w-12 h-12" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
            </svg>
            <p className="text-[13px]">Upload an X-ray to view it here</p>
          </div>
        )}

        {/* Transform wrapper */}
        <div
          className="absolute inset-0 flex items-center justify-center"
          style={{ transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`, transformOrigin: "center center" }}
        >
          {/* Base X-ray — fills the stage */}
          {originalSrc && (
            <img
              src={originalSrc}
              alt="Chest X-Ray"
              className="w-full h-full object-contain pointer-events-none"
              draggable={false}
            />
          )}

          {/* Grad-CAM overlay */}
          {hasHeatmap && heatmapSrc && (
            <img
              src={heatmapSrc}
              alt="Grad-CAM Heatmap"
              className="absolute inset-0 w-full h-full object-contain pointer-events-none mix-blend-screen transition-opacity duration-300"
              style={{ opacity: showHeatmap ? heatmapOpacity : 0 }}
              draggable={false}
            />
          )}
        </div>
      </div>

      {/* Floating toolbar — always at bottom center */}
      <div className="absolute bottom-4 left-1/2 -translate-x-1/2 z-20 flex items-center gap-1 px-2 py-1.5 bg-black/80 backdrop-blur-xl border border-white/10 rounded-full shadow-2xl">
        {/* Zoom Out */}
        <button
          onClick={handleZoomOut}
          title="Zoom out (or scroll down)"
          className="w-9 h-9 rounded-full flex items-center justify-center text-white/60 hover:text-white hover:bg-white/10 transition-colors"
        >
          <ZoomOut className="w-4 h-4" />
        </button>

        {/* Reset */}
        <button
          onClick={handleReset}
          title="Reset view"
          className="w-9 h-9 rounded-full flex items-center justify-center text-white/60 hover:text-white hover:bg-white/10 transition-colors"
        >
          <RotateCcw className="w-3.5 h-3.5" />
        </button>

        {/* Zoom In */}
        <button
          onClick={handleZoomIn}
          title="Zoom in (or scroll up)"
          className="w-9 h-9 rounded-full flex items-center justify-center text-white/60 hover:text-white hover:bg-white/10 transition-colors"
        >
          <ZoomIn className="w-4 h-4" />
        </button>

        {/* Divider + heatmap toggle — only shown on the heatmap viewer */}
        {hasHeatmap && (
          <>
            <div className="w-px h-5 bg-white/10 mx-0.5" />
            <button
              onClick={() => setShowHeatmap(!showHeatmap)}
              title={showHeatmap ? "Hide heatmap overlay" : "Show heatmap overlay"}
              className={`w-9 h-9 rounded-full flex items-center justify-center transition-colors ${
                showHeatmap
                  ? "text-[#0099ff] bg-[#0099ff]/15 hover:bg-[#0099ff]/25"
                  : "text-white/60 hover:text-white hover:bg-white/10"
              }`}
            >
              <Layers className="w-4 h-4" />
            </button>
          </>
        )}
      </div>
    </div>
  );
}
