"use client";

import { useState, useRef, useCallback, useEffect, MouseEvent as ReactMouseEvent } from "react";
import { ZoomIn, ZoomOut, Maximize2, Layers } from "lucide-react";
import { Button } from "@/components/ui/button";

export interface DicomViewerProps {
  imageBase64: string;
  heatmapBase64?: string;
  hasHeatmap?: boolean;
  label?: string;
  viewMode?: "original" | "heatmap" | string;
  heatmapOpacity?: number;
  setHeatmapOpacity?: (opacity: number) => void;
  // Kept for backward compatibility with ScreeningWorkstation
  [key: string]: any; 
}

export default function DicomViewer({
  imageBase64,
  heatmapBase64,
  hasHeatmap = false,
  label = "Nirikshon Viewport",
  viewMode = "original",
  heatmapOpacity = 0.6,
}: DicomViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  // View state
  const [zoom, setZoom] = useState(1.0);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [panStart, setPanStart] = useState({ x: 0, y: 0 });

  // Toggle state
  const [showHeatmap, setShowHeatmap] = useState(viewMode === "heatmap");

  useEffect(() => {
    setShowHeatmap(viewMode === "heatmap");
  }, [viewMode]);

  // Handle interaction
  const handleMouseDown = (e: ReactMouseEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
    setDragStart({ x: e.clientX, y: e.clientY });
    setPanStart({ ...pan });
  };

  const handleMouseMove = (e: ReactMouseEvent<HTMLDivElement>) => {
    if (!isDragging) return;
    const dx = e.clientX - dragStart.x;
    const dy = e.clientY - dragStart.y;
    setPan({
      x: panStart.x + dx,
      y: panStart.y + dy,
    });
  };

  const handleMouseUp = () => setIsDragging(false);

  const handleWheel = useCallback((e: WheelEvent) => {
    e.preventDefault();
    const zoomFactor = 0.1;
    const delta = e.deltaY < 0 ? zoomFactor : -zoomFactor;
    setZoom((z) => Math.min(Math.max(0.1, z + delta), 10.0));
  }, []);

  useEffect(() => {
    const container = containerRef.current;
    if (container) {
      container.addEventListener("wheel", handleWheel, { passive: false });
      return () => container.removeEventListener("wheel", handleWheel);
    }
  }, [handleWheel]);

  const handleReset = () => {
    setZoom(1.0);
    setPan({ x: 0, y: 0 });
  };

  const handleZoomIn = () => setZoom((z) => Math.min(10.0, z + 0.2));
  const handleZoomOut = () => setZoom((z) => Math.max(0.1, z - 0.2));

  return (
    <div className="relative w-full h-full flex flex-col bg-[#090909] rounded-[24px] border border-white/5 overflow-hidden">
      {/* Viewer Header */}
      <div className="absolute top-4 left-4 z-20 flex items-center gap-2">
        <div className="px-3 py-1 bg-black/60 backdrop-blur-md rounded-full border border-white/10 text-xs font-medium text-white/70">
          {label}
        </div>
      </div>

      {/* Main Image Stage */}
      <div 
        ref={containerRef}
        className={`relative flex-1 w-full h-full overflow-hidden flex items-center justify-center ${isDragging ? "cursor-grabbing" : "cursor-grab"}`}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      >
        <div 
          className="relative transition-transform duration-75 origin-center"
          style={{
            transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
            width: '100%',
            height: '100%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}
        >
          {/* Base Image */}
          {imageBase64 && (
            <img
              src={`data:image/jpeg;base64,${imageBase64}`}
              alt="Medical Scan"
              className="absolute max-w-full max-h-full object-contain pointer-events-none"
              style={{ width: "100%", height: "100%" }}
            />
          )}

          {/* Grad-CAM Heatmap Overlay */}
          {hasHeatmap && heatmapBase64 && (
            <img
              src={`data:image/jpeg;base64,${heatmapBase64}`}
              alt="AI Heatmap"
              className={`absolute max-w-full max-h-full object-contain mix-blend-screen pointer-events-none transition-opacity duration-300 ${showHeatmap ? 'opacity-100' : 'opacity-0'}`}
              style={{ width: "100%", height: "100%", opacity: showHeatmap ? heatmapOpacity : 0 }}
            />
          )}
        </div>
      </div>

      {/* Floating Toolbar */}
      <div className="absolute bottom-6 left-1/2 -translate-x-1/2 z-20 flex items-center gap-1 p-1.5 bg-[#141414]/90 backdrop-blur-xl border border-white/10 rounded-full shadow-2xl">
        <Button 
          variant="ghost" 
          size="icon" 
          onClick={handleZoomOut}
          className="rounded-full w-10 h-10 text-white/70 hover:text-white hover:bg-white/10"
        >
          <ZoomOut className="w-4 h-4" />
        </Button>
        <Button 
          variant="ghost" 
          size="icon" 
          onClick={handleReset}
          className="rounded-full w-10 h-10 text-white/70 hover:text-white hover:bg-white/10"
        >
          <Maximize2 className="w-4 h-4" />
        </Button>
        <Button 
          variant="ghost" 
          size="icon" 
          onClick={handleZoomIn}
          className="rounded-full w-10 h-10 text-white/70 hover:text-white hover:bg-white/10"
        >
          <ZoomIn className="w-4 h-4" />
        </Button>

        <div className="w-[1px] h-6 bg-white/10 mx-1" />

        <Button 
          variant="ghost" 
          size="icon" 
          onClick={() => setShowHeatmap(!showHeatmap)}
          disabled={!hasHeatmap}
          className={`rounded-full w-10 h-10 ${showHeatmap ? "text-[#0099ff] bg-[#0099ff]/10" : "text-white/70 hover:text-white hover:bg-white/10"} ${!hasHeatmap && "opacity-50"}`}
        >
          <Layers className="w-4 h-4" />
        </Button>
      </div>
    </div>
  );
}
