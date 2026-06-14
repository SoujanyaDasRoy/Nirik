"use client";

import { useState, useRef, useEffect, MouseEvent, useCallback } from "react";
import { Badge } from "@/components/ui/badge";

export type LungZone = "Apical" | "Mid-zone" | "Basal" | "Pleural";

export const ZONE_COLORS: Record<LungZone, string> = {
  Apical: "#ef4444", 
  "Mid-zone": "#f97316", 
  Basal: "#eab308", 
  Pleural: "#a855f7",
};

export interface Box {
  x: number;
  y: number;
  w: number;
  h: number;
  zone: LungZone;
}

interface AnnotationCanvasProps {
  boxes: Box[];
  setBoxes: React.Dispatch<React.SetStateAction<Box[]>>;
  activeZone: LungZone;
  annotateMode: boolean;
  imgRef: React.RefObject<HTMLImageElement | null>;
  canvasRef: React.RefObject<HTMLCanvasElement | null>;
}

export default function AnnotationCanvas({
  boxes,
  setBoxes,
  activeZone,
  annotateMode,
  imgRef,
  canvasRef
}: AnnotationCanvasProps) {
  const [drawing, setDrawing] = useState<{ x: number; y: number } | null>(null);
  const [currentBox, setCurrentBox] = useState<{ x: number; y: number; w: number; h: number } | null>(null);

  const redraw = useCallback(() => {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (!canvas || !img) return;

    canvas.width = img.offsetWidth || 300;
    canvas.height = img.offsetHeight || 300;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw saved annotations
    boxes.forEach(box => {
      const color = ZONE_COLORS[box.zone];
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.5;
      ctx.setLineDash([]);
      ctx.strokeRect(box.x, box.y, box.w, box.h);
      ctx.fillStyle = color + "22";
      ctx.fillRect(box.x, box.y, box.w, box.h);
      ctx.font = "bold 10px Inter, sans-serif";
      ctx.fillStyle = color;
      ctx.fillText(box.zone, box.x + 4, box.y + 12);
    });

    // Draw active drawing box outline
    if (currentBox) {
      ctx.strokeStyle = ZONE_COLORS[activeZone];
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 3]);
      ctx.strokeRect(currentBox.x, currentBox.y, currentBox.w, currentBox.h);
    }
  }, [boxes, currentBox, activeZone, canvasRef, imgRef]);

  useEffect(() => {
    redraw();
    const handleResize = () => redraw();
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, [boxes, currentBox, activeZone, redraw]);

  const getPos = (e: MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };
    const rect = canvas.getBoundingClientRect();
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  };

  const handleMouseDown = (e: MouseEvent<HTMLCanvasElement>) => {
    if (!annotateMode) return;
    const p = getPos(e);
    setDrawing(p);
  };

  const handleMouseMove = (e: MouseEvent<HTMLCanvasElement>) => {
    if (!annotateMode || !drawing) return;
    const p = getPos(e);
    setCurrentBox({
      x: Math.min(p.x, drawing.x),
      y: Math.min(p.y, drawing.y),
      w: Math.abs(p.x - drawing.x),
      h: Math.abs(p.y - drawing.y)
    });
  };

  const handleMouseUp = (e: MouseEvent<HTMLCanvasElement>) => {
    if (!annotateMode || !drawing) return;
    const p = getPos(e);
    const box: Box = {
      x: Math.min(p.x, drawing.x),
      y: Math.min(p.y, drawing.y),
      w: Math.abs(p.x - drawing.x),
      h: Math.abs(p.y - drawing.y),
      zone: activeZone
    };
    if (box.w > 10 && box.h > 10) {
      setBoxes(prev => [...prev, box]);
    }
    setDrawing(null);
    setCurrentBox(null);
  };

  // Expose static helper function to save/export base64 annotated image overlays
  // Reused inside the exporter to stamp vector bounding boxes onto the source dimensions
  return (
    <div className="absolute inset-0 z-20">
      <canvas
        ref={canvasRef}
        className="w-full h-full block absolute inset-0"
        style={{
          cursor: annotateMode ? "crosshair" : "default",
          pointerEvents: annotateMode ? "all" : "none"
        }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
      />
      {annotateMode && (
        <div className="absolute bottom-2 inset-x-0 flex justify-center pointer-events-none z-30">
          <Badge className="bg-amber-500 hover:bg-amber-500 text-black text-[9px] font-bold shadow-md">
            ✏️ Bounding Box Tool: Click &amp; drag over {activeZone} Zone
          </Badge>
        </div>
      )}
    </div>
  );
}
