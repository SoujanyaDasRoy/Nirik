import React, { useEffect, useState } from 'react';
import { Card, CardContent } from './ui/card';
import { Loader2 } from 'lucide-react';

interface TsnePoint {
  x: number;
  y: number;
  dataset: string;
  label: string;
  patient_id: string;
}

export default function TsnePlot() {
  const [data, setData] = useState<TsnePoint[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000";
    fetch(`${API}/model/tsne`)
      .then(r => {
        if (!r.ok) throw new Error("Failed to load t-SNE data");
        return r.json();
      })
      .then(d => {
        setData(d);
        setLoading(false);
      })
      .catch(e => {
        setError(e.message);
        setLoading(false);
      });
  }, []);

  if (loading) {
    return (
      <Card className="border border-border/50 bg-card/40 backdrop-blur-xl h-64 flex items-center justify-center">
        <Loader2 className="w-8 h-8 animate-spin text-muted-foreground" />
      </Card>
    );
  }

  if (error) {
    return (
      <Card className="border border-destructive/20 bg-destructive/5 h-64 flex items-center justify-center p-4 text-center text-sm text-destructive font-semibold">
        {error}
      </Card>
    );
  }

  // Find bounds
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  data.forEach(p => {
    if (p.x < minX) minX = p.x;
    if (p.x > maxX) maxX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.y > maxY) maxY = p.y;
  });

  // Pad bounds
  const padX = (maxX - minX) * 0.1;
  const padY = (maxY - minY) * 0.1;
  minX -= padX; maxX += padX;
  minY -= padY; maxY += padY;

  const datasetColors: Record<string, string> = {
    "Montgomery": "#3b82f6", // blue-500
    "Shenzhen": "#10b981", // emerald-500
    "TB-Database": "#f59e0b", // amber-500
    "NIRT": "#8b5cf6", // violet-500
  };

  return (
    <Card className="border border-border/50 bg-card/40 backdrop-blur-xl rounded-2xl shadow-[inset_0_1px_1px_rgba(255,255,255,0.05)] overflow-hidden">
      <CardContent className="p-5 space-y-4">
        <div className="flex justify-between items-center border-b border-border/40 pb-3">
          <div>
            <h4 className="text-xs font-bold uppercase tracking-wider text-foreground">Feature Embedding Manifold</h4>
            <p className="text-[10px] text-muted-foreground mt-0.5">t-SNE projection of DenseNet-121 penultimate layer</p>
          </div>
          <div className="flex gap-2 text-[9px] font-bold uppercase">
            {Object.entries(datasetColors).map(([ds, color]) => (
              <div key={ds} className="flex items-center gap-1">
                <div className="w-2 h-2 rounded-full" style={{ backgroundColor: color }} />
                <span className="text-muted-foreground">{ds}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="relative w-full aspect-video bg-black/40 rounded-lg border border-white/5 overflow-hidden">
          <svg viewBox={`${minX} ${minY} ${maxX - minX} ${maxY - minY}`} className="w-full h-full" style={{ transform: "scaleY(-1)" }}>
            {data.map((p, i) => (
              <circle
                key={i}
                cx={p.x}
                cy={p.y}
                r={(maxX - minX) * 0.015} // scale radius based on viewBox width
                fill={datasetColors[p.dataset] || "#ffffff"}
                opacity={p.label === "Tuberculosis" ? 0.9 : 0.4}
                stroke={p.label === "Tuberculosis" ? "#ffffff" : "transparent"}
                strokeWidth={(maxX - minX) * 0.003}
              >
                <title>{`${p.patient_id} (${p.dataset})\nLabel: ${p.label}`}</title>
              </circle>
            ))}
          </svg>
        </div>
        
        <p className="text-[10px] text-muted-foreground italic mt-2 text-center">
          * Filled circles with white borders represent positive TB cases. Faded circles are Normal. Note clustering patterns by source dataset indicating possible domain shift.
        </p>
      </CardContent>
    </Card>
  );
}
