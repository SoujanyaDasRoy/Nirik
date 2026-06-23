"use client";

import { useState, useEffect } from "react";
import { getCookie } from "../hooks/usePrediction";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";

interface HistoryRecord {
  timestamp: string;
  confidence: number;
  is_tb: boolean;
  prediction: string;
  heatmap_b64: string;
  metadata: Record<string, string>;
  clinician_override: string | null;
}

interface LongitudinalTrackerProps {
  patientId: string;
  patientName: string;
  currentResult: {
    confidence: number;
    is_tb: boolean;
    prediction: string;
    heatmap_image: string;
    original_image: string;
    metadata: Record<string, string>;
  };
  onCompare?: (priorRecord: HistoryRecord) => void;
}

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "https://projectmantra-nirikshon-backend.hf.space";

export default function LongitudinalTracker({ patientId, patientName, currentResult, onCompare }: LongitudinalTrackerProps) {
  const [history, setHistory]     = useState<HistoryRecord[]>([]);
  const [isSaving, setIsSaving]   = useState(false);
  const [isSaved, setIsSaved]     = useState(false);
  const [selectedIdx, setSelectedIdx] = useState<number | null>(null);
  const [view, setView]           = useState<"chart" | "compare">("chart");

  const fetchHistory = async () => {
    try {
      const res  = await fetch(`${API_BASE}/patients/${patientId}/history`, {
        credentials: "include"
      });
      const data = await res.json();
      setHistory(data.records || []);
    } catch { /* silent */ }
  };

  useEffect(() => {
    fetchHistory();
    setIsSaved(false);
    setSelectedIdx(null);
  }, [patientId]); // eslint-disable-line react-hooks/exhaustive-deps

  const saveToHistory = async () => {
    setIsSaving(true);
    try {
      const token = getCookie("csrf_token") || "";
      await fetch(`${API_BASE}/patients/${patientId}/save`, {
        method: "POST",
        headers: { 
          "Content-Type": "application/json",
          "X-CSRF-Token": token
        },
        credentials: "include",
        body: JSON.stringify(currentResult),
      });
      setIsSaved(true);
      await fetchHistory();
    } catch { alert("Failed to save. Is the backend running?"); }
    finally { setIsSaving(false); }
  };

  // ── SVG Chart ──────────────────────────────────────────
  const W = 460, H = 140, PX = 40, PY = 16;
  const n = history.length;

  const renderChart = () => {
    if (n === 0) return (
      <div className="flex items-center justify-center h-[140px] text-sm text-muted-foreground">
        No records yet. Save this scan to begin tracking.
      </div>
    );
    const iW = W - PX * 2, iH = H - PY * 2;
    const pts = history.map((r, i) => ({
      x: PX + (n === 1 ? iW / 2 : (i / (n - 1)) * iW),
      y: PY + iH - r.confidence * iH,
      ...r,
    }));
    const polyline  = pts.map(p => `${p.x},${p.y}`).join(" ");
    const areaPath  = `M${pts[0].x} ${PY + iH} ` + pts.map(p => `L${p.x} ${p.y}`).join(" ") + ` L${pts[pts.length - 1].x} ${PY + iH}Z`;
    const threshY   = PY + iH - 0.5 * iH;

    return (
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ height: H }}>
        <defs>
          <linearGradient id="grad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#ef4444" stopOpacity={0.5} />
            <stop offset="100%" stopColor="#10b981" stopOpacity={0.05} />
          </linearGradient>
        </defs>
        {[0, 0.25, 0.5, 0.75, 1].map(pct => {
          const y = PY + iH - pct * iH;
          return (
            <g key={pct}>
              <line x1={PX} y1={y} x2={W - PX} y2={y} stroke="hsl(var(--border))" strokeWidth={1} />
              <text x={PX - 4} y={y + 4} textAnchor="end" fontSize={9} fill="hsl(var(--muted-foreground))">
                {Math.round(pct * 100)}%
              </text>
            </g>
          );
        })}
        <line x1={PX} y1={threshY} x2={W - PX} y2={threshY} stroke="#f59e0b" strokeWidth={1.5} strokeDasharray="4 3" />
        <text x={W - PX + 3} y={threshY + 4} fontSize={9} fill="#f59e0b">50%</text>
        <path d={areaPath} fill="url(#grad)" opacity={0.4} />
        <polyline points={polyline} fill="none" stroke="hsl(var(--primary))" strokeWidth={2.5} strokeLinejoin="round" />
        {pts.map((p, i) => (
          <g key={i} style={{ cursor: "pointer" }} onClick={() => setSelectedIdx(i)}>
            <circle cx={p.x} cy={p.y} r={selectedIdx === i ? 7 : 5}
              fill={p.is_tb ? "#ef4444" : "#10b981"}
              stroke="hsl(var(--background))" strokeWidth={2} />
            <text x={p.x} y={H - 2} textAnchor="middle" fontSize={8} fill="hsl(var(--muted-foreground))">
              {p.timestamp.slice(5, 10)}
            </text>
          </g>
        ))}
      </svg>
    );
  };

  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <div>
          <p className="font-bold text-sm">{patientName}</p>
          <p className="text-xs text-muted-foreground">
            ID: {patientId} · {history.length} scan{history.length !== 1 ? "s" : ""} on record
          </p>
        </div>
        <Button
          onClick={saveToHistory}
          disabled={isSaving || isSaved}
          variant={isSaved ? "outline" : "default"}
          size="sm"
          className={isSaved ? "border-primary text-primary" : ""}
        >
          {isSaving ? "Saving…" : isSaved ? "✓ Saved" : "Save to Record"}
        </Button>
      </div>

      {history.length > 0 && (
        <div className="flex gap-2">
          <Button size="sm" variant={view === "chart" ? "default" : "outline"} onClick={() => setView("chart")}>
            📈 Trend Chart
          </Button>
          <Button size="sm" variant={view === "compare" ? "default" : "outline"} onClick={() => setView("compare")}>
            ⚖ Compare Scans
          </Button>
        </div>
      )}

      {view === "chart" && (
        <Card>
          <CardHeader className="pb-2 pt-4 px-4">
            <CardTitle className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">
              TB Probability Over Time · click a point to inspect
            </CardTitle>
          </CardHeader>
          <CardContent className="px-4 pb-4">
            {renderChart()}
            {selectedIdx !== null && history[selectedIdx] && (
              <>
                <Separator className="my-3" />
                <div className="flex items-center gap-3 text-xs">
                  <span className="font-bold">{history[selectedIdx].timestamp.slice(0, 10)}</span>
                  <Badge variant={history[selectedIdx].is_tb ? "destructive" : "secondary"}
                    className={history[selectedIdx].is_tb ? "" : "bg-emerald-500/15 text-emerald-600 dark:text-emerald-400 border-emerald-500/30"}>
                    {history[selectedIdx].prediction}
                  </Badge>
                  <span className="text-muted-foreground">{(history[selectedIdx].confidence * 100).toFixed(1)}% confidence</span>
                  {history[selectedIdx].clinician_override && (
                    <span className="text-amber-500 text-[10px]">
                      Override: {history[selectedIdx].clinician_override}
                    </span>
                  )}
                </div>
              </>
            )}
          </CardContent>
        </Card>
      )}

      {view === "compare" && history.length > 0 && (
        <div className="overflow-x-auto">
          <div className="flex gap-3 pb-2" style={{ minWidth: "max-content" }}>
            {history.map((rec, i) => (
              <Card key={i} className="overflow-hidden bg-black border-border" style={{ width: 160 }}>
                {rec.heatmap_b64 ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img
                    src={rec.heatmap_b64.startsWith("data:") || rec.heatmap_b64.startsWith("blob:") || rec.heatmap_b64.startsWith("http") ? rec.heatmap_b64 : `data:image/png;base64,${rec.heatmap_b64}`}
                    alt={`Scan ${i + 1}`}
                    className="w-full h-[100px] object-cover"
                  />
                ) : (
                  <div className="w-full h-[100px] flex items-center justify-center text-muted-foreground text-xs">No image</div>
                )}
                <div className="p-2 bg-card">
                  <p className="text-[10px] font-bold text-foreground">{rec.timestamp.slice(0, 10)}</p>
                  <p className={`text-[10px] font-bold ${rec.is_tb ? "text-red-500" : "text-emerald-500"}`}>
                    {rec.prediction} · {(rec.confidence * 100).toFixed(0)}%
                  </p>
                  {onCompare && (
                    <Button 
                      variant="secondary" 
                      size="sm" 
                      className="w-full mt-2 h-6 text-[10px]" 
                      onClick={() => onCompare(rec)}
                    >
                      Compare
                    </Button>
                  )}
                </div>
              </Card>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
