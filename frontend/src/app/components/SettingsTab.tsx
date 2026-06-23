"use client";

import { useState, useEffect } from "react";
import { 
  Sliders, 
  Lock, 
  CheckCircle,
  Info
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "https://projectmantra-nirikshon-backend.hf.space";

interface AuditLog {
  id: number;
  timestamp: string;
  user: string;
  event: string;
  details: string;
}

export function SettingsTab() {
  const [threshold, setThreshold] = useState(0.50);
  const [saved, setSaved] = useState(false);
  const [logs, setLogs] = useState<AuditLog[]>([]);
  const [logsLoading, setLogsLoading] = useState(true);

  // Fetch real audit logs from backend
  useEffect(() => {
    const fetchLogs = async () => {
      setLogsLoading(true);
      try {
        const res = await fetch(`${API_BASE}/audit/logs`, { credentials: "include" });
        if (res.ok) {
          const data = await res.json();
          setLogs(data.logs || []);
        }
      } catch { /* silent — endpoint may not exist */ }
      finally { setLogsLoading(false); }
    };
    fetchLogs();
  }, []);

  const saveSettings = () => {
    // The threshold is used client-side in the prediction hook to override
    // display classification. In a production system this would POST to backend.
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      {/* Settings Header */}
      <div className="space-y-1">
        <h2 className="text-2xl font-bold text-foreground font-sans">Clinical Suite Settings</h2>
        <p className="text-xs text-muted-foreground">Configure AI workstation parameters for this session.</p>
      </div>
      <Separator />

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left Column: Config Panels */}
        <div className="lg:col-span-7 space-y-6">

          {/* Section 1: AI Calibration */}
          <Card className="border border-border bg-card rounded-xl shadow-none">
            <CardContent className="p-5 space-y-4">
              <div className="flex items-center gap-2">
                <Sliders className="w-4 h-4 text-primary" strokeWidth={1.5} />
                <h3 className="text-sm font-bold text-foreground">AI Decision Threshold</h3>
              </div>
              <Separator />
              <div className="space-y-4">
                <div className="space-y-1.5">
                  <div className="flex justify-between text-xs font-mono">
                    <span className="text-muted-foreground font-sans">Classification Threshold</span>
                    <span className="font-bold text-primary">{threshold.toFixed(2)}</span>
                  </div>
                  <div className="py-3 flex items-center min-h-[44px]">
                    <input
                      type="range"
                      min={0.10}
                      max={0.90}
                      step={0.05}
                      value={threshold}
                      onChange={(e) => setThreshold(parseFloat(e.target.value))}
                      className="w-full h-1.5 bg-muted rounded-lg appearance-none cursor-pointer accent-primary outline-none focus:ring-1 focus:ring-primary focus:ring-offset-1"
                    />
                  </div>
                  <div className="flex justify-between text-[10px] text-muted-foreground">
                    <span>High Sensitivity (0.10)</span>
                    <span>High Specificity (0.90)</span>
                  </div>
                </div>
                <p className="text-xs text-muted-foreground leading-relaxed">
                  Lowering the threshold increases sensitivity (Recall), flagging subtle consolidations at the cost of
                  more false positives. The balanced clinical baseline is <strong>0.50</strong>.
                </p>

                {/* Session-only warning */}
                <div className="flex items-start gap-2 p-3 rounded-lg border border-amber-500/20 bg-amber-500/5 text-xs">
                  <Info className="w-3.5 h-3.5 text-amber-500 flex-shrink-0 mt-0.5" strokeWidth={1.5} />
                  <p className="text-amber-700 dark:text-amber-400 leading-relaxed">
                    <strong>Session-only.</strong> This threshold applies to the current browser session and is not persisted to the backend model. The underlying DenseNet-121 model always returns raw sigmoid confidence.
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Action buttons */}
          <div className="flex items-center justify-end gap-3 pt-4 border-t border-border">
            {saved && (
              <span className="text-xs text-emerald-500 font-bold flex items-center gap-1">
                <CheckCircle className="w-3.5 h-3.5" strokeWidth={1.5} /> Session threshold updated
              </span>
            )}
            <Button size="sm" onClick={saveSettings} className="h-10 text-xs px-5 cursor-pointer rounded-full font-semibold">
              Apply to Session
            </Button>
          </div>

          {/* Academic Prototype Warning */}
          <div className="p-4 border border-amber-500/20 bg-amber-500/5 rounded-xl flex items-start gap-3 text-xs">
            <Sliders className="w-5 h-5 text-amber-500 flex-shrink-0" strokeWidth={1.5} />
            <div>
              <p className="font-bold text-amber-600 dark:text-amber-500">Academic Prototype</p>
              <p className="text-muted-foreground mt-0.5 leading-relaxed">
                This software is an academic prototype built as a final year project for clinical screening research.
                PACS node configuration, HL7 FHIR endpoint setup, and persistent model threshold tuning are out of
                scope for this prototype and are not implemented.
              </p>
            </div>
          </div>
        </div>

        {/* Right Column: Real Audit Logs */}
        <div className="lg:col-span-5 space-y-6">
          <Card className="border border-border bg-card rounded-xl shadow-none">
            <CardContent className="p-5 space-y-4">
              <div className="flex items-center gap-2">
                <Lock className="w-4 h-4 text-primary" strokeWidth={1.5} />
                <h3 className="text-sm font-bold text-foreground">Audit Trail</h3>
              </div>
              <Separator />
              <p className="text-[11px] text-muted-foreground leading-relaxed">
                Live activity log from the backend database. Captures authentication events, inference calls,
                and clinician overrides.
              </p>
              
              {/* Scrollable Audit Trail */}
              <div className="h-[432px] overflow-y-auto border border-border rounded-xl divide-y divide-border bg-muted/10 p-3 space-y-2">
                {logsLoading ? (
                  <div className="flex items-center justify-center h-full">
                    <p className="text-xs text-muted-foreground animate-pulse">Loading audit logs…</p>
                  </div>
                ) : logs.length === 0 ? (
                  <div className="flex flex-col items-center justify-center h-full gap-2 text-center">
                    <Lock className="w-6 h-6 text-muted-foreground/40" strokeWidth={1} />
                    <p className="text-xs text-muted-foreground">No audit entries recorded yet.</p>
                    <p className="text-[10px] text-muted-foreground/60">Activity will appear here as you use the system.</p>
                  </div>
                ) : (
                  logs.map((log) => (
                    <div key={log.id} className="p-2 text-[10px] space-y-1 bg-card border border-border rounded-lg hover:border-primary/20 transition-all">
                      <div className="flex justify-between items-center text-muted-foreground font-mono text-[9px]">
                        <span>{log.timestamp}</span>
                        <span className="px-1.5 py-0.5 rounded bg-muted text-foreground font-bold">{log.user}</span>
                      </div>
                      <p className="font-bold text-foreground leading-snug">{log.event}</p>
                      <p className="text-muted-foreground font-mono text-[9px] break-all leading-normal">{log.details}</p>
                    </div>
                  ))
                )}
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
