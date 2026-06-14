"use client";

import { useState } from "react";
import { 
  Sliders, 
  Server, 
  Lock, 
  Database, 
  CheckCircle
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";

interface AuditLog {
  id: number;
  timestamp: string;
  user: string;
  event: string;
  details: string;
}

export function SettingsTab() {
  const [threshold, setThreshold] = useState(0.50);
  const [aeTitle, setAeTitle] = useState("APOLLO_DX");
  const [pacsPort, setPacsPort] = useState(104);
  const [pacsHost, setPacsHost] = useState("127.0.0.1");
  const [fhirUrl, setFhirUrl] = useState("http://localhost:5000/fhir");
  const [saved, setSaved] = useState(false);

  const [logs, setLogs] = useState<AuditLog[]>([
    { id: 1, timestamp: "2026-06-13 12:50:32", user: "Dr. Roy", event: "User login successful", details: "Role: Radiologist · Client IP: 127.0.0.1" },
    { id: 2, timestamp: "2026-06-13 12:51:15", user: "Dr. Roy", event: "PACS Node Status Checked", details: "Node: MONTGOMERY_PACS · status: online" },
    { id: 3, timestamp: "2026-06-13 12:52:01", user: "Dr. Roy", event: "Ingestion: radiograph loaded", details: "File: patient_0918.dcm · Modality: DX" },
    { id: 4, timestamp: "2026-06-13 12:52:45", user: "Dr. Roy", event: "Inference: executed prediction", details: "File: patient_0918.dcm · result: TB (96.7% confidence)" },
    { id: 5, timestamp: "2026-06-13 12:53:10", user: "Dr. Roy", event: "Clinician Feedback Saved", details: "Case: PX-9122 · override: none · notes recorded" },
  ]);

  const saveSettings = () => {
    setSaved(true);
    const now = new Date();
    const timeStr = now.toISOString().replace('T', ' ').slice(0, 19);
    
    // Append a log entry to audit configuration changes
    setLogs(prev => [
      {
        id: Date.now(),
        timestamp: timeStr,
        user: "Dr. Roy",
        event: "Configuration Update",
        details: `Threshold set to ${threshold.toFixed(2)} · PACS Node AE: ${aeTitle} · Port: ${pacsPort}`
      },
      ...prev
    ]);
    setTimeout(() => setSaved(false), 2000);
  };

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      {/* Settings Header */}
      <div className="space-y-1">
        <h2 className="text-2xl font-bold text-foreground font-sans">Clinical Suite Settings</h2>
        <p className="text-xs text-muted-foreground">Configure AI workstation calibration, DICOM connections, and HL7 FHIR endpoints.</p>
      </div>
      <Separator />

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left Column: Config Panels (col-span-7) */}
        <div className="lg:col-span-7 space-y-6">
          {/* Section 1: AI Calibration */}
          <Card className="border border-border bg-card rounded-xl shadow-none">
            <CardContent className="p-5 space-y-4">
              <div className="flex items-center gap-2">
                <Sliders className="w-4 h-4 text-primary" strokeWidth={1.5} />
                <h3 className="text-sm font-bold text-foreground">AI Workstation Calibration</h3>
              </div>
              <Separator />
              <div className="space-y-4">
                <div className="space-y-1.5">
                  <div className="flex justify-between text-xs font-mono">
                    <span className="text-muted-foreground font-sans">Decision Threshold</span>
                    <span className="font-bold text-primary">{threshold.toFixed(2)}</span>
                  </div>
                  {/* Accessible 44px touch target area */}
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
                  Lowering the threshold increases the model's sensitivity (Recall), flagging subtle lung consolidations at the expense of higher false-alarm rates. The balanced clinical baseline is set to <strong>0.50</strong>.
                </p>
              </div>
            </CardContent>
          </Card>

          {/* Section 2: PACS Node Config */}
          <Card className="border border-border bg-card rounded-xl shadow-none">
            <CardContent className="p-5 space-y-4">
              <div className="flex items-center gap-2">
                <Server className="w-4 h-4 text-primary" strokeWidth={1.5} />
                <h3 className="text-sm font-bold text-foreground">DICOM PACS Node Configuration</h3>
              </div>
              <Separator />
              <div className="grid grid-cols-2 gap-4 text-xs">
                <div className="space-y-1.5">
                  <span className="text-muted-foreground font-semibold uppercase text-[10px]">Calling AE Title</span>
                  <input
                    type="text"
                    value={aeTitle}
                    onChange={(e) => setAeTitle(e.target.value)}
                    className="w-full px-4 py-2 border border-border rounded-full bg-card text-foreground focus:ring-1 focus:ring-primary outline-none h-10 text-xs"
                  />
                </div>
                <div className="space-y-1.5">
                  <span className="text-muted-foreground font-semibold uppercase text-[10px]">PACS Port</span>
                  <input
                    type="number"
                    value={pacsPort}
                    onChange={(e) => setPacsPort(parseInt(e.target.value) || 0)}
                    className="w-full px-4 py-2 border border-border rounded-full bg-card text-foreground focus:ring-1 focus:ring-primary outline-none h-10 text-xs"
                  />
                </div>
                <div className="col-span-2 space-y-1.5">
                  <span className="text-muted-foreground font-semibold uppercase text-[10px]">PACS Server Host IP</span>
                  <input
                    type="text"
                    value={pacsHost}
                    onChange={(e) => setPacsHost(e.target.value)}
                    className="w-full px-4 py-2 border border-border rounded-full bg-card text-foreground focus:ring-1 focus:ring-primary outline-none h-10 text-xs"
                  />
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Section 3: FHIR Integration */}
          <Card className="border border-border bg-card rounded-xl shadow-none">
            <CardContent className="p-5 space-y-4">
              <div className="flex items-center gap-2">
                <Database className="w-4 h-4 text-primary" strokeWidth={1.5} />
                <h3 className="text-sm font-bold text-foreground">EHR Integration Endpoint (HL7 FHIR)</h3>
              </div>
              <Separator />
              <div className="space-y-1.5 text-xs">
                <span className="text-muted-foreground font-semibold uppercase text-[10px]">FHIR R4 Server Base URL</span>
                <input
                  type="text"
                  value={fhirUrl}
                  onChange={(e) => setFhirUrl(e.target.value)}
                  className="w-full px-4 py-2 border border-border rounded-full bg-card text-foreground focus:ring-1 focus:ring-primary outline-none h-10 text-xs"
                />
              </div>
            </CardContent>
          </Card>

          {/* Action buttons */}
          <div className="flex items-center justify-end gap-3 pt-4 border-t border-border">
            {saved && (
              <span className="text-xs text-emerald-500 font-bold flex items-center gap-1">
                <CheckCircle className="w-3.5 h-3.5" strokeWidth={1.5} /> Settings Saved
              </span>
            )}
            <Button size="sm" onClick={saveSettings} className="h-10 text-xs px-5 cursor-pointer rounded-full font-semibold">
              Save Changes
            </Button>
          </div>
        </div>

        {/* Right Column: HIPAA Activity Logs (col-span-5) */}
        <div className="lg:col-span-5 space-y-6">
          <Card className="border border-border bg-card rounded-xl shadow-none">
            <CardContent className="p-5 space-y-4">
              <div className="flex items-center gap-2">
                <Lock className="w-4 h-4 text-primary" strokeWidth={1.5} />
                <h3 className="text-sm font-bold text-foreground">System Transaction Logs</h3>
              </div>
              <Separator />
              <p className="text-[11px] text-muted-foreground leading-relaxed">
                Logged transaction audit trail for the active screening session. Updates live on configuration updates.
              </p>
              
              {/* Scrollable Audit Trail */}
              <div className="h-[432px] overflow-y-auto border border-border rounded-xl divide-y divide-border bg-muted/10 p-3 space-y-2">
                {logs.map((log) => (
                  <div key={log.id} className="p-2 text-[10px] space-y-1 bg-card border border-border rounded-lg hover:border-primary/20 transition-all">
                    <div className="flex justify-between items-center text-muted-foreground font-mono text-[9px]">
                      <span>{log.timestamp}</span>
                      <span className="px-1.5 py-0.5 rounded bg-muted text-foreground font-bold">{log.user}</span>
                    </div>
                    <p className="font-bold text-foreground leading-snug">{log.event}</p>
                    <p className="text-muted-foreground font-mono text-[9px] break-all leading-normal">{log.details}</p>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Academic Prototype Warning */}
          <div className="p-4 border border-amber-500/20 bg-amber-500/5 rounded-xl flex items-start gap-3 text-xs">
            <Sliders className="w-5 h-5 text-amber-500 flex-shrink-0" strokeWidth={1.5} />
            <div>
              <p className="font-bold text-amber-600 dark:text-amber-500">Academic Prototype Warning</p>
              <p className="text-muted-foreground mt-0.5 leading-relaxed">
                This software is an academic prototype built as a college final year project for clinical screening validation. It is not certified for diagnostic medical use.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
