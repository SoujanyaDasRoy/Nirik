"use client";

import { Card } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { AnalysisResult } from "../hooks/useFileUpload";

interface AnalyticsTabProps {
  files: File[];
  results: AnalysisResult[];
}

export function AnalyticsTab({ files, results }: AnalyticsTabProps) {
  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Stats overview */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="stat-card p-5 bg-card border border-border rounded-lg shadow-none">
          <p className="text-[10px] text-muted-foreground uppercase font-bold tracking-wider">Session Total</p>
          <h4 className="text-3xl font-bold text-foreground mt-1 font-mono tnum-data">
            {results.filter(r => r.status === "success").length}
          </h4>
          <p className="text-xs text-muted-foreground mt-1">Processed radiographs</p>
        </div>
        <div className="stat-card p-5 bg-card border border-border rounded-lg shadow-none">
          <p className="text-[10px] text-muted-foreground uppercase font-bold tracking-wider">TB Detection Rate</p>
          <h4 className="text-3xl font-bold text-amber-600 dark:text-amber-500 mt-1 font-mono tnum-data">
            {(() => {
              const succ = results.filter(r => r.status === "success");
              if (succ.length === 0) return "0.0%";
              const tbs = succ.filter(r => r.clinician_override ? r.clinician_override === "Tuberculosis" : r.is_tb);
              return `${((tbs.length / succ.length) * 100).toFixed(1)}%`;
            })()}
          </h4>
          <p className="text-xs text-muted-foreground mt-1">Positive classification ratio</p>
        </div>
        <div className="stat-card p-5 bg-card border border-border rounded-lg shadow-none">
          <p className="text-[10px] text-muted-foreground uppercase font-bold tracking-wider">Override Rate</p>
          <h4 className="text-3xl font-bold text-primary mt-1 font-mono tnum-data">
            {(() => {
              const succ = results.filter(r => r.status === "success");
              if (succ.length === 0) return "0.0%";
              const overrides = succ.filter(r => r.clinician_override !== undefined && r.clinician_override !== null);
              return `${((overrides.length / succ.length) * 100).toFixed(1)}%`;
            })()}
          </h4>
          <p className="text-xs text-muted-foreground mt-1">Clinician audit interventions</p>
        </div>
      </div>

      {/* Graphic breakdown */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* AI Agreement Consensus Chart */}
        <Card className="p-5 space-y-4 border border-border bg-card rounded-lg shadow-none">
          <p className="text-xs font-bold text-muted-foreground uppercase tracking-wider">Clinician / AI Consensus</p>
          <div className="h-44 flex flex-col justify-center space-y-4">
            {(() => {
              const succ = results.filter(r => r.status === "success");
              const overrides = succ.filter(r => r.clinician_override !== undefined && r.clinician_override !== null).length;
              const agrees = succ.length - overrides;
              const agreePct = succ.length > 0 ? (agrees / succ.length) * 100 : 0;
              const overridePct = succ.length > 0 ? (overrides / succ.length) * 100 : 0;

              return (
                <div className="space-y-4">
                  <div className="space-y-1.5">
                    <div className="flex justify-between text-xs font-mono">
                      <span className="font-bold text-emerald-600 dark:text-emerald-500">Agreed Scans ({agrees})</span>
                      <span>{agreePct.toFixed(0)}%</span>
                    </div>
                    <Progress value={agreePct} className="h-2 bg-muted [&>div]:bg-emerald-500" />
                  </div>
                  <div className="space-y-1.5">
                    <div className="flex justify-between text-xs font-mono">
                      <span className="font-bold text-amber-600 dark:text-amber-500">Overridden Scans ({overrides})</span>
                      <span>{overridePct.toFixed(0)}%</span>
                    </div>
                    <Progress value={overridePct} className="h-2 bg-muted [&>div]:bg-amber-500" />
                  </div>
                </div>
              );
            })()}
          </div>
        </Card>

        {/* Modality distribution */}
        <Card className="p-5 space-y-4 border border-border bg-card rounded-lg shadow-none">
          <p className="text-xs font-bold text-muted-foreground uppercase tracking-wider">Ingested Formats</p>
          <div className="h-44 flex flex-col justify-center space-y-4">
            {(() => {
              const dcmCount = files.filter(f => f.name.toLowerCase().endsWith(".dcm") || f.name.toLowerCase().endsWith(".dicom")).length;
              const stdCount = files.length - dcmCount;
              const dcmPct = files.length > 0 ? (dcmCount / files.length) * 100 : 0;
              const stdPct = files.length > 0 ? (stdCount / files.length) * 100 : 0;

              return (
                <div className="space-y-4">
                  <div className="space-y-1.5">
                    <div className="flex justify-between text-xs font-mono">
                      <span className="font-bold text-primary">DICOM Radiographs ({dcmCount})</span>
                      <span>{dcmPct.toFixed(0)}%</span>
                    </div>
                    <Progress value={dcmPct} className="h-2 bg-muted [&>div]:bg-primary" />
                  </div>
                  <div className="space-y-1.5">
                    <div className="flex justify-between text-xs font-mono">
                      <span className="font-bold text-muted-foreground">Standard PNG/JPG ({stdCount})</span>
                      <span>{stdPct.toFixed(0)}%</span>
                    </div>
                    <Progress value={stdPct} className="h-2 bg-muted [&>div]:bg-muted-foreground" />
                  </div>
                </div>
              );
            })()}
          </div>
        </Card>
      </div>
    </div>
  );
}
