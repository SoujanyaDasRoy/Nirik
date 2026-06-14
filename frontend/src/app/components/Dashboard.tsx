"use client";

import { useState, useEffect } from "react";
import { 
  Activity, 
  TrendingUp, 
  Clock, 
  CheckCircle, 
  ChevronRight, 
  FolderOpen, 
  Users, 
  AlertTriangle,
  Download
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000";

interface DashboardStats {
  total_cases: number;
  tb_positive_cases: number;
  pending_reviews: number;
  completed_reviews: number;
  disease_distribution: {
    Tuberculosis: number;
    Normal: number;
  };
  confidence_distribution: {
    "0-20%": number;
    "20-40%": number;
    "40-60%": number;
    "60-80%": number;
    "80-100%": number;
  };
  model_performance: {
    date: string;
    accuracy: number;
  }[];
  reviewer_agreement_rate: number;
}

interface DashboardProps {
  onNavigate: (view: "landing" | "diagnose" | "settings" | "patients") => void;
  onOpenWorkbench: () => void;
  hasFiles: boolean;
}

export function Dashboard({ onNavigate, onOpenWorkbench, hasFiles }: DashboardProps) {
  const [stats, setStats] = useState<DashboardStats>({
    total_cases: 0,
    tb_positive_cases: 0,
    pending_reviews: 0,
    completed_reviews: 0,
    disease_distribution: { Tuberculosis: 0, Normal: 0 },
    confidence_distribution: { "0-20%": 0, "20-40%": 0, "40-60%": 0, "60-80%": 0, "80-100%": 0 },
    model_performance: [],
    reviewer_agreement_rate: 100.0
  });
  const [loading, setLoading] = useState(true);

  const fetchStats = async () => {
    try {
      const response = await fetch(`${API_BASE}/dashboard/stats`, {
        credentials: "include"
      });
      if (response.ok) {
        const data = await response.json();
        setStats(data);
      }
    } catch (err) {
      console.error("Failed to load dashboard statistics:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchStats();
    // Poll stats every 30 seconds
    const interval = setInterval(fetchStats, 30000);
    return () => clearInterval(interval);
  }, []);

  const downloadResearchExport = async (format: "json" | "csv") => {
    try {
      const response = await fetch(`${API_BASE}/export/research?format=${format}`, {
        credentials: "include"
      });
      if (response.ok) {
        if (format === "csv") {
          const blob = await response.blob();
          const url = window.URL.createObjectURL(blob);
          const a = document.createElement("a");
          a.href = url;
          a.download = "Nirikshon_research_export.csv";
          document.body.appendChild(a);
          a.click();
          a.remove();
        } else {
          const data = await response.json();
          const jsonString = `data:text/json;charset=utf-8,${encodeURIComponent(
            JSON.stringify(data, null, 2)
          )}`;
          const a = document.createElement("a");
          a.href = jsonString;
          a.download = "Nirikshon_research_export.json";
          document.body.appendChild(a);
          a.click();
          a.remove();
        }
      } else {
        alert("Failed to export research data");
      }
    } catch (err) {
      console.error("Export error:", err);
      alert("Error occurred exporting data");
    }
  };

  // 1. Disease Distribution Donut
  const renderDonutChart = () => {
    const tbCount = stats?.disease_distribution?.Tuberculosis || 0;
    const normalCount = stats?.disease_distribution?.Normal || 0;
    const totalDist = tbCount + normalCount;
    const tbPct = totalDist > 0 ? (tbCount / totalDist) * 100 : 0;
    const normalPct = totalDist > 0 ? (normalCount / totalDist) * 100 : 0;
    
    const r = 40;
    const C = 2 * Math.PI * r;
    const tbDash = (tbPct / 100) * C;
    const normalDash = (normalPct / 100) * C;

    return (
      <svg width="120" height="120" viewBox="0 0 100 100" className="mx-auto select-none">
        <circle cx="50" cy="50" r={r} fill="transparent" stroke="#f1f5f9" strokeWidth="8" />
        <circle
          cx="50"
          cy="50"
          r={r}
          fill="transparent"
          stroke="#10b981"
          strokeWidth="8"
          strokeDasharray={`${normalDash} ${C}`}
          strokeDashoffset="0"
          transform="rotate(-90 50 50)"
        />
        <circle
          cx="50"
          cy="50"
          r={r}
          fill="transparent"
          stroke="#f59e0b"
          strokeWidth="8"
          strokeDasharray={`${tbDash} ${C}`}
          strokeDashoffset={-normalDash}
          transform="rotate(-90 50 50)"
        />
        <text x="50" y="47" textAnchor="middle" dominantBaseline="middle" className="fill-foreground font-bold text-xs font-sans">
          {totalDist}
        </text>
        <text x="50" y="58" textAnchor="middle" dominantBaseline="middle" className="fill-muted-foreground text-[6px] uppercase tracking-wider font-semibold">
          Total Cases
        </text>
      </svg>
    );
  };

  // 2. Confidence Distribution Bar Chart
  const renderConfidenceBarChart = () => {
    const buckets = Object.entries(stats?.confidence_distribution || {});
    const maxCount = Math.max(...buckets.map(([_, v]) => v as number), 1);
    const width = 200, height = 120;
    const paddingX = 24, paddingY = 16;
    const chartWidth = width - paddingX * 2;
    const chartHeight = height - paddingY * 2;

    return (
      <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} className="mx-auto select-none">
        {buckets.map(([label, val], idx) => {
          const barWidth = 18;
          const gap = (chartWidth - barWidth * 5) / 4;
          const x = paddingX + idx * (barWidth + gap);
          const barHeight = (val / maxCount) * chartHeight;
          const y = paddingY + chartHeight - barHeight;
          return (
            <g key={idx} className="group">
              <rect
                x={x}
                y={y}
                width={barWidth}
                height={barHeight}
                fill="#0f766e"
                rx="3"
                className="transition-all duration-300 hover:fill-emerald-500"
              />
              <text x={x + barWidth / 2} y={y - 4} textAnchor="middle" className="fill-foreground text-[8px] font-bold">
                {val}
              </text>
              <text x={x + barWidth / 2} y={paddingY + chartHeight + 12} textAnchor="middle" className="fill-muted-foreground text-[6px] font-semibold">
                {label}
              </text>
            </g>
          );
        })}
      </svg>
    );
  };

  // 3. Performance Line Chart
  const renderPerformanceLineChart = () => {
    const perf = stats?.model_performance || [];
    const minAcc = 85;
    const maxAcc = 100;
    const w = 200, h = 120;
    const px = 24, py = 16;
    const cW = w - px * 2;
    const cH = h - py * 2;
    
    const pts = perf.map((p, i) => {
      const x = px + (perf.length <= 1 ? cW / 2 : (i / (perf.length - 1)) * cW);
      const normY = (p.accuracy - minAcc) / (maxAcc - minAcc);
      const y = py + cH - normY * cH;
      return { x, y, ...p };
    });
    
    const polylinePath = pts.map(p => `${p.x},${p.y}`).join(" ");

    return (
      <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} className="mx-auto select-none">
        <line x1={px} y1={py} x2={px + cW} y2={py} stroke="#f1f5f9" strokeDasharray="3 3" />
        <line x1={px} y1={py + cH / 2} x2={px + cW} y2={py + cH / 2} stroke="#f1f5f9" strokeDasharray="3 3" />
        <line x1={px} y1={py + cH} x2={px + cW} y2={py + cH} stroke="#e2e8f0" />
        {pts.length > 0 && (
          <polyline
            fill="none"
            stroke="#0f766e"
            strokeWidth="2.5"
            points={polylinePath}
          />
        )}
        {pts.map((p, idx) => (
          <g key={idx}>
            <circle cx={p.x} cy={p.y} r="3" className="fill-background stroke-primary stroke-[2px] cursor-pointer" />
            <text x={p.x} y={p.y - 6} textAnchor="middle" className="fill-foreground text-[8px] font-bold">
              {p.accuracy}%
            </text>
            <text x={p.x} y={py + cH + 12} textAnchor="middle" className="fill-muted-foreground text-[6px] font-semibold">
              {p.date}
            </text>
          </g>
        ))}
      </svg>
    );
  };

  // 4. Agreement Gauge
  const renderAgreementGauge = () => {
    const agreement = stats?.reviewer_agreement_rate || 0;
    const gR = 40;
    const gC = 2 * Math.PI * gR;
    const semiC = gC / 2;
    const filledDash = (agreement / 100) * semiC;

    return (
      <svg width="120" height="90" viewBox="0 0 100 60" className="mx-auto select-none">
        <circle
          cx="50"
          cy="50"
          r={gR}
          fill="transparent"
          stroke="#f1f5f9"
          strokeWidth="8"
          strokeDasharray={`${semiC} ${gC}`}
          transform="rotate(-180 50 50)"
        />
        <circle
          cx="50"
          cy="50"
          r={gR}
          fill="transparent"
          stroke="#0f766e"
          strokeWidth="8"
          strokeDasharray={`${filledDash} ${gC}`}
          transform="rotate(-180 50 50)"
          className="transition-all duration-500"
        />
        <text x="50" y="42" textAnchor="middle" className="fill-foreground font-bold text-xs font-sans">
          {agreement}%
        </text>
        <text x="50" y="52" textAnchor="middle" className="fill-muted-foreground text-[6px] uppercase tracking-wider font-semibold">
          Agreement Rate
        </text>
      </svg>
    );
  };

  const tbCount = stats?.disease_distribution?.Tuberculosis || 0;
  const normalCount = stats?.disease_distribution?.Normal || 0;

  return (
    <div className="max-w-5xl mx-auto space-y-8 animate-fadein">
      {/* Welcome & Overview Banner */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
        <div className="space-y-1">
          <h2 className="text-2xl font-bold text-foreground font-sans tracking-tight">Clinical Workstation Dashboard</h2>
          <p className="text-xs text-muted-foreground">Real-time cohort monitoring, diagnostic metrics, and verification backlog stats.</p>
        </div>
        <div className="flex gap-2">
          <Button 
            onClick={() => onNavigate("patients")}
            variant="outline"
            className="rounded-full text-xs font-semibold h-9 gap-1.5 cursor-pointer"
          >
            <Users className="w-3.5 h-3.5" /> Manage Patients
          </Button>
          <Button 
            onClick={hasFiles ? onOpenWorkbench : () => onNavigate("landing")}
            className="rounded-full text-xs font-semibold h-9 gap-1.5 cursor-pointer"
          >
            <Activity className="w-3.5 h-3.5" /> 
            {hasFiles ? "Open Active Workbench" : "Intake New Scan"}
            <ChevronRight className="w-3.5 h-3.5" />
          </Button>
        </div>
      </div>
      <Separator />

      {/* Stats Cards Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {[
          { 
            label: "Total Studies Ingestion", 
            value: stats?.total_cases || 0, 
            icon: <FolderOpen className="w-4 h-4 text-primary" strokeWidth={1.5} />,
            desc: "Historical cases in PACS node"
          },
          { 
            label: "TB Positive Index", 
            value: stats?.tb_positive_cases || 0, 
            icon: <AlertTriangle className="w-4 h-4 text-amber-500" strokeWidth={1.5} />,
            desc: "AI predicted or review confirmed",
            alert: (stats?.tb_positive_cases || 0) > 0
          },
          { 
            label: "Pending Adjudication", 
            value: stats?.pending_reviews || 0, 
            icon: <Clock className="w-4 h-4 text-blue-500" strokeWidth={1.5} />,
            desc: "Reviews awaiting signature"
          },
          { 
            label: "Finalized Reviews", 
            value: stats?.completed_reviews || 0, 
            icon: <CheckCircle className="w-4 h-4 text-emerald-500" strokeWidth={1.5} />,
            desc: "Reports locked & archived"
          }
        ].map((card, i) => (
          <Card key={i} className="border border-border bg-card rounded-xl shadow-none">
            <CardContent className="p-5 flex flex-col justify-between h-32">
              <div className="flex justify-between items-center w-full">
                <span className="text-[10px] font-extrabold uppercase text-muted-foreground tracking-widest">{card.label}</span>
                <div className="w-8 h-8 rounded-full bg-muted/40 flex items-center justify-center">
                  {card.icon}
                </div>
              </div>
              <div className="mt-2 space-y-1">
                {loading ? (
                  <div className="h-8 w-12 bg-muted/50 animate-pulse rounded-md" />
                ) : (
                  <h3 className={`text-2xl font-bold font-sans ${card.alert ? "text-amber-600 dark:text-amber-500" : "text-foreground"}`}>
                    {card.value}
                  </h3>
                )}
                <p className="text-[10px] text-muted-foreground">{card.desc}</p>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Analytics & Charts Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Chart 1: Disease Distribution */}
        <Card className="border border-border bg-card rounded-xl shadow-none p-5 flex flex-col justify-between items-center text-center space-y-3">
          <div className="w-full flex justify-between items-center text-[10px] uppercase font-bold text-muted-foreground tracking-widest">
            <span>Disease Cohort Split</span>
            <span className="text-primary font-mono">• Donut</span>
          </div>
          {renderDonutChart()}
          <div className="flex justify-between w-full text-[10px] font-semibold text-muted-foreground pt-1">
            <span className="flex items-center gap-1"><span className="w-2 h-2 bg-emerald-500 rounded-full" /> Normal ({normalCount})</span>
            <span className="flex items-center gap-1"><span className="w-2 h-2 bg-amber-500 rounded-full" /> TB ({tbCount})</span>
          </div>
        </Card>

        {/* Chart 2: Confidence Distribution */}
        <Card className="border border-border bg-card rounded-xl shadow-none p-5 flex flex-col justify-between items-center text-center space-y-3">
          <div className="w-full flex justify-between items-center text-[10px] uppercase font-bold text-muted-foreground tracking-widest">
            <span>Confidence Brackets</span>
            <span className="text-primary font-mono">• Hist</span>
          </div>
          {renderConfidenceBarChart()}
        </Card>

        {/* Chart 3: Model Performance Line Chart */}
        <Card className="border border-border bg-card rounded-xl shadow-none p-5 flex flex-col justify-between items-center text-center space-y-3">
          <div className="w-full flex justify-between items-center text-[10px] uppercase font-bold text-muted-foreground tracking-widest">
            <span>Weekly Accuracy Trend</span>
            <span className="text-primary font-mono">• Spark</span>
          </div>
          {renderPerformanceLineChart()}
        </Card>

        {/* Chart 4: Reviewer Agreement Dial */}
        <Card className="border border-border bg-card rounded-xl shadow-none p-5 flex flex-col justify-between items-center text-center space-y-3">
          <div className="w-full flex justify-between items-center text-[10px] uppercase font-bold text-muted-foreground tracking-widest">
            <span>Agreement Calibration</span>
            <span className="text-primary font-mono">• Gauge</span>
          </div>
          {renderAgreementGauge()}
        </Card>
      </div>

      {/* Research Export Action Card */}
      <Card className="border border-border bg-card rounded-xl shadow-none p-6 flex flex-col sm:flex-row justify-between items-center gap-4">
        <div className="space-y-1 text-center sm:text-left">
          <h4 className="text-sm font-bold text-foreground">Research Datasets &amp; Validation Export</h4>
          <p className="text-xs text-muted-foreground">Download the entire patient cohort study database and reviewer override reviews in CSV or JSON.</p>
        </div>
        <div className="flex gap-2">
          <Button
            onClick={() => downloadResearchExport("csv")}
            variant="outline"
            className="rounded-full text-xs font-semibold h-9 gap-1.5 cursor-pointer"
          >
            <Download className="w-3.5 h-3.5" /> Export CSV
          </Button>
          <Button
            onClick={() => downloadResearchExport("json")}
            variant="outline"
            className="rounded-full text-xs font-semibold h-9 gap-1.5 cursor-pointer"
          >
            <Download className="w-3.5 h-3.5" /> Export JSON
          </Button>
        </div>
      </Card>
    </div>
  );
}
