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
  AlertTriangle
} from "lucide-react";
import { Separator } from "@/components/ui/separator";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "https://projectmantra-nirikshon-backend.hf.space";

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
    const interval = setInterval(fetchStats, 30000);
    return () => clearInterval(interval);
  }, []);

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
        <circle cx="50" cy="50" r={r} fill="transparent" stroke="#262626" strokeWidth="8" />
        <circle
          cx="50"
          cy="50"
          r={r}
          fill="transparent"
          stroke="#0099ff"
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
          stroke="#ff5577"
          strokeWidth="8"
          strokeDasharray={`${tbDash} ${C}`}
          strokeDashoffset={-normalDash}
          transform="rotate(-90 50 50)"
        />
        <text x="50" y="47" textAnchor="middle" dominantBaseline="middle" className="fill-white font-bold text-[14px] font-sans tracking-tight">
          {totalDist}
        </text>
        <text x="50" y="58" textAnchor="middle" dominantBaseline="middle" className="fill-[#999999] text-[6px] uppercase tracking-widest font-semibold">
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
                fill="#1c1c1c"
                rx="4"
                className="transition-all duration-300 hover:fill-[#0099ff]"
              />
              <text x={x + barWidth / 2} y={y - 4} textAnchor="middle" className="fill-white text-[8px] font-bold">
                {val}
              </text>
              <text x={x + barWidth / 2} y={paddingY + chartHeight + 12} textAnchor="middle" className="fill-[#999999] text-[6px] font-semibold">
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
        <line x1={px} y1={py} x2={px + cW} y2={py} stroke="#262626" strokeDasharray="3 3" />
        <line x1={px} y1={py + cH / 2} x2={px + cW} y2={py + cH / 2} stroke="#262626" strokeDasharray="3 3" />
        <line x1={px} y1={py + cH} x2={px + cW} y2={py + cH} stroke="#404040" />
        {pts.length > 0 && (
          <polyline
            fill="none"
            stroke="#0099ff"
            strokeWidth="2.5"
            points={polylinePath}
          />
        )}
        {pts.map((p, idx) => (
          <g key={idx}>
            <circle cx={p.x} cy={p.y} r="3" className="fill-[#141414] stroke-[#ffffff] stroke-[2px] cursor-pointer" />
            <text x={p.x} y={p.y - 6} textAnchor="middle" className="fill-[#ffffff] text-[8px] font-bold">
              {p.accuracy}%
            </text>
            <text x={p.x} y={py + cH + 12} textAnchor="middle" className="fill-[#999999] text-[6px] font-semibold">
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
          stroke="#262626"
          strokeWidth="8"
          strokeDasharray={`${semiC} ${gC}`}
          transform="rotate(-180 50 50)"
        />
        <circle
          cx="50"
          cy="50"
          r={gR}
          fill="transparent"
          stroke="#6a4cf5" // Framer Violet Accent
          strokeWidth="8"
          strokeDasharray={`${filledDash} ${gC}`}
          transform="rotate(-180 50 50)"
          className="transition-all duration-500"
        />
        <text x="50" y="42" textAnchor="middle" className="fill-white font-bold text-[14px] font-sans tracking-tight">
          {agreement}%
        </text>
        <text x="50" y="52" textAnchor="middle" className="fill-[#999999] text-[6px] uppercase tracking-widest font-semibold">
          Agreement Rate
        </text>
      </svg>
    );
  };

  return (
    <div className="max-w-[1199px] mx-auto space-y-12">
      {/* Welcome & Overview Banner */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-6">
        <div>
          <h1 className="text-[62px] font-medium text-white tracking-[-3.1px] leading-none mb-4">Dashboard</h1>
          <p className="text-[18px] text-[#999999] tracking-[-0.18px] leading-[1.3] max-w-xl">
            Real-time cohort monitoring, diagnostic metrics, and verification backlog stats.
          </p>
        </div>
        <div className="flex gap-3">
          <button 
            onClick={() => onNavigate("patients")}
            className="px-[15px] py-[10px] rounded-[100px] bg-[#141414] text-[#ffffff] text-[14px] font-medium tracking-[-0.14px] transition-transform hover:scale-[1.02] flex items-center gap-2"
          >
            <Users className="w-4 h-4" /> Manage Patients
          </button>
          <button 
            onClick={hasFiles ? onOpenWorkbench : () => onNavigate("landing")}
            className="px-[15px] py-[10px] rounded-[100px] bg-[#ffffff] text-[#000000] text-[14px] font-medium tracking-[-0.14px] transition-transform hover:scale-[1.02] flex items-center gap-2"
          >
            <Activity className="w-4 h-4" /> 
            {hasFiles ? "Open Active Workbench" : "Intake New Scan"}
            <ChevronRight className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Stats Cards Grid - 1 Gradient Spotlight, 3 Charcoal */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
        {[
          { 
            label: "Total Studies Ingestion", 
            value: stats?.total_cases || 0, 
            icon: <FolderOpen className="w-5 h-5 text-[#999999]" strokeWidth={1.5} />,
            desc: "Historical cases in PACS node",
            style: "bg-[#141414] text-white border-[#262626]"
          },
          { 
            label: "TB Positive Index", 
            value: stats?.tb_positive_cases || 0, 
            icon: <AlertTriangle className="w-5 h-5 text-[#ff5577]" strokeWidth={1.5} />,
            desc: "AI predicted or review confirmed",
            style: "bg-[#141414] text-white border-[#262626]"
          },
          { 
            label: "Pending Adjudication", 
            value: stats?.pending_reviews || 0, 
            icon: <Clock className="w-5 h-5 text-white" strokeWidth={1.5} />,
            desc: "Reviews awaiting signature",
            // FRAMER GRADIENT SPOTLIGHT (Violet)
            style: "bg-gradient-to-br from-[#6a4cf5] to-[#4c2bb8] text-white border-transparent" 
          },
          { 
            label: "Finalized Reviews", 
            value: stats?.completed_reviews || 0, 
            icon: <CheckCircle className="w-5 h-5 text-[#22c55e]" strokeWidth={1.5} />,
            desc: "Reports locked & archived",
            style: "bg-[#141414] text-white border-[#262626]"
          }
        ].map((card, i) => (
          <div key={i} className={`border rounded-[20px] p-6 flex flex-col justify-between h-40 transition-transform hover:scale-[1.02] ${card.style}`}>
            <div className="flex justify-between items-start w-full">
              <span className={`text-[12px] font-medium tracking-[-0.12px] ${card.style.includes('gradient') ? 'text-white/80' : 'text-[#999999]'}`}>
                {card.label}
              </span>
              <div className={`w-10 h-10 rounded-full flex items-center justify-center ${card.style.includes('gradient') ? 'bg-white/10' : 'bg-[#1c1c1c]'}`}>
                {card.icon}
              </div>
            </div>
            <div className="mt-4 space-y-1">
              {loading ? (
                <div className="h-8 w-12 bg-white/10 animate-pulse rounded-md" />
              ) : (
                <h3 className="text-[32px] font-medium tracking-[-1.0px] leading-[1.13]">
                  {card.value}
                </h3>
              )}
              <p className={`text-[12px] tracking-[-0.12px] ${card.style.includes('gradient') ? 'text-white/70' : 'text-[#666666]'}`}>
                {card.desc}
              </p>
            </div>
          </div>
        ))}
      </div>

      {/* Analytics & Charts Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {/* Chart 1: Disease Distribution */}
        <div className="border border-[#262626] bg-[#141414] rounded-[20px] p-6 flex flex-col justify-between items-center text-center space-y-4 hover:border-[#404040] transition-colors">
          <div className="w-full flex justify-between items-center text-[12px] font-medium tracking-[-0.12px] text-[#999999]">
            <span>Case Distribution</span>
            <span className="text-[#0099ff]">• Donut</span>
          </div>
          {renderDonutChart()}
        </div>

        {/* Chart 2: Confidence Distribution */}
        <div className="border border-[#262626] bg-[#141414] rounded-[20px] p-6 flex flex-col justify-between items-center text-center space-y-4 hover:border-[#404040] transition-colors">
          <div className="w-full flex justify-between items-center text-[12px] font-medium tracking-[-0.12px] text-[#999999]">
            <span>Confidence Brackets</span>
            <span className="text-[#0099ff]">• Hist</span>
          </div>
          {renderConfidenceBarChart()}
        </div>

        {/* Chart 3: Model Performance Line Chart */}
        <div className="border border-[#262626] bg-[#141414] rounded-[20px] p-6 flex flex-col justify-between items-center text-center space-y-4 hover:border-[#404040] transition-colors">
          <div className="w-full flex justify-between items-center text-[12px] font-medium tracking-[-0.12px] text-[#999999]">
            <span>Weekly Accuracy Trend</span>
            <span className="text-[#0099ff]">• Spark</span>
          </div>
          {renderPerformanceLineChart()}
        </div>
      </div>
    </div>
  );
}
