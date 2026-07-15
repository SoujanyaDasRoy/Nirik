"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import {
  Activity,
  ArrowRight,
  ShieldAlert,
  BookOpen,
  Cpu,
  ChevronDown,
  Play,
  TrendingUp,
  Brain,
  Users,
  FileText,
  Scan,
} from "lucide-react";

export default function AboutPage() {
  const router = useRouter();
  const [accepted, setAccepted] = useState(false);
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [aboutOpen, setAboutOpen] = useState(false);
  const [highlightsOpen, setHighlightsOpen] = useState(true);
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const checkSession = async () => {
      try {
        const API_BASE = process.env.NEXT_PUBLIC_API_URL || "https://projectmantra-nirikshon-backend.hf.space";
        const res = await fetch(`${API_BASE}/session`, { credentials: "include" });
        if (res.ok) {
          const data = await res.json();
          if (data.authenticated) setIsLoggedIn(true);
        }
      } catch {}
    };
    checkSession();

    const handleScroll = () => setScrolled(window.scrollY > 10);
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const canLaunch = isLoggedIn || accepted;

  return (
    <div className="min-h-screen flex flex-col font-sans selection:bg-[#0099ff]/30" style={{ backgroundColor: "#090909", color: "#ffffff" }}>
      
      {/* ── TOP NAV ── */}
      <header
        className={`sticky top-0 z-50 transition-all duration-300 ${scrolled ? 'backdrop-blur-xl' : ''}`}
        style={{
          backgroundColor: scrolled ? "rgba(9, 9, 9, 0.85)" : "transparent",
          borderBottom: scrolled ? "1px solid #262626" : "1px solid transparent",
        }}
      >
        <div className="h-16 max-w-[1199px] mx-auto px-6 flex items-center justify-between">
          <div className="flex items-center gap-8">
            <Link href="/" className="flex items-center gap-2 cursor-pointer relative z-10 group">
              <img src="/nirikshon_logo.png" alt="Nirikshon Logo" className="h-[64px] w-auto object-contain transition-transform hover:scale-105 origin-left" />
            </Link>
            <span className="hidden md:inline-flex items-center px-2.5 py-1 rounded-full text-[11px] font-medium tracking-wide bg-[#141414] text-[#999999] border border-[#262626]">
              Research Prototype
            </span>
          </div>
          
          <div className="flex items-center gap-6">
            <button
              onClick={() => document.getElementById("medical-disclaimer")?.scrollIntoView({ behavior: "smooth", block: "center" })}
              className="text-[14px] font-medium text-[#E0E0E0] hover:text-[#ffffff] transition-colors"
            >
              Disclaimer
            </button>
            <button
              onClick={() => document.getElementById("model-section")?.scrollIntoView({ behavior: "smooth" })}
              className="text-[14px] font-medium text-[#E0E0E0] hover:text-[#ffffff] transition-colors"
            >
              Model Results
            </button>
            <button
              onClick={() => {
                if (canLaunch) router.push("/diagnose");
                else document.getElementById("medical-disclaimer")?.scrollIntoView({ behavior: "smooth", block: "center" });
              }}
              className="flex items-center gap-2 px-5 py-2.5 rounded-full font-semibold text-[14px] transition-all hover:scale-[1.02] shadow-lg shadow-white/5 hover:bg-[#e0e0e0]"
              style={{
                backgroundColor: "#ffffff",
                color: "#000000",
                border: "1px solid transparent"
              }}
            >
              Launch <ArrowRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      </header>

      {/* ── HERO ── */}
      <section className="relative py-[80px] md:py-[100px] flex flex-col items-center text-center px-6 overflow-hidden">
        {/* Subtle background glow */}
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-[#ffffff]/5 blur-[120px] rounded-full pointer-events-none"></div>
        
        <div className="max-w-[900px] mx-auto flex flex-col items-center relative z-10">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full text-[12px] font-medium tracking-wide bg-[#1c1c1c] text-[#ffffff] mb-6 border border-[#262626]">
            Final Year Academic Project · 2025–2026
          </div>
          
          <h1 className="text-[46px] md:text-[68px] font-medium leading-[1.0] tracking-[-2px] md:tracking-[-3.5px] mb-6 text-[#F5F5F5]">
            Nirikhshon.
          </h1>
          
          <p className="text-[16px] md:text-[18px] text-[#E0E0E0] font-normal leading-[1.5] tracking-[-0.2px] max-w-[650px] mb-8">
            AI-assisted pulmonary tuberculosis screening powered by a fine-tuned DenseNet-121 model with Grad-CAM explainability and a clinical review workflow.
          </p>

          <div className="flex flex-wrap justify-center gap-2.5 mb-10">
            {[
              { icon: <Scan className="w-4 h-4" />, label: "DICOM Support" },
              { icon: <Brain className="w-4 h-4" />, label: "Grad-CAM XAI" },
              { icon: <TrendingUp className="w-4 h-4" />, label: "Longitudinal Tracking" },
              { icon: <Users className="w-4 h-4" />, label: "Patient Registry" },
              { icon: <FileText className="w-4 h-4" />, label: "PDF Reports" },
            ].map(f => (
              <span key={f.label} className="flex items-center gap-2 px-3 py-1.5 rounded-full text-[13px] font-medium bg-[#141414] text-[#E0E0E0] border border-[#262626] transition-colors hover:border-[#ffffff]/50 hover:text-[#ffffff] hover:bg-[#ffffff]/10 cursor-default">
                <span className="text-current opacity-70">{f.icon}</span>
                {f.label}
              </span>
            ))}
          </div>

          <div className="flex flex-col sm:flex-row items-center gap-4">
            <button
              onClick={() => {
                if (canLaunch) router.push("/diagnose");
                else document.getElementById("disclaimer-accept")?.focus();
              }}
              className="flex items-center gap-2.5 px-6 py-3 rounded-full font-semibold text-[14px] transition-all hover:scale-[1.02] shadow-xl shadow-white/5 hover:bg-[#e0e0e0] cursor-pointer"
              style={{
                backgroundColor: "#ffffff",
                color: "#000000",
              }}
            >
              <Play className="w-5 h-5 fill-current" />
              Launch Screening Workstation
            </button>
            {!canLaunch && (
              <p className="text-[13px] text-[#999999]">Accept the disclaimer below to enable launch.</p>
            )}
          </div>
        </div>
      </section>

      {/* ── KEY METRICS (Gradient Spotlight Cards) ── */}
      <section className="max-w-[1199px] mx-auto px-6 pb-[120px] w-full">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {[
            { label: "AUC-ROC", value: "99.4%", sub: "Classifier discrimination", bg: "#6a4cf5" }, // Violet
            { label: "TB Recall", value: "96.3%", sub: "Active cases detected", bg: "#d44df0" }, // Magenta
            { label: "Accuracy", value: "96.4%", sub: "Overall test set", bg: "#ff5577" }, // Coral
            { label: "F1 Score", value: "0.943", sub: "Precision-recall harmonic", bg: "#ff7a3d" }, // Orange
          ].map(m => (
            <div key={m.label} className="rounded-[30px] p-8 flex flex-col justify-between aspect-square" style={{ backgroundColor: m.bg }}>
              <p className="text-[14px] font-medium text-[#ffffff] tracking-wide mix-blend-overlay">{m.label}</p>
              <div>
                <p className="text-[62px] font-medium tracking-[-3.1px] text-[#ffffff] leading-[1] mb-2">{m.value}</p>
                <p className="text-[15px] text-[#ffffff] mix-blend-overlay leading-[1.3] tracking-[-0.15px]">{m.sub}</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* ── MEDICAL DISCLAIMER ── */}
      <section id="medical-disclaimer" className="max-w-[900px] mx-auto px-6 pb-[120px] w-full">
        <div className="rounded-[20px] p-8 space-y-6 bg-[#141414] border border-[#262626]">
          <div className="flex items-start gap-4">
            <div className="w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0 bg-[#1c1c1c] border border-[#262626]">
              <ShieldAlert className="w-5 h-5 text-[#ffffff]" />
            </div>
            <div>
              <p className="font-bold text-[18px] text-[#ffffff] tracking-[-0.2px]">Medical Disclaimer — Please Read Before Proceeding</p>
              <p className="text-[14px] mt-1 text-[#999999]">Required for responsible use of this research tool.</p>
            </div>
          </div>
          <div className="space-y-3 text-[15px] leading-[1.4] text-[#999999]">
            <p>• <strong className="text-[#ffffff]">Not a Medical Device.</strong> Nirikhshon is an academic research prototype with no clinical certifications (FDA, CE Mark, CDSCO, HIPAA).</p>
            <p>• <strong className="text-[#ffffff]">Not a Final Diagnosis.</strong> All AI results are preliminary aids. A certified radiologist must confirm all findings.</p>
            <p>• <strong className="text-[#ffffff]">Consult a Doctor.</strong> If any result is positive, please consult a licensed physician for confirmatory testing (GeneXpert, sputum smear).</p>
            <p>• <strong className="text-[#ffffff]">Academic Use Only.</strong> Built for a final year project demonstration. Not for real-world patient deployment.</p>
          </div>
          <div className="flex flex-col sm:flex-row items-center gap-4 pt-6 border-t border-[#262626] justify-between">
            <div className="flex items-center gap-4 flex-1">
              <input
                id="disclaimer-accept"
                type="checkbox"
                checked={accepted}
                onChange={e => setAccepted(e.target.checked)}
                className="w-5 h-5 cursor-pointer appearance-none rounded-[6px] border border-[#262626] bg-[#090909] checked:bg-[#0099ff] checked:border-[#0099ff] transition-colors relative shrink-0"
              />
              <label htmlFor="disclaimer-accept" className="text-[15px] font-medium cursor-pointer text-[#ffffff]">
                I understand this is not a clinical tool and will consult a doctor for medical decisions.
              </label>
            </div>
            {accepted && (
              <button
                onClick={() => router.push("/diagnose")}
                className="flex items-center justify-center gap-2.5 px-6 py-3 rounded-full font-semibold text-[14px] bg-[#ffffff] text-[#000000] hover:scale-[1.02] transition-all cursor-pointer shadow-lg w-full sm:w-auto shrink-0 animate-fadein"
              >
                Enter Workstation
                <Play className="w-4 h-4 fill-current" />
              </button>
            )}
          </div>
        </div>
      </section>

      {/* ── SYSTEM ARCHITECTURE ── */}
      <section id="architecture-section" className="max-w-[1199px] mx-auto px-6 pb-[120px] w-full space-y-12">
        <div className="text-center space-y-4 max-w-[700px] mx-auto">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full text-[13px] font-medium tracking-wide bg-[#141414] text-[#ffffff] border border-[#262626]">
            Platform Architecture
          </div>
          <h2 className="text-[52px] font-medium tracking-[-2.5px] leading-[1]">Decoupled & Modular</h2>
          <p className="text-[18px] text-[#999999] leading-[1.3] tracking-[-0.18px]">
            Engineered for high performance, clinical safety, and seamless LLM integration.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {[
            { 
              title: "Next.js Frontend", 
              desc: "React 19 & Tailwind CSS powers a responsive, scrollable clinical workspace with native DICOM parsing and zero viewport locks.", 
              icon: <Scan className="w-6 h-6 text-[#0099ff]" /> 
            },
            { 
              title: "DenseNet-121 Engine", 
              desc: "PyTorch & ONNX dual-load backend providing 15ms CPU inference and true-gradient backpropagation for Grad-CAM++.", 
              icon: <Cpu className="w-6 h-6 text-[#10b981]" /> 
            },
            { 
              title: "LLM Co-Pilot", 
              desc: "Gemini 2.5 Flash integrated with clinical guardrails. Provides context-aware insights based on the AI prediction.", 
              icon: <Brain className="w-6 h-6 text-[#d44df0]" /> 
            },
            { 
              title: "Flask REST API", 
              desc: "Modular Flask 3.1 architecture with independent blueprints for auth, prediction, explainability, and LLM chat.", 
              icon: <Activity className="w-6 h-6 text-[#ff5577]" /> 
            }
          ].map(arc => (
            <div key={arc.title} className="p-8 rounded-[24px] bg-[#141414] border border-[#262626] hover:bg-[#1c1c1c] transition-colors flex flex-col gap-4 shadow-xl">
              <div className="w-12 h-12 rounded-full bg-[#090909] border border-[#262626] flex items-center justify-center">
                {arc.icon}
              </div>
              <div>
                <p className="text-[18px] font-medium text-[#ffffff] mb-2">{arc.title}</p>
                <p className="text-[14px] text-[#999999] leading-[1.5]">{arc.desc}</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* ── MODEL PERFORMANCE SHOWCASE (BENTO GRID) ── */}
      <section id="model-section" className="max-w-[1199px] mx-auto px-6 pb-[120px] w-full space-y-12">
        <div className="text-center space-y-4 max-w-[700px] mx-auto">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full text-[13px] font-medium tracking-wide bg-[#141414] text-[#ffffff] border border-[#262626]">
            Validated Model Performance
          </div>
          <h2 className="text-[52px] font-medium tracking-[-2.5px] leading-[1]">Real Results on Unseen Data</h2>
          <p className="text-[18px] text-[#999999] leading-[1.3] tracking-[-0.18px]">
            Comprehensive evaluation metrics and visualizations from the held-out test cohort.
          </p>
        </div>

        {/* Bento Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          
          {/* Large Card: Grad-CAM */}
          <div className="lg:col-span-2 rounded-[24px] overflow-hidden bg-[#141414] border border-[#262626] flex flex-col group shadow-xl">
            <div className="px-6 py-5 border-b border-[#262626] flex justify-between items-center">
              <div>
                <p className="text-[18px] font-medium text-[#ffffff]">XAI: Grad-CAM Visualizations</p>
                <p className="text-[14px] text-[#999999]">Gradient-weighted class activation maps across varied cases.</p>
              </div>
              <span className="px-3 py-1.5 rounded-full text-[12px] font-medium bg-[#1c1c1c] text-[#ffffff] border border-[#262626]">Interpretability</span>
            </div>
            <div className="p-6 bg-[#090909] flex-1 flex items-center justify-center">
              <img src="/model-results/gradcam_visualizations.png" alt="Grad-CAM" className="w-full h-auto rounded-[12px] object-contain transition-transform duration-500 group-hover:scale-[1.02]" />
            </div>
          </div>

          {/* Small Card: Confusion Matrix */}
          <div className="rounded-[24px] overflow-hidden bg-[#141414] border border-[#262626] flex flex-col group shadow-xl">
            <div className="px-6 py-5 border-b border-[#262626]">
              <p className="text-[18px] font-medium text-[#ffffff]">Confusion Matrix</p>
              <p className="text-[14px] text-[#999999]">358 TN · 157 TP · 13 FP · 6 FN</p>
            </div>
            <div className="p-6 bg-[#090909] flex-1 flex items-center justify-center">
              <img src="/model-results/confusion_matrix.png" alt="Confusion Matrix" className="w-full h-auto rounded-[12px] object-contain transition-transform duration-500 group-hover:scale-[1.05]" />
            </div>
          </div>

          {/* Small Card: ROC Curves */}
          <div className="rounded-[24px] overflow-hidden bg-[#141414] border border-[#262626] flex flex-col group shadow-xl">
            <div className="px-6 py-5 border-b border-[#262626]">
              <p className="text-[18px] font-medium text-[#ffffff]">ROC & Precision-Recall</p>
              <p className="text-[14px] text-[#999999]">AUC = 0.941</p>
            </div>
            <div className="p-6 bg-[#090909] flex-1 flex items-center justify-center">
              <img src="/model-results/roc_pr_curves.png" alt="ROC Curves" className="w-full h-auto rounded-[12px] object-contain transition-transform duration-500 group-hover:scale-[1.05]" />
            </div>
          </div>

          {/* Large Card: Sample Predictions */}
          <div className="lg:col-span-2 rounded-[24px] overflow-hidden bg-[#141414] border border-[#262626] flex flex-col group shadow-xl">
            <div className="px-6 py-5 border-b border-[#262626]">
              <p className="text-[18px] font-medium text-[#ffffff]">Sample Clinical Predictions</p>
              <p className="text-[14px] text-[#999999]">Demonstrating high-confidence classifications vs ground truth.</p>
            </div>
            <div className="p-6 bg-[#090909] flex-1 flex items-center justify-center">
              <img src="/model-results/sample_predictions.png" alt="Sample Predictions" className="w-full h-auto rounded-[12px] object-contain transition-transform duration-500 group-hover:scale-[1.02]" />
            </div>
          </div>

          {/* Small Card: Performance Comparison */}
          <div className="lg:col-span-2 rounded-[24px] overflow-hidden bg-[#141414] border border-[#262626] flex flex-col group shadow-xl">
            <div className="px-6 py-5 border-b border-[#262626]">
              <p className="text-[18px] font-medium text-[#ffffff]">Architecture Comparison</p>
              <p className="text-[14px] text-[#999999]">DenseNet vs ResNet Performance Benchmark.</p>
            </div>
            <div className="p-6 bg-[#090909] flex-1 flex items-center justify-center">
              <img src="/model-results/performance_comparison.png" alt="Performance" className="w-full max-h-[300px] rounded-[12px] object-contain transition-transform duration-500 group-hover:scale-[1.02]" />
            </div>
          </div>

          {/* Small Card: Deployment Metrics */}
          <div className="rounded-[24px] overflow-hidden bg-[#141414] border border-[#262626] flex flex-col group shadow-xl">
            <div className="px-6 py-5 border-b border-[#262626]">
              <p className="text-[18px] font-medium text-[#ffffff]">Deployment Profiling</p>
              <p className="text-[14px] text-[#999999]">Hardware Latency & Memory.</p>
            </div>
            <div className="p-6 bg-[#090909] flex-1 flex items-center justify-center">
              <img src="/model-results/deployment_metrics.png" alt="Deployment" className="w-full h-auto rounded-[12px] object-contain transition-transform duration-500 group-hover:scale-[1.05]" />
            </div>
          </div>
          
        </div>
      </section>

      {/* ── DETAILS (Collapsibles) ── */}
      <section className="max-w-[900px] mx-auto px-6 pb-[120px] w-full space-y-6">
        <div className="rounded-[20px] overflow-hidden bg-[#141414] border border-[#262626]">
          <button
            onClick={() => setAboutOpen(!aboutOpen)}
            className="w-full px-6 py-5 flex items-center justify-between hover:bg-[#1c1c1c] transition-colors"
          >
            <div className="flex items-center gap-4">
              <BookOpen className="w-5 h-5 text-[#ffffff]" />
              <div className="text-left">
                <p className="text-[18px] font-medium text-[#ffffff] tracking-[-0.2px]">About This Project</p>
                <p className="text-[14px] text-[#999999]">Clinical objectives, training pipeline, and validation metrics</p>
              </div>
            </div>
            <ChevronDown className={`w-5 h-5 text-[#ffffff] transition-transform duration-300 ${aboutOpen ? "rotate-180" : ""}`} />
          </button>
          {aboutOpen && (
            <div className="px-6 pb-8 space-y-8 border-t border-[#262626] animate-fadein">
              <div className="space-y-4 pt-6">
                <p className="text-[13px] font-medium text-[#ffffff] uppercase tracking-wide">Training Pipeline</p>
                {[
                  { phase: "Phase A — Teacher (ResNet-50)", details: "Large network trained on Shenzhen + Montgomery global datasets capturing general pulmonary structures." },
                  { phase: "Phase B — Student (DenseNet-121)", details: "~7M parameter architecture trained via knowledge distillation to mimic teacher logits. 3× smaller, CPU-optimized." },
                  { phase: "Phase C — Indian Domain Adaptation (NIRT)", details: "Fine-tuned on NIRT Chennai dataset, calibrating for local scanner contrast and visual noise patterns." },
                ].map((p, i) => (
                  <div key={i} className="flex gap-5 p-5 rounded-[15px] bg-[#1c1c1c] border border-[#262626]">
                    <div className="w-8 h-8 rounded-full flex items-center justify-center text-[14px] font-medium bg-[#ffffff] text-[#000000] flex-shrink-0">
                      {i + 1}
                    </div>
                    <div className="space-y-1">
                      <p className="text-[15px] font-medium text-[#ffffff]">{p.phase}</p>
                      <p className="text-[14px] leading-[1.4] text-[#999999]">{p.details}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        <div className="rounded-[20px] overflow-hidden bg-[#141414] border border-[#262626]">
          <button
            onClick={() => setHighlightsOpen(!highlightsOpen)}
            className="w-full px-6 py-5 flex items-center justify-between hover:bg-[#1c1c1c] transition-colors"
          >
            <div className="flex items-center gap-4">
              <Cpu className="w-5 h-5 text-[#ffffff]" />
              <div className="text-left">
                <p className="text-[18px] font-medium text-[#ffffff] tracking-[-0.2px]">Tech Stack & Specifications</p>
                <p className="text-[14px] text-[#999999]">Model details and technology used</p>
              </div>
            </div>
            <ChevronDown className={`w-5 h-5 text-[#ffffff] transition-transform duration-300 ${highlightsOpen ? "rotate-180" : ""}`} />
          </button>
          {highlightsOpen && (
            <div className="px-6 pb-6 border-t border-[#262626] animate-fadein">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-8 pt-6">
                <div className="space-y-4">
                  <p className="text-[13px] font-medium text-[#ffffff] uppercase tracking-wide">Model Specs</p>
                  {[
                    ["Architecture", "DenseNet-121 (Student)"],
                    ["Parameters", "~7.04 Million"],
                    ["Model Size", "29.7 MB"],
                    ["Inference", "~15.5 ms/image (CPU)"],
                    ["AUC-ROC", "0.994"],
                    ["Best Threshold", "0.69"],
                  ].map(([k, v]) => (
                    <div key={k} className="flex justify-between items-center text-[14px]">
                      <span className="text-[#999999]">{k}</span>
                      <span className="font-medium text-[#ffffff]">{v}</span>
                    </div>
                  ))}
                </div>
                <div className="space-y-4">
                  <p className="text-[13px] font-medium text-[#ffffff] uppercase tracking-wide">Technology</p>
                  {[
                    ["Backend", "Flask 3.1 + Python 3.14"],
                    ["ML Engine", "PyTorch 2.x + Keras 3.8"],
                    ["Frontend", "Next.js 16 + React 19"],
                    ["Storage", "SQLite3 Database"],
                    ["XAI", "Grad-CAM + LIME + SHAP"],
                  ].map(([k, v]) => (
                    <div key={k} className="flex justify-between items-center text-[14px]">
                      <span className="text-[#999999]">{k}</span>
                      <span className="font-medium text-[#ffffff]">{v}</span>
                    </div>
                  ))}
                </div>
              </div>
              <div className="mt-8 pt-6 border-t border-[#262626]">
                <button
                  onClick={() => {
                    if (canLaunch) router.push("/diagnose");
                    else document.getElementById("medical-disclaimer")?.scrollIntoView({ behavior: "smooth", block: "center" });
                  }}
                  className="w-full flex items-center justify-center gap-2 py-4 rounded-full font-semibold text-[15px] transition-all bg-[#ffffff] text-[#000000] hover:scale-[1.02]"
                >
                  Launch Screening Workstation <ArrowRight className="w-5 h-5" />
                </button>
              </div>
            </div>
          )}
        </div>
      </section>

      {/* ── FOOTER ── */}
      <footer className="mt-auto py-12 px-6 bg-[#090909] border-t border-[#262626]">
        <div className="max-w-[1199px] mx-auto flex flex-col sm:flex-row justify-between items-center gap-4 text-[13px] text-[#999999]">
          <p>© {new Date().getFullYear()} Nirikhshon — Academic Final Year Project Prototype</p>
          <p>Built with Next.js, Flask, and DenseNet-121</p>
        </div>
      </footer>
    </div>
  );
}
