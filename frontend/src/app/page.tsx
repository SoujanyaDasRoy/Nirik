"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  ShieldAlert,
  BookOpen,
  Cpu,
  Layers,
  TrendingUp,
  CheckCircle,
  Play,
  Microscope,
  Scan,
  Brain,
  Users,
  FileText,
  Database,
  ChevronDown,
} from "lucide-react";

import { Badge } from "@/components/ui/badge";

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
    <div className="min-h-screen flex flex-col" style={{ background: "#313338", color: "#F2F3F5", fontFamily: "Inter, sans-serif" }}>

      {/* ── DISCORD-STYLE HEADER ── */}
      <header
        className="sticky top-0 z-50 transition-all duration-200"
        style={{
          background: scrolled ? "rgba(43,45,49,0.95)" : "#2B2D31",
          borderBottom: "1px solid rgba(255,255,255,0.06)",
          backdropFilter: scrolled ? "blur(12px)" : "none",
        }}
      >
        <div className="h-14 max-w-6xl mx-auto px-6 flex items-center justify-between">
          {/* Logo */}
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-full flex items-center justify-center" style={{ background: "#5865F2" }}>
              <Activity className="w-4 h-4 text-white" />
            </div>
            <span className="font-bold text-[15px] text-white tracking-tight">Nirikhshon</span>
            <span className="hidden md:inline-flex items-center px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider" style={{ background: "rgba(254,231,92,0.15)", color: "#FEE75C", border: "1px solid rgba(254,231,92,0.25)" }}>
              Research Prototype
            </span>
          </div>

          {/* Nav */}
          <div className="flex items-center gap-3">
            <button
              onClick={() => {
                document.getElementById("medical-disclaimer")?.scrollIntoView({ behavior: "smooth", block: "center" });
              }}
              className="text-sm font-medium cursor-pointer transition-colors"
              style={{ color: "#949BA4" }}
              onMouseEnter={e => (e.currentTarget.style.color = "#DBDEE1")}
              onMouseLeave={e => (e.currentTarget.style.color = "#949BA4")}
            >
              Disclaimer
            </button>
            <button
              onClick={() => {
                document.getElementById("model-section")?.scrollIntoView({ behavior: "smooth" });
              }}
              className="text-sm font-medium cursor-pointer transition-colors"
              style={{ color: "#949BA4" }}
              onMouseEnter={e => (e.currentTarget.style.color = "#DBDEE1")}
              onMouseLeave={e => (e.currentTarget.style.color = "#949BA4")}
            >
              Model Results
            </button>
            <button
              onClick={() => {
                if (canLaunch) router.push("/diagnose");
                else document.getElementById("medical-disclaimer")?.scrollIntoView({ behavior: "smooth", block: "center" });
              }}
              className="flex items-center gap-1.5 px-4 py-1.5 rounded font-semibold text-sm text-white transition-all cursor-pointer"
              style={{ background: canLaunch ? "#5865F2" : "#4E5058" }}
              onMouseEnter={e => { if (canLaunch) (e.currentTarget as HTMLButtonElement).style.background = "#4752C4"; }}
              onMouseLeave={e => { (e.currentTarget as HTMLButtonElement).style.background = canLaunch ? "#5865F2" : "#4E5058"; }}
            >
              Launch <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      </header>

      {/* ── HERO ── */}
      <section className="py-24 sm:py-32 flex flex-col items-center text-center px-6">
        <div className="max-w-3xl mx-auto space-y-6">
          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-[11px] font-bold uppercase tracking-widest" style={{ background: "rgba(88,101,242,0.15)", color: "#5865F2", border: "1px solid rgba(88,101,242,0.25)" }}>
            Final Year Academic Project · 2025–2026
          </div>

          <h1 className="text-5xl sm:text-6xl lg:text-7xl font-extrabold text-white tracking-tight leading-none">
            Nirik<span style={{ color: "#5865F2" }}>hshon</span>
          </h1>

          <p className="text-lg text-[#949BA4] max-w-xl mx-auto leading-relaxed">
            AI-assisted pulmonary tuberculosis screening powered by a fine-tuned DenseNet-121 model with Grad-CAM explainability and a clinical review workflow.
          </p>

          {/* Feature pills */}
          <div className="flex flex-wrap justify-center gap-2 pt-2">
            {[
              { icon: <Scan className="w-3.5 h-3.5" />, label: "DICOM Support" },
              { icon: <Brain className="w-3.5 h-3.5" />, label: "Grad-CAM XAI" },
              { icon: <TrendingUp className="w-3.5 h-3.5" />, label: "Longitudinal Tracking" },
              { icon: <Users className="w-3.5 h-3.5" />, label: "Patient Registry" },
              { icon: <FileText className="w-3.5 h-3.5" />, label: "PDF Reports" },
            ].map(f => (
              <span key={f.label} className="flex items-center gap-1.5 px-3 py-1.5 rounded-full text-[11px] font-semibold" style={{ background: "#2B2D31", color: "#DBDEE1", border: "1px solid rgba(255,255,255,0.08)" }}>
                <span style={{ color: "#5865F2" }}>{f.icon}</span>
                {f.label}
              </span>
            ))}
          </div>
        </div>
      </section>

      {/* ── KEY METRICS BENTO ── */}
      <section className="max-w-6xl mx-auto px-6 pb-16 w-full">
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {[
            { label: "AUC-ROC", value: "94.1%", sub: "Classifier discrimination", color: "#5865F2", bg: "rgba(88,101,242,0.1)", border: "rgba(88,101,242,0.25)" },
            { label: "TB Recall", value: "81.7%", sub: "Active cases detected", color: "#57F287", bg: "rgba(87,242,135,0.08)", border: "rgba(87,242,135,0.2)" },
            { label: "Accuracy", value: "93.4%", sub: "Overall test set", color: "#EB459E", bg: "rgba(235,69,158,0.08)", border: "rgba(235,69,158,0.2)" },
            { label: "F1 Score", value: "0.844", sub: "Precision-recall harmonic", color: "#FEE75C", bg: "rgba(254,231,92,0.08)", border: "rgba(254,231,92,0.2)" },
          ].map(m => (
            <div key={m.label} className="rounded-lg p-5 flex flex-col gap-1.5" style={{ background: m.bg, border: `1px solid ${m.border}` }}>
              <p className="text-[10px] font-bold uppercase tracking-widest" style={{ color: "#949BA4" }}>{m.label}</p>
              <p className="text-3xl font-extrabold tracking-tight" style={{ color: m.color }}>{m.value}</p>
              <p className="text-[11px]" style={{ color: "#949BA4" }}>{m.sub}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ── MEDICAL DISCLAIMER ── */}
      <section id="medical-disclaimer" className="max-w-6xl mx-auto px-6 pb-16 w-full">
        <div className="rounded-lg p-6 space-y-4" style={{ background: "rgba(254,231,92,0.06)", border: "1px solid rgba(254,231,92,0.2)" }}>
          <div className="flex items-start gap-3">
            <div className="w-9 h-9 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5" style={{ background: "rgba(254,231,92,0.15)" }}>
              <ShieldAlert className="w-4.5 h-4.5" style={{ color: "#FEE75C" }} />
            </div>
            <div>
              <p className="font-bold text-sm" style={{ color: "#FEE75C" }}>Medical Disclaimer — Please Read Before Proceeding</p>
              <p className="text-[11px] mt-0.5" style={{ color: "#949BA4" }}>Required for responsible use of this research tool.</p>
            </div>
          </div>
          <div className="space-y-2 text-xs leading-relaxed" style={{ color: "#949BA4" }}>
            <p>• <strong style={{ color: "#DBDEE1" }}>Not a Medical Device.</strong> Nirikhshon is an academic research prototype with no clinical certifications (FDA, CE Mark, CDSCO, HIPAA).</p>
            <p>• <strong style={{ color: "#DBDEE1" }}>Not a Final Diagnosis.</strong> All AI results are preliminary aids. A certified radiologist must confirm all findings.</p>
            <p>• <strong style={{ color: "#DBDEE1" }}>Consult a Doctor.</strong> If any result is positive, please consult a licensed physician for confirmatory testing (GeneXpert, sputum smear).</p>
            <p>• <strong style={{ color: "#DBDEE1" }}>Academic Use Only.</strong> Built for a final year project demonstration. Not for real-world patient deployment.</p>
          </div>
          <div className="flex items-center gap-3 pt-2" style={{ borderTop: "1px solid rgba(254,231,92,0.15)" }}>
            <input
              id="disclaimer-accept"
              type="checkbox"
              checked={accepted}
              onChange={e => setAccepted(e.target.checked)}
              className="w-4 h-4 cursor-pointer rounded"
              style={{ accentColor: "#5865F2" }}
            />
            <label htmlFor="disclaimer-accept" className="text-xs font-semibold cursor-pointer" style={{ color: "#FEE75C" }}>
              I understand this is not a clinical tool and will consult a doctor for medical decisions.
            </label>
          </div>

          {/* Launch CTA */}
          <div className="flex flex-col sm:flex-row items-start sm:items-center gap-3 pt-2">
            <button
              onClick={() => {
                if (canLaunch) router.push("/diagnose");
                else document.getElementById("disclaimer-accept")?.focus();
              }}
              className="flex items-center gap-2 px-6 py-3 rounded font-bold text-sm text-white transition-all cursor-pointer"
              style={{
                background: canLaunch ? "#5865F2" : "#4E5058",
                opacity: canLaunch ? 1 : 0.6,
              }}
              onMouseEnter={e => { if (canLaunch) (e.currentTarget as HTMLButtonElement).style.background = "#4752C4"; }}
              onMouseLeave={e => { (e.currentTarget as HTMLButtonElement).style.background = canLaunch ? "#5865F2" : "#4E5058"; }}
            >
              <Play className="w-4 h-4 fill-current" />
              Launch Screening Workstation
              <ArrowRight className="w-4 h-4" />
            </button>
            {!canLaunch && (
              <p className="text-[11px]" style={{ color: "#949BA4" }}>Accept the disclaimer above to enable launch.</p>
            )}
          </div>
        </div>
      </section>

      {/* ── MODEL PERFORMANCE SHOWCASE ── */}
      <section id="model-section" className="max-w-6xl mx-auto px-6 pb-20 w-full space-y-12">

        {/* Section header */}
        <div className="text-center space-y-2">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-[10px] font-bold uppercase tracking-widest" style={{ background: "rgba(88,101,242,0.12)", color: "#5865F2", border: "1px solid rgba(88,101,242,0.2)" }}>
            Validated Model Performance
          </div>
          <h2 className="text-3xl sm:text-4xl font-bold text-white">Real Results on Unseen Data</h2>
          <p className="text-sm max-w-xl mx-auto" style={{ color: "#949BA4" }}>
            Evaluated on a held-out test set from the NIRT Chennai cohort — never seen during training.
          </p>
        </div>

        {/* ROC + Confusion Matrix */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {[
            { title: "ROC & Precision-Recall Curves", sub: "AUC = 0.941 · Threshold = 0.870 · Recall = 81.7%", img: "/model-results/roc_pr_curves.png", alt: "ROC Curves" },
            { title: "Confusion Matrix — Test Set", sub: "378 TN · 89 TP · 13 FP · 20 FN", img: "/model-results/confusion_matrix.png", alt: "Confusion Matrix" },
          ].map(card => (
            <div key={card.title} className="rounded-lg overflow-hidden" style={{ background: "#2B2D31", border: "1px solid rgba(255,255,255,0.06)" }}>
              <div className="px-5 py-3.5" style={{ borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
                <p className="text-sm font-bold text-white">{card.title}</p>
                <p className="text-[11px] mt-0.5" style={{ color: "#949BA4" }}>{card.sub}</p>
              </div>
              <div className="p-3" style={{ background: "#1E1F22" }}>
                <img src={card.img} alt={card.alt} className="w-full rounded object-contain" />
              </div>
            </div>
          ))}
        </div>

        {/* Grad-CAM */}
        <div className="rounded-lg overflow-hidden" style={{ background: "#2B2D31", border: "1px solid rgba(255,255,255,0.06)" }}>
          <div className="px-5 py-3.5 flex items-center justify-between" style={{ borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
            <div>
              <p className="text-sm font-bold text-white">Grad-CAM Explainability Visualizations</p>
              <p className="text-[11px] mt-0.5" style={{ color: "#949BA4" }}>Gradient-weighted class activation maps. <span style={{ color: "#ED4245" }}>Red</span> = high attention, <span style={{ color: "#5865F2" }}>Blue</span> = low attention.</p>
            </div>
            <span className="px-2 py-1 rounded text-[10px] font-bold uppercase" style={{ background: "rgba(88,101,242,0.15)", color: "#5865F2", border: "1px solid rgba(88,101,242,0.25)" }}>XAI</span>
          </div>
          <div className="p-3" style={{ background: "#1E1F22" }}>
            <img src="/model-results/gradcam_visualizations.png" alt="Grad-CAM" className="w-full rounded object-contain" />
          </div>
          <div className="grid grid-cols-3 gap-0 text-center" style={{ borderTop: "1px solid rgba(255,255,255,0.06)" }}>
            {[
              { label: "True TB Cases", desc: "Model focuses on upper lobe consolidation and cavity regions — clinically correct.", color: "#57F287" },
              { label: "High-Conf Errors", desc: "False positives show pleural/hilar features sharing visual similarity with TB.", color: "#ED4245" },
              { label: "Low-Conf Correct", desc: "Normal scans with unusual anatomy or portable technique challenge the model.", color: "#FEE75C" },
            ].map((item, i) => (
              <div key={item.label} className="p-4 space-y-1.5" style={{ borderLeft: i > 0 ? "1px solid rgba(255,255,255,0.06)" : "none" }}>
                <p className="text-[10px] font-bold uppercase tracking-wide" style={{ color: item.color }}>{item.label}</p>
                <p className="text-[11px] leading-relaxed" style={{ color: "#949BA4" }}>{item.desc}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Sample Predictions */}
        <div className="rounded-lg overflow-hidden" style={{ background: "#2B2D31", border: "1px solid rgba(255,255,255,0.06)" }}>
          <div className="px-5 py-3.5" style={{ borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
            <p className="text-sm font-bold text-white">Sample Predictions on Held-Out Test Cases</p>
            <p className="text-[11px] mt-0.5" style={{ color: "#949BA4" }}>Row 1: Normal correct · Row 2: TB detected · Row 3: High-confidence errors · Row 4: Low-confidence correct</p>
          </div>
          <div className="p-3" style={{ background: "#1E1F22" }}>
            <img src="/model-results/sample_predictions.png" alt="Sample Predictions" className="w-full rounded object-contain" />
          </div>
        </div>

        {/* Teacher vs Student */}
        <div className="grid grid-cols-1 lg:grid-cols-5 gap-4">
          <div className="lg:col-span-3 rounded-lg overflow-hidden" style={{ background: "#2B2D31", border: "1px solid rgba(255,255,255,0.06)" }}>
            <div className="px-5 py-3.5" style={{ borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
              <p className="text-sm font-bold text-white">Deployment Performance Metrics</p>
              <p className="text-[11px] mt-0.5" style={{ color: "#949BA4" }}>Teacher (ResNet-50) vs Student (DenseNet-121)</p>
            </div>
            <div className="p-3" style={{ background: "#1E1F22" }}>
              <img src="/model-results/deployment_metrics.png" alt="Deployment Metrics" className="w-full rounded object-contain" />
            </div>
          </div>

          <div className="lg:col-span-2 rounded-lg overflow-hidden flex flex-col" style={{ background: "#2B2D31", border: "1px solid rgba(255,255,255,0.06)" }}>
            <div className="px-5 py-3.5" style={{ borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
              <p className="text-sm font-bold text-white">Knowledge Distillation</p>
              <p className="text-[11px] mt-0.5" style={{ color: "#949BA4" }}>Student keeps 99.4% of teacher AUC at 1/16th the size</p>
            </div>
            <div className="flex-1 p-5 flex flex-col justify-center gap-4">
              {[
                { label: "Model Size", teacher: "471.9 MB", student: "29.7 MB", win: true, note: "16× smaller" },
                { label: "Parameters", teacher: "23.6M", student: "7.0M", win: true, note: "3.3× fewer" },
                { label: "AUC-ROC", teacher: "0.947", student: "0.941", win: false, note: "−0.6% only" },
                { label: "Accuracy", teacher: "67.0%", student: "93.4%", win: true, note: "+26.4%" },
                { label: "Inference", teacher: "7.8 ms", student: "15.5 ms", win: false, note: "CPU-only" },
              ].map(row => (
                <div key={row.label} className="flex items-center justify-between text-xs">
                  <span className="w-20" style={{ color: "#949BA4" }}>{row.label}</span>
                  <span className="font-mono line-through text-[10px]" style={{ color: "#4E5058" }}>{row.teacher}</span>
                  <span className="font-mono font-bold" style={{ color: row.win ? "#57F287" : "#FEE75C" }}>{row.student}</span>
                  <span className="px-2 py-0.5 rounded text-[10px] font-bold" style={{ background: row.win ? "rgba(87,242,135,0.1)" : "rgba(254,231,92,0.1)", color: row.win ? "#57F287" : "#FEE75C" }}>
                    {row.note}
                  </span>
                </div>
              ))}
            </div>
            <div className="px-5 py-2.5 text-[10px]" style={{ borderTop: "1px solid rgba(255,255,255,0.06)", color: "#949BA4" }}>
              Runs entirely on CPU — no GPU required for deployment.
            </div>
          </div>
        </div>
      </section>

      {/* ── ABOUT PROJECT COLLAPSIBLE ── */}
      <section className="max-w-6xl mx-auto px-6 pb-20 w-full space-y-4">
        {/* About */}
        <div className="rounded-lg overflow-hidden" style={{ background: "#2B2D31", border: "1px solid rgba(255,255,255,0.06)" }}>
          <button
            onClick={() => setAboutOpen(!aboutOpen)}
            className="w-full px-5 py-4 flex items-center justify-between cursor-pointer transition-colors"
            style={{ background: "transparent" }}
            onMouseEnter={e => (e.currentTarget.style.background = "rgba(255,255,255,0.02)")}
            onMouseLeave={e => (e.currentTarget.style.background = "transparent")}
          >
            <div className="flex items-center gap-3">
              <BookOpen className="w-4 h-4" style={{ color: "#5865F2" }} />
              <div className="text-left">
                <p className="text-sm font-semibold text-white">About This Project</p>
                <p className="text-[11px]" style={{ color: "#949BA4" }}>Clinical objectives, training pipeline, and validation metrics</p>
              </div>
            </div>
            <ChevronDown className={`w-4 h-4 transition-transform duration-200 ${aboutOpen ? "rotate-180" : ""}`} style={{ color: "#949BA4" }} />
          </button>

          {aboutOpen && (
            <div className="px-5 pb-6 space-y-6 animate-fadein" style={{ borderTop: "1px solid rgba(255,255,255,0.06)" }}>
              {/* Training Pipeline */}
              <div className="space-y-3 pt-5">
                <p className="text-[10px] font-bold uppercase tracking-widest" style={{ color: "#5865F2" }}>Training Pipeline</p>
                {[
                  { phase: "Phase A — Teacher (ResNet-50)", details: "Large network trained on Shenzhen + Montgomery global datasets capturing general pulmonary structures." },
                  { phase: "Phase B — Student (DenseNet-121)", details: "~7M parameter architecture trained via knowledge distillation to mimic teacher logits. 3× smaller, CPU-optimized." },
                  { phase: "Phase C — Indian Domain Adaptation (NIRT)", details: "Fine-tuned on NIRT Chennai dataset, calibrating for local scanner contrast and visual noise patterns." },
                ].map((p, i) => (
                  <div key={i} className="flex gap-4 p-4 rounded-lg" style={{ background: "#313338", border: "1px solid rgba(255,255,255,0.05)" }}>
                    <div className="w-6 h-6 rounded-full flex items-center justify-center text-[11px] font-bold flex-shrink-0 mt-0.5" style={{ background: "rgba(88,101,242,0.2)", color: "#5865F2" }}>
                      {i + 1}
                    </div>
                    <div className="space-y-1">
                      <p className="text-xs font-semibold text-white">{p.phase}</p>
                      <p className="text-xs leading-relaxed" style={{ color: "#949BA4" }}>{p.details}</p>
                    </div>
                  </div>
                ))}
              </div>

              {/* Metrics Table */}
              <div className="space-y-3">
                <p className="text-[10px] font-bold uppercase tracking-widest" style={{ color: "#5865F2" }}>Validated Metrics (Unseen Test Set)</p>
                <div className="rounded-lg overflow-hidden" style={{ border: "1px solid rgba(255,255,255,0.06)" }}>
                  <table className="w-full text-xs">
                    <thead>
                      <tr style={{ background: "rgba(255,255,255,0.03)", borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
                        {["Metric", "Value", "Clinical Significance"].map(h => (
                          <th key={h} className="p-3 text-left font-semibold" style={{ color: "#949BA4" }}>{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {[
                        ["AUC-ROC", "94.1%", "High classifier discrimination on unseen cases"],
                        ["TB Recall (Sensitivity)", "81.7%", "Active TB cases flagged for confirmatory testing"],
                        ["Specificity", "96.7%", "Normal cases correctly cleared, minimizing false alarms"],
                        ["Overall Accuracy", "93.4%", "Consistent overall prediction on held-out test set"],
                        ["F1 Score (TB)", "0.844", "Harmonic mean of precision and recall for TB class"],
                      ].map(([m, v, s], i) => (
                        <tr key={m} style={{ borderTop: i > 0 ? "1px solid rgba(255,255,255,0.04)" : "none" }}>
                          <td className="p-3 font-medium text-white">{m}</td>
                          <td className="p-3 font-mono font-bold" style={{ color: "#5865F2" }}>{v}</td>
                          <td className="p-3" style={{ color: "#949BA4" }}>{s}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Limitations */}
              <div className="space-y-3">
                <p className="text-[10px] font-bold uppercase tracking-widest" style={{ color: "#FEE75C" }}>Known Limitations</p>
                {[
                  { title: "Confirmatory Testing Required", text: "Positive findings must be verified with sputum smear, GeneXpert, or molecular assays." },
                  { title: "No Automated Decisions", text: "All verdicts require a signed override by a certified radiologist or supervising physician." },
                  { title: "Single-Site Generalizability", text: "Fine-tuned on NIRT cohort. Scans from different hardware/institutions may vary in results." },
                ].map((w, i) => (
                  <div key={i} className="flex items-start gap-3 p-4 rounded-lg" style={{ background: "rgba(254,231,92,0.05)", border: "1px solid rgba(254,231,92,0.12)" }}>
                    <AlertTriangle className="w-4 h-4 flex-shrink-0 mt-0.5" style={{ color: "#FEE75C" }} />
                    <div>
                      <p className="text-xs font-semibold" style={{ color: "#FEE75C" }}>{w.title}</p>
                      <p className="text-xs mt-0.5 leading-relaxed" style={{ color: "#949BA4" }}>{w.text}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Tech Stack */}
        <div className="rounded-lg overflow-hidden" style={{ background: "#2B2D31", border: "1px solid rgba(255,255,255,0.06)" }}>
          <button
            onClick={() => setHighlightsOpen(!highlightsOpen)}
            className="w-full px-5 py-4 flex items-center justify-between cursor-pointer transition-colors"
            style={{ background: "transparent" }}
            onMouseEnter={e => (e.currentTarget.style.background = "rgba(255,255,255,0.02)")}
            onMouseLeave={e => (e.currentTarget.style.background = "transparent")}
          >
            <div className="flex items-center gap-3">
              <Cpu className="w-4 h-4" style={{ color: "#5865F2" }} />
              <div className="text-left">
                <p className="text-sm font-semibold text-white">Tech Stack & Specifications</p>
                <p className="text-[11px]" style={{ color: "#949BA4" }}>Model details and technology used</p>
              </div>
            </div>
            <ChevronDown className={`w-4 h-4 transition-transform duration-200 ${highlightsOpen ? "rotate-180" : ""}`} style={{ color: "#949BA4" }} />
          </button>

          {highlightsOpen && (
            <div className="px-5 pb-5 animate-fadein" style={{ borderTop: "1px solid rgba(255,255,255,0.06)" }}>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-5">
                <div className="space-y-3">
                  <p className="text-[10px] font-bold uppercase tracking-widest" style={{ color: "#5865F2" }}>Model Specs</p>
                  {[
                    ["Architecture", "DenseNet-121 (Student)"],
                    ["Parameters", "~7.04 Million"],
                    ["Model Size", "29.7 MB"],
                    ["Inference", "~15.5 ms/image (CPU)"],
                    ["AUC-ROC", "0.941"],
                    ["Best Threshold", "0.87"],
                    ["Validation Cohort", "NIRT Chennai, India"],
                  ].map(([k, v]) => (
                    <div key={k} className="flex justify-between items-center text-xs">
                      <span style={{ color: "#949BA4" }}>{k}</span>
                      <span className="font-mono font-semibold text-white">{v}</span>
                    </div>
                  ))}
                </div>
                <div className="space-y-3">
                  <p className="text-[10px] font-bold uppercase tracking-widest" style={{ color: "#5865F2" }}>Technology</p>
                  {[
                    ["Backend", "Flask 3.1 + Python 3.14"],
                    ["ML Engine", "PyTorch 2.x + Keras 3.8"],
                    ["Imaging", "pydicom + OpenCV + Pillow"],
                    ["Frontend", "Next.js 16 + React 19"],
                    ["Storage", "SQLite3 Database"],
                    ["Deployment", "Vercel + Hugging Face Spaces"],
                    ["XAI", "Grad-CAM + LIME + SHAP"],
                  ].map(([k, v]) => (
                    <div key={k} className="flex justify-between items-center text-xs">
                      <span style={{ color: "#949BA4" }}>{k}</span>
                      <span className="font-mono font-semibold text-white">{v}</span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="mt-5 pt-4" style={{ borderTop: "1px solid rgba(255,255,255,0.06)" }}>
                <button
                  onClick={() => {
                    if (canLaunch) router.push("/diagnose");
                    else document.getElementById("medical-disclaimer")?.scrollIntoView({ behavior: "smooth", block: "center" });
                  }}
                  className="w-full flex items-center justify-center gap-2 py-2.5 rounded font-semibold text-sm text-white transition-all cursor-pointer"
                  style={{ background: "#5865F2" }}
                  onMouseEnter={e => { (e.currentTarget as HTMLButtonElement).style.background = "#4752C4"; }}
                  onMouseLeave={e => { (e.currentTarget as HTMLButtonElement).style.background = "#5865F2"; }}
                >
                  Launch Screening Workstation <ArrowRight className="w-4 h-4" />
                </button>
              </div>
            </div>
          )}
        </div>
      </section>

      {/* ── FOOTER ── */}
      <footer className="mt-auto py-6 px-6" style={{ background: "#2B2D31", borderTop: "1px solid rgba(255,255,255,0.06)" }}>
        <div className="max-w-6xl mx-auto flex flex-col sm:flex-row justify-between items-center gap-3 text-xs" style={{ color: "#949BA4" }}>
          <p>© {new Date().getFullYear()} Nirikhshon — Academic Final Year Project Prototype</p>
          <p>Built with Next.js, Flask, and DenseNet-121</p>
        </div>
      </footer>
    </div>
  );
}
