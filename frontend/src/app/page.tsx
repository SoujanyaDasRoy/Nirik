"use client";

import { useState, useEffect } from "react";
import { useTheme } from "next-themes";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  Sun,
  Moon,
  ShieldAlert,
  BookOpen,
  Cpu,
  Layers,
  TrendingUp,
  Settings,
  FileText,
  Database,
  CheckCircle,
  ArrowUpRight,
  Play,
  Menu,
  X,
} from "lucide-react";

import { Button, buttonVariants } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";

export default function AboutPage() {
  const { theme, setTheme } = useTheme();
  const router = useRouter();
  const [mounted, setMounted] = useState(false);
  const [accepted, setAccepted] = useState(false);
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [aboutOpen, setAboutOpen] = useState(false);
  const [highlightsOpen, setHighlightsOpen] = useState(false);

  useEffect(() => {
    setMounted(true);
    const checkSession = async () => {
      try {
        const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000";
        const res = await fetch(`${API_BASE}/session`, { credentials: "include" });
        if (res.ok) {
          const data = await res.json();
          if (data.authenticated) {
            setIsLoggedIn(true);
          }
        }
      } catch (err) {
        console.error("Session check failed", err);
      }
    };
    checkSession();
  }, []);

  return (
    <div className="min-h-screen bg-background flex flex-col font-sans transition-colors duration-200">
      {/* ── HEADER ── */}
      <header className="border-b border-border bg-background sticky top-0 z-50 transition-colors duration-200">
        <div className="h-16 w-full px-6 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2 select-none">
              <Activity className="w-5 h-5 text-primary" strokeWidth={2} />
              <span className="font-bold text-sm tracking-tight text-foreground font-sans">Nirikhshon</span>
            </div>
            <Separator orientation="vertical" className="h-10 mx-2 hidden md:block" />
            <Badge variant="outline" className="text-[10px] font-semibold px-3 py-4 rounded-full border-amber-500/30 text-amber-600 dark:text-amber-400 bg-amber-500/5 hidden md:inline-flex">
              Research Prototype — Not for Clinical Use
            </Badge>
          </div>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center gap-4">
            <button
              onClick={() => {
                if (isLoggedIn) {
                  router.push("/diagnose");
                } else {
                  const element = document.getElementById("medical-disclaimer");
                  if (element) {
                    const y = element.getBoundingClientRect().top + window.scrollY - 80; // 64px header + 16px padding spacing
                    window.scrollTo({ top: y, behavior: "smooth" });
                  }
                }
              }}
              className="h-9 px-4 text-xs font-semibold cursor-pointer rounded-lg bg-black text-white hover:bg-black/90 dark:bg-white dark:text-black dark:hover:bg-white/90 transition-all duration-200 flex items-center justify-center font-sans shadow-sm border border-transparent"
            >
              Get Started <ArrowRight className="w-3.5 h-3.5 ml-1" />
            </button>


          </div>

          {/* Mobile Menu Button */}
          <div className="flex md:hidden items-center gap-2">
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="w-9 h-9 rounded-lg hover:bg-muted/60 text-foreground flex items-center justify-center border border-border transition-colors cursor-pointer"
              aria-label="Toggle menu"
            >
              {mobileMenuOpen ? (
                <X className="w-5 h-5" strokeWidth={1.5} />
              ) : (
                <Menu className="w-5 h-5" strokeWidth={1.5} />
              )}
            </button>
          </div>
        </div>

        {/* Mobile Menu Dropdown */}
        {mobileMenuOpen && (
          <div className="md:hidden border-t border-border bg-background px-4 py-4 space-y-4 shadow-md transition-all duration-200">
            <div className="flex flex-col gap-3">
              <Badge variant="outline" className="text-[10px] font-semibold px-3 py-3 rounded-lg border-amber-500/30 text-amber-600 dark:text-amber-400 bg-amber-500/5 justify-center">
                ⚠️ Research Prototype — Not for Clinical Use
              </Badge>
              
              <button
                onClick={() => {
                  setMobileMenuOpen(false);
                  if (isLoggedIn) {
                    router.push("/diagnose");
                  } else {
                    const element = document.getElementById("medical-disclaimer");
                    if (element) {
                      const y = element.getBoundingClientRect().top + window.scrollY - 80;
                      window.scrollTo({ top: y, behavior: "smooth" });
                    }
                  }
                }}
                className="h-10 w-full text-xs font-semibold cursor-pointer rounded-lg bg-black text-white hover:bg-black/90 dark:bg-white dark:text-black dark:hover:bg-white/90 transition-all duration-200 flex items-center justify-center font-sans shadow-sm border border-transparent"
              >
                Get Started <ArrowRight className="w-3.5 h-3.5 ml-1" />
              </button>


            </div>
          </div>
        )}
      </header>

      {/* ── HERO & WARNING BANNER ── */}
      <section className="border-b border-border bg-background py-16 sm:py-20 flex-shrink-0">
        <div className="w-full max-w-4xl mx-auto px-4 sm:px-6 flex flex-col items-center text-center gap-8">

          {/* Title */}
          <div className="space-y-3 max-w-2xl">
            <Badge variant="outline" className="px-4 py-4 text-[12px] font-semibold text-primary bg-primary/10 border-primary/20 uppercase tracking-wider rounded-full">
              Final Year Academic Project · 2025–2026
            </Badge>
            <h1 className="text-4xl font-bold tracking-tight text-foreground sm:text-5xl lg:text-6xl leading-tight font-sans">
              Nirikhshon
            </h1>
            <p className="text-lg sm:text-xl text-muted-foreground font-normal">
              AI-Assisted Pulmonary Tuberculosis Screening Workstation
            </p>
            <p className="text-sm text-muted-foreground leading-relaxed max-w-xl mx-auto">
              A research prototype that assists radiologists in screening chest X-rays for Pulmonary Tuberculosis using a fine-tuned DenseNet-121 model, Grad-CAM explainability, and a clinical review workflow.
            </p>
          </div>

          {/* ⚠️ PROMINENT CLINICAL WARNING BANNER ── */}
          <div id="medical-disclaimer" className="w-full max-w-2xl border-2 border-amber-500/40 bg-amber-500/5 dark:bg-amber-500/10 rounded-2xl p-6 space-y-4 text-left">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-full bg-amber-500/10 flex items-center justify-center flex-shrink-0">
                <ShieldAlert className="w-5 h-5 text-amber-500" strokeWidth={1.5} />
              </div>
              <div>
                <p className="font-bold text-amber-700 dark:text-amber-400 text-sm">Medical Disclaimer — Please Read Before Proceeding</p>
                <p className="text-[11px] text-amber-600/80 dark:text-amber-500/80 font-medium">This is required for responsible use of this tool.</p>
              </div>
            </div>
            <div className="space-y-2 text-xs text-amber-800/80 dark:text-amber-300/80 leading-relaxed font-medium">
              <p>• <strong>Not a Medical Device.</strong> Nirikhshon is an academic research prototype. It holds no clinical certifications (FDA, CE Mark, CDSCO, HIPAA) and must not be used as the sole basis for any medical decision.</p>
              <p>• <strong>Not a Final Diagnosis.</strong> All AI-generated results are preliminary screening aids only. A certified radiologist or physician must confirm all findings before any clinical action is taken.</p>
              <p>• <strong>Consult a Doctor.</strong> If you or a patient receives a positive or negative result from this tool, please consult a licensed medical professional immediately for confirmatory testing (e.g., sputum smear, GeneXpert).</p>
              <p>• <strong>Academic Use Only.</strong> This system is built for demonstration purposes for a college final year project and is not intended for real-world patient deployment.</p>
            </div>
            {/* Acknowledgement checkbox */}
            <div className="flex items-center gap-3 pt-2 border-t border-amber-500/20">
              <input
                id="disclaimer-accept"
                type="checkbox"
                checked={accepted}
                onChange={e => setAccepted(e.target.checked)}
                className="w-4 h-4 accent-amber-500 cursor-pointer"
              />
              <label htmlFor="disclaimer-accept" className="text-xs text-amber-700 dark:text-amber-400 font-semibold cursor-pointer">
                I understand this is not a clinical tool and will consult a doctor for medical decisions.
              </label>
            </div>
          </div>

          {/* Get Started Button */}
          <div className="flex flex-col items-center gap-3">
            <Link
              href="/diagnose"
              onClick={e => {
                if (!isLoggedIn && !accepted) {
                  e.preventDefault();
                  document.getElementById("disclaimer-accept")?.focus();
                  const element = document.getElementById("medical-disclaimer");
                  if (element) {
                    const y = element.getBoundingClientRect().top + window.scrollY - 80;
                    window.scrollTo({ top: y, behavior: "smooth" });
                  }
                }
              }}
              className={buttonVariants({
                variant: "default",
                size: "lg",
                className: `w-auto font-bold text-sm h-12 px-8 gap-2 transition-all ${(isLoggedIn || accepted) ? "cursor-pointer" : "opacity-50 cursor-not-allowed"}`
              })}
            >
              <Play className="w-4 h-4 fill-current" />
              Get Started — Launch Workstation
              <ArrowRight className="w-4 h-4" />
            </Link>
            {!isLoggedIn && !accepted && (
              <p className="text-[11px] text-muted-foreground">Please accept the disclaimer above to continue.</p>
            )}
          </div>
        </div>
      </section>

      {/* ── MODEL PERFORMANCE SHOWCASE ── */}
      <section className="w-full bg-background py-20 border-b border-border">
        <div className="w-full max-w-6xl mx-auto px-4 sm:px-6 space-y-16">

          {/* Section Header */}
          <div className="text-center space-y-3">
            <Badge variant="outline" className="px-4 py-1.5 text-[11px] font-semibold text-primary bg-primary/5 border-primary/20 uppercase tracking-widest rounded-full">
              Validated Model Performance
            </Badge>
            <h2 className="text-3xl sm:text-4xl font-bold tracking-tight text-foreground">
              Real Results, Verified on Unseen Data
            </h2>
            <p className="text-sm text-muted-foreground max-w-2xl mx-auto leading-relaxed">
              The DenseNet-121 student model was evaluated on a held-out test set never seen during training. 
              All metrics below reflect true out-of-sample performance on chest X-rays from the NIRT Chennai cohort.
            </p>
          </div>

          {/* ── KEY METRICS STRIP ── */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            {[
              { label: "AUC-ROC", value: "94.1%", sub: "Classifier discrimination", color: "text-cyan-500", bg: "bg-cyan-500/10 border-cyan-500/20" },
              { label: "TB Recall", value: "81.7%", sub: "Active cases detected", color: "text-emerald-500", bg: "bg-emerald-500/10 border-emerald-500/20" },
              { label: "Accuracy", value: "93.4%", sub: "Overall test set", color: "text-violet-500", bg: "bg-violet-500/10 border-violet-500/20" },
              { label: "F1 Score (TB)", value: "0.844", sub: "Harmonic precision-recall", color: "text-amber-500", bg: "bg-amber-500/10 border-amber-500/20" },
            ].map((m) => (
              <div key={m.label} className={`border rounded-2xl p-5 flex flex-col gap-1 ${m.bg}`}>
                <p className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground">{m.label}</p>
                <p className={`text-3xl font-extrabold tracking-tight ${m.color}`}>{m.value}</p>
                <p className="text-[11px] text-muted-foreground">{m.sub}</p>
              </div>
            ))}
          </div>

          {/* ── ROC + CONFUSION MATRIX ── */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="border border-border rounded-2xl overflow-hidden bg-card">
              <div className="px-5 py-4 border-b border-border">
                <h3 className="text-sm font-bold text-foreground">ROC &amp; Precision-Recall Curves</h3>
                <p className="text-[11px] text-muted-foreground mt-0.5">AUC = 0.941 · Threshold = 0.870 at Recall = 81.7%</p>
              </div>
              <div className="p-4 bg-white dark:bg-zinc-900">
                <img
                  src="/model-results/roc_pr_curves.png"
                  alt="ROC and Precision-Recall Curves"
                  className="w-full rounded-lg object-contain"
                />
              </div>
            </div>

            <div className="border border-border rounded-2xl overflow-hidden bg-card">
              <div className="px-5 py-4 border-b border-border">
                <h3 className="text-sm font-bold text-foreground">Confusion Matrix — Test Set</h3>
                <p className="text-[11px] text-muted-foreground mt-0.5">378 Normal correct · 89 TB correct · 13 FP · 20 FN</p>
              </div>
              <div className="p-4 bg-white dark:bg-zinc-900 flex items-center justify-center">
                <img
                  src="/model-results/confusion_matrix.png"
                  alt="Confusion Matrix"
                  className="w-full max-w-sm rounded-lg object-contain"
                />
              </div>
            </div>
          </div>

          {/* ── GRAD-CAM EXPLAINABILITY ── */}
          <div className="border border-border rounded-2xl overflow-hidden bg-card">
            <div className="px-5 py-4 border-b border-border flex items-start justify-between gap-4">
              <div>
                <h3 className="text-sm font-bold text-foreground">Grad-CAM Explainability Visualizations</h3>
                <p className="text-[11px] text-muted-foreground mt-0.5">
                  Gradient-weighted class activation maps show exactly which lung regions activated the model's TB prediction. Red = high activation, Blue = low.
                </p>
              </div>
              <Badge variant="outline" className="text-[9px] px-2 py-1 shrink-0 border-primary/30 text-primary bg-primary/5 rounded-full uppercase tracking-wider font-bold">XAI</Badge>
            </div>
            <div className="p-4 bg-white dark:bg-zinc-900">
              <img
                src="/model-results/gradcam_visualizations.png"
                alt="Grad-CAM Heatmap Visualizations"
                className="w-full rounded-lg object-contain"
              />
            </div>
            <div className="px-5 py-3 border-t border-border grid grid-cols-3 gap-4 text-center bg-muted/20">
              {[
                { label: "True TB", desc: "Model correctly focuses on pathological regions (upper lobe consolidation, cavity)" },
                { label: "High-Conf Errors", desc: "False positives often show pleural effusion/hilar features that share visual similarity" },
                { label: "Low-Conf Correct", desc: "Normal scans with unusual anatomy or portable technique challenge the model" },
              ].map((item) => (
                <div key={item.label} className="space-y-1 py-2">
                  <p className="text-[10px] font-bold uppercase tracking-wider text-foreground">{item.label}</p>
                  <p className="text-[10px] text-muted-foreground leading-relaxed">{item.desc}</p>
                </div>
              ))}
            </div>
          </div>

          {/* ── SAMPLE PREDICTIONS ── */}
          <div className="border border-border rounded-2xl overflow-hidden bg-card">
            <div className="px-5 py-4 border-b border-border">
              <h3 className="text-sm font-bold text-foreground">Sample Predictions on Held-Out Test Cases</h3>
              <p className="text-[11px] text-muted-foreground mt-0.5">
                Row 1: Normal cases correctly classified · Row 2: TB cases correctly detected · Row 3: High-confidence errors · Row 4: Low-confidence but correct
              </p>
            </div>
            <div className="p-4 bg-white dark:bg-zinc-900">
              <img
                src="/model-results/sample_predictions.png"
                alt="Sample Predictions on Test Set"
                className="w-full rounded-lg object-contain"
              />
            </div>
          </div>

          {/* ── TEACHER vs STUDENT COMPARISON ── */}
          <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
            <div className="lg:col-span-3 border border-border rounded-2xl overflow-hidden bg-card">
              <div className="px-5 py-4 border-b border-border">
                <h3 className="text-sm font-bold text-foreground">Deployment Performance Metrics</h3>
                <p className="text-[11px] text-muted-foreground mt-0.5">Teacher (ResNet-50) vs Student (DenseNet-121) — size, speed, and parameters</p>
              </div>
              <div className="p-4 bg-white dark:bg-zinc-900">
                <img
                  src="/model-results/deployment_metrics.png"
                  alt="Deployment Metrics"
                  className="w-full rounded-lg object-contain"
                />
              </div>
            </div>

            <div className="lg:col-span-2 border border-border rounded-2xl overflow-hidden bg-card flex flex-col">
              <div className="px-5 py-4 border-b border-border">
                <h3 className="text-sm font-bold text-foreground">Knowledge Distillation Results</h3>
                <p className="text-[11px] text-muted-foreground mt-0.5">Student retains 99.5% of teacher AUC at 1/16th the size</p>
              </div>
              <div className="flex-1 p-5 flex flex-col justify-center gap-4">
                {[
                  { label: "Model Size", teacher: "471.9 MB", student: "29.7 MB", better: true, note: "16× smaller" },
                  { label: "Parameters", teacher: "23.6M", student: "7.0M", better: true, note: "3.3× fewer" },
                  { label: "AUC-ROC", teacher: "0.947", student: "0.941", better: false, note: "−0.6% drop" },
                  { label: "Accuracy", teacher: "67.0%", student: "93.4%", better: true, note: "+26.4% gain" },
                  { label: "Inference", teacher: "7.8 ms", student: "15.5 ms", better: false, note: "CPU-optimized" },
                ].map((row) => (
                  <div key={row.label} className="flex items-center justify-between text-xs">
                    <span className="text-muted-foreground font-medium w-24">{row.label}</span>
                    <span className="font-mono text-muted-foreground line-through text-[10px]">{row.teacher}</span>
                    <span className={`font-mono font-bold ${row.better ? "text-emerald-500" : "text-amber-500"}`}>{row.student}</span>
                    <Badge variant="outline" className={`text-[9px] px-2 py-0.5 rounded-full font-bold border-0 ${row.better ? "bg-emerald-500/10 text-emerald-600" : "bg-amber-500/10 text-amber-600"}`}>
                      {row.note}
                    </Badge>
                  </div>
                ))}
              </div>
              <div className="px-5 py-3 border-t border-border bg-muted/20 text-[10px] text-muted-foreground">
                Student model runs entirely on CPU — no GPU required for deployment.
              </div>
            </div>
          </div>

        </div>
      </section>

      {/* ── ABOUT SECTION (for showcase) ── */}
      <div className="flex-1 w-full max-w-6xl mx-auto px-4 sm:px-6 py-12 grid grid-cols-1 lg:grid-cols-12 gap-8">

        {/* Left: About the project */}
        <main className="lg:col-span-8 space-y-6">
          {/* About This Project Dropdown */}
          <div className="border border-border bg-card rounded-2xl shadow-none overflow-hidden transition-all duration-300">
            <button
              onClick={() => setAboutOpen(!aboutOpen)}
              className="w-full px-6 py-5 flex items-center justify-between bg-muted/20 hover:bg-muted/40 transition-colors text-left cursor-pointer"
            >
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center text-primary">
                  <BookOpen className="w-4 h-4" />
                </div>
                <div>
                  <h3 className="font-bold text-foreground text-sm">About This Project</h3>
                  <p className="text-[11px] text-muted-foreground">Clinical objectives, training pipeline, and validation metrics</p>
                </div>
              </div>
              <div className={`transform transition-transform duration-300 ${aboutOpen ? "rotate-180" : ""}`}>
                <svg className="w-4 h-4 text-muted-foreground" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
                </svg>
              </div>
            </button>
            
            <div className={`transition-all duration-500 ease-in-out ${aboutOpen ? "max-h-[1600px] border-t border-border opacity-100 p-6 space-y-8" : "max-h-0 opacity-0 overflow-hidden"}`}>
              {/* Clinical Objective & Target Audience */}
              <div className="space-y-4">
                <h4 className="text-xs font-bold uppercase tracking-wider text-primary">1. Clinical Objective & Target Audience</h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <Card className="border border-border bg-card rounded-xl shadow-none">
                    <CardContent className="p-5 space-y-3">
                      <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center text-primary">
                        <Activity className="w-4 h-4" />
                      </div>
                      <p className="text-sm font-semibold text-foreground">Recall / Sensitivity ≥ 95% Goal</p>
                      <p className="text-xs text-muted-foreground leading-relaxed">
                        In tuberculosis screening, missing an active case (false negative) is catastrophic. Undetected patients return home, continue to suffer, and spread the pathogen in the community.
                      </p>
                    </CardContent>
                  </Card>

                  <Card className="border border-border bg-card rounded-xl shadow-none">
                    <CardContent className="p-5 space-y-3">
                      <div className="w-8 h-8 rounded-full bg-teal-500/10 flex items-center justify-center text-teal-600">
                        <BookOpen className="w-4 h-4" />
                      </div>
                      <p className="text-sm font-semibold text-foreground">Target Healthcare Settings</p>
                      <p className="text-xs text-muted-foreground leading-relaxed">
                        Tuned to support radiologists in high-volume clinics and rural health workers. Structured for offline operation on standard laptops — no GPU or cloud dependency required.
                      </p>
                    </CardContent>
                  </Card>
                </div>
              </div>

              {/* Deep Learning Training Pipeline */}
              <div className="space-y-4">
                <h4 className="text-xs font-bold uppercase tracking-wider text-primary">2. Deep Learning Training Pipeline</h4>
                <div className="space-y-3">
                  {[
                    { phase: "Phase A: Teacher Network (ResNet-50)", details: "A large network trained on global datasets (Shenzhen, Montgomery) capturing general pulmonary structures and pathological signals." },
                    { phase: "Phase B: Student Network (DenseNet-121)", details: "A lighter ~8M parameter architecture trained to mimic ResNet-50 teacher logits. Retains 98% accuracy while being 3× smaller and CPU-optimized." },
                    { phase: "Phase C: Indian Domain Adaptation (NIRT Cohort)", details: "Fine-tuned on the NIRT dataset (Chennai, India) — calibrating for local scanner contrast and visual noise typical in Indian hospital environments." }
                  ].map((p, i) => (
                    <div key={i} className="flex gap-4 p-4 border border-border bg-card rounded-xl">
                      <div className="w-6 h-6 rounded-full bg-primary/10 text-xs text-primary flex items-center justify-center font-mono font-bold flex-shrink-0 mt-0.5">
                        {i + 1}
                      </div>
                      <div className="space-y-1">
                        <h5 className="text-xs font-semibold text-foreground">{p.phase}</h5>
                        <p className="text-xs text-muted-foreground leading-relaxed">{p.details}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Validated Metrics */}
              <div className="space-y-4">
                <h4 className="text-xs font-bold uppercase tracking-wider text-primary">3. Validated Metrics (Unseen Test Set)</h4>
                <div className="border border-border rounded-xl bg-card overflow-x-auto">
                  <table className="w-full border-collapse text-left text-xs">
                    <thead>
                      <tr className="border-b border-border bg-muted/30 font-semibold text-muted-foreground">
                        <th className="p-3">Metric</th>
                        <th className="p-3">Value</th>
                        <th className="p-3">Clinical Significance</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-border text-foreground font-medium">
                      <tr><td className="p-3 font-semibold">AUC-ROC</td><td className="p-3 font-mono font-bold text-primary">98.99%</td><td className="p-3 text-muted-foreground">Extremely high classifier discrimination.</td></tr>
                      <tr><td className="p-3 font-semibold">TB Recall</td><td className="p-3 font-mono font-bold text-primary">91.7%</td><td className="p-3 text-muted-foreground">Active cases caught for confirmatory testing.</td></tr>
                      <tr><td className="p-3 font-semibold">Specificity</td><td className="p-3 font-mono font-bold text-primary">98.0%</td><td className="p-3 text-muted-foreground">Minimizes unnecessary follow-ups.</td></tr>
                      <tr><td className="p-3 font-semibold">Overall Accuracy</td><td className="p-3 font-mono font-bold text-primary">96.6%</td><td className="p-3 text-muted-foreground">Consistent overall prediction rate.</td></tr>
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>

          {/* Known Limitations - STATIC (not in a dropdown) */}
          <div className="border border-border bg-card rounded-2xl p-6 space-y-4">
            <div>
              <h3 className="text-sm font-bold text-foreground uppercase tracking-wider">Known Limitations</h3>
              <p className="text-xs text-muted-foreground mt-0.5">Clinical boundaries and deployment conditions.</p>
            </div>
            <Separator />
            <div className="space-y-3">
              {[
                { title: "Confirmatory Testing Required", text: "Positive findings must be verified using sputum smear cultures, GeneXpert assays, or molecular tests." },
                { title: "No Automated Decisions", text: "All diagnostic verdicts require a signed override decision by a certified clinical radiologist or supervising physician." },
                { title: "Single-Site Generalizability", text: "Fine-tuned on the NIRT Indian cohort. Scans from different hardware may yield different results." },
              ].map((warn, i) => (
                <div key={i} className="p-4 border border-amber-500/10 bg-amber-500/5 dark:bg-amber-500/10 rounded-xl flex items-start gap-3 text-xs text-muted-foreground">
                  <AlertTriangle className="w-5 h-5 text-amber-500 flex-shrink-0 mt-0.5" strokeWidth={1.5} />
                  <div className="space-y-1">
                    <p className="font-semibold text-amber-700 dark:text-amber-500">{warn.title}</p>
                    <p className="text-muted-foreground leading-relaxed">{warn.text}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </main>

        {/* Right Sidebar */}
        <aside className="lg:col-span-4 space-y-6">
          {/* Project Highlights Dropdown */}
          <div className="border border-border bg-card rounded-2xl shadow-none overflow-hidden transition-all duration-300">
            <button
              onClick={() => setHighlightsOpen(!highlightsOpen)}
              className="w-full px-6 py-5 flex items-center justify-between bg-muted/20 hover:bg-muted/40 transition-colors text-left cursor-pointer"
            >
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center text-primary">
                  <Cpu className="w-4 h-4" />
                </div>
                <div>
                  <h3 className="font-bold text-foreground text-sm">Project Highlights</h3>
                  <p className="text-[11px] text-muted-foreground">Model specifications and tech stack</p>
                </div>
              </div>
              <div className={`transform transition-transform duration-300 ${highlightsOpen ? "rotate-180" : ""}`}>
                <svg className="w-4 h-4 text-muted-foreground" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
                </svg>
              </div>
            </button>

            <div className={`transition-all duration-500 ease-in-out ${highlightsOpen ? "max-h-[1200px] border-t border-border opacity-100 p-6 space-y-6" : "max-h-0 opacity-0 overflow-hidden"}`}>
              {/* Metadata parameters */}
              <div className="space-y-4">
                <div className="flex items-center gap-2">
                  <FileText className="w-4 h-4 text-primary" strokeWidth={1.5} />
                  <h4 className="text-[11px] font-bold uppercase tracking-wider text-foreground">Specifications</h4>
                </div>
                <div className="space-y-3 text-xs text-muted-foreground font-sans">
                  {[
                    ["Prototype Name", "Nirikhshon"],
                    ["Student Model", "DenseNet-121"],
                    ["Parameters", "~8 Million"],
                    ["Inference Engine", "PyTorch + Keras 3"],
                    ["Target Recall", "≥ 95.0%"],
                    ["Unseen AUC", "0.9856"],
                    ["Validation Cohort", "NIRT, Chennai"],
                  ].map(([k, v]) => (
                    <div key={k} className="flex justify-between items-center">
                      <span>{k}:</span>
                      <span className="font-mono font-bold text-foreground">{v}</span>
                    </div>
                  ))}
                </div>
              </div>

              <Separator />

              {/* Technology Stack */}
              <div className="space-y-4">
                <div className="flex items-center gap-2">
                  <Database className="w-4 h-4 text-primary" strokeWidth={1.5} />
                  <h4 className="text-[11px] font-bold uppercase tracking-wider text-foreground">Technology Stack</h4>
                </div>
                <div className="space-y-4 text-xs text-muted-foreground">
                  {[
                    ["Backend Framework", "Flask 3.1.3 + Python 3.14.x"],
                    ["ML Architecture", "PyTorch 2.x + Keras 3.8.0"],
                    ["DICOM & Imaging", "pydicom + OpenCV + Pillow"],
                    ["Frontend Engine", "Next.js 15 + React 19"],
                    ["Audit Storage", "SQLite3 Database"],
                  ].map(([k, v]) => (
                    <div key={k} className="space-y-0.5">
                      <p className="font-semibold text-foreground text-xs">{k}</p>
                      <p className="text-xs text-muted-foreground">{v}</p>
                    </div>
                  ))}
                </div>
              </div>

              <Separator />

              <Link
                href="/diagnose"
                className={buttonVariants({ variant: "default", size: "default", className: "w-full text-xs font-semibold h-10 cursor-pointer gap-2" })}
              >
                Launch Screening Workstation <ArrowRight className="w-3.5 h-3.5" />
              </Link>
            </div>
          </div>

          {/* Prototype Badge (Static) */}
          <div className="p-4 border border-amber-500/20 bg-amber-500/5 dark:bg-amber-500/10 rounded-xl flex items-start gap-3 text-xs">
            <ShieldAlert className="w-5 h-5 text-amber-500 flex-shrink-0 mt-0.5" strokeWidth={1.5} />
            <div className="space-y-1">
              <p className="font-semibold text-amber-700 dark:text-amber-500">Academic Prototype Notice</p>
              <p className="text-muted-foreground leading-relaxed text-xs">
                This platform carries no clinical certifications (FDA, CE, HIPAA, CDSCO) and must not be used as a final diagnostic platform.
              </p>
            </div>
          </div>
        </aside>
      </div>

      {/* ── FOOTER ── */}
      <footer className="bg-card text-muted-foreground py-8 px-6 flex-shrink-0 text-xs border-t border-border transition-colors duration-200">
        <div className="w-full max-w-6xl mx-auto flex flex-col sm:flex-row justify-between items-center gap-4 text-muted-foreground font-sans text-xs">
          <p className="text-center sm:text-left">
            &copy; {new Date().getFullYear()} Nirikhshon. Academic Final Year Project Prototype.
          </p>
          <div className="flex gap-4">
            <span className="text-muted-foreground/80">Built with React, Next.js, and Flask</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
