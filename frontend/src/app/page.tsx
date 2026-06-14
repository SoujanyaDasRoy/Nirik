"use client";

import { useState, useEffect } from "react";
import { useTheme } from "next-themes";
import Link from "next/link";
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
} from "lucide-react";

import { Button, buttonVariants } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";

export default function AboutPage() {
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);
  const [accepted, setAccepted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  return (
    <div className="min-h-screen bg-background flex flex-col font-sans transition-colors duration-200">
      {/* ── HEADER ── */}
      <header className="h-16 border-b border-border bg-background flex items-center flex-shrink-0 sticky top-0 z-50">
        <div className="w-full max-w-6xl mx-auto px-6 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2 select-none">
              <Activity className="w-5 h-5 text-primary" strokeWidth={2} />
              <span className="font-bold text-sm tracking-tight text-foreground font-sans">Nirikhshon</span>
            </div>
            <Separator orientation="vertical" className="h-10 mx-2" />
            <Badge variant="outline" className="text-[10px] font-semibold px-3 py-4 rounded-full border-amber-500/30 text-amber-600 dark:text-amber-400 bg-amber-500/5">
              Research Prototype — Not for Clinical Use
            </Badge>
          </div>

          {/* <div className="flex items-center gap-4">
            <Link
              href="/diagnose"
              className={buttonVariants({ variant: "outline", size: "sm", className: "h-9 px-4 text-xs font-semibold cursor-pointer" })}
            >
              Launch Workspace <ArrowUpRight className="w-3.5 h-3.5 ml-1" />
            </Link>

            <Separator orientation="vertical" className="h-5" />

            <button
              onClick={() => setTheme(mounted && theme === "dark" ? "light" : "dark")}
              className="w-9 h-9 rounded-full bg-muted/40 hover:bg-muted/80 text-foreground flex items-center justify-center border border-border hover:border-primary/50 cursor-pointer transition-colors"
              title={mounted && theme === "dark" ? "Light Mode" : "Dark Mode"}
            >
              {mounted && theme === "dark" ? (
                <Sun className="w-4 h-4" strokeWidth={1.5} />
              ) : (
                <Moon className="w-4 h-4" strokeWidth={1.5} />
              )}
            </button>
          </div> */}
        </div>
      </header>

      {/* ── HERO & WARNING BANNER ── */}
      <section className="border-b border-border bg-background py-20 flex-shrink-0">
        <div className="w-full max-w-4xl mx-auto px-6 flex flex-col items-center text-center gap-8">

          {/* Title */}
          <div className="space-y-3 max-w-2xl">
            <Badge variant="outline" className="px-3 py-1 text-[11px] font-semibold text-primary bg-primary/5 border-primary/20 uppercase tracking-wider rounded-full">
              Final Year Academic Project · 2024–2025
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
          <div className="w-full max-w-2xl border-2 border-amber-500/40 bg-amber-500/5 dark:bg-amber-500/10 rounded-2xl p-6 space-y-4 text-left">
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
              href={accepted ? "/diagnose" : "#disclaimer-accept"}
              onClick={e => {
                if (!accepted) {
                  e.preventDefault();
                  document.getElementById("disclaimer-accept")?.focus();
                  window.scrollTo({ top: 400, behavior: "smooth" });
                }
              }}
              className={buttonVariants({
                variant: "default",
                size: "lg",
                className: `w-auto font-bold text-sm h-12 px-8 gap-2 transition-all ${!accepted ? "opacity-50 cursor-not-allowed" : "cursor-pointer"}`
              })}
            >
              <Play className="w-4 h-4 fill-current" />
              Get Started — Launch Workstation
              <ArrowRight className="w-4 h-4" />
            </Link>
            {!accepted && (
              <p className="text-[11px] text-muted-foreground">Please accept the disclaimer above to continue.</p>
            )}
          </div>
        </div>
      </section>

      {/* ── ABOUT SECTION (for showcase) ── */}
      {/* TODO: Remove this section before public release */}
      <div className="flex-1 w-full max-w-6xl mx-auto px-6 py-12 grid grid-cols-1 lg:grid-cols-12 gap-8 overflow-y-auto">

        {/* Left: About the project */}
        <main className="lg:col-span-8 space-y-8">
          <div className="flex mb-6">
            <p className="text-xs font-bold uppercase tracking-wider text-muted-foreground border-l-2 border-primary pl-3">
              About This Project
            </p>
          </div>

          <div className="space-y-6">
            <div className="space-y-1">
              <h2 className="text-xl font-bold tracking-tight text-foreground">1. Clinical Objective & Target Audience</h2>
              <p className="text-xs text-muted-foreground">Why sensitivity and offline capability are prioritized.</p>
            </div>
            <Separator />
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <Card className="border border-border bg-card rounded-xl shadow-none">
                <CardContent className="p-6 space-y-3">
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
                <CardContent className="p-6 space-y-3">
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

            <div className="space-y-4 pt-4">
              <h3 className="text-sm font-bold text-foreground">2. Deep Learning Training Pipeline</h3>
              <div className="space-y-3">
                {[
                  { phase: "Phase A: Teacher Network (ResNet-50)", details: "A large network trained on global datasets (Shenzhen, Montgomery) capturing general pulmonary structures and pathological signals." },
                  { phase: "Phase B: Student Network (DenseNet-121)", details: "A lighter ~8M parameter architecture trained to mimic ResNet-50 teacher logits. Retains 98% accuracy while being 3× smaller and faster on CPU." },
                  { phase: "Phase C: Indian Domain Adaptation (NIRT Cohort)", details: "Fine-tuned on the NIRT dataset (Chennai, India) — calibrating for local scanner contrast and visual noise typical in Indian hospital environments." }
                ].map((p, i) => (
                  <div key={i} className="flex gap-4 p-4 border border-border bg-card rounded-xl">
                    <div className="w-6 h-6 rounded-full bg-primary/10 text-xs text-primary flex items-center justify-center font-mono font-bold flex-shrink-0 mt-0.5">
                      {i + 1}
                    </div>
                    <div className="space-y-1">
                      <h4 className="text-sm font-semibold text-foreground">{p.phase}</h4>
                      <p className="text-xs text-muted-foreground leading-relaxed">{p.details}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="space-y-4 pt-4">
              <h3 className="text-sm font-bold text-foreground">3. Validated Metrics (Unseen Test Set)</h3>
              <div className="border border-border rounded-xl bg-card overflow-hidden">
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

            <div className="space-y-4 pt-4">
              <h3 className="text-sm font-bold text-foreground">4. Known Limitations</h3>
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
          </div>
        </main>

        {/* Right Sidebar */}
        <aside className="lg:col-span-4 space-y-6">
          <Card className="border border-border bg-card rounded-xl shadow-none">
            <CardContent className="p-6 space-y-4">
              <div className="flex items-center gap-2">
                <FileText className="w-4 h-4 text-primary" strokeWidth={1.5} />
                <h3 className="text-xs font-bold uppercase tracking-wider text-foreground">Project Highlights</h3>
              </div>
              <Separator />
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
              <Separator />
              <Link
                href="/diagnose"
                className={buttonVariants({ variant: "default", size: "default", className: "w-full text-xs font-semibold h-10 cursor-pointer gap-2" })}
              >
                Launch Screening Workstation <ArrowRight className="w-3.5 h-3.5" />
              </Link>
            </CardContent>
          </Card>

          <Card className="border border-border bg-card rounded-xl shadow-none">
            <CardContent className="p-6 space-y-4">
              <div className="flex items-center gap-2">
                <Database className="w-4 h-4 text-primary" strokeWidth={1.5} />
                <h3 className="text-xs font-bold uppercase tracking-wider text-foreground">Technology Stack</h3>
              </div>
              <Separator />
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
            </CardContent>
          </Card>

          {/* Prototype badge */}
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
      <footer className="bg-[#101010] text-[#a1a1aa] py-10 flex-shrink-0 text-xs border-t border-stone-900">
        <div className="w-full max-w-6xl mx-auto px-6 flex flex-col sm:flex-row justify-between items-center gap-4 font-sans">
          <p className="text-stone-400">Nirikhshon · Final Year College Project Prototype · Not for Clinical Use</p>
          <div className="flex gap-4 text-stone-500 font-medium">
            <Link href="/diagnose" className="hover:text-stone-200 hover:underline">Screening Workspace</Link>
          </div>
        </div>
      </footer>
    </div>
  );
}
