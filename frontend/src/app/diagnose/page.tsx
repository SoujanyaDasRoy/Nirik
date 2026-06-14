"use client";

import { useState, useRef, useEffect } from "react";
import { useRouter } from "next/navigation";
import { useTheme } from "next-themes";
import {
  Activity,
  AlertTriangle,
  UploadCloud,
  Play,
  Download,
  Sun,
  Moon,
  Settings,
  Eye,
  Users,
  LayoutDashboard,
  ShieldAlert,
  ChevronRight,
  X,
  LogOut,
  Trash2,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { Badge } from "@/components/ui/badge";

// Hooks
import { useFileUpload } from "../hooks/useFileUpload";
import { usePrediction } from "../hooks/usePrediction";

// Views
import { ScreeningTab } from "../components/ScreeningTab";
import { SettingsTab } from "../components/SettingsTab";
import { PatientsTab } from "../components/PatientsTab";
import { Dashboard } from "../components/Dashboard";
import { NotificationsPanel } from "../components/NotificationsPanel";
import AdminConsole from "../components/AdminConsole";

type ViewState = "upload" | "workbench" | "dashboard" | "patients" | "admin" | "settings";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000";

// ── Sidebar nav items ──────────────────────────────────────────
const navItems: { id: ViewState; icon: React.ReactNode; label: string; adminOnly?: boolean }[] = [
  { id: "upload",    icon: <UploadCloud className="w-5 h-5" />,      label: "Upload" },
  { id: "workbench", icon: <Eye className="w-5 h-5" />,              label: "Workbench" },
  { id: "dashboard", icon: <LayoutDashboard className="w-5 h-5" />,  label: "Dashboard" },
  { id: "patients",  icon: <Users className="w-5 h-5" />,            label: "Patients" },
  { id: "admin",     icon: <ShieldAlert className="w-5 h-5" />,      label: "Admin", adminOnly: true },
  { id: "settings",  icon: <Settings className="w-5 h-5" />,         label: "Settings", adminOnly: true },
];

// ── Breadcrumb labels ──────────────────────────────────────────
const viewLabels: Record<ViewState, string> = {
  upload:    "Upload Radiograph",
  workbench: "Diagnostic Workbench",
  dashboard: "Analytics Dashboard",
  patients:  "Patient Registry",
  admin:     "Administration",
  settings:  "Settings",
};

export default function WorkspacePage() {
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);
  const [viewState, setViewState] = useState<ViewState>("upload");
  const [globalNote, setGlobalNote] = useState("");
  const [sessionUser, setSessionUser] = useState<{ username: string; role: string } | null>(null);
  const [checkingSession, setCheckingSession] = useState(true);
  const router = useRouter();

  const fileUpload = useFileUpload();
  const {
    files, results, setResults, selectedIdx, setSelectedIdx,
    isDragActive, fileInputRef, handleDrag, handleDrop,
    handleFileInput, removeFile, clearAll,
  } = fileUpload;

  const { isBatchProcessing, analyzeFile, analyzeAll } = usePrediction(
    files, results, setResults, setSelectedIdx
  );

  const reportRef = useRef<HTMLDivElement>(null);

  // ── Session check ──────────────────────────────────────────
  useEffect(() => {
    setMounted(true);
    const checkUserSession = async () => {
      try {
        const res = await fetch(`${API_BASE}/session`, { credentials: "include" });
        if (res.ok) {
          const data = await res.json();
          if (data.authenticated) {
            setSessionUser({ username: data.username, role: data.role });
            setCheckingSession(false);
          } else {
            router.push("/login");
          }
        } else {
          router.push("/login");
        }
      } catch {
        router.push("/login");
      }
    };
    checkUserSession();
  }, [router]);

  const handleLogout = async () => {
    try {
      const res = await fetch(`${API_BASE}/logout`, { method: "POST", credentials: "include" });
      if (res.ok) router.push("/login");
    } catch { /* silent */ }
  };

  const handleFeedbackSaved = (override: string | null, note: string, annotatedB64: string, comments?: string, reviewer?: string) => {
    if (selectedIdx === null) return;
    setResults(prev => {
      const next = [...prev];
      next[selectedIdx] = { ...next[selectedIdx], clinician_override: override, clinician_note: note, annotated_image: annotatedB64, review_comments: comments, reviewer_name: reviewer };
      return next;
    });
  };

  const handleSelectHistoryStudy = (record: any) => {
    const mappedResult = {
      filename: `Study: ${record.study_id}`,
      status: "success" as const,
      prediction: record.prediction,
      confidence: record.confidence,
      is_tb: record.is_tb,
      metadata: {
        patient_id: record.metadata.patient_id,
        patient_name: record.metadata.patient_name,
        patient_age: record.metadata.patient_age,
        patient_sex: record.metadata.patient_sex,
        modality: record.metadata.modality,
        study_date: record.metadata.study_date,
      },
      original_image: record.original_b64,
      heatmap_image: record.heatmap_b64,
      clinician_override: record.clinician_override,
      clinician_note: record.clinician_note,
      annotated_image: record.annotation_b64,
      review_comments: record.clinician_reason,
      reviewer_name: record.reviewer_name || "",
      study_id: record.study_id,
      image_quality: record.image_quality,
    };
    setResults([mappedResult]);
    setSelectedIdx(0);
    setViewState("workbench");
  };

  const downloadReport = async () => {};

  const handleNavigate = (view: ViewState) => {
    if (view === "workbench" && files.length === 0) {
      setViewState("upload");
      return;
    }
    setViewState(view);
  };

  const exportCSV = () => {
    let csv = "Filename,Patient ID,Name,Age,Sex,AI Prediction,Confidence,Clinician Status\n";
    results.forEach(r => {
      if (r.status === "success" && r.metadata) {
        const status = r.clinician_override ? `Overridden to ${r.clinician_override}` : "Agreed";
        csv += `"${r.filename}","${r.metadata.patient_id}","${r.metadata.patient_name}","${r.metadata.patient_age}","${r.metadata.patient_sex}","${r.prediction}",${((r.confidence ?? 0) * 100).toFixed(1)}%,"${status}"\n`;
      }
    });
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "nirikhshon_session.csv";
    a.click();
  };

  // ── Loading screen ──────────────────────────────────────────
  if (checkingSession) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center font-sans">
        <div className="flex flex-col items-center gap-4">
          <Activity className="w-8 h-8 text-primary animate-pulse" strokeWidth={1.5} />
          <p className="text-xs text-muted-foreground font-semibold uppercase tracking-wider">Verifying session...</p>
        </div>
      </div>
    );
  }

  const isAdmin = sessionUser?.role === "admin";

  return (
    <div className="min-h-screen bg-background flex flex-col md:flex-row font-sans overflow-hidden" style={{ height: "100vh" }}>

      {/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */}
      {/* LEFT SIDEBAR / BOTTOM MOBILE NAV BAR */}
      {/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */}
      <aside className="fixed bottom-0 left-0 right-0 h-14 w-full border-t border-r-0 md:relative md:bottom-auto md:left-auto md:right-auto md:h-full md:w-16 flex flex-row md:flex-col items-center justify-around md:justify-start py-0 md:py-4 gap-1 bg-card border-border z-40">
        {/* Logo */}
        <div
          onClick={() => window.location.href = "/"}
          className="w-10 h-10 rounded-xl bg-primary/10 items-center justify-center text-primary cursor-pointer mb-4 hover:bg-primary/20 transition-colors hidden md:flex"
          title="Nirikhshon Home"
        >
          <Activity className="w-5 h-5" strokeWidth={1.5} />
        </div>

        <Separator className="w-8 mb-2 hidden md:block" />

        {/* Nav icons */}
        {navItems.map(item => {
          if (item.adminOnly && !isAdmin) return null;
          const isActive = viewState === item.id;
          const isDisabled = item.id === "workbench" && files.length === 0;
          return (
            <div key={item.id} className="relative group">
              <button
                onClick={() => handleNavigate(item.id)}
                disabled={isDisabled}
                className={`w-10 h-10 rounded-xl flex items-center justify-center transition-all cursor-pointer ${
                  isActive
                    ? "bg-primary text-primary-foreground shadow-md shadow-primary/20"
                    : isDisabled
                    ? "text-muted-foreground/30 cursor-not-allowed"
                    : "text-muted-foreground hover:bg-muted hover:text-foreground"
                }`}
                title={item.label}
              >
                {item.icon}
                {/* Active indicator dot */}
                {isActive && <span className="absolute -right-0.5 top-1/2 -translate-y-1/2 w-1 h-4 rounded-full bg-primary hidden md:block" />}
              </button>

              {/* Tooltip */}
              <div className="absolute left-14 top-1/2 -translate-y-1/2 hidden group-hover:flex items-center pointer-events-none z-50">
                <div className="bg-foreground text-background text-[11px] font-semibold px-2.5 py-1 rounded-lg whitespace-nowrap shadow-lg">
                  {item.label}
                  {isDisabled && <span className="text-muted ml-1">(upload first)</span>}
                </div>
              </div>

              {/* File count badge on upload icon */}
              {item.id === "upload" && files.length > 0 && (
                <span className="absolute -top-1 -right-1 w-4 h-4 rounded-full bg-primary text-primary-foreground text-[9px] font-bold flex items-center justify-center">
                  {files.length}
                </span>
              )}
            </div>
          );
        })}

        <div className="mt-0 md:mt-auto flex flex-row md:flex-col gap-1 items-center">
          <Separator className="w-8 mb-2 hidden md:block" />
          {/* Theme toggle */}
          <div className="relative group">
            <button
              onClick={() => setTheme(mounted && theme === "dark" ? "light" : "dark")}
              className="w-10 h-10 rounded-xl text-muted-foreground hover:bg-muted hover:text-foreground flex items-center justify-center transition-all cursor-pointer"
              title="Toggle Theme"
            >
              {mounted && theme === "dark" ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
            </button>
          </div>
          {/* Logout */}
          <div className="relative group">
            <button
              onClick={handleLogout}
              className="w-10 h-10 rounded-xl text-muted-foreground hover:bg-destructive/10 hover:text-destructive flex items-center justify-center transition-all cursor-pointer"
              title="Logout"
            >
              <LogOut className="w-4 h-4" />
            </button>
            <div className="absolute left-14 bottom-0 hidden group-hover:flex items-center pointer-events-none z-50">
              <div className="bg-foreground text-background text-[11px] font-semibold px-2.5 py-1 rounded-lg whitespace-nowrap shadow-lg">
                Logout ({sessionUser?.username})
              </div>
            </div>
          </div>
        </div>
      </aside>

      {/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */}
      {/* MAIN AREA */}
      {/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden pb-14 md:pb-0">

        {/* ── TOP BAR ─────────────────────────────────────────────── */}
        <header className="h-14 border-b border-border bg-background flex items-center px-6 gap-3 flex-shrink-0 z-30">
          {/* Breadcrumb */}
          <div className="flex items-center gap-2 text-sm flex-1 min-w-0">
            <span className="font-bold text-muted-foreground text-xs">Nirikhshon</span>
            <ChevronRight className="w-3.5 h-3.5 text-muted-foreground flex-shrink-0" strokeWidth={2} />
            <span className="font-bold text-foreground text-sm truncate">{viewLabels[viewState]}</span>
          </div>

          {/* Right side: role badge + notifications */}
          <div className="flex items-center gap-3 flex-shrink-0">
            {sessionUser && (
              <Badge variant="outline" className="h-7 text-[11px] font-semibold px-3 rounded-full capitalize hidden sm:flex">
                👤 {sessionUser.username} · {sessionUser.role}
              </Badge>
            )}
            <NotificationsPanel />
          </div>
        </header>

        {/* ── CONTENT AREA ────────────────────────────────────────── */}
        <main className={`flex-1 overflow-y-auto ${viewState === "workbench" ? "p-4" : "p-6 max-w-6xl mx-auto w-full"}`}>

          {/* ─────────────────────────────── UPLOAD VIEW ─────────────────────────────── */}
          {viewState === "upload" && (
            <div className="max-w-2xl mx-auto space-y-6 py-8 animate-fadein">
              <div className="space-y-1">
                <h1 className="text-2xl font-bold tracking-tight text-foreground">Upload Radiograph</h1>
                <p className="text-sm text-muted-foreground">
                  Drop a chest X-ray (DICOM, PNG, or JPEG) to begin AI screening.
                </p>
              </div>

              {/* Drop zone */}
              <div
                onDragEnter={handleDrag}
                onDragOver={handleDrag}
                onDragLeave={handleDrag}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
                className={`w-full py-16 border-2 border-dashed rounded-2xl flex flex-col items-center justify-center text-center cursor-pointer transition-all duration-200 ${
                  isDragActive
                    ? "border-primary bg-primary/5 scale-[1.01]"
                    : "border-border hover:border-primary/50 hover:bg-muted/20"
                }`}
              >
                <input type="file" ref={fileInputRef} onChange={handleFileInput} multiple className="hidden" accept=".dcm,.png,.jpg,.jpeg" />
                <div className={`w-14 h-14 rounded-2xl flex items-center justify-center mb-4 transition-colors ${isDragActive ? "bg-primary text-primary-foreground" : "bg-muted text-muted-foreground"}`}>
                  <UploadCloud className="w-7 h-7" strokeWidth={1.5} />
                </div>
                <p className="text-sm font-semibold text-foreground">
                  {isDragActive ? "Release to upload" : "Drag & drop X-ray files here"}
                </p>
                <p className="text-xs text-muted-foreground mt-1.5">or <span className="text-primary font-semibold">click to browse</span></p>
                <p className="text-[11px] text-muted-foreground/60 mt-3">Supports DICOM (.dcm), PNG, and JPEG · Max 15MB per file</p>
              </div>

              {/* Post-upload warning */}
              {files.length > 0 && (
                <div className="border border-amber-500/30 bg-amber-500/5 rounded-xl p-4 flex items-start gap-3 animate-fadein">
                  <AlertTriangle className="w-4 h-4 text-amber-500 flex-shrink-0 mt-0.5" strokeWidth={1.5} />
                  <p className="text-xs text-amber-700 dark:text-amber-400 leading-relaxed font-medium">
                    <strong>AI results are not a final diagnosis.</strong> Please consult a licensed radiologist or physician before taking any medical action.
                  </p>
                </div>
              )}

              {/* Uploaded files list */}
              {files.length > 0 && (
                <div className="border border-border rounded-2xl overflow-hidden bg-card animate-fadein">
                  <div className="px-5 py-3.5 border-b border-border flex items-center justify-between bg-muted/30">
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-bold text-foreground">Uploaded Files</span>
                      <Badge variant="secondary" className="rounded-full font-bold text-xs h-5 px-2">{files.length}</Badge>
                    </div>
                    <button onClick={clearAll} className="text-xs text-destructive font-semibold hover:underline flex items-center gap-1 cursor-pointer">
                      <Trash2 className="w-3.5 h-3.5" /> Clear all
                    </button>
                  </div>

                  <div className="divide-y divide-border">
                    {files.map((file, idx) => {
                      const res = results[idx];
                      const ext = file.name.split(".").pop()?.toUpperCase() || "?";
                      const sizeMB = (file.size / 1024 / 1024).toFixed(1);
                      return (
                        <div key={idx} className="px-5 py-3.5 flex items-center gap-4 hover:bg-muted/20 transition-colors">
                          {/* Format badge */}
                          <div className="w-10 h-10 rounded-lg bg-primary/5 border border-border flex items-center justify-center flex-shrink-0">
                            <span className="text-[9px] font-bold text-primary font-mono">{ext}</span>
                          </div>

                          {/* File info */}
                          <div className="flex-1 min-w-0">
                            <p className="text-sm font-semibold text-foreground truncate">{file.name}</p>
                            <p className="text-[11px] text-muted-foreground">
                              {sizeMB} MB
                              {res?.metadata?.patient_id && <span className="ml-2">· ID: {res.metadata.patient_id}</span>}
                            </p>
                          </div>

                          {/* Status badge */}
                          <div className="flex-shrink-0">
                            {res?.status === "success" ? (
                              <Badge className={res.is_tb ? "rounded-full font-bold bg-amber-500/10 text-amber-600 border-amber-500/20 hover:bg-amber-500/20" : "rounded-full font-bold bg-emerald-500/10 text-emerald-600 border-emerald-500/20 hover:bg-emerald-500/20"}>
                                {res.is_tb ? "TB Detected" : "Normal"}
                              </Badge>
                            ) : res?.status === "loading" ? (
                              <Badge variant="secondary" className="animate-pulse rounded-full">Analyzing…</Badge>
                            ) : (
                              <Badge variant="outline" className="rounded-full">Pending</Badge>
                            )}
                          </div>

                          {/* Remove */}
                          <button
                            onClick={e => { e.stopPropagation(); removeFile(idx); }}
                            className="w-7 h-7 rounded-lg hover:bg-destructive/10 text-muted-foreground hover:text-destructive flex items-center justify-center transition-colors cursor-pointer flex-shrink-0"
                          >
                            <X className="w-3.5 h-3.5" strokeWidth={1.5} />
                          </button>
                        </div>
                      );
                    })}
                  </div>

                  {/* Action row */}
                  <div className="px-5 py-4 border-t border-border bg-muted/10 flex items-center justify-between gap-3">
                    <Button variant="outline" size="sm" className="text-xs gap-1.5 cursor-pointer" onClick={exportCSV}>
                      <Download className="w-3.5 h-3.5" /> Export CSV
                    </Button>
                    <div className="flex items-center gap-2">
                      <Button
                        size="sm"
                        variant="outline"
                        className="text-xs gap-1.5 cursor-pointer"
                        onClick={analyzeAll}
                        disabled={isBatchProcessing}
                      >
                        <Play className="w-3.5 h-3.5 fill-current" />
                        {isBatchProcessing ? "Analyzing…" : "Analyze All"}
                      </Button>
                      <Button
                        size="sm"
                        className="text-xs gap-1.5 cursor-pointer"
                        onClick={() => {
                          // Auto-analyze any pending files before entering workbench
                          const hasPending = results.some(r => r.status === "pending");
                          if (hasPending) analyzeAll();
                          setSelectedIdx(0);
                          setViewState("workbench");
                        }}
                      >
                        Open Workbench
                        <ChevronRight className="w-3.5 h-3.5" strokeWidth={2} />
                      </Button>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* ─────────────────────────────── WORKBENCH VIEW ─────────────────────────────── */}
          {viewState === "workbench" && (
            <div className="animate-fadein w-full">
              <ScreeningTab
                files={files}
                results={results}
                setResults={setResults}
                selectedIdx={selectedIdx}
                setSelectedIdx={setSelectedIdx}
                isDragActive={isDragActive}
                isBatchProcessing={isBatchProcessing}
                fileInputRef={fileInputRef}
                handleDrag={handleDrag}
                handleDrop={handleDrop}
                handleFileInput={handleFileInput}
                analyzeFile={analyzeFile}
                removeFile={removeFile}
                clearAll={clearAll}
                globalNote={globalNote}
                setGlobalNote={setGlobalNote}
                reportRef={reportRef}
                downloadReport={downloadReport}
                handleFeedbackSaved={handleFeedbackSaved}
              />
            </div>
          )}

          {/* ─────────────────────────────── DASHBOARD ─────────────────────────────── */}
          {viewState === "dashboard" && (
            <div className="animate-fadein">
              <Dashboard
                onNavigate={(v) => setViewState(v as ViewState)}
                onOpenWorkbench={() => { if (files.length > 0) { setViewState("workbench"); setSelectedIdx(0); } }}
                hasFiles={files.length > 0}
              />
            </div>
          )}

          {/* ─────────────────────────────── PATIENTS ─────────────────────────────── */}
          {viewState === "patients" && (
            <div className="animate-fadein space-y-6">
              <div>
                <h1 className="text-2xl font-bold tracking-tight text-foreground">Patient Registry</h1>
                <p className="text-sm text-muted-foreground mt-1">Search and manage patient records and study history.</p>
              </div>
              <Separator />
              <PatientsTab onSelectStudy={handleSelectHistoryStudy} />
            </div>
          )}

          {/* ─────────────────────────────── ADMIN ─────────────────────────────── */}
          {viewState === "admin" && isAdmin && (
            <div className="animate-fadein">
              <AdminConsole />
            </div>
          )}

          {/* ─────────────────────────────── SETTINGS ─────────────────────────────── */}
          {viewState === "settings" && isAdmin && (
            <div className="animate-fadein space-y-6">
              <div>
                <h1 className="text-2xl font-bold tracking-tight text-foreground">System Settings</h1>
                <p className="text-sm text-muted-foreground mt-1">Configure model parameters, thresholds, and system preferences.</p>
              </div>
              <Separator />
              <SettingsTab />
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
