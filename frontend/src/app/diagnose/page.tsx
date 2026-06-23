"use client";

import { useState, useRef, useEffect } from "react";
import { useRouter } from "next/navigation";
import {
  Activity,
  AlertTriangle,
  UploadCloud,
  ChevronRight,
  X,
  LogOut,
  Trash2,
  LayoutDashboard,
  Eye,
  Users,
  ShieldAlert,
  Settings,
  Hash,
} from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";

// Hooks
import { useFileUpload } from "../hooks/useFileUpload";
import { usePrediction } from "../hooks/usePrediction";

// Views
import { ScreeningTab } from "../components/ScreeningTab";
import { SettingsTab } from "../components/SettingsTab";
import { PatientsTab } from "../components/PatientsTab";
import { Dashboard } from "../components/Dashboard";
import AdminConsole from "../components/AdminConsole";

type ViewState = "upload" | "workbench" | "dashboard" | "patients" | "admin" | "settings";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000";

// ── Sidebar nav items ─────────────────────────────────────
const NAV_ITEMS: { id: ViewState; icon: React.ReactNode; label: string; category: string; adminOnly?: boolean }[] = [
  { id: "upload",    icon: <UploadCloud className="w-4 h-4" />,      label: "Upload X-Ray",    category: "workspace" },
  { id: "workbench", icon: <Eye className="w-4 h-4" />,              label: "Workbench",        category: "workspace" },
  { id: "dashboard", icon: <LayoutDashboard className="w-4 h-4" />,  label: "Dashboard",        category: "workspace" },
  { id: "patients",  icon: <Users className="w-4 h-4" />,            label: "Patients",         category: "workspace" },
  { id: "admin",     icon: <ShieldAlert className="w-4 h-4" />,      label: "Admin Console",    category: "admin", adminOnly: true },
  { id: "settings",  icon: <Settings className="w-4 h-4" />,         label: "Settings",         category: "admin", adminOnly: true },
];

const VIEW_LABELS: Record<ViewState, string> = {
  upload: "Upload X-Ray",
  workbench: "Diagnostic Workbench",
  dashboard: "Analytics Dashboard",
  patients: "Patient Registry",
  admin: "Administration",
  settings: "Settings",
};

// Discord-style nav item
function NavItem({
  item, isActive, isDisabled, badge, onClick,
}: {
  item: typeof NAV_ITEMS[0];
  isActive: boolean;
  isDisabled: boolean;
  badge?: number;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      disabled={isDisabled}
      title={item.label}
      className="w-full flex items-center gap-2.5 px-2 py-1.5 rounded text-sm transition-all cursor-pointer relative group"
      style={{
        background: isActive ? "rgba(88,101,242,0.15)" : "transparent",
        color: isActive ? "#FFFFFF" : isDisabled ? "#4E5058" : "#949BA4",
        cursor: isDisabled ? "not-allowed" : "pointer",
        fontWeight: isActive ? 600 : 500,
      }}
      onMouseEnter={e => {
        if (!isActive && !isDisabled) {
          (e.currentTarget as HTMLButtonElement).style.background = "rgba(255,255,255,0.06)";
          (e.currentTarget as HTMLButtonElement).style.color = "#DBDEE1";
        }
      }}
      onMouseLeave={e => {
        if (!isActive && !isDisabled) {
          (e.currentTarget as HTMLButtonElement).style.background = "transparent";
          (e.currentTarget as HTMLButtonElement).style.color = "#949BA4";
        }
      }}
    >
      {/* Active indicator bar */}
      {isActive && (
        <div className="absolute left-0 top-1/2 -translate-y-1/2 w-0.5 h-4 rounded-r-full" style={{ background: "#5865F2" }} />
      )}
      <span style={{ color: isActive ? "#5865F2" : "inherit", opacity: isDisabled ? 0.4 : 1 }}>{item.icon}</span>
      <span className="flex-1 text-left truncate">{item.label}</span>
      {badge !== undefined && badge > 0 && (
        <span className="w-4 h-4 rounded-full text-[10px] font-bold flex items-center justify-center" style={{ background: isActive ? "#5865F2" : "rgba(88,101,242,0.3)", color: "#FFFFFF" }}>
          {badge}
        </span>
      )}
    </button>
  );
}

export default function WorkspacePage() {
  const [mounted, setMounted] = useState(false);
  const [viewState, setViewState] = useState<ViewState>("upload");
  const [workstationMode, setWorkstationMode] = useState<"clinical" | "research" | "xai">("xai");
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

  // ── URL sync ───────────────────────────────────────────
  useEffect(() => {
    setMounted(true);
    const params = new URLSearchParams(window.location.search);
    const tab = params.get("tab") as ViewState | null;
    if (tab && ["upload", "workbench", "dashboard", "patients", "admin", "settings"].includes(tab)) {
      setViewState(tab);
    } else {
      window.history.replaceState({ viewState: "upload" }, "", `${window.location.pathname}?tab=upload`);
    }

    const handlePopState = (e: PopStateEvent) => {
      const p = new URLSearchParams(window.location.search);
      const t = p.get("tab") as ViewState | null;
      if (t && ["upload", "workbench", "dashboard", "patients", "admin", "settings"].includes(t)) {
        setViewState(t);
      } else {
        setViewState("upload");
      }
    };
    window.addEventListener("popstate", handlePopState);
    return () => window.removeEventListener("popstate", handlePopState);
  }, []);

  useEffect(() => {
    if (!mounted) return;
    const params = new URLSearchParams(window.location.search);
    if (params.get("tab") !== viewState) {
      window.history.pushState({ viewState }, "", `${window.location.pathname}?tab=${viewState}`);
    }
  }, [viewState, mounted]);

  // ── Session check ─────────────────────────────────────
  useEffect(() => {
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
    } catch {}
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
    setResults([{
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
      xai_results: record.xai_results || record.xai || null,
    }]);
    setSelectedIdx(0);
    setViewState("workbench");
  };

  const handleNavigate = (view: ViewState) => {
    if (view === "workbench" && files.length === 0) { setViewState("upload"); return; }
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

  // ── Loading screen ────────────────────────────────────
  if (checkingSession) {
    return (
      <div className="min-h-screen flex items-center justify-center" style={{ background: "#313338" }}>
        <div className="flex flex-col items-center gap-4">
          <div className="w-12 h-12 rounded-full flex items-center justify-center animate-pulse" style={{ background: "#5865F2" }}>
            <Activity className="w-6 h-6 text-white" />
          </div>
          <p className="text-xs font-bold uppercase tracking-widest" style={{ color: "#949BA4" }}>Verifying session…</p>
        </div>
      </div>
    );
  }

  const isAdmin = sessionUser?.role === "admin";

  return (
    <div className="flex h-screen overflow-hidden" style={{ background: "#313338", fontFamily: "Inter, sans-serif" }}>

      {/* ── DISCORD SIDEBAR ── */}
      <aside
        className="fixed bottom-0 left-0 right-0 z-40 md:relative md:bottom-auto md:left-auto md:right-auto flex flex-row md:flex-col"
        style={{
          width: undefined,
          minHeight: undefined,
          background: "#2B2D31",
          borderRight: "1px solid rgba(255,255,255,0.04)",
          borderTop: "1px solid rgba(255,255,255,0.04)",
        }}
      >
        {/* Mobile: bottom bar | Desktop: 240px sidebar */}
        <div className="md:hidden h-16 w-full flex flex-row items-center justify-around px-2">
          {NAV_ITEMS.filter(i => !i.adminOnly || isAdmin).slice(0, 5).map(item => {
            const isActive = viewState === item.id;
            const isDisabled = item.id === "workbench" && files.length === 0;
            return (
              <button
                key={item.id}
                onClick={() => handleNavigate(item.id)}
                disabled={isDisabled}
                className="flex flex-col items-center gap-0.5 px-3 py-2 rounded-lg transition-all cursor-pointer"
                style={{ color: isActive ? "#5865F2" : "#949BA4" }}
              >
                {item.icon}
                <span className="text-[9px] font-semibold">{item.label.split(" ")[0]}</span>
              </button>
            );
          })}
        </div>

        {/* Desktop Sidebar */}
        <div className="hidden md:flex flex-col h-full" style={{ width: 240 }}>
          {/* Logo */}
          <div
            className="h-14 flex items-center gap-3 px-4 cursor-pointer flex-shrink-0"
            style={{ borderBottom: "1px solid rgba(255,255,255,0.04)" }}
            onClick={() => router.push("/")}
          >
            <div className="w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0" style={{ background: "#5865F2" }}>
              <Activity className="w-4 h-4 text-white" />
            </div>
            <div>
              <p className="text-sm font-bold text-white leading-tight">Nirikhshon</p>
              <p className="text-[9px] font-bold uppercase tracking-widest" style={{ color: "#949BA4" }}>Workspace</p>
            </div>
          </div>

          {/* Nav sections */}
          <div className="flex-1 overflow-y-auto py-3 px-2 space-y-4">
            {/* Workspace section */}
            <div className="space-y-0.5">
              <p className="px-2 pb-1 text-[10px] font-bold uppercase tracking-widest" style={{ color: "#5C5F66" }}>Workspace</p>
              {NAV_ITEMS.filter(i => i.category === "workspace").map(item => (
                <NavItem
                  key={item.id}
                  item={item}
                  isActive={viewState === item.id}
                  isDisabled={item.id === "workbench" && files.length === 0}
                  badge={item.id === "upload" ? (files.length > 0 ? files.length : undefined) : undefined}
                  onClick={() => handleNavigate(item.id)}
                />
              ))}
            </div>

            {/* Admin section */}
            {isAdmin && (
              <div className="space-y-0.5">
                <p className="px-2 pb-1 text-[10px] font-bold uppercase tracking-widest" style={{ color: "#5C5F66" }}>Administration</p>
                {NAV_ITEMS.filter(i => i.category === "admin" && i.adminOnly).map(item => (
                  <NavItem
                    key={item.id}
                    item={item}
                    isActive={viewState === item.id}
                    isDisabled={false}
                    onClick={() => handleNavigate(item.id)}
                  />
                ))}
              </div>
            )}
          </div>

          {/* User area at bottom */}
          <div
            className="px-3 py-3 flex items-center gap-2.5"
            style={{ borderTop: "1px solid rgba(255,255,255,0.04)", background: "#232428" }}
          >
            <div
              className="w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold text-white uppercase flex-shrink-0"
              style={{ background: "#5865F2" }}
            >
              {sessionUser?.username.charAt(0)}
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-xs font-bold text-white truncate capitalize">{sessionUser?.username}</p>
              <p className="text-[9px] font-semibold capitalize" style={{ color: "#949BA4" }}>{sessionUser?.role}</p>
            </div>
            <button
              onClick={handleLogout}
              className="w-7 h-7 rounded flex items-center justify-center transition-colors cursor-pointer"
              title="Logout"
              style={{ color: "#949BA4" }}
              onMouseEnter={e => {
                (e.currentTarget as HTMLButtonElement).style.background = "rgba(237,66,69,0.15)";
                (e.currentTarget as HTMLButtonElement).style.color = "#ED4245";
              }}
              onMouseLeave={e => {
                (e.currentTarget as HTMLButtonElement).style.background = "transparent";
                (e.currentTarget as HTMLButtonElement).style.color = "#949BA4";
              }}
            >
              <LogOut className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      </aside>

      {/* ── MAIN AREA ── */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden pb-16 md:pb-0" style={{ background: "#313338" }}>

        {/* Discord-style top bar */}
        <header
          className="h-12 flex items-center justify-between px-4 flex-shrink-0"
          style={{ borderBottom: "1px solid rgba(255,255,255,0.06)", background: "#313338" }}
        >
          <div className="flex items-center gap-2">
            <Hash className="w-4 h-4" style={{ color: "#5C5F66" }} />
            <span className="font-bold text-sm text-white">{VIEW_LABELS[viewState]}</span>
            {viewState === "workbench" && files.length > 0 && selectedIdx !== null && results[selectedIdx] && (
              <>
                <ChevronRight className="w-3 h-3" style={{ color: "#5C5F66" }} />
                <span className="text-xs font-mono px-2 py-0.5 rounded" style={{ background: "#2B2D31", color: "#949BA4" }}>
                  {results[selectedIdx]?.study_id || "ST-TEMP"}
                </span>
              </>
            )}
          </div>

          {/* Right — user chip */}
          {sessionUser && (
            <div className="flex items-center gap-2">
              <span className="text-xs hidden sm:block" style={{ color: "#949BA4" }}>{sessionUser.username}</span>
              <div
                className="w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold text-white uppercase"
                style={{ background: "#5865F2" }}
              >
                {sessionUser.username.charAt(0)}
              </div>
            </div>
          )}
        </header>

        {/* Content */}
        <main className={`flex-1 overflow-y-auto ${viewState === "workbench" ? "p-0" : "p-5"}`}>

          {/* ── UPLOAD VIEW ── */}
          {viewState === "upload" && (
            <div className="max-w-2xl mx-auto space-y-5 py-6 animate-fadein">
              <div>
                <h1 className="text-xl font-bold text-white">Upload Radiograph</h1>
                <p className="text-sm mt-0.5" style={{ color: "#949BA4" }}>Drop a chest X-ray (DICOM, PNG, or JPEG) to begin AI screening.</p>
              </div>

              {/* Drop zone */}
              <div
                onDragEnter={handleDrag} onDragOver={handleDrag} onDragLeave={handleDrag} onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
                className="w-full py-16 rounded-lg flex flex-col items-center justify-center text-center cursor-pointer transition-all duration-200"
                style={{
                  border: `2px dashed ${isDragActive ? "#5865F2" : "rgba(255,255,255,0.12)"}`,
                  background: isDragActive ? "rgba(88,101,242,0.08)" : "rgba(255,255,255,0.02)",
                }}
                onMouseEnter={e => {
                  if (!isDragActive) {
                    (e.currentTarget as HTMLDivElement).style.borderColor = "rgba(88,101,242,0.5)";
                    (e.currentTarget as HTMLDivElement).style.background = "rgba(88,101,242,0.04)";
                  }
                }}
                onMouseLeave={e => {
                  if (!isDragActive) {
                    (e.currentTarget as HTMLDivElement).style.borderColor = "rgba(255,255,255,0.12)";
                    (e.currentTarget as HTMLDivElement).style.background = "rgba(255,255,255,0.02)";
                  }
                }}
              >
                <input type="file" ref={fileInputRef} onChange={handleFileInput} multiple className="hidden" accept=".dcm,.png,.jpg,.jpeg" />
                <div className="w-14 h-14 rounded-xl flex items-center justify-center mb-4 transition-colors"
                  style={{ background: isDragActive ? "#5865F2" : "rgba(255,255,255,0.06)" }}>
                  <UploadCloud className="w-7 h-7" style={{ color: isDragActive ? "#FFFFFF" : "#949BA4" }} strokeWidth={1.5} />
                </div>
                <p className="text-sm font-semibold text-white">{isDragActive ? "Release to upload" : "Drag & drop X-ray files here"}</p>
                <p className="text-xs mt-1.5" style={{ color: "#949BA4" }}>or <span style={{ color: "#5865F2", fontWeight: 600 }}>click to browse</span></p>
                <p className="text-[11px] mt-3" style={{ color: "#5C5F66" }}>Supports DICOM (.dcm), PNG, and JPEG · Max 15 MB per file</p>
              </div>

              {/* Medical disclaimer bar */}
              {files.length > 0 && (
                <div className="flex items-start gap-3 p-3.5 rounded-lg animate-fadein"
                  style={{ background: "rgba(254,231,92,0.06)", border: "1px solid rgba(254,231,92,0.2)" }}>
                  <AlertTriangle className="w-4 h-4 flex-shrink-0 mt-0.5" style={{ color: "#FEE75C" }} />
                  <p className="text-xs leading-relaxed" style={{ color: "#FEE75C" }}>
                    <strong>AI results are not a final diagnosis.</strong> Please consult a licensed radiologist before taking any medical action.
                  </p>
                </div>
              )}

              {/* Files list */}
              {files.length > 0 && (
                <div className="rounded-lg overflow-hidden animate-fadein" style={{ background: "#2B2D31", border: "1px solid rgba(255,255,255,0.06)" }}>
                  <div className="px-4 py-3 flex items-center justify-between" style={{ borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-bold text-white">Uploaded Files</span>
                      <span className="w-5 h-5 rounded-full text-[10px] font-bold flex items-center justify-center" style={{ background: "#5865F2", color: "#FFFFFF" }}>
                        {files.length}
                      </span>
                    </div>
                    <button onClick={clearAll} className="flex items-center gap-1 text-xs font-semibold cursor-pointer transition-colors" style={{ color: "#ED4245" }}>
                      <Trash2 className="w-3.5 h-3.5" />Clear all
                    </button>
                  </div>

                  <div className="divide-y" style={{ borderColor: "rgba(255,255,255,0.04)" }}>
                    {files.map((file, idx) => {
                      const res = results[idx];
                      const ext = file.name.split(".").pop()?.toUpperCase() || "?";
                      const sizeMB = (file.size / 1024 / 1024).toFixed(1);
                      return (
                        <div key={idx} className="px-4 py-3 flex items-center gap-3 transition-colors"
                          style={{ borderTop: idx > 0 ? "1px solid rgba(255,255,255,0.04)" : "none" }}
                          onMouseEnter={e => (e.currentTarget as HTMLDivElement).style.background = "rgba(255,255,255,0.02)"}
                          onMouseLeave={e => (e.currentTarget as HTMLDivElement).style.background = "transparent"}
                        >
                          <div className="w-9 h-9 rounded flex items-center justify-center flex-shrink-0" style={{ background: "rgba(88,101,242,0.12)", border: "1px solid rgba(88,101,242,0.2)" }}>
                            <span className="text-[9px] font-bold font-mono" style={{ color: "#5865F2" }}>{ext}</span>
                          </div>
                          <div className="flex-1 min-w-0">
                            <p className="text-sm font-semibold text-white truncate">{file.name}</p>
                            <p className="text-[11px]" style={{ color: "#949BA4" }}>
                              {sizeMB} MB{res?.metadata?.patient_id && <span> · ID: {res.metadata.patient_id}</span>}
                            </p>
                          </div>
                          <div className="flex-shrink-0">
                            {res?.status === "success" ? (
                              <span className="px-2 py-0.5 rounded text-[11px] font-bold" style={res.is_tb
                                ? { background: "rgba(237,66,69,0.15)", color: "#ED4245", border: "1px solid rgba(237,66,69,0.3)" }
                                : { background: "rgba(87,242,135,0.1)", color: "#57F287", border: "1px solid rgba(87,242,135,0.25)" }}>
                                {res.is_tb ? "TB Detected" : "Normal"}
                              </span>
                            ) : res?.status === "loading" ? (
                              <span className="px-2 py-0.5 rounded text-[11px] font-bold animate-pulse" style={{ background: "rgba(88,101,242,0.15)", color: "#5865F2" }}>
                                Analyzing…
                              </span>
                            ) : (
                              <span className="px-2 py-0.5 rounded text-[11px] font-bold" style={{ background: "rgba(255,255,255,0.06)", color: "#949BA4" }}>
                                Pending
                              </span>
                            )}
                          </div>
                          <button
                            onClick={e => { e.stopPropagation(); removeFile(idx); }}
                            className="w-7 h-7 rounded flex items-center justify-center cursor-pointer transition-colors flex-shrink-0"
                            style={{ color: "#4E5058" }}
                            onMouseEnter={e => { (e.currentTarget as HTMLButtonElement).style.background = "rgba(237,66,69,0.1)"; (e.currentTarget as HTMLButtonElement).style.color = "#ED4245"; }}
                            onMouseLeave={e => { (e.currentTarget as HTMLButtonElement).style.background = "transparent"; (e.currentTarget as HTMLButtonElement).style.color = "#4E5058"; }}
                          >
                            <X className="w-3.5 h-3.5" strokeWidth={1.5} />
                          </button>
                        </div>
                      );
                    })}
                  </div>

                  <div className="px-4 py-3 flex justify-end" style={{ borderTop: "1px solid rgba(255,255,255,0.06)", background: "rgba(0,0,0,0.1)" }}>
                    <button
                      onClick={() => {
                        const hasPending = results.some(r => r.status === "pending");
                        if (hasPending) analyzeAll();
                        setSelectedIdx(0);
                        setViewState("workbench");
                      }}
                      className="flex items-center gap-2 px-4 py-2 rounded text-sm font-semibold text-white cursor-pointer transition-all"
                      style={{ background: "#5865F2" }}
                      onMouseEnter={e => { (e.currentTarget as HTMLButtonElement).style.background = "#4752C4"; }}
                      onMouseLeave={e => { (e.currentTarget as HTMLButtonElement).style.background = "#5865F2"; }}
                    >
                      Open Workbench <ChevronRight className="w-4 h-4" strokeWidth={2} />
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* ── WORKBENCH ── */}
          {viewState === "workbench" && (
            <div className="animate-fadein h-full">
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
                downloadReport={async () => {}}
                handleFeedbackSaved={handleFeedbackSaved}
                workstationMode={workstationMode}
                setWorkstationMode={setWorkstationMode}
              />
            </div>
          )}

          {/* ── DASHBOARD ── */}
          {viewState === "dashboard" && (
            <div className="animate-fadein">
              <Dashboard
                onNavigate={v => setViewState(v as ViewState)}
                onOpenWorkbench={() => { if (files.length > 0) { setViewState("workbench"); setSelectedIdx(0); } }}
                hasFiles={files.length > 0}
              />
            </div>
          )}

          {/* ── PATIENTS ── */}
          {viewState === "patients" && (
            <div className="animate-fadein space-y-5">
              <div>
                <h1 className="text-xl font-bold text-white">Patient Registry</h1>
                <p className="text-sm mt-0.5" style={{ color: "#949BA4" }}>Search and manage patient records and study history.</p>
              </div>
              <Separator style={{ borderColor: "rgba(255,255,255,0.06)" }} />
              <PatientsTab onSelectStudy={handleSelectHistoryStudy} />
            </div>
          )}

          {/* ── ADMIN ── */}
          {viewState === "admin" && isAdmin && (
            <div className="animate-fadein"><AdminConsole /></div>
          )}

          {/* ── SETTINGS ── */}
          {viewState === "settings" && isAdmin && (
            <div className="animate-fadein space-y-5">
              <div>
                <h1 className="text-xl font-bold text-white">System Settings</h1>
                <p className="text-sm mt-0.5" style={{ color: "#949BA4" }}>Configure model parameters, thresholds, and system preferences.</p>
              </div>
              <Separator style={{ borderColor: "rgba(255,255,255,0.06)" }} />
              <SettingsTab />
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
