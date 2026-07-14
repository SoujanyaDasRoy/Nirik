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
import { ScreeningWorkstation } from "../components/ScreeningWorkstation";
import { SettingsTab } from "../components/SettingsTab";
import { PatientsTab } from "../components/PatientsTab";
import { Dashboard } from "../components/Dashboard";
import AdminConsole from "../components/AdminConsole";

type ViewState = "upload" | "workbench" | "dashboard" | "patients" | "admin" | "settings";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "https://projectmantra-nirikshon-backend.hf.space";

// ── Sidebar nav items ─────────────────────────────────────
const NAV_ITEMS: { id: ViewState; icon: React.ReactNode; label: string; category: string; adminOnly?: boolean }[] = [
  { id: "upload",    icon: <UploadCloud className="w-4 h-4" />,      label: "Upload X-Ray",    category: "workspace" },
  { id: "workbench", icon: <Eye className="w-4 h-4" />,              label: "Workbench",        category: "workspace" },
  { id: "dashboard", icon: <LayoutDashboard className="w-4 h-4" />,  label: "Dashboard",        category: "workspace" },
  { id: "patients",  icon: <Users className="w-4 h-4" />,            label: "Patients",         category: "workspace" },
  { id: "admin",     icon: <ShieldAlert className="w-4 h-4" />,      label: "Admin Console",    category: "admin", adminOnly: true },
  { id: "settings",  icon: <Settings className="w-4 h-4" />,         label: "Settings",         category: "settings" },
];

const VIEW_LABELS: Record<ViewState, string> = {
  upload: "Upload X-Ray",
  workbench: "Diagnostic Workbench",
  dashboard: "Analytics Dashboard",
  patients: "Patient Registry",
  admin: "Administration",
  settings: "Settings",
};

// Framer-style nav item (Pills)
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
      className="w-full flex items-center gap-3 px-3 py-2 rounded-full text-[14px] transition-all cursor-pointer relative group"
      style={{
        background: isActive ? "#1c1c1c" : "transparent",
        color: isActive ? "#ffffff" : isDisabled ? "#4E5058" : "#999999",
        cursor: isDisabled ? "not-allowed" : "pointer",
        fontWeight: isActive ? 500 : 400,
      }}
      onMouseEnter={e => {
        if (!isActive && !isDisabled) {
          (e.currentTarget as HTMLButtonElement).style.background = "#141414";
          (e.currentTarget as HTMLButtonElement).style.color = "#ffffff";
        }
      }}
      onMouseLeave={e => {
        if (!isActive && !isDisabled) {
          (e.currentTarget as HTMLButtonElement).style.background = "transparent";
          (e.currentTarget as HTMLButtonElement).style.color = "#999999";
        }
      }}
    >
      <span style={{ color: isActive ? "#ffffff" : "inherit", opacity: isDisabled ? 0.4 : 1 }}>{item.icon}</span>
      <span className="flex-1 text-left truncate">{item.label}</span>
      {badge !== undefined && badge > 0 && (
        <span className="px-2 py-0.5 rounded-full text-[11px] font-medium flex items-center justify-center" style={{ background: isActive ? "#ffffff" : "#1c1c1c", color: isActive ? "#000000" : "#ffffff" }}>
          {badge}
        </span>
      )}
    </button>
  );
}

export default function WorkspacePage() {
  const [mounted, setMounted] = useState(false);
  const [viewState, setViewState] = useState<ViewState>("upload");
  const [workstationMode, setWorkstationMode] = useState<"clinical" | "xai">("clinical");
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
      if (res.ok) {
        localStorage.removeItem("nirikshon_user");
        router.push("/login");
      }
    } catch {
      localStorage.removeItem("nirikshon_user");
      router.push("/login");
    }
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
      <div className="min-h-screen flex items-center justify-center" style={{ background: "#090909" }}>
        <div className="flex flex-col items-center gap-4">
          <div className="w-12 h-12 rounded-full flex items-center justify-center animate-pulse" style={{ background: "#1c1c1c" }}>
            <Activity className="w-6 h-6 text-white" />
          </div>
          <p className="text-xs font-medium tracking-wide" style={{ color: "#999999" }}>Verifying session…</p>
        </div>
      </div>
    );
  }

  const isAdmin = sessionUser?.role === "admin";

  return (
    <div className="flex h-screen overflow-hidden selection:bg-[#0099ff]/30 font-sans" style={{ background: "#090909", color: "#ffffff" }}>

      {/* ── FRAMER SIDEBAR ── */}
      <aside
        className="fixed bottom-0 left-0 right-0 z-40 md:relative md:bottom-auto md:left-auto md:right-auto flex flex-row md:flex-col"
        style={{
          width: undefined,
          minHeight: undefined,
          background: "#090909",
          borderRight: "1px solid #262626",
          borderTop: "1px solid #262626",
        }}
      >
        {/* Mobile: bottom bar | Desktop: 260px sidebar */}
        <div className="md:hidden h-16 w-full flex flex-row items-center justify-around px-2 bg-[#090909]">
          {NAV_ITEMS.filter(i => !i.adminOnly || isAdmin).slice(0, 5).map(item => {
            const isActive = viewState === item.id;
            const isDisabled = item.id === "workbench" && files.length === 0;
            return (
              <button
                key={item.id}
                onClick={() => handleNavigate(item.id)}
                disabled={isDisabled}
                className="flex flex-col items-center gap-1 px-3 py-2 rounded-lg transition-all cursor-pointer"
                style={{ color: isActive ? "#ffffff" : "#999999" }}
              >
                {item.icon}
                <span className="text-[10px] font-medium">{item.label.split(" ")[0]}</span>
              </button>
            );
          })}
        </div>

        {/* Desktop Sidebar */}
        <div className="hidden md:flex flex-col h-full" style={{ width: 260 }}>
          {/* Logo */}
          <div
            className="h-[60px] flex items-center gap-3 px-6 cursor-pointer flex-shrink-0"
            style={{ borderBottom: "1px solid #262626" }}
            onClick={() => router.push("/")}
          >
            <div className="w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 bg-[#ffffff]">
              <Activity className="w-4 h-4 text-[#000000]" />
            </div>
            <div>
              <p className="text-[15px] font-bold text-white tracking-[-0.2px] leading-tight">Nirikhshon.</p>
              <p className="text-[11px] font-medium" style={{ color: "#999999" }}>Workspace</p>
            </div>
          </div>

          {/* Nav sections */}
          <div className="flex-1 overflow-y-auto py-6 px-4 space-y-6">
            {/* Workspace section */}
            <div className="space-y-1">
              <p className="px-3 pb-2 text-[12px] font-medium" style={{ color: "#999999" }}>Workspace</p>
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

            {/* Settings section */}
            <div className="space-y-1">
              <p className="px-3 pb-2 text-[12px] font-medium" style={{ color: "#999999" }}>System</p>
              {NAV_ITEMS.filter(i => i.category === "settings").map(item => (
                <NavItem
                  key={item.id}
                  item={item}
                  isActive={viewState === item.id}
                  isDisabled={false}
                  onClick={() => handleNavigate(item.id)}
                />
              ))}
            </div>

            {/* Admin section */}
            {isAdmin && (
              <div className="space-y-1">
                <p className="px-3 pb-2 text-[12px] font-medium" style={{ color: "#999999" }}>Administration</p>
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
            className="p-4 flex items-center gap-3 m-4 rounded-[15px]"
            style={{ background: "#141414", border: "1px solid #262626" }}
          >
            <div
              className="w-9 h-9 rounded-full flex items-center justify-center text-sm font-medium text-black bg-[#ffffff] uppercase flex-shrink-0"
            >
              {sessionUser?.username.charAt(0)}
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-[13px] font-medium text-white truncate capitalize leading-tight">{sessionUser?.username}</p>
              <p className="text-[11px] mt-0.5 capitalize" style={{ color: "#999999" }}>{sessionUser?.role}</p>
            </div>
            <button
              onClick={handleLogout}
              className="w-8 h-8 rounded-full flex items-center justify-center transition-colors cursor-pointer"
              title="Logout"
              style={{ color: "#999999", background: "#1c1c1c" }}
              onMouseEnter={e => {
                (e.currentTarget as HTMLButtonElement).style.color = "#ffffff";
                (e.currentTarget as HTMLButtonElement).style.background = "#262626";
              }}
              onMouseLeave={e => {
                (e.currentTarget as HTMLButtonElement).style.color = "#999999";
                (e.currentTarget as HTMLButtonElement).style.background = "#1c1c1c";
              }}
            >
              <LogOut className="w-4 h-4" />
            </button>
          </div>
        </div>
      </aside>

      {/* ── MAIN AREA ── */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden pb-16 md:pb-0" style={{ background: "#090909" }}>

        {/* Top bar */}
        <header
          className="h-[60px] flex items-center justify-between px-6 flex-shrink-0"
          style={{ borderBottom: "1px solid #262626", background: "#090909" }}
        >
          <div className="flex items-center gap-3">
            <span className="font-medium text-[15px] text-white tracking-[-0.2px]">{VIEW_LABELS[viewState]}</span>
            {viewState === "workbench" && files.length > 0 && selectedIdx !== null && results[selectedIdx] && (
              <>
                <ChevronRight className="w-4 h-4 text-[#999999]" />
                <span className="text-[13px] font-medium px-2.5 py-1 rounded-full border border-[#262626]" style={{ background: "#141414", color: "#999999" }}>
                  {results[selectedIdx]?.study_id || "ST-TEMP"}
                </span>
              </>
            )}
          </div>
        </header>

        {/* Content */}
        <main className={`flex-1 overflow-y-auto ${viewState === "workbench" ? "p-0" : "p-8 md:p-12"}`}>

          {/* ── UPLOAD VIEW ── */}
          {viewState === "upload" && (
            <div className="max-w-2xl mx-auto space-y-8 animate-fadein mt-8">
              <div>
                <h1 className="text-[32px] font-medium text-white tracking-[-1px] leading-tight">Upload Radiograph</h1>
                <p className="text-[15px] mt-2" style={{ color: "#999999" }}>Drop a chest X-ray (DICOM, PNG, or JPEG) to begin AI screening.</p>
              </div>

              {/* Drop zone */}
              <div
                onDragEnter={handleDrag} onDragOver={handleDrag} onDragLeave={handleDrag} onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
                className="w-full py-10 rounded-[20px] flex flex-col items-center justify-center text-center cursor-pointer transition-all duration-300"
                style={{
                  border: `1px dashed ${isDragActive ? "#0099ff" : "#262626"}`,
                  background: isDragActive ? "rgba(0,153,255,0.05)" : "#141414",
                }}
                onMouseEnter={e => {
                  if (!isDragActive) {
                    (e.currentTarget as HTMLDivElement).style.borderColor = "#404040";
                    (e.currentTarget as HTMLDivElement).style.background = "#1c1c1c";
                  }
                }}
                onMouseLeave={e => {
                  if (!isDragActive) {
                    (e.currentTarget as HTMLDivElement).style.borderColor = "#262626";
                    (e.currentTarget as HTMLDivElement).style.background = "#141414";
                  }
                }}
              >
                <input type="file" ref={fileInputRef} onChange={handleFileInput} multiple className="hidden" accept=".dcm,.png,.jpg,.jpeg" />
                <div className="w-16 h-16 rounded-full flex items-center justify-center mb-5 transition-colors border border-[#262626]"
                  style={{ background: isDragActive ? "#0099ff" : "#1c1c1c" }}>
                  <UploadCloud className="w-7 h-7" style={{ color: isDragActive ? "#FFFFFF" : "#ffffff" }} strokeWidth={1.5} />
                </div>
                <p className="text-[18px] font-medium text-white tracking-[-0.2px]">{isDragActive ? "Release to upload" : "Drag & drop X-ray files here"}</p>
                <p className="text-[14px] mt-2" style={{ color: "#999999" }}>or <span className="text-[#ffffff] font-medium">click to browse</span></p>
                <p className="text-[12px] mt-4" style={{ color: "#666666" }}>Supports DICOM (.dcm), PNG, and JPEG · Max 15 MB per file</p>
              </div>

              {/* Medical disclaimer bar */}
              {files.length > 0 && (
                <div className="flex items-start gap-3 p-4 rounded-[15px] animate-fadein"
                  style={{ background: "#1c1c1c", border: "1px solid #262626" }}>
                  <AlertTriangle className="w-5 h-5 flex-shrink-0 mt-0.5" style={{ color: "#ff7a3d" }} />
                  <p className="text-[14px] leading-relaxed" style={{ color: "#999999" }}>
                    <strong className="text-white font-medium">AI results are not a final diagnosis.</strong> Please consult a licensed radiologist before taking any medical action.
                  </p>
                </div>
              )}

              {/* Files list */}
              {files.length > 0 && (
                <div className="rounded-[20px] overflow-hidden animate-fadein" style={{ background: "#141414", border: "1px solid #262626" }}>
                  <div className="px-6 py-4 flex items-center justify-between border-b border-[#262626]">
                    <div className="flex items-center gap-3">
                      <span className="text-[15px] font-medium text-white">Uploaded Files</span>
                      <span className="w-6 h-6 rounded-full text-[12px] font-medium flex items-center justify-center" style={{ background: "#1c1c1c", color: "#ffffff", border: "1px solid #262626" }}>
                        {files.length}
                      </span>
                    </div>
                    <button onClick={clearAll} className="flex items-center gap-1.5 text-[13px] font-medium cursor-pointer transition-colors" style={{ color: "#ff5577" }}>
                      <Trash2 className="w-4 h-4" /> Clear all
                    </button>
                  </div>

                  <div className="divide-y divide-[#262626]">
                    {files.map((file, idx) => {
                      const res = results[idx];
                      const ext = file.name.split(".").pop()?.toUpperCase() || "?";
                      const sizeMB = (file.size / 1024 / 1024).toFixed(1);
                      return (
                        <div key={idx} className="px-6 py-4 flex items-center gap-4 transition-colors hover:bg-[#1c1c1c]">
                          <div className="w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0 bg-[#1c1c1c] border border-[#262626]">
                            <span className="text-[10px] font-medium font-mono text-[#ffffff]">{ext}</span>
                          </div>
                          <div className="flex-1 min-w-0">
                            <p className="text-[14px] font-medium text-white truncate">{file.name}</p>
                            <p className="text-[12px] mt-0.5 text-[#999999]">
                              {sizeMB} MB{res?.metadata?.patient_id && <span> · ID: {res.metadata.patient_id}</span>}
                            </p>
                          </div>
                          <div className="flex-shrink-0">
                            {res?.status === "success" ? (
                              <span className="px-3 py-1 rounded-full text-[12px] font-medium border border-[#262626]" style={res.is_tb
                                ? { background: "#1c1c1c", color: "#ff5577" }
                                : { background: "#1c1c1c", color: "#22c55e" }}>
                                {res.is_tb ? "TB Detected" : "Normal"}
                              </span>
                            ) : res?.status === "loading" ? (
                              <span className="px-3 py-1 rounded-full text-[12px] font-medium border border-[#262626] bg-[#1c1c1c] text-[#ffffff] animate-pulse">
                                Analyzing…
                              </span>
                            ) : (
                              <span className="px-3 py-1 rounded-full text-[12px] font-medium border border-[#262626] bg-[#090909] text-[#999999]">
                                Pending
                              </span>
                            )}
                          </div>
                          <button
                            onClick={e => { e.stopPropagation(); removeFile(idx); }}
                            className="w-8 h-8 rounded-full flex items-center justify-center cursor-pointer transition-colors flex-shrink-0 text-[#999999] hover:bg-[#262626] hover:text-[#ffffff]"
                          >
                            <X className="w-4 h-4" strokeWidth={1.5} />
                          </button>
                        </div>
                      );
                    })}
                  </div>

                  <div className="p-6 flex justify-end border-t border-[#262626] bg-[#090909]">
                    <button
                      onClick={() => {
                        const hasPending = results.some(r => r.status === "pending");
                        if (hasPending) analyzeAll();
                        setSelectedIdx(0);
                        setViewState("workbench");
                      }}
                      className="flex items-center gap-2 px-6 py-3 rounded-full text-[14px] font-medium text-[#000000] cursor-pointer transition-transform hover:scale-[1.02] bg-[#ffffff]"
                    >
                      Start Scanning <ChevronRight className="w-4 h-4" strokeWidth={2} />
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* ── WORKBENCH ── */}
          {viewState === "workbench" && (
            <div className="animate-fadein h-full">
              <ScreeningWorkstation
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

                handleFeedbackSaved={handleFeedbackSaved}
                workstationMode={workstationMode}
                setWorkstationMode={setWorkstationMode}
              />
            </div>
          )}

          {/* ── DASHBOARD ── */}
          {viewState === "dashboard" && (
            <div className="animate-fadein h-full">
              <Dashboard
                onNavigate={v => setViewState(v as ViewState)}
                onOpenWorkbench={() => { if (files.length > 0) { setViewState("workbench"); setSelectedIdx(0); } }}
                hasFiles={files.length > 0}
              />
            </div>
          )}

          {/* ── PATIENTS ── */}
          {viewState === "patients" && (
            <div className="animate-fadein space-y-8 max-w-[1199px] mx-auto">
              <div>
                <h1 className="text-[32px] font-medium text-white tracking-[-1px] leading-tight">Patient Registry</h1>
                <p className="text-[15px] mt-2" style={{ color: "#999999" }}>Search and manage patient records and study history.</p>
              </div>
              <Separator style={{ borderColor: "#262626" }} />
              <PatientsTab onSelectStudy={handleSelectHistoryStudy} />
            </div>
          )}

          {/* ── ADMIN ── */}
          {viewState === "admin" && isAdmin && (
            <div className="animate-fadein max-w-[1199px] mx-auto"><AdminConsole /></div>
          )}

          {/* ── SETTINGS ── */}
          {viewState === "settings" && (
            <div className="animate-fadein space-y-8 max-w-[1199px] mx-auto">
              <div>
                <h1 className="text-[32px] font-medium text-white tracking-[-1px] leading-tight">System Settings</h1>
                <p className="text-[15px] mt-2" style={{ color: "#999999" }}>Configure model parameters, thresholds, and system preferences.</p>
              </div>
              <Separator style={{ borderColor: "#262626" }} />
              <SettingsTab />
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
