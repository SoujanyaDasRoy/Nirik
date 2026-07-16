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
} from "lucide-react";

// Hooks
import { useFileUpload } from "../hooks/useFileUpload";
import { usePrediction } from "../hooks/usePrediction";

// Views
import { ScreeningWorkstation } from "../components/ScreeningWorkstation";
import LlmAssistant from "../components/LlmAssistant";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "https://projectmantra-nirikshon-backend.hf.space";

export default function DiagnosePage() {
  const [mounted, setMounted] = useState(false);
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

  useEffect(() => {
    setMounted(true);
  }, []);

  // ── Session check ─────────────────────────────────────
  useEffect(() => {
    const checkUserSession = () => {
      try {
        const userStr = localStorage.getItem("nirikshon_user");
        if (userStr) {
          const user = JSON.parse(userStr);
          if (user && user.username) {
            setSessionUser({ username: user.username, role: user.role || "reviewer" });
            setCheckingSession(false);
            return;
          }
        }
      } catch (e) {
        console.error("Failed to parse user session", e);
      }
      router.push("/login");
    };
    checkUserSession();
  }, [router]);

  const handleLogout = () => {
    localStorage.removeItem("nirikshon_user");
    router.push("/login");
  };

  const handleFeedbackSaved = (override: string | null, note: string, annotatedB64: string, comments?: string, reviewer?: string) => {
    if (selectedIdx === null) return;
    setResults(prev => {
      const next = [...prev];
      next[selectedIdx] = { ...next[selectedIdx], clinician_override: override, clinician_note: note, annotated_image: annotatedB64, review_comments: comments, reviewer_name: reviewer };
      return next;
    });
  };

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

  return (
    <div className="min-h-screen flex flex-col selection:bg-[#0099ff]/30 font-sans" style={{ background: "#090909", color: "#ffffff" }}>
      
      {/* ── TOP NAVBAR ── */}
      <header className="h-[70px] sticky top-0 z-50 flex items-center justify-between px-6 md:px-12 backdrop-blur-xl bg-[#090909]/80 border-b border-[#262626]">
        <div 
          className="flex items-center gap-3 cursor-pointer"
          onClick={() => router.push("/")}
        >
          <div className="w-10 h-10 rounded-full flex items-center justify-center bg-[#ffffff] shadow-[0_0_15px_rgba(255,255,255,0.2)]">
            <Activity className="w-5 h-5 text-[#000000]" />
          </div>
          <div>
            <p className="text-[18px] font-bold text-white tracking-[-0.2px] leading-tight">Nirikhshon.</p>
            <p className="text-[11px] font-medium text-[#999999]">Diagnostic Platform</p>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <div className="hidden md:flex items-center gap-3 px-4 py-2 rounded-full bg-[#141414] border border-[#262626]">
            <div className="w-6 h-6 rounded-full flex items-center justify-center text-[10px] font-bold text-black bg-[#ffffff] uppercase">
              {sessionUser?.username.charAt(0)}
            </div>
            <div className="flex flex-col">
              <span className="text-[12px] font-medium text-white capitalize leading-none">{sessionUser?.username}</span>
            </div>
          </div>
          
          <button
            onClick={handleLogout}
            className="w-10 h-10 rounded-full flex items-center justify-center transition-all cursor-pointer text-[#999999] bg-[#141414] border border-[#262626] hover:text-white hover:bg-[#262626]"
            title="Logout"
          >
            <LogOut className="w-4 h-4" />
          </button>
        </div>
      </header>

      {/* ── SCROLLABLE MAIN CONTENT ── */}
      <main className="flex-1 w-full max-w-5xl mx-auto p-6 md:p-12 space-y-12">
        
        {/* HERO / UPLOAD SECTION */}
        <section className="space-y-8 animate-fadein">
          <div className="text-center space-y-3">
            <h1 className="text-[40px] md:text-[56px] font-medium text-white tracking-[-2px] leading-tight drop-shadow-lg">
              Start Screening
            </h1>
            <p className="text-[16px] md:text-[18px] text-[#999999] max-w-2xl mx-auto">
              Upload a chest X-ray to begin immediate AI-assisted diagnosis for Tuberculosis.
            </p>
          </div>

          <div
            onDragEnter={handleDrag} onDragOver={handleDrag} onDragLeave={handleDrag} onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className="w-full py-10 rounded-[24px] flex flex-col items-center justify-center text-center cursor-pointer transition-all duration-300 shadow-2xl relative overflow-hidden group"
            style={{
              border: `2px dashed ${isDragActive ? "#0099ff" : "#262626"}`,
              background: isDragActive ? "rgba(0,153,255,0.05)" : "#141414",
            }}
          >
            {/* Glow effect on hover */}
            <div className="absolute inset-0 bg-gradient-to-b from-white/[0.02] to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none" />
            
            <input type="file" ref={fileInputRef} onChange={handleFileInput} multiple className="hidden" accept=".dcm,.png,.jpg,.jpeg" />
            
            <div className="w-16 h-16 rounded-full flex items-center justify-center mb-4 transition-all duration-300 border border-[#262626] group-hover:scale-110 group-hover:shadow-[0_0_30px_rgba(255,255,255,0.1)]"
              style={{ background: isDragActive ? "#0099ff" : "#1c1c1c" }}>
              <UploadCloud className="w-6 h-6" style={{ color: "#ffffff" }} strokeWidth={1.5} />
            </div>
            
            <p className="text-[18px] font-medium text-white tracking-[-0.5px]">
              {isDragActive ? "Drop files now" : "Drag & drop X-ray images here"}
            </p>
            <p className="text-[13px] mt-2" style={{ color: "#999999" }}>
              or <span className="text-[#ffffff] font-medium group-hover:underline">click to browse</span>
            </p>
            <p className="text-[11px] mt-4 px-4 py-1.5 rounded-full bg-[#090909] border border-[#262626]" style={{ color: "#666666" }}>
              Supports DICOM, PNG, JPEG · Max 15 MB
            </p>
          </div>

          {/* Files List */}
          {files.length > 0 && (
            <div className="rounded-[24px] overflow-hidden animate-fadein shadow-2xl" style={{ background: "#141414", border: "1px solid #262626" }}>
              <div className="px-6 py-5 flex items-center justify-between border-b border-[#262626] bg-gradient-to-r from-white/[0.02] to-transparent">
                <div className="flex items-center gap-3">
                  <span className="text-[16px] font-medium text-white">Pending Studies</span>
                  <span className="w-6 h-6 rounded-full text-[12px] font-bold flex items-center justify-center bg-[#ffffff] text-[#000000]">
                    {files.length}
                  </span>
                </div>
                <button onClick={clearAll} className="flex items-center gap-1.5 text-[13px] font-medium cursor-pointer transition-colors hover:text-red-400" style={{ color: "#ff5577" }}>
                  <Trash2 className="w-4 h-4" /> Clear
                </button>
              </div>

              <div className="divide-y divide-[#262626]">
                {files.map((file, idx) => {
                  const res = results[idx];
                  const sizeMB = (file.size / 1024 / 1024).toFixed(1);
                  return (
                    <div key={idx} className="px-6 py-4 flex items-center gap-4 transition-colors hover:bg-[#1c1c1c]">
                      <div className="flex-1 min-w-0">
                        <p className="text-[15px] font-medium text-white truncate">{file.name}</p>
                        <p className="text-[13px] mt-1 text-[#999999]">
                          {sizeMB} MB
                        </p>
                      </div>
                      <button
                        onClick={() => removeFile(idx)}
                        className="w-10 h-10 rounded-full flex items-center justify-center cursor-pointer transition-colors text-[#999999] hover:bg-[#262626] hover:text-[#ffffff]"
                      >
                        <X className="w-5 h-5" strokeWidth={1.5} />
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
                    // Instead of viewState switching, we scroll to workstation
                    setTimeout(() => {
                      document.getElementById("workstation")?.scrollIntoView({ behavior: "smooth" });
                    }, 100);
                  }}
                  className="flex items-center gap-2 px-8 py-4 rounded-full text-[16px] font-bold text-[#000000] cursor-pointer transition-all hover:scale-[1.02] shadow-[0_0_20px_rgba(255,255,255,0.2)] bg-[#ffffff]"
                >
                  Analyze & Generate Results <ChevronRight className="w-5 h-5" strokeWidth={2.5} />
                </button>
              </div>
            </div>
          )}
        </section>

        {/* WORKSTATION SECTION */}
        {files.length > 0 && selectedIdx !== null && (
          <section id="workstation" className="pt-8 pb-16 animate-fadein scroll-mt-24 space-y-12">
            <div className="flex items-center gap-3">
              <h2 className="text-[32px] font-medium text-white tracking-[-1px]">Analysis Results</h2>
              <div className="px-3 py-1 rounded-full border border-yellow-500/30 bg-yellow-500/10 text-yellow-500 text-[11px] uppercase font-bold tracking-wider flex items-center gap-1.5">
                <AlertTriangle className="w-3.5 h-3.5" />
                Not a diagnosis
              </div>
            </div>
            
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

            {/* AI CO-PILOT SECTION */}
            {results[selectedIdx]?.status === "success" && (
              <div className="w-full mt-12 animate-fadein">
                <div className="flex items-center gap-3 mb-6">
                  <h2 className="text-[28px] font-medium text-white tracking-[-1px]">AI Co-Pilot</h2>
                  <div className="px-3 py-1 rounded-full border border-primary/30 bg-primary/10 text-primary text-[11px] uppercase font-bold tracking-wider">
                    Gemini 2.5 Flash
                  </div>
                </div>
                <div className="h-[500px]">
                  <LlmAssistant activeResult={results[selectedIdx]} />
                </div>
              </div>
            )}
          </section>
        )}

      </main>
    </div>
  );
}
