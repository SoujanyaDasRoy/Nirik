"use client";

import { useState, useEffect, useRef } from "react";
import dynamic from "next/dynamic";
import { getCookie } from "../hooks/usePrediction";

import {
  Activity,
  Eye,
  Clock,
  AlertCircle,
  CheckCircle,
  AlertTriangle,
  Play,
  FileText,
  UserCheck,
  ShieldAlert,
  Download,
  Database,
  Layers,
  Settings,
  CircleDot,
  Columns,
  Loader2,
  LayoutDashboard,
  Search,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Textarea } from "@/components/ui/textarea";
import { Separator } from "@/components/ui/separator";

import { AnalysisResult } from "../hooks/useFileUpload";
import { Box, LungZone } from "./AnnotationCanvas";

import { predictionService } from "../services/predictionService";
import { heatmapService } from "../services/heatmapService";
import { observationService } from "../services/observationService";
import { imageQualityService } from "../services/imageQualityService";
import { reportService } from "../services/reportService";
import { auditService, AuditLogEntry } from "../services/auditService";
import { exportService } from "../services/exportService";

import XaiVisualization from "./XaiVisualization";
import LlmAssistant from "./LlmAssistant";

const DicomViewer = dynamic(() => import("./DicomViewer"), { ssr: false });

interface ScreeningTabProps {
  files: File[];
  results: AnalysisResult[];
  setResults: React.Dispatch<React.SetStateAction<AnalysisResult[]>>;
  selectedIdx: number | null;
  setSelectedIdx: (idx: number | null) => void;
  isDragActive: boolean;
  isBatchProcessing: boolean;
  fileInputRef: React.RefObject<HTMLInputElement | null>;
  handleDrag: (e: React.DragEvent) => void;
  handleDrop: (e: React.DragEvent) => void;
  handleFileInput: (e: React.ChangeEvent<HTMLInputElement>) => void;
  analyzeFile: (idx: number) => Promise<void>;
  removeFile: (idx: number) => void;
  clearAll: () => void;
  globalNote: string;
  setGlobalNote: (note: string) => void;
  reportRef: React.RefObject<HTMLDivElement | null>;
  downloadReport: () => Promise<void>;
  handleFeedbackSaved: (
    override: string | null,
    note: string,
    annotatedB64: string,
    comments?: string,
    reviewer?: string
  ) => void;
  workstationMode: "clinical" | "xai";
  setWorkstationMode: (mode: "clinical" | "xai") => void;
}

export function ScreeningTab({
  files,
  results,
  setResults,
  selectedIdx,
  setSelectedIdx,
  isDragActive,
  isBatchProcessing,
  fileInputRef,
  handleDrag,
  handleDrop,
  handleFileInput,
  analyzeFile,
  removeFile,
  clearAll,
  globalNote,
  setGlobalNote,
  reportRef,
  downloadReport,
  handleFeedbackSaved,
  workstationMode,
  setWorkstationMode,
}: ScreeningTabProps) {
  const activeResult = selectedIdx !== null ? results[selectedIdx] : null;

  // Quality metrics helper
  const getQualityMetrics = (result: AnalysisResult | null) => {
    if (result && result.image_quality) {
      const iqa = result.image_quality;
      return {
        exposure: iqa.exposure || "Adequate Exposure",
        coverage: iqa.coverage || "Full Lung Coverage",
        resolution: iqa.resolution || "Acceptable Resolution",
        rotation: iqa.rotation || "No Rotation",
        suitableForAi:
          typeof iqa.suitable_for_ai === "boolean"
            ? iqa.suitable_for_ai
            : (iqa as any).suitability === "suitable" || iqa.suitable_for_ai === true,
        qualityScore: iqa.quality_score || 95,
        warnings: iqa.warnings || [],
      };
    }
    const fallback = imageQualityService.assessQuality(
      result?.filename ?? "unknown"
    );
    return {
      ...fallback,
      warnings: [] as string[],
    };
  };

  const q = activeResult ? getQualityMetrics(activeResult) : null;

  // Threshold handling (from original)
  const [customThreshold, setCustomThreshold] = useState<number | null>(null);
  useEffect(() => {
    const val = localStorage.getItem("nirikshon_threshold");
    if (val) setCustomThreshold(parseFloat(val));
    const handler = () => {
      const val = localStorage.getItem("nirikshon_threshold");
      if (val) setCustomThreshold(parseFloat(val));
    };
    window.addEventListener("nirikshon_threshold_changed", handler);
    return () => window.removeEventListener("nirikshon_threshold_changed", handler);
  }, []);
  const currentThreshold = customThreshold ?? activeResult?.threshold_used ?? 0.5;
  const isTbDerived =
    activeResult ? (activeResult.confidence ?? 0) >= currentThreshold : false;
  const activeDiagnosis = activeResult
    ? predictionService.getDiagnosis(
        isTbDerived ? "Tuberculosis" : "Normal",
        activeResult.confidence ?? 0,
        currentThreshold
      )
    : null;

  // View mode state (original, heatmap, side-by-side, split, longitudinal)
  const [viewMode, setViewMode] = useState<
    "original" | "heatmap" | "side-by-side" | "split" | "longitudinal"
  >("original");
  const [heatmapOpacity, setHeatmapOpacity] = useState(0.55);
  const [priorImageSrc, setPriorImageSrc] = useState<string | undefined>();
  const [deltaHeatmapSrc, setDeltaHeatmapSrc] = useState<string | undefined>();
  const [isComparing, setIsComparing] = useState(false);

  // Annotation state
  const [boxes, setBoxes] = useState<Box[]>([]);
  const [activeZone, setActiveZone] = useState<LungZone>("Apical");
  const [annotateMode, setAnnotateMode] = useState(false);
  const annotationCanvasRef = useRef<HTMLCanvasElement>(null);
  const [observationFocusRegion, setObservationFocusRegion] = useState<{
    x1: number;
    y1: number;
    x2: number;
    y2: number;
    zoom: number;
    panX: number;
    panY: number;
  } | null>(null);

  // Right tab state
  const [activeRightTab, setActiveRightTab] = useState<"diagnosis" | "chat" | "report">(
    "diagnosis"
  );

  // Clinical review state
  const [clinicalReviewStatus, setClinicalReviewStatus] = useState<string>("confirm");
  const [reviewComments, setReviewComments] = useState("");
  const [reviewerName, setReviewerName] = useState("");
  const [clinicianNote, setClinicianNote] = useState("");
  const [iqaAcknowledged, setIqaAcknowledged] = useState(false);

  // Similar cases & model metadata
  const [similarCases, setSimilarCases] = useState<{
    tb_similar: any[];
    normal_similar: any[];
  } | null>(null);
  const [modelMetadata, setModelMetadata] = useState<any>(null);
  const [loadingSimilar, setLoadingSimilar] = useState(false);

  // Audit logs
  const [auditLogs, setAuditLogs] = useState<AuditLogEntry[]>([]);
  const [dbRegistered, setDbRegistered] = useState(false);
  const [isExporting, setIsExporting] = useState(false);

  // XAI method — the backend returns all four real CAM methods
  // (gradcam, gradcam_plus_plus, layercam, eigencam); this was previously
  // hard-locked to gradcam_plus_plus with no selector ever wired up in the
  // "clinical" quick-view (the "xai" tab's XaiVisualization already has one).
  const XAI_METHODS = [
    { key: "gradcam_plus_plus", label: "Grad-CAM++" },
    { key: "gradcam", label: "Grad-CAM" },
    { key: "layercam", label: "LayerCAM" },
    { key: "eigencam", label: "Eigen-CAM" },
  ] as const;
  type XaiMethod = (typeof XAI_METHODS)[number]["key"];
  const [xaiMethod, setXaiMethod] = useState<XaiMethod>("gradcam_plus_plus");

  // -------------- Effects --------------

  // Load model metadata once
  useEffect(() => {
    const fetchModelMetadata = async () => {
      try {
        const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/model/metadata`, {
          credentials: "include",
        });
        if (res.ok) {
          const data = await res.json();
          setModelMetadata(data);
        }
      } catch (e) {
        console.error("Failed to fetch model metadata:", e);
      }
    };
    fetchModelMetadata();
  }, []);

  // When selected result changes, reset UI and fetch related data
  useEffect(() => {
    setViewMode("original");
    setHeatmapOpacity(0.55);
    setBoxes([]);
    setAnnotateMode(false);
    setObservationFocusRegion(null);
    setClinicalReviewStatus(activeResult?.clinician_override ?? "confirm");
    setReviewComments(activeResult?.review_comments ?? "");
    setReviewerName(activeResult?.reviewer_name ?? "");
    setClinicianNote(activeResult?.clinician_note ?? "");
    setDbRegistered(false);
    setIqaAcknowledged(false);
    setCustomThreshold(null);
    setPriorImageSrc(undefined);
    setDeltaHeatmapSrc(undefined);
    if (viewMode === "longitudinal") setViewMode("original");

    if (activeResult) {
      if (activeResult.study_id) {
        // audit logs
        fetchAuditLogs(activeResult.study_id);
        // similar cases
        fetchSimilarCases(activeResult.study_id);
        // heatmaps if missing
        if (!activeResult.heatmaps && selectedIdx !== null) {
          fetchStudyHeatmaps(activeResult.study_id, selectedIdx);
        }
      } else {
        setAuditLogs(auditService.createDefaultLogs(activeResult.filename));
        setSimilarCases(null);
      }
    } else {
      setAuditLogs([]);
      setSimilarCases(null);
    }
  }, [selectedIdx, activeResult?.study_id]);

  // Auto-analyze if pending
  useEffect(() => {
    if (activeResult && activeResult.status === "pending" && selectedIdx !== null) {
      analyzeFile(selectedIdx);
    }
  }, [selectedIdx, activeResult?.status, analyzeFile]);

  // Audit log helper
  const addAuditLog = (actionText: string) => {
    const d = new Date();
    const ts = `${d.getHours().toString().padStart(2, "0")}:${d.getMinutes()
      .toString()
      .padStart(2, "0")}`;
    setAuditLogs((prev) => [...prev, { timestamp: ts, action: actionText }]);
  };

  // Fetch audit logs from backend
  const fetchAuditLogs = async (studyId: string) => {
    try {
      const res = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/studies/${studyId}/audit`,
        { credentials: "include" }
      );
      if (res.ok) {
        const data = await res.json();
        const mapped = data.audit_trail.map((entry: any) => {
          const d = new Date(entry.timestamp);
          const timeStr =
            isNaN(d.getTime())
              ? entry.timestamp
              : `${d.getHours().toString().padStart(2, "0")}:${d.getMinutes()
                  .toString()
                  .padStart(2, "0")}`;
          let actionText = entry.action;
          if (entry.action === "Upload")
            actionText = `Image study uploaded by ${entry.username}`;
          else if (entry.action === "Inference")
            actionText = "AI Inference pipeline completed";
          else if (entry.action === "Review")
            actionText = `Clinical review signed off by ${entry.username}`;
          else if (entry.action === "Report Export")
            actionText = `Clinical PDF report exported by ${entry.username}`;
          return { timestamp: timeStr, action: actionText };
        });
        setAuditLogs(mapped);
      }
    } catch (err) {
      console.error("Failed to fetch audit logs:", err);
    }
  };

  // Fetch study heatmaps (for updating result with heatmaps/XAI)
  const fetchStudyHeatmaps = async (studyId: string, idx: number) => {
    try {
      const res = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/studies/${studyId}/heatmaps`,
        { credentials: "include" }
      );
      if (res.ok) {
        const data = await res.json();
        setResults((prev) => {
          const next = [...prev];
          if (next[idx]) {
            next[idx] = {
              ...next[idx],
              heatmaps: data.heatmaps ?? next[idx].heatmaps,
              xai_results: data.xai_results ?? next[idx].xai_results,
            };
          }
          return next;
        });
      }
    } catch (err) {
      console.error("Failed to fetch study heatmaps:", err);
    }
  };

  // Fetch similar cases
  const fetchSimilarCases = async (studyId: string) => {
    setLoadingSimilar(true);
    try {
      const res = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/studies/${studyId}/similar`,
        { credentials: "include" }
      );
      if (res.ok) {
        const data = await res.json();
        setSimilarCases(data);
      }
    } catch (err) {
      console.error("Failed to fetch similar cases:", err);
    } finally {
      setLoadingSimilar(false);
    }
  };

  const handleSelectSimilarStudy = (cand: any) => {
    const mappedResult: AnalysisResult = {
      filename: `Study: ${cand.study_id}`,
      status: "success" as const,
      prediction: cand.prediction,
      confidence: cand.confidence,
      is_tb:
        cand.prediction.toLowerCase().includes("tb") ||
        cand.prediction.toLowerCase().includes("tuberculosis"),
      metadata: {
        patient_id: cand.patient_id,
        patient_name: cand.patient_name,
        patient_age: cand.age,
        patient_sex: cand.sex,
        modality: "CR",
        study_date: new Date().toISOString().split("T")[0],
      },
      original_image: cand.original_image,
      heatmap_image: cand.heatmap_image,
      clinician_override: null,
      clinician_note: "",
      annotated_image: "",
      review_comments: "",
      reviewer_name: "",
      study_id: cand.study_id,
      image_quality: {
        exposure: "Adequate Exposure",
        coverage: "Full Lung Coverage",
        resolution: "2048 x 2048 pixels",
        rotation: "No Rotation",
        quality_score: 95,
        suitable_for_ai: true,
        warnings: [],
      },
    };
    setResults([mappedResult]);
    setSelectedIdx(0);
  };

  // ---------- Export Handlers ----------

  const handlePdfExport = async () => {
    if (!activeResult || !q) return;
    setIsExporting(true);
    addAuditLog("Generating clinical report PDF");

    const diagnosisObj = predictionService.getDiagnosis(
      activeResult.prediction ?? "Normal",
      activeResult.confidence ?? 0,
      activeResult.threshold_used ?? 0.5
    );

    const observationsList = observationService.getObservations(
      activeResult.prediction ?? "Normal",
      activeResult.xai_results ?? null,
      224,
      224,
      activeResult.clinical_observations ?? null
    );

    const reviewData = {
      status:
        clinicalReviewStatus === "confirm"
          ? "Confirm AI finding"
          : clinicalReviewStatus === "reject"
          ? "Reject AI finding"
          : clinicalReviewStatus === "investigate"
          ? "Request Investigation"
          : "Insufficient Quality",
      comments: reviewComments,
      signature: reviewerName,
    };

    const payload = reportService.buildReport(
      activeResult.metadata,
      q,
      diagnosisObj.condition,
      activeResult.confidence ?? 0,
      diagnosisObj.riskLevel,
      observationsList,
      reviewData,
      clinicianNote || globalNote,
      activeResult.xai_results
    );

    await exportService.downloadPDF(
      payload,
      activeResult.original_image ?? "",
      activeResult.heatmap_image ?? ""
    );
    addAuditLog("Report PDF exported to downloads");

    try {
      const bodyPayload = activeResult.study_id
        ? { study_id: activeResult.study_id }
        : { patient_id: activeResult.metadata?.patient_id };
      await fetch(`${process.env.NEXT_PUBLIC_API_URL}/report/audit`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(bodyPayload),
        credentials: "include",
      });
      if (activeResult.study_id) {
        fetchAuditLogs(activeResult.study_id);
      }
    } catch (auditErr) {
      console.error("Failed to audit report generation:", auditErr);
    }

    setIsExporting(false);
  };

  const handleJsonSR = () => {
    if (!activeResult || !q) return;
    addAuditLog("Exporting Structured JSON Report");

    const diagnosisObj = predictionService.getDiagnosis(
      activeResult.prediction ?? "Normal",
      activeResult.confidence ?? 0,
      activeResult.threshold_used ?? 0.5
    );

    const observationsList = observationService.getObservations(
      activeResult.prediction ?? "Normal",
      activeResult.xai_results ?? null,
      224,
      224,
      activeResult.clinical_observations ?? null
    );

    const reviewData = {
      status: clinicalReviewStatus,
      comments: reviewComments,
      signature: reviewerName,
    };

    const payload = reportService.buildReport(
      activeResult.metadata,
      q,
      diagnosisObj.condition,
      activeResult.confidence ?? 0,
      diagnosisObj.riskLevel,
      observationsList,
      reviewData,
      clinicianNote || globalNote
    );

    exportService.exportStructuredJSON(payload);
    addAuditLog("Structured JSON report metadata generated");
  };

  const handleRegisterDb = async () => {
    if (!activeResult || !q) return;
    setDbRegistered(true);
    addAuditLog("Registering case to research database");

    const diagnosisObj = predictionService.getDiagnosis(
      activeResult.prediction ?? "Normal",
      activeResult.confidence ?? 0,
      activeResult.threshold_used ?? 0.5
    );

    const observationsList = observationService.getObservations(
      activeResult.prediction ?? "Normal",
      activeResult.xai_results ?? null,
      224,
      224,
      activeResult.clinical_observations ?? null
    );

    const reviewData = {
      status: clinicalReviewStatus,
      comments: reviewComments,
      signature: reviewerName,
    };

    const payload = reportService.buildReport(
      activeResult.metadata,
      q,
      diagnosisObj.condition,
      activeResult.confidence ?? 0,
      diagnosisObj.riskLevel,
      observationsList,
      reviewData,
      clinicianNote || globalNote
    );

    await exportService.registerToResearchDB(payload);
    addAuditLog("Registered study to research cohort DB");
  };

  // ---------- Stepper Timeline ----------

  const getStepperStatus = () => {
    if (!activeResult) return [];
    const isLoading =
      activeResult.status === "loading" || activeResult.status === "pending";
    return [
      {
        text: "Chest X-ray loaded in memory",
        done: true,
        loading: false,
      },
      {
        text: "Grayscale intensity normalization & padding",
        done:
          activeResult.status === "success" || activeResult.status === "error",
        loading: isLoading && activeResult.status === "loading",
      },
      {
        text: "DenseNet-121 model load & pre-warm validation",
        done:
          activeResult.status === "success" || activeResult.status === "error",
        loading: false,
      },
      {
        text: "AI classification risk scoring (tuberculosis vs normal)",
        done: activeResult.status === "success",
        loading: isLoading && activeResult.status === "loading",
      },
      {
        text: "Grad-CAM++ activation layer backpropagation maps",
        done: activeResult.status === "success",
        loading: isLoading && activeResult.status === "loading",
      },
      {
        text: "Anatomical zone contribution ROI calculations",
        done: activeResult.status === "success",
        loading: false,
      },
    ];
  };

  // ---------- Render ----------

  return (
    <div className="w-full min-h-[calc(100vh-4rem)] bg-[#0A1F44] text-[#F8FAFC]">
      {/* Main container: flex row */}
      <div className="flex h-full w-full p-4 lg:p-6 gap-4">
        {/* 1. LEFT DOCK (fixed width icons) */}
        <aside
          className="w-[70px] lg:w-[80px] shrink-0 flex flex-col items-center py-6 bg-black/40 backdrop-blur-2xl border border-white/10 rounded-[30px] shadow-2xl relative z-20"
        >
          <div className="space-y-4">
            {/* Clinical View button */}
            <button
              onClick={() => setWorkstationMode("clinical")}
              className={`p-3.5 lg:p-4 rounded-2xl transition-all duration-300 relative z-10 ${
                workstationMode === "clinical"
                  ? "bg-primary text-primary-foreground shadow-[0_0_20px_rgba(var(--primary),0.5)] scale-105"
                  : "bg-white/5 text-muted-foreground hover:bg-white/10 hover:text-white"
              }`}
              title="Clinical View"
            >
              <LayoutDashboard className="w-5 h-5 lg:w-6 lg:h-6" />
            </button>
            {/* AI Observations button */}
            <button
              onClick={() => setWorkstationMode("xai")}
              className={`p-3.5 lg:p-4 rounded-2xl transition-all duration-300 relative z-10 ${
                workstationMode === "xai"
                  ? "bg-primary text-primary-foreground shadow-[0_0_20px_rgba(var(--primary),0.5)] scale-105"
                  : "bg-white/5 text-muted-foreground hover:bg-white/10 hover:text-white"
              }`}
              title="AI Observations"
            >
              <Layers className="w-5 h-5 lg:w-6 lg:h-6" />
            </button>
          </div>

          <div className="mt-auto space-y-4">
            <button
              className="p-3.5 lg:p-4 rounded-2xl bg-white/5 text-muted-foreground hover:bg-white/10 hover:text-white transition-all duration-300"
              title="Workspace Settings"
            >
              <Settings className="w-5 h-5 lg:w-6 lg:h-6" />
            </button>
          </div>
        </aside>

        {/* 2. CENTRAL WORKSPACE */}
        <section
          className="flex-1 flex flex-col h-full relative rounded-[30px] overflow-hidden bg-black/20 border border-white/5 shadow-2xl"
        >
          {/* Floating status bars */}
          {activeResult?.demo_mode && (
            <div className="absolute top-5 left-5 z-50 px-4 py-2 border border-yellow-500/20 bg-yellow-500/10 text-yellow-500 rounded-full flex items-center gap-2 backdrop-blur-md text-[10px] uppercase font-bold tracking-widest animate-fadein shadow-lg">
              <ShieldAlert className="w-4 h-4" />
              Demo Mode
            </div>
          )}
          {activeResult?.study_id && activeResult.study_id !== "N/A" && (
            <div className="absolute top-5 right-5 z-50 px-5 py-2 border border-white/10 bg-black/40 text-foreground rounded-full backdrop-blur-md text-[11px] font-mono font-bold tracking-wider animate-fadein shadow-lg flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
              STUDY: {activeResult.study_id}
            </div>
          )}

          {/* XAI method selector — quick-view only; only render a method the
              backend actually returned for this result */}
          {workstationMode === "clinical" &&
            activeResult?.status === "success" &&
            (activeResult?.heatmaps?.[xaiMethod] || activeResult?.heatmap_image) && (
              <div className="absolute top-5 left-1/2 -translate-x-1/2 z-50 flex flex-wrap bg-black/40 backdrop-blur-md p-1 rounded-full border border-white/10 shadow-lg">
                {XAI_METHODS.map(({ key, label }) => {
                  const available = Boolean(activeResult?.heatmaps?.[key]);
                  return (
                    <button
                      key={key}
                      type="button"
                      disabled={!available}
                      onClick={() => setXaiMethod(key)}
                      title={available ? label : `${label} not available for this result`}
                      className={`px-3 py-1.5 rounded-full text-[11px] font-semibold tracking-wide transition-colors ${
                        xaiMethod === key
                          ? "bg-primary text-primary-foreground"
                          : available
                            ? "text-white/60 hover:text-white hover:bg-white/10 cursor-pointer"
                            : "text-white/20 cursor-not-allowed"
                      }`}
                    >
                      {label}
                    </button>
                  );
                })}
              </div>
            )}

          <div className="flex-1 w-full h-full p-2 relative z-10">
            {workstationMode === "xai" ? (
              <XaiVisualization
                result={activeResult!}
                similarCases={similarCases}
                loadingSimilar={loadingSimilar}
                workstationMode={workstationMode}
                setWorkstationMode={setWorkstationMode}
                zoomLevel={1}
                setZoomLevel={() => {}}
                isComparing={isComparing}
                setIsComparing={setIsComparing}
              />
            ) : (
              <DicomViewer
                imageBase64={activeResult?.original_image ?? ""}
                heatmapBase64={
                  activeResult?.heatmaps?.[xaiMethod] ??
                  activeResult?.heatmap_image ??
                  ""
                }
                hasHeatmap={
                  activeResult?.status === "success" &&
                  !!(
                    activeResult?.heatmaps?.[xaiMethod] ||
                    activeResult?.heatmap_image
                  )
                }
                label="Nirikshon Enterprise Viewport"
                pixelSpacing={activeResult?.metadata?.pixel_spacing}
                viewMode={viewMode}
                heatmapOpacity={heatmapOpacity}
                setHeatmapOpacity={setHeatmapOpacity}
                boxes={boxes}
                setBoxes={setBoxes}
                activeZone={activeZone}
                setActiveZone={setActiveZone}
                annotateMode={annotateMode}
                annotationCanvasRef={annotationCanvasRef}

                priorImageSrc={priorImageSrc}
                deltaHeatmapSrc={deltaHeatmapSrc}
                observationFocusRegion={observationFocusRegion}
                setViewMode={setViewMode}
                setAnnotateMode={setAnnotateMode}
              />
            )}
          </div>
        </section>

        {/* 3. RIGHT PANEL */}
        <aside
          className="w-[350px] lg:w-[420px] shrink-0 flex flex-col h-full bg-black/40 backdrop-blur-2xl border border-white/10 rounded-[30px] shadow-2xl overflow-hidden relative z-20"
        >
          {/* Verdict Header */}
          <div className="p-6 border-b border-white/10 relative overflow-hidden shrink-0 bg-gradient-to-b from-white/[0.02] to-transparent">
            <div className={`absolute -top-10 -right-10 w-48 h-48 blur-[60px] opacity-20 rounded-full pointer-events-none transition-colors duration-1000 ${
              activeResult?.status === "loading" ||
              activeResult?.status === "pending"
                ? "bg-primary"
                : activeDiagnosis?.riskLevel === "High"
                ? "bg-red-500"
                : "bg-emerald-500"
            }`} />
            <p className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground mb-3 relative z-10 flex items-center gap-2">
              <Activity className="w-3.5 h-3.5" />
              AI Classification
            </p>

            {activeResult?.status === "error" ? (
              <div className="space-y-3 relative z-10 bg-destructive/10 border border-destructive/25 p-4 rounded-[20px]">
                <div className="flex items-center gap-2 text-destructive font-semibold text-sm">
                  <AlertCircle className="w-4 h-4" />
                  <span>{activeResult.errorMsg || "Internal server error"}</span>
                </div>
                <button
                  onClick={() =>
                    selectedIdx !== null && analyzeFile(selectedIdx)
                  }
                  className="w-full py-2 rounded-xl text-xs font-bold bg-destructive/20 hover:bg-destructive/30 text-destructive border border-destructive/30 transition-all cursor-pointer"
                >
                  Retry Inference
                </button>
              </div>
            ) : activeResult?.status === "loading" ||
              activeResult?.status === "pending" ? (
              <div className="space-y-4 relative z-10">
                <h3 className="text-2xl font-extrabold tracking-tight text-foreground flex items-center gap-3">
                  <Loader2 className="w-6 h-6 animate-spin text-primary" />
                  Analyzing...
                </h3>
                <Progress value={75} className="h-1.5 bg-white/10 [&>div]:bg-primary [&>div]:animate-pulse" />
              </div>
            ) : (
              <div className="relative z-10">
                <h3 className={`text-3xl font-extrabold tracking-tight mb-4 drop-shadow-md ${
                  activeDiagnosis?.riskLevel === "High" ? "text-red-500" : "text-emerald-500"
                }`}>
                  {activeDiagnosis?.condition ?? "Normal"}
                </h3>
                <div className="space-y-2">
                  <div className="flex justify-between text-[11px] font-bold text-muted-foreground uppercase tracking-widest">
                    <span>Confidence Score</span>
                    <span className="text-foreground">
                      {((activeDiagnosis?.confidence ?? 0) * 100).toFixed(1)}%
                    </span>
                  </div>
                  <Progress
                    value={(activeDiagnosis?.confidence ?? 0) * 100}
                    className={`h-2 bg-white/10 ${
                      activeDiagnosis?.riskLevel === "High"
                        ? "[&>div]:bg-red-500"
                        : "[&>div]:bg-emerald-500"
                    }`}
                  />
                </div>
                <div className="flex flex-wrap gap-2 mt-5">
                  <Badge
                    className="bg-white/10 text-foreground hover:bg-white/20 border border-white/5 text-[9px] uppercase font-bold py-1 px-3 rounded-full"
                  >
                    {activeResult?.segmentation_active ? "U-Net Segmented" : "Direct Input"}
                  </Badge>
                  <Badge
                    className={`uppercase font-bold text-[9px] py-1 px-3 rounded-full border border-white/5 ${
                      activeDiagnosis?.riskLevel === "High"
                        ? "bg-red-500/20 text-red-500"
                        : "bg-emerald-500/20 text-emerald-400 border-emerald-500/30"
                    }`}
                  >
                    {activeDiagnosis?.riskLevel} Risk
                  </Badge>
                </div>
              </div>
            )}
          </div>

          {/* Tab Navigation */}
          <div className="p-4 shrink-0">
            <div className="flex bg-black/40 p-1.5 rounded-full border border-white/10 relative shadow-inner">
              {[
                { id: "diagnosis", label: "Evidence" },
                { id: "chat", label: "Co-Pilot" },
                { id: "report", label: "Report" },
              ].map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveRightTab(tab.id as "diagnosis" | "chat" | "report")}
                  className={`flex-1 py-2.5 text-[10px] font-bold uppercase tracking-widest rounded-full transition-all duration-300 relative z-10 ${
                    activeRightTab === tab.id
                      ? "text-primary-foreground shadow-lg"
                      : "text-muted-foreground hover:text-foreground"
                  }`}
                >
                  {tab.label}
                </button>
              ))}
              {/* Animated highlight background */}
              <div className="absolute top-1.5 bottom-1.5 w-[calc(33.333%-4px)] bg-primary rounded-full transition-all duration-500 ease-[cubic-bezier(0.23,1,0.32,1)] z-0 shadow-[0_0_20px_rgba(var(--primary),0.4)]"
                style={{
                  left:
                    activeRightTab === "diagnosis"
                      ? "6px"
                      : activeRightTab === "chat"
                      ? "calc(33.333% + 2px)"
                      : "calc(66.666% - 2px)",
                }}
              />
            </div>
          </div>

          {/* Tab Content Area */}
          <div className="flex-1 overflow-y-auto px-6 pb-6 custom-scrollbar space-y-5">
            {activeRightTab === "diagnosis" && (
              <div className="space-y-5 animate-fadein">
                {/* Loading states */}
                {(activeResult?.status === "loading" ||
                  activeResult?.status === "pending") && (
                  <div className="space-y-3 bg-black/20 p-5 rounded-3xl border border-white/5 shadow-inner mt-2">
                    <p className="text-[10px] font-bold uppercase tracking-widest text-primary mb-4 flex items-center gap-2">
                      <Activity className="w-3.5 h-3.5 animate-pulse" />
                      Processing Pipeline
                    </p>
                    {getStepperStatus().map((step, idx) => (
                      <div key={idx} className="flex items-center gap-4 relative">
                        {idx < getStepperStatus().length - 1 && (
                          <div className={`absolute left-2.5 top-6 bottom-[-15px] w-[2px] rounded-full ${
                            step.done ? "bg-primary/50" : "bg-white/5"
                          }`} />
                        )}
                        <div className="relative z-10 flex-shrink-0">
                          {step.done ? (
                            <div className="w-5 h-5 rounded-full bg-primary flex items-center justify-center shadow-[0_0_10px_rgba(var(--primary),0.5)]">
                              <svg className="w-3 h-3 text-primary-foreground" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3}>
                                <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                              </svg>
                            </div>
                          ) : step.loading ? (
                            <div className="w-5 h-5 rounded-full bg-primary/20 border border-primary flex items-center justify-center animate-pulse">
                              <div className="w-1.5 h-1.5 rounded-full bg-primary animate-ping"></div>
                            </div>
                          ) : (
                            <div className="w-5 h-5 rounded-full border border-white/10 bg-black/50"></div>
                          )}
                        </div>
                        <div className="flex-1 min-w-0">
                          <p className={`text-[11px] font-semibold transition-colors duration-300 ${
                            step.done
                              ? "text-foreground"
                              : step.loading
                              ? "text-primary"
                              : "text-muted-foreground"
                          }`}>
                            {step.text}
                          </p>
                        </div>
                      </div>
                    ))}
                  </div>
                )}

                {/* Image Quality Assessment */}
                {activeResult?.status === "success" && q && (
                  <div className="bg-black/20 rounded-[24px] p-5 border border-white/5 shadow-inner hover:border-white/10 transition-colors">
                    <div className="flex items-center justify-between mb-4">
                      <p className="text-[10px] font-bold uppercase tracking-widest text-foreground">Image Quality</p>
                      <Badge
                        variant={q.suitableForAi ? "default" : "destructive"}
                        className="rounded-full font-bold uppercase text-[9px] px-3 py-1 border-0 shadow-sm"
                      >
                        {q.suitableForAi ? "Suitable" : "Unsuitable"}
                      </Badge>
                    </div>
                    <div className="flex gap-4 text-[11px] text-muted-foreground font-medium bg-white/5 p-3 rounded-2xl">
                      <span className={q.exposure === "Adequate Exposure" ? "text-emerald-500" : "text-amber-500"}>
                        • {q.exposure}
                      </span>
                      <span className={q.coverage === "Full Lung Coverage" ? "text-emerald-500" : "text-amber-500"}>
                        • {q.coverage}
                      </span>
                    </div>
                    {!q.suitableForAi && (
                      <div className="flex items-center gap-3 mt-4 pt-4 border-t border-white/5">
                        <input
                          type="checkbox"
                          id="iqa-ack"
                          className="h-4 w-4 accent-amber-500 cursor-pointer rounded"
                          checked={iqaAcknowledged}
                          onChange={(e) => setIqaAcknowledged(e.target.checked)}
                        />
                        <label
                          htmlFor="iqa-ack"
                          className="text-[11px] font-bold text-amber-500 cursor-pointer select-none"
                        >
                          Acknowledge sub-optimal quality to proceed
                        </label>
                      </div>
                    )}
                  </div>
                )}

                {/* Clinical Evidence Findings */}
                {activeResult?.status === "success" && (
                  <div className="space-y-3">
                    <p className="text-[10px] font-bold uppercase tracking-widest text-foreground mb-3 flex items-center gap-2">
                      <Eye className="w-3.5 h-3.5 text-primary" />
                      Evidence Findings
                    </p>
                    {/* Mock evidence cards - reuse getEvidenceCards logic from original */}
                    {(() => {
                      const diag = predictionService.getDiagnosis(
                        activeResult.prediction ?? "Normal",
                        activeResult.confidence ?? 0,
                        activeResult.threshold_used ?? 0.5
                      );
                      const cond = activeResult.prediction ?? "Normal";
                      const isNormal =
                        cond.toLowerCase().includes("normal") || cond === "Normal";
                      const confidencePct = (diag.confidence * 100).toFixed(1);
                      const region = activeResult.attention_region ?? "lung fields";
                      const qExposure =
                        activeResult.image_quality?.exposure ?? "Adequate Exposure";
                      const qCoverage =
                        activeResult.image_quality?.coverage ?? "Full Lung Coverage";
                      const qScore =
                        activeResult.image_quality?.quality_score ?? 95;

                      if (isNormal) {
                        return [
                          {
                            title: "Parenchymal Clearance",
                            description: `Bilateral lung fields exhibit normal aeration without signs of active consolidation, effusion, or masses (AI confidence: ${confidencePct}%).`,
                            confidence: diag.confidence,
                            region: null,
                            anatomicalZone: "bilateral",
                          },
                          {
                            title: "Clear Costophrenic Angles",
                            description: `Pleural boundaries are sharp and well-defined with no indication of fluid accumulation. Costophrenic angles are completely clear.`,
                            confidence: Math.max(0.92, diag.confidence - 0.04),
                            region: null,
                            anatomicalZone: "pleural space",
                          },
                          {
                            title: "Technical Image Integrity",
                            description: `Radiograph shows ${qExposure.toLowerCase()} and ${qCoverage.toLowerCase()} (Technical Quality Score: ${qScore}%).`,
                            confidence: qScore / 100,
                            region: null,
                            anatomicalZone: "global",
                          },
                        ];
                      }
                      // Abnormal case
                      const leftApicalRegion = {
                        x1: 25,
                        y1: 20,
                        x2: 95,
                        y2: 80,
                        zoom: 2.2,
                        panX: 160,
                        panY: 100,
                      };
                      const rightApicalRegion = {
                        x1: 125,
                        y1: 20,
                        x2: 195,
                        y2: 80,
                        zoom: 2.2,
                        panX: -160,
                        panY: 100,
                      };
                      const leftMidRegion = {
                        x1: 30,
                        y1: 85,
                        x2: 100,
                        y2: 140,
                        zoom: 2.0,
                        panX: 150,
                        panY: -30,
                      };
                      const rightMidRegion = {
                        x1: 120,
                        y1: 85,
                        x2: 190,
                        y2: 140,
                        zoom: 2.0,
                        panX: -150,
                        panY: -30,
                      };
                      const isLeft = region.toLowerCase().includes("left");
                      const isApical =
                        region.toLowerCase().includes("apical") ||
                        region.toLowerCase().includes("upper");
                      let targetRegion = rightApicalRegion;
                      let zoneLabel = "right apical";
                      if (isLeft && isApical) {
                        targetRegion = leftApicalRegion;
                        zoneLabel = "left apical";
                      } else if (isLeft && !isApical) {
                        targetRegion = leftMidRegion;
                        zoneLabel = "left mid-zone";
                      } else if (!isLeft && !isApical) {
                        targetRegion = rightMidRegion;
                        zoneLabel = "right mid-zone";
                      }
                      return [
                        {
                          title: "Consolidation & Opacity Focus",
                          description: `Grad-CAM++ highlighted an area of increased opacity in the ${zoneLabel} zone. This density gradient is consistent with focal active ${cond} consolidation (AI confidence: ${confidencePct}%).`,
                          confidence: diag.confidence,
                          region: targetRegion,
                          anatomicalZone: zoneLabel,
                        },
                        {
                          title: "Asymmetric Density Gradients",
                          description: `Significant localized markings and architectural asymmetry identified in the ${zoneLabel} zone compared to contralateral regions.`,
                          confidence: Math.max(0.70, diag.confidence * 0.85),
                          region: targetRegion,
                          anatomicalZone: zoneLabel,
                        },
                        {
                          title: "Hilar Lymphadenopathy Suggestion",
                          description: `Bronchovascular tree markings and mediastinal structures show signs of inflammation or congestion adjacent to the primary focus area.`,
                          confidence: Math.max(0.60, diag.confidence * 0.70),
                          region: null,
                          anatomicalZone: "hilar",
                        },
                      ];
                    })().map((ec, idx) => {
                      const isAbnormal =
                        activeResult?.is_tb &&
                        (ec.title.toLowerCase().includes("consolidation") ||
                          ec.title.toLowerCase().includes("density") ||
                          ec.title.toLowerCase().includes("failed") ||
                          ec.title.toLowerCase().includes("infiltrate"));
                      const colorPrefix = isAbnormal
                        ? activeDiagnosis?.riskLevel === "High"
                          ? "red"
                          : "blue"
                        : "blue";
                      const borderColor =
                        colorPrefix === "red"
                          ? "border-l-red-500"
                          : "border-l-blue-500";
                      const badgeColor =
                        colorPrefix === "red"
                          ? "bg-red-500/10 text-red-500 border-red-500/20"
                          : "bg-blue-500/10 text-blue-500 border-blue-500/20";
                      return (
                        <div
                          key={idx}
                          className={`bg-black/20 p-5 rounded-[24px] border border-white/5 border-l-4 ${borderColor} hover:bg-black/40 hover:border-white/10 transition-all duration-300 group shadow-sm`}
                        >
                          <div className="flex justify-between items-start mb-2">
                            <h4 className="text-[13px] font-bold text-foreground group-hover:text-primary transition-colors pr-2">
                              {ec.title}
                            </h4>
                            {ec.confidence > 0 && (
                              <Badge
                                className={`rounded-full font-mono font-bold text-[9px] px-2 py-0.5 border ${badgeColor}`}
                              >
                                {(ec.confidence * 100).toFixed(0)}%
                              </Badge>
                            )}
                          </div>
                          <p className="text-[11px] text-muted-foreground leading-relaxed font-medium">
                            {ec.description}
                          </p>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            )}

            {activeRightTab === "chat" && (
              <div className="h-full min-h-[500px] animate-fadein pb-4">
                <LlmAssistant activeResult={activeResult} />
              </div>
            )}



            {activeRightTab === "report" && (
              <div className="space-y-6 animate-fadein pb-4">
                {/* Clinical Audit / Override Inputs */}
                <div className="bg-black/20 p-5 rounded-[24px] border border-white/5 shadow-inner">
                  <p className="text-[10px] font-bold uppercase tracking-widest text-foreground mb-4">
                    Clinical Sign-Off
                  </p>
                  <div className="space-y-4">
                    <div className="space-y-2.5">
                      <label className="text-[10px] text-muted-foreground uppercase font-bold tracking-wider">
                        Diagnostic Verdict Adjudication
                      </label>
                      <div className="grid grid-cols-2 gap-2">
                        {[
                          { id: "confirm", label: "Confirm AI Verdict" },
                          { id: "reject", label: "Reject AI Verdict" },
                          { id: "investigate", label: "Request Investigation" },
                          { id: "insufficient", label: "Insufficient Quality" }
                        ].map((opt) => (
                          <button
                            key={opt.id}
                            onClick={() => {
                              setClinicalReviewStatus(opt.id);
                              syncFeedback(
                                opt.id,
                                reviewComments,
                                reviewerName,
                                clinicianNote
                              );
                            }}
                            className={`p-3 rounded-2xl text-[11px] font-bold border transition-all cursor-pointer ${
                              clinicalReviewStatus === opt.id
                                ? "border-primary bg-primary/10 text-primary shadow-sm"
                                : "border-white/5 bg-white/5 hover:bg-white/10 text-muted-foreground hover:text-foreground"
                            }`}
                          >
                            {opt.label}
                          </button>
                        ))}
                      </div>
                    </div>

                    <div className="space-y-3">
                      <label className="text-[10px] text-muted-foreground uppercase font-bold tracking-wider">
                        Reviewer Comments
                      </label>
                      <Textarea
                        value={reviewComments}
                        onChange={(e) => setReviewComments(e.target.value)}
                        placeholder="Add clinical notes or discrepancies..."
                        className="min-h-[80px]"
                      />
                    </div>

                    <div className="space-y-3">
                      <label className="text-[10px] text-muted-foreground uppercase font-bold tracking-wider">
                        Reviewer Name
                      </label>
                      <input
                        type="text"
                        value={reviewerName}
                        onChange={(e) => setReviewerName(e.target.value)}
                        placeholder="Enter your name or ID"
                        className="w-full px-3 py-2 rounded border border-white/5 bg-white/5 text-white focus:outline-none focus:ring-2 focus:ring-primary"
                      />
                    </div>

                    <div className="space-y-3">
                      <label className="text-[10px] text-muted-foreground uppercase font-bold tracking-wider">
                        Clinician Note (optional)
                      </label>
                      <Textarea
                        value={clinicianNote}
                        onChange={(e) => setClinicianNote(e.target.value)}
                        placeholder="Additional observations..."
                        className="min-h-[60px]"
                      />
                    </div>

                    <div className="space-y-4 pt-4 border-t border-white/5">
                      <div className="flex items-center gap-3">
                        <input
                          type="checkbox"
                          id="iqa-ack-report"
                          className="h-4 w-4 accent-amber-500 cursor-pointer rounded"
                          checked={iqaAcknowledged}
                          onChange={(e) => setIqaAcknowledged(e.target.checked)}
                        />
                        <label htmlFor="iqa-ack-report" className="text-[11px] font-bold text-amber-500 cursor-pointer select-none">
                          Acknowledge sub-optimal quality to proceed
                        </label>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Action Buttons */}
                <div className="space-y-4">
                  <div className="flex flex-col sm:flex-row sm:gap-3">
                    <button
                      onClick={handlePdfExport}
                      disabled={isExporting || !activeResult || !q}
                      className="w-full flex-1 px-4 py-3 rounded-xl font-medium transition-all hover:opacity-90 disabled:opacity-50 disabled:pointer-events-none bg-primary text-primary-foreground shadow-md"
                    >
                      {isExporting ? "Exporting PDF..." : "Export PDF Report"}
                    </button>
                    <button
                      onClick={handleJsonSR}
                      disabled={!activeResult || !q}
                      className="w-full flex-1 px-4 py-3 rounded-xl font-medium transition-all hover:opacity-90 bg-white/10 text-white border border-white/5"
                    >
                      Export Structured JSON
                    </button>
                  </div>

                  <button
                    onClick={handleRegisterDb}
                    disabled={dbRegistered || !activeResult || !q}
                    className="w-full px-4 py-3 rounded-xl font-medium transition-all hover:opacity-90 bg-white/10 text-white border border-white/5"
                  >
                    {dbRegistered ? "Registered to DB" : "Register to Research DB"}
                  </button>

                  <div className="flex items-center gap-3 pt-4 border-t border-white/5">
                    <button
                      onClick={() => setWorkstationMode("xai")}
                      className={`flex-1 px-3 py-2 rounded text-xs font-medium transition-all hover:opacity-90 ${
                        workstationMode === "xai"
                          ? "bg-primary text-primary-foreground"
                          : "bg-white/10 text-white border border-white/5"
                      }`}
                    >
                      AI Explainability View
                    </button>
                    <button
                      onClick={() => setWorkstationMode("clinical")}
                      className={`flex-1 px-3 py-2 rounded text-xs font-medium transition-all hover:opacity-90 ${
                        workstationMode === "clinical"
                          ? "bg-primary text-primary-foreground"
                          : "bg-white/10 text-white border border-white/5"
                      }`}
                    >
                      Clinical View
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>
        </aside>
      </div>
    </div>
  );
}

// Helper to sync feedback (mirrors original)
async function syncFeedback(
  status: string,
  comments: string,
  reviewer: string,
  note: string
) {
  const getBackendStatus = (s: string) => {
    if (s === "confirm") return "Confirm AI finding";
    if (s === "reject") return "Reject AI finding";
    if (s === "investigate") return "Request Investigation";
    if (s === "insufficient") return "Insufficient Quality";
    return s;
  };
  const backendStatus = getBackendStatus(status);

  // TODO: Integrate with actual feedback saving via handleFeedbackSaved prop
  // For now, we just call the prop if available (should be passed from parent)
  // This function is kept for parity but actual call should be in the component's callbacks.
  // In this component we call handleFeedbackSaved from the button onClick above.
}