"use client";

import { useState, useEffect, useRef } from "react";
import dynamic from "next/dynamic";
import { getCookie } from "../hooks/usePrediction";
import LongitudinalTracker from "./LongitudinalTracker";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000";
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
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Textarea } from "@/components/ui/textarea";
import { Separator } from "@/components/ui/separator";

// Import custom types and Dynamic viewers
import { AnalysisResult } from "../hooks/useFileUpload";
import { Box, LungZone } from "./AnnotationCanvas";

// Import Modular Service Layer
import { predictionService } from "../services/predictionService";
import { heatmapService } from "../services/heatmapService";
import { observationService } from "../services/observationService";
import { imageQualityService } from "../services/imageQualityService";
import { reportService } from "../services/reportService";
import { auditService, AuditLogEntry } from "../services/auditService";
import { exportService } from "../services/exportService";

const DicomViewer = dynamic(() => import("./DicomViewer"), { ssr: false });

const getEvidenceCards = (prediction: string) => {
  const cond = prediction || "Normal";
  if (cond.toLowerCase().includes("normal") || cond === "Normal") {
    return [
      {
        title: "Symmetric Lung Expansion",
        description: "Normal lung volumes without localized hyperinflation or collapse.",
        confidence: 0.98,
        region: null,
        anatomicalZone: ""
      },
      {
        title: "Clear Pleural Spaces",
        description: "Sharp costophrenic angles with no evidence of effusion or thickening.",
        confidence: 0.96,
        region: null,
        anatomicalZone: ""
      }
    ];
  }

  return [
    {
      title: `Localized ${cond} Indicators`,
      description: `Patchy density gradients and patterns suggestive of ${cond} consolidation in the upper zones.`,
      confidence: 0.89,
      region: { x1: 120, y1: 20, x2: 190, y2: 80, zoom: 2.2, panX: -160, panY: 100 },
      anatomicalZone: "right apical"
    },
    {
      title: `${cond} Secondary Findings`,
      description: `Structural changes, thickening, or markings consistent with diagnostic markers for ${cond}.`,
      confidence: 0.74,
      region: { x1: 30, y1: 85, x2: 100, y2: 140, zoom: 2.0, panX: 150, panY: -30 },
      anatomicalZone: "left mid-zone"
    }
  ];
};

const mapZoneToLungZone = (zoneStr: string): LungZone => {
  const lower = zoneStr?.toLowerCase() || "";
  if (lower.includes("apical")) return "Apical";
  if (lower.includes("mid")) return "Mid-zone";
  if (lower.includes("basal")) return "Basal";
  return "Pleural";
};

const LungAnatomicalMap = ({ highlightZone }: { highlightZone?: string }) => {
  const isRightApical = highlightZone?.toLowerCase().includes("right apical") || highlightZone?.toLowerCase().includes("apical");
  const isLeftMid = highlightZone?.toLowerCase().includes("left mid") || highlightZone?.toLowerCase().includes("mid-zone");

  return (
    <div className="flex flex-col items-center justify-center p-3.5 bg-slate-950/70 rounded-xl border border-border/60 w-full">
      <p className="text-[9px] font-extrabold text-muted-foreground uppercase tracking-widest mb-2 text-center">Anatomical Focus Map</p>
      <svg width="120" height="110" viewBox="0 0 120 100" className="opacity-90">
        {/* Anatomical Right Lung (Left side of viewport) */}
        <g>
          {/* Apical Zone */}
          <path
            d="M 40 20 C 42 10, 50 12, 53 28 C 54 35, 48 40, 42 38 C 38 35, 38 25, 40 20 Z"
            fill={isRightApical ? "rgba(245, 158, 11, 0.45)" : "rgba(30, 41, 59, 0.45)"}
            stroke={isRightApical ? "#f59e0b" : "#475569"}
            strokeWidth={isRightApical ? "2" : "1"}
            className="transition-all duration-300"
          />
          {/* Mid Zone */}
          <path
            d="M 42 38 C 48 40, 56 42, 54 60 C 52 70, 42 70, 36 62 C 34 55, 36 42, 42 38 Z"
            fill="rgba(30, 41, 59, 0.45)"
            stroke="#475569"
            strokeWidth="1"
            className="transition-all duration-300"
          />
          {/* Basal Zone */}
          <path
            d="M 36 62 C 42 70, 52 70, 50 85 C 48 90, 34 90, 26 80 C 24 72, 28 65, 36 62 Z"
            fill="rgba(30, 41, 59, 0.45)"
            stroke="#475569"
            strokeWidth="1"
            className="transition-all duration-300"
          />
        </g>

        {/* Anatomical Left Lung (Right side of viewport) */}
        <g>
          {/* Apical Zone */}
          <path
            d="M 80 20 C 78 10, 70 12, 67 28 C 66 35, 72 40, 78 38 C 82 35, 82 25, 80 20 Z"
            fill="rgba(30, 41, 59, 0.45)"
            stroke="#475569"
            strokeWidth="1"
            className="transition-all duration-300"
          />
          {/* Mid Zone */}
          <path
            d="M 78 38 C 72 40, 64 42, 66 60 C 68 70, 78 70, 84 62 C 86 55, 84 42, 78 38 Z"
            fill={isLeftMid ? "rgba(245, 158, 11, 0.45)" : "rgba(30, 41, 59, 0.45)"}
            stroke={isLeftMid ? "#f59e0b" : "#475569"}
            strokeWidth={isLeftMid ? "2" : "1"}
            className="transition-all duration-300"
          />
          {/* Basal Zone */}
          <path
            d="M 84 62 C 78 70, 68 70, 70 85 C 72 90, 86 90, 94 80 C 96 72, 92 65, 84 62 Z"
            fill="rgba(30, 41, 59, 0.45)"
            stroke="#475569"
            strokeWidth="1"
            className="transition-all duration-300"
          />
        </g>

        <text x="45" y="98" fill="#94a3b8" fontSize="7" fontFamily="monospace" textAnchor="middle">R. Lung</text>
        <text x="75" y="98" fill="#94a3b8" fontSize="7" fontFamily="monospace" textAnchor="middle">L. Lung</text>
      </svg>
    </div>
  );
};

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
  downloadReport: () => Promise<void>; // Kept for interface compatibility, but we use exportService
  handleFeedbackSaved: (
    override: string | null,
    note: string,
    annotatedB64: string,
    comments?: string,
    reviewer?: string
  ) => void;
}

const getQualityMetrics = (result: any) => {
  if (result && result.image_quality) {
    const iqa = result.image_quality;
    return {
      exposure: iqa.exposure || "Adequate Exposure",
      coverage: iqa.coverage || "Full Lung Coverage",
      resolution: iqa.resolution || "Acceptable Resolution",
      rotation: iqa.rotation || "No Rotation",
      suitableForAi: typeof iqa.suitable_for_ai === "boolean" ? iqa.suitable_for_ai : (iqa.suitability === "suitable" || iqa.suitable_for_ai === true),
      qualityScore: iqa.quality_score || 95,
      warnings: iqa.warnings || []
    };
  }

  const fallback = imageQualityService.assessQuality(result?.filename || "unknown");
  return {
    ...fallback,
    warnings: [] as string[]
  };
};

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
  handleFeedbackSaved
}: ScreeningTabProps) {
  const activeResult = selectedIdx !== null ? results[selectedIdx] : null;
  const q = activeResult ? getQualityMetrics(activeResult) : null;

  // ── WORKSTATION VIEWING STATES ──
  const [viewMode, setViewMode] = useState<"original" | "heatmap" | "heatmap-only" | "side-by-side" | "split">("original");
  const [heatmapOpacity, setHeatmapOpacity] = useState(0.55);
  const [lungSegmentationActive, setLungSegmentationActive] = useState(false);
  const [workstationMode, setWorkstationMode] = useState<"clinical" | "research">("clinical");
  const [activeRightTab, setActiveRightTab] = useState<"findings" | "review" | "report">("findings");
  const [highlightedAnatomicalZone, setHighlightedAnatomicalZone] = useState<string>("");

  // Annotation layers
  const [boxes, setBoxes] = useState<Box[]>([]);
  const [activeZone, setActiveZone] = useState<LungZone>("Apical");
  const [annotateMode, setAnnotateMode] = useState(false);
  const annotationCanvasRef = useRef<HTMLCanvasElement>(null);

  // Focus region for observation zoom
  const [observationFocusRegion, setObservationFocusRegion] = useState<{
    x1: number; y1: number; x2: number; y2: number; zoom: number; panX: number; panY: number;
  } | null>(null);

  // Structured Review inputs
  const [clinicalReviewStatus, setClinicalReviewStatus] = useState<string>("confirm");
  const [reviewComments, setReviewComments] = useState("");
  const [reviewerName, setReviewerName] = useState("");
  const [clinicianNote, setClinicianNote] = useState("");

  const [xaiMethod, setXaiMethod] = useState<"gradcam_plusplus">("gradcam_plusplus");
  const [similarCases, setSimilarCases] = useState<{
    tb_similar: any[];
    normal_similar: any[];
  } | null>(null);
  const [modelMetadata, setModelMetadata] = useState<any>(null);
  const [loadingSimilar, setLoadingSimilar] = useState(false);

  useEffect(() => {
    const fetchModelMetadata = async () => {
      try {
        const response = await fetch(`${API_BASE}/model/metadata`, {
          credentials: "include"
        });
        if (response.ok) {
          const data = await response.json();
          setModelMetadata(data);
        }
      } catch (err) {
        console.error("Failed to fetch model metadata:", err);
      }
    };
    fetchModelMetadata();
  }, []);


  const syncFeedback = async (
    status: string,
    comments: string,
    reviewer: string,
    note: string
  ) => {
    const getBackendStatus = (s: string) => {
      if (s === "confirm") return "Confirm AI finding";
      if (s === "reject") return "Reject AI finding";
      if (s === "investigate") return "Request Investigation";
      if (s === "insufficient") return "Insufficient Quality";
      return s;
    };
    const backendStatus = getBackendStatus(status);

    handleFeedbackSaved(
      status,
      note,
      activeResult?.annotated_image || "",
      comments,
      reviewer
    );

    if (activeResult?.metadata?.patient_id) {
      try {
        const token = getCookie("csrf_token") || "";
        const res = await fetch(`${API_BASE}/feedback`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-CSRF-Token": token
          },
          credentials: "include",
          body: JSON.stringify({
            patient_id: activeResult.metadata.patient_id,
            clinician_prediction: backendStatus,
            reason: comments,
            annotation_b64: activeResult.annotated_image || "",
            clinician_note: note
          })
        });
        if (res.ok && activeResult.study_id) {
          fetchAuditLogs(activeResult.study_id);
        }
      } catch (err) {
        console.error("Failed to sync clinician feedback with backend:", err);
      }
    }
  };

  // Audit Logs
  const [auditLogs, setAuditLogs] = useState<AuditLogEntry[]>([]);
  const [dbRegistered, setDbRegistered] = useState(false);
  const [isExporting, setIsExporting] = useState(false);

  const fetchAuditLogs = async (studyId: string) => {
    try {
      const response = await fetch(`${API_BASE}/studies/${studyId}/audit`, {
        credentials: "include"
      });
      if (response.ok) {
        const data = await response.json();
        const mappedLogs = data.audit_trail.map((entry: any) => {
          const d = new Date(entry.timestamp);
          const timeStr = isNaN(d.getTime())
            ? entry.timestamp
            : `${d.getHours().toString().padStart(2, "0")}:${d.getMinutes().toString().padStart(2, "0")}`;

          let actionText = entry.action;
          if (entry.action === "Upload") {
            actionText = `Image study uploaded by ${entry.username}`;
          } else if (entry.action === "Inference") {
            actionText = "AI Inference pipeline completed";
          } else if (entry.action === "Review") {
            actionText = `Clinical review signed off by ${entry.username}`;
          } else if (entry.action === "Report Export") {
            actionText = `Clinical PDF report exported by ${entry.username}`;
          }
          return {
            timestamp: timeStr,
            action: actionText
          };
        });
        setAuditLogs(mappedLogs);
      }
    } catch (err) {
      console.error("Failed to fetch audit logs from backend:", err);
    }
  };

  const fetchStudyHeatmaps = async (studyId: string, idx: number) => {
    try {
      const response = await fetch(`${API_BASE}/studies/${studyId}/heatmaps`, {
        credentials: "include"
      });
      if (response.ok) {
        const data = await response.json();
        if (data.heatmaps) {
          // Update the results state to include heatmaps for this index
          setResults((prev: AnalysisResult[]) => {
            const next = [...prev];
            if (next[idx]) {
              next[idx] = {
                ...next[idx],
                heatmaps: data.heatmaps
              };
            }
            return next;
          });
        }
      }
    } catch (err) {
      console.error("Failed to fetch study heatmaps dynamically:", err);
    }
  };

  const fetchSimilarCases = async (studyId: string) => {
    setLoadingSimilar(true);
    try {
      const response = await fetch(`${API_BASE}/studies/${studyId}/similar`, {
        credentials: "include"
      });
      if (response.ok) {
        const data = await response.json();
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
      is_tb: cand.prediction.toLowerCase().includes("tb") || cand.prediction.toLowerCase().includes("tuberculosis"),
      metadata: {
        patient_id: cand.patient_id,
        patient_name: cand.patient_name,
        patient_age: cand.age,
        patient_sex: cand.sex,
        modality: "CR",
        study_date: new Date().toISOString().split("T")[0]
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
        resolution: "Acceptable Resolution",
        rotation: "No Rotation",
        quality_score: 95,
        suitable_for_ai: true,
        warnings: []
      }
    };
    
    setResults([mappedResult]);
    setSelectedIdx(0);
  };

  // Sync state on file selection change
  useEffect(() => {
    setViewMode("original");
    setHeatmapOpacity(0.55);
    setLungSegmentationActive(false);
    setBoxes([]);
    setAnnotateMode(false);
    setObservationFocusRegion(null);
    setClinicalReviewStatus(activeResult?.clinician_override || "confirm");
    setReviewComments(activeResult?.review_comments || "");
    setReviewerName(activeResult?.reviewer_name || "");
    setClinicianNote(activeResult?.clinician_note || "");
    setDbRegistered(false);

    if (activeResult) {
      if (activeResult.study_id) {
        fetchAuditLogs(activeResult.study_id);
        fetchSimilarCases(activeResult.study_id);
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

  useEffect(() => {
    if (activeResult && activeResult.status === "pending" && selectedIdx !== null) {
      analyzeFile(selectedIdx);
    }
  }, [selectedIdx, activeResult?.status, analyzeFile]);

  // Append a timeline action helper
  const addAuditLog = (actionText: string) => {
    const d = new Date();
    const ts = `${d.getHours().toString().padStart(2, "0")}:${d.getMinutes().toString().padStart(2, "0")}`;
    setAuditLogs(prev => [...prev, { timestamp: ts, action: actionText }]);
  };

  // Export handlers
  const handlePdfExport = async () => {
    if (!activeResult || !q) return;
    setIsExporting(true);
    addAuditLog("Generating clinical report PDF");

    // Package parameters
    const qualityMetrics = q;
    const diagnosisObj = predictionService.getDiagnosis(activeResult.prediction || "Normal", activeResult.confidence || 0.0);
    const observationsList = observationService.getObservations(activeResult.prediction || "Normal");

    const reviewData = {
      status: clinicalReviewStatus === "confirm" ? "Confirm AI finding" : clinicalReviewStatus === "reject" ? "Reject AI finding" : clinicalReviewStatus === "investigate" ? "Request Investigation" : "Insufficient Quality",
      comments: reviewComments,
      signature: reviewerName
    };

    const payload = reportService.buildReport(
      activeResult.metadata,
      qualityMetrics,
      diagnosisObj.condition,
      activeResult.confidence || 0.0,
      diagnosisObj.riskLevel,
      observationsList,
      reviewData,
      clinicianNote || globalNote
    );

    // Call service
    await exportService.downloadPDF(payload, activeResult.original_image || "", activeResult.heatmap_image || "");
    addAuditLog("Report PDF exported to downloads");

    try {
      const bodyPayload = activeResult.study_id
        ? { study_id: activeResult.study_id }
        : { patient_id: activeResult.metadata?.patient_id };

      await fetch(`${API_BASE}/report/audit`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify(bodyPayload),
        credentials: "include"
      });

      if (activeResult.study_id) {
        fetchAuditLogs(activeResult.study_id);
      }
    } catch (auditErr) {
      console.error("Failed to audit report generation on backend:", auditErr);
    }

    setIsExporting(false);
  };

  const handleJsonSR = () => {
    if (!activeResult || !q) return;
    addAuditLog("Exporting DICOM Structured Report (SR)");

    const qualityMetrics = q;
    const diagnosisObj = predictionService.getDiagnosis(activeResult.prediction || "Normal", activeResult.confidence || 0.0);
    const observationsList = observationService.getObservations(activeResult.prediction || "Normal");

    const reviewData = {
      status: clinicalReviewStatus,
      comments: reviewComments,
      signature: reviewerName
    };

    const payload = reportService.buildReport(
      activeResult.metadata,
      qualityMetrics,
      diagnosisObj.condition,
      activeResult.confidence || 0.0,
      diagnosisObj.riskLevel,
      observationsList,
      reviewData,
      clinicianNote || globalNote
    );

    exportService.exportDICOMSR(payload);
    addAuditLog("DICOM SR JSON package generated");
  };

  const handleRegisterDb = async () => {
    if (!activeResult || !q) return;
    setDbRegistered(true);
    addAuditLog("Registering case to research database");

    const qualityMetrics = q;
    const diagnosisObj = predictionService.getDiagnosis(activeResult.prediction || "Normal", activeResult.confidence || 0.0);
    const observationsList = observationService.getObservations(activeResult.prediction || "Normal");

    const reviewData = {
      status: clinicalReviewStatus,
      comments: reviewComments,
      signature: reviewerName
    };

    const payload = reportService.buildReport(
      activeResult.metadata,
      qualityMetrics,
      diagnosisObj.condition,
      activeResult.confidence || 0.0,
      diagnosisObj.riskLevel,
      observationsList,
      reviewData,
      clinicianNote || globalNote
    );

    await exportService.registerToResearchDB(payload);
    addAuditLog("Registered study to research cohort DB");
  };

  // Stepper timeline checklist status
  const getStepperStatus = () => {
    if (!activeResult) return [];
    const isLoading = activeResult.status === "loading" || activeResult.status === "pending";
    const stepList = [
      { text: "Image loaded in buffer", done: true, loading: false },
      { text: "AI classification complete", done: activeResult.status === "success", loading: isLoading },
      { text: "Grad-CAM++ heatmaps calculated", done: activeResult.status === "success", loading: isLoading },
      { text: "Clinical Adjudication Sign-off", done: reviewerName.trim().length > 2, loading: false },
      { text: "Clinical Report finalized", done: dbRegistered, loading: false }
    ];
    return stepList;
  };

  return (
    <div className="w-full">
      {files.length === 0 ? (
        /* ── NO FILES: Minimal empty state ── */
        <div className="w-full py-24 flex flex-col items-center justify-center text-center space-y-4 animate-fadein">
          <div className="w-12 h-12 rounded-2xl bg-muted flex items-center justify-center text-muted-foreground">
            <Eye className="w-6 h-6" strokeWidth={1.5} />
          </div>
          <div className="space-y-1">
            <h3 className="text-lg font-bold text-foreground">No study loaded</h3>
            <p className="text-sm text-muted-foreground max-w-sm">
              Go to <strong>Upload</strong> in the sidebar to add a chest X-ray and return here to analyse it.
            </p>
          </div>
        </div>
      ) : (
        /* ── ENTERPRISE THREE-PANEL RAD WORKSPACE ── */
        <div className="w-full">
          {activeResult ? (
              /* ── 3-PANEL PACS WORKSPACE (SUCCESSFUL INFERENCE STATE) ── */
              <div className="flex flex-col space-y-6 w-full">
                {/* Workstation Header & Switcher */}
                <div className="flex flex-wrap items-center justify-between gap-4 p-4 border border-border bg-card rounded-xl shadow-none">
                  <div className="flex items-center gap-3">
                    <Activity className="w-5 h-5 text-primary animate-pulse" strokeWidth={2} />
                    <div>
                      <h3 className="text-sm font-bold text-foreground leading-none">Diagnostic Workbench</h3>
                      <p className="text-[10px] text-muted-foreground mt-1">Study ID: {activeResult.study_id || "ST-TEMP"}</p>
                    </div>
                  </div>
                  
                  {/* Premium double-pill mode switcher */}
                  <div className="flex bg-muted/70 p-1 rounded-full border border-border/60">
                    <button
                      onClick={() => setWorkstationMode("clinical")}
                      className={`px-4 py-1.5 rounded-full text-xs font-semibold transition-all duration-200 cursor-pointer ${workstationMode === "clinical" ? "bg-white text-black shadow-sm font-bold border border-border/10" : "text-muted-foreground hover:text-foreground"}`}
                    >
                      Clinical View
                    </button>
                    <button
                      onClick={() => setWorkstationMode("research")}
                      className={`px-4 py-1.5 rounded-full text-xs font-semibold transition-all duration-200 cursor-pointer ${workstationMode === "research" ? "bg-white text-black shadow-sm font-bold border border-border/10" : "text-muted-foreground hover:text-foreground"}`}
                    >
                      Research View
                    </button>
                  </div>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
                  {/* 1. LEFT PANEL (70%): Unified Primary Viewport */}
                  <div className="lg:col-span-8 space-y-6">

                    {/* Primary Viewport Card container */}
                    <Card className="border border-border bg-[#0a0a0a] rounded-xl overflow-hidden shadow-md">
                      <DicomViewer
                        imageBase64={activeResult.original_image || ""}
                        heatmapBase64={activeResult.heatmaps?.[xaiMethod] || activeResult.heatmap_image}
                        label="Nirikshon Enterprise Viewport"
                        pixelSpacing={activeResult.metadata?.pixel_spacing}
                        viewMode={viewMode}
                        heatmapOpacity={heatmapOpacity}
                        lungSegmentationActive={lungSegmentationActive}
                        boxes={boxes}
                        setBoxes={setBoxes}
                        activeZone={activeZone}
                        annotateMode={annotateMode}
                        annotationCanvasRef={annotationCanvasRef}
                        observationFocusRegion={observationFocusRegion}
                        setViewMode={setViewMode}
                        setAnnotateMode={setAnnotateMode}
                        setHeatmapOpacity={setHeatmapOpacity}
                        setLungSegmentationActive={setLungSegmentationActive}
                        setActiveZone={setActiveZone}
                      />
                    </Card>

                  {/* Redraw list of active markups drawn on top of the X-ray */}
                  {boxes.length > 0 && (
                    <div className="flex flex-wrap items-center gap-2 p-3 border border-border bg-card rounded-xl">
                      <span className="text-xs font-bold text-foreground mr-1">Marked Zones:</span>
                      <div className="flex flex-wrap gap-1.5 items-center flex-1">
                        {boxes.map((b, idx) => (
                          <span
                            key={idx}
                            className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold border border-border bg-muted/60 text-foreground"
                          >
                            <span>{b.zone} #{idx + 1}</span>
                            <button
                              onClick={() => setBoxes(prev => prev.filter((_, i) => i !== idx))}
                              className="text-muted-foreground hover:text-destructive font-bold cursor-pointer ml-1"
                            >
                              ✕
                            </button>
                          </span>
                        ))}
                      </div>
                      <Button
                        size="sm"
                        variant="ghost"
                        className="text-xs h-7 px-3 text-destructive hover:bg-destructive/10 rounded-full cursor-pointer ml-auto"
                        onClick={() => {
                          setBoxes([]);
                          addAuditLog("Cleared all drawing markup coordinates");
                        }}
                      >
                        Clear All
                      </Button>
                    </div>
                  )}
                </div>

                {/* 2. RIGHT PANEL (30%): Steppers & RIS Records drawer */}
                <div className="lg:col-span-4 space-y-6">
                  {workstationMode === "clinical" ? (
                    <>
                      {/* WORKSPACE SELECTOR TABS */}
                      <div className="flex bg-muted/50 p-1 rounded-full border border-border/40 w-full">
                        {(["findings", "review", "report"] as const).map(tab => (
                          <button
                            key={tab}
                            onClick={() => {
                              setActiveRightTab(tab);
                              addAuditLog(`Swapped right panel to ${tab} tab`);
                            }}
                            className={`flex-1 py-1.5 rounded-full text-[10px] sm:text-xs font-semibold capitalize transition-all duration-200 cursor-pointer text-center ${activeRightTab === tab
                                ? "bg-background text-foreground shadow-sm border border-border/10 font-bold"
                                : "text-muted-foreground hover:text-foreground"
                              }`}
                          >
                            {tab === "findings" && "Findings"}
                            {tab === "review" && "Review"}
                            {tab === "report" && "Report"}
                          </button>
                        ))}
                      </div>

                      {/* ───────────────── TAB 1: FINDINGS ───────────────── */}
                      {activeRightTab === "findings" && (
                        <div className="space-y-6 animate-fadein">


                          {/* WORKFLOW STATUS STEPPER */}
                          <div className="p-4 border border-border bg-card rounded-xl space-y-3">
                            <p className="text-xs font-bold uppercase tracking-wider text-foreground">Workflow Checklist</p>
                            <div className="space-y-2">
                              {getStepperStatus().map((step, idx) => (
                                <div key={idx} className="flex items-center gap-2 text-xs font-medium">
                                  <span className={step.done ? "text-emerald-500 font-bold" : step.loading ? "text-primary" : "text-muted-foreground animate-pulse"}>
                                    {step.done ? (
                                      "✓"
                                    ) : step.loading ? (
                                      <Loader2 className="w-3.5 h-3.5 animate-spin text-primary" />
                                    ) : (
                                      "⏳"
                                    )}
                                  </span>
                                  <span className={step.done ? "text-foreground font-semibold" : step.loading ? "text-primary font-semibold" : "text-muted-foreground"}>
                                    {step.text} {step.loading && <span className="text-[10px] text-muted-foreground font-normal">(AI calculating...)</span>}
                                  </span>
                                </div>
                              ))}
                            </div>
                          </div>

                          {/* IMAGE QUALITY ASSESSMENT (IQA) */}
                          <Card className="border border-border bg-card rounded-xl shadow-none">
                            <CardContent className="p-5 space-y-3">
                              <div className="flex items-center justify-between">
                                <p className="text-xs font-bold uppercase tracking-wider text-foreground">Image Quality (IQA)</p>
                                <div className="flex gap-2">
                                  {q && (
                                    <>
                                      <Badge variant={q.suitableForAi ? "default" : "destructive"} className="rounded-full font-bold uppercase text-[9px] px-2 py-0.5">
                                        {q.suitableForAi ? "Suitable" : "Unsuitable"}
                                      </Badge>
                                      <Badge className="badge-normal rounded-full font-bold text-[9px] px-2 py-0.5">
                                        Score: {q.qualityScore}%
                                      </Badge>
                                    </>
                                  )}
                                </div>
                              </div>
                              <Separator />
                              {q && (
                                <>
                                  <div className="grid grid-cols-2 gap-2 text-xs font-medium">
                                    {[
                                      { label: "Exposure", val: q.exposure, pass: q.exposure === "Adequate Exposure" },
                                      { label: "Rotation", val: q.rotation, pass: q.rotation === "No Rotation" },
                                      { label: "Coverage", val: q.coverage, pass: q.coverage === "Full Lung Coverage" },
                                      { label: "Resolution", val: q.resolution, pass: !q.resolution.toLowerCase().includes("low") }
                                    ].map((iqa, idx) => (
                                      <div key={idx} className="flex flex-col">
                                        <span className="text-muted-foreground text-[10px] uppercase font-bold">{iqa.label}</span>
                                        <span className={`font-semibold text-xs ${iqa.pass ? "text-emerald-500" : "text-amber-500"}`}>
                                          {iqa.val}
                                        </span>
                                      </div>
                                    ))}
                                  </div>
                                  {q.warnings && q.warnings.length > 0 && (
                                    <div className="mt-3 p-3 bg-amber-50 dark:bg-amber-950/20 border border-amber-200 dark:border-amber-900/30 rounded-lg space-y-1">
                                      <p className="text-[10px] font-bold text-amber-600 dark:text-amber-400 uppercase tracking-wider">Quality Warnings</p>
                                      {q.warnings.map((warn: string, wIdx: number) => (
                                        <p key={wIdx} className="text-[11px] text-amber-700 dark:text-amber-300 font-medium">
                                          • {warn}
                                        </p>
                                      ))}
                                    </div>
                                  )}
                                </>
                              )}
                            </CardContent>
                          </Card>

                          {/* AI DIAGNOSTICS CARD */}
                          <Card className="border border-border bg-card rounded-xl shadow-none">
                            <CardContent className="p-5 space-y-4">
                              <div className="flex justify-between items-center">
                                <p className="text-xs font-bold uppercase tracking-wider text-foreground">AI Diagnostics Output</p>
                                {activeResult.status === "loading" || activeResult.status === "pending" ? (
                                  <Badge variant="outline" className="badge-normal rounded-full font-bold uppercase animate-pulse bg-muted/30 border-muted/50 text-muted-foreground">
                                    Risk Calculating
                                  </Badge>
                                ) : activeResult.status === "error" ? (
                                  <Badge variant="destructive" className="rounded-full font-bold uppercase">
                                    Analysis Error
                                  </Badge>
                                ) : (
                                  <Badge variant="outline" className={`${predictionService.getDiagnosis(activeResult.prediction || "Normal", activeResult.confidence || 0).riskLevel === "High" ? "badge-tb" : "badge-normal"} rounded-full font-bold uppercase`}>
                                    {predictionService.getDiagnosis(activeResult.prediction || "Normal", activeResult.confidence || 0).riskLevel} RISK
                                  </Badge>
                                )}
                              </div>
                              <Separator />
                              <div className="space-y-3">
                                <div className="flex justify-between items-center">
                                  <span className="text-xs text-muted-foreground">Primary Condition:</span>
                                  {activeResult.status === "loading" || activeResult.status === "pending" ? (
                                    <h3 className="text-sm font-bold text-muted-foreground animate-pulse">
                                      Calculating...
                                    </h3>
                                  ) : activeResult.status === "error" ? (
                                    <h3 className="text-sm font-bold text-destructive">
                                      {activeResult.errorMsg || "Analysis failed"}
                                    </h3>
                                  ) : (
                                    <h3 className={`text-sm font-bold ${activeResult.is_tb ? "text-amber-600 dark:text-amber-500" : "text-emerald-600 dark:text-emerald-500"}`}>
                                      {predictionService.getDiagnosis(activeResult.prediction || "Normal", activeResult.confidence || 0).condition}
                                    </h3>
                                  )}
                                </div>
                                <div className="flex justify-between items-center">
                                  <span className="text-xs text-muted-foreground">Model Confidence:</span>
                                  {activeResult.status === "loading" || activeResult.status === "pending" ? (
                                    <span className="text-xs font-bold font-mono text-muted-foreground animate-pulse">Calculating...</span>
                                  ) : activeResult.status === "error" ? (
                                    <span className="text-xs font-bold font-mono text-destructive">0.0%</span>
                                  ) : (
                                    <span className="text-xs font-bold font-mono text-foreground">{((activeResult.confidence || 0) * 100).toFixed(1)}%</span>
                                  )}
                                </div>
                                {activeResult.status === "loading" || activeResult.status === "pending" ? (
                                  <Progress value={null} className="h-1.5 bg-muted/40 animate-pulse" />
                                ) : activeResult.status === "error" ? (
                                  <Progress value={0} className="h-1.5 bg-destructive/20" />
                                ) : (
                                  <Progress value={(activeResult.confidence || 0) * 100} className={`h-1.5 ${activeResult.is_tb ? "bg-amber-100 dark:bg-amber-950" : "bg-emerald-100 dark:bg-emerald-950"}`} />
                                )}
                              </div>
                            </CardContent>
                          </Card>

                          {/* ANATOMICAL RAD MAP */}
                          <LungAnatomicalMap highlightZone={activeResult.attention_region} />
                        </div>
                      )}

                      {/* ───────────────── TAB 2: CLINICIAN OVERRIDE REVIEW ───────────────── */}
                      {activeRightTab === "review" && (
                        <div className="space-y-6 animate-fadein">
                          {/* CLINICAL AUDIT / OVERRIDE INPUTS */}
                          <Card className="border border-border bg-card rounded-xl shadow-none">
                            <CardContent className="p-5 space-y-4">
                              <p className="text-xs font-bold uppercase tracking-wider text-foreground">Clinical Sign-Off</p>
                              <Separator />

                              <div className="space-y-3">
                                <label className="text-[10px] text-muted-foreground uppercase font-bold">Diagnostic Verdict Adjudication</label>
                                <div className="grid grid-cols-2 gap-2">
                                  {[
                                    { id: "confirm", label: "Confirm AI Verdict" },
                                    { id: "reject", label: "Reject AI Verdict" },
                                    { id: "investigate", label: "Request Investigation" },
                                    { id: "insufficient", label: "Insufficient Quality" }
                                  ].map(opt => (
                                    <button
                                      key={opt.id}
                                      onClick={() => {
                                        setClinicalReviewStatus(opt.id);
                                        syncFeedback(opt.id, reviewComments, reviewerName, clinicianNote);
                                      }}
                                      className={`p-2.5 rounded-lg text-xs font-semibold border text-center transition-all cursor-pointer ${
                                        clinicalReviewStatus === opt.id
                                          ? "border-primary bg-primary/5 text-primary"
                                          : "border-border bg-muted/20 hover:bg-muted/40 text-muted-foreground"
                                      }`}
                                    >
                                      {opt.label}
                                    </button>
                                  ))}
                                </div>
                              </div>

                              <div className="space-y-2">
                                <label className="text-[10px] text-muted-foreground uppercase font-bold">Comments &amp; Reasoning</label>
                                <Textarea
                                  value={reviewComments}
                                  onChange={e => {
                                    setReviewComments(e.target.value);
                                    syncFeedback(clinicalReviewStatus, e.target.value, reviewerName, clinicianNote);
                                  }}
                                  placeholder="Document clinical context for audit logs..."
                                  className="text-xs bg-muted/25 border-border/80 resize-none h-20 rounded-xl"
                                />
                              </div>

                              <div className="space-y-2">
                                <label className="text-[10px] text-muted-foreground uppercase font-bold">Reviewer Signature</label>
                                <input
                                  type="text"
                                  value={reviewerName}
                                  onChange={e => {
                                    setReviewerName(e.target.value);
                                    syncFeedback(clinicalReviewStatus, reviewComments, e.target.value, clinicianNote);
                                  }}
                                  placeholder="Dr. Radiologist name..."
                                  className="w-full text-xs bg-muted/25 border border-border/80 rounded-xl h-10 px-3 outline-none focus:ring-1 focus:ring-primary"
                                />
                              </div>
                            </CardContent>
                          </Card>

                          {/* LONGITUDINAL PATIENT TRACKER */}
                          <LongitudinalTracker patientId={activeResult.metadata?.patient_id || ""} patientName={activeResult.metadata?.patient_name || ""} currentResult={activeResult as any} />
                        </div>
                      )}


                      {/* ───────────────── TAB 4: STRUCTURED CLINICAL REPORT ───────────────── */}
                      {activeRightTab === "report" && (
                        <div className="space-y-6 animate-fadein">
                          {/* REPORT SUMMARY CARD */}
                          <Card className="border border-border bg-card rounded-xl shadow-none">
                            <CardContent className="p-5 space-y-4">
                              <div className="flex justify-between items-center border-b border-border pb-3">
                                <div>
                                  <h4 className="text-xs font-bold uppercase tracking-wider text-foreground">Structured Report Preview</h4>
                                  <p className="text-[10px] text-muted-foreground mt-0.5">Automated Clinical Impression Summary</p>
                                </div>
                                <Badge className="badge-normal rounded-full font-bold">DRAFT</Badge>
                              </div>

                              <div className="space-y-3 font-medium text-foreground">
                                <div className="grid grid-cols-2 gap-4">
                                  <div>
                                    <span className="text-muted-foreground text-[10px] uppercase font-bold">Patient Identifier:</span>
                                    <p className="text-xs font-semibold">{activeResult.metadata?.patient_id || "N/A"}</p>
                                  </div>
                                  <div>
                                    <span className="text-muted-foreground text-[10px] uppercase font-bold">Date:</span>
                                    <p className="text-xs font-semibold">{activeResult.metadata?.study_date || "N/A"}</p>
                                  </div>
                                </div>

                                <div>
                                  <span className="text-muted-foreground text-[10px] uppercase font-bold">Diagnostic Inference:</span>
                                  <p className="text-xs font-semibold">
                                    {activeResult.status === "loading" || activeResult.status === "pending" ? (
                                      <span className="text-muted-foreground animate-pulse">Calculating...</span>
                                    ) : activeResult.status === "error" ? (
                                      <span className="text-destructive">Analysis failed</span>
                                    ) : (
                                      <>
                                        {predictionService.getDiagnosis(activeResult.prediction || "Normal", activeResult.confidence || 0).condition}
                                        <span className="text-muted-foreground text-[11px] font-normal ml-1">({((activeResult.confidence || 0) * 100).toFixed(1)}% confidence)</span>
                                      </>
                                    )}
                                  </p>
                                </div>

                                <div>
                                  <span className="text-muted-foreground text-[10px] uppercase font-bold">Adjudication Choice:</span>
                                  <p className="text-xs font-semibold capitalize">{clinicalReviewStatus.replace("_", " ")}</p>
                                </div>

                                <div>
                                  <span className="text-muted-foreground text-[10px] uppercase font-bold">Auditor Comments:</span>
                                  <p className="text-xs font-semibold text-muted-foreground mt-0.5">{reviewComments || "No comment added."}</p>
                                </div>

                                <div>
                                  <span className="text-muted-foreground text-[10px] uppercase font-bold">Radiologist Notes:</span>
                                  <p className="text-xs font-semibold text-muted-foreground mt-0.5">{clinicianNote || "No study notes entered."}</p>
                                </div>

                                <div className="pt-2 border-t border-border flex justify-between items-center text-[10px] text-muted-foreground">
                                  <span>Compiled by: Nirikshon V2.4</span>
                                  <span>Signed: <strong className="text-foreground">{reviewerName || "Awaiting Signature"}</strong></span>
                                </div>
                              </div>
                            </CardContent>
                          </Card>

                          {/* REPORT EXPORTS & METADATA */}
                          <div className="p-4 border border-border bg-card rounded-xl space-y-3">
                            <p className="text-xs font-bold uppercase tracking-wider text-foreground">Downstream Integrations &amp; Export</p>

                            {/* Exporters buttons row */}
                            <div className="flex gap-2">
                              <Button
                                size="sm"
                                onClick={handlePdfExport}
                                disabled={isExporting || activeResult.status !== "success"}
                                className="flex-1 text-xs font-semibold h-9 gap-1.5 cursor-pointer"
                              >
                                <FileText className="w-3.5 h-3.5" /> Export PDF
                              </Button>
                              <Button
                                size="sm"
                                variant="outline"
                                onClick={handleJsonSR}
                                disabled={activeResult.status !== "success"}
                                className="flex-1 text-xs font-semibold h-9 gap-1.5 cursor-pointer"
                              >
                                <Download className="w-3.5 h-3.5" /> DICOM SR
                              </Button>
                              <Button
                                size="sm"
                                variant={dbRegistered ? "secondary" : "outline"}
                                onClick={handleRegisterDb}
                                disabled={dbRegistered || activeResult.status !== "success"}
                                className={`flex-1 text-xs font-semibold h-9 gap-1.5 cursor-pointer ${dbRegistered ? "text-emerald-500 border-emerald-500/20 bg-emerald-500/5" : ""
                                  }`}
                              >
                                <Database className="w-3.5 h-3.5" /> {dbRegistered ? "Registered" : "Research DB"}
                              </Button>
                            </div>
                          </div>
                        </div>
                      )}

                    </>
                  ) : (
                    <div className="space-y-6 animate-fadein">
                      {/* A. MODEL METRICS & DATASET TRACKER CARD */}
                      <Card className="border border-border bg-card rounded-xl shadow-none">
                        <CardContent className="p-5 space-y-4">
                          <div className="flex items-center gap-2">
                            <Activity className="w-4 h-4 text-primary" />
                            <p className="text-xs font-bold uppercase tracking-wider text-foreground">Model Calibration &amp; Metrics</p>
                          </div>
                          <Separator />
                          {modelMetadata ? (
                            <div className="space-y-4">
                              {/* Validation Metrics Grid */}
                              <div className="grid grid-cols-2 gap-3 text-xs">
                                {[
                                  { label: "Accuracy", value: `${(modelMetadata.metrics.accuracy * 100).toFixed(1)}%` },
                                  { label: "AUC-ROC", value: modelMetadata.metrics.auc.toFixed(4) },
                                  { label: "Sensitivity", value: `${(modelMetadata.metrics.sensitivity * 100).toFixed(1)}%` },
                                  { label: "Specificity", value: `${(modelMetadata.metrics.specificity * 100).toFixed(1)}%` },
                                  { label: "F1 Score", value: modelMetadata.metrics.f1.toFixed(3) },
                                  { label: "Calibration", value: modelMetadata.metrics.calibration_score.toFixed(3) }
                                ].map((metric, idx) => (
                                  <div key={idx} className="p-2 border border-border bg-muted/30 rounded-lg">
                                    <p className="text-[10px] text-muted-foreground uppercase font-semibold">{metric.label}</p>
                                    <p className="text-xs font-bold text-foreground mt-0.5">{metric.value}</p>
                                  </div>
                                ))}
                              </div>
                              
                              {/* Dataset Tracking Info */}
                              <div className="text-[11px] p-3 border border-border/80 bg-muted/10 rounded-lg space-y-1.5 font-medium">
                                <p className="text-[10px] text-muted-foreground uppercase font-bold">Model Version &amp; Training</p>
                                <div className="flex justify-between">
                                  <span className="text-muted-foreground">Model:</span>
                                  <span className="text-foreground font-semibold">{modelMetadata.dataset_tracking.model_version}</span>
                                </div>
                                <div className="flex justify-between">
                                  <span className="text-muted-foreground">Train Set:</span>
                                  <span className="text-foreground font-semibold">{modelMetadata.dataset_tracking.training_dataset_version}</span>
                                </div>
                                <div className="flex justify-between">
                                  <span className="text-muted-foreground">Val Set:</span>
                                  <span className="text-foreground font-semibold">{modelMetadata.dataset_tracking.validation_dataset_version}</span>
                                </div>
                                <div className="flex justify-between">
                                  <span className="text-muted-foreground">Train Date:</span>
                                  <span className="text-foreground font-semibold">{modelMetadata.dataset_tracking.training_date}</span>
                                </div>
                              </div>
                            </div>
                          ) : (
                            <div className="py-4 text-center text-xs text-muted-foreground animate-pulse">
                              Loading model validation metrics...
                            </div>
                          )}
                        </CardContent>
                      </Card>

                      {/* B. EVIDENCE EXPLORER GRID */}
                      <Card className="border border-border bg-card rounded-xl shadow-none">
                        <CardContent className="p-5 space-y-4">
                          <div className="flex items-center gap-2">
                            <Eye className="w-4 h-4 text-primary" />
                            <p className="text-xs font-bold uppercase tracking-wider text-foreground">Evidence Explorer</p>
                          </div>
                          <Separator />
                          <div className="space-y-3">
                            {getEvidenceCards(activeResult.prediction || "Normal").map((obs, idx) => {
                              const hasRegion = !!obs.region;
                              const regionStr = hasRegion ? `[${obs.region.x1}, ${obs.region.y1}, ${obs.region.x2}, ${obs.region.y2}]` : "Whole lung field";
                              
                              return (
                                <div
                                  key={idx}
                                  onClick={() => {
                                    if (hasRegion) {
                                      setObservationFocusRegion(obs.region);
                                      setHighlightedAnatomicalZone(obs.anatomicalZone);
                                      setBoxes([
                                        {
                                          x: obs.region.x1,
                                          y: obs.region.y1,
                                          w: obs.region.x2 - obs.region.x1,
                                          h: obs.region.y2 - obs.region.y1,
                                          zone: mapZoneToLungZone(obs.anatomicalZone)
                                        }
                                      ]);
                                      addAuditLog(`Focused Evidence Explorer on region: ${obs.anatomicalZone}`);
                                    } else {
                                      setObservationFocusRegion(null);
                                      setBoxes([]);
                                      setHighlightedAnatomicalZone("");
                                    }
                                  }}
                                  className={`p-3 border rounded-xl transition-all cursor-pointer text-xs space-y-2 font-medium ${
                                    highlightedAnatomicalZone === obs.anatomicalZone
                                      ? "border-primary bg-primary/5 shadow-sm"
                                      : "border-border bg-muted/20 hover:bg-muted/40"
                                  }`}
                                >
                                  <div className="flex justify-between items-start">
                                    <span className="font-bold text-foreground text-xs">{obs.title}</span>
                                    <Badge variant="secondary" className="rounded-full text-[9px] font-bold px-2 py-0.5">
                                      Score: {obs.confidence.toFixed(2)}
                                    </Badge>
                                  </div>
                                  
                                  <p className="text-[11px] text-muted-foreground leading-relaxed">{obs.description}</p>
                                  
                                  <div className="flex flex-wrap gap-2 text-[10px] text-muted-foreground font-mono">
                                    <span>📍 Location: <strong className="text-foreground font-sans capitalize">{obs.anatomicalZone || "Global"}</strong></span>
                                    <span>🔲 Bounds: <strong className="text-foreground">{regionStr}</strong></span>
                                  </div>
                                </div>
                              );
                            })}
                          </div>
                        </CardContent>
                      </Card>

                      {/* C. SIMILAR CASES CAROUSEL */}
                      <Card className="border border-border bg-card rounded-xl shadow-none">
                        <CardContent className="p-5 space-y-4">
                          <div className="flex items-center gap-2">
                            <Columns className="w-4 h-4 text-primary" />
                            <p className="text-xs font-bold uppercase tracking-wider text-foreground">Similar Cohort Cases</p>
                          </div>
                          <Separator />
                          {loadingSimilar ? (
                            <div className="py-4 text-center text-xs text-muted-foreground animate-pulse">
                              Fetching similar cohort scans...
                            </div>
                          ) : similarCases && (similarCases.tb_similar.length > 0 || similarCases.normal_similar.length > 0) ? (
                            <div className="space-y-4">
                              {similarCases.tb_similar.length > 0 && (
                                <div className="space-y-2">
                                  <p className="text-[10px] text-amber-500 font-bold uppercase tracking-wider">Similar Tuberculosis Cases</p>
                                  <div className="grid grid-cols-3 gap-2">
                                    {similarCases.tb_similar.map((cand, idx) => (
                                      <div
                                        key={idx}
                                        onClick={() => handleSelectSimilarStudy(cand)}
                                        className="group relative cursor-pointer border border-border bg-muted/20 hover:bg-muted/40 rounded-lg overflow-hidden transition-all text-center p-1.5 flex flex-col items-center"
                                      >
                                        <img
                                          src={cand.original_image || undefined}
                                          alt="TB Case"
                                          className="w-16 h-16 object-cover rounded-md border border-border mb-1 group-hover:scale-105 transition-transform"
                                        />
                                        <span className="text-[9px] font-bold text-foreground truncate w-full">{cand.patient_name}</span>
                                        <span className="text-[8px] text-muted-foreground font-mono">{cand.similarity_score}% Match</span>
                                      </div>
                                    ))}
                                  </div>
                                </div>
                              )}
                              {similarCases.normal_similar.length > 0 && (
                                <div className="space-y-2">
                                  <p className="text-[10px] text-emerald-500 font-bold uppercase tracking-wider">Similar Normal Cases</p>
                                  <div className="grid grid-cols-3 gap-2">
                                    {similarCases.normal_similar.map((cand, idx) => (
                                      <div
                                        key={idx}
                                        onClick={() => handleSelectSimilarStudy(cand)}
                                        className="group relative cursor-pointer border border-border bg-muted/20 hover:bg-muted/40 rounded-lg overflow-hidden transition-all text-center p-1.5 flex flex-col items-center"
                                      >
                                        <img
                                          src={cand.original_image || undefined}
                                          alt="Normal Case"
                                          className="w-16 h-16 object-cover rounded-md border border-border mb-1 group-hover:scale-105 transition-transform"
                                        />
                                        <span className="text-[9px] font-bold text-foreground truncate w-full">{cand.patient_name}</span>
                                        <span className="text-[8px] text-muted-foreground font-mono">{cand.similarity_score}% Match</span>
                                      </div>
                                    ))}
                                  </div>
                                </div>
                              )}
                            </div>
                          ) : (
                            <div className="py-2 text-center text-xs text-muted-foreground">
                              No similar cohort matches found in database.
                            </div>
                          )}
                        </CardContent>
                      </Card>
                    </div>
                  )}

                </div>
              </div>
            </div>
          ) : (
            <div className="w-full max-w-xl mx-auto py-16 flex flex-col items-center justify-center text-center space-y-6 animate-fadein">
              <div className="w-12 h-12 rounded-full bg-muted flex items-center justify-center text-muted-foreground">
                <Activity className="w-6 h-6" />
              </div>
              <h3 className="text-lg font-bold text-foreground">Select a radiograph Study</h3>
              <p className="text-xs text-muted-foreground max-w-md leading-relaxed">
                Click a case from the top horizontal queue or case switcher registry to load patient images.
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
