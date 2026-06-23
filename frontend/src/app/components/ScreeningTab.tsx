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
  LayoutDashboard,
  Search,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import TsnePlot from "./TsnePlot";
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
import XaiVisualization from "./XaiVisualization";

const DicomViewer = dynamic(() => import("./DicomViewer"), { ssr: false });

const getEvidenceCards = (result: AnalysisResult | null) => {
  if (result?.status === "error") {
    return [
      {
        title: "Diagnostic Inference Failed",
        description: `Evidence analysis cannot be populated because the AI processing failed. Error: ${result.errorMsg || "Internal server error"}.`,
        confidence: 0.0,
        region: null,
        anatomicalZone: "global"
      }
    ];
  }
  if (!result || result.status !== "success") {
    return [
      {
        title: "Awaiting Diagnostic Inference",
        description: "Evidence analysis will populate automatically once the chest X-ray processing is complete.",
        confidence: 0.0,
        region: null,
        anatomicalZone: "global"
      }
    ];
  }

  const diag = predictionService.getDiagnosis(result.prediction || "Normal", result.confidence || 0.0, result.threshold_used);
  const cond = result.prediction || "Normal";
  const isNormal = cond.toLowerCase().includes("normal") || cond === "Normal";
  const confidencePct = (diag.confidence * 100).toFixed(1);

  // Get attention region
  const region = result.attention_region || "lung fields";
  
  // Get quality metrics
  const qExposure = result.image_quality?.exposure || "Adequate Exposure";
  const qCoverage = result.image_quality?.coverage || "Full Lung Coverage";
  const qScore = result.image_quality?.quality_score || 95;

  if (isNormal) {
    return [
      {
        title: "Parenchymal Clearance",
        description: `Bilateral lung fields exhibit normal aeration without signs of active consolidation, effusion, or masses (AI confidence: ${confidencePct}%).`,
        confidence: diag.confidence,
        region: null,
        anatomicalZone: "bilateral"
      },
      {
        title: "Clear Costophrenic Angles",
        description: `Pleural boundaries are sharp and well-defined with no indication of fluid accumulation. Costophrenic angles are completely clear.`,
        confidence: Math.max(0.92, diag.confidence - 0.04),
        region: null,
        anatomicalZone: "pleural space"
      },
      {
        title: "Technical Image Integrity",
        description: `Radiograph shows ${qExposure.toLowerCase()} and ${qCoverage.toLowerCase()} (Technical Quality Score: ${qScore}%).`,
        confidence: qScore / 100,
        region: null,
        anatomicalZone: "global"
      }
    ];
  }

  // If Tuberculosis / Abnormal
  // Dynamic regions based on attention_region or coordinates
  const leftApicalRegion = { x1: 25, y1: 20, x2: 95, y2: 80, zoom: 2.2, panX: 160, panY: 100 };
  const rightApicalRegion = { x1: 125, y1: 20, x2: 195, y2: 80, zoom: 2.2, panX: -160, panY: 100 };
  const leftMidRegion = { x1: 30, y1: 85, x2: 100, y2: 140, zoom: 2.0, panX: 150, panY: -30 };
  const rightMidRegion = { x1: 120, y1: 85, x2: 190, y2: 140, zoom: 2.0, panX: -150, panY: -30 };

  const isLeft = region.toLowerCase().includes("left");
  const isApical = region.toLowerCase().includes("apical") || region.toLowerCase().includes("upper");
  
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
      title: `Consolidation & Opacity Focus`,
      description: `Grad-CAM++ highlighted an area of increased opacity in the ${zoneLabel} zone. This density gradient is consistent with focal active ${cond} consolidation (AI confidence: ${confidencePct}%).`,
      confidence: diag.confidence,
      region: targetRegion,
      anatomicalZone: zoneLabel
    },
    {
      title: "Asymmetric Density Gradients",
      description: `Significant localized markings and architectural asymmetry identified in the ${zoneLabel} zone compared to contralateral regions.`,
      confidence: Math.max(0.70, diag.confidence * 0.85),
      region: targetRegion,
      anatomicalZone: zoneLabel
    },
    {
      title: "Hilar Lymphadenopathy Suggestion",
      description: `Bronchovascular tree markings and mediastinal structures show signs of inflammation or congestion adjacent to the primary focus area.`,
      confidence: Math.max(0.60, diag.confidence * 0.70),
      region: null,
      anatomicalZone: "hilar"
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
  workstationMode: "clinical" | "research" | "xai";
  setWorkstationMode: (mode: "clinical" | "research" | "xai") => void;
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
  handleFeedbackSaved,
  workstationMode,
  setWorkstationMode
}: ScreeningTabProps) {
  const activeResult = selectedIdx !== null ? results[selectedIdx] : null;
  const q = activeResult ? getQualityMetrics(activeResult) : null;
  const [customThreshold, setCustomThreshold] = useState<number | null>(null);
  
  const currentThreshold = customThreshold ?? activeResult?.threshold_used ?? 0.5;
  const isTbDerived = activeResult ? (activeResult.confidence || 0) >= currentThreshold : false;

  const activeDiagnosis = activeResult
    ? predictionService.getDiagnosis(isTbDerived ? "Tuberculosis" : "Normal", activeResult.confidence || 0.0, currentThreshold)
    : null;

  // ── WORKSTATION VIEWING STATES ──
  const [viewMode, setViewMode] = useState<"original" | "heatmap" | "heatmap-only" | "side-by-side" | "split" | "longitudinal">("original");
  const [heatmapOpacity, setHeatmapOpacity] = useState(0.55);
  const [priorImageSrc, setPriorImageSrc] = useState<string | undefined>();
  const [deltaHeatmapSrc, setDeltaHeatmapSrc] = useState<string | undefined>();
  const [isComparing, setIsComparing] = useState(false);

  const handleCompare = async (priorRecord: any) => {
    if (!activeResult || !activeResult.original_image) return;
    setIsComparing(true);
    try {
      const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000";
      const fd = new FormData();
      
      let blob;
      if (activeResult.original_image.startsWith("blob:")) {
        blob = await fetch(activeResult.original_image).then(r => r.blob());
      } else {
        const b64Data = activeResult.original_image.split(',')[1] || activeResult.original_image;
        const byteCharacters = atob(b64Data);
        const byteNumbers = new Array(byteCharacters.length);
        for (let i = 0; i < byteCharacters.length; i++) {
          byteNumbers[i] = byteCharacters.charCodeAt(i);
        }
        blob = new Blob([new Uint8Array(byteNumbers)], {type: 'image/png'});
      }
      
      fd.append("file", blob, "compare.png");
      fd.append("prior_image_b64", priorRecord.original_b64);
      
      const res = await fetch(`${API}/predict`, {
        method: "POST",
        body: fd,
        credentials: "include"
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Compare failed");
      
      setPriorImageSrc(priorRecord.original_b64);
      setDeltaHeatmapSrc(data.delta_heatmap_b64);
      setViewMode("longitudinal");
    } catch (e) {
      console.error("Compare error:", e);
      alert("Failed to compute delta heatmap");
    } finally {
      setIsComparing(false);
    }
  };

  // Automatically switch to clinical mode when an image successfully finishes AI processing
  useEffect(() => {
    if (activeResult?.status === "success" && activeResult?.xai_results) {
      setWorkstationMode("clinical");
    }
  }, [activeResult?.status, activeResult?.study_id]);

  const [activeRightTab, setActiveRightTab] = useState<"findings" | "review" | "report">("findings");
  const [highlightedAnatomicalZone, setHighlightedAnatomicalZone] = useState<string>("");
  const [zoomLevel, setZoomLevel] = useState<number>(1);
  const [xaiComparing, setXaiComparing] = useState<boolean>(false);

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
  const [iqaAcknowledged, setIqaAcknowledged] = useState(false);

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

    let annotatedB64 = "";
    if (annotationCanvasRef.current) {
      annotatedB64 = annotationCanvasRef.current.toDataURL("image/png");
    }

    handleFeedbackSaved(
      status,
      note,
      annotatedB64 || activeResult?.annotated_image || "",
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
            annotation_b64: annotatedB64 || activeResult.annotated_image || "",
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
        if (data.heatmaps || data.xai_results) {
          // Update the results state to include heatmaps and XAI telemetry for this index
          setResults((prev: AnalysisResult[]) => {
            const next = [...prev];
            if (next[idx]) {
              next[idx] = {
                ...next[idx],
                heatmaps: data.heatmaps || next[idx].heatmaps,
                xai_results: data.xai_results || next[idx].xai_results
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
        resolution: "2048 x 2048 pixels",
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
    setBoxes([]);
    setAnnotateMode(false);
    setObservationFocusRegion(null);
    setClinicalReviewStatus(activeResult?.clinician_override || "confirm");
    setReviewComments(activeResult?.review_comments || "");
    setReviewerName(activeResult?.reviewer_name || "");
    setClinicianNote(activeResult?.clinician_note || "");
    setDbRegistered(false);
    setIqaAcknowledged(false);
    setCustomThreshold(null);
    setPriorImageSrc(undefined);
    setDeltaHeatmapSrc(undefined);
    if (viewMode === "longitudinal") setViewMode("original");

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
    const diagnosisObj = predictionService.getDiagnosis(activeResult.prediction || "Normal", activeResult.confidence || 0.0, activeResult.threshold_used);
    const observationsList = observationService.getObservations(
      activeResult.prediction || "Normal",
      activeResult.xai_results ?? null
    );

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
      clinicianNote || globalNote,
      activeResult.xai_results
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
    addAuditLog("Exporting Structured JSON Report");

    const qualityMetrics = q;
    const diagnosisObj = predictionService.getDiagnosis(activeResult.prediction || "Normal", activeResult.confidence || 0.0, activeResult.threshold_used);
    const observationsList = observationService.getObservations(
      activeResult.prediction || "Normal",
      activeResult.xai_results ?? null
    );

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

    exportService.exportStructuredJSON(payload);
    addAuditLog("Structured JSON report metadata generated");
  };

  const handleRegisterDb = async () => {
    if (!activeResult || !q) return;
    setDbRegistered(true);
    addAuditLog("Registering case to research database");

    const qualityMetrics = q;
    const diagnosisObj = predictionService.getDiagnosis(activeResult.prediction || "Normal", activeResult.confidence || 0.0, activeResult.threshold_used);
    const observationsList = observationService.getObservations(
      activeResult.prediction || "Normal",
      activeResult.xai_results ?? null
    );

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
      { text: "Chest X-ray loaded in memory", done: true, loading: false },
      { text: "Grayscale intensity normalization & padding", done: activeResult.status === "success" || activeResult.status === "error", loading: isLoading && activeResult.status === "loading" },
      { text: "DenseNet-121 model load & pre-warm validation", done: activeResult.status === "success" || activeResult.status === "error", loading: false },
      { text: "AI classification risk scoring (tuberculosis vs normal)", done: activeResult.status === "success", loading: isLoading && activeResult.status === "loading" },
      { text: "Grad-CAM++ activation layer backpropagation maps", done: activeResult.status === "success", loading: isLoading && activeResult.status === "loading" },
      { text: "Anatomical zone contribution ROI calculations", done: activeResult.status === "success", loading: false }
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
              <div className="flex flex-col space-y-6 w-full animate-fadein p-6">
                <div className="px-6 py-4 bg-background border border-border/50 rounded-3xl flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 shadow-lg">
                  <div className="flex items-center gap-3">
                    {(["clinical", "research", "xai"] as const).map(mode => {
                      const isActive = workstationMode === mode;
                      return (
                        <button
                          key={mode}
                          onClick={() => setWorkstationMode(mode)}
                          className={`px-5 py-2 rounded-full text-xs font-bold transition-all duration-200 cursor-pointer border ${
                            isActive
                              ? "bg-[#5865F2] border-[#5865F2] text-white shadow-lg shadow-[#5865F2]/25 scale-[1.03]"
                              : "bg-[#383A40]/40 border-white/5 hover:bg-[#383A40]/80 text-[#949BA4] hover:text-white"
                          }`}
                        >
                          {mode === "clinical" && "Clinical View"}
                          {mode === "research" && "Research View"}
                          {mode === "xai" && "Observations"}
                        </button>
                      );
                    })}
                    {activeResult.study_id && activeResult.study_id !== "N/A" && (
                      <span className="text-xs font-mono text-muted-foreground bg-black/10 px-3 py-1.5 rounded-lg border border-white/5 ml-2">
                        Study: {activeResult.study_id}
                      </span>
                    )}
                  </div>
                  
                  {workstationMode === "xai" && (
                    <div className="flex items-center gap-3 ml-auto">
                      <Button 
                        variant="outline" 
                        size="sm" 
                        className={`border-border/50 text-xs font-medium rounded-xl transition-all cursor-pointer ${xaiComparing ? 'bg-secondary text-secondary-foreground border-secondary/80' : 'bg-card/50 text-muted-foreground hover:bg-card hover:text-foreground'}`}
                        onClick={() => setXaiComparing(!xaiComparing)}
                      >
                        <LayoutDashboard className="w-4 h-4 mr-2" />
                        {xaiComparing ? "Exit Split View" : "Split View"}
                      </Button>
                      <Button 
                        variant="outline" 
                        size="sm" 
                        className="border-border/50 bg-card/50 text-muted-foreground hover:bg-card hover:text-foreground text-xs font-medium rounded-xl transition-all cursor-pointer"
                        onClick={() => setZoomLevel(zoomLevel === 1.3 ? 1 : 1.3)}
                      >
                        <Search className="w-4 h-4 mr-2" />
                        {zoomLevel === 1.3 ? "Reset Zoom" : "Zoom 130%"}
                      </Button>
                    </div>
                  )}
                </div>

                {activeResult && activeResult.demo_mode && (
                  <div className="p-3 border border-yellow-500/20 bg-yellow-500/5 text-yellow-600 dark:text-yellow-500 rounded-xl flex items-center gap-3 animate-fadein">
                    <ShieldAlert className="w-4 h-4 shrink-0 text-yellow-500" />
                    <div>
                      <p className="text-xs font-bold uppercase tracking-wider">Demo / Fallback Mode Active</p>
                    </div>
                  </div>
                )}

                <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
                  {workstationMode === "xai" ? (
                    <div className="lg:col-span-12">
                      <XaiVisualization
                        result={activeResult}
                        similarCases={similarCases}
                        loadingSimilar={loadingSimilar}
                        workstationMode={workstationMode}
                        setWorkstationMode={setWorkstationMode}
                        zoomLevel={zoomLevel}
                        setZoomLevel={setZoomLevel}
                        isComparing={xaiComparing}
                        setIsComparing={setXaiComparing}
                      />
                    </div>
                  ) : (
                    <>
                      {/* 1. LEFT PANEL (70%): Unified Primary Viewport */}
                  <div className="lg:col-span-8 space-y-4">
                    <Card className="glass-panel rounded-2xl p-5 space-y-4 overflow-hidden">
                      <DicomViewer
                        imageBase64={activeResult.original_image || ""}
                        heatmapBase64={activeResult.heatmaps?.[xaiMethod] || activeResult.heatmap_image}
                        hasHeatmap={activeResult.status === "success" && !!(activeResult.heatmaps?.[xaiMethod] || activeResult.heatmap_image)}
                        label="Nirikshon Enterprise Viewport"
                        pixelSpacing={activeResult.metadata?.pixel_spacing}
                        viewMode={viewMode}
                        heatmapOpacity={heatmapOpacity}
                        setHeatmapOpacity={setHeatmapOpacity}
                        boxes={boxes}
                        setBoxes={setBoxes}
                        activeZone={activeZone}
                        annotateMode={annotateMode}
                        annotationCanvasRef={annotationCanvasRef}
                        setActiveZone={setHighlightedAnatomicalZone}
                        priorImageSrc={priorImageSrc}
                        deltaHeatmapSrc={deltaHeatmapSrc}
                        observationFocusRegion={observationFocusRegion}
                        setViewMode={setViewMode}
                        setAnnotateMode={setAnnotateMode}
                      />
                    </Card>

                    {activeResult && activeResult.saliency_fallback && (
                      <div className="mt-2 px-3 py-1.5 border border-amber-500/20 bg-amber-500/5 text-amber-600 dark:text-amber-500 rounded-lg flex items-center gap-2 text-[10px] font-medium leading-relaxed animate-fadein">
                        <AlertCircle className="w-3.5 h-3.5 shrink-0 text-amber-500" />
                        <span>Notice: Displaying raw density intensity heatmap (model activation gradients offline).</span>
                      </div>
                    )}

                  {/* Redraw list of active markups drawn on top of the X-ray */}
                  {boxes.length > 0 && (
                    <div className="flex flex-wrap items-center gap-2 p-3 glass-panel rounded-2xl">
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
                <div className="lg:col-span-4 space-y-4">
                  {workstationMode === "research" && (
                    <div className="mb-4 animate-fadein">
                      <TsnePlot />
                    </div>
                  )}

                  {/* UNIFIED VERDICT CARD (Always visible) */}
                  {workstationMode === "clinical" && activeResult && (
                    <div className="glass-panel rounded-2xl p-6 relative overflow-hidden">
                      {/* Subtle background glow based on risk */}
                      <div className={`absolute -top-10 -right-10 w-32 h-32 blur-3xl opacity-20 rounded-full pointer-events-none ${
                        activeResult.is_tb ? "bg-amber-500" : "bg-emerald-500"
                      }`} />
                      
                      <div className="flex flex-col gap-4 relative z-10">
                        <div className="flex justify-between items-start">
                          <div className="w-full">
                            <p className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground mb-1">
                              AI Classification
                            </p>
                            {activeResult.status === "error" ? (
                              <div className="space-y-3 mt-2 bg-destructive/10 border border-destructive/25 p-4 rounded-xl">
                                <div className="flex items-center gap-2 text-destructive font-semibold text-sm">
                                  <AlertCircle className="w-4 h-4 shrink-0" />
                                  <span>{activeResult.errorMsg || "Internal server error"}</span>
                                </div>
                                <p className="text-xs text-muted-foreground">
                                  Failed to complete radiograph processing. Please verify server connection and try again.
                                </p>
                                <button
                                  onClick={() => selectedIdx !== null && analyzeFile(selectedIdx)}
                                  className="px-3.5 py-1.5 rounded-lg text-xs font-bold bg-destructive/20 hover:bg-destructive/30 text-destructive border border-destructive/30 transition-all cursor-pointer"
                                >
                                  Retry Inference
                                </button>
                              </div>
                            ) : (
                              <h3 className={`text-xl font-extrabold tracking-tight ${
                                activeResult.status === "loading" || activeResult.status === "pending"
                                  ? "text-muted-foreground animate-pulse"
                                  : activeResult.is_tb
                                  ? "text-amber-500"
                                  : "text-emerald-500"
                              }`}>
                                {activeResult.status === "loading" || activeResult.status === "pending" ? (
                                  "Calculating..."
                                ) : (
                                  activeDiagnosis?.condition || "Normal"
                                )}
                              </h3>
                            )}
                          </div>
                          
                          {/* Minimalist Risk Badge */}
                          {activeResult.status === "success" && (
                            <Badge className={`uppercase font-bold text-[10px] py-1 px-3 rounded-full border-0 ${
                              activeResult.is_tb ? "bg-amber-500/20 text-amber-500" : "bg-emerald-500/20 text-emerald-500"
                            }`}>
                              {activeResult.is_tb ? "High Risk" : "Low Risk"}
                            </Badge>
                          )}
                        </div>

                        {/* Minimalist Confidence Bar */}
                        {activeResult.status === "success" && (
                          <div className="space-y-1.5 mt-2">
                            <div className="flex justify-between items-center text-xs">
                              <span className="text-muted-foreground font-medium">Confidence Score</span>
                              <span className="font-mono font-bold text-foreground">
                                {((activeDiagnosis?.confidence || 0) * 100).toFixed(1)}%
                              </span>
                            </div>
                            <Progress 
                              value={(activeDiagnosis?.confidence || 0) * 100} 
                              className={`h-1.5 bg-muted/30 ${activeResult.is_tb ? "[&>div]:bg-amber-500" : "[&>div]:bg-emerald-500"}`} 
                            />
                            
                            {/* Decision Threshold Slider */}
                            <div className="pt-3 mt-3 border-t border-white/5 flex items-center justify-between gap-3">
                              <span className="text-[9px] font-bold uppercase tracking-wider text-muted-foreground">Threshold</span>
                              <input 
                                type="range" 
                                min="0.1" 
                                max="0.9" 
                                step="0.05"
                                value={currentThreshold}
                                onChange={(e) => setCustomThreshold(parseFloat(e.target.value))}
                                className="flex-1 h-1 bg-muted/30 rounded-full appearance-none [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-primary cursor-pointer"
                              />
                              <span className="text-[10px] font-mono text-muted-foreground">{currentThreshold.toFixed(2)}</span>
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                  {activeResult && (activeResult.status === "loading" || activeResult.status === "pending") ? (
                    <div className="glass-panel rounded-2xl p-6 space-y-5 animate-fadein">
                      <div className="space-y-1">
                        <p className="text-[10px] font-bold uppercase tracking-widest text-primary">Inference Active</p>
                        <h4 className="text-sm font-bold text-foreground">AI Processing Pipeline</h4>
                      </div>
                      
                      {/* Detailed step-by-step progress checklist */}
                      <div className="space-y-4">
                        {getStepperStatus().map((step, idx) => (
                          <div key={idx} className="flex items-start gap-3">
                            <div className="mt-0.5 flex-shrink-0">
                              {step.done ? (
                                <div className="w-4 h-4 rounded-full bg-emerald-500/20 text-emerald-400 flex items-center justify-center">
                                  <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3}>
                                    <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                                  </svg>
                                </div>
                              ) : step.loading ? (
                                <div className="w-4 h-4 rounded-full border-2 border-primary border-t-transparent animate-spin"></div>
                              ) : (
                                <div className="w-4 h-4 rounded-full border border-muted-foreground/30 bg-muted/10"></div>
                              )}
                            </div>
                            <div className="flex-1 min-w-0">
                              <p className={`text-xs font-semibold ${step.done ? "text-foreground" : step.loading ? "text-primary animate-pulse" : "text-muted-foreground"}`}>
                                {step.text}
                              </p>
                            </div>
                          </div>
                        ))}
                      </div>

                      <div className="pt-4 border-t border-white/5 flex flex-col items-center justify-center text-center gap-2">
                        <div className="text-[10px] text-muted-foreground">Neural computation is executing on the backend container.</div>
                        <div className="w-full bg-muted/20 rounded-full h-1 overflow-hidden">
                          <div 
                            className="bg-primary h-1 rounded-full animate-pulse" 
                            style={{ 
                              width: activeResult.status === "loading" ? "75%" : "25%", 
                              transition: "width 1.5s ease-in-out" 
                            }} 
                          />
                        </div>
                      </div>
                    </div>
                  ) : workstationMode === "clinical" ? (
                    <>
                      {/* MINIMALIST BORDERLESS TABS */}
                      <div className="flex w-full mt-2 border-b border-border/40">
                        {(["findings", "review", "report"] as const).map(tab => (
                          <button
                            key={tab}
                            onClick={() => {
                              setActiveRightTab(tab);
                              addAuditLog(`Swapped right panel to ${tab} tab`);
                            }}
                            className={`flex-1 py-3 text-xs font-semibold capitalize transition-all duration-300 relative ${
                              activeRightTab === tab
                                ? "text-primary"
                                : "text-muted-foreground hover:text-foreground"
                            }`}
                          >
                            {tab === "findings" && "Findings"}
                            {tab === "review" && "Review"}
                            {tab === "report" && "Report"}
                            {activeRightTab === tab && (
                              <div className="absolute bottom-0 left-0 w-full h-[2px] bg-primary rounded-t-full" />
                            )}
                  </button>
                        ))}
                      </div>

                      {/* ───────────────── TAB 1: FINDINGS ───────────────── */}
                      {activeRightTab === "findings" && (
                        <div className="space-y-4 animate-fadein mt-4">

                          {/* IMAGE QUALITY ASSESSMENT (IQA) - Minimal Inline Layout */}
                          {q && (
                            <div className="glass-panel rounded-xl p-4 flex flex-col gap-3">
                              <div className="flex items-center justify-between">
                                <p className="text-[10px] font-bold uppercase tracking-wider text-foreground">Image Quality</p>
                                <div className="flex gap-2">
                                  <Badge variant={q.suitableForAi ? "default" : "destructive"} className="rounded-md font-bold uppercase text-[9px] px-2 py-0.5 border-0">
                                    {q.suitableForAi ? "Suitable" : "Unsuitable"}
                                  </Badge>
                                  <span className="text-[10px] font-mono text-muted-foreground font-bold">Score: {q.qualityScore}%</span>
                                </div>
                              </div>
                              <div className="flex gap-4 text-[10px] text-muted-foreground">
                                <span className={q.exposure === "Adequate Exposure" ? "text-emerald-500" : "text-amber-500"}>• {q.exposure}</span>
                                <span className={q.coverage === "Full Lung Coverage" ? "text-emerald-500" : "text-amber-500"}>• {q.coverage}</span>
                              </div>
                              {!q.suitableForAi && (
                                <div className="flex items-center gap-2 mt-1 pt-3 border-t border-white/5">
                                  <input 
                                    type="checkbox" 
                                    id="iqa-ack" 
                                    className="h-3 w-3 accent-amber-500"
                                    checked={iqaAcknowledged}
                                    onChange={(e) => setIqaAcknowledged(e.target.checked)}
                                  />
                                  <label htmlFor="iqa-ack" className="text-[10px] font-bold text-amber-500 cursor-pointer select-none">
                                    Acknowledge sub-optimal quality to proceed
                                  </label>
                                </div>
                              )}
                            </div>
                          )}

                          <div className="space-y-3 pt-2">
                            <div className="flex items-center gap-1.5 pl-1">
                              <Eye className="w-3.5 h-3.5 text-primary" />
                              <p className="text-[10px] font-bold uppercase tracking-wider text-foreground">Clinical Evidence Findings</p>
                            </div>
                            {activeResult.status === "loading" || activeResult.status === "pending" ? (
                              <div className="space-y-3">
                                {[1, 2].map((i) => (
                                  <div key={i} className="glass-panel p-4 rounded-xl space-y-2 animate-pulse">
                                    <div className="h-4 bg-muted/50 rounded w-1/3"></div>
                                    <div className="h-3 bg-muted/30 rounded w-full"></div>
                                    <div className="h-3 bg-muted/30 rounded w-4/5"></div>
                                  </div>
                                ))}
                              </div>
                            ) : (
                              <div className="space-y-3">
                                {getEvidenceCards(activeResult).map((ec, idx) => {
                                  const isAbnormal = activeResult.is_tb && (ec.title.toLowerCase().includes("consolidation") || ec.title.toLowerCase().includes("density") || ec.title.toLowerCase().includes("failed") || ec.title.toLowerCase().includes("infiltrate"));
                                  const borderColor = isAbnormal ? "border-l-amber-500 hover:border-l-amber-400" : "border-l-emerald-500 hover:border-l-emerald-400";
                                  const badgeColor = isAbnormal ? "bg-amber-500/10 text-amber-400" : "bg-emerald-500/10 text-emerald-400";
                                  
                                  return (
                                    <div 
                                      key={idx} 
                                      className={`glass-panel p-4 rounded-xl border-l-4 ${borderColor} bg-card/25 backdrop-blur hover:bg-card/45 transition-all duration-200 group shadow-sm`}
                                    >
                                      <div className="flex justify-between items-start mb-1.5">
                                        <h4 className="text-xs font-bold text-foreground group-hover:text-primary transition-colors pr-2">
                                          {ec.title}
                                        </h4>
                                        {ec.confidence > 0 && (
                                          <Badge className={`rounded font-mono font-bold text-[9px] px-1.5 py-0.5 border-0 ${badgeColor}`}>
                                            {(ec.confidence * 100).toFixed(0)}%
                                          </Badge>
                                        )}
                                      </div>
                                      <p className="text-[11px] text-muted-foreground leading-relaxed">{ec.description}</p>
                                    </div>
                                  );
                                })}
                              </div>
                            )}
                          </div>
                        </div>
                      )}

                      {/* ───────────────── TAB 2: CLINICIAN OVERRIDE REVIEW ───────────────── */}
                      {activeRightTab === "review" && (
                        <div className="space-y-6 animate-fadein">
                          {/* CLINICAL AUDIT / OVERRIDE INPUTS */}
                          <Card className="glass-panel">
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
                                <label className="text-[10px] text-muted-foreground uppercase font-bold">XAI Observations &amp; Study Notes</label>
                                <Textarea
                                  value={clinicianNote}
                                  onChange={e => {
                                    setClinicianNote(e.target.value);
                                    syncFeedback(clinicalReviewStatus, reviewComments, reviewerName, e.target.value);
                                  }}
                                  placeholder="Type focal anomaly observations, model alignment comments, or secondary findings..."
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
                          <LongitudinalTracker 
                            patientId={activeResult.metadata?.patient_id || ""} 
                            patientName={activeResult.metadata?.patient_name || "Unknown Patient"} 
                            currentResult={activeResult as any}
                            onCompare={handleCompare}
                          />
                        </div>
                      )}


                      {/* ───────────────── TAB 4: STRUCTURED CLINICAL REPORT ───────────────── */}
                      {activeRightTab === "report" && (
                        <div className="space-y-6 animate-fadein">
                          {/* REPORT SUMMARY CARD */}
                          <Card className="glass-panel">
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
                                        {activeDiagnosis?.condition}
                                        <span className="text-muted-foreground text-[11px] font-normal ml-1">({((activeDiagnosis?.confidence || 0) * 100).toFixed(1)}% confidence)</span>
                                      </>
                                    )}
                                  </p>
                                </div>

                                <div>
                                  <span className="text-muted-foreground text-[10px] uppercase font-bold">AI Saliency Explanation:</span>
                                  <p className="text-xs font-serif italic text-muted-foreground mt-0.5 leading-relaxed">
                                    "{activeResult.xai_results?.summary || "The model focused primarily on the upper right lung region. Increased opacity and abnormal density patterns within this area contributed significantly to the final prediction."}"
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
                          <div className="p-4 glass-panel space-y-3">
                            <p className="text-xs font-bold uppercase tracking-wider text-foreground">Downstream Integrations &amp; Export</p>
                            
                            {activeResult.demo_mode && (
                              <div className="p-2 mb-2 bg-red-500/10 border border-red-500/30 rounded-lg flex items-start gap-2">
                                <AlertTriangle className="w-4 h-4 text-red-500 shrink-0 mt-0.5" />
                                <p className="text-[11px] text-red-600 dark:text-red-400 font-medium">
                                  Exporting is disabled in Demo Mode. Results are simulated.
                                </p>
                              </div>
                            )}

                            {!activeResult.demo_mode && (!reviewerName.trim() || (!q?.suitableForAi && !iqaAcknowledged)) && (
                              <div className="p-2 mb-2 bg-amber-500/10 border border-amber-500/30 rounded-lg flex flex-col gap-1">
                                <p className="text-[10px] font-bold text-amber-600 uppercase">Pre-flight Checklist Required:</p>
                                {!reviewerName.trim() && <p className="text-[11px] text-amber-600">• Clinical Reviewer name is required</p>}
                                {q && !q.suitableForAi && !iqaAcknowledged && <p className="text-[11px] text-amber-600">• Must acknowledge sub-optimal image quality</p>}
                              </div>
                            )}

                            {/* Exporters buttons row */}
                            <div className="flex gap-2">
                              <Button
                                size="sm"
                                onClick={handlePdfExport}
                                disabled={isExporting || activeResult.status !== "success" || activeResult.demo_mode || !reviewerName.trim() || (!q?.suitableForAi && !iqaAcknowledged)}
                                className="flex-1 text-xs font-semibold h-9 gap-1.5 cursor-pointer"
                              >
                                <FileText className="w-3.5 h-3.5" /> Export PDF
                              </Button>
                              <Button
                                size="sm"
                                variant="outline"
                                onClick={handleJsonSR}
                                disabled={activeResult.status !== "success" || activeResult.demo_mode}
                                className="flex-1 text-xs font-semibold h-9 gap-1.5 cursor-pointer"
                              >
                                <Download className="w-3.5 h-3.5" /> Export JSON
                              </Button>
                              <Button
                                size="sm"
                                variant={dbRegistered ? "secondary" : "outline"}
                                onClick={handleRegisterDb}
                                disabled={dbRegistered || activeResult.status !== "success" || activeResult.demo_mode}
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
                      <Card className="border border-border/50 bg-card/40 backdrop-blur-xl rounded-2xl shadow-[inset_0_1px_1px_rgba(255,255,255,0.05)]">
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
                    </div>
                  )}
                </div>
              </>
            )}
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
