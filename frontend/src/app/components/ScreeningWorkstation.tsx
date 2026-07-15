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

interface ScreeningWorkstationProps {
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

export function ScreeningWorkstation({
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

  handleFeedbackSaved,
  workstationMode,
  setWorkstationMode,
}: ScreeningWorkstationProps) {
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

  // XAI method
  const [xaiMethod, setXaiMethod] = useState<"gradcam_plusplus">("gradcam_plusplus");

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
    <div className="w-full max-w-[900px] mx-auto bg-[#090909] text-[#ffffff] flex flex-col gap-10 pb-20">
      
      {/* 1. ORIGINAL X-RAY */}
      <section className="flex flex-col gap-3">
        <h2 className="text-[20px] font-medium tracking-tight flex items-center gap-2">
          <Eye className="w-5 h-5 text-[#999999]" />
          Original X-Ray
        </h2>
        <div className="w-full min-h-[400px] md:h-[600px] rounded-[24px] overflow-hidden bg-[#141414] border border-[#262626] shadow-xl relative">
          <DicomViewer
            imageBase64={activeResult?.original_image ?? ""}
            hasHeatmap={false}
            label="Original Scan"
            viewMode="original"
          />
        </div>
      </section>

      {/* 2. GRAD-CAM OVERLAY */}
      {activeResult?.status === "success" && (activeResult?.heatmaps?.[xaiMethod] || activeResult?.heatmap_image) && (
        <section className="flex flex-col gap-3 animate-fadein">
          <h2 className="text-[20px] font-medium tracking-tight flex items-center gap-2">
            <Layers className="w-5 h-5 text-[#0099ff]" />
            Grad-CAM Overlay
          </h2>
          <div className="w-full min-h-[400px] md:h-[600px] rounded-[24px] overflow-hidden bg-[#141414] border border-[#262626] shadow-xl relative">
            <DicomViewer
              imageBase64={activeResult?.original_image ?? ""}
              heatmapBase64={activeResult?.heatmaps?.[xaiMethod] ?? activeResult?.heatmap_image ?? ""}
              hasHeatmap={true}
              label="Activation Heatmap"
              viewMode="heatmap"
              heatmapOpacity={0.55}
            />
          </div>
        </section>
      )}

      {/* 3. PREDICTION + CONFIDENCE */}
      <section className="flex flex-col gap-3">
        <h2 className="text-[20px] font-medium tracking-tight flex items-center gap-2">
          <Activity className="w-5 h-5 text-[#10b981]" />
          Prediction & Confidence
        </h2>
        <div className="w-full p-8 rounded-[24px] bg-[#141414] border border-[#262626] shadow-xl relative overflow-hidden">
          <div className={`absolute -top-20 -right-20 w-64 h-64 blur-[80px] opacity-20 rounded-full pointer-events-none transition-colors duration-1000 ${
            activeResult?.status === "loading" || activeResult?.status === "pending"
              ? "bg-[#ffffff]"
              : activeDiagnosis?.riskLevel === "High"
              ? "bg-red-500"
              : "bg-blue-500"
          }`} />

          {activeResult?.status === "error" ? (
            <div className="space-y-4 relative z-10 bg-destructive/10 border border-destructive/25 p-6 rounded-3xl">
              <div className="flex items-center gap-3 text-destructive font-semibold text-lg">
                <AlertCircle className="w-6 h-6" />
                <span>{activeResult.errorMsg || "Internal server error"}</span>
              </div>
              <button
                onClick={() => selectedIdx !== null && analyzeFile(selectedIdx)}
                className="px-6 py-3 rounded-full font-bold bg-destructive/20 hover:bg-destructive/30 text-destructive border border-destructive/30 transition-all cursor-pointer"
              >
                Retry Inference
              </button>
            </div>
          ) : activeResult?.status === "loading" || activeResult?.status === "pending" ? (
            <div className="space-y-6 relative z-10">
              <h3 className="text-4xl font-extrabold tracking-[-1.5px] text-[#ffffff] flex items-center gap-4">
                <Loader2 className="w-8 h-8 animate-spin text-[#0099ff]" />
                Analyzing Scan...
              </h3>
              <Progress value={75} className="h-2 bg-[#262626] [&>div]:bg-[#ffffff] [&>div]:animate-pulse" />
            </div>
          ) : (
            <div className="relative z-10">
              <h3 className={`text-6xl md:text-7xl font-bold tracking-[-3px] mb-6 drop-shadow-md ${
                activeDiagnosis?.riskLevel === "High" ? "text-red-500" : "text-blue-500"
              }`}>
                {activeDiagnosis?.condition ?? "Normal"}
              </h3>
              
              <div className="space-y-3 mb-8">
                <div className="flex justify-between text-[14px] font-bold text-[#999999] uppercase tracking-wide">
                  <span>AI Confidence Score</span>
                  <span className="text-[#ffffff]">
                    {((activeDiagnosis?.confidence ?? 0) * 100).toFixed(1)}%
                  </span>
                </div>
                <Progress
                  value={(activeDiagnosis?.confidence ?? 0) * 100}
                  className={`h-3 bg-[#262626] ${
                    activeDiagnosis?.riskLevel === "High"
                      ? "[&>div]:bg-red-500"
                      : "[&>div]:bg-blue-500"
                  }`}
                />
              </div>

              <div className="flex flex-wrap gap-3">
                <Badge className="bg-[#262626] text-[#ffffff] hover:bg-[#333333] border border-[#262626] text-[12px] uppercase font-bold py-1.5 px-4 rounded-full">
                  {activeResult?.segmentation_active ? "U-Net Segmented" : "Direct Input"}
                </Badge>
                <Badge className={`uppercase font-bold text-[12px] py-1.5 px-4 rounded-full border border-[#262626] ${
                  activeDiagnosis?.riskLevel === "High"
                    ? "bg-red-500/20 text-red-500"
                    : "bg-blue-500/20 text-blue-500"
                }`}>
                  {activeDiagnosis?.riskLevel} Risk
                </Badge>
              </div>

              <div className="mt-8 pt-8 border-t border-[#262626] flex justify-end">
                <button
                  onClick={handlePdfExport}
                  disabled={isExporting}
                  className="flex items-center gap-2 px-5 py-2.5 rounded-full font-semibold text-[13px] bg-[#ffffff] text-[#000000] hover:scale-[1.02] transition-transform disabled:opacity-50 disabled:hover:scale-100"
                >
                  <Download className="w-4 h-4" />
                  {isExporting ? "Generating PDF..." : "Export Clinical Report"}
                </button>
              </div>
            </div>
          )}
        </div>
      </section>

      {/* 4. DETAILS OF THE DIAGNOSIS / EXPLANATION */}
      {activeResult?.status === "success" && (
        <section className="flex flex-col gap-3 animate-fadein">
          <h2 className="text-[20px] font-medium tracking-tight flex items-center gap-2">
            <Bot className="w-5 h-5 text-[#8b5cf6]" />
            Details of the Diagnosis
          </h2>
          <div className="w-full p-8 rounded-[24px] bg-[#141414] border border-[#262626] shadow-xl relative overflow-hidden text-[#cccccc] text-[15px] leading-relaxed">
            <div className="space-y-4">
              <p>
                <strong>AI Explanation:</strong> Based on the DenseNet-121 model analysis, the radiograph exhibits features {activeDiagnosis?.riskLevel === "High" ? "strongly indicative of Pulmonary Tuberculosis" : "consistent with normal lung anatomy"}. The confidence level of this prediction is {((activeDiagnosis?.confidence ?? 0) * 100).toFixed(1)}%.
              </p>
              
              {activeDiagnosis?.riskLevel === "High" ? (
                <>
                  <p>
                    <strong>Key Observations:</strong> The Grad-CAM heatmap highlights specific anatomical regions of interest (typically upper lung zones or apical regions) where the neural network detected pathological textures or opacities correlating with active TB infection.
                  </p>
                  <p>
                    <strong>Clinical Recommendation:</strong> Urgent clinical correlation is advised. Sputum AFB smear, GeneXpert MTB/RIF, and further microbiological testing should be considered to confirm active infection.
                  </p>
                </>
              ) : (
                <>
                  <p>
                    <strong>Key Observations:</strong> The Grad-CAM heatmap shows diffuse or minimal activations, with no focal areas of high pathological significance detected by the model. The lung fields appear generally clear of TB-specific opacities.
                  </p>
                  <p>
                    <strong>Clinical Recommendation:</strong> No immediate action is required based solely on this radiograph. However, AI screening does not replace clinical judgement; evaluate the patient for any symptomatic presentations.
                  </p>
                </>
              )}
            </div>
          </div>
        </section>
      )}
    </div>
  );
}
