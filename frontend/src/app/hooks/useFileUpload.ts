import { useState, useRef, DragEvent, ChangeEvent } from "react";

export interface AnalysisResult {
  filename: string;
  status: "pending" | "loading" | "success" | "error";
  prediction?: string;
  confidence?: number;
  is_tb?: boolean;
  threshold_used?: number;
  segmentation_active?: boolean;
  metadata?: {
    patient_id?: string;
    patient_name?: string;
    patient_age?: string;
    patient_sex?: string;
    modality?: string;
    study_date?: string;
    body_part?: string;
    pixel_spacing?: number[] | null;
  };
  original_image?: string;
  heatmap_image?: string;
  errorMsg?: string;
  clinician_override?: string | null;
  clinician_note?: string;
  annotated_image?: string;
  attention_region?: string;
  heatmap_coverage?: number;
  review_comments?: string;
  reviewer_name?: string;
  study_id?: string;
  image_quality?: {
    exposure: string;
    coverage: string;
    resolution: string;
    rotation: string;
    suitable_for_ai: boolean;
    quality_score: number;
    warnings: string[];
  };
  heatmaps?: {
    gradcam: string;
    gradcam_plusplus: string;
    attention: string;
    coverage: string;
    attribution: string;
  };
  demo_mode?: boolean;
  saliency_fallback?: boolean;
  xai_results?: {
    rois: {
      id: string;
      activation_score: number;
      contribution_pct: number;
      location: string;
      bbox: [number, number, number, number];
      circle: [number, number, number];
      contour: [number, number][];
      center: [number, number];
    }[];
    summary: string;
    ranking: {
      region_id: string;
      location: string;
      contribution_pct: number;
    }[];
    metrics: {
      tb_probability: number;
      calibrated_confidence: number;
      reliability: string;
      uncertainty: string;
    };
  };
  quadrant_analysis?: {
    quadrant_scores: {
      upper_left: number;
      upper_right: number;
      lower_left: number;
      lower_right: number;
    };
    upper_fraction: number;
    lower_fraction: number;
    dominant_zone: "upper" | "lower" | "mixed";
    interpretation: string;
    disease_overlap: string[];
  };
  /**
   * Phase 5: Rich clinical observations derived by the backend from the
   * XAI ROI payload. Each entry carries a full multi-clause narrative
   * sentence, recommended follow-up, and differential diagnoses — see
   * backend/utils/observation_builder.py. May be empty for older records
   * written before the column was added; the frontend falls back to the
   * XAI-derived path in that case.
   */
  clinical_observations?: DetailedClinicalObservation[];
}

/**
 * Backend-issued rich clinical observation. Mirrors the dict shape
 * produced by `backend.utils.observation_builder.build_clinical_observations`.
 */
export interface DetailedClinicalObservation {
  /** ROI label, e.g. "A", "B" */
  id: string;
  /** ICD-10-ish short code, e.g. "A15.0" (TB lung), "R91.1" (abnormal CXR) */
  code: string;
  /** Human-readable finding label, e.g. "Focal apical consolidation" */
  label: string;
  /** Free-form location string from the XAI ROI, e.g. "Right Upper Lung Zone" */
  location: string;
  /** Parsed laterality */
  laterality: "Right" | "Left" | "Bilateral";
  /** Parsed lung zone */
  zone: "Upper" | "Middle" | "Lower" | "Pleural";
  /** TB-pattern vs benign pattern */
  descriptor: "TB-specific" | "Non-TB";
  /** High/Moderate/Low clinical weight */
  clinical_significance: "High" | "Moderate" | "Low";
  /** 0..1 evidence score combining activation and contribution */
  evidence_score: number;
  /** 0..100 contribution share of total model attention */
  contribution_pct: number;
  /** 0..100 peak activation at the ROI centroid */
  activation_score: number;
  /** Severity bucket derived from activation */
  severity: "Critical" | "Marked" | "Moderate" | "Mild" | "Background";
  /** Bounding box in image coordinates: [x, y, w, h] */
  bbox: [number, number, number, number];
  /** Enclosing circle: [cx, cy, radius] */
  circle: [number, number, number];
  /** Polygon contour points: [[x, y], ...] */
  contour: [number, number][];
  /** Normalized centre: [nx, ny] in 0..1 */
  center: [number, number];
  /** Pan/zoom hint for DicomViewer (coords on the 224-grid) */
  target_region: {
    x1: number;
    y1: number;
    x2: number;
    y2: number;
    zoom: number;
    panX: number;
    panY: number;
  };
  /** Full multi-clause narrative sentence — never truncated */
  narrative: string;
  /** Recommended clinical follow-up actions */
  recommended_followup: string[];
  /** Differential diagnosis list */
  differential_diagnoses: string[];
}

export function useFileUpload() {
  const [files, setFiles] = useState<File[]>([]);
  const [results, setResults] = useState<AnalysisResult[]>([]);
  const [selectedIdx, setSelectedIdx] = useState<number | null>(null);
  const [isDragActive, setIsDragActive] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDrag = (e: DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setIsDragActive(true);
    } else if (e.type === "dragleave") {
      setIsDragActive(false);
    }
  };

  const handleDrop = (e: DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const added = Array.from(e.dataTransfer.files);
      addFiles(added);
    }
  };

  const handleFileInput = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const added = Array.from(e.target.files);
      addFiles(added);
    }
  };

  const addFiles = (added: File[]) => {
    setFiles(prev => [...prev, ...added]);
    setResults(prev => [
      ...prev,
      ...added.map(f => ({ filename: f.name, status: "pending" as const }))
    ]);
  };

  const removeFile = (idx: number) => {
    setFiles(prev => {
      const next = [...prev];
      next.splice(idx, 1);
      return next;
    });
    setResults(prev => {
      const next = [...prev];
      next.splice(idx, 1);
      return next;
    });
    setSelectedIdx(prev => {
      if (prev === null) return null;
      if (prev === idx) return null;
      if (prev > idx) return prev - 1;
      return prev;
    });
  };

  const clearAll = () => {
    setFiles([]);
    setResults([]);
    setSelectedIdx(null);
  };

  return {
    files,
    setFiles,
    results,
    setResults,
    selectedIdx,
    setSelectedIdx,
    isDragActive,
    setIsDragActive,
    fileInputRef,
    handleDrag,
    handleDrop,
    handleFileInput,
    addFiles,
    removeFile,
    clearAll
  };
}
