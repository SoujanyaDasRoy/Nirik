import { useState, useRef, DragEvent, ChangeEvent } from "react";

export interface AnalysisResult {
  filename: string;
  status: "pending" | "loading" | "success" | "error";
  prediction?: string;
  confidence?: number;
  is_tb?: boolean;
  threshold_used?: number;
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
