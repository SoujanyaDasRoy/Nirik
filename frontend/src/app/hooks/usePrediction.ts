import { useState } from "react";
import { AnalysisResult } from "./useFileUpload";

export function getCookie(name: string): string | null {
  if (typeof document === "undefined") return null;
  const value = `; ${document.cookie}`;
  const parts = value.split(`; ${name}=`);
  if (parts.length === 2) return parts.pop()?.split(";").shift() || null;
  return null;
}

export function usePrediction(
  files: File[],
  results: AnalysisResult[],
  setResults: React.Dispatch<React.SetStateAction<AnalysisResult[]>>,
  setSelectedIdx: React.Dispatch<React.SetStateAction<number | null>>
) {
  const [isBatchProcessing, setIsBatchProcessing] = useState(false);

  const analyzeFile = async (idx: number) => {
    const file = files[idx];
    // Create a local preview URL immediately so DicomViewer shows the image while backend processes
    const localPreviewUrl = URL.createObjectURL(file);
    setResults(prev => {
      const next = [...prev];
      next[idx] = { ...next[idx], status: "loading", original_image: localPreviewUrl };
      return next;
    });

    const API = process.env.NEXT_PUBLIC_API_URL || "https://projectmantra-nirikshon-backend.hf.space";
    const fd = new FormData();
    fd.append("file", file);

    try {
      const res = await fetch(`${API}/predict`, { 
        method: "POST", 
        body: fd,
        credentials: "include" // Send cookies for CSRF double-submit validation
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Inference failed");

      setResults(prev => {
        const next = [...prev];
        next[idx] = {
          filename: file.name,
          status: "success",
          prediction: data.prediction,
          confidence: data.confidence,
          is_tb: data.is_tb,
          metadata: data.metadata,
          original_image: data.original_image,
          heatmap_image: data.heatmap_image,
          study_id: data.study_id,
          image_quality: data.image_quality,
          heatmaps: data.heatmaps,
          xai_results: data.xai_results,
          demo_mode: data.demo_mode,
          saliency_fallback: data.saliency_fallback
        };
        return next;
      });
      setSelectedIdx(idx);
    } catch (err: any) {
      setResults(prev => {
        const next = [...prev];
        next[idx] = {
          ...next[idx],
          status: "error",
          errorMsg: err.message || "Server connection failed"
        };
        return next;
      });
    }
  };

  const analyzeAll = async () => {
    if (isBatchProcessing || results.length === 0) return;
    setIsBatchProcessing(true);

    // Filter indices of files that still need to be processed (not success status)
    const indicesToProcess = results
      .map((res, idx) => (res.status !== "success" ? idx : -1))
      .filter(idx => idx !== -1);

    if (indicesToProcess.length === 0) {
      setIsBatchProcessing(false);
      return;
    }

    let nextIndex = 0;
    const worker = async () => {
      while (nextIndex < indicesToProcess.length) {
        const currentTaskIndex = nextIndex++;
        if (currentTaskIndex >= indicesToProcess.length) break;
        const fileIdx = indicesToProcess[currentTaskIndex];
        try {
          await analyzeFile(fileIdx);
        } catch (err) {
          console.error(`Error analyzing file ${fileIdx}:`, err);
        }
      }
    };

    // Concurrency limit of 3 workers
    const workers = [];
    const concurrencyLimit = Math.min(3, indicesToProcess.length);
    for (let i = 0; i < concurrencyLimit; i++) {
      workers.push(worker());
    }

    await Promise.all(workers);
    setIsBatchProcessing(false);
  };

  return {
    isBatchProcessing,
    analyzeFile,
    analyzeAll
  };
}
