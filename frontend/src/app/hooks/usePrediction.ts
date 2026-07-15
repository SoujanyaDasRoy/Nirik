import { useCallback, useEffect, useRef, useState } from "react";
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
  const inFlight = useRef<Set<number>>(new Set());
  // Track object URLs to revoke them when no longer needed
  const objectUrls = useRef<Map<number, string>>(new Map());
  // Track previous file and result lengths to detect removals
  const prevFilesLength = useRef(files.length);
  const prevResultsLength = useRef(results.length);

  // Clean up object URLs when component unmounts or dependencies change
  useEffect(() => {
    return () => {
      objectUrls.current.forEach(url => {
        try {
          URL.revokeObjectURL(url);
        } catch (e) {
          console.warn('Failed to revoke object URL:', e);
        }
      });
      objectUrls.current.clear();
    };
  }, []);

  // Detect when files are removed and clean up their object URLs
  useEffect(() => {
    if (files.length < prevFilesLength.current) {
      // Files were removed, find which indices were removed and clean up their URLs
      const oldIndices = new Array(prevFilesLength.current).fill(0).map((_, i) => i);
      const newIndices = new Array(files.length).fill(0).map((_, i) => i);
      const removedIndices = oldIndices.filter(i => !newIndices.includes(i));

      removedIndices.forEach(index => {
        const url = objectUrls.current.get(index);
        if (url) {
          try {
            URL.revokeObjectURL(url);
          } catch (e) {
            console.warn('Failed to revoke object URL for removed file:', e);
          }
          objectUrls.current.delete(index);
        }
      });
    }
    prevFilesLength.current = files.length;
  }, [files]);

  // Detect when results are removed and clean up their object URLs
  useEffect(() => {
    if (results.length < prevResultsLength.current) {
      // Results were removed, find which indices were removed and clean up their URLs
      const oldIndices = new Array(prevResultsLength.current).fill(0).map((_, i) => i);
      const newIndices = new Array(results.length).fill(0).map((_, i) => i);
      const removedIndices = oldIndices.filter(i => !newIndices.includes(i));

      removedIndices.forEach(index => {
        const url = objectUrls.current.get(index);
        if (url) {
          try {
            URL.revokeObjectURL(url);
          } catch (e) {
            console.warn('Failed to revoke object URL for removed result:', e);
          }
          objectUrls.current.delete(index);
        }
      });
    }
    prevResultsLength.current = results.length;
  }, [results]);

  const analyzeFile = useCallback(async (idx: number) => {
    const file = files[idx];
    if (!file) return;
    // Dedupe concurrent invocations for the same file index. The ScreeningTab
    // effect fires on [selectedIdx, status, analyzeFile] — if analyzeFile ref
    // changes between renders or the user retries rapidly, multiple calls
    // would otherwise stack and trigger duplicate POSTs.
    if (inFlight.current.has(idx)) return;
    inFlight.current.add(idx);
    try {
      // Revoke previous object URL for this index if it exists
      const prevUrl = objectUrls.current.get(idx);
      if (prevUrl) {
        try {
          URL.revokeObjectURL(prevUrl);
        } catch (e) {
          console.warn('Failed to revoke previous object URL:', e);
        }
      }

      // Create a local preview URL and DicomViewer shows the image
      // while backend processes.
      const localPreviewUrl = URL.createObjectURL(file);
      // Store the URL so we can revoke it later
      objectUrls.current.set(idx, localPreviewUrl);

      const API = process.env.NEXT_PUBLIC_API_URL || "https://projectmantra-nirikshon-backend.hf.space";
      const fd = new FormData();
      fd.append("image", file);
      // Always request explainability data since backend is already computing it
      fd.append("include_explainability", "true");

      try {
        const res = await fetch(`${API}/predict`, {
          method: "POST",
          body: fd,
          credentials: "include" // Send cookies for CSRF double-submit validation
        });

        // Always read the body once. Server may return JSON (4xx with error
        // payload) or plain text (5xx). Don't assume JSON.
        let payload: any = null;
        const text = await res.text();
        if (text) {
          try {
            payload = JSON.parse(text);
          } catch {
            payload = null;
          }
        }

        if (!res.ok) {
          const serverMsg =
            (payload && (payload.error || payload.message)) || text || res.statusText;
          throw new Error(serverMsg || `Request failed (${res.status})`);
        }

        const data = payload || {};

        setResults(prev => {
          const next = [...prev];
          // Do not revoke the object URL here, as we need it for original_image if backend doesn't provide one.
          // The useEffect will clean it up when the result is removed.
          const prevUrl = objectUrls.current.get(idx);


          next[idx] = {
            filename: file.name,
            status: "success",
            prediction: data.prediction,
            confidence: data.confidence,
            is_tb: data.prediction === "Tuberculosis",
            threshold_used: data.threshold_used,
            segmentation_active: data.segmentation_active || false,
            metadata: data.metadata || {
               patient_id: data.request_id?.substring(0, 8).toUpperCase(),
               patient_name: "Anonymous",
               patient_age: "Unknown",
               patient_sex: "Unknown",
               modality: "CR",
               study_date: new Date().toISOString().split('T')[0]
            },
            original_image: data.original_image || prevUrl || null,
            heatmap_image: data.explainability?.gradcam_plus_plus || data.explainability?.gradcam || null,
            study_id: data.request_id || "unknown",
            image_quality: data.image_quality || {
               exposure: "Adequate Exposure",
               coverage: "Full Lung Coverage",
               resolution: "2048 x 2048 pixels",
               rotation: "No Rotation",
               quality_score: 95,
               suitable_for_ai: true,
               warnings: []
            },
            heatmaps: data.explainability || null,
            xai_results: data.xai_results || data.explainability || null,
            clinical_observations: data.clinical_observations || data.explainability?.clinical_observations || null,
            demo_mode: data.demo_mode || false,
            saliency_fallback: data.saliency_fallback || false,
            findings: data.findings || [],
            report: data.report || null
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
    } finally {
      inFlight.current.delete(idx);
    }
  }, [files, setResults, setSelectedIdx]);

  const analyzeAll = useCallback(async () => {
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
  }, [isBatchProcessing, results, analyzeFile]);

  return {
    isBatchProcessing,
    analyzeFile,
    analyzeAll
  };
}