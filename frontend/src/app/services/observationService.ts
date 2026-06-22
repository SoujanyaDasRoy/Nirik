export interface ClinicalObservation {
  id: string;
  text: string;
  location: string;
  evidenceScore: number;
  confidence: number;
  coordinates: { x1: number; y1: number; x2: number; y2: number };
  targetRegion: {
    x1: number;
    y1: number;
    x2: number;
    y2: number;
    zoom: number;
    panX: number;
    panY: number;
  };
}

interface XaiRoi {
  id: string;
  activation_score: number;
  contribution_pct: number;
  location: string;
  bbox: [number, number, number, number];
  circle: [number, number, number];
  contour: [number, number][];
  center: [number, number];
}

interface XaiResults {
  rois: XaiRoi[];
  summary: string;
  ranking: { region_id: string; location: string; contribution_pct: number }[];
  metrics: {
    tb_probability: number;
    calibrated_confidence: number;
    reliability: string;
    uncertainty: string;
  };
}

/**
 * Converts a ROI bbox ([x, y, w, h] in natural image pixels) to a
 * coordinate object usable by the viewer. All values are passed through
 * unchanged — the viewer is responsible for scaling to the displayed size.
 */
function bboxToCoords(bbox: [number, number, number, number]) {
  const [x, y, w, h] = bbox;
  return { x1: x, y1: y, x2: x + w, y2: y + h };
}

/**
 * Generates a target-region viewport hint from a bbox.
 * Adds 20% padding around the region and derives a zoom level from
 * how small the region is relative to an assumed 224×224 grid.
 */
function bboxToTargetRegion(
  bbox: [number, number, number, number],
  imgW = 224,
  imgH = 224
) {
  const [x, y, w, h] = bbox;
  const pad = Math.round(Math.min(w, h) * 0.2);
  const x1 = Math.max(0, x - pad);
  const y1 = Math.max(0, y - pad);
  const x2 = Math.min(imgW, x + w + pad);
  const y2 = Math.min(imgH, y + h + pad);

  // The smaller the region relative to the image, the more we want to zoom in.
  const regionArea = (x2 - x1) * (y2 - y1);
  const imgArea = imgW * imgH;
  const zoom = Math.min(3.0, Math.max(1.0, Math.sqrt(imgArea / (regionArea + 1))));

  // Pan centres the region in the viewer viewport
  const cx = (x1 + x2) / 2 - imgW / 2;
  const cy = (y1 + y2) / 2 - imgH / 2;

  return { x1, y1, x2, y2, zoom, panX: -cx * zoom, panY: -cy * zoom };
}

/**
 * Builds a human-readable observation sentence from a single ROI, tuned
 * to TB vs Normal context.
 */
function roiToObservationText(roi: XaiRoi, isTb: boolean): string {
  const loc = roi.location;
  const act = roi.activation_score.toFixed(1);
  const contrib = roi.contribution_pct.toFixed(1);

  if (isTb) {
    // Describe the radiographic feature implied by high activation
    if (roi.activation_score >= 70) {
      return `High-activation region (${act}% intensity, ${contrib}% of total model attention) detected in the ${loc} — pattern consistent with focal consolidation or infiltrate.`;
    } else if (roi.activation_score >= 40) {
      return `Moderate neural attention focus (${act}% intensity, ${contrib}% contribution) in the ${loc} — may reflect increased parenchymal opacity or texture irregularity.`;
    } else {
      return `Low-level activation (${act}% intensity, ${contrib}% contribution) noted in the ${loc} — secondary pattern without dominant radiographic feature.`;
    }
  } else {
    if (roi.activation_score >= 50) {
      return `Background neural attention (${act}% intensity, ${contrib}% contribution) observed in the ${loc} — no focal abnormality identified; pattern consistent with normal parenchymal texture.`;
    } else {
      return `Diffuse low-level activation (${act}% intensity, ${contrib}% contribution) in the ${loc} — bilateral baseline noise with no pathological focus detected.`;
    }
  }
}

export const observationService = {
  /**
   * Derives clinical observations from the actual XAI ROI payload returned
   * by the backend. Falls back to generic observations only when no XAI data
   * is present, and labels the fallback clearly.
   *
   * @param prediction  - "Tuberculosis" | "Normal" | other
   * @param xaiResults  - The full xai_results object from the backend API
   * @param imgW        - Natural image width in pixels (used for viewport calc)
   * @param imgH        - Natural image height in pixels
   */
  getObservations(
    prediction: string,
    xaiResults?: XaiResults | null,
    imgW = 224,
    imgH = 224
  ): ClinicalObservation[] {
    const isTb = (prediction || "Normal").toLowerCase().includes("tuberculosis");

    // ── Path 1: Real XAI data is available ───────────────────────────────
    if (xaiResults?.rois && xaiResults.rois.length > 0) {
      const obs: ClinicalObservation[] = xaiResults.rois.slice(0, 5).map((roi, idx) => {
        const coords = bboxToCoords(roi.bbox);
        const region = bboxToTargetRegion(roi.bbox, imgW, imgH);

        return {
          id: `obs-xai-${roi.id}`,
          text: roiToObservationText(roi, isTb),
          location: roi.location,
          // evidence & confidence derived from the actual activation metrics
          evidenceScore: Math.min(1.0, roi.activation_score / 100),
          confidence: Math.min(1.0, roi.contribution_pct / 100 + 0.5),
          coordinates: coords,
          targetRegion: region,
        };
      });

      // Add a summary observation from the XAI narrative if present
      if (xaiResults.summary) {
        const n = xaiResults.rois.length;
        obs.push({
          id: "obs-xai-summary",
          text: `Model attention map summary (${n} activation zone${n !== 1 ? "s" : ""} identified): ${xaiResults.summary}`,
          location: "Whole Lung Field",
          evidenceScore: Math.min(1.0, (xaiResults.metrics?.calibrated_confidence ?? 50) / 100),
          confidence: Math.min(1.0, (xaiResults.metrics?.calibrated_confidence ?? 50) / 100),
          coordinates: { x1: 0, y1: 0, x2: imgW, y2: imgH },
          targetRegion: { x1: 0, y1: 0, x2: imgW, y2: imgH, zoom: 1.0, panX: 0, panY: 0 },
        });
      }

      return obs;
    }

    // ── Path 2: No XAI data — return generic fallback, labelled as such ──
    if (!isTb) {
      return [
        {
          id: "obs-fallback-norm-1",
          text: "No focal high-activation regions detected by the model. Lung fields appear clear (XAI data unavailable — observation is generic).",
          location: "Bilateral Lung Fields",
          evidenceScore: 0.80,
          confidence: 0.80,
          coordinates: { x1: 20, y1: 20, x2: 204, y2: 204 },
          targetRegion: { x1: 20, y1: 20, x2: 204, y2: 204, zoom: 1.0, panX: 0, panY: 0 },
        },
        {
          id: "obs-fallback-norm-2",
          text: "Sigmoid output below decision threshold — model does not detect TB-consistent pattern (XAI data unavailable).",
          location: "Whole Lung Field",
          evidenceScore: 0.75,
          confidence: 0.75,
          coordinates: { x1: 0, y1: 0, x2: 224, y2: 224 },
          targetRegion: { x1: 0, y1: 0, x2: 224, y2: 224, zoom: 1.0, panX: 0, panY: 0 },
        },
      ];
    }

    return [
      {
        id: "obs-fallback-tb-1",
        text: "Model sigmoid output exceeds decision threshold — TB-positive classification (XAI data unavailable; exact region location cannot be determined).",
        location: "Lung Field (unlocalized)",
        evidenceScore: 0.80,
        confidence: 0.80,
        coordinates: { x1: 20, y1: 20, x2: 204, y2: 204 },
        targetRegion: { x1: 20, y1: 20, x2: 204, y2: 204, zoom: 1.0, panX: 0, panY: 0 },
      },
      {
        id: "obs-fallback-tb-2",
        text: "Clinical correlation with sputum analysis and chest CT recommended to confirm and localize the finding (XAI data unavailable).",
        location: "Whole Lung Field",
        evidenceScore: 0.75,
        confidence: 0.75,
        coordinates: { x1: 0, y1: 0, x2: 224, y2: 224 },
        targetRegion: { x1: 0, y1: 0, x2: 224, y2: 224, zoom: 1.0, panX: 0, panY: 0 },
      },
    ];
  },
};
