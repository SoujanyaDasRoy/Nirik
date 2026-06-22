import { ImageQualityMetrics } from "./imageQualityService";
import { ClinicalObservation } from "./observationService";

export interface ClinicalReportPayload {
  patientId: string;
  patientName: string;
  ageSex: string;
  modality: string;
  studyDate: string;
  imageQuality: ImageQualityMetrics;
  condition: string;
  confidence: number;
  riskLevel: string;
  observations: ClinicalObservation[];
  reviewStatus: string;
  reviewComments: string;
  reviewerName: string;
  notes: string;
  timestamp: string;
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
}

export const reportService = {
  buildReport(
    metadata: any,
    quality: ImageQualityMetrics,
    prediction: string,
    confidence: number,
    riskLevel: string,
    observations: ClinicalObservation[],
    review: { status: string; comments: string; signature: string },
    notes: string,
    xai_results?: any
  ): ClinicalReportPayload {
    return {
      patientId: metadata?.patient_id || "UNKNOWN",
      patientName: metadata?.patient_name || "Anonymous Patient",
      ageSex: `${metadata?.patient_age || "N/A"} / ${metadata?.patient_sex || "N/A"}`,
      modality: metadata?.modality || "CR (Computed Radiography)",
      studyDate: metadata?.study_date || new Date().toLocaleDateString(),
      imageQuality: quality,
      condition: prediction,
      confidence: confidence,
      riskLevel: riskLevel,
      observations: observations,
      reviewStatus: review.status || "Pending Review",
      reviewComments: review.comments || "",
      reviewerName: review.signature || "Not Signed",
      notes: notes,
      timestamp: new Date().toLocaleString(),
      xai_results: xai_results
    };
  }
};
