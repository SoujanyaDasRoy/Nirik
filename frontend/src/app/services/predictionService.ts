export interface DiagnosisResult {
  condition: string;
  confidence: number;
  riskLevel: "Low" | "Medium" | "High";
  isBorderline?: boolean;
  rawConfidence?: number;
}

export const predictionService = {
  getDiagnosis(prediction: string, confidence: number, thresholdUsed: number = 0.60): DiagnosisResult {
    const isTb = prediction === "Tuberculosis" || prediction.toLowerCase().includes("tuberculosis");
    const riskLevel: "Low" | "Medium" | "High" = isTb ? "High" : "Low";

    // Mathematically calibrate confidence relative to decision threshold
    let calibrated = 0.50;
    if (isTb) {
      if (thresholdUsed < 1.0) {
        calibrated = 0.50 + 0.50 * (confidence - thresholdUsed) / (1.0 - thresholdUsed);
      } else {
        calibrated = 1.0;
      }
    } else {
      if (thresholdUsed > 0.0) {
        calibrated = 0.50 + 0.50 * (thresholdUsed - confidence) / thresholdUsed;
      } else {
        calibrated = 1.0;
      }
    }
    
    // Clamp to [0.50, 1.00]
    calibrated = Math.max(0.50, Math.min(1.00, calibrated));

    return {
      condition: isTb ? "Tuberculosis" : "Normal",
      confidence: calibrated,
      riskLevel,
      isBorderline: false,
      rawConfidence: confidence
    };
  },

  getDistribution(prediction: string, confidence: number): { condition: string; probability: number }[] {
    const primary = confidence;
    const secondary = (1.0 - confidence) * 0.75;
    const tertiary = 1.0 - primary - secondary;

    const cond = prediction || "Normal";
    if (cond === "Normal" || cond.toLowerCase().includes("normal")) {
      return [
        { condition: "Normal", probability: primary },
        { condition: "Infectious Pathology", probability: secondary },
        { condition: "Other Abnormality", probability: tertiary }
      ];
    } else {
      return [
        { condition: cond, probability: primary },
        { condition: "Normal", probability: secondary },
        { condition: "Other Abnormality", probability: tertiary }
      ];
    }
  }
};
