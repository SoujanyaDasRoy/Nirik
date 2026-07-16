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

    // Use the raw class-relative probability (directly matches what the LLM receives/calculates)
    const displayConfidence = isTb ? confidence : (1.0 - confidence);

    return {
      condition: isTb ? "Tuberculosis" : "Normal",
      confidence: displayConfidence,
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
