export interface DiagnosisResult {
  condition: string;
  confidence: number;
  riskLevel: "Low" | "Medium" | "High";
}

export const predictionService = {
  getDiagnosis(prediction: string, confidence: number): DiagnosisResult {
    // Dynamically map diagnosis to condition-agnostic structure
    const confidencePct = confidence * 100;
    let riskLevel: "Low" | "Medium" | "High" = "Low";
    if (confidencePct > 75) riskLevel = "High";
    else if (confidencePct > 45) riskLevel = "Medium";

    return {
      condition: prediction === "Normal" ? "Normal (No active pathology)" : prediction,
      confidence: confidence,
      riskLevel
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
