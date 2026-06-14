export const heatmapService = {
  calculateCoverage(heatmapBase64: string): number {
    if (!heatmapBase64) return 0;
    // Mock calculating activation area coverage ratio from the Grad-CAM matrix
    // In production, this decodes the base64 png, counts pixels above threshold, and returns ratio.
    // We return a calibrated mock percentage depending on the heatmap string hash to remain deterministic.
    let hash = 0;
    for (let i = 0; i < Math.min(100, heatmapBase64.length); i++) {
      hash += heatmapBase64.charCodeAt(i);
    }
    return 10 + (hash % 15); // Returns deterministic coverage between 10% and 25%
  },

  getAttentionRegion(prediction: string): string {
    if (prediction === "Normal") return "None (Homogeneous)";
    return "Right Upper Lung Zone"; // Focal consolidation focus point
  },

  getExplanationConfidence(confidence: number): "Low" | "Moderate" | "High" {
    if (confidence > 0.90) return "High";
    if (confidence > 0.60) return "Moderate";
    return "Low";
  }
};
