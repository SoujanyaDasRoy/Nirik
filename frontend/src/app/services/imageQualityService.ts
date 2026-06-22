export interface ImageQualityMetrics {
  exposure: string;
  coverage: string;
  resolution: string;
  rotation: string;
  suitableForAi: boolean;
  qualityScore: number;
}

export const imageQualityService = {
  assessQuality(filename: string): ImageQualityMetrics {
    // Generate deterministic quality indicators based on filename characters
    let hash = 0;
    for (let i = 0; i < filename.length; i++) {
      hash += filename.charCodeAt(i);
    }

    const hasRotation = hash % 3 === 0;
    const isUnderexposed = hash % 7 === 0;
    const score = 85 + (hash % 13); // Score between 85% and 98%

    return {
      exposure: isUnderexposed ? "Underexposed" : "Adequate Exposure",
      coverage: "Full Lung Coverage",
      resolution: "2048 x 2048 pixels",
      rotation: hasRotation ? "Mild Patient Rotation" : "No Rotation",
      suitableForAi: score > 80,
      qualityScore: score
    };
  }
};
