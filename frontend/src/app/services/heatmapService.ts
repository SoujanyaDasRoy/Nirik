export const heatmapService = {
  // NOTE: `calculateCoverage` and `getAttentionRegion` previously fabricated
  // clinically-specific values in the browser (a hash of the base64 string
  // presented as an "activation coverage %", and a hardcoded
  // "Right Upper Lung Zone" for every TB case). Presenting invented numbers /
  // regions as if they were measured is a clinical-integrity problem, so they
  // have been neutralized to return honest "not available" values. Real ROI /
  // region data comes from the backend (`xai_results.rois` / `quadrant_analysis`);
  // callers should route to that data rather than to these helpers.
  calculateCoverage(_heatmapBase64: string): number | null {
    // Activation-area coverage is not computed on the client. Decoding the
    // Grad-CAM PNG and thresholding pixels is out of scope here, so we return
    // null instead of a fabricated percentage. Callers should render "—".
    return null;
  },

  getAttentionRegion(_prediction: string): string {
    // The dominant attention region is not derived on the client. Use the
    // backend-provided ROI / quadrant analysis instead of a hardcoded region.
    return "Not available";
  },

  getExplanationConfidence(confidence: number): "Low" | "Moderate" | "High" {
    // Legitimate derived label: buckets a real confidence number.
    if (confidence > 0.90) return "High";
    if (confidence > 0.60) return "Moderate";
    return "Low";
  }
};
