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

export const observationService = {
  getObservations(prediction: string): ClinicalObservation[] {
    const cond = prediction || "Normal";
    if (cond === "Normal" || cond.toLowerCase().includes("normal")) {
      return [
        {
          id: "obs-norm-1",
          text: "Clear lung fields bilaterally",
          location: "Bilateral Lung Fields",
          evidenceScore: 0.95,
          confidence: 0.98,
          coordinates: { x1: 50, y1: 50, x2: 250, y2: 250 },
          targetRegion: { x1: 50, y1: 50, x2: 250, y2: 250, zoom: 1.0, panX: 0, panY: 0 }
        },
        {
          id: "obs-norm-2",
          text: "Normal cardiothoracic ratio",
          location: "Mediastinum & Heart",
          evidenceScore: 0.92,
          confidence: 0.96,
          coordinates: { x1: 100, y1: 120, x2: 200, y2: 220 },
          targetRegion: { x1: 100, y1: 120, x2: 200, y2: 220, zoom: 1.3, panX: -20, panY: 20 }
        },
        {
          id: "obs-norm-3",
          text: "Costophrenic angles are sharp and clear",
          location: "Bilateral Plural Spaces",
          evidenceScore: 0.89,
          confidence: 0.94,
          coordinates: { x1: 50, y1: 220, x2: 250, y2: 280 },
          targetRegion: { x1: 50, y1: 220, x2: 250, y2: 280, zoom: 1.5, panX: 0, panY: 50 }
        }
      ];
    }

    return [
      {
        id: "obs-dyn-1",
        text: `Increased opacity consistent with focal ${cond} consolidation`,
        location: "Right Upper Lung Zone",
        evidenceScore: 0.86,
        confidence: 0.91,
        coordinates: { x1: 130, y1: 40, x2: 210, y2: 110 },
        targetRegion: { x1: 130, y1: 40, x2: 210, y2: 110, zoom: 1.8, panX: -60, panY: 60 }
      },
      {
        id: "obs-dyn-2",
        text: `Focal infiltrative patterns identified indicative of ${cond}`,
        location: "Apex of Right Lung",
        evidenceScore: 0.82,
        confidence: 0.88,
        coordinates: { x1: 140, y1: 50, x2: 190, y2: 100 },
        targetRegion: { x1: 140, y1: 50, x2: 190, y2: 100, zoom: 2.2, panX: -80, panY: 80 }
      },
      {
        id: "obs-dyn-3",
        text: `Asymmetric markings detected in areas of model focus for ${cond}`,
        location: "Mid Right Lung Zone",
        evidenceScore: 0.75,
        confidence: 0.84,
        coordinates: { x1: 40, y1: 50, x2: 200, y2: 180 },
        targetRegion: { x1: 40, y1: 50, x2: 200, y2: 180, zoom: 1.3, panX: 20, panY: 20 }
      },
      {
        id: "obs-dyn-4",
        text: `Primary activation focus highlighted by Grad-CAM++ for ${cond}`,
        location: "Right Upper Lung Zone",
        evidenceScore: 0.93,
        confidence: 0.95,
        coordinates: { x1: 130, y1: 30, x2: 220, y2: 120 },
        targetRegion: { x1: 130, y1: 30, x2: 220, y2: 120, zoom: 1.6, panX: -50, panY: 50 }
      }
    ];
  }
};
