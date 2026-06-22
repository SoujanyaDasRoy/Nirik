"use client";

import { Card, CardContent } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { Badge } from "@/components/ui/badge";

export function DocsTab() {
  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <Card className="border border-border bg-card rounded-lg shadow-none">
        <CardContent className="p-6 space-y-5">
          <div className="flex items-start justify-between">
            <h3 className="text-lg font-bold text-foreground">Nirikhshon — System Reference</h3>
            <Badge variant="outline" className="text-[10px] uppercase font-bold">Research Use Only</Badge>
          </div>
          <Separator />
          <div className="space-y-5 text-xs leading-relaxed text-muted-foreground">
            <div className="space-y-1.5">
              <p className="font-bold text-foreground text-sm">1. What This System Does</p>
              <p>
                Nirikhshon is a TB chest X-ray screening assistant. It takes a chest radiograph (PNG, JPEG, or DICOM),
                runs it through a trained <strong className="text-foreground">DenseNet-121</strong> convolutional neural network,
                and returns a binary classification — <em>Tuberculosis</em> or <em>Normal</em> — along with a
                confidence score and a Grad-CAM heatmap showing which image regions most influenced the decision.
              </p>
            </div>

            <div className="space-y-1.5">
              <p className="font-bold text-foreground text-sm">2. Model Architecture</p>
              <p>
                The model is a <strong className="text-foreground">DenseNet-121</strong> pretrained on ImageNet and
                fine-tuned on a dataset of chest radiographs annotated for pulmonary tuberculosis. Input images are
                resized to <code className="text-primary bg-primary/5 px-1 rounded">224 × 224</code> pixels and
                normalised to ImageNet mean/std. The final sigmoid output is compared against a threshold
                (default: 0.50) to produce the binary label.
              </p>
            </div>

            <div className="space-y-1.5">
              <p className="font-bold text-foreground text-sm">3. Grad-CAM Explainability</p>
              <p>
                Gradient-weighted Class Activation Maps (Grad-CAM) are computed by backpropagating the class score
                through the final convolutional block and weighting each feature map channel by its gradient. The
                resulting heatmap is overlaid on the original image using a Jet colormap
                (blue → yellow → red for low → high activation).
              </p>
            </div>

            <div className="space-y-1.5">
              <p className="font-bold text-foreground text-sm">4. Image Acceptance Criteria</p>
              <p>
                The system accepts standard chest PA (posterior-anterior) or AP (anterior-posterior) radiographs.
                DICOM files are decoded server-side and normalised to an 8-bit PNG for inference. Non-X-ray images
                (photographs, CT slices, MRI, etc.) will be rejected by the image quality checker. Minimum
                recommended resolution is <strong className="text-foreground">512 × 512</strong> pixels.
              </p>
            </div>

            <div className="space-y-1.5">
              <p className="font-bold text-foreground text-sm">5. Scope Limitations</p>
              <p>
                This is an <strong className="text-foreground">academic prototype</strong>. It is not a certified
                medical device and must not be used as the sole basis for any clinical decision. The patient registry
                uses simulated demo records. PACS integration, HL7 FHIR connections, and multi-site user management
                are not implemented.
              </p>
            </div>

            <div className="space-y-1.5">
              <p className="font-bold text-foreground text-sm">6. Tech Stack</p>
              <div className="flex flex-wrap gap-2 pt-1">
                {[
                  "Next.js 14 (Frontend)",
                  "Flask 3 (API Backend)",
                  "TensorFlow / Keras",
                  "DenseNet-121",
                  "Grad-CAM",
                  "SQLite (Database)",
                  "jsPDF (Report Export)",
                  "Web Speech API (Dictation)",
                ].map(t => (
                  <span key={t} className="px-2 py-1 rounded-full border border-border bg-muted/30 text-[10px] font-semibold text-foreground">{t}</span>
                ))}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
