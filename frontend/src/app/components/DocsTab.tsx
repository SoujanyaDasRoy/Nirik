"use client";

import { Card, CardContent } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";

export function DocsTab() {
  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <Card className="border border-border bg-card rounded-lg shadow-none">
        <CardContent className="p-6 space-y-4">
          <h3 className="text-lg font-bold text-foreground">TB Diagnostic Assistant Protocol Docs</h3>
          <Separator />
          <div className="space-y-3 text-xs leading-relaxed text-muted-foreground">
            <p>
              <strong>1. Overview:</strong> This platform is designed to assist radiologists and physicians in fast batch screening of chest radiographs for signs of pulmonary tuberculosis (TB).
            </p>
            <p>
              <strong>2. Model Weights:</strong> The deep learning model is fully traced via TorchScript. It scans input pixel shapes `[1, 3, 224, 224]` and outputs a raw sigmoid confidence score representing consolidations.
            </p>
            <p>
              <strong>3. DICOM PACS Integration:</strong> The integration is simulated via mock FHIR R4 bundles and PACS node listings. This is fully extensible to real clinical RIS/PACS environments via DCMTK interfaces.
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
