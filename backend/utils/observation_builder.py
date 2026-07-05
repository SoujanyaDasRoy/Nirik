"""build_clinical_observations — converts the XAI ROI payload into a rich
clinical observation list.

This helper is intentionally pure: no DB, no model, no IO. It exists so
the frontend can render the same shape whether it came from a fresh
/predict call or was read back from the patients database (via
predictions.clinical_observations_json).

The returned dicts are JSON-serializable (lists, strings, numbers, bool)
so the predictions table can store them verbatim.

Field reference (per observation):
    id                    : ROI label, e.g. "A", "B"
    code                  : ICD-10-ish short code
    label                 : Human-readable finding label
    location              : passthrough from XAI ROI
    laterality            : "Right" | "Left" | "Bilateral"
    zone                  : "Upper" | "Middle" | "Lower" | "Pleural"
    descriptor            : "TB-specific" | "Non-TB"
    clinical_significance : "High" | "Moderate" | "Low"
    evidence_score        : 0..1
    contribution_pct      : 0..100
    activation_score      : 0..100
    severity              : "Critical" | "Marked" | "Moderate" | "Mild" | "Background"
    bbox, circle, contour, center : passthrough
    target_region         : {x1, y1, x2, y2, zoom, panX, panY} on a 224x224 grid
    narrative             : full multi-clause sentence (NEVER truncated)
    recommended_followup  : list[str]
    differential_diagnoses: list[str]
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────
# Clinical dictionaries
# ─────────────────────────────────────────────────────────────────────────

# TB-specific label/codes by zone. A real diagnosis still requires lab
# confirmation — these are the *saliency-implied* patterns.
_TB_LABELS = {
    "Upper":   ("Focal apical consolidation",     "A15.0"),
    "Middle":  ("Mid-zone parenchymal infiltrate", "A15.0"),
    "Lower":   ("Lower-zone consolidative opacity", "A15.0"),
    "Pleural": ("Pleural thickening / effusion",   "A15.6"),
}
_TB_FOLLOWUP = [
    "Sputum AFB smear microscopy x 3 (early-morning specimens)",
    "GeneXpert MTB/RIF or Truenat molecular assay",
    "Chest CT for cavitation and tree-in-bud assessment",
    "HIV co-infection testing and CD4 count",
    "Drug-susceptibility testing if AFB positive",
    "Contact tracing per NTEP / WHO guidelines",
]
_TB_DIFFERENTIALS = [
    "Typical pulmonary tuberculosis (Mycobacterium tuberculosis)",
    "Atypical mycobacterial infection (M. avium complex, M. kansasii)",
    "Fungal pneumonia (histoplasmosis, coccidioidomycosis)",
    "Cavitating lung malignancy (squamous cell carcinoma)",
    "Aspiration pneumonia with cavitation",
    "Septic emboli with cavitary lesions",
]

# Non-TB dictionaries — pattern suggested is benign, but radiology
# correlation is still required for symptomatic patients.
_NORMAL_LABELS = {
    "Upper":   ("Mild apical opacity / vascular marking", "R91.1"),
    "Middle":  ("Mid-zone parenchymal texture",           "R91.1"),
    "Lower":   ("Lower-zone vascular shadow",             "R91.1"),
    "Pleural": ("Costophrenic angle / pleural reflection", "R91.8"),
}
_NORMAL_FOLLOWUP = [
    "Clinical correlation with symptoms (cough, fever, weight loss)",
    "Repeat chest radiograph in 4-6 weeks if symptoms persist",
    "Sputum AFB smear only if TB risk factors present",
    "Consider chest CT only if clinical concern escalates",
]
_NORMAL_DIFFERENTIALS = [
    "Normal anatomical variant / vascular shadow",
    "Mild atelectasis or subsegmental collapse",
    "Early interstitial changes (non-specific)",
    "Viral lower-respiratory-tract infection",
    "Residual scarring from prior infection",
]


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────

def _parse_laterality(location: str) -> str:
    """Return "Right", "Left", or "Bilateral" from a location string."""
    loc = (location or "").lower()
    if "bilateral" in loc:
        return "Bilateral"
    if "left" in loc and "right" in loc:
        return "Bilateral"
    if "left" in loc:
        return "Left"
    if "right" in loc:
        return "Right"
    return "Bilateral"


def _parse_zone(location: str) -> str:
    """Return "Upper", "Middle", "Lower", or "Pleural" from a location string."""
    loc = (location or "").lower()
    if "upper" in loc or "apical" in loc or "apex" in loc:
        return "Upper"
    if "mid" in loc or "middle" in loc or "hilar" in loc:
        return "Middle"
    if "lower" in loc or "basal" in loc:
        return "Lower"
    if "pleural" in loc or "costophrenic" in loc:
        return "Pleural"
    return "Middle"


def _severity_from_activation(act: float) -> str:
    if act >= 90.0:
        return "Critical"
    if act >= 70.0:
        return "Marked"
    if act >= 50.0:
        return "Moderate"
    if act >= 30.0:
        return "Mild"
    return "Background"


def _significance(act: float, contrib: float) -> str:
    """Combine activation and contribution into a 3-level significance tag."""
    score = act * 0.6 + contrib * 0.4
    if score >= 60.0:
        return "High"
    if score >= 30.0:
        return "Moderate"
    return "Low"


def _evidence_score(act: float, contrib: float) -> float:
    """0..1 evidence score: act/100 weighted by contribution share."""
    act_n = max(0.0, min(1.0, act / 100.0))
    contrib_n = max(0.0, min(1.0, contrib / 100.0))
    # Weight activation more heavily but reward high contribution.
    return round(min(1.0, act_n * 0.7 + contrib_n * 0.3), 4)


def _target_region(bbox, img_w: int = 224, img_h: int = 224) -> dict:
    """Compute a 224-grid (or custom image) pan/zoom hint from [x, y, w, h].

    Mirrors the frontend's bboxToTargetRegion so the DicomViewer
    normalization (coords / 224) just works.
    """
    try:
        x, y, w, h = [float(v) for v in bbox[:4]]
    except (TypeError, ValueError):
        return {
            "x1": 0, "y1": 0, "x2": img_w, "y2": img_h,
            "zoom": 1.0, "panX": 0.0, "panY": 0.0,
        }
    pad = round(min(w, h) * 0.2)
    x1 = max(0.0, x - pad)
    y1 = max(0.0, y - pad)
    x2 = float(min(img_w, x + w + pad))
    y2 = float(min(img_h, y + h + pad))

    region_area = max(1.0, (x2 - x1) * (y2 - y1))
    img_area = float(img_w * img_h)
    zoom = min(3.0, max(1.0, (img_area / (region_area + 1.0)) ** 0.5))

    cx = (x1 + x2) / 2.0 - img_w / 2.0
    cy = (y1 + y2) / 2.0 - img_h / 2.0
    return {
        "x1": round(x1, 2),
        "y1": round(y1, 2),
        "x2": round(x2, 2),
        "y2": round(y2, 2),
        "zoom": round(zoom, 3),
        "panX": round(-cx * zoom, 2),
        "panY": round(-cy * zoom, 2),
    }


def _build_narrative(
    *,
    region_id: str,
    location: str,
    is_tb: bool,
    confidence: float,
    activation: float,
    contribution: float,
    severity: str,
    laterality: str,
    zone: str,
) -> str:
    """Return a full multi-clause narrative sentence.

    The output NEVER contains "..." or any truncation marker. Tests assert
    that the result ends with a period and exceeds a minimum length.
    """
    act_s = f"{activation:.1f}"
    contrib_s = f"{contribution:.1f}"
    conf_s = f"{confidence * 100:.1f}"

    if is_tb:
        # Multi-clause clinical narrative for TB-positive regions.
        return (
            f"Saliency analysis on Region {region_id} ({act_s}% peak activation, "
            f"{contrib_s}% of total model attention) localises a {severity.lower()} "
            f"high-density opacity within the {location}; the gradient distribution "
            f"and spatial weighting are highly consistent with active pulmonary "
            f"tuberculosis, with secondary {laterality.lower()} {zone.lower()}-zone "
            f"involvement (overall model TB probability {conf_s}%). This pattern "
            f"warrants confirmatory sputum acid-fast bacilli smear microscopy, "
            f"GeneXpert MTB/RIF or Truenat molecular assay, and clinical correlation "
            f"with constitutional symptoms (cough >= 2 weeks, evening fever, night "
            f"sweats, weight loss) per WHO and NTEP screening guidelines; chest CT "
            f"is recommended for cavitation and tree-in-bud assessment if sputum "
            f"studies are positive."
        )
    # Non-TB / normal narrative.
    return (
        f"Saliency analysis on Region {region_id} ({act_s}% peak activation, "
        f"{contrib_s}% of total model attention) shows a {severity.lower()} "
        f"diffuse parenchymal gradient within the {location}, with no focal "
        f"consolidation, cavitation, or asymmetric opacity; the model interprets "
        f"this distribution as background neural attention consistent with normal "
        f"aerated lung parenchyma (overall model TB probability {conf_s}%). "
        f"Although no computable feature suggests active tuberculosis, this does "
        f"not exclude latent infection, very early disease, or non-tuberculous "
        f"respiratory pathology; clinical correlation and follow-up imaging are "
        f"advised for symptomatic patients or high-risk exposures."
    )


# ─────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────

def build_clinical_observations(
    xai_results: dict | None,
    *,
    is_tb: bool,
    confidence: float,
    img_w: int = 224,
    img_h: int = 224,
) -> list[dict]:
    """Build the rich clinical_observations list from an xai_results payload.

    Args:
        xai_results: The dict produced by `core.inference.compute_xai_payload`
            (or its demo-mode equivalent). Must contain a `rois` list.
        is_tb: Top-level prediction from the classifier.
        confidence: Raw classifier sigmoid output (0..1).
        img_w, img_h: Natural image size used for the target_region grid.
            Defaults match DenseNet-121 (224x224) — DicomViewer normalizes
            focus-region coordinates by /224.

    Returns:
        A list of observation dicts, one per ROI, ordered by descending
        contribution. Empty list if no ROIs are present.
    """
    if not xai_results:
        return []
    rois = xai_results.get("rois") or []
    if not rois:
        return []

    out: list[dict] = []
    for roi in rois:
        try:
            region_id = str(roi.get("id") or "?").strip() or "?"
            location = str(roi.get("location") or "Unspecified Lung Field")
            activation = float(roi.get("activation_score") or 0.0)
            contribution = float(roi.get("contribution_pct") or 0.0)
        except (TypeError, ValueError):
            continue

        laterality = _parse_laterality(location)
        zone = _parse_zone(location)

        # Pick label/code by TB vs non-TB.
        if is_tb:
            label, code = _TB_LABELS.get(zone, ("Consolidative opacity", "A15.0"))
            followup = list(_TB_FOLLOWUP)
            differentials = list(_TB_DIFFERENTIALS)
            descriptor = "TB-specific"
        else:
            label, code = _NORMAL_LABELS.get(zone, ("Parenchymal shadow", "R91.1"))
            followup = list(_NORMAL_FOLLOWUP)
            differentials = list(_NORMAL_DIFFERENTIALS)
            descriptor = "Non-TB"

        severity = _severity_from_activation(activation)
        significance = _significance(activation, contribution)
        evidence = _evidence_score(activation, contribution)

        # Re-narrate the contribution fraction in absolute terms when it
        # exceeds 100% (rare but possible if a prior run summed oddly).
        contribution_for_text = min(contribution, 100.0)
        narrative = _build_narrative(
            region_id=region_id,
            location=location,
            is_tb=is_tb,
            confidence=confidence,
            activation=activation,
            contribution=contribution_for_text,
            severity=severity,
            laterality=laterality,
            zone=zone,
        )

        # Safe passthrough of geometry fields. Some paths may have ints.
        try:
            bbox = [int(round(float(v))) for v in (roi.get("bbox") or [0, 0, 0, 0])]
        except (TypeError, ValueError):
            bbox = [0, 0, 0, 0]
        try:
            circle = [int(round(float(v))) for v in (roi.get("circle") or [0, 0, 0])]
        except (TypeError, ValueError):
            circle = [0, 0, 0]
        try:
            contour_raw = roi.get("contour") or []
            contour = [[int(round(float(x))), int(round(float(y)))] for (x, y) in contour_raw]
        except (TypeError, ValueError):
            contour = []
        try:
            center = [float(v) for v in (roi.get("center") or [0.5, 0.5])]
        except (TypeError, ValueError):
            center = [0.5, 0.5]

        out.append({
            "id": region_id,
            "code": code,
            "label": label,
            "location": location,
            "laterality": laterality,
            "zone": zone,
            "descriptor": descriptor,
            "clinical_significance": significance,
            "evidence_score": evidence,
            "contribution_pct": round(contribution, 1),
            "activation_score": round(activation, 1),
            "severity": severity,
            "bbox": bbox,
            "circle": circle,
            "contour": contour,
            "center": center,
            "target_region": _target_region(bbox, img_w=img_w, img_h=img_h),
            "narrative": narrative,
            "recommended_followup": followup,
            "differential_diagnoses": differentials,
        })

    return out
