"""Unit tests for backend.utils.observation_builder.

These tests are intentionally lightweight: they import only the pure helper
module and avoid loading any ML / Keras / torch dependencies. This is why
the helper lives in `utils/` next to `attention_region` rather than inside
`core/`.

Coverage:
  * Full structure is populated (no truncation) for TB and Normal flows
  * Empty / None xai_results -> empty list
  * Laterality and zone are parsed from the location string
  * target_region sits on the 224-grid expected by DicomViewer
  * Narrative ends with a period, has no "..." or "[truncated]"
"""
import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from utils.observation_builder import build_clinical_observations


TB_XAI = {
    "rois": [
        {
            "id": "A",
            "activation_score": 87.3,
            "contribution_pct": 64.5,
            "location": "Right Upper Lung Zone",
            "bbox": [60, 30, 80, 60],
            "circle": [100, 60, 40],
            "contour": [[60, 30], [140, 30], [140, 90], [60, 90]],
            "center": [0.45, 0.27],
        },
        {
            "id": "B",
            "activation_score": 42.1,
            "contribution_pct": 22.0,
            "location": "Left Mid Lung Zone",
            "bbox": [110, 95, 60, 45],
            "circle": [140, 117, 28],
            "contour": [[110, 95], [170, 95], [170, 140], [110, 140]],
            "center": [0.63, 0.52],
        },
    ],
    "summary": "TB-suggestive saliency.",
    "ranking": [],
    "metrics": {"tb_probability": 91.0, "calibrated_confidence": 93.0,
                "reliability": "High", "uncertainty": "Low"},
}

NORMAL_XAI = {
    "rois": [
        {
            "id": "A",
            "activation_score": 18.0,
            "contribution_pct": 60.0,
            "location": "Left Lower Lung Zone",
            "bbox": [120, 140, 60, 50],
            "circle": [150, 165, 30],
            "contour": [[120, 140], [180, 140], [180, 190], [120, 190]],
            "center": [0.67, 0.74],
        },
    ],
    "summary": "Background activation only.",
    "ranking": [],
    "metrics": {"tb_probability": 22.0, "calibrated_confidence": 80.0,
                "reliability": "High", "uncertainty": "Low"},
}

REQUIRED_KEYS = {
    "id", "code", "label", "location", "laterality", "zone", "descriptor",
    "clinical_significance", "evidence_score", "contribution_pct",
    "activation_score", "severity", "bbox", "circle", "contour", "center",
    "target_region", "narrative", "recommended_followup",
    "differential_diagnoses",
}


def _assert_narrative_shape(narrative: str):
    """Hard guards against silent truncation."""
    assert isinstance(narrative, str) and len(narrative) > 80, (
        f"narrative too short or not a string: {narrative!r}"
    )
    assert narrative.rstrip().endswith("."), (
        f"narrative must end with a period: {narrative!r}"
    )
    assert "..." not in narrative, f"narrative must not contain '...': {narrative!r}"
    assert "[truncated]" not in narrative.lower(), (
        f"narrative must not be marked truncated: {narrative!r}"
    )


def test_tb_full_structure_no_truncation():
    obs = build_clinical_observations(
        TB_XAI, is_tb=True, confidence=0.91
    )
    assert isinstance(obs, list)
    assert len(obs) == 2, f"expected 2 observations, got {len(obs)}"

    for o in obs:
        # All required keys present
        missing = REQUIRED_KEYS - set(o.keys())
        assert not missing, f"missing keys: {missing}"

        # TB-specific descriptor and follow-up list
        assert o["descriptor"] == "TB-specific", o["descriptor"]
        assert o["code"].startswith("A15"), o["code"]
        assert len(o["recommended_followup"]) >= 2
        assert len(o["differential_diagnoses"]) >= 2

        # Bbox/circle/contour/center passthrough (numeric types)
        assert len(o["bbox"]) == 4
        assert len(o["circle"]) == 3
        assert isinstance(o["contour"], list) and len(o["contour"]) >= 4
        assert len(o["center"]) == 2

        # Evidence score is in [0, 1]
        assert 0.0 <= o["evidence_score"] <= 1.0

        # target_region is on the 224-grid (DicomViewer normalizes by /224)
        tr = o["target_region"]
        assert 0 <= tr["x1"] <= 224 and 0 <= tr["x2"] <= 224
        assert 0 <= tr["y1"] <= 224 and 0 <= tr["y2"] <= 224
        assert 1.0 <= tr["zoom"] <= 3.0
        assert isinstance(tr["panX"], (int, float))
        assert isinstance(tr["panY"], (int, float))

        # Narrative is a full sentence, never truncated
        _assert_narrative_shape(o["narrative"])


def test_normal_full_structure_no_truncation():
    obs = build_clinical_observations(
        NORMAL_XAI, is_tb=False, confidence=0.22
    )
    assert len(obs) == 1
    o = obs[0]
    assert o["descriptor"] == "Non-TB", o["descriptor"]
    assert o["code"].startswith("R91"), o["code"]
    assert len(o["recommended_followup"]) >= 2
    assert len(o["differential_diagnoses"]) >= 2
    # Even the "background" severity bucket must produce a full narrative
    _assert_narrative_shape(o["narrative"])


def test_empty_xai_returns_empty_list():
    assert build_clinical_observations(
        {"rois": []}, is_tb=False, confidence=0.0
    ) == []
    assert build_clinical_observations(None, is_tb=True, confidence=0.9) == []
    # Missing "rois" key entirely
    assert build_clinical_observations({}, is_tb=True, confidence=0.9) == []


def test_laterality_and_zone_parsed_from_location():
    obs = build_clinical_observations(
        TB_XAI, is_tb=True, confidence=0.9
    )
    by_id = {o["id"]: o for o in obs}
    assert by_id["A"]["laterality"] == "Right"
    assert by_id["A"]["zone"] == "Upper"
    assert by_id["B"]["laterality"] == "Left"
    assert by_id["B"]["zone"] == "Middle"


def test_severity_and_significance_bands():
    # Region A: act=87.3 contrib=64.5 -> severity "Marked", significance "High"
    obs = build_clinical_observations(
        TB_XAI, is_tb=True, confidence=0.9
    )
    by_id = {o["id"]: o for o in obs}
    assert by_id["A"]["severity"] == "Marked"
    assert by_id["A"]["clinical_significance"] == "High"
    # Region B: act=42.1 contrib=22.0 -> severity "Mild", significance "Moderate"
    assert by_id["B"]["severity"] == "Mild"
    assert by_id["B"]["clinical_significance"] == "Moderate"


def test_observation_id_passthrough():
    obs = build_clinical_observations(
        TB_XAI, is_tb=True, confidence=0.9
    )
    ids = [o["id"] for o in obs]
    assert ids == ["A", "B"]


def test_observation_preserves_contribution_and_activation_pct():
    obs = build_clinical_observations(
        TB_XAI, is_tb=True, confidence=0.9
    )
    by_id = {o["id"]: o for o in obs}
    # Values rounded to 1 dp inside the builder
    assert by_id["A"]["activation_score"] == 87.3
    assert by_id["A"]["contribution_pct"] == 64.5


def test_malformed_roi_does_not_crash():
    bad = {
        "rois": [
            {"id": "A", "activation_score": "not a number",
             "contribution_pct": 10, "location": "Right Upper Lung Zone",
             "bbox": [0, 0, 0, 0], "circle": [0, 0, 0],
             "contour": [], "center": [0, 0]},
        ]
    }
    obs = build_clinical_observations(bad, is_tb=True, confidence=0.5)
    # Builder should drop the bad ROI rather than raising
    assert obs == []


def test_returns_json_serializable():
    """Round-trip through json.dumps; this is what the DB column needs."""
    import json
    obs = build_clinical_observations(
        TB_XAI, is_tb=True, confidence=0.9
    )
    blob = json.dumps(obs)
    roundtrip = json.loads(blob)
    assert len(roundtrip) == 2
    assert roundtrip[0]["id"] == "A"
    assert isinstance(roundtrip[0]["target_region"], dict)
    assert isinstance(roundtrip[0]["recommended_followup"], list)
    assert isinstance(roundtrip[0]["differential_diagnoses"], list)
