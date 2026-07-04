"""Unit tests for backend.utils.attention_region.

Regression coverage for Bug #2 (hard-coded attention_region). These tests
import only the lightweight helper module — they do NOT load app.py or any
heavy ML dependencies, so they run on any environment with plain Python.
"""
import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from utils.attention_region import derive_attention_region


def test_top_roi_returned_for_tb():
    rois = [
        {"location": "Left Mid Lung Zone", "contribution_pct": 18.0},
        {"location": "Right Upper Lung Zone", "contribution_pct": 82.0},
    ]
    assert derive_attention_region(rois, is_tb=True) == "Right Upper Lung Zone"


def test_top_roi_returned_for_normal():
    rois = [{"location": "Left Lower Lung Zone", "contribution_pct": 100.0}]
    assert derive_attention_region(rois, is_tb=False) == "Left Lower Lung Zone"


def test_tb_no_rois_falls_back_to_right_apical():
    assert derive_attention_region([], is_tb=True) == "right apical"


def test_normal_no_rois_falls_back_to_clear():
    assert derive_attention_region([], is_tb=False) == "clear"


def test_none_rois_handled():
    assert derive_attention_region(None, is_tb=True) == "right apical"
    assert derive_attention_region(None, is_tb=False) == "clear"


def test_top_roi_picked_by_contribution_not_list_order():
    rois = [
        {"location": "Right Lower Lung Zone", "contribution_pct": 25.0},
        {"location": "Right Upper Lung Zone", "contribution_pct": 75.0},
    ]
    assert derive_attention_region(rois, is_tb=True) == "Right Upper Lung Zone"


def test_roi_missing_location_falls_back():
    rois = [{"contribution_pct": 90.0}]  # no "location" key
    assert derive_attention_region(rois, is_tb=True) == "right apical"


def test_legacy_hardcoded_values_not_returned_when_rois_present():
    """The bug: app.py returned literal "right apical" for every TB.
    Once ROIs are available, the helper must NOT echo that literal
    unless it's also the real top-ROI location.
    """
    rois = [{"location": "Left Mid Lung Zone", "contribution_pct": 100.0}]
    result = derive_attention_region(rois, is_tb=True)
    assert result != "right apical"
    assert result == "Left Mid Lung Zone"