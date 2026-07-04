"""Derive a clinical attention_region string from XAI ROIs.

The /predict endpoint historically hard-coded "right apical" for any TB case
and "clear" for any Normal case. That made the UI evidence cards useless —
they always pointed to the right apex regardless of what the model actually
looked at. This helper surfaces the top-contribution ROI's anatomical
location instead, with a fallback to the legacy defaults when no ROIs are
available (e.g. demo mode, prior to XAI rollout).
"""


def derive_attention_region(rois, is_tb: bool) -> str:
    """Return the human-readable location the model focused on.

    Args:
        rois: list of ROI dicts from xai_results.rois, each with
            ``location`` (str) and ``contribution_pct`` (float).
        is_tb: whether the prediction is tuberculosis.

    Returns:
        Anatomical zone string for the top-contribution ROI, or the
        legacy fallback ("right apical" / "clear") when ROIs are absent.
    """
    if rois:
        top = max(rois, key=lambda r: r.get("contribution_pct", 0.0))
        loc = top.get("location")
        if loc:
            return loc
    return "right apical" if is_tb else "clear"