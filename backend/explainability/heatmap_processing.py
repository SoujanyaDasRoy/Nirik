import numpy as np
import cv2

def calibrate_confidence(prob: float, threshold: float, is_tb: bool) -> float:
    """Mathematically calibrate confidence relative to dynamic threshold."""
    if is_tb:
        calibrated = 0.50 + 0.50 * (prob - threshold) / (1.0 - threshold) if threshold < 1.0 else 1.0
    else:
        calibrated = 0.50 + 0.50 * (threshold - prob) / threshold if threshold > 0.0 else 1.0
    return max(0.50, min(1.00, calibrated))


def validate_explainability(raw_heatmap: np.ndarray, lung_mask: np.ndarray, method_name: str) -> dict:
    """
    Validate explainability heatmap against lung mask.
    Input: raw_heatmap (normalized [0,1]), lung_mask (binary 0/1), method_name.
    Output: dict with status, reason, metrics.
    """
    # Ensure inputs are numpy arrays
    if raw_heatmap is None or lung_mask is None:
        return {
            "status": "Unavailable",
            "reason": "Missing heatmap or lung mask",
            "metrics": {}
        }
    # Ensure same shape
    if raw_heatmap.shape != lung_mask.shape:
        # Resize lung_mask to match heatmap
        lung_mask = cv2.resize(lung_mask.astype(np.uint8), (raw_heatmap.shape[1], raw_heatmap.shape[0]), interpolation=cv2.INTER_NEAREST)
    # Convert lung_mask to boolean
    lung_mask_bool = lung_mask > 0.5
    total_activation = np.sum(raw_heatmap)
    if total_activation == 0:
        activation_inside = 0.0
    else:
        activation_inside = np.sum(raw_heatmap * lung_mask_bool)
    activation_outside = total_activation - activation_inside
    activation_overlap_ratio = activation_inside / total_activation if total_activation > 0 else 0.0
    outside_lung_percentage = (activation_outside / total_activation * 100) if total_activation > 0 else 0.0
    lung_coverage = (activation_inside / np.sum(lung_mask_bool)) * 100 if np.sum(lung_mask_bool) > 0 else 0.0
    activation_density = activation_inside / np.sum(lung_mask_bool) if np.sum(lung_mask_bool) > 0 else 0.0
    # Determine status
    if np.sum(lung_mask_bool) == 0:
        status = "Unavailable"
        reason = "No lung segmentation available."
    elif activation_overlap_ratio >= 0.80 and outside_lung_percentage <= 20.0:
        status = "Valid"
        reason = "High activation overlap with lungs and low outside activation."
    elif activation_overlap_ratio >= 0.60 and outside_lung_percentage <= 40.0:
        status = "Questionable"
        reason = "Moderate activation overlap with lungs moderate outside activation."
    else:
        status = "Invalid"
        reason = "Low activation overlap with lungs or high outside activation."
    return {
        "status": status,
        "reason": reason,
        "metrics": {
            "activation_overlap_ratio": round(activation_overlap_ratio * 100, 1),  # percentage
            "outside_lung_activation_percentage": round(outside_lung_percentage, 1),
            "lung_coverage_percentage": round(lung_coverage, 1),
            "activation_density": round(activation_density, 4)
        }
    }


def extract_evidence(is_tb: bool, prob: float, raw_map_np: np.ndarray, unet_mask: np.ndarray, heatmap_blurred: np.ndarray, validation=None, quadrant_analysis=None, threshold: float = 0.5) -> dict:
    """
    Extract structured evidence from prediction and explainability outputs.
    Returns a dictionary suitable for inclusion in API response under 'evidence' key.
    """
    # Validation
    if validation is None:
        validation = validate_explainability(raw_map_np, unet_mask if unet_mask is not None else np.zeros_like(raw_map_np), "gradcam_plusplus")
    # ROI metrics
    from explainability.roi_extraction import extract_xai_rois
    rois = extract_xai_rois(heatmap_blurred, is_tb)
    roi_count = len(rois)
    total_activation = sum(r["activation_score"] for r in rois)  # activation_score is percentage 0-100
    avg_activation = total_activation / roi_count if roi_count > 0 else 0.0
    # Geometry: average bbox size
    if rois:
        avg_width = sum(r["bbox"][2] for r in rois) / roi_count
        avg_height = sum(r["bbox"][3] for r in rois) / roi_count
    else:
        avg_width = 0.0
        avg_height = 0.0
    # Location from quadrant analysis (use precomputed if available)
    if quadrant_analysis is None:
        from explainability.roi_extraction import compute_quadrant_analysis
        quadrant = compute_quadrant_analysis(raw_map_np, unet_mask)
    else:
        quadrant = quadrant_analysis
    dominant_zone = quadrant["dominant_zone"]
    # Heatmap stats
    hm_min = float(np.min(raw_map_np))
    hm_max = float(np.max(raw_map_np))
    hm_mean = float(np.mean(raw_map_np))
    hm_std = float(np.std(raw_map_np))
    # Timing placeholder (we could compute actual time but skip)
    import datetime
    timestamp = datetime.datetime.utcnow().isoformat() + "Z"
    # Confidence calibration
    from explainability.heatmap_processing import calibrate_confidence
    calibrated_conf = calibrate_confidence(prob, threshold, is_tb)
    # Evidence confidence: combine prediction confidence, validation quality, ROI quality, and localization quality.
    # Formula: evidence_confidence = 0.5 * calibrated_conf + 0.3 * val_score + 0.1 * roi_score + 0.1 * loc_score
    # where:
    #   val_score = 1.0 if validation status is "Valid", 0.5 if "Questionable", 0.0 otherwise
    #   roi_score = min(roi_count / 3.0, 1.0)  # up to 3 ROIs considered full score
    #   loc_score = 1.0 if dominant_zone is not "mixed", else 0.5
    val_score = 1.0 if validation["status"] == "Valid" else 0.5 if validation["status"] == "Questionable" else 0.0
    roi_score = min(roi_count / 3.0, 1.0)  # up to 3 ROIs considered full
    loc_score = 1.0 if dominant_zone != "mixed" else 0.5
    evidence_confidence = (calibrated_conf * 0.5 + val_score * 0.3 + roi_score * 0.1 + loc_score * 0.1)  # weighted sum, ensure <=1
    evidence_confidence = max(0.0, min(1.0, evidence_confidence))
    # Build evidence dict (note: prediction, is_tb, and raw_probability are omitted to avoid duplication with root response)
    evidence = {
        "version": "1.0",
        "timestamp": timestamp,
        "calibrated_confidence": calibrated_conf,
        "validation": validation,
        "roi_metrics": {
            "count": roi_count,
            "total_activation": round(total_activation, 1),
            "average_activation": round(avg_activation, 1),
            "average_bbox_width": round(avg_width, 1),
            "average_bbox_height": round(avg_height, 1)
        },
        "geometry": {
            "dominant_zone": dominant_zone,
            "quadrant_scores": quadrant["quadrant_scores"],
            "upper_fraction": quadrant["upper_fraction"],
            "lower_fraction": quadrant["lower_fraction"]
        },
        "heatmap_stats": {
            "min": round(hm_min, 4),
            "max": round(hm_max, 4),
            "mean": round(hm_mean, 4),
            "std": round(hm_std, 4)
        },
        "evidence_confidence": round(evidence_confidence, 4),
        # Relationships: we could include references but skip for brevity
        "_note": "Evidence object aggregates validation, ROI, localization, and heatmap statistics. Prediction and confidence are available in the root response."
    }
    return evidence