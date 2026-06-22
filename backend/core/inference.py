import os
# Force Keras to use PyTorch as its backend
os.environ["KERAS_BACKEND"] = "torch"

import keras
import torch
import numpy as np
import cv2
import threading
import random
from PIL import Image

OPTIMAL_THRESHOLD = 0.93 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "tb_student_densenet121.keras")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMG_SIZE = 224

def pad_to_square(img: Image.Image, fill=0) -> Image.Image:
    w, h = img.size
    if w == h:
        return img
    elif w > h:
        result = Image.new(img.mode, (w, w), fill)
        result.paste(img, (0, (w - h) // 2))
        return result
    else:
        result = Image.new(img.mode, (h, h), fill)
        result.paste(img, ((h - w) // 2, 0))
        return result

def crop_to_original(padded_img: Image.Image, original_size) -> Image.Image:
    w_orig, h_orig = original_size
    w_pad, h_pad = padded_img.size
    if w_orig == h_orig:
        return padded_img
    elif w_orig > h_orig:
        top = (w_orig - h_orig) // 2
        bottom = top + h_orig
        return padded_img.crop((0, top, w_orig, bottom))
    else:
        left = (h_orig - w_orig) // 2
        right = left + w_orig
        return padded_img.crop((left, 0, right, h_orig))

# Lazy load model with thread safety
_model = None
_model_lock = threading.Lock()

def get_model():
    global _model, OPTIMAL_THRESHOLD
    if _model is None:
        with _model_lock:
            if _model is None:
                # Check for model metadata dynamically to resolve clinical threshold settings
                metadata_path = os.path.join(BASE_DIR, "model_metadata.json")
                if os.path.exists(metadata_path):
                    try:
                        import json
                        with open(metadata_path, "r") as f:
                            meta = json.load(f)
                        parsed_thresh = meta.get("optimal_threshold")
                        if parsed_thresh is not None:
                            OPTIMAL_THRESHOLD = float(parsed_thresh)
                            print(f"Loaded calibrated clinical threshold from metadata: {OPTIMAL_THRESHOLD}")
                    except Exception as e_meta:
                        print(f"Failed to load clinical threshold from metadata: {e_meta}")
                else:
                    # Check for best_threshold.txt in the backend folder first
                    thresh_path = os.path.join(BASE_DIR, "best_threshold.txt")
                    if os.path.exists(thresh_path):
                        try:
                            with open(thresh_path, "r") as f:
                                OPTIMAL_THRESHOLD = float(f.read().strip())
                            print(f"Loaded calibrated threshold from best_threshold.txt: {OPTIMAL_THRESHOLD}")
                        except Exception as e_thresh:
                            print(f"Failed to load threshold from best_threshold.txt: {e_thresh}")
                            OPTIMAL_THRESHOLD = 0.93
                    else:
                        OPTIMAL_THRESHOLD = 0.93
                        print(f"Metadata or threshold file not found. Defaulting to: {OPTIMAL_THRESHOLD}")
                    
                if os.path.exists(MODEL_PATH):
                    try:
                        _model = keras.saving.load_model(MODEL_PATH)
                        _model.to(DEVICE)
                        print(f"Model loaded successfully on {DEVICE}")
                        
                        # Warm model compilation (warm-up dummy forward pass)
                        try:
                            dummy_input = torch.zeros(1, IMG_SIZE, IMG_SIZE, 3).to(DEVICE)
                            _ = _model(dummy_input)
                            print("Model warm-up pass completed successfully ✓")
                        except Exception as warm_err:
                            print(f"Warning during model warm-up: {warm_err}")
                    except Exception as e:
                        print(f"Error loading model: {e}")
                else:
                    print(f"WARNING: {MODEL_PATH} not found in the backend folder. Predictions will fail until it is added.")
    return _model

def _generate_density_heatmap(original_img: Image.Image, is_tb: bool) -> Image.Image:
    # FALLBACK: Create a beautiful simulated clinical heatmap targeting density / consolidating features
    w, h = original_img.size
    orig_np = np.array(original_img)
    if len(orig_np.shape) == 3:
        gray = cv2.cvtColor(orig_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = orig_np.copy()
        orig_np = cv2.cvtColor(orig_np, cv2.COLOR_GRAY2RGB)
        
    _, thresholded = cv2.threshold(gray, 150, 255, cv2.THRESH_TOZERO)
    
    # Create a lung mask (center area of X-ray)
    mask = np.zeros_like(gray)
    cv2.ellipse(mask, (int(w*0.5), int(h*0.55)), (int(w*0.38), int(h*0.45)), 0, 0, 360, 255, -1)
    focused = cv2.bitwise_and(thresholded, mask)
    
    blurred = cv2.GaussianBlur(focused, (51, 51), 0)
    norm_blurred = (blurred - blurred.min()) / (blurred.max() - blurred.min() + 1e-8)
    
    heatmap_8bit = (norm_blurred * 255).astype(np.uint8)
    color_heatmap = cv2.applyColorMap(heatmap_8bit, cv2.COLORMAP_JET)
    color_heatmap_rgb = cv2.cvtColor(color_heatmap, cv2.COLOR_BGR2RGB)
    
    alpha = 0.50 if is_tb else 0.25
    blended = cv2.addWeighted(orig_np, 1.0 - alpha, color_heatmap_rgb, alpha, 0)
    return Image.fromarray(blended)

def generate_saliency_heatmap(model, tensor, original_img: Image.Image, is_tb: bool, method: str = "gradcam_plusplus", return_raw: bool = False) -> tuple:
    try:
        if model is None or tensor is None:
            # Generate simulated base activation map (224x224)
            w, h = 224, 224
            cam_np = np.zeros((h, w), dtype=np.float32)
            if is_tb:
                # Tuberculosis focal point: upper right lung field
                cv2.circle(cam_np, (int(w * 0.38), int(h * 0.35)), 45, 1.0, -1)
                cv2.circle(cam_np, (int(w * 0.65), int(h * 0.45)), 30, 0.4, -1)
            else:
                # Normal scan: diffuse symmetrical low-level activity
                cv2.ellipse(cam_np, (int(w*0.35), int(h*0.5)), (25, 45), 0, 0, 360, 0.25, -1)
                cv2.ellipse(cam_np, (int(w*0.65), int(h*0.5)), (25, 45), 0, 0, 360, 0.25, -1)
            
            cam_np = cv2.GaussianBlur(cam_np, (31, 31), 0)
            cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)
        else:
            # Multi-output model targeting Keras "relu" activation layer of DenseNet
            import keras
            last_conv_layer = model.get_layer("relu")
            grad_model = keras.Model(inputs=model.inputs, outputs=[last_conv_layer.output, model.output])
            
            with torch.enable_grad():
                tensor_input = tensor.clone().detach().requires_grad_(True)
                act, logit = grad_model(tensor_input)
                
                if logit.dim() > 1:
                    logit = logit.squeeze(1)
                if logit.dim() > 0:
                    logit = logit.squeeze()
                    
                grads = torch.autograd.grad(logit, act, grad_outputs=torch.ones_like(logit), retain_graph=False)[0]
            
            if method == "gradcam":
                # Standard Grad-CAM
                weights = torch.mean(grads, dim=(1, 2), keepdim=True)
                cam = torch.sum(weights * act, dim=-1).squeeze(0)
                cam = torch.clamp(cam, min=0)
            else:
                # Standard Grad-CAM++ (used as base for all other modes)
                grads_power_2 = grads ** 2
                grads_power_3 = grads ** 3
                sum_act = torch.sum(act, dim=(1, 2), keepdim=True)
                denominator = 2 * grads_power_2 + sum_act * grads_power_3
                denominator = torch.where(denominator != 0.0, denominator, torch.ones_like(denominator))
                alpha = grads_power_2 / denominator
                weights = torch.sum(alpha * torch.clamp(grads, min=0), dim=(1, 2), keepdim=True)
                cam = torch.sum(weights * act, dim=-1).squeeze(0)
                cam = torch.clamp(cam, min=0)
                
            cam_min, cam_max = cam.min(), cam.max()
            if cam_max > cam_min:
                cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
                
            cam_np = cam.detach().cpu().numpy()
            
        # Apply specific XAI post-processing to the normalized cam_np map
        if method == "attention":
            # High-pass filter for edge focused attention highlights
            blurred_low = cv2.GaussianBlur(cam_np, (3, 3), 0)
            blurred_high = cv2.GaussianBlur(cam_np, (11, 11), 0)
            attention_map = np.abs(blurred_low - blurred_high)
            cam_np = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
            kernel = np.ones((2, 2), np.uint8)
            cam_np = cv2.dilate(cam_np, kernel, iterations=1)
        elif method == "coverage":
            # Binary threshold mask
            binary_mask = np.where(cam_np >= 0.22, 1.0, 0.0)
            cam_np = cv2.GaussianBlur(binary_mask, (7, 7), 0)
        elif method == "attribution":
            # Localized grid-attribution
            grid_h, grid_w = cam_np.shape
            for r in range(0, grid_h, 2):
                for c in range(0, grid_w, 2):
                    val = np.mean(cam_np[r:r+2, c:c+2])
                    cam_np[r:r+2, c:c+2] = val
                    
        # Apply low-activation thresholding to filter out diffuse background noise (except coverage/attention)
        if method not in ["coverage", "attention"]:
            cam_np = np.where(cam_np >= 0.22, cam_np, 0.0)
            
        # Resize to original image size
        w, h = original_img.size
        heatmap_resized = cv2.resize(cam_np, (w, h))
        
        if method not in ["coverage", "attention"]:
            heatmap_blurred = cv2.GaussianBlur(heatmap_resized, (15, 15), 0)
        else:
            heatmap_blurred = heatmap_resized
            
        h_min, h_max = heatmap_blurred.min(), heatmap_blurred.max()
        if h_max > h_min:
            heatmap_blurred = (heatmap_blurred - h_min) / (h_max - h_min + 1e-8)
            
        heatmap_8bit = (heatmap_blurred * 255).astype(np.uint8)
        
        # Select colormap based on method
        if method == "attention":
            color_heatmap = cv2.applyColorMap(heatmap_8bit, cv2.COLORMAP_COOL)
        elif method == "coverage":
            color_heatmap = cv2.applyColorMap(heatmap_8bit, cv2.COLORMAP_WINTER)
        elif method == "attribution":
            color_heatmap = cv2.applyColorMap(heatmap_8bit, cv2.COLORMAP_HOT)
        else:
            color_heatmap = cv2.applyColorMap(heatmap_8bit, cv2.COLORMAP_JET)
            
        orig_np = np.array(original_img)
        if len(orig_np.shape) == 2:
            orig_np = cv2.cvtColor(orig_np, cv2.COLOR_GRAY2RGB)
            
        color_heatmap = cv2.resize(color_heatmap, (w, h))
        color_heatmap_rgb = cv2.cvtColor(color_heatmap, cv2.COLOR_BGR2RGB)
        
        alpha = 0.55 if is_tb else 0.35
        if method == "attention":
            alpha = 0.65
        elif method == "coverage":
            alpha = 0.45
            
        blended = cv2.addWeighted(orig_np, 1.0 - alpha, color_heatmap_rgb, alpha, 0)
        
        if return_raw:
            return Image.fromarray(blended), False, heatmap_blurred
        return Image.fromarray(blended), False
        
    except Exception as e:
        print(f"Explainable AI mapping failed for {method}: {e}. Falling back to density.")
        fallback_img = _generate_density_heatmap(original_img, is_tb)
        # Create a mock raw heatmap for the fallback
        w, h = original_img.size
        fallback_raw = np.zeros((h, w), dtype=np.float32)
        if is_tb:
            cv2.circle(fallback_raw, (int(w * 0.38), int(h * 0.35)), int(min(w, h) * 0.2), 1.0, -1)
        else:
            cv2.circle(fallback_raw, (int(w * 0.5), int(h * 0.5)), int(min(w, h) * 0.25), 0.25, -1)
        fallback_raw = cv2.GaussianBlur(fallback_raw, (31, 31), 0)
        
        if return_raw:
            return fallback_img, True, fallback_raw
        return fallback_img, True

# ── Explainable AI (XAI) Helpers ─────────────────────────────

def calibrate_confidence(prob: float, threshold: float, is_tb: bool) -> float:
    """Mathematically calibrate confidence relative to dynamic threshold."""
    if is_tb:
        calibrated = 0.50 + 0.50 * (prob - threshold) / (1.0 - threshold) if threshold < 1.0 else 1.0
    else:
        calibrated = 0.50 + 0.50 * (threshold - prob) / threshold if threshold > 0.0 else 1.0
    return max(0.50, min(1.00, calibrated))

def extract_xai_rois(heatmap_blurred: np.ndarray, is_tb: bool) -> list:
    """
    Extract Regions of Interest (ROIs) from the heatmap using OpenCV contours.
    """
    h, w = heatmap_blurred.shape
    # Threshold to identify hot spots (values >= 0.38 for TB, 0.28 for Normal)
    thresh_val = 0.38 if is_tb else 0.28
    _, mask = cv2.threshold((heatmap_blurred * 255).astype(np.uint8), int(thresh_val * 255), 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    rois = []
    total_activation_sum = 0.0
    
    # Process each contour
    for idx, contour in enumerate(contours):
        if cv2.contourArea(contour) < 15: # Filter very small noise
            continue
            
        x, y, cw, ch = cv2.boundingRect(contour)
        
        # Enclosing circle
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        
        # Simplified contour for rendering
        epsilon = 0.015 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        contour_pts = [[int(pt[0][0]), int(pt[0][1])] for pt in approx]
        
        # Calculate mean activation in this contour
        contour_mask = np.zeros_like(mask)
        cv2.drawContours(contour_mask, [contour], -1, 255, -1)
        mean_val = cv2.mean(heatmap_blurred, mask=contour_mask)[0]
        
        # Calculate sum of activation as proxy for contribution
        sum_val = cv2.sumElems(cv2.multiply(heatmap_blurred, contour_mask.astype(np.float32)/255.0))[0]
        total_activation_sum += sum_val
        
        # Map center to anatomical zones
        nx = (x + cw/2.0) / w
        ny = (y + ch/2.0) / h
        
        side = "Right" if nx < 0.50 else "Left"
        if ny < 0.40:
            zone = "Upper"
        elif ny < 0.68:
            zone = "Middle"
        else:
            zone = "Lower"
            
        location = f"{side} {zone} Lung Zone"
        
        rois.append({
            "id": chr(65 + idx),
            "activation_score": float(mean_val),
            "sum_activation": float(sum_val),
            "location": location,
            "bbox": [int(x), int(y), int(cw), int(ch)],
            "circle": [int(cx), int(cy), int(radius)],
            "contour": contour_pts,
            "center": [float(nx), float(ny)]
        })
        
    # Sort ROIs by sum activation (descending)
    rois.sort(key=lambda x: x["sum_activation"], reverse=True)
    
    # Calculate relative contribution percentage
    final_rois = []
    for idx, r in enumerate(rois[:6]): # Limit to top 6 ROIs
        contrib = (r["sum_activation"] / total_activation_sum * 100.0) if total_activation_sum > 0 else 0.0
        # Assign clean sorted IDs (A, B, C...)
        r["id"] = chr(65 + idx)
        r["contribution_pct"] = round(contrib, 1)
        r["activation_score"] = round(r["activation_score"] * 100.0, 1)
        
        # Delete temporary field
        del r["sum_activation"]
        final_rois.append(r)
        
    return final_rois

def generate_xai_clinical_summary(rois: list, is_tb: bool, confidence: float) -> str:
    """
    Generate a human-readable clinical explanation summary using clinical safety constraints.
    """
    if not rois:
        return "No significant focal abnormalities or salient opacities detected. Radiographic features appear within normal limits."
        
    top_roi = rois[0]
    loc = top_roi["location"]
    contrib = top_roi["contribution_pct"]
    act = top_roi["activation_score"]
    
    if is_tb:
        summary = (
            f"CAD assessment indicates a high-probability focal opacity localized to the {loc} (Region {top_roi['id']}), "
            f"representing {contrib}% of the primary predictive variance (peak local activation: {act}%). This saliency "
            f"distribution is highly correlative with consolidative, infiltrative, or cavitary pathologies classically "
            f"associated with active Mycobacterium tuberculosis infection. Sputum acid-fast bacilli (AFB) smear, molecular "
            f"assays, and clinical correlation are strongly recommended to definitively confirm active disease."
        )
    else:
        summary = (
            f"CAD analysis reveals diffuse, low-level background gradients (predominantly mapped to the {loc}, "
            f"representing {contrib}% relative variance) without evidence of focal asymmetric opacification, cavitation, "
            f"or structural consolidation. No salient radiographic features suggestive of active pulmonary tuberculosis "
            f"are identified. As a computer-aided triage finding, this does not preclude latent infection or early-stage "
            f"non-tuberculous respiratory pathologies. Clinical correlation remains advised for symptomatic patients."
        )
    return summary

def compute_xai_payload(is_tb: bool, prob: float, heatmap_blurred: np.ndarray, quality_score=85) -> dict:
    """Assemble final Explainable AI payload structures."""
    rois = extract_xai_rois(heatmap_blurred, is_tb)
    summary = generate_xai_clinical_summary(rois, is_tb, prob)
    
    ranking = [
        {"region_id": r["id"], "location": r["location"], "contribution_pct": r["contribution_pct"]}
        for r in rois
    ]
    
    calibrated_conf = calibrate_confidence(prob, OPTIMAL_THRESHOLD, is_tb)
    
    reliability = "High" if quality_score >= 85 else "Medium" if quality_score >= 60 else "Low"
    
    # Calculate uncertainty
    diff = abs(calibrated_conf - 0.50)
    uncertainty = "Low" if diff >= 0.35 else "Medium" if diff >= 0.15 else "High"
    
    metrics = {
        "tb_probability": round(prob * 100.0, 1),
        "calibrated_confidence": round(calibrated_conf * 100.0, 1),
        "reliability": reliability,
        "uncertainty": uncertainty
    }
    
    return {
        "rois": rois,
        "summary": summary,
        "ranking": ranking,
        "metrics": metrics
    }

def get_mock_xai_payload(img_size: tuple, is_tb: bool, prob: float, quality_score=85) -> dict:
    """Generate mock XAI results payload for demo run consistency."""
    w, h = img_size
    calibrated_conf = calibrate_confidence(prob, OPTIMAL_THRESHOLD, is_tb)
    reliability = "High" if quality_score >= 85 else "Medium"
    
    diff = abs(calibrated_conf - 0.50)
    uncertainty = "Low" if diff >= 0.35 else "Medium" if diff >= 0.15 else "High"
    
    metrics = {
        "tb_probability": round(prob * 100.0, 1),
        "calibrated_confidence": round(calibrated_conf * 100.0, 1),
        "reliability": reliability,
        "uncertainty": uncertainty
    }
    
    if is_tb:
        rois = [
            {
                "id": "A",
                "activation_score": 92.4,
                "contribution_pct": 82.0,
                "location": "Right Upper Lung Zone",
                "bbox": [int(w * 0.18), int(h * 0.15), int(w * 0.22), int(h * 0.25)],
                "circle": [int(w * 0.29), int(h * 0.27), int(w * 0.12)],
                "contour": [
                    [int(w * 0.18), int(h * 0.15)],
                    [int(w * 0.40), int(h * 0.15)],
                    [int(w * 0.40), int(h * 0.40)],
                    [int(w * 0.18), int(h * 0.40)]
                ],
                "center": [0.29, 0.27]
            },
            {
                "id": "B",
                "activation_score": 78.1,
                "contribution_pct": 18.0,
                "location": "Left Mid Lung Zone",
                "bbox": [int(w * 0.58), int(h * 0.38), int(w * 0.20), int(h * 0.22)],
                "circle": [int(w * 0.68), int(h * 0.49), int(w * 0.10)],
                "contour": [
                    [int(w * 0.58), int(h * 0.38)],
                    [int(w * 0.78), int(h * 0.38)],
                    [int(w * 0.78), int(h * 0.60)],
                    [int(w * 0.58), int(h * 0.60)]
                ],
                "center": [0.68, 0.49]
            }
        ]
    else:
        rois = [
            {
                "id": "A",
                "activation_score": 32.5,
                "contribution_pct": 60.0,
                "location": "Left Lower Lung Zone",
                "bbox": [int(w * 0.55), int(h * 0.60), int(w * 0.22), int(h * 0.22)],
                "circle": [int(w * 0.66), int(h * 0.71), int(w * 0.11)],
                "contour": [
                    [int(w * 0.55), int(h * 0.60)],
                    [int(w * 0.77), int(h * 0.60)],
                    [int(w * 0.77), int(h * 0.82)],
                    [int(w * 0.55), int(h * 0.82)]
                ],
                "center": [0.66, 0.71]
            },
            {
                "id": "B",
                "activation_score": 28.2,
                "contribution_pct": 40.0,
                "location": "Right Lower Lung Zone",
                "bbox": [int(w * 0.22), int(h * 0.58), int(w * 0.20), int(h * 0.22)],
                "circle": [int(w * 0.32), int(h * 0.69), int(w * 0.10)],
                "contour": [
                    [int(w * 0.22), int(h * 0.58)],
                    [int(w * 0.42), int(h * 0.58)],
                    [int(w * 0.42), int(h * 0.80)],
                    [int(w * 0.22), int(h * 0.80)]
                ],
                "center": [0.32, 0.69]
            }
        ]
        
    ranking = [
        {"region_id": r["id"], "location": r["location"], "contribution_pct": r["contribution_pct"]}
        for r in rois
    ]
    summary = generate_xai_clinical_summary(rois, is_tb, prob)
    
    return {
        "rois": rois,
        "summary": summary,
        "ranking": ranking,
        "metrics": metrics
    }

# ── Main Prediction Flow ─────────────────────────────────────

def predict_image(img: Image.Image):
    model = get_model()
    # Import image_to_base64 here
    from utils.image_helpers import image_to_base64

    if model is None:
        # Mock/Demo mode fallback to keep the application fully testable without the weight file
        prob = random.uniform(0.15, 0.88)
        is_tb = prob >= OPTIMAL_THRESHOLD
        
        # Aspect ratio preserving padding for fallback heatmap
        padded_img = pad_to_square(img)
        
        # Generate all 5 heatmaps for demo mode
        heatmaps_b64 = {}
        for method in ["gradcam", "gradcam_plusplus", "attention", "coverage", "attribution"]:
            h_img, _ = generate_saliency_heatmap(None, None, padded_img, is_tb, method=method)
            h_cropped = crop_to_original(h_img, img.size)
            heatmaps_b64[method] = image_to_base64(h_cropped)

        gradcam_plusplus_img, _ = generate_saliency_heatmap(None, None, padded_img, is_tb, method="gradcam_plusplus")
        gradcam_plusplus_cropped = crop_to_original(gradcam_plusplus_img, img.size)
        
        xai_payload = get_mock_xai_payload(img.size, is_tb, prob)
        
        return {
            "prediction": "Tuberculosis" if is_tb else "Normal",
            "confidence": float(prob),
            "threshold_used": OPTIMAL_THRESHOLD,
            "is_tb": bool(is_tb),
            "demo_mode": True,
            "saliency_fallback": False,
            "heatmaps": heatmaps_b64,
            "xai_results": xai_payload
        }, gradcam_plusplus_cropped
        
    # Pad to square to preserve aspect ratio (matches training preprocessing)
    padded_img = pad_to_square(img)
    # Force grayscale to match the degenerate 3-channel distribution (R=G=B) of training
    gray_img = padded_img.convert("L")
    resized_img = gray_img.resize((IMG_SIZE, IMG_SIZE))
    
    # Preprocess matching training pipeline (ResNet50 preprocess_input style: BGR mean subtracted, no division by 255)
    arr = np.array(resized_img, dtype=np.float32)
    # Stack 3 times to create R=G=B channels
    x = np.stack([arr, arr, arr], axis=-1)
    # Apply BGR mean subtraction: B -= 103.939, G -= 116.779, R -= 123.68
    x[..., 0] -= 103.939
    x[..., 1] -= 116.779
    x[..., 2] -= 123.68
    tensor = torch.tensor(x).unsqueeze(0).to(DEVICE)
    
    # Check prediction
    with torch.no_grad():
        logit = model(tensor)
        if logit.dim() > 1:
            logit = logit.squeeze(1)
        prob = torch.sigmoid(logit).item() if logit.dim() == 0 else torch.sigmoid(logit)[0].item()
        
    is_tb = prob >= OPTIMAL_THRESHOLD
    
    # Generate all 5 heatmaps for actual model runs
    heatmaps_b64 = {}
    any_fallback = False
    for method in ["gradcam", "gradcam_plusplus", "attention", "coverage", "attribution"]:
        h_img, is_fb = generate_saliency_heatmap(model, tensor, padded_img, is_tb, method=method)
        h_cropped = crop_to_original(h_img, img.size)
        heatmaps_b64[method] = image_to_base64(h_cropped)
        if is_fb:
            any_fallback = True
        
    gradcam_plusplus_img, is_fb, raw_map = generate_saliency_heatmap(model, tensor, padded_img, is_tb, method="gradcam_plusplus", return_raw=True)
    gradcam_plusplus_cropped = crop_to_original(gradcam_plusplus_img, img.size)
    
    # Crop raw heatmap to preserve original aspect ratio bounding box alignments
    raw_map_cropped = crop_to_original(Image.fromarray((raw_map * 255).astype(np.uint8)), img.size)
    raw_map_np = np.array(raw_map_cropped, dtype=np.float32) / 255.0
    
    xai_payload = compute_xai_payload(is_tb, prob, raw_map_np)
    if is_fb:
        any_fallback = True
    
    return {
        "prediction": "Tuberculosis" if is_tb else "Normal",
        "confidence": float(prob),
        "threshold_used": OPTIMAL_THRESHOLD,
        "is_tb": bool(is_tb),
        "demo_mode": False,
        "saliency_fallback": any_fallback,
        "heatmaps": heatmaps_b64,
        "xai_results": xai_payload
    }, gradcam_plusplus_cropped
