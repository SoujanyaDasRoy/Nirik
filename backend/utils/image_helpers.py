import io
import base64
import cv2
from PIL import Image

def process_standard_image(file_bytes) -> Image.Image:
    return Image.open(io.BytesIO(file_bytes)).convert("RGB")

def image_to_base64(img: Image.Image) -> str:
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"

def analyze_image_quality(img: Image.Image) -> dict:
    import numpy as np
    # Convert image to grayscale for analysis
    gray_img = img.convert("L")
    width, height = gray_img.size
    pixels = np.array(gray_img)
    
    # 1. Exposure (based on mean pixel intensity)
    mean_val = np.mean(pixels)
    if mean_val > 200:
        exposure = "Overexposed"
    elif mean_val < 50:
        exposure = "Underexposed"
    else:
        exposure = "Adequate Exposure"
        
    # 2. Resolution
    resolution = f"{width} x {height} pixels"
    if width < 512 or height < 512:
        resolution = f"{width} x {height} pixels (Low)"
        
    # 3. Rotation (check symmetry by comparing left and right halves)
    left_half = pixels[:, :width//2]
    right_half = pixels[:, width//2:]
    if right_half.shape[1] != left_half.shape[1]:
        right_half = right_half[:, :left_half.shape[1]]
    mean_left = np.mean(left_half)
    mean_right = np.mean(right_half)
    diff = abs(mean_left - mean_right) / max(1.0, (mean_left + mean_right)/2.0)
    
    if diff > 0.25:
        rotation = "Significant Patient Rotation"
    elif diff > 0.10:
        rotation = "Mild Patient Rotation"
    else:
        rotation = "No Rotation"
        
    # 4. Coverage (heuristic based on body contours & aspect ratio)
    touches_edge = False
    _, binary = cv2.threshold(pixels, 35, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        min_x = width
        max_x = 0
        has_large_contour = False
        for c in contours:
            if cv2.contourArea(c) > (width * height * 0.02):
                x, y, w_c, h_c = cv2.boundingRect(c)
                min_x = min(min_x, x)
                max_x = max(max_x, x + w_c)
                has_large_contour = True
        
        if has_large_contour:
            margin_x = max(2, int(width * 0.015))
            if min_x <= margin_x or max_x >= width - margin_x:
                touches_edge = True

    if touches_edge:
        coverage = "Partial Coverage"
    else:
        aspect_ratio = width / height
        if aspect_ratio < 0.65 or aspect_ratio > 1.35:
            coverage = "Partial Coverage"
        else:
            coverage = "Full Lung Coverage"
        
    # Quality Score calculation (deduct from 100)
    score = 100.0
    warnings = []
    
    if exposure == "Overexposed":
        score -= 20
        warnings.append("Image appears overexposed (bright regions dominant).")
    elif exposure == "Underexposed":
        score -= 20
        warnings.append("Image appears underexposed (dark regions dominant).")
        
    if resolution == "Low Resolution":
        score -= 25
        warnings.append(f"Low image resolution ({width}x{height}). Minimum recommended is 512x512.")
        
    if rotation == "Significant Patient Rotation":
        score -= 15
        warnings.append("Significant patient rotation detected. Structural symmetry compromised.")
    elif rotation == "Mild Patient Rotation":
        score -= 5
        warnings.append("Mild patient rotation detected.")
        
    if coverage == "Partial Coverage":
        score -= 15
        if touches_edge:
            warnings.append("Body structures touch the edge of the image frame. Lateral lung fields may be cropped.")
        else:
            warnings.append("Sub-optimal aspect ratio. Possible cropping of lateral lung fields.")
        
    quality_score = max(10.0, min(100.0, score))
    suitable_for_ai = quality_score >= 70.0
    
    return {
        "exposure": exposure,
        "coverage": coverage,
        "resolution": f"{width}x{height} ({resolution})",
        "rotation": rotation,
        "suitable_for_ai": suitable_for_ai,
        "quality_score": int(quality_score),
        "warnings": warnings
    }

def validate_chest_xray(img: Image.Image) -> tuple[bool, str]:
    import numpy as np
    
    # 1. Size check
    w, h = img.size
    if w < 128 or h < 128:
        return False, "Resolution too low. Image must be at least 128x128 pixels."
        
    # Convert PIL to numpy array
    pixels = np.array(img)
    
    # 2. Grayscale/Color check
    if len(pixels.shape) == 3 and pixels.shape[2] >= 3:
        r = pixels[:, :, 0].astype(np.float32)
        g = pixels[:, :, 1].astype(np.float32)
        b = pixels[:, :, 2].astype(np.float32)
        mean_channel_diff = np.mean(np.abs(r - g) + np.abs(g - b))
        if mean_channel_diff > 12.0:
            return False, "Color detected. Standard chest radiographs are strictly grayscale."
            
    # Convert to grayscale array for structural checks
    gray_img = img.convert("L")
    gray_pixels = np.array(gray_img)
    
    # Crop any black borders (padding) to analyze the actual image content
    non_black = np.argwhere(gray_pixels > 15)
    if len(non_black) > 0:
        min_y, min_x = non_black.min(axis=0)
        max_y, max_x = non_black.max(axis=0)
        # Verify the cropped box represents a significant chest structure (at least 50% of width/height)
        if (max_x - min_x) >= int(w * 0.5) and (max_y - min_y) >= int(h * 0.5):
            analyzed_pixels = gray_pixels[min_y:max_y+1, min_x:max_x+1]
        else:
            analyzed_pixels = gray_pixels
    else:
        analyzed_pixels = gray_pixels
        
    h_a, w_a = analyzed_pixels.shape
    analyzed_pixels_f = analyzed_pixels.astype(np.float32)
    
    # 3. Dynamic range / contrast checks
    std_val = np.std(analyzed_pixels_f)
    if std_val < 20.0:
        return False, "Extremely low contrast. Image does not contain the dynamic range of a chest X-ray."
    if std_val > 110.0:
        return False, "Excessive contrast. Likely a binary diagram, screenshot, or document."
        
    # 4. Average intensity check
    mean_val = np.mean(analyzed_pixels_f)
    if mean_val < 25.0 or mean_val > 220.0:
        return False, "Sub-optimal pixel intensity range. Standard chest X-rays have balanced exposure."
        
    # 5. Extreme values check
    pure_black = np.sum(analyzed_pixels < 5)
    pure_white = np.sum(analyzed_pixels > 250)
    total_pixels = h_a * w_a
    extreme_ratio = (pure_black + pure_white) / total_pixels
    if extreme_ratio > 0.65:
        return False, "Excessive solid black/white regions. Likely a screenshot, chart, or text-heavy graphic."
        
    # Left-Right Structural Symmetry Check and Mediastinum/Spine Center Column Brightness Check
    # are bypassed to prevent rejecting chest radiographs with severe asymmetrical pathologies
    # (e.g. unilateral consolidations, pleural effusions, collapses).
        
    # 8. High Frequency Edge Orientation check (to detect text/diagrams/screenshots)
    grad_x = cv2.Sobel(analyzed_pixels, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(analyzed_pixels, cv2.CV_32F, 0, 1, ksize=3)
    
    abs_grad_x = np.abs(grad_x)
    abs_grad_y = np.abs(grad_y)
    
    strong_edges_x = abs_grad_x > (np.max(abs_grad_x) * 0.15)
    strong_edges_y = abs_grad_y > (np.max(abs_grad_y) * 0.15)
    
    x_edge_ratio = np.sum(strong_edges_x) / total_pixels
    y_edge_ratio = np.sum(strong_edges_y) / total_pixels
    
    if x_edge_ratio > 0.09 or y_edge_ratio > 0.09:
        return False, f"Excessive sharp straight edges detected (X: {x_edge_ratio:.3f}, Y: {y_edge_ratio:.3f}). Screenshots, documents, or diagrams are not supported."

    return True, "Valid chest radiograph"

