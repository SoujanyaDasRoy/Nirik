import io
import base64
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
    if width < 512 or height < 512:
        resolution = "Low Resolution"
    else:
        resolution = "Acceptable Resolution"
        
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
        
    # 4. Coverage (heuristic based on aspect ratio)
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
    # Standard medical chest radiographs are grayscale. If an RGB image has high channel variance,
    # it is likely a regular color photo, graph, screenshot, or other non-medical image.
    if len(pixels.shape) == 3 and pixels.shape[2] >= 3:
        r = pixels[:, :, 0].astype(np.float32)
        g = pixels[:, :, 1].astype(np.float32)
        b = pixels[:, :, 2].astype(np.float32)
        mean_channel_diff = np.mean(np.abs(r - g) + np.abs(g - b))
        if mean_channel_diff > 12.0:
            return False, "Color detected. Standard chest radiographs are strictly grayscale."
            
    # Convert to grayscale array for structural checks
    gray_img = img.convert("L")
    gray_pixels = np.array(gray_img).astype(np.float32)
    
    # 3. Dynamic range / contrast checks
    std_val = np.std(gray_pixels)
    if std_val < 20.0:
        return False, "Extremely low contrast. Image does not contain the dynamic range of a chest X-ray."
    if std_val > 110.0:
        return False, "Excessive contrast. Likely a binary diagram, screenshot, or document rather than a chest radiograph."
        
    # 4. Average intensity check
    mean_val = np.mean(gray_pixels)
    if mean_val < 25.0 or mean_val > 220.0:
        return False, "Sub-optimal pixel intensity range. Standard chest X-rays have balanced exposure."
        
    # 5. Extreme values check (solid backgrounds, screenshots, text grids)
    pure_black = np.sum(gray_pixels < 5)
    pure_white = np.sum(gray_pixels > 250)
    total_pixels = w * h
    extreme_ratio = (pure_black + pure_white) / total_pixels
    if extreme_ratio > 0.65:
        return False, "Excessive solid black/white regions. Likely a screenshot, chart, or text-heavy graphic."
        
    # 6. Anatomical profile heuristic
    h_30 = int(h * 0.3)
    middle_zone = gray_pixels[h_30:h-h_30, :]
    bottom_zone = gray_pixels[h-h_30:, :]
    if np.mean(middle_zone) < 15.0 and np.mean(bottom_zone) < 15.0:
        return False, "Anatomical structure check failed. Image is blank or contains no structures."

    return True, "Valid chest radiograph"

