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

