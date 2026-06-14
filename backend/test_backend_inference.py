import os
import sys

# Ensure backend folder is in path for module resolution
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.inference import get_model, predict_image
from PIL import Image

def run_test():
    print("Starting backend inference self-test...")
    
    # 1. Load model
    model = get_model()
    if model is None:
        print("✗ Error: Model failed to load!")
        sys.exit(1)
    print("✓ Model pre-loaded successfully.")
    
    # 2. Prepare dummy image (non-square, simulating a typical chest X-ray aspect ratio)
    original_size = (800, 600)
    print(f"Creating dummy test image with size {original_size}...")
    dummy_img = Image.new("RGB", original_size, color=(128, 128, 128))
    
    # 3. Predict image
    print("Running prediction and Grad-CAM generation...")
    result, heatmap_img = predict_image(dummy_img)
    
    # 4. Verify outputs
    print("\n--- Test Results ---")
    print("Result Keys:", list(result.keys()))
    print("Prediction:", result.get("prediction"))
    print("Confidence:", result.get("confidence"))
    print("Threshold Used:", result.get("threshold_used"))
    print("Is TB:", result.get("is_tb"))
    print("Heatmap Image Type:", type(heatmap_img))
    print("Heatmap Size:", heatmap_img.size if heatmap_img else None)
    
    # Assertions
    import json
    expected_threshold = 0.93
    metadata_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_metadata.json")
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r") as f:
                expected_threshold = float(json.load(f).get("optimal_threshold", 0.93))
        except Exception:
            expected_threshold = 0.93
    else:
        best_t_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_threshold.txt")
        if os.path.exists(best_t_path):
            try:
                with open(best_t_path, "r") as f:
                    expected_threshold = float(f.read().strip())
            except Exception:
                expected_threshold = 0.93

    assert "prediction" in result, "prediction key missing"
    assert "confidence" in result, "confidence key missing"
    assert "threshold_used" in result, "threshold_used key missing"
    assert "is_tb" in result, "is_tb key missing"
    assert 0.0 <= result["confidence"] <= 1.0, f"confidence {result['confidence']} out of range"
    assert result["threshold_used"] == expected_threshold, f"Expected threshold {expected_threshold}, got {result['threshold_used']}"
    assert heatmap_img is not None, "Heatmap image is None"
    assert heatmap_img.size == original_size, f"Heatmap size {heatmap_img.size} does not match original size {original_size}"
    
    print("\n✓ ALL TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_test()
