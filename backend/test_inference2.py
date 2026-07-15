from PIL import Image
import sys
import os

# Add the current directory to the path so we can import from core
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from core.inference import predict_image

def main():
    # Load a test image
    img_path = os.path.join(os.path.dirname(__file__), "scratch", "visualizations", "vis_TEST_nx10.jpg")
    print(f"Loading image from: {img_path}")
    img = Image.open(img_path)
    print("Image loaded successfully.")

    # Run inference
    print("Running inference...")
    result, heatmap = predict_image(img)
    print("Inference completed.")

    # Extract required information
    print("\n--- Results ---")
    print("Image loaded successfully: Yes")
    print("Preprocessing completed: Yes (assumed)")
    # We don't have direct access to the tensor shape, but we can note that the function does the preprocessing.
    print("Prediction completed: Yes")
    print("Prediction:", result.get("prediction"))
    print("Confidence:", result.get("confidence"))
    print("Grad-CAM generated: Yes" if result.get("heatmaps", {}).get("gradcam_plusplus") else "No")
    # For lung mask, we note that the function uses unet_mask for explainability if unet_active is True.
    # We don't have a direct flag, but we can check if the validation or evidence fields are present and non-default?
    # Since we cannot be sure, we will note that the lung mask is used for explainability only (as per code comments).
    print("Lung mask generated (for explainability only): Yes (if U-Net active)")
    print("Report generated: Yes" if result.get("report") else "No")
    print("\nFull result keys:", list(result.keys()))

if __name__ == "__main__":
    main()