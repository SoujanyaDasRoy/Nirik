import os
import time
import csv

# Force Keras to use PyTorch as its backend
os.environ["KERAS_BACKEND"] = "torch"

import keras
import torch
import numpy as np
from PIL import Image

# Import backend prediction modules
import sys
sys.path.append(r"C:\Users\sdroy\OneDrive\Desktop\Documents\Final Year Project\backend")
from core.inference import predict_image, get_model, OPTIMAL_THRESHOLD

TEST_IMAGES_DIR = r"C:\Users\sdroy\OneDrive\Desktop\Documents\Test images"
OUTPUT_CSV = r"C:\Users\sdroy\OneDrive\Desktop\Documents\Final Year Project\backend\scratch\test_image_results.csv"

# Ensure output dir exists
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

print("Listing test images...")
images = [f for f in os.listdir(TEST_IMAGES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
print(f"Found {len(images)} images: {images}")

results = []

# Heatmap folder for audit visualizations
vis_dir = r"C:\Users\sdroy\OneDrive\Desktop\Documents\Final Year Project\backend\scratch\visualizations"
os.makedirs(vis_dir, exist_ok=True)

model = get_model()
print("Using clinical threshold:", OPTIMAL_THRESHOLD)

for img_name in images:
    img_path = os.path.join(TEST_IMAGES_DIR, img_name)
    print(f"\nProcessing {img_name}...")
    
    start_time = time.time()
    errors = []
    warnings = []
    
    try:
        img = Image.open(img_path)
        
        # Run prediction
        predict_res, h_cropped = predict_image(img)
        elapsed = time.time() - start_time
        
        prediction = predict_res["prediction"]
        raw_prob = predict_res["confidence"]
        
        # In our code, calibrated confidence is stored under xai_results["metrics"]["calibrated_confidence"]
        calibrated_conf = predict_res.get("xai_results", {}).get("metrics", {}).get("calibrated_confidence", raw_prob * 100.0) / 100.0
        
        # Check Grad-CAM status
        gradcam_status = "Success"
        if predict_res.get("demo_mode"):
            warnings.append("Demo mode fallback used (model weights missing?)")
            gradcam_status = "Demo Fallback"
            
        if predict_res.get("saliency_fallback"):
            warnings.append("Saliency map fallback triggered")
            gradcam_status = "Saliency Fallback"
            
        # Save Grad-CAM visualization
        vis_path = os.path.join(vis_dir, f"vis_{img_name}")
        h_cropped.save(vis_path)
        print(f"Saved visualization to {vis_path}")
        
        results.append({
            "Filename": img_name,
            "Prediction": prediction,
            "Raw probability": f"{raw_prob:.4f}",
            "Confidence": f"{calibrated_conf:.4f}",
            "Processing time": f"{elapsed:.4f}",
            "Grad-CAM status": gradcam_status,
            "Errors": "; ".join(errors) if errors else "None",
            "Warnings": "; ".join(warnings) if warnings else "None"
        })
        
        print(f"Result for {img_name}: {prediction} (Prob: {raw_prob:.4f}, Calibrated Conf: {calibrated_conf:.4f}) in {elapsed:.4f}s")
        
    except Exception as e:
        elapsed = time.time() - start_time
        errors.append(str(e))
        results.append({
            "Filename": img_name,
            "Prediction": "FAILED",
            "Raw probability": "0.0",
            "Confidence": "0.0",
            "Processing time": f"{elapsed:.4f}",
            "Grad-CAM status": "Error",
            "Errors": "; ".join(errors),
            "Warnings": "; ".join(warnings) if warnings else "None"
        })
        print(f"✗ Failed processing {img_name}: {e}")

# Write to CSV
with open(OUTPUT_CSV, mode='w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=["Filename", "Prediction", "Raw probability", "Confidence", "Processing time", "Grad-CAM status", "Errors", "Warnings"])
    writer.writeheader()
    writer.writerows(results)
    
print(f"\nWritten results to CSV: {OUTPUT_CSV}")
