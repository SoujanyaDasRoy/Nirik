import sys
import os
from PIL import Image
import json

# Add the backend directory to the path so we can import from core
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from core.inference import predict_image

def main():
    # Define the base directory for the project
    base_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..')
    # Define the image directory
    image_dir = os.path.join(base_dir, 'jaypee_india_dataset', 'images', 'images')

    # List of images: 5 Normal and 5 TB
    # We'll use the first 5 Normal (TEST_nx1.jpg to TEST_nx5.jpg) and first 5 TB (TEST_px27.jpg to TEST_px31.jpg)
    image_files = [
        ("TEST_nx1.jpg", "Normal"),
        ("TEST_nx2.jpg", "Normal"),
        ("TEST_nx3.jpg", "Normal"),
        ("TEST_nx4.jpg", "Normal"),
        ("TEST_nx5.jpg", "Normal"),
        ("TEST_px27.jpg", "Tuberculosis"),
        ("TEST_px28.jpg", "Tuberculosis"),
        ("TEST_px29.jpg", "Tuberculosis"),
        ("TEST_px30.jpg", "Tuberculosis"),
        ("TEST_px31.jpg", "Tuberculosis")
    ]

    results = []
    errors = 0

    for file_name, truth in image_files:
        image_path = os.path.join(image_dir, file_name)
        if not os.path.exists(image_path):
            print(f"ERROR: Image not found: {image_path}")
            errors += 1
            continue

        try:
            # Load image
            img = Image.open(image_path)
            # Run inference
            result_dict, heatmap_img = predict_image(img)

            # Extract required information
            prediction = result_dict.get("prediction")
            confidence = result_dict.get("confidence")
            # Check if heatmap_gradcam_plusplus exists (we'll consider any heatmap as generated)
            heatmap_generated = "Yes" if result_dict.get("heatmaps", {}).get("gradcam_plusplus") else "No"
            # Check if clinical report exists
            report_generated = "Yes" if result_dict.get("report") else "No"

            results.append({
                "file": file_name,
                "truth": truth,
                "prediction": prediction,
                "confidence": confidence,
                "heatmap": heatmap_generated,
                "report": report_generated
            })

            # Print per-image result
            print(f"Image: {file_name}")
            print(f"  Ground truth: {truth}")
            print(f"  Prediction: {prediction}")
            print(f"  Confidence: {confidence:.4f}")
            print(f"  Heatmap generated: {heatmap_generated}")
            print(f"  Clinical report generated: {report_generated}")
            print()

        except Exception as e:
            print(f"ERROR processing {file_name}: {e}")
            errors += 1
            continue

    # Calculate summary metrics
    total = len(results)
    if total > 0:
        normal_correct = sum(1 for r in results if r["truth"] == "Normal" and r["prediction"] == "Normal")
        tb_correct = sum(1 for r in results if r["truth"] == "Tuberculosis" and r["prediction"] == "Tuberculosis")
        overall_accuracy = (normal_correct + tb_correct) / total
        avg_confidence = sum(r["confidence"] for r in results) / total
    else:
        normal_correct = 0
        tb_correct = 0
        overall_accuracy = 0.0
        avg_confidence = 0.0

    # Print summary
    print("SUMMARY:")
    print(f"Total images: {total}")
    print(f"Normal correct: {normal_correct}")
    print(f"TB correct: {tb_correct}")
    print(f"Overall accuracy: {overall_accuracy:.4f}")
    print(f"Average confidence: {avg_confidence:.4f}")
    print(f"Runtime errors: {errors}")

    # Return the results for potential use (but we are just printing)
    return {
        "results": results,
        "summary": {
            "total": total,
            "normal_correct": normal_correct,
            "tb_correct": tb_correct,
            "overall_accuracy": overall_accuracy,
            "avg_confidence": avg_confidence,
            "errors": errors
        }
    }

if __name__ == "__main__":
    main()