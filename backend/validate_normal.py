import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from core.inference import predict_image
from PIL import Image

def main():
    base_dir = os.path.join("C:", os.sep, "Users", "sdroy", "OneDrive", "Desktop", "Documents", "Final Year Project", "jaypee_india_dataset", "images", "images")
    metadata_path = os.path.join("C:", os.sep, "Users", "sdroy", "OneDrive", "Desktop", "Documents", "Final Year Project", "jaypee_india_dataset", "jaypee_metadata.csv")
    print(f"Metadata path: {metadata_path}")
    print(f"Base dir: {base_dir}")
    # Get first 10 normal images based on metadata
    import csv
    normal_files = []
    with open(metadata_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['findings'] == 'False':
                normal_files.append(row['study_id'])
                if len(normal_files) >= 10:
                    break
    print(f"Found {len(normal_files)} normal images: {normal_files}")

    results = []
    for fname in normal_files:
        img_path = os.path.join(base_dir, fname)
        print(f"\nProcessing {fname}...")
        img = Image.open(img_path)
        try:
            result, heatmap = predict_image(img)
            pred = result.get('prediction')
            conf = result.get('confidence')
            gradcam = 'Yes' if result.get('heatmaps', {}).get('gradcam_plusplus') else 'No'
            report = 'Yes' if result.get('report') else 'No'
            print(f"  Prediction: {pred}, Confidence: {conf:.4f}, Grad-CAM: {gradcam}, Report: {report}")
            results.append({
                'file': fname,
                'prediction': pred,
                'confidence': conf,
                'gradcam': gradcam,
                'report': report
            })
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                'file': fname,
                'prediction': 'ERROR',
                'confidence': 0.0,
                'gradcam': 'No',
                'report': 'No',
                'error': str(e)
            })

    # Summary
    total = len(results)
    correct = sum(1 for r in results if r['prediction'] == 'Normal')
    incorrect = total - correct
    confs = [r['confidence'] for r in results if isinstance(r['confidence'], float)]
    avg_conf = sum(confs)/len(confs) if confs else 0
    errors = [r for r in results if 'error' in r]

    print("\n=== SUMMARY ===")
    print(f"Total images tested: {total}")
    print(f"Correct predictions (Normal): {correct}")
    print(f"Incorrect predictions: {incorrect}")
    print(f"Average confidence: {avg_conf:.4f}")
    print(f"Runtime errors: {len(errors)}")
    if errors:
        for e in errors:
            print(f"  {e['file']}: {e.get('error')}")
    print(f"Backend stable for NORMAL cases? {'YES' if len(errors)==0 and incorrect==0 else 'NO'}")

if __name__ == '__main__':
    main()