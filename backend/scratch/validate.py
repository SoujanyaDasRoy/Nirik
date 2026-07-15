#!/usr/bin/env python3
import sys
import os
import csv
from PIL import Image
import numpy as np

# Add the backend directory to path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, backend_dir)
# Also add core for direct imports
sys.path.insert(0, os.path.join(backend_dir, 'core'))

from inference import get_model, predict_image, OPTIMAL_THRESHOLD
# ensure explainability can be imported
# from explainability.gradcam import generate_saliency_heatmap  # not needed

BASE_IMG_DIR = os.path.join(backend_dir, '..', 'jaypee_india_dataset', 'images', 'images')
METADATA_CSV = os.path.join(backend_dir, '..', 'jaypee_india_dataset', 'jaypee_metadata.csv')

# Load metadata
ground_truth = {}
with open(METADATA_CSV, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        fname = row['study_id']
        label = row['findings']
        gt = 1 if label.lower() == 'tuberculosis' else 0
        ground_truth[fname] = gt

# Select 10 of each
normals = [f for f, gt in ground_truth.items() if gt == 0][:10]
tb_cases = [f for f, gt in ground_truth.items() if gt == 1][:10]
selected = normals + tb_cases
print(f"Selected {len(normals)} normal and {len(tb_cases)} TB images.")

results = []
model = get_model()
if model is None:
    print("ERROR: Model not loaded.")
    sys.exit(1)

for fname in selected:
    img_path = os.path.join(BASE_IMG_DIR, fname)
    if not os.path.exists(img_path):
        print(f"WARNING: Image not found: {img_path}")
        continue
    img = Image.open(img_path).convert('L')  # ensure grayscale
    # predict_image expects PIL Image (grayscale or RGB)
    result_dict, _ = predict_image(img)
    pred_label = result_dict['prediction']
    confidence = result_dict['confidence']
    # Map prediction to binary
    pred = 1 if pred_label == 'Tuberculosis' else 0
    gt = ground_truth[fname]
    # GradCAM generated?
    gradcam_gen = bool(result_dict.get('heatmaps', {}).get('gradcam++'))
    results.append({
        'filename': fname,
        'ground_truth': gt,
        'prediction': pred,
        'confidence': confidence,
        'gradcam_generated': gradcam_gen
    })
    print(f"{fname}: GT={'TB' if gt else 'Norm'}, Pred={'TB' if pred else 'Norm'}, Conf={confidence:.3f}, GradCAM={'Yes' if gradcam_gen else 'No'}")

# Compute metrics
y_true = [r['ground_truth'] for r in results]
y_pred = [r['prediction'] for r in results]

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
try:
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
except Exception as e:
    print(f"Error computing metrics: {e}")
    sys.exit(1)

print("\n=== Metrics ===")
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1-score: {f1:.4f}")
print("Confusion Matrix:")
print(cm)
print("Label order: [0=Normal, 1=TB]")

# Also compute per-class metrics
print("\nPer-class:")
print(f"Normal: TP={cm[0,0]}, FN={cm[0,1]}, FP={cm[1,0]}, TN={cm[1,1]}")
print(f"TB: TP={cm[1,1]}, FN={cm[1,0]}, FP={cm[0,1]}, TN={cm[0,0]}")

# GradCAM stats
gradcam_count = sum(1 for r in results if r['gradcam_generated'])
print(f"\nGradCAM generated for {gradcam_count}/{len(results)} images.")