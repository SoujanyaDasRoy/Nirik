import os
import sys
import time
from PIL import Image

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from core.inference import predict_image

def main():
    base_dir = r"C:\Users\sdroy\OneDrive\Desktop\Documents\Final Year Project"
    image_dir = os.path.join(base_dir, 'jaypee_india_dataset', 'images', 'images')

    # Gather images
    all_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
    all_files.sort()

    normal_files = [f for f in all_files if f.startswith('TEST_nx')]
    tb_files = [f for f in all_files if f.startswith('TEST_px')]

    # Take first 10 of each
    selected_normal = normal_files[:10]
    selected_tb = tb_files[:10]

    results = []
    errors = 0

    for file_name in selected_normal + selected_tb:
        image_path = os.path.join(image_dir, file_name)
        truth = 'Normal' if file_name.startswith('TEST_nx') else 'Tuberculosis'
        try:
            start = time.time()
            img = Image.open(image_path)
            result_dict, heatmap_img = predict_image(img)
            elapsed = time.time() - start

            prediction = result_dict.get('prediction')
            confidence = result_dict.get('confidence')  # TB probability
            threshold_used = result_dict.get('threshold_used')
            tb_prob = float(confidence) if confidence is not None else None
            normal_prob = 1.0 - tb_prob if tb_prob is not None else None
            raw_softmax = f"[{tb_prob:.6f}, {normal_prob:.6f}]" if tb_prob is not None and normal_prob is not None else ""
            heatmap_gen = 'Yes' if heatmap_img is not None else 'No'
            report_gen = 'Yes' if result_dict.get('report') else 'No'

            results.append({
                'file': file_name,
                'truth': truth,
                'prediction': prediction,
                'confidence': tb_prob,
                'tb_prob': tb_prob,
                'normal_prob': normal_prob,
                'raw_softmax': raw_softmax,
                'threshold': threshold_used,
                'heatmap': heatmap_gen,
                'report': report_gen,
                'runtime': elapsed
            })

            print(f"Processed {file_name}: truth={truth}, pred={prediction}, TB_prob={tb_prob:.4f}")
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            errors += 1
            continue

    # Compute metrics
    y_true = []
    y_pred = []
    y_score = []  # TB probability
    for r in results:
        if r['tb_prob'] is None:
            continue
        y_true.append(1 if r['truth'] == 'Tuberculosis' else 0)
        y_pred.append(1 if r['prediction'] == 'Tuberculosis' else 0)
        y_score.append(r['tb_prob'])

    if len(y_true) == 0:
        print("No valid results")
        return

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

    try:
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_true, y_score)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    except Exception as e:
        print(f"Error computing metrics: {e}")
        accuracy = precision = recall = f1 = roc_auc = specificity = None
        tn = fp = fn = tp = None

    # Print summary
    print("\n=== Evaluation Summary ===")
    print(f"Images tested: {len(results)}")
    print(f"Runtime errors: {errors}")
    if accuracy is not None:
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall (Sensitivity): {recall:.4f}")
        print(f"Specificity: {specificity:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    else:
        print("Metrics could not be computed.")

    miscount = sum(1 for r in results if r['truth'] != r['prediction'])
    print(f"Number of misclassified images: {miscount}")

    # Save CSV
    import csv
    output_csv = os.path.join(os.path.dirname(__file__), 'evaluation_results.csv')
    fieldnames = ['file', 'truth', 'prediction', 'confidence', 'tb_prob', 'normal_prob',
                  'raw_softmax', 'threshold', 'heatmap', 'report', 'runtime']
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"Results saved to: {output_csv}")

if __name__ == '__main__':
    main()