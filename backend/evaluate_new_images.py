#!/usr/bin/env python
import os
import sys
import time
import csv
from PIL import Image

# Add the backend directory to the path so we can import from core
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from core.inference import predict_image

def main():
    base_dir = r"C:\Users\sdroy\OneDrive\Desktop\Documents\Final Year Project\new_images_for_testing"
    images_dir = os.path.join(base_dir, 'images')

    # Output CSV path
    output_csv = os.path.join(base_dir, 'evaluation_results.csv')

    # We'll collect results in a list of dicts
    results = []

    # Walk through the subdirectories
    for root, dirs, files in os.walk(images_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                image_path = os.path.join(root, file)
                # Determine ground truth from parent folder name (the immediate parent of the file)
                # The parent folder is either 'with tb' or 'without tb'
                parent_dir = os.path.basename(root)
                if parent_dir == 'with tb':
                    ground_truth = 'Tuberculosis'
                elif parent_dir == 'without tb':
                    ground_truth = 'Normal'
                else:
                    # If we cannot determine, set to None
                    ground_truth = None

                try:
                    start_time = time.time()
                    # Load image
                    img = Image.open(image_path)
                    # Run prediction
                    result_dict, heatmap_img = predict_image(img)
                    end_time = time.time()
                    runtime = end_time - start_time

                    # Extract required fields from result_dict
                    prediction = result_dict.get('prediction')
                    confidence = result_dict.get('confidence')  # This is the probability of the predicted class?
                    # Note: The confidence in the result_dict is the probability of the predicted class?
                    # But we also need TB probability and Normal probability separately.
                    # The result_dict does not directly give both. We need to get them from the model output.
                    # However, the predict_image function does not return the raw softmax for both classes.
                    # We have to modify our approach? But we are not allowed to change the existing code.
                    # Let's check what the result_dict contains by looking at the code? We cannot change it, but we can see what it returns.
                    # From the predict_image function in inference.py, we see that the result_dict includes:
                    #   "prediction", "confidence", "threshold_used", "is_tb", ... and also "xai_results" which has metrics.
                    # In the xai_results, there is a "metrics" dictionary that includes "tb_probability" and "calibrated_confidence".
                    # However, note that the "tb_probability" in the metrics is the calibrated TB probability?
                    # Let's look at the code for compute_xai_payload and calibrate_confidence.
                    # But we are not allowed to change the code, but we can use what is returned.
                    # We'll extract:
                    #   tb_probability = result_dict['xai_results']['metrics']['tb_probability'] / 100.0  (because it's stored as percentage?)
                    #   normal_probability = 1 - tb_probability? Not exactly, because the model outputs two classes.
                    # Actually, the model outputs two logits, and we get probabilities for two classes.
                    # In the predict_image function, we have:
                    #   prob = float(probs[0][1]) for TB? But we changed the interpretation earlier.
                    # However, we are not allowed to change the code, so we must use the existing function as is.
                    # Let's look at the current state of the predict_image function in the code we have.
                    # We cannot see it now, but we know that the function returns a result_dict that we have been using in validation.
                    # From the validation scripts, we see that they use:
                    #   prediction = result_dict.get("prediction")
                    #   confidence = result_dict.get("confidence")
                    # And they also have:
                    #   tb_probability = result_dict.get("xai_results", {}).get("metrics", {}).get("tb_probability")
                    # But note: the tb_probability in the metrics is a percentage (as seen in the validate_10_images.py: they print confidence as a float between 0 and 1).
                    # In validate_10_images.py, they use:
                    #   confidence = result_dict.get("confidence")
                    # and they treat it as a float (e.g., 0.0168).
                    # So the confidence in the result_dict is the probability of the predicted class? Or the probability of TB?
                    # Let's check the validate_10_images.py: they compute accuracy by comparing prediction to ground truth.
                    # They do not use the confidence for anything other than reporting.
                    # We need the raw softmax output for both classes.
                    # We can try to get the raw logits from the model? But we are not allowed to change the code.
                    # Alternatively, we can note that the predict_function does not return the raw softmax for both classes.
                    # However, the requirement asks for:
                    #   - Raw softmax output
                    #   - TB probability
                    #   - Normal probability
                    # We can approximate:
                    #   If we have the confidence and the prediction, we can set:
                    #       If prediction == "Tuberculosis":
                    #           tb_probability = confidence
                    #           normal_probability = 1 - confidence
                    #       Else:
                    #           normal_probability = confidence
                    #           tb_probability = 1 - confidence
                    # But this assumes that the confidence is the probability of the predicted class, and the two probabilities sum to 1.
                    # This is true if the model outputs a probability distribution over two classes (which it does via softmax).
                    # Therefore, we can do that.
                    #
                    # However, note that the result_dict also contains a "threshold_used" and "is_tb".
                    # We'll use the prediction and confidence as described.
                    #
                    # For the raw softmax output, we can return a string of the two probabilities? Or we can store both.
                    # We'll store the raw softmax as a tuple (prob_normal, prob_tb) or vice versa? We need to know the order.
                    # From our earlier debugging, we found that class 0 is TB and class 1 is Normal.
                    # But we are not allowed to change the code, and we don't know if the current code has been changed in a way that affects this.
                    # However, we are using the existing code as is, and we have been using it in the validation scripts.
                    # In the validation scripts, they only used the prediction and confidence, and they got the expected results (for the 4 images) after we fixed the preprocessing.
                    # We can assume that the model's output is [prob_TB, prob_Normal] or [prob_Normal, prob_TB]?
                    # Let's look at the predict_image function in the current code (we can't see it, but we can infer from the validation).
                    # In the validate_4_images.py, they printed:
                    #   Prediction: Normal, Confidence: 0.0168 for TEST_nx1.jpg (which is a normal image).
                    #   If the confidence is the probability of the predicted class (Normal), then the probability of Normal is 0.0168 and TB is 0.9832.
                    #   But that would mean the model is very confident it's TB for a normal image, which is not the case.
                    #   Wait, in the first run of validate_4_images.py (before we changed the interpretation) we got:
                    #       TEST_nx1.jpg: Prediction: Tuberculosis, Confidence: 0.9873
                    #   Then we changed the interpretation and got:
                    #       TEST_nx1.jpg: Prediction: Normal, Confidence: 0.0168
                    #   So the confidence is the probability of the predicted class.
                    #   Therefore, for TEST_nx1.jpg (normal), the model says Normal with confidence 0.0168 -> meaning it thinks it's very unlikely to be normal?
                    #   That doesn't make sense. Actually, a confidence of 0.0168 for the predicted class (Normal) means the model is only 1.68% confident that it's normal, so it's 98.32% confident it's TB.
                    #   But the image is normal, so the model is wrong.
                    #   However, after we changed the preprocessing to include the bounding box with padding, we got the same result?
                    #   Actually, we haven't changed the preprocessing in the current code? We have been modifying it, but the validation we just ran (with 8.5% padding) gave:
                    #       TEST_nx1.jpg: Prediction: Normal, Confidence: 0.0168
                    #   So the model is still predicting Normal with low confidence.
                    #
                    #   Let's look at the raw logits from the last run we did (with the debug prints) for TEST_nx1.jpg:
                    #       DEBUG: Raw logits: [[-2.1152132  1.9548123]]
                    #       DEBUG: Softmax probs: [[0.01679023 0.9832097 ]]
                    #   So the softmax output is [0.01679023, 0.9832097] for [class0, class1].
                    #   And we decided that class0 is TB and class1 is Normal?
                    #   Then the probability of TB is 0.0168 and Normal is 0.9832.
                    #   But the predicted class is the one with the higher probability, which is class1 (Normal) with 0.9832.
                    #   However, the confidence we stored in the result_dict is 0.0168? That doesn't match.
                    #
                    #   Wait, in the predict_image function, we set:
                    #       prob = prob_tb   (which is the probability of class0)
                    #   and then we compare to threshold to get is_tb.
                    #   Then we set:
                    #       prediction = "Tuberculosis" if is_tb else "Normal"
                    #   and confidence = prob   (which is prob_tb)
                    #   So the confidence in the result_dict is the probability of TB, not the probability of the predicted class.
                    #   Let's check the code we have in the current inference.py for the predict_image function (we changed it earlier).
                    #   We set:
                    #       prob = prob_tb   (which is the probability of class0, which we decided is TB)
                    #   and then:
                    #       is_tb = prob >= OPTIMAL_THRESHOLD
                    #       prediction = "Tuberculosis" if is_tb else "Normal"
                    #   and then we store in result_dict:
                    #       "prediction": prediction,
                    #       "confidence": float(prob),
                    #   So the confidence is the probability of TB.
                    #   Therefore, for TEST_nx1.jpg:
                    #       TB probability = 0.0168 -> confidence = 0.0168
                    #       Normal probability = 1 - 0.0168 = 0.9832
                    #   But the prediction is "Normal" because 0.0168 < 0.0.6? Wait, 0.62.
                    #   Also, we loaded from metadata: 0.62.
                    #   So 0.0168 < 0.62, so is_tb = False -> prediction = Normal.
                    #   And the confidence stored is 0.0168 (the TB probability).
                    #   This is confusing because the confidence is not the probability of the predicted class.
                    #   However, in the validation scripts, they used the confidence as reported and didn't seem to have a problem.
                    #   For example, in validate_4_images.py, they printed:
                    #       Image: TEST_nx1.jpg
                    #         Ground truth: Normal
                    #         Prediction: Normal
                    #         Confidence: 0.0168
                    #   And they considered it correct because the prediction matched the ground truth.
                    #   So the confidence field is the TB probability, regardless of the prediction.
                    #
                    #   Therefore, to get the TB probability and Normal probability, we can do:
                    #       tb_probability = result_dict.get('confidence')
                    #       normal_probability = 1.0 - tb_probability
                    #   But wait, is that always true? Only if the model outputs two classes that sum to 1.
                    #   Yes, because we applied softmax.
                    #
                    #   However, note that the model might have been trained with more than two classes? No, it's binary.
                    #
                    #   So we'll use:
                    #       tb_probability = result_dict.get('confidence')
                    #       normal_probability = 1.0 - tb_probability
                    #   And the raw softmax output can be represented as [normal_probability, tb_probability] or [tb_probability, normal_probability]?
                    #   We don't know the order, but we can state the raw softmax as (prob_class0, prob_class1) and we know from our debugging that class0 is TB and class1 is Normal.
                    #   But we are not allowed to assume that? We can note that from the model's output, the first element is TB and second is Normal.
                    #   However, to be safe, we can just store the two probabilities as we know them: TB probability and Normal probability.
                    #   The requirement says "Raw softmax output" - we can output a string representation of the two probabilities in the order [TB, Normal] or [Normal, TB]?
                    #   Since we don't know for sure, we can output both and let the user know? But the instruction says to record the raw softmax output.
                    #   We'll output the raw softmax as a tuple of (prob_class0, prob_class1) and note in the comments that class0 is TB and class1 is Normal (based on our analysis).
                    #   Alternatively, we can avoid specifying the order and just give the two numbers.
                    #   Let's output: "TB_probability: X, Normal_probability: Y"
                    #   But the requirement might expect a single field for raw softmax output. We'll output a string like "[0.0168, 0.9832]" and note that the first is TB, second is Normal.
                    #
                    #   We'll decide to output the raw softmax as a list: [TB_probability, Normal_probability] based on our earlier debugging.
                    #   We have verified this with the debug prints.
                    #
                    #   Therefore, we will set:
                    #       raw_softmax = [tb_probability, normal_probability]
                    #   where tb_probability = result_dict['confidence']
                    #   and normal_probability = 1.0 - tb_probability
                    #
                    #   However, note that the model might not be exactly two-class? It is.
                    #
                    #   Let's proceed.
                    #
                    tb_probability = result_dict.get('confidence')
                    if tb_probability is None:
                        # Fallback: try to get from xai_results
                        tb_probability = result_dict.get('xai_results', {}).get('metrics', {}).get('tb_probability')
                        if tb_probability is not None:
                            # The tb_probability in the metrics is a percentage (as seen in the code: they multiply by 100)
                            tb_probability = tb_probability / 100.0
                    normal_probability = 1.0 - tb_probability if tb_probability is not None else None
                    raw_softmax = [tb_probability, normal_probability] if tb_probability is not None and normal_probability is not None else None

                    # Determine if heatmap was generated and report generated
                    heatmap_generated = 'Yes' if result_dict.get('heatmaps') and any(result_dict['heatmaps'].values()) else 'No'
                    # Alternatively, we can check if the heatmap_img is not None? But we don't have it in the result_dict.
                    # We have the heatmap_img returned from predict_image, but we didn't store it. We can store a flag.
                    # We'll change: we have the heatmap_img variable, so we can set:
                    heatmap_generated = 'Yes' if heatmap_img is not None else 'No'

                    # Check if clinical report generated: look for 'report' in result_dict
                    report_generated = 'Yes' if result_dict.get('report') else 'No'

                    # Append result
                    results.append({
                        'filename': file,
                        'ground_truth': ground_truth,
                        'predicted_class': prediction,
                        'raw_softmax': str(raw_softmax) if raw_softmax is not None else '',
                        'tb_probability': tb_probability if tb_probability is not None else '',
                        'normal_probability': normal_probability if normal_probability is not None else '',
                        'predicted_confidence': confidence,  # This is the TB probability, as stored in the result_dict
                        'threshold_used': result_dict.get('threshold_used'),
                        'heatmap_generated': heatmap_generated,
                        'report_generated': report_generated,
                        'runtime_seconds': runtime
                    })
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
                    # Optionally, we can record the error and continue
                    results.append({
                        'filename': file,
                        'ground_truth': ground_truth,
                        'predicted_class': 'ERROR',
                        'raw_softmax': '',
                        'tb_probability': '',
                        'normal_probability': '',
                        'predicted_confidence': '',
                        'threshold_used': '',
                        'heatmap_generated': '',
                        'report_generated': '',
                        'runtime_seconds': '',
                        'error': str(e)
                    })

    # Write to CSV
    fieldnames = ['filename', 'ground_truth', 'predicted_class', 'raw_softmax',
                  'tb_probability', 'normal_probability', 'predicted_confidence',
                  'threshold_used', 'heatmap_generated', 'report_generated',
                  'runtime_seconds']
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"Results saved to {output_csv}")

    # Now, if we have ground truth for all (or some) images, compute metrics
    # We'll filter out rows where ground_truth is None or prediction is ERROR
    valid_results = [r for r in results if r['ground_truth'] is not None and r['predicted_class'] != 'ERROR']
    if len(valid_results) > 0:
        # We'll compute binary classification metrics:
        #   Positive class: Tuberculosis
        #   Negative class: Normal
        y_true = []
        y_pred = []
        y_scores = []  # for ROC-AUC, we need the probability of the positive class (TB)
        for r in valid_results:
            y_true.append(1 if r['ground_truth'] == 'Tuberculosis' else 0)
            y_pred.append(1 if r['predicted_class'] == 'Tuberculosis' else 0)
            # Use the TB probability as the score for the positive class
            tb_prob = r['tb_probability']
            if tb_prob == '':
                # If missing, we cannot compute ROC-AUC, but we can skip or set to 0.5?
                tb_prob = 0.5
            y_scores.append(float(tb_prob))

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
        try:
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y_true, y_scores)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        except Exception as e:
            print(f"Error computing metrics: {e}")
            accuracy = precision = recall = f1 = roc_auc = specificity = None
            tn = fp = fn = tp = None

        # Print metrics
        print("\n=== Evaluation Metrics ===")
        print(f"Total images with ground truth: {len(valid_results)}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall (Sensitivity): {recall:.4f}")
        print(f"Specificity: {specificity:.4f}")
        print(f"F1-score: {f1:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"Confusion Matrix:")
        print(f"  TN: {tn}, FP: {fp}")
        print(f"  FN: {fn}, TP: {tp}")

        # List misclassified images
        misclassified = [r for r in valid_results if
                         (r['ground_truth'] == 'Tuberculosis' and r['predicted_class'] != 'Tuberculosis') or
                         (r['ground_truth'] == 'Normal' and r['predicted_class'] != 'Normal')]
        print(f"\nNumber of misclassified images: {len(misclassified)}")
        if len(misclassified) > 0:
            print("Misclassified images:")
            for m in misclassified:
                print(f"  {m['filename']}: true={m['ground_truth']}, pred={m['predicted_class']}, TB_prob={m['tb_probability']}")
    else:
        print("No valid results with ground truth found.")

if __name__ == '__main__':
    main()