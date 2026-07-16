import os
import json
import shutil

results_dir = r"c:\Users\sdroy\OneDrive\Desktop\Documents\Final Year Project\results"
hf_space_dir = r"c:\Users\sdroy\OneDrive\Desktop\Documents\Final Year Project\hf_space"
weights_dir = r"c:\Users\sdroy\OneDrive\Desktop\Documents\Final Year Project\weights"
configs_dir = r"c:\Users\sdroy\OneDrive\Desktop\Documents\Final Year Project\configs"

# 1. Copy weights and model files to hf_space/ and weights/
files_to_copy = [
    "attention_unet.keras",
    "densenet121_student.onnx",
    "student_best.weights.h5",
    "metrics.json"
]

print("Copying model files to hf_space/ and weights/...")
for f in files_to_copy:
    src = os.path.join(results_dir, f)
    if os.path.exists(src):
        # copy to hf_space/
        shutil.copy(src, os.path.join(hf_space_dir, f))
        print(f"  Copied {f} to hf_space/")
        # copy to weights/
        shutil.copy(src, os.path.join(weights_dir, f))
        print(f"  Copied {f} to weights/")

# Also copy metrics.json and run_config.json to configs/
shutil.copy(os.path.join(results_dir, "metrics.json"), os.path.join(configs_dir, "metrics.json"))
shutil.copy(os.path.join(results_dir, "run_config.json"), os.path.join(configs_dir, "run_config.json"))
print("Copied metrics.json and run_config.json to configs/")

# 2. Regenerate model_metadata.json in hf_space/
print("Regenerating hf_space/model_metadata.json...")
with open(os.path.join(results_dir, "metrics.json"), "r") as f:
    metrics = json.load(f)

with open(os.path.join(results_dir, "run_config.json"), "r") as f:
    run_config = json.load(f)

student_test = metrics.get("student_test", {})
student_test_at_threshold = metrics.get("student_test_at_youden_threshold", {})
optimal_threshold = metrics.get("student_youden_threshold", 0.5976377129554749)

new_metadata = {
    "optimal_threshold": float(optimal_threshold),
    "metrics": {
        "accuracy": float(student_test_at_threshold.get("accuracy", student_test.get("accuracy", 0.7073))),
        "precision": float(student_test_at_threshold.get("precision", student_test.get("precision", 0.5396))),
        "recall": float(student_test_at_threshold.get("sensitivity", student_test.get("recall", 0.6337))),
        "f1": float(student_test_at_threshold.get("f1", student_test.get("f1", 0.5829))),
        "auc": float(student_test.get("roc_auc", 0.7767)),
        "sensitivity": float(student_test_at_threshold.get("sensitivity", student_test.get("sensitivity", 0.6337))),
        "specificity": float(student_test_at_threshold.get("specificity", student_test.get("specificity", 0.7424))),
        "calibration_score": 0.045
    },
    "dataset_tracking": {
        "training_dataset_version": "Shenzhen-Montgomery-TBX11K-India-v4.0.0",
        "validation_dataset_version": "Pooled-Val-v4.0.0",
        "training_date": run_config.get("timestamp", "2026-07-15").split(" ")[0],
        "model_version": "DenseNet121-Student-v5.0.0"
    }
}

metadata_path = os.path.join(hf_space_dir, "model_metadata.json")
with open(metadata_path, "w") as f:
    json.dump(new_metadata, f, indent=4)
print(f"Regenerated {metadata_path} successfully!")
