import os
import shutil

results_dir = r"c:\Users\sdroy\OneDrive\Desktop\Documents\Final Year Project\results"
hf_space_dir = r"c:\Users\sdroy\OneDrive\Desktop\Documents\Final Year Project\hf_space"
backend_dir = r"c:\Users\sdroy\OneDrive\Desktop\Documents\Final Year Project\backend"

files_from_results = [
    "attention_unet.keras",
    "densenet121_student.onnx",
    "student_best.weights.h5",
    "metrics.json"
]

print("Syncing files to legacy backend/ directory...")
for f in files_from_results:
    src = os.path.join(results_dir, f)
    if os.path.exists(src):
        shutil.copy(src, os.path.join(backend_dir, f))
        print(f"  Copied {f} from results/ to backend/")

# Copy model_metadata.json from hf_space/ to backend/
shutil.copy(
    os.path.join(hf_space_dir, "model_metadata.json"),
    os.path.join(backend_dir, "model_metadata.json")
)
print("  Copied model_metadata.json from hf_space/ to backend/")
