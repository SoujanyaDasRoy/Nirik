import os
import json
import shutil
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

results_dir = r"c:\Users\sdroy\OneDrive\Desktop\Documents\Final Year Project\results"
figures_dir = os.path.join(results_dir, "figures")
dest_dir = r"c:\Users\sdroy\OneDrive\Desktop\Documents\Final Year Project\frontend\public\model-results"

os.makedirs(dest_dir, exist_ok=True)

# Load metrics.json
with open(os.path.join(results_dir, "metrics.json"), "r") as f:
    metrics = json.load(f)

# 1. Copy Confusion Matrix
print("Copying Confusion Matrix...")
shutil.copy(
    os.path.join(figures_dir, "densenet121student_test_confusion_matrix.png"),
    os.path.join(dest_dir, "confusion_matrix.png")
)

# 2. Copy Sample Predictions (segmentation_samples)
print("Copying Sample Predictions...")
shutil.copy(
    os.path.join(figures_dir, "segmentation_samples.png"),
    os.path.join(dest_dir, "sample_predictions.png")
)

# 3. Copy Performance Comparison (teacher_vs_student_comparison)
print("Copying Performance Comparison...")
# We will generate the performance_comparison table image dynamically to match the original layout
# Model | Accuracy | AUC | F1 (TB) | Params (M) | Size (MB) | Inference (ms/img)
# Let's extract values
teacher_acc = metrics["teacher_val"]["accuracy"]  # wait, test or val? In the original table, teacher acc was 0.670 (which matches test accuracy? No, wait)
# Let's check teacher test metrics
teacher_test_acc = metrics["teacher_test"]["accuracy"]
teacher_test_auc = metrics["teacher_test"]["roc_auc"]
teacher_test_f1 = metrics["teacher_test"]["f1"]

student_test_acc = metrics["student_test"]["accuracy"]
student_test_auc = metrics["student_test"]["roc_auc"]
student_test_f1 = metrics["student_test"]["f1"]

teacher_params = metrics["teacher_params"] / 1e6
student_params = metrics["student_params"] / 1e6

# Disk sizes
teacher_size_mb = os.path.getsize(os.path.join(results_dir, "teacher_finetune_best.keras")) / (1024 * 1024)
student_size_mb = os.path.getsize(os.path.join(results_dir, "student_best.weights.h5")) / (1024 * 1024)

teacher_latency = metrics["teacher_cpu_latency"]["ms_per_image"]
student_latency = metrics["student_cpu_latency"]["ms_per_image"]

print(f"Teacher size: {teacher_size_mb:.3f} MB, Student size: {student_size_mb:.3f} MB")
print(f"Teacher latency: {teacher_latency:.3f} ms, Student latency: {student_latency:.3f} ms")

# Let's plot the Performance Comparison table
fig, ax = plt.subplots(figsize=(15, 2.5), dpi=300)
ax.axis('off')

# Data matrix
cell_data = [
    ["Teacher (ResNet50)", f"{teacher_test_acc:.3f}", f"{teacher_test_auc:.3f}", f"{teacher_test_f1:.3f}" if not np.isnan(teacher_test_f1) else "nan", f"{teacher_params:.3f}", f"{teacher_size_mb:.3f}", f"{teacher_latency:.3f}"],
    ["Student (DenseNet121)", f"{student_test_acc:.3f}", f"{student_test_auc:.3f}", f"{student_test_f1:.3f}", f"{student_params:.3f}", f"{student_size_mb:.3f}", f"{student_latency:.3f}"]
]
col_labels = ["Model", "Accuracy", "AUC", "F1 (TB)", "Params (M)", "Size (MB)", "Inference (ms/img)"]

# Create table
table = ax.table(
    cellText=cell_data,
    colLabels=col_labels,
    loc='center',
    cellLoc='center'
)

# Style table
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2)

# Color header and columns
# Original header: dark gray, text: white
# Student row: light blue tint
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_text_props(color='white', weight='bold')
        cell.set_facecolor('#404040')
    else:
        if row == 2:  # Student row
            cell.set_facecolor('#d9ecff')
        else:
            cell.set_facecolor('#ffffff')

plt.title("Performance Comparison: Teacher vs Student", y=1.1, fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig(os.path.join(dest_dir, "performance_comparison.png"), bbox_inches='tight', dpi=300)
plt.close()
print("Generated performance_comparison.png")

# 4. Generate Deployment Metrics Bar Charts side-by-side
print("Generating Deployment Metrics...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=300)
labels = ['Teacher\nResNet50', 'Student\nDenseNet121']
colors = ['#ff7f0e', '#1f77b4']

# Parameters
axes[0].bar(labels, [teacher_params, student_params], color=colors, width=0.6)
axes[0].set_title("Parameters", fontsize=12, weight='bold')
axes[0].set_ylabel("Parameters (M)", fontsize=10)
for i, v in enumerate([teacher_params, student_params]):
    axes[0].text(i, v + 0.5, f"{v:.2f}", ha='center', va='bottom', fontsize=10)

# Disk Size
axes[1].bar(labels, [teacher_size_mb, student_size_mb], color=colors, width=0.6)
axes[1].set_title("Disk Size", fontsize=12, weight='bold')
axes[1].set_ylabel("Size (MB)", fontsize=10)
for i, v in enumerate([teacher_size_mb, student_size_mb]):
    axes[1].text(i, v + 10, f"{v:.2f}", ha='center', va='bottom', fontsize=10)

# Inference Latency
axes[2].bar(labels, [teacher_latency, student_latency], color=colors, width=0.6)
axes[2].set_title("Inference Speed", fontsize=12, weight='bold')
axes[2].set_ylabel("Latency (ms/img)", fontsize=10)
for i, v in enumerate([teacher_latency, student_latency]):
    axes[2].text(i, v + 10, f"{v:.2f}", ha='center', va='bottom', fontsize=10)

plt.suptitle("Deployment Performance Metrics", y=1.05, fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig(os.path.join(dest_dir, "deployment_metrics.png"), bbox_inches='tight', dpi=300)
plt.close()
print("Generated deployment_metrics.png")

# 5. Combine ROC and PR Curves side-by-side
print("Combining ROC and PR curves...")
roc_img = Image.open(os.path.join(figures_dir, "densenet121student_test_roc_curve.png"))
pr_img = Image.open(os.path.join(figures_dir, "densenet121student_test_pr_curve.png"))

# Create combined image
w_roc, h_roc = roc_img.size
w_pr, h_pr = pr_img.size

# Target size: side-by-side
combined_w = w_roc + w_pr
combined_h = max(h_roc, h_pr)

combined_img = Image.new('RGB', (combined_w, combined_h), color='white')
combined_img.paste(roc_img, (0, 0))
combined_img.paste(pr_img, (w_roc, 0))

# Resize to match original aspect ratio roughly
combined_img.save(os.path.join(dest_dir, "roc_pr_curves.png"))
print("Generated roc_pr_curves.png")

# 6. Combine Grad-CAM explainability images into 2x4 grid
print("Combining Grad-CAM images...")
explain_imgs = []
for i in range(8):
    path = os.path.join(figures_dir, f"densenet121student_explain_{i:02d}.png")
    if os.path.exists(path):
        explain_imgs.append(Image.open(path))
    else:
        # Check fallback names without leading zeros
        path_fallback = os.path.join(figures_dir, f"densenet121student_explain_{i}.png")
        if os.path.exists(path_fallback):
            explain_imgs.append(Image.open(path_fallback))

if len(explain_imgs) >= 8:
    w_exp, h_exp = explain_imgs[0].size
    grid_w = w_exp * 4
    grid_h = h_exp * 2
    
    grid_img = Image.new('RGB', (grid_w, grid_h), color='white')
    
    for idx, img in enumerate(explain_imgs[:8]):
        col = idx % 4
        row = idx // 4
        grid_img.paste(img, (col * w_exp, row * h_exp))
        
    # Resize grid_img to keep file size reasonable (width 2400)
    target_width = 2400
    target_height = int(grid_h * (target_width / grid_w))
    grid_img_resized = grid_img.resize((target_width, target_height), Image.Resampling.LANCZOS)
    grid_img_resized.save(os.path.join(dest_dir, "gradcam_visualizations.png"))
    print("Generated gradcam_visualizations.png")
else:
    print(f"Warning: Only found {len(explain_imgs)} explain images, skipping gradcam_visualizations grid")

print("All visualizations updated successfully in frontend/public/model-results/!")
