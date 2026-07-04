# Kaggle Notebook 2: Classifier Training (Segmented Data)

This notebook is broken down into cells. Copy and paste each block into a new cell in your Kaggle notebook.

### 📦 Datasets to Import
1. **The output from Notebook 1.** 
   *(In Kaggle, add data and search for your newly created dataset containing `segmented_dataset.zip`)*

---

### Cell 1: Imports and Setup
```python
import os, glob, warnings, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from torch.optim import Adam

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (roc_auc_score, f1_score,
                             confusion_matrix, classification_report,
                             precision_recall_curve, roc_curve)

warnings.filterwarnings("ignore")
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
os.makedirs("/kaggle/working/models", exist_ok=True)
```

### Cell 2: Data Loading
```python
IMG_SIZE   = 224
BATCH_SIZE = 32

# UPDATE THIS TO POINT TO YOUR IMPORTED DATASET FROM NOTEBOOK 1
SEGMENTED_DATA_ROOT = Path("/kaggle/input/your-segmented-dataset-name/segmented_dataset")

def load_simple_dataset(base_path, source_name):
    records = []
    if not base_path.exists(): return records
    for f in base_path.rglob("*.png"):
        # Labels: 1 for TB/Abnormal, 0 for Normal
        # Checks the whole folder path for "abnormal" or "tuberculosis"
        path_str = str(f).lower()
        label = 1 if ("abnormal" in path_str or "tuberculosis" in path_str) else 0
        records.append({"path": str(f), "label": label, "source": source_name, "patient_id": f"{source_name}_{f.stem}"})
    return records

print("Loading dataset paths...")
all_records = []
for source in ["nirt", "tb_database", "nih"]:
    all_records.extend(load_simple_dataset(SEGMENTED_DATA_ROOT / source, source))
df_all = pd.DataFrame(all_records)
print(f"Total images found: {len(df_all)}")

gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
train_val_idx, test_idx = next(gss.split(df_all, groups=df_all['patient_id']))
df_trainval = df_all.iloc[train_val_idx].reset_index(drop=True)
df_test = df_all.iloc[test_idx].reset_index(drop=True)

gss_val = GroupShuffleSplit(n_splits=1, test_size=0.176, random_state=42)
train_idx, val_idx = next(gss_val.split(df_trainval, groups=df_trainval['patient_id']))
df_train = df_trainval.iloc[train_idx].reset_index(drop=True)
df_val = df_trainval.iloc[val_idx].reset_index(drop=True)

print(f"Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")
```

### Cell 3: Datasets & Dataloaders
```python
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE + 20, IMG_SIZE + 20)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

class ChestXRayDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        
    def __len__(self): return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            img = Image.open(row["path"]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), 0)
        if self.transform: img = self.transform(img)
        return img, int(row["label"])

# NIRT is 16K (dominant), so weight it DOWN. Boost smaller datasets so they aren't drowned out.
DOMAIN_WEIGHTS = {"nirt": 0.5, "tb_database": 2.5, "nih": 2.0}
def make_weighted_sampler(df):
    counts  = df["label"].value_counts().to_dict()
    weights = [(1.0 / counts[row["label"]]) * DOMAIN_WEIGHTS.get(row["source"], 1.0) for _, row in df.iterrows()]
    return WeightedRandomSampler(torch.DoubleTensor(weights), len(weights), replacement=True)

NUM_WORKERS = 2
train_loader = DataLoader(ChestXRayDataset(df_train, train_transform), batch_size=BATCH_SIZE, sampler=make_weighted_sampler(df_train), num_workers=NUM_WORKERS)
val_loader   = DataLoader(ChestXRayDataset(df_val, val_transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader  = DataLoader(ChestXRayDataset(df_test, val_transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
```

### Cell 4: Model Architecture & Distillation Logic
```python
def build_teacher() -> nn.Module:
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(model.fc.in_features, 1))
    return model.to(DEVICE)

def build_student() -> nn.Module:
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    model.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(model.classifier.in_features, 1))
    return model.to(DEVICE)

class DistillationLoss(nn.Module):
    def __init__(self, temperature=4.0, alpha=0.7):
        super().__init__()
        self.T, self.alpha, self.bce = temperature, alpha, nn.BCEWithLogitsLoss()
    def forward(self, s_logits, t_logits, labels):
        hard_loss = self.bce(s_logits.squeeze(), labels.float())
        s_soft = torch.sigmoid(s_logits.squeeze() / self.T).clamp(1e-7, 1-1e-7)
        t_soft = torch.sigmoid(t_logits.squeeze() / self.T).clamp(1e-7, 1-1e-7)
        kl_loss = F.kl_div(torch.log(torch.stack([s_soft, 1 - s_soft], dim=1)), torch.stack([t_soft, 1 - t_soft], dim=1), reduction="batchmean") * (self.T ** 2)
        return self.alpha * kl_loss + (1 - self.alpha) * hard_loss

bce_loss = nn.BCEWithLogitsLoss()
```

### Cell 5: Training Loops
```python
def evaluate(model, loader):
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            probs = torch.sigmoid(model(imgs).squeeze()).cpu().numpy()
            all_probs.extend(probs.tolist() if probs.ndim > 0 else [float(probs)])
            all_labels.extend(labels.numpy().tolist())
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else float('nan')
    return auc, all_probs, all_labels

def train_one_epoch(model, loader, optimizer, criterion, teacher_model=None, distill_criterion=None):
    model.train()
    losses = []
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        s_logits = model(imgs).squeeze()
        if teacher_model:
            with torch.no_grad(): t_logits = teacher_model(imgs).squeeze()
            loss = distill_criterion(s_logits, t_logits, labels)
        else: loss = criterion(s_logits, labels.float())
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return np.mean(losses)

# --- EXECUTE TRAINING ---
# NOTE: To save time in testing, reduce NUM_EPOCHS.
student = build_student()
s_opt = Adam(student.parameters(), lr=1e-4, weight_decay=1e-4)

print("\nTraining Student...")
best_s_auc = 0.0
for epoch in range(1, 10 + 1):
    train_loss = train_one_epoch(student, train_loader, s_opt, bce_loss)
    vl_auc, _, _ = evaluate(student, val_loader)
    if vl_auc > best_s_auc:
        best_s_auc = vl_auc
        torch.save(student.state_dict(), "/kaggle/working/models/student_best.pth")
    print(f"Ep {epoch} | Loss: {train_loss:.4f} | Val AUC: {vl_auc:.4f}")

student.load_state_dict(torch.load("/kaggle/working/models/student_best.pth"))
```

### Cell 6: Visualizations (ROC, Confusion Matrix, Validation)
```python
student.eval()
test_auc, test_probs, test_labels = evaluate(student, test_loader)

# 1. Optimal Threshold
precisions, recalls, thresholds = precision_recall_curve(test_labels, test_probs)
target_recall = 0.95
idx = np.where(recalls >= target_recall)[0][-1]
idx = min(idx, len(thresholds) - 1)
optimal_thresh = thresholds[idx]

print(f"Optimal Threshold for 95% Recall: {optimal_thresh:.4f}")
test_preds = (np.array(test_probs) >= optimal_thresh).astype(int)

# 2. Confusion Matrix Plot
cm = confusion_matrix(test_labels, test_preds)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'TB'], yticklabels=['Normal', 'TB'])
plt.title(f'Confusion Matrix (Threshold: {optimal_thresh:.2f})')
plt.ylabel('Actual')
plt.xlabel('Predicted')

# 3. ROC Curve Plot
fpr, tpr, _ = roc_curve(test_labels, test_probs)
plt.subplot(1, 2, 2)
plt.plot(fpr, tpr, label=f'AUC = {test_auc:.4f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('Receiver Operating Characteristic')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.tight_layout()
plt.show()

print("\nClassification Report:")
print(classification_report(test_labels, test_preds, target_names=["Normal", "TB"]))
```

### Cell 7: Clinical Grad-CAM Visualization
```python
def get_gradcam(model, img_tensor):
    """Generate Grad-CAM heatmap from DenseNet121's final dense block."""
    gradients = []
    activations = []
    
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])
    def forward_hook(module, input, output):
        activations.append(output)
        
    target_layer = model.features.denseblock4.denselayer16.conv2
    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)
    
    model.eval()
    out = model(img_tensor)
    prob = torch.sigmoid(out).item()
    
    model.zero_grad()
    out.backward(torch.ones_like(out))
    
    grads = gradients[0].cpu().data.numpy()
    acts = activations[0].cpu().data.numpy()
    
    weights = np.mean(grads, axis=(2, 3))[0, :]
    cam = np.zeros(acts.shape[2:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * acts[0, i, :, :]
        
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = cam - np.min(cam)
    cam = cam / (np.max(cam) + 1e-8)
    
    h1.remove()
    h2.remove()
    return cam, prob

def create_lung_mask_for_gradcam(gray_img):
    """
    Create a binary lung mask from the segmented image.
    Since images are already cropped to the lung bounding box by Notebook 1,
    the lung region is wherever pixel intensity > 0 (non-black).
    """
    _, mask = cv2.threshold(gray_img, 10, 255, cv2.THRESH_BINARY)
    # Smooth edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def classify_lung_zones(cam, mask):
    """
    Divide the lung into 6 clinical zones (Upper/Mid/Lower × Left/Right)
    and report which zones have the highest activation.
    
    Standard radiology convention:
    - Upper zone: above 2nd rib (top 1/3)
    - Mid zone: 2nd to 4th rib (middle 1/3)
    - Lower zone: below 4th rib (bottom 1/3)
    """
    h, w = cam.shape
    mid_x = w // 2
    third_h = h // 3
    
    zones = {
        "Right Upper": (0, third_h, 0, mid_x),
        "Left Upper":  (0, third_h, mid_x, w),
        "Right Mid":   (third_h, 2*third_h, 0, mid_x),
        "Left Mid":    (third_h, 2*third_h, mid_x, w),
        "Right Lower": (2*third_h, h, 0, mid_x),
        "Left Lower":  (2*third_h, h, mid_x, w),
    }
    
    zone_scores = {}
    for name, (y1, y2, x1, x2) in zones.items():
        zone_cam = cam[y1:y2, x1:x2]
        zone_mask = mask[y1:y2, x1:x2]
        # Only count activation WITHIN the lung
        if zone_mask.sum() > 0:
            zone_scores[name] = float(np.mean(zone_cam[zone_mask > 0]))
        else:
            zone_scores[name] = 0.0
    
    return zone_scores, zones

def draw_clinical_gradcam(orig_img, cam, lung_mask, prob, true_label, ax):
    """
    Draw a clinically interpretable Grad-CAM visualization:
    - Heatmap masked to lung regions only (no shoulder/boundary artifacts)
    - Zone grid overlay with activation scores
    - Confidence percentage
    - Clinical finding summary
    """
    # Mask the Grad-CAM to lung regions only
    mask_norm = (lung_mask / 255.0).astype(np.float32)
    cam_masked = cam * mask_norm
    
    # Re-normalize after masking
    if cam_masked.max() > 0:
        cam_masked = cam_masked / cam_masked.max()
    
    # Create heatmap overlay
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_masked), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Zero out heatmap outside lung mask so only lungs are colored
    heatmap[lung_mask == 0] = orig_img[lung_mask == 0]
    
    superimposed = cv2.addWeighted(orig_img, 0.5, heatmap, 0.5, 0)
    
    # Get zone scores
    zone_scores, zones = classify_lung_zones(cam_masked, lung_mask)
    
    ax.imshow(superimposed)
    
    # Draw zone grid lines
    h, w = cam.shape
    mid_x = w // 2
    third_h = h // 3
    ax.axhline(y=third_h, color='white', linewidth=0.8, linestyle='--', alpha=0.6)
    ax.axhline(y=2*third_h, color='white', linewidth=0.8, linestyle='--', alpha=0.6)
    ax.axvline(x=mid_x, color='white', linewidth=0.8, linestyle='--', alpha=0.6)
    
    # Annotate each zone with its activation score
    for name, (y1, y2, x1, x2) in zones.items():
        score = zone_scores[name]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        color = 'red' if score > 0.4 else ('yellow' if score > 0.2 else 'white')
        ax.text(cx, cy, f"{score:.0%}", ha='center', va='center',
                fontsize=8, fontweight='bold', color=color,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6))
    
    # Title with prediction info
    pred_label = "TB Positive" if prob >= optimal_thresh else "Normal"
    true_str = "TB" if true_label == 1 else "Normal"
    correct = "✓" if (prob >= optimal_thresh) == (true_label == 1) else "✗"
    title_color = 'green' if correct == "✓" else 'red'
    
    ax.set_title(f"{correct} Pred: {pred_label} ({prob:.1%}) | True: {true_str}",
                 fontsize=10, fontweight='bold', color=title_color)
    
    # Find top activated zones
    top_zones = sorted(zone_scores.items(), key=lambda x: x[1], reverse=True)[:2]
    finding_text = f"Focus: {top_zones[0][0]} ({top_zones[0][1]:.0%})"
    if top_zones[1][1] > 0.2:
        finding_text += f", {top_zones[1][0]} ({top_zones[1][1]:.0%})"
    ax.set_xlabel(finding_text, fontsize=9, style='italic')
    ax.set_xticks([])
    ax.set_yticks([])

# === Generate clinical Grad-CAM for multiple samples ===
print("\nGenerating Clinical Grad-CAM Visualizations...")
print("="*60)

# Pick 3 TB-positive and 2 Normal samples for a comprehensive view
tb_indices = [i for i, lbl in enumerate(test_labels) if lbl == 1]
norm_indices = [i for i, lbl in enumerate(test_labels) if lbl == 0]

sample_indices = random.sample(tb_indices, min(3, len(tb_indices))) + \
                 random.sample(norm_indices, min(2, len(norm_indices)))

fig, axes = plt.subplots(1, len(sample_indices), figsize=(5 * len(sample_indices), 6))
if len(sample_indices) == 1: axes = [axes]

for ax, sample_idx in zip(axes, sample_indices):
    sample_path = df_test.iloc[sample_idx]["path"]
    orig_img = cv2.imread(sample_path)
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    gray_for_mask = cv2.cvtColor(orig_img, cv2.COLOR_RGB2GRAY)
    orig_img = cv2.resize(orig_img, (224, 224))
    gray_for_mask = cv2.resize(gray_for_mask, (224, 224))
    
    # Create lung mask from the already-segmented image
    lung_mask = create_lung_mask_for_gradcam(gray_for_mask)
    
    # Get Grad-CAM
    img_tensor = val_transform(Image.fromarray(orig_img)).unsqueeze(0).to(DEVICE)
    cam, prob = get_gradcam(student, img_tensor)
    
    true_label = test_labels[sample_idx]
    draw_clinical_gradcam(orig_img, cam, lung_mask, prob, true_label, ax)

fig.suptitle("Clinical Grad-CAM — Lung Zone Activation Map\n"
             "(Red zones = high model attention | Heatmap restricted to lung tissue only)",
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("/kaggle/working/clinical_gradcam.png", dpi=150, bbox_inches='tight')
plt.show()

# Print zone analysis summary
print("\n" + "="*60)
print("ZONE ACTIVATION SUMMARY")
print("="*60)
for sample_idx in sample_indices[:3]:  # Show details for TB samples
    sample_path = df_test.iloc[sample_idx]["path"]
    orig_img = cv2.imread(sample_path)
    gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (224, 224))
    lung_mask = create_lung_mask_for_gradcam(gray)
    
    img_tensor = val_transform(Image.fromarray(
        cv2.cvtColor(cv2.resize(orig_img, (224, 224)), cv2.COLOR_BGR2RGB)
    )).unsqueeze(0).to(DEVICE)
    cam, prob = get_gradcam(student, img_tensor)
    
    cam_masked = cam * (lung_mask / 255.0).astype(np.float32)
    if cam_masked.max() > 0: cam_masked = cam_masked / cam_masked.max()
    zone_scores, _ = classify_lung_zones(cam_masked, lung_mask)
    
    true_label = "TB" if test_labels[sample_idx] == 1 else "Normal"
    print(f"\nSample: {Path(sample_path).name} | True: {true_label} | Pred: {prob:.1%}")
    for zone, score in sorted(zone_scores.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(score * 20)
        flag = " ← HIGH" if score > 0.4 else ""
        print(f"  {zone:>12}: {score:5.1%} {bar}{flag}")

# Final Export
torch.save(student.state_dict(), "/kaggle/working/tb_student_densenet121.pt")
print("\nFinished! Download tb_student_densenet121.pt to your backend/ folder.")
```
