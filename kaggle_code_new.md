# TB Chest X-Ray Classification — Kaggle Notebook (Cell-Split Version)
#
# Each fenced ```python block is a separate Kaggle cell. Copy each block
# into its own cell in your Kaggle notebook (top to bottom, in order).

# ============================================================================
# CELL 1 — Imports, Mixed Precision, Seeds, Paths, GPU Check
# ============================================================================
```python
import os, glob, hashlib, random
import numpy as np
import pandas as pd
import tensorflow as tf
import pydicom
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupShuffleSplit
from tensorflow.keras import layers, models, optimizers, losses, metrics
from tensorflow.keras import mixed_precision

# Enable mixed precision training for faster epochs and lower memory usage
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

IMG_SIZE = 224
BATCH = 32
AUTOTUNE = tf.data.AUTOTUNE

BASE = "/kaggle/input/datasets"
PATHS = {
    "montgomery":  f"{BASE}/raddar/tuberculosis-chest-xrays-montgomery/images",
    "shenzhen":    f"{BASE}/raddar/tuberculosis-chest-xrays-shenzhen/images",
    "tb_database": f"{BASE}/tawsifurrahman/tuberculosis-tb-chest-xray-dataset/TB_Chest_Radiography_Database",
    "nirt_dicom":  f"{BASE}/projectmantra/nirt-india-chest-x-ray-dicom-dataset",
}
for k, v in PATHS.items():
    print(k, "->", "OK" if os.path.isdir(v) else "MISSING", v)
# Cap GPU memory growth — prevents TF from grabbing all VRAM at once (helps P100 OOM)
gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass

print("GPU:", tf.config.list_physical_devices('GPU'))
```

# ============================================================================
# CELL 2 — Collect records from all 4 datasets
# ============================================================================
```python
records = []

def patient_id(p):
    base = os.path.splitext(os.path.basename(p))[0]
    parts = base.split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else base

# 1) Montgomery + Shenzhen: label from filename suffix _0 / _1
for name in ["montgomery", "shenzhen"]:
    root = PATHS[name]
    if not os.path.isdir(root):
        print("MISSING:", name, root); continue
    for f in glob.glob(os.path.join(root, "**", "*.*"), recursive=True):
        if not f.lower().endswith((".png", ".jpg", ".jpeg")): continue
        stem = os.path.splitext(os.path.basename(f))[0]
        if   stem.endswith("_1"): lbl = 1
        elif stem.endswith("_0"): lbl = 0
        else: continue
        records.append({"path": f, "label": lbl,
                        "group": f"{name}:{patient_id(f)}", "is_dicom": False})

# 2) TB database: label from folder name (Normal / Tuberculosis)
tb_root = PATHS["tb_database"]
if os.path.isdir(tb_root):
    for f in glob.glob(os.path.join(tb_root, "**", "*.*"), recursive=True):
        if not f.lower().endswith((".png", ".jpg", ".jpeg")): continue
        pl = f.lower()
        if   "/tuberculosis/" in pl: lbl = 1
        elif "/normal/" in pl:       lbl = 0
        else: continue
        # filename is unique per image -> use it as the group id
        gid = os.path.splitext(os.path.basename(f))[0]
        records.append({"path": f, "label": lbl,
                        "group": f"tb_db:{gid}", "is_dicom": False})

# 3) NIRT DICOM: ext is .dicom, label from path, group = patient folder
nirt = PATHS["nirt_dicom"]
if os.path.isdir(nirt):
    for f in glob.glob(os.path.join(nirt, "**", "*"), recursive=True):
        if os.path.isdir(f): continue
        if not f.lower().endswith((".dicom", ".dcm")): continue
        pl = f.lower()
        if   os.sep + "abnormal" + os.sep in pl: lbl = 1
        elif os.sep + "normal"   + os.sep in pl: lbl = 0
        else: continue
        # patient folder is the parent dir name (e.g. intbtr124) -> group key
        pid = os.path.basename(os.path.dirname(f))
        records.append({"path": f, "label": lbl,
                        "group": f"nirt:{pid}", "is_dicom": True})
```

# ============================================================================
# CELL 3 — Train U-Net Lung Segmenter on Montgomery + Shenzhen
# ============================================================================
```python
import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, metrics
from PIL import Image

# 1) Scan /kaggle/input/ recursively to locate images and masks
print("Scanning /kaggle/input/ recursively for image and mask directories...")
seg_images_dirs = []
seg_masks_dirs = []
mont_left_mask_dirs = []
mont_right_mask_dirs = []

for root, dirs, files in os.walk("/kaggle/input"):
    root_clean = root.replace("\\", "/").lower()
    
    # Match CXR_png or images directories for segmentation
    # Note: exclude classification directories (like normal/abnormal/Tuberculosis) to avoid pollution
    if root_clean.endswith("/cxr_png") or root_clean.endswith("/lung segmentation/cxr_png") or root_clean.endswith("/data/cxr_png"):
        seg_images_dirs.append(root)
    elif root_clean.endswith("/images") and "tuberculosis-chest-xrays" in root_clean:
        seg_images_dirs.append(root)
    # Match masks directories
    elif (root_clean.endswith("/masks") or root_clean.endswith("/mask")) and "left" not in root_clean and "right" not in root_clean:
        seg_masks_dirs.append(root)
    # Match Montgomery Set Left/Right masks
    elif "leftmask" in root_clean or "left_mask" in root_clean:
        mont_left_mask_dirs.append(root)
    elif "rightmask" in root_clean or "right_mask" in root_clean:
        mont_right_mask_dirs.append(root)

print("Resolved image directories:", seg_images_dirs)
print("Resolved mask directories:", seg_masks_dirs)
print("Resolved left mask directories:", mont_left_mask_dirs)
print("Resolved right mask directories:", mont_right_mask_dirs)

seg_pairs = []
matched_bases = set()

# First attempt: Pair images from CXR_png/images with combined masks from masks/ folders
for img_dir in seg_images_dirs:
    # Exclude classification-only datasets (like TB_Chest_Radiography_Database)
    if "tuberculosis-tb-chest-xray-dataset" in img_dir.lower() or "nirt-india-chest-x-ray-dicom-dataset" in img_dir.lower():
        continue
    for f in glob.glob(os.path.join(img_dir, "*.png")):
        base = os.path.basename(f)
        if base in matched_bases:
            continue
        stem = os.path.splitext(base)[0]
        
        # Look for a matching mask in any masks directory
        mask_path = None
        for mask_dir in seg_masks_dirs:
            for candidate in [f"{stem}_mask.png", f"{stem}.png", f"{stem}_mask.PNG", f"{stem}.PNG", base]:
                p = os.path.join(mask_dir, candidate)
                if os.path.exists(p):
                    mask_path = p
                    break
            if mask_path:
                break
        
        if mask_path:
            seg_pairs.append({
                "image_path": f,
                "mask_type": "combined",
                "mask_path": mask_path
            })
            matched_bases.add(base)

# Second attempt: Check if we have Montgomery images and left/right masks
if mont_left_mask_dirs and mont_right_mask_dirs:
    for img_dir in seg_images_dirs:
        if "montgomery" in img_dir.lower():
            for f in glob.glob(os.path.join(img_dir, "*.png")):
                base = os.path.basename(f)
                if base in matched_bases:
                    continue
                
                # Check for left and right masks
                lp_found, rp_found = None, None
                for lp_dir in mont_left_mask_dirs:
                    p = os.path.join(lp_dir, base)
                    if os.path.exists(p):
                        lp_found = p; break
                for rp_dir in mont_right_mask_dirs:
                    p = os.path.join(rp_dir, base)
                    if os.path.exists(p):
                        rp_found = p; break
                
                if lp_found and rp_found:
                    seg_pairs.append({
                        "image_path": f,
                        "mask_type": "montgomery",
                        "left_mask_path": lp_found,
                        "right_mask_path": rp_found
                    })
                    matched_bases.add(base)

print(f"Found {len(seg_pairs)} image-mask pairs for U-Net training.")

# Debug listing and validation check
if len(seg_pairs) == 0:
    print("\n[DEBUG] Listing first 3 levels of /kaggle/input directory tree:")
    for root, dirs, files in os.walk("/kaggle/input"):
        depth = root.replace("/kaggle/input", "").count(os.sep)
        if depth <= 3:
            print(f"{'  ' * depth}📂 {os.path.basename(root) or root}/")
            for d in dirs[:10]:
                print(f"{'  ' * (depth + 1)}📁 {d}")
            if len(dirs) > 10:
                print(f"{'  ' * (depth + 1)}... and {len(dirs) - 10} more dirs")
    raise ValueError(
        "Could not automatically resolve all image and mask paths for U-Net training. "
        "Please ensure you have added nikhilpandey360/chest-xray-masks-and-labels dataset "
        "to your notebook. See the printed directory listing above to debug."
    )

# 2) Preload and preprocess images and masks to memory (fast epochs)
def pad_to_square_pil(img: Image.Image, fill=0) -> Image.Image:
    w, h = img.size
    if w == h:
        return img
    elif w > h:
        result = Image.new(img.mode, (w, w), fill)
        result.paste(img, (0, (w - h) // 2))
        return result
    else:
        result = Image.new(img.mode, (h, h), fill)
        result.paste(img, ((h - w) // 2, 0))
        return result

seg_images = []
seg_masks = []

for pair in seg_pairs:
    try:
        img = Image.open(pair["image_path"]).convert("L")
        img_padded = pad_to_square_pil(img, fill=0)
        img_resized = img_padded.resize((256, 256), Image.Resampling.BILINEAR)
        img_arr = np.array(img_resized, dtype=np.float32) / 255.0
        
        if pair["mask_type"] == "montgomery":
            lm = Image.open(pair["left_mask_path"]).convert("L")
            rm = Image.open(pair["right_mask_path"]).convert("L")
            mask = Image.fromarray(np.maximum(np.array(lm), np.array(rm)))
        else:
            mask = Image.open(pair["mask_path"]).convert("L")
            
        mask_padded = pad_to_square_pil(mask, fill=0)
        mask_resized = mask_padded.resize((256, 256), Image.Resampling.NEAREST)
        mask_arr = np.array(mask_resized, dtype=np.float32) / 255.0
        mask_arr = (mask_arr > 0.5).astype(np.float32)
        
        seg_images.append(img_arr[..., np.newaxis])
        seg_masks.append(mask_arr[..., np.newaxis])
    except Exception as e:
        print(f"Skipping pair due to error loading: {pair['image_path']}, Error: {e}")

X_seg = np.array(seg_images)
y_seg = np.array(seg_masks)
print("Loaded segmentation data shapes:", X_seg.shape, y_seg.shape)

# 3) Split into train/validation sets and build tf.data pipeline with augmentation
indices = np.arange(len(X_seg))
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
np.random.shuffle(indices)

split_idx = int(0.85 * len(indices))
train_idx, val_idx = indices[:split_idx], indices[split_idx:]

X_train_seg, y_train_seg = X_seg[train_idx], y_seg[train_idx]
X_val_seg, y_val_seg = X_seg[val_idx], y_seg[val_idx]

train_seg_ds = tf.data.Dataset.from_tensor_slices((X_train_seg, y_train_seg))
val_seg_ds = tf.data.Dataset.from_tensor_slices((X_val_seg, y_val_seg))

def augment_seg(image, mask):
    concat = tf.concat([image, mask], axis=-1)
    concat = tf.image.random_flip_left_right(concat, seed=SEED)
    return concat[..., :1], concat[..., 1:]

train_seg_ds = train_seg_ds.shuffle(128).map(augment_seg).batch(16).prefetch(tf.data.AUTOTUNE)
val_seg_ds = val_seg_ds.batch(16).prefetch(tf.data.AUTOTUNE)


# 4) Build U-Net Architecture
def build_unet(input_shape=(256, 256, 1)):
    inputs = layers.Input(shape=input_shape)
    
    # Contracting path (encoder)
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    
    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    
    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)
    
    c4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)
    
    # Bottleneck
    c5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c5)
    
    # Expanding path (decoder)
    u6 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c6)
    
    u7 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c7)
    
    u8 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c8)
    
    u9 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c9)
    
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid', dtype='float32')(c9)
    
    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model

def dice_coef(y_true, y_pred):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + 1.0) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + 1.0)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    return losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

# 5) Compile and train
unet = build_unet()
unet.compile(
    optimizer=optimizers.Adam(1e-4),
    loss=bce_dice_loss,
    metrics=[dice_coef, 'accuracy']
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
]

print("Training U-Net Lung Segmenter...")
history_unet = unet.fit(
    train_seg_ds,
    validation_data=val_seg_ds,
    epochs=20,
    callbacks=callbacks,
    verbose=2
)

# Save U-Net model
unet.save('/kaggle/working/unet_lung_segmenter.keras')
print("Saved U-Net model to /kaggle/working/unet_lung_segmenter.keras")

# 6) Visual Verification of U-Net predictions
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
for i in range(min(3, len(val_idx))):
    idx = val_idx[i]
    img = X_seg[idx]
    gt_mask = y_seg[idx]
    pred_mask = unet.predict(img[np.newaxis, ...], verbose=0)[0]
    
    plt.subplot(3, 3, i*3 + 1)
    plt.imshow(img.squeeze(), cmap='gray')
    plt.title("Image")
    plt.axis('off')
    
    plt.subplot(3, 3, i*3 + 2)
    plt.imshow(gt_mask.squeeze(), cmap='gray')
    plt.title("Ground Truth")
    plt.axis('off')
    
    plt.subplot(3, 3, i*3 + 3)
    plt.imshow(pred_mask.squeeze(), cmap='gray')
    plt.title("Predicted Mask")
    plt.axis('off')
plt.tight_layout()
plt.savefig('/kaggle/working/unet_validation_samples.png', dpi=150)
plt.show()
```

# ============================================================================
# CELL 4 — Pre-segment all classification images and save as PNGs
# ============================================================================
```python
import os
import cv2
import hashlib
import numpy as np
import pydicom
from PIL import Image

os.makedirs("/kaggle/working/segmented_images", exist_ok=True)

# Helper function to pad numpy array to square (preserves aspect ratio)
def pad_image_to_square(img_arr, fill_val=0):
    h, w = img_arr.shape[:2]
    if h == w:
        return img_arr
    elif h < w:
        pad_top = (w - h) // 2
        pad_bottom = w - h - pad_top
        padded = np.pad(img_arr, ((pad_top, pad_bottom), (0, 0)), mode='constant', constant_values=fill_val)
        return padded
    else:
        pad_left = (h - w) // 2
        pad_right = h - w - pad_left
        padded = np.pad(img_arr, ((0, 0), (pad_left, pad_right)), mode='constant', constant_values=fill_val)
        return padded

def pre_segment_image(rec, unet_model):
    path = rec["path"]
    is_dicom = rec["is_dicom"]
    
    # 1. Read image as grayscale
    if is_dicom:
        ds = pydicom.dcmread(path)
        arr = ds.pixel_array.astype(np.float32)
        if str(getattr(ds, "PhotometricInterpretation", "")) == "MONOCHROME1":
            arr = arr.max() - arr
        arr = arr - arr.min()
        if arr.max() > 0:
            arr = arr / arr.max()
        arr = (arr * 255.0).astype(np.uint8)
    else:
        arr = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if arr is None:
            raise ValueError(f"Failed to read standard image: {path}")
            
    # 2. Pad to square
    padded = pad_image_to_square(arr, fill_val=0)
    h_pad, w_pad = padded.shape[:2]
    
    # 3. Prepare for U-Net (256x256, normalized to [0,1])
    img_norm = cv2.resize(padded, (256, 256)).astype(np.float32) / 255.0
    seg_tensor = img_norm[np.newaxis, :, :, np.newaxis] # (1, 256, 256, 1)
    
    # 4. Predict mask
    pred = unet_model.predict(seg_tensor, verbose=0)
    pred_np = pred[0, :, :, 0] # (256, 256)
    binary_mask = (pred_np > 0.5).astype(np.uint8)
    
    # 5. Scale mask back to the padded image resolution
    mask_full = cv2.resize(binary_mask, (w_pad, h_pad), interpolation=cv2.INTER_NEAREST)
    
    # 6. Apply mask (bitwise AND)
    masked = cv2.bitwise_and(padded, padded, mask=mask_full)
    
    # 7. Resize to classification input size (224x224)
    final_img = cv2.resize(masked, (224, 224))
    
    # 8. Save image as PNG
    file_id = hashlib.md5(path.encode()).hexdigest()
    out_path = f"/kaggle/working/segmented_images/{file_id}.png"
    cv2.imwrite(out_path, final_img)
    
    return out_path

# Execute masking loop on the 4 classification datasets
print("Starting pre-segmentation masking loop on classification datasets...")
import time
start_time = time.time()
success_count = 0
failed_count = 0

for i, rec in enumerate(records):
    try:
        new_path = pre_segment_image(rec, unet)
        records[i]["path"] = new_path
        records[i]["is_dicom"] = False # now standard PNG
        success_count += 1
    except Exception as ex:
        print(f"Error segmenting {rec['path']}: {ex}")
        failed_count += 1
        # Fallback: pad to square, resize to 224x224, save without U-Net mask
        try:
            path = rec["path"]
            if rec["is_dicom"]:
                ds = pydicom.dcmread(path)
                arr = ds.pixel_array.astype(np.float32)
                if str(getattr(ds, "PhotometricInterpretation", "")) == "MONOCHROME1":
                    arr = arr.max() - arr
                arr = arr - arr.min()
                if arr.max() > 0: arr = arr / arr.max()
                arr = (arr * 255.0).astype(np.uint8)
            else:
                arr = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            padded = pad_image_to_square(arr, fill_val=0)
            final_img = cv2.resize(padded, (224, 224))
            file_id = hashlib.md5(path.encode()).hexdigest()
            out_path = f"/kaggle/working/segmented_images/{file_id}.png"
            cv2.imwrite(out_path, final_img)
            records[i]["path"] = out_path
            records[i]["is_dicom"] = False
            success_count += 1
        except Exception as ex2:
            print(f"Fallback failed for {rec['path']}: {ex2}")
            pass

print(f"Pre-segmentation complete: {success_count} success, {failed_count} fallback/failed.")
print(f"Time elapsed: {time.time() - start_time:.2f} seconds")
```

# ============================================================================
# CELL 5 — Build the master DataFrame + sanity check
# ============================================================================

```python
df = pd.DataFrame(records)
assert len(df) > 0, "No records found - check PATHS in Cell 1"
print(df.groupby(["is_dicom", "label"]).size())
print("\nPer-source counts:")
print(df["group"].str.split(":").str[0].value_counts())
print("\nTotal:", len(df), "| Unlabeled:", df['label'].isna().sum())
```

# ============================================================================
# CELL 6 — Image readers + content-hash deduplication
# ============================================================================
```python
def read_dicom(path):
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array.astype(np.float32)
    if str(getattr(ds, "PhotometricInterpretation", "")) == "MONOCHROME1":
        arr = arr.max() - arr
    arr = arr - arr.min()
    if arr.max() > 0: arr = arr / arr.max()
    arr = (arr * 255.0).astype(np.uint8)
    img = cv2.resize(arr, (IMG_SIZE, IMG_SIZE))
    return np.stack([img]*3, axis=-1)

def read_standard(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return np.stack([img]*3, axis=-1)

def load_image(path, is_dicom):
    return read_dicom(path) if is_dicom else read_standard(path)

def file_hash(path, is_dicom):
    img = load_image(path, is_dicom)
    return hashlib.md5(img.tobytes()).hexdigest()

df = df.dropna(subset=["label"]).reset_index(drop=True)
df["label"] = df["label"].astype(int)

# Content-hash dedup (kills duplicate/near-identical images across datasets)
df["hash"] = [file_hash(p, d) for p, d in zip(df["path"], df["is_dicom"])]
before = len(df)
df = df.drop_duplicates(subset="hash").reset_index(drop=True)
print(f"Removed {before-len(df)} duplicate images, {len(df)} remain")
```

# ============================================================================
# CELL 7 — Patient-group-aware 80/10/10 split
# ============================================================================
```python
from sklearn.model_selection import train_test_split

# Build one row per patient group with its dominant label
grp = (df.groupby("group")
         .agg(label=("label", lambda s: int(s.mean() >= 0.5)),
              n=("label", "size"))
         .reset_index())

# Split groups (stratified by group label): 80/10/10
g_train, g_temp = train_test_split(grp, test_size=0.20,
                                   stratify=grp["label"], random_state=SEED)
g_val, g_test   = train_test_split(g_temp, test_size=0.50,
                                   stratify=g_temp["label"], random_state=SEED)

train_df = df[df["group"].isin(g_train["group"])].reset_index(drop=True)
val_df   = df[df["group"].isin(g_val["group"])].reset_index(drop=True)
test_df  = df[df["group"].isin(g_test["group"])].reset_index(drop=True)

# Zero patient overlap
assert not (set(train_df.group) & set(val_df.group))
assert not (set(train_df.group) & set(test_df.group))
assert not (set(val_df.group)   & set(test_df.group))

print("Split sizes -> train:", len(train_df), "val:", len(val_df), "test:", len(test_df))
for nm, fr in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
    print(f"\n{nm} label %:\n", fr["label"].value_counts(normalize=True).round(3))
```

# ============================================================================
# CELL 8 — tf.data pipelines with augmentation + cache
# ============================================================================
```python
# ── Bug 2 fix: teacher (ResNet50) and student (DenseNet121) need DIFFERENT
# preprocessing. We build two separate pipeline factories:
#   make_ds_resnet  → resnet50.preprocess_input  (caffe mode: BGR mean subtract)
#   make_ds_densenet → densenet.preprocess_input  (torch mode: /255 + mean/std)
# Both read the same raw images; only the normalisation differs.

def _build_ds(frame, preprocess_fn, training=False):
    """Shared pipeline builder — callers pass the correct preprocess_fn."""
    paths  = frame["path"].values
    dicoms = frame["is_dicom"].values.astype(np.int32)
    labels = frame["label"].values.astype(np.float32)

    def _load(path, is_dicom, label):
        img = tf.numpy_function(
            lambda p, d: load_image(p.decode(), bool(d)).astype(np.float32),
            [path, is_dicom], tf.float32)
        img.set_shape([IMG_SIZE, IMG_SIZE, 3])
        img = preprocess_fn(img)
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((paths, dicoms, labels))
    ds = ds.map(_load, num_parallel_calls=AUTOTUNE)

    # NOTE (P100): ds.cache() materialises the whole set in CPU RAM,
    # which can OOM during the error-analysis gallery. Re-enable on T4/A100.
    # ds = ds.cache()

    if training:
        ds = ds.shuffle(2048, seed=SEED)
        aug = tf.keras.Sequential([layers.RandomFlip("horizontal"),
                                   layers.RandomRotation(0.05),
                                   layers.RandomZoom(0.1)])
        ds = ds.map(lambda x, y: (aug(x, training=True), y),
                    num_parallel_calls=AUTOTUNE)
    return ds.batch(BATCH).prefetch(AUTOTUNE)

def make_ds_resnet(frame, training=False):
    """Pipeline for the ResNet50 teacher — caffe-style preprocessing."""
    return _build_ds(frame, tf.keras.applications.resnet50.preprocess_input, training)

def make_ds_densenet(frame, training=False):
    """Pipeline for the DenseNet121 student — torch-style preprocessing."""
    return _build_ds(frame, tf.keras.applications.densenet.preprocess_input, training)

# Teacher datasets (ResNet50 preprocessing)
train_ds_t = make_ds_resnet(train_df, training=True)
val_ds_t   = make_ds_resnet(val_df)
test_ds_t  = make_ds_resnet(test_df)

# Student datasets (DenseNet121 correct preprocessing)
train_ds = make_ds_densenet(train_df, training=True)
val_ds   = make_ds_densenet(val_df)
test_ds  = make_ds_densenet(test_df)
```

# ============================================================================
# CELL 9 — Build teacher (ResNet50), class weights, train
# ============================================================================
```python
from sklearn.utils.class_weight import compute_class_weight

# Build model: ImageNet base + dropout + single logit head
def build_model(base_fn):
    base = base_fn(include_top=False, weights="imagenet",
                   input_shape=(IMG_SIZE, IMG_SIZE, 3), pooling="avg")
    # Force every BatchNorm in the backbone to float32 — prevents the float16 BN
    # activation OOM that P100 hit inside `BatchNormalization.call`.
    float32_policy = tf.keras.mixed_precision.Policy('float32')
    for layer in base.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer._dtype_policy = float32_policy
    x = layers.Dropout(0.3)(base.output)
    out = layers.Dense(1, dtype="float32")(x)  # logits, no activation, float32 for mixed precision stability
    return models.Model(base.input, out)

# Class weights to correct train imbalance (~36% TB)
cw = compute_class_weight("balanced", classes=np.array([0, 1]),
                          y=train_df["label"].values)
class_weight = {0: float(cw[0]), 1: float(cw[1])}
print("Class weights:", class_weight)

# Note: do NOT wrap the model in a custom tf.distribute.Strategy scope here.
# Kaggle's training loop (model.fit) uses the default strategy, and mixing
# strategies raises RuntimeError. The BN-to-float32 pinning in build_model()
# is enough on its own to fix the P100 OOM.
teacher = build_model(tf.keras.applications.ResNet50)
teacher.compile(optimizer=optimizers.Adam(1e-4),
                loss=losses.BinaryCrossentropy(from_logits=True),
                metrics=[metrics.BinaryAccuracy(threshold=0.0, name="acc"),
                         metrics.AUC(from_logits=True, name="auc")])

teacher_history = teacher.fit(train_ds_t, validation_data=val_ds_t, epochs=10,
            class_weight=class_weight,
            callbacks=[tf.keras.callbacks.EarlyStopping(
                monitor="val_auc", mode="max", patience=3,
                restore_best_weights=True)],
            verbose=2)

print("\nTeacher test eval:")
teacher_test_eval_early = teacher.evaluate(test_ds_t)

teacher.save("/kaggle/working/teacher_resnet50.keras")
print("Teacher checkpoint saved to /kaggle/working/teacher_resnet50.keras")

```

# ============================================================================
# CELL 10 — Knowledge Distillation: Distiller class + student (DenseNet121)
# ============================================================================
```python
class Distiller(models.Model):
    def __init__(self, student, teacher, alpha=0.5, T=3.0):
        super().__init__()
        self.student, self.teacher = student, teacher
        self.alpha, self.T = alpha, T
        self.bce = losses.BinaryCrossentropy(from_logits=True)

    def compile(self, optimizer, **kw):
        super().compile(optimizer=optimizer, **kw)
        self.acc_metric = metrics.BinaryAccuracy(threshold=0.0, name="acc")
        self.auc_metric = metrics.AUC(from_logits=True, name="auc")

    @property
    def metrics(self):
        return [self.acc_metric, self.auc_metric]

    def _kd_loss(self, t_logits, s_logits):
        t  = tf.sigmoid(t_logits / self.T)
        s  = tf.math.log_sigmoid(s_logits / self.T)
        s0 = tf.math.log_sigmoid(-s_logits / self.T)
        return tf.reduce_mean(-(t * s + (1 - t) * s0)) * (self.T ** 2)

    def train_step(self, data):
        x, y = data
        t_logits = self.teacher(x, training=False)
        with tf.GradientTape() as tape:
            s_logits = self.student(x, training=True)
            hard = self.bce(y, s_logits)
            soft = self._kd_loss(t_logits, s_logits)
            loss = self.alpha * hard + (1 - self.alpha) * soft
        grads = tape.gradient(loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.student.trainable_variables))
        self.acc_metric.update_state(y, s_logits)
        self.auc_metric.update_state(y, s_logits)
        return {"loss": loss, "acc": self.acc_metric.result(), "auc": self.auc_metric.result()}

    def test_step(self, data):
        x, y = data
        s_logits = self.student(x, training=False)
        val_loss = self.bce(y, s_logits)
        self.acc_metric.update_state(y, s_logits)
        self.auc_metric.update_state(y, s_logits)
        return {"loss": val_loss,
                "acc":  self.acc_metric.result(),
                "auc":  self.auc_metric.result()}

teacher.trainable = False
student = build_model(tf.keras.applications.DenseNet121)
distiller = Distiller(student, teacher, alpha=0.5, T=3.0)
distiller.compile(optimizer=optimizers.Adam(1e-4))

history = distiller.fit(train_ds, validation_data=val_ds, epochs=15,
                        callbacks=[tf.keras.callbacks.EarlyStopping(
                            monitor="val_auc", mode="max", patience=4,
                            restore_best_weights=True)],
                        verbose=2)

student.save("/kaggle/working/tb_student_densenet121.keras")
with open("/kaggle/working/best_threshold.txt", "w") as f:
    f.write("0.5")
print("Student checkpoint + placeholder threshold saved to /kaggle/working/")

```

# ============================================================================
# CELL 11 — Threshold tuning on validation, then test-set evaluation
# ============================================================================
```python
from sklearn.metrics import (classification_report, roc_auc_score,
                             confusion_matrix, f1_score, roc_curve, precision_recall_curve)

def collect(ds):
    yt, yp = [], []
    for x, y in ds:
        yt.extend(y.numpy())
        yp.extend(tf.sigmoid(student(x, training=False)).numpy().ravel())
    return np.array(yt), np.array(yp)

val_y, val_p = collect(val_ds)
thresholds = np.linspace(0.05, 0.95, 91)
best_t = max(thresholds, key=lambda t: f1_score(val_y, (val_p > t).astype(int)))
print(f"Best threshold (val, max F1): {best_t:.3f}")

test_y, test_p = collect(test_ds)
test_pred = (test_p > best_t).astype(int)

print("\nStudent Test AUC:", round(roc_auc_score(test_y, test_p), 4))
print("\nAt tuned threshold:")
print(classification_report(test_y, test_pred,
                            target_names=["Normal", "TB"], digits=3))
print("Confusion matrix [rows=true, cols=pred]:")
print(confusion_matrix(test_y, test_pred))

print("\nAt default 0.5 threshold:")
print(classification_report(test_y, (test_p > 0.5).astype(int),
                            target_names=["Normal", "TB"], digits=3))

```

# ============================================================================
# CELL 12 — Plot functions
# ============================================================================
```python
def _save_and_show(save_path, dpi=300):
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    try:
        from IPython.display import display
        display(plt.gcf())
    except Exception:
        pass
    plt.close()

def plot_training_curves(hist):
    epochs = range(1, len(hist.history['loss']) + 1)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    ax1.plot(epochs, hist.history['loss'], 'b-o', label='Training Loss')
    if 'val_loss' in hist.history:
        ax1.plot(epochs, hist.history['val_loss'], 'r-o', label='Validation Loss')
    ax1.set_title('Training & Validation Loss'); ax1.legend(); ax1.grid(True)
    if 'auc' in hist.history:
        ax2.plot(epochs, hist.history['auc'], 'b-o', label='Training AUC')
    if 'val_auc' in hist.history:
        ax2.plot(epochs, hist.history['val_auc'], 'r-o', label='Validation AUC')
    ax2.set_title('Training & Validation AUC'); ax2.legend(); ax2.grid(True)
    if 'acc' in hist.history:
        ax3.plot(epochs, hist.history['acc'], 'b-o', label='Training Accuracy')
    if 'val_acc' in hist.history:
        ax3.plot(epochs, hist.history['val_acc'], 'r-o', label='Validation Accuracy')
    ax3.set_title('Training & Validation Accuracy'); ax3.legend(); ax3.grid(True)
    if 'val_loss' in hist.history:
        vloss = np.array(hist.history['val_loss'])
        vloss_norm = (vloss - vloss.min()) / (vloss.max() - vloss.min() + 1e-8)
        ax4.plot(epochs, vloss_norm, 'm-o', label='Val Loss (norm)')
    if 'val_auc' in hist.history:
        ax4.plot(epochs, hist.history['val_auc'], 'g-o', label='Val AUC')
    ax4.set_title('Validation Summary'); ax4.legend(); ax4.grid(True)
    plt.tight_layout()
    _save_and_show('/kaggle/working/training_curves.png')

def plot_confusion_matrix_heatmap(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.ylabel('Actual Label'); plt.xlabel('Predicted Label'); plt.title('Confusion Matrix')
    _save_and_show('/kaggle/working/confusion_matrix.png')

def plot_roc_pr_curves(y_true, y_probs, optimal_thresh):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc_score = roc_auc_score(y_true, y_probs)
    precision, recall, thresholds_pr = precision_recall_curve(y_true, y_probs)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {auc_score:.4f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--'); ax1.grid(True)
    ax2.plot(recall, precision, color='blue', lw=2, label='Precision-Recall')
    idx = np.argmin(np.abs(thresholds_pr - optimal_thresh))
    ax2.plot(recall[idx], precision[idx], 'ro', label=f'Thresh={optimal_thresh:.3f}')
    ax2.grid(True); _save_and_show('/kaggle/working/roc_pr_curves.png')

def generate_gradcam_overlay(model, img_arr, last_conv_layer_name=None):
    if last_conv_layer_name is None:
        for candidate in ("relu", "conv5_block3_out"):
            try:
                _ = model.get_layer(candidate)
                last_conv_layer_name = candidate
                break
            except ValueError: continue
    img_tensor = tf.expand_dims(img_arr, axis=0)
    grad_model = tf.keras.models.Model(inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        loss = predictions[0]
    grads = tape.gradient(loss, conv_outputs)[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = tf.reduce_sum(conv_outputs[0] * weights, axis=-1).numpy()
    cam = np.maximum(cam, 0)
    cam_min, cam_max = cam.min(), cam.max()
    if cam_max > cam_min: cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
    cam = np.asarray(cam).squeeze()
    _dn_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _dn_std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    if cam.ndim != 2 or cam.size == 0:
        gray = np.full((224, 224), 128, dtype=np.uint8)
        color_heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        color_heatmap_rgb = cv2.cvtColor(color_heatmap, cv2.COLOR_BGR2RGB)
        img_restore = np.clip((img_arr * _dn_std + _dn_mean) * 255.0, 0, 255).astype(np.uint8)
        return cv2.addWeighted(img_restore, 0.6, color_heatmap_rgb, 0.4, 0)
    cam = np.ascontiguousarray(cam, dtype=np.float32)
    cam_resized = cv2.resize(cam, (224, 224))
    heatmap = np.uint8(255 * cam_resized)
    color_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    color_heatmap_rgb = cv2.cvtColor(color_heatmap, cv2.COLOR_BGR2RGB)

    # Restore visible image from DenseNet-preprocessed input
    # densenet.preprocess_input: /255 then (x - mean) / std  → reverse with *std + mean, *255
    _dn_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _dn_std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_restore = img_arr.copy()
    img_restore = img_restore * _dn_std + _dn_mean  # undo normalisation
    img_restore = np.clip(img_restore * 255.0, 0, 255).astype(np.uint8)

    # Pixel-wise alpha blending: heatmap only shows where activations are strong.
    # Avoids the dark-blue JET colormap flooding low-activation background regions.
    max_alpha = 0.6
    alpha_map = cam_resized * max_alpha          # shape (224, 224), range [0, max_alpha]
    alpha_3ch = np.stack([alpha_map, alpha_map, alpha_map], axis=-1).astype(np.float32)
    orig_f = img_restore.astype(np.float32)
    heat_f = color_heatmap_rgb.astype(np.float32)
    blended = np.clip(orig_f * (1.0 - alpha_3ch) + heat_f * alpha_3ch, 0, 255).astype(np.uint8)
    return blended

def list_saved_plots():
    """Print everything currently on /kaggle/working — useful after a crash
    to see what survived without re-running the whole pipeline."""
    import os
    print("\n/kaggle/working/ contents:")
    for f in sorted(os.listdir("/kaggle/working")):
        path = os.path.join("/kaggle/working", f)
        if os.path.isfile(path):
            print(f"  {os.path.getsize(path):>10,d} bytes  {f}")

def plot_error_analysis_gallery(test_ds, model, optimal_thresh):
    # P100-safe version: stream predictions batch-by-batch, never feed the whole
    # test set as one tensor into the model. Also prints ONE Grad-CAM warning
    # instead of a wall of identical ones if the overlay fails.
    import gc
    all_labels, all_probs, all_imgs = [], [], []
    for batch_x, batch_y in test_ds:
        p = tf.sigmoid(model(batch_x, training=False)).numpy().ravel()
        all_probs.append(p)
        all_labels.append(batch_y.numpy())
        all_imgs.append(batch_x.numpy())          # keep one batch at a time

    labels = np.concatenate(all_labels)
    probs  = np.concatenate(all_probs)
    preds  = (probs >= optimal_thresh).astype(int)

    tp_idx = np.where((labels == 1) & (preds == 1))[0]
    tn_idx = np.where((labels == 0) & (preds == 0))[0]
    fp_idx = np.where((labels == 0) & (preds == 1))[0]
    fn_idx = np.where((labels == 1) & (preds == 0))[0]

    categories = [
        ('True Positives (Correct TB)', tp_idx, 'red'),
        ('True Negatives (Correct Normal)', tn_idx, 'green'),
        ('False Positives (Normal read as TB)', fp_idx, 'orange'),
        ('False Negatives (TB Missed - Critical!)', fn_idx, 'magenta')
    ]

    # Flatten once into a single np array (still large but unavoidable for indexing)
    raw_images = np.concatenate(all_imgs, axis=0)
    del all_imgs
    gc.collect()

    fig, axes = plt.subplots(4, 3, figsize=(12, 16))
    rng = np.random.default_rng(SEED)
    gradcam_failed_once = False   # <-- single-warning flag

    for r_idx, (title, indices, color) in enumerate(categories):
        if len(indices) == 0:
            for c_idx in range(3):
                ax = axes[r_idx, c_idx]
                ax.text(0.5, 0.5, "No Cases", ha="center", va="center")
                ax.axis('off')
            continue

        selected = rng.choice(indices, min(3, len(indices)), replace=False)

        for c_idx in range(3):
            ax = axes[r_idx, c_idx]
            if c_idx < len(selected):
                idx = int(selected[c_idx])
                img_arr = raw_images[idx]
                prob = probs[idx]

                # Generate Grad-CAM one image at a time — no big batched gradient
                try:
                    overlay = generate_gradcam_overlay(model, img_arr)
                    ax.imshow(overlay)
                except Exception as e_gc:
                    img_norm = (img_arr - img_arr.min()) / (img_arr.max() - img_arr.min() + 1e-8)
                    ax.imshow(img_norm)
                    if not gradcam_failed_once:
                        print(f"Grad-CAM overlay generation failed (showing grayscale fallback). First error: {e_gc}")
                        gradcam_failed_once = True

                ax.set_title(f"Prob: {prob:.3f}", color=color, fontsize=10)
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center")
            ax.axis('off')

        axes[r_idx, 0].text(-0.2, 0.5, title, rotation=90, va='center', ha='right',
                            transform=axes[r_idx, 0].transAxes,
                            fontsize=11, fontweight='bold')

    plt.tight_layout()
    # Save AND display inline (Kaggle renderer needs display() to capture the figure)
    _save_and_show('/kaggle/working/error_analysis_gallery.png', dpi=150)

    del raw_images
    gc.collect()

def plot_sample_predictions(test_ds, model, optimal_thresh, n_per_row=4):
    """4-row montage of test images with predicted/true labels.
    Row 0: Normal cases (any confidence)
    Row 1: TB cases (any confidence)
    Row 2: high-confidence wrong predictions (model's blind spots)
    Row 3: low-confidence correct predictions (model's hesitation)"""
    import gc
    all_labels, all_probs, all_imgs = [], [], []
    for batch_x, batch_y in test_ds:
        p = tf.sigmoid(model(batch_x, training=False)).numpy().ravel()
        all_probs.append(p); all_labels.append(batch_y.numpy()); all_imgs.append(batch_x.numpy())
    labels = np.concatenate(all_labels)
    probs  = np.concatenate(all_probs)
    preds  = (probs >= optimal_thresh).astype(int)

    rng = np.random.default_rng(SEED)
    rows = []

    # Row 0: Normal cases
    norm_idx = np.where(labels == 0)[0]
    rows.append(("Normal cases", rng.choice(norm_idx, min(n_per_row, len(norm_idx)), replace=False), 'green'))
    # Row 1: TB cases
    tb_idx = np.where(labels == 1)[0]
    rows.append(("TB cases", rng.choice(tb_idx, min(n_per_row, len(tb_idx)), replace=False), 'red'))
    # Row 2: high-confidence wrong (prob > 0.9 or < 0.1, but pred != label)
    wrong_mask = (preds != labels)
    wrong_idx = np.where(wrong_mask)[0]
    if len(wrong_idx):
        conf = np.where(probs[wrong_idx] >= 0.5, probs[wrong_idx], 1 - probs[wrong_idx])
        top = wrong_idx[np.argsort(-conf)[:n_per_row]]
    else:
        top = np.array([], dtype=int)
    rows.append(("High-confidence errors", top, 'orange'))
    # Row 3: low-confidence correct (0.4 < prob < 0.6, but pred == label)
    mid_mask = (preds == labels) & (np.abs(probs - 0.5) < 0.1)
    mid_idx = np.where(mid_mask)[0]
    if len(mid_idx):
        mid = rng.choice(mid_idx, min(n_per_row, len(mid_idx)), replace=False)
    else:
        mid = np.array([], dtype=int)
    rows.append(("Low-confidence correct", mid, 'purple'))

    raw_images = np.concatenate(all_imgs, axis=0)
    del all_imgs; gc.collect()

    fig, axes = plt.subplots(len(rows), n_per_row, figsize=(3*n_per_row, 3*len(rows)))
    if len(rows) == 1:
        axes = axes[np.newaxis, :]

    for r_idx, (title, indices, color) in enumerate(rows):
        for c_idx in range(n_per_row):
            ax = axes[r_idx, c_idx]
            if c_idx < len(indices):
                idx = int(indices[c_idx])
                img_arr = raw_images[idx]
                # Restore visible image from DenseNet-preprocessed input
                _dn_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                _dn_std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                img_show = np.clip((img_arr.copy() * _dn_std + _dn_mean) * 255.0, 0, 255).astype(np.uint8)
                ax.imshow(img_show)
                true_lbl = "TB" if labels[idx] == 1 else "Normal"
                pred_lbl = "TB" if preds[idx] == 1 else "Normal"
                correct = preds[idx] == labels[idx]
                mark = "OK" if correct else "X"
                ax.set_title(f"True: {true_lbl} | Pred: {pred_lbl} ({probs[idx]:.2f}) {mark}",
                             color=color, fontsize=9)
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center")
            ax.axis('off')
        axes[r_idx, 0].text(-0.25, 0.5, title, rotation=90, va='center', ha='right',
                            transform=axes[r_idx, 0].transAxes,
                            fontsize=11, fontweight='bold')

    plt.suptitle("Sample Predictions on Test Set", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    _save_and_show('/kaggle/working/sample_predictions.png', dpi=150)

    del raw_images; gc.collect()

def plot_gradcam_clean(test_ds, model, optimal_thresh, n_samples=8):
    """A clean, separate Grad-CAM visualization: n_samples randomly chosen
    test images with heatmap overlay, captioned with predicted probability."""
    import gc
    all_labels, all_probs, all_imgs = [], [], []
    for batch_x, batch_y in test_ds:
        p = tf.sigmoid(model(batch_x, training=False)).numpy().ravel()
        all_probs.append(p); all_labels.append(batch_y.numpy()); all_imgs.append(batch_x.numpy())
    labels = np.concatenate(all_labels)
    probs  = np.concatenate(all_probs)
    raw_images = np.concatenate(all_imgs, axis=0)
    del all_imgs; gc.collect()

    cols = 4
    rows = (n_samples + cols - 1) // cols
    # Bug 6 fix: use a seeded RNG for reproducibility (matches every other plot function)
    rng = np.random.default_rng(SEED)
    chosen = rng.choice(len(labels), min(n_samples, len(labels)), replace=False)

    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    axes = np.atleast_2d(axes)

    gradcam_failed_once = False
    for i, idx in enumerate(chosen):
        ax = axes[i // cols, i % cols]
        img_arr = raw_images[idx]
        true_lbl = "TB" if labels[idx] == 1 else "Normal"
        try:
            overlay = generate_gradcam_overlay(model, img_arr)
            ax.imshow(overlay)
        except Exception as e_gc:
            img_show = img_arr.copy()
            img_show[..., 0] += 103.939; img_show[..., 1] += 116.779; img_show[..., 2] += 123.68
            img_show = img_show[..., ::-1]
            img_show = np.clip(img_show, 0, 255).astype(np.uint8)
            ax.imshow(img_show)
            if not gradcam_failed_once:
                print(f"Grad-CAM overlay generation failed (showing raw image). First error: {e_gc}")
                gradcam_failed_once = True
        ax.set_title(f"True: {true_lbl} | Prob(TB): {probs[idx]:.3f}", fontsize=10)
        ax.axis('off')
        ax.axis('off')

    # Hide any unused axes
    for j in range(len(chosen), rows * cols):
        axes[j // cols, j % cols].axis('off')

    plt.suptitle("Grad-CAM Visualizations", fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save_and_show('/kaggle/working/gradcam_visualizations.png', dpi=150)

    del raw_images; gc.collect()

def plot_model_architecture(model, save_path='/kaggle/working/model_architecture.png'):
    """Render the model as a graph using tf.keras.utils.plot_model.
    Requires pydot + graphviz (both pre-installed on Kaggle).
    Also saves a text summary alongside it."""
    try:
        tf.keras.utils.plot_model(
            model, to_file=save_path,
            show_shapes=True, show_layer_names=True, expand_nested=False, dpi=150,
        )
        print(f"Model architecture diagram saved: {save_path}")
        # Display inline
        from IPython.display import Image as IPyImage, display
        display(IPyImage(filename=save_path))
    except Exception as e:
        print(f"plot_model failed ({e}); falling back to model.summary() text.")
        summary_lines = []
        model.summary(print_fn=lambda s: summary_lines.append(s))
        with open('/kaggle/working/model_summary.txt', 'w') as f:
            f.write('\n'.join(summary_lines))
        print("Saved text summary to /kaggle/working/model_summary.txt")

def build_performance_table(teacher, student, teacher_test_eval, student_test_y,
                            student_test_p, student_test_pred, best_t,
                            teacher_test_ds_t=None,
                            teacher_inference_ms=None, student_inference_ms=None,
                            teacher_path='/kaggle/working/teacher_resnet50.keras',
                            student_path='/kaggle/working/tb_student_densenet121.keras'):
    """Compare teacher vs student on AUC, accuracy, F1, params, size, latency.
    Saves performance_comparison.png (matplotlib table) and .csv."""
    from sklearn.metrics import f1_score

    # Unpack teacher test eval result (loss, acc, auc) — but here we already have
    # teacher_test_eval as the return value of teacher.evaluate(test_ds).
    if isinstance(teacher_test_eval, (list, tuple)) and len(teacher_test_eval) >= 3:
        teacher_loss, teacher_acc, teacher_auc = teacher_test_eval[:3]
    else:
        teacher_loss = teacher_acc = teacher_auc = float('nan')

    student_auc = roc_auc_score(student_test_y, student_test_p)
    student_acc = float((student_test_pred == student_test_y).mean())
    student_f1  = f1_score(student_test_y, student_test_pred)

    # Bug 5 fix: compute teacher F1 with a real forward pass on test_ds_t.
    # The previous code had a dead placeholder that always left teacher_f1 = nan.
    teacher_f1 = float('nan')
    if teacher_test_ds_t is not None:
        try:
            t_probs, t_true = [], []
            for bx, by in teacher_test_ds_t:
                t_probs.extend(tf.sigmoid(teacher(bx, training=False)).numpy().ravel())
                t_true.extend(by.numpy())
            t_probs, t_true = np.array(t_probs), np.array(t_true)
            # Use the same optimal threshold from validation (student's best_t is a
            # reasonable proxy — both models rank similar cases)
            t_pred = (t_probs > best_t).astype(int)
            teacher_f1 = f1_score(t_true, t_pred)
        except Exception as _e:
            print(f"Teacher F1 computation failed: {_e}")
    rows = [
        ("Teacher (ResNet50)",  teacher_acc, teacher_auc, teacher_f1,
         teacher.count_params(), os.path.getsize(teacher_path) / 1e6,
         teacher_inference_ms if teacher_inference_ms else float('nan')),
        ("Student (DenseNet121)", student_acc, student_auc, student_f1,
         student.count_params(), os.path.getsize(student_path) / 1e6,
         student_inference_ms if student_inference_ms else float('nan')),
    ]

    # Save CSV
    import csv
    with open('/kaggle/working/performance_comparison.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["Model", "Accuracy", "AUC", "F1 (TB)",
                    "Params (M)", "Size (MB)", "Inference (ms/img)"])
        for r in rows:
            w.writerow([r[0], f"{r[1]:.4f}", f"{r[2]:.4f}", f"{r[3]:.4f}",
                        f"{r[4]/1e6:.2f}",  f"{r[5]:.2f}", f"{r[6]:.2f}"])

    # Render as matplotlib table
    fig, ax = plt.subplots(figsize=(13, 2.5))
    ax.axis('off')
    col_labels = ["Model", "Accuracy", "AUC", "F1 (TB)",
                  "Params (M)", "Size (MB)", "Inference (ms/img)"]
    cell_text = [[r[0], f"{r[1]:.3f}", f"{r[2]:.3f}", f"{r[3]:.3f}",
                  f"{r[4]/1e6:.2f}", f"{r[5]:.2f}", f"{r[6]:.2f}"] for r in rows]
    table = ax.table(cellText=cell_text, colLabels=col_labels,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)
    # Color header row
    for j in range(len(col_labels)):
        table[(0, j)].set_facecolor('#404040')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    # Highlight student row
    for j in range(len(col_labels)):
        table[(2, j)].set_facecolor('#d0e8ff')
    plt.title("Performance Comparison: Teacher vs Student", fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    _save_and_show('/kaggle/working/performance_comparison.png', dpi=200)

def measure_deployment_performance(teacher, student, sample_batch,
                                   teacher_path='/kaggle/working/teacher_resnet50.keras',
                                   student_path='/kaggle/working/tb_student_densenet121.keras'):
    """Measure inference latency on a single GPU batch, model size, parameter count.
    Returns (teacher_ms_per_img, student_ms_per_img) for use in the perf table."""
    import time

    def _time_inference(model, x, n_warmup=10, n_runs=50):
        # Warm-up
        for _ in range(n_warmup):
            _ = model(x, training=False)
        # Timed
        start = time.perf_counter()
        for _ in range(n_runs):
            _ = model(x, training=False)
        elapsed = time.perf_counter() - start
        return (elapsed / n_runs) * 1000.0 / x.shape[0]   # ms per image

    x = sample_batch
    teacher_ms = _time_inference(teacher, x)
    student_ms = _time_inference(student, x)
    teacher_params = teacher.count_params()
    student_params = student.count_params()
    teacher_size_mb = os.path.getsize(teacher_path) / 1e6
    student_size_mb = os.path.getsize(student_path) / 1e6

    # Save txt report
    with open('/kaggle/working/deployment_metrics.txt', 'w') as f:
        f.write("Deployment Performance Metrics\n")
        f.write("==============================\n\n")
        f.write(f"Teacher (ResNet50)\n")
        f.write(f"  Parameters:        {teacher_params:>12,d}  ({teacher_params/1e6:.2f} M)\n")
        f.write(f"  Size on disk:      {teacher_size_mb:>12.2f} MB\n")
        f.write(f"  Inference latency: {teacher_ms:>12.3f} ms / image\n")
        f.write(f"  Throughput:        {1000.0/teacher_ms:>12.1f} images / sec\n\n")
        f.write(f"Student (DenseNet121)\n")
        f.write(f"  Parameters:        {student_params:>12,d}  ({student_params/1e6:.2f} M)\n")
        f.write(f"  Size on disk:      {student_size_mb:>12.2f} MB\n")
        f.write(f"  Inference latency: {student_ms:>12.3f} ms / image\n")
        f.write(f"  Throughput:        {1000.0/student_ms:>12.1f} images / sec\n\n")
        # Bug 4 fix: DenseNet has more sequential ops than ResNet despite fewer params
        # (430 vs 178 layers, concat-heavy vs add-residual). Student may be SLOWER.
        speed_ratio = teacher_ms / student_ms
        if speed_ratio >= 1.0:
            speedup_label = f"{speed_ratio:.2f}x faster than teacher"
        else:
            speedup_label = f"{1.0/speed_ratio:.2f}x SLOWER than teacher (DenseNet dense-connectivity overhead)"
        f.write(f"Speedup: {speedup_label}\n")
        f.write(f"Compression: {teacher_params/student_params:.2f}x fewer params\n")

    # Render bar chart
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    names = ["Teacher\nResNet50", "Student\nDenseNet121"]
    params_m = [teacher_params/1e6, student_params/1e6]
    sizes_mb = [teacher_size_mb, student_size_mb]
    latencies = [teacher_ms, student_ms]
    colors = ["#ff7f0e", "#1f77b4"]

    axes[0].bar(names, params_m, color=colors)
    axes[0].set_ylabel("Parameters (M)"); axes[0].set_title("Model Size: Parameters")
    for i, v in enumerate(params_m): axes[0].text(i, v + 0.1, f"{v:.2f}M", ha='center')
    axes[1].bar(names, sizes_mb, color=colors)
    axes[1].set_ylabel("Size on disk (MB)"); axes[1].set_title("Model Size: Disk")
    for i, v in enumerate(sizes_mb): axes[1].text(i, v + 0.5, f"{v:.1f}MB", ha='center')
    axes[2].bar(names, latencies, color=colors)
    axes[2].set_ylabel("Latency (ms / image)"); axes[2].set_title("Inference Speed (P100)")
    for i, v in enumerate(latencies): axes[2].text(i, v + 0.05, f"{v:.2f}ms", ha='center')

    plt.suptitle("Deployment Performance Metrics", fontsize=13, fontweight='bold')
    plt.tight_layout()
    _save_and_show('/kaggle/working/deployment_metrics.png', dpi=200)

    print("\nDeployment metrics saved to /kaggle/working/deployment_metrics.txt")
    speed_ratio = teacher_ms / student_ms
    speed_str = (f"{speed_ratio:.2f}x faster" if speed_ratio >= 1.0
                 else f"{1.0/speed_ratio:.2f}x slower (DenseNet concat overhead)")
    print(f"Teacher: {teacher_ms:.2f} ms/img | Student: {student_ms:.2f} ms/img | {speed_str}")

    return teacher_ms, student_ms
```

# ============================================================================
# CELL 13 — Run all 10 deliverables: diagnostics + plots + tables + metrics
# ============================================================================
```python
print("\nGenerating all 10 deliverables...")
os.makedirs("/kaggle/working", exist_ok=True)

# --- 1. Training/Validation Loss + AUC + Accuracy curves ---
# Bug 3 fix: plot_training_curves was not defined at Cell 9 run time, so
# teacher curves were never actually saved. We plot both here in Cell 13.
print("  1a. Plotting teacher training curves...")
plot_training_curves(teacher_history)   # saves training_curves.png (teacher)

print("  1b. Plotting student/distiller training curves...")
plot_training_curves(history)           # overwrites with distiller history

# --- 2. Confusion Matrix ---
plot_confusion_matrix_heatmap(test_y, test_pred)

# --- 3. ROC + Precision-Recall curves ---
plot_roc_pr_curves(test_y, test_p, best_t)

# --- 4. Sample Predictions (4-row montage: Normal / TB / errors / low-conf) ---
plot_sample_predictions(test_ds, student, best_t)

# --- 5. Clean Grad-CAM Visualizations (separate from error gallery) ---
plot_gradcam_clean(test_ds, student, best_t, n_samples=8)

# --- 6. Error Analysis Examples (categorized TP/TN/FP/FN with Grad-CAM) ---
plot_error_analysis_gallery(test_ds, student, best_t)

# --- 7. Model Architecture Diagram ---
plot_model_architecture(student)

# --- 8. Deployment Performance Metrics (timing + size) ---
# Pull a sample batch from test_ds for the timing measurement.
_sample_x, _ = next(iter(test_ds))
teacher_ms, student_ms = measure_deployment_performance(teacher, student, _sample_x)

# --- 9. Performance Comparison Table (teacher vs student) ---
# Re-evaluate teacher on its own (ResNet-preprocessed) test set
teacher_test_eval = teacher.evaluate(test_ds_t, verbose=0)
build_performance_table(
    teacher=teacher,
    student=student,
    teacher_test_eval=teacher_test_eval,
    student_test_y=test_y,
    student_test_p=test_p,
    student_test_pred=test_pred,
    best_t=best_t,
    teacher_test_ds_t=test_ds_t,  # Bug 5 fix: pass teacher dataset for real F1
    teacher_inference_ms=teacher_ms,
    student_inference_ms=student_ms,
)

# --- 10. Final save: overwrite earlier placeholder with the real tuned threshold ---
student.save("/kaggle/working/tb_student_densenet121.keras")
with open("/kaggle/working/best_threshold.txt", "w") as f:
    f.write(str(best_t))

print("\nAll 10 deliverables saved to /kaggle/working/")
list_saved_plots()
```





















