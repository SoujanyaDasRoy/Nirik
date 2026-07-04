# Kaggle Notebook 1: U-Net Training & Preprocessing

This notebook is broken down into cells. Copy and paste each block into a new cell in your Kaggle notebook.

### 📦 Datasets to Import
Click "Add Data" in Kaggle and search for:
1. `nikhilpandey360/chest-xray-masks-and-labels` (for U-Net masks)
2. `tawsifurrahman/tuberculosis-tb-chest-xray-dataset`
3. `nih-chest-xrays/data`
4. `projectmantra/nirt-india-chest-x-ray-dicom-dataset`

---

### Cell 1: Imports and Setup
```python
!pip install pydicom -q

import os, glob, cv2, shutil
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from pathlib import Path
from tqdm import tqdm

# Ensure reproducibility
tf.random.set_seed(42)
np.random.seed(42)

IMG_SIZE = 256 # U-Net training resolution
print("TensorFlow Version:", tf.__version__)
print("GPUs Available:", len(tf.config.list_physical_devices('GPU')))
```

### Cell 2: Data Loading (Chest X-Ray Masks)
```python
def load_unet_data():
    X_list, y_list = [], []
    
    # Using the nikhilpandey360/chest-xray-masks-and-labels dataset
    # This dataset contains BOTH Montgomery and Shenzhen images with their masks pre-combined!
    img_files = sorted(glob.glob('/kaggle/input/chest-xray-masks-and-labels/data/Lung Segmentation/CXR_png/*.png'))
    mask_files = sorted(glob.glob('/kaggle/input/chest-xray-masks-and-labels/data/Lung Segmentation/masks/*.png'))
    
    print("Loading dataset... (This will take a few minutes)")
    for img_path, mask_path in tqdm(zip(img_files, mask_files), total=len(img_files)):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None or mask is None: continue
            
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
        
        X_list.append(img)
        y_list.append(mask)

    X = np.array(X_list, dtype=np.float32)[..., np.newaxis] / 255.0
    y = np.array(y_list, dtype=np.float32)[..., np.newaxis] / 255.0
    
    # Binarize mask
    y = (y > 0.5).astype(np.float32)
    return X, y

X_data, y_data = load_unet_data() 

# Shuffle data
indices = np.arange(len(X_data))
np.random.shuffle(indices)
X_data = X_data[indices]
y_data = y_data[indices]

# Split into train/val
split_idx = int(len(X_data) * 0.8)
X_train, X_val = X_data[:split_idx], X_data[split_idx:]
y_train, y_val = y_data[:split_idx], y_data[split_idx:]
print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")
```

### Cell 3: Data Augmentation & tf.data Pipeline
```python
def augment_seg(image, mask):
    concat = tf.concat([image, mask], axis=-1)
    concat = tf.image.random_flip_left_right(concat)
    concat = tf.image.rot90(concat, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
    
    img_ch, mask_ch = concat[..., :1], concat[..., 1:]
    img_ch = tf.image.random_brightness(img_ch, max_delta=0.1)
    img_ch = tf.clip_by_value(img_ch, 0.0, 1.0)
    return img_ch, mask_ch

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_ds = train_ds.shuffle(500).map(augment_seg, num_parallel_calls=tf.data.AUTOTUNE).batch(16).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(16).prefetch(tf.data.AUTOTUNE)
```

### Cell 4: Model Definition (Attention U-Net)
```python
def conv_block(x, filters):
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

def attention_gate(skip, gating, inter_filters):
    """
    Attention Gate: learns WHERE to focus in the encoder features.
    - skip:    encoder feature map (high-resolution detail)
    - gating:  decoder feature map (semantic context from deeper layers)
    - output:  skip * attention_coefficients (only relevant features pass through)
    
    This forces the model to attend to lung boundaries and suppress
    irrelevant features like ribs, clothing artifacts, and background noise.
    """
    # Project both inputs to the same channel dimension
    W_skip = layers.Conv2D(inter_filters, (1, 1), padding='same', use_bias=False)(skip)
    W_skip = layers.BatchNormalization()(W_skip)
    
    W_gate = layers.Conv2D(inter_filters, (1, 1), padding='same', use_bias=False)(gating)
    W_gate = layers.BatchNormalization()(W_gate)
    
    # Additive attention: combine both signals
    combined = layers.Add()([W_skip, W_gate])
    combined = layers.Activation('relu')(combined)
    
    # Produce attention coefficients (0 to 1 per spatial location)
    psi = layers.Conv2D(1, (1, 1), padding='same', use_bias=False)(combined)
    psi = layers.BatchNormalization()(psi)
    psi = layers.Activation('sigmoid')(psi)
    
    # Multiply: only attended features pass through to the decoder
    return layers.Multiply()([skip, psi])

def build_attention_unet(input_shape=(IMG_SIZE, IMG_SIZE, 1)):
    inputs = layers.Input(shape=input_shape)
    
    # ---- Encoder ----
    c1 = conv_block(inputs, 32)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    
    c2 = conv_block(p1, 64)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    
    c3 = conv_block(p2, 128)
    p3 = layers.MaxPooling2D((2, 2))(c3)
    
    c4 = conv_block(p3, 256)
    d4 = layers.Dropout(0.2)(c4)
    p4 = layers.MaxPooling2D((2, 2))(d4)
    
    # ---- Bottleneck ----
    c5 = conv_block(p4, 512)
    d5 = layers.Dropout(0.3)(c5)
    
    # ---- Decoder with Attention Gates ----
    u6 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(d5)
    a4 = attention_gate(skip=d4, gating=u6, inter_filters=128)
    u6 = layers.concatenate([u6, a4])
    c6 = conv_block(u6, 256)
    
    u7 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c6)
    a3 = attention_gate(skip=c3, gating=u7, inter_filters=64)
    u7 = layers.concatenate([u7, a3])
    c7 = conv_block(u7, 128)
    
    u8 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c7)
    a2 = attention_gate(skip=c2, gating=u8, inter_filters=32)
    u8 = layers.concatenate([u8, a2])
    c8 = conv_block(u8, 64)
    
    u9 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c8)
    a1 = attention_gate(skip=c1, gating=u9, inter_filters=16)
    u9 = layers.concatenate([u9, a1])
    c9 = conv_block(u9, 32)
    
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid', dtype='float32')(c9)
    return models.Model(inputs=[inputs], outputs=[outputs])

model = build_attention_unet()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='binary_crossentropy',
              metrics=[tf.keras.metrics.BinaryIoU(target_class_ids=[1], name='iou')])
model.summary()
```

### Cell 5: Training
```python
callbacks = [
    tf.keras.callbacks.ModelCheckpoint("/kaggle/working/unet_lung_segmenter.keras", save_best_only=True, monitor="val_iou", mode="max")
]

print("Starting training...")
history = model.fit(train_ds, validation_data=val_ds, epochs=40, callbacks=callbacks)
print("U-Net training complete! Model saved to /kaggle/working/unet_lung_segmenter.keras")
```

### Cell 6: Visualizing Training Graphs
```python
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('U-Net Training Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['iou'], label='Train IoU')
plt.plot(history.history['val_iou'], label='Val IoU')
plt.title('U-Net Intersection over Union (IoU)')
plt.legend()
plt.tight_layout()
plt.show()
```

### Cell 7: Visualizing U-Net Predictions (Testing the Mask)
```python
best_unet = tf.keras.models.load_model("/kaggle/working/unet_lung_segmenter.keras", compile=False)

def visualize_predictions(model, X_val, y_val, num_samples=3):
    indices = np.random.choice(len(X_val), num_samples, replace=False)
    plt.figure(figsize=(15, 5 * num_samples))
    
    for i, idx in enumerate(indices):
        img = X_val[idx]
        mask_true = y_val[idx]
        mask_pred = model.predict(img[np.newaxis, ...], verbose=0)[0]
        
        plt.subplot(num_samples, 3, i * 3 + 1)
        plt.imshow(img[..., 0], cmap='gray')
        plt.title("Original X-Ray")
        plt.axis('off')
        
        plt.subplot(num_samples, 3, i * 3 + 2)
        plt.imshow(mask_true[..., 0], cmap='gray')
        plt.title("Ground Truth Mask")
        plt.axis('off')
        
        plt.subplot(num_samples, 3, i * 3 + 3)
        plt.imshow(mask_pred[..., 0] > 0.5, cmap='gray')
        plt.title("U-Net Predicted Mask")
        plt.axis('off')
        
    plt.tight_layout()
    plt.show()

visualize_predictions(best_unet, X_val, y_val)
```

### Cell 8: Pre-segmenting the Dataset for Notebook 2
```python
# WARNING: This cell might take 1-2 hours depending on the dataset size.
def segment_and_crop(gray_img, unet_model, pad_percent=0.05):
    """
    Segments the lung region and crops tightly to the bounding box.
    This eliminates the black-border boundary shape that the classifier
    could cheat on (the old "shoulder problem").
    """
    orig_h, orig_w = gray_img.shape[:2]
    
    # Predict mask
    resized = cv2.resize(gray_img, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
    pred = unet_model.predict(resized[np.newaxis, ..., np.newaxis], verbose=0)[0, :, :, 0]
    mask = (pred > 0.5).astype(np.uint8)
    mask_full = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    
    # Apply mask to zero out non-lung regions
    masked = cv2.bitwise_and(gray_img, gray_img, mask=mask_full)
    
    # Find bounding box of the lung mask and crop tightly
    coords = cv2.findNonZero(mask_full)
    if coords is None:
        # Mask failed — return the original image resized
        return cv2.resize(gray_img, (224, 224))
    
    x, y, w, h = cv2.boundingRect(coords)
    
    # Add small padding so we don't clip the lung edges
    pad_x = int(w * pad_percent)
    pad_y = int(h * pad_percent)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(orig_w, x + w + pad_x)
    y2 = min(orig_h, y + h + pad_y)
    
    cropped = masked[y1:y2, x1:x2]
    
    # Resize to classifier input size
    cropped = cv2.resize(cropped, (224, 224))
    return cropped

def process_and_save_dataset(input_dir, output_dir, is_dicom=False):
    import pydicom
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    files = glob.glob(f"{input_dir}/**/*.*", recursive=True)
    for f in tqdm(files, desc=f"Processing {Path(input_dir).name}"):
        try:
            if is_dicom and f.lower().endswith('.dcm'):
                dcm = pydicom.dcmread(f, force=True)
                arr = dcm.pixel_array.astype(np.float32)
                arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
                gray = arr.astype(np.uint8)
            elif f.lower().endswith(('.png', '.jpg', '.jpeg')):
                img = cv2.imread(f)
                if img is None: continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                continue

            # Segment lungs and crop to bounding box
            cropped = segment_and_crop(gray, best_unet)
            
            # PRESERVE DIRECTORY STRUCTURE SO LABELS AREN'T LOST
            rel_path = Path(f).relative_to(input_dir)
            save_path = out_path / rel_path.with_suffix('.png')
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            cv2.imwrite(str(save_path), cropped)
        except Exception as e:
            pass

def process_nih_subset(input_dir, output_dir, max_images=2000):
    """Processes a random subset of the massive NIH dataset."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    files = glob.glob(f"{input_dir}/images_*/**/*.png", recursive=True)
    if not files: return
    
    np.random.seed(42)
    files = np.random.permutation(files)[:max_images]
    
    for f in tqdm(files, desc=f"Processing NIH Subset ({max_images} imgs)"):
        try:
            img = cv2.imread(f)
            if img is None: continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            cropped = segment_and_crop(gray, best_unet)
            
            save_name = Path(f).name
            cv2.imwrite(str(out_path / save_name), cropped)
        except Exception as e:
            pass

# Process the databases
process_and_save_dataset("/kaggle/input/tuberculosis-tb-chest-xray-dataset", "/kaggle/working/segmented_dataset/tb_database")
process_and_save_dataset("/kaggle/input/nirt-india-chest-x-ray-dicom-dataset", "/kaggle/working/segmented_dataset/nirt", is_dicom=True)
process_nih_subset("/kaggle/input/data", "/kaggle/working/segmented_dataset/nih", max_images=2000)

# Quick visual check — show 3 sample segmented images
sample_files = glob.glob("/kaggle/working/segmented_dataset/**/*.png", recursive=True)[:3]
plt.figure(figsize=(15, 5))
for i, sf in enumerate(sample_files):
    plt.subplot(1, 3, i+1)
    plt.imshow(cv2.imread(sf, cv2.IMREAD_GRAYSCALE), cmap='gray')
    plt.title(f"Segmented: {Path(sf).name[:20]}")
    plt.axis('off')
plt.suptitle("Verify: Shoulders should NOT be visible")
plt.tight_layout()
plt.show()

print("Zipping the segmented dataset...")
shutil.make_archive("/kaggle/working/segmented_dataset", 'zip', "/kaggle/working/segmented_dataset")
print("Done! Create a Kaggle Dataset from this output to use in Notebook 2.")
```
