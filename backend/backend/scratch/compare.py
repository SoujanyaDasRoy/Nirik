import sys
sys.path.insert(0, '/c/Users/sdroy/OneDrive/Desktop/Documents/Final Year Project/backend')
import os
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from core.inference import get_model, pad_to_square, segment_lungs, preprocess_for_classifier
import torch

def main():
    # Load model
    model = get_model()
    print("Model loaded")
    
    # Image path
    img_path = os.path.join('/c/Users/sdroy/OneDrive/Desktop/Documents/Final Year Project', 'jaypee_india_dataset', 'images', 'images', 'TEST_px27.jpg')
    img = Image.open(img_path)
    print(f"Original image size: {img.size}, mode: {img.mode}")
    
    # --- BACKEND PIPELINE (with our fix) ---
    # This mimics what predict_image does up to the model call
    padded_img = pad_to_square(img).convert('L')
    gray_arr = np.array(padded_img, dtype=np.float32)
    unet_active = True  # we have U-Net
    arr = segment_lungs(gray_arr)  # (IMG_SIZE, IMG_SIZE) float32
    tensor = preprocess_for_classifier(arr, unet_active=unet_active)  # (1, H, W, 3)
    
    # Run model
    logits = model(tensor, training=False)
    probs = tf.nn.softmax(logits, axis=-1)
    print("\nBackend pipeline:")
    print(f"  Tensor shape: {tensor.shape}")
    print(f"  Tensor min: {tensor.numpy().min()}, max: {tensor.numpy().max()}, mean: {tensor.numpy().mean()}, std: {tensor.numpy().std()}")
    print(f"  Logits: {logits.numpy()}")
    print(f"  Probs: {probs.numpy()}")
    
    # --- SIMULATED ORIGINAL EVALUATION PIPELINE ---
    # We assume the original training data was: lung segmented and cropped to bounding box, then resized to 224x224
    # We'll replicate the segment_and_crop function from Notebook 1 (adapted)
    
    # Load image as grayscale (original, not padded)
    gray_arr_orig = np.array(img.convert('L'), dtype=np.float32)
    h_orig, w_orig = gray_arr_orig.shape
    
    # Get U-Net prediction (same as in segment_lungs but we need the mask)
    seg_size = 256  # from inference.py
    img_norm = cv2.resize(gray_arr_orig, (seg_size, seg_size)).astype(np.float32) / 255.0
    # We need torch tensor for U-Net
    import torch
    seg_tensor = torch.tensor(img_norm[np.newaxis, :, :, np.newaxis], dtype=torch.float32)
    # We need the device and model from inference
    from core.inference import get_unet, DEVICE
    unet = get_unet()
    if unet is None:
        print("U-Net not available")
        return
    with torch.no_grad():
        pred = unet(seg_tensor.to(DEVICE))
    pred_np = (pred.detach().cpu().numpy() if hasattr(pred, "detach") else np.array(pred))[0, :, :, 0]
    binary_mask = (pred_np > 0.5).astype(np.uint8)
    # Resize mask to original image size
    mask_full = cv2.resize(binary_mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
    
    # Apply mask to zero out non-lung regions
    masked = cv2.bitwise_and(gray_arr_orig.astype(np.uint8), gray_arr_orig.astype(np.uint8), mask=mask_full)
    
    # Find bounding box of the mask with padding (as in segment_and_crop)
    coords = cv2.findNonZero(mask_full)
    if coords is None:
        print("No lung mask found")
        crop_box = (0, 0, w_orig, h_orig)
    else:
        x, y, w, h = cv2.boundingRect(coords)
        pad_percent = 0.05
        pad_x = int(w * pad_percent)
        pad_y = int(h * pad_percent)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w_orig, x + w + pad_x)
        y2 = min(h_orig, y + h + pad_y)
        crop_box = (x1, y1, x2, y2)
    
    # Crop the grayscale image (original) to the bounding box
    cropped = gray_arr_orig[y1:y2, x1:x2]
    # Resize to 224x224
    resized = cv2.resize(cropped, (224, 224)).astype(np.float32)
    # Stack to 3 channels
    rgb = np.stack([resized, resized, resized], axis=-1)  # (224, 224, 3)
    # Add batch dimension
    rgb_batch = np.expand_dims(rgb, axis=0)  # (1, 224, 224, 3)
    tensor_orig = tf.convert_to_tensor(rgb_batch, dtype=tf.float32)
    
    # Run model
    logits_orig = model(tensor_orig, training=False)
    probs_orig = tf.nn.softmax(logits_orig, axis=-1)
    print("\nSimulated original evaluation pipeline:")
    print(f"  Tensor shape: {tensor_orig.shape}")
    print(f"  Tensor min: {tensor_orig.numpy().min()}, max: {tensor_orig.numpy().max()}, mean: {tensor_orig.numpy().mean()}, std: {tensor_orig.numpy().std()}")
    print(f"  Logits: {logits_orig.numpy()}")
    print(f"  Probs: {probs_orig.numpy()}")
    
    # Compare
    print("\nComparison:")
    print(f"  Backend probs:  {probs.numpy()}")
    print(f"  Original probs: {probs_orig.numpy()}")
    diff = np.abs(probs.numpy() - probs_orig.numpy())
    print(f"  Absolute difference: {diff}")
    print(f"  Max difference: {diff.max()}")
    if np.allclose(probs.numpy(), probs_orig.numpy(), atol=1e-6):
        print("  Match: Yes (within tolerance)")
    else:
        print("  Match: No")
    
    # Check if the difference is due to preprocessing only (i.e., after the model input)
    # Compare the tensors fed to the model
    print("\nInput tensor comparison:")
    backend_input = tensor.numpy()
    original_input = tensor_orig.numpy()
    print(f"  Backend input shape: {backend_input.shape}")
    print(f"  Original input shape: {original_input.shape}")
    # They should be the same shape (1,224,224,3)
    if backend_input.shape == original_input.shape:
        input_diff = np.abs(backend_input - original_input)
        print(f"  Input max difference: {input_diff.max()}")
        print(f"  Input mean difference: {input_diff.mean()}")
    else:
        print("  Input shapes differ")

if __name__ == "__main__":
    main()
