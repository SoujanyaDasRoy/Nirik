import os
import sys
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

# Add the backend directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from core.inference import preprocess_for_classifier, pad_to_square, segment_lungs, get_model

def main():
    # Load the image
    base_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..')
    image_dir = os.path.join(base_dir, 'jaypee_india_dataset', 'images', 'images')
    image_path = os.path.join(image_dir, 'TEST_px27.jpg')

    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    img = Image.open(image_path)
    print(f"Original image size: {img.size}")
    print(f"Original image mode: {img.mode}")

    # Pad to square
    padded_img = pad_to_square(img).convert('L')
    print(f"After padding size: {padded_img.size}")

    # Convert to grayscale array
    gray_arr = np.array(padded_img, dtype=np.float32)
    print(f"Grayscale array shape: {gray_arr.shape}")
    print(f"Grayscale array dtype: {gray_arr.dtype}")
    print(f"Grayscale array min: {gray_arr.min()}, max: {gray_arr.max()}")

    # Apply lung segmentation
    seg_arr = segment_lungs(gray_arr)
    print(f"After segmentation shape: {seg_arr.shape}")
    print(f"After segmentation dtype: {seg_arr.dtype}")
    print(f"After segmentation min: {seg_arr.min()}, max: {seg_arr.max()}")

    # Check if U-Net is active
    unet_active = get_model() is not None  # This is not correct; we need to check U-Net
    # Actually, we have a function get_unet()
    from core.inference import get_unet
    unet_active = get_unet() is not None
    print(f"U-Net active: {unet_active}")

    # Preprocess for classifier
    tensor = preprocess_for_classifier(seg_arr, unet_active=unet_active)
    print(f"Tensor shape: {tensor.shape}")
    print(f"Tensor dtype: {tensor.dtype}")

    # Convert to numpy for statistics
    tensor_np = tensor.numpy()
    print(f"Tensor min: {tensor_np.min()}, max: {tensor_np.max()}, mean: {tensor_np.mean()}, std: {tensor_np.std()}")

    # Get the model
    model = get_model()
    print(f"Model input shape: {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")

    # Run the model
    # Note: the model expects a tensor of shape (1, H, W, 3)
    logits = model(tensor, training=False)
    print(f"Logits shape: {logits.shape}")
    print(f"Logits: {logits}")

    # Apply softmax to get probabilities
    probs = tf.nn.softmax(logits, axis=-1)
    print(f"Probabilities: {probs}")
    print(f"Probability of Normal (class 0): {probs[0][0].numpy()}")
    print(f"Probability of TB (class 1): {probs[0][1].numpy()}")

    # Also, we can get the raw output as the logits
    print("\nRaw model output (logits):")
    print(logits.numpy())

if __name__ == "__main__":
    main()