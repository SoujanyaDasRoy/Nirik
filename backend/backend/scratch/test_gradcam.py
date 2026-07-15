import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import tensorflow as tf
from PIL import Image
import torch

# Create a dummy model for testing
def create_dummy_model():
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(2, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    return model

# Test the generate_saliency_heatmap function
def test_generate_saliency_heatmap():
    from core.inference import generate_saliency_heatmap
    
    # Create dummy inputs
    model = create_dummy_model()
    # Create a dummy tensor (simulating what preprocess_for_classifier returns)
    # This should be a PyTorch tensor of shape (1, 224, 224, 3)
    dummy_array = np.random.rand(1, 224, 224, 3).astype(np.float32)
    tensor = torch.from_numpy(dummy_array)
    
    # Create a dummy image
    dummy_img = Image.fromarray((np.random.rand(224, 224, 3) * 255).astype(np.uint8))
    
    print("Testing generate_saliency_heatmap with TB case...")
    try:
        heatmap_img, is_fallback = generate_saliency_heatmap(model, tensor, dummy_img, is_tb=True, method="gradcam_plusplus")
        print(f"SUCCESS: TB case - heatmap generated, is_fallback: {is_fallback}")
        print(f"Heatmap image size: {heatmap_img.size}")
    except Exception as e:
        print(f"ERROR in TB case: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nTesting generate_saliency_heatmap with Normal case...")
    try:
        heatmap_img, is_fallback = generate_saliency_heatmap(model, tensor, dummy_img, is_tb=False, method="gradcam")
        print(f"SUCCESS: Normal case - heatmap generated, is_fallback: {is_fallback}")
        print(f"Heatmap image size: {heatmap_img.size}")
    except Exception as e:
        print(f"ERROR in Normal case: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nTesting generate_saliency_heatmap with return_raw=True...")
    try:
        heatmap_img, is_fallback, raw_heatmap = generate_saliency_heatmap(model, tensor, dummy_img, is_tb=True, method="gradcam_plusplus", return_raw=True)
        print(f"SUCCESS: Return raw - heatmap generated, is_fallback: {is_fallback}")
        print(f"Heatmap image size: {heatmap_img.size}")
        print(f"Raw heatmap shape: {raw_heatmap.shape}")
    except Exception as e:
        print(f"ERROR in Return raw case: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_generate_saliency_heatmap()
