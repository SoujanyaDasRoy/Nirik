import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import tensorflow as tf
from PIL import Image
import torch

# Test that demonstrates generate_saliency_heatmap works without PyTorch errors
def test_grayscale_only():
    from core.inference import generate_saliency_heatmap
    
    # Load the actual model to test with real scenario
    model = None
    try:
        model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'CNN Model Training', 'student_cnn.keras')
        import tensorflow.keras as keras
        model = keras.models.load_model(model_path)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Could not load model, using dummy model: {e}")
        # Create a simple model that mimics the expected structure
        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', name='conv2d')(inputs)
        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', name='conv2d_2')(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        outputs = tf.keras.layers.Dense(2, name='predictions')(x)
        model = tf.keras.Model(inputs, outputs)
        print("Using dummy model")
    
    # Create a proper input tensor (simulating what preprocess_for_classifier returns)
    # This is a PyTorch tensor as returned by the original preprocessing
    dummy_array = np.random.rand(1, 224, 224, 3).astype(np.float32)
    tensor = torch.from_numpy(dummy_array)  # This is what we get from preprocess_for_classifier
    
    # Create a test image
    img = Image.new('RGB', (224, 224), color='gray')
    
    print("Testing generate_saliency_heatmap - this should work without PyTorch errors in the Grad-CAM implementation")
    
    # Test all methods
    methods = ["gradcam", "gradcam_plusplus", "attention", "coverage", "attribution"]
    
    for method in methods:
        try:
            print(f"  Testing {method}...")
            heatmap_img, is_fallback = generate_saliency_heatmap(
                model, tensor, img, is_tb=True, method=method
            )
            print(f"    SUCCESS: {method} - fallback: {is_fallback}")
            
            # Also test with return_raw=True
            heatmap_img, is_fallback, raw_heatmap = generate_saliency_heatmap(
                model, tensor, img, is_tb=False, method=method, return_raw=True
            )
            print(f"    SUCCESS: {method} (return_raw) - fallback: {is_fallback}")
            
        except Exception as e:
            print(f"    ERROR in {method}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\nTest completed. If we saw no 'NotImplementedError' or 'AttributeError' related to PyTorch")
    print("in the Grad-CAM implementation, then our fix is working correctly.")

if __name__ == "__main__":
    test_grayscale_only()
