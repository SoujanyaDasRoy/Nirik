import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from PIL import Image
import numpy as np

# Test the fallback case when model or tensor is None
def test_fallback():
    from core.inference import generate_saliency_heatmap
    
    # Create a test image
    img = Image.new('RGB', (224, 224), color='gray')
    
    print("Testing fallback case (model=None or tensor=None)...")
    
    # Test with None model
    try:
        heatmap_img, is_fallback = generate_saliency_heatmap(None, None, img, is_tb=True)
        print(f"SUCCESS: None model - fallback: {is_fallback} (should be False - using simulated model)")
        assert is_fallback == False, "Should NOT fallback (use simulated model) when model is None"
    except Exception as e:
        print(f"ERROR with None model: {e}")
        return False

    # Test with None tensor
    try:
        # Create a dummy model for this test
        import tensorflow as tf
        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = tf.keras.layers.Dense(1)(inputs)
        model = tf.keras.Model(inputs, x)

        heatmap_img, is_fallback = generate_saliency_heatmap(model, None, img, is_tb=False)
        print(f"SUCCESS: None tensor - fallback: {is_fallback} (should be False - using simulated model)")
        assert is_fallback == False, "Should NOT fallback (use simulated model) when tensor is None"
    except Exception as e:
        print(f"ERROR with None tensor: {e}")
        return False
    
    # Test return_raw with fallback
    try:
        # Create a dummy model for this test
        import tensorflow as tf
        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = tf.keras.layers.Dense(1)(inputs)
        model = tf.keras.Model(inputs, x)

        heatmap_img, is_fallback, raw_heatmap = generate_saliency_heatmap(model, None, img, is_tb=True, return_raw=True)
        print(f"SUCCESS: None tensor with return_raw - fallback: {is_fallback} (should be False - using simulated model)")
        assert is_fallback == False, "Should NOT fallback (use simulated model) when tensor is None"
        assert isinstance(raw_heatmap, np.ndarray), "Should return numpy array for raw heatmap"
    except Exception as e:
        print(f"ERROR with None tensor and return_raw: {e}")
        return False
    
    print("All fallback tests passed!")
    return True

if __name__ == "__main__":
    success = test_fallback()
    if success:
        print("\nFallback functionality is working correctly.")
    else:
        print("\nFallback functionality is broken.")
        sys.exit(1)
