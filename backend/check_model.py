import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from tensorflow import keras
import numpy as np

# Patch Dense layer to ignore quantization_config if present
from tensorflow.keras.layers import Dense
_original_dense_init = Dense.__init__
def _patched_dense_init(self, *args, **kwargs):
    kwargs.pop('quantization_config', None)
    return _original_dense_init(self, *args, **kwargs)
Dense.__init__ = _patched_dense_init

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "CNN Model Training", "student_cnn.keras")
model = keras.models.load_model(model_path)

print("Model summary:")
model.summary()
print("\nOutput layer details:")
print(f"Output shape: {model.output_shape}")
print(f"Output activation: {model.layers[-1].activation}")
print(f"Loss function: {model.loss}")

# Get the config of the last layer
last_layer = model.layers[-1]
print(f"\nLast layer config: {last_layer.get_config()}")

# If the model is a Sequential, we can also check the layers
if hasattr(model, 'layers'):
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'output_shape'):
            print(f"Layer {i}: {layer.__class__.__name__}, output shape: {layer.output_shape}")
        else:
            print(f"Layer {i}: {layer.__class__.__name__}, output shape: N/A")