import os
import time

# Force Keras to use PyTorch as its backend
os.environ["KERAS_BACKEND"] = "torch"

import keras
import torch
import numpy as np

MODEL_PATH = r"C:\Users\sdroy\OneDrive\Desktop\Documents\Final Year Project\backend\tb_student_densenet121.keras"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Starting model loading validation...")
start_time = time.time()
try:
    model = keras.saving.load_model(MODEL_PATH)
    model.to(DEVICE)
    load_time = time.time() - start_time
    print(f"SUCCESS: Model loaded successfully on {DEVICE}")
    print(f"Load time: {load_time:.4f} seconds")
    
    # Model parameters
    param_count = sum(np.prod(v.shape) for v in model.trainable_weights) + sum(np.prod(v.shape) for v in model.non_trainable_weights)
    print(f"Parameter count: {param_count:,}")
    
    # Input/Output shapes
    if hasattr(model, "input_shape"):
        print(f"Input shape: {model.input_shape}")
    else:
        print("Input shape: not directly available on model object")
        
    if hasattr(model, "output_shape"):
        print(f"Output shape: {model.output_shape}")
    else:
        print("Output shape: not directly available on model object")
        
    # Dummy forward pass
    dummy_input = torch.zeros(1, 224, 224, 3).to(DEVICE)
    output = model(dummy_input)
    print(f"Warm-up pass output shape: {output.shape}")
    print(f"Dummy output sample: {output.detach().cpu().numpy()}")
    
    # Print layers
    print("\nLayer details:")
    for idx, layer in enumerate(model.layers[:10]):
        print(f"Layer {idx}: {layer.name} ({layer.__class__.__name__})")
    print(f"... total layers: {len(model.layers)}")
    
except Exception as e:
    print("FAILED TO LOAD MODEL:")
    import traceback
    traceback.print_exc()
