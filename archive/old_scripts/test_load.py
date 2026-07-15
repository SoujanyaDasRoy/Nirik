import sys
sys.path.insert(0, 'backend')
from core import inference
import os

print("Trying to load model...")
try:
    model = inference.get_model()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    import traceback
    traceback.print_exc()