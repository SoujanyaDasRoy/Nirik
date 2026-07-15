import os
import sys
# Set Keras backend to PyTorch before importing keras
os.environ["KERAS_BACKEND"] = "torch"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))
import keras

print("Keras version:", keras.__version__)

# Compute BASE_DIR as the backend directory (matching inference.py logic)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # This is wrong for our script location
# Let's compute correctly: we want the backend directory.
# If this script is in the project root, then backend is at ./backend
BASE_DIR = os.path.join(os.path.dirname(__file__), 'backend')
print(f"BASE_DIR: {BASE_DIR}")

MODEL_PATH = os.path.join(BASE_DIR, "..", "CNN Model Training", "student_cnn.keras")
print(f"MODEL_PATH: {MODEL_PATH}")
print(f"Exists: {os.path.exists(MODEL_PATH)}")

try:
    model = keras.saving.load_model(MODEL_PATH)
    print("Model loaded successfully!")
    model.summary()
except Exception as e:
    print(f"Error loading model: {e}")
    import traceback
    traceback.print_exc()