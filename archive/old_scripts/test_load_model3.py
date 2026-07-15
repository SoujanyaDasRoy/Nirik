import os
import sys
# Set Keras backend to PyTorch before importing keras
os.environ["KERAS_BACKEND"] = "torch"
sys.path.insert(0, 'backend')
import keras

print("Keras version:", keras.__version__)

model_path = os.path.join('..', 'CNN Model Training', 'student_cnn.keras')
print(f"Trying to load model from: {model_path}")
print(f"Absolute path: {os.path.abspath(model_path)}")
print(f"Exists: {os.path.exists(model_path)}")

try:
    model = keras.saving.load_model(model_path)
    print("Model loaded successfully!")
    # Print a summary
    model.summary()
except Exception as e:
    print(f"Error loading model: {e}")
    import traceback
    traceback.print_exc()