import os
os.environ["KERAS_BACKEND"] = "torch"
import keras

MODEL_PATH = r"C:\Users\sdroy\OneDrive\Desktop\Documents\Final Year Project\backend\tb_student_densenet121.keras"
model = keras.saving.load_model(MODEL_PATH)
print("Last 10 layers:")
for idx, layer in enumerate(model.layers[-10:]):
    print(f"Layer {idx}: {layer.name} ({layer.__class__.__name__})")
    if hasattr(layer, "activation"):
        print(f"  activation: {layer.activation.__name__ if hasattr(layer.activation, '__name__') else layer.activation}")
