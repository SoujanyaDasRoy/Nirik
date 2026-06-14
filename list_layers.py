import os
os.environ["KERAS_BACKEND"] = "torch"
import keras

model_path = os.path.join("backend", "tb_student_densenet121.keras")
model = keras.saving.load_model(model_path)
print("Model layers count:", len(model.layers))
# Print the last 15 layers
for i, layer in enumerate(model.layers[-15:]):
    print(f"Index {len(model.layers)-15+i}: Name={layer.name}, Class={layer.__class__.__name__}")
