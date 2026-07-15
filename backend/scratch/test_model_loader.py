import os
import tensorflow as tf
import traceback

def main():
    # Print TensorFlow version
    print(f"TensorFlow version: {tf.__version__}")

    # Construct absolute path to the model
    # Assuming this script is in backend/scratch, go up three levels to project root
    # __file__: backend/scratch/test_model_loader.py
    # dirname: backend/scratch
    # dirname: backend
    # dirname: <project_root>
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path = os.path.join(base_dir, 'CNN Model Training', 'student_cnn.keras')

    print(f"Loading model from: {model_path}")

    try:
        # Load the model using TensorFlow's Keras
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully!")
        print(f"Input shape: {model.input_shape}")
        print(f"Output shape: {model.output_shape}")
    except Exception as e:
        print("Failed to load model:")
        print(f"Exception: {e}")
        print("\nFull traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    main()