import os
import torch
import numpy as np
from PIL import Image

# Force Keras to use PyTorch as its backend
os.environ["KERAS_BACKEND"] = "torch"
import keras

MODEL_PATH = r"C:\Users\sdroy\OneDrive\Desktop\Documents\Final Year Project\backend\tb_student_densenet121.keras"
TEST_IMAGES_DIR = r"C:\Users\sdroy\OneDrive\Desktop\Documents\Test images"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = keras.saving.load_model(MODEL_PATH)
model.to(DEVICE)

images = [f for f in os.listdir(TEST_IMAGES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

def preprocess_densenet_inference(img):
    # What was currently in core/inference.py
    w, h = img.size
    # pad to square
    if w > h:
        padded = Image.new(img.mode, (w, w), 0)
        padded.paste(img, (0, (w - h) // 2))
    else:
        padded = Image.new(img.mode, (h, h), 0)
        padded.paste(img, ((h - w) // 2, 0))
        
    gray = padded.convert("L")
    resized = gray.resize((224, 224))
    arr = np.array(resized, dtype=np.float32) / 255.0
    x = np.stack([arr, arr, arr], axis=-1)
    
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x

def preprocess_resnet50_training(img):
    # What was actually used during training
    w, h = img.size
    # pad to square
    if w > h:
        padded = Image.new(img.mode, (w, w), 0)
        padded.paste(img, (0, (w - h) // 2))
    else:
        padded = Image.new(img.mode, (h, h), 0)
        padded.paste(img, ((h - w) // 2, 0))
        
    gray = padded.convert("L")
    resized = gray.resize((224, 224))
    
    # In training, load_image did: read grayscale, resize to 224, stack 3 times
    arr = np.array(resized, dtype=np.float32)
    x = np.stack([arr, arr, arr], axis=-1)
    
    # tf.keras.applications.resnet50.preprocess_input(img)
    # Under the hood, this converts RGB to BGR and subtracts [103.939, 116.779, 123.68]
    # Since our R=G=B channels are identical, RGB -> BGR keeps them identical.
    # So we just subtract the BGR mean: [103.939, 116.779, 123.68]
    # For R, G, B channels:
    x[..., 0] -= 103.939
    x[..., 1] -= 116.779
    x[..., 2] -= 123.68
    return x

for img_name in images:
    img_path = os.path.join(TEST_IMAGES_DIR, img_name)
    img = Image.open(img_path)
    
    # 1. Current DenseNet Inference preprocess
    x_dn = preprocess_densenet_inference(img)
    t_dn = torch.tensor(x_dn).unsqueeze(0).to(DEVICE)
    
    # 2. ResNet50 training preprocess
    x_rn = preprocess_resnet50_training(img)
    t_rn = torch.tensor(x_rn).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        logit_dn = model(t_dn).squeeze().item()
        prob_dn = torch.sigmoid(model(t_dn)).squeeze().item()
        
        logit_rn = model(t_rn).squeeze().item()
        prob_rn = torch.sigmoid(model(t_rn)).squeeze().item()
        
    print(f"\nImage: {img_name}")
    print(f"  DenseNet Preprocessing: Logit = {logit_dn:.4f}, Prob = {prob_dn:.4f}")
    print(f"  ResNet50 Preprocessing: Logit = {logit_rn:.4f}, Prob = {prob_rn:.4f}")
