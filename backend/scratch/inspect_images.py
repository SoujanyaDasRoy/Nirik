import os
from PIL import Image

TEST_IMAGES_DIR = r"C:\Users\sdroy\OneDrive\Desktop\Documents\Test images"
images = [f for f in os.listdir(TEST_IMAGES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

print("Test Image Properties:")
for img_name in images:
    img_path = os.path.join(TEST_IMAGES_DIR, img_name)
    try:
        img = Image.open(img_path)
        print(f"File: {img_name} | Size: {img.size} | Mode: {img.mode} | Format: {img.format}")
    except Exception as e:
        print(f"Error reading {img_name}: {e}")
