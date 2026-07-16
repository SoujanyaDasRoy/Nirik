import os
from PIL import Image

frontend_dir = r"c:\Users\sdroy\OneDrive\Desktop\Documents\Final Year Project\frontend\public\model-results"

for f in os.listdir(frontend_dir):
    if f.endswith('.png'):
        p = os.path.join(frontend_dir, f)
        img = Image.open(p)
        print(f"File: {f}, Size: {img.size}")
