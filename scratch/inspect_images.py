import os
from PIL import Image

results_dir = r"c:\Users\sdroy\OneDrive\Desktop\Documents\Final Year Project\results\figures"
frontend_dir = r"c:\Users\sdroy\OneDrive\Desktop\Documents\Final Year Project\frontend\public\model-results"

print("Frontend images:")
if os.path.exists(frontend_dir):
    for f in os.listdir(frontend_dir):
        if f.endswith('.png'):
            p = os.path.join(frontend_dir, f)
            img = Image.open(p)
            print(f"  {f}: {img.size} {img.mode} {os.path.getsize(p)} bytes")

print("\nResults figures:")
if os.path.exists(results_dir):
    for f in os.listdir(results_dir):
        if f.endswith('.png'):
            p = os.path.join(results_dir, f)
            img = Image.open(p)
            print(f"  {f}: {img.size} {img.mode} {os.path.getsize(p)} bytes")
