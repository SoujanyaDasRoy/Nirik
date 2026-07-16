import os
from PIL import Image

frontend_dir = r"c:\Users\sdroy\OneDrive\Desktop\Documents\Final Year Project\frontend\public\model-results"
results_dir = r"c:\Users\sdroy\OneDrive\Desktop\Documents\Final Year Project\results\figures"

for name in os.listdir(frontend_dir):
    if name.endswith('.png'):
        p = os.path.join(frontend_dir, name)
        img = Image.open(p)
        print(f"--- {name} ({img.size}) ---")
        # Let's check color channels or simple properties
