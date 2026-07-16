import os
import shutil

src_dir = r"c:\Users\sdroy\OneDrive\Desktop\Documents\Final Year Project\frontend\public\model-results"
dest_dir = r"C:\Users\sdroy\.gemini\antigravity-ide\brain\b0a53795-d6c5-419b-9686-0df7108ec2f4"

for f in os.listdir(src_dir):
    if f.endswith('.png'):
        src_path = os.path.join(src_dir, f)
        dest_path = os.path.join(dest_dir, f)
        shutil.copy(src_path, dest_path)
        print(f"Copied {f} to artifacts folder")
