import os
import numpy as np
from PIL import Image

frontend_dir = r"c:\Users\sdroy\OneDrive\Desktop\Documents\Final Year Project\frontend\public\model-results"
results_dir = r"c:\Users\sdroy\OneDrive\Desktop\Documents\Final Year Project\results\figures"

frontend_files = [f for f in os.listdir(frontend_dir) if f.endswith('.png')]
results_files = [f for f in os.listdir(results_dir) if f.endswith('.png')]

print("Comparing frontend images to results figures...")

for ff in frontend_files:
    ff_path = os.path.join(frontend_dir, ff)
    img_ff = Image.open(ff_path).convert('RGB')
    
    best_match = None
    best_score = -1
    
    for rf in results_files:
        rf_path = os.path.join(results_dir, rf)
        img_rf = Image.open(rf_path).convert('RGB')
        
        # Resize rf to match ff
        img_rf_resized = img_rf.resize(img_ff.size)
        
        # Calculate correlation score (MSE or correlation coefficient)
        arr_ff = np.array(img_ff, dtype=np.float32)
        arr_rf = np.array(img_rf_resized, dtype=np.float32)
        
        # Standardize
        arr_ff -= arr_ff.mean()
        arr_rf -= arr_rf.mean()
        
        std_ff = arr_ff.std()
        std_rf = arr_rf.std()
        
        if std_ff > 0.1 and std_rf > 0.1:
            arr_ff /= std_ff
            arr_rf /= std_rf
            score = (arr_ff * arr_rf).mean()
        else:
            score = -1
            
        if score > best_score:
            best_score = score
            best_match = rf
            
    print(f"Frontend: {ff} -> Best Match: {best_match} (Correlation Score: {best_score:.4f})")
