import os
import sys
import json
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.inference import predict_image

def run_tests():
    images = [
        "vis_test_non_tb_1.jpg",
        "vis_TEST_nx10.jpg",
        "vis_TEST_px38.jpg",
        "vis_test_tb_1.jpg"
    ]
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "visualizations"))
    
    for img_name in images:
        img_path = os.path.join(base_dir, img_name)
        if not os.path.exists(img_path):
            print(f"File not found: {img_name}")
            continue
            
        print(f"\n{'='*50}\nResults for: {img_name}\n{'='*50}")
        try:
            img = Image.open(img_path)
            res, _ = predict_image(img)
            
            xai = res.get('xai_results', {})
            
            print(f"Prediction: {res.get('prediction')} (Confidence: {res.get('confidence', 0)*100:.1f}%)")
            
            # Print the summary
            print("\nXAI Summary:")
            print(xai.get('summary', 'No summary available'))
            
            # Print ROIs
            print("\nRegions of Interest (ROIs):")
            rois = xai.get('rois', [])
            if not rois:
                print("No ROIs found.")
            else:
                for roi in rois:
                    print(f"- Region {roi.get('id')}:")
                    print(f"  Location: {roi.get('location')}")
                    print(f"  Contribution: {roi.get('contribution_pct')}%")
                    print(f"  Activation Score: {roi.get('activation_score')}%")
                    print(f"  Center (x,y): {roi.get('center')}")
                    
        except Exception as e:
            print(f"Error processing {img_name}: {str(e)}")

if __name__ == "__main__":
    run_tests()
