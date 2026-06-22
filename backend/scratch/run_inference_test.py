import os
import sys
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.inference import predict_image

def test_inference():
    test_img_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "visualizations", "vis_test_tb_1.jpg"))
    if not os.path.exists(test_img_path):
        print(f"Test image not found at {test_img_path}")
        return

    print(f"Testing with image: {test_img_path}")
    img = Image.open(test_img_path)
    res, gradcam_img = predict_image(img)
    
    print(f"Prediction: {res['prediction']}")
    print(f"Confidence: {res['confidence']:.4f}")
    print(f"Threshold Used: {res['threshold_used']}")
    print(f"Is TB: {res['is_tb']}")
    print(f"Demo Mode: {res['demo_mode']}")
    print(f"Saliency Fallback: {res['saliency_fallback']}")
    
    xai = res.get('xai_results', {})
    print(f"XAI Metrics: {xai.get('metrics')}")
    print(f"XAI Summary: {xai.get('summary')}")
    
    if not res['saliency_fallback']:
        print("Success: Native Saliency maps generated!")
        out_path = os.path.join(os.path.dirname(__file__), "test_output_heatmap.jpg")
        gradcam_img.save(out_path)
        print(f"Saved {out_path}")
    else:
        print("Fallback was triggered!")

if __name__ == "__main__":
    test_inference()
