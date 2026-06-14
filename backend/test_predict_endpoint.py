import io
import urllib.request
import urllib.parse
from PIL import Image
import json

def test_predict():
    print("Testing /predict endpoint...")
    
    # 1. Create dummy image bytes
    img = Image.new("RGB", (400, 300), color=(100, 100, 100))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_bytes = img_byte_arr.getvalue()
    
    # 2. Prepare multipart form data
    boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
    data = []
    data.append(f'--{boundary}'.encode('utf-8'))
    data.append('Content-Disposition: form-data; name="file"; filename="test_xray.png"'.encode('utf-8'))
    data.append('Content-Type: image/png'.encode('utf-8'))
    data.append(b'')
    data.append(img_bytes)
    data.append(f'--{boundary}--'.encode('utf-8'))
    body = b'\r\n'.join(data)
    
    # 3. Send request
    req = urllib.request.Request(
        "http://127.0.0.1:5000/predict",
        data=body,
        headers={
            'Content-Type': f'multipart/form-data; boundary={boundary}',
            'Content-Length': len(body)
        }
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            res_body = response.read().decode('utf-8')
            res_json = json.loads(res_body)
            print("\n--- Endpoint Response ---")
            print("Success:", res_json.get("success"))
            print("Prediction:", res_json.get("prediction"))
            print("Confidence:", res_json.get("confidence"))
            print("Is TB:", res_json.get("is_tb"))
            print("Metadata Keys:", list(res_json.get("metadata", {}).keys()))
            print("Original Image Present?", "original_image" in res_json)
            print("Heatmap Image Present?", "heatmap_image" in res_json)
            
            # Basic validation
            assert res_json.get("success") is True, "Success key is not True"
            assert "prediction" in res_json, "Prediction key is missing"
            assert "confidence" in res_json, "Confidence key is missing"
            assert "is_tb" in res_json, "Is TB key is missing"
            assert "original_image" in res_json, "Original image missing"
            assert "heatmap_image" in res_json, "Heatmap image missing"
            
            print("\n✓ /predict ENDPOINT TEST PASSED SUCCESSFULLY!")
    except Exception as e:
        print("✗ Request failed:", e)
        if hasattr(e, 'read'):
            print("Response details:", e.read().decode('utf-8'))
        raise e

if __name__ == "__main__":
    test_predict()
