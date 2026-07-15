import io
import json
import requests
from PIL import Image

# Base URL
BASE_URL = "http://127.0.0.1:5000"

# Login to get session
login_url = f"{BASE_URL}/login"
login_data = {"username": "reviewer", "password": "password123"}
session = requests.Session()
login_response = session.post(login_url, json=login_data)
if login_response.status_code != 200:
    print(f"Login failed: {login_response.status_code}")
    print(login_response.text)
    exit(1)

print("Login successful")

# Create a simple black image (PNG) in memory
img = Image.new('RGB', (224, 224), color=(0, 0, 0))
img_byte_arr = io.BytesIO()
img.save(img_byte_arr, format='PNG')
img_byte_arr = img_byte_arr.getvalue()

# Prepare the file for upload
files = {'file': ('test.png', io.BytesIO(img_byte_arr), 'image/png')}

# Test without explain
print("\nTesting without explain flag...")
response_no_explain = session.post(f"{BASE_URL}/predict", files=files)
print(f"Status code: {response_no_explain.status_code}")
if response_no_explain.status_code == 200:
    data = response_no_explain.json()
    print("Response keys:", list(data.keys()))
    # Check that explainability fields are NOT present
    explain_fields = ['heatmaps', 'xai_results', 'quadrant_analysis', 'clinical_observations', 'validation', 'evidence', 'reasoning', 'findings', 'report']
    missing = [field for field in explain_fields if field not in data]
    if missing:
        print(f"Expected explainability fields missing (as expected): {missing}")
    else:
        print("WARNING: Explainability fields are present when they should not be")
else:
    print(f"Error: {response_no_explain.text}")

# Test with explain
print("\nTesting with explain flag...")
# We need to add the explain parameter to the form data
data = {'explain': 'true'}
response_explain = session.post(f"{BASE_URL}/predict", files=files, data=data)
print(f"Status code: {response_explain.status_code}")
if response_explain.status_code == 200:
    data = response_explain.json()
    print("Response keys:", list(data.keys()))
    # Check that explainability fields ARE present
    explain_fields = ['heatmaps', 'xai_results', 'quadrant_analysis', 'clinical_observations', 'validation', 'evidence', 'reasoning', 'findings', 'report']
    missing = [field for field in explain_fields if field not in data]
    if not missing:
        print("All explainability fields are present")
        # Check a few specific fields for structure
        if 'heatmaps' in data and isinstance(data['heatmaps'], dict):
            print("  - heatmaps is a dict")
            expected_heatmap_keys = ['gradcam', 'gradcam_plusplus', 'attention', 'coverage', 'attribution']
            for key in expected_heatmap_keys:
                if key in data['heatmaps']:
                    print(f"    - {key}: present")
                else:
                    print(f"    - {key}: missing")
        else:
            print("  - heatmaps is not a dict or missing")
    else:
        print(f"Missing explainability fields: {missing}")
else:
    print(f"Error: {response_explain.text}")

print("\nTest completed.")
