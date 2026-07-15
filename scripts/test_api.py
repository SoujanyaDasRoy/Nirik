import urllib.request
import urllib.parse
import json
import mimetypes

def post_multipart(url, file_path):
    import uuid
    boundary = uuid.uuid4().hex
    
    with open(file_path, 'rb') as f:
        file_content = f.read()
        
    body = (
        f'--{boundary}\r\n'
        f'Content-Disposition: form-data; name="file"; filename="test.jpg"\r\n'
        f'Content-Type: image/jpeg\r\n\r\n'
    ).encode('utf-8') + file_content + f'\r\n--{boundary}--\r\n'.encode('utf-8')
    
    req = urllib.request.Request(url, data=body)
    req.add_header('Content-Type', f'multipart/form-data; boundary={boundary}')
    
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read())
        print("API Response Keys:", list(data.keys()))
        if 'xai_results' in data:
            print("xai_results IS present in API response!")
            print("xai_results keys:", list(data['xai_results'].keys()))
        else:
            print("xai_results IS MISSING from API response!")

post_multipart("http://127.0.0.1:5000/api/predict", "backend/scratch/visualizations/vis_test_tb_1.jpg")
