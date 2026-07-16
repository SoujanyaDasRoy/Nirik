import sys
import os
# Add backend directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

print("Initializing Flask test client...")
from backend.app import app

# Enable testing mode
app.config["TESTING"] = True
app.config["CSRF_DISABLED"] = True

client = app.test_client()

# Mock session login
with client.session_transaction() as sess:
    sess["username"] = "reviewer"
    sess["role"] = "reviewer"

# 1. Test /chat/status
print("\nTesting GET /chat/status...")
resp = client.get("/chat/status")
print(f"Status: {resp.status_code}")
print(f"Body: {resp.get_json()}")

# 2. Test POST /chat (fallback degraded response)
print("\nTesting POST /chat (degraded response fallback)...")
chat_payload = {
    "message": "Why did the model classify this as Tuberculosis?",
    "llm_context": {
        "prediction": "Tuberculosis",
        "confidence": 0.85,
        "patientId": "PT-TEST-999",
        "isTb": True
    }
}
resp = client.post("/chat", json=chat_payload)
print(f"Status: {resp.status_code}")
print(f"Body: {resp.get_json()}")

# 3. Test POST /chat with mocked GEMINI_API_KEY environment variable to test loading / model trying flow
print("\nTesting POST /chat with simulated GEMINI_API_KEY...")
os.environ["GEMINI_API_KEY"] = "AIzaSyFakeKeyForTesting"
try:
    resp = client.post("/chat", json=chat_payload)
    print(f"Status: {resp.status_code}")
    print(f"Body: {resp.get_json()}")
finally:
    # Clean up environment
    if "GEMINI_API_KEY" in os.environ:
        del os.environ["GEMINI_API_KEY"]

print("\nFlask endpoints test completed successfully.")
