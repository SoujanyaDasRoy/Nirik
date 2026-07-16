import sys
import os
# Add the backend directory to sys.path so it matches the environment of app.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

print("Testing local backend LLM imports...")
try:
    from llm.client import generate_chat_response
    from llm.guardrails import enforce_guardrails, build_system_prompt
    print("Success: llm package imported successfully!")
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Check fallback / degraded response without GEMINI_API_KEY
llm_context = {
    "prediction": "Tuberculosis",
    "confidence": 0.88,
    "patientId": "PT-TEST-123",
    "isTb": True
}

print("\nCalling generate_chat_response without API Key...")
res = generate_chat_response(llm_context, "Explain this result")
print("Response text:")
print(res)

print("\nLocal LLM test completed.")
