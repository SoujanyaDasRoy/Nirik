import sys
import os
# Add the backend directory to sys.path so it matches the environment of app.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

print("Testing local backend LLM imports...")
try:
    from llm.client import generate_chat_response, generate_chat_response_stream
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


def test_sliding_window_stream():
    print("\nTesting sliding-window streaming guardrail filter...")
    
    # We will simulate a generator that yields text chunks
    class MockChunk:
        def __init__(self, text):
            self.text = text
            
    # Mocking genai response stream:
    # "confirms" is split: "con" in chunk 0, "firms" in chunk 1.
    # "shows" is split: "sho" in chunk 1, "ws" in chunk 2.
    # Trailing text "the end." is at the very end to check final flush.
    chunks = [
        MockChunk("This scan con"),
        MockChunk("firms the diagnosis of disease. It sho"),
        MockChunk("ws some features. Here is the end.")
    ]
    
    # Mock genai GenerativeModel to return this mock stream
    import google.generativeai as genai
    original_model = genai.GenerativeModel
    
    class MockGenerativeModel:
        def __init__(self, name):
            self.name = name
        def generate_content(self, prompt, stream=False, request_options=None):
            return chunks
            
    genai.GenerativeModel = MockGenerativeModel
    
    # Temporarily set API key to force Gemini code path
    os.environ["GEMINI_API_KEY"] = "AIzaSyFakeKeyForTesting"
    try:
        # Call generate_chat_response_stream
        stream = generate_chat_response_stream(
            llm_context={"prediction": "Tuberculosis", "confidence": 0.85, "isTb": True},
            user_message="test"
        )
        
        # Accumulate the stream output
        streamed_chunks = list(stream)
        full_text = "".join(streamed_chunks)
        
        print("Streamed chunks received:")
        for idx, c in enumerate(streamed_chunks):
            print(f"Chunk {idx}: {repr(c)}")
            
        print(f"\nFull reconstructed text:\n{full_text}")
        
        # Assertions
        assert "confirms" not in full_text, "Assertive term 'confirms' was not replaced!"
        assert "shows" not in full_text, "Assertive term 'shows' was not replaced!"
        assert "suggests" in full_text or "appears consistent with" in full_text or "suspected" in full_text, "Replacement terms missing!"
        assert "the end." in full_text, "Final flush failed: trailing text lost!"
        print("\nSuccess: Sliding window stream checks passed!")
        
    finally:
        # Clean up mock and env
        genai.GenerativeModel = original_model
        if "GEMINI_API_KEY" in os.environ:
            del os.environ["GEMINI_API_KEY"]


test_sliding_window_stream()
print("\nLocal LLM test completed.")
