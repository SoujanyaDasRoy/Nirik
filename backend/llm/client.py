import os
import requests
import json
import traceback
from llm.guardrails import build_system_prompt, enforce_guardrails

try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False
    print("[LLM] WARNING: google-generativeai not installed. Gemini will not be available.")


def generate_template_fallback(llm_context: dict, user_message: str) -> str:
    """
    Generates a structured fallback radiology report template when the third-party
    LLM API is unreachable, times out, or is unconfigured.
    """
    pred_class = llm_context.get('prediction', 'Unknown')
    conf_score = f"{llm_context.get('confidence', 0) * 100:.1f}%"
    is_tb = llm_context.get('isTb', False)
    
    xai = llm_context.get('xaiResults', {}) or {}
    rois = xai.get('rois', []) if xai else []
    roi_str = ", ".join([f"{r.get('location')} (Contribution: {r.get('contribution', 0):.1f}%)" for r in rois]) if rois else "No focal ROI detected"
    
    if is_tb:
        impression = f"AI screening is suspicious for pulmonary tuberculosis based on the detected focal saliency anomalies (Confidence: {conf_score})."
        recommendations = (
            "1. Obtain sputum smear for Acid-Fast Bacilli (AFB) x 3.\n"
            "2. Correlate with molecular tests (GeneXpert MTB/RIF or Truenat).\n"
            "3. Clinical evaluation for constitutional symptoms (cough, fever, weight loss).\n"
            "4. Consult a qualified pulmonologist or radiologist for clinical confirmation."
        )
        patient_summary = "The AI model detected changes in the lungs that are suspicious for tuberculosis. Further laboratory tests and evaluation by a healthcare provider are required to establish a screening result."
    else:
        impression = f"No radiographic evidence of active pulmonary tuberculosis detected by the model (Confidence: {conf_score})."
        recommendations = (
            "1. Clinical correlation with patient's presenting symptoms.\n"
            "2. Repeat chest radiograph in 4-6 weeks if clinical symptoms persist.\n"
            "3. If clinical suspicion remains high, consider further microbiological investigation."
        )
        patient_summary = "The AI model did not detect any changes in the lungs suspicious for tuberculosis. Please consult your healthcare provider if you have persistent symptoms."

    fallback_report = f"""# AI-Assisted Chest X-ray Report (Fallback Template Mode)

## AI Prediction
* Predicted Class: {pred_class}
* Confidence: {conf_score}

## Findings
The chest radiograph was analyzed using the Nirikhshon screening model.
* Lung Saliency Zones: {roi_str}

## Impression
{impression}

## Explainability
The saliency analysis highlights the lung zones contributing most strongly to this screening result. This visualization represents neural attention weights and is intended for clinical correlation only.

## Recommendation
{recommendations}

## Patient-Friendly Summary
{patient_summary}"""

    return fallback_report


def generate_chat_response(llm_context: dict, user_message: str) -> str:
    """
    Generates a response using Gemini, a local LLM, or a fallback template.
    Falls back gracefully with informative error messages.
    """
    use_local = os.environ.get("USE_LOCAL_LLM", "false").lower() == "true"
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()

    # Build prompt
    system_prompt = build_system_prompt()
    # Format input variables to conform to the requested structured format (omitting patientId for PHI minimization)
    age = llm_context.get('patientAge', 'Unknown')
    sex = llm_context.get('patientSex', 'Unknown')
    view = llm_context.get('view', 'PA')
    quality = llm_context.get('imageQuality', {}) or {}
    quality_str = f"Exposure: {quality.get('exposure', 'Unknown')}, Coverage: {quality.get('coverage', 'Unknown')}, Quality Score: {quality.get('qualityScore', 'Unknown')}%" if quality else "Unavailable"
    
    pred_class = llm_context.get('prediction', 'Unknown')
    conf_score = f"{llm_context.get('confidence', 0) * 100:.1f}%"
    
    xai = llm_context.get('xaiResults', {}) or {}
    rois = xai.get('rois', []) if xai else []
    
    left_lung = "Not highlighted"
    right_lung = "Not highlighted"
    for r in rois:
        loc = r.get('location', '').lower()
        if 'left' in loc:
            left_lung = f"Attention highlights {r.get('location')} (Contribution: {r.get('contribution', 0):.1f}%)"
        if 'right' in loc:
            right_lung = f"Attention highlights {r.get('location')} (Contribution: {r.get('contribution', 0):.1f}%)"
            
    roi_str = ", ".join([f"{r.get('id')}: {r.get('location')} (Contribution: {r.get('contribution', 0):.1f}%, Peak Activation: {r.get('activation', 0):.2f})" for r in rois]) if rois else "No focal ROI detected"
    
    heatmap_loc = xai.get('ranking', [{}])[0].get('location', 'Unavailable') if xai and xai.get('ranking') else "Unavailable"
    heatmap_intensity = f"Peak activation centroid: {rois[0].get('activation', 0):.2f}" if rois else "Unavailable"
            
    context_str = f"""Patient Information
* Age: {age}
* Sex: {sex}

Image Information
* View (PA/AP): {view}
* Image Quality: {quality_str}

AI Classification
* Predicted Class: {pred_class}
* Confidence Score: {conf_score}
* Threshold Used: {llm_context.get('threshold', 0.5)}

Saliency & Heatmap Information
* Left Lung Attention: {left_lung}
* Right Lung Attention: {right_lung}
* Heatmap Peak Location: {heatmap_loc}
* Heatmap Peak Intensity: {heatmap_intensity}
* Region of Interest Details: {roi_str}
* XAI Consensus Summary: {xai.get('summary', 'Unavailable')}
"""
    full_prompt = (
        f"{system_prompt}\n\n"
        f"=== CURRENT CASE CONTEXT ===\n{context_str}\n\n"
        f"=== USER QUESTION ===\n{user_message}"
    )

    # --- Option 1: Local LLM (Ollama) ---
    if use_local:
        local_url = os.environ.get("LOCAL_LLM_URL", "http://localhost:11434/api/generate")
        local_model = os.environ.get("LOCAL_LLM_MODEL", "llama3")
        try:
            res = requests.post(local_url, json={
                "model": local_model,
                "prompt": full_prompt,
                "stream": False
            }, timeout=10.0)  # Bounded timeout for local LLM requests
            if res.status_code == 200:
                raw_text = res.json().get("response", "")
                return enforce_guardrails(raw_text)
            else:
                print(f"[LLM] Local LLM returned status code {res.status_code}. Using template fallback.")
                return enforce_guardrails(generate_template_fallback(llm_context, user_message))
        except Exception as e:
            print(f"[LLM] Local LLM request failed: {e}. Using template fallback.")
            return enforce_guardrails(generate_template_fallback(llm_context, user_message))

    # --- Option 2: Gemini API ---
    if not HAS_GENAI:
        print("[LLM] google-generativeai SDK missing. Using template fallback.")
        return enforce_guardrails(generate_template_fallback(llm_context, user_message))

    if not api_key:
        print("[LLM] GEMINI_API_KEY environment variable is not set. Using template fallback.")
        return enforce_guardrails(generate_template_fallback(llm_context, user_message))

    # Try models in order: best available first
    MODELS_TO_TRY = [
        "gemini-3.5-flash",                  # Latest & best (confirmed available)
        "gemini-2.5-flash-preview-05-20",    # 2.5 Flash preview fallback
        "gemini-2.0-flash",                  # Stable fallback
        "gemini-1.5-flash",                  # Always available fallback
    ]

    genai.configure(api_key=api_key)
    last_error = None

    for model_name in MODELS_TO_TRY:
        try:
            print(f"[LLM] Trying model: {model_name}")
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(
                full_prompt,
                request_options={"timeout": 10.0}  # Fixed 10-second timeout to prevent hanging
            )
            print(f"[LLM] Success with model: {model_name}")
            return enforce_guardrails(response.text)
        except Exception as e:
            print(f"[LLM] Model {model_name} failed: {e}")
            last_error = e
            continue

    # All models failed: return structured template fallback
    print(f"[LLM] All models failed. Falling back to template. Error: {last_error}")
    return enforce_guardrails(generate_template_fallback(llm_context, user_message))


def generate_chat_response_stream(llm_context: dict, user_message: str):
    """
    Generates a streaming response using Gemini, a local LLM, or a fallback template.
    Yields chunks of text processed through the sliding window guardrail filter.
    """
    use_local = os.environ.get("USE_LOCAL_LLM", "false").lower() == "true"
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()

    # Build prompt
    system_prompt = build_system_prompt()
    age = llm_context.get('patientAge', 'Unknown')
    sex = llm_context.get('patientSex', 'Unknown')
    view = llm_context.get('view', 'PA')
    quality = llm_context.get('imageQuality', {}) or {}
    quality_str = f"Exposure: {quality.get('exposure', 'Unknown')}, Coverage: {quality.get('coverage', 'Unknown')}, Quality Score: {quality.get('qualityScore', 'Unknown')}%" if quality else "Unavailable"
    
    pred_class = llm_context.get('prediction', 'Unknown')
    conf_score = f"{llm_context.get('confidence', 0) * 100:.1f}%"
    
    xai = llm_context.get('xaiResults', {}) or {}
    rois = xai.get('rois', []) if xai else []
    
    left_lung = "Not highlighted"
    right_lung = "Not highlighted"
    for r in rois:
        loc = r.get('location', '').lower()
        if 'left' in loc:
            left_lung = f"Attention highlights {r.get('location')} (Contribution: {r.get('contribution', 0):.1f}%)"
        if 'right' in loc:
            right_lung = f"Attention highlights {r.get('location')} (Contribution: {r.get('contribution', 0):.1f}%)"
            
    roi_str = ", ".join([f"{r.get('id')}: {r.get('location')} (Contribution: {r.get('contribution', 0):.1f}%, Peak Activation: {r.get('activation', 0):.2f})" for r in rois]) if rois else "No focal ROI detected"
    
    heatmap_loc = xai.get('ranking', [{}])[0].get('location', 'Unavailable') if xai and xai.get('ranking') else "Unavailable"
    heatmap_intensity = f"Peak activation centroid: {rois[0].get('activation', 0):.2f}" if rois else "Unavailable"
            
    context_str = f"""Patient Information
* Age: {age}
* Sex: {sex}

Image Information
* View (PA/AP): {view}
* Image Quality: {quality_str}

AI Classification
* Predicted Class: {pred_class}
* Confidence Score: {conf_score}
* Threshold Used: {llm_context.get('threshold', 0.5)}

Saliency & Heatmap Information
* Left Lung Attention: {left_lung}
* Right Lung Attention: {right_lung}
* Heatmap Peak Location: {heatmap_loc}
* Heatmap Peak Intensity: {heatmap_intensity}
* Region of Interest Details: {roi_str}
* XAI Consensus Summary: {xai.get('summary', 'Unavailable')}
"""
    full_prompt = (
        f"{system_prompt}\n\n"
        f"=== CURRENT CASE CONTEXT ===\n{context_str}\n\n"
        f"=== USER QUESTION ===\n{user_message}"
    )

    # Sliding-window guardrail filter for streaming text
    import re
    replacements = {
        r"\bshows\b": "appears consistent with",
        r"\bshow\b": "may suggest",
        r"\bconfirms\b": "suggests",
        r"\bconfirmed\b": "suspected",
        r"\bdiagnosed\b": "screened positive",
        r"\bdiagnosed with\b": "screened positive for",
        r"\bdiagnose\b": "screen",
        r"\bdiagnosis\b": "screening result",
        r"\bdiagnoses\b": "screening results",
        r"\bis present\b": "appears consistent with",
        r"\bthere is\b": "there may be",
        r"\bthere are\b": "there may be",
        r"\bdefinitely\b": "potentially",
        r"\bcertainly\b": "potentially",
        r"\babsolute certainty\b": "high suspicion",
        r"\bproves\b": "suggests",
        r"\bclear evidence of\b": "findings consistent with"
    }

    def filter_text(text: str) -> str:
        for bad_pattern, good_term in replacements.items():
            pattern = re.compile(bad_pattern, re.IGNORECASE)
            text = pattern.sub(good_term, text)
        return text

    disclaimer_footer = (
        "\n\n---\n"
        "AI-assisted decision support only. All visual observations require independent verification against the source image."
    )

    # --- Option 1: Local LLM (Ollama) Stream ---
    if use_local:
        local_url = os.environ.get("LOCAL_LLM_URL", "http://localhost:11434/api/generate")
        local_model = os.environ.get("LOCAL_LLM_MODEL", "llama3")
        try:
            res = requests.post(local_url, json={
                "model": local_model,
                "prompt": full_prompt,
                "stream": True
            }, timeout=10.0, stream=True)
            if res.status_code == 200:
                buffer = ""
                for line in res.iter_lines():
                    if line:
                        chunk_text = json.loads(line.decode('utf-8')).get("response", "")
                        buffer += chunk_text
                        buffer = filter_text(buffer)
                        if len(buffer) > 40:
                            yield buffer[:-40]
                            buffer = buffer[-40:]
                buffer = filter_text(buffer)
                yield buffer
                yield disclaimer_footer
                return
        except Exception as e:
            print(f"[LLM] Local LLM stream failed: {e}")

    # --- Option 2: Gemini API Stream ---
    if HAS_GENAI and api_key:
        MODELS_TO_TRY = [
            "gemini-3.5-flash",
            "gemini-2.5-flash-preview-05-20",
            "gemini-2.0-flash",
            "gemini-1.5-flash",
        ]
        genai.configure(api_key=api_key)
        
        for model_name in MODELS_TO_TRY:
            try:
                print(f"[LLM] Trying streaming model: {model_name}")
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(
                    full_prompt,
                    stream=True,
                    request_options={"timeout": 10.0}
                )
                
                buffer = ""
                for chunk in response:
                    if chunk.text:
                        buffer += chunk.text
                        buffer = filter_text(buffer)
                        if len(buffer) > 40:
                            yield buffer[:-40]
                            buffer = buffer[-40:]
                buffer = filter_text(buffer)
                yield buffer
                yield disclaimer_footer
                return
            except Exception as e:
                print(f"[LLM] Streaming model {model_name} failed: {e}")
                continue

    # --- Option 3: Fallback Template Stream ---
    fallback_report = generate_template_fallback(llm_context, user_message)
    fallback_report = filter_text(fallback_report)
    yield fallback_report
    yield disclaimer_footer
