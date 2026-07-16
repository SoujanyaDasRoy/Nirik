import os
import requests
import traceback
from llm.guardrails import build_system_prompt, enforce_guardrails

try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False
    print("[LLM] WARNING: google-generativeai not installed. Gemini will not be available.")


def generate_chat_response(llm_context: dict, user_message: str) -> str:
    """
    Generates a response using Gemini, a local LLM, or a mock.
    Falls back gracefully with informative error messages.
    """
    use_local = os.environ.get("USE_LOCAL_LLM", "false").lower() == "true"
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()

    # Build prompt
    system_prompt = build_system_prompt()
    # Format input variables to conform to the requested structured format
    patient_id = llm_context.get('patientId', 'Unknown')
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
    
    observations = llm_context.get('observations', [])
    findings_list = {
        "Opacity": "Not reported",
        "Consolidation": "Not reported",
        "Cavitation": "Not reported",
        "Fibrosis": "Not reported",
        "Pleural Effusion": "Not reported",
        "Calcification": "Not reported",
        "Other Findings": "None"
    }
    
    for obs in observations:
        lbl = obs.get('label', '').lower()
        narrative = obs.get('narrative', '').lower()
        full_text = f"{lbl} {narrative}"
        if 'opacity' in full_text:
            findings_list["Opacity"] = f"Detected: {obs.get('narrative')}"
        if 'consolidation' in full_text:
            findings_list["Consolidation"] = f"Detected: {obs.get('narrative')}"
        if 'cavitation' in full_text or 'cavity' in full_text:
            findings_list["Cavitation"] = f"Detected: {obs.get('narrative')}"
        if 'fibrosis' in full_text or 'fibrotic' in full_text:
            findings_list["Fibrosis"] = f"Detected: {obs.get('narrative')}"
        if 'effusion' in full_text or 'pleural' in full_text:
            findings_list["Pleural Effusion"] = f"Detected: {obs.get('narrative')}"
        if 'calcification' in full_text or 'calcified' in full_text:
            findings_list["Calcification"] = f"Detected: {obs.get('narrative')}"
            
    context_str = f"""Patient Information
* Age: {age}
* Sex: {sex}
* Patient ID: {patient_id}

Image Information
* View (PA/AP): {view}
* Image Quality: {quality_str}

AI Classification
* Predicted Class: {pred_class}
* Confidence Score: {conf_score}

Lung Segmentation
* Left Lung: {left_lung}
* Right Lung: {right_lung}
* Region of Interest: {roi_str}

Grad-CAM Information
* Heatmap Location: {heatmap_loc}
* Heatmap Intensity: {heatmap_intensity}

Detected Radiographic Findings
* Opacity: {findings_list['Opacity']}
* Consolidation: {findings_list['Consolidation']}
* Cavitation: {findings_list['Cavitation']}
* Fibrosis: {findings_list['Fibrosis']}
* Pleural Effusion: {findings_list['Pleural Effusion']}
* Calcification: {findings_list['Calcification']}
* Other Findings: {findings_list['Other Findings']}

Additional Notes
* Threshold Used: {llm_context.get('threshold', 0.5)}
* XAI Summary: {xai.get('summary', 'Unavailable')}
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
            }, timeout=30)
            if res.status_code == 200:
                raw_text = res.json().get("response", "")
                return enforce_guardrails(raw_text)
            else:
                return enforce_guardrails(f"Local LLM returned error {res.status_code}.")
        except Exception as e:
            print(f"[LLM] Local LLM request failed: {e}")
            return enforce_guardrails("Local LLM is unreachable. Please check Ollama is running.")

    # --- Option 2: Gemini API ---
    if not HAS_GENAI:
        return enforce_guardrails(
            "The google-generativeai SDK is not installed on this server. "
            "Please add 'google-generativeai>=0.8.0' to requirements.txt."
        )

    if not api_key:
        print("[LLM] ERROR: GEMINI_API_KEY environment variable is not set.")
        return enforce_guardrails(
            "The AI assistant is not configured (missing API key). "
            f"Based on the model data: the scan was classified as '{llm_context.get('prediction', 'Unknown')}' "
            f"with {llm_context.get('confidence', 0) * 100:.1f}% confidence. "
            "Please consult a qualified radiologist for clinical interpretation."
        )

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
            response = model.generate_content(full_prompt)
            print(f"[LLM] Success with model: {model_name}")
            return enforce_guardrails(response.text)
        except Exception as e:
            print(f"[LLM] Model {model_name} failed: {e}")
            last_error = e
            continue

    # All models failed
    return enforce_guardrails(
        f"The AI assistant encountered an error: {str(last_error)}. "
        f"The scan was classified as '{llm_context.get('prediction', 'Unknown')}' "
        f"with {llm_context.get('confidence', 0) * 100:.1f}% confidence. "
        "Please consult a qualified radiologist for clinical interpretation."
    )
