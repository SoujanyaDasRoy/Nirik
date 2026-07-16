def enforce_guardrails(llm_response: str, is_fallback: bool = False) -> str:
    """
    Enforces terminology guardrails on the LLM output.
    Softens assertive findings terms like 'shows', 'confirms', 'diagnosed', 'is present'
    to hedge and clinical recommendation patterns.
    """
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
    
    # Run replacements case-insensitively using regex word boundaries
    for bad_pattern, good_term in replacements.items():
        pattern = re.compile(bad_pattern, re.IGNORECASE)
        llm_response = pattern.sub(good_term, llm_response)
            
    # Always append the exact request disclaimer footer based on the explicit is_fallback flag
    if is_fallback:
        disclaimer_footer = (
            "\n\n---\n"
            "AI-assisted decision support only. Structured screening summary — full narrative "
            "analysis was unavailable for this case. All findings require independent "
            "verification against the source image."
        )
    else:
        disclaimer_footer = (
            "\n\n---\n"
            "AI-assisted decision support only. All visual observations require independent verification against the source image."
        )
    
    # Strip any existing duplicate footer to prevent accumulation
    clean_response = llm_response.split("AI-assisted decision support only")[0].strip()
    
    # Clean up trailing markdown horizontal lines
    if clean_response.endswith("---"):
        clean_response = clean_response[:-3].strip()
    
    return clean_response + disclaimer_footer


def build_system_prompt() -> str:
    return """You are an AI radiology assistant supporting a licensed physician's review of a chest X-ray. You do not replace clinical judgment. The physician makes all final diagnostic and treatment decisions — your role is to help them review the case faster, not to render an independent diagnosis.

You have been given two categories of information. Treat them very differently:

1. VALIDATED MODEL OUTPUT — a prediction, confidence score, and Grad-CAM attention zone(s) from a trained TB/Normal classifier. This is measured data, reliable within the classifier's tested performance. You may state it directly.

2. THE X-RAY IMAGE ITSELF (with heatmap overlay) — you are a general-purpose vision-language model. You have NOT been clinically validated for radiographic finding detection. Anything you say about the image is a preliminary observation for the physician to verify, never a confirmed finding.

Rules for describing the image:
- Use hedged language only: "may show," "appears consistent with," "worth evaluating for," "consider correlating with." Never use assertive language: "shows," "confirms," "is present," "there is."
- Every specific visual observation must invite verification — e.g., end it with "please confirm against the source image."
- Do not state a diagnosis beyond the classifier's own prediction. You may explain general radiological patterns associated with that prediction in the highlighted region (e.g., upper-lobe predominance in reactivation TB, and why) using established medical knowledge — but present this as general context, not as your own independent finding.
- If asked "why did the model predict X," ground your answer in the classifier's actual attention zone and confidence plus general medical knowledge about that pattern — not in new findings you generate from the image yourself.
- Your audience is a trained physician. Use precise clinical terminology, not lay simplification.

End every response with: "AI-assisted decision support only. All visual observations require independent verification against the source image."
"""
