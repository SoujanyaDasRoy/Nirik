# AI Tuberculosis Detection System API Specification

> **Source of truth:** the model architecture and training pipeline (`CNN Model Training/nirikNetMain.py`), and the two companion documents already in this repository — [`docs/nirikshon_architecture_spec.html`](nirikshon_architecture_spec.html) and [`docs/PROJECT_HANDOFF.md`](PROJECT_HANDOFF.md). The **existing** `backend/`/`hf_space/` code is explicitly **not** authoritative where it conflicts with these — it references an older custom-CNN student and stale thresholds that this specification does not carry forward. Where this document and the existing backend disagree, this document wins, and the backend needs to change to match it, not the other way around.
>
> Anything not yet decided is marked **`Pending Implementation`** rather than guessed. Do not treat a `Pending Implementation` marker as "not needed" — it means "needed, not yet designed or built."

---

## 1. API Overview

### Purpose of the backend

The backend is a thin orchestration layer around one model: a DenseNet-121 student, exported to ONNX, that screens a chest X-ray for findings suspicious of pulmonary tuberculosis. The backend's job is narrowly scoped:

1. Accept an uploaded chest X-ray image.
2. Run it through the **exact** preprocessing pipeline the model was trained with (lung segmentation → crop → CLAHE → resize — see Section 8).
3. Run the DenseNet-121 student and apply the **frozen, training-derived** decision threshold (never a hardcoded 0.5).
4. Optionally generate explainability heatmaps.
5. Optionally hand a structured summary of the result to an LLM for natural-language narration.
6. Return a single structured JSON response to the frontend.

The backend is **not** a diagnostic authority, does not store a definitive medical record, and does not make the clinical decision — per this project's own clinical-safety framing, it produces an "AI Screening Result," never a "diagnosis."

### Overall request flow

```
Frontend  →  Backend  →  Preprocessing  →  Segmentation  →  DenseNet-121  →  Threshold  →  (Explainability)  →  (LLM)  →  Frontend
```

### Relationship with frontend

The frontend (Vercel — see `PROJECT_HANDOFF.md` §4) never touches a model directly. It uploads an image, receives a JSON response, and renders it. All model-specific logic (preprocessing, thresholding, CAM generation) lives behind this API — the frontend has no knowledge of DenseNet-121, ONNX, or the training pipeline's internals.

### Relationship with the AI model

The backend loads **exactly one** classification model: the DenseNet-121 student, as exported ONNX (`densenet121_student.onnx`). It also loads the Attention U-Net (native Keras — no ONNX export exists for it; see Section 7). **The ResNet-50 teacher is never loaded by the backend under any circumstance** — it is a training-time-only artifact and loading it in a serving process would be a deviation from the architecture, not a variant of it.

### Relationship with the LLM

The LLM (Gemini 2.5 Flash, via a Google AI Studio API key — **Decided**, not yet implemented) sits strictly *after* the model produces a result. It receives only the structured output of that result (prediction, confidence, threshold) — never the raw image, never model internals. See Section 6.

---

## 2. System Architecture

```
Frontend (Vercel)
    │  multipart/form-data image upload
    ▼
Backend (Hugging Face Spaces)
    │
    ▼
Preprocessing
    │  read image (raster or DICOM) → grayscale
    ▼
Segmentation
    │  Attention U-Net → binary lung mask → largest-2-components
    │  → morphological cleanup → 8.5%-padded bounding-box crop
    ▼
Attention Module
    │  NOT a separate pipeline stage — attention gates are internal to the
    │  U-Net's decoder (Section 3 of the architecture spec). Nothing
    │  happens here that is architecturally distinct from Segmentation.
    ▼
DenseNet-121 Inference
    │  CLAHE → resize 224×224 → canonical 0–255 RGB → ONNX Runtime session
    │  → softmax [P(Normal), P(Tuberculosis)]
    ▼
(Threshold applied — frozen, Youden-derived, from metrics.json)
    ▼
Explainability (optional, on request)
    │  Grad-CAM / Grad-CAM++ / LayerCAM / EigenCAM
    │  ⚠ Pending Implementation — see Section 3, POST /explain
    ▼
LLM Context Builder
    │  assembles {prediction, confidence, threshold, warnings} into a
    │  structured, minimal payload — never the raw image
    ▼
LLM (Gemini 2.5 Flash) — optional, only on POST /chat
    ▼
Frontend Response
```

**Important correction to this flow, stated plainly:** "Attention Module" is listed in this diagram because it was requested in this shape, but there is no code, no service, and no separate processing step corresponding to it. It is folded entirely into Segmentation. Do not build a separate "attention service" — there is nothing for it to do.

---

## 3. API Endpoints

### `POST /predict`

**Purpose:** Run the full screening pipeline on one uploaded chest X-ray and return a prediction.

| | |
|---|---|
| **Method** | `POST` |
| **URL** | `/predict` |
| **Authentication** | `Pending Implementation` — no auth scheme has been decided. Assume none for a first implementation. |
| **Content-Type** | `multipart/form-data` |

**Expected Request**

| Field | Type | Required | Notes |
|---|---|---|---|
| `image` | file | Yes | Chest X-ray image. Format matches whatever `read_image_grayscale()` supports — standard raster formats via OpenCV, plus DICOM if `pydicom` is available server-side. |
| `include_explainability` | boolean | No (default `false`) | If `true`, bundles all four CAM heatmaps into this response instead of requiring a separate `POST /explain` call. Costs real latency — see Section 10. |

**Expected Response** — `200 OK`

```json
{
  "request_id": "b3f1c9d2-8e4a-4c7f-9a1b-2d5e6f7a8b9c",
  "prediction": "Tuberculosis",
  "confidence": 0.62,
  "probabilities": {
    "normal": 0.38,
    "tuberculosis": 0.62
  },
  "threshold_used": 0.42,
  "screening_note": "AI Screening Result — Suspicious for Pulmonary Tuberculosis",
  "warnings": [],
  "explainability": null,
  "llm_context": {
    "prediction": "Tuberculosis",
    "confidence": 0.62,
    "threshold_used": 0.42
  },
  "processing_time_ms": 812,
  "model_version": "densenet121_student-v1"
}
```

If `include_explainability: true` was sent, `explainability` is populated instead of `null` — see Section 5 for its shape.

**Possible Errors:** `400`, `415`, `422`, `500`, `503` — see Section 9.

**Example Request**

```
POST /predict
Content-Type: multipart/form-data; boundary=----WebKitFormBoundary

------WebKitFormBoundary
Content-Disposition: form-data; name="image"; filename="cxr_0001.png"
Content-Type: image/png

<binary image data>
------WebKitFormBoundary
Content-Disposition: form-data; name="include_explainability"

false
------WebKitFormBoundary--
```

---

### `POST /explain`

**Purpose:** Generate the four explainability heatmaps for an already-uploaded image, without re-running `/predict`'s full response. Kept separate from `/predict` by default because generating all four CAM variants requires four additional gradient-tape forward/backward passes — real latency, not something every screening request should always pay for.

| | |
|---|---|
| **Method** | `POST` |
| **URL** | `/explain` |
| **Authentication** | `Pending Implementation` |
| **Content-Type** | `multipart/form-data` |

**Status: `Pending Implementation` — blocked on an unresolved architectural question.** Grad-CAM and its variants require live TensorFlow gradient access to the Keras model's intermediate layers. The deployed student is ONNX-only. **This endpoint cannot be correctly implemented until one of the following is decided** (see `PROJECT_HANDOFF.md` §10, item 5):

- **Option A:** keep a native Keras copy of the student loaded *alongside* the ONNX one, used only for this endpoint.
- **Option B:** implement an ONNX-compatible CAM method (non-trivial — ONNX Runtime does not expose a gradient-tape equivalent).

This specification assumes **Option A** as the more immediately tractable path, but that choice has not actually been made by the project yet.

**Expected Request**

| Field | Type | Required | Notes |
|---|---|---|---|
| `image` | file | Yes | Same image previously sent to `/predict`. This design is deliberately **stateless** — the endpoint re-runs preprocessing rather than requiring a session/cache lookup, since no request-caching layer exists yet (see Section 7, Caching). This re-preprocesses the image, which is wasted work if it was already scored once; accepted as the simpler design for a first implementation. |
| `methods` | array of strings | No (default: all four) | Any subset of `["gradcam", "gradcam_plus_plus", "layercam", "eigencam"]`. |

**Expected Response** — `200 OK`

```json
{
  "request_id": "b3f1c9d2-8e4a-4c7f-9a1b-2d5e6f7a8b9c",
  "explainability": {
    "gradcam": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
    "gradcam_plus_plus": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
    "layercam": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
    "eigencam": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."
  },
  "processing_time_ms": 2140
}
```

**Possible Errors:** `400`, `415`, `422`, `500`, `503` (specifically: `503` if Option A/B above hasn't been implemented at all yet).

---

### `POST /chat`

**Purpose:** Let the frontend ask the LLM a follow-up question about an already-produced prediction.

**Status: `Pending Implementation`.** No LLM client code, prompt templates, or chat-history storage exist anywhere in this project yet. The provider is decided (Gemini 2.5 Flash, Google AI Studio API key); the endpoint itself is not built. This section specifies the intended contract. See Section 6 for full detail.

| | |
|---|---|
| **Method** | `POST` |
| **URL** | `/chat` |
| **Authentication** | `Pending Implementation` |
| **Content-Type** | `application/json` |

**Expected Request**

```json
{
  "session_id": "b3f1c9d2-8e4a-4c7f-9a1b-2d5e6f7a8b9c",
  "question": "What does this result mean?",
  "llm_context": {
    "prediction": "Tuberculosis",
    "confidence": 0.62,
    "threshold_used": 0.42
  }
}
```

**Expected Response** — `200 OK`

```json
{
  "session_id": "b3f1c9d2-8e4a-4c7f-9a1b-2d5e6f7a8b9c",
  "answer": "This AI screening flagged the X-ray as suspicious for pulmonary tuberculosis, with a confidence of 0.62 against an operating threshold of 0.42. This is a screening result, not a diagnosis — please correlate with clinical findings and confirmatory testing.",
  "disclaimer": "AI-assisted interpretation only. Clinician judgment is final."
}
```

**Possible Errors:** `400`, `422`, `429`, `500`, `503`.

---

### `GET /health`

**Purpose:** Liveness check — also confirms both required models (DenseNet-121 student, Attention U-Net) are actually loaded, not just that the process is running.

| | |
|---|---|
| **Method** | `GET` |
| **URL** | `/health` |
| **Authentication** | None |

**Expected Response** — `200 OK`

```json
{
  "status": "ok",
  "models_loaded": {
    "student": true,
    "segmentation_unet": true
  },
  "uptime_seconds": 4213
}
```

If a model failed to load, respond `503` (see Section 9), not `200` with a misleading `status: "ok"`.

---

### `GET /model/info`

**Purpose:** Expose model metadata for debugging/observability — version, architecture, parameter counts, training-run reference.

| | |
|---|---|
| **Method** | `GET` |
| **URL** | `/model/info` |
| **Authentication** | `Pending Implementation` (recommend: none, or internal-only) |

**Expected Response** — `200 OK`

```json
{
  "student": {
    "architecture": "DenseNet-121",
    "parameters": 7043650,
    "format": "ONNX",
    "opset": 13
  },
  "teacher": {
    "architecture": "ResNet-50",
    "parameters": 23600002,
    "note": "training-time only, never loaded in serving"
  },
  "threshold": 0.42,
  "threshold_source": "metrics.json:student_youden_threshold",
  "trained_on": "run_config.json timestamp — Pending Implementation to surface here"
}
```

---

### `GET /version`

**Purpose:** API/backend version, independent of model version — useful once the backend is actually being iterated on post-launch.

| | |
|---|---|
| **Method** | `GET` |
| **URL** | `/version` |
| **Authentication** | None |

**Expected Response** — `200 OK`

```json
{
  "api_version": "1.0.0",
  "model_version": "densenet121_student-v1"
}
```

---

## 4. Required Endpoints

Only these six. Nothing else is required by the architecture as it currently stands — no batch endpoint, no auth endpoints, no patient-record endpoints. Those are explicitly deferred to Section 12 (Future Extensions).

| Endpoint | Method | Status |
|---|---|---|
| `/predict` | `POST` | Spec complete, ready to implement |
| `/explain` | `POST` | Spec complete, **blocked** on the ONNX-vs-native-Keras explainability decision |
| `/chat` | `POST` | Spec complete, **`Pending Implementation`** — no LLM code exists yet |
| `/health` | `GET` | Spec complete, ready to implement |
| `/model/info` | `GET` | Spec complete, ready to implement |
| `/version` | `GET` | Spec complete, ready to implement |

---

## 5. Prediction Response Schema

Full schema for `POST /predict`'s response body.

| Field | Type | Always present? | Description |
|---|---|---|---|
| `request_id` | string (UUID) | Yes | Unique per request — lets `/explain` or `/chat` reference this result later once a caching layer exists (`Pending Implementation`; currently `/explain` is stateless and doesn't use this). |
| `prediction` | string | Yes | `"Tuberculosis"` or `"Normal"`. |
| `confidence` | float | Yes | Raw `P(Tuberculosis)` softmax value, `0.0`–`1.0`. |
| `probabilities` | object | Yes | `{ "normal": float, "tuberculosis": float }`, sums to `1.0` — the full softmax pair. |
| `threshold_used` | float | Yes | The frozen, Youden-derived operating threshold from the training run's `metrics.json`. **Never** `0.5` by default, **never** recomputed per-request. |
| `screening_note` | string | Yes | Clinical-safety-compliant text — see the mandated terminology in `PROJECT_HANDOFF.md` §1. |
| `warnings` | array of strings | Yes (may be empty) | E.g. `["No distinct lung region detected — full image used without cropping."]`, surfaced from the preprocessing pipeline's own silent-fallback behavior (see Section 8). |
| `explainability` | object or `null` | Yes | `null` unless `include_explainability: true` was requested. Shape matches `POST /explain`'s response (Section 3). |
| `llm_context` | object | Yes | The minimal structured payload that would be sent to `POST /chat` if the user asks a follow-up — exposed here so the frontend doesn't have to reconstruct it. |
| `processing_time_ms` | integer | Yes | Server-side wall-clock time for this request. |
| `model_version` | string | Yes | Identifies which trained model produced this result. |

### Fields requested in the original outline that are **not** included, and why

| Field | Why it's excluded |
|---|---|
| `heatmap` | Not a top-level field — heatmaps live nested under `explainability` (four named variants), since there is no single generic "heatmap" the architecture produces. |
| `attention_map` | **Not supported by the architecture at all.** Attention exists only inside the segmentation U-Net's decoder (internal gating), not as a classifier-level artifact. There is nothing to expose here. Do not build frontend UI expecting this field — it will never be populated. |
| `gradcam` (top-level) | Nested under `explainability.gradcam` instead, alongside its three siblings — a top-level field would misrepresent it as the only CAM method. |

---

## 6. LLM API

### Provider

**Gemini 2.5 Flash**, via a Google AI Studio API key. **Decided; not yet implemented.**

### How the frontend sends a question

`POST /chat` with `{ session_id, question, llm_context }` — see Section 3. The frontend is expected to carry `llm_context` forward from the original `/predict` response rather than re-deriving it.

### What context is sent to the LLM

**Only** the structured `llm_context` object: `prediction`, `confidence`, `threshold_used`. Nothing else. Specifically:

- **Never** the raw uploaded image.
- **Never** model weights, architecture internals, or training data.
- **Never** patient-identifying information beyond what a specific downstream report template strictly requires (no such template is defined yet).

### How previous chat history is stored

**`Pending Implementation`.** No session-storage mechanism (in-memory, database, or otherwise) has been designed. Two reasonable approaches, neither decided:

1. **Stateless:** the frontend resends the full conversation history with every `/chat` call.
2. **Stateful:** the backend stores a short-lived session (`session_id` → message list) server-side.

Recommend starting with (1) for a first implementation — no database dependency, consistent with the rest of this API's currently-stateless design (see `/explain`'s design note in Section 3).

### How prompts are generated

`Pending Implementation` in terms of actual template code, but the constraint is clear: the system prompt must (a) restrict the model to commenting only on the numeric fields it's given, (b) forbid diagnostic-sounding language ("confirmed," "diagnosed"), (c) never instruct the model to independently interpret an image it was never shown.

### How hallucinations are minimized

- The LLM is never given the image — it cannot fabricate a visual finding it never saw, because it never receives visual input at all in this design.
- The LLM is constrained to reference only the fields present in `llm_context` — no external knowledge injection about the specific patient.
- Terminology enforcement happens at the system-prompt level, not left to convention.

### What the LLM must NEVER do

- Never assert a confirmed diagnosis.
- Never receive the raw image.
- Never receive model weights or training data.
- Never receive patient PHI beyond what's strictly required.
- Never be presented as a replacement for clinician judgment — every response should carry the disclaimer shown in the example response above.

### Example request / response

See `POST /chat` in Section 3.

---

## 7. Model Loading

| Aspect | Specification |
|---|---|
| **Loading strategy** | Both required models load **once**, at process/container startup — never per-request, never lazily on first request. Lazy loading is explicitly **not recommended**: it would make the first real user's request pay a large, unpredictable cold-start cost on top of an already multi-second pipeline. |
| **Student weights** | `densenet121_student.onnx`, loaded via an ONNX Runtime `InferenceSession`. |
| **Segmentation weights** | `attention_unet.keras`, loaded via native Keras (`.keras` format) — no ONNX export path exists for it. Whether to add one is `Pending Implementation` / undecided (see `PROJECT_HANDOFF.md` §10, item 6). |
| **Teacher (ResNet-50)** | **Never loaded.** Training-time only. |
| **GPU/CPU handling** | **CPU-only.** This is a confirmed assumption baked directly into the training pipeline's own code (`benchmark_inference_latency()`'s docstring and the ONNX export cell's docstring both cite "the deployment target (HF Spaces free tier) is CPU-only"), not a new constraint introduced for this spec. Use ONNX Runtime's `CPUExecutionProvider`. |
| **Batch size** | Single-image inference (one screening upload at a time). The exported ONNX graph's input signature uses a dynamic batch dimension, so batch-of-N is technically possible, but nothing in the current architecture or product flow calls for it — see Section 12 for batch prediction as a future extension. |
| **Caching** | `Pending Implementation`. No caching strategy (of preprocessed tensors, of repeated identical uploads, or otherwise) has been designed. |
| **Lazy loading** | Not recommended (see Loading strategy above). |
| **Error handling** | If either required model fails to load at startup, the process should **fail its own health check** (`GET /health` → `503`) rather than start "successfully" and fail unpredictably on the first real request. |

---

## 8. Image Processing Pipeline

This must **exactly** replicate `preprocess_xray()` from the training pipeline — this is a hard architectural requirement, not a suggestion, stated repeatedly across this project's documentation.

| Stage | Detail |
|---|---|
| **Supported formats** | Whatever `read_image_grayscale()` accepts: standard raster formats via OpenCV, plus DICOM via `pydicom` if present server-side. |
| **Validation** | An unreadable/corrupted file raises a `ValueError` in the training code's equivalent function — map this to a `422` (see Section 9). |
| **Segmentation** | Attention U-Net → binarize at `>0.5` → keep the largest 2 connected components → morphological open+close (5×5 kernel). |
| **Crop** | Bounding box of the cleaned mask, padded by **8.5%**, then cropped. If no lung region is detected at all, the current training-code behavior is to **silently fall back to the full, uncropped image** rather than error — the backend should surface this as a `warnings` entry (Section 5), not silently hide it from the frontend. |
| **Contrast enhancement** | CLAHE, `clipLimit=2.0`, `tileGridSize=(8, 8)`. |
| **Resize** | 224×224 (classification input size). |
| **Normalization** | **Not** a separate backend step. Each model (teacher and student alike) has its own `Lambda(preprocess_input)` layer baked into its graph — the backend only ever needs to produce the canonical 0–255 RGB tensor; per-model normalization happens inside the exported graph itself. |
| **Attention** | Not a separate step — internal to the segmentation stage (see Section 2). |
| **Inference** | DenseNet-121 student, ONNX Runtime, CPU. Output: softmax pair. |
| **Explainability** | Optional, on request — see `POST /explain` (Section 3), and its unresolved ONNX-vs-native-Keras blocker. |
| **Output generation** | Assembled into the response schema (Section 5) and returned as JSON. |

---

## 9. Error Responses

Consistent envelope for every error:

```json
{
  "error": {
    "code": "IMAGE_UNREADABLE",
    "message": "The uploaded file could not be read as an image.",
    "status": 422
  }
}
```

| Status | When | Example `code` |
|---|---|---|
| `400` | Malformed request (missing `image` field, invalid form data) | `BAD_REQUEST` |
| `401` | `Pending Implementation` — no auth scheme decided yet. Reserved for when one exists. | `UNAUTHORIZED` |
| `403` | `Pending Implementation` — same as above. | `FORBIDDEN` |
| `404` | Unknown route, or (for `/explain`/`/chat`) an unrecognized `request_id`/`session_id` once a caching/session layer exists | `NOT_FOUND` |
| `413` | Uploaded file exceeds the maximum accepted size (limit itself: `Pending Implementation` — see Section 10) | `PAYLOAD_TOO_LARGE` |
| `415` | Uploaded file's format isn't supported by `read_image_grayscale()` | `UNSUPPORTED_MEDIA_TYPE` |
| `422` | File is readable as *a* file but not a valid image (corrupted, zero-byte, etc.) — maps to the training code's own `ValueError` | `IMAGE_UNREADABLE` |
| `429` | Rate limit exceeded — limiting scheme itself is `Pending Implementation` | `RATE_LIMITED` |
| `500` | Unhandled server-side error | `INTERNAL_ERROR` |
| `503` | A required model isn't loaded (see Section 7), or the service is otherwise not ready to serve requests | `SERVICE_UNAVAILABLE` |

**Example — `422`:**

```json
{
  "error": {
    "code": "IMAGE_UNREADABLE",
    "message": "The uploaded file could not be read as an image.",
    "status": 422
  }
}
```

**Example — `503`:**

```json
{
  "error": {
    "code": "SERVICE_UNAVAILABLE",
    "message": "The classification model is not loaded. Please retry shortly.",
    "status": 503
  }
}
```

---

## 10. Performance Expectations

| Aspect | Expectation |
|---|---|
| **Inference time (classification only)** | ~750–870ms/image, measured directly on Kaggle's CPU-forced benchmark (architecture-only timing, independent of how well-trained the weights are). This is a **reference point from the training environment's hardware, not a guarantee for Hugging Face Spaces' actual free-tier CPU** — re-measure once deployed. |
| **Inference time (with all 4 explainability methods)** | Not yet measured end-to-end; expect multiple additional seconds (each CAM method requires its own gradient-tape forward/backward pass). Treat as `Pending Implementation` to actually benchmark once `/explain` is built. |
| **Maximum image size** | `Pending Implementation` — no explicit limit exists in the training code (the model resizes to 224×224 regardless of input size, so a very large upload mainly costs more at file-decode time, not inference time). Recommend a sensible cap (e.g., 20MB) at the API layer to bound worst-case decode time and abuse risk — not yet decided. |
| **GPU requirements** | **None** for serving — CPU-only by design (Section 7). Training required GPU (Kaggle, Tesla P100). |
| **Memory considerations** | `Pending Implementation` — not measured. DenseNet-121 (~7M params) plus the Attention U-Net are both modest by deep-learning standards; the practical constraint is more likely Hugging Face Spaces' free-tier memory ceiling than the models themselves. |
| **Concurrency** | `Pending Implementation`. Hugging Face Spaces' free tier is a known, real constraint on concurrent request handling — do not assume the same concurrency headroom a dedicated server would have. |
| **Timeout recommendation** | Given realistic multi-second latency (segmentation + classification + optional 4× CAM passes), recommend a generous client-side timeout — **15–30 seconds** — rather than a typical sub-5-second API timeout. |

---

## 11. Backend Folder Responsibilities

| Concern | Recommended location |
|---|---|
| **Routes** | `backend/api/routes/` — one file per endpoint group (`predict.py`, `explain.py`, `chat.py`, `health.py`). |
| **Controllers** | `backend/api/controllers/` — request validation, response shaping; delegates actual work to services below. |
| **Services** | `backend/services/` — orchestrates a full request (e.g. `prediction_service.py` calls preprocessing → inference → thresholding in sequence). |
| **Model Loader** | `backend/inference/model_loader.py` — owns startup-time loading of the ONNX student session and the native-Keras U-Net; owns the `GET /health`/`GET /model/info` "is everything loaded" state. |
| **Inference Engine** | `backend/inference/engine.py` — wraps the ONNX Runtime session call; the only place that knows about ONNX Runtime specifically. |
| **Preprocessing** | `backend/inference/preprocessing.py` — must mirror `preprocess_xray()` exactly; treat as a direct port, not a reimplementation from memory. |
| **Grad-CAM** | `backend/explainability/` — houses whichever of Option A/Option B (Section 3) is eventually chosen. Depends on that unresolved decision before it can be properly structured. |
| **Attention** | **No dedicated location** — there is no standalone attention component to house (see Section 2's correction). If anything, attention-gate logic belongs inside the segmentation model's own definition, not a separate module. |
| **LLM** | `backend/llm/` — `client.py` (Gemini API call), `prompts/` (templates — none exist yet), `guardrails.py` (terminology enforcement, input allow-listing). |
| **Configuration** | `backend/config/` — mirrors the training pipeline's `Config` dataclass where relevant (image size, CLAHE params, crop padding), plus the frozen threshold read from `metrics.json`. |
| **Utilities** | `backend/utils/` — shared helpers (image I/O, response envelope formatting, error-code constants). |

---

## 12. Future Extensions

Documented as possibilities only — none of these should influence the current architecture, and none are designed beyond the one-line description here.

| Extension | Description |
|---|---|
| **Multiple diseases** | Extending beyond binary Normal/Tuberculosis to a broader findings taxonomy. Would require retraining, not just a backend change. |
| **Batch prediction** | A `POST /predict/batch` accepting multiple images in one call. The ONNX graph's dynamic batch dimension would support this technically; nothing in the product flow currently calls for it. |
| **Doctor login** | Authentication/authorization for clinician-facing accounts. Directly resolves the `Pending Implementation` `401`/`403` markers throughout this document. |
| **Patient history** | Persisting screening results against a patient record over time. Would require a database layer that does not currently exist anywhere in this architecture. |
| **Audit logs** | Recording who requested what prediction, when, or reviewed which case — a compliance/governance concern layered on top of, not part of, the core screening pipeline. |

---

## 13. API Contract Summary

*(Everything a frontend developer needs, without reading the rest of this document.)*

**Base flow:** upload an image to `POST /predict`, render the response. Optionally call `POST /explain` for heatmaps, or `POST /chat` for LLM narration.

**Endpoints:**

| Endpoint | Method | Status |
|---|---|---|
| `/predict` | `POST` | Ready to implement |
| `/explain` | `POST` | Blocked — explainability/ONNX architecture decision pending |
| `/chat` | `POST` | Pending Implementation — no LLM code exists yet |
| `/health` | `GET` | Ready to implement |
| `/model/info` | `GET` | Ready to implement |
| `/version` | `GET` | Ready to implement |

**`/predict` response, the fields that matter most:**

```json
{
  "prediction": "Tuberculosis",
  "confidence": 0.62,
  "probabilities": { "normal": 0.38, "tuberculosis": 0.62 },
  "threshold_used": 0.42,
  "screening_note": "AI Screening Result — Suspicious for Pulmonary Tuberculosis",
  "warnings": [],
  "explainability": null,
  "llm_context": { "prediction": "Tuberculosis", "confidence": 0.62, "threshold_used": 0.42 },
  "processing_time_ms": 812
}
```

**Rules the frontend must follow:**

- Always display `threshold_used` next to `confidence` — the threshold is rarely 0.5, and showing confidence alone without it reads as arbitrary.
- Use `screening_note`'s language verbatim, or equivalently safe clinical framing — never "diagnosed," never "confirmed."
- There is no `attention_map` field and never will be under this architecture — do not build UI expecting one.
- `explainability` is `null` unless explicitly requested (`include_explainability: true`, or a separate `/explain` call) — don't assume it's always populated.
- Expect multi-second latency, especially with explainability included — design loading states accordingly (15–30 second client timeout recommended).
- `/chat` and the full `/explain` flow are **not built yet** — integrate against this contract, but expect them to be under active development, not stable.
