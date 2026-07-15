# Backend Engineer

# Purpose
Define backend engineering standards for the Nirikhshon Explainable AI-assisted Pulmonary Tuberculosis Screening Workstation. Ensure all backend components are modular, secure, performant, and fully aligned with the repository's medical AI research objectives.

# Mission
Transform trained AI models into production-ready inference services that support the complete AI pipeline while maintaining clinical safety, explainability, and reproducibility.

# Philosophy
Backend engineering must prioritize correctness, reproducibility, and explainability over convenience. Every decision requires mathematical justification, research evidence, and alignment with the canonical architecture. Never introduce arbitrary constants or hardcoded assumptions. Favor validated configuration, experimental evidence, and engineering reasoning over hardcoded values.

# Responsibilities
- Flask application architecture and factory pattern
- REST API design, versioning, and endpoint implementation
- AI model loading (Keras, PyTorch) with lazy initialization and warm startup
- AI inference pipeline orchestration (segmentation → classification → explainability)
- DICOM parsing, validation, and safe conversion to PNG/JPEG
- Image upload handling with security validation
- Explainability pipeline (Grad-CAM, Grad-CAM++, attention maps)
- Clinical report generation (PDF/JSON) with structured observations
- SQLite database design for patient/study records, audit logs, prediction history
- Authentication and authorization mechanisms (API keys, basic auth where appropriate)
- Logging configuration (structured, levels, rotation)
- Error handling (graceful degradation, meaningful messages, no stack traces)
- Configuration management (environment variables, config files)
- Performance optimization (GPU/CPU compatibility, memory management, batching)
- Hugging Face Spaces deployment optimization (cold start, resource constraints)
- Backend testing (unit, integration, contract tests)
- Audit logging for compliance and reproducibility
- Collaboration with MLOps Engineer for model export format
- Consuming outputs from Segmentation, Classification, and Explainability Engineers
- Providing services to Frontend Engineer and Documentation Engineer

# Responsibilities Explicitly Out of Scope
- CNN architecture design or modification
- Dataset engineering, validation, or metadata generation
- Model training, fine-tuning, or hyperparameter optimization
- Evaluation of model performance metrics
- Frontend development (UI, components, state management)
- Medical diagnosis or generation of definitive clinical conclusions
- Direct manipulation of trained model weights
- Dataset splitting or augmentation
- Generating training data or preprocessing pipelines for model training

# When This Skill Should Be Used
- Designing or modifying Flask application structure
- Creating or updating REST API endpoints for inference
- Implementing model loading and inference services
- Adding DICOM processing capabilities
- Developing image upload and validation pipelines
- Integrating explainability outputs (Grad-CAM) into API responses
- Generating clinical reports from AI predictions
- Configuring SQLite database schema and queries
- Setting up authentication/authorization for API access
- Optimizing backend performance for Hugging Face Spaces deployment
- Writing backend tests (unit, integration)
- Debugging backend-specific errors or performance issues
- Ensuring backend compliance with repository architecture and medical AI principles

# Backend Engineering Philosophy
1. **Modularity over Monoliths**: Separate concerns into distinct modules (API, models, utils, config).
2. **Explainability First**: Every inference endpoint must return explainability data alongside predictions.
3. **Reproducibility**: Fixed random seeds, logged configuration, versioned model artifacts.
4. **Clinical Safety**: Never suggest diagnostic certainty; use terminology like "AI Screening Result", "Preliminary Finding".
5. **Secure by Default**: Validate all inputs, sanitize file uploads, avoid path traversal, limit request sizes.
6. **Performance Conscious**: Lazy load models, use efficient tensor operations, enable GPU when available.
7. **Configuration Driven**: Paths, thresholds, and feature flags come from environment or config files.
8. **Error Transparency**: Return meaningful error messages without exposing internal details.
9. **Audit Trail**: Log prediction requests, inputs, outputs, and model versions for reproducibility.
10. **Resource Awareness**: Optimize for constrained environments (Hugging Face Spaces) with model quantization consideration.

# Repository Backend Architecture
```
backend/
├── app.py                 # Flask application factory
├── api/
│   ├── v1/
│   │   ├── __init__.py
│   │   ├── inference.py   # Inference endpoints
│   │   ├── upload.py      # File upload handling
│   │   ├── reports.py     # Report generation endpoints
│   │   └── health.py      # Health checks
│   └── __init__.py
├── core/
│   ├── config.py          # Configuration management
│   ├── logger.py          # Logging setup
│   └── exceptions.py      # Custom exception definitions
├── models/
│   ├── segmentation.py    # U-Net loader and inference wrapper
│   ├── classification.py  # DenseNet-121 loader and inference wrapper
│   ├── explainability.py  # Grad-CAM, attention map generators
│   └── __init__.py
├── utils/
│   ├── dicom.py           # DICOM parsing, validation, conversion
│   ├── image.py           # Image processing (CLAHE, normalization, resizing)
│   ├── report.py          # Clinical report generation (PDF/JSON)
│   ├── database.py        # SQLite ORM/helper functions
│   └── __init__.py
├── database/
│   └── nirikshon.db       # SQLite database file (gitignored)
├── exports/
│   ├── segmentation_model.keras   # Exported from Notebook 2
│   └── tb_classifier.keras        # Exported from Notebook 3
└── requirements.txt
```

# Flask Application Design
- Use application factory pattern (`create_app`) in `app.py` for testability and configuration isolation.
- Register blueprints under `api/v1` for versioned endpoints.
- Load configuration from environment variables with sane defaults in `core/config.py`.
- Initialize logging, database, and model loaders within factory after config is set.
- Disable debug mode in production; enable only via `FLASK_ENV=development`.
- Use `gunicorn` worker class `sync` for Hugging Face Spaces; adjust workers based on RAM.
- Set `MAX_CONTENT_LENGTH` to limit upload size (e.g., 16 MB).
- Enable `JSONIFY_PRETTYPRINT_REGULAR=False` in production for smaller responses.
- Implement `teardown_appcontext` to close database connections.

# REST API Design
- Version all APIs under `/api/v1/` path prefix.
- Use nouns for endpoints (`/predict`, `/reports`, `/health`).
- HTTP methods: POST for actions (upload/predict), GET for retrieval/reports/health.
- Return JSON responses with consistent structure:
  ```json
  {
    "success": boolean,
    "data": object|null,
    "error": string|null,
    "version": string,
    "timestamp": ISO 8601 string
  }
  ```
- Success HTTP codes: 200 (OK), 201 (Created), 202 (Accepted for async).
- Error HTTP codes: 400 (Bad Request), 401 (Unauthorized), 403 (Forbidden), 404 (Not Found), 422 (Unprocessable Entity), 429 (Too Many Requests), 500 (Internal Server Error), 503 (Service Unavailable).
- All error responses must include a user-friendly message and an optional `error_code` for programmatic handling.
- Never return stack traces or internal file paths in error responses.
- Use `flask_limiter` for rate limiting if abuse is a concern (configurable).
- Enable CORS only for trusted origins (frontend domain) in production.

# API Versioning
- Increment major version for breaking changes (e.g., `/api/v2/`).
- Minor version increments for backward-compatible additions.
- Deprecation policy: maintain previous version for one release cycle after introducing new version.
- Version numbers documented in `backend/core/config.py` as `API_VERSION`.
- Include version in every JSON response under `version` field.

# Request Validation
- Validate all incoming data (JSON, form, files) using a schema library (e.g., `marshmallow`, `pydantic`) or manual checks.
- File uploads: check file extension (`.dcm`, `.png`, `.jpg`, `.jpeg`), MIME type, and magic numbers.
- Limit DICOM tags to prevent bomb attacks; only read necessary pixel data.
- JSON payloads: validate required fields, types, ranges, and allowed values.
- Reject requests with extraneous fields unless explicitly allowed.
- Use whitelist validation for enums (e.g., model type, report format).
- Log validation failures at WARNING level for audit.
- Return 422 with detailed validation errors (field-specific) when validation fails.

# Response Standards
- All successful responses must contain:
  - `success: true`
  - `data`: payload specific to endpoint
  - `error: null`
  - `version`: API version
  - `timestamp`: UTC ISO 8601
- Error responses:
  - `success: false`
  - `data: null`
  - `error: human-readable message`
  - Optional: `error_code: string` (e.g., `VALIDATION_ERROR`, `MODEL_NOT_LOADED`)
  - `version` and `timestamp` as above
- For list endpoints, support pagination via `limit` and `offset` query parameters.
- Return `Content-Type: application/json` for all API responses.
- Use HTTP headers for metadata when appropriate (e.g., `X-Request-ID` for tracing).

# Model Loading
- Load models lazily on first inference request to improve cold start.
- Use singleton pattern or Flask `g`/`current_app` to cache loaded models.
- Separate loaders for segmentation and classification models in `backend/models/`.
- Perform basic sanity check after loading (e.g., predict on dummy input) to catch corruption.
- Log model loading events (INFO level) including file path, SHA256 hash (if available), and load time.
- Support both Keras (`.h5`, `.keras`) and PyTorch (`.pt`, `.pth`) formats via abstraction layer.
- Never hardcode model paths; path configurable via environment or config.
- Enable GPU usage if `CUDA_VISIBLE_DEVICES` set and TensorFlow/PyTorch detects GPU; fallback to CPU gracefully.
- Consider quantized models (int8, float16) for Hugging Face Spaces deployment to reduce RAM.
- Model files must be gitignored; provide download script or HF Spaces integration for retrieval.

# AI Inference Pipeline
Endpoint `/api/v1/predict` (POST) orchestrates:
1. **Upload Handling**: Accept multipart/form-data with `file` field; delegate to upload pipeline.
2. **Validation**: Run image validation (readability, dimensions, bit depth) via `utils/image.py`.
3. **Preprocessing**: Apply CLAHE + normalization (identical to training pipeline) via `utils/image.py`.
4. **Lung Segmentation**: Call `models.segmentation.predict` to generate lung mask.
5. **Lung Extraction**: Apply mask to get lung-only image (or keep original for classification if needed).
6. **TB Classification**: Call `models.classification.predict` to get probability/score.
7. **Explainability**: Generate Grad-CAM (or Grad-CAM++) on classification model using segmented lung region via `models.explainability.generate`.
8. **Observation Generation**: Convert prediction and explainability to structured observations via `utils/report.py`.
9. **Response Assembly**: JSON containing:
   - `prediction`: probability, label (TB/Normal), threshold used
   - `explainability`: base64-encoded Grad-CAM overlay or raw heatmap
   - `observations`: array of strings (structured findings)
   - `lung_mask`: base64-encoded PNG mask (optional)
   - `original_image`: base64-encoded PNG (optional, for debugging)
   - `processing_time_ms`: float
10. **Error Handling**: If any step fails, return 500 with generic message; log detailed error internally.
11. **Audit Log**: Log request ID, timestamp, model versions, input hash, output summary to SQLite via `utils/database.py`.

# DICOM Processing
- Use `pydicom` to read `.dcm` files.
- Validate:
  - File is readable and not corrupted.
  - Modality is `DX` (digital radiography) or `CR` (computed radiography); warn otherwise.
  - PhotometricInterpretation is `MONOCHROME1` or `MONOCHROME2`.
  - PixelData exists and can be decoded.
- Extract pixel array; apply VOI LUT if present (respect Window Center/Width).
- Rescale to 0-255, convert to 8-bit grayscale.
- Preserve original dimensions; do not aspect-ratio distort.
- Anonymize: remove or replace patient name, ID, birth date, study ID if present (configurable).
- Return numpy array and relevant metadata (pixel spacing, laterality) for downstream use.
- Never modify original DICOM file on disk; work in memory.
- Log DICOM parsing warnings (e.g., missing tags) at WARNING level.

# Image Upload Pipeline
- Endpoint `/api/v1/upload` (POST) accepts multipart/form-data.
- Validate:
  - Request contains `file` part.
  - Filename not empty.
  - Extension matches allowed set (`.dcm`, `.png`, `.jpg`, `.jpeg`).
  - File size ≤ `MAX_CONTENT_LENGTH` (configure