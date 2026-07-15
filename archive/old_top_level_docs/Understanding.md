# Project Understanding: Nirikhshon TB Screening

> Updated 2026-07-04 after reviewing current `backend/`, `frontend/`, `notebooks/`, and recent git history.

## What This Project Is

**Nirikhshon** is an AI-assisted workstation for screening **Pulmonary Tuberculosis (TB)** from chest X-rays. The repo has matured from a two-stage research prototype into a **deployed clinical screening application** with a FastAPI-style Flask backend, PACS-grade Next.js frontend, persistent audit trail, and a live Hugging Face Space production deployment.

Two pipelines:
1. **Training** — Kaggle notebooks (offline, frozen)
2. **Serving** — Flask backend + Next.js frontend (live, evolved significantly)

---

## 1. Training Pipeline (Kaggle Notebooks)

Two-stage cascade. Each stage uses a different model with a different job.

### Stage 1 — Lung Segmentation (U-Net)
- `notebooks/kaggle_unet_training.md`
- 4-level encoder-decoder U-Net (32→64→128→256→512 channels), conv blocks with BatchNorm + ReLU, dropout 0.2/0.3 in deep layers, sigmoid output, binary cross-entropy loss, IoU metric
- Trained on **Montgomery + Shenzhen** (only public TB CXR datasets with manual lung masks)
- 256×256 grayscale, 80/20 split, augmentations: horizontal flip, 90° rotations, brightness jitter
- Adam lr 1e-4, 40 epochs, save best by val IoU
- After training, U-Net pre-segments TB Database / NIH / NIRT India DICOM data so Stage 2 never sees clavicles, shoulders, or text
- **Now deployed** in the backend at `backend/unet_lung_segmenter.keras` (155 MB)

### Stage 2 — TB Classification (DenseNet-121 student)
- `notebooks/kaggle_classifier_training.md`
- DenseNet-121 (ImageNet pretrained) with `nn.Sequential(Dropout(0.4), Linear)` head, single logit output
- ResNet-50 teacher + KD scaffold defined (`DistillationLoss`, T=4, α=0.7) — KD was the training framework, but the **winning model is the student alone**
- Input: 224×224 RGB, ImageNet normalization, augmentations: random crop with padding, h-flip, ±10° rotation, brightness/contrast jitter
- Splits: **GroupShuffleSplit on `patient_id`** to prevent patient leakage — 70/15/15
- Class + domain imbalance via `WeightedRandomSampler` — NIRT (India) gets 3× weight
- Adam lr 1e-4 + weight decay 1e-4, 10 epochs, save best by val AUC
- **Threshold chosen for 95% recall** in training; production calibration is now **0.62** (see below)
- Grad-CAM on `features.denseblock4.denselayer16.conv2` to verify model focuses on lung tissue (not shoulders/text)
- **Now deployed** in backend at `backend/tb_student_densenet121.keras` (30 MB)

---

## 2. Serving Stack — Backend (`backend/`)

**Single Flask app** (`app.py`, 32 KB) with modular imports from `core/`, `utils/`, and `api_v1`. Runs via `flask_socketio` with threading async mode.

### Inference pipeline (`core/inference.py`)

The inference flow has **two model generations** that auto-select based on whether `unet_lung_segmenter.keras` is present:

| Generation | Trigger | Preprocessing |
|------------|---------|---------------|
| **Gen 1** | No U-Net deployed | ResNet50/caffe-style BGR mean subtraction (matches training bug in the original DenseNet pipeline) |
| **Gen 2** | `unet_lung_segmenter.keras` present | DenseNet/torch mode: ÷255, subtract mean, ÷std. **Current mode — U-Net is deployed** |

**Flow per request:**
1. Auth (session or `Authorization: Bearer user:role` header for cross-origin)
2. CSRF double-submit cookie check (skipped for `/api/v1/` and Bearer-auth)
3. Validate upload is actually a chest X-ray (`validate_chest_xray`)
4. Pad to square → grayscale → U-Net segment lungs (256×256) → resize to 224×224 → preprocess
5. DenseNet forward → sigmoid → threshold → `is_tb` / probability
6. Generate **5 saliency maps** in parallel: `gradcam`, `gradcam_plusplus`, `attention`, `coverage`, `attribution` (each with own colormap, max-alpha, low-activation threshold)
7. Target layer for Grad-CAM is the post-concat **`relu`** layer (1024 channels) — NOT `conv5_block16_2_conv` (only 32 channels, pre-norm). This was a bug fix.
8. For normal scans, gradient target is `-logit` so the map highlights evidence of normality
9. Extract ROIs via OpenCV contours, map to anatomical zones (Upper/Mid/Lower × Left/Right)
10. Compute **quadrant analysis** (upper vs lower activation distribution) → disease overlap inference
11. Generate clinical summary text + calibrated confidence + reliability/uncertainty

### Threshold (calibrated)

- `backend/best_threshold.txt` = **`0.62`** (overrides the in-code default 0.93)
- `backend/model_metadata.json` = `optimal_threshold: 0.62`
- Reasoning: trained at 95% recall, but the Gen 2 retrained student is well-calibrated at a lower threshold

### Clinical metrics (deployment_metrics.txt)

| Model | Params | Size | Latency | Throughput | Acc | AUC | F1 |
|-------|--------|------|---------|------------|-----|-----|-----|
| Teacher (ResNet-50) | 23.59M | 471.92 MB | 7.58 ms | 132 img/s | 0.658 | 0.924 | 0.565 |
| **Student (DenseNet-121)** | 7.04M | 29.66 MB | 14.81 ms | 67.5 img/s | **0.938** | **0.964** | **0.856** |

Student is 3.35× smaller, 1.95× **slower** (DenseNet dense-connectivity overhead) but far more accurate — speedup/compression claim is misleading; this is honestly documented.

### Endpoints (Flask routes)

**Auth & session:**
- `POST /login`, `POST /logout`, `GET /session`
- Session cookie: `SameSite=None; Secure` for Vercel↔HF deployments

**Prediction:**
- `POST /predict` — multipart file upload (PNG/JPG/DICOM), returns full result with metadata, heatmaps, XAI, quadrant analysis
- `GET /studies/<id>/heatmaps` — re-runs heatmaps on stored original
- `POST /studies/<id>/save` — save study result

**Patient & study management:**
- `GET/POST /patients`, `PUT /patients/<id>`, `POST /patients/<id>/archive`
- `GET /patients/<id>/history`
- `GET /studies` (filter by date/status/reviewer)
- `GET /studies/<id>/audit` — full audit trail
- `GET /studies/<id>/similar` — similar cases from DB

**Dashboard / analytics:**
- `GET /dashboard/stats` — total cases, TB positive, pending reviews, completed reviews, disease distribution, confidence distribution, model performance, reviewer agreement
- `GET /model/metadata` — current model threshold, metrics, dataset tracking
- `GET /model/tsne` — t-SNE embeddings (`tsne_embeddings.json`, 41 KB)
- `GET /audit/logs` — admin-only audit trail
- `GET /export/research?format=csv|json` — research export

**Notifications & feedback:**
- `GET /notifications`, `POST /notifications/<id>/read`
- `POST /feedback` — clinician override + annotation, logs to audit
- `POST /report/audit` — PDF report generation log

**Mock FHIR / PACS:**
- `GET /fhir/patients?search=&count=` — FHIR R4 Bundle
- `GET /fhir/pacs/status` — mock PACS/DICOM node statuses

**WebSocket:**
- `socketio.emit("study_updated", ...)` on upload, inference complete, review, report download

**`api_v1` blueprint** (`/api/v1/*`, X-API-Key auth via `api_keys` table):
- `/api/v1/predict` — 501 not implemented
- `/api/v1/studies` — list studies
- `/api/v1/analytics` — system analytics

### Persistence (`utils/patient_db.py`, 38 KB)

- SQLite (`backend/patients.db`, 121 MB — significant data accumulated)
- Tables: patients, studies, results, audit_trail, notifications, api_keys
- Functions: `save_result`, `get_history`, `create_patient`, `update_patient`, `archive_patient`, `search_patients`, `create_study`, `save_prediction_record`, `save_review_record`, `log_audit_event`, `create_notification`, `list_notifications`, `mark_notification_read`, `get_dashboard_stats`, `list_studies`, `get_study_audit_trail`, `get_similar_cases`, `get_research_export_data`

### CORS / Security
- Dynamic allowlist: localhost + any `*.vercel.app` + regex `^https?://.*$` for LAN showcase testing
- `DESKTOP_APP=true` env var adds explicit localhost regex
- CSRF: double-submit cookie, `SameSite=None; Secure` on HTTPS
- Bearer header `Authorization: user:role` for cross-origin clients that block cookies

### Bug fix history (from git log)

- **`d1ee50e` — Fixed 6 bugs in Kaggle training pipeline:** Grad-CAM layer target, preprocessing mismatch, teacher training curves, speedup label, teacher F1, RNG seed
- **`b1d66c1` — login reset, heatmap artifacts, git LFS setup**
- **`32bc306` — Updated optimal threshold to 0.62**, added anatomical quadrant analysis to UI
- **`15fc898` — U-Net lung segmentation added to inference** with auto-detect and graceful fallback
- **`62b8ea0` — Backend Grad-CAM gradient target sign** for normal scans (now uses `-logit`)
- **`02c91f9` — Removed threshold slider**, replaced with calibration preset buttons in settings
- **`4af0dd7` — Diagnosis color logic:** Red high / Amber medium / Green low
- **`46c7ccf` — CORS credentials preflight override** for Bearer auth
- **`dfd6539` — Frontend default API fallback** to live HF backend

---

## 3. Serving Stack — Frontend (`frontend/`)

**Next.js 14 + React 18 + TypeScript + Tailwind** (PACS dark mode, radiology reading room aesthetic).

### Architecture

- `page.tsx` (31 KB) — Landing portal
- `components/ScreeningTab.tsx` — main clinical workspace
- `components/DicomViewer.tsx` (dynamic, no SSR) — DICOM rendering with window/level, pixel spacing ruler, bounding box overlay
- `components/AnnotationCanvas.tsx` — ROI drawing
- `components/XaiVisualization.tsx` — XAI explainer card
- `components/TsnePlot.tsx` — t-SNE visualization
- `components/LongitudinalTracker.tsx` — change-over-time
- `hooks/useFileUpload.ts`, `hooks/usePrediction.ts`, `hooks/useFileUpload.ts`
- `services/` — modular service layer:
  - `predictionService` — diagnosis categorization
  - `heatmapService` — heatmap type selection
  - `observationService` — observation tracking
  - `imageQualityService` — IQA pass-through
  - `reportService` — PDF report generation
  - `auditService` — audit log retrieval
  - `exportService` — research data export
- `__tests__/` — Jest + React Testing Library (`hooks.test.tsx`, `usePrediction.dedup.test.tsx`, `usePrediction.error.test.tsx`)

### Default API URL

`process.env.NEXT_PUBLIC_API_URL` || `https://projectmantra-nirikshon-backend.hf.space`

The frontend **defaults to the live HF Space** — works without local backend running.

### UX features

- Dark mode for radiology reading rooms
- Mouse-interactive window/level controls
- Scale-calibrated physical rulers (using DICOM `pixel_spacing`)
- Bounding box annotations (manually drawn + auto from XAI ROIs)
- Clinical report builder with PDF export
- Color logic: Red high / Amber medium / Green low risk
- Evidence cards with anatomical zones (Upper/Mid/Lower × Left/Right)
- Quadrant analysis display
- **5 heatmap overlays** (gradcam, gradcam_plusplus, attention, coverage, attribution) with method-specific colormaps
- Context-sensitive opacity slider for AI overlay
- Settings tab with calibration preset buttons (replaces old threshold slider)
- Batch upload with concurrency limit of 3 (`analyzeAll`)

### `usePrediction` hook — production hardening

- `inFlight: Set<number>` ref to dedupe concurrent invocations on the same file
- Reads response body as text first, then attempts JSON parse (handles 4xx JSON + 5xx plain text)
- Always local-preview URL via `URL.createObjectURL` so DicomViewer shows image during inference
- `credentials: "include"` for CSRF double-submit validation
- `setResults(prev => ...)` immutable updates
- Clears `inFlight` in `finally`

---

## 4. Data Flow

```
Raw CXR / DICOM
      │
      ▼  (multipart upload, authenticated)
[ Flask: /predict ]
      │
      ▼
[ validate_chest_xray ] → reject if not a CXR
      │
      ▼
[ U-Net Lung Segmenter ]  ← unet_lung_segmenter.keras (auto-detected)
      │  (Generation 2 preprocessing: DenseNet/torch mode)
      ▼
[ DenseNet-121 Classifier ]  ← tb_student_densenet121.keras
      │  threshold = 0.62 (from best_threshold.txt or model_metadata.json)
      ▼
Normal / TB + confidence
      │
      ▼
[ 5 Saliency Maps: gradcam, gradcam++, attention, coverage, attribution ]
      │
      ▼
[ ROI extraction → anatomical zones → quadrant analysis → XAI payload ]
      │
      ▼
[ Calibrated confidence + reliability/uncertainty + clinical summary ]
      │
      ▼
[ SQLite: studies, results, audit_trail, notifications ]
      │
      ▼
[ WebSocket emit "study_updated" ] → frontend
      │
      ▼
[ Next.js ScreeningTab → DicomViewer + XaiVisualization + TsnePlot ]
```

---

## 5. Key Design Decisions

| Decision | Reason |
|----------|--------|
| **Two model generations with auto-switch** | Forward-compatible: U-Net file presence flips to correct preprocessing. Backward-compatible: old model still works |
| **U-Net → classifier cascade** | Without segmentation, classifier latches onto shoulders, mediastinum, or DICOM burned-in text. Explicit "verifying the shoulders issue is fixed" in training notebooks |
| **5 heatmap methods** | Different clinical questions: Grad-CAM (where model looks), Grad-CAM++ (better for multiple objects), attention (edges), coverage (extent), attribution (localized contribution) |
| **Post-concat `relu` as Grad-CAM target** | `conv5_block16_2_conv` is only 32 channels, pre-norm. The 1024-channel post-concat relu feeds the dense layer — that's what the model actually uses |
| **Grad-CAM gradient sign for normal scans** | Use `-logit` so heatmap highlights "evidence of normalcy" not absence of TB |
| **Pixel-wise alpha blending** | Uniform `addWeighted` made JET colormap flood background dark blue. Now opacity scales with activation |
| **Anatomical zone mapping in ROIs** | Right/Left + Upper/Middle/Lower (cutoffs at 0.40 and 0.68 of image height) |
| **Quadrant analysis with disease overlap** | Upper-zone concentration → TB, Aspergillosis, Silicosis; lower → pneumonia, COVID, edema; mixed → miliary TB, sarcoidosis |
| **Calibrated confidence formula** | Linear remap: `(prob - threshold) / (1 - threshold)` for TB; `(threshold - prob) / threshold` for normal. Output clamped to [0.5, 1.0] |
| **Reliability from quality score** | High ≥85, Medium ≥60, Low <60 |
| **Uncertainty from calibrated distance from 0.5** | High <0.15, Medium <0.35, Low ≥0.35 |
| **Threshold calibrated to 0.62** | Tuned against deployment data, not the 0.95 from training (Gen 2 student is well-calibrated) |
| **NIRT 3× sample weight** | Project tuned for Indian cohort; NIRT is most representative domain |
| **GroupShuffleSplit by patient** | Without grouping, same patient lands in train and test → data leak → inflated metrics |
| **CSRF + Bearer header support** | Same-origin (cookie) and cross-origin (Bearer) flows; Vercel frontend ↔ HF backend |
| **No regulatory claims in README** | Explicitly disclaims FDA/CE/CDSCO/HIPAA. Academic prototype |
| **`api_v1` blueprint with API keys** | Third-party integration path separate from session-auth UI flow |
| **WebSocket real-time updates** | `study_updated` event drives notifications panel and dashboard refresh without polling |

---

## 6. Open Questions / Worth Checking

1. **`/api/v1/predict` returns 501** — third-party headless prediction endpoint is stubbed, not implemented. The cookie-authed `/predict` is the only working path.
2. **Speedup label in deployment_metrics.txt** is honest: "1.95x SLOWER than teacher (DenseNet dense-connectivity overhead)". This is good documentation but if it's referenced anywhere as a "speedup win" it should be corrected.
3. **Teacher ResNet-50 model file** is not in `backend/` — only metrics. If you want to re-run distillation or compare, the teacher weights may need to be regenerated or pulled from a different location.
4. **`patients.db` is 121 MB** — large, may be slowing tests. Consider whether test fixtures should use an in-memory or temp DB.
5. **Two untracked test files** in `frontend/src/app/__tests__/`: `usePrediction.dedup.test.tsx`, `usePrediction.error.test.tsx` — these are the new tests for the production-hardened `usePrediction` hook (concurrency dedup, error handling).
6. **Frontend untracked assets**: `assets/`, `docs/`, `india_dataset/`, `metrics_and_logs/`, `scripts/`, `scratch/`, `segmented_images/` — directory exists, content not inventoried.
7. **DICOM input gives full PACS metadata**; standard image inputs get an auto-generated `PX-XXXXX` patient_id — UX-wise, the Standard image path looks "less clinical" than DICOM. Could be unified.
8. **`/audit/logs` requires admin role** — role-based access control is real but uses in-session `session["role"]`, not JWT or a proper RBAC system.
9. **CORS regex `^https?://.*$`** is extremely permissive for LAN showcase testing — fine for dev, risky if accidentally enabled in production. The `ALLOWED_ORIGIN` env var exists to override.
10. **`api_keys` table** in SQLite is referenced in `api_v1.py` but no `POST /api/v1/keys` endpoint to create them — bootstrapping mechanism unclear.

---

## 7. Tech Stack (full)

### Training
- TensorFlow/Keras (U-Net)
- PyTorch (DenseNet-121, ResNet-50, KD)
- scikit-learn (GroupShuffleSplit, metrics)
- OpenCV, pydicom

### Serving
- Flask + flask-socketio (async_mode="threading")
- Keras 3 with PyTorch backend (`os.environ["KERAS_BACKEND"] = "torch"`)
- PyTorch (DenseNet inference path uses torch tensors natively)
- SQLite (audit + patients + studies + notifications)
- pydicom, Pillow, OpenCV

### Frontend
- Next.js 14, React 18, TypeScript
- Tailwind CSS (dark PACS theme)
- lucide-react (icons)
- next-themes, next/dynamic
- Jest + React Testing Library
- Service-layer architecture (7 services)

### Infra
- Docker Compose (Flask + Next.js + Redis + Celery)
- PyInstaller `app.spec` and `dist/` folder (Windows binary packaging)
- Hugging Face Spaces deployment
- Vercel frontend
- Git LFS (model weights)

### Reproducibility
- Seeds set (42) for TF, NumPy, PyTorch
- `model_metadata.json` tracks training/validation dataset versions + training date
- `nirikshon-v3.log` (2.3 MB) — production run log
