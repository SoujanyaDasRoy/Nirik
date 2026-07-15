# Nirikshon — Project Handoff Document

**Purpose:** This document lets a brand-new Claude Code conversation continue this project with zero lost context. It was written at the point where the previous conversation approached its context limit. Read this document fully before doing anything else, then verify facts against the actual current state of `nirikNetMain.py` rather than trusting memory alone — code changes fast, this document is a snapshot.

**Companion document:** [`docs/nirikshon_architecture_spec.html`](nirikshon_architecture_spec.html) (v1.2) is the polished, published architecture specification — the contract backend/frontend/LLM work must follow. This handoff document is broader: it also covers debugging history, pending tasks, and conversation context the architecture spec deliberately excludes.

---

# 1. Project Overview

## Goal

Nirikshon is an AI-assisted pulmonary tuberculosis screening system built on chest X-rays. It is explicitly **not** a diagnostic device — it is a clinical decision-support tool. The clinician always makes the final call; the system's job is to flag suspicious cases, localize findings, and explain its reasoning. This framing (from `CLAUDE.md`) governs terminology throughout: "AI Screening Result," "Suspicious for Pulmonary Tuberculosis," never "Confirmed" or "Diagnosed."

## Current Architecture (high level)

A **teacher-student knowledge distillation** setup:

- **Teacher:** ResNet-50 (ImageNet-pretrained, ~23.6M params). Training-time only — never deployed, never exported, never sees a real uploaded image.
- **Student:** DenseNet-121 (ImageNet-pretrained, ~7.04M params, exact: 7,043,650). **This is the only model deployed for inference**, exported to **ONNX**.
- **Segmentation:** A separate Attention U-Net (256×256 grayscale in, binary lung mask out) crops the lungs out of every X-ray before classification — this runs both at training time and at inference time, for every image.
- **Explainability:** Grad-CAM, Grad-CAM++, LayerCAM, EigenCAM, computed against the student's DenseNet-121 backbone.
- **LLM:** Gemini 2.5 Flash via a Google AI Studio API key — **decided, not yet implemented**.
- **Deployment:** Backend + model on **Hugging Face Spaces** (Docker SDK). Frontend on **Vercel**. Both **decided, not yet reconciled** against the current model architecture.

## Model Design Detail

### Teacher model (ResNet-50)
```
Input (224×224×3, canonical 0–255 RGB)
  → Lambda(resnet50.preprocess_input)   [Caffe-style BGR mean-subtract, internal to the model]
  → ResNet-50 backbone (frozen → conv4_block*/conv5_block* unfrozen for finetune stage)
  → GlobalAveragePooling2D → BatchNormalization → Dropout(0.2)
  → Dense(2, linear logits, name="teacher_logits")
  → Activation("softmax", name="teacher_output")
```
Two training stages: `train_teacher_head` (frozen backbone) → `train_teacher_finetune` (conv4/conv5 unfrozen, BatchNorm layers within those blocks kept frozen).

### Student model (DenseNet-121)
```
Input (224×224×3, canonical 0–255 RGB)
  → Lambda(densenet.preprocess_input)   [Torch-style ImageNet mean/std, internal to the model]
  → DenseNet-121 backbone (frozen head stage → conv5_block* unfrozen for finetune stage)
  → GlobalAveragePooling2D → BatchNormalization → Dropout(0.3)
  → Dense(2, linear logits, name="student_logits")
  → Activation("softmax", name="student_output")
```
Two training stages, mirroring the teacher: `train_student_head` (frozen) → `train_student_finetune` (conv5_block* unfrozen only, BatchNorm within it kept frozen).

**Both models take the exact same canonical 0–255 RGB tensor as input** — normalization is internal via each model's own `Lambda` layer, not applied externally. This was a deliberate design choice so a single shared `tf.data` pipeline (and a single `preprocess_xray()` function) can feed both models without needing to know which one will consume it.

### Knowledge Distillation

Implemented in `DistillationModel`. Both networks end in softmax (for standalone use — evaluation, Grad-CAM), but distillation math operates on **pre-softmax logits**, tapped via a `_logits_submodel()` helper that builds a view into each model's graph at the `*_logits` layer (no new weights — same `tf.Variable` objects).

```
soft_teacher = softmax(teacher_logits / T)
soft_student = softmax(student_logits / T)
distill_loss = KLD(soft_teacher, soft_student) * T²
hard_loss    = CategoricalFocalCrossentropy(y, softmax(student_logits))   [T=1]
total_loss   = alpha * hard_loss + (1 - alpha) * distill_loss

T = 3.0, alpha = 0.5
```

**Why this exact formulation:** an earlier version of this code applied the teacher/student's *already-softmaxed* output through a second softmax (softmax-of-softmax) inside the distillation math — mathematically wrong, since it compresses bounded [0,1] values into a near-uniform distribution before the second softmax, flattening gradients and corrupting both loss terms. This was the very first bug found and fixed this session (see Known Bugs #1).

### Attention Mechanism

**Exists only inside the segmentation U-Net's decoder**, as soft attention gates (`_attention_gate()`) at each of the 4 decoder levels: 1×1 convs on the encoder skip-connection and the up-sampled decoder signal, added, ReLU'd, projected to 1 channel, sigmoid'd, multiplied elementwise against the skip connection.

**Neither ResNet-50 nor DenseNet-121 contains any attention mechanism** — no SE blocks, no CBAM, no self-attention. This is worth restating because it's easy to assume otherwise from the project's high-level pipeline description (which lists "Attention Module" as if it were a separate stage — it is not; it's folded into segmentation).

### Segmentation Pipeline

Attention U-Net, 4-level encoder/decoder, `base_filters=32` doubling per level (32→64→128→256, bottleneck 512). Input 256×256×1 grayscale, output 256×256×1 sigmoid mask. At inference: binarize at >0.5 → keep largest 2 connected components → morphological open+close (5×5 kernel) → compute bounding box with **8.5% padding** → crop.

Trained on a **separate** dataset (nikhilpandey360 chest-xray-masks-and-labels: 800 images / 704 masks) — has no dependency on the TB classification pool, and (deliberately) trains first in `main()`, before the classification dataframe is even built, so the trained U-Net is available to power the mask-coverage quality filter (see Data Pipeline below).

### Explainability Pipeline

Four CAM methods, all operating on the student's DenseNet-121 backbone (target conv layer auto-detected by type — `isinstance(l, keras.Model)` — not by a hardcoded layer name, since that broke once a Lambda preprocessing layer was added before the backbone — see Known Bugs #2):

| Method | Mechanism | Needs class index? |
|---|---|---|
| Grad-CAM | Gradient-weighted class activation mapping | Yes |
| Grad-CAM++ | Refined per-pixel weighting (squared/cubed gradients) | Yes |
| LayerCAM | Elementwise ReLU'd gradient weighting | Yes |
| EigenCAM | Gradient-free — top eigenvector of conv feature covariance | No |

**Important unresolved gap:** these require a live TensorFlow `GradientTape` over the Keras model. **ONNX Runtime cannot do this.** If only the ONNX student is loaded in production, explainability as currently implemented cannot run against it. Not resolved — see Pending Tasks.

**Also flagged, not a bug but a documentation/implementation mismatch:** the project's broader documentation (`CLAUDE.md`) describes a "Consensus CAM" (average of the four normalized CAM outputs) as canonical. **No such computation exists in `nirikNetMain.py`.** Only the four individual variants are generated.

### LLM Integration

**Decided this session:** Gemini 2.5 Flash, called via a Google AI Studio API key. **Nothing has been implemented** — no prompt templates, no client code, no guardrails exist anywhere in this codebase. Design intent (from the architecture spec, Section 9):
- Sits after classification + explainability, before final display — narrates already-computed results, does not produce them.
- Should receive only structured outputs (prediction, confidence, threshold) — **not** the raw image, **not** model weights, **not** training data.
- Must never be allowed to produce diagnostic-sounding language ("confirmed," "diagnosed").
- API key must be server-side only (env var / secrets manager), never bundled into the frontend, never logged.

---

# 2. Design Decisions (with reasoning)

| Decision | Why |
|---|---|
| **DenseNet-121 replaced the custom "NirikNet" CNN as student** | NirikNet trained from random initialization — real overfitting risk on a training pool of only a few thousand images. DenseNet-121 starts from ImageNet weights. Also narrows the teacher/student capacity gap from ~11x (23.6M→2.1M) to ~3.35x (23.6M→7.04M), addressing documented knowledge-distillation degradation under large capacity gaps (Mirzadeh et al. 2020). CheXNet (Rajpurkar et al. 2017) established DenseNet-121 specifically for chest-radiograph classification. Trade-off accepted: CPU inference is *slower* than the old NirikNet student (~750–870ms vs. ~260–490ms/image) because dense-block concatenation doesn't parallelize as cheaply as a plain conv chain — judged acceptable since this is a non-real-time, single-image screening upload, not a live feed. |
| **ResNet-50 stayed as teacher (not swapped to DenseNet-121-as-teacher)** | Isolates the experiment to one variable (student architecture) rather than changing both teacher and student at once — directly addresses a "too many simultaneous changes, no ablation path" critique raised earlier in the design process. The teacher's own training code was already validated and working. |
| **Custom CNN (NirikNet) abandoned rather than kept as a fallback** | User's explicit decision after reviewing capacity-gap analysis and historical (but methodologically contaminated) internal evidence about DenseNet121 vs a from-scratch custom CNN. Project's own novelty framing (`CLAUDE.md`) locates novelty in the explainable end-to-end workflow, not in inventing a bespoke CNN — so this doesn't compromise the thesis's actual claimed contribution. |
| **ONNX export for the deployed student** | Kaggle's training environment runs TF 2.20.0/Keras 3.13.2; the project's canonical `tb_env` spec calls for TF 2.15.1/Keras 2.15.0 — and that exact version **is no longer installable** on Kaggle's current Python (confirmed directly: `pip install tensorflow==2.15.1` fails, available versions start at 2.16.0rc0). Rather than fight that, ONNX decouples the deployment framework/version entirely from whatever trained the model — the serving container needs only `onnxruntime`, not `tensorflow`/`keras` or `torch`. Also typically faster on CPU than either native runtime, which matters for the confirmed CPU-only HF Spaces deployment target. |
| **Patient-wise splitting (not image-level)** | A plain image-level split can put different images of the same patient into different splits, letting the model partially memorize patient-specific anatomy instead of genuinely generalizing. Implemented via `GroupShuffleSplit` on a `patient_id` column (real metadata patient/subject ID where available, source-prefixed filename stem as fallback otherwise), with a hard runtime `RuntimeError` if any patient ID is ever found to overlap across splits. |
| **Montgomery + Shenzhen + TBX11K pooled into training (not held out as a permanent "external test")** | Originally these were held out entirely to claim genuine cross-hospital generalization. A model trained on only 3 sources and never shown a single Montgomery/Shenzhen image has no way to learn those hospitals' scanner characteristics exist (Zech et al. 2018). **Explicit trade-off, user-confirmed:** the resulting held-out test-set accuracy now measures in-distribution generalization across all pooled sources, not generalization to a genuinely unseen hospital — a deliberate, documented choice, not an oversight. (A source-stratified k-fold alternative was proposed and rejected in favor of the simpler flat pooled split, per user instruction — "I was told 70/15/15 is a good thing to keep, even if all datasets are being used as input.") |
| **2:1 Normal:TB pool-wide rebalancing** | Raw pooled ratio was ~3-6:1 in various configurations. Achieved by **downsampling the majority (Normal) class only**, proportionally across every source (not just the largest one), never by duplicating/oversampling the minority class (which would risk the same image appearing in multiple splits). Runs *after* dedup/quality/mask-coverage filtering, not before — cleaning first, then balancing the cleaned pool. |
| **Focal Loss (not just class-weighted CE)** | Class-weighting only reweights the loss — it can't manufacture more diverse minority-class examples. Focal Loss (Lin et al. 2017, gamma=2.0, alpha=0.25) down-weights easy/majority examples so gradient concentrates on hard/minority cases, complementary to (not a replacement for) having more real TB images. Applied to both teacher's own hard-label training and the student's hard-label term inside distillation. Label smoothing left unwired deliberately — stacking it with focal loss is an atypical, unvalidated combination (both soften the loss but for different reasons). |
| **`assume_unmatched_as_normal` flipped from `True` to `False`** | Previously, images with no metadata label match and no filename-suffix hint silently defaulted to Normal. Per explicit user data-cleaning instruction: "fewer images with correct labels beats more images with questionable ones" — an incorrect label (a real TB image silently defaulted to Normal) is worse than losing that image. Every dropped image is still logged for spot-checking. **Side effect:** Jaypee/India's metadata only catalogs confirmed-TB cases, so this source now contributes essentially TB-only (its unmatched, presumably-Normal images are dropped, not rescued). |
| **DA/DB's unmatched images default to TB=1 (opposite of the rule above)** | Intentional, not a leftover inconsistency: DA/DB are curated, TB-positive-only collections by construction, so an unmatched image there is far more likely to be an unlabeled TB case than a stray Normal image. Documented explicitly in-code so it isn't mistaken for a bug later. |
| **Student trains in two stages (frozen → partial unfreeze), mirroring the teacher** | The student previously fully fine-tuned its entire 7M-param backbone from step one via distillation, with no staged freeze at all — a real, unaddressed overfitting risk given the training-pool scale. Fixed by making `build_densenet_student` start frozen and adding `train_student_finetune` to unfreeze only `conv5_block*` (last dense block), BatchNorm within it kept frozen, mirroring the teacher's `conv4_block*`/`conv5_block*` pattern. |
| **Perceptual-hash dedup threshold tightened from 5 to 3, plus cross-label-conflict detection added** | A real Kaggle run showed a kept image matching two *differently-labeled* dropped images (one Normal, one TB) across three independent, unrelated hospital sources — a stronger signal of a hash false-positive than an ordinary same-class match, since chest X-rays are all anatomically similar at the coarse 8×8 grid `aHash` operates on (unlike the varied natural photos aHash was designed for). Tightened in both `deduplicate_pool` and `check_dataset_leakage`. **Not yet verified on a fresh Kaggle run** — see Pending Tasks. |
| **CSV export of the final train/val/test split** | The project's own documentation (`CLAUDE.md`'s canonical Notebook 1 contract) calls for `train.csv`/`validation.csv`/`test.csv` as reproducibility/audit artifacts — nothing in `nirikNetMain.py` produced these before. Added `export_split_csvs()`, called right after `split_dataframe()`. Nothing downstream reads these back; they exist purely so the exact split is inspectable without re-running the whole pipeline. |
| **TBX11K: `health`→Normal, `tb`→TB, `sick` and `uncertain` always excluded** | TBX11K's "Sick, non-TB" category is genuinely abnormal but not TB — forcing it into either binary label would inject wrong ground truth. Folder names matched by **exact** name (case-insensitive), not substring — a substring match on `"tb"` would incorrectly match the `TBX11K` wrapper folder itself (`"tbx11k"` contains `"tb"` as a literal substring) before ever reaching the real leaf folder. `health` is capped (`tbx11k_healthy_cap=1500`, secondary guard) since TBX11K's own raw Healthy:TB ratio is more imbalanced than the pool already is. |
| **TF/Keras version pin attempted but non-fatal on failure** | Originally treated as an error-worthy event; changed so a failed pin (which is expected and unavoidable on Kaggle's current Python) prints one calm status line and continues, rather than an alarming pip error block followed by a "restart the kernel" message that wouldn't have helped anyway. |
| **Deployment: Hugging Face Spaces (backend + model) / Vercel (frontend)** | User decision. Validates rather than introduces the CPU-only assumption already baked into `benchmark_inference_latency()`'s and the ONNX export cell's own docstrings. An `hf_space/` directory with a working `Dockerfile` (`python:3.11-slim`, `EXPOSE 7860` — the standard HF Spaces Docker-SDK port) already exists in the repo as scaffolding, not yet reconciled against this architecture. |

---

# 3. Current Progress

## Completed
- Fixed the softmax-of-softmax distillation bug (root cause of student generalizing worse than teacher despite better val numbers).
- Swapped student from custom "NirikNet" CNN to DenseNet-121; teacher stays ResNet-50.
- Fixed a real regression: `_get_grad_model`'s Grad-CAM reconstruction broke once internal `Lambda` preprocessing layers were added (fixed by splitting layer replay into pre-target/post-target lists based on index, not blanket "everything except target").
- Implemented patient-wise splitting with a hard leakage-proof runtime check.
- Restructured the dataset pool: Montgomery/Shenzhen/TBX11K now pooled into training (not held out), flat patient-wise 70/15/15 split.
- Implemented folder-based TBX11K collection (multiple rounds of fixing — see Known Bugs).
- Switched hard-label loss to Focal Loss (teacher and student both).
- Implemented ONNX export + validation for the deployed student, with a properly-calibrated tolerance (see Known Bugs #8).
- Flipped `assume_unmatched_as_normal` default, documented DA/DB's intentional opposite behavior.
- Implemented pool-wide 2:1 Normal:TB rebalancing (downsampling only, proportional across sources).
- Implemented whole-pool deduplication (MD5 exact pass + perceptual-hash near-duplicate pass) before splitting, distinct from the existing post-split cross-split leakage check.
- Implemented an image-quality filter (blur/corrupted/near-blank) and a lung-mask-coverage outlier filter (with label-stratified drop-rate logging and a loud warning if TB images are disproportionately dropped — a hypothesis about the segmenter struggling with visibly diseased lungs, not yet confirmed either way on real data).
- Implemented staged (frozen → partial unfreeze) training for the student, mirroring the teacher.
- Fixed a real BatchNorm/inference-determinism bug (`training=True` hardcoded into the student's graph, meaning predictions would have varied batch-to-batch even at inference — verified fixed via a repeated-inference-call test).
- Found and fixed multiple real runtime crashes via actual Kaggle execution (see Known Bugs — `layers[1]` index bug, mixed-precision augmentation dtype bug, `compiled_metrics` crash, `metrics` property crash, ONNX tolerance miscalibration, TBX11K folder matching x2).
- Tightened perceptual-hash dedup threshold and added cross-label-conflict detection (not yet verified on a fresh run).
- Added CSV export of the final split.
- Produced and published the full architecture specification (`docs/nirikshon_architecture_spec.html`, v1.2, also published as a Claude Artifact), covering the complete pipeline, model architecture, training/inference architecture, explainability, and a derived backend/frontend/LLM contract.
- Recorded the Gemini 2.5 Flash / Google AI Studio and Hugging Face Spaces / Vercel deployment decisions into that spec.

## Partially completed
- **Kaggle smoke testing (`CFG.smoke_test = True`):** multiple rounds run, each surfacing and fixing a real bug. The **most recent code change (hamming_threshold 5→3 + cross-label-conflict detection) has not yet been verified with a fresh Kaggle run.** This is the immediate next action.

## Remaining (not started)
- The full, non-smoke-test training run (`CFG.smoke_test = False`) — hours long, not yet attempted even once.
- Backend reconciliation (`backend/` and/or `hf_space/`) against the current architecture.
- Frontend build (Vercel).
- LLM integration build (Gemini 2.5 Flash).
- Resolution of the explainability-vs-ONNX architectural gap.
- Decision on whether to ONNX-export the U-Net too.
- Decision on the fate of "Consensus CAM" (implement vs. formally drop from documentation).
- Updating `CLAUDE.md` itself, which still describes an older canonical architecture (EfficientNetV2-M teacher / NirikNet student) that the user explicitly told this session's Claude to disregard in favor of `nirikNetMain.py`'s actual state — `CLAUDE.md` was never actually updated to match, so it's currently self-contradictory relative to the real, current architecture. Flagging this prominently since a future session reading `CLAUDE.md` at face value could get confused.

---

# 4. Backend Status

## Current state
Two existing directories, **neither reconciled against the current architecture**:
- `backend/` — has both TensorFlow/Keras and PyTorch installed and imported; `backend/core/inference.py` is a **hybrid, mid-migration file**: a legacy Keras path (`build_niriknet`, loading `results/student_best.weights.h5` — the OLD custom-CNN student, not DenseNet-121) coexists with a newer PyTorch-based path (`torch.jit.load` for quantized models). A comment inside it already anticipated a DenseNet + "torch mode" preprocessing convention, suggesting someone had already started planning the move this session's architecture actually landed on. `model_metadata.json` inside `backend/` is **stale** — it references `"DenseNet121-Student-v4.0.0"` and a hardcoded threshold of `0.62`, leftover from an even older, since-deprecated DenseNet121 experiment that predates the "NirikNet" custom-CNN era, which has now itself been superseded by *this* session's DenseNet-121 decision. The naming history here is genuinely confusing — worth a clean-slate regeneration rather than trying to patch it.
- `hf_space/` — mirrors `backend/`'s structure, has a working `Dockerfile` (`python:3.11-slim`, `EXPOSE 7860` — correct HF Spaces Docker-SDK convention) and a `requirements.txt` that actually pins `tensorflow==2.15.1`/`keras==2.15.0` (matching the project's canonical `tb_env` spec more tightly than `backend/`'s own requirements file does).

## Required updates
- Remove `build_niriknet` and all references to the old custom-CNN student.
- Load the DenseNet-121 student via ONNX Runtime (recommended) or decide to keep a PyTorch serving path and do a validated weight-transfer (higher risk, more work, not recommended given ONNX is already implemented and validated on the training side).
- Regenerate `model_metadata.json` from a real `metrics.json` once the full training run completes — do not hand-patch the stale one.
- Load the operating threshold from `metrics.json`'s `student_youden_threshold` field, not a hardcoded value.
- Decide the U-Net's serving format (currently native `.keras`, no ONNX export path exists for it).
- Resolve the explainability-vs-ONNX gap before wiring up any CAM endpoint.
- Drop the mixed PyTorch+TensorFlow dependency footprint if moving to ONNX Runtime-only serving (smaller image, faster cold starts on HF Spaces' free tier).

## API expectations (derived contract — see architecture spec Section 7 for full detail)
```
POST /predict   — full pipeline: preprocess → segment → classify → threshold → (optionally) explain
POST /explain    — heatmaps only, for an already-scored image
GET  /health     — liveness + confirms both required models are loaded
```
Response shape:
```json
{
  "prediction": "Tuberculosis" | "Normal",
  "confidence": 0.62,
  "threshold_used": 0.079,
  "screening_note": "AI Screening Result — Suspicious for Pulmonary Tuberculosis",
  "explainability": {
    "gradcam": "<image or url>",
    "gradcam_plus_plus": "<image or url>",
    "layercam": "<image or url>",
    "eigencam": "<image or url>"
  }
}
```
Authentication, patient records, report generation, DICOM-specific handling: **not defined** by the model architecture itself — product decisions, not derivable from the training code.

---

# 5. Frontend Status

## Current state
Not touched this session. No frontend code was reviewed or built. Deployment target decided: **Vercel**, calling the Hugging Face Spaces backend as a remote API — the frontend never loads a model itself.

## Required updates / remaining work (derived contract — architecture spec Section 8)
- Upload workflow (single image, formats matching what `read_image_grayscale()` accepts — standard raster + DICOM if `pydicom` present).
- Prediction display using mandated clinical language ("AI Screening Result — Suspicious for Pulmonary Tuberculosis" / "Normal" — never "Confirmed"/"Diagnosed").
- Confidence display **alongside** the operating threshold it was compared against (observed Youden thresholds were far from 0.5 — e.g. 0.079–0.124 — so showing raw probability without its threshold would read as arbitrary to a clinician).
- Heatmap display for whichever CAM variant(s) the backend returns.
- No "attention visualization" as a classification artifact — doesn't exist (see Attention Mechanism above). If the segmentation mask itself is shown, label it as segmentation/localization, not attention.
- No "Consensus CAM" until it exists in the model pipeline.
- Chat — blocked entirely on Section 9 / LLM integration, itself unbuilt.
- Loading states are functionally necessary, not polish — realistic end-to-end latency (student inference + U-Net segmentation + up to 4 gradient-tape CAM passes) is a multi-second operation, not sub-second.
- Error states — not defined beyond what the backend's own failure surface implies (corrupted image → 4xx; no lung region detected → current code silently falls back to the uncropped image rather than erroring, worth surfacing as a quality caveat in the UI).

---

# 6. Model Status

## Current implementation
See Section 1 (Project Overview) for the full architecture. In one line: ResNet-50 teacher (training-only) distills into a DenseNet-121 student (the only deployed model) via temperature-scaled KL-divergence + Focal Cross-Entropy, both trained in two stages (frozen head → partial unfreeze), on a pooled, deduplicated, quality-filtered, mask-coverage-filtered, 2:1-rebalanced, patient-wise-split dataset from 6 sources, exported to ONNX with a validated round-trip.

## Training status
**Only smoke tests have been run** (`CFG.smoke_test=True`: `max_per_class=75`, 2 epochs per stage, mask-coverage filter skipped since the smoke-test U-Net is deliberately undertrained). Each smoke test run took roughly 11–13 minutes on Kaggle (Tesla P100). Multiple rounds surfaced and fixed real bugs (see Known Bugs). **The full, real training run (`CFG.smoke_test=False`) has never been executed.** The most recent code change (dedup threshold tightening) has not yet been smoke-tested.

## Expected outputs of a real run
Written to `/kaggle/working` (or wherever `CFG.output_dir` points):
- `metrics.json` — full evaluation metrics for teacher and student, on both val and test splits, plus Youden threshold and frozen-threshold metrics.
- `run_config.json` — full config + environment snapshot (TF/Keras/Python versions, GPU info, seed).
- `train.csv` / `val.csv` / `test.csv` — final split membership (new this session).
- `dedup_report.json`, `image_quality_dropped.json`, `mask_coverage_dropped.json`, `leakage_check.json` — data-cleaning audit trails.
- `assumed_normal_<source>.json` per source with unmatched-label fallbacks.
- `*_classification_report.txt`, `*_threshold_analysis.json`, `*_frozen_threshold_metrics.json` — per-model, per-split.
- `figures/` — dozens of PNGs: training curves, overfit-gap plots, LR schedules, confusion matrices, ROC/PR curves, dataset composition, segmentation samples, augmentation preview, architecture diagrams, Grad-CAM/Grad-CAM++/LayerCAM/EigenCAM visualizations, teacher-vs-student comparison, pipeline overview.
- `attention_unet.keras`, `teacher_head_best.keras`, `teacher_finetune_best.keras`, `student_head_best.weights.h5`, `student_best.weights.h5` — training-time checkpoints.
- `densenet121_student.onnx` — the actual deployment artifact.
- `onnx_export_validation.json` — the ONNX round-trip validation result.

---

# 7. Important Constraints

## Libraries
TensorFlow/Keras (training — Kaggle currently runs TF 2.20.0 / Keras 3.13.2, **not** the canonical 2.15.1/2.15.0 pin, because that exact wheel is no longer installable on Kaggle's current Python; the pin attempt fails gracefully and training proceeds on whatever's installed), `tf2onnx` + `onnxruntime` (export + validation), scikit-learn (`GroupShuffleSplit`, evaluation metrics), OpenCV (CLAHE, perceptual hashing, connected components, morphology), `pydicom` (optional DICOM support), NumPy, Pandas.

## Dependencies / environment
Local dev uses `tb_env` (canonical spec: Python 3.10.11, TF 2.15.1, Keras 2.15.0 per `CLAUDE.md` — though the actual local `tb_env` observed during this session's testing was already on Keras 3.12.3, i.e. some drift existed locally too, separate from the Kaggle drift).

## Assumptions
- **HF Spaces free tier is CPU-only** — baked into `benchmark_inference_latency()`'s docstring and the ONNX export cell's docstring, *before* the deployment platform was formally confirmed this session (confirming it just validated an assumption already in the code).
- **ONNX decouples serving from the training framework/version** — the reason the export step exists at all.
- **Preprocessing must be byte-for-byte identical between training and inference** — stated repeatedly, a hard requirement, not a suggestion. Any backend implementation must call the exact same CLAHE (clipLimit=2.0, tileGridSize=8×8) → 8.5%-padded lung-mask crop → resize-to-224×224 → canonical-0–255-RGB pipeline.

## Performance considerations
DenseNet-121 CPU inference measured at **~750–870ms/image** on Kaggle's CPU-forced benchmark (architecture-only timing, independent of how well-trained the weights are) — slower than the old NirikNet student (~260–490ms) despite having far fewer parameters than the teacher, because dense-block concatenation doesn't parallelize as cheaply as a plain convolutional chain. Accepted trade-off for a non-real-time, single-image screening workflow.

---

# 8. Folder Responsibilities

(Derived — reflects what this architecture needs, not necessarily the repository's exact current layout. See architecture spec Section 11 for the canonical version.)

```
model/
  segmentation/          # Attention U-Net: build, train, weights
  teacher/                 # ResNet-50: build, train (head + finetune)
  student/                  # DenseNet-121: build, train (head + finetune)
  distillation/              # DistillationModel, KD loss math
  export/                     # ONNX export + validation
  explainability/              # Grad-CAM / Grad-CAM++ / LayerCAM / EigenCAM

backend/                     # deploys to Hugging Face Spaces (Docker SDK) — DECIDED
  Dockerfile                   # already scaffolded in hf_space/: python:3.11-slim, EXPOSE 7860
  api/                          # /predict, /explain, /health
  inference/                     # preprocessing (must mirror preprocess_xray exactly), model loading
  postprocessing/                 # threshold application, response shaping

frontend/                     # deploys to Vercel — DECIDED, calls the HF Spaces backend as a remote API
  upload/
  results/                      # prediction, confidence, threshold display
  explainability/                 # heatmap rendering
  chat/                            # NOT DEFINED — depends on LLM integration

llm/
  client/                        # Gemini 2.5 Flash call (Google AI Studio API key, server-side only)
  prompts/                         # NOT DEFINED — no templates exist yet
  guardrails/                        # terminology enforcement, input allow-list

config/
  training_config.json              # mirrors Config dataclass in nirikNetMain.py
  metrics.json                       # frozen threshold, evaluation results — read by backend, never regenerated there

exports/
  train.csv / val.csv / test.csv    # final split membership — audit/reproducibility only, nothing reads these back

weights/                     # ships to Hugging Face Spaces alongside backend/ — model is not a separate deployment target
  attention_unet.keras
  teacher_finetune_best.keras        # training artifact only — never shipped
  densenet121_student.onnx            # the only model deployed
```

---

# 9. API Contract

See Section 4 (Backend Status) above and architecture spec Section 7 for the complete derived contract, including error-handling behavior derived from the code's own failure surface. Key invariant: **`threshold_used` must always be read from `metrics.json`'s `student_youden_threshold`, never hardcoded to 0.5, never recomputed server-side.**

---

# 10. Pending Tasks (ordered by priority)

1. **Re-run the Kaggle smoke test** (`CFG.smoke_test=True`) to confirm the hamming_threshold=3 + cross-label-conflict-detection change (the most recent code change) doesn't regress anything. This is the very next action.
2. **Run the full, non-smoke-test training job** (`CFG.smoke_test=False`) on Kaggle. Hours long. Not yet attempted even once.
3. **Review the real `metrics.json`/reports** before doing anything else with the output — sanity-check: did the 2:1 balance actually hold post-cleaning? Is `dedup_report.json`'s label-conflict list empty or does it need manual review? Does `mask_coverage_dropped.json` show a disproportionate TB drop rate (the hypothesized "segmenter struggles with visibly diseased lungs" risk)?
4. **Reconcile `backend/`/`hf_space/`** against the current architecture — remove `build_niriknet` references, decide ONNX-Runtime-only vs. keep-PyTorch serving, regenerate `model_metadata.json`, implement the derived API contract, load the real Youden threshold.
5. **Resolve the explainability-vs-ONNX gap** — decide between (a) keeping a native Keras student loaded alongside the ONNX one purely for CAM generation, or (b) an ONNX-compatible CAM implementation.
6. **Decide the U-Net's serving format** — stay native `.keras`, or add an ONNX export path to match the student.
7. **Build the frontend** (Vercel) per the derived contract.
8. **Build the LLM integration** (Gemini 2.5 Flash / Google AI Studio) — prompt design, guardrails, structured-input-only constraint, follow-up-question handling (currently entirely undefined).
9. **Decide the fate of "Consensus CAM"** — implement it to match `CLAUDE.md`'s documented canonical spec, or formally amend that documentation to drop it.
10. **Update `CLAUDE.md`** itself to reflect the actual current architecture (DenseNet-121 student / ResNet-50 teacher), since it currently still describes an older canonical design (EfficientNetV2-M / NirikNet) that was explicitly superseded this session but never actually corrected in the document itself.

---

# 11. Known Bugs

All of these were **found and fixed** this session (several via actually running the pipeline on Kaggle, not just code review) — listed here as historical record since understanding *why* something is the way it is matters for future changes.

1. **Softmax-of-softmax in `DistillationModel`** — teacher/student final layers already output softmax probabilities, but the distillation train/test step divided by temperature and softmaxed again, corrupting both the KD loss and (since the same tensor fed both) the hard-label loss too. Root cause of the student generalizing worse than the teacher despite better validation numbers. **Fixed** by splitting each model's final Dense into a linear-logits layer + separate softmax Activation layer, and computing distillation math on the pre-softmax logits via a `_logits_submodel` view.
2. **`_get_grad_model`'s layer replay broke once Lambda preprocessing layers were added** — it used to assume every non-target, non-Input layer belonged *after* the target conv layer in the graph; once a `Lambda(preprocess_input)` layer was added *before* the backbone, that assumption became false, and the code tried to feed a conv feature map backward through the preprocessing Lambda. **Fixed** by splitting layer replay into pre-target/post-target lists based on the target's actual index in `model.layers`.
3. **`train_teacher_finetune`'s hardcoded `head_teacher.layers[1]`** — assumed index 1 was the ResNet-50 backbone; once a `teacher_preprocess` Lambda layer was added at that position, index 1 became the Lambda instead, and the unfreeze logic would have crashed the first time it actually ran (Lambda layers have no `.layers` to iterate). **Fixed** by finding the backbone via `isinstance(l, keras.Model)` instead of a fixed index, with a `RuntimeError` guard if more than one match is found.
4. **`build_densenet_student` hardcoded `training=True`** into the graph — meant BatchNorm would use live-batch statistics even at single-image deployment inference (degenerate for batch size 1) and would be inconsistent between the Keras model and its ONNX export. **Fixed** by calling the backbone as `base(x)` with no explicit `training=` argument, letting Keras follow `.trainable` state and ambient context correctly. Verified: repeated inference calls on identical input now give identical output (they didn't before).
5. **Mixed-precision (`mixed_float16`) dtype propagation in `_augment`** — the Keras augmentation layers (`RandomRotation` etc.) compute in float16 under the global mixed-precision policy, but `tf.random.normal` defaults to float32, causing a `TypeError` on the addition; fixing *that* alone then surfaced a second issue — `tf.image.random_contrast`'s `AdjustContrastv2` kernel has **no CPU implementation for float16 at all**, unrelated to dtype-matching. **Fixed** by casting back to float32 immediately after the augmentation layers, before any further op touches the tensor. Reproduced and verified locally under the same `mixed_float16` policy.
6. **`self.compiled_metrics.update_state(...)` crash under Keras 3** — leftover, already-vestigial code (superseded by an explicit `accuracy_tracker`) that turned out to be an outright crash under this Keras version for a subclassed `Model` compiled with `metrics=[]`: `"MeanMetricWrapper.update_state() missing 1 required positional argument: 'y_pred'"`. **Fixed** by deleting the call entirely from both `train_step` and `test_step`.
7. **`DistillationModel.metrics` property crash** — surfaced immediately after fixing #6: `super().metrics + [...]` pulled in an internally auto-created loss metric that's never actually updated (loss is tracked manually), and calling `.result()` on that unbuilt metric raised `"Cannot get result() since the metric has not yet been built."` **Fixed** by returning only `[self.loss_tracker, self.accuracy_tracker]`, not mixed with `super().metrics`.
8. **ONNX export tolerance (`atol=1e-4`) too strict** — a genuinely correct export under the `mixed_float16` policy showed a real Keras-vs-ONNX-Runtime divergence of ~1.42–1.43e-4, expected cross-backend noise (BatchNorm epsilon handling, conv algorithm choice differ between TF and ONNX Runtime for mixed-precision graphs), not a correctness bug. The earlier, stricter tolerance had been implicitly calibrated for pure float32 (an earlier, separate verification got 1.19e-7). **Fixed** by loosening to `atol=2e-3` — still tight (0.2% on a bounded softmax probability), with headroom over the observed diff.
9. **TBX11K folder discovery, round 1** — only checked immediate children of the dataset root (`os.listdir`, non-recursive); the real category folders are nested deeper (under `.../TBX11K/imgs/`). **Fixed** by searching recursively (`os.walk`), picking the shallowest match.
10. **TBX11K folder discovery, round 2** — the assumed category names (`Healthy`/`ActiveTB`/`LatentTB`/`Sick`/`Uncertain`, from the user's own illustrative description) turned out to be wrong; the real mirror uses short lowercase names: `health`, `sick`, `tb` (confirmed by direct inspection, user-provided). Additionally, naive substring matching on `"tb"` would have matched the `TBX11K` **wrapper folder itself** (`"tbx11k"` contains `"tb"` as a literal substring) in preference to the real leaf folder, since it's shallower. **Fixed** by switching to exact (not substring) folder-name matching, verified against a synthetic reproduction of the exact collision scenario before trusting it.
11. **Mask-coverage filter dropping 100% of both classes under smoke_test** — the smoke-test U-Net is deliberately undertrained (~2 epochs, ~34 images, val Dice ~0.14) for speed, not accuracy; its predicted masks are near-random, so the coverage-fraction filter (designed for a properly-trained segmenter) rejected everything. **Fixed** by skipping this filter entirely when `cfg.smoke_test` is `True`.
12. **`NameError: train_lung_segmentation not defined`** — not a code bug; a notebook-execution-order issue (a cell was skipped, or the kernel was restarted and cells re-run out of order/incompletely). Resolved by using Kaggle's "Restart & Run All" instead of manually running cells piecemeal.
13. **Perceptual-hash false-positive risk** — a kept image (Shenzhen, `_1`/TB-suffixed) matched two *differently-labeled* dropped images (one Tawsifur "Normal," one TBX11K "tb") across three unrelated hospital sources, under `hamming_threshold=5`. **Fixed** (not yet re-verified on Kaggle) by tightening to `hamming_threshold=3` in both `deduplicate_pool` and `check_dataset_leakage`, plus adding explicit cross-label-conflict detection/reporting.
14. **TF/Keras version pin (2.15.1/2.15.0) fails on Kaggle** — that exact wheel is no longer built for Kaggle's current Python version (confirmed: available versions start at 2.16.0rc0). **Handled gracefully**, not truly "fixed" (can't be, the wheel doesn't exist) — the pin attempt now prints one calm status line and continues on whatever TF/Keras is already installed, rather than an alarming pip error block.

## Known, not yet fixed (low priority, non-blocking)
- **Degenerate cosine-decay LR schedule under `smoke_test`** — when `len(train_df) < batch_size` (true under smoke_test's tiny pool), `total_steps` and `warmup_steps` both collapse to 1, causing a division-by-zero in the schedule's cosine-decay term → NaN learning rate → NaN `val_loss` at epoch 2 of every stage. **Diagnosed, not fixed**: harmless in practice (each stage's `restore_best_weights=True` reloads the real, non-NaN epoch-1 weights), and structurally impossible at real-data scale (the real pool has thousands of images, `len(train_df) // batch_size` is never zero). A defensive guard would be cheap but wasn't prioritized since it can't affect the real training run.

---

# 12. Technical Debt

- No genuine held-out-hospital validation anymore (flat pooled 70/15/15 chosen over source-stratified k-fold, per explicit user decision) — external-generalization claims are weaker than the project's earlier design intended. Documented, not hidden, but worth remembering when writing up results.
- `CLAUDE.md` itself is stale relative to the actual current architecture (see Pending Tasks #10).
- `backend/` has a confusing, mixed PyTorch+TensorFlow dependency footprint and hybrid legacy code paths — needs real cleanup regardless of which serving framework is ultimately chosen.
- No "Consensus CAM" despite `CLAUDE.md` documenting it as canonical.
- No anatomical zone-mapping (e.g., "right upper lobe") despite `CLAUDE.md` describing it — do not assume this is available as an LLM input until it actually exists.
- Explainability (Grad-CAM family) is architecturally incompatible with ONNX-only serving as currently implemented — unresolved design gap, not just an implementation gap.
- The segmentation U-Net's own train/val split (for training the U-Net itself, not the classifier) is still image-level, not patient-wise — lower priority, flagged early, never fixed.
- The degenerate-LR-schedule-under-smoke-test issue (see Known Bugs, "not yet fixed").

---

# 13. Important Files

| File | What it does |
|---|---|
| `CNN Model Training/nirikNetMain.py` | **The entire training pipeline. This is the single source of truth for the model.** Notebook-cell-structured (`# %%` markers) — segmentation training, dataset discovery/collection/cleaning, teacher training, student training via distillation, evaluation, explainability, ONNX export, all in one file, run top-to-bottom via `main()`. |
| `CLAUDE.md` | Authoritative engineering-principles document for the whole repo — but **currently stale** regarding the canonical model architecture (still describes EfficientNetV2-M/NirikNet in places; the user explicitly directed this session's Claude to treat `nirikNetMain.py`'s actual DenseNet-121/ResNet-50 design as canonical instead). Needs updating (Pending Task #10). |
| `docs/nirikshon_architecture_spec.html` | **The published, polished architecture specification** (v1.2) — the contract backend/frontend/LLM work must follow. Every claim tagged Confirmed / Derived / Decided / Not Defined by epistemic status. Also live as a Claude Artifact. |
| `docs/PROJECT_HANDOFF.md` | This document. |
| `CNN Model Training/Nirikshon_Project_Handoff.md` | An **older**, mostly-superseded handoff document from earlier in the project's life (pre-dates this session). Historical context only — most of its items (metadata-driven labeling, CLAHE, crop padding percentage) have already been addressed in the current `nirikNetMain.py`. Do not treat as current. |
| `backend/`, `hf_space/` | Existing backend scaffolding, **out of sync** with the current architecture — see Section 4 (Backend Status) for exactly what's stale and what needs to change. `hf_space/`'s `Dockerfile`/`requirements.txt` are the more useful starting point (correct HF Spaces port convention, tighter version pins) than `backend/`'s. |
| `results/` (or wherever old Kaggle runs wrote output) | Contains output from **older** runs (pre-dating this session's architecture, some even pre-dating the "NirikNet" era) — will be entirely superseded once the real (non-smoke) training run under the current architecture completes. Don't trust any numbers in there as representative of the current model. |

---

# 14. Next Steps (exactly what to do after opening a new conversation)

1. Read this document fully.
2. Read `CLAUDE.md`, keeping in mind the staleness caveat above (Section 13).
3. Open `CNN Model Training/nirikNetMain.py` directly and re-verify current ground truth for anything load-bearing before acting on it — this handoff document is a snapshot, not a live view.
4. Ask the user directly: **"Has the smoke test with the hamming_threshold=3 change been re-run on Kaggle yet? What was the output?"** — this is the actual next action left pending from the previous session.
5. If the smoke test hasn't been re-run, or if it surfaces a new error: continue the same debugging loop this session used — Restart & Run All on Kaggle with `CFG.smoke_test=True`, get the traceback, fix in `nirikNetMain.py`, verify locally with `tb_env` where possible (`py_compile` at minimum, an isolated reproduction test where the bug is non-trivial), repeat.
6. Once a smoke test passes cleanly end-to-end: get the user's go-ahead to switch to `CFG.smoke_test=False` and run the real training job.
7. After the real run completes: review `metrics.json` and the data-cleaning audit JSONs (Section 6) before doing anything else with the results.
8. Only then proceed to backend reconciliation → frontend build → LLM integration, in that priority order (Section 10).

---

# 15. Conversation Memory (narrative summary of the previous session)

The previous session began with a request to analyze why `nirikNetMain.py`'s trained model performed the way it did — validation accuracy near 97–98%, but external-test accuracy collapsing to 55–74%, with the student (then a custom CNN called "NirikNet") performing *worse* than its own ResNet-50 teacher despite better validation numbers. Root-cause investigation (following a systematic-debugging process) found the cause: a softmax-of-softmax bug in the knowledge-distillation math, where the code treated already-softmaxed model outputs as raw logits, double-softmaxing them and flattening gradients for both the distillation loss and the (shared-tensor) hard-label loss. This was fixed, along with several related issues discovered in the same pass: no real Youden-threshold selection (just a fixed 0.3/0.5/0.7 sweep), 5% lung-crop padding instead of the handoff-specified 8.5%, weak/partially-inert augmentation (a brightness delta calibrated for the wrong pixel-value range), and an overly conservative teacher backbone unfreeze.

The user then asked for a broader review against `Nirikshon_Project_Handoff.md` (an older, separate handoff document) and against general medical-imaging data-quality best practices they'd researched independently. This led to a long back-and-forth about the "NirikNet" custom CNN's viability versus alternatives. After research into knowledge-distillation capacity-gap literature (Mirzadeh et al.) and domain-generalization concerns (Zech et al., on cross-site model generalization), and after checking (and finding methodologically contaminated) historical internal evidence about a much older DenseNet121 experiment in the `backend/` folder, the user decided: **DenseNet-121 replaces the custom CNN as student; ResNet-50 stays as teacher.** This was deliberately framed as isolating one variable (student architecture) rather than changing both teacher and student, directly responding to an earlier critique that too many simultaneous changes had been proposed with no ablation path.

A large implementation session followed, covering (in rough order): TF/Keras version pinning (later found to be unachievable on Kaggle's current Python, handled gracefully instead of as an error), the DenseNet-121 swap itself (with a real regression in Grad-CAM layer-replay logic caught and fixed via a targeted smoke test before it could ship), patient-wise splitting (a leakage bug that had existed even *before* this session, caught during a broader risk audit), restructuring the dataset pool to include Montgomery/Shenzhen/TBX11K in training rather than holding them out (a deliberate, discussed trade-off against genuine cross-hospital generalization claims), TBX11K integration (which required several rounds of correction as real folder-structure assumptions turned out wrong — first assumed CSV-based labeling, then assumed a 5-way Healthy/ActiveTB/LatentTB/Sick/Uncertain folder split that turned out to actually be a simpler 3-way `health`/`sick`/`tb` split, discovered only by asking the user to run `os.listdir` directly rather than guessing further), Focal Loss for class imbalance, and ONNX export with a validated round-trip.

A dedicated "mitigate bugs, issues, and risks" pass then found and fixed several more issues purely through code review and targeted local testing (without needing a live Kaggle run): the double-softmax-adjacent BatchNorm inference-determinism bug in the new DenseNet-121 student, the same `layers[1]`-index fragility in the teacher's own unfreeze logic (which would have crashed on its first real run), a staged-freeze training scheme added for the student to address a real overfitting risk from full-from-step-one fine-tuning, an exact-hash pre-pass added to the perceptual-hash deduplication, label-stratified logging added to the mask-coverage filter (to surface, not silently risk, the possibility that it disproportionately drops severe/visible TB cases), an explicit warning when no real patient-ID metadata column is found for a source, and a clarifying comment distinguishing DA/DB's intentional TB-default behavior from a bug.

The user then asked for actual Kaggle execution, which surfaced a genuine sequence of real runtime bugs across several rounds — a mixed-precision augmentation dtype crash (with a second, deeper float16-CPU-kernel-gap issue found only by locally reproducing the exact failure before trusting the fix), a `compiled_metrics`/Keras-3 crash in the distillation training step (leftover dead code that turned out to be an outright crash, not just inert), a follow-on `metrics` property crash from the same root cause, and an ONNX export-validation tolerance that was miscalibrated for a mixed-precision-trained model (a genuinely correct export was failing its own safety check; the tolerance was loosened with clear justification, not just widened arbitrarily). Each fix was verified — via `py_compile` at minimum, and via isolated local reproduction tests for anything non-trivial — before being declared done, and each round ended with a clear "which cells were touched" answer so the user could re-sync only what changed rather than re-uploading the whole file blindly.

Once a full smoke-test pipeline run completed cleanly end-to-end (through ONNX export), a careful re-read of that "clean" log surfaced one more thing worth fixing proactively rather than waiting for it to cause a problem: a kept image in the deduplication step had matched two differently-labeled dropped images across three unrelated hospital datasets — a stronger signal of a perceptual-hash false positive than an ordinary match, given how visually similar all chest X-rays are at the coarse resolution the hash operates on. The hash threshold was tightened and cross-label-conflict detection was added — **this is the most recent code change, and it has not yet been re-verified with a fresh Kaggle run.**

Separately, the user requested a full architecture specification document — explicitly as a source-of-truth contract for backend, frontend, and LLM work going forward, built strictly from the actual training code with no invented features, and with any genuinely undecided or unimplemented aspect explicitly marked as such rather than guessed. This was produced as a designed HTML document (not a plain markdown file), published as a Claude Artifact and also saved into the repository's `docs/` folder, and has since been updated twice more in the same session: once to record the LLM provider decision (Gemini 2.5 Flash via a Google AI Studio API key — a correction was made here, since the user had attributed this recommendation to Claude, when in fact Claude had never specified a model version; `CLAUDE.md`'s own documentation only mentioned "Gemini" generically) and the addition of the CSV split-export feature, and once more to record the deployment-platform decision (backend + model on Hugging Face Spaces, frontend on Vercel) — which was notable for *confirming*, not introducing, an assumption already baked into the training code's own comments about CPU-only inference being the deployment target.

This handoff document itself was requested at the point the previous conversation approached its context limit, specifically so a new conversation could continue without needing to re-derive any of the above.

---

# 16. Architecture Summary (condensed final specification)

*(Full detail lives in `docs/nirikshon_architecture_spec.html` — this is a condensed, self-contained version so this handoff document doesn't require cross-referencing another file to be useful on its own.)*

| Fact | Status |
|---|---|
| Deployed classifier | **DenseNet-121 student only**, exported to ONNX (opset 13) |
| Teacher (ResNet-50) | Training-time only — never loaded in serving |
| Input contract | 224×224×3 canonical 0–255 RGB; model-internal normalization via each model's own `Lambda` layer |
| Preprocessing | One shared function (`preprocess_xray`), identical for training and inference — CLAHE (2.0/8×8), 8.5%-padded lung crop, resize to 224×224 |
| Segmentation | Attention U-Net, 256×256 grayscale, 4-level encoder/decoder, attention gates in the decoder only |
| Attention mechanism | Exists **only** inside the segmentation U-Net's decoder gates — not a classifier component |
| Knowledge distillation | `T=3.0`, `alpha=0.5`, KL-divergence + Focal Cross-Entropy on pre-softmax logits |
| Decision threshold | Frozen, Youden's-J-derived from validation — never 0.5, never recomputed at inference |
| Explainability | Grad-CAM, Grad-CAM++, LayerCAM, EigenCAM on the student — **unresolved gap**: requires native Keras gradient access, incompatible with ONNX-only serving as currently implemented |
| Consensus CAM | Documented in `CLAUDE.md`, **not implemented** in code |
| Dataset | 6 pooled sources (Tawsifur, Jaypee/India, Montgomery, Shenzhen, DA/DB, TBX11K), patient-wise 70/15/15 split, deduplicated, quality-filtered, mask-coverage-filtered, rebalanced to 2:1 Normal:TB |
| Reproducibility artifacts | `train.csv`/`val.csv`/`test.csv` (final split membership), `metrics.json`, `run_config.json`, various `*_report.json`/`*_dropped.json` audit files |
| LLM layer | Provider **decided** (Gemini 2.5 Flash, Google AI Studio API key) — integration itself not implemented |
| Deployment platforms | Backend + model: **Hugging Face Spaces** (Docker SDK, scaffolded in `hf_space/`). Frontend: **Vercel**, calling HF Spaces as a remote API. Decided; not yet reconciled against this architecture |
| Backend / Frontend code | Not implemented against this architecture — existing `backend/`/`hf_space/` directories are stale and out of sync |
| Training run status | Only smoke tests completed; full training run never executed |

**If any future change contradicts this table, the change is wrong or this document is stale — not both silently coexisting.**
