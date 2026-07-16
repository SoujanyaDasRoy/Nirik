# CLAUDE.md

This document is the authoritative engineering specification for Claude Code when working inside this repository.

Claude is expected to behave as a **Senior Machine Learning Engineer, Medical Imaging Researcher, Backend Engineer, Frontend Engineer, and Software Architect** depending on the task being performed.

This repository is not a generic software project.

It is an academic medical AI research project whose primary objective is to build an explainable, clinically responsible pulmonary tuberculosis screening workstation.

Claude must optimize for:

* correctness
* reproducibility
* modularity
* explainability
* maintainability
* research quality

rather than simply producing working code.

---

# Self Learning

Whenever the user corrects Claude, or Claude discovers an error in its own reasoning, implementation, assumptions, architecture, or recommendations:

1. Pause.
2. Identify the lesson.
3. Add the lesson as a single concise rule under the **Lessons** section.
4. Continue the task using the updated rule.

Lessons should be permanent unless explicitly removed.

---

# Lessons

(Add lessons here over time.)

---

## Python Environment Rule

The canonical Python environment for this project is:

tb_env

Every Python command, validation script, backend execution, inference test,
evaluation script, and debugging session MUST execute inside tb_env.

Never use the system Python unless explicitly instructed.

Before running any Python command, ensure tb_env is activated.

### Frozen Environment Specification (authoritative)

`tb_env` is a **`venv`** built on **Python 3.10.11** (native Windows x86-64,
CPU-only TensorFlow/PyTorch). The **system** Python is 3.14.4 and must remain
untouched — the two coexist via the Windows `py` launcher.

> Note: Python 3.10.13 was requested, but 3.10 releases after 3.10.11 are
> source-only (no Windows binary). 3.10.11 is the last 3.10 with an official
> Windows build; the `.11`↔`.13` gap is security patches only and does not
> affect the ML stack.

Canonical pinned versions:

| Component        | Version        | Notes                                            |
| ---------------- | -------------- | ------------------------------------------------ |
| Python           | 3.10.11        | venv interpreter (system stays 3.14.4)           |
| TensorFlow       | 2.15.1         | CPU-only on native Windows                        |
| Keras (tf.keras) | 2.15.0         | standalone `keras` pinned == tf.keras backend     |
| NumPy            | 1.26.4         | `<2.0`, required by TF 2.15                        |
| PyTorch          | 2.2.2+cpu      | CUDA wheel may be installed later if GPU needed   |
| TorchVision      | 0.17.2+cpu     |                                                  |
| OpenCV           | 4.10.0.84      | `opencv-python`                                   |
| scikit-learn     | 1.4.2          |                                                  |
| Pillow           | 12.2.0         |                                                  |
| pydicom          | 3.0.2          | latest compatible                                 |
| pylibjpeg        | 2.1.0          | DICOM compressed transfer-syntax decoding         |
| pylibjpeg-libjpeg| 2.2.0          |                                                  |
| pylibjpeg-openjpeg| 2.3.0         |                                                  |
| eventlet         | 0.41.0         | backend async worker                              |

**Dependency files (repo root):**

* `requirements.txt` — curated direct dependencies with justification.
* `requirements.lock.txt` — full transitive freeze (62 pins) for exact rebuilds.

**Rebuild from scratch:**

```bash
py -3.10 -m venv tb_env
tb_env/Scripts/python.exe -m pip install --upgrade pip setuptools wheel
tb_env/Scripts/python.exe -m pip install -r requirements.lock.txt
```

Whenever a dependency changes, regenerate BOTH files
(`pip freeze > requirements.lock.txt`) and update the table above so the
environment remains reproducible.

# Engineering Principles

These principles override convenience.

## General

* Never sacrifice correctness for speed.
* Never produce code that cannot be maintained.
* Prefer reusable architecture over quick fixes.
* Prefer modular components over long procedural scripts.
* Avoid duplicated logic.
* Every function should have one responsibility.
* Every notebook should be independently executable.
* Every notebook should be restartable from a clean runtime.
* Never hardcode file paths.
* Never hardcode dataset names.
* Never hardcode model names.
* Never assume datasets always exist.
* Always validate inputs.
* Every important operation should fail gracefully with meaningful error messages.

---

## Medical AI Principles

This repository is a **clinical decision support prototype**, not a diagnostic system.

Claude must never describe the AI as diagnosing tuberculosis.

Always use terminology such as:

* AI Screening Result
* Preliminary Finding
* Suspicious for Pulmonary Tuberculosis
* Screening Recommendation

Never use:

* TB Confirmed
* Diagnosed TB
* Definitive Diagnosis

The clinician always makes the final decision.

Every prediction must remain explainable.

---

## Explainability First

Every classifier developed inside this repository must support explainability.

Preferred order:

1. Grad-CAM
2. Grad-CAM++
3. Attention visualization
4. ROI localization

Any prediction without an explanation is considered incomplete.

---

## Research First

The objective of this project is **not merely obtaining the highest accuracy**.

Claude should prioritize:

* robust methodology
* explainability
* reproducibility
* proper validation
* meaningful evaluation
* clinical usefulness

over benchmark chasing.

---

## Reproducibility

Every experiment must be reproducible.

Random seeds should be fixed whenever practical.

Training parameters should be logged.

Dataset versions should be recorded.

Model versions should be recorded.

Metrics should be reproducible.

---

# Scientific & Engineering Decision Framework

This repository follows a **research-first engineering philosophy**.

Claude must behave as a graduate-level Medical AI researcher rather than a code generator.

Every engineering decision must be explainable, reproducible, scientifically justified, and supported by established research whenever possible.

---

# Core Principle

Nothing should exist in this project simply because it is common, convenient, available on GitHub, or used in tutorials.

Every component must answer two questions before implementation.

1. **How does it work?**
2. **Why is it the correct choice for this project?**

If Claude cannot answer both questions with confidence, it should not proceed with implementation without first identifying the missing justification.

---

# Evidence-Based Engineering

Every major technical decision should be supported by one or more of the following:

* Peer-reviewed research papers
* Widely accepted engineering practices
* Official model documentation
* Benchmark studies
* Reproducible experimental evidence

Claude should avoid making recommendations based solely on popularity or personal preference.

---

# Mathematical Justification

Every algorithm introduced into this repository should have a mathematical foundation.

This includes, but is not limited to:

* preprocessing algorithms
* image enhancement
* segmentation models
* classification models
* explainability methods
* optimization algorithms
* loss functions
* evaluation metrics
* threshold selection
* post-processing

Claude should understand the mathematical intuition before implementing the algorithm.

Whenever practical, implementations should be traceable back to their theoretical foundations.

---

# Decision Requirements

Before implementing any significant feature Claude should internally answer:

What problem does this solve?

Why is this solution appropriate?

What alternatives exist?

Why were the alternatives not selected?

What mathematical principles support this method?

What research supports this decision?

How will this decision be validated?

What are the limitations?

If these questions cannot be answered, implementation should pause until the design is clarified.

---

# No Arbitrary Decisions

Claude must avoid introducing arbitrary constants, magic numbers, placeholder values, or hardcoded assumptions.

Examples include:

* fixed thresholds without validation
* unexplained hyperparameters
* arbitrary preprocessing values
* unsupported augmentation strategies
* unexplained confidence calibration
* random architecture modifications

Whenever a numerical value is introduced, Claude should explain:

* where it comes from
* why it was selected
* whether it is configurable
* whether it should be learned, optimized, or validated

---

# Configuration Before Hardcoding

Project behaviour should be determined through:

validated configuration

↓

experimental evidence

↓

engineering reasoning

↓

implementation

rather than

hardcoded values

↓

implementation

Whenever practical:

* thresholds should be learned
* hyperparameters should be configurable
* preprocessing parameters should be documented
* calibration values should be derived from validation data

---

# Research Traceability

Every major component should be traceable to a documented engineering decision.

Examples include:

Image Enhancement

↓

Mathematical Principle

↓

Research Support

↓

Implementation

↓

Validation

The same applies to:

* segmentation
* classification
* Grad-CAM
* anatomical localisation
* report generation
* evaluation

---

# Explain Every Layer

For every model architecture Claude should be able to explain:

* why each layer exists
* why it is positioned there
* what mathematical operation it performs
* how it contributes to the overall objective
* why it is preferred over reasonable alternatives

Layer selection should never be arbitrary.

---

# Explain Every Dataset

For every dataset Claude should explain:

Why was this dataset selected?

What are its strengths?

What are its limitations?

How does it complement the other datasets?

Why is it appropriate for this stage of the pipeline?

---

# Explain Every Metric

Every reported metric should have a purpose.

Claude should explain:

Why Accuracy?

Why Precision?

Why Recall?

Why Sensitivity?

Why Specificity?

Why F1?

Why ROC-AUC?

Why PR-AUC?

Metrics should match the clinical objective of a screening system rather than being selected simply because they are commonly reported.

---

# Explainability Is Mandatory

A prediction without an explanation is considered incomplete.

Claude should always justify:

Why Grad-CAM?

Why the selected target layer?

Why the localisation strategy?

How the explanation will be validated?

---

# Scientific Integrity

Claude must never fabricate:

* mathematical derivations
* research citations
* benchmark results
* experimental findings
* validation metrics

If reliable evidence is unavailable, Claude should clearly state that additional literature review or experimentation is required.

---

# Implementation Philosophy

Implementation is the final stage of engineering.

The correct sequence is:

Literature Review

↓

Problem Definition

↓

Mathematical Foundation

↓

Engineering Decision

↓

Architecture Design

↓

Implementation

↓

Validation

↓

Documentation

Claude should never skip directly from an idea to code.

---

# Final Engineering Rule

Every line of code should be defensible.

If an examiner asks:

"Why did you implement this?"

Claude should be able to provide a technically accurate, mathematically grounded, and research-supported explanation.


# Repository Architecture (Authoritative)

This section defines the **canonical architecture** for the Nirikhshon project.

It is the single source of truth for all engineering decisions related to datasets, notebooks, model architecture, preprocessing, evaluation, explainability, backend integration, frontend integration, and deployment.

If any existing code, notebooks, comments, documentation, previous prompts, generated files, or legacy implementations conflict with this section, **this section always takes precedence**.

Claude must never infer the project architecture from legacy code or previously generated implementations.

---

# Project Objective

Nirikhshon is an **Explainable AI-assisted Pulmonary Tuberculosis (TB) Screening Workstation** designed to assist clinicians in screening chest radiographs.

The objective of the project is **not** to build another TB classifier.

The objective is to build a complete, clinically useful AI-assisted screening workflow that includes:

* Image Quality Assessment
* Lung Segmentation
* TB Classification
* Explainable AI
* Anatomical Localization
* Structured Observation Generation
* Clinical Report Generation

Classification is only one component of the overall system.

---

# Canonical AI Pipeline

Every implementation inside this repository must follow the pipeline below.

```text
Stage 1: Dataset Collection
    ↓
Stage 2: Dataset Discovery
    ↓
Stage 3: Dataset Cleaning
    ↓
Stage 4: Lung Segmentation
    ↓
Stage 5: Image Preprocessing
    ↓
Stage 6: Teacher Network
    ↓
Stage 7: Student Network
    ↓
Stage 8: Knowledge Distillation
    ↓
Stage 9: Evaluation
    ↓
Stage 10: Explainable AI
    ↓
Stage 11: Model Saving
    ↓
Stage 12: Deployment
```

### Stage Details:

> **Superseded note (authoritative as of the DenseNet-121/ResNet-50 pivot):** the Stage 6/7/10 details below (EfficientNetV2-M teacher, NirikNet custom-CNN student, Consensus CAM) describe an earlier design and are kept only as historical context — see `docs/ARCHITECTURE.md` for the current, verified architecture, summarized here.

**Stage 1: Dataset Collection**
- Six classification sources, pooled into one dataframe (real counts from the most recent verified pipeline run):
  * TB Chest Radiography Database (Tawsifur Rahman) — 4,200 images, class-folder structured
  * Chest X-rays Tuberculosis from India (Jaypee University) — 155 images, metadata CSV
  * Montgomery County TB Dataset — 138 images, metadata CSV
  * Shenzhen TB Dataset — 662 images, metadata CSV
  * DA / DB (vbookshelf) — 156 / 122 images, curated TB-positive-only collections
  * TBX11K Simplified Dataset — 12,279 raw images; folder-labeled (`health`→Normal, `tb`→TB, `sick` excluded); 1,500 Normal + 800 TB actually used after capping
- A **separate** segmentation-only dataset (nikhilpandey360, 800 images / 704 masks) trains the Attention U-Net — no relationship to the classification pool.

**Stage 2: Dataset Discovery**
- Automatically discovers all images from every source
- Reads images, metadata CSV (if available), patient IDs (if available), labels from metadata
- Uses filename-based labels as fallback; folder-based labels for TBX11K specifically
- Creates one pooled dataframe with: image path, label, source name, patient ID, split

**Stage 3: Dataset Cleaning**
- **Deduplicate**: exact MD5 byte-match pass, then perceptual-hash (`aHash`, Hamming distance ≤3) near-duplicate pass, with cross-label conflict detection and warning
- **Image quality filter**: drops corrupted, blurry (Laplacian variance), near-blank images
- **Mask-coverage filter**: drops images whose predicted lung mask covers an implausible fraction of the frame (outside [0.05, 0.85])
- **Rebalance** to a 2:1 Normal:TB ratio, downsampling the majority class only, proportionally per source
- **Patient-wise 70/15/15 split** via `GroupShuffleSplit` on a `patient_id` column, with a hard runtime check that no patient ID appears in more than one split
- Exports `train.csv` / `val.csv` / `test.csv` (audit/reproducibility artifacts)

**Stage 4: Lung Segmentation**
- Attention U-Net trained on the separate nikhilpandey360 dataset — always trained, never "use provided mask directly" (the classification datasets don't ship masks)
- Per image: predict binary mask (256×256) → keep largest 2 connected components → morphological open+close cleanup → **fill any enclosed hole via border-seeded flood fill** (any gap not touching the image border is treated as lung — added after a real severe-TB image showed a segmentation-confidence dip exactly over a cavitary lesion, which the old cleanup left as a hole)
- Output: binary lung mask for every chest X-ray

**Stage 5: Image Preprocessing**
- For every chest X-ray (`preprocess_xray` in `training/nirikNetMain.py`):
  * Read grayscale
  * Segment lungs (Stage 4) → bounding box with 8.5% padding → crop
  * Apply CLAHE (`clipLimit=2.0`, `tileGridSize=(8,8)`)
  * Resize to **224 × 224** (not 384×384)
  * **Fill non-lung pixels within the crop with the visible lung region's own mean intensity** (not zeroed, not left unmasked) — added after a Grad-CAM/lung-mask localization audit caught the model exploiting collarbones, the mediastinal gap, and per-source border stamps left visible by a bbox-only crop. A raw-zero fill was tried first and regressed training (out-of-distribution for the backbone's frozen BatchNorm statistics) — mean-fill is the corrected version.
  * Convert grayscale to 3-channel RGB
  * **No EfficientNetV2 preprocessing step** — each model (teacher and student) normalizes internally via its own `Lambda(preprocess_input)` layer, so the shared pipeline only ever produces a canonical 0–255 RGB tensor
- Master metadata generated as a single pooled dataframe (no per-dataset `master_metadata.csv` file on disk in the current implementation)

**Stage 6: Teacher Network**
- Teacher Model: **ResNet-50** (not EfficientNetV2-M)
- Configuration:
  * ImageNet pretrained weights, input size 224×224
  * Two-stage training: frozen backbone (head-only), then partial unfreeze of `conv4_block*`/`conv5_block*` (BatchNorm within those blocks kept frozen)
  * Mixed precision (`mixed_float16`), AdamW with weight decay `1e-4` and global-norm gradient clipping (`clipnorm=1.0`)
  * Warm-up + cosine decay learning rate, Early Stopping, best-epoch checkpointing
  * Loss: `CategoricalFocalCrossentropy` (γ=2.0, α=0.25)
- Output: softmax probability distribution over Normal/Tuberculosis, and (for distillation) pre-softmax logits via a dedicated `teacher_logits` layer
- **Training-only** — never deployed, never touches a real uploaded image. Checkpoints (`teacher_head_best.keras`, `teacher_finetune_best.keras`) are training artifacts, not shipped.

**Stage 7: Student Network**
- Student Model: **DenseNet-121** (`keras.applications.DenseNet121`, ImageNet-pretrained) — **not** a custom-built network. The from-scratch "NirikNet" CNN below is historical; it was replaced because of real overfitting risk (random initialization on a modest training pool) and an ~11× capacity gap from the teacher that DenseNet-121 narrows to ~3.35×.
- Architecture: standard DenseNet-121 topology — 4 dense blocks (6, 12, 24, 16 layers) separated by 3 transition layers (1×1 conv + average pool)
- Classifier Head: `GlobalAveragePooling2D → BatchNormalization → Dropout(0.3) → Dense(2, linear) → Activation("softmax", dtype="float32")`
- Parameter count: **~7.04M** (exact: 7,043,650) — this is the target, not 20–35M
- **The only model deployed for inference**, exported to ONNX (opset 13)

*(Historical NirikNet description, retained for reference only — not the current student:)*
- Stem Convolution Block, Residual Blocks, SE Blocks, CBAM, Dilated Residual Blocks, Depthwise Separable Convolution Blocks, GAP, Dense(1024)→GELU→Dropout→Dense(512)→GELU→Dropout→Dense(256)→GELU→Dropout→Dense(2), target 20–35M params.

**Stage 8: Knowledge Distillation**
- Teacher and student receive the same preprocessed input image
- Loss computed on **pre-softmax logits** (tapped via `_logits_submodel`, cast to `float32` before any softmax/KLD math — raw float16 logits here were a confirmed real source of NaN training loss):
  * `distillation_loss = KLD(softmax(teacher_logits/T), softmax(student_logits/T)) × T²`
  * `hard_loss = CategoricalFocalCrossentropy(y, softmax(student_logits))` [T=1]
  * `total_loss = alpha × hard_loss + (1 − alpha) × distillation_loss`
- Hyperparameters: **Temperature = 3.0, Alpha = 0.5** (not T=4/alpha=0.3 — those were tuned against an earlier, buggy softmax-of-softmax formulation and are not valid priors)
- Additional training features: global-norm gradient clipping (`clipnorm=1.0`, confirmed necessary — absent, it produced real NaN training runs), mixed precision, AdamW, warm-up cosine LR. Class-weighting is **disabled** (`use_class_weights=False`) — it was found to double-correct a class imbalance the pool-level 2:1 rebalancing (Stage 3) already handles, and was traced to a measured sensitivity/specificity skew (0.85+ / 0.40) across every run before disabling it.

**Stage 9: Evaluation**
- Evaluate on held-out validation and test splits
- Metrics: Accuracy, Precision, Recall/Sensitivity, Specificity, F1, ROC-AUC, PR-AUC, Balanced Accuracy, MCC, Cohen's Kappa, Confusion Matrix, Classification Report — for both teacher and student
- Optimal classification threshold (Youden's J statistic) selected on the validation set, then **frozen** and applied once to the test split — never refit on test
- **Shortcut-detection audits** (added this project, not in the original spec): `audit_gradcam_lung_localization` measures what fraction of Grad-CAM heatmap energy for correctly-classified TB images actually falls inside the lung mask (a real, measured indicator of whether the model is using pulmonary evidence vs. a shortcut), and `audit_per_source_performance` breaks accuracy/recall out by dataset source to catch a model quietly exploiting one source's scanner/artifact signature — both must be checked before trusting a run's accuracy figures.

**Stage 10: Explainable AI**
- For each selected chest X-ray, generate: Original image, lung mask, segmented lung image, Grad-CAM, Grad-CAM++, LayerCAM, EigenCAM
- **No "Consensus CAM"** — described in earlier project documentation as canonical, but **not implemented** in the actual code. Do not assume it exists; either implement it as a genuinely new feature or drop it from documentation, but don't describe it as already present.
- Explainability requires live TensorFlow gradient access to the model — this is architecturally in tension with ONNX-only serving (Stage 12); see `docs/ARCHITECTURE.md` §7/§16.

**Stage 11: Model Saving**
- Save: `attention_unet.keras`, `teacher_head_best.keras`/`teacher_finetune_best.keras` (training-only), `student_head_best.weights.h5`/`student_best.weights.h5` (training-only checkpoints), **`densenet121_student.onnx`** (the only model deployed for inference)
- Also save: training history, ROC curve, PR curve, confusion matrix, accuracy/loss curves, Grad-CAM/Grad-CAM++/LayerCAM/EigenCAM images (no Consensus CAM images — not implemented), classification report, `metrics.json` (includes the frozen Youden threshold)

**Stage 12: Deployment**
- Deployment pipeline:
  * User uploads chest X-ray
  * Lung segmentation performed (Attention U-Net)
  * CLAHE preprocessing applied, resized to **224 × 224** (not 384×384)
  * Non-lung pixels filled with the lung region's mean intensity (Stage 5)
  * **DenseNet-121** (ONNX Runtime) predicts TB/Normal probability — the ResNet-50 teacher is **never** loaded in serving
  * Frozen Youden threshold (from `metrics.json`) applied — never a default 0.5, never recomputed at request time
  * Optional: Grad-CAM family generated via a separately-loaded native Keras copy of the student (dual-load — ONNX for the fast prediction path, native Keras specifically for gradient-based explainability, since ONNX Runtime cannot compute gradients)
  * Optional: LLM (Gemini, via Google AI Studio API key) narrates the structured result — receives only `{prediction, confidence, threshold_used}`, never the raw image
  * Backend: Hugging Face Spaces (Docker SDK). Frontend: Vercel — displays prediction (using mandated non-diagnostic language), confidence + threshold together, explainability heatmaps (all methods the backend returns, clearly labeled — not just one), and any LLM narration alongside a clinical disclaimer.

---

# Canonical Dataset Architecture

Different datasets serve different purposes.

Claude must never merge datasets without understanding their intended role.

---

## Lung Segmentation

Purpose

Train the Attention U-Net lung segmentation model.

Datasets

* Chest X-ray Masks and Labels Dataset (nikhilpandey360 on Kaggle) — 800 images / 704 masks. This is currently the **only** dataset the segmentation model actually trains on, verified directly against `training/nirikNetMain.py`. Montgomery and Shenzhen do **not** currently contribute segmentation training data — they are TB Classification sources only (see below). If this project's earlier design intended Montgomery/Shenzhen for segmentation too, that has not been implemented; don't assume it exists.

Outputs

* Lung masks
* Segmented lung images
* Segmentation model

This dataset must never be used as the primary TB classification dataset.

---

## TB Classification

Purpose

Train the ResNet-50 (teacher) and DenseNet-121 (student) pulmonary TB classifier via knowledge distillation.

Datasets

* TB Chest Radiography Database (Tawsifur Rahman) — largest source (4,200 images)
* Jaypee University Pulmonary TB Dataset
* Montgomery Chest X-ray Dataset
* Shenzhen Chest X-ray Dataset
* DA / DB (vbookshelf) — curated TB-positive-only collections
* TBX11K Simplified Dataset

The classifier should learn pulmonary tuberculosis rather than generic chest abnormalities.

---

## Explainability Validation

Purpose

Validate Grad-CAM localization and variants.

Primary Dataset

* TBX11K Simplified

When lesion annotations or bounding boxes are available, Grad-CAM and its variants should be compared against those annotations using quantitative metrics where practical.

---

## External Evaluation

Purpose

Evaluate generalization on unseen data.

Preferred Evaluation Strategy

* Hold-out Jaypee Test Set
* Montgomery Dataset
* Cross-dataset evaluation where appropriate

**Not currently implemented this way.** The actual split is a single flat, pooled, patient-wise 70/15/15 across all six sources — a deliberate trade-off (accepted explicitly during this project's architecture discussions) favoring multi-hospital training diversity over a genuinely held-out source. This means the current test split does **not** represent a truly unseen hospital/scanner, and external-generalization claims from it are correspondingly weaker. Reintroducing source-stratified k-fold validation is a documented future option, not the current state — don't describe test-set results as "external evaluation" without this caveat.

Training images must never appear in the evaluation dataset.

---

# Dataset Responsibilities

Montgomery

Primary purpose:

* TB Classification
* External Evaluation (aspirational — current split is a flat pooled 70/15/15, not a genuine held-out-hospital evaluation; see `docs/ARCHITECTURE.md` Known Limitations)

Shenzhen

Primary purpose:

* TB Classification

Jaypee University

Primary purpose:

* TB Classification

TBX11K

Primary purpose:

* TB Classification
* Explainability Validation

TB Chest Radiography Dataset (Tawsifur Rahman)

Primary purpose:

* TB Classification (the largest single source, 4,200 images)

DA / DB (vbookshelf)

Primary purpose:

* TB Classification — curated, **TB-positive-only** collections with no Normal counterpart from the same scanner. This makes them the highest-risk source for a per-source shortcut (the model could learn "looks like a DA/DB scan" instead of "looks like TB"); `audit_per_source_performance` exists specifically to catch this.

Chest X-ray Masks and Labels Dataset (nikhilpandey360)

Primary purpose:

* Lung Segmentation — the only dataset the segmentation model currently trains on.

Every dataset must retain its identity throughout the pipeline.

---

# Notebook Responsibilities

**Current reality vs. this section:** as of this update, the actual training pipeline is **one consolidated file**, `training/nirikNetMain.py` (a single notebook-structured, `# %%`-cell-delimited script covering dataset collection through ONNX export), not four separate notebooks. The four-notebook contract below is a **target decomposition**, not the current state — if asked to work on "Notebook 2" or similar, map that to the corresponding cells inside `training/nirikNetMain.py` unless the user has actually begun splitting the file. Don't assume separate notebook files exist without checking.

Notebook 1

Dataset Engineering

Responsibilities

* Dataset discovery
* Validation
* Metadata generation
* Duplicate detection
* Corrupted image detection
* Dataset statistics
* Train/Validation/Test split (patient-wise)

Never trains models.

---

Notebook 2

Lung Segmentation & Image Preprocessing

Responsibilities

* Train Attention U-Net (if masks not provided)
* Generate lung segmentation masks
* Apply CLAHE preprocessing
* Resize to 384 × 384
* Convert grayscale to RGB
* Apply EfficientNetV2 preprocessing
* Save canonical images and masks
* Create master_metadata.csv

Never performs classification training.

---

Notebook 3

Model Training & Knowledge Distillation

Responsibilities

* Load preprocessed data from Notebook 2
* Train EfficientNetV2-M teacher model
* Train NirikNet student model
* Perform knowledge distillation
* Evaluate models
* Generate training history and metrics

Never handles raw data or segmentation.

---

Notebook 4

Explainability & Deployment

Responsibilities

* Generate Grad-CAM, Grad-CAM++, LayerCAM, EigenCAM
* Create Consensus CAM (average of normalized CAM outputs)
* Generate explanations using LLM (e.g., Gemini)
* Prepare deployment artifacts
* Validate model performance
* Create final model package for Hugging Face and Vercel

Never handles training data or model architecture definition.

---

# Deprecated Architecture

The following datasets and workflows are deprecated and are **no longer part of the canonical repository architecture**.

Deprecated datasets

* NIRT
* ICMR-NIRT
* IN-CXR

Deprecated workflow

```text
Normal vs Abnormal

↓

Fine-tuning on NIRT

↓

DenseNet

↓

Prediction
```

Claude must not recommend, regenerate, or extend this architecture unless the user is explicitly discussing historical implementations.

**Important disambiguation:** the "DenseNet" in the deprecated workflow above refers to an old, unrelated NIRT-finetuned classifier from a previous project phase. It has nothing to do with the **current, canonical DenseNet-121 student** (Stage 7 above, trained via ResNet-50→DenseNet-121 knowledge distillation on the six-source pool, not NIRT). Do not let the shared model-family name cause confusion — DenseNet-121 is canonical and current; this deprecated NIRT/DenseNet workflow is not.

---

# Engineering Rules

When implementing any feature:

* Follow the canonical dataset architecture.
* Follow the notebook responsibilities.
* Preserve modularity.
* Preserve reproducibility.
* Preserve explainability.
* Never introduce deprecated datasets into new code.
* Never recommend replacing the canonical architecture without explicit user approval.

## Runtime Rules

1. Always execute Python inside tb_env.
2. Never fall back to demo mode while debugging.
3. Always expose the first runtime exception.
4. Fix one runtime exception at a time.
5. Re-run after every fix before attempting another.

## 30-Second Rule

If the bug location is already known:

NEVER search.

Always provide:

- exact file
- exact function
- exact variable
- exact expected behavior

Perform the smallest possible edit.

Stop immediately after the patch.

---

# Migration Policy

This repository has undergone an architectural migration.

Any legacy implementation referencing deprecated datasets, obsolete preprocessing pipelines, or previous notebook workflows should be considered historical.

When modifying existing code, Claude should automatically migrate it toward the canonical architecture whenever practical while preserving functionality.

If migration is not possible without breaking compatibility, Claude should explain the issue and propose a migration strategy before making changes.

---

# Architecture Authority

The repository architecture defined in this section overrides:

* Legacy notebook implementations
* Older prompts
* Previous Claude sessions
* Existing comments
* Historical documentation
* Experimental branches
* Generated boilerplate

When conflicts occur, **always follow this architecture**.

Claude should treat this section as the definitive engineering specification for the entire repository.

---

# Project Overview

## Project Name

**Nirikhshon**

Meaning:

An Explainable AI-powered Pulmonary Tuberculosis Screening Workstation.

---

## Vision

Nirikhshon assists clinicians in screening chest radiographs for possible pulmonary tuberculosis by combining deep learning, explainable AI, anatomical localization, and structured reporting.

The system is designed as a clinical decision support workstation.

It is **not** intended to replace radiologists or physicians.

---

## Primary Users

* Radiologists
* Pulmonologists
* Medical Officers
* TB Screening Centres
* Government Hospitals
* Medical Colleges
* Rural Health Clinics

Patients are indirect beneficiaries rather than primary users.

---

## Core Objectives

The system should:

* assist clinicians in TB screening
* reduce interpretation time
* improve screening workflow
* provide explainable AI predictions
* localize suspicious pulmonary regions
* generate structured preliminary observations
* support research and education

---

# Canonical Assets & Production Model

## Canonical Classification Model

**Superseded — updated to match `docs/ARCHITECTURE.md`, the verified current source of truth.** `niriknet_best.keras` and the custom "NirikNet" architecture described below are no longer canonical; they were replaced by a ResNet-50 teacher / DenseNet-121 student trained via knowledge distillation (see Stage 6/7 above).

The ONLY production classification model is:

`densenet121_student.onnx` (exported from `training/nirikNetMain.py`; the training-time native-Keras equivalent is `student_best.weights.h5` + `build_densenet_student`, kept **only** for the explainability path — see below)

This model is the single source of truth for the Nirikhshon project.

### Requirements

- Backend inference MUST load only `densenet121_student.onnx` (via ONNX Runtime) for the `/predict` path.
- The ResNet-50 teacher MUST NEVER be loaded in a serving process, for any purpose.
- Grad-CAM and its variants require live TensorFlow gradient access, which ONNX Runtime does not provide. Generating them MUST use a separately-loaded native-Keras copy of the **same trained student weights** (`build_densenet_student` + `student_best.weights.h5`) — never a different model, never the teacher. This dual-load (ONNX for `/predict`, native Keras for `/explain`) is the resolved architecture decision for the explainability-vs-ONNX gap.
- Backend preprocessing MUST exactly match `preprocess_xray()` in `training/nirikNetMain.py` — as of this update, that includes CLAHE (2.0/8×8), 8.5%-padded lung crop with hole-filled mask, resize to 224×224, and non-lung-pixel mean-fill masking. **A real, unfixed discrepancy was found**: `hf_space/inference/preprocessing.py` does not currently implement the mean-fill masking step and adds an extra `pad_to_square` step that doesn't exist in training — this is a genuine train/serve preprocessing mismatch, not a documented variant, and should be treated as an open bug until reconciled.
- Evaluation MUST use `densenet121_student.onnx` (or the equivalent native-Keras weights for methods ONNX can't run, e.g. the Grad-CAM audits).
- Frontend predictions MUST originate from `densenet121_student.onnx`, via the backend.
- Hugging Face Spaces deployment MUST use `densenet121_student.onnx` for `/predict` and the native-Keras student copy for `/explain`.

### Deprecated Models

Any previous classification models are deprecated, including but not limited to:

- **NirikNet** (the custom from-scratch CNN this project used before the DenseNet-121 pivot — now itself deprecated, the inverse of what earlier project documentation stated)
- EfficientNetV2-M (the previous teacher; ResNet-50 is now the teacher)
- Experimental CNNs
- Previous `.keras` or `.h5` classification models not matching the current `build_teacher`/`build_densenet_student` definitions in `training/nirikNetMain.py`

These models must never be used for inference, evaluation, Grad-CAM generation, deployment, or demonstrations unless explicitly requested for historical comparison. **DenseNet-121 and ResNet-50 are the current, canonical architectures** — do not treat their presence in older "deprecated workflow" documentation elsewhere in this file as still applicable; see the clarifying note under Deprecated Architecture below.

### Canonical Output Rule

All generated outputs must originate from the canonical production model.

If the canonical model changes, all previously generated outputs become deprecated and must be regenerated.

This includes:

- Grad-CAM visualizations
- Grad-CAM++ visualizations
- LayerCAM visualizations
- EigenCAM visualizations
- Heatmaps (there is no "Consensus CAM" — see Explainability Standards below; don't regenerate something that was never implemented)
- Segmentation overlays
- Predictions
- Evaluation metrics
- Clinical reports
- Exported artifacts

## System Overview

The project consists of four major components.

### 1. Dataset Engineering Pipeline

Responsible for:

* dataset discovery
* validation
* cleaning (deduplication, image quality filter, mask-coverage filter)
* duplicate detection
* metadata generation
* patient-wise train/validation/test preparation (2:1 Normal:TB rebalancing before the split)
* lung mask generation (always trained — the Attention U-Net, not "if needed")
* CLAHE preprocessing
* resizing to **224 × 224** (not 384×384)
* grayscale to RGB conversion
* non-lung pixel mean-fill masking (no separate "EfficientNetV2 preprocessing" step — each model normalizes internally)

---

### 2. AI Pipeline

The AI pipeline follows the 12-stage process (see Stage Details above for the current, verified specifics):

Stage 1: Dataset Collection
Stage 2: Dataset Discovery
Stage 3: Dataset Cleaning
Stage 4: Lung Segmentation
Stage 5: Image Preprocessing
Stage 6: Teacher Network (ResNet-50)
Stage 7: Student Network (DenseNet-121)
Stage 8: Knowledge Distillation
Stage 9: Evaluation
Stage 10: Explainable AI
Stage 11: Model Saving
Stage 12: Deployment

---

### 3. Backend API

Provides

* inference
* model serving
* DICOM processing
* database
* audit logs
* report generation
* REST API

---

### 4. Frontend Workstation

Provides

* PACS-style viewer
* DICOM viewing
* Heatmaps
* ROI visualization
* Report generation
* Clinical workflow
* Annotation tools
* Integration with Hugging Face backend
* LLM-generated explanations (e.g., Gemini)

---

## Technical Stack

Frontend

* Next.js
* React
* TypeScript
* Tailwind CSS

Backend

* Flask
* Python

Deep Learning

* TensorFlow / Keras
* PyTorch (where appropriate)

Computer Vision

* OpenCV
* NumPy

Medical Imaging

* pydicom

Database

* SQLite

Deployment

* Docker
* Hugging Face Spaces
* Vercel

Development

* Kaggle
* Git
* GitHub

---

## High-Level AI Architecture

The AI pipeline follows the 12-stage process (see Stage Details near the top of this file, and `docs/ARCHITECTURE.md`, for the current specifics):

Stage 1: Dataset Collection
Stage 2: Dataset Discovery
Stage 3: Dataset Cleaning
Stage 4: Lung Segmentation
Stage 5: Image Preprocessing
Stage 6: Teacher Network (ResNet-50)
Stage 7: Student Network (DenseNet-121)
Stage 8: Knowledge Distillation
Stage 9: Evaluation
Stage 10: Explainable AI
Stage 11: Model Saving
Stage 12: Deployment

The clinician always has the final authority over every prediction.

## Repository Structure

Claude must preserve the overall repository architecture.

Unless explicitly instructed otherwise, use the following directory responsibilities.

**Real current structure differs from the illustrative tree below** — as of this update the repo actually has `training/nirikNetMain.py` (one consolidated file, not the `notebooks/` layout shown), `docs/` (architecture and API specifications), and `hf_space/` (the actual current backend implementation, alongside an older `backend/`). See `docs/ARCHITECTURE.md` §11 for the authoritative current folder-responsibility mapping; don't assume the tree below already exists verbatim.

```text
project-root/

backend/
    app.py
    api/
    core/
    models/
    utils/
    database/

frontend/
    app/
    components/
    hooks/
    services/
    lib/
    public/

notebooks/
    01_dataset_preparation.ipynb
    02_lung_segmentation_preprocessing.ipynb
    03_model_training_distillation.ipynb
    04_explainability_deployment.ipynb

datasets/
    raw/
    processed/
    metadata/

experiments/
    experiment_001/
    experiment_002/
    ...

exports/

reports/

tests/

docs/
```

Claude should never create unnecessary folders.

Always reuse the existing architecture whenever possible.

---

# Machine Learning Development Workflow

Machine learning work inside this repository must follow a fixed engineering pipeline.

Never skip stages.

The workflow is:

Dataset Engineering

↓

Data Validation

↓

Metadata Generation

↓

Lung Segmentation & Preprocessing

↓

Model Training & Knowledge Distillation

↓

Evaluation

↓

Explainability & Deployment

↓

Testing

↓

Documentation

Claude must never jump directly into model training without ensuring that the dataset has already been validated and preprocessed.

---

# Notebook Contracts

Each notebook has exactly one responsibility.

**Same caveat as Notebook Responsibilities above: this is a target decomposition, not current reality.** `training/nirikNetMain.py` is currently one file; the exports/paths below (e.g. `master_metadata.csv`, `teacher_efficientnetv2m.keras`, `niriknet_best.keras`) also describe the earlier EfficientNetV2-M/NirikNet design — see the corrected Stage 6/7/11 details near the top of this file and `docs/ARCHITECTURE.md` for what the pipeline actually exports today (`densenet121_student.onnx`, `attention_unet.keras`, `metrics.json`, `train.csv`/`val.csv`/`test.csv`, etc.).

---

## Notebook 1

Dataset Preparation

Responsibilities

* dataset discovery
* validation
* metadata generation
* duplicate detection
* corrupted image detection
* dataset statistics
* train/validation/test split (patient-wise to prevent leakage)

Notebook 1 must NEVER train any model.

Exports

* master_metadata.csv (with paths to raw images, labels, dataset names, patient IDs, mask paths, split)
* dataset_statistics.csv
* segmentation_metadata.csv
* heatmap_metadata.csv
* train.csv
* validation.csv
* test.csv

---

## Notebook 2

Lung Segmentation & Image Preprocessing

Responsibilities

* Load raw images and masks from Notebook 1 outputs
* Train Attention U-Net (if lung masks not provided in dataset)
* Generate lung segmentation masks for all images
* Apply CLAHE preprocessing (clipLimit=2.0, tileGridSize=(8,8))
* Extract largest connected component with 5% padding
* Resize to 384 × 384 pixels
* Convert grayscale to 3-channel RGB
* Apply EfficientNetV2 preprocessing
* Save canonical images to `/kaggle/working/datasets/processed/canon/`
* Save canonical masks to `/kaggle/working/datasets/processed/canon_mask/`
* Update master_metadata.csv with paths to processed images and masks

Notebook 2 must NEVER perform classification training.

Exports

* attention_unet.keras (trained or loaded)
* canonical images in `/kaggle/working/datasets/processed/canon/`
* canonical masks in `/kaggle/working/datasets/processed/canon_mask/`
* updated master_metadata.csv

---

## Notebook 3

Model Training & Knowledge Distillation

Responsibilities

* Load preprocessed data from Notebook 2 outputs
* Load or create EfficientNetV2-M teacher model
* Load or create NirikNet student model
* Freeze teacher backbone initially, then unfreeze top 20% (excluding BatchNorm)
* Implement knowledge distillation with:
  * Temperature = 4
  * Alpha = 0.5 (balance between hard and soft loss)
  * Class weights for imbalance
  * Gradient clipping (clipnorm=1.0)
  * Mixed Precision (mixed_float16)
  * AdamW optimizer
  * Warm-up cosine decay learning rate
* Train models
* Evaluate on validation set
* Save best models based on validation AUC
* Generate training history and metrics

Notebook 3 must NEVER handle raw data or perform segmentation/preprocessing.

Exports

* teacher_efficientnetv2m.keras
* niriknet_best.keras
* niriknet.keras (final model)
* training_history.csv
* validation_metrics.json
* test_metrics.json

---

## Notebook 4

Explainability & Deployment

Responsibilities

* Load best student model (niriknet_best.keras) from Notebook 3
* Generate explainability visualizations:
  * Grad-CAM
  * Grad-CAM++
  * LayerCAM
  * EigenCAM
  * Consensus CAM (average of normalized individual CAM outputs)
* Generate LLM-based explanations (e.g., using Gemini) highlighting:
  * AI-assisted interpretation
  * Not a medical diagnosis
  * Clinician remains final decision maker
* Prepare final model package for:
  * Hugging Face deployment
  * Vercel frontend integration
* Validate deployment readiness

Notebook 4 must NEVER handle training data or modify model architecture.

Exports

* explainability_images/ (Grad-CAM variants and Consensus CAM)
* explanation_texts/ (LLM-generated explanations)
* deployment_package/ (for Hugging Face and Vercel)
* final_metrics.json
* deployment_readiness_report.txt

---

# Dataset Standards

Claude must understand that every dataset serves a different purpose.

Datasets must never be merged blindly.

Each dataset should retain its identity.

---

## Lung Segmentation

Primary datasets

* Montgomery
* Shenzhen
* Chest X-ray Masks and Labels

Purpose

Train Attention U-Net.

---

## TB Classification

Primary datasets

* Jaypee University
* Shenzhen
* TBX11K
* TB Chest Radiography Dataset (Tawsifur Rahman)

Purpose

Train EfficientNetV2-M (teacher) and NirikNet (student).

---

## Explainability Validation

Primary dataset

* TBX11K

Purpose

Compare Grad-CAM localization and variants against lesion annotations whenever annotation files are available.

---

# Dataset Rules

Always

* verify labels
* verify class balance
* verify image readability
* detect duplicate images
* detect corrupted images

Never assume labels from different datasets are identical.

Always normalize labels before training.

Preferred labels

TB

Normal

---

# Dataset Preparation Rules

Notebook 1 must perform

Image validation

↓

Duplicate detection

↓

Corrupted image detection

↓

Metadata generation

↓

Train Validation Test split (patient-wise)

↓

Dataset statistics

↓

Export

Only after these stages are complete should Notebook 2 begin.

---

# Experiment Tracking

Every experiment must receive a unique experiment identifier.

Example

experiment_001

experiment_002

experiment_003

Each experiment folder should contain

```text
model.keras

history.csv

metrics.json

configuration.json

confusion_matrix.png

roc_curve.png

notes.md
```

Never overwrite previous experiments.

Always create a new experiment folder.

---

# Configuration Management

Magic numbers are prohibited.

Hyperparameters must be stored centrally.

Examples

* image size (224 × 224 classification, 256 × 256 segmentation)
* batch size (32)
* epochs (per-stage — see Stage 6-8 above)
* optimizer (AdamW, weight decay 1e-4, gradient clipnorm 1.0)
* learning rate (with warm-up cosine decay)
* distillation temperature (3.0)
* distillation alpha (0.5)
* threshold (Youden's J, frozen from validation)
* augmentation parameters
* CLAHE parameters (clipLimit=2.0, tileGridSize=(8,8))

Claude should recommend configuration files instead of scattered constants.

---

# Logging

Training should never rely only on print statements.

Preferred order

1. logging module
2. tqdm progress bars
3. TensorBoard
4. CSV logs

Every important stage should produce meaningful logs.

---

# Error Handling

Claude should always write defensive code.

Missing datasets

↓

Warn

Missing masks

↓

Warn (will generate via U-Net if needed)

Corrupted images

↓

Skip

Missing annotations

↓

Continue (qualitative analysis still possible)

Training should fail only when absolutely necessary.

The pipeline should recover whenever practical.

---

# Medical AI Engineering Standards

This repository follows research-grade Medical AI engineering principles.

Claude must optimize for:

* explainability
* robustness
* reproducibility
* clinical usefulness

rather than simply maximizing classification accuracy.

The project is intended for academic research and engineering education.

It is **NOT** a diagnostic medical device.

---

# Clinical Safety Principles

The clinician is always the final decision maker.

The AI only assists.

Never generate text suggesting the AI has diagnosed tuberculosis.

Preferred terminology

* AI Screening Result
* Preliminary Finding
* Suspicious for Pulmonary Tuberculosis
* Screening Recommendation

Avoid

* Confirmed TB
* Definitive Diagnosis
* Patient has TB
* AI diagnosed TB

Always communicate uncertainty.

---

# AI Pipeline Contract

Every prediction must follow the 12-stage pipeline.

Stage 1: Dataset Collection
Stage 2: Dataset Discovery
Stage 3: Dataset Cleaning
Stage 4: Lung Segmentation
Stage 5: Image Preprocessing
Stage 6: Teacher Network
Stage 7: Student Network
Stage 8: Knowledge Distillation
Stage 9: Evaluation
Stage 10: Explainable AI
Stage 11: Model Saving
Stage 12: Deployment

Claude should never bypass intermediate stages.

---

# Image Quality Assessment

Before inference every image should be checked for

* readability
* orientation
* brightness
* contrast
* corruption
* supported format

Poor-quality images should generate warnings rather than silent failures.

---

# Image Enhancement Standards

Preferred preprocessing

* CLAHE
* normalization
* resizing
* grayscale preservation where appropriate

Image enhancement should remain deterministic.

Training images and inference images must use identical preprocessing pipelines.

---

# Lung Segmentation Standards

Segmentation exists to reduce shortcut learning.

The classifier should learn pulmonary pathology rather than image artifacts.

Preferred architecture

Attention U-Net

Alternative architectures

* U-Net
* U-Net++

Segmentation outputs should be reusable.

Never retrain segmentation unnecessarily.

Export masks whenever practical.

---

# Classification Standards

Preferred architecture

**ResNet-50 (teacher) and DenseNet-121 (student)** — superseded from an earlier EfficientNetV2-M/NirikNet design; see Stage 6/7 above and `docs/ARCHITECTURE.md` §15 for the full reasoning.

Reason

ResNet-50 is a well-established, ImageNet-pretrained backbone that trains a strong teacher without architecture novelty risk. DenseNet-121 was chosen as the student specifically to replace a from-scratch custom CNN ("NirikNet") that carried real overfitting risk on this project's modest training pool (~2,500 train images) and an ~11× capacity gap from the teacher; DenseNet-121 starts from ImageNet weights and narrows that gap to ~3.35×, with direct precedent in chest-radiograph classification research (CheXNet).

Alternative architectures

* EfficientNet (other variants)
* ConvNeXt
* The original custom "NirikNet" CNN (deprecated — see Canonical Assets & Production Model above)

Vision Transformers should only be recommended when sufficient data and computational resources are available.

---

# Explainability Standards

Explainability is mandatory.

Every prediction should include Grad-CAM and its variants.

Preferred explainability hierarchy

1. Grad-CAM
2. Grad-CAM++
3. LayerCAM
4. EigenCAM
5. ROI Localization

All four CAM methods are actually implemented (`training/nirikNetMain.py`, `explain_predictions`) and generated against the DenseNet-121 student's backbone. **"Attention Maps" and "Consensus CAM" are not implemented and should not be presented as available** — the student has no attention mechanism (CBAM/SE were part of the deprecated NirikNet design, not DenseNet-121; the only attention in this project lives inside the Attention U-Net's segmentation decoder, unrelated to classification explainability), and Consensus CAM (an average of the four methods) exists only as a documentation aspiration, never implemented in code. If either is genuinely wanted, it needs to be built as new work, not assumed to already exist.

Predictions without explanations are considered incomplete. A prediction's explanation is also considered incomplete without evidence it reflects real pulmonary signal, not a shortcut — see the Grad-CAM/lung-mask localization audit below.

---

# Heatmap Validation

Whenever lesion annotations or bounding boxes exist,

Claude should recommend validating Grad-CAM localization against the annotated regions.

Possible validation methods

* IoU
* Dice Similarity
* qualitative comparison
* clinician review
* **`audit_gradcam_lung_localization`** (implemented, `training/nirikNetMain.py`) — this project's actual, working validation: measures the fraction of Grad-CAM heatmap energy falling inside the segmented lung mask (not lesion annotations specifically, but a real, automated, currently-used check), plus a border/edge-concentration flag. Run this — and check `audit_per_source_performance` alongside it — before trusting any training run's Grad-CAM output as clinically meaningful.

Never assume Grad-CAM is clinically meaningful without validation. A completed training run without a checked localization audit does not satisfy this rule.

---

# Anatomical Localization

The application should map findings into anatomical lung regions.

Preferred zones

Left Upper

Left Middle

Left Lower

Right Upper

Right Middle

Right Lower

Whenever possible report

* dominant lung
* dominant zone
* estimated affected area

---

# Observation Generation

The classifier should not output only probabilities.

Whenever possible produce structured observations.

Example

Suspicious opacity detected.

Right upper lung involvement.

Attention concentrated in highlighted region.

Features suspicious for pulmonary tuberculosis.

Recommend clinical correlation.

This report is intended for screening assistance only.

---

# Report Generation

Reports should contain

Patient Information

Study Information

Original Image

Segmented Lung

Grad-CAM

Grad-CAM++

LayerCAM

EigenCAM

AI Confidence

Observations

Recommendation

Clinical Disclaimer

Never generate reports implying confirmed diagnosis.

---

# Evaluation Standards

Every classification experiment should report

Accuracy

Precision

Recall

Sensitivity

Specificity

F1 Score

ROC-AUC

PR-AUC

Confusion Matrix

ROC Curve

Precision Recall Curve

Training Loss

Validation Loss

Training Accuracy

Validation Accuracy

Never report accuracy alone.

---

# Segmentation Evaluation

Preferred metrics

Dice Coefficient

IoU

Pixel Accuracy

Boundary Accuracy

Visual Examples

---

# Explainability Evaluation

Whenever annotation data exists evaluate

Grad-CAM overlap

Grad-CAM++ overlap

LayerCAM overlap

EigenCAM overlap

Whenever annotation data does not exist

Perform qualitative visual inspection, and run `audit_gradcam_lung_localization` (lung-mask overlap, no lesion annotations required) plus `audit_per_source_performance` — both implemented and this project's actual current evaluation method for explainability quality.

---

# Research Standards

Claude should think like a research engineer.

Every implementation should answer

Why is this model chosen?

Why this dataset?

Why this preprocessing?

Why this architecture?

Why these evaluation metrics?

Why this threshold?

Every engineering decision should have a technical justification.

---

# Novelty Requirements

Claude should avoid reproducing common GitHub projects.

Preferred research contributions

Explainable AI (with multiple CAM variants — Grad-CAM, Grad-CAM++, LayerCAM, EigenCAM — plus quantitative lung-localization auditing; no Consensus CAM implemented, see Explainability Standards)

Clinical Workflow Integration (12-stage pipeline)

Structured Reporting

Anatomical Localization

Quality Assessment

Heatmap Validation (with multiple metrics)

Research Reproducibility

The project novelty is the end-to-end explainable TB screening workflow with teacher-student knowledge distillation rather than classification alone.

---

# Reproducibility Requirements

Every experiment should save

Model

Weights

Configuration

Metrics

Training History

Random Seed

Dataset Version

Training Timestamp

Experiment Identifier

Claude should encourage reproducible research practices whenever practical.

---

# Performance Optimisation

Prefer

mixed precision

early stopping

learning rate scheduling

checkpointing

gradient clipping

model checkpoint recovery

Avoid unnecessary GPU memory usage.

Avoid loading datasets repeatedly.

Reuse cached data whenever possible.

---

# Code Quality Standards

Machine learning code should

* be modular
* be documented
* use type hints where practical
* avoid duplicated preprocessing
* avoid notebook spaghetti
* separate configuration from implementation
* separate training from evaluation
* separate evaluation from visualization

Every notebook should execute from top to bottom without manual intervention.

---

# Documentation Requirements

Every notebook should document

Purpose

Inputs

Outputs

Dependencies

Expected Runtime

Exported Files

Failure Conditions

Future notebooks should be able to understand previous notebook outputs without reading the implementation.

---

# Self Review

Before completing any substantial ML implementation Claude should internally verify

✓ Medical terminology is clinically appropriate

✓ Notebook remains reproducible

✓ No duplicated preprocessing exists

✓ Explainability is preserved

✓ Metrics are complete

✓ Code remains modular

✓ Outputs satisfy notebook contracts

✓ Research methodology remains defensible

If any requirement fails,

Claude should improve the implementation before returning it.

---

# Claude Code Autonomous Engineering Workflow

Claude should approach this repository as a long-term engineering project rather than a sequence of unrelated prompts.

Before implementing any feature, Claude must internally perform the following workflow.

Project Understanding

↓

Task Planning

↓

Task Decomposition

↓

Skill Selection

↓

Subagent Delegation

↓

Implementation

↓

Testing

↓

Self Review

↓

Documentation

↓

Completion

Claude should avoid immediately writing code without first understanding the broader engineering objective.

---

# Planning First

Before writing code Claude should answer internally:

What problem is being solved?

What existing architecture already exists?

Will this change break another notebook?

Can this component be reused?

Should this be implemented as a module rather than notebook code?

Can existing code be improved instead of duplicated?

Never duplicate functionality that already exists elsewhere in the repository.

---

# Skills

Claude should think in terms of engineering disciplines.

The following Skills should be treated as independent specialists.

---

## Dataset Engineer

Responsibilities

* dataset discovery
* metadata generation
* duplicate detection
* corrupted image detection
* preprocessing
* dataset versioning
* train validation test creation (patient-wise to prevent leakage)

Never performs model training.

---

## Computer Vision Engineer

Responsibilities

* OpenCV
* preprocessing
* CLAHE (clipLimit=2.0, tileGridSize=(8,8))
* image normalization
* augmentation
* image enhancement
* largest connected component extraction
* morphological cleanup + border-seeded flood-fill hole closing
* 8.5% border padding application
* grayscale to RGB conversion
* non-lung pixel mean-fill masking (post-CLAHE) — not raw zero-fill, and not a separate "EfficientNetV2 preprocessing" step (each model normalizes internally via its own `Lambda` layer)

Responsible only for image processing.

---

## Segmentation Engineer

Responsibilities

* U-Net
* Attention U-Net (primary for this project)
* segmentation metrics
* Dice coefficient
* IoU
* mask generation
* largest connected component analysis
* morphological operations
* padding application

Responsible only for segmentation.

---

## Classification Engineer

Responsibilities

* ResNet-50 (teacher model)
* DenseNet-121 (student model — not the deprecated custom "NirikNet" CNN)
* transfer learning
* fine tuning
* knowledge distillation (temperature-scaled KLD on pre-softmax logits, cast to float32 before any loss math)
* threshold optimisation (Youden's J on validation data, frozen before test)
* layer-wise unfreezing strategies (staged freeze: head-only, then partial backbone unfreeze)
* gradient clipping and mixed-precision numerical stability (a real, confirmed source of NaN training failures on this project — never skip clipnorm or the float32 logits cast when touching `DistillationModel`)

Responsible only for classification. Note: SE/CBAM attention mechanisms belonged to the deprecated NirikNet architecture — DenseNet-121 has no attention mechanism.

---

## Explainability Engineer

Responsibilities

* Grad-CAM
* Grad-CAM++
* LayerCAM
* EigenCAM
* quantitative lung-localization auditing (`audit_gradcam_lung_localization`, `audit_per_source_performance`) — this project's actual current heatmap-quality validation
* ROI extraction
* LLM-based explanation generation (Gemini, via Google AI Studio API key)
* explanation validation (ensuring AI-assisted, not diagnostic)

Responsible only for explainability. Note: no "Consensus CAM" or classification-level "attention maps" exist in the current implementation — see Explainability Standards. The dual-load requirement (ONNX for `/predict`, native Keras for `/explain`) is this role's responsibility to maintain in the backend.

---

## Medical AI Reviewer

Responsibilities

Verify

medical terminology

clinical workflow

report wording

AI safety

diagnostic language

This skill prevents clinically unsafe wording.

---

## Research Engineer

Responsibilities

Research methodology

dataset justification

model justification

evaluation methodology

novelty analysis

paper readiness

Every engineering decision should have a research justification.

---

## Evaluation Engineer

Responsibilities

Confusion Matrix

ROC

AUC

Precision

Recall

Sensitivity

Specificity

F1

Calibration

Threshold optimisation

Never report accuracy alone.

---

## Backend Engineer

Responsibilities

Flask

API

SQLite

DICOM

Authentication

REST

Inference
Hugging Face integration
Vercel integration
Model serving
API endpoints for prediction
Audit logging
Report generation endpoints

---

## Frontend Engineer

Responsibilities

Next.js

React

TypeScript

Tailwind

DICOM Viewer

Heatmap Rendering

Report UI

Vercel deployment
Integration with Hugging Face backend
Display of:
- Prediction
- Confidence score
- Explainability heatmaps (Grad-CAM variants + Consensus CAM)
- LLM-generated explanations
- Clinical disclaimer

---

## DevOps Engineer

Responsibilities

Docker

Deployment

Environment Variables

CI/CD

Performance

Model Packaging

Hugging Face Spaces deployment
Vercel deployment
Container optimization
Environment variable management
Continuous integration/continuous deployment
Model serialization for deployment
Resource optimization
Latency minimization
Throughput maximization

---

## Documentation Engineer

Responsibilities

README

Research Documentation

Notebook Documentation

Architecture Diagrams

Code Comments

API Documentation
Deployment guides
User manuals
Technical specifications
Release notes
Troubleshooting guides

---

# Subagents

Complex work should be divided into specialised subagents.

Claude should reason as though these agents exist even if implementation occurs in a single response.

---

## Dataset Auditor

Checks

duplicates

corrupted images

class balance

resolution

missing files

---

## Dataset Splitter

Creates

train

validation

test

while preventing leakage (patient-wise).

---

## Metadata Builder

Produces

master_metadata.csv

dataset_statistics.csv

segmentation_metadata.csv

heatmap_metadata.csv

---

## Segmentation Trainer

Builds

Attention U-Net

evaluates segmentation

exports masks.

---

## Model Trainer

Builds

ResNet-50 (teacher)

DenseNet-121 (student — not the deprecated custom "NirikNet")

performs transfer learning

performs knowledge distillation

exports trained models (`densenet121_student.onnx` for deployment; teacher checkpoints are training-only, never shipped).

---

## Explainability Validator

Generates

Grad-CAM

Grad-CAM++

LayerCAM

EigenCAM

validates heatmaps via `audit_gradcam_lung_localization` (quantitative lung-mask overlap) and `audit_per_source_performance` (per-source shortcut detection) — the actual implemented validation, not just qualitative review

creates ROI statistics.

Generates LLM-based explanations. Note: no "Consensus CAM" exists — see Explainability Standards.

---

## Deployment Preparer

Prepares

model package for Hugging Face

model package for Vercel

API endpoints

frontend integration

deployment validation

---

## Metrics Auditor

Verifies

metrics

graphs

confusion matrix

ROC

AUC

training history.

validation results

test results

explainability quality

---

## Research Reviewer

Checks

novelty

limitations

future work

methodology

academic quality.

---

## Clinical Safety Validator

Ensures

medically appropriate terminology

absence of diagnostic claims

proper uncertainty communication

clinical disclaimer presence

---

# Task Delegation Strategy

Simple tasks

↓

Single Skill

Medium tasks

↓

Multiple Skills

Large architectural tasks

↓

Multiple Skills

*

Multiple Subagents

Claude should think in terms of engineering collaboration rather than isolated coding.

---

# Engineering Decision Framework

Whenever Claude must choose between two implementations evaluate

Correctness

↓

Clinical Safety

↓

Maintainability

↓

Reusability

↓

Performance

↓

Complexity

↓

Development Speed

Never optimise only for development speed.

---

# Notebook Completion Checklist

Before considering a notebook complete Claude should verify

Dataset Inputs

Outputs

Documentation

Logging

Error Handling

Reproducibility

Configuration

Metrics

Visualisations

Exported Files

Notebook Contract

Future Notebook Compatibility

Every notebook must satisfy all checklist items.

---

# Pull Request Mentality

Every implementation should be treated as if it will undergo professional code review.

Claude should ask internally

Would another ML Engineer understand this?

Is the code reusable?

Is the architecture clean?

Can this scale?

Would this pass peer review?

If the answer is no,

improve the implementation before returning it.

---

# Final Review

Before responding Claude should mentally perform one final engineering review.

Confirm

✓ architecture consistency

✓ notebook compatibility

✓ dataset compatibility

✓ medical terminology

✓ reproducibility

✓ modularity

✓ explainability

✓ research quality

✓ deployment compatibility

✓ documentation completeness

Claude should optimise for producing software that would be acceptable in a graduate-level medical AI research laboratory rather than merely producing code that runs.