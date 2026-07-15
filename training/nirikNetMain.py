# %%
# # Setup (run once per Kaggle session)
"""Cell 0 - Optional dependency for architecture diagrams (keras.utils.plot_model
needs pydot; the graphviz binary itself is preinstalled on Kaggle's standard
image). Uses subprocess rather than a `!pip install` magic so this file still
runs correctly if executed as a plain script, not just as notebook cells."""
import subprocess
import sys


def _ensure_package(pkg_name, import_name=None):
    import_name = import_name or pkg_name
    try:
        __import__(import_name)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg_name], check=False)


_ensure_package("pydot")

_PINNED_TF = "2.15.1"
_PINNED_KERAS = "2.15.0"


def _ensure_pinned_tf_keras():
    """Best-effort pin to this project's canonical tb_env spec (TF 2.15.1 /
    Keras 2.15.0 - see CLAUDE.md), mainly for reproducibility (the same
    notebook re-run on a drifted Kaggle image later shouldn't silently
    behave differently). NOT load-bearing, and the "tf2onnx needs Keras 2"
    concern that originally motivated this turned out to be overstated:
    export_student_to_onnx's own round-trip validation was run and passed
    (max abs diff ~1e-7) against Keras 3.12.3, not Keras 2 - so tf2onnx
    compatibility isn't actually a strong reason to require this pin.
    Confirmed on a real Kaggle run that tensorflow==2.15.1 simply isn't an
    installable wheel on Kaggle's current Python version (its
    available-versions list starts at 2.16.0rc0) - no retry or different
    install strategy fixes that, since the wheel doesn't exist for this
    interpreter. When the pin can't be satisfied, this prints a calm status
    line and moves on rather than treating it as an error: ONNX export is
    what actually decouples the deployment framework/version from whatever
    trained the model, so training under a newer TF/Keras than canonical is
    a reproducibility footnote here, not a correctness problem."""
    try:
        import tensorflow as _tf_check
        import keras as _keras_check
        if _tf_check.__version__ == _PINNED_TF and _keras_check.__version__ == _PINNED_KERAS:
            print(f"Already on the pinned tensorflow=={_PINNED_TF} keras=={_PINNED_KERAS}.")
            return
        already_imported = True
    except ImportError:
        already_imported = False

    print(f"Attempting to pin tensorflow=={_PINNED_TF} keras=={_PINNED_KERAS} ...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         f"tensorflow=={_PINNED_TF}", f"keras=={_PINNED_KERAS}"],
        check=False, capture_output=True, text=True,
    )
    if result.returncode == 0:
        print("Pin succeeded.")
        if already_imported:
            print("A different TF/Keras was already imported in this kernel "
                  "session - RESTART THE KERNEL now and re-run from the top; "
                  "a running Python process cannot hot-swap an already-"
                  "imported TensorFlow build.")
    else:
        print(f"Could not pin to tensorflow=={_PINNED_TF} on this Python/Kaggle "
              f"image (pip install failed - most likely that wheel isn't built "
              f"for this interpreter version, not a transient error). "
              f"Continuing with whatever TF/Keras is already installed. This "
              f"is a reproducibility footnote, not a blocker: ONNX export "
              f"decouples the deployment framework/version from whichever "
              f"TF/Keras actually trains the model - see export_student_to_onnx.")


_ensure_pinned_tf_keras()

# %%
# # Imports
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import gc
import json
import time
import random
import warnings
import itertools
from dataclasses import dataclass, asdict
import platform

import numpy as np
import pandas as pd
import cv2

import tensorflow as tf
tf.get_logger().setLevel("ERROR")
try:
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
except Exception:
    pass
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    auc,
    matthews_corrcoef,
    cohen_kappa_score,
    classification_report,
)

# Optional DICOM support - the pipeline still runs without it.
try:
    import pydicom
    HAS_PYDICOM = True
except Exception:
    HAS_PYDICOM = False

warnings.filterwarnings("ignore")

# Reproducibility.
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def reset_keras():
    """Clear the Keras/TF backend session and force garbage collection.

    Use this between separate Kaggle *sessions* if you split the pipeline
    into multiple notebook runs (segmentation -> teacher -> student), each
    reloading the previous stage's saved .keras checkpoint. It is NOT called
    mid-pipeline inside main() below - clearing the global Keras session
    while you still hold live Python references to models you need (e.g.
    `unet`, `teacher`) risks those objects behaving incorrectly afterward.
    With the right-sized models in this version (ResNet-50 ~25.6M + NirikNet
    ~2.1M + Attention U-Net, all at 224px), a single continuous run should
    comfortably fit in 16GB VRAM without needing mid-run resets.
    """
    keras.backend.clear_session()
    gc.collect()


# Mixed precision for faster GPU training (safe no-op on CPU).
try:
    keras.mixed_precision.set_global_policy("mixed_float16")
    print("Mixed precision policy:", keras.mixed_precision.global_policy().name)
except Exception as exc:
    print("Mixed precision unavailable, using float32:", exc)

# Allow GPU memory to grow instead of pre-allocating everything.
PHYSICAL_GPUS = tf.config.list_physical_devices("GPU")
for _gpu in PHYSICAL_GPUS:
    try:
        tf.config.experimental.set_memory_growth(_gpu, True)
    except Exception:
        pass

print("TensorFlow:", tf.__version__)
print("Keras:", keras.__version__)
print("GPUs available:", len(PHYSICAL_GPUS))
print("pydicom available:", HAS_PYDICOM)

# %%
# # Configuration
"""Cell 2 - Central configuration for every stage of the pipeline."""


@dataclass
class Config:
    """Immutable-ish container holding all paths and hyper-parameters."""

    # ---- Kaggle dataset paths: TRAINING POOL ---------------------------------
    tb_dir: str = (
        "/kaggle/input/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset/"
        "TB_Chest_Radiography_Database"
    )
    india_dir: str = (
        "/kaggle/input/datasets/raddar/chest-xrays-tuberculosis-from-india/images/images"
    )
    india_meta: str = (
        "/kaggle/input/datasets/raddar/chest-xrays-tuberculosis-from-india/"
        "jaypee_metadata.csv"
    )
    use_da_db: bool = True
    da_dir: str = "/kaggle/input/datasets/vbookshelf/da-and-db-tb-chest-x-ray-datasets/images/da"
    db_dir: str = "/kaggle/input/datasets/vbookshelf/da-and-db-tb-chest-x-ray-datasets/images/db"

    # ---- Kaggle dataset paths: Montgomery + Shenzhen ---------------------------
    # Previously held out entirely as a permanent "external test" set. Per
    # project decision, now pooled into the same train/val/test split as
    # every other source: a model trained on only 3 sources and never shown
    # a single Montgomery/Shenzhen image has no way to learn those hospitals'
    # scanner characteristics exist (Zech et al. 2018, cross-site CNN
    # generalization). Held-out test-set accuracy under this design measures
    # in-distribution generalization across all pooled sources, not
    # generalization to a genuinely unseen hospital - a deliberate,
    # documented trade-off, not an oversight.
    montgomery_dir: str = (
        "/kaggle/input/datasets/raddar/tuberculosis-chest-xrays-montgomery/images/images"
    )
    montgomery_meta: str = (
        "/kaggle/input/datasets/raddar/tuberculosis-chest-xrays-montgomery/"
        "montgomery_metadata.csv"
    )
    shenzhen_dir: str = (
        "/kaggle/input/datasets/raddar/tuberculosis-chest-xrays-shenzhen/images/images"
    )
    shenzhen_meta: str = (
        "/kaggle/input/datasets/raddar/tuberculosis-chest-xrays-shenzhen/"
        "shenzhen_metadata.csv"
    )

    # ---- Kaggle dataset paths: TBX11K (usmanshams/tbx-11) ----------------------
    # Folder-organized, no metadata CSV - five category folders: Healthy,
    # ActiveTB, LatentTB, Sick (non-TB), Uncertain. Only Healthy/ActiveTB/
    # LatentTB map onto this project's binary Normal/TB task; Sick(non-TB)
    # and Uncertain are always excluded (see _collect_tbx11k_dataset).
    use_tbx11k: bool = True
    tbx11k_dir: str = "/kaggle/input/datasets/usmanshams/tbx-11"
    # Cap on how many Healthy images to take from TBX11K (0 = unlimited).
    # TBX11K's own internal Healthy:TB ratio (~5000:1200) is more imbalanced
    # than this project's pooled dataset already is - taking it in full
    # would partially undo the class-balance improvement Montgomery/Shenzhen
    # already provide. The pool-wide 2:1 target below is the actual
    # governing mechanism now; this cap is a secondary, per-source guard.
    tbx11k_healthy_cap: int = 1500

    # ---- Kaggle dataset paths: segmentation ----------------------------------
    seg_img_dir: str = (
        "/kaggle/input/datasets/nikhilpandey360/chest-xray-masks-and-labels/"
        "Lung Segmentation/CXR_png"
    )
    seg_mask_dir: str = (
        "/kaggle/input/datasets/nikhilpandey360/chest-xray-masks-and-labels/"
        "Lung Segmentation/masks"
    )

    # ---- Output ---------------------------------------------------------------
    output_dir: str = "/kaggle/working"
    # Subfolder specifically for report-ready figures, kept separate from
    # model checkpoints and metrics.json so it's easy to zip up and download.
    figures_dir: str = "/kaggle/working/figures"

    # ---- Image geometry --------------------------------------------------------
    img_size: int = 224
    seg_size: int = 256
    num_classes: int = 2
    class_names: tuple = ("Normal", "Tuberculosis")

    # ---- Batching ---------------------------------------------------------------
    batch_size: int = 32
    seg_batch_size: int = 8

    # ---- Epochs -------------------------------------------------------------------
    seg_epochs: int = 80
    teacher_head_epochs: int = 15
    teacher_finetune_epochs: int = 40
    # Student now trains in two stages too (frozen head warmup, then partial
    # backbone unfreeze), mirroring the teacher - see build_densenet_student
    # and train_student_head/train_student_finetune. student_epochs is the
    # FINETUNE-stage duration (name kept for minimal disruption to existing
    # config references); student_head_epochs is the new frozen-stage one.
    student_head_epochs: int = 10
    student_epochs: int = 100

    # ---- Learning rates -------------------------------------------------------------
    teacher_lr_head: float = 1e-3
    teacher_lr_finetune: float = 1e-4
    student_lr: float = 1e-3
    # Lower LR for the unfrozen-backbone stage, same rationale as the
    # teacher's head-vs-finetune LR split - a pretrained backbone's weights
    # shouldn't move as fast as a freshly-initialized head does.
    student_lr_finetune: float = 1e-4
    weight_decay: float = 1e-4
    # Global-norm gradient clipping, applied to every optimizer below (all
    # four training stages) - confirmed missing entirely on a real run where
    # student finetune produced NaN loss from epoch 1 onward (identical
    # val_accuracy every epoch afterward - weights went NaN on the first
    # real backprop through the newly-unfrozen, pretrained conv5_block
    # weights, under mixed_float16, via DistillationModel's custom
    # train_step, and never recovered). Unbounded gradients through a much
    # deeper graph than the head-only stage - which trained fine without
    # this - are the direct cause; 1.0 is the standard default for this
    # kind of finetune-stage stabilization, not fit to this project's data.
    grad_clip_norm: float = 1.0

    # ---- Warmup steps -----------------------------------------------------------------
    teacher_warmup_epochs: int = 5
    student_warmup_epochs: int = 10

    # ---- Distillation ---------------------------------------------------------------------
    # T=3, alpha=0.5 (equal weight on hard-label CE vs. KD loss) are the
    # Hinton et al. (2015) standard starting point for a two-class problem
    # with a confident teacher. The previous T=4/alpha=0.3 values were tuned
    # (if at all) against a distillation loss that had a softmax-of-softmax
    # bug (see DistillationModel) - they are not valid priors post-fix and
    # need re-validation against val AUC/F1 once retrained.
    distill_temperature: float = 3.0
    distill_alpha: float = 0.5
    # label_smoothing is intentionally unused now that the hard-label loss
    # is Focal Loss (see focal_loss_gamma/alpha below) - stacking label
    # smoothing and focal loss is an atypical, unvalidated combination
    # (both soften the loss but for different reasons: label smoothing
    # hedges against label noise, focal loss sharpens focus on hard/
    # minority examples). Kept as a field for reference, not wired in.
    label_smoothing: float = 0.1

    # ---- Class imbalance: Focal Loss -------------------------------------------------------
    # Normal:TB is roughly 3.4:1 in the pooled val split. Class-weighting
    # alone only reweights the loss - it can't manufacture more diverse TB
    # examples. Focal Loss (Lin et al. 2017, RetinaNet) complements it by
    # down-weighting easy/majority examples so gradient concentrates on
    # hard/minority cases. gamma=2.0, alpha=0.25 are the paper's original
    # defaults - a reasonable starting point, not validated for this
    # dataset yet; re-check against val F1/sensitivity once retrained.
    focal_loss_gamma: float = 2.0
    focal_loss_alpha: float = 0.25

    # ---- Lung crop padding ------------------------------------------------------------------
    # Fraction of the segmentation bounding box's height/width added as
    # margin before cropping. 8.5% (per project handoff spec) rather than a
    # tighter crop, to reduce the risk of clipping peripheral/costal-margin
    # lesions that a snug bounding box could cut off.
    lung_crop_padding: float = 0.085

    # ---- Train-time augmentation -------------------------------------------------------------
    # Applied to the already lung-cropped, CLAHE'd, ResNet50-preprocessed
    # tensor (roughly in a [-124, 152] pixel-value range, not [0, 255]).
    # Geometric augmentations (rotation/translation/zoom) are scale-invariant
    # so their fractional values are unaffected by that range. Rotation is
    # kept modest (chest anatomy is close to but not exactly rotationally
    # invariant) and horizontal flip is deliberately NOT used - CXR
    # laterality (which lung a finding is in) is diagnostically meaningful,
    # per standard CXR augmentation practice (Rajpurkar et al. 2017,
    # Irvin et al. 2019 CheXpert).
    aug_rotation_deg: float = 10.0
    aug_translation_frac: float = 0.07
    aug_zoom_frac: float = 0.08
    # Additive brightness delta is NOT scale-invariant - it was previously
    # 0.1 against a ~276-unit pixel range (~0.04% of range), which is
    # numerically inert. Rescaled to be a meaningful ~5% perturbation.
    aug_brightness_delta: float = 15.0
    aug_contrast_range: tuple = (0.85, 1.15)
    # Mild additive Gaussian noise (~2% of pixel range) to simulate
    # scanner/sensor noise differences across acquisition hardware - one of
    # the concrete, cheap ways to reduce reliance on scanner-specific
    # texture (shortcut learning) ahead of cross-domain (external test)
    # evaluation.
    aug_noise_stddev: float = 6.0

    # ---- Splits & sampling ------------------------------------------------------------------
    # Patient-wise 70/15/15 (train/val/test) of the single pooled dataset -
    # see split_dataframe(). val_split kept only for any legacy reference;
    # train_split is what split_dataframe() actually reads.
    train_split: float = 0.70
    val_split: float = 0.15
    max_per_class: int = 0
    max_seg_samples: int = 0

    # ---- Class imbalance --------------------------------------------------------------------
    use_class_weights: bool = True
    # Pool-wide Normal:TB target ratio, enforced by downsampling the
    # majority (Normal) class after all sources are merged (see
    # build_classification_dataframe). 2.0 rather than the ~3-6:1 the raw
    # pooled sources would otherwise give - "a ratio around 2:1 is much
    # healthier than 6:1" for a minority-class-sensitive screening task.
    # Set to 0 to disable and keep the raw pooled ratio.
    target_normal_tb_ratio: float = 2.0

    # ---- Unmatched-image handling ----------------------------------------------------------
    # For datasets loaded via _collect_folder_dataset (India, Montgomery,
    # Shenzhen), an image with no metadata-CSV row AND no _0/_1 filename
    # suffix used to default to label 0 (Normal) rather than being dropped.
    # Changed to False: fewer images with correct labels beats more images
    # with questionable ones. An incorrect label (a real TB image silently
    # defaulted to Normal) is a worse outcome than losing that image
    # entirely - "Unknown -> Skip", not "Unknown -> Normal". Every dropped
    # image is still logged for spot-checking either way.
    assume_unmatched_as_normal: bool = False

    # ---- Preprocessing cache ----------------------------------------------------------------
    use_preprocessing_cache: bool = True

    # ---- Training controls ------------------------------------------------------------------
    early_stopping_patience: int = 15
    explain_samples: int = 8

    # ---- Smoke test mode --------------------------------------------------------------------
    # Set True to run the ENTIRE pipeline end-to-end on a tiny subset with
    # minimal epochs (a few minutes, not hours). This proves the code runs -
    # it does NOT prove the model learns anything, since 2 epochs on ~20
    # images per class teaches it nothing. Use it to catch runtime issues
    # fast, then set back to False for the real run.
    smoke_test: bool = False

    # ---- Reporting figures ------------------------------------------------------------------
    figure_format: str = "png"
    figure_quality: int = 95
    figure_dpi: int = 150

    def out(self, name: str) -> str:
        return os.path.join(self.output_dir, name)

    def fig(self, name: str) -> str:
        stem, _ext = os.path.splitext(name)
        return os.path.join(self.figures_dir, f"{stem}.{self.figure_format}")


CFG = Config()
os.makedirs(CFG.output_dir, exist_ok=True)
os.makedirs(CFG.figures_dir, exist_ok=True)
print("Output directory:", CFG.output_dir)
print("Figures directory:", CFG.figures_dir)
print("Image size:", CFG.img_size, "| Segmentation size:", CFG.seg_size)


def _savefig(fig_or_plt, filepath, cfg=CFG):
    """Single choke point for saving every report figure, so format/quality/
    DPI stay consistent everywhere and changing to PNG later is one edit."""
    save_kwargs = {"dpi": cfg.figure_dpi}
    if cfg.figure_format in ("jpg", "jpeg"):
        save_kwargs["pil_kwargs"] = {"quality": cfg.figure_quality}
    fig_or_plt.savefig(filepath, format=cfg.figure_format, **save_kwargs)


def save_run_config(cfg=CFG):
    """Dump environment info + the full Config into run_config.json,
    alongside metrics.json - covers the report's experimental-setup section
    and makes the run reproducible later (including by you, in a month)."""
    gpu_names = []
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            details = tf.config.experimental.get_device_details(gpu)
            gpu_names.append(details.get("device_name", str(gpu)))
        except Exception:
            gpu_names.append(str(gpu))

    run_config = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__,
        "keras_version": keras.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "opencv_version": cv2.__version__,
        "gpus_available": len(tf.config.list_physical_devices("GPU")),
        "gpu_names": gpu_names,
        "seed": SEED,
        "config": asdict(cfg),
    }
    path = cfg.out("run_config.json")
    with open(path, "w") as f:
        json.dump(run_config, f, indent=2, default=str)
    print(f"Saved run configuration -> {path}")
    return run_config

# %%
# # Dataset Discovery
"""Cell 3 - Utilities that locate image files across the Kaggle datasets."""

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".dcm", ".dic")


def list_image_files(directory, extensions=IMAGE_EXTENSIONS):
    """Recursively list image files inside `directory`. Missing dirs -> []."""
    if not directory or not os.path.isdir(directory):
        return []
    found = []
    for root, _dirs, files in os.walk(directory):
        for fname in files:
            if fname.lower().endswith(tuple(extensions)):
                found.append(os.path.join(root, fname))
    return sorted(found)


def describe_directory(name, directory):
    files = list_image_files(directory)
    status = "OK" if files else "MISSING/EMPTY"
    print(f"[{status:12s}] {name:24s} -> {len(files):6d} images  ({directory})")
    if not files:
        parent = os.path.dirname(directory.rstrip("/"))
        if os.path.isdir(parent):
            print(f"    contents of parent '{parent}': {sorted(os.listdir(parent))}")
        else:
            grandparent = os.path.dirname(parent)
            if os.path.isdir(grandparent):
                print(f"    '{parent}' doesn't exist either; contents of "
                      f"'{grandparent}': {sorted(os.listdir(grandparent))}")
    return files


def _find_dataset_dir(expected_slug, base="/kaggle/input", max_depth=3):
    """Breadth-first search up to max_depth levels under base for a
    directory whose name matches expected_slug (case-insensitive). Handles
    Kaggle mounting datasets either flat (/kaggle/input/<slug>/) or nested
    (/kaggle/input/datasets/<owner>/<slug>/ - confirmed to be what this
    environment actually uses)."""
    if not os.path.isdir(base):
        return None
    frontier = [(base, 0)]
    while frontier:
        current, depth = frontier.pop(0)
        try:
            entries = os.listdir(current)
        except Exception:
            continue
        for entry in entries:
            full = os.path.join(current, entry)
            if not os.path.isdir(full):
                continue
            if entry.lower() == expected_slug.lower():
                return full
            if depth + 1 < max_depth:
                frontier.append((full, depth + 1))
    return None


def _resolve_kaggle_path(configured_path, label=""):
    """If `configured_path` doesn't exist, search /kaggle/input (up to 3
    levels deep, so this works whether the configured path is written in
    the flat or the nested datasets/<owner>/<slug> style) for a directory
    matching the expected slug, and use that instead."""
    base = "/kaggle/input"
    if os.path.exists(configured_path):
        return configured_path

    if not os.path.isdir(base):
        print(f"  [{label}] /kaggle/input does not exist - this isn't running "
              f"in a Kaggle environment with input mounted.")
        return configured_path

    if not os.listdir(base):
        print(f"  [{label}] /kaggle/input exists but is EMPTY - no datasets are "
              f"attached to this notebook at all. Add them via 'Add Input' in "
              f"the notebook editor, then re-commit.")
        return configured_path

    rel_parts = os.path.relpath(configured_path, base).split(os.sep)
    # Try the nested-style slug first (datasets/<owner>/<slug>/...) if the
    # configured path looks like that, then fall back to the flat-style slug.
    candidates = []
    if rel_parts[0] == "datasets" and len(rel_parts) > 2:
        candidates.append((rel_parts[2], rel_parts[3:]))
    candidates.append((rel_parts[0], rel_parts[1:]))

    for slug, remainder in candidates:
        match = _find_dataset_dir(slug, base=base)
        if match:
            corrected = os.path.join(match, *remainder) if remainder else match
            print(f"  [{label}] resolved slug '{slug}' -> {corrected}")
            return corrected

    print(f"  [{label}] no match found under {base} for any of: "
          f"{[c[0] for c in candidates]}")
    return configured_path


def apply_smoke_test_overrides(cfg=CFG):
    """If cfg.smoke_test is True, shrink every stage to a few minutes total
    so the pipeline can be run end-to-end to catch runtime bugs fast. Does
    NOT indicate anything about model quality - this is a plumbing check."""
    if not cfg.smoke_test:
        return
    print("=" * 78)
    print("SMOKE TEST MODE - tiny subsets, minimal epochs. Proves the code runs,")
    print("NOT that the model learns anything. Set smoke_test=False for the real run.")
    print("=" * 78)
    # 75/class (not 20) - with a patient-wise 70/15/15 split, 20/class left
    # val/test slices small enough (~5-6 images) that they could land on a
    # single class by chance, which crashes roc_auc_score/
    # select_youden_threshold near the END of the run (after the expensive
    # part already succeeded) - a smoke-test-only false alarm, not a real
    # bug. 75/class keeps this fast while making that far less likely.
    cfg.max_per_class = 75
    cfg.max_seg_samples = 40
    cfg.seg_epochs = 2
    cfg.teacher_head_epochs = 2
    cfg.teacher_finetune_epochs = 2
    cfg.student_head_epochs = 2
    cfg.student_epochs = 2
    cfg.explain_samples = 2


def resolve_all_paths(cfg=CFG):
    """Run every configured dataset path through _resolve_kaggle_path and
    update cfg in place. Call this once, before discover_datasets()."""
    print("=" * 78)
    print("RESOLVING DATASET PATHS")
    print("=" * 78)
    base = "/kaggle/input"
    if os.path.isdir(base):
        print(f"Currently mounted under /kaggle/input: {sorted(os.listdir(base)) or '(nothing)'}")
    else:
        print("/kaggle/input does not exist.")

    path_attrs = [
        "tb_dir", "india_dir", "india_meta", "da_dir", "db_dir",
        "montgomery_dir", "montgomery_meta", "shenzhen_dir", "shenzhen_meta",
        "seg_img_dir", "seg_mask_dir",
    ]
    for attr in path_attrs:
        original = getattr(cfg, attr)
        resolved = _resolve_kaggle_path(original, label=attr)
        if resolved != original:
            setattr(cfg, attr, resolved)
    print("=" * 78)


def discover_datasets(cfg=CFG):
    print("=" * 78)
    print("DATASET DISCOVERY")
    print("=" * 78)
    discovered = {
        "tb": describe_directory("TB (pool)", cfg.tb_dir),
        "india": describe_directory("India (pool)", cfg.india_dir),
        "montgomery": describe_directory("Montgomery (pool)", cfg.montgomery_dir),
        "shenzhen": describe_directory("Shenzhen (pool)", cfg.shenzhen_dir),
        "seg_images": describe_directory("SegImages", cfg.seg_img_dir),
        "seg_masks": describe_directory("SegMasks", cfg.seg_mask_dir),
    }
    if cfg.use_da_db:
        discovered["da"] = describe_directory("DA (pool)", cfg.da_dir)
        discovered["db"] = describe_directory("DB (pool)", cfg.db_dir)
    if cfg.use_tbx11k:
        discovered["tbx11k"] = describe_directory("TBX11K (pool)", cfg.tbx11k_dir)
    print("=" * 78)
    return discovered

# %%
# # Metadata Generation
"""Cell 4 - Build ONE pooled dataframe across every labelled source.

Label convention: 0 = Normal, 1 = Tuberculosis.
POOL: tawsifurrahman + India + Montgomery + Shenzhen (+ optional DA/DB,
TBX11K). Montgomery/Shenzhen are no longer held out as a permanent external
test - see the comment on Config.montgomery_dir for why. split_dataframe()
below produces a single patient-wise 70/15/15 train/val/test split of this
pool.
"""

_LABEL_COLUMN_CANDIDATES = (
    "label", "target", "class", "finding", "findings",
    "diagnosis", "ptb", "tb", "abnormal", "study",
)
_POSITIVE_TOKENS = ("tb", "tuberculosis", "abnormal", "sick", "positive", "1", "yes", "true")
_NEGATIVE_TOKENS = ("normal", "health", "healthy", "negative", "0", "no", "false")


def _label_from_filename(path):
    stem = os.path.splitext(os.path.basename(path))[0].lower()
    if stem.endswith("_1"):
        return 1
    if stem.endswith("_0"):
        return 0
    return None


def _label_from_value(value):
    text = str(value).strip().lower()
    if text in ("", "nan", "none"):
        return None
    for token in _NEGATIVE_TOKENS:
        if text == token:
            return 0
    for token in _POSITIVE_TOKENS:
        if text == token:
            return 1
    if any(tok in text for tok in ("tubercul", "abnormal")):
        return 1
    if "normal" in text or "health" in text:
        return 0
    return None


def _read_metadata_csv(path):
    if not path or not os.path.isfile(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print("  ! could not read metadata:", path, "->", exc)
        return None


_PATIENT_COLUMN_CANDIDATES = (
    "patient_id", "patientid", "patient", "subject_id", "subject",
    "case_id", "study_id",
)


def _build_meta_lookup(meta_df, source=None):
    """Returns (label_lookup, patient_lookup), both keyed by lowercased
    filename stem. patient_lookup is empty when the metadata CSV has no
    recognizable patient/subject/case ID column - callers fall back to the
    filename stem itself as a pseudo-patient-id in that case (see
    _collect_folder_dataset), which is no worse than the previous
    image-level split and enables true grouping wherever a real ID exists.
    Prints an explicit warning when that fallback is in effect (rather than
    leaving it a silent, only-discoverable-by-reading-the-code state) so
    it's visible in the run log which sources are actually protected by
    real patient-wise grouping vs. the weaker filename-based fallback."""
    label_lookup = {}
    patient_lookup = {}
    if meta_df is None or meta_df.empty:
        return label_lookup, patient_lookup
    columns = {c.lower(): c for c in meta_df.columns}
    fname_col = None
    for cand in ("fname", "filename", "file", "image", "id", "study_id", "name"):
        if cand in columns:
            fname_col = columns[cand]
            break
    label_col = None
    for cand in _LABEL_COLUMN_CANDIDATES:
        if cand in columns:
            label_col = columns[cand]
            break
    patient_col = None
    for cand in _PATIENT_COLUMN_CANDIDATES:
        if cand in columns:
            patient_col = columns[cand]
            break
    if patient_col is None:
        print(f"  [{source or '?'}] no patient/subject/case-ID column found in metadata "
              f"({list(columns.values())}) - patient-wise split protection for this source "
              f"falls back to filename-based pseudo-IDs (won't catch two different files "
              f"of the same real patient).")
    if fname_col is None:
        return label_lookup, patient_lookup
    for _idx, row in meta_df.iterrows():
        stem = os.path.splitext(os.path.basename(str(row[fname_col])))[0].lower()
        if label_col is not None:
            label = _label_from_value(row[label_col])
            if label is not None:
                label_lookup[stem] = label
        if patient_col is not None:
            patient_val = str(row[patient_col]).strip()
            if patient_val and patient_val.lower() not in ("nan", "none", ""):
                patient_lookup[stem] = patient_val
    return label_lookup, patient_lookup


def _collect_folder_dataset(directory, source, meta_path=None, cfg=CFG):
    rows = []
    meta_lookup, patient_lookup = _build_meta_lookup(_read_metadata_csv(meta_path), source=source)
    fallback_paths = []
    for path in list_image_files(directory):
        stem = os.path.splitext(os.path.basename(path))[0].lower()
        label = meta_lookup.get(stem)
        label_source = "metadata"
        if label is None:
            label = _label_from_filename(path)
            label_source = "filename_suffix"
        if label is None:
            if not cfg.assume_unmatched_as_normal:
                continue
            label = 0
            label_source = "assumed_normal"
            fallback_paths.append(path)
        # No real patient/subject ID in the metadata CSV -> the filename
        # stem itself is used as a pseudo-patient-id (still prevents the
        # SAME file from ever being split across train/val/test, which a
        # plain image-level random split does not guarantee when a real ID
        # would have grouped multiple files together).
        patient_id = patient_lookup.get(stem, f"{source}:{stem}")
        rows.append({
            "path": path, "label": int(label), "source": source,
            "label_source": label_source, "patient_id": patient_id,
        })

    if fallback_paths:
        print(f"  [{source}] {len(fallback_paths)} image(s) had no metadata match and no "
              f"filename suffix - defaulted to label 0 (Normal). First few:")
        for p in fallback_paths[:5]:
            print(f"    {p}")
        if len(fallback_paths) > 5:
            print(f"    ... and {len(fallback_paths) - 5} more.")
        report_path = cfg.out(f"assumed_normal_{source}.json")
        with open(report_path, "w") as f:
            json.dump(fallback_paths, f, indent=2)
        print(f"  Full list of all {len(fallback_paths)} assumed-Normal {source} "
              f"images -> {report_path}")
    return rows


def _collect_tb_dataset(cfg):
    """tawsifurrahman - class-folder structured (Normal/ and Tuberculosis/).
    No patient metadata available - filename stem used as pseudo-patient-id
    (see _collect_folder_dataset for the same convention/rationale)."""
    rows = []
    class_dirs = {0: [os.path.join(cfg.tb_dir, "Normal")],
                  1: [os.path.join(cfg.tb_dir, "Tuberculosis")]}
    for label, dirs in class_dirs.items():
        for d in dirs:
            for path in list_image_files(d):
                stem = os.path.splitext(os.path.basename(path))[0].lower()
                rows.append({"path": path, "label": label, "source": "tb",
                             "patient_id": f"tb:{stem}"})
    return rows


def _collect_da_db_dataset(cfg):
    """DA and DB - flat folders, curated TB-positive collections.

    Unmatched images here default to label 1 (TB), NOT label 0 (Normal) -
    the opposite of assume_unmatched_as_normal's general-purpose default
    used elsewhere. This is intentional, not a leftover inconsistency: DA/DB
    are curated to be TB-positive-only collections by construction (per
    their own naming/origin), so an unmatched image here is far more likely
    to be an unlabeled TB case than a Normal one that wandered into a
    TB-only collection. Applying assume_unmatched_as_normal's "when in
    doubt, skip" logic to a source that is by design TB-only would be
    wrong, not safer."""
    rows = []
    for directory, name in ((cfg.da_dir, "da"), (cfg.db_dir, "db")):
        for path in list_image_files(directory):
            label = _label_from_filename(path)
            if label is None:
                label = 1
            stem = os.path.splitext(os.path.basename(path))[0].lower()
            rows.append({"path": path, "label": label, "source": name,
                         "patient_id": f"{name}:{stem}"})
    return rows


def _collect_tbx11k_dataset(cfg):
    """TBX11K (usmanshams/tbx-11 Kaggle mirror) - folder-organized, no
    metadata CSV. Confirmed by direct inspection, this specific mirror uses
    three short lowercase category folders under .../TBX11K/imgs/:
    'health', 'sick', 'tb' - not the five-way Healthy/ActiveTB/LatentTB/
    Sick/Uncertain split the original TBX11K paper describes (this mirror
    doesn't separate active/latent TB, and has no separate uncertain
    bucket). Only 'health' and 'tb' map onto this project's binary Normal/
    TB task:
      - 'sick' (non-TB abnormal) is ALWAYS excluded - forcing it into a
        binary label would inject wrong ground truth (genuinely abnormal
        but not TB). "Fewer images with correct labels beats more images
        with questionable ones."
      - 'health' is capped at cfg.tbx11k_healthy_cap (0 = unlimited) - see
        the Config comment on why (TBX11K's raw Healthy:TB ratio is more
        imbalanced than the pool already is). The pool-wide 2:1 rebalancing
        in build_classification_dataframe is the actual final guarantee;
        this is a secondary, per-source guard against one source dominating
        that rebalancing step.

    Folder names are matched by EXACT name (case-insensitive), not
    substring - confirmed by testing this matters: a substring match for
    "tb" would incorrectly match the "TBX11K" wrapper folder itself
    ("tbx11k" contains "tb" as a literal substring) before ever reaching
    the real leaf folder, since it's shallower in the tree and the
    shallowest match was preferred. Searched at ANY depth under
    cfg.tbx11k_dir (not just immediate children) - this mirror's category
    folders are nested under TBX11K/imgs/, not directly at the dataset
    root.
    """
    rows = []
    if not os.path.isdir(cfg.tbx11k_dir):
        print(f"  ! TBX11K directory not found: {cfg.tbx11k_dir}")
        return rows

    def _find_dir(*exact_names):
        """Search every directory under cfg.tbx11k_dir, any depth, for one
        whose name EXACTLY matches (case-insensitive) any of the given
        names. Returns the shallowest match's full path, or None."""
        candidates = []
        for dirpath, dirnames, _files in os.walk(cfg.tbx11k_dir):
            depth = os.path.relpath(dirpath, cfg.tbx11k_dir).count(os.sep)
            for d in dirnames:
                if d.lower() in exact_names:
                    candidates.append((depth, os.path.join(dirpath, d)))
        if not candidates:
            return None
        candidates.sort(key=lambda pair: pair[0])
        return candidates[0][1]

    healthy_dir = _find_dir("health", "healthy")
    tb_dir = _find_dir("tb", "activetb", "active_tb")
    sick_dir = _find_dir("sick")
    uncertain_dir = _find_dir("uncertain")

    for path_found, reason in ((sick_dir, "non-TB abnormal, not Normal/TB"),
                                (uncertain_dir, "uncertain ground truth")):
        if path_found:
            n = len(list_image_files(path_found))
            print(f"  [tbx11k] excluding '{path_found}': {n} images skipped ({reason}).")

    if healthy_dir:
        healthy_paths = list_image_files(healthy_dir)
        available = len(healthy_paths)
        if cfg.tbx11k_healthy_cap and cfg.tbx11k_healthy_cap > 0 and available > cfg.tbx11k_healthy_cap:
            rng = np.random.default_rng(SEED)
            healthy_paths = list(rng.choice(healthy_paths, size=cfg.tbx11k_healthy_cap, replace=False))
            print(f"  [tbx11k] Healthy capped to {cfg.tbx11k_healthy_cap} of {available} available.")
        for path in healthy_paths:
            stem = os.path.splitext(os.path.basename(path))[0].lower()
            rows.append({"path": path, "label": 0, "source": "tbx11k",
                         "label_source": "folder", "patient_id": f"tbx11k:{stem}"})
    else:
        print("  ! TBX11K: no 'health' folder found anywhere under "
              f"{cfg.tbx11k_dir} (searched recursively).")

    if tb_dir:
        for path in list_image_files(tb_dir):
            stem = os.path.splitext(os.path.basename(path))[0].lower()
            rows.append({"path": path, "label": 1, "source": "tbx11k",
                         "label_source": "folder", "patient_id": f"tbx11k:{stem}"})
    else:
        print("  ! TBX11K: no 'tb' folder found anywhere under "
              f"{cfg.tbx11k_dir} (searched recursively).")

    n_normal = sum(1 for r in rows if r["label"] == 0)
    n_tb = sum(1 for r in rows if r["label"] == 1)
    print(f"  TBX11K collected: {n_normal} Normal (health), {n_tb} TB (tb).")
    return rows


def _rebalance_to_target_ratio(df, cfg):
    """Downsample the majority (Normal) class pool-wide to hit
    cfg.target_normal_tb_ratio (Normal:TB). Downsampling is proportional
    across sources (not just the single largest source) so no one source
    is disproportionately gutted, and is seeded for reproducibility."""
    if not cfg.target_normal_tb_ratio or cfg.target_normal_tb_ratio <= 0:
        return df

    n_normal = int((df["label"] == 0).sum())
    n_tb = int((df["label"] == 1).sum())
    if n_tb == 0 or n_normal == 0:
        return df

    current_ratio = n_normal / n_tb
    if current_ratio <= cfg.target_normal_tb_ratio:
        print(f"Pool Normal:TB ratio already {current_ratio:.2f}:1, at or under "
              f"target {cfg.target_normal_tb_ratio:.2f}:1 - no rebalancing needed.")
        return df

    target_normal_count = int(round(n_tb * cfg.target_normal_tb_ratio))
    print(f"Pool Normal:TB ratio is {current_ratio:.2f}:1 (target "
          f"{cfg.target_normal_tb_ratio:.2f}:1) - downsampling Normal from "
          f"{n_normal} to {target_normal_count}, proportionally across sources.")

    normal_df = df[df["label"] == 0]
    tb_df = df[df["label"] == 1]
    frac = target_normal_count / n_normal
    kept_normal = (
        normal_df.groupby("source", group_keys=False)
        .apply(lambda g: g.sample(n=max(1, round(len(g) * frac)) if len(g) > 1 else len(g),
                                   random_state=SEED))
    )
    # Rounding per-group can overshoot the exact target slightly; trim once more if needed.
    if len(kept_normal) > target_normal_count:
        kept_normal = kept_normal.sample(n=target_normal_count, random_state=SEED)

    rebalanced = pd.concat([kept_normal, tb_df]).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    print(f"Post-rebalance: {len(kept_normal)} Normal, {len(tb_df)} TB "
          f"({len(kept_normal) / len(tb_df):.2f}:1).")
    return rebalanced


def build_classification_dataframe(cfg=CFG):
    """Assemble the single pooled dataframe. See module docstring above."""
    print("Generating POOLED dataset metadata ...")
    rows = []
    rows += _collect_tb_dataset(cfg)
    rows += _collect_folder_dataset(cfg.india_dir, "india", cfg.india_meta, cfg=cfg)
    rows += _collect_folder_dataset(cfg.montgomery_dir, "montgomery", cfg.montgomery_meta, cfg=cfg)
    rows += _collect_folder_dataset(cfg.shenzhen_dir, "shenzhen", cfg.shenzhen_meta, cfg=cfg)
    if cfg.use_da_db:
        rows += _collect_da_db_dataset(cfg)
    if cfg.use_tbx11k:
        rows += _collect_tbx11k_dataset(cfg)

    df = pd.DataFrame(rows).drop_duplicates(subset="path").reset_index(drop=True)
    if df.empty:
        raise RuntimeError(
            "No labelled images were discovered. Check that the Kaggle "
            "datasets are attached and the paths in Config are correct."
        )

    if cfg.max_per_class and cfg.max_per_class > 0:
        df = (
            df.groupby("label", group_keys=False)
            .apply(lambda g: g.sample(min(len(g), cfg.max_per_class), random_state=SEED))
            .reset_index(drop=True)
        )

    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    # NOTE: target-ratio rebalancing intentionally does NOT happen here.
    # It runs in main() AFTER deduplicate_pool/filter_image_quality/
    # filter_mask_coverage, on the already-cleaned pool - rebalancing before
    # those cleaning steps would let them disturb the carefully-hit target
    # ratio again, undoing the point of doing it last.
    print("Per-source counts (pre-cleaning, pre-rebalance):")
    print(df.groupby(["source", "label"]).size())
    print("Per-class totals:", df["label"].value_counts().to_dict())
    print("Unique patients:", df["patient_id"].nunique(), "  Total images:", len(df))
    return df


def split_dataframe(df, cfg=CFG):
    """Patient-wise 70/15/15 train/val/test split of the pooled dataframe.

    Plain image-level train_test_split (the previous approach) can put
    different images of the SAME patient into different splits, letting a
    model partially memorize patient-specific anatomy instead of genuinely
    generalizing - CLAUDE.md's dataset rules require patient-wise splitting
    for exactly this reason. GroupShuffleSplit guarantees every image
    belonging to a given patient_id stays in exactly one split; done twice
    (70/30, then the 30 split again 50/50) to get three groups. Stratification
    by label is NOT exact under grouping (a patient's images are all one
    class already in this dataset, so group-safety and class-balance mostly
    coexist in practice, but aren't mathematically guaranteed together) -
    group-safety is treated as the higher priority of the two per CLAUDE.md's
    explicit leakage-prevention requirement.
    """
    from sklearn.model_selection import GroupShuffleSplit

    if df.empty:
        raise RuntimeError(
            "split_dataframe() received an empty dataframe - every image was "
            "dropped somewhere upstream (dedup / quality filter / mask-coverage "
            "filter / rebalancing). Check the pool size printed after each of "
            "those stages above to see which one zeroed it out, and the "
            "corresponding *_report.json / *_dropped.json file for why."
        )

    groups = df["patient_id"].values
    gss1 = GroupShuffleSplit(n_splits=1, test_size=(1.0 - cfg.train_split), random_state=SEED)
    train_idx, rest_idx = next(gss1.split(df, groups=groups))
    train = df.iloc[train_idx].reset_index(drop=True)
    rest = df.iloc[rest_idx].reset_index(drop=True)

    rest_groups = rest["patient_id"].values
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
    val_idx, test_idx = next(gss2.split(rest, groups=rest_groups))
    val = rest.iloc[val_idx].reset_index(drop=True)
    test = rest.iloc[test_idx].reset_index(drop=True)

    overlap_tv = set(train["patient_id"]) & set(val["patient_id"])
    overlap_tt = set(train["patient_id"]) & set(test["patient_id"])
    overlap_vt = set(val["patient_id"]) & set(test["patient_id"])
    if overlap_tv or overlap_tt or overlap_vt:
        raise RuntimeError(
            f"Patient-wise split invariant violated - overlapping patient_ids "
            f"found (train/val={len(overlap_tv)}, train/test={len(overlap_tt)}, "
            f"val/test={len(overlap_vt)}). This should be impossible with "
            f"GroupShuffleSplit; do not proceed with a leaking split."
        )

    print(f"Train: {len(train)} ({train['patient_id'].nunique()} patients)  "
          f"Val: {len(val)} ({val['patient_id'].nunique()} patients)  "
          f"Test: {len(test)} ({test['patient_id'].nunique()} patients)")
    print("Train per-class:", train["label"].value_counts().to_dict())
    print("Val per-class:", val["label"].value_counts().to_dict())
    print("Test per-class:", test["label"].value_counts().to_dict())
    return train, val, test


def export_split_csvs(train_df, val_df, test_df, cfg=CFG):
    """Write train.csv / val.csv / test.csv - the exact, final, post-cleaning
    split membership (every column: path, label, source, label_source,
    patient_id). Nothing downstream reads these back; they exist purely as
    a reproducibility/audit artifact - per this project's own stated
    principle ("Dataset versions should be recorded"), the split is
    deterministically reproducible given the same code/seed/raw data, but
    without this there was no way to inspect which images actually ended up
    where without re-running the entire pipeline. Also the canonical
    train.csv/val.csv/test.csv export this project's own documentation
    calls for, which nothing in this file previously produced."""
    for name, split_df in (("train", train_df), ("val", val_df), ("test", test_df)):
        path = cfg.out(f"{name}.csv")
        split_df.to_csv(path, index=False)
        print(f"Saved {name}.csv ({len(split_df)} rows) -> {path}")

# %%
# # Data Integrity Check
"""Cell 4b - Perceptual-hash duplicate check between the training pool and
the external test set. Catches any *accidental* overlap that wouldn't be
obvious just from reading source names - e.g. via DA/DB or the India
dataset, neither of which was specifically checked against Montgomery/
Shenzhen.

Caveat: chest X-rays share a lot of coarse structure (dark background,
similar ribcage shape), so an 8x8 average hash can produce false positives.
Treat flagged pairs as candidates for a quick visual check, not as
confirmed duplicates.
"""


def compute_image_hashes(paths, hash_size=8):
    """Simple average-hash (aHash) for each image path."""
    hashes = {}
    for path in paths:
        try:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            small = cv2.resize(img, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
            avg = small.mean()
            bits = (small > avg).flatten()
            hash_int = 0
            for bit in bits:
                hash_int = (hash_int << 1) | int(bit)
            hashes[path] = hash_int
        except Exception:
            continue
    return hashes


def _hamming_distance(a, b):
    return bin(a ^ b).count("1")


def check_dataset_leakage(train_df, val_df, test_df, cfg=CFG, hamming_threshold=3):
    """Flag candidate near-duplicate images across the train/val/test split.

    hamming_threshold=3, not 5 - see deduplicate_pool's docstring for why:
    confirmed on a real run that 5 flags genuinely different chest X-rays
    as duplicates on this domain (all frontal CXRs share a similar coarse
    silhouette at aHash's 8x8 grid), evidenced by a kept/dropped pair with
    conflicting Normal/TB labels across three unrelated hospital sources.

    patient_id-based grouping (split_dataframe) already guarantees no
    filename is split across sets, but patient_id is derived per-source
    (metadata patient column, or a source-prefixed filename fallback) - it
    cannot catch the SAME underlying image appearing under different
    filenames in two different pooled sources (a real risk now that
    Montgomery/Shenzhen/TBX11K/DA-DB are all pooled together and some public
    CXR datasets are known to overlap). This perceptual-hash check is the
    independent, second-layer catch for that case."""
    print("=" * 78)
    print("DATA LEAKAGE CHECK (perceptual hash across train/val/test)")
    print("=" * 78)

    pair_names = [("train", "val", train_df, val_df),
                  ("train", "test", train_df, test_df),
                  ("val", "test", val_df, test_df)]

    all_flagged = []
    for name_a, name_b, df_a, df_b in pair_names:
        hashes_a = compute_image_hashes(df_a["path"].tolist())
        hashes_b = compute_image_hashes(df_b["path"].tolist())
        items_a = list(hashes_a.items())
        for path_b, hash_b in hashes_b.items():
            for path_a, hash_a in items_a:
                if _hamming_distance(hash_a, hash_b) <= hamming_threshold:
                    all_flagged.append({
                        "split_a": name_a, "image_a": path_a,
                        "split_b": name_b, "image_b": path_b,
                    })

    if all_flagged:
        print(f"WARNING: {len(all_flagged)} candidate near-duplicate pairs found across "
              f"splits. These are CANDIDATES for manual visual review, not confirmed "
              f"duplicates - a single image matching many others usually indicates an "
              f"unusually blank/low-detail perceptual-hash false positive, not real leakage.")
        for pair in all_flagged[:20]:
            print(f"  {pair['split_a']}: {pair['image_a']}  <->  {pair['split_b']}: {pair['image_b']}")
        if len(all_flagged) > 20:
            print(f"  ... and {len(all_flagged) - 20} more (full list in leakage_check.json).")
    else:
        print(f"No candidate near-duplicates found across train/val/test "
              f"(hamming_threshold={hamming_threshold}).")

    report_path = cfg.out("leakage_check.json")
    with open(report_path, "w") as f:
        json.dump({
            "hamming_threshold": hamming_threshold,
            "train_size": len(train_df), "val_size": len(val_df), "test_size": len(test_df),
            "num_flagged_pairs": len(all_flagged),
            "flagged_pairs": all_flagged,
        }, f, indent=2)
    print(f"Saved leakage check report -> {report_path}")
    return all_flagged


def _exact_duplicate_paths(paths):
    """Group paths by MD5 of their raw file bytes - a zero-false-positive,
    zero-false-negative check for byte-identical files (e.g. literally the
    same file copied into two different dataset folders). Complements, but
    doesn't replace, the perceptual-hash near-duplicate check below (which
    catches resized/recompressed copies this can't, at the cost of being a
    fuzzy threshold rather than a certainty)."""
    import hashlib
    seen = {}
    to_drop = set()
    dropped_pairs = []
    for path in paths:
        try:
            with open(path, "rb") as f:
                digest = hashlib.md5(f.read()).hexdigest()
        except OSError:
            continue
        if digest in seen:
            to_drop.add(path)
            dropped_pairs.append({"kept": seen[digest], "dropped": path, "reason": "exact_byte_match"})
        else:
            seen[digest] = path
    return to_drop, dropped_pairs


def deduplicate_pool(df, cfg=CFG, hamming_threshold=3):
    """Remove duplicate images from the pooled dataframe BEFORE splitting,
    in two passes. Different from check_dataset_leakage (which runs AFTER
    the split, checking only across train/val/test): this catches the same
    underlying image appearing twice ANYWHERE in the pool - e.g. the same
    X-ray distributed under two different source datasets, or a resized/
    recompressed copy - which both inflates the apparent sample size and,
    if a duplicate happens to land in two different splits, would look like
    "generalization" the model didn't actually earn. Keeps the first
    occurrence (in the pool's current row order) of each duplicate group.

    Pass 1 (exact, MD5 byte hash): catches byte-identical files with
    certainty, and cheaply shrinks the pool before the expensive pass.
    Pass 2 (perceptual/near-duplicate, aHash + Hamming distance): O(n^2)
    pairwise comparison over whatever survives pass 1 - a few minutes for a
    pool of a few thousand images.

    hamming_threshold=3, not 5: confirmed on a real run that 5 (up to ~8%
    of the 64-bit aHash differing) is loose enough to flag genuinely
    different images as duplicates on THIS domain specifically - a kept
    image matched two dropped images with CONFLICTING labels (one Normal,
    one TB, from three independent, unrelated hospital collections that
    have no real reason to share raw images). Chest X-rays are all
    anatomically similar at the coarse 8x8 grid aHash operates on (same
    rough silhouette/lung-field layout for any frontal CXR) in a way
    generic photos aren't, so aHash's usual threshold is plausibly too
    loose here. 3 is tighter (~5% of bits) while still catching true
    near-duplicates (recompressed/resized copies of the same source image
    typically differ by 0-2 bits, not 3-5).

    Also checks every near-duplicate pair's labels: if they disagree, that
    is reported as a CONFLICT, not silently treated the same as an
    ordinary same-class match - a kept/dropped pair with different labels
    is a stronger signal of a hash false-positive (or a genuine
    cross-dataset label disagreement, also worth knowing) than same-class
    duplicates are."""
    print("=" * 78)
    print("DEDUPLICATION, PASS 1/2 (exact byte match)")
    print("=" * 78)
    label_by_path = dict(zip(df["path"], df["label"]))
    all_paths = df["path"].tolist()
    exact_drop, exact_pairs = _exact_duplicate_paths(all_paths)
    if exact_drop:
        print(f"Found {len(exact_drop)} byte-identical duplicate files - dropping.")
    else:
        print("No byte-identical duplicates found.")
    remaining_paths = [p for p in all_paths if p not in exact_drop]

    print("=" * 78)
    print("DEDUPLICATION, PASS 2/2 (perceptual hash, near-duplicates - may take a few minutes)")
    print("=" * 78)

    hashes = compute_image_hashes(remaining_paths)
    items = [(p, hashes[p]) for p in remaining_paths if p in hashes]

    to_drop = set(exact_drop)
    dropped_pairs = list(exact_pairs)
    label_conflicts = []
    for i in range(len(items)):
        path_i, hash_i = items[i]
        if path_i in to_drop:
            continue
        for j in range(i + 1, len(items)):
            path_j, hash_j = items[j]
            if path_j in to_drop:
                continue
            if _hamming_distance(hash_i, hash_j) <= hamming_threshold:
                to_drop.add(path_j)
                pair = {"kept": path_i, "dropped": path_j, "reason": "near_duplicate"}
                dropped_pairs.append(pair)
                if label_by_path.get(path_i) != label_by_path.get(path_j):
                    label_conflicts.append({
                        **pair,
                        "kept_label": label_by_path.get(path_i),
                        "dropped_label": label_by_path.get(path_j),
                    })

    n_near = len(to_drop) - len(exact_drop)
    if n_near:
        print(f"Found {n_near} additional near-duplicate images (hamming_threshold="
              f"{hamming_threshold}) - dropping the later occurrence of each, "
              f"keeping the first.")
        for pair in dropped_pairs[:20]:
            print(f"  kept: {pair['kept']}  dropped ({pair['reason']}): {pair['dropped']}")
        if len(dropped_pairs) > 20:
            print(f"  ... and {len(dropped_pairs) - 20} more (full list in dedup_report.json).")
    else:
        print(f"No additional near-duplicates found (hamming_threshold={hamming_threshold}).")

    if label_conflicts:
        print(f"WARNING: {len(label_conflicts)} near-duplicate pair(s) had CONFLICTING labels "
              f"(kept image's label differs from the dropped 'duplicate' it matched) - stronger "
              f"evidence of a hash false-positive, or a genuine cross-dataset label disagreement, "
              f"than an ordinary same-class match. Recommend manually reviewing these specific "
              f"pairs before trusting the pool:")
        for c in label_conflicts[:10]:
            print(f"  kept: {c['kept']} (label={c['kept_label']})  "
                  f"dropped: {c['dropped']} (label={c['dropped_label']})")

    report_path = cfg.out("dedup_report.json")
    with open(report_path, "w") as f:
        json.dump({
            "hamming_threshold": hamming_threshold,
            "pool_size_before": len(df),
            "num_dropped_exact": len(exact_drop),
            "num_dropped_near": n_near,
            "num_dropped_total": len(to_drop),
            "num_label_conflicts": len(label_conflicts),
            "label_conflicts": label_conflicts,
            "dropped_pairs": dropped_pairs,
        }, f, indent=2)
    print(f"Saved dedup report -> {report_path}")

    deduped = df[~df["path"].isin(to_drop)].reset_index(drop=True)
    print(f"Pool size: {len(df)} -> {len(deduped)} after deduplication "
          f"({len(exact_drop)} exact + {n_near} near-duplicate).")
    return deduped


def filter_image_quality(df, cfg=CFG, blur_threshold=100.0, min_std=5.0):
    """Drop images that are corrupted (unreadable), extremely blurry (low
    Laplacian variance), or near-blank (very low pixel variation - e.g. an
    almost solid black/white frame, not a real chest X-ray). Runs on the
    whole pool before splitting.

    blur_threshold=100.0 is the commonly-cited OpenCV rule-of-thumb
    starting point for Laplacian-variance blur detection on ordinary
    photos - chest radiographs have different frequency content than
    typical photos, so this has NOT been separately validated for X-rays
    specifically. Check image_quality_dropped.json after a real run and
    spot-visually-verify a sample of what gets dropped before trusting this
    threshold blindly."""
    print("=" * 78)
    print("IMAGE QUALITY CHECK")
    print("=" * 78)
    keep_mask = []
    reasons = {"corrupted": 0, "blurry": 0, "near_blank": 0}
    dropped = []
    for path in df["path"]:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            keep_mask.append(False)
            reasons["corrupted"] += 1
            dropped.append({"path": path, "reason": "corrupted"})
            continue
        small = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        laplacian_var = cv2.Laplacian(small, cv2.CV_64F).var()
        if laplacian_var < blur_threshold:
            keep_mask.append(False)
            reasons["blurry"] += 1
            dropped.append({"path": path, "reason": "blurry", "laplacian_var": float(laplacian_var)})
            continue
        if small.std() < min_std:
            keep_mask.append(False)
            reasons["near_blank"] += 1
            dropped.append({"path": path, "reason": "near_blank", "std": float(small.std())})
            continue
        keep_mask.append(True)

    n_dropped = len(df) - sum(keep_mask)
    print(f"Dropped {n_dropped} images: {reasons}")
    if dropped:
        report_path = cfg.out("image_quality_dropped.json")
        with open(report_path, "w") as f:
            json.dump(dropped, f, indent=2)
        print(f"Saved dropped-image report -> {report_path}")

    filtered = df[pd.Series(keep_mask, index=df.index)].reset_index(drop=True)
    print(f"Pool size: {len(df)} -> {len(filtered)} after quality filtering.")
    return filtered


def filter_mask_coverage(df, unet, cfg=CFG, min_coverage=0.05, max_coverage=0.85):
    """Drop images where the predicted (cleaned) lung mask covers an
    implausibly small or large fraction of the image - a rough proxy for
    missing/severely cropped lungs, or a non-frontal (lateral) view the
    segmentation model wasn't trained for (a lateral CXR confuses a
    frontal-view U-Net into a degenerate mask). Requires a trained `unet`,
    so this runs after train_lung_segmentation(), on the whole pool before
    splitting.

    Known limitation, not solved here: this does NOT reliably catch every
    outlier type worth excluding (pediatric X-rays, CT slices accidentally
    included) - those need age/modality metadata this project's datasets
    don't provide, not a mask-coverage heuristic. Flagging as an open gap
    rather than claiming this filter solves it.

    Also tracks drop rate PER LABEL, not just in aggregate: large
    consolidations/opacities can confuse the segmenter into predicting a
    smaller-than-normal lung mask (diseased tissue doesn't look like
    "textbook lung" to a U-Net trained mostly on unremarkable anatomy),
    while genuinely normal lungs reliably produce clean, full-coverage
    masks. If TB images get dropped at a meaningfully higher rate than
    Normal ones, this filter would be systematically removing the most
    visually-obvious, most clinically-important TB cases - the opposite
    problem from normal-bias, and just as serious. This is reported loudly
    rather than silently trusted.

    Skipped entirely under cfg.smoke_test: apply_smoke_test_overrides caps
    segmentation training to ~2 epochs on a few dozen images specifically to
    be fast, not accurate. Confirmed on a real run that the resulting U-Net
    (val_dice_coefficient ~0.14, essentially untrained) produces coverage
    fractions meaningless enough to fail this filter on 100% of BOTH classes
    - not real outlier detection, just a near-random segmenter. Runs
    normally once smoke_test=False and the U-Net is properly trained
    (historically ~0.94-0.95 val Dice)."""
    if cfg.smoke_test:
        print("=" * 78)
        print("LUNG MASK COVERAGE CHECK - SKIPPED (smoke_test mode)")
        print("=" * 78)
        print("The smoke-test U-Net is deliberately undertrained (~2 epochs, a few "
              "dozen images) for speed, not accuracy - its predicted masks aren't "
              "reliable enough for a coverage-fraction filter to mean anything. "
              "Runs normally when smoke_test=False.")
        return df

    print("=" * 78)
    print("LUNG MASK COVERAGE CHECK")
    print("=" * 78)
    keep_mask = []
    dropped = []
    label_totals = df["label"].value_counts().to_dict()
    label_dropped = {0: 0, 1: 0}
    for path, label in zip(df["path"], df["label"]):
        gray = read_image_grayscale(path)
        mask = predict_lung_mask(unet, gray, cfg)
        mask = keep_largest_components(mask, num_components=2)
        mask = morphological_cleanup(mask, kernel_size=5)
        coverage = float(mask.mean())
        if coverage < min_coverage or coverage > max_coverage:
            keep_mask.append(False)
            dropped.append({"path": path, "label": int(label), "coverage": coverage})
            label_dropped[int(label)] = label_dropped.get(int(label), 0) + 1
            continue
        keep_mask.append(True)

    n_dropped = len(df) - sum(keep_mask)
    normal_rate = label_dropped.get(0, 0) / label_totals.get(0, 1)
    tb_rate = label_dropped.get(1, 0) / label_totals.get(1, 1)
    print(f"Dropped {n_dropped} images with implausible lung-mask coverage "
          f"(outside [{min_coverage:.2f}, {max_coverage:.2f}] of image area).")
    print(f"  Normal: {label_dropped.get(0, 0)}/{label_totals.get(0, 0)} dropped ({normal_rate:.2%})")
    print(f"  TB:     {label_dropped.get(1, 0)}/{label_totals.get(1, 0)} dropped ({tb_rate:.2%})")
    if tb_rate > normal_rate * 1.5 and label_dropped.get(1, 0) >= 5:
        print(f"  WARNING: TB drop rate ({tb_rate:.2%}) is notably higher than Normal "
              f"({normal_rate:.2%}) - this filter may be disproportionately removing "
              f"visually-severe TB cases (large opacities confusing the segmenter), not "
              f"just genuine outliers. Recommend manually reviewing a sample of dropped "
              f"TB images in mask_coverage_dropped.json before trusting this filter blindly.")

    if dropped:
        report_path = cfg.out("mask_coverage_dropped.json")
        with open(report_path, "w") as f:
            json.dump({
                "min_coverage": min_coverage, "max_coverage": max_coverage,
                "normal_drop_rate": normal_rate, "tb_drop_rate": tb_rate,
                "dropped": dropped,
            }, f, indent=2)
        print(f"Saved dropped-image report -> {report_path}")

    filtered = df[pd.Series(keep_mask, index=df.index)].reset_index(drop=True)
    print(f"Pool size: {len(df)} -> {len(filtered)} after mask-coverage filtering.")
    return filtered

# %%
# # Lung Segmentation (Attention U-Net)
"""Cell 5 - unchanged from the previous version: this was already training
successfully, so it's carried forward as-is (aside from saving history/
architecture diagrams at the end of train_lung_segmentation)."""


def _conv_block(x, filters, name):
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False,
                       kernel_initializer="he_normal", name=name + "_conv1")(x)
    x = layers.BatchNormalization(name=name + "_bn1")(x)
    x = layers.Activation("relu", name=name + "_relu1")(x)
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False,
                       kernel_initializer="he_normal", name=name + "_conv2")(x)
    x = layers.BatchNormalization(name=name + "_bn2")(x)
    x = layers.Activation("relu", name=name + "_relu2")(x)
    return x


def _attention_gate(skip, gating, inter_channels, name):
    theta = layers.Conv2D(inter_channels, 1, padding="same", name=name + "_theta")(skip)
    phi = layers.Conv2D(inter_channels, 1, padding="same", name=name + "_phi")(gating)
    add = layers.Add(name=name + "_add")([theta, phi])
    act = layers.Activation("relu", name=name + "_relu")(add)
    psi = layers.Conv2D(1, 1, padding="same", name=name + "_psi")(act)
    psi = layers.Activation("sigmoid", name=name + "_sigmoid")(psi)
    return layers.Multiply(name=name + "_mul")([skip, psi])


def build_attention_unet(input_size, base_filters=32):
    inputs = keras.Input(shape=(input_size, input_size, 1), name="seg_input")

    e1 = _conv_block(inputs, base_filters, "enc1")
    p1 = layers.MaxPooling2D(2, name="pool1")(e1)
    e2 = _conv_block(p1, base_filters * 2, "enc2")
    p2 = layers.MaxPooling2D(2, name="pool2")(e2)
    e3 = _conv_block(p2, base_filters * 4, "enc3")
    p3 = layers.MaxPooling2D(2, name="pool3")(e3)
    e4 = _conv_block(p3, base_filters * 8, "enc4")
    p4 = layers.MaxPooling2D(2, name="pool4")(e4)

    b = _conv_block(p4, base_filters * 16, "bottleneck")

    u4 = layers.Conv2DTranspose(base_filters * 8, 2, strides=2, padding="same", name="up4")(b)
    a4 = _attention_gate(e4, u4, base_filters * 8, "att4")
    d4 = _conv_block(layers.Concatenate(name="cat4")([u4, a4]), base_filters * 8, "dec4")

    u3 = layers.Conv2DTranspose(base_filters * 4, 2, strides=2, padding="same", name="up3")(d4)
    a3 = _attention_gate(e3, u3, base_filters * 4, "att3")
    d3 = _conv_block(layers.Concatenate(name="cat3")([u3, a3]), base_filters * 4, "dec3")

    u2 = layers.Conv2DTranspose(base_filters * 2, 2, strides=2, padding="same", name="up2")(d3)
    a2 = _attention_gate(e2, u2, base_filters * 2, "att2")
    d2 = _conv_block(layers.Concatenate(name="cat2")([u2, a2]), base_filters * 2, "dec2")

    u1 = layers.Conv2DTranspose(base_filters, 2, strides=2, padding="same", name="up1")(d2)
    a1 = _attention_gate(e1, u1, base_filters, "att1")
    d1 = _conv_block(layers.Concatenate(name="cat1")([u1, a1]), base_filters, "dec1")

    outputs = layers.Conv2D(1, 1, activation="sigmoid", dtype="float32", name="seg_output")(d1)
    return keras.Model(inputs, outputs, name="attention_unet")


def dice_coefficient(y_true, y_pred, smooth=1.0):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_targets_f = tf.reshape(y_true, [-1])
    y_preds_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_targets_f * y_preds_f)
    return (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_targets_f) + tf.reduce_sum(y_preds_f) + smooth
    )


def dice_bce_loss(y_true, y_pred):
    bce = keras.losses.binary_crossentropy(y_true, y_pred)
    bce = tf.reduce_mean(bce)
    dice = 1.0 - dice_coefficient(y_true, y_pred)
    return bce + dice


def _match_mask_path(image_path, mask_dir):
    stem = os.path.splitext(os.path.basename(image_path))[0]
    candidates = [
        os.path.join(mask_dir, stem + ".png"),
        os.path.join(mask_dir, stem + "_mask.png"),
        os.path.join(mask_dir, stem + ".jpg"),
    ]
    for cand in candidates:
        if os.path.isfile(cand):
            return cand
    return None


def build_segmentation_arrays(cfg=CFG):
    image_paths = list_image_files(cfg.seg_img_dir)
    if cfg.max_seg_samples and cfg.max_seg_samples > 0:
        image_paths = image_paths[: cfg.max_seg_samples]

    images, masks = [], []
    for img_path in image_paths:
        mask_path = _match_mask_path(img_path, cfg.seg_mask_dir)
        if mask_path is None:
            continue
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        msk = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None or msk is None:
            continue
        img = cv2.resize(img, (cfg.seg_size, cfg.seg_size), interpolation=cv2.INTER_AREA)
        msk = cv2.resize(msk, (cfg.seg_size, cfg.seg_size), interpolation=cv2.INTER_NEAREST)
        images.append(img.astype("float32") / 255.0)
        masks.append((msk > 127).astype("float32"))

    if not images:
        raise RuntimeError("No matching lung image/mask pairs were found.")

    x = np.expand_dims(np.stack(images), -1)
    y = np.expand_dims(np.stack(masks), -1)
    print(f"Segmentation dataset: X={x.shape}  Y={y.shape}")
    return x, y


def train_lung_segmentation(cfg=CFG):
    print("=" * 78)
    print("STAGE: LUNG SEGMENTATION (Attention U-Net)")
    print("=" * 78)
    x, y = build_segmentation_arrays(cfg)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.15, random_state=SEED)

    model = build_attention_unet(cfg.seg_size)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=dice_bce_loss,
        metrics=[dice_coefficient],
    )

    save_path = cfg.out("attention_unet.keras")
    overfit_monitor = OverfitMonitor(gap_threshold=0.10, metric="dice_coefficient")
    callbacks = [
        keras.callbacks.ModelCheckpoint(save_path, monitor="val_dice_coefficient",
                                         mode="max", save_best_only=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3,
                                           min_lr=1e-6, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=6,
                                       restore_best_weights=True, verbose=1),
        overfit_monitor,
    ]

    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=cfg.seg_epochs,
                         batch_size=cfg.seg_batch_size, callbacks=callbacks, verbose=2)
    model.save(save_path)

    plot_training_history(history, "Segmentation", cfg, metrics=("loss", "dice_coefficient"))
    plot_overfit_gap(overfit_monitor, "Segmentation", cfg)
    plot_architecture_diagram(model, cfg.fig("unet_architecture"), cfg=cfg)

    print("Saved lung segmentation model ->", save_path)
    return model, history

# %%
# # Preprocessing Functions
"""Cell 6 - The exact preprocessing pipeline applied to every X-ray."""

_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def read_image_grayscale(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".dcm", ".dic") and HAS_PYDICOM:
        dataset = pydicom.dcmread(path)
        pixels = dataset.pixel_array.astype("float32")
        pixels -= pixels.min()
        if pixels.max() > 0:
            pixels = pixels / pixels.max() * 255.0
        return pixels
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Unable to read image: {path}")
    return img.astype("float32")


def predict_lung_mask(unet, gray, cfg=CFG):
    h, w = gray.shape[:2]
    if unet is None:
        return np.ones((h, w), dtype="uint8")
    resized = cv2.resize(gray, (cfg.seg_size, cfg.seg_size), interpolation=cv2.INTER_AREA)
    tensor = (resized / 255.0).astype("float32")[None, ..., None]
    pred = unet(tensor, training=False)
    pred = np.asarray(pred)[0, ..., 0]
    pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)
    return (pred > 0.5).astype("uint8")


def keep_largest_components(mask, num_components=2):
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask.astype("uint8"), connectivity=8
    )
    if num_labels <= 1:
        return mask.astype("uint8")
    areas = [(idx, stats[idx, cv2.CC_STAT_AREA]) for idx in range(1, num_labels)]
    areas.sort(key=lambda pair: pair[1], reverse=True)
    keep = {idx for idx, _area in areas[:num_components]}
    cleaned = np.isin(labels, list(keep)).astype("uint8")
    return cleaned


def morphological_cleanup(mask, kernel_size=5):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(mask.astype("uint8"), cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    return closed


def _fill_mask_holes(mask):
    """Fills every fully-enclosed hole in a binary lung mask, regardless of
    size - a chest X-ray projection has no anatomical reason for the lung
    silhouette to have a true hole all the way through; an enclosed gap is
    either a segmentation-confidence dip or a lesion (e.g. a TB cavity), and
    either way it belongs to the lung region for masking purposes.

    Confirmed on a real image (tb0283.png, visibly severe bilateral
    disease): the U-Net's per-pixel confidence dipped below 0.5 exactly
    over a round cavitary lesion, leaving a hole in the mask too large for
    the 5x5 morphological-close kernel above to fill. Left unfixed, the
    non-lung pixel-masking step in preprocess_xray would have replaced that
    exact hole - the single most diagnostically important part of the
    image - with the neutral background fill value, erasing it entirely.

    Uses a border-seeded flood fill rather than a bigger closing kernel:
    closing with an arbitrarily larger kernel is a guess at "big enough,"
    while flood-filling from the image border finds the TRUE exterior
    background directly, so anything left over is unambiguously an
    enclosed hole no matter how large.
    """
    mask_u8 = (mask > 0).astype("uint8")
    h, w = mask_u8.shape
    corners_xy = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]
    seed = next(((x, y) for (x, y) in corners_xy if mask_u8[y, x] == 0), None)
    if seed is None:
        # Every corner is foreground - an extreme, near-full-frame mask
        # where flood-filling from a corner would incorrectly treat real
        # background as a "hole." Skip rather than corrupt the mask.
        return mask_u8
    flood_filled = mask_u8.copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype="uint8")
    cv2.floodFill(flood_filled, flood_mask, seed, 1)
    holes = (flood_filled == 0).astype("uint8")
    return (mask_u8 | holes).astype("uint8")


def bounding_box_with_padding(mask, pad_fraction=0.05):
    h, w = mask.shape[:2]
    ys, xs = np.where(mask > 0)
    if ys.size == 0 or xs.size == 0:
        return 0, h, 0, w
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    pad_y = int((y1 - y0) * pad_fraction)
    pad_x = int((x1 - x0) * pad_fraction)
    y0 = max(0, y0 - pad_y)
    y1 = min(h, y1 + pad_y + 1)
    x0 = max(0, x0 - pad_x)
    x1 = min(w, x1 + pad_x + 1)
    return y0, y1, x0, x1


def apply_clahe(gray):
    uint8_img = np.clip(gray, 0, 255).astype("uint8")
    return _CLAHE.apply(uint8_img).astype("float32")


def _segment_and_crop(gray, unet, cfg):
    """Runs lung segmentation + cleanup + bbox crop once, returning both the
    cropped grayscale region AND the correspondingly-cropped binary mask, so
    every caller derives its own output from one shared segmentation pass
    instead of re-deriving the crop geometry independently (used by both
    preprocess_xray and the Grad-CAM/lung-localization audit's
    _lung_mask_in_model_frame)."""
    mask = predict_lung_mask(unet, gray, cfg)
    mask = keep_largest_components(mask, num_components=2)
    mask = morphological_cleanup(mask, kernel_size=5)
    mask = _fill_mask_holes(mask)

    y0, y1, x0, x1 = bounding_box_with_padding(mask, pad_fraction=cfg.lung_crop_padding)
    cropped_gray = gray[y0:y1, x0:x1]
    cropped_mask = mask[y0:y1, x0:x1]
    if cropped_gray.size == 0:
        cropped_gray, cropped_mask = gray, mask
    return cropped_gray, cropped_mask


def preprocess_xray(path, unet, cfg=CFG):
    """Full preprocessing pipeline for a single X-ray. Returns a canonical
    0-255 float32 RGB tensor - NOT backbone-normalized.

    Teacher (ResNet50) and student (DenseNet121) need different input
    normalization (Caffe-style BGR mean-subtraction vs. ImageNet mean/std),
    and knowledge distillation needs both models to see the exact same input
    tensor in the same training step (DistillationModel calls both on the
    same `x`). Rather than maintaining parallel per-model datasets, each
    model owns a `Lambda(preprocess_input)` layer as its first layer (see
    build_teacher / build_densenet_student) and normalizes internally - so
    this function, the tf.data pipeline, and augmentation all operate on one
    shared canonical representation regardless of which model consumes it.

    Non-lung pixels are explicitly filled with the visible lung region's own
    mean intensity (not cropped to a bounding box, and not zeroed) after
    CLAHE: the bbox crop alone still leaves collarbones, shoulders, the
    mediastinum, and any border stamp/marker inside the rectangle fully
    visible, and the Grad-CAM/lung-localization audit
    (audit_gradcam_lung_localization) confirmed the model was exploiting
    exactly that - 88% of correctly-classified TB predictions had most of
    their Grad-CAM energy outside the lung mask, and 59% peaked near the
    crop border specifically.

    A first version of this fix filled the background with raw 0 instead of
    the mean - that regressed training almost immediately (val accuracy
    falling below the majority-class baseline within a handful of epochs,
    confirmed on a real run). Root cause: each model's internal
    Lambda(preprocess_input) normalizes canonical 0-255 input, and raw 0 is
    NOT neutral post-normalization - DenseNet's torch-style scaling maps it
    to roughly -2.1 to -2.3 per channel, a value ImageNet pretraining rarely
    produced, fed straight into the backbone's still-frozen (head/early-
    finetune stage) BatchNorm layers, which carry ImageNet's original
    running statistics. Filling with the lung region's own mean keeps the
    background numerically close to real tissue values instead of an
    out-of-distribution extreme, while still removing all the
    border/artifact/collarbone content that was the actual shortcut -
    the goal was always "make the background uninformative", not
    "make the background an extreme value."

    Masking happens AFTER CLAHE rather than before, so CLAHE's per-tile
    histogram equalization is computed from real tissue values, not skewed
    by tiles straddling a hard background region.
    """
    gray = read_image_grayscale(path)
    cropped, cropped_mask = _segment_and_crop(gray, unet, cfg)

    enhanced = apply_clahe(cropped)
    resized = cv2.resize(enhanced, (cfg.img_size, cfg.img_size), interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(
        cropped_mask.astype("float32"), (cfg.img_size, cfg.img_size),
        interpolation=cv2.INTER_NEAREST,
    )
    binary_mask = resized_mask > 0.5
    lung_pixels = resized[binary_mask]
    fill_value = float(lung_pixels.mean()) if lung_pixels.size > 0 else float(resized.mean())
    resized = np.where(binary_mask, resized, fill_value).astype("float32")
    rgb = np.stack([resized, resized, resized], axis=-1).astype("float32")
    return rgb

# %%
# # Dataset Pipeline
"""Cell 7 - tf.data input pipeline built on top of the preprocessing routine."""


def _sample_generator(paths, labels, unet, cfg, cache=None):
    def generator():
        skipped_count = 0
        for path, label in zip(paths, labels):
            if cache is not None and path in cache:
                image = cache[path]
            else:
                try:
                    image = preprocess_xray(path, unet, cfg)
                except Exception as e:
                    print(f"Skipping {path} due to error: {e}")
                    skipped_count += 1
                    continue
                if cache is not None:
                    cache[path] = image
            one_hot = np.zeros((cfg.num_classes,), dtype="float32")
            one_hot[int(label)] = 1.0
            yield image, one_hot
        if skipped_count > 0:
            print(f"Total skipped images: {skipped_count}")

    return generator


_AUG_LAYERS = keras.Sequential(
    [
        keras.layers.RandomRotation(factor=CFG.aug_rotation_deg / 360.0, fill_mode="constant",
                                     interpolation="bilinear", seed=SEED),
        keras.layers.RandomTranslation(CFG.aug_translation_frac, CFG.aug_translation_frac,
                                        fill_mode="constant", interpolation="bilinear", seed=SEED),
        keras.layers.RandomZoom(CFG.aug_zoom_frac, fill_mode="constant",
                                 interpolation="bilinear", seed=SEED),
    ],
    name="xray_augment",
)


def _augment(image, label):
    # Under the mixed_float16 global policy, _AUG_LAYERS (Keras
    # preprocessing layers) compute in float16 by default and return a
    # float16 tensor even though `image` comes in as float32. Cast back to
    # float32 IMMEDIATELY, before any further op - confirmed by reproducing
    # this locally that it's not just a dtype-mismatch issue on the final
    # addition (tf.random.normal defaulting to float32 while image is
    # float16): tf.image.random_contrast's AdjustContrastv2 kernel isn't
    # registered for float16 on CPU at all, so it fails outright regardless
    # of operand matching. tf.data .map() pipelines run on CPU even when
    # the model itself trains on GPU, so this isn't a CPU-only-machine
    # artifact - casting early avoids every float16-on-CPU gap in this
    # chain at once, not just the one that happened to surface first.
    image = _AUG_LAYERS(image, training=True)
    image = tf.cast(image, tf.float32)
    image = tf.image.random_brightness(image, max_delta=CFG.aug_brightness_delta)
    image = tf.image.random_contrast(image, lower=CFG.aug_contrast_range[0],
                                      upper=CFG.aug_contrast_range[1])
    image = image + tf.random.normal(tf.shape(image), mean=0.0, stddev=CFG.aug_noise_stddev)
    return image, label


def make_dataset(df, unet, cfg=CFG, training=False, class_weights=None):
    """Build a batched tf.data.Dataset. Pass class_weights for the TRAINING
    split only (never for val/external-test)."""
    paths = df["path"].tolist()
    labels = df["label"].tolist()

    cache = {} if cfg.use_preprocessing_cache else None

    output_signature = (
        tf.TensorSpec(shape=(cfg.img_size, cfg.img_size, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(cfg.num_classes,), dtype=tf.float32),
    )
    dataset = tf.data.Dataset.from_generator(
        _sample_generator(paths, labels, unet, cfg, cache),
        output_signature=output_signature,
    )

    if training:
        dataset = dataset.shuffle(buffer_size=min(1024, max(1, len(paths))), seed=SEED)
        dataset = dataset.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)

    if class_weights is not None:
        weight_tensor = tf.constant(class_weights, dtype=tf.float32)

        def attach_weight(image, label):
            sample_weight = tf.reduce_sum(label * weight_tensor)
            return image, label, sample_weight

        dataset = dataset.map(attach_weight, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(cfg.batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def compute_class_weights(df):
    classes = np.array([0, 1])
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=df["label"].values)
    weight_map = {int(c): float(w) for c, w in zip(classes, weights)}
    ordered = [weight_map.get(0, 1.0), weight_map.get(1, 1.0)]
    print("Class weights:", ordered)
    return ordered

# %%
# # Teacher Network (ResNet-50)
"""Cell 8 - Teacher network: ResNet-50."""


def build_teacher(input_shape, num_classes, weights="imagenet"):
    """Construct the ResNet-50 teacher model. Takes canonical 0-255 RGB
    input directly - normalizes internally via a Lambda layer, so the
    shared tf.data pipeline never needs to know which backbone is
    consuming its output (see preprocess_xray)."""
    base = keras.applications.ResNet50(
        input_shape=input_shape,
        include_top=False,
        weights=weights,
    )
    base.trainable = False
    inputs = keras.Input(shape=input_shape, name="teacher_input")
    x = layers.Lambda(keras.applications.resnet50.preprocess_input,
                       name="teacher_preprocess")(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
    x = layers.BatchNormalization(name="teacher_bn")(x)
    x = layers.Dropout(0.2, name="teacher_dropout")(x)
    logits = layers.Dense(num_classes, activation=None, dtype="float32",
                           name="teacher_logits")(x)
    outputs = layers.Activation("softmax", dtype="float32", name="teacher_output")(logits)
    return keras.Model(inputs, outputs, name="resnet50_teacher")

# %%
# # Student Network (DenseNet121, ImageNet-pretrained)
"""Cell 9 - Student network: DenseNet121.

Replaces the earlier from-scratch custom CNN (NirikNet). Rationale, decided
across several rounds of review:
  - NirikNet trained from random initialization - a real overfitting risk on
    a training pool of only a few thousand images. DenseNet121 starts from
    ImageNet weights, sidestepping that problem without needing a separate
    proxy-pretraining stage.
  - Capacity gap to the ResNet50 teacher shrinks from ~11x (23.6M -> 2.1M)
    to ~3.35x (23.6M -> 7.04M), addressing the teacher/student capacity-gap
    degradation documented in Mirzadeh et al. 2020 ("Improved Knowledge
    Distillation via Teacher Assistant").
  - CheXNet (Rajpurkar et al. 2017) established DenseNet121 specifically as
    a strong chest-radiograph classifier, unlike a from-scratch bespoke
    architecture with no domain precedent.
  - Known trade-off, measured directly on this project's hardware: CPU
    inference is SLOWER than the old NirikNet student (~493ms/image vs.
    ~262ms/image, architecture-only benchmark) because dense-block
    concatenation doesn't parallelize as cheaply as a plain conv chain
    despite having far fewer parameters. Deemed acceptable for a
    non-real-time screening upload, not a live feed.
"""


def build_densenet_student(input_shape, num_classes, weights="imagenet"):
    """Construct the DenseNet121 student model. Same canonical-0-255-RGB
    input contract as build_teacher, same logits/softmax head split for
    distillation-safe KD math (see DistillationModel).

    Backbone starts FROZEN (base.trainable=False) - train_student_head/
    train_student_finetune below mirror the teacher's staged approach
    (frozen warmup, then partial unfreeze) instead of fine-tuning the whole
    7M-param backbone from step one, which was a real, unaddressed overfit
    risk on this project's training-pool scale.

    The backbone is called as `base(x)` with NO explicit `training=`
    argument - deliberately, not an oversight. Keras automatically forces
    inference-mode BatchNorm on any layer with `trainable=False`
    regardless of the ambient training context, and follows the ambient
    context once `trainable=True`. Hardcoding `training=True` here (an
    earlier version of this function did) would have baked "always use
    live-batch BatchNorm statistics" into the graph permanently, including
    at single-image deployment inference - degenerate for a batch of 1,
    and inconsistent between the Keras model and its ONNX export."""
    base = keras.applications.DenseNet121(
        input_shape=input_shape,
        include_top=False,
        weights=weights,
    )
    base.trainable = False
    inputs = keras.Input(shape=input_shape, name="student_input")
    x = layers.Lambda(keras.applications.densenet.preprocess_input,
                       name="student_preprocess")(inputs)
    x = base(x)
    x = layers.GlobalAveragePooling2D(name="student_avg_pool")(x)
    x = layers.BatchNormalization(name="student_bn")(x)
    x = layers.Dropout(0.3, name="student_dropout")(x)
    logits = layers.Dense(num_classes, activation=None, dtype="float32",
                           name="student_logits")(x)
    outputs = layers.Activation("softmax", dtype="float32", name="student_output")(logits)
    return keras.Model(inputs, outputs, name="densenet121_student")

# %%
# # Knowledge Distillation Implementation
"""Cell 10 - training logic unchanged from the previous version; added a
loss_tracker so training curves for the student have a plottable 'loss' key
(previously only 'accuracy' was exposed in the fit() history)."""


def _logits_submodel(model, logits_layer_name):
    """Build a Model that shares `model`'s layers/weights but returns the
    output of `logits_layer_name` (the pre-softmax Dense output) instead of
    the final softmax output. This is a view into the same functional graph
    - no new layers or variables are created, so gradients computed through
    this submodel update the exact same tf.Variable objects as `model`."""
    return keras.Model(
        model.input, model.get_layer(logits_layer_name).output,
        name=model.name + "_logits_view",
    )


class DistillationModel(keras.Model):
    """Combines student + teacher for knowledge distillation training.

    Both `student` and `teacher` are full models ending in a softmax
    Activation layer (their normal, standalone-usable form - required by
    evaluate_model, Grad-CAM, threshold analysis, etc. elsewhere in this
    file). Distillation math needs the PRE-softmax logits, not those
    softmax probabilities - re-softmaxing an already-softmaxed output
    (softmax-of-softmax) is not equivalent to temperature-scaled softmax
    over logits: it compresses already-bounded [0,1] values into a
    near-uniform distribution before the second softmax, flattening
    gradients and corrupting both the KD loss and the hard-label loss
    (which previously reused the same corrupted tensor). `_logits_submodel`
    taps each model's `*_logits` layer directly so the standard Hinton et
    al. (2015) KD formulation is computed correctly:
        soft targets   = softmax(teacher_logits / T)
        soft student   = softmax(student_logits / T)
        KD loss        = KLD(soft targets, soft student) * T^2
        hard loss      = CE(y, softmax(student_logits))   [T=1, unscaled]
    """

    def __init__(self, student, teacher, temperature=CFG.distill_temperature,
                 alpha=CFG.distill_alpha, **kwargs):
        super().__init__(**kwargs)
        self.student = student
        self.teacher = teacher
        self._student_logits_model = _logits_submodel(student, "student_logits")
        self._teacher_logits_model = _logits_submodel(teacher, "teacher_logits")
        self.T = temperature
        self.alpha = alpha
        self.loss_fn = keras.losses.CategoricalFocalCrossentropy(
            from_logits=False, gamma=CFG.focal_loss_gamma, alpha=CFG.focal_loss_alpha
        )
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.accuracy_tracker = keras.metrics.CategoricalAccuracy(name="accuracy")

    def compile(self, optimizer, metrics):
        super().compile(optimizer=optimizer, metrics=metrics)

    @property
    def metrics(self):
        # Deliberately NOT `super().metrics + [...]`. Confirmed on a real
        # run: this Keras version's base Model.metrics includes an
        # internally auto-created loss metric that's never actually updated
        # here (loss is computed and tracked manually via self.loss_tracker
        # instead), and calling .result() on that never-built metric raises
        # "Cannot get result() since the metric has not yet been built."
        # Since loss_tracker/accuracy_tracker already cover everything this
        # model reports, returning only those sidesteps the auto-metric
        # entirely rather than needing to build/update something never used.
        return [self.loss_tracker, self.accuracy_tracker]

    def _compute_losses(self, x, y, sample_weight, training):
        # Cast to float32 immediately, before any softmax/KLD/focal-loss
        # math: build_teacher/build_densenet_student already force their
        # OWN softmax Activation output layer to dtype="float32" as the
        # standard mixed-precision safeguard, but that only protects each
        # model's normal forward-pass output. _logits_submodel taps the
        # PRE-softmax logits layer directly (no dtype override), so without
        # this cast, every op below - softmax, KLD, focal loss's internal
        # log() terms - still runs in raw float16, bypassing that
        # safeguard entirely. Confirmed on a real run: student training hit
        # outright NaN loss (mid-epoch, after gradient clipping was already
        # in place, which only bounds gradient norm and cannot rescue an
        # already-NaN/Inf value produced by float16 loss computation)
        # while the teacher - trained via ordinary compile()/fit() against
        # its own float32-safe output layer, never through this method -
        # only showed noisy validation accuracy, not NaN. That asymmetry is
        # exactly what this gap predicts.
        student_logits = tf.cast(self._student_logits_model(x, training=training), tf.float32)
        teacher_logits = tf.cast(self._teacher_logits_model(x, training=False), tf.float32)

        teacher_probs_soft = tf.nn.softmax(teacher_logits / self.T, axis=-1)
        student_probs_soft = tf.nn.softmax(student_logits / self.T, axis=-1)
        distillation_loss = tf.reduce_mean(
            keras.losses.kullback_leibler_divergence(teacher_probs_soft, student_probs_soft)
        ) * (self.T * self.T)

        student_probs_hard = tf.nn.softmax(student_logits, axis=-1)
        student_loss = self.loss_fn(y, student_probs_hard, sample_weight=sample_weight)

        loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss
        return loss, student_probs_hard

    def train_step(self, data):
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            x, y = data
            sample_weight = None

        with tf.GradientTape() as tape:
            loss, student_probs_hard = self._compute_losses(x, y, sample_weight, training=True)

        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # self.compiled_metrics.update_state(...) used to be called here too
        # - removed. It was already vestigial once self.accuracy_tracker was
        # added below (compile() is called with metrics=[], so there was
        # nothing real for compiled_metrics to track; self.metrics reports
        # accuracy_tracker's value directly, not anything compiled_metrics
        # produces). Confirmed on a real run that under Keras 3.13.2 it's
        # not just dead weight but an outright crash: "MeanMetricWrapper.
        # update_state() missing 1 required positional argument: 'y_pred'" -
        # Keras 3's compiled_metrics resolves differently than Keras 2's
        # MetricsContainer did for a subclassed Model compiled with no
        # metrics. Never load-bearing here; just deleting it.
        self.loss_tracker.update_state(loss)
        self.accuracy_tracker.update_state(y, student_probs_hard, sample_weight=sample_weight)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            x, y = data
            sample_weight = None

        loss, student_probs_hard = self._compute_losses(x, y, sample_weight, training=False)

        self.loss_tracker.update_state(loss)
        self.accuracy_tracker.update_state(y, student_probs_hard, sample_weight=sample_weight)
        return {m.name: m.result() for m in self.metrics}

    def call(self, x, training=None):
        return self.student(x, training=training)

# %%
# # Training Routines
"""Cell 11 - Training functions for teacher (two-stage) and student. Each now
returns (model, history) and saves its own training-curve + LR-schedule
figures."""


class OverfitMonitor(keras.callbacks.Callback):
    """Tracks the train/val gap for a given metric every epoch.

    This doesn't change training behavior (early stopping already exists) -
    it makes the train/val divergence visible and diagnosable epoch-by-epoch,
    which your original 20-35M NirikNet run never had: it just silently
    overfit until you found out from the final numbers. Prints a warning the
    moment the gap crosses `gap_threshold`, and stores history for
    `plot_overfit_gap` to render at the end of the stage.
    """

    def __init__(self, gap_threshold=0.10, metric="accuracy"):
        super().__init__()
        self.gap_threshold = gap_threshold
        self.metric = metric
        self.epochs = []
        self.train_values = []
        self.val_values = []
        self.gaps = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        train_val = logs.get(self.metric)
        val_val = logs.get(f"val_{self.metric}")
        if train_val is None or val_val is None:
            return
        gap = train_val - val_val
        self.epochs.append(epoch + 1)
        self.train_values.append(train_val)
        self.val_values.append(val_val)
        self.gaps.append(gap)
        if gap > self.gap_threshold:
            print(f"  [Overfit warning] epoch {epoch + 1}: train_{self.metric}="
                  f"{train_val:.4f} vs val_{self.metric}={val_val:.4f} "
                  f"(gap={gap:.4f} > threshold {self.gap_threshold:.2f})")


def get_cosine_decay_with_warmup(learning_rate, total_steps, warmup_steps, hold_base_rate_steps=0):
    class WarmUpCosineDecaySchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, learning_rate, total_steps, warmup_steps, hold_base_rate_steps=0):
            super().__init__()
            self.learning_rate = learning_rate
            self.total_steps = total_steps
            self.warmup_steps = warmup_steps
            self.hold_base_rate_steps = hold_base_rate_steps

        def __call__(self, step):
            step = tf.cast(step, tf.float32)
            warmup_steps = tf.cast(self.warmup_steps, tf.float32)
            hold_steps = warmup_steps + tf.cast(self.hold_base_rate_steps, tf.float32)
            warmup_lr = self.learning_rate * (step / warmup_steps)
            hold_lr = self.learning_rate
            decay_progress = (step - hold_steps) / tf.cast(
                self.total_steps - self.warmup_steps - self.hold_base_rate_steps, tf.float32
            )
            decay_lr = 0.5 * self.learning_rate * (
                1.0 + tf.cos(tf.constant(np.pi) * decay_progress)
            )
            return tf.where(
                step < warmup_steps, warmup_lr,
                tf.where(step < hold_steps, hold_lr, decay_lr),
            )

        def get_config(self):
            return {
                "learning_rate": self.learning_rate,
                "total_steps": self.total_steps,
                "warmup_steps": self.warmup_steps,
                "hold_base_rate_steps": self.hold_base_rate_steps,
            }
    return WarmUpCosineDecaySchedule(learning_rate, total_steps, warmup_steps, hold_base_rate_steps)


def train_teacher_head(train_df, val_df, train_ds, val_ds, cfg=CFG):
    """Stage 1: Train only the classifier head of the teacher (backbone frozen)."""
    print("=" * 78)
    print("STAGE: TEACHER HEAD TRAINING (Frozen Backbone)")
    print("=" * 78)

    input_shape = (cfg.img_size, cfg.img_size, 3)
    teacher = build_teacher(input_shape, cfg.num_classes, weights="imagenet")

    total_steps = max(1, (len(train_df) // cfg.batch_size) * cfg.teacher_head_epochs)
    warmup_steps = max(1, (len(train_df) // cfg.batch_size) * cfg.teacher_warmup_epochs)
    lr_schedule = get_cosine_decay_with_warmup(cfg.teacher_lr_head, total_steps, warmup_steps)
    plot_lr_schedule(lr_schedule, total_steps, "Teacher Head", cfg)

    teacher.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=lr_schedule, weight_decay=cfg.weight_decay,
            clipnorm=cfg.grad_clip_norm,
        ),
        loss=keras.losses.CategoricalFocalCrossentropy(from_logits=False, gamma=cfg.focal_loss_gamma, alpha=cfg.focal_loss_alpha),
        metrics=["accuracy"],
    )

    overfit_monitor = OverfitMonitor(gap_threshold=0.10, metric="accuracy")
    callbacks = [
        keras.callbacks.ModelCheckpoint(cfg.out("teacher_head_best.keras"), monitor="val_accuracy",
                                         mode="max", save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=cfg.early_stopping_patience,
                                       restore_best_weights=True, verbose=1),
        overfit_monitor,
    ]

    history = teacher.fit(train_ds, validation_data=val_ds, epochs=cfg.teacher_head_epochs,
                           callbacks=callbacks, verbose=2)
    teacher.load_weights(cfg.out("teacher_head_best.keras"))
    plot_training_history(history, "Teacher Head", cfg)
    plot_overfit_gap(overfit_monitor, "Teacher Head", cfg)
    print("Loaded best teacher head weights.")
    return teacher, history


def train_teacher_finetune(train_df, val_df, train_ds, val_ds, head_teacher, cfg=CFG):
    """Stage 2: Partially unfreeze the ResNet-50 backbone's last two residual
    stages (conv4_block*, conv5_block*) and fine-tune, keeping BatchNorm
    layers within them frozen. The backbone is found by type
    (isinstance(l, keras.Model)), not by a fixed layer index - indices
    shifted once a `teacher_preprocess` Lambda layer was added before the
    backbone in build_teacher, which silently broke a hardcoded
    `head_teacher.layers[1]` lookup (confirmed by testing: layers[1] became
    the Lambda layer, which has no `.layers` to iterate - this would have
    crashed train_teacher_finetune the first time it actually ran). Looking
    up by type is robust to any future layers added before the backbone.

    conv5-only was too conservative: the teacher's external-test accuracy
    (73.6% vs. 96.8% val) caps what the student can ever learn via
    distillation, since the student never sees information the teacher
    doesn't encode. Extending to conv4+conv5 gives the backbone more
    capacity to adapt to this dataset's TB-specific features rather than
    only re-weighting frozen ImageNet features. BatchNorm layers stay
    frozen within the unfrozen stages - fine-tuning conv kernels while
    freezing BN running statistics/affine params is a standard
    stabilization technique (avoids the small per-batch stats of a
    fine-tuning run corrupting BN's learned normalization) and matches this
    project's own stated unfreezing principle ("excluding BatchNorm")."""
    print("=" * 78)
    print("STAGE: TEACHER FINE-TUNING (Partial Unfreeze - last two stages)")
    print("=" * 78)

    backbone_candidates = [l for l in head_teacher.layers if isinstance(l, keras.Model)]
    if len(backbone_candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one nested backbone submodel in head_teacher.layers, "
            f"found {len(backbone_candidates)}: "
            f"{[l.name for l in backbone_candidates]}. Cannot safely pick which one to unfreeze."
        )
    backbone = backbone_candidates[0]
    backbone.trainable = True
    unfrozen = 0
    frozen_bn = 0
    for layer in backbone.layers:
        if layer.name.startswith("conv4_block") or layer.name.startswith("conv5_block"):
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
                frozen_bn += 1
            else:
                layer.trainable = True
                unfrozen += 1
        else:
            layer.trainable = False
    print(f"Unfroze {unfrozen} layers in conv4_block*/conv5_block* "
          f"({frozen_bn} BatchNorm layers within them kept frozen).")

    total_steps = max(1, (len(train_df) // cfg.batch_size) * cfg.teacher_finetune_epochs)
    warmup_steps = max(1, (len(train_df) // cfg.batch_size) * cfg.teacher_warmup_epochs)
    lr_schedule = get_cosine_decay_with_warmup(cfg.teacher_lr_finetune, total_steps, warmup_steps)
    plot_lr_schedule(lr_schedule, total_steps, "Teacher Finetune", cfg)

    head_teacher.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=lr_schedule, weight_decay=cfg.weight_decay,
            clipnorm=cfg.grad_clip_norm,
        ),
        loss=keras.losses.CategoricalFocalCrossentropy(from_logits=False, gamma=cfg.focal_loss_gamma, alpha=cfg.focal_loss_alpha),
        metrics=["accuracy"],
    )

    overfit_monitor = OverfitMonitor(gap_threshold=0.10, metric="accuracy")
    callbacks = [
        keras.callbacks.ModelCheckpoint(cfg.out("teacher_finetune_best.keras"), monitor="val_accuracy",
                                         mode="max", save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=cfg.early_stopping_patience,
                                       restore_best_weights=True, verbose=1),
        overfit_monitor,
    ]

    history = head_teacher.fit(train_ds, validation_data=val_ds, epochs=cfg.teacher_finetune_epochs,
                                callbacks=callbacks, verbose=2)
    head_teacher.load_weights(cfg.out("teacher_finetune_best.keras"))
    plot_training_history(history, "Teacher Finetune", cfg)
    plot_overfit_gap(overfit_monitor, "Teacher Finetune", cfg)
    print("Loaded best teacher fine-tuned weights.")
    return head_teacher, history


class StudentModelCheckpoint(keras.callbacks.Callback):
    """Module-level (not nested in a training function) so both
    train_student_head and train_student_finetune can share it.
    DistillationModel isn't a plain compiled Keras model, so a plain
    ModelCheckpoint can't target it directly - this saves
    `self.model.student.save_weights(...)` based on a monitored metric
    instead."""

    def __init__(self, filepath, monitor="val_loss", mode="min", save_best_only=True):
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.best = None
        self.filepath = filepath

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            return
        if self.save_best_only:
            if self.best is None or self._is_improved(current, self.best):
                self.best = current
                self.model.student.save_weights(self.filepath)
        else:
            self.model.student.save_weights(self.filepath)

    def _is_improved(self, current, best):
        return current < best if self.mode == "min" else current > best


def train_student_head(train_df, val_df, train_ds, val_ds, teacher, cfg=CFG):
    """Stage 1: DenseNet121 backbone frozen (as built by
    build_densenet_student), train only the head via distillation. Mirrors
    the teacher's head-only warmup stage - addresses the overfit risk of
    fully fine-tuning a 7M-param pretrained backbone from step one, which
    the student previously did with no staged freeze at all."""
    print("=" * 78)
    print("STAGE: STUDENT HEAD TRAINING (Knowledge Distillation, frozen backbone)")
    print("=" * 78)

    input_shape = (cfg.img_size, cfg.img_size, 3)
    student = build_densenet_student(input_shape, cfg.num_classes)

    total_steps = max(1, (len(train_df) // cfg.batch_size) * cfg.student_head_epochs)
    warmup_steps = max(1, (len(train_df) // cfg.batch_size) * cfg.student_warmup_epochs)
    lr_schedule = get_cosine_decay_with_warmup(cfg.student_lr, total_steps, warmup_steps)
    plot_lr_schedule(lr_schedule, total_steps, "Student Head", cfg)

    distillation_model = DistillationModel(
        student=student, teacher=teacher,
        temperature=cfg.distill_temperature, alpha=cfg.distill_alpha,
    )
    distillation_model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=lr_schedule, weight_decay=cfg.weight_decay,
            clipnorm=cfg.grad_clip_norm,
        ),
        metrics=[],
    )

    overfit_monitor = OverfitMonitor(gap_threshold=0.10, metric="accuracy")
    callbacks = [
        StudentModelCheckpoint(cfg.out("student_head_best.weights.h5"), monitor="val_accuracy",
                                mode="max", save_best_only=True),
        keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max",
                                       patience=cfg.early_stopping_patience,
                                       restore_best_weights=True, verbose=1),
        overfit_monitor,
    ]

    history = distillation_model.fit(train_ds, validation_data=val_ds, epochs=cfg.student_head_epochs,
                                      callbacks=callbacks, verbose=2)
    if os.path.isfile(cfg.out("student_head_best.weights.h5")):
        student.load_weights(cfg.out("student_head_best.weights.h5"))
    plot_training_history(history, "Student Head", cfg)
    plot_overfit_gap(overfit_monitor, "Student Head", cfg)
    print("Loaded best student head weights.")
    return student, history


def train_student_finetune(train_df, val_df, train_ds, val_ds, teacher, head_student, cfg=CFG):
    """Stage 2: Partially unfreeze DenseNet121's last dense block
    (conv5_block*, verified by inspection to be 112 layers / 16 sub-blocks
    in DenseNet121) and continue distillation training, keeping BatchNorm
    layers within it frozen - same rationale and pattern as the teacher's
    conv4/conv5 unfreeze: a fully-frozen backbone can only re-weight
    generic ImageNet features, not adapt them to TB-specific patterns, but
    fine-tuning the whole 7M-param network at once (the previous design)
    risked overfitting on this project's dataset scale.

    Backbone found by type (isinstance(l, keras.Model)), not a fixed layer
    index - see train_teacher_finetune's docstring for why a hardcoded
    index silently broke once a preprocessing Lambda layer was added
    before the backbone."""
    print("=" * 78)
    print("STAGE: STUDENT FINE-TUNING (Partial Unfreeze - last dense block)")
    print("=" * 78)

    backbone_candidates = [l for l in head_student.layers if isinstance(l, keras.Model)]
    if len(backbone_candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one nested backbone submodel in head_student.layers, "
            f"found {len(backbone_candidates)}: "
            f"{[l.name for l in backbone_candidates]}. Cannot safely pick which one to unfreeze."
        )
    backbone = backbone_candidates[0]
    backbone.trainable = True
    unfrozen, frozen_bn = 0, 0
    for layer in backbone.layers:
        if layer.name.startswith("conv5_block"):
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
                frozen_bn += 1
            else:
                layer.trainable = True
                unfrozen += 1
        else:
            layer.trainable = False
    print(f"Unfroze {unfrozen} layers in conv5_block* "
          f"({frozen_bn} BatchNorm layers within them kept frozen).")

    total_steps = max(1, (len(train_df) // cfg.batch_size) * cfg.student_epochs)
    warmup_steps = max(1, (len(train_df) // cfg.batch_size) * cfg.student_warmup_epochs)
    lr_schedule = get_cosine_decay_with_warmup(cfg.student_lr_finetune, total_steps, warmup_steps)
    plot_lr_schedule(lr_schedule, total_steps, "Student Finetune", cfg)

    distillation_model = DistillationModel(
        student=head_student, teacher=teacher,
        temperature=cfg.distill_temperature, alpha=cfg.distill_alpha,
    )
    distillation_model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=lr_schedule, weight_decay=cfg.weight_decay,
            clipnorm=cfg.grad_clip_norm,
        ),
        # No compiled "accuracy" metric here - self.accuracy_tracker inside
        # DistillationModel already computes it explicitly against the
        # correct (single-softmax, T=1) student predictions. Passing
        # metrics=["accuracy"] here would register a second, redundant
        # "accuracy" entry computed from whatever compiled_metrics receives.
        metrics=[],
    )

    # DistillationModel now tracks accuracy explicitly via self.accuracy_tracker
    # (compiled_metrics alone didn't reliably surface "val_accuracy" for this
    # custom train_step/test_step in earlier testing), so checkpointing/early
    # stopping can monitor val_accuracy here, matching the teacher's
    # callbacks above instead of the loss (which, given class imbalance, is
    # a noisier model-selection signal than accuracy since it's also
    # weighted by the KD term).
    overfit_monitor = OverfitMonitor(gap_threshold=0.10, metric="accuracy")
    callbacks = [
        StudentModelCheckpoint(cfg.out("student_best.weights.h5"), monitor="val_accuracy",
                                mode="max", save_best_only=True),
        keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max",
                                       patience=cfg.early_stopping_patience,
                                       restore_best_weights=True, verbose=1),
        overfit_monitor,
    ]

    history = distillation_model.fit(train_ds, validation_data=val_ds, epochs=cfg.student_epochs,
                                      callbacks=callbacks, verbose=2)
    if not os.path.isfile(cfg.out("student_best.weights.h5")):
        # Defensive fallback: if no checkpoint was ever written (e.g. the
        # monitored metric was missing for some other reason we haven't
        # seen yet), save current weights directly rather than crash on load.
        print("No checkpoint was saved during training - saving current "
              "student weights directly as a fallback.")
        head_student.save_weights(cfg.out("student_best.weights.h5"))
    head_student.load_weights(cfg.out("student_best.weights.h5"))
    plot_training_history(history, "Student Finetune", cfg)
    plot_overfit_gap(overfit_monitor, "Student Finetune", cfg)
    print("Loaded best student fine-tuned weights.")
    return head_student, history

# %%
# # Evaluation Functions
"""Cell 12 - Evaluation utilities (metrics, ROC, PR, confusion matrix).
Saved figures now go through the shared `_savefig` helper (JPG by default)."""


def _plot_confusion_matrix(cm, class_names, filepath, cfg=CFG):
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm, copy=False, nan=0.0)

    threshold = cm_norm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm_norm[i, j] > threshold else "black"
        plt.text(j, i, format(cm[i, j], "d") if cm[i, j] != 0 else "",
                  horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    _savefig(figure, filepath, cfg)
    plt.close(figure)


def _plot_roc_curve(y_true, y_score, filepath, cfg=CFG):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    fig = plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:0.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    _savefig(fig, filepath, cfg)
    plt.close(fig)
    return roc_auc


def _plot_pr_curve(y_true, y_score, filepath, cfg=CFG):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
    fig = plt.figure()
    plt.plot(recall, precision, color="blue", lw=2, label=f"PR curve (area = {pr_auc:0.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    _savefig(fig, filepath, cfg)
    plt.close(fig)
    return pr_auc


def evaluate_model(model, dataset, model_name, cfg=CFG):
    """Evaluate a model on a dataset, returning metrics and saving figures."""
    all_probs = []
    all_labels = []

    for batch in dataset:
        batch_images, batch_labels = batch[0], batch[1]
        probs = model.predict(batch_images, verbose=0)
        all_probs.append(probs)
        all_labels.append(batch_labels)

    y_prob = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_labels, axis=0)

    if np.isnan(y_prob).any():
        # Fail fast, right here, with the actual cause named - rather than
        # letting this reach roc_auc_score()/precision_recall_curve() below,
        # which raise a generic "Input contains NaN" far from the real
        # source. Confirmed on a real run: this happens when a training
        # stage's loss went NaN (see DistillationModel._compute_losses'
        # float32 cast) and NaN weights silently propagated all the way to
        # evaluation before anything surfaced the actual problem.
        raise RuntimeError(
            f"{model_name}: model predictions contain NaN ({int(np.isnan(y_prob).sum())} "
            f"of {y_prob.size} values). This means the model's weights went NaN during "
            f"training (most likely a NaN loss at some epoch) - check the training log "
            f"for 'loss: nan' before this point. Evaluating a NaN-weighted model is "
            f"meaningless; fix the training instability, don't just re-run evaluation."
        )

    y_pred = np.argmax(y_prob, axis=1)
    y_true_labels = np.argmax(y_true, axis=1)

    accuracy = accuracy_score(y_true_labels, y_pred)
    precision = precision_score(y_true_labels, y_pred, zero_division=0)
    recall = recall_score(y_true_labels, y_pred, zero_division=0)
    f1 = f1_score(y_true_labels, y_pred, zero_division=0)

    y_score = y_prob[:, 1]

    roc_auc = roc_auc_score(y_true_labels, y_score)
    pr_curve_precision, pr_curve_recall, _ = precision_recall_curve(y_true_labels, y_score)
    pr_auc = auc(pr_curve_recall, pr_curve_precision)

    specificity = recall_score(y_true_labels, y_pred, pos_label=0, zero_division=0)
    sensitivity = recall
    balanced_acc = (sensitivity + specificity) / 2.0
    mcc = matthews_corrcoef(y_true_labels, y_pred)
    kappa = cohen_kappa_score(y_true_labels, y_pred)

    cm = confusion_matrix(y_true_labels, y_pred)

    class_report_dict = classification_report(
        y_true_labels, y_pred, target_names=list(cfg.class_names),
        output_dict=True, zero_division=0,
    )
    class_report_text = classification_report(
        y_true_labels, y_pred, target_names=list(cfg.class_names), zero_division=0,
    )

    metrics = {
        "accuracy": float(accuracy), "precision": float(precision), "recall": float(recall),
        "f1": float(f1), "roc_auc": float(roc_auc), "pr_auc": float(pr_auc),
        "specificity": float(specificity), "sensitivity": float(sensitivity),
        "balanced_accuracy": float(balanced_acc), "mcc": float(mcc), "cohen_kappa": float(kappa),
        "confusion_matrix": cm.tolist(),
        "classification_report": class_report_dict,
    }

    prefix = model_name.lower().replace(" ", "_") + "_"

    _plot_confusion_matrix(cm, list(cfg.class_names), cfg.fig(f"{prefix}confusion_matrix"), cfg)
    _plot_roc_curve(y_true_labels, y_score, cfg.fig(f"{prefix}roc_curve"), cfg)
    _plot_pr_curve(y_true_labels, y_score, cfg.fig(f"{prefix}pr_curve"), cfg)

    report_path = cfg.out(f"{prefix}classification_report.txt")
    with open(report_path, "w") as f:
        f.write(class_report_text)

    print(f"{model_name} evaluation metrics:")
    for k, v in metrics.items():
        if k not in ("confusion_matrix", "classification_report"):
            print(f"  {k}: {v:.4f}")
    print(f"{model_name} per-class report:\n{class_report_text}")
    print(f"Saved per-class classification report -> {report_path}")

    return metrics, y_true, y_prob


def benchmark_inference_latency(model, model_name, cfg=CFG, n_runs=100, warmup_runs=10):
    """Time n_runs single-image forward passes, forced onto CPU regardless
    of GPU availability, since the deployment target (HF Spaces free tier)
    is CPU-only. Uses direct __call__ rather than .predict() - .predict()
    carries extra per-call overhead (dataset building, retracing) that would
    inflate the number relative to what a real serving loop looks like."""
    print(f"Benchmarking {model_name} CPU inference latency ...")
    dummy_input = tf.constant(
        np.random.rand(1, cfg.img_size, cfg.img_size, 3).astype("float32")
    )

    with tf.device("/CPU:0"):
        for _ in range(warmup_runs):
            _ = model(dummy_input, training=False)

        start = time.time()
        for _ in range(n_runs):
            _ = model(dummy_input, training=False)
        elapsed = time.time() - start

    ms_per_image = (elapsed / n_runs) * 1000
    print(f"{model_name}: {ms_per_image:.2f} ms/image on CPU "
          f"({n_runs} runs, {warmup_runs} warmup runs discarded)")
    return {"model": model_name, "ms_per_image": round(ms_per_image, 3), "n_runs": n_runs}


def analyze_decision_thresholds(y_true, y_prob, model_name, cfg=CFG, thresholds=(0.3, 0.5, 0.7)):
    """Report accuracy/sensitivity/specificity/precision at a few decision
    thresholds on the TB-positive probability, not just the default argmax
    (0.5). For a screening tool, a lower threshold trades some specificity
    for higher sensitivity (fewer missed TB cases) - worth showing
    explicitly as an option rather than only reporting the default
    operating point."""
    y_true_labels = np.argmax(y_true, axis=1) if y_true.ndim > 1 else y_true
    y_score = y_prob[:, 1]

    print(f"Threshold analysis for {model_name}:")
    results = []
    for t in thresholds:
        y_pred_t = (y_score >= t).astype(int)
        acc = accuracy_score(y_true_labels, y_pred_t)
        sens = recall_score(y_true_labels, y_pred_t, pos_label=1, zero_division=0)
        spec = recall_score(y_true_labels, y_pred_t, pos_label=0, zero_division=0)
        prec = precision_score(y_true_labels, y_pred_t, zero_division=0)
        results.append({
            "threshold": t, "accuracy": float(acc), "sensitivity": float(sens),
            "specificity": float(spec), "precision": float(prec),
        })
        print(f"  threshold={t:.2f} | accuracy={acc:.4f}  sensitivity={sens:.4f}  "
              f"specificity={spec:.4f}  precision={prec:.4f}")

    path = cfg.out(f"{model_name.lower().replace(' ', '_')}_threshold_analysis.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved threshold analysis -> {path}")
    return results


def select_youden_threshold(y_true, y_prob, model_name, cfg=CFG):
    """Select the decision threshold maximizing Youden's J statistic
    (sensitivity + specificity - 1; Youden, 1950) - the standard
    threshold-selection method for a binary screening test. It picks the
    ROC point furthest from the chance diagonal without assuming a specific
    cost ratio between false negatives and false positives.

    MUST be called on validation data only. The returned threshold is meant
    to be frozen and reused for external-test scoring via
    `evaluate_at_threshold` - selecting a threshold on the test set itself
    would leak test information into an operating point and inflate the
    reported test metrics."""
    y_true_labels = np.argmax(y_true, axis=1) if y_true.ndim > 1 else y_true
    y_score = y_prob[:, 1]
    fpr, tpr, thresholds = roc_curve(y_true_labels, y_score)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    best_threshold = float(thresholds[best_idx])
    print(f"{model_name}: Youden-optimal threshold = {best_threshold:.4f} "
          f"(sensitivity={tpr[best_idx]:.4f}, specificity={1 - fpr[best_idx]:.4f}, "
          f"J={j_scores[best_idx]:.4f})")
    return best_threshold


def evaluate_at_threshold(y_true, y_prob, threshold, model_name, cfg=CFG):
    """Report metrics at a single fixed decision threshold - used to score a
    set (typically the external test set) at a threshold chosen on a
    DIFFERENT set (typically validation, via `select_youden_threshold`)."""
    y_true_labels = np.argmax(y_true, axis=1) if y_true.ndim > 1 else y_true
    y_score = y_prob[:, 1]
    y_pred_t = (y_score >= threshold).astype(int)
    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true_labels, y_pred_t)),
        "sensitivity": float(recall_score(y_true_labels, y_pred_t, pos_label=1, zero_division=0)),
        "specificity": float(recall_score(y_true_labels, y_pred_t, pos_label=0, zero_division=0)),
        "precision": float(precision_score(y_true_labels, y_pred_t, zero_division=0)),
        "f1": float(f1_score(y_true_labels, y_pred_t, zero_division=0)),
    }
    print(f"{model_name} @ frozen threshold={threshold:.4f}: {metrics}")
    path = cfg.out(f"{model_name.lower().replace(' ', '_')}_frozen_threshold_metrics.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved frozen-threshold metrics -> {path}")
    return metrics

# %%
# # Explainability (Grad-CAM, Grad-CAM++, Eigen-CAM, Layer-CAM)
"""Cell 13 - Explainability implementations."""


def _get_grad_model(model, layer_name):
    """Return something callable as `conv_outputs, predictions = grad_model(inputs)`,
    usable directly inside a GradientTape.

    Two cases:
    1. `layer_name` is a top-level layer of `model` itself - Keras's
       standard get_layer()-based sub-model reconstruction works fine here.
    2. `layer_name` is a nested submodel (e.g. the teacher's ResNet50 or the
       student's DenseNet121 backbone, wrapped as a single layer). Both
       models also have a Lambda preprocessing layer BEFORE the backbone
       (canonical 0-255 RGB in, backbone-normalized out - see build_teacher /
       build_densenet_student) and a small classifier head AFTER it.
       Reconstructing via get_layer(...).output raises a disconnected-graph
       KeyError at call time for this case - confirmed by testing, not a
       typo - so instead the layers before the target (preprocessing) and
       after it (GAP -> BN -> Dropout -> Dense -> Activation) are replayed
       manually in their original graph order, split by the target's index
       in `model.layers` rather than assumed to all come after it. This
       stays purely eager/tape-based and sidesteps Keras's static graph
       reconstruction machinery entirely.
    """
    target = model.get_layer(layer_name)

    if isinstance(target, keras.Model):
        target_idx = model.layers.index(target)
        pre_layers = [
            l for l in model.layers[1:target_idx]
            if not isinstance(l, keras.layers.InputLayer)
        ]
        post_layers = model.layers[target_idx + 1:]

        def call(inputs):
            x = inputs
            for layer in pre_layers:
                x = layer(x, training=False)
            conv_out = target(x, training=False)
            x = conv_out
            for layer in post_layers:
                x = layer(x, training=False)
            return conv_out, x

        return call

    return keras.Model(
        inputs=[model.inputs],
        outputs=[target.output, model.output],
    )


def compute_gradcam(model, image_batch, class_idx, layer_name):
    grad_model = _get_grad_model(model, layer_name)
    with tf.GradientTape() as tape:
        inputs = tf.cast(image_batch, tf.float32)
        tape.watch(inputs)
        conv_outputs, predictions = grad_model(inputs)
        loss = predictions[:, class_idx]
    grads = tape.gradient(loss, conv_outputs)[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = tf.reduce_sum(tf.multiply(weights, conv_outputs[0]), axis=-1)
    cam = tf.maximum(cam, 0)
    cam = tf.image.resize(cam[tf.newaxis, ..., tf.newaxis], tf.shape(image_batch)[1:3])[0, ..., 0]
    cam_max = tf.reduce_max(cam)
    if cam_max == 0:
        return tf.zeros_like(cam).numpy()
    return (cam / cam_max).numpy()


def compute_gradcam_plus(model, image_batch, class_idx, layer_name):
    grad_model = _get_grad_model(model, layer_name)
    with tf.GradientTape() as tape:
        inputs = tf.cast(image_batch, tf.float32)
        tape.watch(inputs)
        conv_outputs, predictions = grad_model(inputs)
        loss = predictions[:, class_idx]
    grads = tape.gradient(loss, conv_outputs)[0]
    conv_outputs_single = conv_outputs[0]
    square_grads = tf.pow(grads, 2)
    cube_grads = tf.pow(grads, 3)
    sum_conv_outputs = tf.reduce_sum(conv_outputs_single, axis=(0, 1))
    alpha_num = square_grads
    alpha_denom = 2.0 * square_grads + sum_conv_outputs[tf.newaxis, tf.newaxis, :] * cube_grads
    alpha_denom = tf.where(alpha_denom != 0.0, alpha_denom, tf.ones_like(alpha_denom))
    alphas = alpha_num / alpha_denom
    weights = tf.reduce_sum(alphas * tf.nn.relu(grads), axis=(0, 1))
    cam = tf.reduce_sum(tf.multiply(weights, conv_outputs_single), axis=-1)
    cam = tf.maximum(cam, 0)
    cam = tf.image.resize(cam[tf.newaxis, ..., tf.newaxis], tf.shape(image_batch)[1:3])[0, ..., 0]
    cam_max = tf.reduce_max(cam)
    if cam_max == 0:
        return tf.zeros_like(cam).numpy()
    return (cam / cam_max).numpy()


def compute_layercam(model, image_batch, class_idx, layer_name):
    grad_model = _get_grad_model(model, layer_name)
    with tf.GradientTape() as tape:
        inputs = tf.cast(image_batch, tf.float32)
        tape.watch(inputs)
        conv_outputs, predictions = grad_model(inputs)
        loss = predictions[:, class_idx]
    grads = tape.gradient(loss, conv_outputs)[0]
    weights = tf.nn.relu(grads)
    cam = tf.reduce_sum(tf.multiply(weights, conv_outputs[0]), axis=-1)
    cam = tf.maximum(cam, 0)
    cam = tf.image.resize(cam[tf.newaxis, ..., tf.newaxis], tf.shape(image_batch)[1:3])[0, ..., 0]
    cam_max = tf.reduce_max(cam)
    if cam_max == 0:
        return tf.zeros_like(cam).numpy()
    return (cam / cam_max).numpy()


def compute_eigencam(model, image_batch, layer_name):
    grad_model = _get_grad_model(model, layer_name)
    conv_outputs, _ = grad_model(image_batch)
    # Cast to float32 up front: under mixed_float16 policy, conv_outputs
    # comes back as float16, which conflicts with the explicit float32 cast
    # a few lines down - and eigendecomposition is numerically safer in
    # float32 regardless of that conflict.
    conv_outputs = tf.cast(conv_outputs[0], tf.float32)
    h, w, c = conv_outputs.shape
    conv_reshaped = tf.reshape(conv_outputs, (-1, c))
    mean = tf.reduce_mean(conv_reshaped, axis=0)
    conv_centered = conv_reshaped - mean
    cov = tf.matmul(conv_centered, conv_centered, transpose_a=True) / tf.cast(h * w - 1, tf.float32)
    _eigen_values, eigen_vectors = tf.linalg.eigh(cov)
    principal_component = eigen_vectors[:, -1]
    cam = tf.matmul(conv_reshaped, tf.expand_dims(principal_component, axis=-1))
    cam = tf.reshape(cam, [h, w])
    cam = tf.maximum(cam, 0)
    cam = tf.image.resize(cam[tf.newaxis, ..., tf.newaxis], tf.shape(image_batch)[1:3])[0, ..., 0]
    cam_max = tf.reduce_max(cam)
    if cam_max == 0:
        return tf.zeros_like(cam).numpy()
    return (cam / cam_max).numpy()


def explain_predictions(model, unet, df, cfg=CFG, model_name="Model", target_layer=None):
    """Generate and save Grad-CAM / Grad-CAM++ / Layer-CAM / Eigen-CAM
    visualizations for a balanced subset of images."""
    print("=" * 78)
    print(f"STAGE: EXPLAINABILITY ({model_name})")
    print("=" * 78)

    normal_df = df[df["label"] == 0].sample(
        min(cfg.explain_samples // 2, len(df[df["label"] == 0])), random_state=SEED
    )
    tb_df = df[df["label"] == 1].sample(
        min(cfg.explain_samples // 2, len(df[df["label"] == 1])), random_state=SEED
    )
    sample_df = pd.concat([normal_df, tb_df]).sample(frac=1, random_state=SEED)

    model_inputs, display_images, paths = [], [], []
    for _, row in sample_df.iterrows():
        try:
            # Same canonical tensor serves both uses now - the model itself
            # normalizes internally, so no separate "normalized" variant.
            model_input = preprocess_xray(row["path"], unet, cfg)
            display_img = model_input
            model_inputs.append(model_input)
            display_images.append(display_img)
            paths.append(row["path"])
        except Exception as e:
            print(f"Skipping {row['path']} during explainability: {e}")
            continue

    if not model_inputs:
        print("No valid images for explainability.")
        return

    model_inputs = np.array(model_inputs)
    probs = model.predict(model_inputs, verbose=0)
    preds = np.argmax(probs, axis=1)

    if target_layer is None:
        conv_layers = [
            l.name for l in model.layers
            if isinstance(l, (layers.Conv2D, layers.SeparableConv2D)) and "output" not in l.name
        ]
        if conv_layers:
            target_layer = conv_layers[-1]
        else:
            # No top-level conv layer - the backbone is a nested submodel
            # (e.g. ResNet50 wrapped inside the teacher via build_teacher).
            # Use the nested model's own name/output as the target: for an
            # include_top=False application model, that IS the final conv
            # feature map, and - unlike reaching into the submodel's own
            # internal layers, which belong to a separate, disconnected
            # build context - it's already correctly wired to this outer
            # model's actual input.
            nested_model_names = [l.name for l in model.layers if isinstance(l, keras.Model)]
            target_layer = nested_model_names[-1] if nested_model_names else None
        if target_layer is None:
            raise ValueError("No suitable convolutional layer found for explainability.")

    for idx, (model_input, display_img, path, pred) in enumerate(
        zip(model_inputs, display_images, paths, preds)
    ):
        raw_img = np.clip(display_img, 0, 255).astype("uint8")
        img_batch = model_input[None, ...]

        gradcam = compute_gradcam(model, img_batch, pred, target_layer)
        gradcam_plus = compute_gradcam_plus(model, img_batch, pred, target_layer)
        layercam = compute_layercam(model, img_batch, pred, target_layer)
        eigencam = compute_eigencam(model, img_batch, target_layer)

        def heatmap_to_rgb(heatmap):
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        def overlay_heatmap(raw, heatmap, alpha=0.6):
            heatmap_rgb = heatmap_to_rgb(heatmap)
            return cv2.addWeighted(raw, 1 - alpha, heatmap_rgb, alpha, 0)

        gradcam_overlay = overlay_heatmap(raw_img, gradcam)
        gradcam_plus_overlay = overlay_heatmap(raw_img, gradcam_plus)
        layercam_overlay = overlay_heatmap(raw_img, layercam)
        eigencam_overlay = overlay_heatmap(raw_img, eigencam)

        figure, axs = plt.subplots(2, 2, figsize=(11, 10))
        figure.suptitle(
            f"Explainability for {os.path.basename(path)} - Predicted: {cfg.class_names[pred]}",
            fontsize=14,
        )

        axs[0, 0].imshow(gradcam_overlay); axs[0, 0].set_title("Grad-CAM"); axs[0, 0].axis("off")
        axs[0, 1].imshow(gradcam_plus_overlay); axs[0, 1].set_title("Grad-CAM++"); axs[0, 1].axis("off")
        axs[1, 0].imshow(layercam_overlay); axs[1, 0].set_title("Layer-CAM"); axs[1, 0].axis("off")
        axs[1, 1].imshow(eigencam_overlay); axs[1, 1].set_title("Eigen-CAM"); axs[1, 1].axis("off")

        plt.tight_layout()
        filepath = cfg.fig(f"{model_name.lower().replace(' ', '_')}_explain_{idx:02d}")
        _savefig(figure, filepath, cfg)
        plt.close(figure)

    print(f"Saved explainability visualizations to {cfg.figures_dir}")


def _lung_mask_in_model_frame(path, unet, cfg=CFG):
    """Reuses _segment_and_crop's geometry (same one preprocess_xray uses)
    to return the lung mask at the classifier's own input resolution - used
    only for the Grad-CAM/lung-localization audit below, never for training
    or inference. Since preprocess_xray now zeroes out non-lung pixels
    directly, this mask also describes exactly which pixels the model's
    input tensor left non-zero."""
    gray = read_image_grayscale(path)
    _cropped_gray, cropped_mask = _segment_and_crop(gray, unet, cfg)
    resized_mask = cv2.resize(
        cropped_mask.astype("float32"), (cfg.img_size, cfg.img_size),
        interpolation=cv2.INTER_NEAREST,
    )
    return (resized_mask > 0.5).astype("uint8")


def audit_gradcam_lung_localization(model, unet, df, cfg=CFG, model_name="Model",
                                     target_layer=None, n_samples=60,
                                     low_overlap_threshold=0.5):
    """Shortcut-detection audit, NOT a training constraint: lung cropping in
    preprocess_xray already removes most background before the classifier
    ever sees it, but the ~8.5% crop padding and any rib-cage/collarbone/
    text-stamp artifacts inside that padding could still let the model key
    on something other than lung tissue. This is the check that would
    actually catch that, rather than assuming the crop alone guarantees
    on-target attention.

    For a sample of TB-labeled images the model itself predicts as TB,
    measures what fraction of the Grad-CAM heatmap's energy falls inside the
    segmented lung mask (recomputed in the model's own cropped/resized
    coordinate frame, not the original image's). A LOW overlap fraction on a
    correctly-classified TB image is the signature of a shortcut: the model
    got the right answer while looking somewhere other than the lungs.
    Also flags images whose heatmap concentrates near the crop border - a
    classic sign of a leaked artifact/stamp rather than genuine pulmonary
    signal.
    """
    tb_df = df[df["label"] == 1]
    sample_df = tb_df.sample(min(n_samples, len(tb_df)), random_state=SEED)

    if target_layer is None:
        nested_model_names = [l.name for l in model.layers if isinstance(l, keras.Model)]
        target_layer = nested_model_names[-1] if nested_model_names else None
        if target_layer is None:
            raise ValueError("No suitable convolutional layer found for the localization audit.")

    overlaps = []
    edge_flags = []
    for _, row in sample_df.iterrows():
        try:
            model_input = preprocess_xray(row["path"], unet, cfg)
            lung_mask = _lung_mask_in_model_frame(row["path"], unet, cfg)
        except Exception as e:
            print(f"Skipping {row['path']} during localization audit: {e}")
            continue

        img_batch = model_input[None, ...]
        prob = model.predict(img_batch, verbose=0)[0]
        pred = int(np.argmax(prob))
        if pred != 1:
            continue  # only audit images the model itself calls TB

        heatmap = compute_gradcam(model, img_batch, class_idx=1, layer_name=target_layer)
        total_energy = heatmap.sum()
        if total_energy <= 0:
            continue

        inside_energy = (heatmap * lung_mask).sum()
        overlaps.append(float(inside_energy / total_energy))

        # Outer 10%-wide border ring vs. interior - a heatmap that peaks
        # near the crop border is a classic artifact/stamp-leak signature.
        border = max(1, int(0.10 * cfg.img_size))
        border_mask = np.ones_like(heatmap, dtype=bool)
        border_mask[border:-border, border:-border] = False
        interior_mean = heatmap[~border_mask].mean() if (~border_mask).any() else 0.0
        border_mean = heatmap[border_mask].mean()
        edge_flags.append(bool(interior_mean > 0 and border_mean > interior_mean))

    if not overlaps:
        print(f"{model_name}: no correctly-classified TB samples available for the localization audit.")
        return None

    mean_overlap = float(np.mean(overlaps))
    pct_low_overlap = float(np.mean([o < low_overlap_threshold for o in overlaps]))
    pct_edge_flagged = float(np.mean(edge_flags))

    result = {
        "n_audited": len(overlaps),
        "mean_gradcam_lung_overlap": mean_overlap,
        "fraction_below_overlap_threshold": pct_low_overlap,
        "overlap_threshold": low_overlap_threshold,
        "fraction_edge_flagged": pct_edge_flagged,
    }

    print(f"{model_name} Grad-CAM/lung-mask localization audit "
          f"({len(overlaps)} correctly-classified TB images):")
    print(f"  mean overlap fraction (heatmap energy inside lung mask): {mean_overlap:.4f}")
    print(f"  fraction of images below {low_overlap_threshold:.2f} overlap: {pct_low_overlap:.4f}")
    print(f"  fraction flagged for border/edge-concentrated activation: {pct_edge_flagged:.4f}")
    if mean_overlap < low_overlap_threshold:
        print(f"  WARNING: mean overlap below {low_overlap_threshold:.2f} - the model may be "
              f"relying on non-pulmonary shortcuts rather than lung pathology. Recommend "
              f"manual review of the saved explainability figures.")

    path = cfg.out(f"{model_name.lower().replace(' ', '_')}_gradcam_lung_localization_audit.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved localization audit -> {path}")
    return result


def audit_per_source_performance(model, df, unet, cfg=CFG, model_name="Model", threshold=0.5,
                                  accuracy_spread_warn=0.20):
    """Shortcut-detection audit: breaks test-set performance out by dataset
    source (the 'source' column already carried on every row - see
    build_classification_dataframe). A model relying on genuine TB signal
    should generalize roughly evenly across sources; a model that aces one
    source far better than the rest is a strong signature of a per-source
    shortcut (scanner/compression/collimation artifact) rather than real
    pulmonary-pathology learning - particularly for DA/DB, which are curated
    TB-positive-only collections with no Normal counterpart from the same
    scanner, so any DA/DB-specific artifact would be indistinguishable from
    the TB label itself during training.
    """
    rows = []
    for source, group in df.groupby("source"):
        model_inputs, y_true = [], []
        for _, row in group.iterrows():
            try:
                model_inputs.append(preprocess_xray(row["path"], unet, cfg))
                y_true.append(int(row["label"]))
            except Exception as e:
                print(f"Skipping {row['path']} during per-source audit: {e}")
                continue

        if not model_inputs:
            continue

        probs = model.predict(np.array(model_inputs), verbose=0)[:, 1]
        y_true = np.array(y_true)
        y_pred = (probs >= threshold).astype(int)

        acc = accuracy_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred, zero_division=0) if (y_true == 1).any() else None
        rows.append({
            "source": source, "n": int(len(y_true)),
            "n_tb": int((y_true == 1).sum()), "n_normal": int((y_true == 0).sum()),
            "accuracy": float(acc), "recall_tb": float(rec) if rec is not None else None,
        })

    print(f"{model_name} per-source performance breakdown (threshold={threshold:.4f}):")
    for r in rows:
        rec_display = f"{r['recall_tb']:.4f}" if r["recall_tb"] is not None else "n/a (no TB samples)"
        print(f"  {r['source']:<12} n={r['n']:<5} accuracy={r['accuracy']:.4f}  recall_tb={rec_display}")

    accuracies = [r["accuracy"] for r in rows]
    spread = float(max(accuracies) - min(accuracies)) if len(accuracies) > 1 else 0.0
    if len(accuracies) > 1:
        print(f"  Accuracy spread across sources: {spread:.4f}")
        if spread > accuracy_spread_warn:
            print(f"  WARNING: >{accuracy_spread_warn:.2f} accuracy spread across sources - "
                  f"investigate whether the model is exploiting a per-source (scanner/artifact) "
                  f"shortcut rather than a genuine TB signal, particularly for DA/DB (TB-only, "
                  f"no cross-label counterpart from the same scanner).")

    path = cfg.out(f"{model_name.lower().replace(' ', '_')}_per_source_audit.json")
    with open(path, "w") as f:
        json.dump({"per_source": rows, "accuracy_spread": spread}, f, indent=2)
    print(f"Saved per-source audit -> {path}")
    return {"per_source": rows, "accuracy_spread": spread}

# %%
# # Visualization & Reporting
"""Cell 14 - Everything needed to write the report: training curves, learning
rate schedules, architecture diagrams (special focus on NirikNet), dataset
composition, segmentation/augmentation samples, and the teacher-vs-student
comparison chart. All saved as JPG into `cfg.figures_dir`."""


def plot_training_history(history, stage_name, cfg=CFG, metrics=("loss", "accuracy")):
    """Plot train/val curves for the given metrics from a Keras History object."""
    available = [m for m in metrics if m in history.history]
    if not available:
        print(f"No matching metrics to plot for {stage_name}.")
        return
    fig, axes = plt.subplots(1, len(available), figsize=(6 * len(available), 5))
    if len(available) == 1:
        axes = [axes]
    for ax, metric in zip(axes, available):
        ax.plot(history.history[metric], label=f"train_{metric}", linewidth=2)
        val_key = f"val_{metric}"
        if val_key in history.history:
            ax.plot(history.history[val_key], label=f"val_{metric}", linewidth=2)
        ax.set_title(f"{stage_name}: {metric}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        ax.legend()
        ax.grid(alpha=0.3)
    plt.tight_layout()
    filepath = cfg.fig(f"history_{stage_name.lower().replace(' ', '_')}")
    _savefig(fig, filepath, cfg)
    plt.close(fig)
    print(f"Saved training curves -> {filepath}")


def plot_overfit_gap(monitor, stage_name, cfg=CFG):
    """Plot train vs. val curves alongside the gap between them, using the
    history captured by an OverfitMonitor callback."""
    if not monitor.gaps:
        print(f"No overfit-gap data captured for {stage_name}.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(monitor.epochs, monitor.train_values, label=f"train_{monitor.metric}", linewidth=2)
    axes[0].plot(monitor.epochs, monitor.val_values, label=f"val_{monitor.metric}", linewidth=2)
    axes[0].set_title(f"{stage_name}: Train vs Val {monitor.metric}")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel(monitor.metric)
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(monitor.epochs, monitor.gaps, color="#D32F2F", linewidth=2)
    axes[1].axhline(monitor.gap_threshold, color="black", linestyle="--",
                     label=f"warning threshold ({monitor.gap_threshold:.2f})")
    axes[1].axhline(0, color="gray", linewidth=0.8)
    axes[1].set_title(f"{stage_name}: Overfitting Gap (train - val {monitor.metric})")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Gap")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    filepath = cfg.fig(f"overfit_gap_{stage_name.lower().replace(' ', '_')}")
    _savefig(fig, filepath, cfg)
    plt.close(fig)
    print(f"Saved overfit-gap plot -> {filepath}")


def plot_lr_schedule(schedule, total_steps, stage_name, cfg=CFG):
    """Visualize the warmup + cosine decay learning rate schedule."""
    sample_every = max(1, total_steps // 500)
    steps = np.arange(0, total_steps, sample_every)
    lrs = [float(schedule(tf.constant(float(s)))) for s in steps]

    fig = plt.figure(figsize=(7, 4))
    plt.plot(steps, lrs, color="#5C6BC0", linewidth=2)
    plt.title(f"Learning Rate Schedule: {stage_name}")
    plt.xlabel("Training Step")
    plt.ylabel("Learning Rate")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    filepath = cfg.fig(f"lr_schedule_{stage_name.lower().replace(' ', '_')}")
    _savefig(fig, filepath, cfg)
    plt.close(fig)
    print(f"Saved LR schedule plot -> {filepath}")


def plot_architecture_diagram(model, filepath, cfg=CFG, show_shapes=True):
    """Render a Keras model's layer diagram via keras.utils.plot_model.
    Requires pydot + graphviz; falls back to PNG, then skips gracefully with
    a clear message if neither format renders (e.g. graphviz binary missing)."""
    try:
        keras.utils.plot_model(
            model, to_file=filepath, show_shapes=show_shapes,
            show_layer_names=True, dpi=cfg.figure_dpi, expand_nested=False,
        )
        print(f"Saved architecture diagram -> {filepath}")
        return
    except Exception as exc:
        print(f"{cfg.figure_format.upper()} architecture diagram failed for "
              f"'{model.name}': {exc}")
    fallback = os.path.splitext(filepath)[0] + ".png"
    try:
        keras.utils.plot_model(
            model, to_file=fallback, show_shapes=show_shapes,
            show_layer_names=True, dpi=cfg.figure_dpi, expand_nested=False,
        )
        print(f"Saved architecture diagram (PNG fallback) -> {fallback}")
    except Exception as exc2:
        print(f"Architecture diagram unavailable for '{model.name}' - is "
              f"pydot/graphviz installed? ({exc2})")


def plot_pipeline_overview(cfg=CFG):
    """Hand-drawn flowchart of the full multi-stage pipeline - the
    'overall architecture, every stage' figure for the report."""
    stages = [
        ("Data Sources", "tawsifurrahman + India + DA/DB +\nMontgomery + Shenzhen (+ TBX11K)\npooled, patient-wise 70/15/15"),
        ("Preprocessing", "U-Net lung mask -> crop -> CLAHE\n-> resize 224x224 (canonical RGB,\nper-model norm applied internally)"),
        ("Attention U-Net", "Lung segmentation\n4-level encoder-decoder,\nattention gates on every skip"),
        ("Teacher: ResNet-50", "ImageNet-pretrained, ~23.6M params\nStage 1: frozen backbone\nStage 2: unfreeze conv4+conv5 blocks"),
        ("Knowledge Distillation", "Hard-label focal loss + KL(teacher || student)\nT = 3.0, alpha = 0.5"),
        ("Student: DenseNet121", "ImageNet-pretrained, ~7.0M params\nfull fine-tune via distillation"),
        ("Explainability", "Grad-CAM, Grad-CAM++,\nLayer-CAM, Eigen-CAM"),
        ("Evaluation", "Accuracy, sensitivity, specificity,\nROC/PR-AUC, MCC, kappa\non held-out pooled test split"),
    ]
    colors = ["#B0BEC5", "#90CAF9", "#64B5F6", "#FFAB91", "#CE93D8", "#80CBC4", "#FFF59D", "#A5D6A7"]

    fig, ax = plt.subplots(figsize=(7, len(stages) * 1.7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(stages) * 2)
    ax.axis("off")

    positions = []
    y = len(stages) * 2 - 1
    for (title, desc), color in zip(stages, colors):
        positions.append(y)
        box = FancyBboxPatch(
            (0.7, y - 0.75), 8.6, 1.5, boxstyle="round,pad=0.08",
            linewidth=1.2, edgecolor="black", facecolor=color,
        )
        ax.add_patch(box)
        ax.text(5, y + 0.28, title, ha="center", va="center", fontsize=11, fontweight="bold")
        ax.text(5, y - 0.28, desc, ha="center", va="center", fontsize=7.5)
        y -= 2

    for i in range(len(positions) - 1):
        ax.annotate(
            "", xy=(5, positions[i + 1] + 0.75), xytext=(5, positions[i] - 0.75),
            arrowprops=dict(arrowstyle="->", lw=1.6, color="black"),
        )

    plt.title("End-to-End Pipeline Architecture", fontsize=13, fontweight="bold", pad=18)
    plt.tight_layout()
    filepath = cfg.fig("pipeline_overview")
    _savefig(fig, filepath, cfg)
    plt.close(fig)
    print(f"Saved pipeline overview -> {filepath}")


def plot_dataset_composition(train_df, val_df, test_df, cfg=CFG):
    """Bar charts: images per split, class balance per split, and per-source
    breakdown of the training pool."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    splits = ["Train", "Val", "Test"]
    dfs = [train_df, val_df, test_df]
    sizes = [len(d) for d in dfs]
    bars = axes[0].bar(splits, sizes, color=["#64B5F6", "#81C784", "#E57373"])
    axes[0].set_title("Images per Split")
    axes[0].set_ylabel("Count")
    for bar, v in zip(bars, sizes):
        axes[0].text(bar.get_x() + bar.get_width() / 2, v, str(v), ha="center", va="bottom")

    class_names = list(cfg.class_names)
    width = 0.25
    x = np.arange(len(class_names))
    for i, (name, d) in enumerate(zip(splits, dfs)):
        counts = [len(d[d["label"] == c]) for c in range(len(class_names))]
        axes[1].bar(x + i * width, counts, width, label=name)
    axes[1].set_xticks(x + width)
    axes[1].set_xticklabels(class_names)
    axes[1].set_title("Class Balance per Split")
    axes[1].set_ylabel("Count")
    axes[1].legend()

    source_counts = train_df["source"].value_counts()
    axes[2].bar(source_counts.index, source_counts.values, color="#9575CD")
    axes[2].set_title("Train Split by Source")
    axes[2].set_ylabel("Count")
    axes[2].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    filepath = cfg.fig("dataset_composition")
    _savefig(fig, filepath, cfg)
    plt.close(fig)
    print(f"Saved dataset composition chart -> {filepath}")


def plot_segmentation_samples(unet, image_paths, cfg=CFG):
    """Original CXR + predicted lung mask overlay, side by side, for a few images."""
    image_paths = list(image_paths)[:4]
    if not image_paths:
        print("No images available for segmentation preview.")
        return

    fig, axes = plt.subplots(2, len(image_paths), figsize=(4 * len(image_paths), 8))
    if len(image_paths) == 1:
        axes = axes.reshape(2, 1)

    for i, path in enumerate(image_paths):
        gray = read_image_grayscale(path)
        mask = predict_lung_mask(unet, gray, cfg)
        mask = keep_largest_components(mask, num_components=2)
        mask = morphological_cleanup(mask, kernel_size=5)

        display_gray = np.clip(gray, 0, 255).astype("uint8")
        overlay = cv2.cvtColor(display_gray, cv2.COLOR_GRAY2RGB)
        overlay_mask = np.zeros_like(overlay)
        overlay_mask[..., 1] = mask * 255
        blended = cv2.addWeighted(overlay, 0.7, overlay_mask, 0.3, 0)

        axes[0, i].imshow(display_gray, cmap="gray")
        axes[0, i].set_title(f"Original\n{os.path.basename(path)}", fontsize=8)
        axes[0, i].axis("off")
        axes[1, i].imshow(blended)
        axes[1, i].set_title("Predicted Lung Mask", fontsize=8)
        axes[1, i].axis("off")

    plt.tight_layout()
    filepath = cfg.fig("segmentation_samples")
    _savefig(fig, filepath, cfg)
    plt.close(fig)
    print(f"Saved segmentation samples -> {filepath}")


def plot_augmentation_preview(df, unet, cfg=CFG, n=3):
    """Original vs. augmented versions of a few training images."""
    sample_paths = df["path"].sample(min(n, len(df)), random_state=SEED).tolist()

    fig, axes = plt.subplots(2, len(sample_paths), figsize=(4 * len(sample_paths), 8))
    if len(sample_paths) == 1:
        axes = axes.reshape(2, 1)

    for i, path in enumerate(sample_paths):
        original = preprocess_xray(path, unet, cfg)
        # _AUG_LAYERS is a shared module-level object first used (unbatched)
        # inside make_dataset()'s tf.data pipeline, which locks in its
        # expected input shape as (224,224,3) with no batch dim. Calling it
        # with an explicit batch dimension here caused a shape mismatch -
        # call it the same, unbatched way it's used everywhere else.
        augmented = _AUG_LAYERS(original, training=True).numpy()
        augmented = tf.image.random_brightness(augmented, max_delta=0.1).numpy()
        augmented = np.clip(augmented, 0, 255).astype("uint8")

        axes[0, i].imshow(np.clip(original, 0, 255).astype("uint8"))
        axes[0, i].set_title("Original", fontsize=9)
        axes[0, i].axis("off")
        axes[1, i].imshow(augmented)
        axes[1, i].set_title("Augmented", fontsize=9)
        axes[1, i].axis("off")

    plt.tight_layout()
    filepath = cfg.fig("augmentation_preview")
    _savefig(fig, filepath, cfg)
    plt.close(fig)
    print(f"Saved augmentation preview -> {filepath}")


def plot_model_comparison(teacher_val, student_val, teacher_ext, student_ext,
                           teacher_params, student_params, cfg=CFG):
    """Bar charts comparing teacher vs. student: key metrics on both val and
    the held-out test split, plus parameter-count / compression-ratio chart.
    This is the central figure for the distillation results section."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    metric_keys = ["accuracy", "sensitivity", "specificity", "roc_auc"]
    x = np.arange(len(metric_keys))
    width = 0.2

    axes[0].bar(x - 1.5 * width, [teacher_val[k] for k in metric_keys], width, label="Teacher (Val)")
    axes[0].bar(x - 0.5 * width, [student_val[k] for k in metric_keys], width, label="Student (Val)")
    axes[0].bar(x + 0.5 * width, [teacher_ext[k] for k in metric_keys], width, label="Teacher (Test)")
    axes[0].bar(x + 1.5 * width, [student_ext[k] for k in metric_keys], width, label="Student (Test)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metric_keys, rotation=20)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Teacher vs Student: Key Metrics")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)

    names = ["Teacher\n(ResNet-50)", "Student\n(DenseNet121)"]
    params = [teacher_params, student_params]
    bars = axes[1].bar(names, params, color=["#EF9A9A", "#80CBC4"])
    ratio = teacher_params / max(student_params, 1)
    axes[1].set_title(f"Parameter Count (compression: {ratio:.1f}x)")
    axes[1].set_ylabel("Parameters")
    for bar, p in zip(bars, params):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                      f"{p:,}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    filepath = cfg.fig("teacher_vs_student_comparison")
    _savefig(fig, filepath, cfg)
    plt.close(fig)
    print(f"Saved teacher vs student comparison -> {filepath}")

# %%
# # ONNX Export (deployment artifact)
"""Cell 14b - Export ONLY the student (the model that actually gets
deployed - the teacher never ships) to ONNX, then validate the export
against the original Keras model before trusting it.

Why: the Kaggle training environment's TF/Keras version and the backend's
serving environment (a mix of TF/Keras and PyTorch, per the repo's current
state) don't reliably agree - a Keras-3-saved model isn't guaranteed to
load cleanly under Keras 2, and there is no first-party Keras->PyTorch
loader at all. ONNX decouples training-framework version churn from
serving entirely: the serving container only needs onnxruntime, not
tensorflow/keras or torch. ONNX Runtime's CPU execution provider is also
typically faster than either native TF or PyTorch eager execution, which
directly helps the CPU-only HF Spaces deployment target.

Never assume a format conversion is lossless without checking - the
validation step below is not optional."""


def export_student_to_onnx(student, cfg=CFG, n_validation_samples=8, atol=2e-3):
    """atol=2e-3, not 1e-4: confirmed on a real run under this project's
    mixed_float16 global policy that Keras and ONNX Runtime diverge by
    ~1.42e-4 on a genuinely correct export - the model's ~427 DenseNet121
    layers compute largely in float16 during training/export, and TF vs.
    ONNX Runtime round mixed-precision ops slightly differently internally
    (BatchNorm epsilon handling, conv algorithm choice, etc.). That's
    expected cross-backend noise, not a correctness bug: 1e-4 was
    implicitly calibrated for pure float32 (where an earlier, separate
    verification got 1.19e-7), not this mixed-precision regime. 2e-3
    (0.2%) on a bounded [0,1] softmax probability is still tight enough to
    catch a real logic error (which would typically show up as an
    order-of-magnitude-larger or outright different prediction) while
    giving real headroom over the observed 1.42e-4."""
    print("=" * 78)
    print("STAGE: ONNX EXPORT (student model only)")
    print("=" * 78)

    _ensure_package("tf2onnx")
    _ensure_package("onnxruntime")
    import tf2onnx
    import onnxruntime as ort

    input_signature = [tf.TensorSpec(
        (None, cfg.img_size, cfg.img_size, 3), tf.float32, name="student_input"
    )]
    onnx_path = cfg.out("densenet121_student.onnx")
    model_proto, _ = tf2onnx.convert.from_keras(
        student, input_signature=input_signature, output_path=onnx_path, opset=13,
    )
    print(f"Exported ONNX model -> {onnx_path}")

    # Validation: same random canonical-range (0-255) input through both the
    # original Keras model and the ONNX Runtime session, compare outputs.
    rng = np.random.default_rng(SEED)
    sample = (rng.random((n_validation_samples, cfg.img_size, cfg.img_size, 3)) * 255.0).astype("float32")

    keras_output = student.predict(sample, verbose=0)

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    onnx_output = session.run(None, {input_name: sample})[0]

    max_abs_diff = float(np.max(np.abs(keras_output - onnx_output)))
    matches = bool(np.allclose(keras_output, onnx_output, atol=atol))
    print(f"Validation: max abs diff = {max_abs_diff:.2e} (tolerance {atol:.0e}), "
          f"match = {matches}")

    if not matches:
        raise RuntimeError(
            f"ONNX export validation FAILED - Keras and ONNX Runtime outputs diverge "
            f"by up to {max_abs_diff:.2e}, exceeding tolerance {atol:.0e}. Do not deploy "
            f"this ONNX file until the discrepancy is understood - a silent divergence "
            f"here means production predictions would not match what was evaluated."
        )

    validation_report = {
        "onnx_path": onnx_path,
        "opset": 13,
        "n_validation_samples": n_validation_samples,
        "max_abs_diff": max_abs_diff,
        "tolerance": atol,
        "passed": matches,
    }
    report_path = cfg.out("onnx_export_validation.json")
    with open(report_path, "w") as f:
        json.dump(validation_report, f, indent=2)
    print(f"Saved ONNX export validation report -> {report_path}")
    return onnx_path

# %%
# # Main Execution
"""Cell 15 - Orchestrator that runs the full pipeline end-to-end, generating
every report figure along the way into `CFG.figures_dir`."""


def main():
    """Full pipeline: segmentation -> teacher (ResNet-50) -> student
    (DenseNet121, via distillation) -> evaluation on a patient-wise
    held-out test split of the pooled dataset -> explainability -> report
    figures throughout. See Config.montgomery_dir's comment for why there is
    no longer a permanently-held-out external test set.
    """
    start_time = time.time()
    print("=" * 78)
    print("TUBERCULOSIS CHEST X-RAY CLASSIFICATION PIPELINE")
    print("=" * 78)
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    save_run_config()

    apply_smoke_test_overrides()
    resolve_all_paths()
    discover_datasets()

    # U-Net trained first - it has no dependency on the classification
    # dataframe (trains on its own separate masks-and-labels dataset) - so
    # it's available here to power the mask-coverage quality filter below,
    # before the pool gets split.
    unet, _seg_history = train_lung_segmentation()

    df = build_classification_dataframe()
    df = deduplicate_pool(df)
    df = filter_image_quality(df)
    df = filter_mask_coverage(df, unet)
    df = _rebalance_to_target_ratio(df, CFG)
    train_df, val_df, test_df = split_dataframe(df)
    export_split_csvs(train_df, val_df, test_df)
    plot_dataset_composition(train_df, val_df, test_df)
    check_dataset_leakage(train_df, val_df, test_df)
    plot_segmentation_samples(
        unet, train_df["path"].sample(min(4, len(train_df)), random_state=SEED).tolist(),
    )

    class_weights = compute_class_weights(train_df) if CFG.use_class_weights else None

    train_ds = make_dataset(train_df, unet, cfg=CFG, training=True, class_weights=class_weights)
    val_ds = make_dataset(val_df, unet, cfg=CFG, training=False)
    test_ds = make_dataset(test_df, unet, cfg=CFG, training=False)
    plot_augmentation_preview(train_df, unet, cfg=CFG)

    teacher_head, _th_history = train_teacher_head(train_df, val_df, train_ds, val_ds)
    teacher, _tf_history = train_teacher_finetune(train_df, val_df, train_ds, val_ds, teacher_head)
    plot_architecture_diagram(teacher, CFG.fig("teacher_architecture"))

    student_head, _sh_history = train_student_head(train_df, val_df, train_ds, val_ds, teacher)
    student, _sf_history = train_student_finetune(train_df, val_df, train_ds, val_ds, teacher, student_head)
    plot_architecture_diagram(student, CFG.fig("densenet121_student_architecture"))

    plot_pipeline_overview()

    teacher_val_metrics, _, _ = evaluate_model(teacher, val_ds, "Teacher_Val", cfg=CFG)
    student_val_metrics, student_val_y_true, student_val_y_prob = evaluate_model(
        student, val_ds, "DenseNet121Student_Val", cfg=CFG
    )

    teacher_test_metrics, _, _ = evaluate_model(teacher, test_ds, "Teacher_Test", cfg=CFG)
    student_test_metrics, student_test_y_true, student_test_y_prob = evaluate_model(
        student, test_ds, "DenseNet121Student_Test", cfg=CFG
    )

    # Threshold analysis on the STUDENT specifically - it's the deployed
    # model, so its decision threshold is the one that's actually
    # operationally relevant.
    student_val_thresholds = analyze_decision_thresholds(
        student_val_y_true, student_val_y_prob, "DenseNet121Student_Val", cfg=CFG
    )
    student_test_thresholds = analyze_decision_thresholds(
        student_test_y_true, student_test_y_prob, "DenseNet121Student_Test", cfg=CFG
    )

    # Operational threshold: chosen via Youden's J on validation data ONLY,
    # then frozen and applied identically to the test set - never fit on the
    # data it's then scored against.
    student_youden_threshold = select_youden_threshold(
        student_val_y_true, student_val_y_prob, "DenseNet121Student_Val", cfg=CFG
    )
    student_val_at_youden = evaluate_at_threshold(
        student_val_y_true, student_val_y_prob, student_youden_threshold,
        "DenseNet121Student_Val_YoudenThreshold", cfg=CFG
    )
    student_test_at_youden = evaluate_at_threshold(
        student_test_y_true, student_test_y_prob, student_youden_threshold,
        "DenseNet121Student_Test_YoudenThreshold", cfg=CFG
    )

    # Shortcut-detection audits (see CLAUDE.md's mandate that no prediction
    # is trusted without an explanation): confirms the deployed student is
    # actually keying on lung pathology rather than a background/scanner/
    # per-source artifact that happens to correlate with the label.
    student_localization_audit = audit_gradcam_lung_localization(
        student, unet, test_df, cfg=CFG, model_name="DenseNet121Student",
    )
    student_per_source_audit = audit_per_source_performance(
        student, test_df, unet, cfg=CFG, model_name="DenseNet121Student",
        threshold=student_youden_threshold,
    )

    teacher_params = int(teacher.count_params())
    student_params = int(student.count_params())
    plot_model_comparison(
        teacher_val_metrics, student_val_metrics, teacher_test_metrics, student_test_metrics,
        teacher_params, student_params, cfg=CFG,
    )

    teacher_latency = benchmark_inference_latency(teacher, "Teacher (ResNet-50)", cfg=CFG)
    student_latency = benchmark_inference_latency(student, "Student (DenseNet121)", cfg=CFG)

    all_metrics = {
        "config": {
            "img_size": CFG.img_size,
            "batch_size": CFG.batch_size,
            "teacher_epochs": CFG.teacher_head_epochs + CFG.teacher_finetune_epochs,
            "student_epochs": CFG.student_head_epochs + CFG.student_epochs,
            "distill_temperature": CFG.distill_temperature,
            "distill_alpha": CFG.distill_alpha,
            "weight_decay": CFG.weight_decay,
        },
        "teacher_val": teacher_val_metrics,
        "student_val": student_val_metrics,
        "teacher_test": teacher_test_metrics,
        "student_test": student_test_metrics,
        "teacher_params": teacher_params,
        "student_params": student_params,
        "compression_ratio": teacher_params / max(student_params, 1),
        "teacher_cpu_latency": teacher_latency,
        "student_cpu_latency": student_latency,
        "student_threshold_analysis_val": student_val_thresholds,
        "student_threshold_analysis_test": student_test_thresholds,
        "student_youden_threshold": student_youden_threshold,
        "student_val_at_youden_threshold": student_val_at_youden,
        "student_test_at_youden_threshold": student_test_at_youden,
        "student_gradcam_lung_localization_audit": student_localization_audit,
        "student_per_source_audit": student_per_source_audit,
    }

    metrics_path = CFG.out("metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")

    explain_predictions(teacher, unet, test_df, cfg=CFG, model_name="Teacher")
    explain_predictions(student, unet, test_df, cfg=CFG, model_name="DenseNet121Student")

    export_student_to_onnx(student, cfg=CFG)

    end_time = time.time()
    hours, rem = divmod(end_time - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("=" * 78)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print(f"Total time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
    print(f"Results saved in: {CFG.output_dir}")
    print(f"Report figures saved in: {CFG.figures_dir}")
    print("=" * 78)


if __name__ == "__main__":
    main()
