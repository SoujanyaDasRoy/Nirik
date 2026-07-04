"""
Regression test for Bug #1: heatmap focus preprocessing.

Root cause: the Generation 1 path (no U-Net deployed) takes a grayscale
image, stacks it 3× to make a degenerate RGB, then subtracts per-channel
BGR means (103.939, 116.779, 123.680) from channels 0/1/2 respectively.
Because all three input channels carry the SAME value (grayscale),
this introduces a per-channel offset that breaks channel equality and
systematically biases the DenseNet-121 input distribution away from what
the model was trained on. The Grad-CAM heatmap ends up focused on the
artificial channel-shift artefacts rather than the lung fields.

Correct behaviour: when the input is grayscale stacked 3×, each channel
should remain equal after preprocessing (a global bias is absorbed by
the network). The fix subtracts the AVERAGE of the three BGR means so
all three channels stay equal.
"""
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# We import inference lazily / carefully — torch + keras may not be present.
try:
    import torch  # noqa: F401
except ImportError:
    torch = None


IMG_SIZE = 224  # mirror inference module default


def _import_preprocess():
    try:
        from core.inference import preprocess_for_classifier
        return preprocess_for_classifier
    except Exception as exc:  # pragma: no cover - environment dependent
        import pytest
        pytest.skip(f"inference module not importable: {exc}")


def test_grayscale_uniform_input_keeps_channel_equality_in_generation1():
    """Uniform grayscale stacked 3× must yield equal channels after
    preprocessing (Generation 1 path)."""
    if torch is None:
        import pytest
        pytest.skip("torch not installed")

    preprocess = _import_preprocess()

    # A uniform gray image at mid-intensity.
    arr = np.full((IMG_SIZE, IMG_SIZE), 128.0, dtype=np.float32)
    tensor = preprocess(arr, unet_active=False)

    # tensor shape: (1, 224, 224, 3)
    assert tensor.shape == (1, IMG_SIZE, IMG_SIZE, 3), tensor.shape
    arr_out = tensor[0].cpu().numpy()
    ch0 = arr_out[..., 0]
    ch1 = arr_out[..., 1]
    ch2 = arr_out[..., 2]

    # All three channels must remain numerically equal — the current bug
    # introduces per-channel offset (128-103.939, 128-116.779, 128-123.680)
    # which fails this assertion.
    diff01 = float(np.max(np.abs(ch0 - ch1)))
    diff02 = float(np.max(np.abs(ch0 - ch2)))
    diff12 = float(np.max(np.abs(ch1 - ch2)))

    assert diff01 < 1e-4, (
        f"channel 0 vs channel 1 differ by {diff01:.4f} after preprocessing "
        "of uniform grayscale; Generation 1 path leaks BGR per-channel means "
        "into a single-channel image and corrupts the network input."
    )
    assert diff02 < 1e-4, f"channel 0 vs channel 2 differ by {diff02:.4f}"
    assert diff12 < 1e-4, f"channel 1 vs channel 2 differ by {diff12:.4f}"


def test_grayscale_input_in_generation2_keeps_channel_equality():
    """Same invariant for Generation 2 (with U-Net) — densenet preprocess
    applies per-channel (mean, std), but input is still grayscale stacked,
    so channels must remain equal."""
    if torch is None:
        import pytest
        pytest.skip("torch not installed")

    preprocess = _import_preprocess()

    arr = np.full((IMG_SIZE, IMG_SIZE), 200.0, dtype=np.float32)
    tensor = preprocess(arr, unet_active=True)
    arr_out = tensor[0].cpu().numpy()
    ch0, ch1, ch2 = arr_out[..., 0], arr_out[..., 1], arr_out[..., 2]

    diff01 = float(np.max(np.abs(ch0 - ch1)))
    diff02 = float(np.max(np.abs(ch0 - ch2)))
    assert diff01 < 1e-4 and diff02 < 1e-4, (
        f"Generation 2 path also broke channel equality: {diff01=}, {diff02=}"
    )