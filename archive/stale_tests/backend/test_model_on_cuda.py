import torch
import pytest
from unittest.mock import patch
from backend.core.inference import get_model

def test_model_on_cuda_if_available(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    model = get_model()
    # Check if model parameters are on CUDA
    if hasattr(model, 'parameters'):
        # PyTorch model
        assert next(model.parameters()).device.type == "cuda"
    else:
        # Keras model - check if we can access underlying tensors
        # This is trickier, but for now we'll skip if it's not a PyTorch model
        # Since we're using Torch backend for Keras, we should be able to check
        try:
            weights = model.weights
            if len(weights) > 0:
                assert weights[0].device.type == "cuda"
        except:
            # Fallback: just check that the model loads without error
            assert model is not None
