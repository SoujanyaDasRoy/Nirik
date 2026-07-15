import os
import torch
from unittest.mock import patch, MagicMock
from backend.core.inference import get_model

def test_load_quantized_model(monkeypatch):
    monkeypatch.setenv("USE_QUANTIZED_MODEL", "1")
    monkeypatch.setenv("QUANTIZED_MODEL_PATH", "dummy.pt")
    # Mock torch.jit.load to avoid actually loading a file
    with patch('torch.jit.load') as mock_load:
        mock_model = MagicMock()
        mock_load.return_value = mock_model
        # Call get_model and check that it attempts to load quantized path
        model = get_model()
        mock_load.assert_called_once_with("dummy.pt")
        assert model == mock_model

if __name__ == "__main__":
    # For manual testing
    class MockMonkeyPatch:
        def setenv(self, key, value):
            os.environ[key] = value
        def undo(self):
            pass
    monkeypatch = MockMonkeyPatch()
    test_load_quantized_model(monkeypatch)
    print("Test completed")
