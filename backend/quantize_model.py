import torch
import torch.nn as nn
from pathlib import Path

def quantize_model(model_path: str, cal_dir: str, output_path: str):
    # Dummy implementation: copy model as placeholder
    Path(output_path).write_bytes(Path(model_path).read_bytes())