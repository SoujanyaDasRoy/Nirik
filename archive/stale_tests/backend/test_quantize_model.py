import os
import shutil
from backend.quantize_model import quantize_model

def test_quantize_model_output_exists():
    # Define paths
    model_source = "backend/tb_student_densenet121.keras"
    model_dest = "fixtures/tb_student_densenet121.keras"
    cal_dir = "fixtures/calibration"
    output_path = "fixtures/tb_student_densenet121_quantized.pt"

    # Ensure fixtures directory exists
    os.makedirs("fixtures", exist_ok=True)

    # Copy model to fixtures if not already there
    if not os.path.exists(model_dest):
        shutil.copy(model_source, model_dest)

    # Create calibration directory
    os.makedirs(cal_dir, exist_ok=True)

    # Run the quantization function
    quantize_model(model_dest, cal_dir, output_path)

    # Check that the output file exists
    assert os.path.exists(output_path)