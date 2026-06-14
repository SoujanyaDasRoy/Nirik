import io
import pydicom
import numpy as np
from PIL import Image

def safe_float_parse(val):
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, pydicom.multival.MultiValue):
        return float(val[0])
    val_str = str(val).strip()
    if '\\' in val_str:
        try:
            return float(val_str.split('\\')[0])
        except ValueError:
            return None
    try:
        return float(val_str)
    except ValueError:
        return None

def process_dicom(file_bytes):
    dcm = pydicom.dcmread(io.BytesIO(file_bytes), force=True)
    arr = dcm.pixel_array.astype(np.float32)
    
    intercept = safe_float_parse(getattr(dcm, 'RescaleIntercept', 0))
    slope = safe_float_parse(getattr(dcm, 'RescaleSlope', 1))
    if intercept is None: intercept = 0.0
    if slope is None: slope = 1.0
    arr = arr * slope + intercept
    
    center = getattr(dcm, 'WindowCenter', None)
    width = getattr(dcm, 'WindowWidth', None)
    
    if center is not None and width is not None:
        try:
            c_val = safe_float_parse(center)
            w_val = safe_float_parse(width)
            if c_val is not None and w_val is not None:
                vmin, vmax = c_val - (w_val / 2), c_val + (w_val / 2)
                arr = np.clip(arr, vmin, vmax)
                arr = (arr - vmin) / (w_val + 1e-8) * 255
            else:
                arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
        except Exception:
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
    else:
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
        
    # Invert pixel values if PhotometricInterpretation is MONOCHROME1 (radiology standard standardization)
    if getattr(dcm, 'PhotometricInterpretation', '') == 'MONOCHROME1':
        arr = 255.0 - arr
        
    img = Image.fromarray(arr.astype(np.uint8))
    return img.convert("RGB") if img.mode != "RGB" else img, dcm

def extract_metadata(dcm) -> dict:
    metadata = {}
    tags = [
        ('PatientID', 'patient_id', 'Patient ID'),
        ('PatientName', 'patient_name', 'Patient Name'),
        ('PatientAge', 'patient_age', 'Patient Age'),
        ('PatientSex', 'patient_sex', 'Patient Sex'),
        ('Modality', 'modality', 'Modality'),
        ('StudyDate', 'study_date', 'Study Date'),
        ('BodyPartExamined', 'body_part', 'Body Part')
    ]
    for tag, key, label in tags:
        val = getattr(dcm, tag, None)
        if val is not None:
            val_str = str(val).strip()
            if key == 'patient_name':
                val_str = val_str.replace('^', ', ')
            metadata[key] = val_str
        else:
            metadata[key] = "Unknown"
            
    # Extract PixelSpacing for frontend ruler calibration
    pixel_spacing = getattr(dcm, 'PixelSpacing', getattr(dcm, 'ImagerPixelSpacing', None))
    if pixel_spacing is not None:
        try:
            if isinstance(pixel_spacing, (list, tuple)) or hasattr(pixel_spacing, '__iter__'):
                spacing_vals = [float(x) for x in pixel_spacing]
            else:
                spacing_vals = [float(pixel_spacing), float(pixel_spacing)]
            metadata['pixel_spacing'] = spacing_vals
        except Exception:
            metadata['pixel_spacing'] = None
    else:
        metadata['pixel_spacing'] = None
        
    return metadata
