import torch
import numpy as np
from pathlib import Path
from monai.transforms import LoadImage, Orientation, EnsureChannelFirst
from dicom_utils import load_dicom_volume_robust

def compare_loaders(patient_id="15094"):
    data_root = Path("../../../venturalitica-sdk-samples-extra/scenarios/surgery-dicom-tcia/data/dicom").resolve()
    patient_dir = data_root / patient_id
    
    # Identify non-SEG files
    files = sorted(list(patient_dir.rglob("*.dcm")))
    input_files = [str(f) for f in files if "SEG" not in str(f).upper() and "seg" not in str(f).lower()]
    
    # 1. Custom Load
    print(f"--- Custom Loader ---")
    custom_mt = load_dicom_volume_robust(input_files)
    print(f"Shape: {custom_mt.shape}")
    print(f"Affine:\n{custom_mt.affine}")
    
    # 2. MONAI Load (using the parent directory of images, as it handles series)
    # Actually MONAI LoadImage on a list of files works too
    print(f"\n--- MONAI LoadImage ---")
    monai_loader = LoadImage(image_only=True)
    try:
        monai_mt = monai_loader(input_files)
        # Ensure channel first for comparison
        monai_mt = EnsureChannelFirst()(monai_mt)
        print(f"Shape: {monai_mt.shape}")
        print(f"Affine:\n{monai_mt.affine}")
    except Exception as e:
        print(f"MONAI Load failed: {e}")
        return

    # 3. Check Orientation Transform result
    tf_ras = Orientation(axcodes="RAS")
    
    custom_ras = tf_ras(custom_mt)
    monai_ras = tf_ras(monai_mt)
    
    print(f"\n--- After Orientation(RAS) ---")
    print(f"Custom Shape: {custom_ras.shape}")
    print(f"MONAI Shape:  {monai_ras.shape}")
    
    # Check if they are nearly equal
    if custom_ras.shape == monai_ras.shape:
        mse = torch.mean((custom_ras - monai_ras)**2).item()
        print(f"MSE between volumes: {mse:.6f}")
    else:
        print(f"Shape Mismatch after RAS orientation!")

if __name__ == "__main__":
    compare_loaders()
