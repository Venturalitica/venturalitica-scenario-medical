import pydicom
from pathlib import Path
import numpy as np

def inspect_series_z_spacing(series_dir, name="Series"):
    files = sorted(list(series_dir.rglob("*.dcm")))
    if not files:
        print(f"  ⚠️ No files in {series_dir}")
        return

    # Extract Z positions
    z_positions = []
    valid_files = []
    
    first_ds = None

    for f in files:
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True)
            if hasattr(ds, "ImagePositionPatient"):
                z = float(ds.ImagePositionPatient[2])
                z_positions.append((z, f))
                valid_files.append(f)
                if first_ds is None: first_ds = ds
        except Exception:
            pass
            
    if not z_positions:
        print(f"  ⚠️ No valid ImagePositionPatient z-coords found in {len(files)} files.")
        return

    # Sort by Z
    z_positions.sort(key=lambda x: x[0])
    zs = [z for z, f in z_positions]
    
    print(f"  ℹ️ {name}: Found {len(zs)} slices. Range Z: {min(zs):.2f} to {max(zs):.2f}")
    
    if len(zs) > 1:
        diffs = np.diff(zs)
        mean_spacing = np.mean(diffs)
        std_spacing = np.std(diffs)
        min_spacing = np.min(diffs)
        max_spacing = np.max(diffs)
        
        print(f"     Spacing: Mean={mean_spacing:.4f}, Std={std_spacing:.4f}")
        print(f"     Min Spacing={min_spacing:.4f}, Max Spacing={max_spacing:.4f}")
        
        # Check for non-uniformity
        if std_spacing > 0.1: # 0.1 mm tolerance
             print(f"     ❌ VARIABLE SPACING DETECTED! (Std > 0.1)")
        
        # Check for potential duplicates (spacing ~ 0)
        if min_spacing < 0.01:
             print(f"     ❌ DUPLICATE ZIP POSITIONS DETECTED! (Min < 0.01)")
             
    # Return sorted file list
    return [f for z, f in z_positions]

def inspect_patient(patient_dir):
    pid = patient_dir.name
    inner_dir = patient_dir / pid 
    if not inner_dir.exists():
        inner_dir = patient_dir

    print(f"\nPATIENT {pid}")
    
    ct_dirs = list(inner_dir.glob("CT-*"))
    if ct_dirs:
         inspect_series_z_spacing(ct_dirs[0], "CT")
    else:
         # Try root
         inspect_series_z_spacing(inner_dir, "Fallback CT")

def main():
    root = Path("../../../venturalitica-sdk-samples-extra/scenarios/surgery-dicom-tcia/data/dicom").resolve()
    # Iterate all directories
    patients = sorted([p for p in root.iterdir() if p.is_dir()])
    print(f"Scanning {len(patients)} patients for Z-alignment...")
    for p in patients:
        inspect_patient(p)

if __name__ == "__main__":
    main()
