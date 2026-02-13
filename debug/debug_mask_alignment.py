import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import torch
import pydicom
from pathlib import Path
from tqdm import tqdm

# MONAI Imports
from monai.transforms import (
    Compose, Orientation, Spacing, LoadImage, ResampleToMatch
)
from monai.data import MetaTensor

# Local Imports
# Ensure we can import from local modules
sys.path.append(str(Path(__file__).parent))
from dicom_utils import load_dicom_volume_robust

# Configuration
DATA_ROOT = Path("../../../venturalitica-sdk-samples-extra/scenarios/surgery-dicom-tcia/data/dicom").resolve()
DEBUG_OUTPUT_DIR = Path("debug_outputs/alignment_checks")
DEBUG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def find_ct_and_seg_files(patient_dir):
    """
    Heuristic to separate CT series from SEG series/files.
    Assumes CT series are in folders NOT named 'SEG'.
    Assumes SEG files might be in folders named 'SEG' or have 'SEG' in filename.
    """
    ct_files = []
    seg_candidates = []

    # Recursive search
    all_files = sorted(list(patient_dir.rglob("*.dcm")))
    
    for f in all_files:
        is_seg = "SEG" in str(f).upper() or "SEG" in f.parent.name.upper()
        if is_seg:
            seg_candidates.append(f)
        else:
            ct_files.append(f)
            
    # If no dcm segs, check for nii/nii.gz? (Not for this specific dataset, but good practice)
    # For this task, we focus on the provided request context which implies DICOMs.
    
    return ct_files, seg_candidates

def get_largest_connected_component_centroid(mask_vol):
    """
    Finds the centroid of the largest connected component in a binary mask.
    Returns (z, y, x) coordinates (int).
    """
    if mask_vol.sum() == 0:
        return None
        
    labeled_mask, num_labels = ndimage.label(mask_vol > 0.5)
    if num_labels == 0:
        return None
    
    # Bincount to find largest label (skip 0=background)
    sizes = np.bincount(labeled_mask.ravel())
    largest_label = sizes[1:].argmax() + 1
    
    # Center of Mass
    com = ndimage.center_of_mass(mask_vol, labeled_mask, largest_label)
    return tuple(map(int, com))

def plot_mpr(ct_vol, gt_vol, patient_id):
    """
    Generates and saves a 3x3 MPR plot.
    ct_vol, gt_vol: 3D numpy arrays [H, W, D] (Y, X, Z)
    """
    # Calculate Centroid
    centroid = get_largest_connected_component_centroid(gt_vol)
    
    if centroid is None:
        print(f"      ⚠️ No mask content found for {patient_id}. Centering on volume center.")
        H, W, D = ct_vol.shape
        cx, cy, cz = H//2, W//2, D//2
    else:
        cx, cy, cz = centroid
        # Clamp
        cx = max(0, min(cx, ct_vol.shape[0]-1))
        cy = max(0, min(cy, ct_vol.shape[1]-1))
        cz = max(0, min(cz, ct_vol.shape[2]-1))
        
    print(f"      📍 Centering view at: ({cx}, {cy}, {cz})")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Patient {patient_id} - Mask Alignment Check", fontsize=16)

    # Views (Assuming image is Y, X, Z)
    # Dim 0 = Y (Rows/Height)
    # Dim 1 = X (Cols/Width)
    # Dim 2 = Z (Slice/Depth)
    
    # 1. Axial (XY plane, constant Z)
    # Shows dim0 (Y) vs dim1 (X).
    ax = axes[0]
    ax.set_title(f"Axial (Z={cz})")
    # imshow expects (Rows, Cols). Here (Y, X). 
    # Usually X is horiz, Y is vert.
    # To get correct orientation, usually need origin='lower' + potentially transpose if stored as (X,Y)
    # If stored as (Y, X), imshow displays Y as vertical.
    ax.imshow(ct_vol[:, :, cz], cmap='gray', origin='lower', aspect='equal')
    if gt_vol[:, :, cz].max() > 0:
        mask_slice = np.zeros((*gt_vol[:, :, cz].shape, 4))
        mask_slice[gt_vol[:, :, cz] > 0] = [0, 1, 0, 0.4] 
        ax.imshow(mask_slice, origin='lower', aspect='equal')

    # 2. Coronal (XZ plane, constant Y)
    # Coronal View: Frontal. Fixed Y (Dim 1).
    ax = axes[1]
    ax.set_title(f"Coronal (Y={cy})")
    ax.imshow(ct_vol[:, cy, :].T, cmap='gray', origin='lower', aspect='auto')
    if gt_vol[:, cy, :].max() > 0:
        mask_slice = np.zeros((*gt_vol[:, cy, :].T.shape, 4))
        mask_slice[gt_vol[:, cy, :].T > 0] = [0, 1, 0, 0.4]
        ax.imshow(mask_slice, origin='lower', aspect='auto')

    # 3. Sagittal (YZ plane, constant X)
    # Sagittal View: Side. Fixed X (Dim 0).
    ax = axes[2]
    ax.set_title(f"Sagittal (X={cx})")
    ax.imshow(ct_vol[cx, :, :].T, cmap='gray', origin='lower', aspect='auto')
    if gt_vol[cx, :, :].max() > 0:
        mask_slice = np.zeros((*gt_vol[cx, :, :].T.shape, 4))
        mask_slice[gt_vol[cx, :, :].T > 0] = [0, 1, 0, 0.4]
        ax.imshow(mask_slice, origin='lower', aspect='auto')

    save_path = DEBUG_OUTPUT_DIR / f"{patient_id}_alignment.png"
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"      📸 Saved visualization to {save_path}")

def build_seg_affine(seg_ds, ct_affine):
    """
    Constructs the Affine Matrix for the Reconstructed Segmentation.
    Hybrid approach:
    - X/Y (In-Plane): Uses SEG's ImageOrientationPatient and PixelSpacing.
    - Z   (Through):  Uses CT's Z-direction and Spacing (since we aligned indices).
    - Origin:         Uses SEG's X/Y Origin, but CT's Z Origin (since index 0 = CT start).
    """
    # 1. Get SEG In-Plane Meta
    # Multi-frame SEG commonly puts this in SharedFunctionalGroupsSequence
    shared = getattr(seg_ds, "SharedFunctionalGroupsSequence", None)
    
    # Defaults
    iop = [1, 0, 0, 0, 1, 0] # Identity
    spacing = [1.0, 1.0]
    origin = [0.0, 0.0, 0.0]
    
    # Extract from Shared Groups if available
    if shared and len(shared) > 0:
        grp = shared[0]
        if 'PlaneOrientationSequence' in grp:
            iop = grp.PlaneOrientationSequence[0].ImageOrientationPatient
        if 'PixelMeasuresSequence' in grp:
            spacing = grp.PixelMeasuresSequence[0].PixelSpacing
            
    # Or check PerFrame for origin of Frame 0 (if valid)
    per_frame = getattr(seg_ds, "PerFrameFunctionalGroupsSequence", None)
    if per_frame and len(per_frame) > 0:
        if 'PlanePositionSequence' in per_frame[0]:
            origin = per_frame[0].PlanePositionSequence[0].ImagePositionPatient
            
    # If standard tags exist at top level (sometimes in single-frame converted)
    if hasattr(seg_ds, "ImageOrientationPatient"): iop = seg_ds.ImageOrientationPatient
    if hasattr(seg_ds, "PixelSpacing"): spacing = seg_ds.PixelSpacing
    if hasattr(seg_ds, "ImagePositionPatient"): origin = seg_ds.ImagePositionPatient
    
    # 2. Extract Components
    rx, ry, rz, cx, cy, cz = iop
    row_spacing, col_spacing = spacing
    ox, oy, oz = origin
    
    # 3. Build Matrices
    # CT Affine allows us to extract Z-vector and Z-spacing/origin
    # ct_affine is 4x4 tensor/numpy
    ct_aff = ct_affine if isinstance(ct_affine, np.ndarray) else ct_affine.numpy()
    
    # CT Z-Vector (Column 2)
    # [ zx*sz, zy*sz, zz*sz, 0 ]
    z_vec = ct_aff[:3, 2]
    
    # CT Z-Origin (Component 2 of Column 3)
    # If we mapped index 0 of SEG to index 0 of CT, Z-origin should match CT.
    ct_z_origin = ct_aff[2, 3] # Z-coord
    
    # 4. Construct New Affine
    mat = np.eye(4)
    
    # Col 0: Index 0 (Rows) -> 'c' vector (down) * RowSpacing? 
    # WAIT. Standard DICOM: RowDir is vector of Row Increase? No, RowDir is vector ALONG the Row (across columns).
    # "The first three values are the direction cosine of the first row (Top row, Left to Right)" -> This is Column Index Increase direction.
    # "The next three are the direction cosine of the first column (Left column, Top to Bottom)" -> This is Row Index Increase direction.
    # So IOP[0:3] = X_dir (Col 1 in affine), IOP[3:6] = Y_dir (Col 0 in affine). (Assuming coords x,y,z)
    
    # IOP = [Rx, Ry, Rz, Cx, Cy, Cz]
    # R = Direction of Row (across cols) -> Corresponds to Affine Column 1 (J index)
    # C = Direction of Col (down rows)   -> Corresponds to Affine Column 0 (I index)
    
    # Col 0 (I form): Vector C * RowSpacing (Vertical spacing)
    mat[0, 0] = cx * row_spacing
    mat[1, 0] = cy * row_spacing
    mat[2, 0] = cz * row_spacing
    
    # Col 1 (J form): Vector R * ColSpacing (Horizontal spacing)
    mat[0, 1] = rx * col_spacing
    mat[1, 1] = ry * col_spacing
    mat[2, 1] = rz * col_spacing
    
    # Col 2 (K form): Matches CT Z-vector exactly
    mat[0, 2] = z_vec[0]
    mat[1, 2] = z_vec[1]
    mat[2, 2] = z_vec[2]
    
    # Col 3 (Origin): Uses SEG X/Y, CT Z
    mat[0, 3] = ox
    mat[1, 3] = oy
    mat[2, 3] = ct_z_origin # Override Z origin with CT's to ensure slice 0 matches
    
    return mat

def load_dicom_seg_reconstructed(seg_path, ct_files, target_shape, ct_affine):
    """
    Manually reconstructs a Segmentation Volume from a DICOM SEG file.
    Returns:
        vol_tensor: (1, H, W, D) aligned segmentation mask (raw spacing).
        affine: 4x4 numpy array (constructed)
        seg_affine: 4x4 numpy array (constructed)
    """
    import pydicom
    
    print(f"      DEBUG: Reconstructing SEG from {seg_path}...")
    
    # 0. Ensure CT Files are Sorted by Z
    from dicom_utils import sort_dicom_files
    ct_files = sort_dicom_files(ct_files)

    # 1. Map CT Ids to Meta (Indices and Origin/Orientation)
    sop_to_meta = {}
    for idx, f in enumerate(ct_files):
        try:
            d = pydicom.dcmread(f, stop_before_pixels=True)
            sop_to_meta[d.SOPInstanceUID] = {
                'idx': idx,
                'ipp': np.array(d.ImagePositionPatient, dtype=float),
                'iop': np.array(d.ImageOrientationPatient, dtype=float),
                'rows': d.Rows,
                'cols': d.Columns
            }
        except Exception:
            pass
            
    print(f"      DEBUG: Indexed {len(sop_to_meta)} CT slices.")
    
    # 2. Read SEG
    seg_ds = pydicom.dcmread(seg_path)
    
    # 2.5 Build Affine
    seg_affine = build_seg_affine(seg_ds, ct_affine)
    print(f"      DEBUG: Constructed Hybrid SEG Affine:\n{seg_affine}")
    
    rows = seg_ds.Rows
    cols = seg_ds.Columns
    num_frames = getattr(seg_ds, "NumberOfFrames", 1)
    
    # Depth derived from CT
    D_ct = target_shape[-1]
    
    # Alloc reconstruction buffer [H_seg, W_seg, D_ct]
    recon_vol = np.zeros((rows, cols, D_ct), dtype=np.uint8)
    
    # 3. Iterate Frames and Check Alignment
    per_frame = getattr(seg_ds, "PerFrameFunctionalGroupsSequence", None)
    pixel_array = seg_ds.pixel_array # (Frames, H, W)
    
    matched_frames = 0
    diff_ipp_list = []
    diff_iop_list = []
    
    for i in range(num_frames):
        # Find Ref UDP
        ref_uid = None
        current_frame_ipp = None
        current_frame_iop = None
        
        # Check PerFrame
        if per_frame and i < len(per_frame):
             # DerivationImageSequence -> SourceImageSequence -> ReferencedSOPInstanceUID
             grp = per_frame[i]
             if 'DerivationImageSequence' in grp:
                 for deriv in grp.DerivationImageSequence:
                     if 'SourceImageSequence' in deriv:
                         for src in deriv.SourceImageSequence:
                             if 'ReferencedSOPInstanceUID' in src:
                                 ref_uid = src.ReferencedSOPInstanceUID
                                 break
             if 'PlanePositionSequence' in grp:
                 current_frame_ipp = np.array(grp.PlanePositionSequence[0].ImagePositionPatient, dtype=float)
             # IOP is usually shared, but check per-frame first if it exists
             if 'PlaneOrientationSequence' in grp: 
                 current_frame_iop = np.array(grp.PlaneOrientationSequence[0].ImageOrientationPatient, dtype=float)
        
        # Check Shared if per-frame IOP missing
        if current_frame_iop is None and hasattr(seg_ds, "SharedFunctionalGroupsSequence"):
             shared = seg_ds.SharedFunctionalGroupsSequence[0]
             if 'PlaneOrientationSequence' in shared:
                 current_frame_iop = np.array(shared.PlaneOrientationSequence[0].ImageOrientationPatient, dtype=float)

        if ref_uid and ref_uid in sop_to_meta:
            ct_meta = sop_to_meta[ref_uid]
            z_idx = ct_meta['idx']
            
            # Compare Metadata
            if current_frame_ipp is not None:
                diff_ipp_list.append(current_frame_ipp - ct_meta['ipp'])
            if current_frame_iop is not None:
                diff_iop_list.append(np.abs(current_frame_iop - ct_meta['iop'])) # Abs diff
            
            # Accumulate mask (Logical OR if overlap)
            mask_slice = pixel_array[i]
            if mask_slice.max() > 0:
                 recon_vol[:, :, z_idx] = np.maximum(recon_vol[:, :, z_idx], mask_slice)
            matched_frames += 1
            
    # 5. Convert to Tensor
    vol_tensor = torch.tensor(recon_vol.astype(np.float32)).unsqueeze(0) # (1, H_s, W_s, D_ct)
    
    # Resizing - Wait!
    # If we constructed the affine to match rows/cols, we should NOT resize blindly.
    # The 'Spacing' transform will handle the resampling from H_s, W_s to 1mm.
    # So we return the raw tensor size.
    # if vol_tensor.shape != target_shape:
    #     print(f"      DEBUG: Resizing in-plane from {vol_tensor.shape} to {target_shape}")
    #     vol_tensor = torch.nn.functional.interpolate(vol_tensor, size=target_shape[1:], mode='nearest')
        
    # For now, we return the constructed affine.
    # The instruction mentioned "suggested_flip" but the provided snippet was incomplete.
    # We will stick to returning the tensor and the affine as per the original function signature,
    # and let the calling function decide how to use the debug info.
    return vol_tensor

def auto_align_orientation(ct_tensor, seg_tensor):
    """
    Heuristic Alignment: Defines the best orientation by maximizing overlap 
    between the Segmentation Mask and Bone-like structures in CT.
    
    Args:
        ct_tensor (Tensor): (1, H, W, D) aligned CT data.
        seg_tensor (Tensor): (1, H, W, D) candidate SEG data.
        
    Returns:
        best_seg_tensor (Tensor): Optimally oriented SEG.
    """
    # Bone Threshold (Hounsfield Units). 
    # CT is usually rescaled. If raw meta tensor was loaded via MONAI, might be shifted?
    # Assume generic HU range. Bone ~ > 200.
    
    # We work on CPU/Numpy for check
    ct_np = ct_tensor[0].cpu().numpy()
    seg_np = seg_tensor[0].cpu().numpy()
    
    # Generate Thresholded Bone Mask
    # Safe check: if CT is not HU, this might fail. 
    # But usually medical CT is HU.
    bone_mask = (ct_np > 250).astype(np.float32)
    
    transforms_to_try = [
        ([], "Original"),
        ([0], "Flip Y (Height)"),
        ([1], "Flip X (Width)"),
        ([0, 1], "Flip XY (Both)")
    ]
    
    best_score = -1.0
    best_seg = seg_tensor
    best_name = "Original"
    
    print("      DEBUG: Running Heuristic Orientation Alignment (Bone Overlap)...")
    
    for axes, name in transforms_to_try:
        if not axes:
            candidate = seg_np
        else:
            candidate = np.flip(seg_np, axis=axes)
            
        # Compute Overlap Fraction: (Mask & Bone) / Mask
        # How much of the mask lands on bone?
        mask_sum = candidate.sum()
        if mask_sum == 0:
            score = 0
        else:
            overlap = (candidate * bone_mask).sum()
            score = overlap / mask_sum
            
        print(f"      - {name}: Overlap Score = {score:.4f} (Pixels: {mask_sum})")
        
        if score > best_score:
            best_score = score
            # Re-create tensor for the best one to ensure gradients/device if needed (though here just data)
            # We flipped numpy, need to put back to Tensor
            best_seg = torch.tensor(candidate.copy()).unsqueeze(0)
            best_name = name
            
    print(f"      ✅ Selected Orientation: {best_name} (Score: {best_score:.4f})")
    
    if best_score < 0.3:
        print("      ⚠️ WARNING: Low bone overlap. Threshold might be wrong or mask is not bone.")
        
    return best_seg

def process_patient(p_dir):
    pid = p_dir.name
    print(f"\n🔍 Processing Patient: {pid}")

    ct_files, seg_files = find_ct_and_seg_files(p_dir)
    
    if not ct_files:
        print("      ⚠️ No CT files found. Skipping.")
        return
        
    if not seg_files:
        print("      ⚠️ No Segmentation files found. Skipping.")
        return
        
    print(f"      Found {len(ct_files)} CT files and {len(seg_files)} SEG files.")

    try:
        # 1. Load CT Volume (Raw, no transforms yet)
        # Note: load_dicom_volume_robust sorts the files internally to build the volume,
        # but we need the SORTED file list to map indices for the SEG.
        from dicom_utils import sort_dicom_files
        ct_files_sorted = sort_dicom_files(ct_files)
        
        # We assume load_dicom_volume_robust uses the same logic. 
        # Ideally we'd pass the sorted list to it, but it takes paths.
        raw_ct_meta_tensor = load_dicom_volume_robust(ct_files_sorted)
        print(f"      DEBUG: Raw CT Shape: {raw_ct_meta_tensor.shape}")

        # 2. Reconstruct GT Mask in RAW Space (matching original Z-indices)
        try:
             seg_file = seg_files[0]
             
             # Reconstruct using RAW CT shape
             # We passed ct_affine so the loader *could* use it, but we are ignoring the returned affine for now to inspect diffs
             raw_seg_tensor = load_dicom_seg_reconstructed(seg_file, ct_files_sorted, target_shape=raw_ct_meta_tensor.shape, ct_affine=raw_ct_meta_tensor.affine)
             print(f"      DEBUG: Raw Reconstructed SEG Shape: {raw_seg_tensor.shape}")
             
             # HEURISTIC ALIGNMENT: Optimize Orientation against Bone
             # This bypasses tricky metadata logic by anchoring to physics
             raw_seg_tensor = auto_align_orientation(raw_ct_meta_tensor, raw_seg_tensor)
             
             # Wrap in MetaTensor to preserve Affine for transforms
             # We use CT affine. If it's flipped, we'll see it in diff logs and apply Flip later.
             from monai.data import MetaTensor
             raw_seg_meta_tensor = MetaTensor(
                 raw_seg_tensor, 
                 affine=raw_ct_meta_tensor.affine, 
                 meta=raw_ct_meta_tensor.meta
             )
             
        except Exception as e:
             print(f"      ❌ Custom SEG Loading Failed: {e}")
             import traceback
             traceback.print_exc()
             return
             
        # 3. Apply Transforms to BOTH (Sync)
        # We need a consistent coordinate system for visualization
        # Use bilinear for CT, nearest for SEG
        
        ct_transforms = Compose([
            Orientation(axcodes="RAS"),
            Spacing(pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        ])
        
        seg_transforms = Compose([
            Orientation(axcodes="RAS"),
            Spacing(pixdim=(1.0, 1.0, 1.0), mode="nearest"),
        ])
        
        print("      DEBUG: Resampling CT to RAS 1mm...")
        ct_vol_ras = ct_transforms(raw_ct_meta_tensor)
        print(f"      CT Shape (RAS, 1mm): {ct_vol_ras.shape}")

        print("      DEBUG: Resampling SEG to RAS 1mm...")
        aligned_seg = seg_transforms(raw_seg_meta_tensor)
        print(f"      SEG Shape (RAS, 1mm): {aligned_seg.shape}")

        print(f"      DEBUG: Aligned SEG Non-zero count: {(aligned_seg > 0).sum()}")
        
        # 4. Convert to Numpy for Plotting
        ct_np = ct_vol_ras[0].cpu().numpy() # [H, W, D]
        seg_np = aligned_seg[0].cpu().numpy() # [H, W, D]
        
        # Binarize
        seg_np = (seg_np > 0.5).astype(np.float32)
        
        # 5. Visualize
        plot_mpr(ct_np, seg_np, pid)

    except Exception as e:
        print(f"      ❌ Error processing patient {pid}: {e}")
        import traceback
        traceback.print_exc()

        
        # 4. Convert to Numpy for Plotting
        ct_np = ct_vol_ras[0].cpu().numpy() # [H, W, D] (assuming channel 0)
        seg_np = aligned_seg[0].cpu().numpy() # [H, W, D]
        
        # Binarize
        seg_np = (seg_np > 0.5).astype(np.float32)
        
        # 5. Visualize
        plot_mpr(ct_np, seg_np, pid)

    except Exception as e:
        print(f"      ❌ Error processing patient {pid}: {e}")
        import traceback
        traceback.print_exc()

def main():
    if not DATA_ROOT.exists():
        print(f"❌ Data root not found: {DATA_ROOT}")
        return

    patients = sorted([p for p in DATA_ROOT.iterdir() if p.is_dir()])
    print(f"🚀 Starting Mask Alignment Debugger on {len(patients)} patients...")
    print(f"📂 Output Directory: {DEBUG_OUTPUT_DIR}")

    for p in patients:
        process_patient(p)

if __name__ == "__main__":
    main()
