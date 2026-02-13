# 01_model_evaluation.py
# Refactored for Modularity and Robustness

import warnings
warnings.filterwarnings("ignore") # Suppress FutureWarnings from MONAI/Torch

import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from monai.bundle import download
from monai.networks.nets import SegResNet
from monai.inferers import SlidingWindowInferer
from monai.transforms import (
    Compose, 
    LoadImage, 
    EnsureChannelFirst, 
    Orientation, 
    Spacing,
    ResizeWithPadOrCrop,
    NormalizeIntensity,
    ScaleIntensity,
    ToTensor,
    ThresholdIntensity,
    GaussianSmooth
)
from monai.data import MetaTensor

# Custom Utilities
from dicom_utils import load_dicom_volume_robust, get_annotated_spine_indices
from viz_utils import create_spine_audit_panel

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BUNDLE_NAME = "wholeBody_ct_segmentation"
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

class SpineEvaluator:
    def __init__(self):
        self.device = DEVICE
        print(f"⚙️ Initializing Evaluator on {self.device}...")
        
        # Download Real Model
        if not (MODEL_DIR / BUNDLE_NAME).exists():
            print(f"⬇️ Downloading MONAI Bundle: {BUNDLE_NAME}...")
            download(name=BUNDLE_NAME, bundle_dir=MODEL_DIR)
        
        # Load Model
        # This model uses SegResNet architecture (from inference.json)
        self.net = SegResNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=105, # Total classes
            init_filters=32,
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            dropout_prob=0.2,
        ).to(self.device)
        
        # Load Weights
        weights_path = MODEL_DIR / f"{BUNDLE_NAME}/models/model_lowres.pt" # Or model.pt for highres
        # Actually standard bundle usually has model.pt
        weights_path = MODEL_DIR / f"{BUNDLE_NAME}/models/model.pt"
        if not weights_path.exists():
            # Fallback
            weights_path = MODEL_DIR / f"{BUNDLE_NAME}/models/model_lowres.pt"
            
        print(f"✅ Real Model Loaded: {weights_path.name}")
        self.net.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.net.eval()

        # Sliding Window Inferer
        self.inferer = SlidingWindowInferer(
            roi_size=(96, 96, 96), 
            sw_batch_size=1, 
            overlap=0.25,
            mode="gaussian", # Smoother blending
            device="cpu", 
            sw_device="cuda" 
        )
        
        # Debug Output Dir
        self.debug_dir = Path("debug_outputs")
        self.debug_dir.mkdir(exist_ok=True)

        
        # CPU Transforms (Synchronized with bundle's inference.json)
        self.cpu_transforms = Compose([
            # The bundle uses NormalizeIntensity with nonzero=True
            # followed by ScaleIntensity to [-1, 1]
            NormalizeIntensity(nonzero=True),
            ScaleIntensity(minv=-1.0, maxv=1.0),
            ToTensor()
        ])

    def run_inference(self, dicom_input, patient_id="unknown", gt_dice=None, gt_seg_input=None):
        """Run real inference on a DICOM series (list of files)."""
        torch.cuda.empty_cache()
        try:
            if not dicom_input:
                return None, 0.0

            # 1. Load & Resample (CPU) via Robust Loader
            print(f"      DEBUG [Image]: Loading {len(dicom_input)} files with robust loader...")
            try:
                raw_meta_tensor = load_dicom_volume_robust(dicom_input)
            except Exception as e:
                print(f"      ❌ Load Error: {e}")
                import traceback
                traceback.print_exc()
                return None, 0.0

            print(f"      DEBUG [Raw]: Shape: {raw_meta_tensor.shape}, Min: {raw_meta_tensor.min():.2f}, Max: {raw_meta_tensor.max():.2f}")

            # Save Raw Affine for SEG reconstruction later
            raw_ct_affine = raw_meta_tensor.affine
            
            # Apply Spatial Transforms (Orientation RAS, Spacing)
            tf_spat = Compose([
                Orientation(axcodes="RAS"),
                Spacing(pixdim=(1.5, 1.5, 1.5), mode="bilinear"),
            ])
            input_data = tf_spat(raw_meta_tensor)
            print(f"      DEBUG [Spatial]: Shape: {input_data.shape}, Min: {input_data.min():.2f}, Max: {input_data.max():.2f}")

            # 2. Intensity Normalization (CPU)
            # Apply CPU transforms (Normalization, Scaling to [-1, 1])
            input_data = self.cpu_transforms(input_data)
            
            print(f"      DEBUG [Intensity]: Min: {input_data.min():.2f}, Max: {input_data.max():.2f}, Mean: {input_data.mean():.2f}")

            input_data = input_data.float()

            # Full spine CT inference
            inputs = input_data.unsqueeze(0) # Add batch dim
            print(f"      DEBUG [Inference]: Model Input Shape: {inputs.shape}")
            
            # Determine which spine classes to include dynamically based on GT
            # Default: Vertebrae (18-41) + Sacrum (92)
            default_indices = list(range(18, 42)) + [92]
            target_spine_indices = default_indices
            
            if gt_seg_input:
                annotated_indices = get_annotated_spine_indices(gt_seg_input[0])
                if annotated_indices:
                    target_spine_indices = annotated_indices
                    print(f"      DEBUG [Filter]: Dynamic Filtering enabled. Classes: {target_spine_indices}")

            def spine_predictor(x):
                # x is the patch (B, 1, 96, 96, 96)
                out = self.net(x) # Logits (B, 105, 96, 96, 96)
                
                # Use Argmax to find winner per-voxel
                pred = torch.argmax(out, dim=1, keepdim=True) # (B, 1, 96, 96, 96)
                
                # Dynamic Spine Mask: Include only requested indices
                mask = torch.zeros_like(pred, dtype=torch.bool)
                for idx in target_spine_indices:
                    mask |= (pred == idx)
                
                return mask.float() # Slav Window averages these (Probability of being 'Spine')

            with torch.no_grad():
                # Inferer returns a 1-channel volume: [SpineProb]
                outputs = self.inferer(inputs, spine_predictor)
            
            # Extract probability map and ensure it's 3D [H, W, D]
            spine_prob = outputs[0, 0, ...]
            print(f"      DEBUG: Input range: [{inputs.min():.2f}, {inputs.max():.2f}]")
            print(f"      DEBUG: Max spine prob: {spine_prob.max().item():.4f}")

            # Calculate metrics
            spine_mask = (spine_prob > 0.5).float()
            vol = int(torch.sum(spine_mask).item())
            # 4. Load & Resample Real GT (if available)
            real_gt_vol = None
            if gt_seg_input:
                try:
                    # Need sorted CT files for mapping
                    from dicom_utils import sort_dicom_files, load_dicom_seg_reconstructed, auto_align_orientation
                    
                    # 1. Sort CT
                    ct_files_sorted = sort_dicom_files(dicom_input)
                    seg_file = gt_seg_input[0] # Assuming first SEG

                    # 2. Reconstruct in RAW space
                    # We pass raw_ct_affine but we ignore the returned affine (if any) because we use heuristic.
                    # Note: load_dicom_seg_reconstructed in utils returns only tensor.
                    raw_seg_tensor = load_dicom_seg_reconstructed(seg_file, ct_files_sorted, target_shape=raw_meta_tensor.shape)
                    
                    # 3. Heuristic Alignment (Bone Overlap)
                    # Note: raw_meta_tensor is the CT in raw space
                    raw_seg_tensor = auto_align_orientation(raw_meta_tensor, raw_seg_tensor)
                    
                    # 4. Wrap in MetaTensor for Transforms
                    # Use raw_ct_affine (captured earlier)
                    raw_seg_meta_tensor = MetaTensor(
                        raw_seg_tensor, 
                        affine=raw_ct_affine, 
                        meta=raw_meta_tensor.meta
                    )
                    
                    # 5. Apply SAME Transforms as CT (Spacing + Orientation)
                    seg_transforms = Compose([
                        Orientation(axcodes="RAS"),
                        Spacing(pixdim=(1.5, 1.5, 1.5), mode="nearest"),
                    ])
                    
                    seg_vol = seg_transforms(raw_seg_meta_tensor)
                    
                    # Ensure binary
                    seg_vol = (seg_vol > 0.5).float()
                    
                    # Handle Shape Mismatch (e.g. padding/cropping)
                    # Force SEG to match CT shape exactly
                    if seg_vol.shape != inputs.shape[2:]:  
                         # Resample to match exact spatial size
                         # Use interpolate for robustness
                         seg_vol = torch.nn.functional.interpolate(
                             seg_vol.unsqueeze(0), 
                             size=inputs.shape[2:], 
                             mode='nearest'
                         ).squeeze(0)

                    real_gt_vol = seg_vol[0] # (H, W, D) Tensor
                    gt_voxels = int(real_gt_vol.sum().item())
                    print(f"      DEBUG: Aligned SEG Shape: {real_gt_vol.shape}, Non-zero (GT): {gt_voxels}")
                    
                except Exception as e:
                    print(f"      ❌ Custom SEG Loading Failed: {e}")
                    import traceback
                    traceback.print_exc()
                    # Fallback or leave as None

            # 5. Calculate Metrics
            # Confidence is the mean of the probability map where the mask is 1
            confidence = float(spine_prob[spine_mask > 0].mean().item()) if vol > 0 else 0.0
            
            dice = 0.0
            jaccard = 0.0
            if real_gt_vol is not None:
                # Binary comparison
                pred_bin = spine_mask.bool()
                gt_bin = real_gt_vol.bool()
                
                intersection = (pred_bin & gt_bin).float().sum().item()
                union = (pred_bin | gt_bin).float().sum().item()
                sum_total = pred_bin.float().sum().item() + gt_bin.float().sum().item()
                
                if sum_total > 0:
                    dice = (2.0 * intersection) / sum_total
                if union > 0:
                    jaccard = intersection / union
                    
                print(f"      📈 Metrics: Dice={dice:.4f} | Jaccard={jaccard:.4f}")

            # 6. Save High-Fidelity 3x3 Diagnostic Panel
            # Standardize all volumes to 3D [H, W, D] (Numpy) for Viz Utils
            ct_vol_np = inputs[0, 0].cpu().numpy() # [H, W, D]
            spine_prob_np = spine_prob.cpu().numpy() # [H, W, D]
            real_gt_vol_np = real_gt_vol.cpu().numpy() if real_gt_vol is not None else None # [H, W, D] OR None

            _ = create_spine_audit_panel(
                ct_vol=ct_vol_np,
                spine_prob=spine_prob_np,
                real_gt_vol=real_gt_vol_np,
                patient_id=str(patient_id),
                debug_dir=self.debug_dir
            )

            return vol, confidence, dice, jaccard
            
        except Exception as e:
            print(f"      ⚠️ Batch Error: {e}")
            import traceback
            traceback.print_exc()
            return None, 0.0, 0.0, 0.0

    def evaluate_cohort(self, df, limit=None):
        from dicom_utils import find_ct_and_seg_files
        print(f"🚀 Starting Real Inference on {len(df)} patients...")
        
        results = []
        processed_pids = set()
        
        # Check for existing results to resume
        if Path("cohort_results.csv").exists():
            print("🔄 Found existing cohort_results.csv. Resuming...")
            existing_df = pd.read_csv("cohort_results.csv")
            results = existing_df.to_dict('records')
            processed_pids = set(existing_df['PatientID'].astype(str).tolist())
            print(f"⏩ Skipping {len(processed_pids)} already processed patients.")

        count = 0
        for _, row in df.iterrows():
            if limit and count >= limit: break
            
            pid = str(row['PatientID'])
            
            if pid in processed_pids:
                continue

            # We need the Patient Directory to find files robustly
            # The 'FilePaths' in row are just what we found in main(), which might be incomplete or mixed.
            # Let's derive the patient directory from the first file.
            sample_file = Path(row['FilePaths'][0])
            # Robust way: Use the input file's parent's parent... 
            # But let's trust the debug script logic:
            patient_dir = sample_file.parent.parent # Series -> Patient Inner (e.g. 10543/10543)
            
            print(f"   😷 [Patient {len(results)+1}/{len(df)}] Processing {pid}...")
            
            # Use Robust Loader Logic
            ct_files, seg_files = find_ct_and_seg_files(patient_dir)
            if not ct_files:
                print(f"      ⚠️ No CT files found in {patient_dir}")
                continue
                
            print(f"      files found: CT={len(ct_files)}, SEG={len(seg_files)}")
            
            # Check GT consistency
            gt_dice = None
            if len(seg_files) > 0:
                 # Calculate Dice from metadata if available? 
                 # Actually we calculate it in run_inference vs the loaded SEG.
                 pass
            
            # Run Inference
            # We pass ct_files (list) and seg_files (list)
            vol, conf, dice, jaccard = self.run_inference(ct_files, patient_id=pid, gt_dice=gt_dice, gt_seg_input=seg_files)
            
            if vol is not None:
                results.append({
                    "PatientID": pid,
                    "SpineVol": vol,
                    "Confidence": conf,
                    "Dice": dice,
                    "Jaccard": jaccard
                })
                print(f"      ✓ Metrics: Vol={vol} | Dice={dice:.4f} | Jaccard={jaccard:.4f}")
            
            # Incremental save
            pd.DataFrame(results).to_csv("cohort_results.csv", index=False)
            
            count += 1
            
        return pd.DataFrame(results)

def main():
    print("🏥 DICOM Spine Segmentation - Real Model Inference (Refactored)")
    print("==================================================")
    
    # Locate Data
    data_root = Path("../../../venturalitica-sdk-samples-extra/scenarios/surgery-dicom-tcia/data/dicom").resolve()
    if not data_root.exists():
        print(f"❌ Data root not found: {data_root}")
        return

    # Scan Metadata
    patients = []
    target_pid = "15094" # Focused Verification
    
    for p_dir in data_root.iterdir():
        if p_dir.is_dir():
            pid = p_dir.name
                
            # Gather all DICOMs for this patient (simplistic)
            files = sorted(list(p_dir.rglob("*.dcm")))
            input_files = [f for f in files if "SEG" not in str(f).upper() and "seg" not in str(f).lower()]
            
            if input_files:
                patients.append({
                    "PatientID": p_dir.name,
                    "FilePaths": input_files
                })
                
    df = pd.DataFrame(patients)
    print(f"✅ Loaded {len(df)} patients from metadata.")
    
    # Run
    evaluator = SpineEvaluator()
    # Run
    evaluator = SpineEvaluator()
    results_df = evaluator.evaluate_cohort(df, limit=None) 
    
    # Save
    results_df.to_csv("cohort_results.csv", index=False)
    print("\n✅ Cohort Processing Complete. Results saved to cohort_results.csv")

if __name__ == "__main__":
    main()
