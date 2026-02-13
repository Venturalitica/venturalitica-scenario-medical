import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from pathlib import Path

def create_spine_audit_panel(ct_vol, spine_prob, real_gt_vol, patient_id, debug_dir):
    """
    Generates a 3x3 Clinical Audit Panel (Axial, Coronal, Sagittal).
    
    Args:
        ct_vol (np.ndarray): 3D Volume of CT Image [H, W, D]
        spine_prob (np.ndarray): 3D Volume of Spine Probability [H, W, D]
        real_gt_vol (np.ndarray | None): 3D Volume of Ground Truth Mask [H, W, D] (Optional)
        patient_id (str): Patient ID for filenames/titles.
        debug_dir (Path): Output directory.
    
    Returns:
        Path: Saved image path.
    """
    try:
        # Strictly enforce 3D shapes
        if ct_vol.ndim != 3:
            raise ValueError(f"CT Volume must be 3D [H, W, D], got {ct_vol.shape}")
        if spine_prob.ndim != 3:
            raise ValueError(f"Prob Volume must be 3D [H, W, D], got {spine_prob.shape}")
        if real_gt_vol is not None and real_gt_vol.ndim != 3:
            raise ValueError(f"GT Volume must be 3D [H, W, D], got {real_gt_vol.shape}")

        spine_mask = (spine_prob > 0.5).astype(np.float32)

        # Find spatial center of detection using Largest Connected Component
        # This focuses the view on the actual spine, not empty space
        mask_np = spine_mask # Already verified 3D
        labeled_mask, num_labels = ndimage.label(mask_np)
        
        # Determine centering target
        # Preference: GT > Pred
        if real_gt_vol is not None and real_gt_vol.max() > 0:
             centering_mask = real_gt_vol
             labeled_mask, num_labels = ndimage.label(centering_mask)
        else:
             centering_mask = mask_np
             labeled_mask, num_labels = ndimage.label(centering_mask)

        # Get COM or Center
        H, W, D = centering_mask.shape
        if num_labels > 0:
            label_sizes = np.bincount(labeled_mask.ravel())
            largest_label = label_sizes[1:].argmax() + 1
            com = ndimage.center_of_mass(centering_mask, labeled_mask, largest_label)
            best_x, best_y, best_z = int(com[0]), int(com[1]), int(com[2])
        else:
            best_x, best_y, best_z = H//2, W//2, D//2
        
        # Robust Bounds Check
        best_x = max(0, min(best_x, H - 1))
        best_y = max(0, min(best_y, W - 1))
        best_z = max(0, min(best_z, D - 1))
        
        print(f"      DEBUG [Viz]: Center Coord: ({best_x}, {best_y}, {best_z}) on Volume Shape: ({H}, {W}, {D})")
        
        fig, axes = plt.subplots(3, 3, figsize=(18, 18))
        
        # Convenience wrappers
        mask_vol = spine_mask
        gt_vol = real_gt_vol

        # Slice extraction for each view
        # Note: Medical images are typically (H, W, D). imshow uses (Y, X). often need transpose .T
        
        axial_view = (ct_vol[:, :, best_z], mask_vol[:, :, best_z], spine_prob[:, :, best_z], 
                      gt_vol[:, :, best_z] if gt_vol is not None else None, "Axial", best_z)
        
        coronal_view = (ct_vol[:, best_y, :], mask_vol[:, best_y, :], spine_prob[:, best_y, :], 
                        gt_vol[:, best_y, :] if gt_vol is not None else None, "Coronal", best_y)
        
        sagittal_view = (ct_vol[best_x, :, :], mask_vol[best_x, :, :], spine_prob[best_x, :, :], 
                         gt_vol[best_x, :, :] if gt_vol is not None else None, "Sagittal", best_x)

        views = [axial_view, coronal_view, sagittal_view]

        for col, (img, pred, prob, gt, title, idx) in enumerate(views):
            aspect = 'auto' if title != "Axial" else 'equal'
            
            # Match Orientation of Debug Script:
            # Axial: No Transpose (H, W) -> (Y, X)
            # Coronal: Transpose (H, D) -> (D, H)
            # Sagittal: Transpose (W, D) -> (D, W)
            
            if title != "Axial":
                img = img.T
                pred = pred.T
                prob = prob.T
                if gt is not None: gt = gt.T

            # --- ROW 0: Ground Truth (Green Overlay) ---
            ax = axes[0, col]
            ax.set_title(f"GT - {title} ({idx})")
            ax.imshow(img, cmap='gray', vmin=-0.8, vmax=0.8, origin='lower', aspect=aspect)
            
            if gt is not None:
                if gt.max() > 0:
                    m_gt = np.zeros((*gt.shape, 4))
                    m_gt[gt > 0] = [0, 1, 0, 0.5] # Green
                    ax.imshow(m_gt, origin='lower', aspect=aspect)
            else:
                ax.text(10, 20, "No GT Mask found", color='yellow', fontweight='bold')

            # --- ROW 1: Prediction (Red Overlay) ---
            ax = axes[1, col]
            ax.set_title(f"Pred - {title} ({idx})")
            ax.imshow(img, cmap='gray', vmin=-0.8, vmax=0.8, origin='lower', aspect=aspect)
            if pred.max() > 0:
                m_pred = np.zeros((*pred.shape, 4))
                m_pred[pred > 0] = [1, 0, 0, 0.5] # Red
                ax.imshow(m_pred, origin='lower', aspect=aspect)

            # --- ROW 2: Error (Heatmap) ---
            ax = axes[2, col]
            ax.set_title(f"Diff/Error - {title}")
            
            if gt is not None:
                # Absolute Difference Heatmap
                error_data = np.abs(pred - gt)
                im = ax.imshow(error_data, cmap='inferno', origin='lower', aspect=aspect)
            else:
                # Fallback: Probability Map (Uncertainty Proxy)
                error_data = prob
                im = ax.imshow(error_data, cmap='hot', origin='lower', aspect=aspect, vmin=0, vmax=1)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()
        save_path = debug_dir / f"spine_audit_3x3_{patient_id}.png"
        plt.savefig(save_path)
        plt.close()
        print(f"      📸 Clinical 3x3 Audit Panel saved: {save_path}")
        return save_path
        
    except Exception as e:
        print(f"      ⚠️ Visualization Error: {e}")
        import traceback
        traceback.print_exc()
        return None
