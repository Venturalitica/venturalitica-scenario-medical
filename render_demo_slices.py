"""Render three coronal PNGs for the demo deck "sistema-box" slide.

For a single patient (default: 10352), produce 3 identical-frame coronal
PNGs:
  - ct-coronal-<pid>.png        — pure CT, no overlay
  - monai-coronal-<pid>.png     — same CT + MONAI prediction (red)
  - totalseg-coronal-<pid>.png  — same CT + TotalSegmentator prediction (red)

All three PNGs share: same slice index, same crop box, same CT window,
same pixel dimensions. Only the overlay differs.

The script reuses the existing MONAI inference path (base_medical/
model_evaluation.py) and TotalSegmentator path (base_medical/
totalseg_evaluation.py); both pipelines are re-run *without* writing their
own audit artefacts and the in-memory masks are projected into one shared
reference frame (the MONAI RAS+1.5mm grid) so the overlays line up.

Usage:
    .venv/bin/python render_demo_slices.py --patient 10352 \
        --out venturalitica-scenario-medical/demo_slices
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import scipy.ndimage as ndimage
import torch
from monai.bundle import download
from monai.data import MetaTensor
from monai.inferers import SlidingWindowInferer
from monai.networks.nets import SegResNet
from monai.transforms import Compose, NormalizeIntensity, Orientation, ScaleIntensity, Spacing, ToTensor

warnings.filterwarnings("ignore")

# Make base_medical importable
BASE_MEDICAL = Path(__file__).parent / "base_medical"
sys.path.insert(0, str(BASE_MEDICAL))

from dicom_utils import (  # noqa: E402
    auto_align_orientation,
    find_ct_and_seg_files,
    get_annotated_spine_indices,
    load_dicom_seg_reconstructed,
    load_dicom_volume_robust,
    sort_dicom_files,
)
from totalseg_evaluation import (  # noqa: E402
    _VERTEBRA_LABEL_IDS_TS,
    _annotated_ts_labels,
    _dicom_ct_to_nifti,
    _run_totalsegmentator,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BUNDLE_NAME = "wholeBody_ct_segmentation"
SCENARIO_ROOT = Path(__file__).parent
MODEL_DIR = SCENARIO_ROOT / "shared_data" / "models"
DICOM_ROOT = SCENARIO_ROOT / "shared_data" / "dicom"


def load_monai_model() -> tuple[SegResNet, SlidingWindowInferer]:
    if not (MODEL_DIR / BUNDLE_NAME).exists():
        print(f"  ↓ Downloading MONAI bundle: {BUNDLE_NAME}")
        download(name=BUNDLE_NAME, bundle_dir=MODEL_DIR)
    net = SegResNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=105,
        init_filters=32,
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        dropout_prob=0.2,
    ).to(DEVICE)
    weights = MODEL_DIR / BUNDLE_NAME / "models" / "model.pt"
    if not weights.exists():
        weights = MODEL_DIR / BUNDLE_NAME / "models" / "model_lowres.pt"
    net.load_state_dict(torch.load(weights, map_location=DEVICE))
    net.eval()
    inferer = SlidingWindowInferer(
        roi_size=(96, 96, 96),
        sw_batch_size=1,
        overlap=0.25,
        mode="gaussian",
        device="cpu",
        sw_device=str(DEVICE),
    )
    return net, inferer


def run_monai(ct_files: list[Path], seg_files: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (ct_np, monai_pred_np, ras_affine, gt_np_or_zeros), all in
    RAS + 1.5 mm grid."""
    net, inferer = load_monai_model()

    raw = load_dicom_volume_robust([str(p) for p in ct_files])
    raw_affine = raw.affine

    spat = Compose([Orientation(axcodes="RAS"), Spacing(pixdim=(1.5, 1.5, 1.5), mode="bilinear")])
    spatial = spat(raw)
    ras_affine_t = spatial.affine
    cpu_tf = Compose([NormalizeIntensity(nonzero=True), ScaleIntensity(minv=-1.0, maxv=1.0), ToTensor()])
    input_data = cpu_tf(spatial).float()

    # Determine target classes from GT annotation
    default_indices = list(range(18, 42)) + [92]
    target = default_indices
    if seg_files:
        annotated = get_annotated_spine_indices(seg_files[0])
        if annotated:
            target = annotated

    def predictor(x):
        out = net(x)
        pred = torch.argmax(out, dim=1, keepdim=True)
        mask = torch.zeros_like(pred, dtype=torch.bool)
        for idx in target:
            mask |= pred == idx
        return mask.float()

    with torch.no_grad():
        outputs = inferer(input_data.unsqueeze(0), predictor)
    spine_prob = outputs[0, 0].cpu().numpy()
    monai_mask = (spine_prob > 0.5).astype(np.uint8)
    ct_np = input_data[0].cpu().numpy()

    # Ground truth (optional) in RAS+1.5mm grid
    gt_np = np.zeros_like(monai_mask, dtype=np.uint8)
    if seg_files:
        try:
            ct_sorted = sort_dicom_files([str(p) for p in ct_files])
            raw_seg = load_dicom_seg_reconstructed(seg_files[0], ct_sorted, target_shape=raw.shape)
            raw_seg = auto_align_orientation(raw, raw_seg)
            raw_seg_meta = MetaTensor(raw_seg, affine=raw_affine, meta=raw.meta)
            seg_tf = Compose([Orientation(axcodes="RAS"), Spacing(pixdim=(1.5, 1.5, 1.5), mode="nearest")])
            gt_meta = seg_tf(raw_seg_meta)
            gt_arr = (gt_meta > 0.5).float()
            if gt_arr.shape[1:] != monai_mask.shape:
                gt_arr = torch.nn.functional.interpolate(
                    gt_arr.unsqueeze(0), size=monai_mask.shape, mode="nearest"
                ).squeeze(0)
            gt_np = gt_arr[0].cpu().numpy().astype(np.uint8)
        except Exception as exc:
            print(f"  ⚠ GT alignment failed: {exc}")

    ras_affine_np = ras_affine_t.cpu().numpy() if torch.is_tensor(ras_affine_t) else np.asarray(ras_affine_t)
    return ct_np, monai_mask, ras_affine_np, gt_np


def run_totalseg(ct_files: list[Path], seg_files: list[Path]) -> tuple[np.ndarray, "nib.Nifti1Image"]:
    """Returns (ts_binary_mask, ts_nibabel_img) in TotalSegmentator's
    output grid (= raw CT grid)."""
    nifti_path, _ = _dicom_ct_to_nifti([Path(p) for p in ct_files])
    ml_mask, _, _, ml_img = _run_totalsegmentator(nifti_path)

    annotated_monai = get_annotated_spine_indices(seg_files[0]) if seg_files else None
    if annotated_monai:
        ts_subset = _annotated_ts_labels(annotated_monai) or _VERTEBRA_LABEL_IDS_TS
    else:
        ts_subset = _VERTEBRA_LABEL_IDS_TS
    binary = np.isin(ml_mask, ts_subset).astype(np.uint8)
    return binary, ml_img


def resample_ts_to_monai_grid(
    ts_mask: np.ndarray, ts_img: "nib.Nifti1Image", ras_affine: np.ndarray, target_shape: tuple[int, int, int]
) -> np.ndarray:
    """Resample TotalSegmentator binary mask into MONAI's RAS+1.5mm grid."""
    from nibabel.processing import resample_from_to

    ts_nii = nib.Nifti1Image(ts_mask.astype(np.uint8), ts_img.affine)
    target_img = nib.Nifti1Image(np.zeros(target_shape, dtype=np.uint8), ras_affine)
    resampled = resample_from_to(ts_nii, target_img, order=0)
    return np.asarray(resampled.dataobj).astype(np.uint8)


def pick_coronal_index(mask: np.ndarray) -> int:
    """Return Y-index (axis 1) of the COM of the largest connected
    component in mask. Falls back to volume center if empty."""
    if mask.max() == 0:
        return mask.shape[1] // 2
    labelled, n = ndimage.label(mask)
    sizes = np.bincount(labelled.ravel())
    largest = sizes[1:].argmax() + 1
    com = ndimage.center_of_mass(mask, labelled, largest)
    return int(np.clip(round(com[1]), 0, mask.shape[1] - 1))


def compute_crop_bbox(mask: np.ndarray, slice_y: int, padding: int = 25) -> tuple[int, int, int, int]:
    """Return (x0, x1, z0, z1) bounding box of mask on coronal slice with
    padding, clipped to volume. Falls back to full slice if mask empty."""
    coronal = mask[:, slice_y, :]
    if coronal.max() == 0:
        H, _, D = mask.shape
        return 0, H, 0, D
    xs, zs = np.where(coronal > 0)
    x0 = max(0, xs.min() - padding)
    x1 = min(mask.shape[0], xs.max() + padding + 1)
    z0 = max(0, zs.min() - padding)
    z1 = min(mask.shape[2], zs.max() + padding + 1)
    return x0, x1, z0, z1


def render_panel(
    ct_slice: np.ndarray,
    mask_slice: np.ndarray | None,
    out_path: Path,
    size_px: int = 600,
    overlay_color: tuple[float, float, float] = (1.0, 0.20, 0.20),
    overlay_alpha: float = 0.55,
) -> None:
    """Write a single PNG: CT (grayscale) + optional red mask overlay.
    Black background, no axes, fixed pixel size.
    """
    # Transpose so coronal slice displays with anatomical "up" pointing up.
    img = ct_slice.T
    fig = plt.figure(figsize=(size_px / 100, size_px / 100), dpi=100, facecolor="black")
    ax = fig.add_axes([0, 0, 1, 1])  # full-bleed
    ax.set_facecolor("black")
    ax.imshow(img, cmap="gray", vmin=-0.8, vmax=0.8, origin="lower", aspect="equal", interpolation="bilinear")
    if mask_slice is not None and mask_slice.max() > 0:
        m = mask_slice.T
        overlay = np.zeros((*m.shape, 4))
        r, g, b = overlay_color
        overlay[m > 0] = [r, g, b, overlay_alpha]
        ax.imshow(overlay, origin="lower", aspect="equal", interpolation="nearest")
    ax.axis("off")
    fig.savefig(out_path, facecolor="black", edgecolor="none", dpi=100)
    plt.close(fig)
    print(f"  ✓ {out_path.name}  ({size_px}×{size_px}px)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient", default="10352", help="Patient ID (folder under shared_data/dicom)")
    parser.add_argument(
        "--out", default=str(SCENARIO_ROOT / "demo_slices"), help="Output directory for the 3 PNGs"
    )
    parser.add_argument("--size", type=int, default=600, help="Pixel size of each output PNG (square)")
    args = parser.parse_args()

    pid = args.patient
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pat_dir = DICOM_ROOT / pid
    if not pat_dir.exists():
        raise SystemExit(f"Patient directory not found: {pat_dir}")

    print(f"=== Rendering demo slices for patient {pid} ===")
    ct_files, seg_files = find_ct_and_seg_files(pat_dir)
    print(f"  CT={len(ct_files)} slices · SEG={len(seg_files)} files")
    if not ct_files:
        raise SystemExit("No CT files found")

    # 1) MONAI — gives us CT in RAS+1.5mm grid + MONAI mask + GT in same grid
    print("→ Running MONAI inference …")
    ct_np, monai_mask, ras_affine, gt_np = run_monai(ct_files, seg_files)
    print(f"  CT shape {ct_np.shape} · MONAI mask voxels {int(monai_mask.sum()):,} · GT voxels {int(gt_np.sum()):,}")

    # 2) TotalSegmentator — runs on raw CT NIfTI, then resample into MONAI grid
    print("→ Running TotalSegmentator inference …")
    ts_raw_mask, ts_img = run_totalseg(ct_files, seg_files)
    print(f"  TS raw mask voxels {int(ts_raw_mask.sum()):,} on grid {ts_raw_mask.shape}")
    ts_mask = resample_ts_to_monai_grid(ts_raw_mask, ts_img, ras_affine, ct_np.shape)
    print(f"  TS resampled mask voxels {int(ts_mask.sum()):,} on MONAI grid {ts_mask.shape}")

    # 3) Pick coronal slice — use GT center if available, otherwise MONAI mask
    pivot_mask = gt_np if gt_np.max() > 0 else monai_mask
    if pivot_mask.max() == 0:
        pivot_mask = ts_mask
    slice_y = pick_coronal_index(pivot_mask)
    print(f"  Selected coronal slice Y={slice_y} (of {ct_np.shape[1]})")

    # 4) Crop bbox common to all three
    union_mask = (monai_mask | ts_mask | gt_np).astype(np.uint8)
    x0, x1, z0, z1 = compute_crop_bbox(union_mask, slice_y, padding=30)
    print(f"  Crop bbox X[{x0}:{x1}] Z[{z0}:{z1}]")

    ct_slice = ct_np[x0:x1, slice_y, z0:z1]
    monai_slice = monai_mask[x0:x1, slice_y, z0:z1]
    ts_slice = ts_mask[x0:x1, slice_y, z0:z1]

    # 5) Render the 3 PNGs
    render_panel(ct_slice, None, out_dir / f"ct-coronal-{pid}.png", size_px=args.size)
    render_panel(ct_slice, monai_slice, out_dir / f"monai-coronal-{pid}.png", size_px=args.size)
    render_panel(ct_slice, ts_slice, out_dir / f"totalseg-coronal-{pid}.png", size_px=args.size)
    print(f"\nDone. PNGs written to {out_dir}/")


if __name__ == "__main__":
    main()
