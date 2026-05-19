"""TotalSegmentator v2 vertebra-segmentation inference wrapper.

Mirrors the contract of `base_medical/model_evaluation.py` (which runs the
MONAI `wholeBody_ct_segmentation` SegResNet bundle) so the same audit
pipeline can score both models on the same TCIA Spine-Mets-CT-SEG cohort
and emit a side-by-side comparison.

Output: `shared_data/cohort_results_totalseg.csv` with columns
    PatientID, Dice, Jaccard, SpineVol, Confidence
— same shape as `cohort_results.csv` produced by MONAI inference.

TotalSegmentator v2.13 (Wasserthal et al., Feb 2025) is the closest
public model to the SOTA for vertebra segmentation on CT: nnU-Net
backbone, individual labels for the 25 vertebral bodies (C1–S1), Apache
2.0. We restrict to the `total` task with `roi_subset` limited to
`vertebrae_*` so the comparison is apples-to-apples with the MONAI
bundle (which we only consume the 24 vertebra channels of).
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from monai.data import MetaTensor
from monai.transforms import Compose, Orientation, Spacing

# Local utilities (re-used from MONAI pipeline so Dice is computed
# identically against the TCIA ground-truth SEG).
from dicom_utils import (
    auto_align_orientation,
    find_ct_and_seg_files,
    get_annotated_spine_indices,
    load_dicom_seg_reconstructed,
    load_dicom_volume_robust,
    sort_dicom_files,
)

warnings.filterwarnings("ignore")


# TotalSegmentator vertebra label ids in the `total` task (v2.13).
# Range 26 (S1) → 50 (C1) plus 25 (sacrum bone). Keep aligned with
# totalsegmentator.map_to_binary if upgrading versions.
_VERTEBRA_LABEL_IDS_TS = list(range(25, 51))


# ─── MONAI bundle v0.2.7 ↔ TotalSegmentator v2.13 label mapping ────────────
# Both models share the TotalSegmentator-derived vertebra naming, but the
# integer IDs differ between MONAI's wholeBody_ct_segmentation channel_def
# (TotalSegmentator v1 scheme) and TotalSegmentator v2's class_map.
# The map is BY ANATOMICAL NAME and is the source of truth for filtering
# TS predictions down to the subset that the TCIA SEG annotator labelled
# (the SEG file uses the MONAI scheme — `get_annotated_spine_indices`
# returns MONAI ids).
_MONAI_TO_TS: dict[int, list[int]] = {
    18: [27],  # L5
    19: [28],  # L4
    20: [29],  # L3
    21: [30],  # L2
    22: [31],  # L1
    23: [32],  # T12
    24: [33],  # T11
    25: [34],  # T10
    26: [35],  # T9
    27: [36],  # T8
    28: [37],  # T7
    29: [38],  # T6
    30: [39],  # T5
    31: [40],  # T4
    32: [41],  # T3
    33: [42],  # T2
    34: [43],  # T1
    35: [44],  # C7
    36: [45],  # C6
    37: [46],  # C5
    38: [47],  # C4
    39: [48],  # C3
    40: [49],  # C2
    41: [50],  # C1
    # MONAI's `sacrum` (92) lumps the sacral bone together with the S1
    # vertebral body. TS v2 keeps them separate (25 = sacrum bone,
    # 26 = vertebrae_S1) — to make the masks compatible we take the
    # union of both TS labels whenever the GT marks MONAI's 92.
    92: [25, 26],
}


def _annotated_ts_labels(annotated_monai: list[int]) -> list[int]:
    """Translate the MONAI-scheme indices found in a TCIA SEG into the
    set of TotalSegmentator-scheme indices that cover the same anatomy.
    Used to restrict the TS prediction to the cohort the GT actually
    annotated — fair-comparison prerequisite."""
    out: list[int] = []
    for m in annotated_monai:
        if m in _MONAI_TO_TS:
            out.extend(_MONAI_TO_TS[m])
    return sorted(set(out))


def _dicom_ct_to_nifti(ct_files: list[Path]) -> tuple[Path, MetaTensor]:
    """Save the DICOM CT series as a NIfTI in its **raw** orientation so
    TotalSegmentator (which performs its own RAS-orientation / resampling
    internally and emits output in the exact input grid) can be compared
    against the TCIA SEG ground truth in the same coordinate system.

    Earlier versions of this wrapper pre-transformed the CT to
    RAS + 1.5 mm before saving; the resulting pred and the GT (resampled
    via a different code path) ended up on subtly different grids and
    the Dice collapsed to zero. Letting TotalSegmentator handle
    orientation removes that whole class of alignment bugs.
    """
    raw_meta = load_dicom_volume_robust([str(p) for p in ct_files])

    tmp_dir = Path(tempfile.mkdtemp(prefix="vlts_"))
    nifti_path = tmp_dir / "ct.nii.gz"

    affine = raw_meta.affine.cpu().numpy() if torch.is_tensor(raw_meta.affine) else np.asarray(raw_meta.affine)
    array = raw_meta.cpu().numpy() if torch.is_tensor(raw_meta) else np.asarray(raw_meta)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    nib.save(nib.Nifti1Image(array.astype(np.int16), affine), str(nifti_path))
    return nifti_path, raw_meta


def _run_totalsegmentator(nifti_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, "nib.Nifti1Image"]:
    """Run TotalSegmentator on a CT NIfTI; return (multilabel_mask, affine,
    softmax_probs_or_none).

    We pass `roi_subset=['vertebrae_*']` so only vertebra labels are kept
    — keeps the comparison apples-to-apples with the MONAI bundle's
    spine-only prediction path.
    """
    from totalsegmentator.python_api import totalsegmentator

    # With ml=True TotalSegmentator writes a SINGLE multilabel NIfTI file
    # whose path is taken verbatim from the `output` argument (it ignores
    # the directory-vs-file distinction and appends .nii if missing). So
    # we pass an explicit .nii.gz file path rather than a folder.
    multilabel_path = nifti_path.parent / "ts_out.nii.gz"
    from totalsegmentator.map_to_binary import class_map
    roi_subset = [v for v in class_map["total"].values() if v.startswith("vertebrae_")]

    totalsegmentator(
        input=str(nifti_path),
        output=str(multilabel_path),
        task="total",
        roi_subset=roi_subset,
        ml=True,
        fast=False,
        verbose=False,
        quiet=True,
    )
    # Fall back to any sibling .nii / .nii.gz the API may have produced.
    if not multilabel_path.exists():
        candidates = sorted(nifti_path.parent.glob("ts_out*.ni*"))
        if not candidates:
            raise FileNotFoundError(
                f"TotalSegmentator produced no output near {multilabel_path}"
            )
        multilabel_path = candidates[0]

    img = nib.load(str(multilabel_path))
    mask = np.asarray(img.dataobj).astype(np.int32)
    affine = img.affine
    # Return the loaded nibabel image too — callers need it as the
    # reference grid when resampling the GT into the prediction's frame.
    return mask, affine, None, img


def _compute_dice(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    """Same Dice + Jaccard formulation as the MONAI pipeline."""
    pred_b = (pred > 0).astype(np.uint8)
    gt_b = (gt > 0).astype(np.uint8)
    inter = int(np.logical_and(pred_b, gt_b).sum())
    union = int(np.logical_or(pred_b, gt_b).sum())
    pred_sum = int(pred_b.sum())
    gt_sum = int(gt_b.sum())
    dice = (2.0 * inter) / (pred_sum + gt_sum) if (pred_sum + gt_sum) > 0 else 0.0
    jaccard = inter / union if union > 0 else 0.0
    return float(dice), float(jaccard)


def evaluate_patient(patient_dir: Path) -> dict | None:
    """Run TotalSegmentator on one patient and compute Dice against the
    TCIA SEG ground truth. Returns the metrics row or None on hard failure."""
    pid = patient_dir.name
    print(f"\n  ▶ {pid}")
    try:
        ct_files, seg_files = find_ct_and_seg_files(patient_dir)
    except Exception as exc:
        print(f"      ❌ DICOM walk failed: {exc}")
        return None
    if not ct_files:
        print(f"      ⏭  no CT series — skipping")
        return None

    tmp_root = None
    try:
        nifti_path, nifti_meta = _dicom_ct_to_nifti([Path(f) for f in ct_files])
        tmp_root = nifti_path.parent
        ml_mask, ml_affine, _, ml_img = _run_totalsegmentator(nifti_path)

        # We restrict the TS prediction to the same subset of vertebrae
        # the TCIA SEG annotated (otherwise TS predicts all 25 vertebrae
        # and the unannotated ones inflate the union → Dice collapses).
        # `get_annotated_spine_indices` returns MONAI-scheme ids; we
        # translate to TS ids before filtering.
        annotated_monai = get_annotated_spine_indices(seg_files[0]) if seg_files else None
        if annotated_monai:
            ts_subset = _annotated_ts_labels(annotated_monai) or _VERTEBRA_LABEL_IDS_TS
        else:
            ts_subset = _VERTEBRA_LABEL_IDS_TS
        spine_mask = np.isin(ml_mask, ts_subset).astype(np.uint8)
        spine_vol = int(spine_mask.sum())
        print(
            f"      ↳ predicted spine volume: {spine_vol} voxels  "
            f"(restricted to {len(ts_subset)} TS labels mapped from "
            f"{len(annotated_monai) if annotated_monai else 0} GT-annotated MONAI labels)"
        )

        # Compute Dice against the TCIA SEG ground truth — but the GT
        # arrives in the raw CT grid (post auto_align_orientation
        # heuristic) while the TS prediction lives in TotalSegmentator's
        # output grid (orientation reconciled via the DICOM affine).
        # `auto_align`'s bone-overlap heuristic occasionally picks the
        # wrong flip for sparse-SEG patients, so we use nibabel's
        # affine-aware `resample_from_to` to put the GT exactly on the
        # prediction's grid. That eliminates orientation + spacing drift
        # in one operation.
        dice, jaccard = 0.0, 0.0
        if seg_files:
            from nibabel.processing import resample_from_to

            ct_files_sorted = sort_dicom_files([str(p) for p in ct_files])
            raw_seg_tensor = load_dicom_seg_reconstructed(
                seg_files[0], ct_files_sorted, target_shape=nifti_meta.shape
            )
            raw_seg_tensor = auto_align_orientation(nifti_meta, raw_seg_tensor)
            gt_array = raw_seg_tensor.cpu().numpy() if torch.is_tensor(raw_seg_tensor) else np.asarray(raw_seg_tensor)
            if gt_array.ndim == 4 and gt_array.shape[0] == 1:
                gt_array = gt_array[0]

            # The GT lives in the raw CT grid — wrap it as a nibabel image
            # with the raw CT affine so resample_from_to can project it
            # into the prediction's frame.
            raw_ct_affine = (
                nifti_meta.affine.cpu().numpy()
                if torch.is_tensor(nifti_meta.affine)
                else np.asarray(nifti_meta.affine)
            )
            gt_img = nib.Nifti1Image(gt_array.astype(np.int32), raw_ct_affine)
            gt_in_pred_grid = resample_from_to(gt_img, ml_img, order=0)  # NN — preserve label ids
            gt_resampled = np.asarray(gt_in_pred_grid.dataobj).astype(np.int32)

            # TCIA SEG voxels are stored as a BINARY mask of the annotated
            # vertebrae (the per-level identity lives in the SEG metadata
            # consumed by `get_annotated_spine_indices`, not in the voxel
            # values). We mirror MONAI's evaluator: filter the PREDICTION
            # to the annotated levels (done above) and compare against
            # the binary GT as-is.
            gt_binary = (gt_resampled > 0).astype(np.uint8)
            dice, jaccard = _compute_dice(spine_mask, gt_binary)
            print(
                f"      ↳ Dice={dice:.4f} | Jaccard={jaccard:.4f}  "
                f"(pred {int(spine_mask.sum()):,} vox / gt {int(gt_binary.sum()):,} vox / "
                f"shared shape {spine_mask.shape})"
            )
        else:
            print(f"      ⏭  no SEG — Dice unavailable")

        # TotalSegmentator's python_api does not surface per-voxel softmax,
        # so we report a deterministic Confidence proxy: the fraction of
        # the multilabel volume that received a vertebra label. This is a
        # weaker calibration signal than MONAI's softmax-derived
        # Confidence — flagged in the audit warnings.
        proxy_conf = float(spine_vol) / float(ml_mask.size) if ml_mask.size else 0.0

        return {
            "PatientID": pid,
            "Dice": dice,
            "Jaccard": jaccard,
            "SpineVol": spine_vol,
            "Confidence": proxy_conf,
        }
    except Exception as exc:
        print(f"      ❌ {exc.__class__.__name__}: {exc}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        if tmp_root and tmp_root.exists():
            shutil.rmtree(tmp_root, ignore_errors=True)


def main(model_path: str | None = None, data_path: str | None = None,
         output_csv: str | None = None) -> None:
    """Drop-in alternative to base_medical/model_evaluation.py::main."""
    data_dir = Path(data_path) if data_path else Path(__file__).parent.parent / "shared_data" / "dicom"
    out_path = Path(output_csv) if output_csv else (
        Path(__file__).parent.parent / "shared_data" / "cohort_results_totalseg.csv"
    )
    if not data_dir.exists():
        raise SystemExit(f"DICOM directory not found: {data_dir}")

    patient_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir()])
    print(f"=== TotalSegmentator v2 vertebra evaluation — {len(patient_dirs)} patients ===")

    rows = []
    for pat_dir in patient_dirs:
        row = evaluate_patient(pat_dir)
        if row:
            rows.append(row)

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\n  ✓ Wrote {len(df)} rows to {out_path}")
    if not df.empty:
        print(f"    mean Dice: {df['Dice'].mean():.4f} | min: {df['Dice'].min():.4f} | max: {df['Dice'].max():.4f}")


if __name__ == "__main__":
    data_arg = sys.argv[1] if len(sys.argv) > 1 else None
    main(data_path=data_arg)
