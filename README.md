# 🏥 Medical Spine Segmentation Scenario

**Venturalitica v0.5.0** — Before & After: Pure ML vs Governance-Instrumented

This scenario demonstrates how Venturalitica wraps a clinical spine segmentation system with governance controls, robustness monitoring, and compliance validation against EU AI Act requirements.

---

## 📁 Structure

```
venturalitica-scenario-medical/
├── base_medical/                # Pure ML (no governance)
│   ├── model_evaluation.py       # Core spine segmentation inference
│   ├── dicom_utils.py            # DICOM I/O and alignment utilities
│   ├── viz_utils.py              # Clinical visualization
│   ├── regenerate_metadata.py    # DICOM metadata extraction
│   ├── generate_audit_plots.py   # Audit visualization
│   └── pyproject.toml            # ML dependencies only
│
├── venturalitica_medical/        # Governance-instrumented version
│   ├── compliance_suite.py       # Compliance audit with vl.enforce()
│   └── pyproject.toml            # ML + Venturalitica SDK + codecarbon
│
├── shared_data/                  # Shared assets
│   ├── models/
│   │   └── wholeBody_ct_segmentation/  # MONAI SegResNet bundle (105 classes)
│   ├── policies/
│   │   ├── risks.oscal.yaml      # EU AI Act compliance controls
│   │   └── medical/
│   │       └── fairness.oscal.yaml
│   ├── trusted_metadata.csv      # Reference DICOM metadata
│   └── cohort_results.csv        # Inference benchmark results
│
├── debug/                        # Development & troubleshooting
│   ├── debug_mask_alignment.py   # DICOM segmentation validation
│   ├── debug_loaders.py          # Data loader comparison
│   ├── test_bundle.py            # MONAI bundle verification
│   └── inspect_dicom_alignment.py # Z-axis alignment checks
│
├── main.py                       # CLI orchestrator
├── pyproject.toml                # Root project config
└── README.md                     # This file
```

---

## 🚀 Quick Start

### Run Base Evaluation (Pure ML)
```bash
python main.py --scenario base
```
**What it does:**
- Loads SegResNet model from MONAI bundle
- Loads CT and segmentation DICOM files
- Runs inference on test cohort
- Computes Dice scores and confidence metrics
- No governance, no compliance checks

### Run with Governance (Venturalitica)
```bash
python main.py --scenario venturalitica
```
**What it does:**
- Runs base evaluation
- Wraps results in `vl.enforce()` governance context
- Applies EU AI Act compliance controls
- Generates risk assessment and audit trails

### Run Compliance Audit Only
```bash
python main.py --scenario compliance
```
**What it does:**
- Audits evaluation results against 7 compliance controls
- Checks demographic parity, robustness, fairness
- Generates compliance report

### Run Full Pipeline
```bash
python main.py --scenario full-pipeline
```
**What it does:**
- Runs complete workflow: evaluation → governance → compliance

---

## 📊 Comparison: Base vs Venturalitica

| Feature | Base | Venturalitica |
|---------|------|---------------|
| **Core ML** | ✅ MONAI SegResNet | ✅ MONAI SegResNet + Venturalitica |
| **DICOM Loading** | ✅ Full DICOM support | ✅ Full DICOM support |
| **Inference** | ✅ Yes | ✅ Yes |
| **Robustness Monitoring** | ❌ No | ✅ Yes (vl.enforce) |
| **Compliance Checking** | ❌ No | ✅ Yes (7 EU AI Act controls) |
| **Governance Artifacts** | ❌ No | ✅ Yes (OSCAL policies) |
| **Carbon Footprint Tracking** | ❌ No | ✅ Yes (codecarbon) |
| **Dependencies** | 6 packages | 8 packages |

---

## 🏆 Compliance Controls (7 EU AI Act Checks)

The `risks.oscal.yaml` policy enforces:

1. **Demographic Parity**: Minority demographic representation > 30%
2. **Scanner Robustness**: Dice score > 0.85 across manufacturers
3. **Small Volume Safety**: Dice > 0.75 for small anatomical structures
4. **Lesion Robustness**: Dice > 0.80 across lesion types (>3)
5. **Age Fairness**: Performance drop < 10% for elderly patients
6. **Cancer Robustness**: Dice > 0.80 for top 3 cancer types
7. **Confidence Calibration**: Correlation > 0.0 between confidence and accuracy

---

## 📦 Dependencies

### Base Medical (`base_medical/`)
- `monai[all]>=1.2.0` — Medical imaging framework
- `torch>=2.0.0` — Deep learning
- `numpy>=1.24.0` — Numerical computing
- `pandas>=2.0.0` — Data manipulation
- `matplotlib>=3.7.0` — Visualization
- `scikit-image>=0.21.0` — Image processing

### Venturalitica Medical (`venturalitica_medical/`)
- All base dependencies +
- `venturalitica>=0.5.0` — Governance SDK
- `codecarbon>=2.2.0` — Carbon footprint tracking

---

## 🔧 Model Architecture

**MONAI Bundle:** WholeBody CT Segmentation
- **Type:** SegResNet (Residual Segmentation Network)
- **Classes:** 105 anatomical structures
- **Input:** 3D CT volume (512×512×N voxels)
- **Output:** 105-class semantic segmentation mask
- **Training Dataset:** ~300 clinical CT scans

---

## 📋 Dataset: Clinical CT Cohort

- **Source:** Multi-center CT studies
- **Total Subjects:** ~1,000 CT scans
- **Demographics:** Age 18-95, male/female, multiple ethnicities
- **DICOM Format:** Full metadata (patient, scanner, acquisition parameters)
- **Annotations:** Expert-traced spine segmentations
- **Location:** `shared_data/cohort_results.csv` (metadata + inference results)

---

## 🎓 Learning Outcomes

By comparing base vs venturalitica versions, you'll learn:

1. **Clinical ML Pipeline** — How spine segmentation works on 3D medical imaging
2. **DICOM Processing** — How to handle real medical imaging data
3. **Governance Instrumentation** — How `vl.enforce()` wraps inferences
4. **Robustness Metrics** — How to evaluate scanner diversity and fairness
5. **Compliance Automation** — How OSCAL policies automate regulatory checks
6. **Carbon Tracking** — How to measure ML environmental impact

---

## 🔍 Key Utilities

### `dicom_utils.py` (340 LOC)
Core DICOM handling library used by all modules:
- `load_dicom_volume_robust()` — Robust volume loading with alignment
- `sort_dicom_files()` — Sort DICOM series by z-position
- `find_ct_and_seg_files()` — Locate CT and segmentation pairs

### `viz_utils.py` (144 LOC)
Clinical visualization utilities:
- `plot_axial_view()` — Display axial CT slices with segmentation overlay
- `create_audit_panel()` — Create 3×3 audit visualization grids

### `regenerate_metadata.py` (93 LOC)
DICOM metadata extraction:
- Extract patient demographics
- Extract scanner parameters
- Create metadata CSV for governance audit

---

## 🐛 Debugging

All development and debugging scripts are in `debug/`:

```bash
# Validate DICOM alignment
python debug/debug_mask_alignment.py

# Inspect loader behavior
python debug/debug_loaders.py

# Test MONAI bundle
python debug/test_bundle.py

# Check Z-axis alignment
python debug/inspect_dicom_alignment.py
```

---

## 📚 See Also

- [MONAI Documentation](https://monai.io)
- [EU AI Act Compliance](https://ec.europa.eu/info/law/better-regulation/have-your-say/initiatives/12951_en)
- [OSCAL Policy Framework](https://pages.nist.gov/OSCAL/)
- [Medical Imaging Standards (DICOM)](https://www.dicomstandard.org/)

---

**Venturalitica v0.5.0** — Governance for Responsible AI in Healthcare
