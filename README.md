# Medical Spine Segmentation Scenario

**Venturalitica SDK v0.5.1** — Before & After: Pure ML vs Governance-Instrumented

End-to-end demo: download real DICOM data from TCIA, run GPU inference with MONAI SegResNet, then audit compliance against EU AI Act controls using `venturalitica.enforce()`.

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/Venturalitica/venturalitica-scenario-medical.git
cd venturalitica-scenario-medical

# Full GPU pipeline (MONAI + PyTorch + CUDA + Venturalitica SDK)
uv sync --extra gpu
```

> Model weights (~75 MB) are stored via Git LFS. Run `git lfs pull` if weights are missing.

### 2. Download DICOM Data from TCIA

```bash
# Download 1 patient (~800 files, ~1 min)
uv run download_data.py --patients 10543 --skip-clinical

# Download 5 patients for a broader cohort
uv run download_data.py --limit 5

# Download all patients (~50 GB)
uv run download_data.py
```

Data is saved to `shared_data/dicom/`. Downloads are resumable.

### 3. Run GPU Inference + Compliance Audit

```bash
# Full pipeline: inference + governance monitoring + compliance audit
python main.py --scenario venturalitica --data-path shared_data/dicom
```

This runs:
1. MONAI SegResNet inference on the downloaded DICOM volumes (GPU)
2. `vl.monitor()` — 7 governance probes (hardware, carbon, BOM, trace, integrity, artifact, handshake)
3. DICOM metadata extraction (demographics, scanner parameters)
4. `vl.enforce()` — 10 EU AI Act compliance controls
5. Evidence vault at `.venturalitica/runs/`

### 4. Inspect Results

After the pipeline completes:

```
shared_data/
├── cohort_results.csv          # Inference metrics (generated)
├── trusted_metadata.csv        # DICOM metadata (generated)
└── dicom/                      # Downloaded DICOM data

compliance_report_sdk.md        # Compliance audit report (generated)
.venturalitica/runs/            # Evidence vault (generated)
```

---

## Alternative Scenarios

### Base Evaluation (Pure ML, no governance)
```bash
python main.py --scenario base --data-path shared_data/dicom
```

### Compliance Audit Only (after inference has run)
```bash
python main.py --scenario compliance
```

### Full Pipeline (inference + compliance in single monitor session)
```bash
python main.py --scenario full-pipeline --data-path shared_data/dicom
```

---

## Project Structure

```
venturalitica-scenario-medical/
├── main.py                       # CLI orchestrator
├── download_data.py              # TCIA DICOM downloader (PEP 723)
├── pyproject.toml                # Root project config
│
├── base_medical/                 # Pure ML (no governance)
│   ├── model_evaluation.py       # Core spine segmentation inference
│   ├── dicom_utils.py            # DICOM I/O and alignment
│   ├── viz_utils.py              # Clinical visualization
│   └── regenerate_metadata.py    # DICOM metadata extraction
│
├── venturalitica_medical/        # Governance-instrumented version
│   └── compliance_suite.py       # Compliance audit with vl.enforce()
│
├── shared_data/                  # Runtime data (generated, not committed)
│   ├── dicom/                    # Downloaded DICOM volumes
│   ├── models/                   # MONAI model bundle (Git LFS)
│   │   └── wholeBody_ct_segmentation/
│   └── policies/
│       └── risks.oscal.yaml      # EU AI Act compliance controls
│
└── debug/                        # Development utilities
```

---

## Compliance Controls (10 EU AI Act Checks)

The `risks.oscal.yaml` policy enforces:

**Article 10 — Data Governance:**
1. Demographic Parity: Minority sex representation > 30%
2. Scanner Robustness: Min Dice > 0.85 across manufacturers
3. Small Volume Safety: Dice > 0.75 for bottom 25% volume cases
4. Lesion Type Robustness: Dice > 0.80 across Lytic/Blastic phenotypes

**Article 15 — Accuracy & Robustness:**
5. Global Accuracy: Mean Dice > 0.85
6. Gender Fairness: Male/Female performance gap < 5%
7. Age Fairness: Performance drop < 10% for elderly (>70)
8. Cancer Robustness: Dice > 0.80 for top 3 cancer types
9. Data Leakage Check: Max single Dice < 0.99

**Article 15 — Safety:**
10. Confidence Calibration: Correlation > 0.5 between confidence and accuracy

---

## Comparison: Base vs Venturalitica

| Feature | Base | Venturalitica |
|---------|------|---------------|
| Core ML | MONAI SegResNet | MONAI SegResNet |
| Inference | Yes | Yes |
| Governance Monitoring | No | `vl.monitor()` (7 probes) |
| Compliance Checking | No | `vl.enforce()` (10 controls) |
| Carbon Tracking | No | codecarbon |
| Evidence Vault | No | `.venturalitica/runs/` |

---

## Requirements

- Python >= 3.11
- CUDA-capable GPU (for inference)
- Git LFS (for model weights)
- ~1 GB disk per patient (DICOM data)

---

**Venturalitica v0.5.1** — Governance for Responsible AI in Healthcare
