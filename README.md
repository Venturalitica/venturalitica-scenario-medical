# Medical Spine Segmentation Scenario

**Venturalitica SDK v0.5** — Three-Layer Compliance: Data Governance + Model Performance + Annex IV.1

End-to-end demo: download real DICOM data from TCIA, run GPU inference with MONAI SegResNet, then audit compliance against EU AI Act using the SDK's three-phase architecture.

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
# Download 1 patient (~800 files, ~1 min) + clinical metadata
uv run download_data.py --patients 10543

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
4. **Phase 1** — `vl.enforce(data=df)` — SDK-computed fairness & privacy metrics (4 controls)
5. **Phase 2** — `vl.enforce(metrics={...})` — Domain-specific model performance (7 controls)
6. **Phase 3** — Annex IV.1 system description identity card
7. Evidence vault at `.venturalitica/runs/`

### 4. Inspect Results

After the pipeline completes:

```
shared_data/
├── cohort_results.csv          # Inference metrics (generated)
├── trusted_metadata.csv        # DICOM metadata (generated)
├── annex_iv1.yaml              # Annex IV.1 system description
├── dicom/                      # Downloaded DICOM data
└── policies/
    ├── data_policy.oscal.yaml  # Art. 10 Data Governance (4 controls)
    ├── model_policy.oscal.yaml # Art. 15 Model Performance (7 controls)
    └── medical/
        └── fairness.oscal.yaml # Reference catalog (not used at runtime)

compliance_report_sdk.md        # Consolidated audit report (generated)
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
│   └── compliance_suite.py       # Three-phase compliance audit
│
├── shared_data/                  # Runtime data (generated, not committed)
│   ├── annex_iv1.yaml            # Annex IV.1 system description
│   ├── dicom/                    # Downloaded DICOM volumes
│   ├── models/                   # MONAI model bundle (Git LFS)
│   └── policies/
│       ├── data_policy.oscal.yaml   # Art. 10 (SDK-computed)
│       └── model_policy.oscal.yaml  # Art. 15 (manual metrics)
│
└── debug/                        # Development utilities
```

---

## Compliance Controls (11 EU AI Act Checks)

### Data Policy — Art. 10 Data Governance (4 controls, SDK-computed)

The SDK computes these metrics automatically from the patient DataFrame via `vl.enforce(data=df)`:

| Control | Metric | Threshold | What it measures |
|---------|--------|-----------|------------------|
| `fairness-sex-disparate-impact` | `disparate_impact` | > 0.80 | Four-fifths rule across biological sex |
| `fairness-age-demographic-parity` | `demographic_parity_diff` | < 0.15 | Outcome parity across age groups |
| `privacy-k-anonymity` | `k_anonymity` | >= 3 | Min quasi-identifier group size (Age, Sex, Manufacturer) |
| `data-quality-completeness` | `data_completeness` | >= 0.90 | Non-null fraction across all columns |

### Model Policy — Art. 15 Accuracy & Robustness (7 controls, domain metrics)

Domain-specific aggregations computed manually and enforced via `vl.enforce(metrics={...})`:

| Control | Metric | Threshold | What it measures |
|---------|--------|-----------|------------------|
| `model-accuracy-global` | `global_dice` | > 0.85 | Mean Dice across cohort |
| `data-leakage-check` | `max_single_dice` | < 0.99 | Train-test leakage detection |
| `robustness-scanner-bias` | `min_scanner_dice` | > 0.85 | Worst Dice across scanner manufacturers |
| `robustness-lesion-type` | `min_lesion_type_dice` | > 0.80 | Worst Dice across Lytic/Blastic phenotypes |
| `robustness-cancer-types` | `min_cancer_dice` | > 0.80 | Worst Dice across top 3 cancer types |
| `safety-small-volume` | `small_vol_dice` | > 0.75 | Dice for bottom 25% volume cases |
| `safety-calibration` | `confidence_correlation` | > 0.50 | Confidence-accuracy correlation |

### Annex IV.1 — System Description

System identity card rendered from `annex_iv1.yaml` (provider: NovaMed Robotics, system: SpineGuard AI v1.0).

---

## Comparison: Base vs Venturalitica

| Feature | Base | Venturalitica |
|---------|------|---------------|
| Core ML | MONAI SegResNet | MONAI SegResNet |
| Inference | Yes | Yes |
| Governance Monitoring | No | `vl.monitor()` (7 probes) |
| Data Governance | No | `vl.enforce(data=df)` (4 controls) |
| Model Performance | No | `vl.enforce(metrics={...})` (7 controls) |
| Annex IV.1 | No | System description card |
| Carbon Tracking | No | codecarbon |
| Evidence Vault | No | `.venturalitica/runs/` |

---

## Requirements

- Python >= 3.11
- CUDA-capable GPU (for inference)
- Git LFS (for model weights)
- ~1 GB disk per patient (DICOM data)

---

**Venturalitica v0.5** — Governance for Responsible AI in Healthcare
