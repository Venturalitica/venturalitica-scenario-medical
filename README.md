# Medical Spine Segmentation — EU AI Act Compliance Demo

**Venturalitica SDK v0.5** — Three-layer compliance audit on a real medical imaging pipeline.

This repository demonstrates how to add EU AI Act compliance controls to an existing ML system — without changing the model or the inference code. It downloads real DICOM data from a public cancer imaging archive, runs GPU inference with a pre-trained MONAI model, and then audits the entire pipeline against 11 regulatory controls using the Venturalitica SDK.

---

## Data & Model Sources

Everything in this demo comes from publicly available, peer-reviewed medical research:

### Dataset: Spine-Mets-CT-Seg (TCIA)

| | |
|---|---|
| **Source** | [The Cancer Imaging Archive (TCIA)](https://www.cancerimagingarchive.net/collection/spine-mets-ct-seg/) |
| **Content** | CT volumes of patients with spinal metastatic disease, with expert segmentation annotations |
| **Modality** | CT (DICOM format) |
| **Clinical metadata** | Age, sex, primary cancer type, lesion classification (lytic/blastic/mixed), vertebral levels |
| **License** | [TCIA Data Usage Policy](https://www.cancerimagingarchive.net/data-usage-policies-and-restrictions/) — CC BY 4.0 |
| **Citation** | Hallinan JTPD, et al. *"Spine-Mets-CT-Seg"*, The Cancer Imaging Archive, 2023 |

The downloader script (`download_data.py`) uses the official [NBIA Toolkit](https://github.com/jjjermiah/nbia-toolkit) to fetch DICOM volumes and the associated clinical spreadsheet directly from TCIA.

### Model: MONAI wholeBody_ct_segmentation

| | |
|---|---|
| **Source** | [MONAI Model Zoo](https://github.com/Project-MONAI/model-zoo) — `wholeBody_ct_segmentation` bundle |
| **Architecture** | SegResNet (3D) — 104 whole-body anatomical structures |
| **Training data** | [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) dataset |
| **Weights** | ~75 MB, stored via Git LFS in `shared_data/models/` |
| **License** | Apache 2.0 |
| **Citation** | Wasserthal J, et al. *"TotalSegmentator: robust segmentation of 104 anatomic structures in CT images"*, Radiology: AI, 2023 |

We use a subset of the 104 predicted segments — specifically the 24 vertebral body labels (C1–S1) — to compute spine-specific metrics (Dice, Jaccard) per patient.

### Policies: OSCAL format

The compliance policies are defined as [OSCAL](https://pages.nist.gov/OSCAL/) assessment-plan YAML files in `shared_data/policies/`. Each control specifies a metric key, threshold, comparison operator, and input bindings. The SDK reads these policies at runtime — no compliance logic is hardcoded.

---

## How It Works

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│  TCIA DICOM     │────▶│  MONAI SegResNet  │────▶│  Venturalitica SDK  │
│  (real patient   │     │  (104-structure   │     │  (3-phase audit)    │
│   CT volumes)   │     │   segmentation)   │     │                     │
└─────────────────┘     └──────────────────┘     └─────────────────────┘
                                                    │
                              ┌──────────────────────┤
                              ▼                      ▼
                   ┌─────────────────┐    ┌──────────────────┐
                   │ Phase 1: Data   │    │ Phase 2: Model   │
                   │ Art.10 Fairness │    │ Art.15 Accuracy  │
                   │ (SDK-computed)  │    │ (domain metrics) │
                   └─────────────────┘    └──────────────────┘
                              │                      │
                              ▼                      ▼
                   ┌──────────────────────────────────────────┐
                   │ Phase 3: Annex IV.1 System Description   │
                   │ (identity card from annex_iv1.yaml)      │
                   └──────────────────────────────────────────┘
                              │
                              ▼
                   ┌──────────────────────────────────────────┐
                   │ Evidence Vault: .venturalitica/runs/     │
                   │ + compliance_report_sdk.md               │
                   └──────────────────────────────────────────┘
```

**Phase 1 — Data Governance (Art. 10):** The SDK automatically computes fairness and privacy metrics from the patient DataFrame: disparate impact across sex, demographic parity across age groups, k-anonymity, and data completeness. You only pass the data — no metric code needed.

**Phase 2 — Model Performance (Art. 15):** Domain-specific metrics (global Dice, scanner bias, lesion-type robustness, small-volume safety, calibration) are aggregated from inference results and enforced against OSCAL thresholds.

**Phase 3 — Annex IV.1:** A structured system description (identity card) is rendered from `annex_iv1.yaml`, covering all subsections (a)–(h) required by the EU AI Act for high-risk system documentation.

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
# Download 1 patient (~800 DICOM files, ~1 min)
uv run download_data.py --patients 10543

# Download 5 patients for a broader cohort
uv run download_data.py --limit 5

# Download all patients (~50 GB)
uv run download_data.py
```

Data is saved to `shared_data/dicom/`. Downloads are resumable.

### 3. Run GPU Inference + Compliance Audit

```bash
# Full pipeline: inference → governance monitoring → 3-phase compliance audit
python main.py --scenario venturalitica --data-path shared_data/dicom
```

This runs:
1. **MONAI SegResNet inference** on the downloaded DICOM volumes (GPU)
2. **`vl.monitor()`** — 7 governance probes (hardware, carbon, BOM, trace, integrity, artifact, handshake)
3. **DICOM metadata extraction** (demographics, scanner parameters)
4. **Phase 1** — `vl.enforce(data=df)` — SDK-computed fairness & privacy metrics (4 controls)
5. **Phase 2** — `vl.enforce(metrics={...})` — Domain-specific model performance (7 controls)
6. **Phase 3** — Annex IV.1 system description identity card
7. **Evidence vault** at `.venturalitica/runs/`

### 4. Launch the Dashboard

```bash
uv run venturalitica ui
```

Opens a Streamlit dashboard at `localhost:8501` showing compliance results, policy details, and the system description card. Run from the project root directory.

### 5. Inspect Results

```
shared_data/
├── cohort_results.csv          # Inference metrics per patient (generated)
├── trusted_metadata.csv        # DICOM metadata per patient (generated)
├── clinical_metadata.csv       # Clinical data from TCIA spreadsheet
├── annex_iv1.yaml              # Annex IV.1 system description
├── dicom/                      # Downloaded DICOM volumes
├── models/                     # MONAI model bundle (Git LFS)
│   └── wholeBody_ct_segmentation/
└── policies/
    ├── data_policy.oscal.yaml  # Art. 10 Data Governance (4 controls)
    └── model_policy.oscal.yaml # Art. 15 Model Performance (7 controls)

compliance_report_sdk.md        # Consolidated audit report (generated)
.venturalitica/runs/            # Evidence vault with trace + results JSON
```

---

## Alternative Scenarios

```bash
# Pure ML evaluation (no governance)
python main.py --scenario base --data-path shared_data/dicom

# Compliance audit only (after inference has already run)
python main.py --scenario compliance

# Full pipeline (inference + compliance in single session)
python main.py --scenario full-pipeline --data-path shared_data/dicom
```

---

## Compliance Controls (11 EU AI Act Checks)

### Phase 1 — Data Governance, Art. 10 (4 controls, SDK-computed)

The SDK computes these metrics automatically from the patient DataFrame via `vl.enforce(data=df)`:

| Control | Metric | Threshold | What it measures |
|---------|--------|-----------|------------------|
| `fairness-sex-disparate-impact` | `disparate_impact` | > 0.80 | Four-fifths rule across biological sex |
| `fairness-age-demographic-parity` | `demographic_parity_diff` | < 0.15 | Outcome parity across age groups |
| `privacy-k-anonymity` | `k_anonymity` | >= 3 | Min quasi-identifier group size (Age, Sex, Manufacturer) |
| `data-quality-completeness` | `data_completeness` | >= 0.90 | Non-null fraction across all columns |

### Phase 2 — Model Performance, Art. 15 (7 controls, domain metrics)

Domain-specific aggregations computed from inference results and enforced via `vl.enforce(metrics={...})`:

| Control | Metric | Threshold | What it measures |
|---------|--------|-----------|------------------|
| `model-accuracy-global` | `global_dice` | > 0.85 | Mean Dice coefficient across cohort |
| `data-leakage-check` | `max_single_dice` | < 0.99 | Train-test leakage detection |
| `robustness-scanner-bias` | `min_scanner_dice` | > 0.85 | Worst Dice across scanner manufacturers |
| `robustness-lesion-type` | `min_lesion_type_dice` | > 0.80 | Worst Dice across Lytic/Blastic phenotypes |
| `robustness-cancer-types` | `min_cancer_dice` | > 0.80 | Worst Dice across top 3 cancer types |
| `safety-small-volume` | `small_vol_dice` | > 0.75 | Dice for bottom 25% tumor volume cases |
| `safety-calibration` | `confidence_correlation` | > 0.50 | Confidence-accuracy correlation (Pearson r) |

### Phase 3 — Annex IV.1 System Description

Structured identity card rendered from `annex_iv1.yaml`, covering all EU AI Act Annex IV Section 1 subsections: (a) purpose, (b) hardware/software interaction, (c) dependencies, (d) market form, (e) hardware requirements, (f) external features, (g) UI, (h) instructions for use, plus foreseeable misuse.

> The system description uses fictional names (NovaMed Robotics / SpineGuard AI) for demonstration purposes.

---

## Demo Guide (Step by Step)

This walkthrough is designed for a live demo or self-guided exploration.

### Prerequisites

```bash
cd venturalitica-scenario-medical
uv sync --extra gpu          # Install all dependencies
git lfs pull                 # Ensure model weights are present
```

### Step 1 — Download Patient Data

```bash
uv run download_data.py --limit 5
```

This fetches 5 real patient CT volumes from TCIA plus the clinical metadata spreadsheet. Takes ~5 min depending on connection speed.

**What to show:** The `shared_data/dicom/` folder with real DICOM files, the `clinical_metadata.csv` with patient demographics (age, sex, cancer type, lesion morphology).

### Step 2 — Run Base Inference (Pure ML, No Governance)

```bash
python main.py --scenario base --data-path shared_data/dicom
```

This runs the SegResNet model on each patient's CT volume and produces `cohort_results.csv` with per-patient Dice and Jaccard scores.

**What to show:** The inference output — real medical image segmentation with no compliance overhead. This is the "before" baseline.

### Step 3 — Run Compliance Audit

```bash
python main.py --scenario compliance
```

This runs the 3-phase audit using the existing inference results (no GPU needed):

- **Phase 1** output: 4 data governance controls with pass/fail verdicts
- **Phase 2** output: 7 model performance controls with actual metric values
- **Phase 3** output: Full Annex IV.1 system identity card

**Key demo points:**
- The compliance code (`compliance_suite.py`) is ~300 lines — it only calls `vl.monitor()`, `vl.enforce()`, and loads YAML policies
- The model code (`base_medical/`) was not modified at all
- Controls that fail (e.g., `safety-calibration`) show that the system is honest — it catches real issues

### Step 4 — Launch the Dashboard

```bash
uv run venturalitica ui
```

Opens `http://localhost:8501`. Navigate through:

1. **Session selector** — pick the latest compliance audit run
2. **Policy view** — see the OSCAL controls with thresholds
3. **Results table** — pass/fail for each control with actual values
4. **System description** — the rendered Annex IV.1 identity card

### Step 5 — Inspect the Evidence Vault

```bash
ls .venturalitica/runs/
cat .venturalitica/runs/*/results.json | python -m json.tool | head -50
```

Every audit run produces a timestamped evidence folder with:
- `results.json` — machine-readable control results
- `trace_*.json` — execution trace (what ran, when, with what inputs)
- Linked to the CycloneDX software BOM

### Step 6 — Review the Generated Report

```bash
cat compliance_report_sdk.md
```

Markdown report with the consolidated verdict: how many controls passed, which failed, and whether the system can be deployed under EU AI Act.

---

## Comparison: Base vs Venturalitica

| Feature | Base | Venturalitica |
|---------|------|---------------|
| Core ML | MONAI SegResNet | MONAI SegResNet (unchanged) |
| Inference | Yes | Yes |
| Governance monitoring | — | `vl.monitor()` (7 probes) |
| Data governance | — | `vl.enforce(data=df)` (4 controls) |
| Model performance | — | `vl.enforce(metrics={...})` (7 controls) |
| Annex IV.1 | — | System description card |
| Carbon tracking | — | codecarbon integration |
| Software BOM | — | CycloneDX 1.5 ML profile |
| Evidence vault | — | `.venturalitica/runs/` |

---

## Project Structure

```
venturalitica-scenario-medical/
├── main.py                       # CLI orchestrator (4 scenarios)
├── download_data.py              # TCIA DICOM downloader (PEP 723, standalone)
├── pyproject.toml                # Project config with [gpu] extra
│
├── base_medical/                 # Pure ML pipeline (no governance)
│   ├── model_evaluation.py       # SpineEvaluator: SegResNet inference
│   ├── dicom_utils.py            # DICOM I/O, volume loading, alignment
│   ├── viz_utils.py              # Clinical visualization utilities
│   └── regenerate_metadata.py    # DICOM header → trusted_metadata.csv
│
├── venturalitica_medical/        # Governance-instrumented layer
│   └── compliance_suite.py       # 3-phase compliance audit (~300 LOC)
│
├── shared_data/                  # Data directory
│   ├── annex_iv1.yaml            # Annex IV.1 system description
│   ├── clinical_metadata.csv     # TCIA clinical spreadsheet
│   ├── cohort_results.csv        # Inference results (generated)
│   ├── trusted_metadata.csv      # DICOM metadata (generated)
│   ├── dicom/                    # Downloaded DICOM volumes
│   ├── models/                   # MONAI model bundle (Git LFS)
│   │   └── wholeBody_ct_segmentation/
│   └── policies/
│       ├── data_policy.oscal.yaml
│       └── model_policy.oscal.yaml
│
└── debug/                        # Development utilities
```

---

## Requirements

- Python >= 3.11
- CUDA-capable GPU (for inference; compliance audit runs on CPU)
- Git LFS (for model weights, ~75 MB)
- ~1 GB disk per patient (DICOM data)

---

## References

1. Hallinan JTPD, et al. *Spine-Mets-CT-Seg*, The Cancer Imaging Archive, 2023. [Link](https://www.cancerimagingarchive.net/collection/spine-mets-ct-seg/)
2. Wasserthal J, et al. *TotalSegmentator: robust segmentation of 104 anatomic structures in CT images*, Radiology: AI, 2023. [Link](https://doi.org/10.1148/ryai.230024)
3. Myronenko A. *3D MRI brain tumor segmentation using autoencoder regularization*, BrainLes 2018. [Link](https://arxiv.org/abs/1810.11654) (SegResNet architecture)
4. MONAI Model Zoo — `wholeBody_ct_segmentation` bundle. [Link](https://github.com/Project-MONAI/model-zoo)
5. NIST OSCAL — Open Security Controls Assessment Language. [Link](https://pages.nist.gov/OSCAL/)

---

**[Venturalitica](https://venturalitica.ai)** — AI Governance SDK for EU AI Act Compliance
