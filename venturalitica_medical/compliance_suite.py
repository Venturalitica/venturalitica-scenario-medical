"""
Venturalitica Medical Compliance Suite — SDK v0.5
Three-phase audit: Data Governance → Model Performance → Annex IV.1

Phase 1 — Data Governance (Art. 10)
  SDK-computed: disparate_impact, demographic_parity_diff, k_anonymity, data_completeness

Phase 2 — Model Performance (Art. 15)
  Domain-specific manual aggregations enforced via vl.enforce(metrics={...})

Phase 3 — Annex IV.1
  System description identity card from annex_iv1.yaml
"""

import shutil
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import venturalitica
from venturalitica.models import SystemDescription


# Resolve paths relative to the scenario root (not CWD)
SCENARIO_ROOT = Path(__file__).parent.parent
SHARED_DATA = SCENARIO_ROOT / "shared_data"

# Canonical policy/annex locations
DATA_POLICY = SHARED_DATA / "policies" / "data_policy.oscal.yaml"
MODEL_POLICY = SHARED_DATA / "policies" / "model_policy.oscal.yaml"
ANNEX_IV = SHARED_DATA / "annex_iv1.yaml"


def stage_for_dashboard():
    """Copy policy and annex files to SCENARIO_ROOT so `venturalitica ui` finds them.

    The Streamlit dashboard checks CWD for:
      system_description.yaml, model_policy.oscal.yaml, data_policy.oscal.yaml
    """
    for src, dst_name in [
        (DATA_POLICY, "data_policy.oscal.yaml"),
        (MODEL_POLICY, "model_policy.oscal.yaml"),
        (ANNEX_IV, "system_description.yaml"),
    ]:
        dst = SCENARIO_ROOT / dst_name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            print(f"  Staged {dst_name} for dashboard")


def load_clinical_metadata() -> pd.DataFrame | None:
    """Load clinical metadata downloaded from TCIA."""
    clinical_path = SHARED_DATA / "clinical_metadata.csv"
    if not clinical_path.exists():
        return None

    raw = pd.read_csv(clinical_path)

    # TCIA Excel has definition rows before data — find first numeric Case
    first_data_idx = None
    for i, val in enumerate(raw["Case"]):
        try:
            int(float(val))
            first_data_idx = i
            break
        except (ValueError, TypeError):
            continue

    if first_data_idx is None:
        return None

    df = raw.iloc[first_data_idx:].copy()
    df["PatientID"] = df["Case"].astype(float).astype(int).astype(str)
    return df


def prepare_dataframe(results_df: pd.DataFrame, trusted_path: Path) -> pd.DataFrame:
    """Merge inference results with DICOM metadata, add derived columns."""
    results_df["PatientID"] = results_df["PatientID"].astype(str)

    # Merge trusted DICOM metadata
    if trusted_path.exists():
        trusted_df = pd.read_csv(trusted_path)
        trusted_df["PatientID"] = trusted_df["PatientID"].astype(str)
        merged_df = results_df.merge(trusted_df, on="PatientID", how="left")
    else:
        print("  WARNING: No DICOM metadata — demographic controls will use available data only.")
        merged_df = results_df.copy()

    # Merge clinical metadata
    clinical_df = load_clinical_metadata()
    if clinical_df is not None:
        clinical_cols = ["PatientID", "Primary cancer", "Lytic", "Blastic", "Mixed"]
        available_cols = [c for c in clinical_cols if c in clinical_df.columns]
        merged_df = merged_df.merge(clinical_df[available_cols], on="PatientID", how="left")
        print(f"  Clinical metadata loaded ({len(clinical_df)} patients).")
    else:
        print("  WARNING: No clinical metadata. Cancer/lesion model controls will FAIL.")

    # Drop rows without Dice (no ground truth)
    merged_df = merged_df.dropna(subset=["Dice"]).copy()

    # Derived columns for fairness metrics
    merged_df["is_successful"] = (merged_df["Dice"] > 0.7).astype(int)

    if "Age" in merged_df.columns:
        merged_df["age_group"] = pd.cut(
            merged_df["Age"].astype(float),
            bins=[0, 50, 70, 120],
            labels=["Young", "Adult", "Senior"],
        )

    return merged_df


def phase1_data_governance(merged_df: pd.DataFrame) -> list:
    """Phase 1 — Data Governance (Art. 10): SDK-computed fairness & privacy metrics."""
    print("\n" + "-" * 60)
    print("  PHASE 1 — Data Governance (Art. 10)")
    print("-" * 60)

    print(f"  Policy: data_policy.oscal.yaml")
    print(f"  SDK computes: disparate_impact, demographic_parity_diff, k_anonymity, data_completeness")
    print(f"  DataFrame: {len(merged_df)} rows x {len(merged_df.columns)} columns")

    results = venturalitica.enforce(
        data=merged_df,
        policy=str(DATA_POLICY),
    )

    _print_phase_results(results, "Data Governance")
    return results


def phase2_model_performance(merged_df: pd.DataFrame) -> list:
    """Phase 2 — Model Performance (Art. 15): domain-specific manual aggregations."""
    print("\n" + "-" * 60)
    print("  PHASE 2 — Model Performance (Art. 15)")
    print("-" * 60)

    n_valid = len(merged_df)
    audit_metrics = {}
    warnings = []

    # Global accuracy
    audit_metrics["global_dice"] = float(merged_df["Dice"].mean())
    audit_metrics["max_single_dice"] = float(merged_df["Dice"].max())

    # Scanner robustness
    if "Manufacturer" in merged_df.columns and merged_df["Manufacturer"].dropna().nunique() >= 1:
        mfr_dice = merged_df.groupby("Manufacturer")["Dice"].mean()
        audit_metrics["min_scanner_dice"] = float(mfr_dice.min())
        if len(mfr_dice) == 1:
            warnings.append(f"  Only 1 scanner manufacturer ({mfr_dice.index[0]}) — no cross-scanner validation")
    else:
        audit_metrics["min_scanner_dice"] = 0.0
        warnings.append("  No scanner manufacturer data — cannot prove scanner robustness")

    # Small volume safety
    if "SpineVol" in merged_df.columns and n_valid >= 4:
        vol_q1 = merged_df["SpineVol"].quantile(0.25)
        small_subset = merged_df[merged_df["SpineVol"] <= vol_q1]
        audit_metrics["small_vol_dice"] = float(small_subset["Dice"].mean()) if len(small_subset) > 0 else 0.0
    elif "SpineVol" in merged_df.columns:
        audit_metrics["small_vol_dice"] = float(merged_df["Dice"].min())
        warnings.append(f"  Only {n_valid} patients — using worst-case Dice for small volume safety")
    else:
        audit_metrics["small_vol_dice"] = 0.0
        warnings.append("  No volume data — cannot prove small volume safety")

    # Lesion type robustness
    lesion_dices = []
    for ltype in ["Lytic", "Blastic", "Mixed"]:
        if ltype in merged_df.columns and merged_df[ltype].notna().any():
            subset = merged_df[merged_df[ltype].notna()]
            if len(subset) >= 1:
                lesion_dices.append(float(subset["Dice"].mean()))
    audit_metrics["min_lesion_type_dice"] = min(lesion_dices) if lesion_dices else 0.0
    if not lesion_dices:
        warnings.append("  No lesion type annotations — cannot prove lesion robustness")

    # Cancer type robustness
    if "Primary cancer" in merged_df.columns and merged_df["Primary cancer"].notna().any():
        top_cancers = merged_df["Primary cancer"].dropna().value_counts().nlargest(3).index
        cancer_dices = []
        for cancer in top_cancers:
            d = merged_df[merged_df["Primary cancer"] == cancer]["Dice"].mean()
            if not np.isnan(d):
                cancer_dices.append(float(d))
        audit_metrics["min_cancer_dice"] = min(cancer_dices) if cancer_dices else 0.0
    else:
        audit_metrics["min_cancer_dice"] = 0.0
        warnings.append("  No cancer type data — cannot prove cancer robustness")

    # Confidence calibration
    if n_valid >= 3 and "Confidence" in merged_df.columns:
        corr = merged_df["Confidence"].corr(merged_df["Dice"])
        audit_metrics["confidence_correlation"] = float(corr) if not pd.isna(corr) else 0.0
    else:
        audit_metrics["confidence_correlation"] = 0.0
        warnings.append(f"  Insufficient data for correlation analysis")

    # Print metrics
    print("\n  Computed domain metrics:")
    for k, v in audit_metrics.items():
        print(f"    {k}: {v:.4f}")

    if warnings:
        print(f"\n  Data gaps ({len(warnings)}):")
        for w in warnings:
            print(f"    {w}")

    # Enforce
    print(f"\n  Policy: model_policy.oscal.yaml")

    results = venturalitica.enforce(
        metrics=audit_metrics,
        policy=str(MODEL_POLICY),
    )

    _print_phase_results(results, "Model Performance")
    return results


def phase3_annex_iv(merged_df: pd.DataFrame) -> SystemDescription:
    """Phase 3 — Annex IV.1: Load and display system description identity card."""
    print("\n" + "-" * 60)
    print("  PHASE 3 — Annex IV.1 System Description")
    print("-" * 60)

    if not ANNEX_IV.exists():
        print(f"  WARNING: {ANNEX_IV} not found — skipping Annex IV.1")
        return None

    with open(ANNEX_IV) as f:
        sd = SystemDescription(**yaml.safe_load(f))

    print(f"\n  {'=' * 56}")
    print(f"  SYSTEM IDENTITY CARD — EU AI Act Annex IV.1")
    print(f"  {'=' * 56}")
    print(f"  System:     {sd.name} v{sd.version}")
    print(f"  Provider:   {sd.provider_name}")
    print(f"  {'-' * 56}")
    print(f"  (a) Purpose:        {sd.intended_purpose[:80]}...")
    print(f"  (b) Interaction:    {sd.interaction_description[:80]}...")
    print(f"  (c) Dependencies:   {sd.software_dependencies[:80]}...")
    print(f"  (d) Market form:    {sd.market_placement_form[:80]}...")
    print(f"  (e) Hardware:       {sd.hardware_description[:80]}...")
    print(f"  (f) External:       {sd.external_features[:80]}...")
    print(f"  (g) UI:             {sd.ui_description[:80]}...")
    print(f"  (h) Instructions:   {sd.instructions_for_use[:80]}...")
    print(f"  (*) Misuse risks:   {sd.potential_misuses[:80]}...")
    print(f"  {'=' * 56}")

    return sd


def _print_phase_results(results: list, phase_name: str):
    """Print pass/fail summary for a phase."""
    if not results:
        print(f"  No results returned for {phase_name}.")
        return

    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)

    print(f"\n  Results ({phase_name}):")
    for check in results:
        icon = "PASS" if check.passed else "FAIL"
        print(
            f"    [{icon}] {check.control_id}: "
            f"{check.actual_value:.4f} {check.operator} {check.threshold} "
            f"({check.severity})"
        )
    print(f"  -> {passed} passed / {failed} failed")


def generate_report(
    data_results: list,
    model_results: list,
    system_desc: SystemDescription | None,
    n_patients: int,
    n_valid: int,
):
    """Generate consolidated markdown compliance report."""
    report_path = SCENARIO_ROOT / "compliance_report_sdk.md"
    all_results = (data_results or []) + (model_results or [])
    passed = sum(1 for r in all_results if r.passed)
    failed = sum(1 for r in all_results if not r.passed)
    overall = "COMPLIANT" if failed == 0 else "NON-COMPLIANT"

    with open(report_path, "w") as f:
        f.write("# Venturalitica Compliance Audit Report\n\n")
        f.write(f"**SDK v{venturalitica.__version__}** | **Cohort:** {n_patients} patients ({n_valid} with GT)\n\n")

        f.write("## Executive Summary\n\n")
        f.write(f"- **Overall Status**: {overall}\n")
        f.write(f"- **Total Controls**: {len(all_results)}\n")
        f.write(f"- **Passed**: {passed}\n")
        f.write(f"- **Failed**: {failed}\n\n")

        # Phase 1
        if data_results:
            f.write("## Phase 1 — Data Governance (Art. 10)\n\n")
            f.write("SDK-computed metrics via `vl.enforce(data=df)`.\n\n")
            _write_results_table(f, data_results)

        # Phase 2
        if model_results:
            f.write("## Phase 2 — Model Performance (Art. 15)\n\n")
            f.write("Domain-specific aggregations via `vl.enforce(metrics={...})`.\n\n")
            _write_results_table(f, model_results)

        # Phase 3
        if system_desc:
            f.write("## Phase 3 — Annex IV.1 System Description\n\n")
            f.write(f"| Field | Value |\n|---|---|\n")
            f.write(f"| **System** | {system_desc.name} v{system_desc.version} |\n")
            f.write(f"| **Provider** | {system_desc.provider_name} |\n")
            f.write(f"| **(a) Purpose** | {system_desc.intended_purpose.strip()[:120]}... |\n")
            f.write(f"| **(e) Hardware** | {system_desc.hardware_description.strip()[:120]}... |\n")
            f.write(f"| **(h) Instructions** | {system_desc.instructions_for_use.strip()[:120]}... |\n")
            f.write(f"| **Misuse risks** | {system_desc.potential_misuses.strip()[:120]}... |\n\n")

    print(f"\n  Report saved to {report_path}")


def _write_results_table(f, results: list):
    """Write a markdown table of compliance results."""
    f.write("| Control | Metric | Result | Threshold | Verdict | Severity |\n")
    f.write("|---|---|---|---|---|---|\n")
    for r in results:
        icon = "PASS" if r.passed else "FAIL"
        f.write(
            f"| {r.control_id} | `{r.metric_key}` | {r.actual_value:.4f} "
            f"| {r.operator} {r.threshold} | {icon} | {r.severity} |\n"
        )
    f.write("\n")


def run_compliance_suite():
    print("=" * 60)
    print("  Venturalitica Compliance Audit — EU AI Act (High-Risk)")
    print("  SDK v0.5 — Three-Phase Architecture")
    print("=" * 60)

    # Stage files for dashboard visibility
    stage_for_dashboard()

    # Load inference results
    results_path = SHARED_DATA / "cohort_results.csv"
    if not results_path.exists():
        print(f"\n  ERROR: No inference results found at {results_path}")
        print("  Run inference first: python main.py --scenario venturalitica --data-path shared_data/dicom")
        return

    results_df = pd.read_csv(results_path)
    n_patients = len(results_df)

    # Prepare merged DataFrame with derived columns
    merged_df = prepare_dataframe(results_df, SHARED_DATA / "trusted_metadata.csv")
    n_valid = len(merged_df)
    print(f"  Cohort: {n_patients} patients | {n_valid} with Dice scores.")

    if n_valid == 0:
        print("\n  FATAL: No valid Dice scores. Cannot audit.")
        return

    # Run three phases inside a single monitor session
    with venturalitica.monitor(
        name="Spine-Mets Compliance Audit",
        label="EU AI Act Medical Device — High Risk (Art. 6.1)",
    ):
        data_results = phase1_data_governance(merged_df)
        model_results = phase2_model_performance(merged_df)
        system_desc = phase3_annex_iv(merged_df)

    # Consolidated verdict
    all_results = (data_results or []) + (model_results or [])
    passed = sum(1 for r in all_results if r.passed)
    failed = sum(1 for r in all_results if not r.passed)

    print("\n" + "=" * 60)
    print(f"  CONSOLIDATED VERDICT: {passed} passed / {failed} failed out of {len(all_results)} controls")
    if failed > 0:
        print(f"  This model CANNOT be deployed under EU AI Act without addressing {failed} violations.")
    else:
        print("  All controls passed. Model is compliant for deployment.")
    print("=" * 60)

    # Generate report
    generate_report(data_results, model_results, system_desc, n_patients, n_valid)


def main():
    """Alias for orchestrator compatibility."""
    run_compliance_suite()


if __name__ == "__main__":
    run_compliance_suite()
