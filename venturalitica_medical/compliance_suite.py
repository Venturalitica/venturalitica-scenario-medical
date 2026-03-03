import pandas as pd
import numpy as np
from pathlib import Path
import venturalitica


# Resolve paths relative to the scenario root (not CWD)
SCENARIO_ROOT = Path(__file__).parent.parent
SHARED_DATA = SCENARIO_ROOT / "shared_data"


def run_compliance_suite():
    print("=" * 60)
    print("  Venturalitica Compliance Audit Suite (EU AI Act)")
    print("=" * 60)

    # 1. Load Data
    results_path = SHARED_DATA / "cohort_results.csv"
    if not results_path.exists():
        print(f"  Results file not found: {results_path}")
        print(f"  Run --scenario base first to generate inference results.")
        return
    results_df = pd.read_csv(results_path)

    # Load Trusted Metadata (Generated from DICOMs)
    trusted_path = SHARED_DATA / "trusted_metadata.csv"
    if trusted_path.exists():
        trusted_df = pd.read_csv(trusted_path)
        trusted_df["PatientID"] = trusted_df["PatientID"].astype(str)
        merged_df = results_df.astype({"PatientID": str}).merge(
            trusted_df, on="PatientID", how="left"
        )
    else:
        print(f"  Trusted metadata not found at {trusted_path}.")
        merged_df = results_df.astype({"PatientID": str})

    # Load Clinical Metadata (optional — cancer type data)
    clinical_metadata_path = (
        SCENARIO_ROOT.parent
        / "venturalitica-sdk-samples-extra"
        / "scenarios"
        / "surgery-dicom-tcia"
        / "data"
        / "combined_metadata.csv"
    )
    if clinical_metadata_path.exists():
        meta_df = pd.read_csv(clinical_metadata_path)
        meta_df["PatientID"] = (
            pd.to_numeric(meta_df["Case"], errors="coerce")
            .fillna(0)
            .astype(int)
            .astype(str)
        )
        clinical_cols = ["PatientID", "Primary cancer", "Lytic", "Blastic", "Mixed"]
        available_cols = [c for c in clinical_cols if c in meta_df.columns]
        merged_df = merged_df.merge(meta_df[available_cols], on="PatientID", how="left")
    else:
        print(f"  Clinical metadata not found (optional). Skipping cancer-type metrics.")

    # Drop rows with NaN Dice (patients without ground truth)
    valid_df = merged_df.dropna(subset=["Dice"])
    print(f"  Loaded {len(merged_df)} records ({len(valid_df)} with valid Dice scores).")

    # 2. Compute Custom Metrics
    print("\n  Pre-computing audit metrics...")
    audit_metrics = {}

    # Sex Stats
    sex_col = "Sex" if "Sex" in valid_df.columns else None
    if sex_col:
        sex_dist = valid_df[sex_col].dropna().value_counts(normalize=True)
        audit_metrics["minority_sex_prop"] = float(sex_dist.min()) if len(sex_dist) > 1 else 0.0
    else:
        audit_metrics["minority_sex_prop"] = 0.0

    # Global Dice
    global_dice = valid_df["Dice"].mean()
    audit_metrics["global_dice"] = global_dice
    audit_metrics["max_single_dice"] = valid_df["Dice"].max()

    # Gender Gap
    if sex_col and sex_col in valid_df.columns:
        male_dice = valid_df[valid_df[sex_col] == "M"]["Dice"].mean()
        female_dice = valid_df[valid_df[sex_col] == "F"]["Dice"].mean()
        if not pd.isna(male_dice) and not pd.isna(female_dice):
            audit_metrics["gender_gap"] = abs(male_dice - female_dice)
        else:
            audit_metrics["gender_gap"] = 0.0
    else:
        audit_metrics["gender_gap"] = 0.0

    # Scanner Bias
    if "Manufacturer" in valid_df.columns:
        manufacturers = valid_df["Manufacturer"].value_counts()
        valid_mfrs = manufacturers[manufacturers > 5].index
        min_scanner_dice = 1.0
        for mfr in valid_mfrs:
            d = valid_df[valid_df["Manufacturer"] == mfr]["Dice"].mean()
            if d < min_scanner_dice:
                min_scanner_dice = d
        audit_metrics["min_scanner_dice"] = min_scanner_dice
    else:
        audit_metrics["min_scanner_dice"] = 1.0

    # Small Volume Safety (Bottom 25% volume)
    if "SpineVol" in valid_df.columns:
        vol_q1 = valid_df["SpineVol"].quantile(0.25)
        small_vol_dice = valid_df[valid_df["SpineVol"] < vol_q1]["Dice"].mean()
        audit_metrics["small_vol_dice"] = small_vol_dice if not pd.isna(small_vol_dice) else 1.0
    else:
        audit_metrics["small_vol_dice"] = 1.0

    # Lesion Type Robustness
    min_lesion_dice = 1.0
    for ltype in ["Lytic", "Blastic"]:
        if ltype in valid_df.columns and valid_df[ltype].notna().any():
            subset = valid_df[valid_df[ltype].notna()]
            if len(subset) > 3:
                d = subset["Dice"].mean()
                if d < min_lesion_dice:
                    min_lesion_dice = d
    audit_metrics["min_lesion_type_dice"] = min_lesion_dice

    # Age Bias
    age_col = "Age" if "Age" in valid_df.columns else None
    if age_col:
        bins = [0, 50, 70, 100]
        labels = ["<50", "50-70", ">70"]
        valid_df = valid_df.copy()
        valid_df["AgeGroup"] = pd.cut(valid_df[age_col], bins=bins, labels=labels)
        elderly_dice = valid_df[valid_df["AgeGroup"] == ">70"]["Dice"].mean()
        if pd.isna(elderly_dice) or global_dice == 0:
            audit_metrics["age_bias"] = 0.0
        else:
            audit_metrics["age_bias"] = (global_dice - elderly_dice) / global_dice
    else:
        audit_metrics["age_bias"] = 0.0

    # Cancer Robustness
    if "Primary cancer" in valid_df.columns and valid_df["Primary cancer"].notna().any():
        top_cancers = valid_df["Primary cancer"].value_counts().nlargest(3).index
        min_c_dice = 1.0
        for cancer in top_cancers:
            c_dice = valid_df[valid_df["Primary cancer"] == cancer]["Dice"].mean()
            if not np.isnan(c_dice) and c_dice < min_c_dice:
                min_c_dice = c_dice
        audit_metrics["min_cancer_dice"] = min_c_dice
    else:
        audit_metrics["min_cancer_dice"] = 1.0

    # Calibration
    corr = valid_df["Confidence"].corr(valid_df["Dice"])
    audit_metrics["confidence_correlation"] = corr if not pd.isna(corr) else 0.0

    # Cast to python float
    audit_metrics = {k: float(v) for k, v in audit_metrics.items()}

    print()
    for k, v in audit_metrics.items():
        print(f"    {k}: {v:.4f}")

    # 3. Enforce Compliance with monitoring
    policy_path = str(SHARED_DATA / "policies" / "risks.oscal.yaml")
    print(f"\n  Enforcing policy: {policy_path}")

    with venturalitica.monitor(
        name="Spine-Mets Compliance Audit",
        label="EU AI Act Medical Device",
    ):
        compliance_results = venturalitica.enforce(
            metrics=audit_metrics,
            policy=policy_path,
        )

    if not compliance_results:
        print("  No compliance results returned. Check policy file.")
        return

    # 4. Print Results
    print("\n" + "=" * 60)
    print("  AUDIT RESULTS")
    print("=" * 60)

    passed_count = sum(1 for r in compliance_results if r.passed)
    failed_count = sum(1 for r in compliance_results if not r.passed)

    for check in compliance_results:
        icon = "PASS" if check.passed else "FAIL"
        print(
            f"  [{icon}] {check.control_id}: "
            f"{check.actual_value:.4f} {check.operator} {check.threshold} "
            f"({check.severity})"
        )
        print(f"         {check.description}")

    print(f"\n  Summary: {passed_count} passed / {failed_count} failed out of {len(compliance_results)} controls")

    # 5. Generate Markdown Report
    report_path = SCENARIO_ROOT / "compliance_report_sdk.md"
    with open(report_path, "w") as f:
        f.write("# Venturalitica Compliance Audit Report\n\n")
        f.write("Generated via `venturalitica.enforce(policy='risks.oscal.yaml')`\n\n")

        f.write("## Executive Summary\n\n")
        overall = "PASS" if failed_count == 0 else "FAIL"
        f.write(f"- **Overall Status**: {overall}\n")
        f.write(f"- **Controls Checked**: {len(compliance_results)}\n")
        f.write(f"- **Passed**: {passed_count}\n")
        f.write(f"- **Failed**: {failed_count}\n\n")

        f.write("## Detailed Findings\n\n")
        for check in compliance_results:
            icon = "PASS" if check.passed else "FAIL"
            f.write(f"### [{icon}] {check.control_id}\n\n")
            f.write(f"- **Description**: {check.description}\n")
            f.write(f"- **Severity**: {check.severity}\n")
            f.write(f"- **Metric**: `{check.metric_key}`\n")
            f.write(f"- **Result**: {check.actual_value:.4f} {check.operator} {check.threshold}\n")
            f.write(f"- **Verdict**: {'Compliant' if check.passed else 'Non-compliant'}\n\n")

    print(f"\n  Report saved to {report_path}")


def main():
    """Alias for orchestrator compatibility."""
    run_compliance_suite()


if __name__ == "__main__":
    run_compliance_suite()
