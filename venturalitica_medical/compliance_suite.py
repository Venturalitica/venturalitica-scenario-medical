import pandas as pd
import numpy as np
from pathlib import Path
import venturalitica


# Resolve paths relative to the scenario root (not CWD)
SCENARIO_ROOT = Path(__file__).parent.parent
SHARED_DATA = SCENARIO_ROOT / "shared_data"

# EU AI Act High-Risk: if you can't prove compliance, you FAIL.
# No silent defaults. No fake passes. Every metric must be real.
INSUFFICIENT = "INSUFFICIENT_DATA"


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


def run_compliance_suite():
    print("=" * 60)
    print("  Venturalitica Compliance Audit — EU AI Act (High-Risk)")
    print("=" * 60)

    # ── 1. Load inference results ──
    results_path = SHARED_DATA / "cohort_results.csv"
    if not results_path.exists():
        print(f"\n  ERROR: No inference results found at {results_path}")
        print(f"  Run inference first: python main.py --scenario venturalitica --data-path shared_data/dicom")
        return
    results_df = pd.read_csv(results_path)
    results_df["PatientID"] = results_df["PatientID"].astype(str)
    n_patients = len(results_df)

    # ── 2. Load trusted DICOM metadata ──
    trusted_path = SHARED_DATA / "trusted_metadata.csv"
    if trusted_path.exists():
        trusted_df = pd.read_csv(trusted_path)
        trusted_df["PatientID"] = trusted_df["PatientID"].astype(str)
        merged_df = results_df.merge(trusted_df, on="PatientID", how="left")
    else:
        print(f"\n  WARNING: No DICOM metadata at {trusted_path}")
        print(f"  Demographic and scanner controls will FAIL (no data to prove compliance).")
        merged_df = results_df

    # ── 3. Load clinical metadata from TCIA ──
    clinical_df = load_clinical_metadata()
    if clinical_df is not None:
        clinical_cols = ["PatientID", "Primary cancer", "Lytic", "Blastic", "Mixed"]
        available_cols = [c for c in clinical_cols if c in clinical_df.columns]
        merged_df = merged_df.merge(clinical_df[available_cols], on="PatientID", how="left")
        print(f"  Clinical metadata loaded ({len(clinical_df)} patients).")
    else:
        print(f"  WARNING: No clinical metadata. Cancer/lesion controls will FAIL.")

    # Drop rows without Dice (no ground truth)
    valid_df = merged_df.dropna(subset=["Dice"]).copy()
    n_valid = len(valid_df)
    print(f"  Cohort: {n_patients} patients | {n_valid} with Dice scores.")

    if n_valid == 0:
        print("\n  FATAL: No valid Dice scores. Cannot audit.")
        return

    # ── 4. Compute metrics — NO DEFAULTS, real values only ──
    print("\n  Computing audit metrics from real data...")
    audit_metrics = {}
    warnings = []

    # Global Dice (always computable if we have valid data)
    global_dice = float(valid_df["Dice"].mean())
    audit_metrics["global_dice"] = global_dice
    audit_metrics["max_single_dice"] = float(valid_df["Dice"].max())

    # ── Demographic Parity (Art. 10) ──
    sex_col = "Sex" if "Sex" in valid_df.columns else None
    if sex_col and valid_df[sex_col].dropna().nunique() >= 2:
        sex_dist = valid_df[sex_col].dropna().value_counts(normalize=True)
        audit_metrics["minority_sex_prop"] = float(sex_dist.min())
    elif sex_col and valid_df[sex_col].dropna().nunique() == 1:
        # Only one sex in cohort — that's a real demographic failure
        audit_metrics["minority_sex_prop"] = 0.0
        warnings.append(f"  Only {valid_df[sex_col].dropna().iloc[0]} patients — zero minority representation")
    else:
        audit_metrics["minority_sex_prop"] = 0.0
        warnings.append("  No sex data available — cannot prove demographic parity")

    # ── Gender Gap (Art. 15) ──
    if sex_col and valid_df[sex_col].dropna().nunique() >= 2:
        male_dice = valid_df[valid_df[sex_col] == "M"]["Dice"].mean()
        female_dice = valid_df[valid_df[sex_col] == "F"]["Dice"].mean()
        audit_metrics["gender_gap"] = float(abs(male_dice - female_dice))
    else:
        # Can't prove fairness → worst case (max gap)
        audit_metrics["gender_gap"] = 1.0
        warnings.append("  Cannot compute gender gap — insufficient sex diversity")

    # ── Scanner Robustness (Art. 15) ──
    if "Manufacturer" in valid_df.columns and valid_df["Manufacturer"].dropna().nunique() >= 1:
        mfr_counts = valid_df["Manufacturer"].dropna().value_counts()
        # Use any manufacturer with at least 1 patient (real data, not thresholded away)
        min_scanner_dice = float("inf")
        for mfr in mfr_counts.index:
            d = valid_df[valid_df["Manufacturer"] == mfr]["Dice"].mean()
            if d < min_scanner_dice:
                min_scanner_dice = float(d)
        audit_metrics["min_scanner_dice"] = min_scanner_dice
        n_mfrs = len(mfr_counts)
        if n_mfrs == 1:
            warnings.append(f"  Only 1 scanner manufacturer ({mfr_counts.index[0]}) — no cross-scanner validation")
    else:
        audit_metrics["min_scanner_dice"] = 0.0
        warnings.append("  No scanner manufacturer data — cannot prove scanner robustness")

    # ── Small Volume Safety (Art. 15) ──
    if "SpineVol" in valid_df.columns and n_valid >= 4:
        vol_q1 = valid_df["SpineVol"].quantile(0.25)
        small_subset = valid_df[valid_df["SpineVol"] <= vol_q1]
        if len(small_subset) > 0:
            audit_metrics["small_vol_dice"] = float(small_subset["Dice"].mean())
        else:
            audit_metrics["small_vol_dice"] = 0.0
    elif "SpineVol" in valid_df.columns and n_valid < 4:
        # Too few patients for quartile — use the worst individual score
        audit_metrics["small_vol_dice"] = float(valid_df["Dice"].min())
        warnings.append(f"  Only {n_valid} patients — using worst-case Dice for small volume safety")
    else:
        audit_metrics["small_vol_dice"] = 0.0
        warnings.append("  No volume data — cannot prove small volume safety")

    # ── Lesion Type Robustness (Art. 15) ──
    lesion_computed = False
    min_lesion_dice = float("inf")
    for ltype in ["Lytic", "Blastic", "Mixed"]:
        if ltype in valid_df.columns and valid_df[ltype].notna().any():
            subset = valid_df[valid_df[ltype].notna()]
            if len(subset) >= 1:
                d = float(subset["Dice"].mean())
                if d < min_lesion_dice:
                    min_lesion_dice = d
                lesion_computed = True
    if lesion_computed:
        audit_metrics["min_lesion_type_dice"] = min_lesion_dice
    else:
        audit_metrics["min_lesion_type_dice"] = 0.0
        warnings.append("  No lesion type annotations — cannot prove lesion robustness")

    # ── Age Bias (Art. 15) ──
    age_col = "Age" if "Age" in valid_df.columns else None
    if age_col and valid_df[age_col].dropna().any():
        valid_df["AgeGroup"] = pd.cut(
            valid_df[age_col].astype(float),
            bins=[0, 50, 70, 120],
            labels=["<50", "50-70", ">70"],
        )
        elderly = valid_df[valid_df["AgeGroup"] == ">70"]
        non_elderly = valid_df[valid_df["AgeGroup"] != ">70"]
        if len(elderly) >= 1 and len(non_elderly) >= 1 and global_dice > 0:
            elderly_dice = float(elderly["Dice"].mean())
            audit_metrics["age_bias"] = float((global_dice - elderly_dice) / global_dice)
        elif len(elderly) == 0:
            audit_metrics["age_bias"] = 1.0
            warnings.append("  No elderly (>70) patients in cohort — cannot prove age fairness")
        else:
            audit_metrics["age_bias"] = 1.0
            n_age = len(valid_df[age_col].dropna())
            warnings.append(f"  Only {n_age} patient(s) with age data — need both elderly and non-elderly to compute bias")
    else:
        audit_metrics["age_bias"] = 1.0
        warnings.append("  No age data in DICOM metadata — cannot prove age fairness")

    # ── Cancer Robustness (Art. 15) ──
    if "Primary cancer" in valid_df.columns and valid_df["Primary cancer"].notna().any():
        top_cancers = valid_df["Primary cancer"].dropna().value_counts().nlargest(3).index
        min_c_dice = float("inf")
        for cancer in top_cancers:
            c_dice = valid_df[valid_df["Primary cancer"] == cancer]["Dice"].mean()
            if not np.isnan(c_dice) and c_dice < min_c_dice:
                min_c_dice = float(c_dice)
        audit_metrics["min_cancer_dice"] = min_c_dice if min_c_dice < float("inf") else 0.0
    else:
        audit_metrics["min_cancer_dice"] = 0.0
        warnings.append("  No cancer type data — cannot prove cancer robustness")

    # ── Confidence Calibration (Art. 15) ──
    if n_valid >= 3:
        corr = valid_df["Confidence"].corr(valid_df["Dice"])
        audit_metrics["confidence_correlation"] = float(corr) if not pd.isna(corr) else 0.0
    else:
        audit_metrics["confidence_correlation"] = 0.0
        warnings.append(f"  Only {n_valid} patients — insufficient for correlation analysis")

    # Print computed metrics
    print()
    for k, v in audit_metrics.items():
        print(f"    {k}: {v:.4f}")

    if warnings:
        print(f"\n  Data gaps detected ({len(warnings)}):")
        for w in warnings:
            print(f"    {w}")

    # ── 5. Enforce compliance ──
    policy_path = str(SHARED_DATA / "policies" / "risks.oscal.yaml")
    print(f"\n  Enforcing OSCAL policy: {Path(policy_path).name}")

    with venturalitica.monitor(
        name="Spine-Mets Compliance Audit",
        label="EU AI Act Medical Device — High Risk (Art. 6.1)",
    ):
        compliance_results = venturalitica.enforce(
            metrics=audit_metrics,
            policy=policy_path,
        )

    if not compliance_results:
        print("  No compliance results returned. Check policy file.")
        return

    # ── 6. Print results ──
    print("\n" + "=" * 60)
    print("  EU AI ACT COMPLIANCE AUDIT — RESULTS")
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

    if failed_count > 0:
        print(f"\n  This model CANNOT be deployed under EU AI Act without addressing {failed_count} violations.")
    else:
        print(f"\n  All controls passed. Model is compliant for deployment.")

    # ── 7. Generate report ──
    report_path = SCENARIO_ROOT / "compliance_report_sdk.md"
    with open(report_path, "w") as f:
        f.write("# Venturalitica Compliance Audit Report\n\n")
        f.write(f"**Policy:** `risks.oscal.yaml` | **Cohort:** {n_patients} patients ({n_valid} with GT)\n\n")
        f.write(f"Generated via `venturalitica.enforce()` | SDK v{venturalitica.__version__}\n\n")

        f.write("## Executive Summary\n\n")
        overall = "COMPLIANT" if failed_count == 0 else "NON-COMPLIANT"
        f.write(f"- **Overall Status**: {overall}\n")
        f.write(f"- **Controls Checked**: {len(compliance_results)}\n")
        f.write(f"- **Passed**: {passed_count}\n")
        f.write(f"- **Failed**: {failed_count}\n\n")

        if warnings:
            f.write("## Data Gaps\n\n")
            f.write("The following metrics could not be fully evaluated due to insufficient cohort data:\n\n")
            for w in warnings:
                f.write(f"- {w.strip()}\n")
            f.write("\n")

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
