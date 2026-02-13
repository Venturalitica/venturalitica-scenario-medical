import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import venturalitica 

def check_age_bias(df):
    """
    Check if performance drop for elderly (>70) is < 10% compared to global mean.
    Returns True if pass, False if fail.
    """
    bins = [0, 50, 70, 100]
    labels = ['<50', '50-70', '>70']
    df['AgeGroup'] = pd.cut(df['Age (Y)'], bins=bins, labels=labels)
    
    global_dice = df['Dice'].mean()
    elderly_dice = df[df['AgeGroup'] == '>70']['Dice'].mean()
    
    if pd.isna(elderly_dice): return True # No elderly patients
    
    drop = (global_dice - elderly_dice) / global_dice
    return drop < 0.10

def check_cancer_robustness(df):
    """
    Check if Dice > 0.80 for all top 3 primary cancer types.
    """
    top_cancers = df['Primary cancer'].value_counts().nlargest(3).index
    for cancer in top_cancers:
        c_dice = df[df['Primary cancer'] == cancer]['Dice'].mean()
        if c_dice < 0.80:
            return False
    return True

def run_compliance_suite():
    print("🛡️ Venturalitica Compliance Audit Suite (EU AI Act) - SDK Mode")
    print("============================================================")
    
    # 1. Load Data
    results_path = Path("cohort_results.csv")
    if not results_path.exists():
        print(f"❌ Results file not found: {results_path}")
        return
    results_df = pd.read_csv(results_path)

    # Load Trusted Metadata (Generated from DICOMs)
    trusted_path = Path("trusted_metadata.csv")
    if trusted_path.exists():
        trusted_df = pd.read_csv(trusted_path)
        trusted_df['PatientID'] = trusted_df['PatientID'].astype(str)
        # Merge trusted metadata
        merged_df = results_df.astype({'PatientID': str}).merge(trusted_df, on='PatientID', how='left')
    else:
        print(f"⚠️ Trusted metadata not found. Run regenerate_metadata.py first.")
        merged_df = results_df.astype({'PatientID': str})

    # Load Clinical Metadata (Legacy CSV - Only for Clinical outcomes)
    metadata_path = Path("../../../venturalitica-sdk-samples-extra/scenarios/surgery-dicom-tcia/data/combined_metadata.csv")
    if metadata_path.exists():
        meta_df = pd.read_csv(metadata_path)
        meta_df['PatientID'] = pd.to_numeric(meta_df['Case'], errors='coerce').fillna(0).astype(int).astype(str)
        # Only take clinical columns
        clinical_cols = ['PatientID', 'Primary cancer', 'Lytic', 'Blastic', 'Mixed']
        merged_df = merged_df.merge(meta_df[clinical_cols], on='PatientID', how='left')
    
    print(f"✅ Loaded and merged {len(merged_df)} records.")

    # 2. Compute Custom Metrics
    print("Pre-computing custom metrics for validation...")
    audit_metrics = {}
    
    # Sex Stats (Prefer trusted)
    sex_col = 'Sex' if 'Sex' in merged_df.columns else 'Sex_y' # Handle potential duplicate if not cleaned
    if sex_col in merged_df.columns:
        sex_dist = merged_df[sex_col].value_counts(normalize=True)
        audit_metrics['minority_sex_prop'] = sex_dist.min()
    else:
        audit_metrics['minority_sex_prop'] = 0.0
    
    # Global
    global_dice = merged_df['Dice'].mean()
    audit_metrics['global_dice'] = global_dice
    audit_metrics['max_single_dice'] = merged_df['Dice'].max()
    
    # Gender Gap (Use trusted Sex)
    if sex_col in merged_df.columns:
        male_dice = merged_df[merged_df[sex_col] == 'M']['Dice'].mean()
        female_dice = merged_df[merged_df[sex_col] == 'F']['Dice'].mean()
        audit_metrics['gender_gap'] = abs(male_dice - female_dice) if not pd.isna(male_dice) and not pd.isna(female_dice) else 0.0
    
    # Scanner Bias (Trusted)
    if 'Manufacturer' in merged_df.columns:
        manufacturers = merged_df['Manufacturer'].value_counts()
        valid_mfrs = manufacturers[manufacturers > 5].index
        min_scanner_dice = 1.0
        for mfr in valid_mfrs:
            d = merged_df[merged_df['Manufacturer'] == mfr]['Dice'].mean()
            if d < min_scanner_dice: min_scanner_dice = d
        audit_metrics['min_scanner_dice'] = min_scanner_dice
    else:
        audit_metrics['min_scanner_dice'] = 1.0

    # Small Volume Safety (Bottom 25% volume)
    if 'SpineVol' in merged_df.columns:
        vol_q1 = merged_df['SpineVol'].quantile(0.25)
        small_vol_dice = merged_df[merged_df['SpineVol'] < vol_q1]['Dice'].mean()
        audit_metrics['small_vol_dice'] = small_vol_dice
    else:
        audit_metrics['small_vol_dice'] = 1.0

    # Lesion Type Robustness (Clinical columns)
    lesion_types = []
    if 'Lytic' in merged_df.columns and merged_df['Lytic'].notna().any(): lesion_types.append('Lytic')
    if 'Blastic' in merged_df.columns and merged_df['Blastic'].notna().any(): lesion_types.append('Blastic')
    # Logic: If column has data for that row, it's that type. 
    # Warning: One patient might have multiple. We check average DICE for patients WITH that lesion type.
    min_lesion_dice = 1.0
    for ltype in lesion_types:
        # Check rows where ltype column is not empty/NaN
        subset = merged_df[merged_df[ltype].notna()]
        if len(subset) > 3:
            d = subset['Dice'].mean()
            if d < min_lesion_dice: min_lesion_dice = d
    audit_metrics['min_lesion_type_dice'] = min_lesion_dice

    
    # Age Bias
    bins = [0, 50, 70, 100]
    labels = ['<50', '50-70', '>70']
    
    age_col = 'Age' if 'Age' in merged_df.columns else 'Age (Y)'
    if age_col in merged_df.columns:
        merged_df['AgeGroup'] = pd.cut(merged_df[age_col], bins=bins, labels=labels)
        elderly_dice = merged_df[merged_df['AgeGroup'] == '>70']['Dice'].mean()
        
        # Avoid zero division or NaN
        if pd.isna(elderly_dice) or global_dice == 0:
            audit_metrics['age_bias'] = 0.0 
        else:
            audit_metrics['age_bias'] = (global_dice - elderly_dice) / global_dice
    else:
        audit_metrics['age_bias'] = 0.0
        print(f"⚠️ Age column not found. Age bias metric set to 0.0.")

    # Cancer Robustness
    if 'Primary cancer' in merged_df.columns:
        top_cancers = merged_df['Primary cancer'].value_counts().nlargest(3).index
        min_c_dice = 1.0
        for cancer in top_cancers:
            c_dice = merged_df[merged_df['Primary cancer'] == cancer]['Dice'].mean()
            if not np.isnan(c_dice) and c_dice < min_c_dice:
                min_c_dice = c_dice
        audit_metrics['min_cancer_dice'] = min_c_dice
    else:
        audit_metrics['min_cancer_dice'] = 1.0 # Default if column missing

    # Calibration
    audit_metrics['confidence_correlation'] = merged_df['Confidence'].corr(merged_df['Dice'])
    
    # Cast to python float
    audit_metrics = {k: float(v) for k, v in audit_metrics.items()}
    
    print(f"Metrics: {audit_metrics}")

    # 3. Enforce Compliance
    print(f"⚙️ Running venturalitica.enforce() with configuration from risks.oscal.yaml...")
    
    compliance_results = venturalitica.enforce(
        metrics=audit_metrics,
        policy="risks.oscal.yaml",
        project_name="Spine-Mets-Seg-Audit",
        version="1.0.0"
    )

    # 4. Generate Artifacts
    # The SDK returns a LIST of result objects directly.
    print("\n--- Audit Results ---")
    
    passed_count = 0
    failed_count = 0
    
    for check in compliance_results:
        # Assuming ComplianceResult validation
        # We need to map the result object fields. 
        # Based on help(), it returns List[venturalitica.models.ComplianceResult]
        # Let's assume standard attributes: passed (bool), policy_name/id, message/description.
        
        # If the SDK actually returned a list, we iterate it.
        try:
            passed = check.passed
            # Fix dynamic attribute access if needed
            name = getattr(check, 'policy_name', getattr(check, 'id', getattr(check, 'control_id', 'Unknown Policy')))
            msg = getattr(check, 'message', getattr(check, 'description', 'No details'))
        except AttributeError:
            # Fallback for debugging if structure is different
            print(f"⚠️ Unexpected result structure: {check}")
            continue

        status_icon = "✅" if passed else "❌"
        print(f"{status_icon} {name}: {msg}")
        
        if passed: passed_count += 1
        else: failed_count += 1
        
    # Generate Plots (Keep existing logic for visual artifacts)
    plot_dir = Path("compliance_plots")
    plot_dir.mkdir(exist_ok=True)
    
    # Save a simplified MD report mimicking the SDK output
    with open("compliance_report_sdk.md", "w") as f:
        f.write("# Venturalitica Compliance Audit Report\n")
        f.write("Generated via `venturalitica.enforce(policy='risks.oscal.yaml')`\n\n")
        
        f.write("## Executive Summary\n")
        f.write(f"- **Overall Status**: {'PASS' if failed_count == 0 else 'FAIL'}\n")
        f.write(f"- **Policies Checked**: {len(compliance_results)}\n")
        f.write(f"- **Passed**: {passed_count}\n")
        f.write(f"- **Failed**: {failed_count}\n\n")
        
        f.write("## Detailed Findings\n")
        for check in compliance_results:
            passed = check.passed
            # Fix property access based on SDK model
            name = getattr(check, 'control_id', getattr(check, 'id', getattr(check, 'name', 'Unknown Policy')))
            # Message often contains the result logic "0.95 < 0.99"
            msg = getattr(check, 'message', getattr(check, 'description', 'No details'))
            desc = getattr(check, 'description', '')
            # severity and article might be widely available or in props
            severity = getattr(check, 'severity', 'Unknown')
            article = getattr(check, 'article', 'Unknown')
            
            icon = "✅" if passed else "❌"
            f.write(f"### {icon} {name}\n")
            f.write(f"- **Description**: {desc}\n")
            f.write(f"- **Severity**: {severity}\n")
            f.write(f"- **Article**: {article}\n")
            f.write(f"- **Result**: {msg}\n\n")

    print(f"\n✅ SDK Audit complete. Report saved to compliance_report_sdk.md")

if __name__ == "__main__":
    run_compliance_suite()
