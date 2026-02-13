import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def set_style():
    plt.style.use('bmh') # Use a built-in matplotlib style that looks good
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'

def generate_plots():
    set_style()
    output_dir = Path("compliance_plots")
    output_dir.mkdir(exist_ok=True)

    # Load Data
    try:
        results = pd.read_csv("cohort_results.csv")
        trusted = pd.read_csv("trusted_metadata.csv")
        trusted['PatientID'] = trusted['PatientID'].astype(str)
        results['PatientID'] = results['PatientID'].astype(str)
        # Fix merge logic
        df = results.merge(trusted, on='PatientID', how='left')
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 1. Scanner Bias Plot (Boxplot)
    plt.figure(figsize=(10, 6))
    
    # Filter for main manufacturers
    mfr_counts = df['Manufacturer'].value_counts()
    valid_mfrs = mfr_counts[mfr_counts > 2].index
    plot_df = df[df['Manufacturer'].isin(valid_mfrs)].copy()
    
    # Clean Manufacturer Names
    plot_df['Manufacturer'] = plot_df['Manufacturer'].replace({
        'GE MEDICAL SYSTEMS': 'GE',
        'SIEMENS': 'Siemens'
    })

    # Group data for boxplot
    data_to_plot = []
    labels = []
    colors = ['#3498db', '#e74c3c']
    for mfr in plot_df['Manufacturer'].unique():
        data_to_plot.append(plot_df[plot_df['Manufacturer'] == mfr]['Dice'].values)
        labels.append(mfr)

    bplot = plt.boxplot(data_to_plot, patch_artist=True, labels=labels, widths=0.5)
    
    # Color boxes
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
        
    # Jitter plot
    for i, mfr in enumerate(labels):
        y = plot_df[plot_df['Manufacturer'] == mfr]['Dice']
        x = np.random.normal(i+1, 0.04, size=len(y))
        plt.plot(x, y, 'k.', alpha=0.3)
    
    plt.axhline(y=0.85, color='r', linestyle='--', label='Threshold (0.85)')
    plt.title('Scanner Robustness: Siemens vs GE', fontsize=14, pad=20)
    plt.ylabel('Dice Similarity Coefficient')
    plt.ylim(0.0, 1.0)
    plt.legend(loc='lower right')
    
    plt.savefig(output_dir / "scanner_bias.png")
    print("Generated scanner_bias.png")
    plt.close()

    # 2. Demographic Parity (Pie Chart)
    plt.figure(figsize=(8, 8))
    
    sex_clean = df['Sex'].fillna('Unknown').replace({'': 'Unknown', 'nan': 'Unknown'})
    sex_counts = sex_clean.value_counts()
    
    plt.pie(sex_counts, labels=sex_counts.index, autopct='%1.1f%%', colors=['#ff9999','#66b3ff','#99ff99'])
    plt.title('Cohort Sex Distribution', fontsize=14)
    plt.savefig(output_dir / "sex_distribution.png")
    print("Generated sex_distribution.png")
    plt.close()

    # 3. Lesion Type Robustness (Bar Chart)
    try:
        meta_path = Path("../../../venturalitica-sdk-samples-extra/scenarios/surgery-dicom-tcia/data/combined_metadata.csv")
        if meta_path.exists():
            clinical_df = pd.read_csv(meta_path)
            clinical_df['Case'] = pd.to_numeric(clinical_df['Case'], errors='coerce').fillna(0).astype(int).astype(str)
            df_full = df.merge(clinical_df, left_on='PatientID', right_on='Case', how='left')
            
            lesion_perf = {}
            for ltype in ['Lytic', 'Blastic', 'Mixed']:
                 if ltype in df_full.columns:
                     perf = df_full[df_full[ltype].notna()]['Dice'].mean()
                     # If nan, skip
                     if not np.isnan(perf):
                        lesion_perf[ltype] = perf
            
            if lesion_perf:
                plt.figure(figsize=(8, 5))
                bars = plt.bar(lesion_perf.keys(), lesion_perf.values(), color='#2ecc71')
                plt.axhline(y=0.80, color='gray', linestyle='--', label='Threshold (0.80)')
                plt.ylim(0.0, 1.0)
                plt.title('Performance by Lesion Phenotype', fontsize=14, pad=15)
                plt.ylabel('Mean Dice Score')
                plt.legend()
                
                # Add labels
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.2f}',
                            ha='center', va='bottom')
                
                plt.savefig(output_dir / "lesion_robustness.png")
                print("Generated lesion_robustness.png")
                plt.close()
    except Exception as e:
        print(f"Skipped lesion plot: {e}")

    # 4. Correlation Plot (Safety)
    plt.figure(figsize=(8, 6))
    plt.scatter(df['Confidence'], df['Dice'], alpha=0.5, color='blue')
    
    # Fit line
    m, b = np.polyfit(df['Confidence'], df['Dice'], 1)
    plt.plot(df['Confidence'], m*df['Confidence'] + b, color='red')
    
    plt.title('Safety Calibration: Confidence vs Performance', fontsize=14)
    plt.xlabel('Model Confidence')
    plt.ylabel('Dice Score')
    plt.xlim(0.8, 1.0)
    plt.ylim(0.0, 1.0)
    plt.savefig(output_dir / "safety_calibration.png")
    print("Generated safety_calibration.png")
    plt.close()

if __name__ == "__main__":
    generate_plots()
