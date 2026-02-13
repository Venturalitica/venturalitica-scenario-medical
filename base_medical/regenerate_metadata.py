
import os
import pydicom
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import dicom_utils

def safe_float(value, default=0.0):
    try:
        if value is None: return default
        return float(value)
    except (ValueError, TypeError):
        return default

def extract_metadata(data_dir):
    """
    Walks through the data directory, finds the first valid CT slice for each patient,
    and extracts trusted metadata directly from DICOM headers.
    """
    data_path = Path(data_dir)
    patient_folders = [d for d in data_path.iterdir() if d.is_dir()]
    
    metadata_list = []
    
    print(f"🔍 Scanning {len(patient_folders)} patient folders in {data_dir}...")
    
    for patient_dir in tqdm(patient_folders):
        try:
            # Reuse robustness logic to find CT files
            ct_files, _ = dicom_utils.find_ct_and_seg_files(patient_dir)
            
            if not ct_files:
                print(f"⚠️ No CT files found for {patient_dir.name}")
                continue
                
            # Sort to get the first file safely
            ct_files = sorted(ct_files)
            first_ct = ct_files[0]
            
            ds = pydicom.dcmread(first_ct, stop_before_pixels=True)
            
            # Extract tags safely
            record = {
                'PatientID': str(getattr(ds, 'PatientID', patient_dir.name)).strip(),
                'Sex': str(getattr(ds, 'PatientSex', 'Unknown')).strip(),
                'Age': str(getattr(ds, 'PatientAge', '000Y')).strip(),
                'Manufacturer': str(getattr(ds, 'Manufacturer', 'Unknown')).strip(),
                'ModelName': str(getattr(ds, 'ManufacturerModelName', 'Unknown')).strip(),
                'KVP': safe_float(getattr(ds, 'KVP', 0.0)),
                'SliceThickness': safe_float(getattr(ds, 'SliceThickness', 0.0)),
                'Exposure': safe_float(getattr(ds, 'Exposure', 0.0)),
                'StudyDate': str(getattr(ds, 'StudyDate', 'Unknown')),
                'PixelSpacing0': safe_float(ds.PixelSpacing[0]) if hasattr(ds, 'PixelSpacing') and ds.PixelSpacing else 0.0,
                'PixelSpacing1': safe_float(ds.PixelSpacing[1]) if hasattr(ds, 'PixelSpacing') and ds.PixelSpacing else 0.0,
                'PatientWeight': safe_float(getattr(ds, 'PatientWeight', 0.0)),
                'PatientSize': safe_float(getattr(ds, 'PatientSize', 0.0)),
            }
            
            # Clean Age "055Y" -> 55
            age_str = record['Age']
            if age_str.endswith('Y') and age_str[:-1].isdigit():
                record['Age'] = int(age_str[:-1])
            elif age_str.isdigit():
                 record['Age'] = int(age_str)
            else:
                 record['Age'] = None
            
            metadata_list.append(record)
            
        except Exception as e:
            print(f"❌ Error processing {patient_dir.name}: {e}")
            
    # Convert to DataFrame
    df = pd.DataFrame(metadata_list)
    output_path = Path("trusted_metadata.csv")
    df.to_csv(output_path, index=False)
    
    print(f"✅ Extracted metadata for {len(df)} patients.")
    print(f"💾 Saved to {output_path.absolute()}")
    print("\n--- Summary ---")
    print(df[['Sex', 'Manufacturer', 'ModelName']].describe())
    if 'Sex' in df.columns:
        print("\nSex Distribution:\n", df['Sex'].value_counts())
    if 'Manufacturer' in df.columns:
        print("\nManufacturers:\n", df['Manufacturer'].value_counts())

if __name__ == "__main__":
    # Point to the actual DICOM directory
    # Based on previous exploration: ../../../venturalitica-sdk-samples-extra/scenarios/surgery-dicom-tcia/data/dicom
    dicom_root = "../../../venturalitica-sdk-samples-extra/scenarios/surgery-dicom-tcia/data/dicom"
    extract_metadata(dicom_root)
