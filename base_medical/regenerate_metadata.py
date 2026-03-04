
import pydicom
import pandas as pd
from pathlib import Path

SCENARIO_ROOT = Path(__file__).parent.parent
SHARED_DATA = SCENARIO_ROOT / "shared_data"


def safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except (ValueError, TypeError):
        return default


def extract_metadata(data_dir, output_path=None):
    """
    Walk the data directory, find the first CT slice per patient,
    and extract trusted metadata from DICOM headers.
    """
    import dicom_utils

    data_path = Path(data_dir)
    patient_folders = sorted([d for d in data_path.iterdir() if d.is_dir()])

    if not patient_folders:
        print(f"  No patient folders found in {data_dir}")
        return pd.DataFrame()

    metadata_list = []
    print(f"  Extracting DICOM metadata from {len(patient_folders)} patients...")

    for patient_dir in patient_folders:
        try:
            ct_files, _ = dicom_utils.find_ct_and_seg_files(patient_dir)
            if not ct_files:
                continue

            first_ct = sorted(ct_files)[0]
            ds = pydicom.dcmread(first_ct, stop_before_pixels=True)

            record = {
                "PatientID": str(getattr(ds, "PatientID", patient_dir.name)).strip(),
                "Sex": str(getattr(ds, "PatientSex", "")).strip(),
                "Age": str(getattr(ds, "PatientAge", "")).strip(),
                "Manufacturer": str(getattr(ds, "Manufacturer", "Unknown")).strip(),
                "ModelName": str(getattr(ds, "ManufacturerModelName", "Unknown")).strip(),
                "KVP": safe_float(getattr(ds, "KVP", 0.0)),
                "SliceThickness": safe_float(getattr(ds, "SliceThickness", 0.0)),
                "Exposure": safe_float(getattr(ds, "Exposure", 0.0)),
                "StudyDate": str(getattr(ds, "StudyDate", "Unknown")),
                "PixelSpacing0": safe_float(ds.PixelSpacing[0]) if hasattr(ds, "PixelSpacing") and ds.PixelSpacing else 0.0,
                "PixelSpacing1": safe_float(ds.PixelSpacing[1]) if hasattr(ds, "PixelSpacing") and ds.PixelSpacing else 0.0,
                "PatientWeight": safe_float(getattr(ds, "PatientWeight", 0.0)),
                "PatientSize": safe_float(getattr(ds, "PatientSize", 0.0)),
            }

            # Clean Age "055Y" -> 55
            age_str = record["Age"]
            if age_str.endswith("Y") and age_str[:-1].isdigit():
                record["Age"] = int(age_str[:-1])
            elif age_str.isdigit():
                record["Age"] = int(age_str)
            else:
                record["Age"] = None

            # Clean empty Sex
            if record["Sex"] not in ("M", "F"):
                record["Sex"] = None

            metadata_list.append(record)

        except Exception as e:
            print(f"    Error processing {patient_dir.name}: {e}")

    df = pd.DataFrame(metadata_list)

    out = output_path or (SHARED_DATA / "trusted_metadata.csv")
    df.to_csv(out, index=False)
    print(f"  Saved metadata for {len(df)} patients to {out}")

    if "Manufacturer" in df.columns:
        print(f"  Manufacturers: {dict(df['Manufacturer'].value_counts())}")
    if "Sex" in df.columns:
        print(f"  Sex distribution: {dict(df['Sex'].value_counts())}")

    return df


if __name__ == "__main__":
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else str(SHARED_DATA / "dicom")
    extract_metadata(data_dir)
