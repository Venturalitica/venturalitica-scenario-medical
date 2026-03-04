#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "nbiatoolkit",
#     "pandas",
#     "requests",
#     "openpyxl",
# ]
# ///
"""
Download SPINE-METS-CT-SEG DICOM data from The Cancer Imaging Archive (TCIA).

Dataset: https://www.cancerimagingarchive.net/collection/spine-mets-ct-seg/

Usage:
    uv run download_data.py                   # Download all patients (~50 GB)
    uv run download_data.py --limit 5         # Download first 5 patients only
    uv run download_data.py --patients 10543,12459   # Download specific patients
"""

import argparse
import io
import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from nbiatoolkit import NBIAClient

COLLECTION = "Spine-Mets-CT-Seg"
CLINICAL_XLSX_URL = "https://www.cancerimagingarchive.net/wp-content/uploads/Spine-Mets-CT-SEG_Clinical.xlsx"
SCENARIO_ROOT = Path(__file__).parent
SHARED_DATA = SCENARIO_ROOT / "shared_data"
DICOM_DIR = SHARED_DATA / "dicom"
PROGRESS_FILE = SHARED_DATA / ".progress" / "downloaded_series.json"


def with_retries(func, *args, retries=3, backoff=1.0, **kwargs):
    for attempt in range(1, retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt == retries:
                raise
            wait = backoff * attempt
            print(f"   Retry {attempt}/{retries}: {e}. Waiting {wait}s...")
            time.sleep(wait)


def load_progress() -> dict:
    try:
        if PROGRESS_FILE.exists():
            return json.loads(PROGRESS_FILE.read_text())
    except Exception:
        pass
    return {}


def save_progress(progress: dict):
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))


def get_cohort_patient_ids() -> list[str]:
    """Get patient IDs from cohort_results.csv."""
    csv_path = SHARED_DATA / "cohort_results.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        return df["PatientID"].astype(str).tolist()
    return []


def download_clinical_metadata() -> pd.DataFrame:
    """Download clinical spreadsheet from TCIA."""
    print(f"\n  Downloading clinical metadata...")
    try:
        resp = requests.get(CLINICAL_XLSX_URL, timeout=30)
        resp.raise_for_status()
        sheets = pd.read_excel(io.BytesIO(resp.content), sheet_name=None)
        dfs = [df for df in sheets.values() if isinstance(df, pd.DataFrame)]
        if not dfs:
            print("    No tabular sheets found")
            return pd.DataFrame()
        clinical_df = pd.concat(dfs, ignore_index=True, sort=False)
        # Save
        out_path = SHARED_DATA / "clinical_metadata.csv"
        clinical_df.to_csv(out_path, index=False)
        print(f"    Saved {len(clinical_df)} rows to {out_path.name}")
        return clinical_df
    except Exception as e:
        print(f"    Failed: {e}")
        return pd.DataFrame()


def download_dicoms(patient_ids: list[str]):
    """Download DICOM files for given patient IDs from TCIA."""
    DICOM_DIR.mkdir(parents=True, exist_ok=True)
    client = NBIAClient()
    progress = load_progress()
    total = len(patient_ids)

    print(f"\n  Downloading DICOMs for {total} patients to {DICOM_DIR}/")

    for i, pid in enumerate(patient_ids, 1):
        patient_dir = DICOM_DIR / str(pid)

        # Skip if already fully downloaded
        if pid in progress and len(progress[pid]) > 0:
            existing_files = list(patient_dir.rglob("*.dcm")) if patient_dir.exists() else []
            if existing_files:
                print(f"  [{i}/{total}] {pid} -- already downloaded ({len(existing_files)} files), skipping")
                continue

        print(f"  [{i}/{total}] {pid} -- fetching series list...")

        try:
            series = with_retries(client.getSeries, Collection=COLLECTION, PatientID=pid)
            if not series:
                print(f"    No series found for {pid}")
                continue

            patient_dir.mkdir(parents=True, exist_ok=True)
            for sidx, s in enumerate(series, 1):
                series_uid = s.get("SeriesInstanceUID") if isinstance(s, dict) else s
                print(f"    Series [{sidx}/{len(series)}]: {str(series_uid)[:30]}...", end=" ")
                try:
                    with_retries(
                        client.downloadSeries,
                        SeriesInstanceUID=series_uid,
                        downloadDir=str(patient_dir),
                    )
                    progress.setdefault(pid, [])
                    progress[pid].append(str(series_uid))
                    save_progress(progress)
                    # Count downloaded files
                    n_files = len(list(patient_dir.rglob("*.dcm")))
                    print(f"OK ({n_files} files)")
                except Exception as e:
                    print(f"FAILED: {e}")

        except Exception as e:
            print(f"    Error: {e}")

    # Summary
    total_files = len(list(DICOM_DIR.rglob("*.dcm")))
    total_patients = len([d for d in DICOM_DIR.iterdir() if d.is_dir()])
    print(f"\n  Download complete: {total_patients} patients, {total_files} DICOM files")


def main():
    parser = argparse.ArgumentParser(description="Download SPINE-METS-CT-SEG data from TCIA")
    parser.add_argument("--limit", type=int, help="Limit number of patients to download")
    parser.add_argument(
        "--patients",
        type=str,
        help="Comma-separated patient IDs (e.g., 10543,12459)",
    )
    parser.add_argument(
        "--skip-clinical",
        action="store_true",
        help="Skip clinical metadata download",
    )
    parser.add_argument(
        "--skip-dicom",
        action="store_true",
        help="Skip DICOM download (metadata only)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  SPINE-METS-CT-SEG Data Download")
    print(f"  Source: {COLLECTION} (TCIA)")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Determine patient list
    if args.patients:
        patient_ids = [p.strip() for p in args.patients.split(",")]
        print(f"\n  Target: {len(patient_ids)} specific patients")
    else:
        # Use cohort patient IDs or fetch from TCIA
        patient_ids = get_cohort_patient_ids()
        if patient_ids:
            print(f"\n  Using {len(patient_ids)} patients from cohort_results.csv")
        else:
            print("\n  Fetching patient list from TCIA...")
            client = NBIAClient()
            raw_patients = with_retries(client.getPatients, Collection=COLLECTION)
            # NBIAClient returns dicts like {"PatientId": "10543", ...} — extract IDs
            patient_ids = []
            for p in raw_patients:
                if isinstance(p, dict):
                    pid = p.get("PatientId") or p.get("PatientID") or p.get("patientId")
                    if pid:
                        patient_ids.append(str(pid))
                else:
                    patient_ids.append(str(p))
            print(f"  Found {len(patient_ids)} patients in collection")

        if args.limit:
            patient_ids = patient_ids[: args.limit]
            print(f"  Limited to {args.limit} patients")

    # Step 1: Clinical metadata
    if not args.skip_clinical:
        download_clinical_metadata()

    # Step 2: DICOM files
    if not args.skip_dicom:
        download_dicoms(patient_ids)
    else:
        print("\n  Skipping DICOM download (--skip-dicom)")

    print(f"\n  Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
