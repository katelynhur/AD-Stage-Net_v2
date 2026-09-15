#!/usr/bin/env python3
"""
merge_oasis3.py

Merges OASIS-3's CDR/diagnosis, demographics, and FreeSurfer volumetric CSVs
into one flat visit-level table, ready for the subject-level train/val/test
split and model training.

Sources used (paths relative to OASIS3_data_files/scans/):
    UDSb4-Form_B4.../resources/csv/files/OASIS3_UDSb4_cdr.csv
        -> MMSE, CDRSUM, CDRTOT (stage label), dx1-dx5 (etiology label)
    demo-demographics/resources/csv/files/OASIS3_demographics.csv
        -> AgeatEntry, GENDER, EDUC, APOE, daddem, momdem, HAND (subject-level, one row per subject)
    FS-Freesurfer_output/resources/csv/files/OASIS3_Freesurfer_output.csv
        -> hippocampal/ventricle volumes, cortical thickness, ICV (one row per MR session)

Usage:
    python merge_oasis3.py \
        --data_root $AD_STAGE_NET_ROOT/data/raw/oasis3/OASIS3_data_files/scans \
        --output $AD_STAGE_NET_ROOT/data/processed/oasis3_merged.csv \
        --mri_match_tolerance_days 365
"""

import argparse
import re
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_day(session_label: str) -> float:
    """Pulls the integer day-from-entry out of an OASIS session label like
    'OAS30001_UDSb4_d0000' or 'OAS30001_MR_d0129' -> 0, 129."""
    if pd.isna(session_label):
        return float("nan")
    match = re.search(r"_d(\d+)$", str(session_label))
    return float(match.group(1)) if match else float("nan")


# ---------------------------------------------------------------------------
# Step 1: CDR / diagnosis table (primary label source)
# ---------------------------------------------------------------------------

def load_cdr(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={
        "OASISID": "subject_id",
        "days_to_visit": "days_from_entry",
        "age at visit": "age_at_visit",
        "CDRTOT": "cdr_global",
        "CDRSUM": "cdr_sum_of_boxes",
        "MMSE": "mmse",
        "dx1": "dx1",
    })
    keep = ["subject_id", "days_from_entry", "age_at_visit", "mmse",
            "cdr_global", "cdr_sum_of_boxes", "dx1"]
    df = df[keep].copy()
    df["days_from_entry"] = pd.to_numeric(df["days_from_entry"], errors="coerce")
    df["cdr_global"] = pd.to_numeric(df["cdr_global"], errors="coerce")
    df["cdr_sum_of_boxes"] = pd.to_numeric(df["cdr_sum_of_boxes"], errors="coerce")
    df["mmse"] = pd.to_numeric(df["mmse"], errors="coerce")
    df = df.dropna(subset=["subject_id", "days_from_entry", "cdr_global"])
    return df


# ---------------------------------------------------------------------------
# Step 2: Subject-level demographics
# ---------------------------------------------------------------------------

def load_demographics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={
        "OASISID": "subject_id",
        "AgeatEntry": "age_at_entry",
        "GENDER": "sex",
        "EDUC": "education_years",
        "APOE": "apoe",
        "daddem": "father_dementia",
        "momdem": "mother_dementia",
        "HAND": "handedness",
    })
    keep = ["subject_id", "age_at_entry", "sex", "education_years",
            "apoe", "father_dementia", "mother_dementia", "handedness"]
    df = df[keep].copy()
    df = df.drop_duplicates(subset="subject_id")
    return df


# ---------------------------------------------------------------------------
# Step 3: FreeSurfer volumetrics (needs nearest-day matching, not exact join)
# ---------------------------------------------------------------------------

FS_FEATURE_MAP = {
    "IntraCranialVol": "intracranial_volume",
    "TotalGrayVol": "total_gray_matter_volume",
    "Left-Hippocampus_volume": "hippocampal_volume_l",
    "Right-Hippocampus_volume": "hippocampal_volume_r",
    "TOTAL_HIPPOCAMPUS_VOLUME": "hippocampal_volume_total",
    "Left-Lateral-Ventricle_volume": "ventricle_volume_l",
    "Right-Lateral-Ventricle_volume": "ventricle_volume_r",
    "lh_entorhinal_thickness": "entorhinal_thickness_l",
    "rh_entorhinal_thickness": "entorhinal_thickness_r",
}


def load_freesurfer(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Keep only QC-passed segmentations -- failed QC volumes are not reliable
    # features and will add noise rather than signal.
    if "FS QC Status" in df.columns:
        before = len(df)
        df = df[df["FS QC Status"].astype(str).str.strip().str.lower() == "passed"]
        print(f"[freesurfer] Kept {len(df)}/{before} rows with QC status 'Passed'")

    df["subject_id"] = df["Subject"]
    df["days_from_entry"] = df["MR_session"].apply(extract_day)

    rename = {k: v for k, v in FS_FEATURE_MAP.items() if k in df.columns}
    missing = [k for k in FS_FEATURE_MAP if k not in df.columns]
    if missing:
        print(f"[freesurfer] WARNING - expected columns not found, skipping: {missing}")

    keep = ["subject_id", "days_from_entry"] + list(rename.keys())
    df = df[keep].rename(columns=rename)
    df = df.dropna(subset=["subject_id", "days_from_entry"])
    return df


# ---------------------------------------------------------------------------
# Step 4: Nearest-day merge of CDR visits <-> FreeSurfer sessions, per subject
# ---------------------------------------------------------------------------

def nearest_merge_freesurfer(cdr_df: pd.DataFrame, fs_df: pd.DataFrame,
                              tolerance_days: float) -> pd.DataFrame:
    """
    For each CDR visit, attaches the FreeSurfer volumetrics from the closest
    MR session for that SAME subject, as long as it's within tolerance_days.
    Visits with no MRI close enough simply get NaN for all MRI columns --
    this is exactly the missingness the tree models are designed to handle.
    """
    cdr_sorted = cdr_df.sort_values("days_from_entry").reset_index(drop=True)
    fs_sorted = fs_df.sort_values("days_from_entry").reset_index(drop=True)

    cdr_sorted["days_from_entry"] = cdr_sorted["days_from_entry"].astype("float64")
    fs_sorted["days_from_entry"] = fs_sorted["days_from_entry"].astype("float64")

    merged = pd.merge_asof(
        cdr_sorted,
        fs_sorted,
        on="days_from_entry",
        by="subject_id",
        direction="nearest",
        tolerance=tolerance_days,
    )
    n_matched = merged["intracranial_volume"].notna().sum() if "intracranial_volume" in merged.columns else 0
    print(f"[merge] {n_matched}/{len(merged)} visits matched to an MRI session within {tolerance_days} days")
    return merged


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True,
                         help="Path to OASIS3_data_files/scans/ (contains the UDSb4-..., demo-..., FS-... folders)")
    parser.add_argument("--output", required=True, help="Where to write the merged CSV")
    parser.add_argument("--mri_match_tolerance_days", type=float, default=365,
                         help="Max days between a clinical visit and an MRI session to still match them")
    args = parser.parse_args()

    root = Path(args.data_root)

    cdr_path = next(root.glob("UDSb4-Form_B4*/resources/csv/files/*.csv"))
    demo_path = next(root.glob("demo-demographics/resources/csv/files/*.csv"))
    fs_path = next(root.glob("FS-Freesurfer_output/resources/csv/files/*.csv"))

    print(f"Loading CDR/diagnosis: {cdr_path}")
    cdr_df = load_cdr(cdr_path)
    print(f"  -> {len(cdr_df)} visits, {cdr_df['subject_id'].nunique()} subjects")

    print(f"Loading demographics: {demo_path}")
    demo_df = load_demographics(demo_path)
    print(f"  -> {len(demo_df)} subjects")

    print(f"Loading FreeSurfer: {fs_path}")
    fs_df = load_freesurfer(fs_path)
    print(f"  -> {len(fs_df)} MR sessions, {fs_df['subject_id'].nunique()} subjects")

    print("Merging FreeSurfer onto CDR visits (nearest-day match)...")
    merged = nearest_merge_freesurfer(cdr_df, fs_df, args.mri_match_tolerance_days)

    print("Merging static demographics (by subject)...")
    merged = merged.merge(demo_df, on="subject_id", how="left")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)

    print(f"\nDone. Merged table: {len(merged)} visit-rows, {merged['subject_id'].nunique()} subjects")
    print(f"Saved to: {out_path}")
    print("\nColumn summary (non-null counts):")
    print(merged.notna().sum().sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
