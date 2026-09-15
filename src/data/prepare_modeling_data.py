#!/usr/bin/env python3
"""
prepare_modeling_data.py

Takes the subject-split train/val/test CSVs (already built by
apply_subject_split.py) and produces clean, leakage-free feature/label
tables for this project's two prediction tasks:

    progression  -- predict progresses_next_visit (any-cause CDR increase
                    at the next visit)
    development  -- predict converts_to_ad_preclinical, restricted to
                    CDR=0 (cognitively normal) baseline rows with a defined
                    label (AD-specific dementia evidence within 3 years)

LEAKAGE COLUMNS (excluded from features in both tasks, always):
    progresses_next_visit          -- target for the progression task
    days_to_next_visit             -- derived from the NEXT visit; if a row
                                       has this populated, a next visit
                                       necessarily existed, which itself
                                       leaks progression info
    converts_to_ad_preclinical     -- target for the development task
    days_to_ad_evidence_preclinical -- derived from a LATER visit's diagnosis
    subject_id                     -- identifier; risk of the model
                                       memorizing individual subjects rather
                                       than generalizing

Both tasks keep cdr_global, cdr_sum_of_boxes, dx1, and mmse as ordinary
CURRENT-VISIT features -- using what's known NOW to predict what happens
LATER is exactly the point, and is not leakage.

CATEGORICAL COLUMNS:
    dx1, sex, handedness, apoe are cast to pandas 'category' dtype so
    LightGBM can use its native categorical handling instead of needing
    one-hot encoding.

Usage:
    python prepare_modeling_data.py \
        --splits_dir $AD_STAGE_NET_ROOT/data/processed/splits \
        --output_dir $AD_STAGE_NET_ROOT/data/processed/modeling
"""

import argparse
from pathlib import Path

import pandas as pd

LEAKAGE_COLUMNS = [
    "progresses_next_visit",
    "days_to_next_visit",
    "converts_to_ad_preclinical",
    "days_to_ad_evidence_preclinical",
    "subject_id",
]

CATEGORICAL_COLUMNS = ["dx1", "sex", "handedness", "apoe"]

TASK_TARGETS = {
    "progression": "progresses_next_visit",
    "development": "converts_to_ad_preclinical",
}


def cast_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


def build_progression_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Only rows with a DEFINED progresses_next_visit label are usable --
    rows where it's NaN are censored (subject's last recorded visit) and
    must be dropped, not treated as a negative."""
    valid = df.dropna(subset=[TASK_TARGETS["progression"]]).copy()
    exclude = set(LEAKAGE_COLUMNS)
    feature_cols = [c for c in valid.columns if c not in exclude and c != TASK_TARGETS["progression"]]
    out = valid[feature_cols + [TASK_TARGETS["progression"]]].copy()
    out = out.rename(columns={TASK_TARGETS["progression"]: "target"})
    out["target"] = out["target"].astype(int)
    return cast_categoricals(out)


def build_development_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Restricted to CDR=0 visits with a DEFINED converts_to_ad_preclinical
    label. (build_labels.py already only populates this label for CDR=0
    rows with valid follow-up, but the explicit filter here makes the
    restriction visible in this script too.)"""
    valid = df[df["cdr_global"] == 0.0].dropna(subset=[TASK_TARGETS["development"]]).copy()
    exclude = set(LEAKAGE_COLUMNS)
    feature_cols = [c for c in valid.columns if c not in exclude and c != TASK_TARGETS["development"]]
    out = valid[feature_cols + [TASK_TARGETS["development"]]].copy()
    out = out.rename(columns={TASK_TARGETS["development"]: "target"})
    out["target"] = out["target"].astype(int)
    return cast_categoricals(out)


TASK_BUILDERS = {
    "progression": build_progression_dataset,
    "development": build_development_dataset,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits_dir", required=True,
                         help="Directory containing train.csv, val.csv, test.csv")
    parser.add_argument("--output_dir", required=True,
                         help="Where to write data/processed/modeling/<task>/<split>.csv")
    args = parser.parse_args()

    splits_dir = Path(args.splits_dir)
    output_dir = Path(args.output_dir)

    splits = {}
    for split_name in ["train", "val", "test"]:
        path = splits_dir / f"{split_name}.csv"
        splits[split_name] = pd.read_csv(path)
        print(f"Loaded {split_name}: {len(splits[split_name])} rows")

    for task_name, builder in TASK_BUILDERS.items():
        print(f"\n=== Task: {task_name} ===")
        task_dir = output_dir / task_name
        task_dir.mkdir(parents=True, exist_ok=True)

        for split_name, df in splits.items():
            task_df = builder(df)
            out_path = task_dir / f"{split_name}.csv"
            task_df.to_csv(out_path, index=False)

            n_subjects = df.loc[task_df.index, "subject_id"].nunique() if "subject_id" in df.columns else None
            print(f"  {split_name}: {len(task_df)} rows"
                  + (f", {n_subjects} subjects" if n_subjects is not None else ""))
            print(f"    target distribution: {task_df['target'].value_counts().to_dict()}")

    print(f"\nAll modeling datasets written under: {output_dir}")


if __name__ == "__main__":
    main()
