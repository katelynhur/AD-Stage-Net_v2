#!/usr/bin/env python3
"""
apply_subject_split.py

Loads the labeled OASIS-3 data and assigns each row/visit to train,
validation, or test based on subject_id from the pre-computed
subject_split.json (split_subjects.py). Verifies zero subject overlap
between splits before writing anything out.

Usage:
    AD_STAGE_NET_ROOT=/path/to/your/oasis3-working-directory \
        python apply_subject_split.py
"""

import json
import os
from pathlib import Path
from typing import Dict, Set

import pandas as pd

ROOT = Path(os.environ.get("AD_STAGE_NET_ROOT", "/path/to/your/oasis3-working-directory"))
DATA_DIR = ROOT / "data" / "processed"
SPLITS_DIR = DATA_DIR / "splits"

LABELED_CSV = DATA_DIR / "oasis3_labeled.csv"
SUBJECT_SPLIT_JSON = SPLITS_DIR / "subject_split.json"

TRAIN_CSV = SPLITS_DIR / "train.csv"
VAL_CSV = SPLITS_DIR / "val.csv"
TEST_CSV = SPLITS_DIR / "test.csv"


def load_data():
    print(f"Loading labeled data from: {LABELED_CSV}")
    df = pd.read_csv(LABELED_CSV)
    print(f"  Total rows: {len(df)}")

    print(f"\nLoading subject split from: {SUBJECT_SPLIT_JSON}")
    with open(SUBJECT_SPLIT_JSON) as f:
        split_data = json.load(f)

    train_subjects = set(split_data["train_subjects"])
    val_subjects = set(split_data["val_subjects"])
    test_subjects = set(split_data["test_subjects"])

    print(f"  Seed: {split_data['seed']}")
    print(f"  Train subjects: {len(train_subjects)}")
    print(f"  Val subjects: {len(val_subjects)}")
    print(f"  Test subjects: {len(test_subjects)}")

    return df, {"train": train_subjects, "val": val_subjects, "test": test_subjects}


def assign_splits(df: pd.DataFrame, subject_splits: Dict[str, Set[str]]):
    train_mask = df["subject_id"].isin(subject_splits["train"])
    val_mask = df["subject_id"].isin(subject_splits["val"])
    test_mask = df["subject_id"].isin(subject_splits["test"])
    unassigned_mask = ~(train_mask | val_mask | test_mask)

    splits = {
        "train": df[train_mask].copy(),
        "val": df[val_mask].copy(),
        "test": df[test_mask].copy(),
    }
    return splits, unassigned_mask


def verify_no_overlap(splits: Dict[str, pd.DataFrame]) -> bool:
    train_subjects = set(splits["train"]["subject_id"].unique())
    val_subjects = set(splits["val"]["subject_id"].unique())
    test_subjects = set(splits["test"]["subject_id"].unique())

    overlaps = {
        "train/val": train_subjects & val_subjects,
        "train/test": train_subjects & test_subjects,
        "val/test": val_subjects & test_subjects,
    }
    overlap_found = any(overlaps.values())
    for name, ov in overlaps.items():
        if ov:
            print(f"ERROR: {len(ov)} subjects appear in both {name}: {sorted(ov)[:5]}...")
    if not overlap_found:
        print("CONFIRMED: zero subject overlap between train/val/test")
    return not overlap_found


def save_splits(splits: Dict[str, pd.DataFrame]):
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for split_name, split_df in splits.items():
        out_path = SPLITS_DIR / f"{split_name}.csv"
        split_df.to_csv(out_path, index=False)
        print(f"  Saved {len(split_df)} rows to: {out_path}")


def main():
    df, subject_splits = load_data()
    splits, unassigned_mask = assign_splits(df, subject_splits)
    no_overlap = verify_no_overlap(splits)

    n_unassigned = int(unassigned_mask.sum())
    if n_unassigned:
        print(f"WARNING: {n_unassigned} rows belong to subjects not in any split")

    save_splits(splits)

    if not no_overlap:
        raise RuntimeError("Subject overlap detected between splits -- see errors above")
    print("Success: subject-based split applied with zero cross-split leakage.")


if __name__ == "__main__":
    main()
