"""
split_subjects.py

Creates a SUBJECT-LEVEL train/val/test split for OASIS-3 and saves it to
disk. Run this ONCE, early, on the merged clinical table. Never regenerate
this file once created -- every downstream script (baseline models,
MRI/ROI-feature models) must load the SAME split so results stay
comparable, and no individual may cross between splits.

Why subject-level (not visit-level):
    OASIS-3 has multiple visits per person. If visits from the same subject
    land in both train and test, the model can partly "memorize" that
    person from training and get an inflated, non-generalizing test score.
    Every split must group by subject_id first, then assign the WHOLE
    subject to one split.

Why stratified (not pure random):
    Outcome classes are imbalanced (few CDR=2/3 subjects; a minority of
    CDR=0 subjects actually develop AD within 3 years). Pure random subject
    splitting risks a test set with almost no positive cases, making
    metrics noisy and unstable. Stratifying by each subject's BASELINE CDR
    class keeps the class balance roughly consistent across splits.

Usage:
    python split_subjects.py --input $AD_STAGE_NET_ROOT/data/processed/oasis3_merged.csv \
        --output_dir $AD_STAGE_NET_ROOT/data/processed/splits --seed 42
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def build_subject_level_table(df: pd.DataFrame, subject_col: str, cdr_col: str,
                                days_col: str) -> pd.DataFrame:
    """
    Reduces the visit-level table to one row per subject, using each
    subject's EARLIEST visit CDR as the stratification label.
    """
    df_sorted = df.sort_values([subject_col, days_col])
    baseline = df_sorted.groupby(subject_col).first().reset_index()
    return baseline[[subject_col, cdr_col]].rename(columns={cdr_col: "baseline_cdr"})


def make_split(subject_table: pd.DataFrame, subject_col: str, seed: int,
               train_frac: float = 0.70, val_frac: float = 0.15):
    """70/15/15 subject-level split, stratified by baseline CDR."""
    test_frac = 1.0 - train_frac - val_frac

    train_ids, temp_ids = train_test_split(
        subject_table,
        train_size=train_frac,
        stratify=subject_table["baseline_cdr"],
        random_state=seed,
    )

    val_relative_frac = val_frac / (val_frac + test_frac)
    val_ids, test_ids = train_test_split(
        temp_ids,
        train_size=val_relative_frac,
        stratify=temp_ids["baseline_cdr"],
        random_state=seed,
    )

    return (
        sorted(train_ids[subject_col].tolist()),
        sorted(val_ids[subject_col].tolist()),
        sorted(test_ids[subject_col].tolist()),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to merged OASIS-3 visit-level CSV")
    parser.add_argument("--output_dir", required=True, help="Where to save the split JSON")
    parser.add_argument("--subject_col", default="subject_id")
    parser.add_argument("--cdr_col", default="cdr_global")
    parser.add_argument("--days_col", default="days_from_entry")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_frac", type=float, default=0.70)
    parser.add_argument("--val_frac", type=float, default=0.15)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    subject_table = build_subject_level_table(df, args.subject_col, args.cdr_col, args.days_col)

    print(f"Total subjects: {len(subject_table)}")
    print("Baseline CDR distribution:")
    print(subject_table["baseline_cdr"].value_counts())

    train_ids, val_ids, test_ids = make_split(
        subject_table, args.subject_col, args.seed, args.train_frac, args.val_frac
    )

    split = {
        "seed": args.seed,
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        "test_frac": round(1 - args.train_frac - args.val_frac, 3),
        "train_subjects": train_ids,
        "val_subjects": val_ids,
        "test_subjects": test_ids,
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "subject_split.json"

    if out_path.exists():
        raise FileExistsError(
            f"{out_path} already exists. This split is meant to be created ONCE and reused. "
            "Delete it manually if you're certain you want to regenerate it (this will make "
            "all prior results non-comparable)."
        )

    with open(out_path, "w") as f:
        json.dump(split, f, indent=2)

    print(f"\nSplit saved to {out_path}")
    print(f"  Train: {len(train_ids)} subjects")
    print(f"  Val:   {len(val_ids)} subjects")
    print(f"  Test:  {len(test_ids)} subjects")


if __name__ == "__main__":
    main()
