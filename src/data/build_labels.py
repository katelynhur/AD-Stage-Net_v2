#!/usr/bin/env python3
"""
build_labels.py

Adds the two outcome labels used in this project to the merged OASIS-3
visit-level table:

  1. progresses_next_visit / days_to_next_visit  ("progression")
     Does CDR increase at the subject's NEXT recorded visit? Etiology-
     agnostic -- catches any worsening.

  2. converts_to_ad_preclinical / days_to_ad_evidence_preclinical  ("development")
     For visits where the subject's CDR is 0 (cognitively normal) at that
     visit, does dx1 show AD-dementia evidence at any LATER visit within
     a 1,095-day (3-year) window? Internally named "preclinical" in this
     codebase; this is the "development" outcome referenced in the paper
     and README: development of Alzheimer's disease within three years
     among individuals beginning at CDR 0.

Both labels are NaN when there's no valid future visit to check against --
this is real censoring (short follow-up), not missing data, and should be
excluded from training/eval for that label rather than treated as a
negative case. See the printed summary at the end for how many rows this
affects.

dx1 in this data release uses ~50 granular AD-subtype strings (e.g.
"AD dem w/depresss, not contribut", "AD dem Language dysf after") plus a
separate "DAT" (Dementia of Alzheimer's Type) prefix used in some visits.
A plain string-equality check against "AD Dementia" would miss most of
these. This script matches case-insensitively on the "ad dem" / "dat"
prefix instead.

Usage:
    python build_labels.py --input $AD_STAGE_NET_ROOT/data/processed/oasis3_merged.csv \
        --output $AD_STAGE_NET_ROOT/data/processed/oasis3_labeled.csv \
        --development_window_days 1095
"""

import argparse

import pandas as pd


def is_ad_dementia(dx1_series: pd.Series) -> pd.Series:
    """True for any dx1 value indicating AD dementia (any subtype),
    matching both the 'AD dem...' and 'DAT...' (Dementia of Alzheimer's
    Type) naming conventions used across this dataset's visit years."""
    s = dx1_series.astype(str).str.strip().str.lower()
    return s.str.startswith("ad dem") | s.str.startswith("dat")


def build_progression_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each visit, looks at that SAME subject's next recorded visit
    (however far away it is) and flags whether CDR increased.
    NaN if there is no next visit (last visit for that subject / censored).
    """
    df = df.sort_values(["subject_id", "days_from_entry"]).reset_index(drop=True)

    grouped = df.groupby("subject_id")
    df["next_cdr_global"] = grouped["cdr_global"].shift(-1)
    df["next_visit_days_from_entry"] = grouped["days_from_entry"].shift(-1)
    has_next = df["next_cdr_global"].notna()

    df["days_to_next_visit"] = df["next_visit_days_from_entry"] - df["days_from_entry"]
    df["progresses_next_visit"] = pd.NA
    df.loc[has_next, "progresses_next_visit"] = (
        df.loc[has_next, "next_cdr_global"] > df.loc[has_next, "cdr_global"]
    ).astype(int)

    df = df.drop(columns=["next_cdr_global", "next_visit_days_from_entry"])
    return df


def build_development_label(df: pd.DataFrame, window_days: float = 1095.0) -> pd.DataFrame:
    """
    "Development" outcome (internal name: converts_to_ad_preclinical).
    For each visit where cdr_global == 0.0 (cognitively normal baseline),
    checks all LATER visits for that subject within window_days and flags
    whether dx1 shows AD dementia at any of them.

    converts_to_ad_preclinical = 1  -> AD dementia evidence found in the follow-up window
    converts_to_ad_preclinical = 0  -> subject has follow-up in the window but no AD dementia evidence
    converts_to_ad_preclinical = NaN -> either this visit isn't a CDR=0 baseline, or
                                        there's no follow-up visit in the window to check
                                        (censored -- don't treat as a negative)
    """
    df = df.sort_values(["subject_id", "days_from_entry"]).reset_index(drop=True)
    df["is_ad_dx"] = is_ad_dementia(df["dx1"])

    df["converts_to_ad_preclinical"] = pd.NA
    df["days_to_ad_evidence_preclinical"] = pd.NA

    for subject_id, group in df.groupby("subject_id"):
        idx = group.index.tolist()
        for pos, row_idx in enumerate(idx):
            if df.at[row_idx, "cdr_global"] != 0.0:
                continue  # only cognitively-normal (CDR=0) baselines get this label

            baseline_day = df.at[row_idx, "days_from_entry"]
            future_idx = idx[pos + 1:]
            if not future_idx:
                continue  # no follow-up at all -- stays NaN (censored)

            future = df.loc[future_idx]
            if window_days is not None:
                future = future[future["days_from_entry"] - baseline_day <= window_days]

            if future.empty:
                continue  # no follow-up WITHIN the window -- stays NaN (censored)

            if future["is_ad_dx"].any():
                df.at[row_idx, "converts_to_ad_preclinical"] = 1
                first_ad_day = future.loc[future["is_ad_dx"], "days_from_entry"].min()
                df.at[row_idx, "days_to_ad_evidence_preclinical"] = first_ad_day - baseline_day
            else:
                df.at[row_idx, "converts_to_ad_preclinical"] = 0

    df = df.drop(columns=["is_ad_dx"])
    return df


def print_summary(df: pd.DataFrame) -> None:
    n_subjects = df["subject_id"].nunique()
    single_visit_subjects = (df.groupby("subject_id").size() == 1).sum()

    print(f"\nTotal visits: {len(df)}  |  Total subjects: {n_subjects}")
    print(f"Subjects with only 1 visit (no progression/development label possible): {single_visit_subjects}")

    print("\n--- progresses_next_visit (progression) ---")
    print(df["progresses_next_visit"].value_counts(dropna=False))

    print("\n--- converts_to_ad_preclinical (development, only defined for CDR=0 baseline visits) ---")
    n_baseline = (df["cdr_global"] == 0.0).sum()
    print(f"CDR=0 baseline visits: {n_baseline}")
    print(df["converts_to_ad_preclinical"].value_counts(dropna=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--development_window_days", type=float, default=1095.0,
                         help="Only count AD-dementia evidence within this many days "
                              "(default 1,095 = 3 years) of the CDR=0 baseline visit.")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} visits, {df['subject_id'].nunique()} subjects")

    df = build_progression_label(df)
    df = build_development_label(df, window_days=args.development_window_days)

    df.to_csv(args.output, index=False)
    print(f"\nSaved labeled table to: {args.output}")

    print_summary(df)


if __name__ == "__main__":
    main()
