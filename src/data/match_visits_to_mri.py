#!/usr/bin/env python3
"""
match_visits_to_mri.py -- causal visit-to-MRI-session matching, and the
CDR-based labels used to pretrain the whole-volume / ROI 3D CNN backbones.

Two outputs:
  1. data/processed/visit_scan_matches.csv
     For every clinical visit, the causally-matched MR session (nearest scan
     with scan_day <= visit_day, within --tolerance_days), with a signed
     offset for auditing. Causal everywhere: no scan may post-date its
     visit. Default 730-day lookback.

  2. data/processed/mr_finetune_labels.csv
     Labels for MR sessions of train/val-split subjects: the CDR of the
     session's NEAREST matched visit (sessions matched to several visits
     are deduplicated to one label). CDR {0, 0.5, 1, 2} map to 4 classes;
     CDR 3 visits are excluded (too rare to train a class on). These labels
     are used ONLY to pretrain the whole-volume and ROI CNN backbones
     (Stage classification is a representation-learning step here, not a
     reported outcome -- see docs/methods.md); the resulting penultimate-
     layer embeddings are what feed the progression/development models.

Usage:
    AD_STAGE_NET_ROOT=/path/to/your/oasis3-working-directory \
        python match_visits_to_mri.py --tolerance_days 730
"""

import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ.get("AD_STAGE_NET_ROOT", "/path/to/your/oasis3-working-directory"))
CDR_TO_CLASS = {0.0: 0, 0.5: 1, 1.0: 2, 2.0: 3}


def build_mr_inventory(imaging_root: Path) -> pd.DataFrame:
    """One row per MR session that has at least one T1w file.
    Session dirs are named <subject>_MR_d<day>."""
    rows = []
    for sess_dir in sorted((imaging_root / "MR").iterdir()):
        m = re.match(r"^(OAS3\d+)_MR_d(\d+)$", sess_dir.name)
        if not m:
            continue
        has_t1w = any(sess_dir.rglob("*_T1w.nii.gz"))
        if has_t1w:
            rows.append({"subject_id": m.group(1), "scan_day": float(m.group(2)),
                        "session": sess_dir.name})
    return pd.DataFrame(rows)


def match_visits_to_scans(visits: pd.DataFrame, scans: pd.DataFrame,
                          tolerance_days: float) -> pd.DataFrame:
    """Returns rows: visit_idx, scan_day, session, offset_days for every
    visit that has a nearest-prior scan within tolerance."""
    out = []
    scans_by_subj = {s: g.sort_values("scan_day") for s, g in scans.groupby("subject_id")}
    for idx, row in visits.iterrows():
        g = scans_by_subj.get(row["subject_id"])
        if g is None:
            continue
        prior = g[g["scan_day"] <= row["days_from_entry"]]
        if prior.empty:
            continue
        nearest = prior.iloc[(prior["scan_day"] - row["days_from_entry"]).abs().argmin()]
        off = nearest["scan_day"] - row["days_from_entry"]
        if abs(off) <= tolerance_days:
            out.append({"visit_idx": idx, "scan_day": nearest["scan_day"],
                        "session": nearest["session"], "offset_days": off})
    return pd.DataFrame(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged", default=str(ROOT / "data" / "processed" / "oasis3_labeled.csv"))
    parser.add_argument("--imaging_root", default=str(ROOT / "data" / "raw" / "oasis3" / "imaging"))
    parser.add_argument("--split_json",
                        default=str(ROOT / "data" / "processed" / "splits" / "subject_split.json"))
    parser.add_argument("--out_dir", default=str(ROOT / "data" / "processed"))
    parser.add_argument("--tolerance_days", type=float, default=730.0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.merged).sort_values(["subject_id", "days_from_entry"]).reset_index(drop=True)
    with open(args.split_json) as f:
        split = json.load(f)
    split_map = ({**{s: "train" for s in split["train_subjects"]},
                  **{s: "val" for s in split["val_subjects"]},
                  **{s: "test" for s in split["test_subjects"]}})
    df["split"] = df["subject_id"].map(split_map)
    assert df["split"].notna().all()

    print("Building MR scan inventory...")
    mr = build_mr_inventory(Path(args.imaging_root))
    print(f"  MR sessions w/ T1w: {len(mr)}")

    print(f"Causal matching (<= {args.tolerance_days:.0f}d lookback)...")
    mri_m = match_visits_to_scans(df, mr, args.tolerance_days)

    visits = df.reset_index()[["subject_id", "days_from_entry", "cdr_global", "split"]]
    visits.index.name = "visit_idx"
    mri_tbl = mri_m.merge(visits, on="visit_idx", how="left") if not mri_m.empty else mri_m

    matches = mri_tbl.rename(columns={"scan_day": "mri_scan_day", "session": "mri_session",
                                      "offset_days": "mri_offset_days"})

    match_path = out_dir / "visit_scan_matches.csv"
    matches.to_csv(match_path, index=False)
    n_mri = matches["mri_session"].notna().sum()
    print(f"Matched: MRI {n_mri}/{len(df)} visits")
    print(f"Saved match table: {match_path}")

    # ---------------- fine-tune labels (MR only, nearest visit, CDR 0-2) -----
    lab = (mri_tbl.dropna(subset=["mri_session"])
           .sort_values("mri_offset_days", key=lambda s: s.abs())  # nearest visit first
           .drop_duplicates(subset="mri_session", keep="first"))
    lab = lab[lab["cdr_global"].isin(CDR_TO_CLASS)]
    lab = lab.assign(label=lab["cdr_global"].map(CDR_TO_CLASS).astype(int))

    label_rows = []
    for _, r in lab.iterrows():
        label_rows.append({"session": r["mri_session"], "subject_id": r["subject_id"],
                           "split": r["split"], "cdr_global": r["cdr_global"],
                           "label": r["label"]})
    labels = pd.DataFrame(label_rows)
    label_path = out_dir / "mr_finetune_labels.csv"
    labels.to_csv(label_path, index=False)
    print(f"\nBackbone-pretraining label set: {len(labels)} MR sessions")
    print(labels.groupby(["split", "label"]).size().unstack(fill_value=0))
    print(f"Saved: {label_path}")


if __name__ == "__main__":
    main()
