#!/usr/bin/env python3
"""
cognitive_features.py -- integrate the UDS cognitive battery (Form C1) into
the canonical clinical table, matched causally (no post-baseline
information) for the progression/development prediction tasks.

Coverage decisions (documented, not a blind merge):
    - Kept the 29 subtest columns with >=50% row coverage in the C1 file
      (memory: SRT total/free + learning trials, Logical Memory; executive:
      Trails A/B, digit symbol, letter-number + switching variants; language:
      category fluency ANIMALS/VEG, Boston Naming; attention/working memory:
      digits forward/backward, mental control; psychometric composites
      PSY019/PSY021).
    - DROPPED for sparsity: the MoCA family, Craft verbal, MINT naming, and
      udsver/udsbent/udspatial batteries (<50% coverage) -- these are the
      UDS3-era replacements administered only in later protocol years.

Matching rule ("split by task type"):
    - cog_* block     : CAUSAL nearest-prior assessment within 365 days
                        (no post-baseline information). Used by both
                        progression and development (future prediction).
    - cog730_* block  : causal nearest-prior within 730 days -- sensitivity
                        variant only.
    Audit columns cog_*_offset_days record the signed offset for every match.

Usage:
    AD_STAGE_NET_ROOT=/path/to/your/oasis3-working-directory \
        python cognitive_features.py
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("AD_STAGE_NET_ROOT", "/path/to/your/oasis3-working-directory"))
COG_CSV = (ROOT / "data" / "raw" / "oasis3" / "OASIS3_data_files" / "scans" /
           "pychometrics-Form_C1__Cognitive_Assessments" / "resources" / "csv" / "files" /
           "OASIS3_UDSc1_cognitive_assessments.csv")
LABELED_CSV = ROOT / "data" / "processed" / "oasis3_labeled.csv"
OUT_DIR = ROOT / "data" / "processed"

# 29 subtests with >=50% coverage (see module docstring)
KEEP = ["ANIMALS", "tma", "tmb", "srtfree", "srttotal", "asscmem", "srt1f",
        "srt1c", "srt2f", "srt2c", "srt3f", "srt3c", "mentcont", "PSY019",
        "digsym", "VEG", "PSY021", "lettnum", "simon", "switch", "digfor",
        "digback", "bnt", "switchCV", "switchmixed", "simonnumber",
        "LOGIMEM", "MEMUNITS"]
RENAME = {c: f"cog_{c.lower()}" for c in KEEP}


def match(visits, cog_by, window):
    """Causal nearest-prior match within `window` days. Returns (values, offsets)."""
    vals = pd.DataFrame(index=visits.index, columns=list(RENAME.values()), dtype="float32")
    offs = pd.Series(np.nan, index=visits.index)
    for idx, row in visits.iterrows():
        g = cog_by.get(row["subject_id"])
        if g is None:
            continue
        cand = g[g["days"] <= row["days_from_entry"]]
        if cand.empty:
            continue
        i = (cand["days"] - row["days_from_entry"]).abs().idxmin()
        off = cand.at[i, "days"] - row["days_from_entry"]
        if abs(off) <= window:
            vals.loc[idx] = cand.loc[i, list(RENAME.values())].to_numpy(dtype="float32")
            offs.at[idx] = off
    return vals, offs


def main():
    cog = pd.read_csv(COG_CSV, low_memory=False)
    cog["days"] = pd.to_numeric(cog["days_to_visit"], errors="coerce")
    cog = cog.dropna(subset=["OASISID", "days"])
    print(f"C1 battery: {len(cog)} assessments, {cog['OASISID'].nunique()} subjects")

    cov = cog[KEEP].notna().mean()
    assert (cov >= 0.5).all(), f"coverage dropped below 50%: {cov[cov < 0.5].to_dict()}"
    cog = cog.rename(columns=RENAME)[["OASISID", "days"] + list(RENAME.values())]
    cog_by = {s: g.sort_values("days") for s, g in cog.groupby("OASISID")}

    base = pd.read_csv(LABELED_CSV)
    print(f"canonical table: {len(base)} visits")

    blocks = {}
    for block, window in [("cog", 365), ("cog730", 730)]:
        vals, offs = match(base, cog_by, window)
        vals.columns = [f"{block}_{c.replace('cog_', '', 1)}" for c in vals.columns]
        blocks[block] = (vals, offs, window)
        print(f"  {block:7s} (causal, <= {window}d): {offs.notna().sum()} "
              f"visits matched ({offs.notna().mean():.1%})")

    out = base.copy()
    for block, (vals, offs, window) in blocks.items():
        out = pd.concat([out, vals, offs.rename(f"{block}_offset_days")], axis=1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "oasis3_cognition.parquet"
    out.to_parquet(out_path)
    print(f"saved {out_path}  ({out.shape[0]} rows x {out.shape[1]} cols)")

    rep = {"kept_subtests": KEEP}
    (OUT_DIR / "cognition_coverage.json").write_text(json.dumps(rep, indent=2))
    print("coverage decisions written to cognition_coverage.json")


if __name__ == "__main__":
    main()
