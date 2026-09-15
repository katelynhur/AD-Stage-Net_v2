#!/usr/bin/env python3
"""
trajectory_features.py -- change-over-time features, added on top of the
cognitive feature table. This becomes the tabular-only feature set that
every whole-volume/ROI imaging comparison in this project is measured
against.

For every visit with a prior visit for the same subject:
    traj_days_since_prior_visit
    traj_delta_mmse, traj_delta_cdr_sb, traj_delta_hippo_total
    traj_delta_cog_<subtest>   (29 deltas from the causal cog block)
    traj_has_prior             (binary flag; all deltas NaN on first visits)

Leakage note: every delta uses only CURRENT-visit and PRIOR-visit values --
both legitimately available at prediction time.

Usage:
    AD_STAGE_NET_ROOT=/path/to/your/oasis3-working-directory \
        python trajectory_features.py
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("AD_STAGE_NET_ROOT", "/path/to/your/oasis3-working-directory"))
DATA_DIR = ROOT / "data" / "processed"


def main():
    src = DATA_DIR / "oasis3_cognition.parquet"
    df = pd.read_parquet(src).sort_values(["subject_id", "days_from_entry"]).reset_index(drop=True)

    cog_cols = sorted(c for c in df.columns
                       if c.startswith("cog_") and not c.startswith("cog730_")
                       and not c.endswith("_offset_days"))
    base_cols = ["mmse", "cdr_sum_of_boxes", "hippocampal_volume_total"]
    delta_specs = ([(c, f"traj_delta_{c}") for c in base_cols]
                   + [(c, f"traj_delta_cog_{c[len('cog_'):]}") for c in cog_cols])

    new_cols = {name: np.full(len(df), np.nan, dtype="float32")
                for _, name in delta_specs}
    new_cols["traj_days_since_prior_visit"] = np.full(len(df), np.nan, dtype="float32")
    has_prior = np.zeros(len(df), dtype=bool)

    for _, g in df.groupby("subject_id", sort=False):
        idx = g.index.tolist()
        prev = None
        for i in idx:
            if prev is not None:
                has_prior[i] = True
                new_cols["traj_days_since_prior_visit"][i] = (
                    df.at[i, "days_from_entry"] - df.at[prev, "days_from_entry"])
                for src_col, name in delta_specs:
                    cur, pri = df.at[i, src_col], df.at[prev, src_col]
                    if pd.notna(cur) and pd.notna(pri):
                        new_cols[name][i] = float(cur) - float(pri)
            prev = i

    traj = pd.DataFrame(new_cols)
    traj["traj_has_prior"] = has_prior
    out = pd.concat([df, traj], axis=1)

    dst = DATA_DIR / "oasis3_trajectory.parquet"
    out.to_parquet(dst)
    n = int(has_prior.sum())
    print(f"saved {dst}  ({out.shape[0]} rows x {out.shape[1]} cols)")
    print(f"visits with a prior visit: {n}/{len(out)} ({n/len(out):.1%})")


if __name__ == "__main__":
    main()
