#!/usr/bin/env python3
"""
train_clinical_cognitive_baseline.py -- the clinical/cognitive LightGBM
baseline (longitudinal clinical + cognitive-battery + trajectory features),
for progression and development. This model's per-task AUCs are the
"tabular-only" baseline every whole-volume/ROI imaging comparison in this
project is measured against.

Usage:
    AD_STAGE_NET_ROOT=/path/to/your/oasis3-working-directory \
        python train_clinical_cognitive_baseline.py
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
from common import (  # noqa: E402
    LEAKAGE_COLUMNS, ROOT, add_split_column, build_task_frame,
    summarize_seeds, train_and_evaluate,
)

TASKS = ["progression", "preclinical"]  # preclinical == development


def feature_columns(df):
    cols = [c for c in df.columns
            if c not in set(LEAKAGE_COLUMNS) | {"target", "cdr_global"}
            and not c.startswith(("mri_emb_", "mri3d_emb_", "cog730_"))
            and c not in ("has_mri_embedding", "has_mri3d_embedding")]
    base = [c for c in cols if not c.startswith(("cog_", "traj_"))]
    cog = [c for c in cols if c.startswith("cog_") and not c.endswith("_offset_days")]
    traj = [c for c in cols if c.startswith("traj_")]
    return base, cog, traj


def main():
    df = add_split_column(pd.read_parquet(
        ROOT / "data" / "processed" / "oasis3_trajectory.parquet"))

    results_dir = ROOT / "results" / "clinical_cognitive_baseline"
    results_dir.mkdir(parents=True, exist_ok=True)
    model_dir = ROOT / "models" / "clinical_cognitive_baseline"
    preds_dir = ROOT / "predictions" / "clinical_cognitive_baseline"

    all_results = {}
    for task in TASKS:
        base, cog, traj = feature_columns(df)
        cols = base + cog + traj
        frame = build_task_frame(df, task)
        print(f"\n=== {task}: {len(base)} base + {len(cog)} cog + {len(traj)} trajectory ===")
        res = train_and_evaluate(task, frame, cols, model_dir, preds_dir,
                                 tag=f"clinical_cognitive_{task}")
        all_results[task] = {"per_seed": res, "summary": summarize_seeds(res),
                             "n_features": len(cols)}

    with open(results_dir / "metrics.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n=== Clinical/cognitive baseline (test, mean over seeds) ===")
    for task in TASKS:
        s = all_results[task]["summary"]
        got, sd = s["test_roc_auc"]["mean"], s["test_roc_auc"]["std"]
        print(f"{task:<13} AUC {got:.4f} ± {sd:.4f} | macroF1 {s['test_macro_f1']['mean']:.3f}")
    print(f"\nmetrics -> {results_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
