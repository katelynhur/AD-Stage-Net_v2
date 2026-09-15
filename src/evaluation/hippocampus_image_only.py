#!/usr/bin/env python3
"""
hippocampus_image_only.py -- image-only (hippocampus ROI CNN embeddings
alone, LightGBM, no clinical/cognitive features) evaluation for progression
and development. Does not retrain the CNN -- uses the existing hippocampus
embeddings as-is.

Usage:
    AD_STAGE_NET_ROOT=/path/to/your/oasis3-working-directory \
        python hippocampus_image_only.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
from common import ROOT, add_split_column, build_task_frame, train_and_evaluate  # noqa: E402

TASKS = ["progression", "preclinical"]  # preclinical == development
TRAIN_SEEDS = [42, 202, 303, 404]
REPORT_SEEDS = [202, 303, 404]


def summarize(results, seeds, key):
    vals = [results[s][key] for s in seeds]
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
            "n_seeds": len(vals), "per_seed": {s: results[s][key] for s in seeds}}


def main():
    RES = ROOT / "results" / "hippocampus_image_only"
    PRED = ROOT / "predictions" / "hippocampus_image_only"
    MODELS = ROOT / "models" / "hippocampus_image_only"
    RES.mkdir(parents=True, exist_ok=True)
    PRED.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    base = add_split_column(pd.read_parquet(
        ROOT / "data" / "processed" / "oasis3_trajectory.parquet"))
    emb = pd.read_parquet(ROOT / "data" / "processed" / "embeddings" / "roi_single.parquet")
    img_emb_cols = [c for c in emb.columns if c.startswith("roi1a_emb_")]
    flag_col = "has_roi1a_embedding"
    img_cols = img_emb_cols + [flag_col]
    joined = pd.concat([base.reset_index(drop=True), emb.reset_index(drop=True)], axis=1)

    out = {"train_seeds": TRAIN_SEEDS, "report_seeds": REPORT_SEEDS,
           "n_with_real_embedding": int(emb[flag_col].sum()), "embedding_dim": len(img_emb_cols),
           "source": "hippocampus ROI CNN (roi_single.parquet)"}

    for task in TASKS:
        frame = build_task_frame(joined, task)
        test = frame[frame["split"] == "test"]

        print(f"\n=== {task}: hippocampus-only image ({len(img_cols)} cols) ===")
        img_res = train_and_evaluate(task, frame, img_cols, MODELS, PRED,
                                     tag=f"hip_{task}_image_only", seeds=TRAIN_SEEDS)
        img_summary = summarize(img_res, REPORT_SEEDS, "test_roc_auc")

        preds = pd.read_parquet(PRED / f"hip_{task}_image_only_s202.parquet")
        pt = preds[preds.split == "test"].reset_index(drop=True)
        flags = test[flag_col].reset_index(drop=True)
        merged = pd.concat([pt, pd.DataFrame({flag_col: flags.values})], axis=1)
        with_e = merged[merged[flag_col]]
        sub_with = roc_auc_score(with_e.y_true, with_e.proba) if with_e.y_true.nunique() > 1 else None

        out[task] = {
            "n_test": int(len(test)),
            "n_test_with_embedding": int(test[flag_col].sum()),
            "image_only_full_test": img_summary,
            "image_only_with_embedding_subset_seed202": sub_with,
        }
        (RES / "image_only_multiseed.json").write_text(json.dumps(out, indent=2, default=str))
        print(f"[{task}] full={img_summary['mean']:.4f} (n={len(test)})  with_emb_subset={sub_with}")

    print("\nsaved", RES / "image_only_multiseed.json")


if __name__ == "__main__":
    main()
