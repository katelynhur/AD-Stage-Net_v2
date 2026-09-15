#!/usr/bin/env python3
"""
hippocampus_fused_ablation.py -- hippocampal ROI approach: full fusion
(clinical/cognitive/trajectory + Stage-1a single-region hippocampus ROI CNN
embeddings) for progression and development, plus the inference-time
MRI-masking ablation (same trained fused model, image columns zeroed at
predict time -- the model itself is not retrained).

Also reports the "with-embedding subset" breakdown at seed 202: many test
rows have no real hippocampus embedding (no scan close enough to that
visit), so the headline full-model / MRI-masked AUCs are computed on the
SAME scan-eligible subset for both the full and masked model, matching
this project's reported test n.

Usage:
    AD_STAGE_NET_ROOT=/path/to/your/oasis3-working-directory \
        python hippocampus_fused_ablation.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
from common import (  # noqa: E402
    ROOT, add_split_column, build_task_frame, cast_categoricals,
    summarize_seeds, train_and_evaluate,
)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from wholevolume_fusion_ablation import blocks as tabular_blocks  # noqa: E402

TASKS = ["progression", "preclinical"]  # preclinical == development
TRAIN_SEEDS = [42, 202, 303, 404]
REPORT_SEEDS = [202, 303, 404]


def summarize(results, seeds, key):
    vals = [results[s][key] for s in seeds]
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
            "n_seeds": len(vals), "per_seed": {s: results[s][key] for s in seeds}}


def main():
    RES = ROOT / "results" / "hippocampus_fused"
    PRED = ROOT / "predictions" / "hippocampus_fused"
    MODELS = ROOT / "models" / "hippocampus_fused"
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
           "source": "hippocampus ROI CNN (roi_single.parquet)",
           "n_with_real_embedding": int(emb[flag_col].sum())}

    for task in TASKS:
        tab_cols, _ = tabular_blocks(base, task)
        frame = build_task_frame(joined, task)
        test = frame[frame["split"] == "test"]
        y_test = test["target"].to_numpy()

        print(f"\n=== {task}: hippocampus fused ({len(tab_cols) + len(img_cols)} cols) ===")
        fused_res = train_and_evaluate(task, frame, tab_cols + img_cols, MODELS, PRED,
                                       tag=f"hipfused_{task}", seeds=TRAIN_SEEDS)

        masked_rows = []
        with_flags = test[flag_col].reset_index(drop=True)
        for seed in REPORT_SEEDS:
            b = lgb.Booster(model_file=str(MODELS / f"hipfused_{task}_s{seed}.txt"))
            X = cast_categoricals(test[tab_cols + img_cols]).copy()
            X[img_emb_cols] = np.nan
            X[flag_col] = 0
            proba = b.predict(X)
            for i, (yt, p) in enumerate(zip(y_test, proba)):
                masked_rows.append({"seed": seed, "y_true": int(yt), "proba": float(p),
                                    "has_roi1a_embedding": bool(with_flags.iloc[i])})
        masked_df = pd.DataFrame(masked_rows)
        masked_df.to_parquet(PRED / f"hipfused_{task}_masked_test.parquet")

        metric_key = "test_roc_auc"
        fused_summary = summarize(fused_res, REPORT_SEEDS, metric_key)

        # ---- with/without-embedding subset breakdown, seed 202 ----
        fused_preds = pd.read_parquet(PRED / f"hipfused_{task}_s202.parquet")
        fpt = fused_preds[fused_preds.split == "test"].reset_index(drop=True)
        fmerged = pd.concat([fpt, pd.DataFrame({flag_col: with_flags.values})], axis=1)
        f_with = fmerged[fmerged[flag_col]]

        m202 = masked_df[masked_df.seed == 202].reset_index(drop=True)
        m_with = m202[m202[flag_col]]

        fused_with = roc_auc_score(f_with.y_true, f_with.proba) if f_with.y_true.nunique() > 1 else None
        masked_with = roc_auc_score(m_with.y_true, m_with.proba) if m_with.y_true.nunique() > 1 else None

        out[task] = {
            "n_test": int(len(test)),
            "n_test_with_embedding": int(test[flag_col].sum()),
            "fused_full_test": fused_summary,
            "with_embedding_subset_seed202": {
                "n": int(test[flag_col].sum()),
                "full_model_auc": fused_with,
                "mri_masked_auc": masked_with,
                "mri_contribution_delta": (fused_with - masked_with)
                    if (fused_with is not None and masked_with is not None) else None,
            },
        }
        (RES / "fusion_multiseed.json").write_text(json.dumps(out, indent=2, default=str))
        print(f"[{task}] full={fused_with}  masked={masked_with}  (n={int(test[flag_col].sum())})")

    print("\nsaved", RES / "fusion_multiseed.json")


if __name__ == "__main__":
    main()
