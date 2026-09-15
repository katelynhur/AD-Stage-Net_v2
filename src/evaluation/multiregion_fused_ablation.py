#!/usr/bin/env python3
"""
multiregion_fused_ablation.py -- multi-region ROI approach: image-only and
full fusion (clinical/cognitive/trajectory + Stage-1b multi-region ROI CNN
embeddings: hippocampus, ventricles, amygdala, entorhinal) for progression
and development, plus the inference-time MRI-masking ablation (same
trained fused model, image columns zeroed at predict time). Does not
retrain the CNN -- uses the existing multi-region embeddings as-is.

As with the hippocampal approach, the headline full-model / MRI-masked AUCs
are computed on the scan-eligible subset (rows with a real multi-region
embedding) at seed 202, using the already-trained fused model's saved
per-visit predictions -- the same "with-embedding subset" methodology
applied in hippocampus_fused_ablation.py, so the two ROI approaches are
evaluated on directly comparable, same-sized test populations.

Usage:
    AD_STAGE_NET_ROOT=/path/to/your/oasis3-working-directory \
        python multiregion_fused_ablation.py
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
    RES = ROOT / "results" / "multiregion_fused"
    PRED = ROOT / "predictions" / "multiregion_fused"
    MODELS = ROOT / "models" / "multiregion_fused"
    RES.mkdir(parents=True, exist_ok=True)
    PRED.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    base = add_split_column(pd.read_parquet(
        ROOT / "data" / "processed" / "oasis3_trajectory.parquet"))
    emb = pd.read_parquet(ROOT / "data" / "processed" / "embeddings" / "roi_multiregion.parquet")
    img_emb_cols = [c for c in emb.columns if "_emb_" in c]
    flag_col = "has_roi1b_embedding"
    img_cols = img_emb_cols + [flag_col]
    joined = pd.concat([base.reset_index(drop=True), emb.reset_index(drop=True)], axis=1)

    out = {"train_seeds": TRAIN_SEEDS, "report_seeds": REPORT_SEEDS,
           "source": "multi-region ROI CNN: hippocampus, ventricles, amygdala, entorhinal "
                     "(roi_multiregion.parquet)",
           "n_with_real_embedding": int(emb[flag_col].sum())}

    for task in TASKS:
        tab_cols, _ = tabular_blocks(base, task)
        frame = build_task_frame(joined, task)
        test = frame[frame["split"] == "test"]
        y_test = test["target"].to_numpy()

        print(f"\n=== {task}: image-only ({len(img_cols)} cols) ===")
        img_res = train_and_evaluate(task, frame, img_cols, MODELS, PRED,
                                     tag=f"multiregion_{task}_image_only", seeds=TRAIN_SEEDS)

        print(f"\n=== {task}: fused ({len(tab_cols) + len(img_cols)} cols) ===")
        fused_res = train_and_evaluate(task, frame, tab_cols + img_cols, MODELS, PRED,
                                       tag=f"multiregion_{task}_fused", seeds=TRAIN_SEEDS)

        # ---- masked ablation: same fused model per seed, images zeroed ----
        masked_rows = []
        with_flags = test[flag_col].reset_index(drop=True)
        for seed in REPORT_SEEDS:
            b = lgb.Booster(model_file=str(MODELS / f"multiregion_{task}_fused_s{seed}.txt"))
            X = cast_categoricals(test[tab_cols + img_cols]).copy()
            X[img_emb_cols] = np.nan
            X[flag_col] = 0
            proba = b.predict(X)
            for i, (yt, p) in enumerate(zip(y_test, proba)):
                masked_rows.append({"seed": seed, "y_true": int(yt), "proba": float(p),
                                    flag_col: bool(with_flags.iloc[i])})
        masked_df = pd.DataFrame(masked_rows)
        masked_df.to_parquet(PRED / f"multiregion_{task}_masked_test.parquet")

        img_summary = summarize(img_res, REPORT_SEEDS, "test_roc_auc")
        fused_summary = summarize(fused_res, REPORT_SEEDS, "test_roc_auc")

        # ---- with/without-embedding subset breakdown, seed 202 (same
        # methodology as hippocampus_fused_ablation.py, applied here so
        # both ROI approaches report on the same scan-eligible subset) ----
        fused_preds = pd.read_parquet(PRED / f"multiregion_{task}_fused_s202.parquet")
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
            "image_only_full_test": img_summary,
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
        print(f"[{task}] image_only={img_summary['mean']:.4f}  full={fused_with}  "
              f"masked={masked_with}  (n={int(test[flag_col].sum())})")

    print("\nsaved", RES / "fusion_multiseed.json")


if __name__ == "__main__":
    main()
