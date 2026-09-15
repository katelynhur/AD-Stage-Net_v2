#!/usr/bin/env python3
"""
wholevolume_fusion_ablation.py -- whole-volume MRI approach: trains
tabular-only / image-only / fused LightGBM variants for progression and
development on the clinical/cognitive/trajectory table + whole-volume
mri3d_emb_* embeddings, then re-scores the trained fused model with the
MRI embedding block masked out (inference-time ablation; the model is not
retrained). Reports the full-model AUC and the MRI-masked AUC on the same
test set for each task.

Usage:
    AD_STAGE_NET_ROOT=/path/to/your/oasis3-working-directory \
        python wholevolume_fusion_ablation.py
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
    LEAKAGE_COLUMNS, ROOT, add_split_column, build_task_frame,
    cast_categoricals, sens_spec, summarize_seeds, train_and_evaluate,
)

TASKS = ["progression", "preclinical"]  # preclinical == development


def blocks(df, task):
    """tabular vs. whole-volume-MRI feature blocks (no PET -- out of scope)."""
    drop = set(LEAKAGE_COLUMNS) | {
        "target", "mri3d_session", "mri3d_offset_days",
        "cog_offset_days", "cog730_offset_days",
    }
    cols = [c for c in df.columns if c not in drop and not c.startswith("mri3d_emb_")]
    base = [c for c in cols if not c.startswith(("cog_", "cog730_", "traj_"))]
    cog = [c for c in cols if c.startswith("cog_") and not c.startswith("cog730_")
           and not c.endswith("_offset_days")]
    traj = [c for c in cols if c.startswith("traj_")]
    tab = base + cog + traj
    img = [c for c in df.columns if c.startswith("mri3d_emb_")] + ["has_mri3d_embedding"]
    return tab, img


def apply_mask(X, img_emb_cols, flag_col):
    X = X.copy()
    X[img_emb_cols] = np.nan
    if flag_col in X.columns:
        X[flag_col] = 0
    return X


def main():
    df = add_split_column(pd.read_parquet(
        ROOT / "data" / "processed" / "oasis3_wholevolume_embeddings.parquet"))

    res_dir = ROOT / "results" / "wholevolume_fusion"
    res_dir.mkdir(parents=True, exist_ok=True)
    model_dir = ROOT / "models" / "wholevolume_fusion"
    preds_dir = ROOT / "predictions" / "wholevolume_fusion"

    grid_out, ablation_rows = {}, []
    for task in TASKS:
        tab, img = blocks(df, task)
        frame = build_task_frame(df, task)
        test = frame[frame["split"] == "test"]
        y_test = test["target"].to_numpy()
        img_emb_cols = [c for c in img if c.startswith("mri3d_emb_")]

        for variant, cols in [("tabular", tab), ("image", img), ("fused", tab + img)]:
            print(f"\n=== {task} / {variant} ({len(cols)} features) ===")
            res = train_and_evaluate(task, frame, cols, model_dir, preds_dir,
                                     tag=f"wholevol_{task}_{variant}")
            grid_out[f"{task}_{variant}"] = {"per_seed": res, "summary": summarize_seeds(res),
                                             "n_features": len(cols)}

        # ---- masked ablation on the fused model, seed 42 (paper's reported seed) ----
        seed = 42
        b = lgb.Booster(model_file=str(model_dir / f"wholevol_{task}_fused_s{seed}.txt"))
        X_full = cast_categoricals(test[tab + img])
        p_full = b.predict(X_full)
        p_masked = b.predict(apply_mask(X_full, img_emb_cols, "has_mri3d_embedding"))

        auc_full = roc_auc_score(y_test, p_full)
        auc_masked = roc_auc_score(y_test, p_masked)
        sens_f, spec_f, _ = sens_spec(y_test, p_full)
        sens_m, spec_m, _ = sens_spec(y_test, p_masked)
        ablation_rows.append({
            "task": task, "n_test": int(len(test)),
            "full_auc": round(float(auc_full), 4), "full_sensitivity": round(sens_f, 4),
            "full_specificity": round(spec_f, 4),
            "mri_masked_auc": round(float(auc_masked), 4), "masked_sensitivity": round(sens_m, 4),
            "masked_specificity": round(spec_m, 4),
            "mri_contribution_delta": round(float(auc_masked - auc_full), 4),
        })
        print(f"[{task}] full AUC {auc_full:.4f} | MRI-masked AUC {auc_masked:.4f} | "
              f"delta {auc_masked - auc_full:+.4f}")

    (res_dir / "grid_metrics.json").write_text(json.dumps(grid_out, indent=2))
    (res_dir / "mri_ablation.json").write_text(json.dumps(ablation_rows, indent=2))
    print(f"\nsaved {res_dir}/grid_metrics.json and mri_ablation.json")


if __name__ == "__main__":
    main()
