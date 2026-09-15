#!/usr/bin/env python3
"""
common.py -- shared LightGBM training/eval machinery for the tabular,
whole-volume-fusion, and ROI-fusion stages. Supports this project's two
outcomes only: progression and development (internal column name
"preclinical", see src/data/build_labels.py).

Conventions:
    - subject split loaded from data/processed/splits/subject_split.json,
      NEVER regenerated
    - multi-seed training (42/202/303/404) for every reported model
    - per-visit predictions saved for every model
      (subject_id, days_from_entry, split, y_true, y_pred, proba)
    - macro precision/recall/F1 alongside headline metrics
    - test set scored once per trained model (post-config, no selection)
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import (
    precision_recall_fscore_support, roc_auc_score,
)

ROOT = Path(os.environ.get("AD_STAGE_NET_ROOT", "/path/to/your/oasis3-working-directory"))
SPLIT_JSON = ROOT / "data" / "processed" / "splits" / "subject_split.json"

CATEGORICAL_COLUMNS = ["dx1", "sex", "handedness", "apoe"]

LEAKAGE_COLUMNS = [
    "converts_to_ad_preclinical", "progresses_next_visit", "days_to_next_visit",
    "days_to_ad_evidence_preclinical", "subject_id",
    "mri_session", "mri_offset_days", "mri3d_session", "mri3d_offset_days",
    "split",
]

TASKS = ["progression", "preclinical"]  # "preclinical" == the development outcome
SEEDS = [42, 202, 303, 404]

BASELINE_PARAMS = dict(
    n_estimators=500, learning_rate=0.05, class_weight="balanced",
    random_state=42,
)


def load_split_map():
    import json
    with open(SPLIT_JSON) as f:
        split = json.load(f)
    return ({**{s: "train" for s in split["train_subjects"]},
             **{s: "val" for s in split["val_subjects"]},
             **{s: "test" for s in split["test_subjects"]}})


def add_split_column(df):
    df = df.copy()
    df["split"] = df["subject_id"].map(load_split_map())
    assert df["split"].notna().all()
    return df


def build_task_frame(df, task):
    """task: 'progression' or 'preclinical' (development, CDR=0 baseline)."""
    if task == "progression":
        out = df.dropna(subset=["progresses_next_visit"]).copy()
        out["target"] = out["progresses_next_visit"].astype(int)
    elif task == "preclinical":
        out = df[(df["cdr_global"] == 0.0) & df["converts_to_ad_preclinical"].notna()].copy()
        out["target"] = out["converts_to_ad_preclinical"].astype(int)
    else:
        raise ValueError(f"unsupported task: {task!r} (this project only implements "
                         f"progression and development/'preclinical')")
    return out


def cast_categoricals(X):
    X = X.copy()
    for col in CATEGORICAL_COLUMNS:
        if col in X.columns:
            X[col] = X[col].astype("category")
    return X


def sens_spec(y_true, y_proba, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    return sens, spec, (tn, fp, fn, tp)


def macro_prf(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro",
        labels=sorted(set(y_true) | set(y_pred)), zero_division=0)
    return float(p), float(r), float(f)


def train_and_evaluate(task, frame, feature_cols, model_dir, preds_dir, tag,
                       seeds=SEEDS, params_override=None):
    """Trains one binary model config across seeds; returns per-seed metrics
    dict. Models saved as {model_dir}/{tag}_s{seed}.txt; per-visit
    predictions as {preds_dir}/{tag}_s{seed}.parquet. Test scored once per
    seed."""
    model_dir, preds_dir = Path(model_dir), Path(preds_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(parents=True, exist_ok=True)

    train = frame[frame["split"] == "train"]
    val = frame[frame["split"] == "val"]
    test = frame[frame["split"] == "test"]
    X_tr, X_va, X_te = (cast_categoricals(train[feature_cols]),
                        cast_categoricals(val[feature_cols]),
                        cast_categoricals(test[feature_cols]))
    y_tr, y_va, y_te = train["target"], val["target"], test["target"]

    results = {}
    for seed in seeds:
        params = dict(BASELINE_PARAMS)
        params["random_state"] = seed
        if params_override:
            params.update(params_override)
        params.update(objective="binary")
        model = lgb.LGBMClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                  callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)])
        model.booster_.save_model(str(model_dir / f"{tag}_s{seed}.txt"))

        m = {"task": task, "tag": tag, "seed": seed,
             "n_train": len(train), "n_val": len(val), "n_test": len(test)}
        parts = []
        for split_name, Xs, ys, sub in [("train", X_tr, y_tr, train),
                                        ("val", X_va, y_va, val),
                                        ("test", X_te, y_te, test)]:
            proba = model.predict_proba(Xs)[:, 1]
            pred = (proba >= 0.5).astype(int)
            y = ys.to_numpy()
            m[f"{split_name}_roc_auc"] = roc_auc_score(y, proba) if len(np.unique(y)) == 2 else float("nan")
            s, sp, cm = sens_spec(y, proba)
            m[f"{split_name}_sensitivity"], m[f"{split_name}_specificity"] = s, sp
            m[f"{split_name}_confusion"] = list(cm)
            mp, mr, mf = macro_prf(y, pred)
            m[f"{split_name}_macro_precision"], m[f"{split_name}_macro_recall"], \
                m[f"{split_name}_macro_f1"] = mp, mr, mf
            out = pd.DataFrame({
                "subject_id": sub["subject_id"].values,
                "days_from_entry": sub["days_from_entry"].values,
                "split": split_name, "y_true": y, "y_pred": pred,
                "proba": proba})
            parts.append(out)
        pd.concat(parts, ignore_index=True).to_parquet(preds_dir / f"{tag}_s{seed}.parquet")
        results[seed] = m

        print(f"[{tag} s{seed}] test AUC {m['test_roc_auc']:.4f} | "
              f"macroF1 {m['test_macro_f1']:.3f} | "
              f"train-val gap {m['train_roc_auc'] - m['val_roc_auc']:.3f}", flush=True)
    return results


def summarize_seeds(results):
    """mean/std across seeds for every numeric metric."""
    first_key = next(iter(results))
    keys = [k for k in results[first_key] if isinstance(results[first_key].get(k), (int, float))]
    out = {}
    for k in keys:
        vals = [r[k] for r in results.values() if r.get(k) is not None and not (isinstance(r.get(k), float) and np.isnan(r[k]))]
        if vals:
            out[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n_seeds": len(vals)}
    return out
