#!/usr/bin/env python3
"""
verify_wholevolume_baseline.py -- reproduces the whole-volume CNN's reported
validation balanced accuracy from its saved checkpoints (forward passes
only, no training), as a sanity gate before training the ROI models on top
of it. Exits nonzero if any checkpoint fails to reproduce within tolerance.

Usage:
    AD_STAGE_NET_ROOT=/path/to/your/oasis3-working-directory \
        python verify_wholevolume_baseline.py
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader

ROOT = Path(os.environ.get("AD_STAGE_NET_ROOT", "/path/to/your/oasis3-working-directory"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_wholevolume_cnn import DEVICE, VolDataset, build_arch  # noqa: E402

# Fill in with your own sweep's reported (arch, seed) -> expected val bacc.
EXPECT = {("r2plus1d_18", 42): None}


@torch.no_grad()
def main():
    lab = pd.read_csv(ROOT / "data" / "processed" / "mr_finetune_labels.csv")
    val_df = lab[lab["split"] == "val"]
    test_df = lab[lab["split"] == "test"]
    vol_dir = ROOT / "data" / "processed" / "mri3d"
    va = VolDataset(val_df, vol_dir, False)
    te = VolDataset(test_df, vol_dir, False)
    va_dl = DataLoader(va, batch_size=8, shuffle=False, num_workers=4)
    te_dl = DataLoader(te, batch_size=8, shuffle=False, num_workers=4)
    print(f"val sessions={len(va)}  test sessions={len(te)}")

    ok = True
    results = {}
    for (arch, seed), expected in EXPECT.items():
        model = build_arch(arch).to(DEVICE).eval()
        sd = torch.load(ROOT / "models" / "wholevolume_cnn" / f"{arch}_s{seed}.pt",
                        map_location="cpu", weights_only=True)
        model.load_state_dict(sd, strict=True)
        preds, ys = [], []
        for xb, yb in va_dl:
            xb = xb.to(DEVICE)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"):
                out = model(xb)
            preds.extend(out.argmax(1).cpu().tolist())
            ys.extend(yb.tolist())
        vacc = balanced_accuracy_score(ys, preds)
        tp, ty = [], []
        for xb, yb in te_dl:
            xb = xb.to(DEVICE)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"):
                out = model(xb)
            tp.extend(out.argmax(1).cpu().tolist())
            ty.extend(yb.tolist())
        tacc = balanced_accuracy_score(ty, tp)
        match = expected is None or abs(vacc - expected) < 0.002
        ok &= match
        results[f"{arch}_s{seed}"] = {"val": round(vacc, 4), "expected_val": expected,
                                      "test": round(tacc, 4), "reproduced": bool(match)}
        print(f"{arch} s{seed}: val {vacc:.4f} (expected {expected}) "
              f"{'REPRODUCED' if match else '*** MISMATCH ***'} | test {tacc:.4f}")
        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    out = ROOT / "results" / "wholevolume_cnn" / "verification.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"saved {out}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
