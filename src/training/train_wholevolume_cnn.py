#!/usr/bin/env python3
"""
train_wholevolume_cnn.py -- whole-volume 3D CNN architecture search on
preprocessed OASIS-3 T1 volumes (src/data/preprocess_mri.py).

Search: 7 architectures, one seed (42), identical protocol each:
    MONAI 3D ResNet (basic [1,1,1,1] / [2,2,2,2] / [3,3,3,3], bottleneck
    [3,4,6,3]), MONAI 3D DenseNet121, torchvision r3d_18 and r2plus1d_18,
    all trained from scratch (single-channel input; Kinetics-400 RGB
    weights don't transfer cleanly). r2plus1d_18 is the winning
    architecture used for this project's reported whole-volume results.

Backbone training objective: 4-class CDR staging (0/0.5/1/2) on MR
sessions matched to train/val-split visits. This is a representation-
learning step, not a reported outcome of this project -- the resulting
penultimate-layer embeddings (extract_wholevolume_embeddings.py) are what
feed the progression/development LightGBM fusion models actually reported
in the paper. See docs/methods.md.

Protocol: AdamW lr 1e-4, weight decay 1e-4, dropout 0.3 head, label
smoothing 0.1, mixup 0.2, grad clip 1.0, sqrt-balanced class weights, bf16
autocast, <=40 epochs, early stop patience 8 on val balanced accuracy.
Train-split subjects only; val-split for early stopping; test-split
volumes never touched here. Augmentation: random L/R flip, intensity
scale 0.9-1.1, shift +/-0.1.

After the sweep: top-2 architectures by val balanced accuracy are
retrained with seeds 42/202/303/404 (multi-seed confirmation).

Usage:
    AD_STAGE_NET_ROOT=/path/to/your/oasis3-working-directory \
        python train_wholevolume_cnn.py sweep
    AD_STAGE_NET_ROOT=/path/to/your/oasis3-working-directory \
        python train_wholevolume_cnn.py multiseed --archs r2plus1d_18
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader, Dataset

ROOT = Path(os.environ.get("AD_STAGE_NET_ROOT", "/path/to/your/oasis3-working-directory"))

HP = dict(lr=1e-4, weight_decay=1e-4, dropout=0.3, label_smoothing=0.1,
          mixup_alpha=0.2, grad_clip=1.0, max_epochs=40, patience=8,
          batch_size=8, seed=42)
SEEDS = [42, 202, 303, 404]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _with_dropout_head(model, fc_attr="fc"):
    fc = getattr(model, fc_attr)
    if isinstance(fc, nn.Linear):
        setattr(model, fc_attr, nn.Sequential(nn.Dropout(HP["dropout"]), fc))
    return model


def build_arch(name: str) -> nn.Module:
    from monai.networks.nets import ResNet as MonaiResNet, DenseNet
    from torchvision import models as tvm

    if name == "resnet10_3d":
        m = MonaiResNet(block="basic", layers=[1, 1, 1, 1],
                        block_inplanes=(32, 64, 128, 256), spatial_dims=3,
                        n_input_channels=1, num_classes=4)
        return _with_dropout_head(m)
    if name == "resnet18_3d":
        m = MonaiResNet(block="basic", layers=[2, 2, 2, 2],
                        block_inplanes=(64, 128, 256, 512), spatial_dims=3,
                        n_input_channels=1, num_classes=4)
        return _with_dropout_head(m)
    if name == "resnet34_3d":
        m = MonaiResNet(block="basic", layers=[3, 3, 3, 3],
                        block_inplanes=(64, 128, 256, 512), spatial_dims=3,
                        n_input_channels=1, num_classes=4)
        return _with_dropout_head(m)
    if name == "resnet50_3d":
        m = MonaiResNet(block="bottleneck", layers=[3, 4, 6, 3],
                        block_inplanes=(64, 128, 256, 512), spatial_dims=3,
                        n_input_channels=1, num_classes=4)
        return _with_dropout_head(m)
    if name == "densenet121_3d":
        return DenseNet(spatial_dims=3, in_channels=1, out_channels=4,
                        dropout_prob=HP["dropout"])
    if name in ("r3d_18", "r2plus1d_18"):
        ctor = tvm.video.r3d_18 if name == "r3d_18" else tvm.video.r2plus1d_18
        m = ctor(weights=None)
        fc_in = m.fc.in_features
        m.fc = nn.Sequential(nn.Dropout(HP["dropout"]), nn.Linear(fc_in, 4))
        m.stem[0] = nn.Conv3d(1, m.stem[0].out_channels, kernel_size=(3, 7, 7),
                              stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        return m
    raise ValueError(name)


class VolDataset(Dataset):
    """Volumes cached in RAM as float16; augmentation applied per batch."""

    def __init__(self, df, vol_dir, is_train):
        self.vols, self.labels, self.sessions = [], [], []
        for _, r in df.iterrows():
            p = Path(vol_dir) / f"{r['session']}.npy"
            if p.exists():
                self.vols.append(p)
                self.labels.append(int(r["label"]))
                self.sessions.append(r["session"])
        self.is_train = is_train

    def __len__(self):
        return len(self.vols)

    def __getitem__(self, i):
        v = np.load(self.vols[i]).astype(np.float32)
        x = torch.from_numpy(v).unsqueeze(0)  # 1,x,y,z
        if self.is_train:
            if random.random() < 0.5:
                x = torch.flip(x, dims=[3])  # L/R flip (x axis)
            x = x * (0.9 + 0.2 * random.random()) + (random.random() - 0.5) * 0.2
        return x, self.labels[i]


def load_labels():
    lab = pd.read_csv(ROOT / "data" / "processed" / "mr_finetune_labels.csv")
    return lab[lab["split"].isin(["train", "val"])].copy()


def mixup(x, y, alpha):
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def run_training(name, seed, train_df, val_df, vol_dir, out_dir, log):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

    tr = VolDataset(train_df, vol_dir, True)
    va = VolDataset(val_df, vol_dir, False)
    log(f"{name} s{seed}: train vols={len(tr)} val vols={len(va)}")
    tr_dl = DataLoader(tr, batch_size=HP["batch_size"], shuffle=True,
                       num_workers=4, pin_memory=True, drop_last=True,
                       persistent_workers=True)
    va_dl = DataLoader(va, batch_size=HP["batch_size"], shuffle=False,
                       num_workers=2, pin_memory=True)

    model = build_arch(name).to(DEVICE)
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    counts = np.bincount(tr.labels, minlength=4).astype(float)
    total = counts.sum()
    w = np.array([np.sqrt(total / c) if c > 0 else 0 for c in counts])
    w = w / w[w > 0].mean()
    crit = nn.CrossEntropyLoss(
        weight=torch.tensor(w, dtype=torch.float32, device=DEVICE),
        label_smoothing=HP["label_smoothing"])
    opt = torch.optim.AdamW(model.parameters(), lr=HP["lr"],
                            weight_decay=HP["weight_decay"])

    best, best_ep, patience_left = -1.0, 0, HP["patience"]
    history, t0 = [], time.time()
    for ep in range(1, HP["max_epochs"] + 1):
        model.train()
        cor = seen = 0
        run_loss = 0.0
        for xb, yb in tr_dl:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            if random.random() < 0.5:
                xm, ya, yb2, lam = mixup(xb, yb, HP["mixup_alpha"])
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"):
                    out = model(xm)
                out = out.float()
                loss = lam * crit(out, ya) + (1 - lam) * crit(out, yb2)
            else:
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"):
                    out = model(xb)
                out = out.float()
                loss = crit(out, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), HP["grad_clip"])
            opt.step()
            run_loss += loss.item() * xb.size(0)
            cor += (out.argmax(1) == yb).sum().item()
            seen += xb.size(0)

        model.eval()
        preds, ys = [], []
        with torch.no_grad():
            for xb, yb in va_dl:
                xb = xb.to(DEVICE, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"):
                    out = model(xb)
                preds.extend(out.argmax(1).cpu().tolist())
                ys.extend(yb.tolist())
        vbacc = balanced_accuracy_score(ys, preds)
        tbacc = cor / max(seen, 1)
        history.append({"epoch": ep, "train_loss": run_loss / max(seen, 1),
                        "train_acc": tbacc, "val_balanced_acc": vbacc})
        log(f"[{name} s{seed} ep{ep:02d}] loss {run_loss/max(seen,1):.3f} "
            f"train_acc {tbacc:.3f} val_bacc {vbacc:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if vbacc > best:
            best, best_ep, patience_left = vbacc, ep, HP["patience"]
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), out_dir / f"{name}_s{seed}.pt")
        else:
            patience_left -= 1
            if patience_left <= 0:
                log(f"[{name} s{seed}] early stop @ep{ep} (best {best:.4f} @ep{best_ep})")
                break

    return {"arch": name, "seed": seed, "params_m": round(n_par, 1),
            "best_val_balanced_acc": best, "best_epoch": best_ep,
            "epochs_run": len(history), "history": history,
            "final_train_acc": history[-1]["train_acc"] if history else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["sweep", "multiseed"])
    ap.add_argument("--archs", nargs="*", default=None)
    ap.add_argument("--out", default="s3_multiseed.json",
                    help="multiseed output filename (allows parallel per-arch runs)")
    args = ap.parse_args()

    vol_dir = ROOT / "data" / "processed" / "mri3d"
    out_dir = ROOT / "models" / "wholevolume_cnn"
    res_dir = ROOT / "results" / "wholevolume_cnn"
    res_dir.mkdir(parents=True, exist_ok=True)
    log_path = res_dir / f"{args.mode}.log"

    def log(msg, **_):
        print(msg, flush=True)
        with open(log_path, "a") as f:
            f.write(str(msg) + "\n")

    lab = load_labels()
    train_df = lab[lab["split"] == "train"]
    val_df = lab[lab["split"] == "val"]

    if args.mode == "sweep":
        archs = args.archs or ["resnet10_3d", "resnet18_3d", "resnet34_3d",
                               "resnet50_3d", "densenet121_3d", "r3d_18",
                               "r2plus1d_18"]
        results = []
        for name in archs:
            r = run_training(name, 42, train_df, val_df, vol_dir, out_dir, log)
            results.append(r)
            (res_dir / "sweep.json").write_text(json.dumps(results, indent=2))
        ranking = sorted(results, key=lambda r: -r["best_val_balanced_acc"])
        log("=== sweep ranking (val balanced accuracy) ===")
        for r in ranking:
            log(f"  {r['arch']:<16} {r['best_val_balanced_acc']:.4f} "
                f"@ep{r['best_epoch']} ({r['params_m']}M params, {r['epochs_run']} epochs)")
    else:
        archs = args.archs
        assert archs, "multiseed requires --archs"
        sweep_path = res_dir / "sweep.json"
        sweep_records = json.loads(sweep_path.read_text()) if sweep_path.exists() else []
        results = [r for r in sweep_records
                   if r["arch"] in archs and r["seed"] == 42
                   and (out_dir / f"{r['arch']}_s42.pt").exists()]
        if results:
            log(f"reusing sweep seed-42 records for: {[r['arch'] for r in results]}")
        for name in archs:
            seeds_to_run = [s for s in SEEDS if s != 42] if any(r["arch"] == name for r in results) else list(SEEDS)
            for seed in seeds_to_run:
                r = run_training(name, seed, train_df, val_df, vol_dir, out_dir, log)
                results.append(r)
                (res_dir / args.out).write_text(json.dumps(results, indent=2))
        for name in archs:
            rs = [r for r in results if r["arch"] == name]
            b = [r["best_val_balanced_acc"] for r in rs]
            log(f"{name}: val bacc {np.mean(b):.4f} ± {np.std(b):.4f} over {len(rs)} seeds")


if __name__ == "__main__":
    main()
