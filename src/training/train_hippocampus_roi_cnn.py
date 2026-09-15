#!/usr/bin/env python3
"""
train_hippocampus_roi_cnn.py -- single-region (hippocampus) 64^3 ROI 3D CNN.
Backbone training objective is 4-class CDR staging (a representation-
learning step -- see docs/methods.md), same split as the whole-volume
model, but on much smaller crops.

Architectures (all smaller than the whole-volume r2plus1d_18@31M):
    roi_resnet8    MONAI 3D ResNet, layers [1,1,1,1], planes (32,64,128,256)  ~1.0M
    roi_resnet18h  MONAI 3D ResNet, layers [2,2,2,2], planes (32,64,128,256)  ~2.4M (half width)
    roi_plain3     shallow plain 3D CNN (3 conv blocks + head, no residual)   ~0.3M

Protocol: AdamW 1e-4, wd 1e-4, dropout 0.3 head, label smoothing 0.1, mixup
0.2, grad clip 1.0, bf16, <=40 epochs, early stop patience 8 on val
balanced accuracy, sqrt-balanced class weights, LR-flip + intensity
augmentation, seeds 202/303/404.

Crop-size ablation: best architecture re-run at 96^3 (surrounding cortex).

Usage:
    AD_STAGE_NET_ROOT=/path/to/your/oasis3-working-directory \
        python train_hippocampus_roi_cnn.py sweep
    AD_STAGE_NET_ROOT=/path/to/your/oasis3-working-directory \
        python train_hippocampus_roi_cnn.py size_ablation
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
LABELS = ROOT / "data" / "processed" / "mr_finetune_labels.csv"
CROPS = ROOT / "data" / "processed" / "roi_crops"
RES = ROOT / "results" / "hippocampus_roi_cnn"
MODELS = ROOT / "models" / "hippocampus_roi_cnn"

HP = dict(lr=1e-4, weight_decay=1e-4, dropout=0.3, label_smoothing=0.1,
          mixup_alpha=0.2, grad_clip=1.0, max_epochs=40, patience=8,
          batch_size=16)
SEEDS = [202, 303, 404]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_roi_arch(name: str) -> nn.Module:
    from monai.networks.nets import ResNet as MonaiResNet

    def wrap(m):
        fc = m.fc
        m.fc = nn.Sequential(nn.Dropout(HP["dropout"]), fc)
        return m

    if name == "roi_resnet8":
        return wrap(MonaiResNet(block="basic", layers=[1, 1, 1, 1],
                                block_inplanes=(32, 64, 128, 256), spatial_dims=3,
                                n_input_channels=1, num_classes=4))
    if name == "roi_resnet18h":
        return wrap(MonaiResNet(block="basic", layers=[2, 2, 2, 2],
                                block_inplanes=(32, 64, 128, 256), spatial_dims=3,
                                n_input_channels=1, num_classes=4))
    if name == "roi_plain3":

        class Plain3(nn.Module):
            def __init__(self):
                super().__init__()
                ch = [1, 32, 64, 128]
                self.blocks = nn.ModuleList()
                for i in range(3):
                    self.blocks.append(nn.Sequential(
                        nn.Conv3d(ch[i], ch[i + 1], 3, padding=1, bias=False),
                        nn.BatchNorm3d(ch[i + 1]), nn.ReLU(inplace=True),
                        nn.MaxPool3d(2)))
                self.head = nn.Sequential(nn.Dropout(HP["dropout"]), nn.Linear(128, 4))

            def forward(self, x):
                for b in self.blocks:
                    x = b(x)
                return self.head(x.mean(dim=(2, 3, 4)))
        return Plain3()
    raise ValueError(name)


class CropDS(Dataset):
    def __init__(self, df, crop_dir, size_name, is_train):
        self.items = []
        for _, r in df.iterrows():
            p = crop_dir / f"{r['session']}.npy"
            if p.exists():
                self.items.append((str(p), int(r["label"])))
        self.is_train = is_train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        v = np.load(self.items[i][0]).astype(np.float32)
        x = torch.from_numpy(v)[None]
        x = x / 255.0 * 2.0 - 1.0  # conformed intensities 0-255
        if self.is_train:
            if random.random() < 0.5:
                x = torch.flip(x, dims=[3])
            x = x * (0.9 + 0.2 * random.random()) + (random.random() - 0.5) * 0.2
        return x, self.items[i][1]


def mixup(x, y, alpha):
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def run(arch, size_name, seed, log):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    lab = pd.read_csv(LABELS)
    crop_dir = CROPS / f"hippocampus_{size_name}"
    tr_ds = CropDS(lab[lab.split == "train"], crop_dir, size_name, True)
    va_ds = CropDS(lab[lab.split == "val"], crop_dir, size_name, False)
    log(f"[{arch}@{size_name} s{seed}] train={len(tr_ds)} val={len(va_ds)} crops")
    tr = DataLoader(tr_ds, batch_size=HP["batch_size"], shuffle=True,
                    num_workers=4, pin_memory=True, drop_last=True,
                    persistent_workers=True)
    va = DataLoader(va_ds, batch_size=HP["batch_size"], shuffle=False,
                    num_workers=2, pin_memory=True)

    model = build_roi_arch(arch).to(DEVICE)
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    counts = np.bincount([y for _, y in tr_ds.items], minlength=4).astype(float)
    total = counts.sum()
    w = np.array([np.sqrt(total / c) if c > 0 else 0 for c in counts])
    w = w / w[w > 0].mean()
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=DEVICE),
                               label_smoothing=HP["label_smoothing"])
    opt = torch.optim.AdamW(model.parameters(), lr=HP["lr"], weight_decay=HP["weight_decay"])

    best, best_ep, patience = -1.0, 0, HP["patience"]
    history, t0 = [], time.time()
    for ep in range(1, HP["max_epochs"] + 1):
        model.train()
        run_l = seen = cor = 0
        for xb, yb in tr:
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
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
            run_l += loss.item() * xb.size(0)
            cor += (out.argmax(1) == yb).sum().item()
            seen += xb.size(0)
        model.eval()
        ps, ys = [], []
        with torch.no_grad():
            for xb, yb in va:
                xb = xb.to(DEVICE, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"):
                    out = model(xb)
                ps.extend(out.argmax(1).cpu().tolist()); ys.extend(yb.tolist())
        vb = balanced_accuracy_score(ys, ps)
        ta = cor / max(seen, 1)
        history.append({"epoch": ep, "train_loss": run_l / max(seen, 1),
                        "train_acc": ta, "val_balanced_acc": vb, "gap": ta - vb})
        log(f"[{arch}@{size_name} s{seed} ep{ep:02d}] loss {run_l/max(seen,1):.3f} "
            f"tr {ta:.3f} val {vb:.4f} gap {ta-vb:+.3f} ({time.time()-t0:.0f}s)", flush=True)
        if vb > best:
            best, best_ep, patience = vb, ep, HP["patience"]
            MODELS.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), MODELS / f"{arch}_{size_name}_s{seed}.pt")
        else:
            patience -= 1
            if patience <= 0:
                log(f"[{arch}@{size_name} s{seed}] early stop ep{ep} best {best:.4f}")
                break
    return {"arch": arch, "size": size_name, "seed": seed,
            "params_m": round(n_par, 2), "best_val": best,
            "best_epoch": best_ep, "epochs_run": len(history),
            "final_gap": history[-1]["gap"], "history": history}


def log_factory(path):
    def log(m, **_):
        print(m, flush=True)
        with open(path, "a") as f:
            f.write(str(m) + "\n")
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["sweep", "size_ablation"])
    a = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)
    log = log_factory(RES / "train.log")
    results = []
    out = RES / "sweep.json"
    if out.exists():
        results = json.loads(out.read_text())

    def has(arch, size, seed):
        return any(r["arch"] == arch and r["size"] == size and r["seed"] == seed for r in results)

    if a.mode == "sweep":
        jobs = [(arch, "64") for arch in ["roi_resnet8", "roi_resnet18h", "roi_plain3"]]
    else:
        prev = json.loads(out.read_text())
        best64 = max((r for r in prev if r["size"] == "64"), key=lambda r: r["best_val"])
        jobs = [(best64["arch"], "96")]
        log(f"size ablation: {best64['arch']} @96 (64^3 best was {best64['best_val']:.4f})")
    for arch, size in jobs:
        for seed in SEEDS:
            if has(arch, size, seed):
                continue
            results.append(run(arch, size, seed, log))
            out.write_text(json.dumps(results, indent=2))

    import collections
    agg = collections.defaultdict(list)
    for r in results:
        agg[(r["arch"], r["size"])].append(r)
    log("=== summary (val bacc mean±sd over seeds; final train-val gap) ===")
    for (arch, size), rs in sorted(agg.items()):
        b = [r["best_val"] for r in rs]
        g = [r["final_gap"] for r in rs]
        log(f"  {arch}@{size}: {np.mean(b):.4f} ± {np.std(b):.4f} (n={len(rs)}) gap {np.mean(g):+.3f}")


if __name__ == "__main__":
    main()
