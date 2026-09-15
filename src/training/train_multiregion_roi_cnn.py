#!/usr/bin/env python3
"""
train_multiregion_roi_cnn.py -- multi-region ROI 3D CNN over four regions
(hippocampus, ventricles, amygdala, entorhinal cortex). Backbone training
objective is 4-class CDR staging, same representation-learning role as
train_hippocampus_roi_cnn.py -- see docs/methods.md.

GATE: only trains if train_hippocampus_roi_cnn.py's best 64^3 single-region
config beat both the whole-volume CNN and the clinical/cognitive tabular
staging accuracy (an internal engineering checkpoint on the backbone's
representation quality, not a reported outcome of this project).

Model: one branch per region, branch = the hippocampus model's winning
architecture's trunk; embeddings concatenated before the classifier head.
Each branch also has a small per-region linear projection head
(`region_proj`) that is part of the classifier architecture and is
preserved unmodified; this repository does not compute or report any
downstream interpretability analysis from those per-region scores.
WEIGHT-SHARING ABLATION: shared (one trunk applied to all four crops) vs.
independent (four trunks) -- both run, seeds 202/303/404.

Usage:
    AD_STAGE_NET_ROOT=/path/to/your/oasis3-working-directory \
        python train_multiregion_roi_cnn.py
"""

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
RES = ROOT / "results" / "multiregion_roi_cnn"
S1A_RES = ROOT / "results" / "hippocampus_roi_cnn" / "sweep.json"
MODELS = ROOT / "models" / "multiregion_roi_cnn"

REGIONS = ["hippocampus", "ventricles", "amygdala", "entorhinal"]
HP = dict(lr=1e-4, weight_decay=1e-4, dropout=0.3, label_smoothing=0.1,
          max_epochs=40, patience=8, batch_size=8)
SEEDS = [202, 303, 404]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GATE_WHOLEVOLUME_CNN = 0.4224   # r2plus1d_18 mean val over 4 seeds (backbone staging accuracy)
GATE_TABULAR_STAGING = 0.4592   # clinical/cognitive tabular staging accuracy (backbone gate only)


def trunk_from_hippocampus_model():
    """The hippocampus model's winning 64^3 architecture with its head
    removed; penultimate dim recorded."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_hippocampus_roi_cnn import build_roi_arch
    s1a = json.loads(S1A_RES.read_text())
    best64 = max((r for r in s1a if r["size"] == "64"), key=lambda r: r["best_val"])
    arch_name = best64["arch"]
    full = build_roi_arch(arch_name)
    fc = full.fc  # Sequential(Dropout, Linear)
    in_dim = fc[-1].in_features
    full.fc = nn.Identity()
    return full, in_dim, arch_name


class MultiRegionDS(Dataset):
    """Yields the four region crops (64^3) for a session, in fixed order."""

    def __init__(self, df, is_train):
        self.items = []
        for _, r in df.iterrows():
            paths = [CROPS / f"{reg}_64" / f"{r['session']}.npy" for reg in REGIONS]
            if all(p.exists() for p in paths):
                self.items.append(([str(p) for p in paths], int(r["label"])))
        self.is_train = is_train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        paths, y = self.items[i]
        vols = [np.load(p).astype(np.float32) / 255.0 * 2 - 1 for p in paths]
        xs = [torch.from_numpy(v)[None] for v in vols]
        if self.is_train:
            if random.random() < 0.5:
                xs = [torch.flip(x, dims=[3]) for x in xs]  # SAME flip all
            sc = 0.9 + 0.2 * random.random()
            sh = (random.random() - 0.5) * 0.2
            xs = [x * sc + sh for x in xs]
        return torch.stack(xs), y  # (4, 1, 64, 64, 64)


class MultiRegionNet(nn.Module):
    def __init__(self, shared: bool):
        super().__init__()
        trunk, in_dim, _ = trunk_from_hippocampus_model()
        self.in_dim = in_dim
        self.shared = shared
        if shared:
            self.trunk = trunk
        else:
            import copy
            self.trunk = nn.ModuleList([copy.deepcopy(trunk) for _ in REGIONS])
        # per-region linear projection heads, part of the classifier
        self.region_proj = nn.ModuleList([nn.Linear(in_dim, 1) for _ in REGIONS])
        self.head = nn.Sequential(
            nn.Dropout(HP["dropout"]),
            nn.Linear(in_dim * len(REGIONS) + len(REGIONS), 4))

    def forward(self, x):  # x: (B, 4, 1, 64, 64, 64)
        embs, scores = [], []
        for r in range(len(REGIONS)):
            xr = x[:, r]
            z = self.trunk(xr) if self.shared else self.trunk[r](xr)
            embs.append(z)
            scores.append(self.region_proj[r](z))
        cat = torch.cat(embs + scores, dim=1)
        return self.head(cat)


def run(shared, seed, log):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    lab = pd.read_csv(LABELS)
    tr = MultiRegionDS(lab[lab.split == "train"], True)
    va = MultiRegionDS(lab[lab.split == "val"], False)
    tag = "shared" if shared else "indep"
    log(f"[{tag} s{seed}] train={len(tr)} val={len(va)}")
    tr_dl = DataLoader(tr, batch_size=HP["batch_size"], shuffle=True,
                       num_workers=4, pin_memory=True, drop_last=True,
                       persistent_workers=True)
    va_dl = DataLoader(va, batch_size=HP["batch_size"], shuffle=False, num_workers=2)
    model = MultiRegionNet(shared).to(DEVICE)
    counts = np.bincount([y for _, y in tr.items], minlength=4).astype(float)
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
        rl = seen = cor = 0
        for xb, yb in tr_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"):
                out = model(xb)
            out = out.float()
            loss = crit(out, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            rl += loss.item() * len(yb); seen += len(yb)
            cor += (out.argmax(1) == yb).sum().item()
        model.eval()
        ps, ys = [], []
        with torch.no_grad():
            for xb, yb in va_dl:
                xb2 = xb.to(DEVICE)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"):
                    out = model(xb2)
                ps.extend(out.argmax(1).cpu().tolist()); ys.extend(yb.tolist())
        vb = balanced_accuracy_score(ys, ps)
        ta = cor / max(seen, 1)
        history.append({"epoch": ep, "train_loss": rl / max(seen, 1),
                        "train_acc": ta, "val_balanced_acc": vb, "gap": ta - vb})
        log(f"[{tag} s{seed} ep{ep:02d}] loss {rl/max(seen,1):.3f} tr {ta:.3f} "
            f"val {vb:.4f} gap {ta-vb:+.3f} ({time.time()-t0:.0f}s)", flush=True)
        if vb > best:
            best, best_ep, patience = vb, ep, HP["patience"]
            MODELS.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), MODELS / f"multiregion_{tag}_s{seed}.pt")
        else:
            patience -= 1
            if patience <= 0:
                log(f"[{tag} s{seed}] early stop ep{ep} best {best:.4f}@{best_ep}")
                break
    return {"shared": shared, "seed": seed, "best_val": best,
            "best_epoch": best_ep, "epochs_run": len(history),
            "final_gap": history[-1]["gap"], "history": history}


def main():
    RES.mkdir(parents=True, exist_ok=True)
    logp = RES / "train.log"

    def log(m, **_):
        print(m, flush=True)
        with open(logp, "a") as f:
            f.write(str(m) + "\n")

    if not S1A_RES.exists():
        log("GATE: hippocampus sweep.json missing -> cannot evaluate gate; abort")
        return
    s1a = json.loads(S1A_RES.read_text())
    import collections
    agg = collections.defaultdict(list)
    for r in s1a:
        if r["size"] == "64":
            agg[r["arch"]].append(r["best_val"])
    best_arch, best_vals = max(agg.items(), key=lambda kv: np.mean(kv[1]))
    mean64 = float(np.mean(best_vals))
    log(f"GATE: hippocampus best 64^3 = {best_arch} {mean64:.4f}±{np.std(best_vals):.4f}; "
        f"must beat whole-volume CNN {GATE_WHOLEVOLUME_CNN} AND tabular {GATE_TABULAR_STAGING}")
    if mean64 <= max(GATE_WHOLEVOLUME_CNN, GATE_TABULAR_STAGING):
        log("GATE FAILED -> multi-region training not run.")
        (RES / "gate_verdict.json").write_text(json.dumps(
            {"passed": False, "best_arch": best_arch, "mean_val": mean64}, indent=2))
        return
    log("GATE PASSED")
    (RES / "gate_verdict.json").write_text(json.dumps(
        {"passed": True, "best_arch": best_arch, "mean_val": mean64}, indent=2))

    out = RES / "results.json"
    results = json.loads(out.read_text()) if out.exists() else []
    for shared in [True, False]:
        for seed in SEEDS:
            if any(r["shared"] == shared and r["seed"] == seed for r in results):
                continue
            results.append(run(shared, seed, log))
            out.write_text(json.dumps(results, indent=2))
    for shared, tag in [(True, "shared"), (False, "indep")]:
        b = [r["best_val"] for r in results if r["shared"] == shared]
        if b:
            log(f"=== {tag}: {np.mean(b):.4f} ± {np.std(b):.4f} over {len(b)} seeds")


if __name__ == "__main__":
    main()
