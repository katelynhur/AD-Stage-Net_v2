#!/usr/bin/env python3
"""
extract_roi_embeddings.py -- penultimate embeddings from the best
hippocampus ROI model (-> embeddings/roi_single.parquet) and, if the
multi-region gate passed, the best multi-region model
(-> embeddings/roi_multiregion.parquet). Forward passes only, one parquet
per block, visit_idx-aligned, NaN where no crop/model exists.

roi_single block   : roi1a_emb_0000..  + has_roi1a_embedding
roi_multiregion    : roi1b_<region>_emb_0000.. per region (4 blocks) +
                     has_roi1b_embedding

Usage:
    AD_STAGE_NET_ROOT=/path/to/your/oasis3-working-directory \
        python extract_roi_embeddings.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(os.environ.get("AD_STAGE_NET_ROOT", "/path/to/your/oasis3-working-directory"))
CROPS = ROOT / "data" / "processed" / "roi_crops"
MATCHES = ROOT / "data" / "processed" / "visit_scan_matches.csv"
MODELS1A = ROOT / "models" / "hippocampus_roi_cnn"
MODELS1B = ROOT / "models" / "multiregion_roi_cnn"
S1A_RES = ROOT / "results" / "hippocampus_roi_cnn" / "sweep.json"
S1B_RES = ROOT / "results" / "multiregion_roi_cnn" / "results.json"
GATE = ROOT / "results" / "multiregion_roi_cnn" / "gate_verdict.json"
OUT = ROOT / "data" / "processed" / "embeddings"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sys.path.insert(0, str(Path(__file__).resolve().parent))


def penult_hook(model, head_name="fc"):
    holder = {}
    getattr(model, head_name).register_forward_pre_hook(
        lambda m, i: holder.__setitem__("x", i[0].detach()))
    return holder


@torch.no_grad()
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(MATCHES)
    n = len(pd.read_parquet(ROOT / "data" / "processed" / "oasis3_trajectory.parquet"))
    s1a = json.loads(S1A_RES.read_text())
    best = max(s1a, key=lambda r: r["best_val"])
    arch_name, size, seed = best["arch"], best["size"], best["seed"]
    print(f"roi_single: {arch_name}@{size} s{seed} (val {best['best_val']:.4f})")

    from train_hippocampus_roi_cnn import build_roi_arch
    model = build_roi_arch(arch_name).to(DEVICE).eval()
    sd = torch.load(MODELS1A / f"{arch_name}_{size}_s{seed}.pt",
                    map_location="cpu", weights_only=True)
    model.load_state_dict(sd, strict=True)
    holder = penult_hook(model, "fc")

    sessions = sorted(m["mri_session"].dropna().unique())
    crop_dir = CROPS / f"hippocampus_{size}"
    emb = {}
    for i, s in enumerate(sessions, 1):
        p = crop_dir / f"{s}.npy"
        if not p.exists():
            continue
        x = torch.from_numpy(np.load(p).astype(np.float32) / 255.0 * 2 - 1)
        xb = x[None, None].to(DEVICE)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"):
            model(xb)
        emb[s] = holder["x"].float().squeeze(0).cpu().numpy()
        if i % 400 == 0:
            print(f"  {i}/{len(sessions)}")
    dim = len(next(iter(emb.values())))
    mat = np.full((n, dim), np.nan, dtype=np.float32)
    has = np.zeros(n, dtype=bool)
    for _, r in m.iterrows():
        vi = int(r["visit_idx"])
        if pd.notna(r.get("mri_session")) and r["mri_session"] in emb:
            mat[vi] = emb[r["mri_session"]]
            has[vi] = True
    out = pd.DataFrame(mat, columns=[f"roi1a_emb_{j:04d}" for j in range(dim)])
    out["has_roi1a_embedding"] = has
    out.to_parquet(OUT / "roi_single.parquet")
    print(f"roi_single: dim={dim}, {int(has.sum())} visits -> roi_single.parquet")

    # ---- multi-region embeddings (only if gate passed) ----
    if not (GATE.exists() and json.loads(GATE.read_text()).get("passed")):
        print("multi-region gate not passed (or missing) -> no roi_multiregion block")
        return
    s1b = json.loads(S1B_RES.read_text())
    import collections
    agg = collections.defaultdict(list)
    for r in s1b:
        agg[r["shared"]].append(r["best_val"])
    shared = max(agg, key=lambda k: np.mean(agg[k]))
    best_b = max((r for r in s1b if r["shared"] == shared), key=lambda r: r["best_val"])
    print(f"roi_multiregion: {'shared' if shared else 'indep'} "
          f"s{best_b['seed']} (val {best_b['best_val']:.4f})")

    from train_multiregion_roi_cnn import MultiRegionNet, REGIONS
    mm = MultiRegionNet(shared).to(DEVICE).eval()
    sd = torch.load(MODELS1B / f"multiregion_{'shared' if shared else 'indep'}_s{best_b['seed']}.pt",
                    map_location="cpu", weights_only=True)
    mm.load_state_dict(sd, strict=True)

    emb_b = {}  # session -> (4, dim)
    for i, s in enumerate(sessions, 1):
        paths = [CROPS / f"{reg}_64" / f"{s}.npy" for reg in REGIONS]
        if not all(p.exists() for p in paths):
            continue
        vols = [torch.from_numpy(np.load(p).astype(np.float32) / 255.0 * 2 - 1) for p in paths]
        zs = []
        for ridx, v in enumerate(vols):
            xb = v[None, None].to(DEVICE)
            h = mm.trunk if shared else mm.trunk[ridx]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"):
                z = h(xb)  # trunk's fc is nn.Identity() -> output IS the embedding
            zs.append(z.float().squeeze(0).cpu().numpy())
        emb_b[s] = np.stack(zs)
        if i % 200 == 0:
            print(f"  multiregion {i}/{len(sessions)}")
    dim_b = emb_b[next(iter(emb_b))].shape[-1]
    cols = {}
    for ridx, reg in enumerate(REGIONS):
        mat = np.full((n, dim_b), np.nan, dtype=np.float32)
        for _, r in m.iterrows():
            vi = int(r["visit_idx"])
            if pd.notna(r.get("mri_session")) and r["mri_session"] in emb_b:
                mat[vi] = emb_b[r["mri_session"]][ridx]
        for j in range(dim_b):
            cols[f"roi1b_{reg}_emb_{j:04d}"] = mat[:, j]
    hasb = np.zeros(n, dtype=bool)
    for _, r in m.iterrows():
        vi = int(r["visit_idx"])
        if pd.notna(r.get("mri_session")) and r["mri_session"] in emb_b:
            hasb[vi] = True
    outb = pd.DataFrame(cols)
    outb["has_roi1b_embedding"] = hasb
    outb.to_parquet(OUT / "roi_multiregion.parquet")
    print(f"roi_multiregion: 4 x dim={dim_b}, {int(hasb.sum())} visits -> roi_multiregion.parquet")


if __name__ == "__main__":
    main()
