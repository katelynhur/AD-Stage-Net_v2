#!/usr/bin/env python3
"""
extract_wholevolume_embeddings.py -- extract penultimate-layer 3D embeddings
from the best (per-arch, best-val seed) whole-volume CNN checkpoint for
every MRI-matched visit, and attach them to the clinical/cognitive/
trajectory table as a new block. One forward pass per scan (whole volume --
no slice averaging), no training. New columns: mri3d_emb_0000.. ,
has_mri3d_embedding, mri3d_session, mri3d_offset_days.

Usage:
    AD_STAGE_NET_ROOT=/path/to/your/oasis3-working-directory \
        python extract_wholevolume_embeddings.py --spec r2plus1d_18:303
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_wholevolume_cnn import DEVICE, build_arch  # noqa: E402

ROOT = Path(os.environ.get("AD_STAGE_NET_ROOT", "/path/to/your/oasis3-working-directory"))


class Penult:
    def __init__(self, model, head_name):
        self.feat = None
        getattr(model, head_name).register_forward_pre_hook(self._h)

    def _h(self, module, inputs):
        self.feat = inputs[0].detach()


@torch.no_grad()
def embed_sessions(arch, seed, sessions, vol_dir, batch=8):
    model = build_arch(arch).to(DEVICE).eval()
    sd = torch.load(ROOT / "models" / "wholevolume_cnn" / f"{arch}_s{seed}.pt",
                    map_location="cpu", weights_only=True)
    model.load_state_dict(sd, strict=True)
    head = "classifier" if arch == "densenet121_3d" else "fc"
    grab = Penult(model, head)
    out = {}
    buf, names = [], []

    def flush():
        if not buf:
            return
        x = torch.stack(buf).to(DEVICE)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"):
            model(x)
        for n, f in zip(names, grab.feat.float().cpu().numpy()):
            out[n] = f.astype(np.float32)
        buf.clear(); names.clear()

    for s in sessions:
        p = vol_dir / f"{s}.npy"
        if p.exists():
            buf.append(torch.from_numpy(np.load(p).astype(np.float32)).unsqueeze(0))
            names.append(s)
        if len(buf) >= batch:
            flush()
    flush()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", action="append", required=True,
                    help="arch:seed, e.g. r2plus1d_18:303 (repeatable)")
    args = ap.parse_args()

    vol_dir = ROOT / "data" / "processed" / "mri3d"
    matches = pd.read_csv(ROOT / "data" / "processed" / "visit_scan_matches.csv")
    sessions = sorted(matches["mri_session"].dropna().unique())
    print(f"{len(sessions)} MRI-matched sessions")

    blocks = []
    for spec in args.spec:
        arch, seed = spec.split(":")
        print(f"embedding with {arch} (seed {seed})...")
        emb = embed_sessions(arch, int(seed), sessions, vol_dir)
        dim = len(next(iter(emb.values()))) if emb else 0
        print(f"  {len(emb)} sessions embedded, dim={dim}")
        blocks.append((arch, seed, emb))

    dim = sum(len(next(iter(b[2].values()))) for b in blocks)
    base = pd.read_parquet(ROOT / "data" / "processed" / "oasis3_trajectory.parquet")
    n = len(base)
    mat = np.full((n, dim), np.nan, dtype=np.float32)
    has = np.zeros(n, dtype=bool)
    sess = np.array([None] * n, dtype=object)
    off = np.full(n, np.nan)

    for _, r in matches.iterrows():
        vi = int(r["visit_idx"])
        if pd.isna(r.get("mri_session")):
            continue
        vecs = [b[2][r["mri_session"]] for b in blocks if r["mri_session"] in b[2]]
        if len(vecs) == len(blocks):
            mat[vi] = np.concatenate(vecs)
            has[vi] = True
            sess[vi] = r["mri_session"]
            off[vi] = r["mri_offset_days"]

    emb_df = pd.DataFrame(mat, columns=[f"mri3d_emb_{j:04d}" for j in range(dim)])
    audit = pd.DataFrame({"has_mri3d_embedding": has, "mri3d_session": sess,
                          "mri3d_offset_days": off})
    out = pd.concat([base.reset_index(drop=True), audit, emb_df], axis=1)
    assert len(out) == len(base)
    dst = ROOT / "data" / "processed" / "oasis3_wholevolume_embeddings.parquet"
    out.to_parquet(dst)
    print(f"saved {dst}: {int(has.sum())}/{len(matches)} visits with 3D MRI "
          f"embedding, dim={dim}, cols={out.shape[1]}")


if __name__ == "__main__":
    main()
