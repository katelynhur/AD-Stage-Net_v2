#!/usr/bin/env python3
"""
extract_roi_crops.py -- ROI crop extraction from FastSurfer segmentations,
parameterized by region. Reads data/processed/fs_synth/<session>/
{aparc+aseg.mgz, orig.mgz} (produced by segment_rois_fastsurfer.py).

Regions (bilateral, centroid of the COMBINED L+R label mask, voxel
coordinates), per crop size (64^3 primary, 96^3 with surrounding cortex):
  - crop the RAW INTENSITY volume (orig.mgz -- conformed 1mm 256^3), never
    the segmentation
  - center the box on the centroid; zero-pad when the box exceeds bounds
    and log every padded case
  - save float16 .npy under data/processed/roi_crops/<region>_<size>/<session>.npy
  - one manifest row per (session, region, size): subject, session, region,
    size, centroid (i,j,k), padded flag, label volume (voxels)

Entorhinal caveat: records label-volume distribution + outlier flags
(implausibly small/large vs cohort median), reported rather than absorbed.

Usage:
    AD_STAGE_NET_ROOT=/path/to/your/oasis3-working-directory \
        python extract_roi_crops.py
"""

import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("AD_STAGE_NET_ROOT", "/path/to/your/oasis3-working-directory"))
FS_DIR = ROOT / "data" / "processed" / "fs_synth"
OUT = ROOT / "data" / "processed" / "roi_crops"
LABELS = ROOT / "data" / "processed" / "mr_finetune_labels.csv"
SIZES = {"64": 64, "96": 96}

REGIONS = {
    # region -> (aseg labels for L+R, source segmentation file)
    "hippocampus": ([17, 53], "aparc+aseg.mgz"),
    "ventricles": ([4, 43], "aparc+aseg.mgz"),
    "amygdala": ([18, 54], "aparc+aseg.mgz"),
    "entorhinal": ([1006, 2006], "aparc+aseg.mgz"),
}


def crop_box(vol_shape, centroid, size):
    """Return (slices, pads, padded) for a size^3 box centered on centroid."""
    pads = []
    slices = []
    for c, dim in zip(centroid, vol_shape):
        lo = int(round(c - size // 2))
        hi = lo + size
        pad_lo = max(0, -lo)
        pad_hi = max(0, hi - dim)
        lo_c, hi_c = max(0, lo), min(dim, hi)
        slices.append(slice(lo_c, hi_c))
        pads.append((pad_lo, pad_hi))
    padded = any(p != (0, 0) for p in pads)
    return slices, pads, padded


def main():
    lab = pd.read_csv(LABELS)
    sessions = sorted(lab["session"].unique())
    print(f"{len(sessions)} sessions")

    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    entorhinal_volumes = {}
    counts = {"ok": 0, "missing_seg": 0, "empty_label": 0}
    for n, session in enumerate(sessions, 1):
        sdir = FS_DIR / session
        aseg_p = sdir / "aparc+aseg.mgz"
        orig_p = sdir / "orig.mgz"
        if not (aseg_p.exists() and orig_p.exists()):
            counts["missing_seg"] += 1
            continue
        seg = np.asarray(nib.load(str(aseg_p)).dataobj)
        orig = np.asarray(nib.load(str(orig_p)).dataobj).astype(np.float32)
        subject = session.split("_MR")[0]

        for region, (labels, _) in REGIONS.items():
            mask = np.isin(seg, labels)
            nvox = int(mask.sum())
            if region == "entorhinal":
                entorhinal_volumes[session] = nvox
            if nvox < 10:
                counts["empty_label"] += 1
                manifest.append({"subject_id": subject, "session": session,
                                 "region": region, "size": None,
                                 "centroid_i": None, "centroid_j": None,
                                 "centroid_k": None, "padded": None,
                                 "label_voxels": nvox, "status": "empty_label"})
                continue
            ci, cj, ck = [int(round(v)) for v in
                          np.array(np.nonzero(mask)).mean(axis=1)]
            for size_name, size in SIZES.items():
                slices, pads, padded = crop_box(orig.shape, (ci, cj, ck), size)
                box = orig[slices[0], slices[1], slices[2]]
                out = np.zeros((size, size, size), dtype=np.float32)
                tpl = tuple(slice(p[0], p[0] + box.shape[i]) for i, p in enumerate(pads))
                out[tpl] = box
                d = OUT / f"{region}_{size_name}"
                d.mkdir(parents=True, exist_ok=True)
                np.save(d / f"{session}.npy", out.astype(np.float16))
                manifest.append({"subject_id": subject, "session": session,
                                 "region": region, "size": int(size_name),
                                 "centroid_i": ci, "centroid_j": cj,
                                 "centroid_k": ck, "padded": bool(padded),
                                 "label_voxels": nvox, "status": "ok"})
        counts["ok"] += 1
        if n % 100 == 0:
            print(f"  {n}/{len(sessions)} {counts}", flush=True)

    mdf = pd.DataFrame(manifest)
    mdf.to_csv(OUT / "manifest.csv", index=False)
    print(counts)

    # ---- entorhinal reliability report ----
    ev = pd.Series(entorhinal_volumes)
    med = ev.median()
    rel = {
        "n_sessions": int(len(ev)),
        "empty_lt10": int((ev < 10).sum()),
        "median_voxels": float(med),
        "p5": float(ev.quantile(0.05)), "p95": float(ev.quantile(0.95)),
        "outlier_flagged_lt_half_median": int((ev < med / 2).sum()),
        "outlier_flagged_gt_double_median": int((ev > med * 2).sum()),
    }
    (ROOT / "results" / "entorhinal_reliability.json").parent.mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "entorhinal_reliability.json").write_text(json.dumps(rel, indent=2))
    print("entorhinal:", rel)
    print(f"manifest rows: {len(mdf)} -> {OUT / 'manifest.csv'}")


if __name__ == "__main__":
    main()
