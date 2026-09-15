#!/usr/bin/env python3
"""
preprocess_mri.py -- preprocess OASIS-3 T1w volumes for whole-volume 3D CNN
training/embedding extraction.

Pipeline per volume:
    1. nibabel load, reorient to closest canonical (RAS)
    2. SimpleITK affine registration to MNI152 T1 2mm (Mattes mutual
       information, multi-resolution, gradient descent w/ physical-shift
       scales) -- NO skull stripping (documented limitation; no FSL/
       FreeSurfer skull-stripped volumes used at this stage)
    3. Resample onto the template grid, zero-padded to 96x112x96 (x,y,z)
    4. Intensity: clip to per-volume [p1, p99] over non-zero voxels, z-score
    5. Cache as float16 .npy under data/processed/mri3d/<session>.npy

A failed registration writes NOTHING and returns 'regfail' -- the session
then simply has no 3D embedding downstream. No silent zero volumes.

Usage:
    AD_STAGE_NET_ROOT=/path/to/your/oasis3-working-directory \
        python preprocess_mri.py --workers 14
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("AD_STAGE_NET_ROOT", "/path/to/your/oasis3-working-directory"))
IMG_ROOT = ROOT / "data" / "raw" / "oasis3" / "imaging"
OUT_DIR = ROOT / "data" / "processed" / "mri3d"
MATCHES_CSV = ROOT / "data" / "processed" / "visit_scan_matches.csv"
SHAPE = (96, 112, 96)  # x, y, z (2mm grid, zero-padded from MNI 91x109x91)

_TEMPLATE = None  # per-worker cache


def _fixed_template():
    global _TEMPLATE
    if _TEMPLATE is None:
        import SimpleITK as sitk
        from nilearn import datasets
        mni = datasets.load_mni152_template(resolution=2)
        mni_arr = mni.get_fdata(dtype=np.float32)
        fixed = sitk.GetImageFromArray(mni_arr.transpose(2, 1, 0).astype(np.float32))
        mz = np.sqrt((mni.affine[:3, :3] ** 2).sum(axis=0))
        fixed.SetSpacing((float(mz[0]), float(mz[1]), float(mz[2])))
        fixed.SetOrigin(tuple(float(v) for v in mni.affine[:3, 3]))
        _TEMPLATE = fixed
    return _TEMPLATE


def pick_t1w(session_dir: Path):
    t1s = sorted(session_dir.rglob("*_T1w.nii.gz"))
    if not t1s:
        return None
    for f in t1s:
        if "run-01" in f.name:
            return f
    return t1s[0]


def crop_pad(a: np.ndarray, shape) -> np.ndarray:
    """Center-crop long axes, zero-pad short axes, to `shape`."""
    out = np.zeros(shape, dtype=np.float32)
    for d, s in enumerate(shape):
        if a.shape[d] > s:
            sl = (a.shape[d] - s) // 2
            a = a.take(indices=range(sl, sl + s), axis=d)
    slices = tuple(slice(0, a.shape[i]) for i in range(3))
    out[slices] = a[slices]
    return out


def preprocess_one(session: str) -> str:
    import nibabel as nib
    import SimpleITK as sitk

    out = OUT_DIR / f"{session}.npy"
    if out.exists():
        return "skip"
    nifti = pick_t1w(IMG_ROOT / "MR" / session)
    if nifti is None:
        return "no_t1w"

    img = nib.as_closest_canonical(nib.load(str(nifti)))
    arr = img.get_fdata(dtype=np.float32)

    sitk_img = sitk.GetImageFromArray(arr.transpose(2, 1, 0).astype(np.float32))
    z = np.sqrt((img.affine[:3, :3] ** 2).sum(axis=0))
    sitk_img.SetSpacing((float(z[0]), float(z[1]), float(z[2])))
    sitk_img.SetOrigin(tuple(float(v) for v in img.affine[:3, 3]))

    fixed = _fixed_template()

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.REGULAR)
    reg.SetMetricSamplingPercentage(0.15)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsGradientDescent(
        learningRate=1.0, numberOfIterations=120, convergenceMinimumValue=1e-6,
        convergenceWindowSize=10)
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2.0, 1.0, 0.0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    initial = sitk.CenteredTransformInitializer(
        fixed, sitk_img, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY)
    reg.SetInitialTransform(initial, inPlace=False)

    try:
        final = reg.Execute(fixed, sitk_img)
        resampled = sitk.Resample(
            sitk_img, fixed, final, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    except RuntimeError:
        return "regfail"

    out_arr = sitk.GetArrayFromImage(resampled).astype(np.float32)  # z,y,x
    out_arr = crop_pad(out_arr.transpose(2, 1, 0), SHAPE)           # -> x,y,z

    fg = out_arr[out_arr > 0]
    if fg.size < 1000:
        return "empty"
    lo, hi = np.percentile(fg, [1, 99])
    out_arr = np.clip(out_arr, lo, hi)
    out_arr = (out_arr - out_arr.mean()) / (out_arr.std() + 1e-8)
    np.save(out, out_arr.astype(np.float16))
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=14)
    args = ap.parse_args()

    matches = pd.read_csv(MATCHES_CSV)
    sessions = sorted(matches["mri_session"].dropna().unique())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"{len(sessions)} MR sessions to preprocess -> {OUT_DIR}", flush=True)

    from tqdm import tqdm
    counts = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in tqdm(ex.map(preprocess_one, sessions, chunksize=2), total=len(sessions)):
            counts[r] = counts.get(r, 0) + 1
    print(counts)


if __name__ == "__main__":
    main()
