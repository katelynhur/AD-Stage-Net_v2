from pathlib import Path
import os
import subprocess
import tempfile
import threading

import nibabel as nib
import numpy as np

CACHE_ROOT = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
FASTSURFER_HOME = Path(
    os.environ.get("FASTSURFER_HOME", CACHE_ROOT / "adstagenet" / "FastSurfer")
)

_SETUP_LOCK = threading.Lock()

# FreeSurfer/FastSurfer aseg-DKT label IDs.
# Bilateral labels are merged to compute one centroid per region.
REGION_LABELS = {
    "hippocampus": (17, 53),
    "entorhinal": (1006, 2006),
    "amygdala": (18, 54),
    # Match the ventricular family used in the project's multi-region ROI:
    # left/right lateral + inferior-lateral ventricles.
    "ventricles": (4, 5, 43, 44),
}


def _run(cmd, cwd=None, env=None, timeout=None):
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        tail = "\n".join(proc.stdout.splitlines()[-80:])
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(map(str, cmd))}\n\n{tail}"
        )
    return proc.stdout


def ensure_fastsurfer():
    """
    Install the FastSurfer source/checkpoints into the persistent HF cache.

    The Space build remains a normal Gradio/ZeroGPU build; FastSurfer is fetched
    lazily because it is not a simple PyPI package.
    """
    with _SETUP_LOCK:
        runner = FASTSURFER_HOME / "run_fastsurfer.sh"
        if not runner.exists():
            FASTSURFER_HOME.parent.mkdir(parents=True, exist_ok=True)
            _run([
                "git", "clone", "--depth", "1", "--branch", "stable",
                "https://github.com/Deep-MI/FastSurfer.git",
                str(FASTSURFER_HOME),
            ], timeout=300)

        checkpoint_sentinel = FASTSURFER_HOME / ".adstagenet_checkpoints_ready"
        if not checkpoint_sentinel.exists():
            env = os.environ.copy()
            env["PYTHONPATH"] = (
                str(FASTSURFER_HOME)
                + os.pathsep
                + env.get("PYTHONPATH", "")
            )
            _run(
                [
                    "python",
                    str(FASTSURFER_HOME / "FastSurferCNN" / "download_checkpoints.py"),
                    "--all",
                ],
                cwd=FASTSURFER_HOME,
                env=env,
                timeout=900,
            )
            checkpoint_sentinel.touch()

    return FASTSURFER_HOME


def _center_crop_or_pad(arr, center, shape):
    shape = np.asarray(shape, dtype=int)
    center = np.asarray(center, dtype=float)
    start = np.floor(center - shape / 2.0).astype(int)
    end = start + shape

    src_start = np.maximum(start, 0)
    src_end = np.minimum(end, np.asarray(arr.shape))

    out = np.zeros(tuple(shape), dtype=np.float32)

    dst_start = src_start - start
    dst_end = dst_start + (src_end - src_start)

    src_slices = tuple(slice(int(a), int(b)) for a, b in zip(src_start, src_end))
    dst_slices = tuple(slice(int(a), int(b)) for a, b in zip(dst_start, dst_end))

    out[dst_slices] = arr[src_slices]
    return out


def _region_centroid(seg, label_ids, region):
    mask = np.isin(seg, np.asarray(label_ids))
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise RuntimeError(
            f"FastSurfer did not produce any voxels for {region} "
            f"(expected label IDs {label_ids})."
        )
    centroid = coords.mean(axis=0)
    return centroid, int(coords.shape[0])


def _run_fastsurfer_segmentation(t1_path, workdir, device="cuda"):
    fs_home = ensure_fastsurfer()
    subjects_dir = Path(workdir) / "fastsurfer_subjects"
    subject_id = "uploaded_scan"

    env = os.environ.copy()
    env["FASTSURFER_HOME"] = str(fs_home)
    env["PYTHONPATH"] = str(fs_home) + os.pathsep + env.get("PYTHONPATH", "")

    cmd = [
        "bash",
        str(fs_home / "run_fastsurfer.sh"),
        "--t1", str(Path(t1_path).resolve()),
        "--sid", subject_id,
        "--sd", str(subjects_dir.resolve()),
        "--seg_only",
        "--device", device,
        "--viewagg_device", "cpu",
        "--no_cereb",
        "--no_hypothal",
        "--no_cc",
        "--no_biasfield",
    ]

    _run(cmd, cwd=fs_home, env=env, timeout=1200)

    mri_dir = subjects_dir / subject_id / "mri"
    seg_path = mri_dir / "aparc.DKTatlas+aseg.deep.mgz"
    intensity_path = mri_dir / "orig.mgz"

    if not seg_path.exists():
        raise RuntimeError(f"FastSurfer segmentation was not created: {seg_path}")
    if not intensity_path.exists():
        raise RuntimeError(f"FastSurfer conformed T1 image was not created: {intensity_path}")

    return seg_path, intensity_path


def extract_rois_from_t1(t1_path, crop_shape=(64, 64, 64), device="cuda"):
    """
    Run FastSurfer once, then extract all anatomy-focused AD-Stage-Net crops.

    FastSurfer's segmentation and conformed T1 are on the same voxel grid.
    The segmentation labels locate each region; the returned arrays are intensity
    crops from the T1 image, not segmentation masks.
    """
    with tempfile.TemporaryDirectory(prefix="adstagenet_roi_") as tmp:
        seg_path, intensity_path = _run_fastsurfer_segmentation(
            t1_path,
            tmp,
            device=device,
        )

        seg_img = nib.load(str(seg_path))
        t1_img = nib.load(str(intensity_path))

        seg = np.asarray(seg_img.get_fdata(), dtype=np.int32)
        t1 = np.asarray(t1_img.get_fdata(), dtype=np.float32)

        if seg.shape != t1.shape:
            raise RuntimeError(
                f"FastSurfer segmentation shape {seg.shape} does not match "
                f"conformed T1 shape {t1.shape}."
            )

        rois = {}
        qc = {}

        for region, ids in REGION_LABELS.items():
            centroid, nvox = _region_centroid(seg, ids, region)
            rois[region] = _center_crop_or_pad(t1, centroid, crop_shape)
            qc[region] = {
                "label_ids": list(ids),
                "label_voxels": nvox,
                "centroid": [round(float(x), 1) for x in centroid],
            }

        return rois, qc
