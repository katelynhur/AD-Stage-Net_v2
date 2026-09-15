#!/usr/bin/env python3
"""
segment_rois_fastsurfer.py -- run FastSurferVINN segmentation (aparc+aseg +
aseg + conformed orig + mask) for every label-complete MRI session. FastSurfer
is a third-party tool (https://github.com/Deep-MI/FastSurfer, own license)
and is NOT vendored in this repository -- install it separately and point
FASTSURFER_HOME at your checkout / its own Python environment.

Per session outputs -> data/processed/fs_synth/<MR_session>/
    aparc+aseg.mgz  (DK atlas labels incl. entorhinal 1006/2006)
    aseg.mgz, mask.mgz, orig.mgz (conformed 1mm raw intensity, 256^3)

N parallel worker subprocesses share the GPU; each session writes to a
separate output dir, so no write conflicts. T1 selection matches the
project rule: prefer run-01, else first T1w.

Usage:
    AD_STAGE_NET_ROOT=/path/to/your/oasis3-working-directory \
    FASTSURFER_HOME=/path/to/FastSurfer \
        python segment_rois_fastsurfer.py --workers 3
"""

import argparse
import os
import subprocess
import threading
import time
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ.get("AD_STAGE_NET_ROOT", "/path/to/your/oasis3-working-directory"))
FASTSURFER_HOME = Path(os.environ.get("FASTSURFER_HOME", "/path/to/FastSurfer"))
FASTSURFER_PY = FASTSURFER_HOME / "venv" / "bin" / "python"  # or your own FastSurfer env's python
RUNPRED = FASTSURFER_HOME / "FastSurferCNN" / "run_prediction.py"
IMG_MR = ROOT / "data" / "raw" / "oasis3" / "imaging" / "MR"
LABELS = ROOT / "data" / "processed" / "mr_finetune_labels.csv"
OUT = ROOT / "data" / "processed" / "fs_synth"
LOG = ROOT / "results" / "segmentation.log"

lock = threading.Lock()
counts = {"ok": 0, "skip": 0, "fail": 0, "no_t1": 0}


def pick_t1w(session_dir: Path):
    t1s = sorted(session_dir.rglob("*_T1w.nii.gz"))
    if not t1s:
        return None
    for f in t1s:
        if "run-01" in f.name:
            return f
    return t1s[0]


def worker(queue, total):
    while True:
        try:
            session = queue.get_nowait()
        except Exception:
            return
        dest = OUT / session
        if ((dest / "aparc+aseg.mgz").exists() and (dest / "orig.mgz").exists()
                and (dest / "aseg.mgz").exists()):
            with lock:
                counts["skip"] += 1
            queue.task_done()
            continue
        t1 = pick_t1w(IMG_MR / session)
        if t1 is None:
            with lock:
                counts["no_t1"] += 1
                LOG.parent.mkdir(parents=True, exist_ok=True)
                with open(LOG, "a") as f:
                    f.write(f"NO_T1 {session}\n")
            queue.task_done()
            continue
        dest.mkdir(parents=True, exist_ok=True)
        cmd = [str(FASTSURFER_PY), str(RUNPRED),
               "--t1", str(t1.resolve()), "--sid", session,
               "--sd", str(OUT),
               "--asegdkt_segfile", "aparc+aseg.mgz",
               "--conformed_name", "orig.mgz",
               "--brainmask_name", "mask.mgz", "--aseg_name", "aseg.mgz",
               "--device", "cuda", "--vox_size", "1"]
        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
        ok = (dest / "aparc+aseg.mgz").exists()
        with lock:
            counts["ok" if ok else "fail"] += 1
            with open(LOG, "a") as f:
                f.write(f"{'OK' if ok else 'FAIL'} {session} {time.time()-t0:.0f}s\n")
                if not ok:
                    f.write(r.stderr[-500:] + "\n")
            n = sum(counts.values())
            if n % 25 == 0:
                print(f"{n}/{total} {counts}", flush=True)
        queue.task_done()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    lab = pd.read_csv(LABELS)
    sessions = sorted(lab["session"].unique())
    print(f"{len(sessions)} sessions to segment, {args.workers} workers")

    import queue as q
    qq = q.Queue()
    for s in sessions:
        qq.put(s)
    threads = [threading.Thread(target=worker, args=(qq, len(sessions)), daemon=True)
               for _ in range(args.workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print("DONE", counts)


if __name__ == "__main__":
    main()
