from pathlib import Path
import json
from functools import lru_cache

import lightgbm as lgb
import numpy as np
import torch

APP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = APP_DIR / "model_config.json"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    MODEL_SPECS = json.load(f)

# ZeroGPU provides CUDA emulation at module load and a real CUDA device inside
# @spaces.GPU functions.
DEVICE = torch.device("cuda")


def _center_crop_or_pad(arr, expected_shape):
    out = np.asarray(arr, dtype=np.float32)
    slices = []
    for n, target in zip(out.shape, expected_shape):
        if n > target:
            start = (n - target) // 2
            slices.append(slice(start, start + target))
        else:
            slices.append(slice(0, n))
    out = out[tuple(slices)]

    pad = []
    for n, target in zip(out.shape, expected_shape):
        deficit = max(0, target - n)
        before = deficit // 2
        after = deficit - before
        pad.append((before, after))
    if any(a or b for a, b in pad):
        out = np.pad(out, pad, mode="constant")
    return out


def normalize_roi(arr, expected_shape=(64, 64, 64)):
    """
    Per-volume z-score normalization.

    IMPORTANT: replace this function if the saved training pipeline used a
    different intensity normalization.
    """
    arr = _center_crop_or_pad(arr, expected_shape)
    finite = np.isfinite(arr)
    if not finite.any():
        raise ValueError("ROI contains no finite voxels.")

    values = arr[finite]
    mean = float(values.mean())
    std = float(values.std())
    if std < 1e-8:
        std = 1.0

    arr = (arr - mean) / std
    arr[~finite] = 0.0
    return arr.astype(np.float32)


@lru_cache(maxsize=4)
def load_roi_model(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    try:
        model = torch.jit.load(str(path), map_location=DEVICE)
        model.eval()
        return model
    except Exception as exc:
        raise RuntimeError(
            f"{path.name} is not a TorchScript encoder. Export the exact "
            "embedding-producing training model with torch.jit.save(), or adapt "
            "model_adapter.py to reconstruct its state_dict architecture."
        ) from exc


@lru_cache(maxsize=4)
def load_fusion_model(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return lgb.Booster(model_file=str(path))


def _to_tensor(arr):
    return torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0).to(DEVICE)


@torch.no_grad()
def extract_embedding(model, roi_arrays, spec):
    regions = spec["regions"]
    tensors = [_to_tensor(roi_arrays[r]) for r in regions]

    if len(tensors) == 1:
        out = model(tensors[0])
    else:
        out = model(*tensors)

    if isinstance(out, (tuple, list)):
        out = out[0]
    elif isinstance(out, dict):
        for key in ("embedding", "features", "feat"):
            if key in out:
                out = out[key]
                break
        else:
            raise ValueError("Encoder returned a dict without an embedding/features key.")

    if not torch.is_tensor(out):
        out = torch.as_tensor(out)

    embedding = out.detach().float().cpu().numpy().reshape(-1)
    if embedding.size == 0:
        raise ValueError("MRI encoder returned an empty embedding.")
    return embedding
