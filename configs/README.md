# configs

This project's real, previously-run pipeline (the original research pipeline) does not use a
separate YAML/JSON configuration layer — hyperparameters, seeds, and
architecture choices are inline constants in each script, and that is
preserved here rather than introducing a config system that never
existed in the original code. What's actually configurable, and where:

| Configuration | Where it lives |
|---|---|
| Data root / FastSurfer install path | `.env.example` (repository root) — `AD_STAGE_NET_ROOT`, `FASTSURFER_HOME` |
| Training seeds, LightGBM hyperparameters | `src/models/common.py` (`SEEDS`, `BASELINE_PARAMS`) |
| 3D CNN hyperparameters (per approach) | the `HP` dict at the top of each `src/training/train_*.py` script |
| ROI region → FreeSurfer/FastSurfer label IDs | `src/data/extract_roi_crops.py` (`REGIONS`) |
| Subject-level split ratios / seed | `src/data/split_subjects.py` CLI arguments (`--seed`, `--train_frac`, `--val_frac`) |

`subject_split.schema.json` in this directory documents the structure of
the one artifact these scripts share across pipeline stages
(`data/processed/splits/subject_split.json`, produced once by
`split_subjects.py` and never regenerated) — with placeholder values only,
since the real file contains OASIS-3 subject identifiers and must never be
committed to this repository (see the root `.gitignore`).
