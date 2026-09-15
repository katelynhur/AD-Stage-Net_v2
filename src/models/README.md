# src/models

`common.py` is the single shared LightGBM training/evaluation routine
used by every fusion and image-only evaluation script in `src/evaluation/`
and by `src/training/train_clinical_cognitive_baseline.py`: subject-split
loading, per-task feature-frame construction, multi-seed training with
early stopping, per-visit prediction export, and standard metrics
(ROC AUC, sensitivity/specificity, macro precision/recall/F1).

## Provenance

| File | Ported from (source pipeline) | What was changed |
|---|---|---|
| `common.py` | `03_multimodal_phase3/src/common.py` | The real file trains four tasks (4-class CDR "stage" classification, "progression", CDR=0.5-baseline "conversion to AD", and "preclinical"/development) and includes PET/CT leakage-column names. This project reports only progression and development, so the multiclass-staging branch and the "stage"/"conversion" task paths were removed, along with the PET/CT-specific leakage columns. The binary training/evaluation logic, LightGBM hyperparameters, and metrics themselves are unchanged. `PHASE3_ROOT`/`REPO_ROOT` (phase-relative, `01_tabular_baseline/data/processed/splits/...`) were replaced with the documented `$AD_STAGE_NET_ROOT`-relative layout. |

This is a real, previously-run shared module — not a new abstraction
invented for this repository. The real `03_multimodal_phase3/src/
s4_fusion_grid.py` and `04_longitudinal_roi_phase4/src/s4b_*.py` scripts
import this exact module (the phase-4 scripts reach it via `sys.path`
into phase 3); this repository's `src/evaluation/*.py` scripts reproduce
that same dependency by importing `src/models/common.py` directly.
