# src/features

Clinical/cognitive feature engineering on top of the merged, labeled,
split OASIS-3 table: the UDS cognitive battery and visit-to-visit
trajectory deltas that feed the LightGBM baseline (`src/training/
train_clinical_cognitive_baseline.py`) and every fusion model.

## Provenance

| Script | What it does | Ported from (source pipeline) | What was changed |
|---|---|---|---|
| `cognitive_features.py` | Causal (nearest-prior, no post-baseline information) matching of the 29-item UDS Form-C1 cognitive battery to each visit | `03_multimodal_phase3/src/s1_cognitive_features.py` | Path constants only |
| `trajectory_features.py` | Visit-to-visit deltas (MMSE, CDR sum-of-boxes, hippocampal volume, each cognitive subtest) since the prior visit | `03_multimodal_phase3/src/s2_trajectory_features.py` | Path constants only |

Both scripts are direct ports of real code from this project's original
research pipeline (not included here) — nothing here is invented or
placeholder logic.
