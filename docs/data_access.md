# Data access and privacy

## OASIS-3

This project uses OASIS-3 only. OASIS-3 is not included in, and cannot be
downloaded from, this repository. Access requires a data use agreement
through the OASIS project / NITRC-IR (https://www.nitrc.org/,
https://www.oasis-brains.org/) -- request access there before running any
script in this repository.

No OASIS-3 imaging data, raw or processed clinical/cognitive records,
subject/session/scan identifiers or manifests, participant-level
prediction exports, or trained model checkpoints are, or should ever be,
committed to this repository. See `.gitignore`.

## Expected directory layout

Set `AD_STAGE_NET_ROOT` (see `.env.example`) to a directory you control,
laid out as:

```
$AD_STAGE_NET_ROOT/
  data/
    raw/oasis3/OASIS3_data_files/scans/...   # from NITRC-IR, per OASIS-3's own layout
    raw/oasis3/imaging/MR/<session>/...      # T1w NIfTI sessions
    processed/...                            # script outputs (parquet/csv, embeddings, crops)
  models/...                                 # trained checkpoints (LightGBM .txt, CNN .pt)
  results/...                                # metrics JSON
  predictions/...                            # per-visit predictions (parquet)
```

All scripts under `src/` read and write relative to `AD_STAGE_NET_ROOT`,
never relative to a path inside this repository.

## FastSurfer

ROI segmentation (`src/data/segment_rois_fastsurfer.py`) depends on
FastSurfer (https://github.com/Deep-MI/FastSurfer), a third-party tool
with its own license. It is not vendored in this repository -- install it
separately and set `FASTSURFER_HOME` to point at your checkout / its own
Python environment.

## Reproduction order

```
1. src/data/merge_oasis3.py
2. src/data/build_labels.py
3. src/data/split_subjects.py            (run once; never regenerate)
4. src/data/apply_subject_split.py
5. src/features/cognitive_features.py
6. src/features/trajectory_features.py
7. src/training/train_clinical_cognitive_baseline.py

8. src/data/match_visits_to_mri.py
9. src/data/preprocess_mri.py
10. src/training/train_wholevolume_cnn.py sweep
11. src/training/train_wholevolume_cnn.py multiseed --archs r2plus1d_18
12. src/training/extract_wholevolume_embeddings.py --spec r2plus1d_18:<best_seed>
13. src/evaluation/wholevolume_fusion_ablation.py

14. src/data/segment_rois_fastsurfer.py
15. src/data/extract_roi_crops.py
16. src/training/train_hippocampus_roi_cnn.py sweep
17. src/training/train_hippocampus_roi_cnn.py size_ablation
18. src/training/train_multiregion_roi_cnn.py
19. src/training/extract_roi_embeddings.py
20. src/evaluation/hippocampus_fused_ablation.py
21. src/evaluation/hippocampus_image_only.py
22. src/evaluation/multiregion_fused_ablation.py
```

Every script prints its own inputs/outputs; see each module's docstring
for the exact CLI options.
