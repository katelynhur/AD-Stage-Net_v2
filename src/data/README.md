# src/data

OASIS-3 cohort merge, label construction, subject-level splitting, and
whole-volume/ROI MRI preprocessing. Every script here is a direct port of
real, previously-run code from this project's original research pipeline
(not included here) — restricted to the two outcomes this
project reports (progression, development) and to structural MRI, with
every other modality/outcome/phase-specific detail removed. No script in
this directory is a placeholder, a rewrite-from-scratch, or invented.

## Provenance

| Script | What it does | Ported from (source pipeline) | What was changed |
|---|---|---|---|
| `merge_oasis3.py` | Merges OASIS-3 CDR/diagnosis, demographics, and FreeSurfer CSVs into one visit-level table | `01_tabular_baseline/scripts/merge_oasis3.py` | Path constants only |
| `build_labels.py` | Progression label (any-cause CDR increase at next visit) and development label (AD-dementia evidence within 3 years of a CDR=0 baseline) | `01_tabular_baseline/scripts/build_labels.py` (progression) + `02_fusion_phase2/src/fusion/p1_build_labels_v2.py` (development, there named `converts_to_ad_preclinical`) | Consolidated two files into one; the real files' third label, CDR=0.5-baseline "conversion to AD" (a different outcome not reported by this project), was removed |
| `split_subjects.py` | Fixed, one-time, subject-level 70/15/15 train/val/test split stratified by baseline CDR | `01_tabular_baseline/scripts/split_subjects.py` | Path constants only |
| `apply_subject_split.py` | Applies the fixed split to the labeled visit table; asserts zero subject overlap | `01_tabular_baseline/scripts/apply_subject_split.py` | Path constants only |
| `prepare_modeling_data.py` | Builds leakage-free per-task feature/label tables from the split CSVs | `01_tabular_baseline/scripts/prepare_modeling_data.py` | Path constants only |
| `match_visits_to_mri.py` | Causal (no-future-leakage) visit-to-MR-session matching, and the 4-class CDR backbone-pretraining labels | `02_fusion_phase2/src/fusion/p2a_prepare_oasis3_slices.py` | The real file also causally matches PET sessions and extracts 2D axial-slice PNGs for the earlier AD-Stage-Net v1 fine-tuning approach; both were removed — this project uses whole-volume 3D MRI only |
| `preprocess_mri.py` | Registers T1 volumes to MNI152, resamples/normalizes, caches as float16 `.npy` | `03_multimodal_phase3/src/s3_preprocess_mri.py` | Path constants only |
| `segment_rois_fastsurfer.py` | Runs FastSurfer (external, not vendored) to segment each MRI session | `04_longitudinal_roi_phase4/src/s1_segment_all.py` | Path constants only |
| `extract_roi_crops.py` | Extracts 64³/96³ crops for the four ROIs (hippocampus, entorhinal cortex, amygdala, ventricles) from FastSurfer output | `04_longitudinal_roi_phase4/src/s1_extract_rois.py` | Path constants only |

"Path constants only" means: the real scripts hardcode paths relative to the
old numbered-phase directory layout (e.g. `PHASE3_ROOT`, or in one case a
literal absolute local filesystem path) — that layout does not exist in
this repository, so those constants were replaced with the documented
`$AD_STAGE_NET_ROOT`-relative layout (`docs/data_access.md`). No algorithm,
hyperparameter, or computed value was changed.
