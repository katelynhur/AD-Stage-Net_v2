# AD-Stage-Net v2

**Author:** Katelyln Hur

Code and aggregated results for evaluating whether structural MRI adds predictive value beyond clinical and cognitive data when predicting Alzheimer’s disease progression and development.

## Research question

Does structural MRI add predictive value beyond clinical and cognitive
data when predicting Alzheimer's disease progression and development?

This repository evaluates that question on OASIS-3 across three modeling
approaches -- a clinical/cognitive tabular baseline, a whole-volume
structural MRI CNN, and hippocampal / multi-region ROI CNNs -- each fused
with the tabular baseline and compared against it via an inference-time
masked-image ablation.

## Why overall multimodal AUC does not prove MRI adds unique information

A fused model's AUC reflects everything feeding it; a high fused AUC is
fully consistent with MRI contributing nothing beyond what the
clinical/cognitive features already capture. The evidence for or against a
*unique* MRI contribution is the masked-image ablation: the same trained
fused model, re-scored on the same test rows with only the MRI embedding
columns zeroed out. The delta between the full and masked AUC -- not the
fused AUC on its own -- is what isolates MRI's marginal value. This is why
the whole-volume model's strong fused AUC (~0.79-0.80) coexists with an
essentially flat masked-ablation delta: overall performance and MRI's
unique contribution are different questions. See `docs/methods.md` for the
full methodological writeup.

## Dataset

OASIS-3 only.

| | n |
|---|---|
| Clinical/cognitive visits | 8,625 |
| Clinical/cognitive participants | 1,377 |
| Labeled T1 MRI sessions | 1,964 |
| MRI participants | 1,106 |

Split: fixed subject-level train/val/test (70/15/15, stratified by
baseline CDR), generated once and reused everywhere -- no individual
crosses between splits, and every feature used at a given visit is
information available on or before that visit.

## Outcomes

1. **Progression** -- binary: does Clinical Dementia Rating (CDR) increase
   at the subject's next available visit? (Any cause; not AD-specific.)
2. **Development** -- binary, restricted to CDR=0 (cognitively normal)
   baseline visits: does the subject develop Alzheimer's disease (AD-
   specific dementia diagnosis evidence) within three years? Internally
   named `preclinical` in the code; always called "development" here.

Both labels are excluded from training/eval (not treated as negative)
when there isn't enough follow-up to determine the outcome -- censoring,
not missing data. Full definitions: `docs/methods.md`,
`src/data/build_labels.py`.

## Models

**Clinical/cognitive baseline** -- LightGBM on demographics, FreeSurfer
structural volumes, current-visit CDR/MMSE/diagnosis, a UDS cognitive
battery (causal-matched, no post-baseline information), and visit-to-visit
trajectory deltas. This is the tabular-only baseline every imaging
comparison below is measured against.
(`src/training/train_clinical_cognitive_baseline.py`)

**Whole-volume MRI** -- `r2plus1d_18` 3D CNN over registered, whole-brain
T1 volumes (selected from a 7-architecture sweep); penultimate-layer
embeddings fused with the clinical/cognitive baseline in LightGBM.
(`src/training/train_wholevolume_cnn.py`,
`src/evaluation/wholevolume_fusion_ablation.py`)
**Provenance note:** the whole-volume numbers in the Results table below
are inherited from the original project's production model, whose image
block combined whole-volume MRI **and PET** embeddings together, not MRI
alone -- see [`results/whole_volume/`](results/whole_volume/) for full
disclosure, including a lower-rigor but genuinely MRI-only comparison
point.

**Hippocampal ROI** -- 64x64x64-voxel hippocampal crop, 3D ResNet-18-style
CNN, embeddings fused with the clinical/cognitive baseline.
(`src/training/train_hippocampus_roi_cnn.py`,
`src/evaluation/hippocampus_fused_ablation.py`,
`src/evaluation/hippocampus_image_only.py`)

**Multi-region ROI** -- four 64^3 crops (hippocampus, entorhinal cortex,
amygdala, ventricles), one shared CNN trunk, embeddings fused with the
clinical/cognitive baseline.
(`src/training/train_multiregion_roi_cnn.py`,
`src/evaluation/multiregion_fused_ablation.py`)

For every fusion model, the reported "MRI-masked" number re-scores the
*same trained model* with the MRI embedding block zeroed at inference
time -- not an independently retrained clinical-only model.

## Results

All numbers below are read directly from `results/summary.json` /
`results/*/*.json`; nothing here was estimated or back-calculated. See
[`results/README.md`](results/README.md) for metric definitions and full
provenance, including two caveats flagged below.

| Approach | Task | Test n | Image-only AUC | Full-model AUC | MRI-masked AUC | MRI contribution (Δ AUC) |
|---|---|---|---|---|---|---|
| Whole-volume¹ | Progression | 1,073 | 0.5818 | 0.7925 | 0.7887 | -0.0040 |
| Whole-volume¹ | Development | 745 | 0.5776 | 0.8017 | 0.8067 | +0.0050 |
| Hippocampal ROI | Progression | 381² | 0.5546³ | 0.8034 | 0.7809 | +0.0225 |
| Hippocampal ROI | Development | 286² | 0.6242³ | 0.8582 | 0.8046 | +0.0536 |
| Multi-region ROI⁴ | Progression | 381² | 0.5571³ | 0.8028 | 0.7630 | +0.0399 |
| Multi-region ROI⁴ | Development | 286² | 0.5871³ | 0.8602 | 0.8268 | +0.0334 |

Supporting files: [`results/whole_volume/`](results/whole_volume/) ·
[`results/hippocampal_roi/`](results/hippocampal_roi/) ·
[`results/multiregion_roi/`](results/multiregion_roi/)

(Whole-volume Δ is reported as masked − full, following the ablation
convention in `src/evaluation/wholevolume_fusion_ablation.py`; ROI Δ is
full − masked. Both read as "how much AUC is at stake in the MRI block";
see `docs/methods.md` for why the two approaches use this convention and
why negative/near-zero whole-volume deltas and positive ROI deltas are
comparable evidence of the same underlying pattern.)

¹ **Provenance caveat:** these whole-volume numbers come from the
original project's production model, whose image block combined
whole-volume MRI **and PET** embeddings (PET covered ~25% of visits in
that cohort) -- not an isolated MRI-only model. This repository's own
whole-volume pipeline is MRI-only but has not been re-run to regenerate
these figures. See [`results/whole_volume/`](results/whole_volume/) for
full disclosure and a lower-rigor, genuinely MRI-only comparison point.
² Full-model/MRI-masked/Δ columns use the scan-eligible subset (test rows
with a real ROI embedding) at seed 202, not the full test population.
³ Image-only AUC for the ROI approaches is computed on the **full** test
population (1,073 / 745), not the scan-eligible subset used for the other
three columns in the same row -- the denominators differ.
⁴ **Provenance caveat:** the multi-region full-model/MRI-masked/Δ values
are retained from previous publication; the source file implementing the
same scan-eligible-subset methodology used for hippocampal ROI could not
be located during a recursive audit of the original research pipeline.
The only located multi-region result file uses the full test population
and reports different fused-model AUCs (0.7550 progression, 0.8027
development) -- an unresolved discrepancy, documented in
[`results/multiregion_roi/`](results/multiregion_roi/). Image-only values
for multi-region ARE verified and unaffected by this caveat.

### Interpretation

- Clinical and cognitive variables provided a strong baseline on their
  own (AUC ≈0.75-0.80 across tasks).
- Whole-volume MRI added negligible independent predictive value once
  fused with that baseline (|Δ| ≤ 0.005 on both tasks).
- Anatomy-focused (ROI) MRI recovered a materially stronger imaging
  signal than the whole volume did.
- The largest MRI contribution measured in this project was the
  hippocampal-ROI development result: **+0.0536 AUC**.
- The highest overall fused AUC was the multi-region development result:
  **0.8602**.
- These are not the same finding: the model with the highest fused AUC
  (multi-region development, 0.8602) is not the one with the largest
  *unique* MRI contribution (hippocampal development, +0.0536) -- overall
  fused performance and MRI's marginal contribution answer different
  questions.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

A CUDA-capable GPU is required for the 3D CNN training/inference scripts
(`src/training/*_cnn.py`, `src/data/preprocess_mri.py`,
`src/data/segment_rois_fastsurfer.py`); the LightGBM scripts run on CPU.

FastSurfer (ROI segmentation) is a separate third-party install -- see
`docs/data_access.md`.

The Hugging Face Space application under `app/` has its own, partially
different dependency set (Gradio UI, TorchScript ROI encoders) -- see
`app/requirements.txt` and `app/README.md`, not this section, to run it.

## Authorized OASIS-3 data setup

OASIS-3 is not included in this repository. See `docs/data_access.md` for
the data use agreement, expected directory layout
(`$AD_STAGE_NET_ROOT/...`), and full reproduction order. In short:

1. Request OASIS-3 access via NITRC-IR / the OASIS project.
2. Copy `.env.example` to `.env`, set `AD_STAGE_NET_ROOT` (and
   `FASTSURFER_HOME` if running the ROI pipeline) to your own paths.
3. Run the scripts in the order listed in `docs/data_access.md`.

No raw or processed OASIS data, participant-level records, or model
checkpoints are, or should be, committed to this repository at any point.

## Reproducibility workflow

Every stage below is real, previously-run code ported from this project's
working research repository (not included here), restricted to structural
MRI and to the two outcomes this project reports. See each `src/*/README.md`
for the exact original source path and what was removed (excluded
modalities/outcomes, or a hardcoded local path). **This repository is not
claimed to be fully reproducible end-to-end**: it requires your own
authorized OASIS-3 download, a FastSurfer install, and GPU time it does not
provide -- see [Limitations](#limitations).

| # | Stage | Script |
|---|---|---|
| 1 | OASIS-3 cohort merge | `src/data/merge_oasis3.py` |
| 1 | Progression + development label construction | `src/data/build_labels.py` |
| 2 | Fixed subject-level train/val/test split | `src/data/split_subjects.py` → `src/data/apply_subject_split.py` |
| 3 | Clinical/cognitive feature preparation | `src/features/cognitive_features.py`, `src/features/trajectory_features.py` |
| 4 | Clinical/cognitive LightGBM baseline | `src/training/train_clinical_cognitive_baseline.py` |
| 5 | Visit↔MRI matching + whole-volume preprocessing | `src/data/match_visits_to_mri.py` → `src/data/preprocess_mri.py` |
| 6 | `r2plus1d_18` 3D CNN training + embedding extraction | `src/training/train_wholevolume_cnn.py` → `src/training/extract_wholevolume_embeddings.py` |
| 7 | Whole-volume MRI + LightGBM fusion, image-only, masked ablation | `src/evaluation/wholevolume_fusion_ablation.py` |
| 8 | FastSurfer segmentation + hippocampal/multi-region 64³ ROI extraction | `src/data/segment_rois_fastsurfer.py` → `src/data/extract_roi_crops.py` |
| 9 | Hippocampal 3D ResNet-18-style training + embedding extraction | `src/training/train_hippocampus_roi_cnn.py` → `src/training/extract_roi_embeddings.py` |
| 10 | Multi-region (hippocampus, entorhinal cortex, amygdala, ventricles) training + embedding extraction | `src/training/train_multiregion_roi_cnn.py` → `src/training/extract_roi_embeddings.py` |
| 12–13 | Hippocampal fusion, image-only, masked ablation | `src/evaluation/hippocampus_fused_ablation.py`, `src/evaluation/hippocampus_image_only.py` |
| 12–14 | Multi-region fusion, image-only, masked ablation, final metrics | `src/evaluation/multiregion_fused_ablation.py` → `results/summary.json` |

## Reproduction commands

```bash
# Clinical/cognitive pipeline + baseline
python src/data/merge_oasis3.py --data_root $AD_STAGE_NET_ROOT/data/raw/oasis3/OASIS3_data_files/scans \
    --output $AD_STAGE_NET_ROOT/data/processed/oasis3_merged.csv
python src/data/build_labels.py --input $AD_STAGE_NET_ROOT/data/processed/oasis3_merged.csv \
    --output $AD_STAGE_NET_ROOT/data/processed/oasis3_labeled.csv
python src/data/split_subjects.py --input $AD_STAGE_NET_ROOT/data/processed/oasis3_merged.csv \
    --output_dir $AD_STAGE_NET_ROOT/data/processed/splits
python src/data/apply_subject_split.py
python src/features/cognitive_features.py
python src/features/trajectory_features.py
python src/training/train_clinical_cognitive_baseline.py

# Whole-volume MRI
python src/data/match_visits_to_mri.py
python src/data/preprocess_mri.py
python src/training/train_wholevolume_cnn.py sweep
python src/training/train_wholevolume_cnn.py multiseed --archs r2plus1d_18
python src/training/extract_wholevolume_embeddings.py --spec r2plus1d_18:<best_seed>
python src/evaluation/wholevolume_fusion_ablation.py

# Hippocampal + multi-region ROI
python src/data/segment_rois_fastsurfer.py
python src/data/extract_roi_crops.py
python src/training/train_hippocampus_roi_cnn.py sweep
python src/training/train_hippocampus_roi_cnn.py size_ablation
python src/training/train_multiregion_roi_cnn.py
python src/training/extract_roi_embeddings.py
python src/evaluation/hippocampus_fused_ablation.py
python src/evaluation/hippocampus_image_only.py
python src/evaluation/multiregion_fused_ablation.py
```

## Repository structure

```
AD-Stage-Net-v2-repo/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── configs/            config approach (inline constants + .env.example) and a
│                       schema-only example of subject_split.json (see README.md)
├── src/
│   ├── data/           OASIS-3 merge, labels, split, MRI preprocessing,
│   │                   FastSurfer ROI extraction (see README.md for provenance)
│   ├── features/       clinical/cognitive feature engineering (UDS battery
│   │                   matching, trajectory deltas)
│   ├── models/         common.py -- shared LightGBM training/eval machinery
│   ├── training/       CNN backbones (whole-volume, hippocampal, multi-region),
│   │                   embedding extraction
│   └── evaluation/     fusion + image-only + masked-image ablation for each
│                       approach
├── app/                Hugging Face Space application source (see app/README.md)
├── results/
│   └── summary.json    verified summary metrics (no participant data)
└── docs/
    ├── methods.md
    └── data_access.md
```

Every `src/*` subdirectory has its own `README.md` naming the exact
original source file each script was ported from and what, if anything,
was removed (excluded modalities/outcomes, or a hardcoded local path) --
see those files rather than assuming any script here is a rewrite.

## Limitations

- The whole-volume and ROI CNN backbones are pretrained via 4-class CDR
  staging classification, purely as a representation-learning step -- not
  a reported outcome of this project. See `docs/methods.md`.
- No skull stripping in the whole-volume MRI preprocessing pipeline
  (documented, not silently resolved).
- ROI segmentations come from FastSurfer, not OASIS-3's shipped
  FreeSurfer outputs -- FastSurfer is validated against FreeSurfer but is
  a different tool; region boundaries (especially the thin entorhinal
  cortex) may differ from the shipped segmentations.
- Full-model / MRI-masked ROI ablation numbers are reported at a single
  seed (202) on the scan-eligible subset; multi-seed mean±SD is available
  in the per-model result JSON files for the broader (zero-imputed) test
  population but is not the headline number reported here.
- This repository evaluates OASIS-3 only; no external cohort validation
  is included.
- Full reproduction requires your own authorized OASIS-3 download (not
  redistributed here), a separate FastSurfer install, and a CUDA GPU for
  the 3D CNN stages -- none of which this repository provides. No trained
  checkpoints are included, so the exact reported numbers can only be
  regenerated by re-running the full pipeline, not by loading a shipped
  model.

## Hugging Face Space

**Deployed app:** https://huggingface.co/spaces/katelynhur/AD-Stage-Net_v2

**Application source:** [`app/`](app/) -- the Gradio application source
synced from the live Space's Git repository, supporting the same
progression/development tasks and hippocampal/multi-region ROI strategies
described above (whole-volume T1 MRI upload -> automatic FastSurfer
segmentation and ROI extraction -> CNN embedding -> clinical/cognitive
fusion). Trained model checkpoints are intentionally not included in this
GitHub repository -- see `app/README.md` for what was excluded and why, and
for local installation/run instructions. The live Space repository remains
the canonical source for the deployed model artifacts.

## Citation / acknowledgment

This project uses data from the OASIS-3 study:

> LaMontagne PJ, et al. OASIS-3: Longitudinal Neuroimaging, Clinical, and
> Cognitive Dataset for Normal Aging and Alzheimer's Disease. medRxiv,
> 2019. https://www.oasis-brains.org/

OASIS-3 data collection was supported by NIH grants and the Knight
Alzheimer Disease Research Center. Access requires a data use agreement
through the OASIS project -- see `docs/data_access.md`.
