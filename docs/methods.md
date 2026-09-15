# Methods

## Cohort and labels

All data is OASIS-3. The clinical/cognitive table covers 8,625 visits from
1,377 participants; the imaging arms use 1,964 labeled T1 MRI sessions from
1,106 individuals. Splitting is subject-level (70/15/15, stratified by
baseline CDR) and fixed once (`src/data/split_subjects.py`) -- no
individual crosses between train/val/test, and every feature used at a
given visit is available on or before that visit (no post-baseline
information).

- **Progression**: binary, does CDR increase at the subject's next
  recorded visit (any cause).
- **Development**: binary, restricted to CDR=0 (cognitively normal)
  baseline visits -- does the subject show AD-specific dementia diagnosis
  evidence (`dx1` matching an "AD dem..."/"DAT..." pattern) within a
  1,095-day (3-year) window. Internally named `preclinical` /
  `converts_to_ad_preclinical` in the code (`src/data/build_labels.py`);
  always called "development" in this documentation.

Both labels are `NaN` (excluded from training/eval, not treated as
negative) when there's insufficient follow-up to determine the outcome --
real censoring, not missing data.

## Clinical/cognitive baseline

LightGBM, trained per task, on: demographics, FreeSurfer structural
volumes shipped with OASIS-3, current-visit CDR/MMSE/diagnosis, a 29-item
UDS Form-C1 cognitive battery block (causal nearest-prior match, <=365
days, no post-baseline information), and visit-to-visit trajectory deltas
(change in MMSE, CDR sum-of-boxes, hippocampal volume, and each cognitive
subtest since the prior visit). This is the tabular-only baseline that
every imaging comparison below is measured against
(`src/training/train_clinical_cognitive_baseline.py`).

## Whole-volume MRI approach

T1 volumes are registered to MNI152 (2mm, no skull stripping -- documented
limitation), resampled to a 96x112x96 grid, intensity-normalized
(`src/data/preprocess_mri.py`). A 3D CNN architecture sweep (MONAI 3D
ResNet variants, 3D DenseNet121, torchvision `r3d_18`/`r2plus1d_18`, all
trained from scratch) selects `r2plus1d_18` as the best architecture
(`src/training/train_wholevolume_cnn.py`). Its penultimate-layer embedding
per scan is extracted (`extract_wholevolume_embeddings.py`) and fused with
the clinical/cognitive/trajectory table in LightGBM
(`src/evaluation/wholevolume_fusion_ablation.py`), which also reports the
tabular-only and image-only variants and the masked-image ablation
(fused model re-scored with MRI embeddings zeroed, same trained model,
same test set).

## Hippocampal and multi-region ROI approach

FastSurfer (third-party, not vendored -- see `docs/data_access.md`)
segments each MRI session (`src/data/segment_rois_fastsurfer.py`); ROI
crops are extracted at 64^3 and 96^3 voxels, centered on the bilateral
label centroid, for four regions: hippocampus, ventricles, amygdala, and
entorhinal cortex (`src/data/extract_roi_crops.py`).

Two CNN backbones are trained on the 64^3 crops:
- **Hippocampal**: single-region 3D ResNet-18-style CNN
  (`src/training/train_hippocampus_roi_cnn.py`).
- **Multi-region**: one shared trunk (the hippocampal model's winning
  architecture) applied to all four region crops, embeddings concatenated
  before the classifier head, gated on the hippocampal model clearing a
  minimum representation-quality bar
  (`src/training/train_multiregion_roi_cnn.py`). Each region branch also
  has a small per-region linear projection head that is part of the
  classifier architecture; this repository does not compute or report any
  interpretability analysis from those per-region scores.

**Backbone training objective, and why it is not one of this project's
reported outcomes.** Both ROI backbones (and the whole-volume CNN above)
are trained via 4-class CDR staging classification on MR sessions matched
to train/val-split visits -- this is a representation-learning step used
only to produce useful penultimate-layer embeddings, not a predictor this
project evaluates or reports. The embeddings extracted from these
backbones (`extract_roi_embeddings.py`) are what actually feed the
progression and development LightGBM fusion models
(`src/evaluation/hippocampus_fused_ablation.py`,
`hippocampus_image_only.py`, `multiregion_fused_ablation.py`), which are
this project's reported outcomes.

## MRI-masking ablation and the "scan-eligible subset"

For each fusion model (whole-volume, hippocampal, multi-region), the
*same trained model* is re-scored on the test set with the MRI embedding
columns zeroed out at inference time -- this is not an independently
retrained clinical-only model, and the tabular block is untouched.

Many test visits have no MRI (or no usable ROI segmentation) close enough
to be matched -- for those, the fusion model reduces to a zero-imputed
image block. The full-model and MRI-masked AUCs reported in
`results/summary.json` for the two ROI approaches are computed on the
scan-eligible subset only (visits with a real embedding), so the
comparison is not diluted by rows where there was never any image
information to remove. The whole-volume ablation is reported on its full
matched test population, since its MRI-matching coverage does not
similarly bifurcate the test set. Reproduction: the hippocampal script
already isolates this subset explicitly; `multiregion_fused_ablation.py`
applies the identical subset-filtering method to the multi-region model's
own saved predictions, so both ROI approaches are compared on the same
methodology.

## Why overall multimodal AUC does not prove MRI adds unique information

A fused model's AUC combines everything in it; a high fused AUC is
consistent with MRI contributing nothing beyond what clinical/cognitive
features already capture. The masked-image ablation isolates MRI's
*marginal* contribution by holding the trained model and every other
input fixed and only removing the image block -- the delta between the
full and masked AUC on the same test rows is the actual evidence for or
against unique MRI value, not the fused AUC by itself. This is why the
whole-volume model's fused AUC (0.79-0.80) looks strong even though its
own masked-ablation delta is effectively zero.
