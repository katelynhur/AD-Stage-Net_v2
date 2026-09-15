# src/training

Trains the clinical/cognitive LightGBM baseline and every 3D CNN backbone
(whole-volume, hippocampal, multi-region), and extracts the penultimate-
layer embeddings that feed the fusion models in `src/evaluation/`.

## Backbone pretraining objective

The whole-volume, hippocampal, and multi-region 3D CNNs are all trained
via 4-class CDR staging classification (0/0.5/1/2) on MR sessions matched
to train/val-split visits. This is a representation-learning step used
only to produce useful embeddings — **it is not a reported outcome of
this project.** See `docs/methods.md`.

## Provenance

| Script | What it does | Ported from (source pipeline) | What was changed |
|---|---|---|---|
| `train_clinical_cognitive_baseline.py` | LightGBM clinical/cognitive baseline, progression + development | Uses `src/models/common.py` (see `src/models/README.md`) | n/a — this script itself is this repository's own thin driver over the shared, ported `common.py` |
| `train_wholevolume_cnn.py` | 7-architecture 3D CNN sweep + multi-seed confirmation on whole-volume T1 | `03_multimodal_phase3/src/s3_search.py` | Path constants only; a CPU fallback (`torch.cuda.is_available()`) was added for portability; a stale docstring claim about Kinetics-400 pretrained weights was corrected to match what the real code actually does (`weights=None`, trained from scratch for every architecture) |
| `extract_wholevolume_embeddings.py` | Extracts the winning whole-volume CNN's penultimate-layer embedding per scan | `03_multimodal_phase3/src/s3_extract.py` | Path constants only |
| `verify_wholevolume_baseline.py` | Reproduces the whole-volume CNN's reported validation accuracy from its saved checkpoint, as a sanity gate | `04_longitudinal_roi_phase4/src/s0_verify.py` | Path constants only |
| `train_hippocampus_roi_cnn.py` | Single-region (hippocampus) 64³/96³ ROI 3D CNN sweep + crop-size ablation | `04_longitudinal_roi_phase4/src/s1a_train_roi.py` | Path constants only |
| `train_multiregion_roi_cnn.py` | Shared-trunk 3D CNN over all four ROIs (hippocampus, entorhinal cortex, amygdala, ventricles), gated on the hippocampal model's representation quality | `04_longitudinal_roi_phase4/src/s1b_train_multiregion.py` | Path constants only |
| `extract_roi_embeddings.py` | Extracts penultimate-layer embeddings from the best hippocampal and multi-region models | `04_longitudinal_roi_phase4/src/s1c_extract_embeddings.py` | Path constants only |

"Path constants only" means the real scripts hardcode paths relative to
the old numbered-phase directory layout (or, in one case, a literal
absolute path) that does not exist in this repository; those were
replaced with the documented `$AD_STAGE_NET_ROOT`-relative layout
(`docs/data_access.md`). Architectures, hyperparameters, training loops,
and seeds are unchanged from the real, previously-run code.
