# src/evaluation

Multimodal LightGBM fusion for progression and development, and the
inference-time MRI-masking ablation that isolates MRI's unique
contribution (see the root `README.md` and `docs/methods.md` for why
overall fused AUC alone does not answer that question).

Every script here reports three numbers per task: image-only (where
applicable), full fused model, and the same trained fused model re-scored
with the MRI embedding columns zeroed at inference time (never an
independently retrained clinical-only model).

## Provenance

| Script | What it does | Ported from (source pipeline) | What was changed |
|---|---|---|---|
| `wholevolume_fusion_ablation.py` | Tabular/image-only/fused LightGBM for the whole-volume approach, + masked ablation | `03_multimodal_phase3/src/s4_fusion_grid.py` (feature blocks) + `s4_ablation.py` (masked ablation) | The real `blocks()` function mixes `mri3d_emb_*` **and** `pet2d_emb_*` into the image block, and both files run four tasks (stage/progression/conversion/preclinical). PET was removed from the image block entirely; stage and conversion were removed. The MRI ablation logic and LightGBM training itself are unchanged. |
| `hippocampus_fused_ablation.py` | Full fusion (tabular + hippocampal ROI embedding) + masked ablation, with the scan-eligible-subset breakdown used in this project's reported results | `04_longitudinal_roi_phase4/src/s4b_hippocampus_fused.py` | The real file trains all four tasks and hardcodes an absolute local filesystem path to the working research repository. Stage and conversion were removed; the hardcoded path was replaced with `$AD_STAGE_NET_ROOT`. Fusion/masking/subset logic unchanged. |
| `hippocampus_image_only.py` | Hippocampal-embedding-only LightGBM (no clinical/cognitive features) | `04_longitudinal_roi_phase4/src/s4b_hippocampus_image_only.py` | Same task trim + path change |
| `multiregion_fused_ablation.py` | Image-only + full fusion (tabular + 4-region ROI embeddings) + masked ablation | `04_longitudinal_roi_phase4/src/s4b_multiregion_full.py` | Same task trim + path change |

"Task trim" means: the real files also train a 4-class CDR "stage" model
and a CDR=0.5-baseline "conversion to AD" model, neither of which is one
of this project's two reported outcomes (progression, development). Those
task branches were removed; the progression/development training,
LightGBM hyperparameters, and masked-ablation methodology are otherwise
identical to the real, previously-run code.
