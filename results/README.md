# results

Aggregate, participant-level-data-free result files backing the root
`README.md` Results table. Every file here is a metrics/summary JSON copied
(and, where noted, trimmed) from the original research pipeline's saved
outputs -- no raw predictions, participant identifiers, or model checkpoints
are included.

## Metric definitions

- **Test n** -- size of the held-out test split a metric was computed on.
  For the ROI approaches (hippocampal, multi-region), most cells use the
  *scan-eligible subset* (test rows that actually have a real ROI
  embedding, not a zero-imputed placeholder); see each file's
  `with_embedding_subset_seed202` (hippocampal) or note below
  (multi-region).
- **Image-only AUC** -- LightGBM trained on imaging-embedding features
  only (no clinical/cognitive/trajectory features). For the ROI
  approaches this is computed on the **full** test population (not the
  scan-eligible subset used for the other three columns in the same row)
  -- the two are not on the same denominator; that's a real limitation of
  how these were originally computed, not a typo.
- **Full-model AUC** -- the fused (tabular + imaging) LightGBM model's AUC.
- **MRI-masked AUC** -- the *same trained fused model*, re-scored with the
  imaging embedding columns zeroed at inference time (no retraining).
  Isolates what the model actually leans on the image block for.
- **MRI contribution (Δ AUC)** -- full-model AUC minus MRI-masked AUC
  (sign convention varies slightly by file; see each JSON's own delta
  field for exactly how it was computed).
- `preclinical` in every source file below is this project's internal
  code/data name for the outcome documented everywhere public-facing
  (root `README.md`, this file) as **development**: does a CDR=0
  (cognitively normal) baseline participant develop AD-dementia evidence
  within three years.

## Files

### `hippocampal_roi/`

- `fusion_multiseed.json` -- full fusion (tabular + hippocampal ROI CNN
  embedding) + inference-time masked-image ablation, multi-seed. Trimmed
  from the original file to keep only the `progression` and `preclinical`
  (development) tasks -- the original also covered `stage` (4-class CDR
  staging) and `conversion` (a separate CDR=0.5-baseline "conversion to
  AD" outcome not reported by this project), both removed.
- `image_only_multiseed.json` -- hippocampal-embedding-only LightGBM
  (no clinical/cognitive features), same trimming.

Ported/rerun by `src/evaluation/hippocampus_fused_ablation.py` and
`src/evaluation/hippocampus_image_only.py`. This is the cleanest-provenance
result set here: single coherent MRI-only pipeline, every headline number
in the root README traces exactly to a field in one of these two files.

### `multiregion_roi/`

- `fusion_multiseed_full_test_population.json` -- image-only, fused, and
  masked-tabular-equivalent LightGBM results on the **full** test
  population (not the scan-eligible subset), for the multi-region
  (hippocampus + entorhinal cortex + amygdala + ventricles) ROI embedding.
  Trimmed the same way as the hippocampal files (stage/conversion
  removed).

**Provenance caveat:** the root README's multi-region **full-model AUC,
MRI-masked AUC, and Δ AUC** figures (0.8028 / 0.7630 / +0.0399 for
progression; 0.8602 / 0.8268 / +0.0334 for development) were retained
from previous publication. During this audit we could not locate a source
file implementing the same "scan-eligible subset at seed 202" methodology
used for the hippocampal ROI approach -- the only multi-region result file
found (included above) uses the full test population and reports
different fused-model AUCs (progression 0.7550, development 0.8027). This
is a known, unresolved discrepancy pending re-verification by actually
re-running `src/evaluation/multiregion_fused_ablation.py`. The
**image-only** AUC values (0.5571 progression, 0.5871 development) ARE
verified against the included file and are not affected by this caveat.

### `whole_volume/`

- `mri_pet_production_fusion.json` -- whole-volume fusion/ablation numbers
  for progression and development.

**Provenance caveat:** this repository's whole-volume approach is
documented (root README, `src/evaluation/wholevolume_fusion_ablation.py`)
as MRI-only. The image-only/full-model/MRI-masked AUC values published
here, however, are inherited from the original project's *production*
Stage-4 model, whose image block combined **whole-volume MRI AND PET**
embeddings together (PET coverage in that cohort was ~25% of visits). This
repository's own MRI-only whole-volume script has not actually been
re-run against embeddings extracted without PET, and PET is not part of
this repository's own data pipeline at all. The included JSON documents
this plainly, including a lower-rigor but genuinely MRI-only alternative
(single seed, untuned, using this repo's actual `r2plus1d_18`
architecture) for comparison. Treat the published whole-volume numbers as
representative of what the original combined pipeline measured, not as an
isolated test of MRI alone. This file also includes sensitivity/specificity
at the default 0.5 threshold for the full and masked scenarios, since
those were saved alongside AUC in the source pipeline. The hippocampal and
multi-region ROI source files did not save sensitivity/specificity,
balanced accuracy, F1, or confidence intervals -- only AUC-based metrics
are available for those two approaches.

## What's intentionally not here

Per this repository's scope, no raw or processed MRI/PET data, OASIS
clinical records, subject/participant identifiers, split manifests, model
checkpoints, cache files, or logs are included. Results for excluded
topics (Braak staging, Siamese longitudinal models, PET/CT-only models,
survival analysis, 4-class CDR staging, OASIS-4, or the project's prior
v1 iteration) are out of scope for this repository and are not
represented here.
