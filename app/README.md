# Hugging Face application

**Deployed Space:** https://huggingface.co/spaces/katelynhur/AD-Stage-Net_v2

This directory contains the application source for the deployed Space above,
pulled from the Space's live Git repository
(https://huggingface.co/spaces/katelynhur/AD-Stage-Net_v2/tree/main), which
is the canonical source for the deployed model artifacts. **Trained model
checkpoints are intentionally not included here** -- see
[Model artifacts](#model-artifacts-not-included) below.

## What the application does

A Gradio app that takes one whole-volume T1 MRI plus clinical/cognitive
inputs and runs this project's multimodal fusion pipeline end to end:

```
whole-volume T1 MRI
        ↓
FastSurfer segmentation (fetched at runtime, not vendored)
        ↓
automatic 64x64x64 ROI extraction (hippocampus, and -- for the
multi-region strategy -- entorhinal cortex, amygdala, ventricles)
        ↓
ROI CNN encoder -> MRI embedding
        ↓
clinical/cognitive + MRI-embedding LightGBM fusion
        ↓
binary prediction
```

## Supported tasks

Both outcomes are **binary**, matching this project's two authorized
prediction tasks (`docs/methods.md` in the repository root) -- there is no
four-stage output in this application:

- **Progression** -- does CDR increase at the subject's next visit?
- **Development** -- does the subject develop Alzheimer's disease within
  three years, starting from a CDR = 0 baseline?

Each task can be run with either MRI strategy:

- **Hippocampal ROI** -- single-region hippocampus encoder
- **Multi-Region ROI** -- hippocampus + entorhinal cortex + amygdala +
  ventricles, shared encoder trunk

## Required inputs

- **MRI:** one whole-volume T1 NIfTI file (`.nii` or `.nii.gz`). There is no
  2D image / pre-extracted-ROI upload path -- FastSurfer segmentation and ROI
  extraction happen automatically inside the app.
- **Clinical/cognitive features:** entered in a fixed-row table (age,
  sex, education, APOE, MMSE, CDR global/sum-of-boxes, FreeSurfer structural
  volumes) and/or supplied as an optional CSV with `Feature,Value` columns
  (or a single-row wide CSV) for the full cognitive-battery feature set used
  by training. Any feature left blank is passed to the LightGBM fusion model
  as missing, which it handles natively.

## How the inputs are used

1. `roi_extractor.py` runs FastSurfer on the uploaded MRI (segmentation
   only: `--seg_only`, cerebellum/hypothalamus/corpus-callosum/bias-field
   disabled to keep inference lighter) and crops each region to 64^3 voxels
   from the conformed T1 intensity volume, centered on the bilateral label
   centroid.
2. `model_adapter.py` loads the TorchScript ROI encoder for the selected
   strategy and produces one MRI embedding per scan.
3. The clinical/cognitive values are merged with the MRI embedding into a
   single feature row, matched to the fusion LightGBM booster's exact
   trained feature names, and scored. The booster returns one probability
   (the positive-class probability for the selected binary task); the app
   never treats this as a four-class softmax.

## Model artifacts (not included)

The live Space's `models/` directory holds:

- `*_roi_encoder.pt` / `roi_resnet18h_64_s303.pt` / `multiregion_shared_s303.pt`
  -- TorchScript/PyTorch ROI CNN checkpoints (Git-LFS-tracked on the Space)
- `*.txt` LightGBM fusion boosters (e.g. `stage4b_hipfused_progression_s303.txt`)

**None of these are copied into this GitHub repository** -- they are
trained model weights, excluded per this project's data/model policy (see
the repository root `.gitignore`). They remain hosted on the deployed
Space and are downloaded/loaded there via `model_adapter.py`'s
`load_roi_model()` / `load_fusion_model()`, which read from the Space's own
local `models/` directory at runtime. No download URL or loading behavior
has been invented here beyond what those two functions already implement.

## Installing dependencies

```bash
pip install -r app/requirements.txt
```

`app/requirements.txt` is copied verbatim from the live Space's
`requirements.txt`, plus `gradio`, `spaces`, and `torch` appended at the
bottom -- those three are not in the Space's own requirements file because
the Hugging Face Spaces/ZeroGPU runtime provides them directly (`gradio`
via the Space's declared `sdk_version: "5.49.1"`, `torch` via the ZeroGPU
base image). No version has been invented for `torch`/`spaces`; install a
`torch` build matching your own CUDA setup from https://pytorch.org/.

FastSurfer itself is not a pip package: `roi_extractor.py` clones
`https://github.com/Deep-MI/FastSurfer` (stable branch) into the Hugging
Face cache and downloads its checkpoints on first use, the same way the
live Space does. This requires `git` and network access at runtime
(`app/packages.txt` lists the `apt` packages the Space installs for this:
`git`, `wget`, `ca-certificates`, `file`).

## Running locally

Without the trained checkpoints above, the app **cannot produce real
predictions** -- `model_adapter.py` raises a clear `FileNotFoundError` /
`RuntimeError` rather than fabricating output when a checkpoint is
missing, and the app itself raises a `gr.Error` naming the missing file. To
run it locally with real weights:

```bash
cd app
pip install -r requirements.txt
mkdir -p models
# copy your own copies of the checkpoints listed above into app/models/
python app.py
```

`app.py`'s inference function is wrapped in `@spaces.GPU(duration=300)`
(Hugging Face ZeroGPU); the `spaces` package's decorator is a no-op outside
actual Spaces hardware, so this also runs on a normal local CUDA GPU (or,
slowly, on CPU) without modification.

## Canonical source

The live Hugging Face Space repository
(https://huggingface.co/spaces/katelynhur/AD-Stage-Net_v2/tree/main) is the
canonical source for the deployed application and its model artifacts. If
the Space is updated, re-sync this directory from there rather than editing
the deployed Space from a stale copy of this one.
