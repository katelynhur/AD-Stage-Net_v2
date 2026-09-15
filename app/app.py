from pathlib import Path
import traceback

import gradio as gr
import spaces
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from model_adapter import (
    MODEL_SPECS,
    load_roi_model,
    load_fusion_model,
    extract_embedding,
    normalize_roi,
)
from roi_extractor import extract_rois_from_t1


APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "models"


# Binary outcome labels per prediction task (progression / development are
# both binary; there is no four-stage output in this project -- see
# docs/methods.md in the repository root for the exact label definitions).
BINARY_LABELS = {
    "Progression": ("No progression predicted", "Progression predicted"),
    "Development": ("No development predicted", "Development predicted"),
}

BINARY_COLORS = {
    "No progression predicted": "#22c55e",
    "Progression predicted": "#ef4444",
    "No development predicted": "#22c55e",
    "Development predicted": "#ef4444",
}


TASK_OPTIONS = [
    "Progression",
    "Development",
]


MRI_STRATEGY_OPTIONS = [
    "Multi-Region ROI",
    "Hippocampal ROI",
]


CORE_FEATURES = [
    "age_at_visit",
    "sex",
    "education_years",
    "apoe",
    "mmse",
    "cdr_global",
    "cdr_sum_of_boxes",
    "intracranial_volume",
    "total_gray_matter_volume",
    "hippocampal_volume_l",
    "hippocampal_volume_r",
    "hippocampal_volume_total",
    "ventricle_volume_l",
    "ventricle_volume_r",
    "entorhinal_thickness_l",
    "entorhinal_thickness_r",
]


DEFAULT_CORE_ROWS = [[name, ""] for name in CORE_FEATURES]


PROJECT_MD = r"""
# AD-Stage-Net v2: Multimodal Alzheimer’s Disease Prediction

**AD-Stage-Net v2** combines structural MRI with clinical and cognitive
information to evaluate future Alzheimer’s disease outcomes.

Upload a **whole-volume T1 MRI** (`.nii` / `.nii.gz`). The application automatically
performs anatomical segmentation and ROI extraction before multimodal inference.

### Prediction Tasks

- **Progression** — predicts future disease progression
- **Development** — predicts future development of Alzheimer’s-related impairment

### MRI Strategies

- **Multi-Region ROI:** hippocampus + entorhinal cortex + amygdala + ventricles
- **Hippocampal ROI:** focused hippocampal representation

> **Research demonstration only — not a medical device or clinical diagnostic tool.**
"""


# ============================================================
# Basic helpers
# ============================================================

def _to_float(value):
    if value is None:
        return np.nan

    if isinstance(value, str):
        s = value.strip()

        if not s:
            return np.nan

        low = s.lower()

        if low in {"m", "male"}:
            return 1.0

        if low in {"f", "female"}:
            return 0.0

        if low in {"yes", "true", "positive"}:
            return 1.0

        if low in {"no", "false", "negative"}:
            return 0.0

        try:
            return float(s)

        except ValueError:
            return np.nan

    try:
        return float(value)

    except Exception:
        return np.nan


def _core_table_to_dict(core_df):
    if core_df is None:
        return {}

    if not isinstance(core_df, pd.DataFrame):
        core_df = pd.DataFrame(
            core_df,
            columns=["Feature", "Value"],
        )

    output = {}

    for _, row in core_df.iterrows():
        key = str(row.iloc[0]).strip()

        if key:
            output[key] = _to_float(row.iloc[1])

    return output


def _read_extra_features(extra_csv):
    if extra_csv is None:
        return {}

    df = pd.read_csv(Path(extra_csv))

    if df.empty:
        return {}

    lower_cols = {
        str(c).strip().lower(): c
        for c in df.columns
    }

    if "feature" in lower_cols and "value" in lower_cols:
        feature_col = lower_cols["feature"]
        value_col = lower_cols["value"]

        return {
            str(row[feature_col]).strip(): _to_float(row[value_col])
            for _, row in df.iterrows()
            if str(row[feature_col]).strip()
        }

    row = df.iloc[0]

    return {
        str(column): _to_float(row[column])
        for column in df.columns
    }


def _is_nifti(path):
    name = str(path).lower()

    return (
        name.endswith(".nii")
        or name.endswith(".nii.gz")
    )


# ============================================================
# Result display
# ============================================================

def _make_prob_chart(probabilities):
    labels = list(probabilities.keys())
    values = [
        float(probabilities[label]) * 100
        for label in labels
    ]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=[
                BINARY_COLORS.get(label, "#64748b")
                for label in labels
            ],
            text=[
                f"{value:.1f}%"
                for value in values
            ],
            textposition="outside",
        )
    )

    fig.update_layout(
        xaxis_title="Probability (%)",
        xaxis_range=[0, 105],
        height=280,
        margin=dict(
            l=20,
            r=30,
            t=20,
            b=30,
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    return fig


def _stage_html(
    stage,
    confidence,
    task_name,
    strategy_name,
):
    color = BINARY_COLORS.get(
        stage,
        "#64748b",
    )

    return f"""
    <div style="
        border:1px solid {color};
        border-radius:14px;
        padding:18px 20px;
        background:{color}12;
        text-align:center;
        min-height:120px;
    ">
        <div style="
            font-size:.8rem;
            opacity:.7;
        ">
            Predicted Result
        </div>

        <div style="
            font-size:1.6rem;
            font-weight:750;
            color:{color};
            margin-top:4px;
        ">
            {stage}
        </div>

        <div style="
            font-size:1rem;
            margin-top:5px;
        ">
            {confidence * 100:.1f}% confidence
        </div>

        <div style="
            font-size:.78rem;
            opacity:.65;
            margin-top:7px;
        ">
            {task_name} · {strategy_name}
        </div>
    </div>
    """


def _binary_probabilities(raw):
    """
    Both progression and development are binary outcomes (see
    docs/methods.md), so the fusion LightGBM booster returns a single
    probability of the positive class per row. Returns [p_negative,
    p_positive].
    """
    arr = np.asarray(
        raw,
        dtype=np.float64,
    ).reshape(-1)

    if len(arr) == 1:
        p_positive = float(np.clip(arr[0], 0.0, 1.0))
        return np.array([1.0 - p_positive, p_positive])

    if len(arr) == 2 and np.all(arr >= 0) and 0.98 <= arr.sum() <= 1.02:
        return arr

    raise ValueError(
        f"Expected a single binary probability, got shape {np.asarray(raw).shape}."
    )


# ============================================================
# Task + MRI strategy routing
# ============================================================

def _resolve_model_key(
    task_choice,
    mri_strategy,
):

    if mri_strategy == "Multi-Region ROI":
        strategy_key = "multi_region"

    else:
        strategy_key = "hippocampal"

    if task_choice == "Progression":
        return f"progression_{strategy_key}"

    if task_choice == "Development":
        return f"development_{strategy_key}"

    raise gr.Error(
        f"Unknown prediction task: {task_choice}"
    )


def _task_output_description(task_choice):
    if task_choice == "Progression":
        return (
            "Predict future disease progression using information "
            "available at the current visit."
        )

    if task_choice == "Development":
        return (
            "Predict whether the participant will develop future "
            "Alzheimer’s-related impairment."
        )

    return ""


# ============================================================
# Model helpers
# ============================================================

def _required_files_message(model_key):
    spec = MODEL_SPECS[model_key]

    missing = []

    for filename in (
        spec["roi_checkpoint"],
        spec["fusion_model"],
    ):

        if not (
            MODELS_DIR / filename
        ).exists():

            missing.append(filename)

    if not missing:
        return ""

    return (
        "Missing model file(s): "
        + ", ".join(
            f"models/{filename}"
            for filename in missing
        )
    )


def _build_feature_frame(
    booster,
    clinical_values,
    embedding,
    spec,
):

    feature_names = booster.feature_name()

    values = {
        name: np.nan
        for name in feature_names
    }

    for name, value in clinical_values.items():
        if name in values:
            values[name] = value

    configured_names = spec.get(
        "embedding_feature_names",
        [],
    )

    if configured_names:

        if len(configured_names) != len(embedding):
            raise ValueError(
                "Configured MRI embedding contains "
                f"{len(configured_names)} names, but encoder returned "
                f"{len(embedding)} values."
            )

        for name, value in zip(
            configured_names,
            embedding,
        ):

            if name in values:
                values[name] = float(value)

    else:

        prefixes = (
            "img_",
            "mri_",
            "emb_",
            "embedding_",
            "roi_",
        )

        candidates = [
            name
            for name in feature_names
            if name.lower().startswith(prefixes)
        ]

        if len(candidates) != len(embedding):
            raise ValueError(
                "Could not automatically map the MRI embedding "
                "into the LightGBM model. Add the exact MRI "
                "embedding feature names to embedding_feature_names "
                "in model_config.json."
            )

        for name, value in zip(
            candidates,
            embedding,
        ):
            values[name] = float(value)

    return pd.DataFrame(
        [
            [
                values[name]
                for name in feature_names
            ]
        ],
        columns=feature_names,
    )


# ============================================================
# Multimodal inference
# ============================================================

def _run_multimodal(
    task_choice,
    mri_strategy,
    mri_path,
    core_df,
    extra_csv,
):

    if not _is_nifti(mri_path):
        raise gr.Error(
            "Please upload a whole-volume T1 MRI "
            "in .nii or .nii.gz format."
        )

    model_key = _resolve_model_key(
        task_choice,
        mri_strategy,
    )

    if model_key not in MODEL_SPECS:
        raise gr.Error(
            f"No deployment model is configured for "
            f"'{task_choice}' using '{mri_strategy}'. "
            f"Add '{model_key}' to model_config.json "
            "using the exact trained model files."
        )

    spec = MODEL_SPECS[model_key]

    missing = _required_files_message(
        model_key
    )

    if missing:
        raise gr.Error(missing)

    # --------------------------------------------------------
    # Automatic segmentation + ROI extraction
    # --------------------------------------------------------

    roi_arrays, roi_qc = extract_rois_from_t1(
        mri_path,
        crop_shape=tuple(
            spec["crop_shape"]
        ),
        device="cuda",
    )

    selected_rois = {}

    for region in spec["regions"]:

        selected_rois[region] = normalize_roi(
            roi_arrays[region],
            expected_shape=tuple(
                spec["crop_shape"]
            ),
        )

    # --------------------------------------------------------
    # MRI encoder
    # --------------------------------------------------------

    encoder = load_roi_model(
        MODELS_DIR
        / spec["roi_checkpoint"]
    )

    embedding = extract_embedding(
        encoder,
        selected_rois,
        spec,
    )

    # --------------------------------------------------------
    # Clinical / cognitive fusion
    # --------------------------------------------------------

    booster = load_fusion_model(
        MODELS_DIR
        / spec["fusion_model"]
    )

    clinical_values = _core_table_to_dict(
        core_df
    )

    clinical_values.update(
        _read_extra_features(
            extra_csv
        )
    )

    X = _build_feature_frame(
        booster,
        clinical_values,
        embedding,
        spec,
    )

    raw_prediction = booster.predict(X)

    raw_array = np.asarray(
        raw_prediction
    )

    if raw_array.ndim == 2:
        raw_array = raw_array[0]

    probabilities = _binary_probabilities(
        raw_array
    )

    negative_label, positive_label = BINARY_LABELS[task_choice]

    probability_map = dict(
        zip(
            (negative_label, positive_label),
            probabilities.tolist(),
        )
    )

    top_index = int(
        np.argmax(probabilities)
    )

    stage = (negative_label, positive_label)[
        top_index
    ]

    confidence = float(
        probabilities[top_index]
    )

    regions = ", ".join(
        spec["regions"]
    )

    filled_features = int(
        np.sum(
            ~pd.isna(X.iloc[0])
        )
    )

    total_features = len(
        X.columns
    )

    qc_text = "; ".join(
        (
            f"{region}: "
            f"{roi_qc[region]['label_voxels']} voxels"
        )
        for region in spec["regions"]
    )

    status = (
        f"**Prediction task:** {task_choice}  \n"
        f"**Task meaning:** {_task_output_description(task_choice)}  \n"
        f"**MRI strategy:** {mri_strategy}  \n"
        f"**Input:** whole-volume T1 MRI  \n"
        f"**Automatic processing:** FastSurfer → ROI extraction  \n"
        f"**Regions used:** {regions}  \n"
        f"**ROI size:** "
        f"{spec['crop_shape'][0]}×"
        f"{spec['crop_shape'][1]}×"
        f"{spec['crop_shape'][2]}  \n"
        f"**Fusion features populated:** "
        f"{filled_features}/{total_features}  \n"
        f"**ROI QC:** {qc_text}"
    )

    return (
        stage,
        confidence,
        probability_map,
        status,
        X,
    )


# ============================================================
# ZeroGPU wrapper
# ============================================================

@spaces.GPU(duration=300)
def run_inference(
    task_choice,
    mri_strategy,
    mri_file,
    core_df,
    extra_csv,
):

    if mri_file is None:
        raise gr.Error(
            "Please upload a whole-volume T1 MRI."
        )

    try:

        (
            stage,
            confidence,
            probabilities,
            status,
            feature_frame,
        ) = _run_multimodal(
            task_choice,
            mri_strategy,
            mri_file,
            core_df,
            extra_csv,
        )

        return (
            _stage_html(
                stage,
                confidence,
                task_choice,
                mri_strategy,
            ),
            _make_prob_chart(
                probabilities
            ),
            status,
            feature_frame,
        )

    except gr.Error:
        raise

    except Exception as error:

        traceback.print_exc()

        raise gr.Error(
            f"Inference failed: {error}"
        )


# ============================================================
# Dynamic UI helper
# ============================================================

def selection_help(
    task_choice,
    mri_strategy,
):

    if mri_strategy == "Hippocampal ROI":

        region_text = (
            "**Automatic ROI:** hippocampus"
        )

    else:

        region_text = (
            "**Automatic ROIs:** hippocampus + "
            "entorhinal cortex + amygdala + ventricles"
        )

    if task_choice == "Progression":

        task_text = (
            "**Task:** predict future disease progression"
        )

    else:

        task_text = (
            "**Task:** predict future development "
            "of Alzheimer’s-related impairment"
        )

    return (
        f"{task_text}  \n"
        "**Input:** whole-volume T1 MRI (`.nii` / `.nii.gz`)  \n"
        f"{region_text}  \n"
        "**Clinical/cognitive data:** used in multimodal fusion"
    )


# ============================================================
# Styling
# ============================================================

CSS = """
.gradio-container {
    max-width: 1500px !important;
    width: 94vw !important;
    margin: 0 auto !important;
}

.hero-card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 22px 26px;
    margin-bottom: 12px;
}

#result-card {
    min-height: 130px;
}

footer {
    visibility: hidden;
}
"""


# ============================================================
# App layout
# ============================================================

with gr.Blocks(
    theme=gr.themes.Soft(),
    css=CSS,
    title="AD-Stage-Net Multimodal",
) as demo:

    # ========================================================
    # Header
    # ========================================================

    with gr.Row():

        with gr.Column():

            gr.Markdown(
                PROJECT_MD,
                elem_classes=[
                    "hero-card"
                ],
            )

    # ========================================================
    # Main layout
    # ========================================================

    with gr.Row(
        equal_height=False
    ):

        # ====================================================
        # LEFT SIDE
        # ====================================================

        with gr.Column(
            scale=4,
            min_width=700,
        ):

            gr.Markdown(
                "## Run a Prediction"
            )

            # --------------------------------------------
            # Controls
            # --------------------------------------------

            task_choice = gr.Dropdown(
                choices=TASK_OPTIONS,
                value="Progression",
                label="Prediction Task",
                allow_custom_value=False,
            )

            mri_strategy = gr.Dropdown(
                choices=MRI_STRATEGY_OPTIONS,
                value="Multi-Region ROI",
                label="MRI Strategy",
                allow_custom_value=False,
            )

            model_note = gr.Markdown(
                selection_help(
                    "Progression",
                    "Multi-Region ROI",
                )
            )

            # --------------------------------------------
            # MRI Upload
            # --------------------------------------------

            gr.Markdown(
                "### 1. Upload MRI"
            )

            mri_file = gr.File(
                label="Upload whole-volume T1 MRI",
                file_types=[
                    ".nii",
                    ".gz",
                ],
                type="filepath",
            )

            gr.Markdown(
                "**Supported:** NIfTI whole-volume MRI "
                "(`.nii`, `.nii.gz`)."
            )

            clear_btn = gr.Button(
                "Clear MRI",
                variant="secondary",
            )

            # --------------------------------------------
            # Clinical / Cognitive
            # --------------------------------------------

            gr.Markdown(
                "### 2. Clinical / Cognitive Information"
            )

            gr.Markdown(
                "Enter the measurements available for the participant. "
                "Leave unavailable values blank."
            )

            core_df = gr.Dataframe(
                headers=[
                    "Feature",
                    "Value",
                ],
                datatype=[
                    "str",
                    "str",
                ],
                value=DEFAULT_CORE_ROWS,
                row_count=(
                    len(DEFAULT_CORE_ROWS),
                    "fixed",
                ),
                col_count=(
                    2,
                    "fixed",
                ),
                interactive=True,
                label="Clinical / cognitive features",
            )

            extra_csv = gr.File(
                label="Optional complete cognitive-feature CSV",
                file_types=[
                    ".csv"
                ],
                type="filepath",
            )

            # --------------------------------------------
            # Run
            # --------------------------------------------

            gr.Markdown(
                "### 3. Run Model"
            )

            run_btn = gr.Button(
                "Run inference",
                variant="primary",
                size="lg",
            )

        # ====================================================
        # RIGHT SIDE
        # ====================================================

        with gr.Column(
            scale=1,
            min_width=300,
        ):

            gr.Markdown(
                """
## How the pipeline works

**Whole-volume T1 MRI**  
↓  
**FastSurfer segmentation**  
↓  
**Automatic ROI extraction**  
↓  
**MRI embedding**  
↓  
**Clinical + cognitive fusion**  
↓  
**Prediction**

### Prediction Tasks

**Progression**  
Predicts future disease progression using information available at the current visit.

**Development**  
Predicts whether the participant will develop future Alzheimer’s-related impairment.

### MRI Strategies

**Multi-Region ROI**  
Hippocampus, entorhinal cortex, amygdala, and ventricles.

**Hippocampal ROI**  
Hippocampus only.
"""
            )

    # ========================================================
    # Results
    # ========================================================

    gr.Markdown("---")

    gr.Markdown(
        "## Prediction Results"
    )

    with gr.Row():

        with gr.Column(
            scale=2
        ):

            stage_output = gr.HTML(
                elem_id="result-card"
            )

        with gr.Column(
            scale=3
        ):

            chart_output = gr.Plot()

    status_output = gr.Markdown()

    with gr.Accordion(
        "Show exact multimodal fusion input",
        open=False,
    ):

        feature_frame = gr.Dataframe(
            interactive=False
        )

    # ========================================================
    # Events
    # ========================================================

    task_choice.change(
        fn=selection_help,
        inputs=[
            task_choice,
            mri_strategy,
        ],
        outputs=model_note,
    )

    mri_strategy.change(
        fn=selection_help,
        inputs=[
            task_choice,
            mri_strategy,
        ],
        outputs=model_note,
    )

    clear_btn.click(
        fn=lambda: None,
        inputs=None,
        outputs=mri_file,
    )

    run_btn.click(
        fn=run_inference,
        inputs=[
            task_choice,
            mri_strategy,
            mri_file,
            core_df,
            extra_csv,
        ],
        outputs=[
            stage_output,
            chart_output,
            status_output,
            feature_frame,
        ],
    )


# ============================================================
# Launch
# ============================================================

if __name__ == "__main__":
    demo.launch()