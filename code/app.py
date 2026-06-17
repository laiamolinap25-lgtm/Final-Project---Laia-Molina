import os
import pandas as pd
import numpy as np
import nibabel as nib
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
from pathlib import Path



# PROJECT PATHS

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
DKT_DIR = DATA_DIR / "PET_parcellated" / "dkt"
ASSETS_DIR = BASE_DIR / "assets"
CODE_DIR = BASE_DIR / "code"

RESULTS_DIR = DKT_DIR / "results analysis"
PNG_DIR = DKT_DIR / "receptor_maps_corticalandsubcortical" / "png"
NIFTI_DIR = DKT_DIR / "receptor_maps_corticalandsubcortical" / "nifti"

RECEPTOR_TABLE_PATH = DKT_DIR / "DKT_receptors_table_corticalandsubcortical.csv"

WORKFLOW_IMG = ASSETS_DIR / "tech-roadmap-svg.jpg"
AD_CONTINUUM_IMG = ASSETS_DIR / "Captura de pantalla 2026-06-11 145900.png"
SYNAPSE_IMG = ASSETS_DIR / "Captura de pantalla 2026-06-11 145958.png"

ADNI_FIGURES_DIR = DKT_DIR / "adni analysis"
HEATMAP_DIR = CODE_DIR / "html_correlation_heatmaps_significant"

ALL_CORRELATIONS_PATH = CODE_DIR / "all_correlations.csv"

FILES = {
    "all_correlations": "all_correlations.csv",
    "significant": "significant_correlations_with_direction.csv",
    "direction_system": "direction_by_neurotransmitter_system.csv",
    "direction_system_biomarker": "direction_by_system_and_biomarker.csv",
    "direction_region": "direction_by_region_type.csv",
    "direction_condition": "direction_by_condition.csv",
    "adni_summary": "adni_global_summary.csv",
    "nt_receptor_summary": "nt_receptor_descriptive_summary.csv",
    "nt_system_table": "nt_systems_receptor_count.csv",
}

COLORS = {
    "petroleum": "#1F4E5F",
    "petroleum_dark": "#143642",
    "scientific_blue": "#2F80ED",
    "white": "#FFFFFF",
    "background": "#F7F9FB",
    "text": "#2D3748",
    "muted": "#4A5568",
    "light_border": "#E2E8F0",
    "terracotta": "#C96A45",
    "sand": "#F4E8E1",
    "card": "#FFFFFF",

    "positive": "#4F8A6D",
    "negative": "#C96A45",
    "neutral": "#718096",

    "con": "#4A5568",
    "mci": "#C99A5B",
    "ad": "#C96A45",
}

DIRECTION_COLORS = {
    "positive": COLORS["positive"],
    "negative": COLORS["negative"],
}


# PAGE CONFIG

st.set_page_config(
    page_title="Neurochemical Pattern Explorer",
    layout="wide",
    initial_sidebar_state="expanded",
)


# CUSTOM CSS


st.markdown(
    """
    <style>
    .stApp {
        background-color: #F7F9FB;
        color: #2D3748;
        font-family: "Inter", "Manrope", "Segoe UI", sans-serif;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1320px;
    }

    h1 {
        color: #1F4E5F !important;
        font-weight: 750 !important;
        letter-spacing: -0.03em;
    }

    h2, h3 {
        color: #1F4E5F !important;
        font-weight: 650 !important;
    }

    p, li, label, span {
        color: #2D3748;
    }

    header[data-testid="stHeader"] {
        background-color: #143642 !important;
    }

    section[data-testid="stSidebar"] {
        background-color: #143642 !important;
    }

    section[data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        color: #FFFFFF !important;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label span {
        color: #FFFFFF !important;
    }

    div[role="radiogroup"] svg {
        fill: #C96A45 !important;
    }

    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        padding: 1rem;
        border-radius: 16px;
        box-shadow: 0px 4px 16px rgba(31, 78, 95, 0.06);
    }

    div[data-testid="stMetric"] label {
        color: #4A5568 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #1F4E5F !important;
        font-family: "JetBrains Mono", monospace;
    }

    .info-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 18px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
        box-shadow: 0px 4px 16px rgba(31, 78, 95, 0.06);
        color: #2D3748;
    }

    .info-card h3 {
        color: #1F4E5F !important;
        margin-top: 0;
    }

    .small-note {
        color: #4A5568;
        font-size: 0.9rem;
    }

    button[data-baseweb="tab"] {
        color: #4A5568 !important;
        background-color: transparent !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #C96A45 !important;
        border-bottom: 3px solid #C96A45 !important;
        font-weight: 650 !important;
    }

    button[data-baseweb="tab"] p {
        color: inherit !important;
    }

    div[role="radiogroup"] label {
        color: #2D3748 !important;
    }

    div[role="radiogroup"] label span {
        color: #2D3748 !important;
    }

    div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        border-color: #E2E8F0 !important;
        color: #2D3748 !important;
        border-radius: 10px !important;
    }

    div[data-baseweb="select"] span {
        color: #2D3748 !important;
    }

    ul[role="listbox"] {
        background-color: #FFFFFF !important;
    }

    ul[role="listbox"] li {
        color: #2D3748 !important;
    }

    div[data-testid="stSlider"] label {
        color: #2D3748 !important;
    }

    div[data-testid="stCheckbox"] label {
        color: #2D3748 !important;
    }

    .stDataFrame {
        background-color: #FFFFFF !important;
    }

    div[data-testid="stCaptionContainer"] {
        color: #4A5568 !important;
    }

    div[data-testid="stAlert"] {
        border-radius: 14px;
        border: 1px solid #E2E8F0;
    }

    .stButton button {
        background-color: #C96A45 !important;
        color: #FFFFFF !important;
        border-radius: 10px !important;
        border: none !important;
        font-weight: 600 !important;
    }

    div[data-testid="stDownloadButton"] button {
        background-color: #C96A45 !important;
        color: #FFFFFF !important;
        border-radius: 10px !important;
        border: none !important;
        font-weight: 600 !important;
    }

    hr {
        border: none;
        border-top: 1px solid #E2E8F0;
        margin: 1.2rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
    

# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_csv(filename):
    path = os.path.join(RESULTS_DIR, filename)

    if not os.path.exists(path):
        return None

    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1")
    
@st.cache_data
def load_receptor_matrix():
    if not os.path.exists(RECEPTOR_TABLE_PATH):
        return None

    try:
        df = pd.read_csv(RECEPTOR_TABLE_PATH)

        if "region" not in df.columns:
            df = pd.read_csv(RECEPTOR_TABLE_PATH, sep=";", decimal=",")

    except Exception:
        df = pd.read_csv(RECEPTOR_TABLE_PATH, sep=";", decimal=",")

    df.columns = df.columns.str.strip()

    if "region" not in df.columns:
        return None

    non_receptor_cols = {"label", "region", "name", "id", "index"}
    receptor_cols = [c for c in df.columns if c not in non_receptor_cols]

    for col in receptor_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", ".", regex=False)
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df
@st.cache_data
def load_csv_from_path(path):
    if not os.path.exists(path):
        return None

    try:
        df = pd.read_csv(path, encoding="utf-8")

        # If the file was read as one single column, try semicolon format
        if len(df.columns) == 1 and ";" in df.columns[0]:
            df = pd.read_csv(path, sep=";", decimal=",", encoding="utf-8")

    except UnicodeDecodeError:
        df = pd.read_csv(path, sep=";", decimal=",", encoding="latin1")

    df.columns = df.columns.str.strip()

    return df

all_correlations = load_csv_from_path(ALL_CORRELATIONS_PATH)
significant = all_correlations[all_correlations["sig_fdr"].astype(str).str.lower().isin(["true", "1", "yes"])].copy() if all_correlations is not None and "sig_fdr" in all_correlations.columns else None
if significant is not None and "rho" in significant.columns and "direction" not in significant.columns: significant["direction"] = np.where(pd.to_numeric(significant["rho"], errors="coerce") >= 0, "positive", "negative")
st.sidebar.write("FDR significant from all_correlations:", len(significant) if significant is not None else "None")

if all_correlations is not None and "sig_fdr" in all_correlations.columns:
    all_correlations["sig_fdr"] = (
        all_correlations["sig_fdr"]
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    significant = all_correlations[all_correlations["sig_fdr"]].copy()

    if "rho" in significant.columns:
        significant["rho"] = pd.to_numeric(significant["rho"], errors="coerce")

    if "direction" not in significant.columns and "rho" in significant.columns:
        significant["direction"] = np.where(
            significant["rho"] >= 0,
            "positive",
            "negative",
        )

else:
    significant = None

direction_system = load_csv(FILES["direction_system"])
direction_system_biomarker = load_csv(FILES["direction_system_biomarker"])
direction_region = load_csv(FILES["direction_region"])
direction_condition = load_csv(FILES["direction_condition"])
adni_summary = load_csv(FILES["adni_summary"])
nt_receptor_summary = load_csv(FILES["nt_receptor_summary"])
nt_system_table = load_csv(FILES["nt_system_table"])
receptor_matrix = load_receptor_matrix()


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def find_heatmap_html(directory, biomarker, region_type):
    if not os.path.exists(directory):
        return None

    html_files = sorted(
        [
            f for f in os.listdir(directory)
            if f.lower().endswith(".html")
        ]
    )

    # Expected pattern from the heatmap script
    expected_names = [
        f"heatmap_{biomarker}_{region_type}_significance.html",
        f"heatmap_significant_{biomarker}_{region_type}.html",
        f"heatmap_{biomarker}_{region_type}.html",
    ]

    for expected in expected_names:
        if expected in html_files:
            return os.path.join(directory, expected)

    # Fallback: biomarker and region type contained in filename
    for file in html_files:
        file_lower = file.lower()
        if biomarker.lower() in file_lower and region_type.lower() in file_lower:
            return os.path.join(directory, file)

    return None

def count_fdr_significant(df):
    if df is None or "sig_fdr" not in df.columns:
        return "NA"

    sig = (
        df["sig_fdr"]
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    return int(sig.sum())

def clean_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.22, color="#CBD5E0")


def show_missing_file_warning(label, filename):
    st.warning(f"Missing file for {label}: `{filename}`")


def safe_dataframe(df):
    if df is None:
        st.warning("File not found or not loaded.")
    else:
        st.dataframe(df, use_container_width=True)


def fallback_receptor_table():
    return pd.DataFrame(
        [
            {
                "System": "Dopamine",
                "Receptors / transporters": "D1, D2, DAT",
                "n_maps": 3,
            },
            {
                "System": "Acetylcholine",
                "Receptors / transporters": "VAChT, M1, A4B2",
                "n_maps": 3,
            },
            {
                "System": "GABA",
                "Receptors / transporters": "GABAa-bz, GABAa",
                "n_maps": 2,
            },
            {
                "System": "Serotonin",
                "Receptors / transporters": "5HT1a, 5HT1b, 5HT2a, 5HT4, 5HT6, 5HTT",
                "n_maps": 6,
            },
            {
                "System": "Glutamate",
                "Receptors / transporters": "mGluR5, NMDA",
                "n_maps": 2,
            },
        ]
    )


def plot_positive_negative(df, index_col, title, xlabel):
    if df is None:
        st.warning("Data not available.")
        return

    required = {index_col, "positive", "negative"}

    if not required.issubset(df.columns):
        st.warning(f"Required columns missing: {required}")
        return

    plot_df = df.set_index(index_col)[["positive", "negative"]]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    plot_df.plot(
        kind="bar",
        ax=ax,
        color=[DIRECTION_COLORS["positive"], DIRECTION_COLORS["negative"]],
        edgecolor="white",
        linewidth=0.8,
    )

    ax.set_title(title, pad=12, color=COLORS["text"])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of FDR-significant associations")
    ax.legend(["Positive", "Negative"], frameon=False)
    clean_axis(ax)

    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    st.pyplot(fig)


def show_metric_row(df):
    if df is None or len(df) == 0:
        col1, col2, col3 = st.columns(3)
        col1.metric("Associations", "0")
        col2.metric("Positive", "0")
        col3.metric("Negative", "0")
        return

    total = len(df)

    if "direction" in df.columns:
        positive = int((df["direction"] == "positive").sum())
        negative = int((df["direction"] == "negative").sum())
    else:
        positive = "NA"
        negative = "NA"

    col1, col2, col3 = st.columns(3)
    col1.metric("Associations", total)
    col2.metric("Positive", positive)
    col3.metric("Negative", negative)
  
def get_receptor_columns(df):
    if df is None:
        return []

    non_receptor_cols = {"label", "region", "name", "id", "index"}
    return [c for c in df.columns if c not in non_receptor_cols]


def find_receptor_file(directory, receptor, extensions):
    if not os.path.exists(directory):
        return None

    files = sorted([
        f for f in os.listdir(directory)
        if f.lower().endswith(extensions)
    ])

    # Expected pattern: receptor_DKT_cortical_subcortical_z.*
    for file in files:
        if file.startswith(f"{receptor}_"):
            return os.path.join(directory, file)

    # Fallback: receptor name contained in filename
    for file in files:
        if receptor.lower() in file.lower():
            return os.path.join(directory, file)

    return None

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("Navigation")
dark_mode = st.sidebar.toggle("Dark mode", value=False)

if dark_mode:
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #0F1F26 !important;
            color: #EAF2F5 !important;
        }

        .block-container {
            background-color: #0F1F26 !important;
        }

        h1, h2, h3 {
            color: #EAF2F5 !important;
        }

        p, li, label, span {
            color: #DDEAF0 !important;
        }

        .info-card {
            background-color: #172E38 !important;
            border: 1px solid #2F4A56 !important;
            color: #EAF2F5 !important;
            box-shadow: 0px 4px 16px rgba(0, 0, 0, 0.25) !important;
        }

        .info-card h3 {
            color: #FFFFFF !important;
        }

        .small-note {
            color: #B8CBD3 !important;
        }

        div[data-testid="stMetric"] {
            background-color: #172E38 !important;
            border: 1px solid #2F4A56 !important;
            box-shadow: 0px 4px 16px rgba(0, 0, 0, 0.25) !important;
        }

        div[data-testid="stMetric"] label {
            color: #B8CBD3 !important;
        }

        div[data-testid="stMetricValue"] {
            color: #FFFFFF !important;
        }

        div[data-baseweb="select"] > div {
            background-color: #172E38 !important;
            border-color: #2F4A56 !important;
            color: #EAF2F5 !important;
        }

        div[data-baseweb="select"] span {
            color: #EAF2F5 !important;
        }

        div[data-testid="stDataFrame"] {
            background-color: #172E38 !important;
        }

        div[data-testid="stAlert"] {
            background-color: #172E38 !important;
            color: #EAF2F5 !important;
            border-color: #2F4A56 !important;
        }

        hr {
            border-top: 1px solid #2F4A56 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
page = st.sidebar.radio(
    "Go to",
    [
        "Home",
        "Understand Alzheimer and neurotransmitters",
        "Understand the data",
        "Density explorer",
        "Correlation explorer"
        
    ],
)




# ============================================================
# PAGE 1 — HOME
# ============================================================


if page == "Home":
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #143642 0%, #1F4E5F 100%); border-radius: 24px; padding: 2.5rem 2.6rem; margin-bottom: 1.5rem; box-shadow: 0px 8px 28px rgba(31,78,95,0.18);">
            <div style="font-size: 0.82rem; color: #F4E8E1; font-weight: 700; margin-bottom: 0.7rem; text-transform: uppercase; letter-spacing: 0.06em;">Dashboard Prototype</div>
            <div style="font-size: 2.25rem; color: #FFFFFF; font-weight: 750; line-height: 1.12; max-width: 1050px; margin-bottom: 1rem;">Dashboard for Alzheimer’s Neurotransmitter and Biomarker Associations</div>
            <div style="font-size: 1.05rem; color: #EAF2F5; max-width: 980px; line-height: 1.6; margin-bottom: 1rem;">A visualization interface for exploring PET-derived neurotransmitter receptor maps, ADNI multi-modal neuroimaging biomarker summaries and FDR-significant receptor–biomarker associations.</div>
            <div style="font-size: 0.9rem; color: #F4E8E1; max-width: 1050px; line-height: 1.5;">Developed as part of the project: Development of a computational pipeline for PET-based neurotransmitter receptor mapping and multi-modal neuroimaging biomarker correlation analysis in Alzheimer’s disease.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    n_sig = count_fdr_significant(all_correlations)

    if receptor_matrix is not None:
        n_receptors = len(get_receptor_columns(receptor_matrix))
    else:
        n_receptors = 16

    col1, col2, col3= st.columns(3)

    col1.metric("Neurotransmitter systems", "5")
    col2.metric("Receptor maps", n_receptors)
    col3.metric("ADNI biomarkers", "4")
  


    st.markdown("")

    st.subheader("What the dashboard connects")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            """
            <div class="info-card">
                <h3>Neurochemical architecture</h3>
                <p>
                PET-derived receptor and transporter maps represent regional
                neurotransmitter density patterns across cortical and subcortical brain areas.
                </p>
                <p class="small-note">
                16 maps · 5 neurotransmitter systems
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="info-card">
                <h3>ADNI biomarkers</h3>
                <p>
                Multimodal neuroimaging biomarkers capture metabolic, structural
                and microstructural patterns related to Alzheimer’s disease.
                </p>
                <p class="small-note">
                FDG · T1/T2 · Volume · Cortical thickness
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


    st.subheader("General pipeline workflow")

    if os.path.exists(WORKFLOW_IMG):
        st.image(WORKFLOW_IMG, use_container_width=True)
        st.caption(
            "General workflow of the computational pipeline, from PET receptor maps and ADNI biomarkers "
            "to spatial correlation analysis and interactive visualization."
        )
    else:
        st.warning(f"Workflow image not found: `{WORKFLOW_IMG}`")

    st.subheader("Scope of the tool")

    left, right = st.columns(2)

    with left:
        st.markdown(
            """
            <div class="info-card">
                <h3>What this dashboard does</h3>
                <ul>
                    <li>Explores FDR-significant receptor–biomarker associations.</li>
                    <li>Summarizes positive and negative spatial patterns.</li>
                    <li>Compares neurotransmitter systems, biomarkers and region types.</li>
                    <li>Supports transparent visual inspection of intermediate data.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            """
            <div class="info-card">
                <h3>What this dashboard does not do</h3>
                <ul>
                    <li>It does not diagnose Alzheimer’s disease.</li>
                    <li>It does not infer causality.</li>
                    <li>It does not estimate patient-specific neurotransmitter loss.</li>
                    <li>It does not replace expert interpretation.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.info(
        "Interpretation note: receptor maps are spatial density maps, not patient-specific longitudinal neurotransmitter measurements."
    )




# ============================================================
# PAGE 2 — UNDERSTAND ALZHEIMER AND NEUROTRANSMITTERS
# ============================================================

if page == "Understand Alzheimer and neurotransmitters":
    st.title("Understand Alzheimer and neurotransmitters")

    st.markdown(
        """
        This section introduces the biological rationale behind the dashboard. 
        Alzheimer's disease is not represented here as a single isolated process, 
        but as a multifactorial disorder involving metabolic dysfunction, structural 
        atrophy, microstructural tissue changes, synaptic dysfunction and neurotransmitter imbalance.
        """
    )

    st.markdown(
        """
        <div class="info-card">
            <h3>Core idea of the project</h3>
            <p>
            Alzheimer's disease-related brain alterations are not uniformly distributed across 
            the brain. Different regions show different levels of metabolic dysfunction, tissue 
            damage and structural atrophy. At the same time, neurotransmitter receptors and 
            transporters also follow region-specific spatial distributions.
            </p>
            <p>
            This dashboard connects both dimensions by comparing ADNI-derived neuroimaging 
            biomarker patterns with PET-derived neurotransmitter receptor density maps at the 
            same anatomical resolution.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Alzheimer's disease as a biological continuum")

    if os.path.exists(AD_CONTINUUM_IMG):
        st.image(
            AD_CONTINUUM_IMG,
            use_container_width=True,
            caption=(
                "Representation of the Alzheimer’s disease continuum, "
                "from early biological alterations to symptomatic disease stages."
            ),
        )
    else:
        st.info("AD continuum image not found in the assets folder.")

    st.subheader("Alzheimer's disease background")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="info-card">
                <h3>Clinical continuum</h3>
                <p>
                Alzheimer's disease can be understood as a continuum. Biological alterations 
                may appear before clear cognitive symptoms, followed by mild cognitive impairment 
                and, in later stages, dementia with relevant loss of autonomy.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="info-card">
                <h3>Biological complexity</h3>
                <p>
                Although amyloid-β plaques and tau tangles are central pathological hallmarks, 
                AD also involves synaptic dysfunction, neuroinflammation, metabolic alterations, 
                vascular dysfunction and neuronal loss.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="info-card">
            <h3>Why neurotransmission matters</h3>
            <p>
            Synaptic dysfunction helps explain how molecular pathology becomes cognitive decline. 
            Neurotransmitter systems regulate communication between neurons, synaptic plasticity, 
            excitation/inhibition balance, memory, attention, motivation, mood and behavioural control.
            </p>
            <p>
            For this reason, neurotransmitter receptor maps can be used as a neurochemical reference 
            to investigate whether AD-related biomarker alterations preferentially affect regions with 
            specific receptor or transporter profiles.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Synaptic dysfunction and neurotransmission")

    if os.path.exists(SYNAPSE_IMG):
        st.image(
            SYNAPSE_IMG,
            use_container_width=True,
            caption=(
                "Schematic representation of synaptic dysfunction and neurotransmitter-related "
                "alterations involved in Alzheimer’s disease."
            ),
        )
    else:
        st.info("Synaptic dysfunction image not found in the assets folder.")

    st.subheader("ADNI biomarkers used in the project")

    biomarker_table = pd.DataFrame(
        [
            {
                "Biomarker": "FDG-PET",
                "What it represents": "Regional glucose metabolism",
                "Role in AD": "Hypometabolism reflects functional and synaptic impairment",
            },
            {
                "Biomarker": "T1w/T2w ratio",
                "What it represents": "MRI-derived tissue contrast",
                "Role in AD": "Proxy of microstructural / myelin-related tissue alteration",
            },
            {
                "Biomarker": "Grey matter volume",
                "What it represents": "Amount of regional grey matter tissue",
                "Role in AD": "Structural marker of atrophy and neuronal loss",
            },
            {
                "Biomarker": "Cortical thickness",
                "What it represents": "Thickness of cortical grey matter",
                "Role in AD": "Sensitive marker of cortical atrophy and disease progression",
            },
        ]
    )

    st.dataframe(biomarker_table, use_container_width=True)

    st.subheader("Neurotransmitter systems considered")

    nt_table = pd.DataFrame(
        [
            {
                "System": "Cholinergic",
                "Markers used": "VAChT, M1, α4β2",
                "Main functions": "Attention, learning, memory encoding, cortical and hippocampal modulation",
                "Relevance in AD": "Basal forebrain degeneration, reduced acetylcholine availability, cholinergic hypothesis",
            },
            {
                "System": "Glutamatergic",
                "Markers used": "NMDA, mGluR5",
                "Main functions": "Excitatory signalling, synaptic plasticity, learning and memory",
                "Relevance in AD": "Excitotoxicity, impaired LTP/LTD balance, synaptic damage, reduced mGluR5 availability",
            },
            {
                "System": "GABAergic",
                "Markers used": "GABA-A, GABA-A BZ",
                "Main functions": "Inhibitory signalling, excitation/inhibition balance, network stability",
                "Relevance in AD": "Potential E/I imbalance, altered inhibitory architecture, network hyperexcitability",
            },
            {
                "System": "Dopaminergic",
                "Markers used": "D1, D2, DAT",
                "Main functions": "Motivation, reward, attention, working memory, executive function",
                "Relevance in AD": "Possible contribution to apathy, executive dysfunction and reward-related deficits",
            },
            {
                "System": "Serotonergic",
                "Markers used": "5-HT1A, 5-HT1B, 5-HT2A, 5-HT4, 5-HT6, 5-HTT",
                "Main functions": "Mood, sleep, attention, memory, behavioural control",
                "Relevance in AD": "Neuropsychiatric symptoms, altered receptor binding and transporter availability",
            },
        ]
    )

    st.dataframe(nt_table, use_container_width=True)

    st.subheader("System-specific interpretation")

    with st.expander("Cholinergic system"):
        st.markdown(
            """
            The cholinergic system is strongly linked to attention, learning and memory. 
            In AD, degeneration of basal forebrain cholinergic neurons and reduced acetylcholine 
            availability are associated with cognitive decline. In this project, VAChT represents 
            a presynaptic marker, while M1 and α4β2 represent postsynaptic cholinergic receptor maps.
            """
        )

    with st.expander("Glutamatergic system"):
        st.markdown(
            """
            The glutamatergic system is the main excitatory neurotransmitter system and is essential 
            for synaptic plasticity. NMDA receptors and mGluR5 are relevant because they are involved 
            in learning and memory mechanisms, but abnormal glutamatergic activation may contribute 
            to calcium overload, excitotoxicity and synaptic damage in AD.
            """
        )

    with st.expander("GABAergic system"):
        st.markdown(
            """
            The GABAergic system provides inhibitory control and helps maintain the balance between 
            excitation and inhibition. In this project, GABA-A and GABA-A benzodiazepine-sensitive maps 
            provide complementary information about inhibitory receptor availability and regional 
            inhibitory architecture.
            """
        )

    with st.expander("Dopaminergic system"):
        st.markdown(
            """
            Dopamine is involved in motivation, reward processing, attention, working memory and 
            executive control. Although it is not usually considered the primary neurotransmitter 
            system in AD, dopaminergic alterations may contribute to symptoms such as apathy, reduced 
            motivation and executive dysfunction.
            """
        )

    with st.expander("Serotonergic system"):
        st.markdown(
            """
            Serotonin modulates mood, sleep, attention, memory and behavioural control. In AD, 
            serotonergic alterations may contribute both to cognitive symptoms and to behavioural 
            and psychological symptoms such as depression, anxiety, agitation or sleep disturbances.
            """
        )

    st.subheader("How to interpret the dashboard")

    st.markdown(
        """
        <div class="info-card">
            <ul>
                <li><b>Receptor density maps</b> are normative PET-derived spatial maps, not patient-specific measurements.</li>
                <li><b>ADNI biomarkers</b> represent regional imaging patterns related to metabolism, tissue structure and atrophy.</li>
                <li><b>Spatial correlations</b> quantify whether both profiles tend to be high or low in similar brain regions.</li>
                <li><b>Significant associations</b> are those that survive the statistical controls applied in the pipeline.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.warning(
        "Important: this dashboard is intended for exploratory visualization of spatial patterns. "
        "It does not diagnose Alzheimer's disease, infer causality, or estimate patient-specific neurotransmitter loss."
    )

# ============================================================
# PAGE 3 — UNDERSTAND THE DATA
# ============================================================

if page == "Understand the data":
    st.title("Understand the data")

    st.markdown(
        """
        This section summarizes the two main data components used in the project: 
        PET-derived neurotransmitter receptor density maps and ADNI-derived neuroimaging biomarkers. 
        The objective is to provide a compact overview of the input data before moving to the 
        interactive density and correlation explorers.
        """
    )

    section = st.radio(
        "Select data component",
        ["Neurotransmitter density", "ADNI data"],
        horizontal=True,
    )

    # ========================================================
    # NEUROTRANSMITTER DENSITY DATA
    # ========================================================

    if section == "Neurotransmitter density":
        st.subheader("Neurotransmitter density data")

        st.markdown(
            """
            PET-derived receptor and transporter maps were transformed into regional z-scored 
            density profiles using the DKT cortical and subcortical atlas. These regional profiles 
            define the neurochemical architecture used later in the receptor–biomarker correlation analysis.
            """
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                """
                <div class="info-card">
                    <h3>Regional receptor matrix</h3>
                    <p>
                    Each receptor or transporter map was reduced to a regional vector. 
                    Rows correspond to DKT brain regions and columns correspond to neurotransmitter 
                    receptor or transporter maps.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                """
                <div class="info-card">
                    <h3>Use in the pipeline</h3>
                    <p>
                    These regional receptor profiles are later spatially correlated with ADNI 
                    biomarker profiles to identify receptor–biomarker associations.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("### Neurotransmitter systems and receptor maps")

        if nt_system_table is not None:
            st.dataframe(nt_system_table, use_container_width=True)
        else:
            st.dataframe(fallback_receptor_table(), use_container_width=True)

        st.markdown("### Receptor density descriptive summary")

        if nt_receptor_summary is not None:
            st.dataframe(nt_receptor_summary, use_container_width=True)
        else:
            st.warning(
                "Receptor density summary table not found. "
                "Expected file: nt_receptor_descriptive_summary.csv"
            )

        st.caption(
            "Density values correspond to z-scored regional receptor density values. "
            "This section summarizes the input neurochemical maps; detailed receptor-by-receptor "
            "inspection is available in the Density explorer."
        )

    # ========================================================
    # ADNI DATA
    # ========================================================

    if section == "ADNI data":
        st.subheader("ADNI biomarker data")

        st.markdown(
            """
            ADNI biomarker data were characterized before the receptor–biomarker association analysis. 
            Baseline analyses included participants with at least one valid visit. Delta analyses required 
            at least two valid timepoints and positive follow-up duration.
            """
        )

        st.markdown("### ADNI exploratory figures")

        st.markdown(
            """
            These figures summarize cohort availability before the receptor–biomarker 
            correlation analysis. They describe the number of valid patients, visit availability 
            and longitudinal follow-up structure for each biomarker and diagnostic group.
            """
        )

        if not os.path.exists(ADNI_FIGURES_DIR):
            st.warning(f"ADNI figures directory not found: `{ADNI_FIGURES_DIR}`")

        else:
            figure_files = sorted(
                [
                    f for f in os.listdir(ADNI_FIGURES_DIR)
                    if f.lower().endswith((".png", ".jpg", ".jpeg"))
                ]
            )

            if len(figure_files) == 0:
                st.warning("No ADNI figures found in the selected folder.")

            else:
                figure_type = st.selectbox(
                    "Select figure type",
                    [
                        "Longitudinal eligibility summary",
                        "Baseline vs delta valid patients",
                        "Timepoint availability",
                        "Visits per patient",
                        "All figures",
                    ],
                )

                if figure_type == "Longitudinal eligibility summary":
                    selected_files = [
                        f for f in figure_files
                        if "delta_valid_percent" in f
                    ]

                elif figure_type == "Baseline vs delta valid patients":
                    selected_files = [
                        f for f in figure_files
                        if "baseline_delta_valid" in f
                    ]

                elif figure_type == "Timepoint availability":
                    selected_files = [
                        f for f in figure_files
                        if "timepoint_availability" in f
                    ]

                elif figure_type == "Visits per patient":
                    selected_files = [
                        f for f in figure_files
                        if "visits_per_patient" in f
                    ]

                else:
                    selected_files = figure_files

                if len(selected_files) == 0:
                    st.info("No figures found for this category.")

                else:
                    selected_fig = st.selectbox(
                        "Select figure",
                        selected_files,
                    )

                    fig_path = os.path.join(ADNI_FIGURES_DIR, selected_fig)

                    st.image(
                        fig_path,
                        use_container_width=True,
                    )

                    st.caption(
                        "Exploratory ADNI cohort figure generated during data availability analysis. "
                        "These figures are used to characterize the input cohort and should not be "
                        "interpreted as primary receptor–biomarker correlation results."
                    )
# ============================================================
# PAGE 4 — DENSITY EXPLORER
# ============================================================

if page == "Density explorer":
    st.title("Density explorer")

    st.markdown(
        """
        This section allows interactive exploration of PET-derived neurotransmitter 
        receptor density maps. For each receptor or transporter, the dashboard displays 
        its regional density values and provides both PNG and NIfTI-based visualization.
        """
    )

    if receptor_matrix is None:
        st.warning(
            "Receptor matrix not found. "
            "Expected file: DKT_receptors_table_corticalandsubcortical.csv"
        )

    else:
        receptor_cols = get_receptor_columns(receptor_matrix)

        if len(receptor_cols) == 0:
            st.warning("No receptor columns found in the receptor matrix.")

        else:
            selected_receptor = st.selectbox(
                "Select receptor / transporter",
                receptor_cols,
                key="density_selected_receptor",
            )

            region_col = "region" if "region" in receptor_matrix.columns else None

            if region_col is None:
                st.warning("Column `region` was not found in the receptor matrix.")

            else:
                density_df = receptor_matrix[[region_col, selected_receptor]].copy()

                density_df = density_df.rename(
                    columns={
                        region_col: "region",
                        selected_receptor: "density_z_score",
                    }
                )

                density_df["density_z_score"] = pd.to_numeric(
                    density_df["density_z_score"],
                    errors="coerce",
                )

                density_df = density_df.dropna(subset=["density_z_score"])

                density_df["region_type"] = np.where(
                    density_df["region"].astype(str).str.startswith("ctx-"),
                    "cortical",
                    "subcortical / other",
                )

            

                st.markdown("---")
                st.subheader("Regional density lookup")

                col_region, col_sort = st.columns([1.2, 1])

                region_options = (
                    density_df["region"]
                    .dropna()
                    .astype(str)
                    .sort_values()
                    .unique()
                    .tolist()
                )

                if len(region_options) == 0:
                    st.warning("No valid brain regions found for this receptor.")
                else:
                    selected_region = st.selectbox(
                        "Select brain region",
                        region_options,
                        key="density_selected_region",
                    )

                    selected_row = density_df[
                        density_df["region"].astype(str) == selected_region
                    ]

                    if not selected_row.empty:
                        selected_density = selected_row["density_z_score"].iloc[0]
                        selected_region_type = selected_row["region_type"].iloc[0]

                        st.markdown(
                            f"""
                            <div class="info-card">
                                <h3>Selected regional density</h3>
                                <p><b>Receptor / transporter:</b> {selected_receptor}</p>
                                <p><b>Region:</b> {selected_region}</p>
                                <p><b>Region type:</b> {selected_region_type}</p>
                                <p><b>Density z-score:</b> {selected_density:.4f}</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                # ------------------------------------------------
                # VISUALIZATION
                # ------------------------------------------------

                st.markdown("---")
                st.subheader("Visualization")

                tab_png, tab_nifti = st.tabs(
                    [
                        "PNG receptor map",
                        "NIfTI viewer",
                    ]
                )

                # --------------------------------------------
                # TAB 1 — PNG MAP
                # --------------------------------------------

                with tab_png:
                    st.markdown("### Pre-rendered receptor density map")

                    png_path = find_receptor_file(
                        PNG_DIR,
                        selected_receptor,
                        extensions=(".png",),
                    )

                    if png_path is None:
                        st.warning(f"No PNG map found for receptor `{selected_receptor}`.")
                    else:
                        st.image(
                            png_path,
                            use_container_width=True,
                        )

                        st.caption(
                            "Pre-rendered receptor density map reconstructed from the regional "
                            "z-scored receptor profile."
                        )

                # --------------------------------------------
                # TAB 2 — NIFTI VIEWER
                # --------------------------------------------

                with tab_nifti:
                    st.markdown("### Interactive NIfTI slice viewer")

                    nifti_path = find_receptor_file(
                        NIFTI_DIR,
                        selected_receptor,
                        extensions=(".nii", ".nii.gz"),
                    )

                    if nifti_path is None:
                        st.warning(f"No NIfTI map found for receptor `{selected_receptor}`.")

                    else:
                        try:
                            img = nib.load(nifti_path)
                            data = np.squeeze(img.get_fdata())

                            if data.ndim != 3:
                                st.warning(
                                    f"This NIfTI file has shape {data.shape}. Expected a 3D image."
                                )

                            else:
                                st.write(f"Image shape: `{data.shape}`")

                                col_controls, col_viewer = st.columns([1, 2])

                                with col_controls:
                                    view = st.radio(
                                        "View",
                                        ["Axial", "Coronal", "Sagittal"],
                                        horizontal=False,
                                        key="density_nifti_view",
                                    )

                                    rotate_k = st.selectbox(
                                        "Rotate display",
                                        [0, 1, 2, 3],
                                        format_func=lambda x: f"{x * 90}°",
                                        key="density_rotate",
                                    )

                                    flip_lr = st.checkbox(
                                        "Flip left-right",
                                        key="density_flip_lr",
                                    )

                                    flip_ud = st.checkbox(
                                        "Flip up-down",
                                        key="density_flip_ud",
                                    )

                                    cmap = st.selectbox(
                                        "Colormap",
                                        ["magma", "viridis", "BrBG", "PiYG", "gray"],
                                        index=0,
                                        key="density_cmap",
                                    )

                                    fig_size = st.slider(
                                        "Figure size",
                                        min_value=3.0,
                                        max_value=7.0,
                                        value=4.5,
                                        step=0.5,
                                        key="density_fig_size",
                                    )

                                    show_colorbar = st.checkbox(
                                        "Show colorbar",
                                        value=True,
                                        key="density_show_colorbar",
                                    )

                                    mask_zeros = st.checkbox(
                                        "Hide zero background",
                                        value=True,
                                        key="density_mask_zeros",
                                    )

                                if view == "Axial":
                                    max_slice = data.shape[2] - 1
                                    slice_idx = st.slider(
                                        "Axial slice",
                                        0,
                                        max_slice,
                                        max_slice // 2,
                                        key="density_axial_slice",
                                    )
                                    slice_data = data[:, :, slice_idx]

                                elif view == "Coronal":
                                    max_slice = data.shape[1] - 1
                                    slice_idx = st.slider(
                                        "Coronal slice",
                                        0,
                                        max_slice,
                                        max_slice // 2,
                                        key="density_coronal_slice",
                                    )
                                    slice_data = data[:, slice_idx, :]

                                else:
                                    max_slice = data.shape[0] - 1
                                    slice_idx = st.slider(
                                        "Sagittal slice",
                                        0,
                                        max_slice,
                                        max_slice // 2,
                                        key="density_sagittal_slice",
                                    )
                                    slice_data = data[slice_idx, :, :]

                                display_data = np.rot90(slice_data, k=rotate_k)

                                if flip_lr:
                                    display_data = np.fliplr(display_data)

                                if flip_ud:
                                    display_data = np.flipud(display_data)

                                display_data = display_data.astype(float)

                                if mask_zeros:
                                    display_data[display_data == 0] = np.nan

                                finite_values = display_data[np.isfinite(display_data)]

                                if finite_values.size > 0:
                                    vmin = np.nanpercentile(finite_values, 5)
                                    vmax = np.nanpercentile(finite_values, 95)
                                else:
                                    vmin, vmax = None, None

                                with col_viewer:
                                    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

                                    im = ax.imshow(
                                        display_data,
                                        cmap=cmap,
                                        vmin=vmin,
                                        vmax=vmax,
                                        interpolation="nearest",
                                    )

                                    ax.axis("off")
                                    ax.set_title(
                                        f"{view} slice {slice_idx}",
                                        pad=8,
                                        fontsize=10,
                                    )

                                    if show_colorbar:
                                        cbar = fig.colorbar(
                                            im,
                                            ax=ax,
                                            fraction=0.035,
                                            pad=0.02,
                                        )
                                        cbar.set_label("Density", fontsize=9)
                                        cbar.ax.tick_params(labelsize=8)

                                    plt.tight_layout()
                                    st.pyplot(fig, use_container_width=False)

                                with open(nifti_path, "rb") as f:
                                    st.download_button(
                                        label="Download selected NIfTI",
                                        data=f,
                                        file_name=os.path.basename(nifti_path),
                                        mime="application/octet-stream",
                                        key="density_download_nifti",
                                    )

                        except Exception as e:
                            st.error(f"Could not load NIfTI file: {e}")

                # ------------------------------------------------
                # REGIONAL TABLE
                # ------------------------------------------------

                st.markdown("---")
                st.subheader("Regional density table")

                col_filter, col_order = st.columns(2)

                with col_filter:
                    region_filter = st.selectbox(
                        "Region type",
                        ["All", "cortical", "subcortical / other"],
                        key="density_region_filter",
                    )

                with col_order:
                    sort_option = st.selectbox(
                        "Sort regions by",
                        [
                            "Highest density",
                            "Lowest density",
                            "Region name",
                        ],
                        key="density_sort_option",
                    )

                filtered_density = density_df.copy()

                if region_filter != "All":
                    filtered_density = filtered_density[
                        filtered_density["region_type"] == region_filter
                    ]

                if sort_option == "Highest density":
                    filtered_density = filtered_density.sort_values(
                        "density_z_score",
                        ascending=False,
                    )

                elif sort_option == "Lowest density":
                    filtered_density = filtered_density.sort_values(
                        "density_z_score",
                        ascending=True,
                    )

                else:
                    filtered_density = filtered_density.sort_values("region")

                st.dataframe(
                    filtered_density,
                    use_container_width=True,
                )

                csv = filtered_density.to_csv(index=False).encode("utf-8")

                st.download_button(
                    "Download receptor regional density table",
                    data=csv,
                    file_name=f"{selected_receptor}_regional_density.csv",
                    mime="text/csv",
                    key="density_download_table",
                )

# ============================================================
# PAGE 5 — CORRELATION EXPLORER
# ============================================================

if page == "Correlation explorer":
    st.title("Correlation explorer")

    st.markdown(
        """
        This section allows exploration of receptor–biomarker spatial associations. 
        The heatmap view provides a compact visual summary, while the filtered table 
        allows detailed inspection of FDR-significant associations.
        """
    )

    view_mode = st.radio(
        "Select correlation view",
        [
            "Interactive heatmaps",
            "Filtered significant associations",
        ],
        horizontal=True,
        key="correlation_view_mode",
    )

    # ========================================================
    # VIEW 1 — INTERACTIVE HEATMAPS
    # ========================================================

    if view_mode == "Interactive heatmaps":
        st.subheader("Interactive significant correlation heatmaps")

        st.markdown(
            """
            Heatmaps summarize correlation results by biomarker, receptor and analysis type. 
            Coloured cells represent FDR-significant associations, while non-significant 
            associations are shown in grey when available in the HTML output.
            """
        )

        if not os.path.exists(HEATMAP_DIR):
            st.warning(f"Heatmap directory not found: `{HEATMAP_DIR}`")

        else:
            html_files = sorted(
                [
                    f for f in os.listdir(HEATMAP_DIR)
                    if f.lower().endswith(".html")
                    and f.lower() != "index.html"
                ]
            )

            if len(html_files) == 0:
                st.warning("No heatmap HTML files found in the heatmap directory.")

            else:
                # Use significant table to define clean selector options when available
                if significant is not None:
                    biomarker_options = sorted(
                        significant["biomarker"].dropna().astype(str).unique().tolist()
                    ) if "biomarker" in significant.columns else []

                    region_options = sorted(
                        significant["region_type"].dropna().astype(str).unique().tolist()
                    ) if "region_type" in significant.columns else []
                else:
                    biomarker_options = []
                    region_options = []

                # Fallback options
                if len(biomarker_options) == 0:
                    biomarker_options = ["fdg", "t1t2", "volume", "thickness"]

                if len(region_options) == 0:
                    region_options = ["cortical", "subcortical"]

                col_biomarker, col_region = st.columns(2)

                with col_biomarker:
                    selected_biomarker = st.selectbox(
                        "Select biomarker",
                        biomarker_options,
                        key="heatmap_biomarker",
                    )

                with col_region:
                    selected_region_type = st.selectbox(
                        "Select region type",
                        region_options,
                        key="heatmap_region_type",
                    )

                heatmap_path = find_heatmap_html(
                    HEATMAP_DIR,
                    selected_biomarker,
                    selected_region_type,
                )

                if heatmap_path is None:
                    st.warning(
                        f"No heatmap found for biomarker `{selected_biomarker}` "
                        f"and region type `{selected_region_type}`."
                    )

                    with st.expander("Available heatmap files"):
                        st.write(html_files)

                else:
                    with open(heatmap_path, "r", encoding="utf-8") as f:
                        html_content = f.read()

                    components.html(
                        html_content,
                        height=800,
                        scrolling=True,
                    )

                    st.caption(
                        f"Loaded heatmap: `{os.path.basename(heatmap_path)}`"
                    )

    # ========================================================
    # VIEW 2 — FILTERED SIGNIFICANT ASSOCIATIONS
    # ========================================================

    if view_mode == "Filtered significant associations":
        st.subheader("Filtered significant associations")

        if significant is None:
             st.warning(
                "No FDR-significant correlations were found in all_correlations.csv, "
                "or all_correlations.csv could not be loaded."
            )

        else:
            st.markdown(
                """
                This table contains receptor–biomarker associations that survived FDR correction. 
                Use the filters to inspect specific biomarkers, neurotransmitter systems, receptors, 
                region types, analysis conditions or correlation directions.
                """
            )

            filtered = significant.copy()

            filter_col1, filter_col2, filter_col3 = st.columns(3)
            filter_col4, filter_col5, filter_col6 = st.columns(3)

            def select_filter(dataframe, column, label, container, key):
                if column not in dataframe.columns:
                    return "All"

                values = ["All"] + sorted(
                    dataframe[column]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )

                with container:
                    return st.selectbox(
                        label,
                        values,
                        key=key,
                    )

            selected_biomarker = select_filter(
                filtered,
                "biomarker",
                "Biomarker",
                filter_col1,
                "corr_filter_biomarker",
            )

            selected_system = select_filter(
                filtered,
                "neurotransmitter",
                "Neurotransmitter system",
                filter_col2,
                "corr_filter_system",
            )

            selected_receptor = select_filter(
                filtered,
                "receptor",
                "Receptor / transporter",
                filter_col3,
                "corr_filter_receptor",
            )

            selected_region = select_filter(
                filtered,
                "region_type",
                "Region type",
                filter_col4,
                "corr_filter_region",
            )

            selected_condition = select_filter(
                filtered,
                "correlation_type",
                "Analysis condition",
                filter_col5,
                "corr_filter_condition",
            )

            selected_direction = select_filter(
                filtered,
                "direction",
                "Direction",
                filter_col6,
                "corr_filter_direction",
            )

            if selected_biomarker != "All":
                filtered = filtered[
                    filtered["biomarker"].astype(str) == selected_biomarker
                ]

            if selected_system != "All":
                filtered = filtered[
                    filtered["neurotransmitter"].astype(str) == selected_system
                ]

            if selected_receptor != "All":
                filtered = filtered[
                    filtered["receptor"].astype(str) == selected_receptor
                ]

            if selected_region != "All":
                filtered = filtered[
                    filtered["region_type"].astype(str) == selected_region
                ]

            if selected_condition != "All":
                filtered = filtered[
                    filtered["correlation_type"].astype(str) == selected_condition
                ]

            if selected_direction != "All":
                filtered = filtered[
                    filtered["direction"].astype(str) == selected_direction
                ]

            show_metric_row(filtered)

            st.markdown("### Results table")

            preferred_cols = [
                "biomarker",
                "correlation_type",
                "neurotransmitter",
                "receptor",
                "region_type",
                "direction",
                "rho",
                "pvalue",
                "p_spin",
                "pvalue_fdr",
                "n",
            ]

            available_cols = [
                col for col in preferred_cols
                if col in filtered.columns
            ]

            if len(available_cols) > 0:
                display_df = filtered[available_cols].copy()
            else:
                display_df = filtered.copy()

            st.dataframe(
                display_df,
                use_container_width=True,
            )

            csv = filtered.to_csv(index=False).encode("utf-8")

            st.download_button(
                "Download filtered significant associations",
                data=csv,
                file_name="filtered_significant_associations.csv",
                mime="text/csv",
                key="corr_download_filtered",
            )

            st.markdown("---")

            st.subheader("Selected association detail")

            if len(filtered) == 0:
                st.info("No significant associations match the selected filters.")

            else:
                if "receptor" in filtered.columns and "biomarker" in filtered.columns:
                    filtered = filtered.reset_index(drop=True)

                    option_labels = []

                    for i, row in filtered.iterrows():
                        biomarker = row.get("biomarker", "NA")
                        receptor = row.get("receptor", "NA")
                        region_type = row.get("region_type", "NA")
                        condition = row.get("correlation_type", "NA")
                        rho = row.get("rho", np.nan)

                        try:
                            rho_text = f"{float(rho):.3f}"
                        except Exception:
                            rho_text = "NA"

                        option_labels.append(
                            f"{i + 1}. {biomarker} | {receptor} | {region_type} | {condition} | rho={rho_text}"
                        )

                    selected_label = st.selectbox(
                        "Select association",
                        option_labels,
                        key="corr_selected_association",
                    )

                    selected_idx = option_labels.index(selected_label)
                    selected_row = filtered.iloc[selected_idx]

                    detail_col1, detail_col2, detail_col3, detail_col4 = st.columns(4)

                    detail_col1.metric(
                        "Spearman rho",
                        f"{float(selected_row['rho']):.3f}"
                        if "rho" in selected_row and pd.notna(selected_row["rho"])
                        else "NA",
                    )

                    detail_col2.metric(
                        "p-value",
                        f"{float(selected_row['pvalue']):.3g}"
                        if "pvalue" in selected_row and pd.notna(selected_row["pvalue"])
                        else "NA",
                    )

                    detail_col3.metric(
                        "p-spin",
                        f"{float(selected_row['p_spin']):.3g}"
                        if "p_spin" in selected_row and pd.notna(selected_row["p_spin"])
                        else "NA",
                    )

                    detail_col4.metric(
                        "q-FDR",
                        f"{float(selected_row['pvalue_fdr']):.3g}"
                        if "pvalue_fdr" in selected_row and pd.notna(selected_row["pvalue_fdr"])
                        else "NA",
                    )

                    st.markdown(
                        f"""
                        <div class="info-card">
                            <h3>Association summary</h3>
                            <p><b>Biomarker:</b> {selected_row.get("biomarker", "NA")}</p>
                            <p><b>Analysis condition:</b> {selected_row.get("correlation_type", "NA")}</p>
                            <p><b>Neurotransmitter system:</b> {selected_row.get("neurotransmitter", "NA")}</p>
                            <p><b>Receptor / transporter:</b> {selected_row.get("receptor", "NA")}</p>
                            <p><b>Region type:</b> {selected_row.get("region_type", "NA")}</p>
                            <p><b>Direction:</b> {selected_row.get("direction", "NA")}</p>
                            <p><b>Number of regions:</b> {selected_row.get("n", "NA")}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
