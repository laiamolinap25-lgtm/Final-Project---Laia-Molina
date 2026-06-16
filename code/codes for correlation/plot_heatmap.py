# =============================================================================
# plot_heatmap.py
#
# Generates categorical correlation heatmaps and uses a local Ollama LLM
# to create exploratory scientific interpretations.
#
# Dependencies:  pip install pandas plotly requests
# Ollama model:  ollama pull qwen2.5:7b
# Usage:         python plot_heatmap.py
# =============================================================================

import os
import pandas as pd
import plotly.express as px
import requests


# =============================================================================
# CONFIGURATION
# =============================================================================

CSV_PATH = "all_correlations.csv"
GLOBAL_REPORT = "llm_heatmap_report.md"

MODEL = "qwen2.5:7b"
OLLAMA_URL = "http://localhost:11434/api/generate"

RHO_THRESHOLD = 0.50
FALLBACK_TOP_N = 3

BIOMARKER_LABELS = {
    "fdg": "FDG-PET (cerebral glucose metabolism)",
    "thickness": "Cortical thickness (proxy for cortical atrophy)",
    "volume": "Grey matter volume",
    "t1t2": "T1w/T2w ratio (myelin / tissue integrity)",
}

ANALYSIS_LABELS = {
    "baseline_all": "cross-sectional, all participants",
    "baseline_declining": "cross-sectional, clinically declining participants",
    "delta_all": "longitudinal rate of change, all participants",
    "delta_declining": "longitudinal rate of change, clinically declining participants",
}

COLOR_SCALE = [
    [0.00, "#253a6b"],
    [0.25, "#5c626f"],
    [0.50, "#f0f0f0"],
    [0.75, "#da8b6c"],
    [1.00, "#de5f2d"],
]

CATEGORY_LABELS = {
    -2: "High negative  (ρ < −0.5)",
    -1: "Moderate negative  (−0.5 ≤ ρ < −0.2)",
    0: "Negligible  (|ρ| < 0.2)",
    1: "Moderate positive  (0.2 < ρ ≤ 0.5)",
    2: "High positive  (ρ > 0.5)",
}

CORR_ORDER = [
    "baseline_all",
    "baseline_declining",
    "delta_all",
    "delta_declining",
]

REQUIRED_COLUMNS = {
    "biomarker",
    "correlation_type",
    "neurotransmitter",
    "receptor",
    "region_type",
    "rho",
}


# =============================================================================
# 1. CORRELATION CATEGORISATION
# =============================================================================

def categorise_rho(rho: float) -> int:
    if rho > 0.5:
        return 2
    if rho > 0.2:
        return 1
    if rho >= -0.2:
        return 0
    if rho >= -0.5:
        return -1
    return -2


# =============================================================================
# 2. DATA LOADING
# =============================================================================

def load_correlations(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Correlation table '{csv_path}' not found. "
            "Run the correlation pipeline first."
        )

    df = pd.read_csv(csv_path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["rho"] = pd.to_numeric(df["rho"], errors="coerce")

    if df["rho"].isna().any():
        raise ValueError("Column 'rho' contains missing or non-numeric values.")

    df["abs_rho"] = df["rho"].abs()
    df["category"] = df["rho"].apply(categorise_rho)

    return df


# =============================================================================
# 3. HEATMAP GENERATION
# =============================================================================

def generate_heatmap(
    df_subset: pd.DataFrame,
    biomarker: str,
    region: str,
) -> str | None:

    data = df_subset[df_subset["region_type"] == region]

    if data.empty:
        return None

    pivot = data.pivot_table(
        index="receptor",
        columns="correlation_type",
        values="category",
        aggfunc="first",
    )

    pivot = pivot[[c for c in CORR_ORDER if c in pivot.columns]]

    label = BIOMARKER_LABELS.get(biomarker, biomarker.upper())

    fig = px.imshow(
        pivot,
        labels=dict(x="Analysis type", y="Receptor"),
        color_continuous_scale=COLOR_SCALE,
        range_color=[-2, 2],
        title=f"{label} — {region.capitalize()}",
    )

    fig.update_layout(
        coloraxis_colorbar=dict(
            tickvals=list(CATEGORY_LABELS.keys()),
            ticktext=list(CATEGORY_LABELS.values()),
        ),
        width=700,
        height=800,
    )

    output_path = f"heatmap_{biomarker}_{region}.html"
    fig.write_html(output_path)

    print(f"    Heatmap saved → '{output_path}'")

    return output_path


# =============================================================================
# 4. PROMPT CONSTRUCTION
# =============================================================================

def build_prompt(df_subset: pd.DataFrame, biomarker: str, region: str) -> str:
    label = BIOMARKER_LABELS.get(biomarker, biomarker.upper())
    data = df_subset[df_subset["region_type"] == region].copy()

    strong = data[data["abs_rho"] >= RHO_THRESHOLD]

    if strong.empty:
        strong = data.nlargest(FALLBACK_TOP_N, "abs_rho")
        note = (
            f"no association exceeded |ρ| = {RHO_THRESHOLD}; "
            f"top-{FALLBACK_TOP_N} shown"
        )
    else:
        note = f"associations with |ρ| ≥ {RHO_THRESHOLD}"

    lines = [f"  ({note})"]

    for _, row in strong.sort_values("abs_rho", ascending=False).iterrows():
        direction = "positive" if row["rho"] > 0 else "negative"
        analysis = ANALYSIS_LABELS.get(
            row["correlation_type"],
            row["correlation_type"],
        )

        lines.append(
            f"  · Receptor {row['receptor']} | "
            f"system: {row['neurotransmitter']} | "
            f"ρ = {row['rho']:+.3f} ({direction}) | "
            f"analysis: {analysis}"
        )

    findings = "\n".join(lines)

    prompt = f"""
You are an expert in computational neuroscience and Alzheimer's disease research.

CONTEXT
-------
The following results come from a biomedical engineering thesis investigating
the spatial relationship between regional neurotransmitter receptor density and
neuroimaging biomarkers in the ADNI cohort.

Receptor density maps were derived from a publicly available PET atlas and
parcellated with the Desikan-Killiany-Tourville cortical atlas.

Spearman rank correlations were computed between mean regional biomarker values
and receptor density across brain regions.

Analysis types:
- baseline_all: cross-sectional associations, all participants.
- baseline_declining: cross-sectional associations, clinically declining participants.
- delta_all: longitudinal annualised rate of change, all participants.
- delta_declining: longitudinal annualised rate of change, clinically declining participants.

BIOMARKER: {label}
REGION: {region.upper()}

STRONGEST ASSOCIATIONS
----------------------
{findings}

TASK
----
Write one concise scientific paragraph of 4–6 sentences in the style of a
peer-reviewed neuroscience Results or Discussion section.

The paragraph must:
1. Name the receptor system with the strongest association, including ρ and direction.
2. State whether the association is stronger cross-sectionally or longitudinally.
3. State whether the pattern is specific to clinically declining participants or observed across the full cohort.
4. Provide a cautious neurobiological interpretation grounded in Alzheimer's disease literature.

IMPORTANT CONSTRAINTS
---------------------
- Do not describe associations as statistically significant unless corrected p-values are provided.
- Treat these findings as exploratory and hypothesis-generating.
- Avoid causal language.
- Do not overstate biological interpretation.
- Use cautious wording such as "may suggest", "is compatible with", or "could reflect".
- Do not use bullet points.
- Do not add a heading.
- Write in English.
""".strip()

    return prompt


# =============================================================================
# 5. LOCAL OLLAMA CALL
# =============================================================================

def query_ollama(prompt: str) -> str:
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                },
            },
            timeout=900,
        )

        response.raise_for_status()
        data = response.json()

        return data["response"].strip()

    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            "Could not connect to Ollama. "
            "Make sure Ollama is running before executing this script."
        ) from exc


# =============================================================================
# 6. MAIN PIPELINE
# =============================================================================

def generate_heatmaps_with_llm(csv_path: str = CSV_PATH) -> None:
    df = load_correlations(csv_path)

    report_lines = [
        "# LLM Interpretation Report — Neurotransmitter Receptor × ADNI Biomarker Correlations\n",
        f"> Generated automatically using local Ollama model `{MODEL}`.\n",
        "> All interpretations are exploratory and must be reviewed by domain experts.\n",
        "---\n",
    ]

    for biomarker, df_bm in df.groupby("biomarker"):
        label = BIOMARKER_LABELS.get(biomarker, biomarker.upper())

        print(f"\n{'=' * 60}")
        print(f"  Biomarker: {label}")
        print(f"{'=' * 60}")

        report_lines.append(f"\n## {label}\n")

        for region in ["cortical", "subcortical"]:
            html_path = generate_heatmap(df_bm, biomarker, region)

            if html_path is None:
                print(f"    No data for region: {region} — skipping.")
                continue

            print(f"    Querying local Ollama model for {region} interpretation...")
            prompt = build_prompt(df_bm, biomarker, region)
            interpretation = query_ollama(prompt)

            md_path = html_path.replace(".html", "_interpretation.md")

            with open(md_path, "w", encoding="utf-8") as fh:
                fh.write(f"# {label} — {region.capitalize()}\n\n")
                fh.write(interpretation + "\n")

            print(f"    Interpretation saved → '{md_path}'")

            report_lines.append(f"### {region.capitalize()}\n")
            report_lines.append(interpretation + "\n")

    with open(GLOBAL_REPORT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(report_lines))

    print(f"\n{'=' * 60}")
    print(f"  Global report saved → '{GLOBAL_REPORT}'")
    print(f"{'=' * 60}\nDone.\n")


if __name__ == "__main__":
    generate_heatmaps_with_llm()