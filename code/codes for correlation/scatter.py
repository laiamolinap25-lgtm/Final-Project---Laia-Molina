import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from config import biomarkers
from analysis import (
    read_file,
    get_cols,
    calc_delta,
    calc_delta_declining,
    calc_baseline,
    normalize_ctx_index,
    normalize_sub_index,
    load_receptors,
)

# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = r"C:\Users\laiam\OneDrive\Escritorio\Practicas Canada\neurotransmitters\code\scatter_plots_main_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALL_CORR_PATH = r"C:\Users\laiam\OneDrive\Escritorio\Practicas Canada\neurotransmitters\code\all_correlations.csv"

# ============================================================
# CASES TO PLOT
# ============================================================

CASES = [
    {
        "label": "a",
        "biomarker": "t1t2",
        "correlation_type": "baseline_all",
        "receptor": "5HTT",
        "region_type": "subcortical",
        "title": "5HTT vs T1/T2 (subcortical, baseline all)"
    },
    {
        "label": "b",
        "biomarker": "volume",
        "correlation_type": "baseline_all",
        "receptor": "D2",
        "region_type": "subcortical",
        "title": "D2 vs volume (subcortical, baseline all)"
    },
    {
        "label": "c",
        "biomarker": "fdg",
        "correlation_type": "baseline_all",
        "receptor": "5HT1a",
        "region_type": "cortical",
        "title": "5HT1a vs FDG (cortical, baseline all)"
    },
    {
        "label": "d",
        "biomarker": "thickness",
        "correlation_type": "delta_all",
        "receptor": "5HT1a",
        "region_type": "cortical",
        "title": "5HT1a vs thickness (cortical, delta all)"
    },
]

# ============================================================
# HELPERS
# ============================================================

def load_all_correlations(path):
    if not os.path.exists(path):
        return None

    try:
        df = pd.read_csv(path)
        if len(df.columns) == 1 and ";" in df.columns[0]:
            df = pd.read_csv(path, sep=";", decimal=",")
    except Exception:
        df = pd.read_csv(path, sep=";", decimal=",")

    df.columns = df.columns.str.strip()
    return df


def get_stat_row(all_corr, biomarker, correlation_type, receptor, region_type):
    if all_corr is None:
        return None

    tmp = all_corr.copy()
    for col in ["biomarker", "correlation_type", "receptor", "region_type"]:
        if col in tmp.columns:
            tmp[col] = tmp[col].astype(str).str.strip()

    row = tmp[
        (tmp["biomarker"] == biomarker) &
        (tmp["correlation_type"] == correlation_type) &
        (tmp["receptor"] == receptor) &
        (tmp["region_type"] == region_type)
    ]

    if len(row) == 0:
        return None

    return row.iloc[0]


def build_biomarker_profile(biomarker_name, correlation_type, region_type):
    """
    Returns:
        mean_profile : pd.Series
        x_label      : str
        ctx_mode     : str
    """
    cfg = biomarkers[biomarker_name]
    fmt = cfg["file_format"]

    df_con = read_file(cfg["file_con"], fmt)
    df_mci = read_file(cfg["file_mci"], fmt)
    df_ad  = read_file(cfg["file_ad"], fmt)

    cols_ctx, cols_sub = get_cols(df_con, cfg)

    if region_type == "cortical":
        cols = cols_ctx
    else:
        cols = cols_sub

    if correlation_type == "delta_all":
        prof = pd.concat([calc_delta(d, cols) for d in [df_con, df_mci, df_ad]])
        x_label = cfg["x_axis_delta"]

    elif correlation_type == "delta_declining":
        dx_col = cfg.get("declining_dx_col", "dementia_dx")

        con_dec = df_con[
            (df_con["dementia_dx_bl"] == "CON") &
            (df_con[dx_col].isin(["MCI", "AD"]))
        ]
        mci_dec = df_mci[
            (df_mci["dementia_dx_bl"] == "MCI") &
            (df_mci[dx_col] == "AD")
        ]

        prof = pd.concat([
            calc_delta_declining(con_dec, df_con, cols),
            calc_delta_declining(mci_dec, df_mci, cols)
        ])
        x_label = cfg["x_axis_delta"]

    elif correlation_type == "baseline_all":
        prof = pd.concat([calc_baseline(d, cols) for d in [df_con, df_mci, df_ad]])
        x_label = cfg["x_axis_base"]

    elif correlation_type == "baseline_declining":
        dx_col = cfg.get("declining_dx_col", "dementia_dx")

        con_dec = df_con[
            (df_con["dementia_dx_bl"] == "CON") &
            (df_con[dx_col].isin(["MCI", "AD"]))
        ]
        mci_dec = df_mci[
            (df_mci["dementia_dx_bl"] == "MCI") &
            (df_mci[dx_col] == "AD")
        ]

        prof = pd.concat([
            calc_baseline(con_dec, cols),
            calc_baseline(mci_dec, cols)
        ])
        x_label = cfg["x_axis_base"]

    else:
        raise ValueError(f"Unknown correlation_type: {correlation_type}")

    mean_profile = prof.mean()

    if region_type == "cortical":
        mean_profile = normalize_ctx_index(mean_profile, cfg)
    else:
        mean_profile = normalize_sub_index(mean_profile)

    return mean_profile, x_label, cfg["ctx_index_mode"]


def get_scatter_data(biomarker_name, correlation_type, receptor, region_type):
    mean_profile, x_label, ctx_index_mode = build_biomarker_profile(
        biomarker_name, correlation_type, region_type
    )

    df_rec_ctx, df_rec_sub = load_receptors(ctx_index_mode)

    if region_type == "cortical":
        df_rec = df_rec_ctx
    else:
        df_rec = df_rec_sub

    common = df_rec.index.intersection(mean_profile.index)

    x = mean_profile.loc[common].astype(float)
    y = df_rec.loc[common, receptor].astype(float)

    mask = x.notna() & y.notna()
    x = x[mask]
    y = y[mask]
    regions = x.index

    return x, y, regions, x_label


# ============================================================
# LOAD ALL CORRELATIONS
# ============================================================

all_corr = load_all_correlations(ALL_CORR_PATH)

# ============================================================
# PLOT 2x2 FIGURE
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

for ax, case in zip(axes, CASES):
    biomarker = case["biomarker"]
    correlation_type = case["correlation_type"]
    receptor = case["receptor"]
    region_type = case["region_type"]

    x, y, regions, x_label = get_scatter_data(
        biomarker_name=biomarker,
        correlation_type=correlation_type,
        receptor=receptor,
        region_type=region_type,
    )

    # Statistics from actual plotted data
    rho, pval = stats.spearmanr(x, y)

    # Optional: recover q-value from all_correlations.csv
    stat_row = get_stat_row(all_corr, biomarker, correlation_type, receptor, region_type)
    if stat_row is not None and "pvalue_fdr" in stat_row.index:
        qval = stat_row["pvalue_fdr"]
    else:
        qval = np.nan

    # Scatter
    ax.scatter(x, y, s=55, alpha=0.8)

    # Trend line (visual only)
    if len(x) >= 2:
        slope, intercept = np.polyfit(x, y, 1)
        xx = np.linspace(x.min(), x.max(), 100)
        yy = slope * xx + intercept
        ax.plot(xx, yy, linewidth=2)

    ax.set_xlabel(x_label)
    ax.set_ylabel(f"{receptor} density (z-score)")

    title = f"({case['label']}) {case['title']}"
    ax.set_title(title, fontsize=11)

    stat_text = f"rho = {rho:.3f}\np = {pval:.3g}"
    if pd.notna(qval):
        stat_text += f"\nq = {float(qval):.3g}"

    ax.text(
        0.03, 0.97,
        stat_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", alpha=0.15)
    )

    ax.grid(True, alpha=0.25)

plt.tight_layout()
combined_path = os.path.join(OUTPUT_DIR, "main_4_scatterplots.png")
plt.savefig(combined_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved combined figure to:\n{combined_path}")

# ============================================================
# OPTIONAL: SAVE EACH PANEL SEPARATELY
# ============================================================

for case in CASES:
    biomarker = case["biomarker"]
    correlation_type = case["correlation_type"]
    receptor = case["receptor"]
    region_type = case["region_type"]

    x, y, regions, x_label = get_scatter_data(
        biomarker_name=biomarker,
        correlation_type=correlation_type,
        receptor=receptor,
        region_type=region_type,
    )

    rho, pval = stats.spearmanr(x, y)

    stat_row = get_stat_row(all_corr, biomarker, correlation_type, receptor, region_type)
    if stat_row is not None and "pvalue_fdr" in stat_row.index:
        qval = stat_row["pvalue_fdr"]
    else:
        qval = np.nan

    plt.figure(figsize=(6, 5))
    plt.scatter(x, y, s=60, alpha=0.8)

    if len(x) >= 2:
        slope, intercept = np.polyfit(x, y, 1)
        xx = np.linspace(x.min(), x.max(), 100)
        yy = slope * xx + intercept
        plt.plot(xx, yy, linewidth=2)

    plt.xlabel(x_label)
    plt.ylabel(f"{receptor} density (z-score)")
    plt.title(case["title"])

    stat_text = f"rho = {rho:.3f}\np = {pval:.3g}"
    if pd.notna(qval):
        stat_text += f"\nq = {float(qval):.3g}"

    plt.text(
        0.03, 0.97,
        stat_text,
        transform=plt.gca().transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", alpha=0.15)
    )

    plt.grid(True, alpha=0.25)
    plt.tight_layout()

    out_name = f"{biomarker}_{correlation_type}_{receptor}_{region_type}.png"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out_path}")