"""
analysis.py — Lógica central compartida por todos los biomarcadores.
No necesitas editar este archivo.
"""

import os
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import (
    TIMEPOINT_TO_MONTHS, SUBCORTICAL_COLS,
    COLOR_CORTICAL, COLOR_SUBCORTICAL, COLOR_TRENDLINE,
    RECEPTORS_CSV
)


# ─── CARGA DE DATOS ──────────────────────────────────────────────────────────

def read_file(path: str, fmt: str) -> pd.DataFrame:
    """Lee CSV o XLSX según el formato especificado en config."""
    if fmt == "xlsx":
        return pd.read_excel(path)
    return pd.read_csv(path)


def get_cols(df: pd.DataFrame, cfg: dict) -> tuple:
    """Detecta columnas corticales y subcorticales usando las funciones de config."""
    cols_ctx = cfg["ctx_col_filter"](df.columns.tolist())
    cols_sub = [c for c in SUBCORTICAL_COLS if c in df.columns] if cfg["has_subcortical"] else []
    return cols_ctx, cols_sub


# ─── CÁLCULO DE MÉTRICAS ─────────────────────────────────────────────────────

def calc_delta(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Δ anualizado = (último - primero) / años."""
    df = df.copy()
    df['tp_months'] = df['timepoint'].map(TIMEPOINT_TO_MONTHS)
    df = df.sort_values('tp_months')
    df_bl   = df.groupby('patient_id').first()[cols + ['tp_months']]
    df_last = df.groupby('patient_id').last()[cols + ['tp_months']]
    months_diff = (df_last['tp_months'] - df_bl['tp_months']).replace(0, float('nan'))
    years_diff  = months_diff / 12
    return (df_last[cols] - df_bl[cols]).div(years_diff, axis=0)


def calc_delta_declining(df_decline: pd.DataFrame, df_all: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Δ anualizado solo para pacientes que empeoran."""
    ids = df_decline['patient_id'].unique()
    df_all = df_all[df_all['patient_id'].isin(ids)].copy()
    df_all['tp_months'] = df_all['timepoint'].map(TIMEPOINT_TO_MONTHS)
    df_all = df_all.sort_values('tp_months')
    df_bl   = df_all.groupby('patient_id').first()[cols + ['tp_months']]
    df_last = df_all.groupby('patient_id').last()[cols + ['tp_months']]
    months_diff = (df_last['tp_months'] - df_bl['tp_months']).replace(0, float('nan'))
    years_diff  = months_diff / 12
    return (df_last[cols] - df_bl[cols]).div(years_diff, axis=0)


def calc_baseline(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Valor en baseline (primer timepoint)."""
    df = df.copy()
    df['tp_months'] = df['timepoint'].map(TIMEPOINT_TO_MONTHS)
    df = df.sort_values('tp_months')
    return df.groupby('patient_id').first()[cols]


# ─── NORMALIZACIÓN DE ÍNDICES ─────────────────────────────────────────────────

def normalize_ctx_index(mean_ctx: pd.Series, cfg: dict) -> pd.Series:
    """Aplica la transformación de índice cortical definida en config."""
    mean_ctx = mean_ctx.copy()
    mean_ctx.index = cfg["ctx_index_transform"](mean_ctx.index)
    return mean_ctx


def normalize_sub_index(mean_sub: pd.Series) -> pd.Series:
    """Normaliza el índice subcortical para coincidir con el CSV de receptores."""
    mean_sub = mean_sub.copy()
    mean_sub.index = mean_sub.index.str.replace('-', ' ').str.title().str.replace(' ', '-')
    mean_sub.index = mean_sub.index.map(lambda x: {
        'Left-Accumbens-Area':  'Left-Accumbens-area',
        'Right-Accumbens-Area': 'Right-Accumbens-area',
        'Left-Ventraldc':       'Left-VentralDC',
        'Right-Ventraldc':      'Right-VentralDC',
    }.get(x, x))
    return mean_sub


def load_receptors(ctx_index_mode: str) -> tuple:
    """
    Carga el CSV de receptores separado en cortical y subcortical.

    ctx_index_mode:
      "standard"   → index = columna 'region'  [FDG, T1T2, Thickness]
      "region_key" → quita 'ctx-', reemplaza '-' por '_'  [Volume]
    """
    df_rec = pd.read_csv(RECEPTORS_CSV)

    if ctx_index_mode == "region_key":
        df_rec_ctx = df_rec[df_rec['region'].str.startswith('ctx-', na=False)].copy()
        df_rec_ctx['region_key'] = (
            df_rec_ctx['region']
            .str.replace('ctx-', '', regex=False)
            .str.replace('-', '_', regex=False)
        )
        df_rec_ctx = df_rec_ctx.set_index('region_key')
    else:
        df_rec_ctx = df_rec[df_rec['region'].str.startswith('ctx-', na=False)].set_index('region')

    df_rec_sub = (
        df_rec[~df_rec['region'].str.startswith('ctx-', na=False)]
        .dropna(subset=['region'])
        .set_index('region')
    )
    return df_rec_ctx, df_rec_sub


# ─── GENERACIÓN DE FIGURA ─────────────────────────────────────────────────────

def make_figure(
    mean_ctx, mean_sub,
    df_rec_ctx, df_rec_sub,
    common_ctx, common_sub,
    receptor_cols: list,
    x_label: str,
    title: str,
    has_subcortical: bool,
) -> go.Figure:
    """Genera la figura Plotly con scatter + trendline."""

    n_cols = len(receptor_cols)
    n_rows = 2 if has_subcortical else 1

    if has_subcortical:
        subplot_titles = (
            [f'{r} - Cortical' for r in receptor_cols] +
            [f'{r} - Subcortical' for r in receptor_cols]
        )
        datasets = [
            (1, mean_ctx, df_rec_ctx, common_ctx, 'Cortical',    COLOR_CORTICAL),
            (2, mean_sub, df_rec_sub, common_sub, 'Subcortical', COLOR_SUBCORTICAL),
        ]
        height = 800
    else:
        subplot_titles = [f'{r} - Cortical' for r in receptor_cols]
        datasets = [
            (1, mean_ctx, df_rec_ctx, common_ctx, 'Cortical', COLOR_CORTICAL),
        ]
        height = 450

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        shared_xaxes='rows',
        shared_yaxes='rows',
        subplot_titles=subplot_titles
    )

    for row, mean, df_rec_plot, common, label, color in datasets:
        for col_idx, receptor in enumerate(receptor_cols, start=1):
            x = mean.loc[common].astype(float)
            y = df_rec_plot.loc[common, receptor].astype(float)
            mask = x.notna() & y.notna()
            x, y = x[mask], y[mask]
            region_names = common[mask]

            rho, _ = stats.spearmanr(x, y)

            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode='markers',
                marker=dict(color=color, size=7, opacity=0.7),
                text=region_names,
                hovertemplate=f'<b>%{{text}}</b><br>{x_label}: %{{x:.4f}}<br>Receptor: %{{y:.4f}}<extra></extra>',
                name=f'{label} (n={mask.sum()})',
                showlegend=(col_idx == 1)
            ), row=row, col=col_idx)

            if len(x) >= 2:
                m, b = np.polyfit(x, y, 1)
                x_sorted = np.sort(x)
                fig.add_trace(go.Scatter(
                    x=x_sorted, y=m * x_sorted + b,
                    mode='lines',
                    line=dict(color=COLOR_TRENDLINE, width=1.5),
                    showlegend=False,
                    hoverinfo='skip'
                ), row=row, col=col_idx)

            fig.update_xaxes(title_text=x_label, row=row, col=col_idx)
            fig.update_yaxes(title_text='density z-score', row=row, col=col_idx)

            ann_idx = (row - 1) * n_cols + col_idx
            fig.layout.annotations[ann_idx - 1].text = f'{receptor} - {label}  ρ={rho:.3f}'

    width = max(900, n_cols * 400)
    fig.update_layout(title=title, height=height, width=width, template='plotly_white')
    return fig


# ─── FUNCIÓN PRINCIPAL ────────────────────────────────────────────────────────

def run_analysis(
    biomarker_name: str,
    biomarker_cfg: dict,
    neurotransmitter_name: str,
    receptor_cols: list,
    correlation_type: str,
    output_dir: str = "."
):
    """
    Ejecuta el análisis completo para una combinación de:
      biomarcador × neurotransmisor × tipo de correlación

    correlation_type: "delta_all" | "delta_declining" | "baseline_all" | "baseline_declining"
    """

    print(f"\n{'='*60}")
    print(f"  Biomarcador : {biomarker_name.upper()}")
    print(f"  Neurotr.    : {neurotransmitter_name}")
    print(f"  Correlación : {correlation_type}")
    print(f"{'='*60}")

    fmt    = biomarker_cfg["file_format"]
    df_con = read_file(biomarker_cfg["file_con"], fmt)
    df_mci = read_file(biomarker_cfg["file_mci"], fmt)
    df_ad  = read_file(biomarker_cfg["file_ad"],  fmt)

    cols_ctx, cols_sub = get_cols(df_con, biomarker_cfg)
    has_sub = biomarker_cfg["has_subcortical"]

    # ── Calcula métrica ──────────────────────────────────────────────────────
    if correlation_type == "delta_all":
        delta_ctx = pd.concat([calc_delta(d, cols_ctx) for d in [df_con, df_mci, df_ad]])
        delta_sub = pd.concat([calc_delta(d, cols_sub) for d in [df_con, df_mci, df_ad]]) if has_sub else None
        x_label, population = biomarker_cfg["x_axis_delta"], "All Patients"

    elif correlation_type == "delta_declining":
        con_dec = df_con[(df_con['dementia_dx_bl'] == 'CON') & (df_con['dementia_dx'].isin(['MCI', 'AD']))]
        mci_dec = df_mci[(df_mci['dementia_dx_bl'] == 'MCI') & (df_mci['dementia_dx'] == 'AD')]
        delta_ctx = pd.concat([
            calc_delta_declining(con_dec, df_con, cols_ctx),
            calc_delta_declining(mci_dec, df_mci, cols_ctx)
        ])
        delta_sub = pd.concat([
            calc_delta_declining(con_dec, df_con, cols_sub),
            calc_delta_declining(mci_dec, df_mci, cols_sub)
        ]) if has_sub else None
        x_label, population = biomarker_cfg["x_axis_delta"], "Declining Patients"

    elif correlation_type == "baseline_all":
        delta_ctx = pd.concat([calc_baseline(d, cols_ctx) for d in [df_con, df_mci, df_ad]])
        delta_sub = pd.concat([calc_baseline(d, cols_sub) for d in [df_con, df_mci, df_ad]]) if has_sub else None
        x_label, population = biomarker_cfg["x_axis_base"], "All Patients"

    elif correlation_type == "baseline_declining":
        con_dec = df_con[(df_con['dementia_dx_bl'] == 'CON') & (df_con['dementia_dx'].isin(['MCI', 'AD']))]
        mci_dec = df_mci[(df_mci['dementia_dx_bl'] == 'MCI') & (df_mci['dementia_dx'] == 'AD')]
        delta_ctx = pd.concat([calc_baseline(con_dec, cols_ctx), calc_baseline(mci_dec, cols_ctx)])
        delta_sub = pd.concat([calc_baseline(con_dec, cols_sub), calc_baseline(mci_dec, cols_sub)]) if has_sub else None
        x_label, population = biomarker_cfg["x_axis_base"], "Declining Patients"

    else:
        raise ValueError(f"correlation_type desconocido: '{correlation_type}'. "
                         "Opciones: delta_all | delta_declining | baseline_all | baseline_declining")

    mean_ctx = delta_ctx.mean()
    mean_sub = delta_sub.mean() if delta_sub is not None else None

    # ── Normaliza índices ────────────────────────────────────────────────────
    mean_ctx = normalize_ctx_index(mean_ctx, biomarker_cfg)
    if mean_sub is not None:
        mean_sub = normalize_sub_index(mean_sub)

    # ── Carga receptores ─────────────────────────────────────────────────────
    df_rec_ctx, df_rec_sub = load_receptors(biomarker_cfg["ctx_index_mode"])

    available = [r for r in receptor_cols if r in df_rec_ctx.columns]
    if not available:
        print(f"  ⚠ Ningún receptor de {receptor_cols} encontrado en el CSV. Saltando.")
        return

    common_ctx = df_rec_ctx.index.intersection(mean_ctx.index)
    common_sub = df_rec_sub.index.intersection(mean_sub.index) if mean_sub is not None else pd.Index([])

    print(f"  Matched cortical:    {len(common_ctx)}")
    if has_sub:
        print(f"  Matched subcortical: {len(common_sub)}")

    # ── Genera figura ────────────────────────────────────────────────────────
    title = (f"Spearman ρ: {x_label} vs "
             f"{neurotransmitter_name.title()} Receptor Density — {population}")

    fig = make_figure(
        mean_ctx=mean_ctx,
        mean_sub=mean_sub,
        df_rec_ctx=df_rec_ctx,
        df_rec_sub=df_rec_sub,
        common_ctx=common_ctx,
        common_sub=common_sub,
        receptor_cols=available,
        x_label=x_label,
        title=title,
        has_subcortical=has_sub,
    )

    # ── Guarda HTML ──────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    fname = f"{correlation_type}_{biomarker_name}_{neurotransmitter_name}.html"
    fpath = os.path.join(output_dir, fname)
    fig.write_html(fpath)
    print(f"  ✓ Guardado: {fpath}")
    fig.show()
