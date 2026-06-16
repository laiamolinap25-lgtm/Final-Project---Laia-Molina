"""
config.py — Configuración central para todos los biomarcadores.
Edita las rutas y columnas de cada biomarcador aquí.
"""

import os

# ─── RUTA BASE ───────────────────────────────────────────────────────────────
BASE_DIR = r"C:\Users\laiam\OneDrive\Escritorio\Practicas Canada\neurotransmitters\data"

# ─── ARCHIVO DE RECEPTORES (común a todos) ───────────────────────────────────
RECEPTORS_CSV = os.path.join(
    BASE_DIR,
    r"PET_parcellated\dkt\DKT_receptors_table_corticalandsubcortical.csv"
)

# ─── NEUROTRANSMISORES ───────────────────────────────────────────────────────
NEUROTRANSMITTERS = {
    "dopamine":    ["D1", "D2", "DAT"],
    "ach":         ["VAChT", "M1", "A4B2"],
    "gaba":        ["GABAa-bz", "GABAa"],
    "serotonin":   ["5HT1a", "5HT1b", "5HT2a", "5HT4", "5HT6", "5HTT"],
    "glutamate":   ["mGluR5", "NMDA"],   # tal como aparecen en el CSV de receptores
}

# ─── TIMEPOINTS ──────────────────────────────────────────────────────────────
TIMEPOINT_TO_MONTHS = {
    'bl': 0, 'm06': 6, 'm12': 12, 'm24': 24,
    'm36': 36, 'm48': 48, 'm60': 60, 'm72': 72,
    'm84': 84, 'm96': 96, 'm108': 108, 'm120': 120,
    'm132': 132, 'm144': 144
}

# ─── COLUMNAS SUBCORTICALES ───────────────────────────────────────────────────
SUBCORTICAL_COLS = [
    'left-thalamus-proper', 'left-caudate', 'left-putamen', 'left-pallidum',
    'left-hippocampus', 'left-amygdala', 'left-accumbens-area', 'left-ventraldc',
    'left-cerebellum-cortex', 'right-thalamus-proper', 'right-caudate',
    'right-putamen', 'right-pallidum', 'right-hippocampus', 'right-amygdala',
    'right-accumbens-area', 'right-ventraldc', 'right-cerebellum-cortex', 'brain-stem'
]

# ─── CONFIGURACIÓN POR BIOMARCADOR ───────────────────────────────────────────
#
# Campos obligatorios:
#   file_con, file_mci, file_ad  : rutas a los archivos de datos
#   file_format                  : "csv" o "xlsx"
#   ctx_col_filter               : función lambda que filtra columnas corticales
#   ctx_index_transform          : función lambda que normaliza el índice cortical
#                                  para que coincida con el CSV de receptores
#   has_subcortical              : True/False (Thickness solo tiene cortical)
#   ctx_index_mode               : "standard" → set_index('region')
#                                  "region_key" → transformación especial de Volume
#   x_axis_delta                 : etiqueta eje X para correlación delta
#   x_axis_base                  : etiqueta eje X para correlación baseline

BIOMARKERS = {

    # ── FDG ──────────────────────────────────────────────────────────────────
    # Columnas: lh_superiorfrontal, rh_... → índice: ctx-lh-superiorfrontal
    "fdg": {
        "file_con":    os.path.join(BASE_DIR, "adni_fdg_con_Laia.csv"),
        "file_mci":    os.path.join(BASE_DIR, "adni_fdg_mci_Laia.csv"),
        "file_ad":     os.path.join(BASE_DIR, "adni_fdg_ad_Laia.csv"),
        "file_format": "csv",
        "ctx_col_filter":      lambda cols: [c for c in cols if c.startswith(('lh_', 'rh_'))],
        "ctx_index_transform": lambda idx: 'ctx-' + idx.str.replace('_', '-'),
        "has_subcortical": True,
        "ctx_index_mode":  "standard",
        "x_axis_delta": "Δ FDG (per year)",
        "x_axis_base":  "FDG (baseline)",
    },

    # ── T1/T2 ────────────────────────────────────────────────────────────────
    # Mismos archivos y estructura que FDG
    "t1t2": {
        "file_con":    os.path.join(BASE_DIR, "adni_fdg_con_Laia.csv"),
        "file_mci":    os.path.join(BASE_DIR, "adni_fdg_mci_Laia.csv"),
        "file_ad":     os.path.join(BASE_DIR, "adni_fdg_ad_Laia.csv"),
        "file_format": "csv",
        "ctx_col_filter":      lambda cols: [c for c in cols if c.startswith(('lh_', 'rh_'))],
        "ctx_index_transform": lambda idx: 'ctx-' + idx.str.replace('_', '-'),
        "has_subcortical": True,
        "ctx_index_mode":  "standard",
        "x_axis_delta": "Δ T1/T2 (per year)",
        "x_axis_base":  "T1/T2 (baseline)",
    },

    # ── VOLUME ───────────────────────────────────────────────────────────────
    # Columnas: lh_superiorfrontal_grayvol → índice: lh_superiorfrontal
    # El CSV de receptores usa region_key (sin ctx-, con _)
    "volume": {
        "file_con":    os.path.join(BASE_DIR, "adni_dkt_thick_con_Laia.xlsx"),  # ← AJUSTA nombre si es distinto
        "file_mci":    os.path.join(BASE_DIR, "adni_dkt_thick_mci_Laia.xlsx"),  # ← AJUSTA
        "file_ad":     os.path.join(BASE_DIR, "adni_dkt_thick_ad_Laia.xlsx"),   # ← AJUSTA
        "file_format": "xlsx",
        "ctx_col_filter":      lambda cols: [c for c in cols if 'grayvol' in c and not c.startswith(('subcort', 'total'))],
        "ctx_index_transform": lambda idx: idx.str.replace('_grayvol', ''),
        "has_subcortical": True,
        "ctx_index_mode":  "region_key",   # índice especial: ctx- quitado, - → _
        "x_axis_delta": "Δ Volume (per year)",
        "x_axis_base":  "Volume (baseline)",
    },

    # ── THICKNESS ────────────────────────────────────────────────────────────
    # Columnas: lh_superiorfrontal_thickavg → índice: ctx-lh-superiorfrontal
    # Solo cortical (sin subcortical)
    "thickness": {
        "file_con":    os.path.join(BASE_DIR, "adni_dkt_thick_con_Laia.xlsx"),
        "file_mci":    os.path.join(BASE_DIR, "adni_dkt_thick_mci_Laia.xlsx"),
        "file_ad":     os.path.join(BASE_DIR, "adni_dkt_thick_ad_Laia.xlsx"),
        "file_format": "xlsx",
        "ctx_col_filter":      lambda cols: [c for c in cols if c.endswith('thickavg')],
        "ctx_index_transform": lambda idx: 'ctx-' + idx.str.replace('_', '-').str.replace('-thickavg', ''),
        "has_subcortical": False,   # ← Thickness no tiene subcortical
        "ctx_index_mode":  "standard",
        "x_axis_delta": "Δ Thickness (per year)",
        "x_axis_base":  "Thickness (baseline)",
    },
}

# ─── COLORES ─────────────────────────────────────────────────────────────────
COLOR_CORTICAL    = '#253a6b'
COLOR_SUBCORTICAL = '#de5f2d'
COLOR_TRENDLINE   = 'green'
