"""
run_fdg.py — Correlaciones para el biomarcador FDG.

Ejecuta las 4 correlaciones × 5 neurotransmisores = 20 combinaciones.
Puedes comentar las que no necesites.
"""

from config import BIOMARKERS, NEUROTRANSMITTERS
from analysis import run_analysis

BIOMARKER  = "fdg"
CFG        = BIOMARKERS[BIOMARKER]
OUTPUT_DIR = "output/fdg"

# ─── SELECCIÓN DE CORRELACIONES ──────────────────────────────────────────────
# Comenta las que no quieras ejecutar.
CORRELATION_TYPES = [
    "delta_all",
    "delta_declining",
    "baseline_all",
    "baseline_declining",
]

# ─── SELECCIÓN DE NEUROTRANSMISORES ──────────────────────────────────────────
# Comenta los que no quieras ejecutar.
NEURO_SELECTION = [
    "dopamine",
    "ach",
    "gaba",
    "serotonin",
    "glutamate",
]

# ─── EJECUCIÓN ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for corr_type in CORRELATION_TYPES:
        for neuro_name in NEURO_SELECTION:
            run_analysis(
                biomarker_name=BIOMARKER,
                biomarker_cfg=CFG,
                neurotransmitter_name=neuro_name,
                receptor_cols=NEUROTRANSMITTERS[neuro_name],
                correlation_type=corr_type,
                output_dir=OUTPUT_DIR,
            )

    print("\n✅ FDG — análisis completado.")
