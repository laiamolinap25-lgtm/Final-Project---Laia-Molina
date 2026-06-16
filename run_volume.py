"""
run_volume.py — Correlaciones para el biomarcador Volume.

⚠ ANTES DE EJECUTAR: edita config.py y ajusta:
  - BIOMARKERS["volume"]["csv_con/mci/ad"]  → rutas reales
  - BIOMARKERS["volume"]["ctx_prefix"]       → prefijo de columnas corticales
"""

from config import BIOMARKERS, NEUROTRANSMITTERS
from analysis import run_analysis

BIOMARKER  = "volume"
CFG        = BIOMARKERS[BIOMARKER]
OUTPUT_DIR = "output/volume"

CORRELATION_TYPES = [
    "delta_all",
    "delta_declining",
    "baseline_all",
    "baseline_declining",
]

NEURO_SELECTION = [
    "dopamine",
    "ach",
    "gaba",
    "serotonin",
    "glutamate",
]

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

    print("\n✅ Volume — análisis completado.")
