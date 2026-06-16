from config import biomarkers, neurotransmitters
from analysis import run_analysis


biomarker  = "volume"
cfg        = biomarkers[biomarker]
output_dir = "output/volume"

correlation_types = [
    "delta_all",
    "delta_declining",
    "baseline_all",
    "baseline_declining",
]

neurotransmitter_selection = [
    "dopamine",
    "ach",
    "gaba",
    "serotonin",
    "glutamate",
]

if __name__ == "__main__":
    for corr_type in correlation_types:
        for neuro_name in neurotransmitter_selection:
            run_analysis(
                biomarker_name=biomarker,
                biomarker_cfg=cfg,
                neurotransmitter_name=neuro_name,
                receptor_cols=neurotransmitters[neuro_name],
                correlation_type=corr_type,
                output_dir=output_dir,
            )
