import os
base_dir = r"C:\Users\laiam\OneDrive\Escritorio\Practicas Canada\neurotransmitters\data"
receptors_csv = os.path.join(
    base_dir,
    r"PET_parcellated\dkt\DKT_receptors_table_corticalandsubcortical.csv"
)

neurotransmitters = {
    "dopamine":    ["D1", "D2", "DAT"],
    "ach":         ["VAChT", "M1", "A4B2"],
    "gaba":        ["GABAa-bz", "GABAa"],
    "serotonin":   ["5HT1a", "5HT1b", "5HT2a", "5HT4", "5HT6", "5HTT"],
    "glutamate":   ["mGluR5", "NMDA"],  
}

timepoint_to_months = {
    'bl': 0, 'm06': 6, 'm12': 12, 'm24': 24,
    'm36': 36, 'm48': 48, 'm60': 60, 'm72': 72,
    'm84': 84, 'm96': 96, 'm108': 108, 'm120': 120,
    'm132': 132, 'm144': 144
}

subcortical_cols = [
    'left-thalamus-proper', 'left-caudate', 'left-putamen', 'left-pallidum',
    'left-hippocampus', 'left-amygdala', 'left-accumbens-area', 'left-ventraldc',
    'left-cerebellum-cortex', 'right-thalamus-proper', 'right-caudate',
    'right-putamen', 'right-pallidum', 'right-hippocampus', 'right-amygdala',
    'right-accumbens-area', 'right-ventraldc', 'right-cerebellum-cortex', 'brain-stem'
]


biomarkers = {
    "fdg": {
        "file_con":    os.path.join(base_dir, "adni_fdg_con_Laia.csv"),
        "file_mci":    os.path.join(base_dir, "adni_fdg_mci_Laia.csv"),
        "file_ad":     os.path.join(base_dir, "adni_fdg_ad_Laia.csv"),
        "file_format": "csv",
        "ctx_col_filter":      lambda cols: [c for c in cols if c.startswith(('lh_', 'rh_'))],
        "ctx_index_transform": lambda idx: 'ctx-' + idx.str.replace('_', '-'),
        "has_subcortical": True,
        "ctx_index_mode":  "standard",
        "x_axis_delta": "Δ FDG (per year)",
        "x_axis_base":  "FDG (baseline)",
    },

    "t1t2": {
        "file_con":    os.path.join(base_dir, "adni_t1t2_con_Laia.csv"),
        "file_mci":    os.path.join(base_dir, "adni_t1t2_mci_Laia.csv"),
        "file_ad":     os.path.join(base_dir, "adni_t1t2_ad_Laia.csv"),
        "file_format": "csv",
        "ctx_col_filter":      lambda cols: [c for c in cols if c.startswith(('lh_', 'rh_'))],
        "ctx_index_transform": lambda idx: 'ctx-' + idx.str.replace('_', '-'),
        "has_subcortical": True,
        "ctx_index_mode":  "standard",
        "x_axis_delta": "Δ T1/T2 (per year)",
        "x_axis_base":  "T1/T2 (baseline)",
    },

    "volume": {
        "file_con":    os.path.join(base_dir, "adni_dkt_thick_con_Laia.xlsx"),  
        "file_mci":    os.path.join(base_dir, "adni_dkt_thick_mci_Laia.xlsx"), 
        "file_ad":     os.path.join(base_dir, "adni_dkt_thick_ad_Laia.xlsx"),   
        "file_format": "xlsx",
        "ctx_col_filter":      lambda cols: [c for c in cols if 'grayvol' in c and not c.startswith(('subcort', 'total'))],
        "ctx_index_transform": lambda idx: idx.str.replace('_grayvol', ''),
        "has_subcortical": True,
        "ctx_index_mode":  "region_key",   
        "x_axis_delta": "Δ Volume (per year)",
        "x_axis_base":  "Volume (baseline)",
    },

    "thickness": {
        "file_con":    os.path.join(base_dir, "adni_dkt_thick_con_Laia.xlsx"),
        "file_mci":    os.path.join(base_dir, "adni_dkt_thick_mci_Laia.xlsx"),
        "file_ad":     os.path.join(base_dir, "adni_dkt_thick_ad_Laia.xlsx"),
        "file_format": "xlsx",
        "ctx_col_filter":      lambda cols: [c for c in cols
                                             if c.startswith(('lh_', 'rh_'))
                                             and c.endswith(('_thickavg', '_thickness'))],
        "ctx_index_transform": lambda idx: (
            'ctx-' + idx.str.replace('_', '-')
                         .str.replace('-thickavg', '', regex=False)
                         .str.replace('-thickness', '', regex=False)
        ),
        "has_subcortical":  False,
        "ctx_index_mode":   "standard",
        "declining_dx_col": "dementia_dx_last",
        "x_axis_delta": "Δ Thickness (per year)",
        "x_axis_base":  "Thickness (baseline)",
    },
}


color_cortical = '#253a6b'
color_subcortical = '#de5f2d'
color_trendline   = "#6EB940"
