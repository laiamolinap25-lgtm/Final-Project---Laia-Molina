import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


excel_by_group = {
    'Control': r"C:\Users\laiam\OneDrive\Escritorio\Practicas Canada\neurotransmitters\data\adni_dkt_thick_con_Laia.xlsx",
    'MCI':     r"C:\Users\laiam\OneDrive\Escritorio\Practicas Canada\neurotransmitters\data\adni_dkt_thick_mci_Laia.xlsx",
    'AD':      r"C:\Users\laiam\OneDrive\Escritorio\Practicas Canada\neurotransmitters\data\adni_dkt_thick_ad_Laia.xlsx",
}
PET_PATH = r"C:\Users\laiam\OneDrive\Escritorio\Practicas Canada\neurotransmitters\data\PET_parcellated\dkt\DKT_receptors_table_corticalandsubcortical.csv"

# Load PET
df_rec = pd.read_csv(PET_PATH)
receptor_cols = ['VAChT', 'M1', 'A4B2']

# Load each group and select baseline
data_by_group = {}
for group, path in excel_by_group.items():
    df = pd.read_excel(path)
    data_by_group[group] = df[df['timepoint.1'] == 'sc']

# Region columns
cols_ctx = [c for c in df.columns if 'grayvol' in c and not c.startswith(('subcort', 'total'))]
cols_sub = [c for c in [
    'left-thalamus-proper', 'left-caudate', 'left-putamen', 'left-pallidum',
    'left-hippocampus', 'left-amygdala', 'left-accumbens-area', 'left-ventraldc',
    'left-cerebellum-cortex', 'right-thalamus-proper', 'right-caudate',
    'right-putamen', 'right-pallidum', 'right-hippocampus', 'right-amygdala',
    'right-accumbens-area', 'right-ventraldc', 'right-cerebellum-cortex', 'brain-stem'
] if c in df.columns]

# Mean volume per region per group
mean_ctx_by_group = {}
mean_sub_by_group = {}

for group, df_bl in data_by_group.items():
    mean_ctx = df_bl[cols_ctx].mean()
    mean_ctx.index = mean_ctx.index.str.replace('_grayvol', '')
    mean_ctx_by_group[group] = mean_ctx

    mean_sub = df_bl[cols_sub].mean()
    mean_sub.index = (mean_sub.index.str.replace('-', ' ').str.title().str.replace(' ', '-'))
    mean_sub.index = mean_sub.index.map(lambda x: {
        'Left-Accumbens-Area':  'Left-Accumbens-area',
        'Right-Accumbens-Area': 'Right-Accumbens-area',
        'Left-Ventraldc':       'Left-VentralDC',
        'Right-Ventraldc':      'Right-VentralDC',
    }.get(x, x))
    mean_sub_by_group[group] = mean_sub

# Split PET into cortical and subcortical
df_rec_ctx = df_rec[df_rec['region'].str.startswith('ctx-', na=False)].copy()
df_rec_ctx['region_key'] = df_rec_ctx['region'].str.replace('ctx-', '').str.replace('-', '_')
df_rec_ctx = df_rec_ctx.set_index('region_key')

df_rec_sub = df_rec[~df_rec['region'].str.startswith('ctx-', na=False)].copy()
df_rec_sub = df_rec_sub.dropna(subset=['region'])
df_rec_sub = df_rec_sub[df_rec_sub['region'] != 'nan']
df_rec_sub = df_rec_sub.set_index('region')

# Match regions
common_ctx = df_rec_ctx.index.intersection(mean_ctx_by_group['Control'].index)
common_sub = df_rec_sub.index.intersection(mean_sub_by_group['Control'].index)

print(f"Matched cortical:    {len(common_ctx)}")
print(f"Matched subcortical: {len(common_sub)}")

group_colors = {
    'Control': '#253a6b',
    'MCI':     '#9bc745',
    'AD':      '#de5f2d',
}

fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharey='row', sharex='row')

region_configs = [
    (0, common_ctx, df_rec_ctx, mean_ctx_by_group, 'Cortical'),
    (1, common_sub, df_rec_sub, mean_sub_by_group, 'Subcortical'),
]

for row_idx, common, df_rec_plot, mean_by_group, region_title in region_configs:
    for col_idx, receptor in enumerate(receptor_cols):
        ax = axes[row_idx, col_idx]
        y = df_rec_plot.loc[common, receptor].astype(float)

        for group in mean_by_group:
            x = mean_by_group[group].loc[common].astype(float)
            mask = x.notna() & y.notna()
            x_m, y_m = x[mask], y[mask]

            if len(x_m) < 2:
                continue

            rho, _ = spearmanr(x_m, y_m)
            m, b = np.polyfit(x_m, y_m, 1)
            xs = np.sort(x_m)

            ax.scatter(x_m, y_m, color=group_colors[group], alpha=0.5, s=30)
            ax.plot(xs, m * xs + b, color=group_colors[group], linewidth=2,
                    label=f'{group} (ρ={rho:.2f})')

        ax.set_xlabel('Mean gray volume (mm³)', fontsize=9)
        ax.set_ylabel(f'{receptor} z-score', fontsize=9)
        ax.set_title(f'{region_title} – {receptor}', fontsize=11)
        ax.legend(fontsize=8)

plt.suptitle('Spearman ρ: gray volume vs acetylcholine receptor density', fontsize=13)
plt.tight_layout()
plt.savefig('spearman_acetylcholine.png', dpi=150, bbox_inches='tight')
plt.show()