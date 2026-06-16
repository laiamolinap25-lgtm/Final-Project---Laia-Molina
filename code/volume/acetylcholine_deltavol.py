import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np

# Colors
color_ctx = '#253a6b'
color_sub = '#de5f2d'

# Load data
df_control = pd.read_excel(r"C:\Users\laiam\OneDrive\Escritorio\Practicas Canada\neurotransmitters\data\adni_dkt_thick_con_Laia.xlsx")
df_mci     = pd.read_excel(r"C:\Users\laiam\OneDrive\Escritorio\Practicas Canada\neurotransmitters\data\adni_dkt_thick_mci_Laia.xlsx")

# Region columns
cols_ctx = [c for c in df_control.columns if 'grayvol' in c and not c.startswith(('subcort', 'total'))]
cols_sub = [c for c in [
    'left-thalamus-proper', 'left-caudate', 'left-putamen', 'left-pallidum',
    'left-hippocampus', 'left-amygdala', 'left-accumbens-area', 'left-ventraldc',
    'left-cerebellum-cortex', 'right-thalamus-proper', 'right-caudate',
    'right-putamen', 'right-pallidum', 'right-hippocampus', 'right-amygdala',
    'right-accumbens-area', 'right-ventraldc', 'right-cerebellum-cortex', 'brain-stem'
] if c in df_control.columns]

# Select declining subjects
control_decline = df_control[(df_control['dementia_dx_bl'] == 'CON') & (df_control['dementia_dx_last'].isin(['MCI', 'AD']))]
mci_decline     = df_mci[(df_mci['dementia_dx_bl'] == 'MCI') & (df_mci['dementia_dx_last'] == 'AD')]

# Delta volume per region (last - baseline)
timepoint_order = ['sc', 'm06', 'm12', 'm24', 'm36', 'm48', 'm60', 'm72']
times = {t: i for i, t in enumerate(timepoint_order)}

def calc_delta(df, cols):
    df = df.copy()
    cols = [c for c in cols if c in df.columns]
    df['tp_order'] = df['timepoint'].map(times)
    df_sorted = df.sort_values('tp_order')
    df_bl   = df_sorted.groupby('patient_id').first()[cols]
    df_last = df_sorted.groupby('patient_id').last()[cols]
    return df_last - df_bl

mean_ctx = pd.concat([calc_delta(control_decline, cols_ctx), calc_delta(mci_decline, cols_ctx)]).mean()
mean_ctx.index = mean_ctx.index.str.replace('_grayvol', '')

mean_sub = pd.concat([calc_delta(control_decline, cols_sub), calc_delta(mci_decline, cols_sub)]).mean()
mean_sub.index = (mean_sub.index.str.replace('-', ' ').str.title().str.replace(' ', '-'))

# Fix capitalization mismatches
mean_sub.index = mean_sub.index.map(lambda x: {
    'Left-Accumbens-Area':  'Left-Accumbens-area',
    'Right-Accumbens-Area': 'Right-Accumbens-area',
    'Left-Ventraldc':       'Left-VentralDC',
    'Right-Ventraldc':      'Right-VentralDC',
}.get(x, x))

# Load receptor table
df_rec = pd.read_csv(r"C:\Users\laiam\OneDrive\Escritorio\Practicas Canada\neurotransmitters\data\PET_parcellated\dkt\DKT_receptors_table_corticalandsubcortical.csv")
receptor_cols = ['VAChT', 'M1', 'A4B2']

# Split receptor table into cortical and subcortical
df_rec_ctx = df_rec[df_rec['region'].str.startswith('ctx-', na=False)].copy()
df_rec_ctx['region_key'] = df_rec_ctx['region'].str.replace('ctx-', '').str.replace('-', '_')
df_rec_ctx = df_rec_ctx.set_index('region_key')

df_rec_sub = df_rec[~df_rec['region'].str.startswith('ctx-', na=False)].copy()
df_rec_sub = df_rec_sub.dropna(subset=['region'])
df_rec_sub = df_rec_sub[df_rec_sub['region'] != 'nan']
df_rec_sub = df_rec_sub.set_index('region')

# Match regions
common_ctx = df_rec_ctx.index.intersection(mean_ctx.index)
common_sub = df_rec_sub.index.intersection(mean_sub.index)
print(f"Matched cortical:    {len(common_ctx)}")
print(f"Matched subcortical: {len(common_sub)}")
print(f"NOT matched: {mean_sub.index.difference(df_rec_sub.index).tolist()}")

# Global axis limits
x_all, y_all = [], []
for receptor in receptor_cols:
    for mean, rec, common in [(mean_ctx, df_rec_ctx, common_ctx), (mean_sub, df_rec_sub, common_sub)]:
        x = mean.loc[common].astype(float)
        y = rec.loc[common, receptor].astype(float)
        mask = x.notna() & y.notna()
        x_all.extend(x[mask].tolist())
        y_all.extend(y[mask].tolist())

x_pad = (max(x_all) - min(x_all)) * 0.05
y_pad = (max(y_all) - min(y_all)) * 0.05

fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharex='row', sharey='row')

datasets = [
    (0, mean_ctx, df_rec_ctx, common_ctx, 'Cortical',    color_ctx),
    (1, mean_sub, df_rec_sub, common_sub, 'Subcortical', color_sub),
]

for row, mean, df_rec_plot, common, label, color in datasets:
    for col, receptor in enumerate(receptor_cols):
        ax = axes[row, col]

        x = mean.loc[common].astype(float)
        y = df_rec_plot.loc[common, receptor].astype(float)
        mask = x.notna() & y.notna()
        x, y = x[mask], y[mask]

        rho, _ = stats.spearmanr(x, y)

        ax.scatter(x, y, color=color, alpha=0.7, s=40, label=f'{label} (n={len(x)})', zorder=2)

        m, b = np.polyfit(x, y, 1)
        x_sorted = np.sort(x)
        ax.plot(x_sorted, m * x_sorted + b, color='green', linewidth=1.5, zorder=1)

        ax.set_xlabel('Δ gray volume (mm³)', fontsize=9)
        ax.set_ylabel(f'{receptor} z-score', fontsize=9)
        ax.set_title(f'{receptor}  ρ={rho:.3f}', fontsize=11)
        ax.legend(fontsize=8)

plt.suptitle('Spearman ρ:Delta gray volume vs receptor density', fontsize=12)
plt.tight_layout()
plt.show()