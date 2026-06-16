import pandas as pd
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

df_control = pd.read_csv(r"C:\Users\laiam\OneDrive\Escritorio\Practicas Canada\neurotransmitters\data\adni_fdg_con_Laia.csv")
df_mci     = pd.read_csv(r"C:\Users\laiam\OneDrive\Escritorio\Practicas Canada\neurotransmitters\data\adni_fdg_mci_Laia.csv")

control_decline = df_control[(df_control['dementia_dx_bl'] == 'CON') & (df_control['dementia_dx'].isin(['MCI', 'AD']))]
mci_decline     = df_mci[(df_mci['dementia_dx_bl'] == 'MCI') & (df_mci['dementia_dx'] == 'AD')]

timepoint_to_months = {'bl': 0, 'm06': 6, 'm12': 12, 'm24': 24,
                       'm36': 36, 'm48': 48, 'm60': 60, 'm72': 72,
                       'm84': 84, 'm96': 96, 'm108': 108, 'm120': 120,
                       'm132': 132, 'm144': 144}

cols_ctx = [c for c in df_control.columns if c.startswith(('lh_', 'rh_'))]
cols_sub = [c for c in [
    'left-thalamus-proper', 'left-caudate', 'left-putamen', 'left-pallidum',
    'left-hippocampus', 'left-amygdala', 'left-accumbens-area', 'left-ventraldc',
    'left-cerebellum-cortex', 'right-thalamus-proper', 'right-caudate',
    'right-putamen', 'right-pallidum', 'right-hippocampus', 'right-amygdala',
    'right-accumbens-area', 'right-ventraldc', 'right-cerebellum-cortex', 'brain-stem'
] if c in df_control.columns]

def calc_baseline(df, cols):
    df = df.copy()
    df['tp_months'] = df['timepoint'].map(timepoint_to_months)
    df = df.sort_values('tp_months')
    return df.groupby('patient_id').first()[cols]

delta_ctx = pd.concat([calc_baseline(control_decline, cols_ctx), calc_baseline(mci_decline, cols_ctx)])
delta_sub = pd.concat([calc_baseline(control_decline, cols_sub), calc_baseline(mci_decline, cols_sub)])

mean_ctx = delta_ctx.mean()
mean_sub = delta_sub.mean()

df_rec = pd.read_csv(r"C:\Users\laiam\OneDrive\Escritorio\Practicas Canada\neurotransmitters\data\PET_parcellated\dkt\DKT_receptors_table_corticalandsubcortical.csv")

mean_sub.index = mean_sub.index.str.replace('-', ' ').str.title().str.replace(' ', '-')
mean_sub.index = mean_sub.index.map(lambda x: {
    'Left-Accumbens-Area':  'Left-Accumbens-area',
    'Right-Accumbens-Area': 'Right-Accumbens-area',
    'Left-Ventraldc':       'Left-VentralDC',
    'Right-Ventraldc':      'Right-VentralDC',
}.get(x, x))

mean_ctx.index = 'ctx-' + mean_ctx.index.str.replace('_', '-')

df_rec_ctx = df_rec[df_rec['region'].str.startswith('ctx-', na=False)].set_index('region')
df_rec_sub = df_rec[~df_rec['region'].str.startswith('ctx-', na=False)].dropna(subset=['region']).set_index('region')

common_ctx = df_rec_ctx.index.intersection(mean_ctx.index)
common_sub = df_rec_sub.index.intersection(mean_sub.index)

receptor_cols = ['GABAa-bz', 'GABAa']
color_ctx = '#253a6b'
color_sub = '#de5f2d'
n_cols = len(receptor_cols)

fig = make_subplots(
    rows=2, cols=n_cols,
    shared_xaxes='rows',
    shared_yaxes='rows',
    subplot_titles=[f'{r} - Cortical' for r in receptor_cols] +
                   [f'{r} - Subcortical' for r in receptor_cols]
)

datasets = [
    (1, mean_ctx, df_rec_ctx, common_ctx, 'Cortical',    color_ctx),
    (2, mean_sub, df_rec_sub, common_sub, 'Subcortical', color_sub),
]

for row, mean, df_rec_plot, common, label, color in datasets:
    for col, receptor in enumerate(receptor_cols, start=1):
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
            hovertemplate='<b>%{text}</b><br>FDG baseline: %{x:.4f}<br>Receptor: %{y:.4f}<extra></extra>',
            name=f'{label} (n={mask.sum()})',
            showlegend=(col == 1)
        ), row=row, col=col)

        m, b = np.polyfit(x, y, 1)
        x_sorted = np.sort(x)
        fig.add_trace(go.Scatter(
            x=x_sorted, y=m * x_sorted + b,
            mode='lines',
            line=dict(color='green', width=1.5),
            showlegend=False,
            hoverinfo='skip'
        ), row=row, col=col)

        fig.update_xaxes(title_text='FDG baseline', row=row, col=col)
        fig.update_yaxes(title_text=f'{receptor} density z-score', row=row, col=col)

        idx = (row - 1) * n_cols + col
        fig.layout.annotations[idx - 1].text = f'{receptor} - {label}  ρ={rho:.3f}'

fig.update_layout(
    title='Spearman ρ:   FDG baseline  vs  Receptor Density',
    height=800, width=900,
    template='plotly_white'
)

fig.write_html('baseline_fdg_GABA_declining_patients.html')
fig.show()