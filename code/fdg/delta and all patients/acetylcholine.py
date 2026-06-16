import pandas as pd
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

df_control = pd.read_csv(r"C:\Users\laiam\OneDrive\Escritorio\Practicas Canada\neurotransmitters\data\adni_fdg_con_Laia.csv")
df_mci     = pd.read_csv(r"C:\Users\laiam\OneDrive\Escritorio\Practicas Canada\neurotransmitters\data\adni_fdg_mci_Laia.csv")
df_ad      = pd.read_csv(r"C:\Users\laiam\OneDrive\Escritorio\Practicas Canada\neurotransmitters\data\adni_fdg_ad_Laia.csv")

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

def calc_delta_all(df, cols):
    df = df.copy()
    df['tp_months'] = df['timepoint'].map(timepoint_to_months)
    df = df.sort_values('tp_months')

    df_bl   = df.groupby('patient_id').first()[cols + ['tp_months']]
    df_last = df.groupby('patient_id').last()[cols + ['tp_months']]

    months_diff = (df_last['tp_months'] - df_bl['tp_months']).replace(0, float('nan'))
    years_diff  = months_diff / 12

    delta = (df_last[cols] - df_bl[cols]).div(years_diff, axis=0)
    return delta

delta_ctx = pd.concat([
    calc_delta_all(df_control, cols_ctx),
    calc_delta_all(df_mci,     cols_ctx),
    calc_delta_all(df_ad,      cols_ctx)
])

delta_sub = pd.concat([
    calc_delta_all(df_control, cols_sub),
    calc_delta_all(df_mci,     cols_sub),
    calc_delta_all(df_ad,      cols_sub)
])

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

print(f"Matched cortical:    {len(common_ctx)}")
print(f"Matched subcortical: {len(common_sub)}")

receptor_cols = ['VAChT', 'M1', 'A4B2']
color_ctx = '#253a6b'
color_sub = '#de5f2d'

fig = make_subplots(rows=2, cols=3,
                    shared_xaxes='rows',
                    shared_yaxes='rows',
                    subplot_titles=[f'{r} - Cortical' for r in receptor_cols] +
                                   [f'{r} - Subcortical' for r in receptor_cols])

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
            hovertemplate='<b>%{text}</b><br>Δ FDG: %{x:.4f}<br>Receptor: %{y:.4f}<extra></extra>',
            name=f'{label} (n={mask.sum()})',
            showlegend=(col == 1)
        ), row=row, col=col)

        m, b = np.polyfit(x, y, 1)
        x_sorted = np.sort(x)
        fig.add_trace(go.Scatter(
            x=x_sorted, y=m * x_sorted + b,
            mode='lines',
            line=dict(color='green', width=1.5),
            showlegend=False
        ), row=row, col=col)

        fig.update_xaxes(title_text='Δ FDG (per year)', row=row, col=col)
        fig.update_yaxes(title_text='density z-score', row=row, col=col)

        idx = (row - 1) * 3 + col
        fig.layout.annotations[idx - 1].text = f'{receptor} - {label}  ρ={rho:.3f}'

fig.update_layout(
    title='Spearman ρ: Δ FDG (per year) vs Dopaminergic Receptor Density — All Patients',
    height=800, width=1200
)

fig.write_html('delta_fdg_Ach_all_patients.html')
fig.show()