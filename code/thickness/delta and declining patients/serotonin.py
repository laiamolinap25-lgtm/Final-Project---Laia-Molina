import pandas as pd
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

df_control = pd.read_excel(r"C:\Users\laiam\OneDrive\Escritorio\Practicas Canada\neurotransmitters\data\adni_dkt_thick_con_Laia.xlsx")
df_mci     = pd.read_excel(r"C:\Users\laiam\OneDrive\Escritorio\Practicas Canada\neurotransmitters\data\adni_dkt_thick_mci_Laia.xlsx")

control_decline = df_control[(df_control['dementia_dx_bl'] == 'CON') & (df_control['dementia_dx'].isin(['MCI', 'AD']))]
mci_decline     = df_mci[(df_mci['dementia_dx_bl'] == 'MCI') & (df_mci['dementia_dx'] == 'AD')]

timepoint_to_months = {'bl': 0, 'm06': 6, 'm12': 12, 'm24': 24,
                       'm36': 36, 'm48': 48, 'm60': 60, 'm72': 72,
                       'm84': 84, 'm96': 96, 'm108': 108, 'm120': 120,
                       'm132': 132, 'm144': 144}


cols_ctx = [c for c in df_control.columns
            if c.startswith(('lh_', 'rh_')) and c.endswith(('_thickness', '_thickavg'))]

def calc_delta(df, all_df, cols):
    ids = df['patient_id'].unique()
    all_df = all_df[all_df['patient_id'].isin(ids)].copy()
    all_df['tp_months'] = all_df['timepoint'].map(timepoint_to_months)
    all_df = all_df.sort_values('tp_months')

    df_bl   = all_df.groupby('patient_id').first()[cols + ['tp_months']]
    df_last = all_df.groupby('patient_id').last()[cols + ['tp_months']]

    months_diff = (df_last['tp_months'] - df_bl['tp_months']).replace(0, float('nan'))
    years_diff  = months_diff / 12

    delta = (df_last[cols] - df_bl[cols]).div(years_diff, axis=0)
    return delta

delta_ctx = pd.concat([
    calc_delta(control_decline, df_control, cols_ctx),
    calc_delta(mci_decline,     df_mci,     cols_ctx)
])

mean_ctx = delta_ctx.mean()


df_rec = pd.read_csv(r"C:\Users\laiam\OneDrive\Escritorio\Practicas Canada\neurotransmitters\data\PET_parcellated\dkt\DKT_receptors_table_corticalandsubcortical.csv")

# Map lh_fusiform_thickness  →  ctx-lh-fusiform
mean_ctx.index = ('ctx-' +
                  mean_ctx.index
                          .str.replace('_thickavg$', '', regex=True)  # strip -thickavg
                          .str.replace('_thickness$', '', regex=True) # fallback for _thickness
                          .str.replace('_', '-'))

df_rec_ctx = (df_rec[df_rec['region'].str.startswith('ctx-', na=False)]
              .set_index('region'))

common_ctx = df_rec_ctx.index.intersection(mean_ctx.index)
print(f"Matched cortical: {len(common_ctx)}")


receptor_cols =['5HT1a', '5HT1b', '5HT2a', '5HT4', '5HT6', '5HTT']
color_ctx = '#253a6b'
n_cols = len(receptor_cols)

fig = make_subplots(rows=1, cols=n_cols,
                    shared_yaxes=True,
                    subplot_titles=receptor_cols)

for col, receptor in enumerate(receptor_cols, start=1):
    x = mean_ctx.loc[common_ctx].astype(float)
    y = df_rec_ctx.loc[common_ctx, receptor].astype(float)
    mask = x.notna() & y.notna()
    x, y = x[mask], y[mask]
    region_names = common_ctx[mask]

    rho, _ = stats.spearmanr(x, y)

    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='markers',
        marker=dict(color=color_ctx, size=7, opacity=0.7),
        text=region_names,
        hovertemplate='<b>%{text}</b><br>Δ Thickness: %{x:.4f}<br>Receptor: %{y:.4f}<extra></extra>',
        name=f'Cortical (n={mask.sum()})',
        showlegend=(col == 1)
    ), row=1, col=col)

    m, b = np.polyfit(x, y, 1)
    x_sorted = np.sort(x)
    fig.add_trace(go.Scatter(
        x=x_sorted, y=m * x_sorted + b,
        mode='lines',
        line=dict(color='green', width=1.5),
        showlegend=False
    ), row=1, col=col)

    fig.update_xaxes(title_text='Δ Thickness (mm/year)', row=1, col=col)
    fig.update_yaxes(title_text='density z-score',       row=1, col=col)

    fig.layout.annotations[col - 1].text = f'{receptor}  ρ={rho:.3f}  '

fig.update_layout(
    title='Spearman ρ: Δ Cortical Thickness (per year) vs Serotonin Receptor Density — Declining Patients',
    height=500, width=1800
)

fig.write_html('delta_thickness_serotonin_declining_patients.html')
fig.show()