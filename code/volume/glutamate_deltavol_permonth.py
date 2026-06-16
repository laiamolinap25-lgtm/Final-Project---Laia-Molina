import pandas as pd
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

# Timepoint mapping
timepoint_months = {
    'sc': 0,  'bl': 0,
    'm03': 3,  'm06': 6,  'm12': 12, 'm18': 18,
    'm24': 24, 'm36': 36, 'm48': 48, 'm60': 60,
    'm72': 72, 'm84': 84, 'm96': 96, 'm108': 108,
    'm120': 120, 'm132': 132, 'm144': 144, 'm156': 156,
    'm168': 168
}

def calc_delta_per_month(df, cols): #calculate the change in volume per month for each patient and region and then average across patients to get the mean change per month for each region 
    df = df.copy()
    cols = [c for c in cols if c in df.columns]
    df['months'] = df['timepoint.1'].map(timepoint_months)
    df = df.dropna(subset=['months']) #make  sure to drop rows where the timepoint is not recognized to avoid errors
    df_sorted = df.sort_values('months') #sort the dataframe by the number of months to ensure that the baseline and last measurements are correctly identified for each patient
    df_bl     = df_sorted.groupby('patient_id').first()[cols + ['months']] #get the baseline measurements for each patient by taking the first entry for each patient_id after sorting by months
    df_last   = df_sorted.groupby('patient_id').last()[cols + ['months']] #get the last measurements for each patient by taking the last entry for each patient_id after sorting by months
    n_months  = df_last['months'] - df_bl['months']
    valid     = n_months > 0 #only consider patients who have a positive number of months between baseline and last measurement to avoid division by zero or negative time intervals
    delta     = (df_last.loc[valid, cols] - df_bl.loc[valid, cols]).div(n_months[valid], axis=0) #calculate the change in volume per month by taking the difference between last and baseline measurements and dividing by the number of months for valid patients
    return delta

mean_ctx = pd.concat([calc_delta_per_month(control_decline, cols_ctx),#calculate the mean change in volume per month for cortical regions by concatenating the results from control and MCI declining groups and then taking the mean across patients for each region
                      calc_delta_per_month(mci_decline, cols_ctx)]).mean()
mean_ctx.index = mean_ctx.index.str.replace('_grayvol', '') #clean up the region names by removing the '_grayvol' suffix to match the format used in the receptor table for easier merging later on

mean_sub = pd.concat([calc_delta_per_month(control_decline, cols_sub),
                      calc_delta_per_month(mci_decline, cols_sub)]).mean()
mean_sub.index = mean_sub.index.str.replace('-', ' ').str.title().str.replace(' ', '-')
mean_sub.index = mean_sub.index.map(lambda x: {
    'Left-Accumbens-Area':  'Left-Accumbens-area',
    'Right-Accumbens-Area': 'Right-Accumbens-area',
    'Left-Ventraldc':       'Left-VentralDC',
    'Right-Ventraldc':      'Right-VentralDC',
}.get(x, x))

# Load receptor table
df_rec = pd.read_csv(r"C:\Users\laiam\OneDrive\Escritorio\Practicas Canada\neurotransmitters\data\PET_parcellated\dkt\DKT_receptors_table_corticalandsubcortical.csv")
receptor_cols = ['NMDA', 'mGluR5']

df_rec_ctx = df_rec[df_rec['region'].str.startswith('ctx-', na=False)].copy() #split the receptor table into cortical and subcortical regions based on the prefix 'ctx-' in the region names to facilitate matching with the mean volume change data for cortical and subcortical regions separately
df_rec_ctx['region_key'] = df_rec_ctx['region'].str.replace('ctx-', '').str.replace('-', '_') #clean up the region names in the cortical receptor table by removing the 'ctx-' prefix and replacing hyphens with underscores to create a consistent format for merging with the mean volume change data
df_rec_ctx = df_rec_ctx.set_index('region_key') #set the cleaned region names as the index for the cortical receptor table to allow for easy lookup when matching with the mean volume change data

df_rec_sub = df_rec[~df_rec['region'].str.startswith('ctx-', na=False)].copy() #filter the receptor table to get only subcortical regions by selecting rows where the region name does not start with 'ctx-'
df_rec_sub = df_rec_sub.dropna(subset=['region']) # drop rows where the region name is missing in the subcortical receptor table to ensure that all entries have valid region names for matching with the mean volume change data
df_rec_sub = df_rec_sub[df_rec_sub['region'] != 'nan'] # further filter the subcortical receptor table to remove any rows where the region name is the string 'nan' 
df_rec_sub = df_rec_sub.set_index('region')

common_ctx = df_rec_ctx.index.intersection(mean_ctx.index) #find the common regions between the cortical receptor table and the mean volume change data for cortical regions by taking the intersection of their indices to ensure that only regions present in both datasets are included in the correlation analysis
common_sub = df_rec_sub.index.intersection(mean_sub.index) #find the common regions between the subcortical receptor table and the mean volume change data for subcortical regions by taking the intersection of their indices to ensure that only regions present in both datasets are included in the correlation analysis


mean_all = pd.concat([mean_ctx, mean_sub]) #combine the mean volume change data for cortical and subcortical regions into a single series to facilitate a global correlation analysis across all regions regardless of their cortical or subcortical classification
df_rec_all = pd.concat([df_rec_ctx, df_rec_sub])#combine the cortical and subcortical receptor tables into a single dataframe to facilitate a global correlation analysis across all regions regardless of their cortical or subcortical classification
common_all = df_rec_all.index.intersection(mean_all.index) #find the common regions between the combined receptor table and the combined mean volume change data by taking the intersection of their indices to ensure that only regions present in both datasets are included in the global correlation analysis


labels = pd.Series('Cortical', index=mean_ctx.index) #create a series of labels to indicate whether each region is cortical or subcortical based on the index of the mean volume change data for cortical regions, which will be used for coloring the points in the scatter plots and for grouping in the legend
labels = pd.concat([labels, pd.Series('Subcortical', index=mean_sub.index)]) 


fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=receptor_cols,
    horizontal_spacing=0.08,
    shared_xaxes=True,  
    shared_yaxes=True,  
)

for col, receptor in enumerate(receptor_cols):
    x_ctx = mean_ctx.loc[common_ctx].astype(float)
    y_ctx = df_rec_ctx.loc[common_ctx, receptor].astype(float)
    mask_ctx = x_ctx.notna() & y_ctx.notna()

    x_sub = mean_sub.loc[common_sub].astype(float)
    y_sub = df_rec_sub.loc[common_sub, receptor].astype(float)
    mask_sub = x_sub.notna() & y_sub.notna()

    # Combine cortical and subcortical data for regression and correlation
    x_all = pd.concat([x_ctx[mask_ctx], x_sub[mask_sub]])
    y_all = pd.concat([y_ctx[mask_ctx], y_sub[mask_sub]])
    rho, _ = stats.spearmanr(x_all, y_all)

    #correlation plot for cortical regions
    fig.add_trace(go.Scatter(
        x=x_ctx[mask_ctx], y=y_ctx[mask_ctx],
        mode='markers',
        marker=dict(color=color_ctx, size=7, opacity=0.8),
        text=common_ctx[mask_ctx],
        hovertemplate='<b>%{text}</b><br>Δ vol/month: %{x:.4f}<br>' + f'{receptor}: %{{y:.3f}}<extra></extra>',
        name='Cortical',
        showlegend=(col == 0),
        legendgroup='Cortical',
    ), row=1, col=col+1)

    # subcortical correlation plot
    fig.add_trace(go.Scatter(
        x=x_sub[mask_sub], y=y_sub[mask_sub],
        mode='markers',
        marker=dict(color=color_sub, size=7, opacity=0.8),
        text=common_sub[mask_sub],
        hovertemplate='<b>%{text}</b><br>Δ vol/month: %{x:.4f}<br>' + f'{receptor}: %{{y:.3f}}<extra></extra>',
        name='Subcortical',
        showlegend=(col == 0),
        legendgroup='Subcortical',
    ), row=1, col=col+1)

    # Add regression line using the combined data for cortical and subcortical regions to get a single regression line that represents the overall trend across all regions
    m, b = np.polyfit(x_all, y_all, 1)
    x_sorted = np.sort(x_all)
    fig.add_trace(go.Scatter(
        x=x_sorted, y=m * x_sorted + b,
        mode='lines',
        line=dict(color='green', width=1.5),
        showlegend=False,
        hoverinfo='skip',
    ), row=1, col=col+1)

    
    fig.layout.annotations[col].text = f'{receptor} | ρ={rho:.3f}'

fig.update_layout(
    title='Spearman ρ: Δ gray volume per month vs receptor density',
    height=500,
    width=1200,
    template='plotly_white',
)

fig.update_xaxes(title_text='Δ gray volume (mm³/month)')
fig.update_yaxes(title_text='z-score', col=1)

fig.show()
fig.write_html(r"C:\Users\laiam\OneDrive\Escritorio\Practicas Canada\neurotransmitters\data\volume_correlation\deltavol_correlation_glutamate_permonth.html")