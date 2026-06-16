import os
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# NT receptor matrix exploratory figures
#
# Outputs:
#   1. Boxplot of receptor density z-scores
#   2. Spearman receptor-receptor spatial correlation heatmap
# =============================================================================


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

input_file = r"C:\Users\laiam\OneDrive\Escritorio\Practicas Canada\neurotransmitters\data\PET_parcellated\dkt\DKT_receptors_table_corticalandsubcortical.csv"

output_dir = r"C:\Users\laiam\OneDrive\Escritorio\Practicas Canada\neurotransmitters\data\PET_parcellated\dkt\NT analysis figures"

os.makedirs(output_dir, exist_ok=True)


# -----------------------------------------------------------------------------
# Receptors to include
# -----------------------------------------------------------------------------

receptors_ordered = [
    "D1", "D2", "DAT",
    "VAChT", "M1", "A4B2",
    "GABAa-bz", "GABAa",
    "5HT1a", "5HT1b", "5HT2a", "5HT4", "5HT6", "5HTT",
    "mGluR5", "NMDA"
]


# -----------------------------------------------------------------------------
# Load receptor matrix robustly
# -----------------------------------------------------------------------------

def load_receptor_matrix(path: str) -> pd.DataFrame:
    """
    Loads the receptor matrix.
    Handles both comma-separated and semicolon-separated CSV files.
    """

    df = pd.read_csv(path)

    # If everything was loaded as one single column, retry with semicolon separator
    if df.shape[1] == 1:
        df = pd.read_csv(path, sep=";", decimal=",")

    return df


df = load_receptor_matrix(input_file)


# -----------------------------------------------------------------------------
# Check that all selected receptors exist
# -----------------------------------------------------------------------------

missing_receptors = [r for r in receptors_ordered if r not in df.columns]

if missing_receptors:
    raise ValueError(
        "The following receptors are missing from the receptor matrix:\n"
        f"{missing_receptors}\n\n"
        f"Available columns are:\n{df.columns.tolist()}"
    )


# -----------------------------------------------------------------------------
# Convert receptor columns to numeric
# -----------------------------------------------------------------------------

for receptor in receptors_ordered:
    df[receptor] = (
        df[receptor]
        .astype(str)
        .str.replace(",", ".", regex=False)
    )
    df[receptor] = pd.to_numeric(df[receptor], errors="coerce")


plot_df = df[receptors_ordered].dropna(how="all")


# =============================================================================
# 1. Boxplot of receptor density values
# =============================================================================

plt.figure(figsize=(13, 5))

plot_df.boxplot(
    column=receptors_ordered,
    showfliers=False
)

plt.axhline(0, linestyle="--", linewidth=1)

plt.ylabel("Receptor density z-score")
plt.xlabel("Receptor / transporter")
plt.title("Distribution of regional PET-derived receptor density values")

plt.xticks(rotation=45, ha="right")
plt.tight_layout()

boxplot_path = os.path.join(
    output_dir,
    "receptor_density_boxplot_clean.png"
)

plt.savefig(boxplot_path, dpi=300)
plt.close()

print(f"Boxplot saved in: {boxplot_path}")


# =============================================================================
# 2. Spearman receptor-receptor correlation heatmap
# =============================================================================

corr = plot_df.corr(method="spearman")

corr_matrix_path = os.path.join(
    output_dir,
    "receptor_receptor_spearman_correlation_matrix.csv"
)

corr.to_csv(corr_matrix_path)

plt.figure(figsize=(9, 8))

plt.imshow(
    corr,
    vmin=-1,
    vmax=1,
    cmap="coolwarm"
)

plt.colorbar(label="Spearman correlation")

plt.xticks(
    range(len(receptors_ordered)),
    receptors_ordered,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(receptors_ordered)),
    receptors_ordered
)

plt.title("Spatial Spearman correlation between receptor density maps")

plt.tight_layout()

heatmap_path = os.path.join(
    output_dir,
    "receptor_receptor_spearman_heatmap.png"
)

plt.savefig(heatmap_path, dpi=300)
plt.close()

print(f"Spearman correlation matrix saved in: {corr_matrix_path}")
print(f"Spearman heatmap saved in: {heatmap_path}")