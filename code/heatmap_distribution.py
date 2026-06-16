import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------
path = r"C:\Users\laiam\OneDrive\Escritorio\Practicas Canada\neurotransmitters\data\PET_parcellated\dkt\DKT_receptors_table_cortical.csv"
df = pd.read_csv(path)

regions = df["region"].tolist()
receptors = [c for c in df.columns if c != "region"]

Z = df[receptors].values

# ------------------------------------------------------------------
# Order regions: LH first, RH second
# ------------------------------------------------------------------
lh = [i for i,r in enumerate(regions) if r.startswith("ctx-lh")]
rh = [i for i,r in enumerate(regions) if r.startswith("ctx-rh")]
order = lh + rh

Z = Z[order]
regions = [regions[i] for i in order]

# ------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------
plt.figure(figsize=(1 + 0.7*len(receptors), 10))

v = np.max(np.abs(Z))  # symmetric scale around zero

im = plt.imshow(Z, aspect="auto", interpolation="nearest",
                cmap="coolwarm", vmin=-v, vmax=v)

plt.colorbar(im, label="Receptor density (z-score)")

plt.xticks(range(len(receptors)), receptors, rotation=45, ha="right")
plt.yticks([])

plt.title("Cortical neurotransmitter receptor distribution (DKT)")
plt.tight_layout()
plt.show()