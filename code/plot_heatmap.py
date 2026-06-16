import numpy as np
import matplotlib.pyplot as plt

# Paths -------------------------------------------------------------
matrix_path = "data/PET_parcellated/dkt/DKT_receptors_z_matrix.csv"
names_path  = "data/PET_parcellated/dkt/DKT_receptors_names.txt"

# Load data ---------------------------------------------------------
Z = np.loadtxt(matrix_path, delimiter=",", dtype=float)

with open(names_path, "r") as f:
    receptors = [line.strip() for line in f]

n_regions, n_receptors = Z.shape

print("Matrix shape:", Z.shape)
print("Receptors:", receptors)

# Plot --------------------------------------------------------------
plt.figure(figsize=(12, 8))
im = plt.imshow(Z, aspect="auto", interpolation="nearest")

plt.colorbar(im, label="Z-score")

plt.xticks(np.arange(n_receptors), receptors, rotation=45, ha="right")
plt.yticks(np.arange(n_regions), np.arange(1, n_regions+1))

plt.xlabel("Receptors")
plt.ylabel("Cortical regions")
plt.title("Cortical neurotransmitter receptor atlas (z-score)")

plt.tight_layout()

# Save --------------------------------------------------------------
outpath = "data/PET_parcellated/dkt/DKT_heatmap.png"
plt.savefig(outpath, dpi=300)
plt.show()

print("Saved:", outpath)
