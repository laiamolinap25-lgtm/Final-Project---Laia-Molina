import pandas as pd
#We obtained 62 cortical regions (31 per hemisphere) because the atlas used is the DKT parcellation in volumetric form, 
# not the classic Desikan–Killiany atlas that yields 68 regions. In addition, when the atlas is resampled to a volumetric space, 
# some very small or surface-defined cortical labels may contain no voxels and therefore do
# not appear in the final dataset. As a result, the  atlas used for parcellation contains 31 cortical regions per hemisphere (62 total).
-
path = r"C:\Users\laiam\OneDrive\Escritorio\Practicas Canada\neurotransmitters\data\PET_parcellated\dkt\DKT_receptors_table.csv"
df = pd.read_csv(path)

cortical_mask = df["region"].str.startswith("ctx-lh") | df["region"].str.startswith("ctx-rh")
df_cortical = df[cortical_mask].copy()


print("Total cortical regions:", len(df_cortical))
print("Left hemisphere regions:", df_cortical["region"].str.startswith("ctx-lh").sum())
print("Right hemisphere regions:", df_cortical["region"].str.startswith("ctx-rh").sum())


duplicates = df_cortical["region"][df_cortical["region"].duplicated(keep=False)]
print("\nDuplicated cortical labels (if any):")
print(duplicates.sort_values().unique())


unexpected = df_cortical[
    ~(df_cortical["region"].str.startswith("ctx-lh-") |
      df_cortical["region"].str.startswith("ctx-rh-"))
]

print("\nRegions with unexpected naming (if any):")
print(unexpected["region"].tolist())