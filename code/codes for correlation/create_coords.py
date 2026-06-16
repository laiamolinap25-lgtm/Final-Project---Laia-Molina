from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import center_of_mass

from config import receptors_csv


base_dir = Path(
    r"C:\Users\laiam\OneDrive\Escritorio\Practicas Canada\neurotransmitters"
)

data_dir = base_dir / "data" / "PET_parcellated" / "dkt"

atlas_path = base_dir / "data" / "aparc.DKTatlas+aseg.deep.withCC_resampled.nii"
labels_path = data_dir / "DKT_labels_order.csv"
output_path = data_dir / "dkt_cortical_coordinates.csv"


def require_file(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")


def find_region_column(df):
    possible_cols = [
        "region",
        "regions",
        "Region",
        "Regions",
        "label",
        "labels",
        "Label",
        "Labels",
        "name",
        "names",
        "Name",
        "Names",
        "Unnamed: 0",
    ]

    for col in possible_cols:
        if col in df.columns:
            return col

    raise ValueError(
        "No region column found in receptor table.\n"
        f"Available columns are:\n{df.columns.tolist()}"
    )


def load_labels(path):
    labels = np.loadtxt(path, delimiter=",", dtype=int).flatten()

    if labels.size == 0:
        raise ValueError("No labels were found in DKT_labels_order.csv.")

    return labels


def main():
    require_file(atlas_path)
    require_file(receptors_csv)
    require_file(labels_path)

    print("Loading atlas...")
    atlas_img = nib.load(str(atlas_path))
    atlas_data = atlas_img.get_fdata()

    print("Loading receptor table...")
    receptor_df = pd.read_csv(receptors_csv, sep=";")

    region_col = find_region_column(receptor_df)
    print(f"Using region column: {region_col}")

    print("Loading labels...")
    labels = load_labels(labels_path)

    if len(labels) != len(receptor_df):
        raise ValueError(
            "The number of labels does not match the number of rows in the receptor table.\n"
            f"labels: {len(labels)}\n"
            f"receptor rows: {len(receptor_df)}\n"
            "Check that DKT_labels_order.csv and DKT_receptors_table_corticalandsubcortical.csv are aligned."
        )

    rows = []

    for label, region in zip(labels, receptor_df[region_col]):
        region = str(region)

        if not region.startswith("ctx-"):
            continue

        mask = atlas_data == label

        if not np.any(mask):
            print(f"Warning: label {label} for region {region} not found in atlas.")
            continue

        voxel_center = center_of_mass(mask)

        world_center = nib.affines.apply_affine(
            atlas_img.affine,
            voxel_center,
        )

        if region.startswith("ctx-lh-"):
            hemisphere = "lh"
        elif region.startswith("ctx-rh-"):
            hemisphere = "rh"
        else:
            hemisphere = ""

        rows.append(
            {
                "region": region,
                "hemisphere": hemisphere,
                "label": int(label),
                "x": float(world_center[0]),
                "y": float(world_center[1]),
                "z": float(world_center[2]),
            }
        )

    coords_df = pd.DataFrame(rows)

    if coords_df.empty:
        raise ValueError(
            "No cortical regions were found. "
            "Check whether your receptor region names start with 'ctx-'."
        )

    coords_df.to_csv(output_path, index=False)

    print("\nDone.")
    print(f"Saved coordinates to:\n{output_path}")
    print(f"Number of cortical regions: {len(coords_df)}")
    print(coords_df.head())


if __name__ == "__main__":
    main()