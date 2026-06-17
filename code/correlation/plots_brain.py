# =============================================================================
# generate_receptor_volumes_and_pngs.py
#
# Generate volumetric NIfTI maps and PNG figures of neurotransmitter receptor
# density using DKT cortical + subcortical parcellated receptor values.
# =============================================================================

from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib

from nilearn import plotting
from matplotlib.colors import LinearSegmentedColormap


# =============================================================================
# Configuration
# =============================================================================

base_dir = Path(
    r"C:\Users\laiam\OneDrive\Escritorio\Practicas Canada\neurotransmitters"
)

data_dir = base_dir / "data" / "PET_parcellated" / "dkt"

atlas_path = (
    base_dir
    / "data"
    / "aparc.DKTatlas+aseg.deep.withCC_resampled.nii"
)

receptor_table_path = data_dir / "DKT_receptors_table_corticalandsubcortical.csv"
labels_path = data_dir / "DKT_labels_order.csv"

output_dir = data_dir / "receptor_maps_corticalandsubcortical"
nifti_dir = output_dir / "nifti"
png_dir = output_dir / "png"

nifti_dir.mkdir(parents=True, exist_ok=True)
png_dir.mkdir(parents=True, exist_ok=True)

cmap = LinearSegmentedColormap.from_list(
    "receptor_density",
    [
        "#253a6b",   # dark blue
        "#4daf4a",   # green
        "#ffffff",   # white
        "#fdae61",   # orange
        "#d73027",   # red
    ],
)

cut_coords = [-30, -20, -10, 0, 10, 20, 30, 40]


# =============================================================================
# Helper functions
# =============================================================================

def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def load_receptor_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Remove non-receptor columns
    columns_to_drop = [
        "region",
        "regions",
        "label",
        "labels",
        "id",
        "index",
        "structure",
        "structures",
        "name",
        "names",
    ]

    existing_cols = [c for c in columns_to_drop if c in df.columns]

    if existing_cols:
        df = df.drop(columns=existing_cols)

    # Keep only numeric receptor columns
    df = df.select_dtypes(include=[np.number])

    if df.empty:
        raise ValueError("No numeric receptor columns were found.")

    return df


def load_labels(path: Path) -> np.ndarray:
    labels = np.loadtxt(path, delimiter=",", dtype=int).flatten()

    if labels.size == 0:
        raise ValueError("No atlas labels were found.")

    return labels


def receptor_vector_to_volume(
    values: np.ndarray,
    labels: np.ndarray,
    atlas_data: np.ndarray,
) -> np.ndarray:

    if len(values) != len(labels):
        raise ValueError(
            f"Receptor vector length ({len(values)}) does not match "
            f"number of atlas labels ({len(labels)})."
        )

    output_data = np.zeros(atlas_data.shape, dtype=np.float32)

    for label, value in zip(labels, values):
        output_data[atlas_data == label] = float(value)

    return output_data


def save_receptor_nifti(
    receptor_name: str,
    values: np.ndarray,
    labels: np.ndarray,
    atlas_img,
    atlas_data: np.ndarray,
) -> Path:

    output_data = receptor_vector_to_volume(
        values=values,
        labels=labels,
        atlas_data=atlas_data,
    )

    output_img = nib.Nifti1Image(
        output_data,
        affine=atlas_img.affine,
        header=atlas_img.header,
    )

    output_path = nifti_dir / f"{receptor_name}_DKT_cortical_subcortical_z.nii.gz"

    nib.save(output_img, output_path)

    print(f"Saved NIfTI: {output_path}")

    return output_path


def save_receptor_png(
    receptor_name: str,
    nifti_path: Path,
    values: np.ndarray,
) -> None:

    vmax = float(np.nanmax(np.abs(values)))

    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0

    output_path = png_dir / f"{receptor_name}_DKT_cortical_subcortical_z.png"

    display = plotting.plot_stat_map(
        str(nifti_path),
        display_mode="z",
        cut_coords=cut_coords,
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
        colorbar=True,
        threshold=None,
        title=f"{receptor_name} receptor density z-score",
        annotate=True,
        black_bg=False,
    )

    display.savefig(str(output_path), dpi=300)
    display.close()

    print(f"Saved PNG:   {output_path}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print("\nGenerating cortical + subcortical receptor maps\n")

    require_file(atlas_path)
    require_file(receptor_table_path)
    require_file(labels_path)

    receptor_df = load_receptor_table(receptor_table_path)
    labels = load_labels(labels_path)

    if receptor_df.shape[0] != len(labels):
        raise ValueError(
            "The number of rows in the receptor table does not match the "
            "number of atlas labels.\n"
            f"Receptor table rows: {receptor_df.shape[0]}\n"
            f"Atlas labels: {len(labels)}\n"
            "Check that DKT_receptors_table_corticalandsubcortical.csv and "
            "DKT_labels_order.csv are aligned."
        )

    atlas_img = nib.load(str(atlas_path))
    atlas_data = atlas_img.get_fdata()

    atlas_labels = np.unique(atlas_data).astype(int)
    atlas_labels = atlas_labels[atlas_labels != 0]

    missing_labels = sorted(set(labels) - set(atlas_labels))

    if missing_labels:
        raise ValueError(
            f"{len(missing_labels)} labels from DKT_labels_order.csv were not "
            f"found in the atlas image. Examples: {missing_labels[:10]}"
        )

    print(f"Number of regions: {len(labels)}")
    print(f"Number of receptors: {receptor_df.shape[1]}")
    print(f"Receptors: {receptor_df.columns.tolist()}\n")

    for receptor in receptor_df.columns:
        values = receptor_df[receptor].to_numpy(dtype=float)

        nifti_path = save_receptor_nifti(
            receptor_name=receptor,
            values=values,
            labels=labels,
            atlas_img=atlas_img,
            atlas_data=atlas_data,
        )

        save_receptor_png(
            receptor_name=receptor,
            nifti_path=nifti_path,
            values=values,
        )

    print("\nFinished.")
    print(f"NIfTI files saved in: {nifti_dir}")
    print(f"PNG figures saved in: {png_dir}")


if __name__ == "__main__":
    main()