from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
from brainspace.null_models import SpinPermutations

from config import biomarkers, receptors_csv
from analysis import (
    read_file,
    get_cols,
    calc_baseline,
    calc_delta,
    calc_delta_declining,
    normalize_ctx_index,
)


n_spins = 1000
random_state = 42
alpha = 0.05

dkt_dir = Path(receptors_csv).parent

possible_significant_files = [
    dkt_dir / "results analysis" / "correlations lists.xlsx",
    dkt_dir / "results analysis" / "significant_correlations_with_direction.csv",
    dkt_dir / "correlations lists.xlsx",
    dkt_dir / "significant_correlations_with_direction.csv",
    Path("correlations lists.xlsx"),
    Path("significant_correlations_with_direction.csv"),
]

coords_file = dkt_dir / "dkt_cortical_coordinates.csv"

output_file = (
    dkt_dir
    / "results analysis"
    / "significant_correlations_with_spin_test.csv"
)

output_file.parent.mkdir(parents=True, exist_ok=True)



def to_numeric_series(series):
    """
    Converts numeric columns safely, including European decimal commas.
    Example: '0,402071037' -> 0.402071037
    """
    return pd.to_numeric(
        series.astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    )


def clean_boolean_series(series):
    """
    Converts TRUE/FALSE-like values into booleans.
    """
    return series.astype(str).str.strip().str.lower().isin(
        ["true", "1", "yes", "y"]
    )

def find_existing_file(possible_files):
    for file_path in possible_files:
        if file_path.exists():
            return file_path

    raise FileNotFoundError(
        "No significant correlations file was found.\n"
        "Checked these paths:\n"
        + "\n".join(str(p) for p in possible_files)
    )


def read_table_auto(path):
    path = Path(path)

    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)

    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, sep=";")


def read_receptor_table(path):
    """
    Reads receptor table.
    Your file is semicolon-separated and uses decimal commas.
    """
    df = pd.read_csv(path, sep=";", decimal=",")

    if "region" not in df.columns:
        raise ValueError(
            "The receptor table must contain a 'region' column after reading.\n"
            f"Available columns are:\n{df.columns.tolist()}"
        )

    non_numeric_cols = ["label", "region"]

    for col in df.columns:
        if col not in non_numeric_cols:
            df[col] = to_numeric_series(df[col])

    return df


def load_significant_correlations(path):
    df = read_table_auto(path)

    required_cols = {
        "biomarker",
        "correlation_type",
        "receptor",
        "region_type",
        "rho",
    }

    missing_cols = required_cols - set(df.columns)

    if missing_cols:
        raise ValueError(
            "The significant correlations file is missing required columns:\n"
            f"{sorted(missing_cols)}\n\n"
            f"Available columns:\n{df.columns.tolist()}"
        )

    df = df.copy()

    df["region_type"] = df["region_type"].astype(str).str.strip().str.lower()

    df = df[df["region_type"] == "cortical"].copy()

    if "sig_fdr" in df.columns:
        df = df[clean_boolean_series(df["sig_fdr"])].copy()

    if df.empty:
        raise ValueError(
            "No cortical significant correlations were found after filtering."
        )

    df["rho"] = to_numeric_series(df["rho"])

    if "pvalue" in df.columns:
        df["pvalue"] = to_numeric_series(df["pvalue"])

    if "pvalue_fdr" in df.columns:
        df["pvalue_fdr"] = to_numeric_series(df["pvalue_fdr"])

    return df


def load_coordinates(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Coordinate file not found:\n{path}\n\n"
            "Run generate_dkt_cortical_coordinates.py first."
        )

    coords = pd.read_csv(path)

    required_cols = {"region", "x", "y", "z"}
    missing_cols = required_cols - set(coords.columns)

    if missing_cols:
        raise ValueError(
            "The coordinate file is missing required columns:\n"
            f"{sorted(missing_cols)}\n\n"
            f"Available columns:\n{coords.columns.tolist()}"
        )

    coords = coords.copy()
    coords["region"] = coords["region"].astype(str)

    for col in ["x", "y", "z"]:
        coords[col] = to_numeric_series(coords[col])

    if "hemisphere" not in coords.columns:
        coords["hemisphere"] = coords["region"].apply(get_hemisphere)

    return coords



def get_hemisphere(region):
    region = str(region)

    if (
        region.startswith("ctx-lh-")
        or region.startswith("lh_")
        or region.startswith("lh-")
    ):
        return "lh"

    if (
        region.startswith("ctx-rh-")
        or region.startswith("rh_")
        or region.startswith("rh-")
    ):
        return "rh"

    return None


def to_standard_region_name(region):
    region = str(region)

    if region.startswith("ctx-lh-") or region.startswith("ctx-rh-"):
        return region

    if region.startswith("lh_"):
        return "ctx-lh-" + region.replace("lh_", "", 1).replace("_", "-")

    if region.startswith("rh_"):
        return "ctx-rh-" + region.replace("rh_", "", 1).replace("_", "-")

    if region.startswith("lh-"):
        return "ctx-lh-" + region.replace("lh-", "", 1)

    if region.startswith("rh-"):
        return "ctx-rh-" + region.replace("rh-", "", 1)

    return region


def load_biomarker_groups(biomarker_cfg):
    df_con = read_file(
        biomarker_cfg["file_con"],
        biomarker_cfg["file_format"],
    )

    df_mci = read_file(
        biomarker_cfg["file_mci"],
        biomarker_cfg["file_format"],
    )

    df_ad = read_file(
        biomarker_cfg["file_ad"],
        biomarker_cfg["file_format"],
    )

    return df_con, df_mci, df_ad


def get_biomarker_cortical_series(biomarker_name, correlation_type):
    biomarker_cfg = biomarkers[biomarker_name]

    df_con, df_mci, df_ad = load_biomarker_groups(biomarker_cfg)

    cols_ctx, _ = get_cols(df_con, biomarker_cfg)

    if len(cols_ctx) == 0:
        raise ValueError(
            f"No cortical columns found for biomarker: {biomarker_name}"
        )

    if correlation_type == "delta_all":
        values_ctx = pd.concat(
            [
                calc_delta(df_con, cols_ctx),
                calc_delta(df_mci, cols_ctx),
                calc_delta(df_ad, cols_ctx),
            ]
        )

    elif correlation_type == "baseline_all":
        values_ctx = pd.concat(
            [
                calc_baseline(df_con, cols_ctx),
                calc_baseline(df_mci, cols_ctx),
                calc_baseline(df_ad, cols_ctx),
            ]
        )

    elif correlation_type == "delta_declining":
        dx_col = biomarker_cfg.get("declining_dx_col", "dementia_dx")

        con_dec = df_con[
            (df_con["dementia_dx_bl"] == "CON")
            & (df_con[dx_col].isin(["MCI", "AD"]))
        ]

        mci_dec = df_mci[
            (df_mci["dementia_dx_bl"] == "MCI")
            & (df_mci[dx_col] == "AD")
        ]

        values_ctx = pd.concat(
            [
                calc_delta_declining(con_dec, df_con, cols_ctx),
                calc_delta_declining(mci_dec, df_mci, cols_ctx),
            ]
        )

    elif correlation_type == "baseline_declining":
        dx_col = biomarker_cfg.get("declining_dx_col", "dementia_dx")

        con_dec = df_con[
            (df_con["dementia_dx_bl"] == "CON")
            & (df_con[dx_col].isin(["MCI", "AD"]))
        ]

        mci_dec = df_mci[
            (df_mci["dementia_dx_bl"] == "MCI")
            & (df_mci[dx_col] == "AD")
        ]

        values_ctx = pd.concat(
            [
                calc_baseline(con_dec, cols_ctx),
                calc_baseline(mci_dec, cols_ctx),
            ]
        )

    else:
        raise ValueError(f"Unknown correlation_type: {correlation_type}")

    mean_ctx = values_ctx.mean(axis=0, skipna=True)
    mean_ctx = normalize_ctx_index(mean_ctx, biomarker_cfg)

    mean_ctx = to_numeric_series(mean_ctx)

    return mean_ctx


def get_receptor_cortical_series(biomarker_name, receptor_name):
    biomarker_cfg = biomarkers[biomarker_name]

    receptor_df = read_receptor_table(receptors_csv)

    if receptor_name not in receptor_df.columns:
        raise ValueError(
            f"Receptor '{receptor_name}' not found in receptor table.\n"
            f"Available columns:\n{receptor_df.columns.tolist()}"
        )

    receptor_df = receptor_df[
        receptor_df["region"].astype(str).str.startswith("ctx-", na=False)
    ].copy()

    if biomarker_cfg["ctx_index_mode"] == "region_key":
        receptor_df["region_key"] = (
            receptor_df["region"]
            .str.replace("ctx-", "", regex=False)
            .str.replace("-", "_", regex=False)
        )

        receptor_series = receptor_df.set_index("region_key")[receptor_name]

    else:
        receptor_series = receptor_df.set_index("region")[receptor_name]

    receptor_series = to_numeric_series(receptor_series)

    return receptor_series


def prepare_common_data(receptor_series, biomarker_series, coords):
    receptor_df = pd.DataFrame(
        {
            "region": receptor_series.index.astype(str),
            "receptor_value": receptor_series.values,
        }
    )

    biomarker_df = pd.DataFrame(
        {
            "region": biomarker_series.index.astype(str),
            "biomarker_value": biomarker_series.values,
        }
    )

    common_df = receptor_df.merge(
        biomarker_df,
        on="region",
        how="inner",
    )

    if common_df.empty:
        raise ValueError(
            "No common regions found between receptor and biomarker series."
        )

    common_df["receptor_value"] = to_numeric_series(
        common_df["receptor_value"]
    )

    common_df["biomarker_value"] = to_numeric_series(
        common_df["biomarker_value"]
    )

    common_df["standard_region"] = common_df["region"].apply(
        to_standard_region_name
    )

    common_df["hemisphere"] = common_df["standard_region"].apply(
        get_hemisphere
    )

    coords = coords.copy()
    coords["standard_region"] = coords["region"].apply(
        to_standard_region_name
    )

    common_df = common_df.merge(
        coords[["standard_region", "x", "y", "z"]],
        on="standard_region",
        how="left",
    )

    common_df = common_df.dropna(
        subset=[
            "receptor_value",
            "biomarker_value",
            "x",
            "y",
            "z",
            "hemisphere",
        ]
    )

    common_df = common_df[
        common_df["hemisphere"].isin(["lh", "rh"])
    ].copy()

    if common_df.empty:
        raise ValueError(
            "No valid cortical regions remained after matching coordinates."
        )

    return common_df


def normalize_points_to_sphere(points):
    points = np.asarray(points, dtype=float)

    center = points.mean(axis=0)
    points = points - center

    norm = np.linalg.norm(points, axis=1)

    if np.any(norm == 0):
        raise ValueError(
            "Some coordinate points have zero norm after centering."
        )

    points = points / norm[:, None]

    return points


def compute_spin_test(common_df):
    lh_df = common_df[common_df["hemisphere"] == "lh"].copy()
    rh_df = common_df[common_df["hemisphere"] == "rh"].copy()

    if len(lh_df) < 4 or len(rh_df) < 4:
        raise ValueError(
            "Not enough left/right cortical regions for spin test.\n"
            f"lh regions: {len(lh_df)}\n"
            f"rh regions: {len(rh_df)}"
        )

    receptor_lh = lh_df["receptor_value"].to_numpy(dtype=float)
    receptor_rh = rh_df["receptor_value"].to_numpy(dtype=float)

    biomarker_lh = lh_df["biomarker_value"].to_numpy(dtype=float)
    biomarker_rh = rh_df["biomarker_value"].to_numpy(dtype=float)

    coords_lh = lh_df[["x", "y", "z"]].to_numpy(dtype=float)
    coords_rh = rh_df[["x", "y", "z"]].to_numpy(dtype=float)

    coords_lh = normalize_points_to_sphere(coords_lh)
    coords_rh = normalize_points_to_sphere(coords_rh)

    receptor_values = np.concatenate([receptor_lh, receptor_rh])
    biomarker_values = np.concatenate([biomarker_lh, biomarker_rh])

    observed_rho, _ = spearmanr(
        receptor_values,
        biomarker_values,
    )

    spin_model = SpinPermutations(
        n_rep=n_spins,
        random_state=random_state,
    )

    spin_model.fit(
        coords_lh,
        points_rh=coords_rh,
    )

    spun_lh, spun_rh = spin_model.randomize(
        receptor_lh,
        x_rh=receptor_rh,
    )

    null_rhos = []

    for spin_index in range(n_spins):
        spun_receptor = np.concatenate(
            [
                spun_lh[spin_index],
                spun_rh[spin_index],
            ]
        )

        spin_rho, _ = spearmanr(
            spun_receptor,
            biomarker_values,
        )

        null_rhos.append(spin_rho)

    null_rhos = np.asarray(null_rhos, dtype=float)

    p_spin = (
        np.sum(np.abs(null_rhos) >= abs(observed_rho)) + 1
    ) / (n_spins + 1)

    return {
        "rho_spin_observed": float(observed_rho),
        "p_spin": float(p_spin),
        "spin_significant": bool(p_spin < alpha),
        "n_spin_regions": int(len(common_df)),
        "n_spin_lh": int(len(lh_df)),
        "n_spin_rh": int(len(rh_df)),
        "n_spins": int(n_spins),
    }


def run_spin_for_row(row, coords):
    biomarker_name = row["biomarker"]
    correlation_type = row["correlation_type"]
    receptor_name = row["receptor"]

    biomarker_series = get_biomarker_cortical_series(
        biomarker_name=biomarker_name,
        correlation_type=correlation_type,
    )

    receptor_series = get_receptor_cortical_series(
        biomarker_name=biomarker_name,
        receptor_name=receptor_name,
    )

    common_df = prepare_common_data(
        receptor_series=receptor_series,
        biomarker_series=biomarker_series,
        coords=coords,
    )

    return compute_spin_test(common_df)


def main():
    significant_file = find_existing_file(possible_significant_files)

    print("\nSpin-test sensitivity analysis")
    print("=" * 72)
    print(f"Significant correlations file:\n{significant_file}")
    print(f"Coordinates file:\n{coords_file}")
    print(f"Output file:\n{output_file}")
    print(f"n_spins: {n_spins}")
    print("=" * 72)

    significant_df = load_significant_correlations(significant_file)
    coords = load_coordinates(coords_file)

    print(f"\nCortical significant correlations selected: {len(significant_df)}")

    output_rows = []

    for row_number, (_, row) in enumerate(
        significant_df.iterrows(),
        start=1,
    ):
        print(
            f"\n[{row_number}/{len(significant_df)}] "
            f"{row['biomarker']} | {row['correlation_type']} | "
            f"{row.get('neurotransmitter', 'NA')} | {row['receptor']}"
        )

        row_result = row.to_dict()

        try:
            spin_result = run_spin_for_row(row, coords)
            row_result.update(spin_result)
            row_result["spin_error"] = ""

            print(
                f"rho = {spin_result['rho_spin_observed']:.4f} | "
                f"p_spin = {spin_result['p_spin']:.4f} | "
                f"spin significant = {spin_result['spin_significant']}"
            )

        except Exception as exc:
            row_result.update(
                {
                    "rho_spin_observed": np.nan,
                    "p_spin": np.nan,
                    "spin_significant": False,
                    "n_spin_regions": np.nan,
                    "n_spin_lh": np.nan,
                    "n_spin_rh": np.nan,
                    "n_spins": n_spins,
                    "spin_error": str(exc),
                }
            )

            print(f"ERROR: {exc}")

        output_rows.append(row_result)

    spin_df = pd.DataFrame(output_rows)

    valid_mask = spin_df["p_spin"].notna()

    spin_df["q_spin_fdr"] = np.nan
    spin_df["spin_fdr_significant"] = False

    if valid_mask.any():
        reject, q_values, _, _ = multipletests(
            spin_df.loc[valid_mask, "p_spin"].to_numpy(),
            alpha=alpha,
            method="fdr_bh",
        )

        spin_df.loc[valid_mask, "q_spin_fdr"] = q_values
        spin_df.loc[valid_mask, "spin_fdr_significant"] = reject

    spin_df.to_csv(output_file, index=False, encoding="utf-8")

    print("\nDone.")
    print(f"Saved output to:\n{output_file}")

    print("\nSummary")
    print("-" * 72)
    print(f"Rows tested: {len(spin_df)}")
    print(f"Valid spin tests: {int(valid_mask.sum())}")
    print(f"p_spin < {alpha}: {int(spin_df['spin_significant'].sum())}")
    print(f"q_spin_fdr < {alpha}: {int(spin_df['spin_fdr_significant'].sum())}")
    print("-" * 72)


if __name__ == "__main__":
    main()