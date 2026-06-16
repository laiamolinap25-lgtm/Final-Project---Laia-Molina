import os

import numpy as np
import pandas as pd

from pathlib import Path
from scipy import stats
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
from brainspace.null_models import SpinPermutations

from config import (
    timepoint_to_months,
    subcortical_cols,
    receptors_csv,
)

# ============================================================
# PARAMETERS
# ============================================================

alpha = 0.05
n_spins = 1000
random_state = 42

dkt_dir = Path(receptors_csv).parent
coords_file = dkt_dir / "dkt_cortical_coordinates.csv"
results_file = r"C:\Users\laiam\OneDrive\Escritorio\Practicas Canada\neurotransmitters\data\PET_parcellated\dkt\results analysis\all_correlations.csv"


# ============================================================
# FILE / COLUMN HELPERS
# ============================================================

def read_file(path: str, fmt: str) -> pd.DataFrame:
    if fmt == "xlsx":
        return pd.read_excel(path)
    return pd.read_csv(path)


def get_cols(df: pd.DataFrame, cfg: dict) -> tuple:
    cols_ctx = cfg["ctx_col_filter"](df.columns.tolist())
    cols_sub = (
        [c for c in subcortical_cols if c in df.columns]
        if cfg["has_subcortical"]
        else []
    )
    return cols_ctx, cols_sub


# ============================================================
# BASELINE / DELTA
# ============================================================

def calc_delta(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Delta per year = (last - first) / years"""
    df = df.copy()
    df["tp_months"] = df["timepoint"].map(timepoint_to_months)
    df = df.sort_values("tp_months")
    df_bl = df.groupby("patient_id").first()[cols + ["tp_months"]]
    df_last = df.groupby("patient_id").last()[cols + ["tp_months"]]
    months_diff = (df_last["tp_months"] - df_bl["tp_months"]).replace(0, float("nan"))
    years_diff = months_diff / 12
    return (df_last[cols] - df_bl[cols]).div(years_diff, axis=0)


def calc_delta_declining(df_decline: pd.DataFrame, df_all: pd.DataFrame, cols: list) -> pd.DataFrame:
    ids = df_decline["patient_id"].unique()
    df_all = df_all[df_all["patient_id"].isin(ids)].copy()
    df_all["tp_months"] = df_all["timepoint"].map(timepoint_to_months)
    df_all = df_all.sort_values("tp_months")
    df_bl = df_all.groupby("patient_id").first()[cols + ["tp_months"]]
    df_last = df_all.groupby("patient_id").last()[cols + ["tp_months"]]
    months_diff = (df_last["tp_months"] - df_bl["tp_months"]).replace(0, float("nan"))
    years_diff = months_diff / 12
    return (df_last[cols] - df_bl[cols]).div(years_diff, axis=0)


def calc_baseline(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    df = df.copy()
    df["tp_months"] = df["timepoint"].map(timepoint_to_months)
    df = df.sort_values("tp_months")
    return df.groupby("patient_id").first()[cols]


# ============================================================
# INDEX NORMALIZATION
# ============================================================

def normalize_ctx_index(mean_ctx: pd.Series, cfg: dict) -> pd.Series:
    mean_ctx = mean_ctx.copy()
    mean_ctx.index = cfg["ctx_index_transform"](mean_ctx.index)
    return mean_ctx


def normalize_sub_index(mean_sub: pd.Series) -> pd.Series:
    mean_sub = mean_sub.copy()
    mean_sub.index = mean_sub.index.str.replace("-", " ").str.title().str.replace(" ", "-")
    mean_sub.index = mean_sub.index.map(lambda x: {
        "Left-Accumbens-Area": "Left-Accumbens-area",
        "Right-Accumbens-Area": "Right-Accumbens-area",
        "Left-Ventraldc": "Left-VentralDC",
        "Right-Ventraldc": "Right-VentralDC",
    }.get(x, x))
    return mean_sub


# ============================================================
# RECEPTOR TABLE
# ============================================================

def load_receptors(ctx_index_mode: str) -> tuple:
    df_rec = pd.read_csv(receptors_csv, sep=";", decimal=",")

    df_rec.columns = df_rec.columns.str.strip()

    if "region" not in df_rec.columns:
        raise KeyError(
            "Column 'region' was not found in the receptor matrix.\n"
            f"Available columns are: {df_rec.columns.tolist()}"
        )

    if ctx_index_mode == "region_key":
        df_rec_ctx = df_rec[df_rec["region"].str.startswith("ctx-", na=False)].copy()
        df_rec_ctx["region_key"] = (
            df_rec_ctx["region"]
            .str.replace("ctx-", "", regex=False)
            .str.replace("-", "_", regex=False)
        )
        df_rec_ctx = df_rec_ctx.set_index("region_key")
    else:
        df_rec_ctx = (
            df_rec[df_rec["region"].str.startswith("ctx-", na=False)]
            .copy()
            .set_index("region")
        )

    df_rec_sub = (
        df_rec[~df_rec["region"].str.startswith("ctx-", na=False)]
        .dropna(subset=["region"])
        .copy()
        .set_index("region")
    )

    return df_rec_ctx, df_rec_sub

# ============================================================
# REGION HELPERS (for spin test coordinates)
# ============================================================

def get_hemisphere(region):
    region = str(region)
    if region.startswith(("ctx-lh-", "lh_", "lh-")):
        return "lh"
    if region.startswith(("ctx-rh-", "rh_", "rh-")):
        return "rh"
    return None


def load_coordinates(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Coordinate file not found:\n{path}\n\nRun create_coords.py first."
        )
    coords = pd.read_csv(path)
    coords["region"] = coords["region"].astype(str)
    for col in ["x", "y", "z"]:
        coords[col] = pd.to_numeric(coords[col], errors="coerce")
    return coords


# ============================================================
# SPIN TEST
# ============================================================

def normalize_points_to_sphere(points):
    points = np.asarray(points, dtype=float)
    center = points.mean(axis=0)
    points = points - center
    norm = np.linalg.norm(points, axis=1)
    if np.any(norm == 0):
        raise ValueError("Some coordinate points have zero norm after centering.")
    points = points / norm[:, None]
    return points


def compute_spin_pvalue(receptor_vals, biomarker_vals, region_names, coords, observed_rho):
    """Spin-test p-value for one cortical receptor-biomarker pair."""
    df = pd.DataFrame({
        "region": region_names.astype(str),
        "receptor_value": np.asarray(receptor_vals, dtype=float),
        "biomarker_value": np.asarray(biomarker_vals, dtype=float),
    })
    df["hemisphere"] = df["region"].apply(get_hemisphere)
    df = df.merge(coords[["region", "x", "y", "z"]], on="region", how="left")
    df = df.dropna(subset=["receptor_value", "biomarker_value", "x", "y", "z", "hemisphere"])
    df = df[df["hemisphere"].isin(["lh", "rh"])].copy()

    lh_df = df[df["hemisphere"] == "lh"]
    rh_df = df[df["hemisphere"] == "rh"]
    if len(lh_df) < 4 or len(rh_df) < 4:
        return float("nan")

    receptor_lh = lh_df["receptor_value"].to_numpy(dtype=float)
    receptor_rh = rh_df["receptor_value"].to_numpy(dtype=float)
    biomarker_lh = lh_df["biomarker_value"].to_numpy(dtype=float)
    biomarker_rh = rh_df["biomarker_value"].to_numpy(dtype=float)

    coords_lh = normalize_points_to_sphere(lh_df[["x", "y", "z"]].to_numpy(dtype=float))
    coords_rh = normalize_points_to_sphere(rh_df[["x", "y", "z"]].to_numpy(dtype=float))
    biomarker_values = np.concatenate([biomarker_lh, biomarker_rh])

    spin_model = SpinPermutations(n_rep=n_spins, random_state=random_state)
    spin_model.fit(coords_lh, points_rh=coords_rh)
    spun_lh, spun_rh = spin_model.randomize(receptor_lh, x_rh=receptor_rh)

    null_rhos = []
    for i in range(n_spins):
        spun_receptor = np.concatenate([spun_lh[i], spun_rh[i]])
        spin_rho, _ = spearmanr(spun_receptor, biomarker_values)
        null_rhos.append(spin_rho)

    null_rhos = np.asarray(null_rhos, dtype=float)
    p_spin = (np.sum(np.abs(null_rhos) >= abs(observed_rho)) + 1) / (n_spins + 1)
    return float(p_spin)


# ============================================================
# BIOMARKER MEANS
# ============================================================

def build_biomarker_means(biomarker_cfg, correlation_type, cols_ctx, cols_sub, has_sub,
                          df_con, df_mci, df_ad):
    if correlation_type == "delta_all":
        v_ctx = pd.concat([calc_delta(d, cols_ctx) for d in [df_con, df_mci, df_ad]])
        v_sub = pd.concat([calc_delta(d, cols_sub) for d in [df_con, df_mci, df_ad]]) if has_sub else None

    elif correlation_type == "delta_declining":
        dx_col = biomarker_cfg.get("declining_dx_col", "dementia_dx")
        con_dec = df_con[(df_con["dementia_dx_bl"] == "CON") & (df_con[dx_col].isin(["MCI", "AD"]))]
        mci_dec = df_mci[(df_mci["dementia_dx_bl"] == "MCI") & (df_mci[dx_col] == "AD")]
        v_ctx = pd.concat([
            calc_delta_declining(con_dec, df_con, cols_ctx),
            calc_delta_declining(mci_dec, df_mci, cols_ctx),
        ])
        v_sub = pd.concat([
            calc_delta_declining(con_dec, df_con, cols_sub),
            calc_delta_declining(mci_dec, df_mci, cols_sub),
        ]) if has_sub else None

    elif correlation_type == "baseline_all":
        v_ctx = pd.concat([calc_baseline(d, cols_ctx) for d in [df_con, df_mci, df_ad]])
        v_sub = pd.concat([calc_baseline(d, cols_sub) for d in [df_con, df_mci, df_ad]]) if has_sub else None

    elif correlation_type == "baseline_declining":
        dx_col = biomarker_cfg.get("declining_dx_col", "dementia_dx")
        con_dec = df_con[(df_con["dementia_dx_bl"] == "CON") & (df_con[dx_col].isin(["MCI", "AD"]))]
        mci_dec = df_mci[(df_mci["dementia_dx_bl"] == "MCI") & (df_mci[dx_col] == "AD")]
        v_ctx = pd.concat([calc_baseline(con_dec, cols_ctx), calc_baseline(mci_dec, cols_ctx)])
        v_sub = pd.concat([calc_baseline(con_dec, cols_sub), calc_baseline(mci_dec, cols_sub)]) if has_sub else None

    else:
        raise ValueError(f"Unknown correlation_type: '{correlation_type}'")

    mean_ctx = v_ctx.mean()
    mean_sub = v_sub.mean() if v_sub is not None else None
    return mean_ctx, mean_sub


# ============================================================
# RAW CORRELATIONS (Spearman + spin for cortex), NO FDR yet
# ============================================================

def compute_raw_correlations(
    biomarker_name, biomarker_cfg, neurotransmitter_name,
    receptor_cols, correlation_type, coords,
):
    fmt = biomarker_cfg["file_format"]
    df_con = read_file(biomarker_cfg["file_con"], fmt)
    df_mci = read_file(biomarker_cfg["file_mci"], fmt)
    df_ad = read_file(biomarker_cfg["file_ad"], fmt)

    cols_ctx, cols_sub = get_cols(df_con, biomarker_cfg)
    has_sub = biomarker_cfg["has_subcortical"]

    mean_ctx, mean_sub = build_biomarker_means(
        biomarker_cfg, correlation_type, cols_ctx, cols_sub, has_sub,
        df_con, df_mci, df_ad,
    )

    mean_ctx = normalize_ctx_index(mean_ctx, biomarker_cfg)
    if mean_sub is not None:
        mean_sub = normalize_sub_index(mean_sub)

    df_rec_ctx, df_rec_sub = load_receptors(biomarker_cfg["ctx_index_mode"])
    available = [r for r in receptor_cols if r in df_rec_ctx.columns]
    if not available:
        return []

    common_ctx = df_rec_ctx.index.intersection(mean_ctx.index)
    common_sub = df_rec_sub.index.intersection(mean_sub.index) if mean_sub is not None else pd.Index([])

    records = []

    # cortical: Spearman + spin
    for receptor in available:
        bm = mean_ctx.loc[common_ctx].astype(float)
        rc = df_rec_ctx.loc[common_ctx, receptor].astype(float)
        mask = bm.notna() & rc.notna()
        n = int(mask.sum())
        if n <= 2:
            continue
        rho, pval = stats.spearmanr(bm[mask], rc[mask])
        p_spin = compute_spin_pvalue(
            receptor_vals=rc[mask].values,
            biomarker_vals=bm[mask].values,
            region_names=common_ctx[mask],
            coords=coords,
            observed_rho=rho,
        )
        records.append(dict(
            biomarker=biomarker_name, correlation_type=correlation_type,
            neurotransmitter=neurotransmitter_name, receptor=receptor,
            region_type="cortical", n=n,
            rho=round(float(rho), 4), pvalue=float(pval), p_spin=p_spin,
        ))

    # subcortical: Spearman only
    if has_sub and not common_sub.empty:
        for receptor in available:
            bm = mean_sub.loc[common_sub].astype(float)
            rc = df_rec_sub.loc[common_sub, receptor].astype(float)
            mask = bm.notna() & rc.notna()
            n = int(mask.sum())
            if n <= 2:
                continue
            rho, pval = stats.spearmanr(bm[mask], rc[mask])
            records.append(dict(
                biomarker=biomarker_name, correlation_type=correlation_type,
                neurotransmitter=neurotransmitter_name, receptor=receptor,
                region_type="subcortical", n=n,
                rho=round(float(rho), 4), pvalue=float(pval), p_spin=float("nan"),
            ))

    return records


# ============================================================
# FDR grouped by biomarker + correlation_type + region_type
# ============================================================

def apply_grouped_fdr(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pvalue_fdr"] = np.nan
    df["sig_fdr"] = False

    # cortical FDR uses p_spin; subcortical FDR uses pvalue
    for (bm, ct, rt), idx in df.groupby(
        ["biomarker", "correlation_type", "region_type"]
    ).groups.items():
        sub = df.loc[idx]
        pcol = "p_spin" if rt == "cortical" else "pvalue"
        valid = sub[pcol].notna()
        if valid.sum() == 0:
            continue
        pvals = sub.loc[valid, pcol].to_numpy()
        reject, q, _, _ = multipletests(pvals, alpha=alpha, method="fdr_bh")
        df.loc[sub.index[valid], "pvalue_fdr"] = q
        df.loc[sub.index[valid], "sig_fdr"] = reject

    return df


# ============================================================
# ORCHESTRATION
# ============================================================

def run_full_pipeline(biomarkers, neurotransmitters,
                      correlation_types, biomarker_list, neurotransmitter_selection):
    coords = load_coordinates(coords_file)

    all_records = []
    for biomarker in biomarker_list:
        cfg = biomarkers[biomarker]
        for corr_type in correlation_types:
            for neuro_name in neurotransmitter_selection:
                recs = compute_raw_correlations(
                    biomarker_name=biomarker,
                    biomarker_cfg=cfg,
                    neurotransmitter_name=neuro_name,
                    receptor_cols=neurotransmitters[neuro_name],
                    correlation_type=corr_type,
                    coords=coords,
                )
                all_records.extend(recs)

    df = pd.DataFrame(all_records)
    if df.empty:
        print("No correlations computed.")
        return

    df = apply_grouped_fdr(df)
    df.to_csv(results_file, index=False)
    print(f"Saved {len(df)} correlations to {results_file}")