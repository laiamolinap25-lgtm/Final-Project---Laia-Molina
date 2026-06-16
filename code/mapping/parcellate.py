import os
import re
import numpy as np
from nilearn.image import load_img
from neuromaps.parcellate import Parcellater


path = r"C:\Users\laiam\OneDrive\Escritorio\Practicas Canada\neurotransmitters\data\PET_nifti_images"
outpath = r"C:\Users\laiam\OneDrive\Escritorio\Practicas Canada\neurotransmitters\data\PET_parcellated\dkt"
dkt_path = r"C:\Users\laiam\OneDrive\Escritorio\Practicas Canada\neurotransmitters\data\aparc.DKTatlas+aseg.deep.withCC_resampled.nii"

os.makedirs(outpath, exist_ok=True)


def strip_nii_ext(fname: str) -> str:
    base = os.path.basename(fname)
    return base.replace(".nii.gz", "").replace(".nii", "")

def get_receptor_and_n(fname: str):
    stem = strip_nii_ext(fname)
    receptor = stem.split("_")[0]
    m = re.search(r"_hc(\d+)_", stem)
    n = int(m.group(1)) if m else 1
    return receptor, n

atlas_img = load_img(dkt_path)


parcellater = Parcellater(
    parcellation=atlas_img,
    space="MNI152",
    resampling_target="data"
)


receptors_nii = [
    os.path.join(path, "5HT1a_way_hc36_savli.nii"),
    os.path.join(path, "5HT1a_cumi_hc8_beliveau.nii"),
    os.path.join(path, "5HT1b_az_hc36_beliveau.nii"),
    os.path.join(path, "5HT1b_p943_hc22_savli.nii"),
    os.path.join(path, "5HT1b_p943_hc65_gallezot.nii.gz"),
    os.path.join(path, "5HT2a_cimbi_hc29_beliveau.nii"),
    os.path.join(path, "5HT2a_alt_hc19_savli.nii"),
    os.path.join(path, "5HT2a_mdl_hc3_talbot.nii.gz"),
    os.path.join(path, "5HT4_sb20_hc59_beliveau.nii"),
    os.path.join(path, "5HT6_gsk_hc30_radhakrishnan.nii.gz"),
    os.path.join(path, "5HTT_dasb_hc100_beliveau.nii"),
    os.path.join(path, "5HTT_dasb_hc30_savli.nii"),
    os.path.join(path, "A4B2_flubatine_hc30_hillmer.nii.gz"),
    os.path.join(path, "D1_SCH23390_hc13_kaller.nii"),
    os.path.join(path, "D2_fallypride_hc49_jaworska.nii"),
    os.path.join(path, "D2_flb457_hc37_smith.nii.gz"),
    os.path.join(path, "D2_flb457_hc55_sandiego.nii.gz"),
    os.path.join(path, "D2_raclopride_hc7_alakurtti.nii"),
    os.path.join(path, "DAT_fpcit_hc174_dukart_spect.nii"),
    os.path.join(path, "DAT_fepe2i_hc6_sasaki.nii.gz"),
    os.path.join(path, "GABAa-bz_flumazenil_hc16_norgaard.nii"),
    os.path.join(path, "GABAa_flumazenil_hc6_dukart.nii"),
    os.path.join(path, "M1_lsn_hc24_naganawa.nii.gz"),
    os.path.join(path, "mGluR5_abp_hc22_rosaneto.nii"),
    os.path.join(path, "mGluR5_abp_hc28_dubois.nii"),
    os.path.join(path, "mGluR5_abp_hc73_smart.nii"),
    os.path.join(path, "NMDA_ge179_hc29_galovic.nii.gz"),
    os.path.join(path, "VAChT_feobv_hc4_tuominen.nii"),
    os.path.join(path, "VAChT_feobv_hc5_bedard_sum.nii"),
    os.path.join(path, "VAChT_feobv_hc18_aghourian_sum.nii"),
]


parcellated = {}

for nii_path in receptors_nii:
    try:
        print("Processing:", os.path.basename(nii_path))
        img = load_img(nii_path)

        vec = parcellater.fit_transform(img, "MNI152", True)
        vec = np.asarray(vec).ravel()

        parcellated[nii_path] = vec

    except Exception as e:
        print("ERROR:", nii_path, "->", e)


by_rec = {}
weights = {}

for fpath, vec in parcellated.items():
    receptor, n = get_receptor_and_n(fpath)
    by_rec.setdefault(receptor, []).append(vec)
    weights.setdefault(receptor, []).append(n)

for receptor in sorted(by_rec.keys()):

    X = np.vstack(by_rec[receptor])   # k x n_regions
    w = np.array(weights[receptor], dtype=float)
    w = w / w.sum() if w.sum() != 0 else np.ones_like(w) / len(w)

    # z-score each template across regions
    mu = np.nanmean(X, axis=1, keepdims=True)
    sd = np.nanstd(X, axis=1, keepdims=True)
    Xz = (X - mu) / sd
    Xz = np.nan_to_num(Xz, nan=0.0, posinf=0.0, neginf=0.0)
    combined_z = (w[:, None] * Xz).sum(axis=0)

    # save numeric vector only
    np.savetxt(
        os.path.join(outpath, f"{receptor}_DKT_z.csv"),
        combined_z,
        delimiter=","
    )

    print(f"Saved {receptor}")

print(dkt_path)


