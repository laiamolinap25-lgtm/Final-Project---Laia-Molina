import os
import pandas as pd
import matplotlib.pyplot as plt

output_dir = r"C:\Users\laiam\OneDrive\Escritorio\Practicas Canada\neurotransmitters\data\PET_parcellated\dkt\adni analysis"
os.makedirs(output_dir, exist_ok=True)

timepoint_to_months = {
    "bl": 0,
    "m06": 6,
    "m12": 12,
    "m18": 18,
    "m24": 24,
    "m36": 36,
    "m48": 48,
    "m60": 60,
    "m72": 72,
    "m84": 84,
    "m96": 96,
    "m108": 108,
    "m120": 120,
    "m132": 132,
    "m144": 144
}

files = {
    "CON": r"C:\Users\laiam\OneDrive\Escritorio\Practicas Canada\neurotransmitters\data\adni_dkt_thick_con_Laia.xlsx",
    "MCI": r"C:\Users\laiam\OneDrive\Escritorio\Practicas Canada\neurotransmitters\data\adni_dkt_thick_mci_Laia.xlsx",
    "AD": r"C:\Users\laiam\OneDrive\Escritorio\Practicas Canada\neurotransmitters\data\adni_dkt_thick_ad_Laia.xlsx"
}

dfs = []

for group, file in files.items():
    df = pd.read_excel(file)
    df["group"] = group
    df["biomarker"] = "dkt_thick"
    dfs.append(df)

dkt_thick = pd.concat(dfs, ignore_index=True)

dkt_thick["tp_months"] = dkt_thick["timepoint"].map(timepoint_to_months)

# Check unknown timepoints
unknown = dkt_thick.loc[dkt_thick["tp_months"].isna(), "timepoint"].unique()


patient_summary = (
    dkt_thick.groupby(["group", "patient_id"])
    .agg(
        n_visits=("timepoint", "count"),
        first_month=("tp_months", "min"),
        last_month=("tp_months", "max"),
        sex=("sex", "first"),
        dementia_dx_bl=("dementia_dx_bl", "first")
    )
    .reset_index()
)

patient_summary["followup_months"] = (
    patient_summary["last_month"] - patient_summary["first_month"]
)

patient_summary["followup_years"] = patient_summary["followup_months"] / 12

patient_summary["valid_baseline"] = patient_summary["n_visits"] >= 1

patient_summary["valid_delta"] = (
    (patient_summary["n_visits"] >= 2) &
    (patient_summary["followup_months"] > 0)
)

patient_summary.to_csv(
    os.path.join(output_dir, "dkt_thickvol_patient_summary_by_group.csv"),
    index=False
)

summary = (
    patient_summary.groupby("group")
    .agg(
        n_patients=("patient_id", "nunique"),
        baseline_valid=("valid_baseline", "sum"),
        delta_valid=("valid_delta", "sum"),
        median_visits=("n_visits", "median"),
        mean_visits=("n_visits", "mean")
    )
    .reset_index()
)

# Add percentage of longitudinally eligible patients
summary["delta_valid_percent"] = (
    100 * summary["delta_valid"] / summary["n_patients"]
)

# Follow-up only among valid delta patients
followup_valid = (
    patient_summary[patient_summary["valid_delta"]]
    .groupby("group")
    .agg(
        median_followup_years=("followup_years", "median"),
        mean_followup_years=("followup_years", "mean")
    )
    .reset_index()
)

summary = summary.merge(followup_valid, on="group", how="left")
summary.to_csv(
    os.path.join(output_dir, "dkt_thickvol_patient_summary_group_level.csv"),
    index=False
)


timepoint_summary = (
    dkt_thick.groupby(["group", "timepoint"])
    .size()
    .reset_index(name="n_visits")
)

timepoint_summary["months"] = timepoint_summary["timepoint"].map(timepoint_to_months)

timepoint_summary.to_csv(
    os.path.join(output_dir, "dkt_thickvol_timepoint_summary_by_group.csv"),
    index=False
)

x = range(len(summary))
width = 0.35

plt.figure(figsize=(7, 4))
plt.bar(
    [i - width / 2 for i in x],
    summary["baseline_valid"],
    width=width,
    label="Baseline valid"
)
plt.bar(
    [i + width / 2 for i in x],
    summary["delta_valid"],
    width=width,
    label="Delta valid"
)

plt.xticks(x, summary["group"])
plt.ylabel("Number of patients")
plt.title("DKT Thick patients valid for baseline and delta analyses")
plt.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(output_dir, "dkt_thickvol_baseline_delta_valid_by_group.png"),
    dpi=300
)
plt.close()

pivot = timepoint_summary.pivot(
    index="timepoint",
    columns="group",
    values="n_visits"
).fillna(0)

# Sort timepoints by months
order = (
    timepoint_summary[["timepoint", "months"]]
    .drop_duplicates()
    .sort_values("months")["timepoint"]
)

pivot = pivot.loc[order]

pivot.plot(kind="bar", figsize=(8, 4))

plt.ylabel("Number of visits")
plt.xlabel("Timepoint")
plt.title("DKT Thick timepoint availability by diagnostic group")
plt.tight_layout()
plt.savefig(
    os.path.join(output_dir, "dkt_thickvol_timepoint_availability_by_group.png"),
    dpi=300
)
plt.close()

visits_dist = (
    patient_summary.groupby(["group", "n_visits"])
    .size()
    .reset_index(name="n_patients")
)

for group in visits_dist["group"].unique():
    tmp = visits_dist[visits_dist["group"] == group]

    plt.figure(figsize=(6, 4))
    plt.bar(tmp["n_visits"].astype(str), tmp["n_patients"])
    plt.xlabel("Number of visits")
    plt.ylabel("Number of patients")
    plt.title(f"DKT Thick visits per patient - {group}")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"dkt_thickvol_visits_per_patient_{group}.png"),
        dpi=300
    )
    plt.close()

