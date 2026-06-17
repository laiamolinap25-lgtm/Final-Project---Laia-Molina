import os
import pandas as pd
import matplotlib.pyplot as plt

output_dir = r"C:\Users\laiam\OneDrive\Escritorio\Practicas Canada\neurotransmitters\data\PET_parcellated\dkt\adni analysis"
os.makedirs(output_dir, exist_ok=True)

timepoint_to_months = {
    "bl": 0, "m06": 6, "m12": 12, "m18": 18, "m24": 24,
    "m36": 36, "m48": 48, "m60": 60, "m72": 72,
    "m84": 84, "m96": 96, "m108": 108, "m120": 120,
    "m132": 132, "m144": 144
}

# Config: biomarker -> {group -> filename}
files = {
    "fdg": {
        "CON": r"C:\Users\laiam\OneDrive\Escritorio\Practicas Canada\neurotransmitters\data\adni_fdg_con_Laia.csv",
        "MCI": r"C:\Users\laiam\OneDrive\Escritorio\Practicas Canada\neurotransmitters\data\adni_fdg_mci_Laia.csv",
        "AD": r"C:\Users\laiam\OneDrive\Escritorio\Practicas Canada\neurotransmitters\data\adni_fdg_ad_Laia.csv"
    },
    "t1t2": {
        "CON": r"C:\Users\laiam\OneDrive\Escritorio\Practicas Canada\neurotransmitters\data\adni_t1t2_con_Laia.csv",
        "MCI": r"C:\Users\laiam\OneDrive\Escritorio\Practicas Canada\neurotransmitters\data\adni_t1t2_mci_Laia.csv",
        "AD": r"C:\Users\laiam\OneDrive\Escritorio\Practicas Canada\neurotransmitters\data\adni_t1t2_ad_Laia.csv"
    },
    "volume": {
        "CON": r"C:\Users\laiam\OneDrive\Escritorio\Practicas Canada\neurotransmitters\data\adni_dkt_thick_con_Laia.xlsx",
        "MCI": r"C:\Users\laiam\OneDrive\Escritorio\Practicas Canada\neurotransmitters\data\adni_dkt_thick_mci_Laia.xlsx",
        "AD": r"C:\Users\laiam\OneDrive\Escritorio\Practicas Canada\neurotransmitters\data\adni_dkt_thick_ad_Laia.xlsx"
    },
    "thickness": {
        "CON": r"C:\Users\laiam\OneDrive\Escritorio\Practicas Canada\neurotransmitters\data\adni_dkt_thick_con_Laia.xlsx",
        "MCI": r"C:\Users\laiam\OneDrive\Escritorio\Practicas Canada\neurotransmitters\data\adni_dkt_thick_mci_Laia.xlsx",
        "AD": r"C:\Users\laiam\OneDrive\Escritorio\Practicas Canada\neurotransmitters\data\adni_dkt_thick_ad_Laia.xlsx"
    }
}

all_patient_summaries = []

for biomarker, group_files in files.items():
    for group, path in group_files.items():
        # Load
        if path.endswith(".csv"):
            df = pd.read_csv(path)
        else:
            df = pd.read_excel(path)

        df["group"] = group
        df["biomarker"] = biomarker
        df["tp_months"] = df["timepoint"].map(timepoint_to_months)

        # Patient summary
        patient_summary = (
            df.sort_values("tp_months")
              .groupby(["biomarker", "group", "patient_id"])
              .agg(
                  n_visits=("timepoint", "count"),
                  first_month=("tp_months", "min"),
                  last_month=("tp_months", "max"),
                  sex=("sex", "first"),
                  diagnosis_baseline=("dementia_dx_bl", "first")
              )
              .reset_index()
        )
        patient_summary["followup_months"] = patient_summary["last_month"] - patient_summary["first_month"]
        patient_summary["followup_years"] = patient_summary["followup_months"] / 12
        patient_summary["valid_baseline"] = patient_summary["n_visits"] >= 1
        patient_summary["valid_delta"] = (patient_summary["n_visits"] >= 2) & (patient_summary["followup_months"] > 0)

        patient_summary.to_csv(
            os.path.join(output_dir, f"{biomarker}_{group}_patient_summary.csv"),
            index=False
        )

        all_patient_summaries.append(patient_summary)

# Concatenate all biomarkers for a global summary
global_df = pd.concat(all_patient_summaries, ignore_index=True)

# Summary by biomarker & group
summary = (
    global_df.groupby(["biomarker", "group"])
    .agg(
        n_patients=("patient_id", "nunique"),
        baseline_valid=("valid_baseline", "sum"),
        delta_valid=("valid_delta", "sum"),
        median_visits=("n_visits", "median"),
        mean_visits=("n_visits", "mean")
    )
    .reset_index()
)

summary["baseline_valid_percent"] = 100 * summary["baseline_valid"] / summary["n_patients"]
summary["delta_valid_percent"] = 100 * summary["delta_valid"] / summary["n_patients"]

summary.to_csv(os.path.join(output_dir, "adni_global_summary.csv"), index=False)
print(summary)

# Plot delta_valid_percent by biomarker & group
pivot = summary.pivot(index="biomarker", columns="group", values="delta_valid_percent")
pivot = pivot[["CON", "MCI", "AD"]]

pivot.plot(kind="bar", figsize=(8, 5))
plt.ylabel("Patients valid for delta (%)")
plt.xlabel("Biomarker")
plt.title("Longitudinal eligibility by biomarker and diagnostic group")
plt.ylim(0, 100)
plt.xticks(rotation=0)
plt.legend(title="Group")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "adni_delta_valid_percent_by_biomarker_group.png"), dpi=300)
plt.close()