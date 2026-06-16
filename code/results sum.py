
import os
import pandas as pd
import matplotlib.pyplot as plt


input_file = r"C:\Users\laiam\OneDrive\Escritorio\Practicas Canada\neurotransmitters\data\correlations lists.xlsx"
output_dir = r"C:\Users\laiam\OneDrive\Escritorio\Practicas Canada\neurotransmitters\data\PET_parcellated\dkt\results analysis"



os.makedirs(output_dir, exist_ok=True)

df = pd.read_excel(input_file)

df["rho"] = pd.to_numeric(df["rho"], errors="coerce")
df = df.dropna(subset=["rho"])

# Direction of the significant correlation
df["direction"] = df["rho"].apply(lambda x: "positive" if x > 0 else "negative")

# Optional: baseline/delta and all/declining
df["time_type"] = df["correlation_type"].apply(
    lambda x: "baseline" if str(x).startswith("baseline") else "delta"
)

df["population"] = df["correlation_type"].apply(
    lambda x: "declining" if str(x).endswith("declining") else "all"
)

# Save enriched file
df.to_csv(
    os.path.join(output_dir, "significant_correlations_with_direction.csv"),
    index=False
)

# =========================
# 1. POSITIVE / NEGATIVE BY NEUROTRANSMITTER SYSTEM
# =========================

system_direction = (
    df.groupby(["neurotransmitter", "direction"])
    .size()
    .reset_index(name="n")
)

system_pivot = system_direction.pivot(
    index="neurotransmitter",
    columns="direction",
    values="n"
).fillna(0)

if "positive" not in system_pivot.columns:
    system_pivot["positive"] = 0

if "negative" not in system_pivot.columns:
    system_pivot["negative"] = 0

system_pivot["total"] = system_pivot["positive"] + system_pivot["negative"]
system_pivot["percent_positive"] = 100 * system_pivot["positive"] / system_pivot["total"]
system_pivot["percent_negative"] = 100 * system_pivot["negative"] / system_pivot["total"]

system_pivot = system_pivot.reset_index()

system_pivot.to_csv(
    os.path.join(output_dir, "direction_by_neurotransmitter_system.csv"),
    index=False
)

print("\nDirection by neurotransmitter system:")
print(system_pivot)

# =========================
# 2. POSITIVE / NEGATIVE BY SYSTEM AND BIOMARKER
# =========================

system_biomarker_direction = (
    df.groupby(["neurotransmitter", "biomarker", "direction"])
    .size()
    .reset_index(name="n")
)

system_biomarker_pivot = system_biomarker_direction.pivot_table(
    index=["neurotransmitter", "biomarker"],
    columns="direction",
    values="n",
    fill_value=0
).reset_index()

if "positive" not in system_biomarker_pivot.columns:
    system_biomarker_pivot["positive"] = 0

if "negative" not in system_biomarker_pivot.columns:
    system_biomarker_pivot["negative"] = 0

system_biomarker_pivot["total"] = (
    system_biomarker_pivot["positive"] + system_biomarker_pivot["negative"]
)

system_biomarker_pivot["percent_positive"] = (
    100 * system_biomarker_pivot["positive"] / system_biomarker_pivot["total"]
)

system_biomarker_pivot["percent_negative"] = (
    100 * system_biomarker_pivot["negative"] / system_biomarker_pivot["total"]
)

system_biomarker_pivot.to_csv(
    os.path.join(output_dir, "direction_by_system_and_biomarker.csv"),
    index=False
)

print("\nDirection by system and biomarker:")
print(system_biomarker_pivot)

# =========================
# 3. POSITIVE / NEGATIVE BY CONDITION
# =========================

condition_direction = (
    df.groupby(["correlation_type", "direction"])
    .size()
    .reset_index(name="n")
)

condition_pivot = condition_direction.pivot(
    index="correlation_type",
    columns="direction",
    values="n"
).fillna(0)

if "positive" not in condition_pivot.columns:
    condition_pivot["positive"] = 0

if "negative" not in condition_pivot.columns:
    condition_pivot["negative"] = 0

condition_pivot["total"] = condition_pivot["positive"] + condition_pivot["negative"]
condition_pivot["percent_positive"] = 100 * condition_pivot["positive"] / condition_pivot["total"]
condition_pivot["percent_negative"] = 100 * condition_pivot["negative"] / condition_pivot["total"]

condition_pivot = condition_pivot.reset_index()

condition_pivot.to_csv(
    os.path.join(output_dir, "direction_by_condition.csv"),
    index=False
)

print("\nDirection by condition:")
print(condition_pivot)

# =========================
# 4. POSITIVE / NEGATIVE BY BIOMARKER
# =========================

biomarker_direction = (
    df.groupby(["biomarker", "direction"])
    .size()
    .reset_index(name="n")
)

biomarker_pivot = biomarker_direction.pivot(
    index="biomarker",
    columns="direction",
    values="n"
).fillna(0)

if "positive" not in biomarker_pivot.columns:
    biomarker_pivot["positive"] = 0

if "negative" not in biomarker_pivot.columns:
    biomarker_pivot["negative"] = 0

biomarker_pivot["total"] = biomarker_pivot["positive"] + biomarker_pivot["negative"]
biomarker_pivot["percent_positive"] = 100 * biomarker_pivot["positive"] / biomarker_pivot["total"]
biomarker_pivot["percent_negative"] = 100 * biomarker_pivot["negative"] / biomarker_pivot["total"]

biomarker_pivot = biomarker_pivot.reset_index()

biomarker_pivot.to_csv(
    os.path.join(output_dir, "direction_by_biomarker.csv"),
    index=False
)

print("\nDirection by biomarker:")
print(biomarker_pivot)
# =========================
# REGION TYPE DIRECTION ANALYSIS
# =========================

# 1. Direction by region type
region_direction = (
    df.groupby(["region_type", "direction"])
    .size()
    .reset_index(name="n")
)

region_pivot = region_direction.pivot(
    index="region_type",
    columns="direction",
    values="n"
).fillna(0)

if "positive" not in region_pivot.columns:
    region_pivot["positive"] = 0

if "negative" not in region_pivot.columns:
    region_pivot["negative"] = 0

region_pivot["total"] = region_pivot["positive"] + region_pivot["negative"]
region_pivot["percent_positive"] = 100 * region_pivot["positive"] / region_pivot["total"]
region_pivot["percent_negative"] = 100 * region_pivot["negative"] / region_pivot["total"]

region_pivot = region_pivot.reset_index()

region_pivot.to_csv(
    os.path.join(output_dir, "direction_by_region_type.csv"),
    index=False
)

print("\nDirection by region type:")
print(region_pivot)


# 2. Direction by neurotransmitter system and region type
system_region_direction = (
    df.groupby(["neurotransmitter", "region_type", "direction"])
    .size()
    .reset_index(name="n")
)

system_region_pivot = system_region_direction.pivot_table(
    index=["neurotransmitter", "region_type"],
    columns="direction",
    values="n",
    fill_value=0
).reset_index()

if "positive" not in system_region_pivot.columns:
    system_region_pivot["positive"] = 0

if "negative" not in system_region_pivot.columns:
    system_region_pivot["negative"] = 0

system_region_pivot["total"] = (
    system_region_pivot["positive"] + system_region_pivot["negative"]
)

system_region_pivot["percent_positive"] = (
    100 * system_region_pivot["positive"] / system_region_pivot["total"]
)

system_region_pivot["percent_negative"] = (
    100 * system_region_pivot["negative"] / system_region_pivot["total"]
)

system_region_pivot.to_csv(
    os.path.join(output_dir, "direction_by_system_and_region_type.csv"),
    index=False
)

print("\nDirection by system and region type:")
print(system_region_pivot)


# 3. Direction by biomarker and region type
biomarker_region_direction = (
    df.groupby(["biomarker", "region_type", "direction"])
    .size()
    .reset_index(name="n")
)

biomarker_region_pivot = biomarker_region_direction.pivot_table(
    index=["biomarker", "region_type"],
    columns="direction",
    values="n",
    fill_value=0
).reset_index()

if "positive" not in biomarker_region_pivot.columns:
    biomarker_region_pivot["positive"] = 0

if "negative" not in biomarker_region_pivot.columns:
    biomarker_region_pivot["negative"] = 0

biomarker_region_pivot["total"] = (
    biomarker_region_pivot["positive"] + biomarker_region_pivot["negative"]
)

biomarker_region_pivot["percent_positive"] = (
    100 * biomarker_region_pivot["positive"] / biomarker_region_pivot["total"]
)

biomarker_region_pivot["percent_negative"] = (
    100 * biomarker_region_pivot["negative"] / biomarker_region_pivot["total"]
)

biomarker_region_pivot.to_csv(
    os.path.join(output_dir, "direction_by_biomarker_and_region_type.csv"),
    index=False
)

print("\nDirection by biomarker and region type:")
print(biomarker_region_pivot)
GROUP_COLORS = {
    "positive": "#59A14F",
    "negative": "#E15759"
}

region_plot = region_pivot.set_index("region_type")[["positive", "negative"]]

ax = region_plot.plot(
    kind="bar",
    figsize=(6, 4),
    color=[GROUP_COLORS["positive"], GROUP_COLORS["negative"]],
    edgecolor="white",
    linewidth=0.8
)

plt.ylabel("Number of FDR-significant associations")
plt.xlabel("Region type")
plt.title("Direction of significant associations by region type")
plt.xticks(rotation=0)
plt.legend(["Positive", "Negative"], frameon=False)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.25)

plt.tight_layout()
plt.savefig(
    os.path.join(output_dir, "bar_direction_by_region_type.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()