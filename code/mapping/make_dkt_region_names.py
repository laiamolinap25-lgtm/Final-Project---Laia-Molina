import os
import numpy as np


outpath = r"C:\Users\laiam\OneDrive\Escritorio\Practicas Canada\neurotransmitters\data\PET_parcellated\dkt"
labels_path = os.path.join(outpath, "DKT_labels_order.csv")
lut_path = os.path.join(outpath, "label_names_list.csv")


labels = np.loadtxt(labels_path, delimiter=",", dtype=int)


def load_label_names(lut_file):
    lut = {}
    with open(lut_file, "r", encoding="utf-8", errors="replace") as f:
        header = f.readline().strip()  
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(";", 1)
            if len(parts) != 2:
                continue

            lab_str = parts[0].strip()
            name_str = parts[1].strip()

            # clean repeated/extra tail after ",,"
            name_clean = name_str.split(",,")[0].strip()

            try:
                lab = int(lab_str)
            except ValueError:
                continue

            lut[lab] = name_clean

    return lut

if not os.path.exists(lut_path):
    raise FileNotFoundError(f"Missing LUT file: {lut_path}")

lut = load_label_names(lut_path)


region_names = [lut.get(int(lab), f"Unknown_{int(lab)}") for lab in labels]


names_txt = os.path.join(outpath, "DKT_region_names.txt")
with open(names_txt, "w", encoding="utf-8") as f:
    for nm in region_names:
        f.write(nm + "\n")


table_csv = os.path.join(outpath, "DKT_label_name_table.csv")
with open(table_csv, "w", encoding="utf-8") as f:
    f.write("label,name\n")
    for lab, nm in zip(labels, region_names):
        # quote names just in case there are commas
        f.write(f'{int(lab)},"{nm}"\n')


print("Saved:", names_txt)
print("Saved:", table_csv)
print("\nSanity check (first 10):")
for i in range(min(10, len(labels))):
    print(f"{labels[i]} -> {region_names[i]}")


missing = [int(lab) for lab in labels if int(lab) not in lut]
if missing:
    print(f"\nWarning: {len(missing)} labels not found in LUT. Example:", missing[:10])
else:
    print("\nAll labels found in LUT. ")
