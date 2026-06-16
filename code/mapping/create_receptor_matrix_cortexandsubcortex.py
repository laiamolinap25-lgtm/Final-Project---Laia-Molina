import os
import glob
import csv
import numpy as np
import pandas as pd


path = r"C:\Users\laiam\OneDrive\Escritorio\Practicas Canada\neurotransmitters\data\PET_parcellated\dkt"

name_table_file = os.path.join(path, "DKT_label_name_table.csv")
labels_order_file = os.path.join(path, "DKT_labels_order.csv")  

files = sorted(glob.glob(os.path.join(path, "*_DKT_z.csv")))
if len(files) == 0:
    raise FileNotFoundError(f"No *_DKT_z.csv files found in: {path}")

receptor_names = [os.path.basename(f).replace("_DKT_z.csv", "") for f in files]


vectors = [] # list to hold each receptor vector
for f in files: # load each receptor vector
    v = np.loadtxt(f, delimiter=",").astype(float).ravel() # ensure it's 1D
    vectors.append(v)# check all vectors have the same length

Z = np.column_stack(vectors) 
print("Matrix shape:", Z.shape) 


df_names = pd.read_csv(name_table_file) # load region names

if "name" not in df_names.columns: # check for 'name' column
    raise ValueError(f"'name' column not found in {name_table_file}. Columns: {df_names.columns.tolist()}") # ensure names are strings

regions = df_names["name"].astype(str).tolist() # check alignment of regions and matrix rows
print("Regions:", len(regions))# print first few regions to verify

if len(regions) != Z.shape[0]: # check alignment of regions and matrix rows
    raise ValueError(
        f"Region names and vectors are not aligned: names={len(regions)} vs matrix_rows={Z.shape[0]}" # ensure labels are integers
    )

df_labels = pd.read_csv(labels_order_file, header=None, names=["label"]) # load label indices
labels = df_labels["label"].astype(int).tolist()# check alignment of labels and matrix rows

if len(labels) != len(regions): # check alignment of labels and matrix rows
    raise ValueError(
        f"Labels and regions are not aligned: labels={len(labels)} vs regions={len(regions)}" # ensure labels are integers
    )


out_combined = os.path.join(path, "DKT_receptors_table_corticalandsubcortical.csv") # output file with labels, regions, and receptor values

with open(out_combined, "w", encoding="utf-8", newline="") as f:# save combined table with labels, regions, and receptor values
    writer = csv.writer(f) 

    writer.writerow(["label", "region"] + receptor_names) # write header with label, region, and receptor names
    for i in range(len(regions)): 
        writer.writerow([labels[i], regions[i]] + Z[i, :].tolist()) 
 
print("Saved combined:", out_combined)