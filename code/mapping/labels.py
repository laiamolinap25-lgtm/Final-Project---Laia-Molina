from nilearn.image import load_img
import numpy as np

dkt_path = r"C:\Users\laiam\OneDrive\Escritorio\Practicas Canada\neurotransmitters\data\aparc.DKTatlas+aseg.deep.withCC_resampled.nii"

atlas = load_img(dkt_path)
data = atlas.get_fdata()

labels = np.unique(data)
labels = labels[labels != 0]  #  remove background

print("Number of labels:", len(labels))
print(labels.astype(int))
print(len(labels))