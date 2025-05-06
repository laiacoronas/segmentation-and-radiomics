# -*- coding: utf-8 -*-
"""
Milestone 1 (Segmentation): Generation of an Annotated Dataset.
 1. Extract VOI (Volume of Interest) from the CTs (intensity and mask).
 2. Produce a single annotation for each lesion from the 4 radiologists’ annotations using Max-Voting.
 3. Make Max-Voting to obtain the “Diagnosis”: if two or more radiologists have characterized the nodule with a Malignancy score > 3, then Diagnosis=1 (malignant), otherwise Diagnosis=0 (benign).
"""

#%% Environment onfiguration

from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu
from skimage.morphology import ball, opening
import matplotlib as plt
import pandas as pd
import numpy as np
import nibabel as nib
import os

current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)
os.chdir(current_dir)

#%% Upload metadata

df = pd.read_excel('MetadatabyAnnotation.xlsx')

#%% Extract VOIs

def load_nifti(path):
    nii = nib.load(path)
    return nii.get_fdata(), nii.affine, nii.header

def save_nifti(volume, affine, header, path):
    nib.save(nib.Nifti1Image(volume.astype(np.uint8), affine, header), path)

def segment_voi(ct_path, output_path, sigma=1, radius=2):
    # Load CT
    ct, affine, header = load_nifti(ct_path)

    #  Preprocessing: Gaussian smoothing
    smoothed = gaussian_filter(ct, sigma=sigma)

    # Otsu Thresholding
    threshold = threshold_otsu(smoothed)
    mask = smoothed > threshold

    # Morphological Opening
    struct_elem = ball(radius)
    cleaned_mask = opening(mask, struct_elem)

    # Save result
    save_nifti(cleaned_mask, affine, header, output_path)
    print(f"Saved: {output_path}")

# Paths
base_path = "data/sample/CT/image"
output_dir = "output/segmented_vois"
os.makedirs(output_dir, exist_ok=True)

ct_ids = ["LIDC-IDRI-0001", "LIDC-IDRI-0003", "LIDC-IDRI-0005"]
for ct_id in ct_ids:
    input_path = os.path.join(base_path, f"{ct_id}.nii.gz")
    output_path = os.path.join(output_dir, f"{ct_id}_pred_mask.nii.gz")
    segment_voi(input_path, output_path)
    
#%% Visualize results

# Load images
ct1, affine1, header = load_nifti(os.path.join(current_dir,"data/full_data/VOIs/image/LIDC-IDRI-0001_R_1.nii.gz"))
r1, affine_r1, header = load_nifti(os.path.join(current_dir,"data/full_data/VOIs/nodule_mask/LIDC-IDRI-0001_R_1.nii.gz"))


# Visualize a middle slice 

slice_index = ct1.shape[2] // 2

# Create triple subplot
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(ct1[:, :,  ct1.shape[2] // 2], cmap='gray')
axes[0].set_title('Full CT Image')
axes[0].axis('off')

axes[1].imshow(r1[:, :, r1.shape[2]//2], cmap='gray')
axes[1].set_title('VOI Image')
axes[1].axis('off')

plt.tight_layout()
plt.show()
    