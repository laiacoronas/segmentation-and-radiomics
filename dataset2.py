#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Milestone 1 (Segmentation): Generation of an Annotated Dataset.
 1. Extract VOI (Volume of Interest) from the CTs (intensity and mask).
 2. Produce a single annotation for each lesion from the 4 radiologists’ annotations using Max-Voting.
 3. Make Max-Voting to obtain the “Diagnosis”: if two or more radiologists have characterized the nodule with a Malignancy score > 3, then Diagnosis=1 (malignant), otherwise Diagnosis=0 (benign).
"""

#%% Environment configuration

import os
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from nibabel.affines import apply_affine

# Change working directory
current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)
os.chdir(current_dir)

#%% Load metadata

df = pd.read_excel('data/sample/MetadatabyAnnotation.xlsx')

#%% Utility functions

def load_nifti(path):
    nii = nib.load(path)
    return nii.get_fdata(), nii.affine, nii.header

def save_nifti(volume, affine, header, path):
    nib.save(nib.Nifti1Image(volume.astype(np.uint8), affine, header), path)

def extract_voi_from_coords(ct_path, centercoords, diameter_mm, output_path, padding_mm=4.0):
    ct, affine, header = load_nifti(ct_path)

    margin = diameter_mm / 2 + padding_mm
    coordX, coordY, coordZ = centercoords

    # Manual conversion using affine matrix
    voxel_x_low = int((coordX - margin - affine[0, 3]) / affine[0, 0])
    voxel_x_high = int((coordX + margin - affine[0, 3]) / affine[0, 0])
    voxel_y_low = int((coordY - margin - affine[1, 3]) / affine[1, 1])
    voxel_y_high = int((coordY + margin - affine[1, 3]) / affine[1, 1])
    voxel_z_low = int((coordZ + margin - affine[2, 3]) / affine[2, 2])
    voxel_z_high = int((coordZ - margin - affine[2, 3]) / affine[2, 2])


    print(f"Voxel box low:  ({voxel_x_low}, {voxel_y_low}, {voxel_z_low})")
    print(f"Voxel box high: ({voxel_x_high}, {voxel_y_high}, {voxel_z_high})")

    ct_voi = ct[:, voxel_y_low:voxel_y_high, voxel_x_low: voxel_x_high]
    save_nifti(ct_voi, affine, header, output_path)

    print(f"VOI saved to {output_path} with shape {ct_voi.shape}")

#%% Compute VOIs

ct_base_path = "data/sample/CT/image"
output_dir_VOIs = "output/VOIs"
os.makedirs(output_dir_VOIs, exist_ok=True)

last_nodule_id = None
last_ct_id = None
voi_counter = 1

for idx, row in df.iterrows():
    ct_id = row['patient_id']
    nodule_id = row['nodule_id']

    if ct_id != last_ct_id:
        voi_counter = 1
        last_ct_id = ct_id

    if nodule_id != last_nodule_id:
        last_nodule_id = nodule_id

        coordX, coordY, coordZ = row['coordX'], row['coordY'], row['coordZ']
        diameter = row['diameter_mm']
        ct_path = os.path.join(ct_base_path, f"{ct_id}.nii.gz")

        # LIDC-IDRI-0003 has a known off-by-one inconsistency in ground truth file naming
        gt_index = voi_counter + 1 if ct_id == "LIDC-IDRI-0003" else voi_counter
        gt_path = os.path.join("data/full_data/VOIs/image/", f"{ct_id}_R_{gt_index}.nii.gz")

        output_path = os.path.join(output_dir_VOIs, f"{ct_id}_R_{voi_counter}.nii.gz")

        if os.path.exists(ct_path):
            extract_voi_from_coords(ct_path, (coordX, coordY, coordZ), diameter, output_path, padding_mm=4)

        voi_counter += 1

#%% Visualize example

ct1, _, _ = load_nifti("data/sample/CT/image/LIDC-IDRI-0001.nii.gz")
voi_pred, _, _ = load_nifti("output/VOIs/LIDC-IDRI-0001_R_1.nii.gz")
voi_gt, _, _ = load_nifti("data/full_data/VOIs/image/LIDC-IDRI-0001_R_1.nii.gz")

# Visualize central slice of each
fig, axes = plt.subplots(1, 3, figsize=(12, 5))

mid_ct = ct1.shape[2] // 2
mid_pred = 10
mid_gt = voi_gt.shape[2] // 2

axes[0].imshow(ct1[:, :, mid_ct], cmap='gray')
axes[0].set_title("Full CT (middle slice)")
axes[0].axis('off')

axes[1].imshow(voi_gt[:, :, mid_gt], cmap='gray')
axes[1].set_title("Ground Truth VOI")
axes[1].axis('off')

axes[2].imshow(voi_pred[:, :, mid_pred], cmap='gray')
axes[2].set_title("Extracted VOI")
axes[2].axis('off')

plt.tight_layout()
plt.show()