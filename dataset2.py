#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Milestone 1 (Segmentation): Generation of an Annotated Dataset.
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
print(df.head())

#%% Utility functions

def load_nifti(path):
    nii = nib.load(path)
    data = nii.get_fdata()
    affine = nii.affine
    header = nii.header
    return data, affine, header

def save_nifti(volume, affine, header, path):
    nib.save(nib.Nifti1Image(volume, affine, header), path)

def extract_voi_from_coords(ct_path, centercoords, diameter_mm, output_path, padding_mm=4.0):
    ct, affine, header = load_nifti(ct_path)

    margin = diameter_mm / 2 + padding_mm
    coordX, coordY, coordZ = centercoords
    

    # Compute low and high mm bounding box
    mm_coords_high = (coordX + margin, coordY + margin, coordZ + margin)
    mm_coords_low = (coordX - margin, coordY - margin, coordZ - margin)

    print(f"   - Bounding box (mm) low: {mm_coords_low}")
    print(f"   - Bounding box (mm) high: {mm_coords_high}")

    # Convert to homogeneous coordinates
    mm_coords_hom_low = np.append(mm_coords_low, 1)
    mm_coords_hom_high = np.append(mm_coords_high, 1)
    
    
    # Transform mm → voxel
    voxel_coords_low = np.linalg.inv(affine) @ mm_coords_hom_low
    voxel_coords_high = np.linalg.inv(affine) @ mm_coords_hom_high
    voxel_indices_low = voxel_coords_low[:3].astype(int)
    voxel_indices_high = voxel_coords_high[:3].astype(int)

    print(f"   - Bounding box (voxel) low (before fix): {voxel_indices_low}")
    print(f"   - Bounding box (voxel) high (before fix): {voxel_indices_high}")

    
    # INVERTIR los índices respecto a las dimensiones
    shape_x, shape_y, shape_z = ct.shape

    # x0 = shape_x - voxel_indices_low[0] - 50
    # x1 = shape_x - voxel_indices_high[0] - 25
    # y0 = shape_y - voxel_indices_low[1]- 40
    # y1 = shape_y - voxel_indices_high[1] -5
    
    x0 = shape_x - voxel_indices_low[0]
    x1 = shape_x - voxel_indices_high[0]
    y0 = shape_y - voxel_indices_low[1]
    y1 = shape_y - voxel_indices_high[1]
    
    z0 = voxel_indices_low[2]
    z1 = voxel_indices_high[2]
    
    # CORRECCIÓN: ordenar índices
    voxel_indices_low = (x0, y0, z0)
    voxel_indices_high = (x1, y1, z1)
    
    voxel_indices_low_fixed = np.minimum(voxel_indices_low, voxel_indices_high)
    voxel_indices_high_fixed = np.maximum(voxel_indices_low, voxel_indices_high)

    print(f"   - Bounding box (voxel) low (fixed): {voxel_indices_low_fixed}")
    print(f"   - Bounding box (voxel) high (fixed): {voxel_indices_high_fixed}")
    
    x0, y0, z0 = voxel_indices_low_fixed
    x1, y1, z1 = voxel_indices_high_fixed

    # Validación
    if x0 >= x1 or y0 >= y1 or z0 >= z1:
        print("ERROR: Invalid bounding box after correction")
        return

    # Extract VOI
    ct_voi = ct[x0:x1, y0:y1, z0:z1]
    print(f"Extracted VOI shape: {ct_voi.shape}")

    # ✅ CORRECCIÓN: calcular nueva affine
    new_affine = affine.copy()
    translation_offset = affine[:3, :3] @ [x0, y0, z0]
    new_affine[:3, 3] = affine[:3, 3] + translation_offset

    print(f"New affine for VOI:\n{new_affine}")

    save_nifti(ct_voi, new_affine, header, output_path)

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

        output_path = os.path.join(output_dir_VOIs, f"{ct_id}_R_{voi_counter}.nii.gz")

        if os.path.exists(ct_path):
            print(f"\n📝 Processing {ct_id}, nodule {nodule_id}")
            print(f"   - Center mm: ({coordX}, {coordY}, {coordZ}), Diameter: {diameter}mm")
            extract_voi_from_coords(ct_path, (coordX, coordY, coordZ), diameter, output_path, padding_mm=20)
            continue
        voi_counter += 1


#%% Visualize example

ct1, _, _ = load_nifti("data/sample/CT/image/LIDC-IDRI-0001.nii.gz")
voi_gt, _, _ = load_nifti("data/full_data/VOIs/image/LIDC-IDRI-0003_R_2.nii.gz")
voi_pred, _, _ = load_nifti("output/VOIs/LIDC-IDRI-0003_R_1.nii.gz")

# Visualize central slice of each
fig, axes = plt.subplots(1, 3, figsize=(12, 5))

mid_ct = ct1.shape[2] // 2
mid_pred = voi_pred.shape[2] // 2
mid_gt = voi_gt.shape[2] // 2

axes[0].imshow(ct1[:, :, mid_ct], cmap='gray')
axes[0].set_title("Full CT (middle slice)")
axes[0].axis('off')

axes[1].imshow(voi_gt[:, :, mid_gt], cmap='gray')
axes[1].set_title("Ground Truth VOI")
axes[1].axis('off')


axes[2].imshow(voi_pred[:,:, mid_pred], cmap='gray')
axes[2].set_title("Extracted VOI")
axes[2].axis('off')

plt.tight_layout()
plt.show()