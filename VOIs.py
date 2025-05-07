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

# Change working directory
current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)
os.chdir(current_dir)

#%% Load metadata

df = pd.read_excel('data/sample/MetadatabyAnnotation.xlsx')

#%% Utility functions

def load_nifti(path):
    nii = nib.load(path)
    data = nii.get_fdata()
    affine = nii.affine
    header = nii.header
    return data, affine, header

def save_nifti(volume, affine, header, path):
    nib.save(nib.Nifti1Image(volume, affine, header), path)

def extract_voi_from_coords(ct_path, bbox_low_mm, bbox_high_mm, output_path, padding_mm=10):
    ct, affine, header = load_nifti(ct_path)
    
    # Expand bounding box (padding in mm)
    bbox_low_mm = np.array(bbox_low_mm) - padding_mm
    bbox_high_mm = np.array(bbox_high_mm) + padding_mm
    
    # Convert LPS (DICOM) → RAS (NIFTI) (flip X and Y axes)
    bbox_low_mm_ras = bbox_low_mm.copy()
    bbox_high_mm_ras = bbox_high_mm.copy()
    bbox_low_mm_ras[0] = -bbox_low_mm_ras[0]
    bbox_low_mm_ras[1] = -bbox_low_mm_ras[1]
    bbox_high_mm_ras[0] = -bbox_high_mm_ras[0]
    bbox_high_mm_ras[1] = -bbox_high_mm_ras[1]

    # Homogeneous coordinates
    mm_coords_hom_low = np.append(bbox_low_mm_ras, 1)
    mm_coords_hom_high = np.append(bbox_high_mm_ras, 1)
    
    # Transform mm → voxel
    voxel_coords_low = np.linalg.inv(affine) @ mm_coords_hom_low
    voxel_coords_high = np.linalg.inv(affine) @ mm_coords_hom_high
    voxel_indices_low = np.round(voxel_coords_low[:3]).astype(int)
    voxel_indices_high = np.round(voxel_coords_high[:3]).astype(int)
    
    # Ensure indices are ordered
    voxel_indices_low_fixed = np.minimum(voxel_indices_low, voxel_indices_high)
    voxel_indices_high_fixed = np.maximum(voxel_indices_low, voxel_indices_high)
    
    x0, y0, z0 = voxel_indices_low_fixed
    x1, y1, z1 = voxel_indices_high_fixed

    # Clip to image bounds
    x0 = max(0, x0)
    y0 = max(0, y0)
    z0 = max(0, z0)
    x1 = min(ct.shape[0], x1)
    y1 = min(ct.shape[1], y1)
    z1 = min(ct.shape[2], z1)

    if x0 >= x1 or y0 >= y1 or z0 >= z1:
        return  # Invalid box → skip

    ct_voi = ct[x0:x1, y0:y1, z0:z1]

    # Compute new affine
    new_affine = affine.copy()
    translation_offset = affine[:3, :3] @ [x0, y0, z0]
    new_affine[:3, 3] = affine[:3, 3] + translation_offset

    save_nifti(ct_voi, new_affine, header, output_path)

#%% Compute VOIs

ct_base_path = "data/sample/CT/image"
output_dir_VOIs = "output/VOIs"
os.makedirs(output_dir_VOIs, exist_ok=True)

last_ct_id = None
last_nodule_id = None
voi_counter = 1

for idx, row in df.iterrows():
    ct_id = row['patient_id']
    nodule_id = row['nodule_id']

    if ct_id != last_ct_id:
        if ct_id == "LIDC-IDRI-0003":
            voi_counter = 2
        else:
            voi_counter = 1
        last_ct_id = ct_id
        
    if nodule_id != last_nodule_id:
        last_nodule_id = nodule_id
        
        bboxLowX, bboxLowY, bboxLowZ = row['bboxLowX'], row['bboxLowY'], row['bboxLowZ']
        bboxHighX, bboxHighY, bboxHighZ = row['bboxHighX'], row['bboxHighY'], row['bboxHighZ']

        ct_path = os.path.join(ct_base_path, f"{ct_id}.nii.gz")
        output_path = os.path.join(output_dir_VOIs, f"{ct_id}_R_{voi_counter}.nii.gz")

        if os.path.exists(ct_path):
            extract_voi_from_coords(ct_path, (bboxLowX, bboxLowY, bboxLowZ), (bboxHighX, bboxHighY, bboxHighZ), output_path)
            voi_counter += 1
        else:
            continue

#%% Visualization

ct_path = "data/sample/CT/image"
output_dir_VOIs = "output/VOIs"
ground_truth_dir = "data/full_data/VOIs/image"

ct_files = [f for f in os.listdir(ct_path) if f.endswith('.nii.gz')]
voi_files = [f for f in os.listdir(output_dir_VOIs) if f.endswith('.nii.gz')]

for i,voi_file in enumerate(voi_files):
    
    if "LIDC-IDRI-0001" in voi_file:
        ct ,_ ,_ = load_nifti(os.path.join(ct_path, ct_files[0]))
    if "LIDC-IDRI-0003" in voi_file:
        ct ,_ ,_ = load_nifti(os.path.join(ct_path, ct_files[1]))
    if "LIDC-IDRI-0005" in voi_file:
        ct ,_ ,_ = load_nifti(os.path.join(ct_path, ct_files[2]))
    
    voi_pred ,_ ,_ = load_nifti(os.path.join(output_dir_VOIs, voi_file))
    voi_gt ,_ ,_ = load_nifti(os.path.join(ground_truth_dir, voi_file))
    
    mid_ct = ct.shape[2] // 2
    mid_pred = voi_pred.shape[2] // 2
    mid_gt = voi_gt.shape[2] // 2

    fig, axes = plt.subplots(1, 3, figsize=(8, 4))
    
    axes[0].imshow(ct[:, :, mid_ct], cmap='gray')
    axes[0].set_title("Full CT (middle slice)")
    axes[0].axis('off')
    
    axes[1].imshow(voi_gt[:, :, mid_gt], cmap='gray')
    axes[1].set_title("Ground Truth VOI")
    axes[1].axis('off')
    
    axes[2].imshow(voi_pred[:, :, mid_pred], cmap='gray')
    axes[2].set_title(f"Extracted VOI")
    axes[2].axis('off')
    
    plt.suptitle(f"Visualization for {voi_file}", fontsize=16)
    plt.tight_layout()
    plt.show()