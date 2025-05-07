# -*- coding: utf-8 -*-
"""
Milestone 1 (Segmentation): Generation of an Annotated Dataset.
 1. Extract VOI (Volume of Interest) from the CTs (intensity and mask).
 2. Produce a single annotation for each lesion from the 4 radiologists’ annotations using Max-Voting.
 3. Make Max-Voting to obtain the “Diagnosis”: if two or more radiologists have characterized the nodule with a Malignancy score > 3, then Diagnosis=1 (malignant), otherwise Diagnosis=0 (benign).
"""

#%% Environment onfiguration

from scipy.ndimage import gaussian_filter, label
from skimage.filters import threshold_otsu
from skimage.morphology import ball, opening
from nibabel.affines import apply_affine
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import nibabel as nib
import os

current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)
os.chdir(current_dir)

#%% Upload metadata

df = pd.read_excel('data/sample/MetadatabyAnnotation.xlsx')

#%% Extract VOIs

def load_nifti(path):
    nii = nib.load(path)
    return nii.get_fdata(), nii.affine, nii.header

def save_nifti(volume, affine, header, path):
    nib.save(nib.Nifti1Image(volume.astype(np.uint8), affine, header), path)

def extract_voi_from_coords(ct_path, gt_path, low_coords, high_coords, output_path):
    ct, affine, header = load_nifti(ct_path)
    gt, _, _ = load_nifti(gt_path)

    
    # Convert world (mm) → voxel indices
    low_vox = np.round(apply_affine(np.linalg.inv(affine), low_coords)).astype(int)
    high_vox = np.round(apply_affine(np.linalg.inv(affine), high_coords)).astype(int)
    
    z1, y1, x1 = low_vox
    z2, y2, x2 = high_vox
    
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])
    z1, z2 = sorted([z1, z2])
    
    print(low_vox)
    print(high_vox)
    
    ct_voi = ct[z1:z2, y1:y2, x1:x2]
    save_nifti(ct_voi, affine, header, output_path)
    shape = ct_voi.shape
    gt_shape = gt.shape

    print(f"VOI saved to {output_path} with shape {shape}")
    print(f"Ground truth shape {gt_shape}")


# Computing VOIs

ct_base_path = "data/sample/CT/image"
output_dir_VOIs = "output/VOIs"
os.makedirs(output_dir_VOIs, exist_ok=True)

last_nodule_id = None
last_ct_id = None
voi_counter = 1

for idx, row in df.iterrows():
    ct_id = row['patient_id']
    nodule_id = row['nodule_id']
    
    # Reset nodule counter
    if ct_id != last_ct_id:
        voi_counter = 1
        last_ct_id = ct_id

    # Compute only for different nodules
    if nodule_id != last_nodule_id:
        last_nodule_id = nodule_id
        
        x_low, y_low, z_low = row['bboxLowX'], row['bboxLowY'], row['bboxLowZ']
        x_high, y_high, z_high = row['bboxHighX'], row['bboxHighY'], row['bboxHighZ']
        ct_path = os.path.join(ct_base_path, f"{ct_id}.nii.gz")
        
        if ct_id == "LIDC-IDRI-0003":
            gt_path = os.path.join("data/full_data/VOIs/image/", f"{ct_id}_R_{voi_counter+1}.nii.gz")
        else:
            gt_path = os.path.join("data/full_data/VOIs/image/", f"{ct_id}_R_{voi_counter}.nii.gz")
        
        if os.path.exists(ct_path):
            output_path = os.path.join(output_dir_VOIs, f"{ct_id}_R_{voi_counter}.nii.gz")
            extract_voi_from_coords(ct_path, gt_path, (x_low, y_low, z_low), (x_high, y_high, z_high), output_path)        
        else:
            continue

        voi_counter += 1
    
        ct_path = os.path.join(ct_base_path, f"{ct_id}.nii.gz")
#%% Visualize results

# Load images
ct1, affine1, header = load_nifti(os.path.join(current_dir,"data/sample/CT/image/LIDC-IDRI-0001.nii.gz"))
r1, affine_r1, header_r1 = load_nifti(os.path.join(current_dir,"output/VOIs/LIDC-IDRI-0001_R_1.nii.gz"))
r1_gt, affine_r1_gt, header_r1_gt = load_nifti(os.path.join(current_dir,"data/full_data/VOIs/image/LIDC-IDRI-0001_R_1.nii.gz"))

# Visualize a middle slice 

# Create subplots
fig, axes = plt.subplots(1, 3, figsize=(10, 5))

axes[0].imshow(ct1[:, :,  ct1.shape[2] // 2], cmap='gray')
axes[0].set_title('Full CT')
axes[0].axis('off')

axes[1].imshow(r1_gt[:, :,  r1_gt.shape[2] // 2], cmap='gray')
axes[1].set_title('Ground truth')
axes[1].axis('off')

axes[2].imshow(r1[:, :, 0, cmap='gray')
axes[2].set_title('Computed mask')
axes[2].axis('off')

plt.tight_layout()
plt.show()
    