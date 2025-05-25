# -*- coding: utf-8 -*-
"""
Milestone 2 (Classification): Extraction of Radiomic Features. 
1. Extract GLCM texture features using the PyRadiomics library.
"""

#%% Preparing enviroment

# Import libraries
import os
import numpy as np
import nibabel as nib
import pandas as pd
from radiomics import featureextractor
import SimpleITK as sitk

#%% Load data and setting directories

# Define loading function
def load_nii(filepath):
    """Load NIfTI image and return array and affine"""
    img = nib.load(filepath)
    data = img.get_fdata()
    affine = img.affine
    return data, affine

# Load annotations for diagnosis
annotations = pd.read_excel("output/Annotations_MaxVote.xlsx")
annotations['Diagnosis_label'] = annotations['Diagnosis_label'].str.capitalize()

# Set the different directories
image_dir = r"data/full_data/VOIs/image"
mask_dir = r"data/full_data/VOIs/nodule_mask"
current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, "feature_spaces")

#%% Extraction of features

# Feature extractor
extractor = featureextractor.RadiomicsFeatureExtractor()
extractor.enableFeaturesByName(glcm=[])

# Storage
slice_features = []
slice_meta = []

# Loop through cases
for filename in os.listdir(image_dir):
    if not filename.endswith(".nii.gz"):
        continue

    base_name = filename.replace(".nii.gz", "")
    patient_id = base_name.split("_")[0]
    num = int(patient_id.split("-")[-1])
    nodule_id = int(base_name.split("_")[-1])

    image_path = os.path.join(image_dir, filename)
    mask_path = os.path.join(mask_dir, filename)

    image_3d = load_nii(image_path)
    mask_3d = load_nii(mask_path)

    # Find slices with mask > 0 or voxels aligned
    for z in range(mask_3d.shape[2]):
      n_voxels = np.count_nonzero(mask_3d[:, :, z])
      nonzero_rows = np.any(mask_3d[:, :, z], axis=1).sum()
      nonzero_cols = np.any(mask_3d[:, :, z], axis=0).sum()

      if n_voxels <= 1 or nonzero_rows == 1 or nonzero_cols == 1:
          continue

      # Convert slice to SimpleITK image
      img_slice = sitk.GetImageFromArray(image_3d[:, :, z])
      mask_slice = sitk.GetImageFromArray(mask_3d[:, :, z])

      # Extract features
      result = extractor.execute(img_slice, mask_slice)

      # Extract only GLCM features
      glcm_values = [v for k, v in result.items() if "glcm" in k]
      glcm_floats = [float(x) for arr in glcm_values for x in np.ravel(arr)]
      slice_features.append(glcm_floats)

      # Compose slice ID and diagnosis
      slice_id = f"{patient_id}_GT1_{nodule_id}"
      diagnosis_row = annotations[
          (annotations["patient_id"] == patient_id) &
          (annotations["nodule_id"] == nodule_id)
      ]
      diagnosis = diagnosis_row["Diagnosis_label"].values[0] if not diagnosis_row.empty else "Unknown"

      slice_meta.append([slice_id, num, nodule_id, diagnosis])

#%% Save data

np.savez(
    os.path.join(output_dir, "slice_glcm_features.npz"),
    slice_features=slice_features,
    slice_meta=slice_meta)

print(f"Saved slice-wise features to {output_dir}/slice_glcm_features.npz")