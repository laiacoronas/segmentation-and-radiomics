# -*- coding: utf-8 -*-
"""
Milestone 2 (Classification): Extraction of Radiomic Features. 
1. Extract GLCM texture features using the PyRadiomics library.
"""

#%% Preparing enviroment

# Import libraries
import nibabel as nib
import numpy as np
import os
import numpy as np
import SimpleITK as sitk
from radiomics import featureextractor
import pandas as pd

#%% Load data and setting directories

# Define loading function
def load_nii(filepath):
    """Load NIfTI image and return array and affine"""
    img = nib.load(filepath)
    data = img.get_fdata()
    affine = img.affine
    return data, affine

# Set the different directories
image_dir = r"C:\Users\Maria Fité\Documents\MSC - HEALTH DATA SCIENCE\Q2\Machine Learning (ML)\Challange 2\Dataset\VOIs\image\"
mask_dir = r"C:\Users\Maria Fité\Documents\MSC - HEALTH DATA SCIENCE\Q2\Machine Learning (ML)\Challange 2\Dataset\VOIs\nodule_mask\"
output_dir = r"C:\Users\Maria Fité\Documents\MSC - HEALTH DATA SCIENCE\Q2\Machine Learning (ML)\Challange 2\Milestone 2"

#%% Extract features

# Collect features from all images
all_features = []

for filename in os.listdir(input_folder):
    # Extract image ID
    base_name = filename.replace(".nii.gz", "")
    
    # Define paths
    image_path = os.path.join(image_dir, filename)
    mask_path = os.path.join(mask_dir, filename)
   
    # Load image
    image, affine_image = load_nii(image_path)
    mask, affine_mask = load_nii(mask_path)

    # Set up the feature extractor
    extractor = featureextractor.RadiomicsFeatureExtractor()
    
    # Enable GLCM features extraction
    extractor.enableFeaturesByName('glcm')
    
    # Extract features
    features = extractor.execute(image, mask)
    
    # Keep only GLCM features
    glcm_features = {k: v for k, v in features.items() if "glcm" in k}
    glcm_features['image'] = base_name  
    
    # Save features for the image
    all_features.append(glcm_features)
    
    
#%% Save data in a csv

# Create a DataFrame
df = pd.DataFrame(all_features)
#df = df.set_index('image') 

# Save to CSV
csv_path = os.path.join(output_dir, "glcm_features.csv")
df.to_csv(csv_path)

print(f"Saved all GLCM features to {csv_path}")