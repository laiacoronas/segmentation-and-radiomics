# -*- coding: utf-8 -*-

"""
Milestone 1 (Segmentation): Nodule Segmentation. 
Apply unsupervised techniques to obtain a segmentation of lesions in VOIs:
 1. Use a classic standard pipeline over intensity volumes.
 2. Use Otsu threholding and different morphological operations.
 3. Quantify the performance using fair segmentation metrics.
 4. Use kmeans over classic filter banks.
 5. Compare between different unsupervised methods.
"""

#%% Preparing enviroment

# Import libraries
import nibabel as nib
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu
from skimage import morphology as Morpho

# Set working directory
path = os.chdir(r"C:\Users\Maria Fité\Documents\MSC - HEALTH DATA SCIENCE\Q2\Machine Learning (ML)\Challange 2\Dataset\VOIs")

#%% Load data

# Define loading function
def load_nii(filepath):
    """Load NIfTI image and return array and affine"""
    img = nib.load(filepath)
    data = img.get_fdata()
    affine = img.affine
    return data, affine

# Load images
data1, affine1 =load_nii(r"C:\Users\Maria Fité\Documents\MSC - HEALTH DATA SCIENCE\Q2\Machine Learning (ML)\Challange 2\Dataset\VOIs\image\LIDC-IDRI-0001_R_1.nii.gz")
data_gt, affine_gt =load_nii(r"C:\Users\Maria Fité\Documents\MSC - HEALTH DATA SCIENCE\Q2\Machine Learning (ML)\Challange 2\Dataset\VOIs\nodule_mask\LIDC-IDRI-0001_R_1.nii.gz")

# Visualize a middle slice 
slice_index = data1.shape[2] // 2
plt.imshow(data1[:, :, slice_index], cmap='gray')
plt.title(f'Axial slice {slice_index}')
plt.axis('off')
plt.show()

#%% Pre-processing

# Apply Gaussian Filter
smoothed_data = gaussian_filter(data1, sigma=0.5)  # Optimal to remove noise and les blurried

# Visualize a middle slice comparison
slice_idx = data1.shape[2] // 2
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(data1[:, :, slice_idx], cmap='gray')
plt.title('Original Slice')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(smoothed_data[:, :, slice_idx], cmap='gray')
plt.title('Smoothed with Gaussian (σ=0.5)')
plt.axis('off')

plt.tight_layout()
plt.show()

#%% Simple thresholding

# Plot histogram of intensity values
plt.figure(figsize=(6, 4))
plt.hist(smoothed_data.flatten(), bins=100)
plt.title("Intensity Histogram (Smoothed Image)")
plt.xlabel("Intensity")
plt.ylabel("Frequency")
plt.show()

# Biniarize the image
simple_threshold = -400  # Adjusted after seeing the histogram
binary_simpleth = (smoothed_data > simple_threshold).astype(np.uint8)

# Show binary mask of an slide
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.imshow(data1[:, :, slice_idx], cmap='gray')
plt.title('Original Slice')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(binary_simpleth[:, :, slice_idx], cmap='gray')
plt.title(f'Binarized Image (th = {simple_threshold})')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(data_gt[:, :, slice_idx], cmap='gray')
plt.title(f'Binarized Image Ground Truth')
plt.axis('off')

plt.tight_layout()
plt.show()

#%% Otsu thresholding 

# Find otsu threshold
otsu_threshold = threshold_otsu(smoothed_data)
print(f"Otsu threshold: {otsu_threshold:.3f}")

# Binarize image using Otsu threshold
binary_otsu = (smoothed_data > otsu_threshold).astype(np.uint8)

# Show binary mask of an slide
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.imshow(data1[:, :, slice_idx], cmap='gray')
plt.title('Original Slice')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(binary_otsu[:, :, slice_idx], cmap='gray')
plt.title(f'Binarized Image (th = {otsu_threshold})')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(data_gt[:, :, slice_idx], cmap='gray')
plt.title(f'Binarized Image Ground Truth')
plt.axis('off')

plt.tight_layout()
plt.show()

#%% Post-processing

# We will apply an oppening 
size_kernel = 3
kernel = Morpho.cube(size_kernel)
binary_otsuOpen = Morpho.binary_opening(binary_otsu, kernel)

# Show binary mask of an slide
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.imshow(binary_otsuOpen[:, :, slice_idx], cmap='gray')
plt.title('Opening')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(binary_otsu[:, :, slice_idx], cmap='gray')
plt.title(f'Binarized Image (th = {otsu_threshold})')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(data_gt[:, :, slice_idx], cmap='gray')
plt.title(f'Binarized Image Ground Truth')
plt.axis('off')

plt.tight_layout()
plt.show()


#%% Metrics
