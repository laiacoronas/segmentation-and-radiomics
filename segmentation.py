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
#path = os.chdir(r"C:\Users\Maria Fité\Documents\MSC - HEALTH DATA SCIENCE\Q2\Machine Learning (ML)\Challange 2\Dataset\VOIs")
 
#%% Load data

# Define loading function
def load_nii(filepath):
    """Load NIfTI image and return array and affine"""
    img = nib.load(filepath)
    data = img.get_fdata()
    affine = img.affine
    return data, affine

# Load images
#data1, affine1 =load_nii(r"C:\Users\Maria Fité\Documents\MSC - HEALTH DATA SCIENCE\Q2\Machine Learning (ML)\Challange 2\Dataset\VOIs\image\LIDC-IDRI-0001_R_1.nii.gz")
#data_gt, affine_gt =load_nii(r"C:\Users\Maria Fité\Documents\MSC - HEALTH DATA SCIENCE\Q2\Machine Learning (ML)\Challange 2\Dataset\VOIs\nodule_mask\LIDC-IDRI-0001_R_1.nii.gz")

data1, affine1 =load_nii(r"C:\Users\lclai\Desktop\VOIs\VOIs\image\LIDC-IDRI-0001_R_1.nii.gz")
data_gt, affine_gt =load_nii(r"C:\Users\lclai\Desktop\VOIs\VOIs\nodule_mask\LIDC-IDRI-0001_R_1.nii.gz")


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

##PROVA

from skimage.measure import label, regionprops

# Tomamos el slice procesado
opened_slice = binary_otsuOpen[:, :, slice_idx]

# Etiquetar regiones conectadas
labeled_slice = label(opened_slice)

# Obtener propiedades de cada región
regions = regionprops(labeled_slice)

# Seleccionar la región más grande (o la más redonda, si prefieres)
if regions:
    largest_region = max(regions, key=lambda r: r.area)
    mask_clean = (labeled_slice == largest_region.label).astype(np.uint8)
else:
    mask_clean = np.zeros_like(opened_slice)

# Mostrar resultado limpio
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(opened_slice, cmap='gray')
plt.title("Post-opening (conectado)")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(mask_clean, cmap='gray')
plt.title("Región mayor (bolita)")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(data_gt[:, :, slice_idx], cmap='gray')
plt.title("Ground Truth")
plt.axis("off")

plt.tight_layout()
plt.show()


#%% Metrics

from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from skimage.filters import sobel, prewitt, roberts
from sklearn.preprocessing import StandardScaler

#%% Metrics

def compute_metrics(pred, gt):
    """Compute segmentation metrics given prediction and ground truth"""
    pred_flat = pred.flatten()
    gt_flat = gt.flatten()
    
    TP = np.sum((pred_flat == 1) & (gt_flat == 1))
    TN = np.sum((pred_flat == 0) & (gt_flat == 0))
    FP = np.sum((pred_flat == 1) & (gt_flat == 0))
    FN = np.sum((pred_flat == 0) & (gt_flat == 1))

    eps = 1e-7
    iou = TP / (TP + FP + FN + eps)
    dice = 2 * TP / (2 * TP + FP + FN + eps)
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)

    return {"IoU": iou, "Dice": dice, "Precision": precision, "Recall": recall}

# Evaluate simple thresholding
metrics_simple = compute_metrics(binary_simpleth, data_gt)
metrics_otsu = compute_metrics(binary_otsuOpen, data_gt)

print("\n--- Metrics (Simple Threshold) ---")
for k, v in metrics_simple.items():
    print(f"{k}: {v:.3f}")

print("\n--- Metrics (Otsu + Opening) ---")
for k, v in metrics_otsu.items():
    print(f"{k}: {v:.3f}")


#%% KMeans over filter banks

# Prepare filter bank
slice_data = smoothed_data[:, :, slice_idx]

features = []
features.append(slice_data.flatten())                      # Original intensity
features.append(sobel(slice_data).flatten())               # Sobel
features.append(prewitt(slice_data).flatten())             # Prewitt
features.append(roberts(slice_data).flatten())             # Roberts

X = np.array(features).T

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply k-means
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)
kmeans_seg = kmeans_labels.reshape(slice_data.shape)

# Optionally invert if foreground/background flipped
if np.mean(kmeans_seg[data_gt[:, :, slice_idx] == 1]) < 0.5:
    kmeans_seg = 1 - kmeans_seg

# Show results
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.imshow(slice_data, cmap='gray')
plt.title("Original Slice")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(kmeans_seg, cmap='gray')
plt.title("K-Means Filter Bank")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(data_gt[:, :, slice_idx], cmap='gray')
plt.title("Ground Truth")
plt.axis('off')

plt.tight_layout()
plt.show()

# Evaluate KMeans
metrics_kmeans = compute_metrics(kmeans_seg, data_gt[:, :, slice_idx])
print("\n--- Metrics (KMeans) ---")
for k, v in metrics_kmeans.items():
    print(f"{k}: {v:.3f}")


#%% Summary of Methods

import pandas as pd

summary_df = pd.DataFrame({
    "Method": ["Simple Threshold", "Otsu + Morphology", "KMeans + Filters"],
    "IoU": [metrics_simple["IoU"], metrics_otsu["IoU"], metrics_kmeans["IoU"]],
    "Dice": [metrics_simple["Dice"], metrics_otsu["Dice"], metrics_kmeans["Dice"]],
    "Precision": [metrics_simple["Precision"], metrics_otsu["Precision"], metrics_kmeans["Precision"]],
    "Recall": [metrics_simple["Recall"], metrics_otsu["Recall"], metrics_kmeans["Recall"]],
})

print("\n=== Summary of Methods ===")
print(summary_df.round(3))
