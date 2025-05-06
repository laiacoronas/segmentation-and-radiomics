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
from skimage.morphology import closing, opening, disk
from skimage.measure import label, regionprops
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from skimage.filters import sobel, prewitt, roberts
from skimage.feature import canny
from sklearn.preprocessing import StandardScaler
<<<<<<< HEAD
from skimage.filters import laplace, gaussian
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.util import img_as_ubyte
from skimage.segmentation import watershed
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
=======

#%% Environment onfiguration

current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)
os.chdir(current_dir)
>>>>>>> 0247c4da04ef0f5e56215887e13f42f7f623e154
 
#%% Load data

# Define loading function
def load_nii(filepath):
    """Load NIfTI image and return array and affine"""
    img = nib.load(filepath)
    data = img.get_fdata()
    affine = img.affine
    return data, affine

# Load images
<<<<<<< HEAD
#data1, affine1 =load_nii(r"C:\Users\Maria Fité\Documents\MSC - HEALTH DATA SCIENCE\Q2\Machine Learning (ML)\Challange 2\Dataset\VOIs\image\LIDC-IDRI-0001_R_1.nii.gz")
#data_gt, affine_gt =load_nii(r"C:\Users\Maria Fité\Documents\MSC - HEALTH DATA SCIENCE\Q2\Machine Learning (ML)\Challange 2\Dataset\VOIs\nodule_mask\LIDC-IDRI-0001_R_1.nii.gz")

#data1, affine1 =load_nii(r"C:\Users\lclai\Desktop\VOIs\VOIs\image\LIDC-IDRI-0001_R_1.nii.gz")
#data_gt, affine_gt =load_nii(r"C:\Users\lclai\Desktop\VOIs\VOIs\nodule_mask\LIDC-IDRI-0001_R_1.nii.gz")
=======
data1, affine1 =load_nii(os.path.join(current_dir,"data/full_data/VOIs/image/LIDC-IDRI-0001_R_1.nii.gz"))
ct1, affine_ct_1 =load_nii(os.path.join(current_dir,"data/sample/CT/image/LIDC-IDRI-0001.nii.gz"))
data_gt, affine_gt =load_nii(os.path.join(current_dir,"data/full_data/VOIs/nodule_mask/LIDC-IDRI-0001_R_1.nii.gz"))
>>>>>>> 0247c4da04ef0f5e56215887e13f42f7f623e154

data1, affine1 =load_nii(r"C:\Users\CLAUDIA\Desktop\Clau\MHEDAS\PRIMER\SEGON SEM\ML\challenge2\image\LIDC-IDRI-0007_R_1.nii.gz")
data_gt, affine_gt =load_nii(r"C:\Users\CLAUDIA\Desktop\Clau\MHEDAS\PRIMER\SEGON SEM\ML\challenge2\nodule_mask\LIDC-IDRI-0007_R_1.nii.gz")

# Visualize a middle slice 

slice_index = data1.shape[2] // 2

# Create triple subplot
fig, axes = plt.subplots(1, 3, figsize=(10, 5))

axes[0].imshow(ct1[:, :,  ct1.shape[2] // 2], cmap='gray')
axes[0].set_title('Full CT Image')
axes[0].axis('off')

axes[1].imshow(data1[:, :, slice_index], cmap='gray')
axes[1].set_title('VOI Image')
axes[1].axis('off')

axes[2].imshow(data_gt[:, :, slice_index], cmap='gray')
axes[2].set_title('Nodule Mask')
axes[2].axis('off')

plt.tight_layout()
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

#%% Post-processing Simple threshoolding

# We will apply an oppening 
size_kernel = 3
kernel = Morpho.cube(size_kernel)
binary_thOpen = Morpho.binary_opening(binary_simpleth, kernel)


# Initialize clean volume
mask_clean_volume_th = np.zeros_like(binary_thOpen, dtype=np.uint8)

# Run for all the volume
for idx in range(binary_thOpen.shape[2]):
    opened_slice = binary_thOpen[:, :, idx] # select slice
    labeled_slice = label(opened_slice) # label regions
    regions = regionprops(labeled_slice) # obtain region properties
    min_area = 10
    if regions: # select biggest region
        largest_region = max(regions, key=lambda r: r.area)
        mask_clean_volume_th[:, :, idx] = (labeled_slice == largest_region.label).astype(np.uint8)
    else:
        mask_clean_volume_th[:, :, idx] = 0


# Show results
plt.figure(figsize=(12, 4))

plt.subplot(1, 4, 1)
plt.imshow(binary_simpleth[:, :, slice_idx], cmap='gray')
plt.title('Binarized Image')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(binary_thOpen[:, :, slice_idx], cmap='gray')
plt.title("Post-opening")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(mask_clean_volume_th[:, :, slice_idx], cmap='gray')
plt.title("Cleaned mask")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(data_gt[:, :, slice_idx], cmap='gray')
plt.title("Ground Truth")
plt.axis("off")

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

#%% Post-processing Otsu

# We will apply an oppening 
size_kernel = 3
kernel = Morpho.cube(size_kernel)
binary_otsuOpen = Morpho.binary_opening(binary_otsu, kernel)

# Initialize clean volume
mask_clean_volume_otsu = np.zeros_like(binary_otsuOpen, dtype=np.uint8)

# Run for all the volume
for idx in range(binary_otsuOpen.shape[2]):
    opened_slice = binary_otsuOpen[:, :, idx] # select slice
    labeled_slice = label(opened_slice) # label regions
    regions = regionprops(labeled_slice) # obtain region properties
    min_area = 10
    if regions: # select biggest region
        largest_region = max(regions, key=lambda r: r.area)
        mask_clean_volume_th[:, :, idx] = (labeled_slice == largest_region.label).astype(np.uint8)
    else:
        mask_clean_volume_th[:, :, idx] = 0


# Show results
plt.figure(figsize=(12, 4))

plt.subplot(1, 4, 1)
plt.imshow(binary_otsu[:, :, slice_idx], cmap='gray')
plt.title('Binarized Image Otsu')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(binary_otsuOpen[:, :, slice_idx], cmap='gray')
plt.title("Post-opening")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(mask_clean_volume_otsu[:, :, slice_idx], cmap='gray')
plt.title("Cleaned mask")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(data_gt[:, :, slice_idx], cmap='gray')
plt.title("Ground Truth")
plt.axis("off")

plt.tight_layout()
plt.show()

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
metrics_simple = compute_metrics(mask_clean_volume_th, data_gt)
metrics_otsu = compute_metrics(mask_clean_volume_otsu, data_gt)

print("\n--- Metrics (Simple Threshold) ---")
for k, v in metrics_simple.items():
    print(f"{k}: {v:.3f}")

print("\n--- Metrics (Otsu) ---")
for k, v in metrics_otsu.items():
    print(f"{k}: {v:.3f}")

#%% KMeans over filter banks

binary_kmeans = np.zeros_like(smoothed_data, dtype=np.uint8)
kmeans_seg = np.zeros_like(smoothed_data, dtype=np.uint8)

for idx in range(smoothed_data.shape[2]):
    slice_data = smoothed_data[:, :, idx]
    
    # Filter banks
    features = []
    features.append(slice_data.flatten())                                                  # Intensity
    features.append(sobel(slice_data).flatten())                                           # Sobel
    features.append(prewitt(slice_data).flatten())                                         # Prewitt
    features.append(roberts(slice_data).flatten())                                         # Roberts
    features.append(laplace(gaussian(slice_data, sigma=1)).flatten())                      # laplace
    normalized_slice = np.clip(slice_data, -1, 1)                                          
    features.append(entropy(img_as_ubyte(normalized_slice), disk(3)).flatten())            # Entropy
    features.append(gaussian(slice_data, sigma=2).flatten())                               # Gauss

    X = np.array(features).T

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply KMeans
    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    kmeans_seg_slice = kmeans_labels.reshape(slice_data.shape)
    kmeans_seg[:, :, idx] = kmeans_seg_slice
    
    # Identify the cluster with the highest mean intensity
    cluster_means = [slice_data[kmeans_seg_slice == i].mean() for i in range(4)]
    background_cluster = np.argmin(cluster_means)


    # Create binary mask
    binary_kmeans[:, :, idx] = (kmeans_seg_slice != background_cluster).astype(np.uint8)

# Show results
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.imshow(kmeans_seg[:, :, slice_idx], cmap='gray')
plt.title("K-Means Segmentation")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(binary_kmeans[:, :, slice_idx], cmap='gray')
plt.title("Binary result")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(data_gt[:, :, slice_idx], cmap='gray')
plt.title("Ground Truth")
plt.axis('off')

plt.tight_layout()
plt.show()

#%% Post processing and evaluation

# We will apply an oppening 
size_kernel = 3
kernel = Morpho.cube(size_kernel)
binary_kmeansOpen = Morpho.binary_opening(binary_kmeans, kernel)

# Initialize clean volume
mask_clean_volume_kmeans = np.zeros_like(binary_kmeans, dtype=np.uint8)


# Run for all the volume
for idx in range(binary_kmeansOpen.shape[2]):
    opened_slice = binary_kmeansOpen[:, :, idx]  # select slice
    labeled_slice = label(opened_slice)          # label regions
    regions = regionprops(labeled_slice)         # obtain region properties
    min_area = 10  # Instead of keeping only the largest region, keep all regions above a minimum area threshold.
    for region in regions:
        if region.area >= min_area:
            mask_clean_volume_kmeans[:, :, idx][labeled_slice == region.label] = 1
        else:
            mask_clean_volume_kmeans[:, :, idx] = 0

# Show results
plt.figure(figsize=(12, 4))

plt.subplot(1, 4, 1)
plt.imshow(binary_kmeans[:, :, slice_idx], cmap='gray')
plt.title('Binarized Image K-Means')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(binary_kmeansOpen[:, :, slice_idx], cmap='gray')
plt.title("Post-opening")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(mask_clean_volume_kmeans[:, :, slice_idx], cmap='gray')
plt.title("Cleaned mask")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(data_gt[:, :, slice_idx], cmap='gray')
plt.title("Ground Truth")
plt.axis("off")

plt.tight_layout()
plt.show()

# Evaluate KMeans
metrics_kmeans = compute_metrics(mask_clean_volume_kmeans, data_gt)
print("\n--- Metrics (KMeans) ---")
for k, v in metrics_kmeans.items():
    print(f"{k}: {v:.3f}")


#%% Summary of all Methods

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