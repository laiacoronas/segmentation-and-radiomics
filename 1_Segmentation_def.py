#!/usr/bin/env python3
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
from skimage.measure import label, regionprops
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from skimage.filters import sobel, prewitt, roberts
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import label, center_of_mass
from skimage.morphology import binary_opening, cube
from skimage.filters import laplace, gaussian
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.util import img_as_ubyte
import pandas as pd
import matplotlib.pyplot as plt
 
#%% Load data

# Define loading function
def load_nii(filepath):
    """Load NIfTI image and return array and affine"""
    img = nib.load(filepath)
    data = img.get_fdata()
    affine = img.affine
    return data, affine

# Define saving function
def save_nii(data, affine, output_path):
    """Save NIfTI image"""
    img = nib.Nifti1Image(data, affine)
    nib.save(img, output_path)

# Change working directory
current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)
os.chdir(current_dir)

# Load images
data1, affine1 =load_nii(r"data/full_data/VOIs/image/LIDC-IDRI-0017_R_1.nii.gz")
data_gt, affine_gt =load_nii(r"data/full_data/VOIs/nodule_mask/LIDC-IDRI-0017_R_1.nii.gz")

# Visualize a middle slice 
slice_index = data1.shape[2] // 2
plt.imshow(data1[:, :, slice_index], cmap='gray')
plt.title(f'Axial slice {slice_index}')
plt.axis('off')
plt.show()

#%% Pre-processing

# Define preprocessing
def preprocessing (data1):
    # Apply Gaussian Filter
    smoothed_data = gaussian_filter(data1, sigma=0.5)  # Optimal to remove noise and les blurried
    return smoothed_data 

# Apply preprocessing
smoothed_data = preprocessing(data1)

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

# Define simple threshoold
def simple_threshoold(smoothed_data):
    # Biniarize the image
    simple_threshold = -400  # Adjusted after seeing the histogram
    binary_simpleth = (smoothed_data > simple_threshold).astype(np.uint8)
    return binary_simpleth

# Apply simple threshoold
binary_simpleth = simple_threshoold(smoothed_data)

# Show binary mask of an slide
plt.figure(figsize=(10, 5))
simple_threshold = -400

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

# Function to find central region
def get_central_region(mask):
    labeled_mask, num_features = label(mask)
    if num_features == 0:
        return np.zeros_like(mask), 0

    center = np.array(mask.shape) / 2
    centroids = center_of_mass(mask, labeled_mask, range(1, num_features + 1))
    distances = [np.linalg.norm(np.array(c) - center) for c in centroids]
    central_label = np.argmin(distances) + 1
    region = (labeled_mask == central_label).astype(np.uint8)
    return region, region.sum()

# Function for the post processing
def postprocessing(binary):
    # Parameters and initial mask
    kernel = cube(3)
    threshold = data1.size // 7  
    n1 = 1
    current_mask = binary.copy()
    # Start iterating
    while True:
        central_region, area = get_central_region(current_mask)     # Extract central region
        if area < threshold or area == 0:
            break  # Stop
        if n1 > 20:
            break     # Stop iterating
        if n1 > 10 :    # Change kernel size
            kernel = kernel = cube(5)
        current_mask = binary_opening(central_region, kernel) # Apply opening and continue
        n1 = n1 + 1
    # Making final result
    cleaned = central_region  
    kernel_2 = Morpho.cube(2)
    cleaned_mask = Morpho.binary_dilation(cleaned, kernel_2) # Dilation
    return cleaned_mask


# We will apply the post processing
mask_clean_volume_th = postprocessing(binary_simpleth)

# Show results
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(binary_simpleth[:, :, slice_idx], cmap='gray')
plt.title('Binarized Image Otsu')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(mask_clean_volume_th[:, :, slice_idx], cmap='gray')
plt.title("Cleaned mask")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(data_gt[:, :, slice_idx], cmap='gray')
plt.title("Ground Truth")
plt.axis("off")

plt.tight_layout()
plt.show()

#%% Otsu thresholding 

# Define otsu binarization
def otsu_threshoold (smoothed_data):
    otsu_threshold = threshold_otsu(smoothed_data) # Find otsu threshold
    binary_otsu = (smoothed_data > otsu_threshold).astype(np.uint8) # Binarize image using Otsu threshold
    return binary_otsu, otsu_threshold

# Apply otsu method 
binary_otsu, otsu_threshold = otsu_threshoold (smoothed_data)

# Show binary mask of an slide
print(f"Otsu threshold: {otsu_threshold:.3f}")
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

# We will apply the post processing
mask_clean_volume_otsu = postprocessing(binary_otsu)

# Show results
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(binary_otsu[:, :, slice_idx], cmap='gray')
plt.title('Binarized Image Otsu')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(mask_clean_volume_otsu[:, :, slice_idx], cmap='gray')
plt.title("Cleaned mask")
plt.axis("off")

plt.subplot(1, 3, 3)
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

# Evaluate methods
metrics_simple = compute_metrics(mask_clean_volume_th, data_gt)
metrics_otsu = compute_metrics(mask_clean_volume_otsu, data_gt)

print("\n--- Metrics (Simple Threshold) ---")
for k, v in metrics_simple.items():
    print(f"{k}: {v:.3f}")

print("\n--- Metrics (Otsu) ---")
for k, v in metrics_otsu.items():
    print(f"{k}: {v:.3f}")

#%% KMeans over filter banks 

# define function with kmeans
def kmeans_segmentation(smoothed_data):

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
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans_labels = kmeans.fit_predict(X_scaled)
        kmeans_seg_slice = kmeans_labels.reshape(slice_data.shape)
        kmeans_seg[:, :, idx] = kmeans_seg_slice
        # Identify the cluster with the highest mean intensity
        cluster_means = [slice_data[kmeans_seg_slice == i].mean() for i in range(3)]
        background_cluster = np.argmin(cluster_means)
        # Create binary mask
        binary_kmeans[:, :, idx] = (kmeans_seg_slice != background_cluster).astype(np.uint8)
    return binary_kmeans, kmeans_seg

# Apply kmeans 
binary_kmeans, kmeans_seg = kmeans_segmentation(smoothed_data)

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

# We will apply the post processing
mask_clean_volume_kmeans = postprocessing(binary_kmeans)

# Show results
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(binary_kmeans[:, :, slice_idx], cmap='gray')
plt.title('Binarized Image K-Means')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(mask_clean_volume_kmeans[:, :, slice_idx], cmap='gray')
plt.title("Cleaned mask")
plt.axis("off")

plt.subplot(1, 3, 3)
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

#%% Loop for all the images

# Paths
input_folder = r"data/full_data/VOIs/image"
input_folder_gt = r"data/full_data/VOIs/nodule_mask"
output_folder_sth = r"output/Maks generated/simple_th"
output_folder_otsu = r"output/Maks generated/otsu"
output_folder_kmeans = r"output/Maks generated/k_means"

# Ensure folders exist
os.makedirs(output_folder_sth, exist_ok=True)
os.makedirs(output_folder_otsu, exist_ok=True)
os.makedirs(output_folder_kmeans, exist_ok=True)

# Lista para almacenar métricas
metrics_sth_list = []
metrics_otsu_list = []
metrics_kmeans_list = []

# Process each image .nii o .nii.gz
for filename in os.listdir(input_folder):
    if filename.endswith('.nii') or filename.endswith('.nii.gz'):
        
        # Define paths
        input_path = os.path.join(input_folder, filename)
        input_path_gt = os.path.join(input_folder_gt, filename)
        output_path_sth = os.path.join(output_folder_sth, filename)
        output_path_otsu = os.path.join(output_folder_otsu, filename)
        output_path_kmeans = os.path.join(output_folder_kmeans, filename)

        # Load image
        data, affine = load_nii(input_path)
        data_gt, affine_gt = load_nii(input_path_gt)
        print ('Processing image')
        
        # Apply preprocessing
        smoothed_data = preprocessing(data)
        
        # Apply binarization
        binary_simpleth = simple_threshoold(smoothed_data) 
        binary_otsu, _ = otsu_threshoold(smoothed_data)
        binary_kmeans, _ = kmeans_segmentation(smoothed_data)
        
        # Apply post processing
        mask_clean_volume_th = postprocessing(binary_simpleth)
        mask_clean_volume_otsu = postprocessing(binary_otsu)
        mask_clean_volume_kmeans = postprocessing(binary_kmeans)
        
        # Compute metrics
        metrics_simple = compute_metrics(mask_clean_volume_th, data_gt)
        metrics_otsu = compute_metrics(mask_clean_volume_otsu, data_gt)
        metrics_kmeans = compute_metrics(mask_clean_volume_kmeans, data_gt)
        print ('Processing DONE')

        # Save processed images
        save_nii(mask_clean_volume_th, affine, output_path_sth)
        save_nii(mask_clean_volume_otsu, affine, output_path_otsu)
        save_nii(mask_clean_volume_kmeans, affine, output_path_kmeans)

        # Save metrics
        metrics_sth_list.append(metrics_simple)
        metrics_otsu_list.append(metrics_otsu)
        metrics_kmeans_list.append(metrics_kmeans)

#%% Summary of all methods

# list to df
df_simple = pd.DataFrame(metrics_sth_list)
df_otsu = pd.DataFrame(metrics_otsu_list)
df_kmeans = pd.DataFrame(metrics_kmeans_list)

# Table mean
mean_df = pd.DataFrame({
    "Method": ["Simple Threshold", "Otsu + Morphology", "KMeans + Filters"],
    "IoU": [df_simple["IoU"].mean(), df_otsu["IoU"].mean(), df_kmeans["IoU"].mean()],
    "Dice": [df_simple["Dice"].mean(), df_otsu["Dice"].mean(), df_kmeans["Dice"].mean()],
    "Precision": [df_simple["Precision"].mean(), df_otsu["Precision"].mean(), df_kmeans["Precision"].mean()],
    "Recall": [df_simple["Recall"].mean(), df_otsu["Recall"].mean(), df_kmeans["Recall"].mean()],
})

# Table std
std_df = pd.DataFrame({
    "Method": ["Simple Threshold", "Otsu + Morphology", "KMeans + Filters"],
    "IoU": [df_simple["IoU"].std(), df_otsu["IoU"].std(), df_kmeans["IoU"].std()],
    "Dice": [df_simple["Dice"].std(), df_otsu["Dice"].std(), df_kmeans["Dice"].std()],
    "Precision": [df_simple["Precision"].std(), df_otsu["Precision"].std(), df_kmeans["Precision"].std()],
    "Recall": [df_simple["Recall"].std(), df_otsu["Recall"].std(), df_kmeans["Recall"].std()],
})

# Display tables
print("Tabla de medias:")
print(mean_df.round(4))
print("\nTabla de desviaciones estándar:")
print(std_df.round(4))

# Violin plot for simple threshold
metric_names = df_simple.columns.tolist()
data = [df_simple[metric].values for metric in metric_names]
plt.figure(figsize=(8, 6))
parts = plt.violinplot(data, showmeans=False, showmedians=True, showextrema=True)
for pc in parts['bodies']:
    pc.set_facecolor('#1f77b4')
    pc.set_edgecolor('black')
    pc.set_alpha(0.4)
for i, y in enumerate(data):
    x = np.random.normal(loc=i + 1, scale=0.05, size=len(y)) 
    plt.scatter(x, y, color='black', s=1, alpha=0.6)
plt.xticks(ticks=range(1, len(metric_names) + 1), labels=metric_names)
plt.title("Distribution of metrics for the simple threshoold method")
plt.ylabel("Value")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Violin plot for otsu
metric_names = df_otsu.columns.tolist()
data = [df_otsu[metric].values for metric in metric_names]
plt.figure(figsize=(8, 6))
parts = plt.violinplot(data, showmeans=False, showmedians=True, showextrema=True)
for pc in parts['bodies']:
    pc.set_facecolor('#1f77b4')
    pc.set_edgecolor('black')
    pc.set_alpha(0.4)
for i, y in enumerate(data):
    x = np.random.normal(loc=i + 1, scale=0.05, size=len(y)) 
    plt.scatter(x, y, color='black', s=1, alpha=0.6)
plt.xticks(ticks=range(1, len(metric_names) + 1), labels=metric_names)
plt.title("Distribution of metrics for the Otsu method")
plt.ylabel("Value")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Violin plot for kmeans
metric_names = df_kmeans.columns.tolist()
data = [df_kmeans[metric].values for metric in metric_names]
plt.figure(figsize=(8, 6))
parts = plt.violinplot(data, showmeans=False, showmedians=True, showextrema=True)
for pc in parts['bodies']:
    pc.set_facecolor('#1f77b4')
    pc.set_edgecolor('black')
    pc.set_alpha(0.4)
for i, y in enumerate(data):
    x = np.random.normal(loc=i + 1, scale=0.05, size=len(y))  
    plt.scatter(x, y, color='black', s=1, alpha=0.6)
plt.xticks(ticks=range(1, len(metric_names) + 1), labels=metric_names)
plt.title("Distribution of metrics for the K-Means method")
plt.ylabel("Value")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
