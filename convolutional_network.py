# -*- coding: utf-8 -*-
"""
Milestone 3 (Classification): Feature Extraction using a Pre-trained
Convolutional Network.

1. Load a predefined VGG model and modify it to extract the features of the 1st fully connected (FC) layer.

2. Apply techniques for reduction of dimensionality.

Deliverable: A zip file containing code for the extraction and selection of VGG
features.
"""

#%% 0. Import necessary libraries
import os
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image

# Set working directory to current script's location
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#%% 1. Define function to extract ROI from the central slice

def extract_roi(image_3d, mask_3d):
    z_center = image_3d.shape[2] // 2  # Use central slice
    
    img_slice = image_3d[:, :, z_center]
    mask_slice = mask_3d[:, :, z_center]
    
    roi = img_slice * (mask_slice > 0)

    coords = np.argwhere(mask_slice > 0)
    
    # If no ROI, return normalized full slice
    if coords.size == 0:
        slice_norm = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min())
        slice_uint8 = (slice_norm * 255).astype(np.uint8)
        return slice_uint8

    # Crop to bounding box of the mask
    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0) + 1 
    roi_cropped = roi[x0:x1, y0:y1]

    # Normalize and convert to uint8 image
    roi_norm = (roi_cropped - roi_cropped.min()) / (roi_cropped.max() - roi_cropped.min())
    roi_uint8 = (roi_norm * 255).astype(np.uint8)
    
    return roi_uint8

#%% 2. Convert 3D NIfTI volumes to 2D PNG ROI slices

nii_dir = "data/full_data/VOIs/image"      
mask_dir = "data/full_data/VOIs/nodule_mask" 
png_dir = "data/full_data/VOIs/image_processed"

os.makedirs(png_dir, exist_ok=True)

for file in os.listdir(nii_dir):
    if file.endswith(".nii.gz"):
        
        # Load image and corresponding mask
        nii_img = nib.load(os.path.join(nii_dir, file))
        img_data = nii_img.get_fdata()
        nii_mask = nib.load(os.path.join(mask_dir, file))
        mask_data = nii_mask.get_fdata()
        
        # Extract and preprocess the ROI
        roi_uint8 = extract_roi(img_data, mask_data)
        
        # Convert and resize to 224x224 RGB image
        im = Image.fromarray(roi_uint8)
        im = im.convert("RGB")
        im = im.resize((224, 224))

        # Save as PNG
        im.save(os.path.join(png_dir, file.replace(".nii.gz", ".png")))

print("ROI to PNG conversion complete.")

#%% 3. Extract deep features using VGG16

# Initialize the model and set its output to the first fully connected layer
base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

features_list = []
image_names = []

for filename in os.listdir(png_dir):
    if filename.endswith('.png'):
        
        # Load and preprocess the image for VGG16
        img_path = os.path.join(png_dir, filename)
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        
        # Extract features from 'fc1' layer
        features = model.predict(x)
        features_list.append(features.flatten())
        image_names.append(filename)

# Save features to CSV
df_features = pd.DataFrame(features_list)
df_features.insert(0, 'image', image_names)
df_features.to_csv('feature_spaces/CN_features.csv', index=False)
print("Features extracted and saved to CN_features.csv")

#%% 4. Hybrid Feature Selection (parallel filter + wrapper)

# Load annotations and map labels
annotations_df = pd.read_excel("output/Annotations_MaxVote.xlsx")
annotations_df['image'] = annotations_df['patient_id'].astype(str) + '_R_' + annotations_df['nodule_id'].astype(str) + ".png"
image_to_label = dict(zip(annotations_df['image'], annotations_df['Diagnosis_value']))

df_features['label'] = df_features['image'].map(image_to_label)
df_features = df_features.dropna(subset=['label'])

X = df_features.drop(columns=['image', 'label'])
y = df_features['label'].astype(int)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Filter method: SelectKBest (ANOVA)
kbest = SelectKBest(score_func=f_classif, k=100)
kbest.fit(X_scaled, y)
kbest_mask = kbest.get_support()

# Wrapper method: SelectFromModel (Random Forest)
rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42), threshold='median')
rf_selector.fit(X_scaled, y)
rf_mask = rf_selector.get_support()

# Combine selected features (union of masks)
combined_mask = kbest_mask | rf_mask
X_selected = X_scaled[:, combined_mask]

# Save selected features
selected_df = pd.DataFrame(X_selected)
selected_df.insert(0, 'image', df_features['image'].values)
os.makedirs("feature_spaces", exist_ok=True)
selected_df.to_csv("feature_spaces/selected_CN_features.csv", index=False)

print("Hybrid-selected features saved to feature_spaces/selected_CN_features.csv")
