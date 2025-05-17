import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import nibabel as nib
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from PIL import Image

def extract_roi(image_3d, mask_3d):
    z_center = image_3d.shape[2] // 2
    
    img_slice = image_3d[:, :, z_center]
    mask_slice = mask_3d[:, :, z_center]
    
    roi = img_slice * (mask_slice > 0)

    coords = np.argwhere(mask_slice > 0)
    if coords.size == 0:
     
        slice_norm = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min())
        slice_uint8 = (slice_norm * 255).astype(np.uint8)
        return slice_uint8

    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0) + 1 
    
    roi_cropped = roi[x0:x1, y0:y1]

    roi_norm = (roi_cropped - roi_cropped.min()) / (roi_cropped.max() - roi_cropped.min())
    roi_uint8 = (roi_norm * 255).astype(np.uint8)
    
    return roi_uint8

nii_dir = "data/full_data/VOIs/image"      
mask_dir = "data/full_data/VOIs/nodule_mask" 
png_dir = "data/full_data/VOIs/image_processed"

os.makedirs(png_dir, exist_ok=True)

for file in os.listdir(nii_dir):
    if file.endswith(".nii.gz"):
        
        nii_img = nib.load(os.path.join(nii_dir, file))
        img_data = nii_img.get_fdata()
        nii_mask = nib.load(os.path.join(mask_dir, file))
        mask_data = nii_mask.get_fdata()
        
        roi_uint8 = extract_roi(img_data, mask_data)
        
       
        im = Image.fromarray(roi_uint8)
        im = im.convert("RGB")  
        im = im.resize((224, 224))
        

        im.save(os.path.join(png_dir, file.replace(".nii.gz", ".png")))

print("Conversión a PNG con ROI completada.")


base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

features_list = []
image_names = []

for filename in os.listdir(png_dir):
    if filename.endswith('.png'):
        img_path = os.path.join(png_dir, filename)
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = model.predict(x)
        features_list.append(features.flatten())
        image_names.append(filename)

df_features = pd.DataFrame(features_list)
df_features.insert(0, 'image', image_names)
df_features.to_csv('CN_features.csv', index=False)
print("Features extraídas y guardadas en CN_features.csv")

X = df_features.drop(columns=['image'])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

df_reduced = pd.DataFrame({
    'PC1': X_pca[:,0],
    'PC2': X_pca[:,1],
    'TSNE1': X_tsne[:,0],
    'TSNE2': X_tsne[:,1],
    'image': image_names
})
df_reduced.to_csv('reduced_CN_features.csv', index=False)
print("Reducción dimensional guardada en reduced_CN_features.csv")

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.scatter(df_reduced['PC1'], df_reduced['PC2'], alpha=0.7)
plt.title('PCA of VGG16 FC1 features')
plt.xlabel('PC1')
plt.ylabel('PC2')

plt.subplot(1,2,2)
plt.scatter(df_reduced['TSNE1'], df_reduced['TSNE2'], alpha=0.7, c='orange')
plt.title('t-SNE of VGG16 FC1 features')
plt.xlabel('TSNE1')
plt.ylabel('TSNE2')

plt.tight_layout()
plt.show()
