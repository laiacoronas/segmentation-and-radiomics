# -*- coding: utf-8 -*-
"""
Milestone 2 (Classification): Data Exploration
 1. Use different unsupervised techniques (eg. hierarchical clustering) and 
statistical tests to get correlations across radiological descriptions and 
also detect those annotations more relevant to the diagnosis.
"""

#%% Preparing enviroment

# Load libraries
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_ind
import numpy as np


#%% Loading data

# Load the data
radiomics_df = pd.read_csv(r"glcm_features.csv")
annotations_df = pd.read_excel(r"output/Annotations_MaxVote.xlsx")

# Remove the first column
radiomics_df = radiomics_df.drop(radiomics_df.columns[0], axis=1)

# Add image in the annotations_df
annotations_df['image'] = annotations_df['patient_id'].astype(str) + '_R_' + annotations_df['nodule_id'].astype(str)

# Add patient_id and nodule_id in the radiomics_df
radiomics_df[['patient_id', 'nodule_id']] = radiomics_df['image'].str.extract(r'^(LIDC-IDRI-\d+)_R_(\d+)$')
radiomics_df['nodule_id'] = radiomics_df['nodule_id'].astype(int)

# Filter data frames that have same image
common_images = set(radiomics_df['image']) & set(annotations_df['image'])
radiomics_df = radiomics_df[radiomics_df['image'].isin(common_images)].copy()
annotations_df = annotations_df[annotations_df['image'].isin(common_images)].copy()

# Display head and columns for radiomics
print("Radiomics DataFrame Head:")
print(radiomics_df.head(), "\n")
print("Radiomics Columns:")
print(radiomics_df.columns.tolist(), "\n")

# Display head and columns for annotations
print("Annotations DataFrame Head:")
print(annotations_df.head(), "\n")
print("Annotations Columns:")
print(annotations_df.columns.tolist())

#%% First inspection of radiomic features

# df caractheristics
print("Shape:", radiomics_df.shape)
print("Columns:\n", radiomics_df.columns.tolist())
print("\nData types:\n", radiomics_df.dtypes)
print("\nMissing values:\n", radiomics_df.isnull().sum())

# Summary statistics
print("=== Summary Statistics: Radiomics ===")
print(radiomics_df.describe(include='all'))

#%% Unsupervised thecniques for radiomic features

# Drop non usefull features
X_radiomics = radiomics_df.drop(columns=['image', 'patient_id', 'nodule_id'])

# Step 1: Dimensionality Reduction using PCA
scaler = StandardScaler()
X = scaler.fit_transform(X_radiomics) # Scale features
pca = PCA(n_components=2)
components = pca.fit_transform(X)

# Step 2: Hierarchical Clustering
hc = AgglomerativeClustering(n_clusters=2)
hc_clusters = hc.fit_predict(components)

# Step 3: KMeans Clustering
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans_clusters = kmeans.fit_predict(components)

# Step 4: DBSCAN Clustering
dbscan = DBSCAN(eps=1.5, min_samples=50)
dbscan_clusters = dbscan.fit_predict(components)

# Step 5: Visualizing the results
# Create a DataFrame for easy visualization
df_pca = pd.DataFrame(components, columns=['PCA1', 'PCA2'])

# Visualize Hierarchical Clustering
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df_pca['PCA1'], y=df_pca['PCA2'], hue=hc_clusters, palette="viridis", s=100)
plt.title("Hierarchical Clustering using PCA")
plt.legend(title="Cluster")
plt.show()

# Visualize KMeans Clustering
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df_pca['PCA1'], y=df_pca['PCA2'], hue=kmeans_clusters, palette="viridis", s=100)
plt.title("KMeans Clustering using PCA")
plt.legend(title="Cluster")
plt.show()

# Visualize DBSCAN Clustering
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df_pca['PCA1'], y=df_pca['PCA2'], hue=dbscan_clusters, palette="viridis", s=100)
plt.title("DBSCAN Clustering using PCA")
plt.legend(title="Cluster")
plt.show()

#%% Statistical tests for radiomic features


# Add the diagnosis
radiomics_df = radiomics_df.merge(
    annotations_df[['image', 'Diagnosis_value']],
    on='image',
    how='left'
)

X_radiomics = radiomics_df.drop(columns=['image', 'patient_id', 'nodule_id'])

# T-test
group1 = X_radiomics[X_radiomics['Diagnosis_value'] == 0]
group2 = X_radiomics[X_radiomics['Diagnosis_value'] == 1]
results = {}
for col in X_radiomics.drop(columns='Diagnosis_value'):
    group1_data = group1[col]
    group2_data = group2[col]
    stat, p_value = ttest_ind(group1_data, group2_data, equal_var=True)
    results[col] = p_value

# Identify most important features
results_radiomic_df = pd.DataFrame(list(results.items()), columns=['Variable', 'P-value'])
selected_variables_radiomic = results_radiomic_df[results_radiomic_df['P-value'] < 0.01]
print("Selected variables (p-value < 0.01):")
print(selected_variables_radiomic)

# Correlation matrix of important features
important_cols_rad = selected_variables_radiomic['Variable'].tolist()
selected_features_rad_df = X_radiomics[important_cols_rad]
plt.figure(figsize=(10, 8))
sns.heatmap(selected_features_rad_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap of Selected Features')
plt.show()

# See highly correlated features
corr_matrix = selected_features_rad_df.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
print(f"Highly correlated features (corr > 0.9): {to_drop}")
kept_features = [col for col in selected_features_rad_df.columns if col not in to_drop]
print(f"Relevant features selected: {kept_features}")

#%% First inspection of annotations

# df caractheristics
print("Shape:", annotations_df.shape)
print("Columns:\n", annotations_df.columns.tolist())
print("\nData types:\n", annotations_df.dtypes)
print("\nMissing values:\n", annotations_df.isnull().sum())

# Summary statistics
print("\n=== Summary Statistics: Annotations ===")
print(annotations_df.describe(include='all'))

#%% Unsupervised thecniques for annotations

# Select the numerical features that are relevant for PCA and clustering
X_annotations = annotations_df.select_dtypes(include='number').drop(columns=['nodule_id', 'Diagnosis_value'])

# Step 1: Dimensionality Reduction using PCA
scaler = StandardScaler()
X = scaler.fit_transform(X_annotations) # Scale features
pca = PCA(n_components=2)
components = pca.fit_transform(X)

# Step 2: Hierarchical Clustering
hc = AgglomerativeClustering(n_clusters=2)
hc_clusters = hc.fit_predict(components)

# Step 3: KMeans Clustering
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans_clusters = kmeans.fit_predict(components)

# Step 4: DBSCAN Clustering
dbscan = DBSCAN(eps=1, min_samples=50)
dbscan_clusters = dbscan.fit_predict(components)

# Step 5: Visualizing the results
# Create a DataFrame for easy visualization
df_pca = pd.DataFrame(components, columns=['PCA1', 'PCA2'])

# Visualize Hierarchical Clustering
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df_pca['PCA1'], y=df_pca['PCA2'], hue=hc_clusters, palette="viridis", s=100)
plt.title("Hierarchical Clustering using PCA")
plt.legend(title="Cluster")
plt.show()

# Visualize KMeans Clustering
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df_pca['PCA1'], y=df_pca['PCA2'], hue=kmeans_clusters, palette="viridis", s=100)
plt.title("KMeans Clustering using PCA")
plt.legend(title="Cluster")
plt.show()

# Visualize DBSCAN Clustering
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df_pca['PCA1'], y=df_pca['PCA2'], hue=dbscan_clusters, palette="viridis", s=100)
plt.title("DBSCAN Clustering using PCA")
plt.legend(title="Cluster")
plt.show()

#%% Statistical tests for annotations

# Add the diagnosis
X_annotations = annotations_df.select_dtypes(include='number').drop(columns=['nodule_id'])

# T-test
group1 = X_annotations[X_annotations['Diagnosis_value'] == 0]
group2 = X_annotations[X_annotations['Diagnosis_value'] == 1]
results = {}
for col in X_annotations.drop(columns='Diagnosis_value'):
    group1_data = group1[col]
    group2_data = group2[col]
    if np.std(group1_data) < 1e-8 or np.std(group2_data) < 1e-8: # Skip columns with very low variance in either group
        print(f"Skipping {col} due to near-zero variance.")
        continue
    stat, p_value = ttest_ind(group1_data, group2_data, equal_var=True)
    results[col] = p_value

# Identify most important features
results_annotations_df = pd.DataFrame(list(results.items()), columns=['Variable', 'P-value'])
selected_variables_annotations = results_annotations_df[results_annotations_df['P-value'] < 0.01]
print("Selected variables (p-value < 0.01):")
print(selected_variables_annotations)

# Correlation matrix of important features
important_cols = selected_variables_annotations['Variable'].tolist()
selected_features_df = X_annotations[important_cols]
plt.figure(figsize=(10, 8))
sns.heatmap(selected_features_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap of Selected Features')
plt.show()