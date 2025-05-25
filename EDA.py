# -*- coding: utf-8 -*-
"""
Milestone 2 (Classification): Data Exploration
Use different unsupervised techniques (eg. hierarchical clustering) and 
statistical tests to get correlations across radiological descriptions and 
also detect those annotations more relevant to the diagnosis.
"""

# %% import libraries

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from scipy.stats import ttest_ind, mannwhitneyu
from statsmodels.stats.multitest import multipletests
from umap import UMAP

# %% Load data

annotations_df = pd.read_excel("output/Annotations_MaxVote.xlsx")

def load_data(radiomics_path, annotations_path):
    # Load radiomics data
    radiomics_df = pd.read_csv(radiomics_path).drop(columns=["Unnamed: 0"], errors="ignore")
    # Extract patient and nodule IDs
    radiomics_df[['patient_id', 'nodule_id']] = radiomics_df['image'].str.extract(r'^(LIDC-IDRI-\d+)_R_(\d+)$')
    radiomics_df['nodule_id'] = radiomics_df['nodule_id'].astype(int)

    # Load annotations data
    annotations_df = pd.read_excel(annotations_path)
    annotations_df['image'] = annotations_df['patient_id'].astype(str) + '_R_' + annotations_df['nodule_id'].astype(str)
    
    # Filter common images
    common_images = set(radiomics_df['image']) & set(annotations_df['image'])
    radiomics_df = radiomics_df[radiomics_df['image'].isin(common_images)].copy()
    annotations_df = annotations_df[annotations_df['image'].isin(common_images)].copy()
    
    return radiomics_df, annotations_df

path_radiomics = "glcm_features.csv"
path_annotations = "output/Annotations_MaxVote.xlsx"
radiomics_df, annotations_df = load_data(path_radiomics, path_annotations)
radiomics_df = radiomics_df.merge(
    annotations_df[['image', 'Diagnosis_value']],
    on='image',
    how='left'
)

# %% Tests and correlations

# Correlations
def correlations(X_annotations):
    plt.figure(figsize=(10, 8))
    corr_matrix = X_annotations.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Heatmap of Selected Features')
    plt.show()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.corr().shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]
    print(f"Highly correlated features (corr > 0.8): {to_drop}")


# T-test
def ttest(XY_annotations):
    group0 = XY_annotations[XY_annotations['Diagnosis_value'] == 0]
    group1 = XY_annotations[XY_annotations['Diagnosis_value'] == 1]
    results = {}
    for col in XY_annotations.drop(columns=['Diagnosis_value']):
        group0_data = group0[col]
        group1_data = group1[col]
        if np.std(group0_data) < (0.05/(XY_annotations.shape[1]-1)) or np.std(group1_data) < (0.05/(XY_annotations.shape[1]-1)) : # Skip columns with very low variance in either group
            continue
        stat, p_value = ttest_ind(group0_data, group1_data, equal_var=True)
        results[col] = p_value
        
    results_significant = {k: v for k, v in results.items() if v < (0.05/(XY_annotations.shape[1]-1))}
    
    print(f"Signifficant features for diagnosis: {results_significant}")


# %% Clustering deffinition

def clustering_radiological(X):
    
    # Step 1: Standardize
    X_scaled = StandardScaler().fit_transform(X)
    
    # Step 2: Dimensionality reductions
    reducers = {
        "PCA": PCA(n_components=2),
        "t-SNE": TSNE(n_components=2, perplexity=200, n_iter=500, random_state=42),
        "UMAP": UMAP(n_components=2, n_neighbors=100, min_dist=0.7, random_state=42)
    }
    
    # Step 3: Clustering algorithms
    clusterers = {
        "KMeans": KMeans(n_clusters=2, random_state=42, init='k-means++'),
        "DBSCAN": DBSCAN(eps=0.5, min_samples=10),
        "Hierarchical": AgglomerativeClustering(n_clusters=2)
    }
    
    # Step 4: Apply combinations and plot
    plt.figure(figsize=(15, 12))
    plot_num = 1
    
    for reducer_name, reducer in reducers.items():
        X_reduced = reducer.fit_transform(X_scaled)
        X_reduced = StandardScaler().fit_transform(X_reduced) # Scale the new space for clustering
    
        for clusterer_name, clusterer in clusterers.items():
            labels = clusterer.fit_predict(X_reduced)
    
            plt.subplot(len(reducers), len(clusterers), plot_num)
            for label in set(labels):
                mask = labels == label
                plt.scatter(X_reduced[mask, 0], X_reduced[mask, 1], label=f'Cluster {label}', s=10)
            plt.title(f'{reducer_name} + {clusterer_name}')
            plt.xticks([])
            plt.yticks([])
            plot_num += 1
    
    plt.tight_layout()
    plt.show()
    
#%% Application to Radiological features
X_annotations = annotations_df.filter(regex='_value$').drop(columns=['Diagnosis_value', 'Malignancy_value'], errors='ignore')
XY_annotations = annotations_df.filter(regex='_value$').drop(columns=['Malignancy_value'])
correlations(X_annotations)
ttest(XY_annotations)
clustering_radiological(X_annotations)