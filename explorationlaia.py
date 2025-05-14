# -*- coding: utf-8 -*-
"""
Milestone 2 (Classification): Data Exploration
1. Use different unsupervised techniques (eg. hierarchical clustering) and 
   statistical tests to get correlations across radiological descriptions and 
   also detect those annotations more relevant to the diagnosis.
"""

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


def preprocess_data(df, drop_cols):
    return df.drop(columns=drop_cols, errors="ignore").select_dtypes(include="number")


def run_clustering(X, n_clusters=2, eps=1.5, min_samples=50, method='pca', title_suffix='All Features'):
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Dimensionality reduction
    if method == 'pca':
        reducer = PCA(n_components=2)
    elif method == 'tsne':
        reducer = TSNE(n_components=2, perplexity=30, random_state=42)
    elif method == 'umap':
        reducer = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    else:
        raise ValueError(f"Unknown method: {method}")
    components = reducer.fit_transform(X_scaled)
    df_reduced = pd.DataFrame(components, columns=['Component 1', 'Component 2'])
    
    # Run clustering algorithms
    cluster_results = {
        "Hierarchical": AgglomerativeClustering(n_clusters=n_clusters).fit_predict(components),
        "KMeans": KMeans(n_clusters=n_clusters, random_state=42).fit_predict(components),
        "DBSCAN": DBSCAN(eps=eps, min_samples=min_samples).fit_predict(components)
    }
    
    # Visualize results
    for cluster_name, labels in cluster_results.items():
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=df_reduced['Component 1'], y=df_reduced['Component 2'], hue=labels, palette="viridis", s=100)
        plt.title(f"{cluster_name} Clustering ({method.upper()}) - {title_suffix}")
        plt.legend(title="Cluster")
        plt.show()
    
    return df_reduced, cluster_results


def statistical_tests(X, target_col, alpha=0.01, correction="bonferroni"):
    group1 = X[X[target_col] == 0]
    group2 = X[X[target_col] == 1]
    p_values = {}
    
    for col in X.drop(columns=target_col):
        if group1[col].std() < 1e-8 or group2[col].std() < 1e-8:
            continue
        stat, p = ttest_ind(group1[col], group2[col], equal_var=True)
        p_values[col] = p
    
    # Multiple testing correction
    features, p_vals = list(p_values.keys()), list(p_values.values())
    reject, corrected_p_vals, _, _ = multipletests(p_vals, alpha=alpha, method=correction)
    significant_features = [f for f, r in zip(features, reject) if r]
    non_significant = [(f, p) for f, r, p in zip(features, reject, corrected_p_vals) if not r]
    print(f"Non-significant features: {non_significant}")
    
    return significant_features, dict(zip(features, corrected_p_vals))


def remove_highly_correlated_features(df, features, threshold=0.75):
    corr_matrix = df[features].corr().abs()
    
    # Plot the heatmap
    plt.figure(figsize=(15, 15))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar_kws={"shrink": 0.75})
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.show()
    
    # See highly correlated features
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    print(f"Highly correlated features (corr > {threshold}): {to_drop}")
    return [col for col in features if col not in to_drop]


def main():
    radiomics_df, annotations_df = load_data("glcm_features.csv", "output/Annotations_MaxVote.xlsx")
    radiomics_df = radiomics_df.merge(
        annotations_df[['image', 'Diagnosis_value']],
        on='image',
        how='left'
    )

    # Radiomic features
    X_radiomics = preprocess_data(radiomics_df, ['image', 'patient_id', 'nodule_id'])
    for title_suffix, features in [
        ('All Features', X_radiomics),
        ('Significant Features', X_radiomics[statistical_tests(X_radiomics, 'Diagnosis_value')[0]]),
        ('Significant Features (No High Corr)', X_radiomics[remove_highly_correlated_features(X_radiomics, statistical_tests(X_radiomics, 'Diagnosis_value')[0], threshold=0.8)])
    ]:
        for method in ['pca', 'tsne', 'umap']:
            run_clustering(features, method=method, title_suffix=title_suffix)

    # Annotation features
    X_annotations = preprocess_data(annotations_df, ['image', 'patient_id', 'nodule_id'])
    for title_suffix, features in [
        ('All Features', X_annotations),
        ('Significant Features', X_annotations[statistical_tests(X_annotations, 'Diagnosis_value')[0]]),
        ('Significant Features (No High Corr)', X_annotations[remove_highly_correlated_features(X_annotations, statistical_tests(X_annotations, 'Diagnosis_value')[0], threshold=0.8)])
    ]:
        for method in ['pca', 'tsne', 'umap']:
            run_clustering(features, method=method, title_suffix=title_suffix)

if __name__ == "__main__":
    main()
