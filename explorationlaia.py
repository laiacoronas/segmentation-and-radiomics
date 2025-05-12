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
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from scipy.stats import ttest_ind, mannwhitneyu
from statsmodels.stats.multitest import multipletests


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


def run_clustering(X, n_clusters=2, eps=1.5, min_samples=50):
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA for dimensionality reduction
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)
    df_pca = pd.DataFrame(components, columns=['PCA1', 'PCA2'])
    
    # Run clustering algorithms
    cluster_results = {
        "Hierarchical": AgglomerativeClustering(n_clusters=n_clusters).fit_predict(components),
        "KMeans": KMeans(n_clusters=n_clusters, random_state=42).fit_predict(components),
        "DBSCAN": DBSCAN(eps=eps, min_samples=min_samples).fit_predict(components)
    }
    
    return df_pca, cluster_results


def visualize_clusters(df_pca, cluster_results):
    for method, labels in cluster_results.items():
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=df_pca['PCA1'], y=df_pca['PCA2'], hue=labels, palette="viridis", s=100)
        plt.title(f"{method} Clustering using PCA")
        plt.legend(title="Cluster")
        plt.show()


def statistical_tests(X, target_col, alpha=0.01, correction="bonferroni"):
    group1 = X[X[target_col] == 0]
    group2 = X[X[target_col] == 1]
    p_values = {}
    
    for col in X.drop(columns=target_col):
        # Use non-parametric test if the data is not normally distributed
        if group1[col].std() < 1e-8 or group2[col].std() < 1e-8:
            continue
        stat, p = ttest_ind(group1[col], group2[col], equal_var=True)
        p_values[col] = p
    
    # Multiple testing correction
    features, p_vals = list(p_values.keys()), list(p_values.values())
    reject, corrected_p_vals, _, _ = multipletests(p_vals, alpha=alpha, method=correction)
    significant_features = [f for f, r in zip(features, reject) if r]
    
    return significant_features, dict(zip(features, corrected_p_vals))


def visualize_correlation(df, features):
    plt.figure(figsize=(12, 10))
    corr_matrix = df[features].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Heatmap of Selected Features')
    plt.show()


def main():
    radiomics_df, annotations_df = load_data("glcm_features.csv", "output/Annotations_MaxVote.xlsx")
    
    # Merge radiomics and annotations
    radiomics_df = radiomics_df.merge(
        annotations_df[['image', 'Diagnosis_value']],
        on='image',
        how='left'
    )
    
    # Preprocess data
    X_radiomics = preprocess_data(radiomics_df, ['image', 'patient_id', 'nodule_id'])
    
    # Run clustering
    df_pca, cluster_results = run_clustering(X_radiomics)
    visualize_clusters(df_pca, cluster_results)
    
    # Statistical tests
    significant_features, corrected_p_vals = statistical_tests(X_radiomics, 'Diagnosis_value')
    print("Significant Features:", significant_features)
    
    # Visualize correlation of selected features
    if significant_features:
        visualize_correlation(X_radiomics, significant_features)


if __name__ == "__main__":
    main()
