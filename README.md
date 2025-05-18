# Pulmonary Lesion Analysis with Machine Learning

## Project Overview
This project implements a complete machine learning pipeline for pulmonary mass analysis using CT scans. It spans unsupervised lesion segmentation, radiomic and deep feature extraction, supervised classification, and hyperparameter optimization. The goal is to compare traditional radiomics-based approaches with deep learning representations for diagnostic classification tasks.

## Objectives
- Segment pulmonary nodules from CT scans using unsupervised techniques.
- Extract radiomic features (GLCM) and deep features using VGG16.
- Apply feature selection methods to identify the most relevant features.
- Evaluate classifiers using cross-validation grouped by slice and by nodule.
- Tune hyperparameters using Grid Search, Random Search, and Optuna.

## Directory Structure
<pre>
├── data/               # CT volumes, masks, and annotations
├── feature_spaces/     # Extracted and selected features
├── output/             # Intermediate files, results, and annotations
└── README.md           # Project description
</pre>


## Milestones
- Milestone 1: Segmentation using unsupervised methods and VOI extraction
- Milestone 2: Feature extraction (GLCM and VGG16) and unsupervised data analysis
- Milestone 3: Deep feature extraction, feature selection and supervised classification
- Milestone 4: Hyperparameter optimization using Grid Search, Random Search, and Optuna

## Tools and Libraries
- nibabel, SimpleITK: medical image I/O
- scikit-learn: ML pipeline, feature selection, cross-validation
- PyRadiomics: radiomic feature extraction
- TensorFlow / Keras: VGG16 feature extraction
- Optuna: Bayesian hyperparameter optimization
- matplotlib, seaborn: plotting and visualization
