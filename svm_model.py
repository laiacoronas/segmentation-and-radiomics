# -*- coding: utf-8 -*-
"""
Milestone 3 (Classification): Experimental Design and Data Splitting.
Use a classifier (like SVM) with default parameters and GLCM features and 
implement a validation using:

1. K-folds by slice (StratifiedKFold).
2. K-folds grouping by nodule (StratifiedGroupKFold).

Deliverable: A zip file containing code for the experimental designs.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline


class RadiomicsClassifier:
    def __init__(self, radiomics_path, annotations_path, n_splits=5, random_state=42):
        self.radiomics_path = radiomics_path
        self.annotations_path = annotations_path
        self.n_splits = n_splits
        self.random_state = random_state
        self.data = None

    def load_and_prepare_data(self):
        # Load radiomics data
        radiomics_df = pd.read_csv(self.radiomics_path).drop(columns=["Unnamed: 0"], errors="ignore")
        radiomics_df[['patient_id', 'nodule_id']] = radiomics_df['image'].str.extract(r'^(LIDC-IDRI-\d+)_R_(\d+)$')
        radiomics_df['nodule_id'] = radiomics_df['nodule_id'].astype(int)

        # Load annotations data
        annotations_df = pd.read_excel(self.annotations_path)
        annotations_df['image'] = annotations_df['patient_id'].astype(str) + '_R_' + annotations_df['nodule_id'].astype(str)

        # Merge data
        self.data = radiomics_df.merge(annotations_df[['image', 'Diagnosis_value']], on='image', how='left')

    def split_and_train(self, group_by_nodule=False):
        X = self.data.drop(columns=['image', 'patient_id', 'nodule_id', 'Diagnosis_value'])
        y = self.data['Diagnosis_value']
        groups = self.data['nodule_id'] if group_by_nodule else None

        # Set up the classifier pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC(probability=True))
        ])

        # Select the K-fold strategy
        if group_by_nodule:
            kfold = StratifiedGroupKFold(n_splits=self.n_splits)
            print("Using StratifiedGroupKFold (Grouping by nodule)")
        else:
            kfold = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            print("Using StratifiedKFold (Grouping by slice)")

        # Perform cross-validation
        for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y, groups)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)[:, 1]
            
            # Classification report
            print(f"\nFold {fold + 1} Results:")
            print(confusion_matrix(y_test, y_pred))
            print(classification_report(y_test, y_pred))
            
            # Confusion matrix plot
            cm_display = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap='Blues')
            cm_display.ax_.set_title(f"Confusion Matrix - Fold {fold + 1}")
            plt.show()
            
            # ROC Curve plot
            roc_display = RocCurveDisplay.from_predictions(y_test, y_proba)
            roc_display.ax_.set_title(f"ROC Curve - Fold {fold + 1}")
            plt.show()


def main():
    classifier = RadiomicsClassifier("glcm_features.csv", "output/Annotations_MaxVote.xlsx")
    classifier.load_and_prepare_data()
    classifier.split_and_train(group_by_nodule=False)  # K-folds by slice
    classifier.split_and_train(group_by_nodule=True)   # K-folds by nodule


if __name__ == "__main__":
    main()
