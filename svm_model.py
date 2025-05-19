# -*- coding: utf-8 -*-
"""
Milestone 3 (Classification): Experimental Design and Data Splitting.
Use a classifier (like SVM) with default parameters and GLCM features and 
implement a validation using:

1. K-folds by slice (StratifiedKFold).
2. K-folds grouping by nodule (StratifiedGroupKFold).

Deliverable: A zip file containing code for the experimental designs.
"""

#%% 0. Import necessary libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

#%% 1. Define classifier class

class RadiomicsClassifier:
    def __init__(self, radiomics_path, n_splits=5, random_state=42):
        self.radiomics_path = radiomics_path
        self.n_splits = n_splits
        self.random_state = random_state
        self.data = None

    #%% 2. Load and prepare data
    
    def load_and_prepare_data(self):
        
        # Load radiomics data
        npz = np.load(self.radiomics_path, allow_pickle=True)
        
        # Separate arrays
        slice_features = npz['slice_features']
        slice_meta = npz['slice_meta']

        # Convert types and join
        slice_meta = pd.DataFrame(slice_meta, columns=["filename", "patient_id", "nodule_id", "diagnosis"])
        slice_features = pd.DataFrame(slice_features)
        slice_features = slice_features.join(slice_meta)

        # Mapping diagnosis values
        slice_features['diagnosis'] = slice_features['diagnosis'].map({
            'Malignant': 1,
            'Benign': 0,
            'NoNod': np.nan
        })
        radiomics_df = slice_features.dropna(subset=['diagnosis'])
        
        # # Balancear clases por undersampling
        malignant = radiomics_df[radiomics_df['diagnosis'] == 1.0]
        benign = radiomics_df[radiomics_df['diagnosis'] == 0.0]
        malignant_sampled = malignant.sample(n=len(benign), random_state= self.random_state)
        radiomics_df = pd.concat([malignant_sampled, benign], axis=0).sample(frac=1, random_state= self.random_state).reset_index(drop=True)
        
        # Separar features y labels
        X = radiomics_df.drop(columns=['filename', 'patient_id', 'nodule_id', 'diagnosis'])
        y = radiomics_df['diagnosis']
        meta = radiomics_df[['filename', 'patient_id', 'nodule_id']]
        
        self.data = (meta, X, y)

    #%% 3. Perform cross-validation and training
    
    def split_and_train(self, group_by_nodule=False):
        
        # Get data and groups
        X = self.data[1]
        y = self.data[2]
        radiomics_df = self.data[0]
        groups = radiomics_df['nodule_id'] if group_by_nodule else None

        # Set up pipeline with scaler and SVM classifier
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC(probability=True, class_weight='balanced'))
        ])

        # Choose cross-validation strategy
        if group_by_nodule:
            kfold = StratifiedGroupKFold(n_splits=self.n_splits)
            print("Using StratifiedGroupKFold (Grouping by nodule)")
        else:
            kfold = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            print("Using StratifiedKFold (Grouping by slice)")
            
        # Initialize lists to store metrics
        all_auc = []
        all_acc = []

        # Train and evaluate model for each fold
        for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y, groups)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # smote = SMOTE(random_state=self.random_state)
            # X_train, y_train = smote.fit_resample(X_train, y_train)
            
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)[:, 1]

            # Compute and store metrics
            auc = roc_auc_score(y_test, y_proba)
            acc = np.mean(y_pred == y_test)
            all_auc.append(auc)
            all_acc.append(acc)
            
            # Print evaluation metrics
            print(f"\nFold {fold + 1} Results:")
            print(confusion_matrix(y_test, y_pred))
            print(classification_report(y_test, y_pred))
            print(f"ROC AUC: {auc:.4f}")
            print(f"Accuracy: {acc:.4f}")
            
            # Plot confusion matrix
            cm_display = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap='Blues')
            cm_display.ax_.set_title(f"Confusion Matrix - Fold {fold + 1}")
            plt.show()
            
            # Plot ROC curve
            roc_display = RocCurveDisplay.from_predictions(y_test, y_proba)
            roc_display.ax_.set_title(f"ROC Curve - Fold {fold + 1}")
            plt.show()
            
        # Final summary
        print("\n=== Cross-validation summary ===")
        print(f"Mean ROC AUC: {np.mean(all_auc):.4f} ± {np.std(all_auc):.4f}")
        print(f"Mean Accuracy: {np.mean(all_acc):.4f} ± {np.std(all_acc):.4f}")

#%% 4. Run main workflow

def main():
    
    classifier = RadiomicsClassifier("feature_spaces/slice_glcm1d.npz")
    classifier.load_and_prepare_data()
    classifier.split_and_train(group_by_nodule=False)  # K-folds by slice
    classifier.split_and_train(group_by_nodule=True)   # K-folds by nodule

if __name__ == "__main__":
    main()

