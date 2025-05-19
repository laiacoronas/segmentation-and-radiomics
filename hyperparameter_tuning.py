# -*- coding: utf-8 -*-
"""
Milestone 3 (Classification): Hyper-parameters Optimization
1. Use a brute force grid search
2. Use a random search using sklearn.
3. Use Optuna.

Deliverable: A zip file containing code for the optimization of hyper-parameters
"""

#%% 0. Importing necessary packages

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.pipeline import Pipeline
from optuna.samplers import TPESampler

#%% 1. Load and prepare data

def load_and_prepare_data(radiomics_path, rd_state):
        # Load radiomics data
        npz = np.load(radiomics_path, allow_pickle=True)
        
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
        malignant_sampled = malignant.sample(n=len(benign), random_state= rd_state)
        radiomics_df = pd.concat([malignant_sampled, benign], axis=0).sample(frac=1, random_state= rd_state).reset_index(drop=True)
        
        # Separar features y labels
        X = radiomics_df.drop(columns=['filename', 'patient_id', 'nodule_id', 'diagnosis'])
        y = radiomics_df['diagnosis']
        meta = radiomics_df[['filename', 'patient_id', 'nodule_id']]
        
        return (meta, X, y)

#%% 2. Define helper to evaluate a model

def evaluate_model(X, y, meta, model, group_by_nodule=False, n_splits=5, random_state=42, method_name=""):
    
    groups = meta['nodule_id'] if group_by_nodule else None
    kfold = StratifiedGroupKFold(n_splits=n_splits) if group_by_nodule else StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    all_auc = []
    all_acc = []

    # Train and evaluate model for each fold
    for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y, groups)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # smote = SMOTE(random_state=self.random_state)
        # X_train, y_train = smote.fit_resample(X_train, y_train)
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

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

#%% 3. Grid Search

def grid_search(X, y):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(probability=True))
    ])

    param_grid = {
        'svc__C': [0.1, 1, 10],
        'svc__gamma': ['scale', 'auto'],
        'svc__kernel': ['linear', 'rbf']
    }

    grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc', verbose=1)
    grid.fit(X, y)

    print("Best GridSearchCV parameters:", grid.best_params_)
    return grid.best_estimator_

#%% 4. Random Search

def random_search(X, y):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(probability=True))
    ])

    param_dist = {
        'svc__C': np.logspace(-2, 2, 20),
        'svc__gamma': ['scale', 'auto'],
        'svc__kernel': ['linear', 'rbf', 'poly', 'sigmoid']
    }

    rand = RandomizedSearchCV(pipeline, param_dist, n_iter=20, cv=5, scoring='roc_auc', random_state=42, verbose=1)
    rand.fit(X, y)

    print("Best RandomizedSearchCV parameters:", rand.best_params_)
    return rand.best_estimator_

#%% 5. Optuna optimization

def optuna_search(X, y):
    def objective(trial):
        C = trial.suggest_loguniform('C', 1e-2, 1e2)
        kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly', 'sigmoid'])
        gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svc', SVC(C=C, kernel=kernel, gamma=gamma, probability=True))
        ])

        scores = []
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            pipeline.fit(X_train, y_train)
            y_proba = pipeline.predict_proba(X_test)[:, 1]
            scores.append(roc_auc_score(y_test, y_proba))

        return np.mean(scores)

    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(lambda trial: objective(trial), n_trials=30)

    print("Best Optuna parameters:", study.best_params)
    best_params = study.best_params

    best_model = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(
            C=best_params['C'],
            kernel=best_params['kernel'],
            gamma=best_params['gamma'],
            probability=True
        ))
    ])

    return best_model

#%% 6. Main

if __name__ == "__main__":
    meta, X, y = load_and_prepare_data("feature_spaces/slice_glcm1d.npz", 42)

    print("\n Grid search results:")
    best_grid = grid_search(X, y)
    evaluate_model(X, y, meta, best_grid, group_by_nodule=True, method_name="Grid Search")

    print("\n Random search results:")
    best_random = random_search(X, y)
    evaluate_model(X, y, meta, best_random, group_by_nodule=True, method_name="Random Search")

    print("\n Optuna search results:")
    best_optuna = optuna_search(X, y)
    evaluate_model(X, y, meta, best_optuna, group_by_nodule=True, method_name="Optuna Search")
