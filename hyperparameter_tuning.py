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
def load_and_prepare_data(radiomics_path, annotations_path):
    radiomics_df = pd.read_csv(radiomics_path).drop(columns=["Unnamed: 0"], errors="ignore")
    radiomics_df[['patient_id', 'nodule_id']] = radiomics_df['image'].str.extract(r'^(LIDC-IDRI-\d+)_R_(\d+)$')
    radiomics_df['nodule_id'] = radiomics_df['nodule_id'].astype(int)

    annotations_df = pd.read_excel(annotations_path)
    annotations_df['image'] = annotations_df['patient_id'].astype(str) + '_R_' + annotations_df['nodule_id'].astype(str)

    data = radiomics_df.merge(annotations_df[['image', 'Diagnosis_value']], on='image', how='left')
    return data

#%% 2. Define helper to run and evaluate a model

def evaluate_model(X, y, model, group_by_nodule=False, n_splits=5, random_state=42, method_name=""):
    groups = X['nodule_id'] if group_by_nodule else None
    features = X.drop(columns=['image', 'patient_id', 'nodule_id', 'Diagnosis_value'])

    kfold = StratifiedGroupKFold(n_splits=n_splits) if group_by_nodule else StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for fold, (train_idx, test_idx) in enumerate(kfold.split(features, y, groups)):
        X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        print(f"\nFold {fold + 1}")
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")

        # Confusion Matrix plot
        cm_display = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap='Blues')
        cm_display.ax_.set_title(f"Confusion Matrix - Fold {fold + 1} ({method_name})")
        plt.show()

        # ROC Curve plot
        roc_display = RocCurveDisplay.from_predictions(y_test, y_proba)
        roc_display.ax_.set_title(f"ROC Curve - Fold {fold + 1} ({method_name})")
        plt.show()

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
    grid.fit(X.drop(columns=['image', 'patient_id', 'nodule_id']), y)

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
    rand.fit(X.drop(columns=['image', 'patient_id', 'nodule_id']), y)

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
    data = load_and_prepare_data("glcm_features.csv", "output/Annotations_MaxVote.xlsx")
    X = data
    y = data["Diagnosis_value"]

    print("\n Grid search results:")
    best_grid = grid_search(X, y)
    evaluate_model(X, y, best_grid, group_by_nodule=True, method_name="Grid Search")

    print("\n Random search results:")
    best_random = random_search(X, y)
    evaluate_model(X, y, best_random, group_by_nodule=True, method_name="Random Search")

    print("\n Optuna search results:")
    best_optuna = optuna_search(X.drop(columns=['image', 'patient_id', 'nodule_id']), y)
    evaluate_model(X, y, best_optuna, group_by_nodule=True, method_name="Optuna Search")
