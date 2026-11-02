"""
Train and export production inference artifact for Liver Disease.

Candidate Model: Logistic Regression with 5-fold Isotonic Calibration
Trained strictly on X_train (456 samples) from Indian Liver Patient Dataset (ILPD).
X_test (114 samples) is used strictly to verify experimental reproduction.
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import shap

RANDOM_STATE = 42

def train_and_export():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, '..'))
    data_path = os.path.join(project_root, 'datasets', 'raw', 'liver', 'Indian Liver Patient Dataset (ILPD).csv')

    print(f"Loading raw dataset from: {data_path}")
    column_names = [
        'Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin', 'Alkphos',
        'Sgpt', 'Sgot', 'Total_Proteins', 'Albumin', 'A_G_Ratio', 'Selector'
    ]
    df_raw = pd.read_csv(data_path, header=None, names=column_names)

    df_processed = df_raw.copy()
    df_processed['target'] = df_processed['Selector'].map({1: 1, 2: 0})
    df_dedup = df_processed.drop_duplicates(subset=column_names).copy()

    feature_cols = [
        'Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin', 'Alkphos',
        'Sgpt', 'Sgot', 'Total_Proteins', 'Albumin', 'A_G_Ratio'
    ]
    X = df_dedup[feature_cols].copy()
    y = df_dedup['target'].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE
    )

    num_features = [
        'Age', 'Total_Bilirubin', 'Direct_Bilirubin', 'Alkphos',
        'Sgpt', 'Sgot', 'Total_Proteins', 'Albumin', 'A_G_Ratio'
    ]
    cat_features = ['Gender']

    preprocessor_scaled = ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='if_binary'), cat_features)
    ])

    base_pipeline = Pipeline([
        ('preprocessor', preprocessor_scaled),
        ('classifier', LogisticRegression(random_state=RANDOM_STATE, max_iter=1000))
    ])

    print("Fitting CalibratedClassifierCV (method='isotonic', cv=5) on X_train...")
    cal_iso = CalibratedClassifierCV(estimator=base_pipeline, method='isotonic', cv=5)
    cal_iso.fit(X_train, y_train)

    probs_iso = cal_iso.predict_proba(X_test)[:, 1]
    brier = brier_score_loss(y_test, probs_iso)
    ll = log_loss(y_test, probs_iso)
    roc = roc_auc_score(y_test, probs_iso)
    pr = average_precision_score(y_test, probs_iso)

    print(f"Verified Holdout Metrics:")
    print(f"  Brier Score: {brier:.4f} (expected: 0.1576)")
    print(f"  Log Loss:    {ll:.4f} (expected: 0.4472)")
    print(f"  ROC-AUC:     {roc:.4f} (expected: 0.8122)")
    print(f"  PR-AUC:      {pr:.4f} (expected: 0.9273)")

    assert np.isclose(brier, 0.1576, atol=1e-3), f"Brier mismatch: {brier}"
    assert np.isclose(ll, 0.4472, atol=1e-3), f"Log Loss mismatch: {ll}"
    assert np.isclose(roc, 0.8122, atol=1e-3), f"ROC-AUC mismatch: {roc}"
    assert np.isclose(pr, 0.9273, atol=1e-3), f"PR-AUC mismatch: {pr}"

    # Also fit base pipeline for standalone explainer & feature transformation
    print("Fitting base pipeline on X_train for SHAP explainer...")
    base_pipeline.fit(X_train, y_train)
    fitted_preprocessor = base_pipeline.named_steps['preprocessor']
    fitted_lr = base_pipeline.named_steps['classifier']

    X_train_scaled = fitted_preprocessor.transform(X_train)
    transformed_feature_names = fitted_preprocessor.get_feature_names_out().tolist()

    print("Creating SHAP LinearExplainer on X_train background...")
    lr_masker = shap.maskers.Independent(X_train_scaled, max_samples=len(X_train_scaled))
    lr_explainer = shap.LinearExplainer(fitted_lr, lr_masker)

    # Save artifacts
    artifacts_dir = os.path.join(base_dir, 'artifacts', 'liver')
    os.makedirs(artifacts_dir, exist_ok=True)

    artifact_bundle = {
        'model_name': 'Logistic Regression (Isotonic Calibrated)',
        'model_version': '1.0.0',
        'calibrated_model': cal_iso,
        'base_pipeline': base_pipeline,
        'preprocessor': fitted_preprocessor,
        'lr_classifier': fitted_lr,
        'shap_explainer': lr_explainer,
        'feature_names_raw': feature_cols,
        'feature_names_transformed': transformed_feature_names,
        'expected_input_schema': {
            'Age': 'int (1-120)',
            'Gender': 'string ("Male" | "Female")',
            'Total_Bilirubin': 'float (>0)',
            'Direct_Bilirubin': 'float (>=0)',
            'Alkphos': 'int (>0)',
            'Sgpt': 'int (>0)',
            'Sgot': 'int (>0)',
            'Total_Proteins': 'float (>0)',
            'Albumin': 'float (>0)',
            'A_G_Ratio': 'float (>0)'
        }
    }

    model_path = os.path.join(artifacts_dir, 'liver_model_bundle.joblib')
    joblib.dump(artifact_bundle, model_path)
    print(f"Saved model bundle to: {model_path}")

    metadata = {
        'disease': 'LIVER_DISEASE',
        'model_name': 'Logistic Regression + Isotonic Calibration',
        'model_version': '1.0.0',
        'training_samples': int(len(X_train)),
        'holdout_samples': int(len(X_test)),
        'random_state': RANDOM_STATE,
        'experimental_holdout_metrics': {
            'brier_score': round(float(brier), 4),
            'log_loss': round(float(ll), 4),
            'roc_auc': round(float(roc), 4),
            'pr_auc': round(float(pr), 4)
        },
        'transformed_features': transformed_feature_names,
        'disclaimer': (
            'Research screening tool. Not clinically validated for diagnostic use. '
            'Feature contributions explain model behavior and do not establish biological causation.'
        )
    }

    metadata_path = os.path.join(artifacts_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to: {metadata_path}")

if __name__ == '__main__':
    train_and_export()
