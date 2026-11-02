"""
Train and export production inference artifact for Heart Disease (Phase 8 Productionization).

Candidate Model: Logistic Regression (Robust L1, C=1.0)
Trained strictly on X_train (242 samples) from Cleveland Heart Disease dataset.
X_test (61 samples) is used strictly to verify experimental reproduction and additivity.

Bundle Contents:
- Fitted preprocessing pipeline (RobustScaler + OneHotEncoder drop='first')
- Fitted Logistic Regression classifier (penalty='l1', C=1.0, solver='liblinear')
- SHAP LinearExplainer fitted on training background distribution
- Configured model classification threshold (t=0.31 from Phase 6 threshold analysis)
- Clinical feature mapping and human-readable metadata
- Expected input validation schema
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss
import shap

from app.heart_preprocessing import (
    verify_raw_dataset_integrity,
    load_heart_raw_data,
    create_modeling_dataset,
    split_heart_data,
    build_logistic_regression_preprocessor,
    get_heart_feature_names,
    FEATURE_COLUMNS,
)
from app.heart_shap import HEART_FEATURE_DESCRIPTIONS, get_heart_feature_mapping

RANDOM_STATE = 42
CONFIGURED_THRESHOLD = 0.31  # Phase 6 High-Sensitivity Operating Point


def train_and_export():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, ".."))
    data_path = os.path.join(project_root, "datasets", "raw", "heart", "processed.cleveland.data")

    print("=" * 70)
    print("HEART DISEASE — PRODUCTION ARTIFACT TRAINING & EXPORT")
    print("=" * 70)

    # 1. Verify Dataset Integrity
    print(f"\n[Step 1] Verifying raw dataset from: {data_path}")
    is_valid, md5_hash, sha_hash, file_sz = verify_raw_dataset_integrity(data_path)
    if not is_valid:
        raise ValueError(f"Raw dataset cryptographic mismatch! Aborting. MD5={md5_hash}")
    print(f"Dataset verified: MD5={md5_hash}, SHA256={sha_hash}, Size={file_sz} bytes.")

    # 2. Load & Split Data
    print("\n[Step 2] Loading data and performing stratified 80/20 train/test split...")
    df_raw = load_heart_raw_data(data_path)
    X, y = create_modeling_dataset(df_raw)
    X_train, X_test, y_train, y_test = split_heart_data(X, y, test_size=0.20, random_state=RANDOM_STATE)
    assert len(X_train) == 242 and len(X_test) == 61
    print(f"Training set: {len(X_train)} samples ({y_train.sum()} positive, {(y_train == 0).sum()} negative)")
    print(f"Holdout set:  {len(X_test)} samples ({y_test.sum()} positive, {(y_test == 0).sum()} negative)")

    # 3. Fit Preprocessing Pipeline strictly on X_train
    print("\n[Step 3] Fitting RobustScaler + OneHotEncoder preprocessor on X_train...")
    preprocessor = build_logistic_regression_preprocessor(scaler="robust", drop="first")
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)
    transformed_feature_names = get_heart_feature_names(preprocessor)
    print(f"Transformed features ({len(transformed_feature_names)}): {transformed_feature_names}")

    # 4. Fit Finalist Logistic Regression Model strictly on X_train
    print("\n[Step 4] Fitting Logistic Regression (penalty='l1', C=1.0, solver='liblinear') on X_train...")
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(
        penalty="l1",
        C=1.0,
        solver="liblinear",
        random_state=RANDOM_STATE,
        max_iter=1000,
    )
    clf.fit(X_train_trans, y_train)

    # 5. Verify Experimental Holdout Reproduction (Phases 5 & 7 benchmarks)
    print("\n[Step 5] Verifying reproduction on locked 61-sample holdout test set...")
    test_probs = clf.predict_proba(X_test_trans)[:, 1]
    brier = brier_score_loss(y_test, test_probs)
    ll = log_loss(y_test, test_probs)
    roc = roc_auc_score(y_test, test_probs)
    pr = average_precision_score(y_test, test_probs)

    print(f"Holdout Metrics Reproduction:")
    print(f"  Brier Score: {brier:.4f} (expected Phase 7: 0.0886)")
    print(f"  Log Loss:    {ll:.4f} (expected Phase 7: 0.3019)")
    print(f"  ROC-AUC:     {roc:.4f} (expected Phase 5: 0.9513)")
    print(f"  PR-AUC:      {pr:.4f} (expected Phase 5: 0.9274)")

    assert np.isclose(brier, 0.0886, atol=1e-3), f"Brier mismatch: {brier}"
    assert np.isclose(ll, 0.3019, atol=1e-3), f"Log Loss mismatch: {ll}"
    assert np.isclose(roc, 0.9513, atol=1e-3), f"ROC-AUC mismatch: {roc}"
    assert np.isclose(pr, 0.9274, atol=1e-3), f"PR-AUC mismatch: {pr}"
    print("Exact reproduction verified against Phase 5 and Phase 7 benchmarks.")

    # 6. Fit SHAP LinearExplainer on Training Background
    print("\n[Step 6] Initializing SHAP LinearExplainer with independent training background...")
    lr_masker = shap.maskers.Independent(X_train_trans, max_samples=len(X_train_trans))
    lr_explainer = shap.LinearExplainer(clf, lr_masker)

    # Verify Additivity Axiom
    exp_test = lr_explainer(X_test_trans)
    reconstructed = exp_test.base_values + exp_test.values.sum(axis=1)
    actual_outputs = clf.decision_function(X_test_trans)
    discrepancy = np.max(np.abs(reconstructed - actual_outputs))
    print(f"SHAP Additivity Verification: Max Discrepancy = {discrepancy:.2e}")
    assert discrepancy < 1e-10, "SHAP additivity verification failed!"

    # 7. Construct Feature Mapping
    mapping_df = get_heart_feature_mapping("LogisticRegression", preprocessor=preprocessor)

    # 8. Package Production Bundle
    artifacts_dir = os.path.join(base_dir, "artifacts", "heart")
    os.makedirs(artifacts_dir, exist_ok=True)

    artifact_bundle = {
        "model_name": "Logistic Regression (Robust L1)",
        "model_version": "1.0.0",
        "disease": "HEART_DISEASE",
        "preprocessor": preprocessor,
        "classifier": clf,
        "shap_explainer": lr_explainer,
        "threshold": CONFIGURED_THRESHOLD,
        "threshold_description": (
            "Configured model classification threshold (t=0.31) targeting >= 90% sensitivity "
            "established in Phase 6 threshold analysis."
        ),
        "calibration_method": (
            "Uncalibrated Baseline (showed comparatively good empirical calibration on this dataset: "
            "ECE=0.0383, Slope=1.0704; post-hoc calibration did not demonstrate consistent improvement)."
        ),
        "output_space": "log-odds",
        "feature_names_raw": FEATURE_COLUMNS,
        "feature_names_transformed": transformed_feature_names,
        "feature_descriptions": HEART_FEATURE_DESCRIPTIONS,
        "expected_input_schema": {
            "age": "float/int (1-120)",
            "sex": "float/int (1.0=Male, 0.0=Female)",
            "cp": "float/int (1.0, 2.0, 3.0, 4.0)",
            "trestbps": "float/int (50-250 mm Hg)",
            "chol": "float/int (50-600 mg/dl)",
            "fbs": "float/int (0.0 or 1.0)",
            "restecg": "float/int (0.0, 1.0, 2.0)",
            "thalach": "float/int (50-250 bpm)",
            "exang": "float/int (0.0 or 1.0)",
            "oldpeak": "float (0.0-10.0)",
            "slope": "float/int (1.0, 2.0, 3.0)",
            "ca": "float/int (0.0, 1.0, 2.0, 3.0)",
            "thal": "float/int (3.0, 6.0, 7.0)",
        },
    }

    model_path = os.path.join(artifacts_dir, "heart_model_bundle.joblib")
    joblib.dump(artifact_bundle, model_path)
    print(f"\n[Step 7] Saved Heart model bundle to: {model_path}")
    print(f"Bundle size: {os.path.getsize(model_path):,} bytes")

    # 9. Save Metadata JSON
    metadata = {
        "disease": "HEART_DISEASE",
        "model_name": "Logistic Regression (Robust L1)",
        "model_version": "1.0.0",
        "config_id": "LR_robust_l1_C1.0",
        "training_samples": len(X_train),
        "holdout_samples": len(X_test),
        "random_state": RANDOM_STATE,
        "configured_threshold": CONFIGURED_THRESHOLD,
        "threshold_policy": "High-Sensitivity Screening (target >= 90% sensitivity on training OOF)",
        "calibration_status": (
            "Comparatively good empirical calibration on this dataset (Brier=0.1188, ECE=0.0383, Slope=1.0704); "
            "post-hoc calibration did not demonstrate consistent improvement."
        ),
        "output_space": "log-odds",
        "experimental_holdout_metrics": {
            "brier_score": round(float(brier), 4),
            "log_loss": round(float(ll), 4),
            "roc_auc": round(float(roc), 4),
            "pr_auc": round(float(pr), 4),
        },
        "transformed_features": transformed_feature_names,
        "dataset_integrity": {
            "md5": md5_hash,
            "sha256": sha_hash,
            "file_size": file_sz,
        },
        "disclaimer": (
            "Research screening decision support tool. Model outputs are statistical probability estimates "
            "and do not constitute clinical diagnosis or medical proof of coronary artery disease. "
            "Feature attributions explain model behavior and do not establish biological causation."
        ),
    }

    metadata_path = os.path.join(artifacts_dir, "production_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved production metadata to: {metadata_path}")

    print("\n" + "=" * 70)
    print("HEART PRODUCTION ARTIFACT CREATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    train_and_export()
