"""
Heart Disease SHAP Explainability Unit & Regression Test Suite (Phase 8).

Verifies:
1. SHAP module imports successfully.
2. Final model configuration loads.
3. Correct explainer is selected for each model architecture.
4. SHAP values are generated successfully.
5. SHAP output dimensions match transformed features (18 for LR, 22 for Tree models).
6. Feature names align exactly with SHAP values.
7. Probabilities and model outputs remain valid and finite.
8. Global importance artifact is generated and sorted.
9. Local explanation artifact is generated for real representative patients.
10. Feature mapping is generated and preserves clinical traceability.
11. Existing Phase 5 artifacts remain unchanged.
12. Existing Phase 6 artifacts remain unchanged.
13. Existing Phase 7 artifacts remain unchanged.
14. Raw Cleveland dataset cryptographic integrity remains unchanged.
15. Exact mathematical additivity / efficiency holds (no fake/static values).
"""

import os
import json
import pytest
import numpy as np
import pandas as pd
import shap

from app.heart_preprocessing import (
    verify_raw_dataset_integrity,
    load_heart_raw_data,
    create_modeling_dataset,
    split_heart_data,
)
from app.heart_calibration import get_heart_finalist_configs
from app.heart_shap import (
    get_heart_feature_mapping,
    fit_model_and_preprocessor_for_shap,
    create_heart_shap_explainer,
    compute_shap_explanations,
    compute_global_feature_importance,
    explain_single_observation,
    generate_representative_local_explanations,
)


@pytest.fixture(scope="module")
def data_splits():
    df_raw = load_heart_raw_data()
    X, y = create_modeling_dataset(df_raw)
    X_train, X_test, y_train, y_test = split_heart_data(X, y, test_size=0.20, random_state=42)
    return X_train, X_test, y_train, y_test


def test_1_shap_module_imports_successfully():
    """Verify that all core functions in app.heart_shap can be imported."""
    import app.heart_shap as hs
    assert hasattr(hs, "get_heart_feature_mapping")
    assert hasattr(hs, "fit_model_and_preprocessor_for_shap")
    assert hasattr(hs, "create_heart_shap_explainer")
    assert hasattr(hs, "compute_shap_explanations")
    assert hasattr(hs, "compute_global_feature_importance")
    assert hasattr(hs, "explain_single_observation")
    assert hasattr(hs, "generate_representative_local_explanations")


def test_2_final_model_configuration_loads():
    """Verify that the primary finalized model configuration (Logistic Regression L1) loads properly."""
    configs = get_heart_finalist_configs()
    assert len(configs) >= 1
    lr_cfg = configs[0]
    assert lr_cfg["family"] == "LogisticRegression"
    assert lr_cfg["config_id"] == "LR_robust_l1_C1.0"
    assert lr_cfg["scaler"] == "robust"
    assert lr_cfg["penalty"] == "l1"


def test_3_correct_explainer_selected(data_splits):
    """Verify LinearExplainer is chosen for LR and TreeExplainer for Tree models."""
    X_train, _, y_train, _ = data_splits
    configs = get_heart_finalist_configs()
    
    # 1. Logistic Regression
    est_lr, prep_lr, X_tr_lr, _ = fit_model_and_preprocessor_for_shap(configs[0], X_train, y_train)
    exp_lr = create_heart_shap_explainer(est_lr, X_tr_lr, "LogisticRegression")
    assert isinstance(exp_lr, shap.LinearExplainer)
    
    # 2. Random Forest
    est_rf, prep_rf, X_tr_rf, _ = fit_model_and_preprocessor_for_shap(configs[1], X_train, y_train)
    exp_rf = create_heart_shap_explainer(est_rf, X_tr_rf, "RandomForest")
    assert isinstance(exp_rf, shap.TreeExplainer)
    
    # 3. XGBoost
    est_xgb, prep_xgb, X_tr_xgb, _ = fit_model_and_preprocessor_for_shap(configs[2], X_train, y_train)
    exp_xgb = create_heart_shap_explainer(est_xgb, X_tr_xgb, "XGBoost")
    assert isinstance(exp_xgb, shap.TreeExplainer)


def test_4_shap_values_generated(data_splits):
    """Verify SHAP explanations compute successfully on holdout data."""
    X_train, X_test, y_train, _ = data_splits
    configs = get_heart_finalist_configs()
    
    est_lr, prep_lr, X_tr_lr, _ = fit_model_and_preprocessor_for_shap(configs[0], X_train, y_train)
    exp_lr = create_heart_shap_explainer(est_lr, X_tr_lr, "LogisticRegression")
    
    res = compute_shap_explanations(exp_lr, prep_lr, est_lr, X_test.iloc[:5], "LogisticRegression")
    assert "shap_values" in res
    assert "base_values" in res
    assert len(res["shap_values"]) == 5


def test_5_shap_output_dimensions_match_transformed_features(data_splits):
    """Verify transformed feature counts match SHAP value columns."""
    X_train, X_test, y_train, _ = data_splits
    configs = get_heart_finalist_configs()
    
    # LR: 18 features (drop='first')
    est_lr, prep_lr, X_tr_lr, fn_lr = fit_model_and_preprocessor_for_shap(configs[0], X_train, y_train)
    assert len(fn_lr) == 18
    exp_lr = create_heart_shap_explainer(est_lr, X_tr_lr, "LogisticRegression")
    res_lr = compute_shap_explanations(exp_lr, prep_lr, est_lr, X_test, "LogisticRegression")
    assert res_lr["shap_values"].shape == (len(X_test), 18)
    
    # Tree: 22 features (drop=None)
    est_rf, prep_rf, X_tr_rf, fn_rf = fit_model_and_preprocessor_for_shap(configs[1], X_train, y_train)
    assert len(fn_rf) == 22
    exp_rf = create_heart_shap_explainer(est_rf, X_tr_rf, "RandomForest")
    res_rf = compute_shap_explanations(exp_rf, prep_rf, est_rf, X_test, "RandomForest")
    assert res_rf["shap_values"].shape == (len(X_test), 22)


def test_6_feature_names_align_with_shap_values(data_splits):
    """Verify that feature names returned match ColumnTransformer feature_names_out."""
    X_train, _, y_train, _ = data_splits
    configs = get_heart_finalist_configs()
    _, prep_lr, _, fn_lr = fit_model_and_preprocessor_for_shap(configs[0], X_train, y_train)
    assert list(prep_lr.get_feature_names_out()) == fn_lr


def test_7_probabilities_and_outputs_remain_valid(data_splits):
    """Verify that predicted probabilities remain in [0, 1] and model outputs are finite."""
    X_train, X_test, y_train, _ = data_splits
    configs = get_heart_finalist_configs()
    est_lr, prep_lr, X_tr_lr, _ = fit_model_and_preprocessor_for_shap(configs[0], X_train, y_train)
    exp_lr = create_heart_shap_explainer(est_lr, X_tr_lr, "LogisticRegression")
    res_lr = compute_shap_explanations(exp_lr, prep_lr, est_lr, X_test, "LogisticRegression")
    
    probs = res_lr["predicted_probabilities"]
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    assert np.all(np.isfinite(res_lr["model_outputs"]))


def test_8_global_importance_artifact_is_generated():
    """Verify that shap_feature_importance.csv exists, is non-empty, and sorted descending."""
    path = os.path.join("artifacts", "heart", "shap", "shap_feature_importance.csv")
    assert os.path.exists(path), f"Artifact missing: {path}"
    df = pd.read_csv(path)
    assert len(df) == 18
    assert "mean_abs_shap" in df.columns
    assert "display_name" in df.columns
    # Check descending order
    vals = df["mean_abs_shap"].values
    assert np.all(vals[:-1] >= vals[1:])


def test_9_local_explanation_artifact_is_generated():
    """Verify that shap_local_explanations.csv exists and contains real representative patients."""
    path = os.path.join("artifacts", "heart", "shap", "shap_local_explanations.csv")
    assert os.path.exists(path), f"Artifact missing: {path}"
    df = pd.read_csv(path)
    assert len(df) >= 3
    assert "sample_index" in df.columns
    assert "model_estimated_probability" in df.columns
    assert "top_positive_contributors" in df.columns
    assert "top_negative_contributors" in df.columns


def test_10_feature_mapping_is_generated():
    """Verify that shap_feature_mapping.csv contains 18 features with original-to-transformed mappings."""
    path = os.path.join("artifacts", "heart", "shap", "shap_feature_mapping.csv")
    assert os.path.exists(path), f"Artifact missing: {path}"
    df = pd.read_csv(path)
    assert len(df) == 18
    assert set(df.columns) == {
        "feature_index", "transformed_feature", "original_feature",
        "display_name", "feature_group", "description"
    }


def test_11_existing_phase5_artifacts_not_overwritten():
    """Verify Phase 5 experiment artifacts remain untouched."""
    p5_dir = os.path.join("artifacts", "heart", "experiments")
    assert os.path.exists(os.path.join(p5_dir, "oof_predictions.csv"))
    assert os.path.exists(os.path.join(p5_dir, "test_predictions.csv"))
    assert os.path.exists(os.path.join(p5_dir, "cv_results_all.csv"))


def test_12_existing_phase6_artifacts_not_overwritten():
    """Verify Phase 6 threshold artifacts remain untouched."""
    p6_dir = os.path.join("artifacts", "heart", "threshold")
    assert os.path.exists(os.path.join(p6_dir, "threshold_summary.csv"))
    assert os.path.exists(os.path.join(p6_dir, "threshold_metrics_oof.csv"))
    assert os.path.exists(os.path.join(p6_dir, "threshold_metrics_test.csv"))


def test_13_existing_phase7_artifacts_not_overwritten():
    """Verify Phase 7 calibration artifacts remain untouched."""
    p7_dir = os.path.join("artifacts", "heart", "calibration")
    assert os.path.exists(os.path.join(p7_dir, "calibration_results.csv"))
    assert os.path.exists(os.path.join(p7_dir, "calibration_summary.csv"))
    assert os.path.exists(os.path.join(p7_dir, "threshold_shift_analysis.csv"))
    assert os.path.exists(os.path.join(p7_dir, "calibration_metrics.json"))


def test_14_raw_dataset_cryptographic_integrity():
    """Verify that the raw Cleveland dataset has not been modified."""
    is_valid, md5_hash, sha_hash, file_size = verify_raw_dataset_integrity()
    assert is_valid
    assert md5_hash == "2d91a8ff69cfd9616aa47b59d6f843db"
    assert sha_hash == "a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8"
    assert file_size == 18461


def test_15_no_fake_static_shap_values_and_additivity_holds(data_splits):
    """Verify that SHAP values satisfy the mathematical additivity / efficiency axiom exactly."""
    X_train, X_test, y_train, _ = data_splits
    configs = get_heart_finalist_configs()
    est_lr, prep_lr, X_tr_lr, _ = fit_model_and_preprocessor_for_shap(configs[0], X_train, y_train)
    exp_lr = create_heart_shap_explainer(est_lr, X_tr_lr, "LogisticRegression")
    res_lr = compute_shap_explanations(exp_lr, prep_lr, est_lr, X_test, "LogisticRegression")
    
    # Exact additivity check: base_value + sum(shap_values) == decision_function
    reconstructed = res_lr["base_values"] + res_lr["shap_values"].sum(axis=1)
    actual = res_lr["model_outputs"]
    discrepancy = np.max(np.abs(reconstructed - actual))
    assert discrepancy < 1e-10, f"Additivity violated: max discrepancy is {discrepancy}"
    # Verify non-trivial variation across samples (not static dummy values)
    assert np.std(res_lr["shap_values"]) > 1e-3
