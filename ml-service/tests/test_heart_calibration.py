"""
Heart Disease Probability Calibration Unit & Regression Test Suite (Phase 7).

Verifies:
1. Calibration module imports successfully.
2. Expected model configurations are available.
3. Training/holdout separation is preserved.
4. No holdout labels are used during calibration fitting (leakage-free).
5. Sigmoid calibration executes successfully.
6. Isotonic calibration executes successfully.
7. Calibrated probabilities remain strictly within [0, 1].
8. Brier score calculation is valid.
9. Log loss calculation is valid.
10. ECE calculation is valid and matches mathematical bounds.
11. Reliability curve and binning data are properly generated.
12. Calibration artifacts are written correctly to artifacts/heart/calibration/.
13. Existing Phase 5 artifacts are untouched and not overwritten.
14. Existing Phase 6 artifacts are untouched and not overwritten.
15. Raw Cleveland dataset cryptographic integrity remains intact.
"""

import os
import json
import pytest
import numpy as np
import pandas as pd

from app.heart_preprocessing import (
    verify_raw_dataset_integrity,
    load_heart_raw_data,
    create_modeling_dataset,
    split_heart_data,
)
from app.heart_calibration import (
    get_heart_finalist_configs,
    compute_ece,
    compute_calibration_intercept_slope,
    calculate_calibration_metrics,
    run_training_calibration_cv,
    fit_calibrated_models_on_train,
    evaluate_calibrated_models_on_holdout,
    analyze_threshold_shifts,
)


@pytest.fixture(scope="module")
def data_splits():
    df_raw = load_heart_raw_data()
    X, y = create_modeling_dataset(df_raw)
    X_train, X_test, y_train, y_test = split_heart_data(X, y, test_size=0.20, random_state=42)
    return X_train, X_test, y_train, y_test


def test_1_calibration_module_imports_successfully():
    """Verify that all core functions in app.heart_calibration can be imported without errors."""
    import app.heart_calibration as hc
    assert hasattr(hc, "get_heart_finalist_configs")
    assert hasattr(hc, "compute_ece")
    assert hasattr(hc, "calculate_calibration_metrics")
    assert hasattr(hc, "run_training_calibration_cv")
    assert hasattr(hc, "fit_calibrated_models_on_train")
    assert hasattr(hc, "evaluate_calibrated_models_on_holdout")
    assert hasattr(hc, "analyze_threshold_shifts")


def test_2_expected_model_configurations_available():
    """Verify that the 3 finalist model configurations from Phase 5 are correctly defined."""
    configs = get_heart_finalist_configs()
    assert len(configs) == 3
    
    families = [c["family"] for c in configs]
    assert "LogisticRegression" in families
    assert "RandomForest" in families
    assert "XGBoost" in families
    
    lr_cfg = next(c for c in configs if c["family"] == "LogisticRegression")
    assert lr_cfg["config_id"] == "LR_robust_l1_C1.0"
    assert lr_cfg["scaler"] == "robust"
    assert lr_cfg["penalty"] == "l1"
    
    rf_cfg = next(c for c in configs if c["family"] == "RandomForest")
    assert rf_cfg["config_id"] == "RF_n200_dNone_leaf5_cwNone"
    assert rf_cfg["n_estimators"] == 200
    assert rf_cfg["min_samples_leaf"] == 5
    
    xgb_cfg = next(c for c in configs if c["family"] == "XGBoost")
    assert xgb_cfg["config_id"] == "XGB_n100_d2_lr0.05_sub0.8_col0.8"
    assert xgb_cfg["learning_rate"] == 0.05
    assert xgb_cfg["max_depth"] == 2


def test_3_training_holdout_separation_preserved(data_splits):
    """Verify training set is exactly 242 rows and holdout is exactly 61 rows with no overlap."""
    X_train, X_test, y_train, y_test = data_splits
    assert len(X_train) == 242
    assert len(X_test) == 61
    assert len(y_train) == 242
    assert len(y_test) == 61
    
    train_indices = set(X_train.index)
    test_indices = set(X_test.index)
    assert len(train_indices.intersection(test_indices)) == 0


def test_4_no_holdout_labels_used_during_calibration_fitting(data_splits):
    """Verify that fitting calibration pipelines accepts strictly X_train and y_train."""
    X_train, X_test, y_train, y_test = data_splits
    configs = get_heart_finalist_configs()
    
    # Fit calibrators on X_train only
    fitted = fit_calibrated_models_on_train(X_train, y_train, configs, cv=3)
    
    for cfg in configs:
        c_name = cfg["candidate_name"]
        assert "Uncalibrated" in fitted[c_name]
        assert "Sigmoid" in fitted[c_name]
        assert "Isotonic" in fitted[c_name]
        
        # Test models produce predictions on X_test without fitting on X_test
        probs = fitted[c_name]["Sigmoid"].predict_proba(X_test)[:, 1]
        assert len(probs) == len(X_test)


def test_5_sigmoid_calibration_executes_successfully(data_splits):
    """Verify Sigmoid / Platt calibration executes and fits properly."""
    X_train, X_test, y_train, y_test = data_splits
    configs = get_heart_finalist_configs()[:1]  # Test with Logistic Regression
    
    fitted = fit_calibrated_models_on_train(X_train, y_train, configs, cv=3)
    sig_model = fitted[configs[0]["candidate_name"]]["Sigmoid"]
    
    train_probs = sig_model.predict_proba(X_train)[:, 1]
    test_probs = sig_model.predict_proba(X_test)[:, 1]
    
    assert len(train_probs) == 242
    assert len(test_probs) == 61
    assert np.all(train_probs >= 0.0) and np.all(train_probs <= 1.0)
    assert np.all(test_probs >= 0.0) and np.all(test_probs <= 1.0)


def test_6_isotonic_calibration_executes_successfully(data_splits):
    """Verify Isotonic regression calibration executes and fits properly."""
    X_train, X_test, y_train, y_test = data_splits
    configs = get_heart_finalist_configs()[:1]
    
    fitted = fit_calibrated_models_on_train(X_train, y_train, configs, cv=3)
    iso_model = fitted[configs[0]["candidate_name"]]["Isotonic"]
    
    train_probs = iso_model.predict_proba(X_train)[:, 1]
    test_probs = iso_model.predict_proba(X_test)[:, 1]
    
    assert len(train_probs) == 242
    assert len(test_probs) == 61
    assert np.all(train_probs >= 0.0) and np.all(train_probs <= 1.0)
    assert np.all(test_probs >= 0.0) and np.all(test_probs <= 1.0)


def test_7_calibrated_probabilities_remain_within_unit_interval(data_splits):
    """Verify that all calibrated probabilities strictly adhere to [0, 1]."""
    X_train, X_test, y_train, y_test = data_splits
    configs = get_heart_finalist_configs()
    fitted = fit_calibrated_models_on_train(X_train, y_train, configs, cv=3)
    
    for cfg in configs:
        c_name = cfg["candidate_name"]
        for method in ["Uncalibrated", "Sigmoid", "Isotonic"]:
            model = fitted[c_name][method]
            probs = model.predict_proba(X_test)[:, 1]
            assert np.all(probs >= 0.0), f"{c_name} {method} has probs < 0"
            assert np.all(probs <= 1.0), f"{c_name} {method} has probs > 1"


def test_8_brier_score_calculation_valid():
    """Verify Brier score calculation matches theoretical bounds and values."""
    # Perfect predictions: Brier score = 0
    y_true = np.array([0, 1, 0, 1])
    y_prob_perfect = np.array([0.0, 1.0, 0.0, 1.0])
    m_perfect = calculate_calibration_metrics(y_true, y_prob_perfect)
    assert np.isclose(m_perfect["brier_score"], 0.0)
    
    # Completely wrong predictions: Brier score = 1
    y_prob_wrong = np.array([1.0, 0.0, 1.0, 0.0])
    m_wrong = calculate_calibration_metrics(y_true, y_prob_wrong)
    assert np.isclose(m_wrong["brier_score"], 1.0)
    
    # Uninformative baseline p=0.5: Brier score = 0.25
    y_prob_half = np.array([0.5, 0.5, 0.5, 0.5])
    m_half = calculate_calibration_metrics(y_true, y_prob_half)
    assert np.isclose(m_half["brier_score"], 0.25)


def test_9_log_loss_calculation_valid():
    """Verify Log loss calculation is strictly positive and finite."""
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.2, 0.8])
    m = calculate_calibration_metrics(y_true, y_prob)
    assert m["log_loss"] > 0.0
    assert np.isfinite(m["log_loss"])


def test_10_ece_calculation_valid():
    """Verify ECE calculation satisfies [0, 1] bounds and correctly handles binning."""
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.7, 0.3, 0.85, 0.15])
    
    ece, mce, bin_df = compute_ece(y_true, y_prob, n_bins=5, strategy="uniform")
    assert 0.0 <= ece <= 1.0
    assert 0.0 <= mce <= 1.0
    assert len(bin_df) == 5
    assert "mean_pred_prob" in bin_df.columns
    assert "observed_pos_freq" in bin_df.columns
    assert "sample_count" in bin_df.columns


def test_11_reliability_data_generated():
    """Verify calibration curve / bin data includes expected columns and matches sample total."""
    y_true = np.array([0, 1] * 50)
    y_prob = np.random.RandomState(42).uniform(0, 1, 100)
    ece, mce, bin_df = compute_ece(y_true, y_prob, n_bins=10, strategy="uniform")
    
    total_samples = bin_df["sample_count"].sum()
    assert total_samples == 100
    assert set(bin_df.columns) == {
        "bin_index", "bin_lower", "bin_upper", "sample_count",
        "mean_pred_prob", "observed_pos_freq", "calibration_gap", "weight"
    }


def test_12_calibration_artifacts_written_correctly():
    """Verify all required Phase 7 calibration artifacts exist and are populated."""
    artifacts_dir = os.path.join("artifacts", "heart", "calibration")
    eval_dir = os.path.join("evaluation", "heart")
    
    expected_files = [
        os.path.join(artifacts_dir, "calibration_results.csv"),
        os.path.join(artifacts_dir, "calibration_summary.csv"),
        os.path.join(artifacts_dir, "calibration_metrics.json"),
        os.path.join(artifacts_dir, "calibration_metadata.json"),
        os.path.join(artifacts_dir, "oof_calibrated_predictions.csv"),
        os.path.join(artifacts_dir, "test_calibrated_predictions.csv"),
        os.path.join(artifacts_dir, "threshold_shift_analysis.csv"),
        os.path.join(artifacts_dir, "calibration_bins_detail.csv"),
        os.path.join(artifacts_dir, "reliability_diagrams_train_cv.png"),
        os.path.join(artifacts_dir, "reliability_diagrams_holdout.png"),
        os.path.join(artifacts_dir, "probability_distribution_comparison.png"),
        os.path.join(eval_dir, "heart_calibration_comparison.csv"),
    ]
    
    for f in expected_files:
        assert os.path.exists(f), f"Expected artifact {f} is missing!"
        assert os.path.getsize(f) > 0, f"Artifact {f} is empty!"
        
    # Verify metadata JSON structure
    with open(os.path.join(artifacts_dir, "calibration_metadata.json"), "r") as fp:
        meta = json.load(fp)
    assert meta["disease"] == "Heart Disease (Cleveland)"
    assert meta["phase"] == "Phase 7: Probability Calibration"
    assert meta["n_train"] == 242
    assert meta["n_test"] == 61


def test_13_existing_phase5_artifacts_not_overwritten():
    """Verify that existing Phase 5 experiment artifacts remain untouched."""
    p5_dir = os.path.join("artifacts", "heart", "experiments")
    assert os.path.exists(p5_dir)
    
    oof_p5 = os.path.join(p5_dir, "oof_predictions.csv")
    test_p5 = os.path.join(p5_dir, "test_predictions.csv")
    cv_p5 = os.path.join(p5_dir, "cv_results_all.csv")
    
    assert os.path.exists(oof_p5)
    assert os.path.exists(test_p5)
    assert os.path.exists(cv_p5)
    
    df_oof = pd.read_csv(oof_p5)
    df_test = pd.read_csv(test_p5)
    assert len(df_oof) == 242
    assert len(df_test) == 61


def test_14_existing_phase6_artifacts_not_overwritten():
    """Verify that existing Phase 6 threshold artifacts remain untouched."""
    p6_dir = os.path.join("artifacts", "heart", "threshold")
    assert os.path.exists(p6_dir)
    
    summary_p6 = os.path.join(p6_dir, "threshold_summary.csv")
    metrics_oof_p6 = os.path.join(p6_dir, "threshold_metrics_oof.csv")
    metrics_test_p6 = os.path.join(p6_dir, "threshold_metrics_test.csv")
    
    assert os.path.exists(summary_p6)
    assert os.path.exists(metrics_oof_p6)
    assert os.path.exists(metrics_test_p6)
    
    df_p6_oof = pd.read_csv(metrics_oof_p6)
    assert len(df_p6_oof) == 297  # 3 models * 99 thresholds


def test_15_raw_dataset_cryptographic_integrity():
    """Verify that the raw processed.cleveland.data dataset remains bit-for-bit unchanged."""
    is_valid, md5_hash, sha_hash, file_size = verify_raw_dataset_integrity()
    assert is_valid
    assert md5_hash == "2d91a8ff69cfd9616aa47b59d6f843db"
    assert sha_hash == "a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8"
    assert file_size == 18461
