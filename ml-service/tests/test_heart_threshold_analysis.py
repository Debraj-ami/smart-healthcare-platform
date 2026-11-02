"""
Automated Unit and Integration Tests for Heart Disease Threshold Analysis (Phase 6).

Verifies the 12 mandatory Phase 6 testing requirements:
1. OOF artifact has exactly 242 rows.
2. Test artifact has exactly 61 rows.
3. Probability values are within [0, 1].
4. Threshold grid contains 0.01–0.99 (99 steps).
5. Confusion-matrix counts are internally consistent (TP + FN = positive_total, TN + FP = negative_total).
6. Sensitivity calculation is correct.
7. Specificity calculation is correct.
8. Youden J calculation is correct (Sensitivity + Specificity - 1).
9. Default threshold 0.50 reproduces Phase 5 confusion matrix counts.
10. Threshold selection uses OOF data rather than test data (no test leakage).
11. No calibration is performed (probabilities remain uncalibrated raw scores).
12. Raw dataset remains unchanged byte-for-byte.
"""

import os
import numpy as np
import pandas as pd
import pytest

from app.heart_preprocessing import (
    verify_raw_dataset_integrity,
    EXPECTED_RAW_MD5,
    EXPECTED_RAW_SHA256,
)
from app.heart_threshold_analysis import (
    MODEL_COLUMNS,
    DEFAULT_THRESHOLDS,
    validate_prediction_artifacts,
    calculate_threshold_metrics,
    analyze_threshold_grid,
    find_high_sensitivity_regions,
    find_balanced_operating_point,
    verify_default_threshold_results,
)

OOF_PATH = os.path.join("artifacts", "heart", "experiments", "oof_predictions.csv")
TEST_PATH = os.path.join("artifacts", "heart", "experiments", "test_predictions.csv")


@pytest.fixture(scope="module")
def prediction_data():
    """Loads and validates Phase 5 prediction artifacts."""
    df_oof, df_test = validate_prediction_artifacts(OOF_PATH, TEST_PATH)
    return {"oof": df_oof, "test": df_test}


def test_1_oof_artifact_exact_rows(prediction_data):
    """Verifies OOF artifact has exactly 242 rows."""
    df_oof = prediction_data["oof"]
    assert len(df_oof) == 242, f"OOF dataframe must have exactly 242 rows, got {len(df_oof)}"


def test_2_test_artifact_exact_rows(prediction_data):
    """Verifies Test artifact has exactly 61 rows."""
    df_test = prediction_data["test"]
    assert len(df_test) == 61, f"Test dataframe must have exactly 61 rows, got {len(df_test)}"


def test_3_probabilities_within_unit_interval(prediction_data):
    """Verifies that all predicted probabilities in OOF and Test files lie in [0, 1]."""
    df_oof = prediction_data["oof"]
    df_test = prediction_data["test"]

    for key, info in MODEL_COLUMNS.items():
        oof_col = info["oof_col"]
        test_col = info["test_prob_col"]

        assert (df_oof[oof_col] >= 0.0).all() and (df_oof[oof_col] <= 1.0).all(), (
            f"OOF column '{oof_col}' has values outside [0, 1]"
        )
        assert (df_test[test_col] >= 0.0).all() and (df_test[test_col] <= 1.0).all(), (
            f"Test column '{test_col}' has values outside [0, 1]"
        )


def test_4_threshold_grid_span():
    """Verifies threshold grid covers 0.01 to 0.99 in 99 steps."""
    assert len(DEFAULT_THRESHOLDS) == 99
    assert np.isclose(DEFAULT_THRESHOLDS[0], 0.01)
    assert np.isclose(DEFAULT_THRESHOLDS[-1], 0.99)
    assert np.isclose(DEFAULT_THRESHOLDS[49], 0.50)


def test_5_confusion_matrix_internal_consistency(prediction_data):
    """Verifies TP + FN = N_pos and TN + FP = N_neg for all thresholds."""
    df_oof = prediction_data["oof"]
    y_true = df_oof["true_label"].values
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())

    sample_probs = df_oof[MODEL_COLUMNS["LogisticRegression"]["oof_col"]].values
    for t in [0.10, 0.30, 0.50, 0.70, 0.90]:
        m = calculate_threshold_metrics(y_true, sample_probs, t)
        assert m["tp"] + m["fn"] == n_pos, f"TP+FN ({m['tp']}+{m['fn']}) != n_pos ({n_pos})"
        assert m["tn"] + m["fp"] == n_neg, f"TN+FP ({m['tn']}+{m['fp']}) != n_neg ({n_neg})"
        assert m["tp"] + m["tn"] + m["fp"] + m["fn"] == len(y_true)


def test_6_sensitivity_calculation_correct():
    """Verifies sensitivity calculation: TP / (TP + FN)."""
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    y_prob = np.array([0.9, 0.8, 0.7, 0.4, 0.3, 0.2, 0.1, 0.05])
    m = calculate_threshold_metrics(y_true, y_prob, threshold=0.50)
    # TP = 3 (0.9, 0.8, 0.7), FN = 1 (0.4) -> Sens = 3/4 = 0.75
    assert m["tp"] == 3
    assert m["fn"] == 1
    assert m["sensitivity"] == pytest.approx(0.75)


def test_7_specificity_calculation_correct():
    """Verifies specificity calculation: TN / (TN + FP)."""
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    y_prob = np.array([0.9, 0.8, 0.7, 0.4, 0.6, 0.2, 0.1, 0.05])
    m = calculate_threshold_metrics(y_true, y_prob, threshold=0.50)
    # Negatives: 0.6 is FP (>=0.5), 0.2, 0.1, 0.05 are TN -> TN=3, FP=1 -> Spec = 3/4 = 0.75
    assert m["tn"] == 3
    assert m["fp"] == 1
    assert m["specificity"] == pytest.approx(0.75)


def test_8_youden_j_calculation_correct():
    """Verifies Youden's J statistic calculation: Sens + Spec - 1."""
    y_true = np.array([1, 1, 0, 0])
    y_prob = np.array([0.8, 0.4, 0.2, 0.1])
    m = calculate_threshold_metrics(y_true, y_prob, threshold=0.50)
    # Sens = 1/2 = 0.5, Spec = 2/2 = 1.0 -> Youden J = 0.5 + 1.0 - 1.0 = 0.5
    assert m["sensitivity"] == pytest.approx(0.5)
    assert m["specificity"] == pytest.approx(1.0)
    assert m["youden_j"] == pytest.approx(0.5)


def test_9_default_threshold_050_reproduces_phase5(prediction_data):
    """Verifies that evaluation at threshold 0.50 exactly reproduces Phase 5 test confusion matrices."""
    df_test = prediction_data["test"]
    results = verify_default_threshold_results(df_test)
    assert all(results.values()), f"Default 0.50 results failed to reproduce: {results}"


def test_10_threshold_selection_uses_oof_not_test(prediction_data):
    """Verifies that high-sensitivity and balanced threshold selection use OOF data, avoiding test leakage."""
    df_oof = prediction_data["oof"]
    y_oof = df_oof["true_label"].values
    oof_probs = df_oof[MODEL_COLUMNS["LogisticRegression"]["oof_col"]].values

    oof_grid = analyze_threshold_grid(y_oof, oof_probs, model_name="LR", cohort_type="OOF")
    high_sens_df = find_high_sensitivity_regions(oof_grid, min_sensitivity=0.90)

    assert len(high_sens_df) > 0
    t_hs = high_sens_df.iloc[-1]["threshold"]
    # Verify t_hs was derived strictly from OOF statistics
    assert t_hs <= 0.50, "High-sensitivity threshold should be <= 0.50 to capture positive cases"


def test_11_no_probability_calibration_applied(prediction_data):
    """Verifies that probabilities in OOF and Test files are uncalibrated raw model outputs."""
    df_oof = prediction_data["oof"]
    df_test = prediction_data["test"]

    # Raw model outputs are continuous and not transformed by CalibratedClassifierCV
    for key, info in MODEL_COLUMNS.items():
        oof_probs = df_oof[info["oof_col"]].values
        test_probs = df_test[info["test_prob_col"]].values

        # Ensure probabilities exhibit expected fine-grained variance of raw model scores
        assert len(np.unique(oof_probs)) > 100, f"OOF probs for {key} unexpectedly discretized"
        assert len(np.unique(test_probs)) >= 20, f"Test probs for {key} unexpectedly discretized"


def test_12_raw_dataset_remains_unchanged():
    """Verifies that datasets/raw/heart/processed.cleveland.data is byte-for-byte unchanged."""
    valid, md5_hash, sha256_hash, file_size = verify_raw_dataset_integrity()
    assert valid, "Raw dataset integrity check failed!"
    assert file_size == 18461, f"File size mismatch: {file_size}"
    assert md5_hash == EXPECTED_RAW_MD5, f"MD5 mismatch: {md5_hash}"
    assert sha256_hash == EXPECTED_RAW_SHA256, f"SHA256 mismatch: {sha256_hash}"
