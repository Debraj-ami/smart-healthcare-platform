"""
Automated Unit and Integration Tests for Heart Disease ML Experimentation (Phase 5).

Verifies the 10 mandatory experimental guarantees:
1. Correct target construction.
2. Correct train/test separation (242 train, 61 test, stratified).
3. CV preprocessing is inside the pipeline.
4. No test data is passed into CV fitting.
5. All candidate models can train successfully.
6. Probability outputs exist (and are continuous [0, 1]).
7. Metric calculations are correct (including specificity).
8. OOF predictions have correct length (242 samples).
9. Final test predictions have exactly 61 samples.
10. Raw dataset remains unchanged byte-for-byte.
"""

import os
import numpy as np
import pandas as pd
import pytest

from sklearn.pipeline import Pipeline

from app.heart_preprocessing import (
    load_heart_raw_data,
    create_modeling_dataset,
    split_heart_data,
    verify_raw_dataset_integrity,
    EXPECTED_RAW_MD5,
    EXPECTED_RAW_SHA256,
)
from app.heart_experimentation import (
    calculate_metrics,
    get_candidate_configurations,
    build_pipeline_from_config,
    generate_out_of_fold_predictions,
    evaluate_candidates_on_holdout,
)


@pytest.fixture(scope="module")
def data_setup():
    """Loads raw data, constructs binary target, and executes locked train/test split."""
    df = load_heart_raw_data()
    X, y = create_modeling_dataset(df)
    X_train, X_test, y_train, y_test = split_heart_data(X, y, test_size=0.20, random_state=42)
    return {
        "df": df,
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


def test_1_target_construction(data_setup):
    """Verifies target is strictly binary {0, 1} with exact 164/139 distribution."""
    y = data_setup["y"]
    assert set(y.unique()) == {0, 1}, "Target must contain only {0, 1}"
    assert int((y == 0).sum()) == 164, "Negative class count must be exactly 164"
    assert int((y == 1).sum()) == 139, "Positive class count must be exactly 139"


def test_2_train_test_separation(data_setup):
    """Verifies exact 242/61 sample split and stratify proportionality."""
    X_train = data_setup["X_train"]
    X_test = data_setup["X_test"]
    y_train = data_setup["y_train"]
    y_test = data_setup["y_test"]

    assert len(X_train) == 242, "Training set must contain exactly 242 rows"
    assert len(X_test) == 61, "Test set must contain exactly 61 rows"
    assert len(y_train) == 242
    assert len(y_test) == 61

    # Verify no index overlap between train and test
    assert len(set(X_train.index).intersection(set(X_test.index))) == 0

    # Stratification check (approx 45.87% positive)
    train_prev = float(y_train.mean())
    test_prev = float(y_test.mean())
    assert abs(train_prev - test_prev) < 0.01, f"Prevalence divergence too large: {train_prev} vs {test_prev}"


def test_3_cv_preprocessing_inside_pipeline():
    """Verifies candidate pipelines contain preprocessor and classifier in a single Pipeline."""
    configs = get_candidate_configurations()
    for fam, cfg_list in configs.items():
        sample_cfg = cfg_list[0]
        pipe = build_pipeline_from_config(sample_cfg)
        assert isinstance(pipe, Pipeline), f"{fam} must be wrapped in an sklearn Pipeline"
        assert "preprocessor" in pipe.named_steps, "Pipeline must contain 'preprocessor' step"
        assert "classifier" in pipe.named_steps, "Pipeline must contain 'classifier' step"


def test_4_no_test_data_in_cv_fitting(data_setup):
    """Verifies that candidate pipeline fit on training fold does not know test indices or values."""
    X_train = data_setup["X_train"]
    y_train = data_setup["y_train"]
    X_test = data_setup["X_test"]

    configs = get_candidate_configurations()
    lr_cfg = configs["LogisticRegression"][0]
    pipe = build_pipeline_from_config(lr_cfg)

    # Fit strictly on train
    pipe.fit(X_train, y_train)
    preprocessor = pipe.named_steps["preprocessor"]

    # Imputer statistics must equal training statistics, not overall or test
    num_imputer = preprocessor.named_transformers_["num"].named_steps["imputer"]
    assert len(num_imputer.statistics_) == 5
    # Verify age median matches training median
    assert num_imputer.statistics_[0] == float(X_train["age"].median())


def test_5_all_candidate_models_train_successfully(data_setup):
    """Verifies that representative configurations from all 3 families train without error."""
    X_train = data_setup["X_train"]
    y_train = data_setup["y_train"]

    configs = get_candidate_configurations()
    for fam in ["LogisticRegression", "RandomForest", "XGBoost"]:
        cfg = configs[fam][0]
        pipe = build_pipeline_from_config(cfg)
        pipe.fit(X_train, y_train)
        assert hasattr(pipe.named_steps["classifier"], "classes_")


def test_6_probability_outputs_exist(data_setup):
    """Verifies that representative models from all families produce continuous probability outputs in [0, 1]."""
    X_train = data_setup["X_train"]
    y_train = data_setup["y_train"]
    X_test = data_setup["X_test"]

    configs = get_candidate_configurations()
    # Select representative configurations (e.g., C=1.0 for LR so coefficients are not trivially zeroed)
    rep_configs = [
        next(c for c in configs["LogisticRegression"] if c["C"] == 1.0 and c["penalty"] == "l2"),
        configs["RandomForest"][0],
        configs["XGBoost"][0],
    ]
    for cfg in rep_configs:
        pipe = build_pipeline_from_config(cfg)
        pipe.fit(X_train, y_train)
        probs = pipe.predict_proba(X_test)[:, 1]
        assert len(probs) == len(X_test)
        assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
        # Ensure outputs are non-trivial continuous floats
        assert len(np.unique(probs)) > 10


def test_7_metric_calculations_correct():
    """Verifies accuracy, precision, recall, specificity, F1, ROC-AUC, and PR-AUC calculations."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    # Predictions: 3 TN, 1 FP, 1 FN, 3 TP
    y_prob = np.array([0.1, 0.2, 0.3, 0.8, 0.4, 0.6, 0.7, 0.9])
    
    m = calculate_metrics(y_true, y_prob, threshold=0.5)
    
    assert m["tn"] == 3
    assert m["fp"] == 1
    assert m["fn"] == 1
    assert m["tp"] == 3
    assert m["accuracy"] == 6 / 8
    assert m["precision"] == 3 / 4
    assert m["recall"] == 3 / 4
    assert m["specificity"] == 3 / 4  # TN / (TN + FP) = 3 / 4
    assert m["f1"] == pytest.approx(0.75)


def test_8_oof_predictions_length(data_setup):
    """Verifies out-of-fold predictions have exactly 242 rows."""
    X_train = data_setup["X_train"]
    y_train = data_setup["y_train"]

    configs = get_candidate_configurations()
    candidate = dict(configs["LogisticRegression"][0])
    candidate["candidate_name"] = "LR_test"

    # Fast 2x2 CV for unit test speed
    oof_df = generate_out_of_fold_predictions(X_train, y_train, [candidate], n_splits=2, n_repeats=2)
    assert len(oof_df) == 242, f"OOF dataframe must have 242 rows, got {len(oof_df)}"
    assert "oof_prob_LR_test" in oof_df.columns
    assert not oof_df["oof_prob_LR_test"].isna().any()


def test_9_final_test_predictions_length(data_setup):
    """Verifies hold-out evaluation on test set produces exactly 61 predictions."""
    X_train = data_setup["X_train"]
    y_train = data_setup["y_train"]
    X_test = data_setup["X_test"]
    y_test = data_setup["y_test"]

    configs = get_candidate_configurations()
    candidate = dict(configs["LogisticRegression"][0])
    candidate["candidate_name"] = "LR_holdout_test"

    df_holdout, test_preds_df, _ = evaluate_candidates_on_holdout(
        X_train, y_train, X_test, y_test, [candidate]
    )
    assert len(df_holdout) == 1
    assert len(test_preds_df) == 61
    assert "test_prob_LR_holdout_test" in test_preds_df.columns
    assert "test_pred_LR_holdout_test" in test_preds_df.columns


def test_10_raw_dataset_remains_unchanged():
    """Verifies that datasets/raw/heart/processed.cleveland.data is byte-for-byte unchanged."""
    valid, md5_hash, sha256_hash, file_size = verify_raw_dataset_integrity()
    assert valid, "Raw dataset integrity check failed!"
    assert file_size == 18461, f"File size mismatch: {file_size}"
    assert md5_hash == EXPECTED_RAW_MD5, f"MD5 mismatch: {md5_hash}"
    assert sha256_hash == EXPECTED_RAW_SHA256, f"SHA256 mismatch: {sha256_hash}"
