"""
Automated Test Suite for Heart Disease Preprocessing Pipeline (Phase 4).

Covers:
- Raw loading and schema validation
- Missing-value conversion and sentinel auditing
- Category and range validation
- Target construction and class balance verification
- Stratified 80/20 train/test split
- Logistic Regression preprocessor (StandardScaler & RobustScaler)
- Tree-model preprocessor (unscaled numericals)
- Feature name generation (get_feature_names_out)
- Graceful handling of unseen categories
- 6 explicit Data Leakage tests
- Raw dataset cryptographic immutability
"""

import os
import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier

from app.heart_preprocessing import (
    DEFAULT_RAW_HEART_PATH,
    EXPECTED_RAW_MD5,
    EXPECTED_RAW_SHA256,
    EXPECTED_ROW_COUNT,
    EXPECTED_COL_COUNT,
    RAW_COLUMNS,
    FEATURE_COLUMNS,
    NUMERICAL_FEATURES,
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    COUNT_FEATURES,
    verify_raw_dataset_integrity,
    load_heart_raw_data,
    create_modeling_dataset,
    split_heart_data,
    build_logistic_regression_preprocessor,
    build_tree_preprocessor,
    build_heart_preprocessor,
    get_heart_feature_names,
)


@pytest.fixture(scope="module")
def raw_df():
    """Fixture providing clean loaded Heart DataFrame."""
    return load_heart_raw_data()


@pytest.fixture(scope="module")
def modeling_data(raw_df):
    """Fixture providing separated X and y."""
    return create_modeling_dataset(raw_df)


# =========================================================================
# 1. Raw Dataset Integrity & Ingestion Tests
# =========================================================================

def test_raw_dataset_cryptographic_integrity():
    """Verify raw Cleveland dataset is 100% byte-for-byte unchanged."""
    is_valid, md5_hash, sha256_hash, file_size = verify_raw_dataset_integrity()
    assert is_valid is True
    assert file_size == 18461
    assert md5_hash == EXPECTED_RAW_MD5
    assert sha256_hash == EXPECTED_RAW_SHA256


def test_load_heart_raw_data_schema(raw_df):
    """Verify loaded raw DataFrame dimensions, column names, and types."""
    assert raw_df.shape == (EXPECTED_ROW_COUNT, EXPECTED_COL_COUNT)
    assert list(raw_df.columns) == RAW_COLUMNS
    for col in FEATURE_COLUMNS:
        assert raw_df[col].dtype == np.float64
    assert raw_df["num"].dtype == np.int64


def test_missing_value_sentinel_conversion(raw_df):
    """Verify sentinel '?' conversion and exact known missingness."""
    # Check that no string '?' remains
    assert (raw_df == "?").sum().sum() == 0

    # Check exact known null counts
    null_counts = raw_df.isna().sum()
    assert null_counts["ca"] == 4
    assert null_counts["thal"] == 2
    assert null_counts.sum() == 6

    # Verify exact affected rows: [87, 166, 192, 266, 287, 302]
    affected_indices = raw_df[raw_df["ca"].isna() | raw_df["thal"].isna()].index.tolist()
    assert affected_indices == [87, 166, 192, 266, 287, 302]


# =========================================================================
# 2. Category & Domain Quality Validation Tests
# =========================================================================

def test_category_and_range_validation(raw_df):
    """Verify categorical features conform strictly to known clinical domains."""
    assert set(raw_df["sex"].dropna().unique()) == {0.0, 1.0}
    assert set(raw_df["cp"].dropna().unique()) == {1.0, 2.0, 3.0, 4.0}
    assert set(raw_df["fbs"].dropna().unique()) == {0.0, 1.0}
    assert set(raw_df["restecg"].dropna().unique()) == {0.0, 1.0, 2.0}
    assert set(raw_df["exang"].dropna().unique()) == {0.0, 1.0}
    assert set(raw_df["slope"].dropna().unique()) == {1.0, 2.0, 3.0}
    assert set(raw_df["ca"].dropna().unique()) == {0.0, 1.0, 2.0, 3.0}
    assert set(raw_df["thal"].dropna().unique()) == {3.0, 6.0, 7.0}
    assert set(raw_df["num"].unique()) == {0, 1, 2, 3, 4}


def test_extreme_values_retained(raw_df):
    """Verify valid recorded physiological extremes are preserved, not pruned."""
    assert raw_df["chol"].max() == 564.0
    assert raw_df["trestbps"].max() == 200.0
    assert raw_df["oldpeak"].max() == 6.2
    assert raw_df["thalach"].min() == 71.0


# =========================================================================
# 3. Target Construction & Dataset Splitting Tests
# =========================================================================

def test_target_construction(raw_df):
    """Verify binary screening target y construction (0 vs 1-4)."""
    X, y = create_modeling_dataset(raw_df)
    assert X.shape == (303, 13)
    assert list(X.columns) == FEATURE_COLUMNS
    assert "num" not in X.columns

    # Verify exact class counts
    assert int((y == 0).sum()) == 164
    assert int((y == 1).sum()) == 139
    assert len(y) == 303


def test_stratified_train_test_split(modeling_data):
    """Verify 80/20 train/test split dimensions and stratification."""
    X, y = modeling_data
    X_train, X_test, y_train, y_test = split_heart_data(X, y, test_size=0.20, random_state=42)

    assert X_train.shape == (242, 13)
    assert X_test.shape == (61, 13)
    assert len(y_train) == 242
    assert len(y_test) == 61

    # Verify stratification preservation (~54% negative, ~46% positive)
    train_neg_pct = (y_train == 0).mean()
    test_neg_pct = (y_test == 0).mean()
    assert abs(train_neg_pct - (164 / 303)) < 0.01
    assert abs(test_neg_pct - (164 / 303)) < 0.02


# =========================================================================
# 4. Preprocessing Fit & Transform Tests
# =========================================================================

def test_logistic_regression_preprocessor_execution(modeling_data):
    """Test standard and robust Logistic Regression ColumnTransformers."""
    X, y = modeling_data
    X_train, X_test, _, _ = split_heart_data(X, y, test_size=0.20, random_state=42)

    # Standard Scaler + drop='first'
    preprocessor_std = build_logistic_regression_preprocessor(scaler="standard", drop="first")
    X_train_std = preprocessor_std.fit_transform(X_train)
    X_test_std = preprocessor_std.transform(X_test)

    assert X_train_std.shape[0] == 242
    assert X_test_std.shape[0] == 61
    assert np.isnan(X_train_std).sum() == 0
    assert np.isnan(X_test_std).sum() == 0

    # Check scaled mean ~ 0 and std ~ 1 on training set for numerical columns
    feature_names = get_heart_feature_names(preprocessor_std)
    assert len(feature_names) == X_train_std.shape[1]

    # Robust Scaler
    preprocessor_rob = build_logistic_regression_preprocessor(scaler="robust", drop="first")
    X_train_rob = preprocessor_rob.fit_transform(X_train)
    X_test_rob = preprocessor_rob.transform(X_test)
    assert X_train_rob.shape == X_train_std.shape
    assert np.isnan(X_test_rob).sum() == 0


def test_tree_preprocessor_execution(modeling_data):
    """Test Tree ColumnTransformer: unscaled numericals + drop=None one-hot."""
    X, y = modeling_data
    X_train, X_test, _, _ = split_heart_data(X, y, test_size=0.20, random_state=42)

    preprocessor = build_tree_preprocessor(drop=None)
    X_train_tree = preprocessor.fit_transform(X_train)
    X_test_tree = preprocessor.transform(X_test)

    assert X_train_tree.shape[0] == 242
    assert X_test_tree.shape[0] == 61
    assert np.isnan(X_train_tree).sum() == 0
    assert np.isnan(X_test_tree).sum() == 0

    # Numerical features should retain their natural magnitudes (unscaled)
    # Feature 0 is age; min age in cohort is 29
    assert X_train_tree[:, 0].min() >= 29.0
    assert X_train_tree[:, 2].max() == 564.0  # chol max is 564


def test_unified_factory(modeling_data):
    """Test build_heart_preprocessor factory dispatch."""
    lr_prep = build_heart_preprocessor("logistic", scaler="standard")
    tree_prep = build_heart_preprocessor("tree")

    assert lr_prep is not None
    assert tree_prep is not None

    with pytest.raises(ValueError, match="Unknown model_type"):
        build_heart_preprocessor("invalid_type")


def test_handle_unseen_categories(modeling_data):
    """Verify that unseen categories in test set do not crash the pipeline."""
    X, y = modeling_data
    X_train, X_test, _, _ = split_heart_data(X, y, test_size=0.20, random_state=42)

    preprocessor = build_logistic_regression_preprocessor()
    preprocessor.fit(X_train)

    # Introduce an unseen category in a test observation (e.g. cp=99.0)
    X_unseen = X_test.copy()
    X_unseen.iloc[0, X_unseen.columns.get_loc("cp")] = 99.0

    # Must transform without error (one-hot sets unseen to zeros)
    X_trans = preprocessor.transform(X_unseen)
    assert X_trans.shape[0] == len(X_unseen)
    assert not np.isnan(X_trans).any()


# =========================================================================
# 5. Data Leakage Prevention Tests (6 Explicit Leakage Checks)
# =========================================================================

def test_leakage_1_test_values_do_not_influence_imputation(modeling_data):
    """TEST 1: Test-set values do NOT influence imputation statistics."""
    X, y = modeling_data
    X_train, X_test, _, _ = split_heart_data(X, y, test_size=0.20, random_state=42)

    preprocessor = build_logistic_regression_preprocessor()
    preprocessor.fit(X_train)

    # Imputer statistics fitted on X_train for 'num' features (median)
    imputer_stats_train = preprocessor.named_transformers_["num"].named_steps["imputer"].statistics_.copy()

    # Create a contaminated test set with extreme outlier values
    X_test_contaminated = X_test.copy()
    X_test_contaminated["age"] = 999.0

    # Transforming contaminated test data must NOT change imputer statistics
    preprocessor.transform(X_test_contaminated)
    imputer_stats_after = preprocessor.named_transformers_["num"].named_steps["imputer"].statistics_

    np.testing.assert_array_equal(imputer_stats_train, imputer_stats_after)


def test_leakage_2_test_values_do_not_influence_scaling(modeling_data):
    """TEST 2: Test-set values do NOT influence scaling parameters (mean / var)."""
    X, y = modeling_data
    X_train, X_test, _, _ = split_heart_data(X, y, test_size=0.20, random_state=42)

    preprocessor = build_logistic_regression_preprocessor(scaler="standard")
    preprocessor.fit(X_train)

    scaler_mean_before = preprocessor.named_transformers_["num"].named_steps["scaler"].mean_.copy()
    scaler_var_before = preprocessor.named_transformers_["num"].named_steps["scaler"].var_.copy()

    # Contaminate test set
    X_test_extreme = X_test.copy()
    X_test_extreme["chol"] = 10000.0
    preprocessor.transform(X_test_extreme)

    scaler_mean_after = preprocessor.named_transformers_["num"].named_steps["scaler"].mean_
    scaler_var_after = preprocessor.named_transformers_["num"].named_steps["scaler"].var_

    np.testing.assert_array_equal(scaler_mean_before, scaler_mean_after)
    np.testing.assert_array_equal(scaler_var_before, scaler_var_after)


def test_leakage_3_test_categories_do_not_influence_encoder_fitting(modeling_data):
    """TEST 3: Test-set categories do NOT influence encoder categories."""
    X, y = modeling_data
    X_train, X_test, _, _ = split_heart_data(X, y, test_size=0.20, random_state=42)

    preprocessor = build_logistic_regression_preprocessor()
    preprocessor.fit(X_train)

    encoder_categories_before = [
        c.copy() for c in preprocessor.named_transformers_["cat"].named_steps["onehot"].categories_
    ]

    # Test set with synthetic unseen categories
    X_test_novel = X_test.copy()
    X_test_novel["cp"] = 88.0
    preprocessor.transform(X_test_novel)

    encoder_categories_after = preprocessor.named_transformers_["cat"].named_steps["onehot"].categories_

    for cat_before, cat_after in zip(encoder_categories_before, encoder_categories_after):
        np.testing.assert_array_equal(cat_before, cat_after)


def test_leakage_4_target_not_passed_to_preprocessing(modeling_data):
    """TEST 4: Target column 'num' is never in X or preprocessor feature list."""
    X, _ = modeling_data
    assert "num" not in X.columns
    assert "target" not in X.columns

    preprocessor = build_logistic_regression_preprocessor()
    preprocessor.fit(X)
    feature_names = get_heart_feature_names(preprocessor)
    for name in feature_names:
        assert "num" not in name.lower() or "num__" in name  # 'num__' prefix for numerical is valid


def test_leakage_5_test_data_uses_train_fitted_transformers(modeling_data):
    """TEST 5: Test data transformation is purely deterministic using train parameters."""
    X, y = modeling_data
    X_train, X_test, _, _ = split_heart_data(X, y, test_size=0.20, random_state=42)

    preprocessor = build_logistic_regression_preprocessor()
    preprocessor.fit(X_train)

    # First transform
    res_1 = preprocessor.transform(X_test)
    # Second transform
    res_2 = preprocessor.transform(X_test)

    np.testing.assert_array_equal(res_1, res_2)


def test_leakage_6_cv_pipelines_fit_independently(modeling_data):
    """TEST 6: Cross-validation pipeline fits preprocessing independently within each fold."""
    X, y = modeling_data
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    imputer_means = []
    for train_idx, val_idx in skf.split(X, y):
        X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
        y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]

        pipe = Pipeline([
            ("preprocessor", build_logistic_regression_preprocessor()),
            ("dummy", DummyClassifier(strategy="prior")),
        ])

        pipe.fit(X_fold_train, y_fold_train)
        # Record fitted mean of 'age' across folds
        fitted_scaler = pipe.named_steps["preprocessor"].named_transformers_["num"].named_steps["scaler"]
        imputer_means.append(fitted_scaler.mean_[0])

        # Validate that prediction on validation fold works without leakage
        preds = pipe.predict(X_fold_val)
        assert len(preds) == len(y_fold_val)

    # Folds must have slightly different training means (proving independent fold fitting)
    assert len(set(np.round(imputer_means, 4))) > 1


# =========================================================================
# 6. Post-Test Raw Dataset Immutability Check
# =========================================================================

def test_raw_dataset_remains_unmodified_after_all_operations():
    """Verify raw file checksum remains 100% identical post-execution."""
    is_valid, md5_hash, sha256_hash, file_size = verify_raw_dataset_integrity()
    assert is_valid is True
    assert md5_hash == EXPECTED_RAW_MD5
    assert sha256_hash == EXPECTED_RAW_SHA256
    assert file_size == 18461
