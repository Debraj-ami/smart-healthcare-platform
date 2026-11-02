"""
Heart Disease Preprocessing & Dataset Preparation Pipeline (Phase 4).

This module provides a reproducible, leakage-safe data loading, validation,
and ColumnTransformer construction framework for Heart Disease risk modeling.

Constraints:
- Treats datasets/raw/heart/processed.cleveland.data as strictly read-only.
- Does NOT train any ML models.
- Implements strict fit-on-train discipline.
"""

import os
import hashlib
from typing import Tuple, List, Optional, Dict, Any, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder

# Raw dataset default relative path from project root
DEFAULT_RAW_HEART_PATH = os.path.join("datasets", "raw", "heart", "processed.cleveland.data")

# Expected raw checksums
EXPECTED_RAW_MD5 = "2d91a8ff69cfd9616aa47b59d6f843db"
EXPECTED_RAW_SHA256 = "a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8"
EXPECTED_ROW_COUNT = 303
EXPECTED_COL_COUNT = 14

# Raw 14-column Cleveland schema
RAW_COLUMNS: List[str] = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
]

# The 13 predictive clinical features
FEATURE_COLUMNS: List[str] = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

# Feature groups established in Phase 3
NUMERICAL_FEATURES: List[str] = ["age", "trestbps", "chol", "thalach", "oldpeak"]
BINARY_FEATURES: List[str] = ["sex", "fbs", "exang"]
CATEGORICAL_FEATURES: List[str] = ["cp", "restecg", "slope", "thal"]
COUNT_FEATURES: List[str] = ["ca"]

# Valid discrete domains for categorical and binary features
VALID_DOMAINS: Dict[str, set] = {
    "sex": {0.0, 1.0},
    "cp": {1.0, 2.0, 3.0, 4.0},
    "fbs": {0.0, 1.0},
    "restecg": {0.0, 1.0, 2.0},
    "exang": {0.0, 1.0},
    "slope": {1.0, 2.0, 3.0},
    "ca": {0.0, 1.0, 2.0, 3.0},
    "thal": {3.0, 6.0, 7.0},
    "num": {0, 1, 2, 3, 4},
}


def verify_raw_dataset_integrity(file_path: Optional[str] = None) -> Tuple[bool, str, str, int]:
    """
    Computes cryptographic checksums of the raw Cleveland dataset and verifies
    that the file is unchanged.
    """
    path = file_path or DEFAULT_RAW_HEART_PATH
    if not os.path.exists(path):
        # Also try locating from workspace root or ml-service root
        alt_path = os.path.join("..", path)
        if os.path.exists(alt_path):
            path = alt_path
        else:
            raise FileNotFoundError(f"Raw Heart dataset not found at {path} or {alt_path}")

    with open(path, "rb") as f:
        bytes_data = f.read()

    file_size = len(bytes_data)
    md5_hash = hashlib.md5(bytes_data).hexdigest()
    sha256_hash = hashlib.sha256(bytes_data).hexdigest()

    is_valid = (
        file_size == 18461
        and md5_hash == EXPECTED_RAW_MD5
        and sha256_hash == EXPECTED_RAW_SHA256
    )
    return is_valid, md5_hash, sha256_hash, file_size


def load_heart_raw_data(file_path: Optional[str] = None) -> pd.DataFrame:
    """
    Loads and normalizes the raw Cleveland Heart Disease dataset.

    Steps:
    1. Reads raw file with header=None and assigns the 14 standard column names.
    2. Validates exact dimensions (303 rows, 14 columns).
    3. Replaces sentinel '?' characters with np.nan.
    4. Converts ca and thal to float64.
    5. Validates expected domains and absence of malformed categories.
    6. Ensures raw source file remains completely untouched.
    """
    path = file_path or DEFAULT_RAW_HEART_PATH
    if not os.path.exists(path):
        alt_path = os.path.join("..", path)
        if os.path.exists(alt_path):
            path = alt_path
        else:
            raise FileNotFoundError(f"Raw Heart dataset not found at {path}")

    # Read raw comma-delimited data
    df = pd.read_csv(path, header=None, names=RAW_COLUMNS, dtype=str)

    if df.shape != (EXPECTED_ROW_COUNT, EXPECTED_COL_COUNT):
        raise ValueError(
            f"Unexpected dataset dimensions {df.shape}; expected ({EXPECTED_ROW_COUNT}, {EXPECTED_COL_COUNT})"
        )

    # Convert '?' sentinels to NaN
    df.replace("?", np.nan, inplace=True)

    # Cast numerical and categorical columns to float64
    for col in FEATURE_COLUMNS:
        df[col] = df[col].astype(float)

    # Cast num to int64
    df["num"] = df["num"].astype(int)

    # Validate categories against expected domain sets
    for col, valid_set in VALID_DOMAINS.items():
        observed_non_null = set(df[col].dropna().unique())
        invalid_values = observed_non_null - valid_set
        if invalid_values:
            raise ValueError(f"Column '{col}' contains unexpected values: {invalid_values}")

    # Validate physiological continuous domains
    assert (df["age"] > 0).all(), "Invalid negative/zero age detected"
    assert (df["trestbps"] > 0).all(), "Invalid negative/zero resting BP detected"
    assert (df["chol"] > 0).all(), "Invalid negative/zero cholesterol detected"
    assert (df["thalach"] > 0).all(), "Invalid negative/zero max heart rate detected"
    assert (df["oldpeak"] >= 0).all(), "Invalid negative ST depression detected"

    return df


def create_modeling_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separates predictors X from binary screening target y.

    Binary Mapping:
    - num == 0 -> y = 0 (Negative / Absence of CAD, N=164)
    - num >= 1 -> y = 1 (Positive / Presence of CAD, N=139)

    Returns:
    - X: DataFrame containing only the 13 predictors.
    - y: Series containing binary classification target {0, 1}.
    """
    if "num" not in df.columns:
        raise ValueError("DataFrame must contain raw target column 'num'")

    # Construct binary target
    y = (df["num"] >= 1).astype(int)
    y.name = "target"

    # Verify verified Phase 0 class balance
    neg_count = int((y == 0).sum())
    pos_count = int((y == 1).sum())
    if neg_count != 164 or pos_count != 139:
        raise ValueError(
            f"Class distribution mismatch: observed ({neg_count} neg, {pos_count} pos), expected (164 neg, 139 pos)"
        )

    # Extract exactly the 13 feature columns
    X = df[FEATURE_COLUMNS].copy()

    return X, y


def split_heart_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Performs stratified train/test split.

    Default: 80% train (242 samples), 20% test (61 samples), random_state=42.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    return X_train, X_test, y_train, y_test


def build_logistic_regression_preprocessor(
    scaler: str = "standard",
    drop: Optional[str] = "first",
) -> ColumnTransformer:
    """
    Constructs a ColumnTransformer suitable for Logistic Regression:
    - Numerical features: Median imputation + Scaling (StandardScaler or RobustScaler).
    - Binary features: Most-frequent imputation + Passthrough.
    - Categorical features: Most-frequent imputation + OneHotEncoder(drop='first' by default).
    - Count feature (ca): Median imputation + Scaling.
    """
    if scaler == "standard":
        scaler_transform = StandardScaler()
    elif scaler == "robust":
        scaler_transform = RobustScaler()
    else:
        raise ValueError(f"Unsupported scaler '{scaler}'. Choose 'standard' or 'robust'.")

    # Numerical pipeline: impute with median, then scale
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", scaler_transform),
    ])

    # Binary pipeline: impute with mode, passthrough as {0, 1}
    bin_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])

    # Categorical pipeline: impute with mode, then one-hot encode
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop=drop, sparse_output=False)),
    ])

    # Count pipeline: median impute + scale
    count_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", scaler_transform),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERICAL_FEATURES),
            ("bin", bin_pipeline, BINARY_FEATURES),
            ("cat", cat_pipeline, CATEGORICAL_FEATURES),
            ("count", count_pipeline, COUNT_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    return preprocessor


def build_tree_preprocessor(
    drop: Optional[str] = None,
) -> ColumnTransformer:
    """
    Constructs a ColumnTransformer suitable for Tree-based models (Random Forest, XGBoost):
    - Numerical features: Median imputation only (NO SCALING).
    - Binary features: Most-frequent imputation + Passthrough.
    - Categorical features: Most-frequent imputation + OneHotEncoder (drop=None by default).
    - Count feature (ca): Median imputation only (NO SCALING).
    """
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    bin_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop=drop, sparse_output=False)),
    ])

    count_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERICAL_FEATURES),
            ("bin", bin_pipeline, BINARY_FEATURES),
            ("cat", cat_pipeline, CATEGORICAL_FEATURES),
            ("count", count_pipeline, COUNT_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    return preprocessor


def build_heart_preprocessor(
    model_type: str = "logistic",
    scaler: str = "standard",
    drop: Optional[str] = "first",
) -> ColumnTransformer:
    """
    Unified factory for building leakage-safe ColumnTransformers.

    Args:
        model_type: 'logistic' or 'tree'
        scaler: 'standard' or 'robust' (applicable when model_type='logistic')
        drop: 'first' or None for OneHotEncoder
    """
    model_type_clean = model_type.strip().lower()
    if model_type_clean in ("logistic", "linear", "lr"):
        return build_logistic_regression_preprocessor(scaler=scaler, drop=drop)
    elif model_type_clean in ("tree", "rf", "random_forest", "xgboost", "xgb"):
        return build_tree_preprocessor(drop=drop)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose 'logistic' or 'tree'.")


def get_heart_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """
    Extracts output feature names from a fitted ColumnTransformer.
    """
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception as e:
        raise RuntimeError(f"Cannot get feature names. Preprocessor must be fitted first: {e}")
