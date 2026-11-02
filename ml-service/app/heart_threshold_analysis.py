"""
Heart Disease Threshold Analysis & Operating-Point Evaluation Module (Phase 6).

This module provides functions for evaluating classification performance across the
continuous probability spectrum [0.01, 0.99] without retraining models or modifying
the locked test set.

Strict Phase Boundaries:
- Uses existing Phase 5 Out-Of-Fold (OOF) predictions for threshold development.
- Uses existing Phase 5 hold-out test predictions strictly for final descriptive evaluation.
- No probability calibration.
- No SHAP explainability.
- No production model bundling or API exposure.
"""

import os
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd

# Canonical Model Identifiers matching Phase 5
MODEL_COLUMNS = {
    "LogisticRegression": {
        "name": "Logistic Regression (Robust L1)",
        "oof_col": "oof_prob_LogisticRegression (Robust L1)",
        "test_prob_col": "test_prob_LogisticRegression (Robust L1)",
        "test_pred_col": "test_pred_LogisticRegression (Robust L1)",
        "phase5_expected_test_cm": {"tn": 28, "fp": 5, "fn": 3, "tp": 25},
    },
    "RandomForest": {
        "name": "Random Forest (n=200, leaf=5)",
        "oof_col": "oof_prob_RandomForest (n=200, leaf=5)",
        "test_prob_col": "test_prob_RandomForest (n=200, leaf=5)",
        "test_pred_col": "test_pred_RandomForest (n=200, leaf=5)",
        "phase5_expected_test_cm": {"tn": 28, "fp": 5, "fn": 2, "tp": 26},
    },
    "XGBoost": {
        "name": "XGBoost (n=100, depth=2, lr=0.05)",
        "oof_col": "oof_prob_XGBoost (n=100, depth=2, lr=0.05)",
        "test_prob_col": "test_prob_XGBoost (n=100, depth=2, lr=0.05)",
        "test_pred_col": "test_pred_XGBoost (n=100, depth=2, lr=0.05)",
        "phase5_expected_test_cm": {"tn": 29, "fp": 4, "fn": 2, "tp": 26},
    },
}

DEFAULT_THRESHOLDS = np.linspace(0.01, 0.99, 99)


def validate_prediction_artifacts(
    oof_path: str,
    test_path: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Validates integrity and format of Phase 5 prediction artifacts:
    - OOF file: Exactly 242 rows, true labels in {0, 1}, continuous probs in [0, 1], no missing values.
    - Test file: Exactly 61 rows, true labels in {0, 1}, continuous probs in [0, 1], no missing values.
    """
    if not os.path.exists(oof_path):
        raise FileNotFoundError(f"OOF predictions artifact not found at {oof_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test predictions artifact not found at {test_path}")

    df_oof = pd.read_csv(oof_path)
    df_test = pd.read_csv(test_path)

    # 1. Row count verification
    if len(df_oof) != 242:
        raise ValueError(f"OOF predictions row count mismatch: got {len(df_oof)}, expected 242")
    if len(df_test) != 61:
        raise ValueError(f"Test predictions row count mismatch: got {len(df_test)}, expected 61")

    # 2. Duplicate index check
    if df_oof["sample_index"].duplicated().any():
        raise ValueError("Duplicate sample indices detected in OOF predictions")
    if df_test["sample_index"].duplicated().any():
        raise ValueError("Duplicate sample indices detected in Test predictions")

    # 3. Label verification
    oof_labels = set(df_oof["true_label"].unique())
    test_labels = set(df_test["true_label"].unique())
    if oof_labels != {0, 1}:
        raise ValueError(f"Unexpected OOF labels: {oof_labels}; expected {{0, 1}}")
    if test_labels != {0, 1}:
        raise ValueError(f"Unexpected Test labels: {test_labels}; expected {{0, 1}}")

    # Class balance check
    if int((df_oof["true_label"] == 0).sum()) != 131 or int((df_oof["true_label"] == 1).sum()) != 111:
        raise ValueError("OOF class distribution mismatch (expected 131 neg, 111 pos)")
    if int((df_test["true_label"] == 0).sum()) != 33 or int((df_test["true_label"] == 1).sum()) != 28:
        raise ValueError("Test class distribution mismatch (expected 33 neg, 28 pos)")

    # 4. Probability checks
    for key, info in MODEL_COLUMNS.items():
        oof_col = info["oof_col"]
        test_col = info["test_prob_col"]

        if oof_col not in df_oof.columns:
            raise KeyError(f"Missing column '{oof_col}' in OOF predictions")
        if test_col not in df_test.columns:
            raise KeyError(f"Missing column '{test_col}' in Test predictions")

        if df_oof[oof_col].isna().any():
            raise ValueError(f"NaN values detected in OOF column '{oof_col}'")
        if df_test[test_col].isna().any():
            raise ValueError(f"NaN values detected in Test column '{test_col}'")

        if (df_oof[oof_col] < 0.0).any() or (df_oof[oof_col] > 1.0).any():
            raise ValueError(f"Probabilities out of [0, 1] in OOF column '{oof_col}'")
        if (df_test[test_col] < 0.0).any() or (df_test[test_col] > 1.0).any():
            raise ValueError(f"Probabilities out of [0, 1] in Test column '{test_col}'")

    return df_oof, df_test


def calculate_threshold_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    """
    Computes confusion matrix counts and screening metrics at decision cutoff threshold t:
    - predicted_positive = y_prob >= threshold
    - predicted_negative = y_prob < threshold

    Handles zero denominators safely:
    - Precision: 1.0 if (tp + fp) == 0 else tp / (tp + fp)
    - NPV: 1.0 if (tn + fn) == 0 else tn / (tn + fn)
    - F1: 0.0 if (precision + recall) == 0 else 2*p*r / (p+r)
    """
    pred_pos = (y_prob >= threshold).astype(int)
    
    tp = int(np.sum((y_true == 1) & (pred_pos == 1)))
    tn = int(np.sum((y_true == 0) & (pred_pos == 0)))
    fp = int(np.sum((y_true == 0) & (pred_pos == 1)))
    fn = int(np.sum((y_true == 1) & (pred_pos == 0)))
    total = len(y_true)

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Safe precision & NPV handling
    prec = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if tp == 0 and fp == 0 else 0.0)
    npv = tn / (tn + fn) if (tn + fn) > 0 else (1.0 if tn == 0 and fn == 0 else 0.0)

    acc = (tp + tn) / total if total > 0 else 0.0
    f1 = (2.0 * prec * sens) / (prec + sens) if (prec + sens) > 0 else 0.0

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    youden_j = sens + spec - 1.0

    return {
        "threshold": round(float(threshold), 4),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "sensitivity": float(sens),
        "specificity": float(spec),
        "precision": float(prec),
        "npv": float(npv),
        "accuracy": float(acc),
        "f1": float(f1),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "youden_j": float(youden_j),
    }


def analyze_threshold_grid(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    cohort_type: str = "OOF",
    thresholds: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Evaluates threshold metrics across the 99-step grid [0.01, 0.99].
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    records = []
    for t in thresholds:
        m = calculate_threshold_metrics(y_true, y_prob, t)
        m["model_name"] = model_name
        m["cohort_type"] = cohort_type
        records.append(m)

    df_grid = pd.DataFrame(records)
    return df_grid


def find_high_sensitivity_regions(
    threshold_df: pd.DataFrame,
    min_sensitivity: float = 0.90,
) -> pd.DataFrame:
    """
    Identifies thresholds achieving Sensitivity >= min_sensitivity (e.g. 0.90).
    """
    sub = threshold_df[threshold_df["sensitivity"] >= min_sensitivity].copy()
    sub.sort_values(by="threshold", inplace=True)
    return sub


def find_balanced_operating_point(
    threshold_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Identifies the balanced candidate threshold maximizing Youden's J statistic
    (Sensitivity + Specificity - 1).
    """
    # Sort by Youden's J descending, break ties by accuracy then F1
    sorted_df = threshold_df.sort_values(by=["youden_j", "accuracy", "f1"], ascending=[False, False, False])
    return sorted_df.iloc[0].to_dict()


def summarize_operating_points(
    oof_grid_df: pd.DataFrame,
    test_grid_df: pd.DataFrame,
    model_name: str,
    high_sens_threshold: float,
    balanced_threshold: float,
) -> pd.DataFrame:
    """
    Summarizes performance for:
    1. High-Sensitivity operating point (derived from OOF data)
    2. Balanced operating point (derived from OOF data maximizing Youden's J)
    3. Default reference point (t = 0.50)

    Reports metrics on both OOF training data and hold-out test set.
    """
    operating_points = [
        ("High-Sensitivity", high_sens_threshold),
        ("Balanced (Youden J)", balanced_threshold),
        ("Default (Reference)", 0.50),
    ]

    records = []
    for label, t in operating_points:
        oof_row = oof_grid_df[np.isclose(oof_grid_df["threshold"], t, atol=1e-3)].iloc[0]
        test_row = test_grid_df[np.isclose(test_grid_df["threshold"], t, atol=1e-3)].iloc[0]

        records.append({
            "model_name": model_name,
            "operating_point": label,
            "threshold": round(t, 2),
            # OOF Metrics
            "oof_sensitivity": oof_row["sensitivity"],
            "oof_specificity": oof_row["specificity"],
            "oof_precision": oof_row["precision"],
            "oof_npv": oof_row["npv"],
            "oof_f1": oof_row["f1"],
            "oof_accuracy": oof_row["accuracy"],
            "oof_fn": int(oof_row["fn"]),
            "oof_fp": int(oof_row["fp"]),
            "oof_youden_j": oof_row["youden_j"],
            # Hold-out Test Metrics
            "test_sensitivity": test_row["sensitivity"],
            "test_specificity": test_row["specificity"],
            "test_precision": test_row["precision"],
            "test_npv": test_row["npv"],
            "test_f1": test_row["f1"],
            "test_accuracy": test_row["accuracy"],
            "test_fn": int(test_row["fn"]),
            "test_fp": int(test_row["fp"]),
            "test_youden_j": test_row["youden_j"],
        })

    return pd.DataFrame(records)


def verify_default_threshold_results(test_preds_df: pd.DataFrame) -> Dict[str, bool]:
    """
    Verifies that the default threshold 0.50 exactly reproduces Phase 5 test results:
    - Logistic Regression: TN=28, FP=5, FN=3, TP=25
    - Random Forest: TN=28, FP=5, FN=2, TP=26
    - XGBoost: TN=29, FP=4, FN=2, TP=26
    """
    results = {}
    y_true = test_preds_df["true_label"].values

    for key, info in MODEL_COLUMNS.items():
        name = info["name"]
        prob_col = info["test_prob_col"]
        expected = info["phase5_expected_test_cm"]

        probs = test_preds_df[prob_col].values
        m = calculate_threshold_metrics(y_true, probs, threshold=0.50)

        is_match = (
            m["tn"] == expected["tn"]
            and m["fp"] == expected["fp"]
            and m["fn"] == expected["fn"]
            and m["tp"] == expected["tp"]
        )
        results[name] = is_match
        if not is_match:
            raise AssertionError(
                f"Default 0.50 mismatch for {name}: got TN={m['tn']}, FP={m['fp']}, FN={m['fn']}, TP={m['tp']}; "
                f"expected TN={expected['tn']}, FP={expected['fp']}, FN={expected['fn']}, TP={expected['tp']}"
            )

    return results
