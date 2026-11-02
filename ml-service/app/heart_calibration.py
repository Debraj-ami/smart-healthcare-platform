"""
Heart Disease Probability Calibration Module (Phase 7).

Provides reproducible, leakage-free empirical calibration evaluation for the 
three Cleveland Heart Disease finalist architectures from Phase 5:
1. Logistic Regression (Robust L1)
2. Random Forest (n=200, leaf=5)
3. XGBoost (n=100, depth=2, lr=0.05)

Methodologies Evaluated:
A. Uncalibrated / Raw Model Probability
B. Sigmoid / Platt Calibration (parametric logistic transformation)
C. Isotonic Calibration (non-parametric monotonic step transformation)

Key Principles:
- Strict training/holdout separation: The 61-sample test set is NEVER used for fitting calibrators,
  selecting hyperparameters, or tuning thresholds.
- Cross-validation on training cohort (N=242) for calibration development and comparison.
- Explicit documentation of isotonic step-boundary overfitting risks given N=242.
- Quantitative evaluation using Brier Score, Log Loss, Expected Calibration Error (ECE),
  and Calibration Intercept/Slope.
- Shift analysis between Phase 6 decision thresholds and calibrated probability scales.
- No clinical efficacy or diagnostic certainty claims.
"""

import os
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from app.heart_preprocessing import (
    build_logistic_regression_preprocessor,
    build_tree_preprocessor,
)
from app.heart_experimentation import (
    build_pipeline_from_config,
    get_candidate_configurations,
)


def get_heart_finalist_configs() -> List[Dict[str, Any]]:
    """
    Returns the exact configurations of the three Phase 5 finalist architectures.
    """
    all_configs = get_candidate_configurations()
    
    # 1. Logistic Regression: Robust L1, C=1.0
    lr_cfg = next(c for c in all_configs["LogisticRegression"] if c["config_id"] == "LR_robust_l1_C1.0")
    lr_cfg = dict(lr_cfg)
    lr_cfg["candidate_name"] = "Logistic Regression (Robust L1)"
    lr_cfg["short_name"] = "Logistic Regression"
    
    # 2. Random Forest: n=200, max_depth=None, min_samples_leaf=5
    rf_cfg = next(c for c in all_configs["RandomForest"] if c["config_id"] == "RF_n200_dNone_leaf5_cwNone")
    rf_cfg = dict(rf_cfg)
    rf_cfg["candidate_name"] = "Random Forest (n=200, leaf=5)"
    rf_cfg["short_name"] = "Random Forest"
    
    # 3. XGBoost: n=100, depth=2, lr=0.05, subsample=0.8, colsample_bytree=0.8
    xgb_cfg = next(c for c in all_configs["XGBoost"] if c["config_id"] == "XGB_n100_d2_lr0.05_sub0.8_col0.8")
    xgb_cfg = dict(xgb_cfg)
    xgb_cfg["candidate_name"] = "XGBoost (n=100, depth=2, lr=0.05)"
    xgb_cfg["short_name"] = "XGBoost"
    
    return [lr_cfg, rf_cfg, xgb_cfg]


def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> Tuple[float, float, pd.DataFrame]:
    """
    Calculates Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
    
    Bins predictions into n_bins intervals, calculating:
      ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
      MCE = max_{m=1}^M |acc(B_m) - conf(B_m)|
      
    Returns:
      (ece, mce, bin_table_dataframe)
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    N = len(y_true)
    
    if strategy == "uniform":
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy == "quantile":
        # Quantile binning ensures roughly equal sample counts per bin
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        bin_edges = np.percentile(y_prob, quantiles * 100.0)
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 2:
            bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
        
    bin_records = []
    total_weighted_gap = 0.0
    max_gap = 0.0
    
    for i in range(len(bin_edges) - 1):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]
        
        # Include left edge; include right edge for the last bin
        if i == len(bin_edges) - 2:
            mask = (y_prob >= lower) & (y_prob <= upper)
        else:
            mask = (y_prob >= lower) & (y_prob < upper)
            
        count = int(np.sum(mask))
        if count > 0:
            conf = float(np.mean(y_prob[mask]))
            acc = float(np.mean(y_true[mask]))
            gap = float(abs(acc - conf))
            weight = count / N
            total_weighted_gap += weight * gap
            if gap > max_gap:
                max_gap = gap
        else:
            conf = float((lower + upper) / 2.0)
            acc = 0.0
            gap = 0.0
            weight = 0.0
            
        bin_records.append({
            "bin_index": i + 1,
            "bin_lower": float(lower),
            "bin_upper": float(upper),
            "sample_count": count,
            "mean_pred_prob": round(conf, 4),
            "observed_pos_freq": round(acc, 4),
            "calibration_gap": round(gap, 4),
            "weight": round(weight, 4),
        })
        
    bin_df = pd.DataFrame(bin_records)
    return float(total_weighted_gap), float(max_gap), bin_df


def compute_calibration_intercept_slope(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    eps: float = 1e-5,
) -> Tuple[float, float]:
    """
    Computes calibration intercept (calibration-in-the-large) and calibration slope
    by regressing binary outcomes on logit-transformed predicted probabilities:
      logit(P(Y=1)) = alpha + beta * logit(y_prob)
      
    Ideal values:
      intercept alpha = 0 (overall probability level matches incidence)
      slope beta = 1 (probability spread is appropriately dispersed)
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.clip(np.asarray(y_prob).astype(float), eps, 1.0 - eps)
    
    logit_p = np.log(y_prob / (1.0 - y_prob)).reshape(-1, 1)
    
    # Fit unpenalized or weakly penalized logistic regression
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    try:
        lr.fit(logit_p, y_true)
        intercept = float(lr.intercept_[0])
        slope = float(lr.coef_[0][0])
    except Exception:
        # Fallback with weak L2 penalty if unpenalized fails
        lr = LogisticRegression(C=1e4, solver="lbfgs", max_iter=1000)
        lr.fit(logit_p, y_true)
        intercept = float(lr.intercept_[0])
        slope = float(lr.coef_[0][0])
        
    return intercept, slope


def calculate_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Computes a comprehensive dictionary of probability calibration and discrimination metrics.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.clip(np.asarray(y_prob).astype(float), 1e-15, 1.0 - 1e-15)
    
    brier = float(brier_score_loss(y_true, y_prob))
    ll = float(log_loss(y_true, y_prob))
    roc_auc = float(roc_auc_score(y_true, y_prob))
    pr_auc = float(average_precision_score(y_true, y_prob))
    
    ece, mce, _ = compute_ece(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    intercept, slope = compute_calibration_intercept_slope(y_true, y_prob)
    
    # Threshold at 0.5 for diagnostic accuracy context
    y_pred_05 = (y_prob >= 0.5).astype(int)
    acc = float(accuracy_score(y_true, y_pred_05))
    prec = float(precision_score(y_true, y_pred_05, zero_division=0))
    rec = float(recall_score(y_true, y_pred_05, zero_division=0))
    f1 = float(f1_score(y_true, y_pred_05, zero_division=0))
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_05).ravel()
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    
    return {
        "brier_score": round(brier, 4),
        "log_loss": round(ll, 4),
        "ece": round(ece, 4),
        "mce": round(mce, 4),
        "cal_intercept": round(intercept, 4),
        "cal_slope": round(slope, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "accuracy_at_05": round(acc, 4),
        "precision_at_05": round(prec, 4),
        "sensitivity_at_05": round(rec, 4),
        "specificity_at_05": round(spec, 4),
        "f1_at_05": round(f1, 4),
        "mean_prob": round(float(np.mean(y_prob)), 4),
        "std_prob": round(float(np.std(y_prob)), 4),
        "min_prob": round(float(np.min(y_prob)), 4),
        "max_prob": round(float(np.max(y_prob)), 4),
    }


def run_training_calibration_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    candidate_configs: List[Dict[str, Any]],
    n_splits: int = 5,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Conducts leakage-free Stratified K-Fold Cross-Validation on the 242-sample training cohort
    to evaluate Uncalibrated, Sigmoid, and Isotonic calibration during development.
    
    Leakage Protection:
    - Splits only X_train, y_train. Test set is strictly absent.
    - Preprocessing is fitted strictly on fold training portions.
    - Out-Of-Fold probability predictions are collected across all 242 training instances.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = list(skf.split(X_train, y_train))
    
    oof_probs_dict = {
        "sample_index": X_train.index.values,
        "true_label": y_train.values,
    }
    
    # Initialize containers for each candidate and method
    for cfg in candidate_configs:
        cand_name = cfg["candidate_name"]
        oof_probs_dict[f"{cand_name}__Uncalibrated"] = np.zeros(len(X_train))
        oof_probs_dict[f"{cand_name}__Sigmoid"] = np.zeros(len(X_train))
        oof_probs_dict[f"{cand_name}__Isotonic"] = np.zeros(len(X_train))
        
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        X_fold_tr = X_train.iloc[train_idx]
        y_fold_tr = y_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_val = y_train.iloc[val_idx]
        
        for cfg in candidate_configs:
            cand_name = cfg["candidate_name"]
            
            # 1. Uncalibrated Base Pipeline
            pipe_uncal = build_pipeline_from_config(cfg)
            pipe_uncal.fit(X_fold_tr, y_fold_tr)
            val_probs_uncal = pipe_uncal.predict_proba(X_fold_val)[:, 1]
            oof_probs_dict[f"{cand_name}__Uncalibrated"][val_idx] = val_probs_uncal
            
            # 2. Sigmoid Calibration fitted via cross-fitting on fold training portion
            cal_sig = CalibratedClassifierCV(
                estimator=build_pipeline_from_config(cfg),
                method="sigmoid",
                cv=3,
            )
            cal_sig.fit(X_fold_tr, y_fold_tr)
            val_probs_sig = cal_sig.predict_proba(X_fold_val)[:, 1]
            oof_probs_dict[f"{cand_name}__Sigmoid"][val_idx] = val_probs_sig
            
            # 3. Isotonic Calibration fitted via cross-fitting on fold training portion
            cal_iso = CalibratedClassifierCV(
                estimator=build_pipeline_from_config(cfg),
                method="isotonic",
                cv=3,
            )
            cal_iso.fit(X_fold_tr, y_fold_tr)
            val_probs_iso = cal_iso.predict_proba(X_fold_val)[:, 1]
            oof_probs_dict[f"{cand_name}__Isotonic"][val_idx] = val_probs_iso
            
    oof_df = pd.DataFrame(oof_probs_dict)
    
    # Compute metrics for each model and method across the full 242 OOF predictions
    summary_rows = []
    y_train_arr = y_train.values
    
    for cfg in candidate_configs:
        cand_name = cfg["candidate_name"]
        short_name = cfg["short_name"]
        
        for method in ["Uncalibrated", "Sigmoid", "Isotonic"]:
            col_name = f"{cand_name}__{method}"
            probs = oof_df[col_name].values
            m = calculate_calibration_metrics(y_train_arr, probs, n_bins=10)
            m["model"] = short_name
            m["candidate_name"] = cand_name
            m["calibration_method"] = method
            m["dataset_split"] = "train_oof_cv"
            m["sample_size"] = len(y_train)
            summary_rows.append(m)
            
    summary_df = pd.DataFrame(summary_rows)
    return summary_df, oof_df


def fit_calibrated_models_on_train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    candidate_configs: List[Dict[str, Any]],
    cv: int = 5,
) -> Dict[str, Dict[str, Any]]:
    """
    Fits final calibrated model instances on the entire 242-sample training cohort:
    - Base uncalibrated pipeline fit on X_train, y_train.
    - Sigmoid CalibratedClassifierCV(cv=5) fit on X_train, y_train.
    - Isotonic CalibratedClassifierCV(cv=5) fit on X_train, y_train.
    
    Returns a dictionary of fitted models keyed by model name and calibration method.
    """
    fitted_models = {}
    
    for cfg in candidate_configs:
        cand_name = cfg["candidate_name"]
        fitted_models[cand_name] = {}
        
        # 1. Uncalibrated
        pipe_uncal = build_pipeline_from_config(cfg)
        pipe_uncal.fit(X_train, y_train)
        fitted_models[cand_name]["Uncalibrated"] = pipe_uncal
        
        # 2. Sigmoid
        cal_sig = CalibratedClassifierCV(
            estimator=build_pipeline_from_config(cfg),
            method="sigmoid",
            cv=cv,
        )
        cal_sig.fit(X_train, y_train)
        fitted_models[cand_name]["Sigmoid"] = cal_sig
        
        # 3. Isotonic
        cal_iso = CalibratedClassifierCV(
            estimator=build_pipeline_from_config(cfg),
            method="isotonic",
            cv=cv,
        )
        cal_iso.fit(X_train, y_train)
        fitted_models[cand_name]["Isotonic"] = cal_iso
        
    return fitted_models


def evaluate_calibrated_models_on_holdout(
    fitted_models: Dict[str, Dict[str, Any]],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    candidate_configs: List[Dict[str, Any]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Conducts strict descriptive evaluation of the calibrated models on the untouched 61-sample holdout.
    
    IMPORTANT:
    This function is descriptive only. Test set predictions are NOT used to select calibration methods.
    """
    test_preds_dict = {
        "sample_index": X_test.index.values,
        "true_label": y_test.values,
    }
    
    summary_rows = []
    y_test_arr = y_test.values
    
    for cfg in candidate_configs:
        cand_name = cfg["candidate_name"]
        short_name = cfg["short_name"]
        
        for method in ["Uncalibrated", "Sigmoid", "Isotonic"]:
            model_obj = fitted_models[cand_name][method]
            test_probs = model_obj.predict_proba(X_test)[:, 1]
            
            col_name = f"{cand_name}__{method}"
            test_preds_dict[col_name] = test_probs
            
            m = calculate_calibration_metrics(y_test_arr, test_probs, n_bins=10)
            m["model"] = short_name
            m["candidate_name"] = cand_name
            m["calibration_method"] = method
            m["dataset_split"] = "holdout_test"
            m["sample_size"] = len(y_test)
            summary_rows.append(m)
            
    test_preds_df = pd.DataFrame(test_preds_dict)
    test_summary_df = pd.DataFrame(summary_rows)
    return test_summary_df, test_preds_df


def analyze_threshold_shifts(
    y_train: np.ndarray,
    uncal_oof_probs: np.ndarray,
    cal_oof_probs: np.ndarray,
    phase6_high_sens_thresh: float,
    phase6_youden_thresh: float,
    model_name: str,
    cal_method_name: str,
) -> Dict[str, Any]:
    """
    Investigates how empirical probability calibration shifts the numerical decision threshold
    associated with specific clinical operating points (High-Sensitivity and Balanced Youden).
    
    Examines:
    - Default threshold 0.50 behavior before vs after calibration.
    - High-sensitivity threshold: the calibrated threshold required to achieve >= 90% sensitivity on training data.
    - Balanced threshold: the calibrated threshold maximizing Youden's J on training data.
    """
    y_true = np.asarray(y_train).astype(int)
    thresholds = np.linspace(0.01, 0.99, 99)
    
    # 1. Uncalibrated operating points
    # High sens point
    uncal_pred_hs = (uncal_oof_probs >= phase6_high_sens_thresh).astype(int)
    uncal_rec_hs = recall_score(y_true, uncal_pred_hs, zero_division=0)
    uncal_tn, uncal_fp, _, _ = confusion_matrix(y_true, uncal_pred_hs).ravel()
    uncal_spec_hs = uncal_tn / (uncal_tn + uncal_fp)
    
    # Default 0.50
    uncal_pred_05 = (uncal_oof_probs >= 0.50).astype(int)
    uncal_rec_05 = recall_score(y_true, uncal_pred_05, zero_division=0)
    uncal_tn5, uncal_fp5, _, _ = confusion_matrix(y_true, uncal_pred_05).ravel()
    uncal_spec_05 = uncal_tn5 / (uncal_tn5 + uncal_fp5)
    
    # Calibrated scan across thresholds to find:
    # A. Calibrated threshold achieving >= 90% sensitivity (highest threshold >= 0.90 sens)
    cal_rec_list = []
    cal_spec_list = []
    cal_youden_list = []
    
    for t in thresholds:
        pred_t = (cal_oof_probs >= t).astype(int)
        r = recall_score(y_true, pred_t, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_true, pred_t).ravel()
        sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        cal_rec_list.append(r)
        cal_spec_list.append(sp)
        cal_youden_list.append(r + sp - 1.0)
        
    cal_rec_arr = np.array(cal_rec_list)
    cal_spec_arr = np.array(cal_spec_list)
    cal_youden_arr = np.array(cal_youden_list)
    
    # Target >= 90% sensitivity
    valid_hs_indices = np.where(cal_rec_arr >= 0.90)[0]
    if len(valid_hs_indices) > 0:
        best_hs_idx = valid_hs_indices[-1]  # Highest threshold meeting >= 90% sensitivity
        cal_high_sens_thresh = thresholds[best_hs_idx]
        cal_high_sens_rec = cal_rec_arr[best_hs_idx]
        cal_high_sens_spec = cal_spec_arr[best_hs_idx]
    else:
        best_hs_idx = 0
        cal_high_sens_thresh = thresholds[0]
        cal_high_sens_rec = cal_rec_arr[0]
        cal_high_sens_spec = cal_spec_arr[0]
        
    # Balanced Youden point
    best_youden_idx = int(np.argmax(cal_youden_arr))
    cal_youden_thresh = thresholds[best_youden_idx]
    cal_youden_rec = cal_rec_arr[best_youden_idx]
    cal_youden_spec = cal_spec_arr[best_youden_idx]
    cal_youden_val = cal_youden_arr[best_youden_idx]
    
    # Calibrated at 0.50
    cal_pred_05 = (cal_oof_probs >= 0.50).astype(int)
    cal_rec_05 = recall_score(y_true, cal_pred_05, zero_division=0)
    cal_tn5, cal_fp5, _, _ = confusion_matrix(y_true, cal_pred_05).ravel()
    cal_spec_05 = cal_tn5 / (cal_tn5 + cal_fp5)
    
    return {
        "model": model_name,
        "calibration_method": cal_method_name,
        "phase6_uncal_high_sens_threshold": phase6_high_sens_thresh,
        "uncal_high_sens_recall": round(float(uncal_rec_hs), 4),
        "uncal_high_sens_specificity": round(float(uncal_spec_hs), 4),
        "calibrated_high_sens_threshold": round(float(cal_high_sens_thresh), 2),
        "calibrated_high_sens_recall": round(float(cal_high_sens_rec), 4),
        "calibrated_high_sens_specificity": round(float(cal_high_sens_spec), 4),
        "phase6_uncal_youden_threshold": phase6_youden_thresh,
        "calibrated_youden_threshold": round(float(cal_youden_thresh), 2),
        "calibrated_youden_recall": round(float(cal_youden_rec), 4),
        "calibrated_youden_specificity": round(float(cal_youden_spec), 4),
        "calibrated_youden_j": round(float(cal_youden_val), 4),
        "uncal_at_050_recall": round(float(uncal_rec_05), 4),
        "uncal_at_050_specificity": round(float(uncal_spec_05), 4),
        "cal_at_050_recall": round(float(cal_rec_05), 4),
        "cal_at_050_specificity": round(float(cal_spec_05), 4),
    }
