"""
Heart Disease ML Experimentation Module (Phase 5).

This module implements a reproducible, leakage-safe experimental framework for
comparing candidate classification models on the Cleveland Heart Disease dataset.

Boundaries:
- Experimentation only; no production deployment or threshold tuning.
- 5-repeat 5-fold Stratified CV (25 validation folds) on 242 training samples.
- Preprocessing strictly inside CV pipelines.
- Hold-out test set (61 samples) evaluated strictly once on final candidates.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)

from app.heart_preprocessing import (
    load_heart_raw_data,
    create_modeling_dataset,
    split_heart_data,
    build_logistic_regression_preprocessor,
    build_tree_preprocessor,
    get_heart_feature_names,
    verify_raw_dataset_integrity,
)


def calculate_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """
    Calculates primary screening evaluation metrics.
    Note: threshold=0.5 is an evaluation convention only, not a clinical threshold.
    """
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    roc_auc = float(roc_auc_score(y_true, y_prob))
    pr_auc = float(average_precision_score(y_true, y_prob))
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    
    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "specificity": spec,
        "f1": f1,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def get_candidate_configurations() -> Dict[str, List[Dict[str, Any]]]:
    """
    Returns the controlled experimental configurations for all three model families:
    - Logistic Regression: 20 configurations (2 scalers x 2 penalties x 5 C values)
    - Random Forest: 12 configurations (compact, defensible search space testing depth, leaf, trees, cw)
    - XGBoost: 12 configurations (compact search space testing depth, lr, n_est, subsample, colsample)
    Total: 44 configurations.
    """
    # 1. Logistic Regression: 2 scalers x 2 penalties x 5 C values = 20 configs
    lr_configs = []
    c_values = [0.01, 0.1, 1.0, 10.0, 100.0]
    scalers = ["standard", "robust"]
    penalties = ["l1", "l2"]
    
    for scaler in scalers:
        for penalty in penalties:
            for c in c_values:
                config_id = f"LR_{scaler}_{penalty}_C{c}"
                lr_configs.append({
                    "family": "LogisticRegression",
                    "config_id": config_id,
                    "scaler": scaler,
                    "penalty": penalty,
                    "C": c,
                    "solver": "liblinear",
                    "random_state": 42,
                    "description": f"LR ({scaler} scaler, penalty={penalty}, C={c})"
                })
                
    # 2. Random Forest: 12 compact, defensible configurations
    rf_configs = [
        # Depth & leaf exploration at n=200, cw=None
        {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 1, "class_weight": None, "config_id": "RF_n200_dNone_leaf1_cwNone"},
        {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 2, "class_weight": None, "config_id": "RF_n200_dNone_leaf2_cwNone"},
        {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 5, "class_weight": None, "config_id": "RF_n200_dNone_leaf5_cwNone"},
        {"n_estimators": 200, "max_depth": 5, "min_samples_leaf": 1, "class_weight": None, "config_id": "RF_n200_d5_leaf1_cwNone"},
        {"n_estimators": 200, "max_depth": 5, "min_samples_leaf": 2, "class_weight": None, "config_id": "RF_n200_d5_leaf2_cwNone"},
        {"n_estimators": 200, "max_depth": 5, "min_samples_leaf": 5, "class_weight": None, "config_id": "RF_n200_d5_leaf5_cwNone"},
        {"n_estimators": 200, "max_depth": 10, "min_samples_leaf": 1, "class_weight": None, "config_id": "RF_n200_d10_leaf1_cwNone"},
        {"n_estimators": 200, "max_depth": 10, "min_samples_leaf": 2, "class_weight": None, "config_id": "RF_n200_d10_leaf2_cwNone"},
        # Class weighting comparison
        {"n_estimators": 200, "max_depth": 5, "min_samples_leaf": 2, "class_weight": "balanced", "config_id": "RF_n200_d5_leaf2_cwbalanced"},
        {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 2, "class_weight": "balanced", "config_id": "RF_n200_dNone_leaf2_cwbalanced"},
        # Estimator scaling comparison (n=500)
        {"n_estimators": 500, "max_depth": 5, "min_samples_leaf": 2, "class_weight": None, "config_id": "RF_n500_d5_leaf2_cwNone"},
        {"n_estimators": 500, "max_depth": None, "min_samples_leaf": 2, "class_weight": None, "config_id": "RF_n500_dNone_leaf2_cwNone"},
    ]
    for cfg in rf_configs:
        cfg["family"] = "RandomForest"
        cfg["random_state"] = 42
        cfg["description"] = f"RF (n={cfg['n_estimators']}, depth={cfg['max_depth']}, leaf={cfg['min_samples_leaf']}, cw={cfg['class_weight']})"
        
    # 3. XGBoost: 12 compact, defensible configurations
    xgb_configs = [
        # Depth exploration at lr=0.05, n=100, sub=0.8, col=0.8
        {"n_estimators": 100, "max_depth": 2, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "config_id": "XGB_n100_d2_lr0.05_sub0.8_col0.8"},
        {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "config_id": "XGB_n100_d3_lr0.05_sub0.8_col0.8"},
        {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "config_id": "XGB_n100_d4_lr0.05_sub0.8_col0.8"},
        # Learning rate exploration
        {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8, "config_id": "XGB_n100_d3_lr0.03_sub0.8_col0.8"},
        {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8, "config_id": "XGB_n100_d3_lr0.1_sub0.8_col0.8"},
        # Estimators exploration (n=200, n=300)
        {"n_estimators": 200, "max_depth": 2, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "config_id": "XGB_n200_d2_lr0.05_sub0.8_col0.8"},
        {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "config_id": "XGB_n200_d3_lr0.05_sub0.8_col0.8"},
        {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "config_id": "XGB_n300_d3_lr0.05_sub0.8_col0.8"},
        # Subsampling and feature sampling exploration
        {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05, "subsample": 1.0, "colsample_bytree": 0.8, "config_id": "XGB_n100_d3_lr0.05_sub1.0_col0.8"},
        {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 1.0, "config_id": "XGB_n100_d3_lr0.05_sub0.8_col1.0"},
        {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05, "subsample": 1.0, "colsample_bytree": 1.0, "config_id": "XGB_n100_d3_lr0.05_sub1.0_col1.0"},
        {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05, "subsample": 1.0, "colsample_bytree": 1.0, "config_id": "XGB_n200_d3_lr0.05_sub1.0_col1.0"},
    ]
    for cfg in xgb_configs:
        cfg["family"] = "XGBoost"
        cfg["eval_metric"] = "logloss"
        cfg["random_state"] = 42
        cfg["description"] = f"XGB (n={cfg['n_estimators']}, depth={cfg['max_depth']}, lr={cfg['learning_rate']}, sub={cfg['subsample']}, col={cfg['colsample_bytree']})"
        
    return {
        "LogisticRegression": lr_configs,
        "RandomForest": rf_configs,
        "XGBoost": xgb_configs,
    }


def build_pipeline_from_config(config: Dict[str, Any]) -> Pipeline:
    """
    Constructs a complete sklearn Pipeline with leakage-safe preprocessor and classifier.
    """
    family = config["family"]
    if family == "LogisticRegression":
        preprocessor = build_logistic_regression_preprocessor(
            scaler=config["scaler"],
            drop="first"
        )
        clf = LogisticRegression(
            penalty=config["penalty"],
            C=config["C"],
            solver=config["solver"],
            random_state=config["random_state"],
            max_iter=1000
        )
    elif family == "RandomForest":
        preprocessor = build_tree_preprocessor(drop=None)
        clf = RandomForestClassifier(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            min_samples_leaf=config["min_samples_leaf"],
            class_weight=config["class_weight"],
            random_state=config["random_state"],
            n_jobs=1
        )
    elif family == "XGBoost":
        preprocessor = build_tree_preprocessor(drop=None)
        clf = XGBClassifier(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            learning_rate=config["learning_rate"],
            subsample=config["subsample"],
            colsample_bytree=config["colsample_bytree"],
            eval_metric=config["eval_metric"],
            random_state=config["random_state"],
            n_jobs=1
        )
    else:
        raise ValueError(f"Unknown family: {family}")
        
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", clf)
    ])


def run_cross_validation_experiments(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
    n_splits: int = 5,
    n_repeats: int = 5,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Runs leakage-safe Repeated Stratified K-Fold CV across all candidate configurations.
    
    Preprocessing is fit strictly inside each validation fold on fold training data.
    """
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    splits = list(rskf.split(X_train, y_train))
    n_folds = len(splits)  # 25 folds
    
    config_dict = get_candidate_configurations()
    all_configs = []
    for fam in ["LogisticRegression", "RandomForest", "XGBoost"]:
        all_configs.extend(config_dict[fam])
        
    print(f"Starting Repeated Stratified K-Fold CV: {n_folds} folds across {len(all_configs)} configurations...")
    
    # Pre-fit preprocessors per fold to optimize runtime while maintaining strict leakage isolation
    # 3 preprocessor types: LR-Standard, LR-Robust, Tree
    fold_data = []
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        X_tr = X_train.iloc[train_idx]
        y_tr = y_train.iloc[train_idx].values
        X_va = X_train.iloc[val_idx]
        y_va = y_train.iloc[val_idx].values
        
        # 1. LR Standard
        prep_std = build_logistic_regression_preprocessor(scaler="standard", drop="first")
        X_tr_std = prep_std.fit_transform(X_tr)
        X_va_std = prep_std.transform(X_va)
        
        # 2. LR Robust
        prep_rob = build_logistic_regression_preprocessor(scaler="robust", drop="first")
        X_tr_rob = prep_rob.fit_transform(X_tr)
        X_va_rob = prep_rob.transform(X_va)
        
        # 3. Tree (Random Forest & XGBoost)
        prep_tree = build_tree_preprocessor(drop=None)
        X_tr_tree = prep_tree.fit_transform(X_tr)
        X_va_tree = prep_tree.transform(X_va)
        
        fold_data.append({
            "train_idx": train_idx,
            "val_idx": val_idx,
            "y_tr": y_tr,
            "y_va": y_va,
            "std": (X_tr_std, X_va_std),
            "rob": (X_tr_rob, X_va_rob),
            "tree": (X_tr_tree, X_va_tree),
        })
        
    results_records = []
    
    for cfg in all_configs:
        fam = cfg["family"]
        fold_metrics = []
        
        for f_idx, fd in enumerate(fold_data):
            y_tr = fd["y_tr"]
            y_va = fd["y_va"]
            
            if fam == "LogisticRegression":
                scaler_type = cfg["scaler"]
                X_tr_f, X_va_f = fd["std"] if scaler_type == "standard" else fd["rob"]
                model = LogisticRegression(
                    penalty=cfg["penalty"],
                    C=cfg["C"],
                    solver=cfg["solver"],
                    random_state=cfg["random_state"],
                    max_iter=1000
                )
            elif fam == "RandomForest":
                X_tr_f, X_va_f = fd["tree"]
                model = RandomForestClassifier(
                    n_estimators=cfg["n_estimators"],
                    max_depth=cfg["max_depth"],
                    min_samples_leaf=cfg["min_samples_leaf"],
                    class_weight=cfg["class_weight"],
                    random_state=cfg["random_state"],
                    n_jobs=1
                )
            elif fam == "XGBoost":
                X_tr_f, X_va_f = fd["tree"]
                model = XGBClassifier(
                    n_estimators=cfg["n_estimators"],
                    max_depth=cfg["max_depth"],
                    learning_rate=cfg["learning_rate"],
                    subsample=cfg["subsample"],
                    colsample_bytree=cfg["colsample_bytree"],
                    eval_metric=cfg["eval_metric"],
                    random_state=cfg["random_state"],
                    n_jobs=1
                )
            else:
                continue
                
            model.fit(X_tr_f, y_tr)
            y_prob = model.predict_proba(X_va_f)[:, 1]
            m = calculate_metrics(y_va, y_prob)
            fold_metrics.append(m)
            
        # Aggregate across 25 folds
        roc_aucs = [m["roc_auc"] for m in fold_metrics]
        pr_aucs = [m["pr_auc"] for m in fold_metrics]
        accs = [m["accuracy"] for m in fold_metrics]
        precs = [m["precision"] for m in fold_metrics]
        recs = [m["recall"] for m in fold_metrics]
        specs = [m["specificity"] for m in fold_metrics]
        f1s = [m["f1"] for m in fold_metrics]
        
        results_records.append({
            "family": fam,
            "config_id": cfg["config_id"],
            "description": cfg["description"],
            "mean_roc_auc": float(np.mean(roc_aucs)),
            "std_roc_auc": float(np.std(roc_aucs)),
            "mean_pr_auc": float(np.mean(pr_aucs)),
            "std_pr_auc": float(np.std(pr_aucs)),
            "mean_accuracy": float(np.mean(accs)),
            "std_accuracy": float(np.std(accs)),
            "mean_precision": float(np.mean(precs)),
            "std_precision": float(np.std(precs)),
            "mean_recall": float(np.mean(recs)),
            "std_recall": float(np.std(recs)),
            "mean_specificity": float(np.mean(specs)),
            "std_specificity": float(np.std(specs)),
            "mean_f1": float(np.mean(f1s)),
            "std_f1": float(np.std(f1s)),
            "n_folds": n_folds,
        })
        
    df_results = pd.DataFrame(results_records)
    print(f"Completed CV for all {len(df_results)} configurations.")
    return df_results, {"fold_data": fold_data, "splits": splits}


def generate_out_of_fold_predictions(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    candidate_configs: List[Dict[str, Any]],
    random_state: int = 42,
    n_splits: int = 5,
    n_repeats: int = 5,
) -> pd.DataFrame:
    """
    Generates out-of-fold predicted probabilities on the 242 training samples
    using Repeated Stratified K-Fold CV (averaging across the 5 repeats).
    """
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    n_samples = len(X_train)
    
    oof_df = pd.DataFrame({
        "sample_index": X_train.index,
        "true_label": y_train.values,
    })
    
    for cfg in candidate_configs:
        model_name = cfg["candidate_name"]
        sum_probs = np.zeros(n_samples, dtype=float)
        repeat0_probs = np.zeros(n_samples, dtype=float)
        
        for repeat_idx, (train_idx, val_idx) in enumerate(rskf.split(X_train, y_train)):
            rep = repeat_idx // n_splits
            X_tr = X_train.iloc[train_idx]
            y_tr = y_train.iloc[train_idx]
            X_va = X_train.iloc[val_idx]
            
            pipe = build_pipeline_from_config(cfg)
            pipe.fit(X_tr, y_tr)
            probs = pipe.predict_proba(X_va)[:, 1]
            
            sum_probs[val_idx] += probs
            if rep == 0:
                repeat0_probs[val_idx] = probs
                
        oof_df[f"oof_prob_{model_name}"] = sum_probs / float(n_repeats)
        oof_df[f"oof_prob_{model_name}_repeat0"] = repeat0_probs
        
    return oof_df


def evaluate_candidates_on_holdout(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    candidate_configs: List[Dict[str, Any]],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Pipeline]]:
    """
    Evaluates final candidate models ONCE on the untouched 61-sample test set.
    Fits preprocessor and classifier strictly on the full 242 training samples.
    """
    metrics_list = []
    test_preds_df = pd.DataFrame({
        "sample_index": X_test.index,
        "true_label": y_test.values,
    })
    fitted_pipelines = {}
    
    for cfg in candidate_configs:
        cand_name = cfg["candidate_name"]
        pipe = build_pipeline_from_config(cfg)
        pipe.fit(X_train, y_train)
        fitted_pipelines[cand_name] = pipe
        
        y_test_prob = pipe.predict_proba(X_test)[:, 1]
        m = calculate_metrics(y_test.values, y_test_prob, threshold=0.5)
        m["candidate_name"] = cand_name
        m["family"] = cfg["family"]
        m["config_id"] = cfg["config_id"]
        m["description"] = cfg["description"]
        m["n_train"] = len(X_train)
        m["n_test"] = len(X_test)
        m["n_pos_test"] = int(y_test.sum())
        m["n_neg_test"] = int((y_test == 0).sum())
        metrics_list.append(m)
        
        test_preds_df[f"test_prob_{cand_name}"] = y_test_prob
        test_preds_df[f"test_pred_{cand_name}"] = (y_test_prob >= 0.5).astype(int)
        
    df_holdout = pd.DataFrame(metrics_list)
    return df_holdout, test_preds_df, fitted_pipelines


def extract_feature_importance_analysis(
    fitted_pipelines: Dict[str, Pipeline]
) -> pd.DataFrame:
    """
    Extracts model-native coefficients and feature importances mapped
    to transformed feature names via get_heart_feature_names().
    """
    records = []
    for cand_name, pipe in fitted_pipelines.items():
        preprocessor = pipe.named_steps["preprocessor"]
        clf = pipe.named_steps["classifier"]
        feature_names = get_heart_feature_names(preprocessor)
        
        if hasattr(clf, "coef_"):
            coefs = clf.coef_[0]
            for f_name, c_val in zip(feature_names, coefs):
                records.append({
                    "candidate_name": cand_name,
                    "model_type": "LogisticRegression",
                    "feature": f_name,
                    "importance_or_coef": float(c_val),
                    "abs_magnitude": float(abs(c_val)),
                })
        elif hasattr(clf, "feature_importances_"):
            imps = clf.feature_importances_
            m_type = "RandomForest" if "RF" in cand_name else "XGBoost"
            for f_name, imp_val in zip(feature_names, imps):
                records.append({
                    "candidate_name": cand_name,
                    "model_type": m_type,
                    "feature": f_name,
                    "importance_or_coef": float(imp_val),
                    "abs_magnitude": float(imp_val),
                })
                
    return pd.DataFrame(records)
