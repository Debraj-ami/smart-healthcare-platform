"""
Heart Disease ML Experimentation Runner (Phase 5).

Executes reproducible, leakage-safe experimentation on the Cleveland Heart Disease dataset:
- Evaluates 44 candidate configurations across 3 model families (Logistic Regression, Random Forest, XGBoost)
- 5-repeat 5-fold Stratified Cross-Validation (25 validation folds) strictly on 242 training samples
- Preprocessing fit strictly inside CV folds
- Generates Out-Of-Fold (OOF) predictions on the 242 training samples
- Evaluates finalists ONCE on the locked 61-sample holdout test set
- Extracts mapped feature importance / coefficients
- Saves structured experiment artifacts and publication-quality plots
- Validates raw dataset integrity before and after
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any

from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from app.heart_preprocessing import (
    verify_raw_dataset_integrity,
    load_heart_raw_data,
    create_modeling_dataset,
    split_heart_data,
    get_heart_feature_names,
)
from app.heart_experimentation import (
    run_cross_validation_experiments,
    generate_out_of_fold_predictions,
    evaluate_candidates_on_holdout,
    extract_feature_importance_analysis,
    build_pipeline_from_config,
    get_candidate_configurations,
    calculate_metrics,
)


def main():
    print("=" * 70)
    print("HEART DISEASE — PHASE 5: ML EXPERIMENTATION & MODEL COMPARISON")
    print("=" * 70)
    
    # 1. Pre-experiment raw dataset integrity check
    print("\n[Step 1] Verifying raw dataset integrity...")
    valid, md5_pre, sha256_pre, size_pre = verify_raw_dataset_integrity()
    assert valid, f"Raw dataset corrupted! MD5: {md5_pre}, SHA256: {sha256_pre}"
    print(f"Raw dataset verified: 18,461 bytes | MD5: {md5_pre} | SHA-256: {sha256_pre}")
    
    # 2. Ingestion & Target Construction
    print("\n[Step 2] Ingesting raw Cleveland data and constructing binary target...")
    df_raw = load_heart_raw_data()
    X, y = create_modeling_dataset(df_raw)
    print(f"Dataset shape: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")
    print(f"Class distribution: y=0 (Absence): {(y == 0).sum()} | y=1 (Presence): {(y == 1).sum()}")
    
    # 3. Stratified Train / Test Split
    print("\n[Step 3] Performing stratified 80/20 train/test split (random_state=42)...")
    X_train, X_test, y_train, y_test = split_heart_data(X, y, test_size=0.20, random_state=42)
    print(f"Training split: {len(X_train)} samples (neg: {(y_train == 0).sum()}, pos: {(y_train == 1).sum()})")
    print(f"Test split:     {len(X_test)} samples (neg: {(y_test == 0).sum()}, pos: {(y_test == 1).sum()})")
    print("Hold-out test set is LOCKED during all cross-validation and hyperparameter selection.")
    
    # 4. Cross-Validation Experiments on Training Set
    print("\n[Step 4] Running 25-fold Repeated Stratified CV across 44 model configurations...")
    df_cv_results, _ = run_cross_validation_experiments(X_train, y_train, random_state=42, n_splits=5, n_repeats=5)
    
    # Sort by mean ROC-AUC
    df_cv_results.sort_values(by="mean_roc_auc", ascending=False, inplace=True)
    df_cv_results.reset_index(drop=True, inplace=True)
    
    print("\n--- TOP CANDIDATE PER MODEL FAMILY (CV METRICS) ---")
    for fam in ["LogisticRegression", "RandomForest", "XGBoost"]:
        top_fam = df_cv_results[df_cv_results["family"] == fam].iloc[0]
        print(f"[{fam}] {top_fam['config_id']}")
        print(f"  ROC-AUC:     {top_fam['mean_roc_auc']:.4f} ± {top_fam['std_roc_auc']:.4f}")
        print(f"  PR-AUC:      {top_fam['mean_pr_auc']:.4f} ± {top_fam['std_pr_auc']:.4f}")
        print(f"  Accuracy:    {top_fam['mean_accuracy']:.4f} ± {top_fam['std_accuracy']:.4f}")
        print(f"  Sensitivity: {top_fam['mean_recall']:.4f} ± {top_fam['std_recall']:.4f}")
        print(f"  Specificity: {top_fam['mean_specificity']:.4f} ± {top_fam['std_specificity']:.4f}")
        print(f"  F1-Score:    {top_fam['mean_f1']:.4f} ± {top_fam['std_f1']:.4f}")
        
    # 5. Define Representative Finalist Candidates
    all_configs_dict = get_candidate_configurations()
    
    # Top LR
    lr_best_cfg = next(c for c in all_configs_dict["LogisticRegression"] if c["config_id"] == "LR_robust_l1_C1.0")
    lr_best_cfg = dict(lr_best_cfg)
    lr_best_cfg["candidate_name"] = "LogisticRegression (Robust L1)"
    
    # Top RF
    rf_best_cfg = next(c for c in all_configs_dict["RandomForest"] if c["config_id"] == "RF_n200_dNone_leaf5_cwNone")
    rf_best_cfg = dict(rf_best_cfg)
    rf_best_cfg["candidate_name"] = "RandomForest (n=200, leaf=5)"
    
    # Top XGBoost
    xgb_best_cfg = next(c for c in all_configs_dict["XGBoost"] if c["config_id"] == "XGB_n100_d2_lr0.05_sub0.8_col0.8")
    xgb_best_cfg = dict(xgb_best_cfg)
    xgb_best_cfg["candidate_name"] = "XGBoost (n=100, depth=2, lr=0.05)"
    
    final_candidates = [lr_best_cfg, rf_best_cfg, xgb_best_cfg]
    
    # 6. Generate Out-Of-Fold (OOF) Predictions on Training Data
    print("\n[Step 5] Generating Out-Of-Fold (OOF) training predictions for finalists...")
    oof_df = generate_out_of_fold_predictions(X_train, y_train, final_candidates, random_state=42)
    print(f"OOF predictions generated for {len(oof_df)} training samples.")
    
    # 7. Final Hold-Out Evaluation on 61-sample Test Set
    print("\n[Step 6] Evaluating finalists ONCE on the locked 61-sample test set...")
    df_holdout, test_preds_df, fitted_pipelines = evaluate_candidates_on_holdout(
        X_train, y_train, X_test, y_test, final_candidates
    )
    
    print("\n--- FINAL TEST SET PERFORMANCE (61 SAMPLES: 33 NEG, 28 POS) ---")
    for idx, row in df_holdout.iterrows():
        print(f"Candidate: {row['candidate_name']}")
        print(f"  Test ROC-AUC:     {row['roc_auc']:.4f}")
        print(f"  Test PR-AUC:      {row['pr_auc']:.4f}")
        print(f"  Test Accuracy:    {row['accuracy']:.4f}")
        print(f"  Test Precision:   {row['precision']:.4f}")
        print(f"  Test Sensitivity: {row['recall']:.4f}")
        print(f"  Test Specificity: {row['specificity']:.4f}")
        print(f"  Test F1-Score:    {row['f1']:.4f}")
        print(f"  Confusion Matrix: TN={row['tn']}, FP={row['fp']}, FN={row['fn']}, TP={row['tp']}")
        
    # 8. Feature Analysis
    print("\n[Step 7] Extracting model-native feature importance and coefficients...")
    df_features = extract_feature_importance_analysis(fitted_pipelines)
    
    # 9. Save Structured Artifacts
    artifacts_dir = os.path.join("artifacts", "heart", "experiments")
    eval_dir = os.path.join("evaluation", "heart")
    os.makedirs(artifacts_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    
    print(f"\n[Step 8] Saving experiment artifacts to {artifacts_dir}...")
    
    # Save CV results
    cv_csv_path = os.path.join(artifacts_dir, "cv_results_all.csv")
    df_cv_results.to_csv(cv_csv_path, index=False)
    print(f"Saved: {cv_csv_path}")
    
    # Save Candidate Summary
    cand_summary_records = []
    for cand in final_candidates:
        c_id = cand["config_id"]
        cv_row = df_cv_results[df_cv_results["config_id"] == c_id].iloc[0]
        test_row = df_holdout[df_holdout["config_id"] == c_id].iloc[0]
        cand_summary_records.append({
            "candidate_name": cand["candidate_name"],
            "config_id": c_id,
            "family": cand["family"],
            "cv_roc_auc_mean": cv_row["mean_roc_auc"],
            "cv_roc_auc_std": cv_row["std_roc_auc"],
            "cv_pr_auc_mean": cv_row["mean_pr_auc"],
            "cv_pr_auc_std": cv_row["std_pr_auc"],
            "cv_f1_mean": cv_row["mean_f1"],
            "cv_f1_std": cv_row["std_f1"],
            "test_roc_auc": test_row["roc_auc"],
            "test_pr_auc": test_row["pr_auc"],
            "test_accuracy": test_row["accuracy"],
            "test_precision": test_row["precision"],
            "test_sensitivity": test_row["recall"],
            "test_specificity": test_row["specificity"],
            "test_f1": test_row["f1"],
            "test_tn": test_row["tn"],
            "test_fp": test_row["fp"],
            "test_fn": test_row["fn"],
            "test_tp": test_row["tp"],
        })
    df_cand_summary = pd.DataFrame(cand_summary_records)
    cand_summary_path = os.path.join(artifacts_dir, "candidate_summary.csv")
    df_cand_summary.to_csv(cand_summary_path, index=False)
    print(f"Saved: {cand_summary_path}")
    
    # Save Model Comparison CSV in evaluation/heart/ (matching Liver conventions)
    model_comp_path = os.path.join(eval_dir, "model_comparison.csv")
    df_cand_summary.to_csv(model_comp_path, index=False)
    print(f"Saved: {model_comp_path}")
    
    # Save OOF predictions
    oof_path = os.path.join(artifacts_dir, "oof_predictions.csv")
    oof_df.to_csv(oof_path, index=False)
    print(f"Saved: {oof_path}")
    
    # Save Test predictions
    test_preds_path = os.path.join(artifacts_dir, "test_predictions.csv")
    test_preds_df.to_csv(test_preds_path, index=False)
    print(f"Saved: {test_preds_path}")
    
    # Save Holdout metrics JSON
    holdout_json_path = os.path.join(artifacts_dir, "final_holdout_metrics.json")
    with open(holdout_json_path, "w") as f:
        json.dump(df_holdout.to_dict(orient="records"), f, indent=2)
    print(f"Saved: {holdout_json_path}")
    
    # Save Feature importance CSV
    feat_path = os.path.join(artifacts_dir, "feature_importance.csv")
    df_features.to_csv(feat_path, index=False)
    print(f"Saved: {feat_path}")
    
    # Save metadata JSON
    meta_path = os.path.join(artifacts_dir, "experiment_metadata.json")
    metadata = {
        "dataset": "processed.cleveland.data",
        "dataset_rows": 303,
        "dataset_cols": 14,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "md5_checksum": md5_pre,
        "sha256_checksum": sha256_pre,
        "random_state": 42,
        "cv_strategy": "RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)",
        "n_cv_folds": 25,
        "n_configurations_evaluated": len(df_cv_results),
        "finalist_candidates": [c["candidate_name"] for c in final_candidates],
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved: {meta_path}")
    
    # 10. Generate Visualizations
    print("\n[Step 9] Generating visual evaluation plots...")
    sns.set_theme(style="whitegrid", font_scale=1.1)
    
    # A. Confusion Matrices Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for i, cand in enumerate(final_candidates):
        cand_name = cand["candidate_name"]
        pred_col = f"test_pred_{cand_name}"
        cm = confusion_matrix(y_test, test_preds_df[pred_col], labels=[0, 1])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=axes[i],
            xticklabels=["Absence (0)", "Presence (1)"],
            yticklabels=["Absence (0)", "Presence (1)"],
        )
        axes[i].set_title(f"{cand_name}\n(Test N=61, Default Thr=0.5)")
        axes[i].set_xlabel("Predicted Label")
        axes[i].set_ylabel("True Label")
    plt.tight_layout()
    cm_plot_path = os.path.join(artifacts_dir, "confusion_matrices.png")
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()
    print(f"Saved: {cm_plot_path}")
    
    # B. ROC Curves Plot
    plt.figure(figsize=(8, 7))
    for cand in final_candidates:
        cand_name = cand["candidate_name"]
        prob_col = f"test_prob_{cand_name}"
        fpr, tpr, _ = roc_curve(y_test, test_preds_df[prob_col])
        score = roc_auc_score(y_test, test_preds_df[prob_col])
        plt.plot(fpr, tpr, lw=2.2, label=f"{cand_name} (AUC = {score:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.7, label="Chance Level (AUC = 0.500)")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity / Recall)")
    plt.title("Receiver Operating Characteristic (ROC) Curves\n(Hold-Out Test Set, N=61)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_plot_path = os.path.join(artifacts_dir, "roc_curves.png")
    plt.savefig(roc_plot_path, dpi=300)
    plt.close()
    print(f"Saved: {roc_plot_path}")
    
    # C. Precision-Recall Curves Plot
    plt.figure(figsize=(8, 7))
    test_prevalence = float((y_test == 1).mean())
    for cand in final_candidates:
        cand_name = cand["candidate_name"]
        prob_col = f"test_prob_{cand_name}"
        prec, rec, _ = precision_recall_curve(y_test, test_preds_df[prob_col])
        score = average_precision_score(y_test, test_preds_df[prob_col])
        plt.plot(rec, prec, lw=2.2, label=f"{cand_name} (PR-AUC = {score:.3f})")
    plt.axhline(test_prevalence, color="k", linestyle="--", lw=1.5, alpha=0.7, label=f"Prevalence Baseline ({test_prevalence:.3f})")
    plt.xlim([-0.02, 1.02])
    plt.ylim([0.3, 1.05])
    plt.xlabel("Recall (Sensitivity)")
    plt.ylabel("Precision (Positive Predictive Value)")
    plt.title("Precision-Recall (PR) Curves\n(Hold-Out Test Set, N=61)")
    plt.legend(loc="lower left")
    plt.tight_layout()
    pr_plot_path = os.path.join(artifacts_dir, "precision_recall_curves.png")
    plt.savefig(pr_plot_path, dpi=300)
    plt.close()
    print(f"Saved: {pr_plot_path}")
    
    # D. CV Metric Comparison Barplot
    plt.figure(figsize=(10, 6))
    comp_data = []
    for cand in final_candidates:
        c_id = cand["config_id"]
        row = df_cv_results[df_cv_results["config_id"] == c_id].iloc[0]
        comp_data.append({
            "Candidate": cand["candidate_name"],
            "Metric": "ROC-AUC",
            "Mean": row["mean_roc_auc"],
            "Std": row["std_roc_auc"],
        })
        comp_data.append({
            "Candidate": cand["candidate_name"],
            "Metric": "PR-AUC",
            "Mean": row["mean_pr_auc"],
            "Std": row["std_pr_auc"],
        })
        comp_data.append({
            "Candidate": cand["candidate_name"],
            "Metric": "F1-Score",
            "Mean": row["mean_f1"],
            "Std": row["std_f1"],
        })
    df_comp = pd.DataFrame(comp_data)
    
    x = np.arange(len(final_candidates))
    width = 0.25
    metrics_list = ["ROC-AUC", "PR-AUC", "F1-Score"]
    
    plt.figure(figsize=(10, 6))
    for i, m in enumerate(metrics_list):
        subset = df_comp[df_comp["Metric"] == m]
        plt.bar(
            x + (i - 1) * width,
            subset["Mean"],
            width,
            yerr=subset["Std"],
            capsize=5,
            label=m,
            alpha=0.85
        )
    plt.xticks(x, [c["candidate_name"] for c in final_candidates], rotation=10, ha="right")
    plt.ylabel("Cross-Validation Score (Mean ± Std)")
    plt.ylim([0.65, 1.0])
    plt.title("Repeated Stratified 5x5 Cross-Validation Performance (25 Folds)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    cv_plot_path = os.path.join(artifacts_dir, "cv_comparison_plot.png")
    plt.savefig(cv_plot_path, dpi=300)
    plt.close()
    print(f"Saved: {cv_plot_path}")
    
    # E. Feature Importance Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    for i, cand in enumerate(final_candidates):
        cand_name = cand["candidate_name"]
        subset = df_features[df_features["candidate_name"] == cand_name].copy()
        subset.sort_values(by="abs_magnitude", ascending=False, inplace=True)
        top10 = subset.head(10)
        
        if "LogisticRegression" in cand_name:
            sns.barplot(data=top10, x="importance_or_coef", y="feature", ax=axes[i], palette="vlag")
            axes[i].set_title(f"{cand_name}\nTop Coefficients (L1 Penalty)")
            axes[i].set_xlabel("Coefficient Value")
        else:
            sns.barplot(data=top10, x="abs_magnitude", y="feature", ax=axes[i], palette="viridis")
            axes[i].set_title(f"{cand_name}\nTop Gini/Split Importances")
            axes[i].set_xlabel("Relative Importance")
            
        axes[i].set_ylabel("Transformed Feature")
    plt.tight_layout()
    feat_plot_path = os.path.join(artifacts_dir, "feature_importance_plot.png")
    plt.savefig(feat_plot_path, dpi=300)
    plt.close()
    print(f"Saved: {feat_plot_path}")
    
    # 11. Post-experiment raw dataset integrity check
    print("\n[Step 10] Verifying post-experiment raw dataset integrity...")
    valid_post, md5_post, sha256_post, size_post = verify_raw_dataset_integrity()
    assert valid_post, f"Raw dataset corrupted post-experiment! MD5: {md5_post}"
    assert md5_post == md5_pre and sha256_post == sha256_pre, "Checksum mismatch post-experiment!"
    print(f"Raw dataset verified unchanged: MD5 {md5_post} | SHA-256 {sha256_post}")
    
    print("\n" + "=" * 70)
    print("PHASE 5 MACHINE LEARNING EXPERIMENTATION COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
