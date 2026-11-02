"""
Heart Disease Calibration Execution Runner (Phase 7).

Executes reproducible, leakage-free calibration evaluation for the Cleveland Heart Disease
finalist models:
1. Validates raw dataset cryptographic integrity (MD5, SHA-256).
2. Loads modeling dataset and performs stratified 80/20 train/test split (242 train, 61 holdout).
3. Evaluates Uncalibrated, Sigmoid, and Isotonic calibration on the 242-sample training cohort
   using 5-fold Stratified Cross-Validation (development stage).
4. Fits final calibrated models on the full 242-sample training cohort.
5. Evaluates calibrated models strictly once and descriptively on the 61-sample locked holdout.
6. Calculates Brier score, Log loss, ECE, MCE, calibration intercept and slope.
7. Conducts threshold shift analysis connecting Phase 6 operating points to calibrated probabilities.
8. Generates reliability diagrams, probability distribution plots, and structured CSV/JSON artifacts.
9. Exports platform evaluation comparison to evaluation/heart/heart_calibration_comparison.csv.
10. Validates raw dataset integrity before and after execution.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve

from app.heart_preprocessing import (
    verify_raw_dataset_integrity,
    load_heart_raw_data,
    create_modeling_dataset,
    split_heart_data,
)
from app.heart_calibration import (
    get_heart_finalist_configs,
    compute_ece,
    calculate_calibration_metrics,
    run_training_calibration_cv,
    fit_calibrated_models_on_train,
    evaluate_calibrated_models_on_holdout,
    analyze_threshold_shifts,
)

# Phase 6 thresholds
PHASE6_OPERATING_POINTS = {
    "Logistic Regression": {"high_sens_thresh": 0.31, "youden_thresh": 0.56},
    "Random Forest": {"high_sens_thresh": 0.32, "youden_thresh": 0.55},
    "XGBoost": {"high_sens_thresh": 0.24, "youden_thresh": 0.62},
}


def main():
    print("=" * 70)
    print("HEART DISEASE — PHASE 7: PROBABILITY CALIBRATION ANALYSIS")
    print("=" * 70)

    # 1. Dataset Integrity Verification
    print("\n[Step 1] Verifying raw Cleveland Heart dataset integrity...")
    is_valid, md5_hash, sha_hash, file_size = verify_raw_dataset_integrity()
    if not is_valid:
        print(f"CRITICAL ERROR: Raw dataset checksum mismatch! Aborting.")
        print(f"MD5: {md5_hash}, SHA256: {sha_hash}, Size: {file_size}")
        sys.exit(1)
    print(f"Dataset integrity verified: MD5={md5_hash}, Size={file_size} bytes.")

    # 2. Data Loading & Splitting
    print("\n[Step 2] Loading dataset and performing stratified split...")
    df_raw = load_heart_raw_data()
    X, y = create_modeling_dataset(df_raw)
    X_train, X_test, y_train, y_test = split_heart_data(X, y, test_size=0.20, random_state=42)
    print(f"Training cohort: {len(X_train)} samples ({y_train.sum()} positive, {(y_train == 0).sum()} negative)")
    print(f"Hold-out cohort: {len(X_test)} samples ({y_test.sum()} positive, {(y_test == 0).sum()} negative)")
    assert len(X_train) == 242
    assert len(X_test) == 61

    # 3. Model Configuration
    candidate_configs = get_heart_finalist_configs()
    print(f"\n[Step 3] Loaded {len(candidate_configs)} finalist configurations:")
    for cfg in candidate_configs:
        print(f"  - {cfg['candidate_name']} ({cfg['family']})")

    # 4. Calibration Development via 5-Fold Stratified CV on Training Set (N=242)
    print("\n[Step 4] Running 5-fold Stratified CV on training cohort for calibration development...")
    train_summary_df, oof_probs_df = run_training_calibration_cv(
        X_train, y_train, candidate_configs, n_splits=5, random_state=42
    )
    print("\n--- TRAINING COHORT (CV OUT-OF-FOLD) CALIBRATION METRICS ---")
    cols_to_show = ["model", "calibration_method", "brier_score", "log_loss", "ece", "cal_intercept", "cal_slope", "roc_auc", "pr_auc"]
    print(train_summary_df[cols_to_show].to_string(index=False))

    # 5. Fit Calibrated Pipelines on Full Training Cohort (N=242)
    print("\n[Step 5] Fitting final calibrated pipelines strictly on full training cohort...")
    fitted_models = fit_calibrated_models_on_train(X_train, y_train, candidate_configs, cv=5)
    print("Fitted Uncalibrated, Sigmoid, and Isotonic pipelines for all 3 finalists.")

    # 6. Descriptive Evaluation on Locked Hold-Out Test Set (N=61)
    print("\n[Step 6] Evaluating calibrated models descriptively on locked 61-sample holdout...")
    holdout_summary_df, test_preds_df = evaluate_calibrated_models_on_holdout(
        fitted_models, X_test, y_test, candidate_configs
    )
    print("\n--- HOLDOUT COHORT (61 SAMPLES) CALIBRATION METRICS ---")
    print(holdout_summary_df[cols_to_show].to_string(index=False))

    # 7. Threshold Shift Analysis (Phase 6 Operating Points)
    print("\n[Step 7] Analyzing threshold shifts connecting Phase 6 operating points...")
    shift_records = []
    for cfg in candidate_configs:
        c_name = cfg["candidate_name"]
        s_name = cfg["short_name"]
        uncal_oof = oof_probs_df[f"{c_name}__Uncalibrated"].values
        hs_t = PHASE6_OPERATING_POINTS[s_name]["high_sens_thresh"]
        yd_t = PHASE6_OPERATING_POINTS[s_name]["youden_thresh"]

        for method in ["Sigmoid", "Isotonic"]:
            cal_oof = oof_probs_df[f"{c_name}__{method}"].values
            shift = analyze_threshold_shifts(
                y_train.values, uncal_oof, cal_oof, hs_t, yd_t, s_name, method
            )
            shift_records.append(shift)

    shift_df = pd.DataFrame(shift_records)
    print("\n--- THRESHOLD SHIFT ANALYSIS SUMMARY ---")
    shift_display_cols = [
        "model", "calibration_method",
        "phase6_uncal_high_sens_threshold", "calibrated_high_sens_threshold",
        "calibrated_high_sens_recall", "calibrated_high_sens_specificity",
        "phase6_uncal_youden_threshold", "calibrated_youden_threshold",
        "calibrated_youden_recall", "calibrated_youden_specificity"
    ]
    print(shift_df[shift_display_cols].to_string(index=False))

    # 8. Calibration Binning Details (ECE breakdown)
    print("\n[Step 8] Computing calibration binning details (N_bins=10)...")
    bin_details_records = []
    
    # Training OOF Bins
    for cfg in candidate_configs:
        c_name = cfg["candidate_name"]
        s_name = cfg["short_name"]
        for method in ["Uncalibrated", "Sigmoid", "Isotonic"]:
            probs = oof_probs_df[f"{c_name}__{method}"].values
            _, _, b_df = compute_ece(y_train.values, probs, n_bins=10, strategy="uniform")
            b_df["model"] = s_name
            b_df["calibration_method"] = method
            b_df["dataset_split"] = "train_oof_cv"
            bin_details_records.append(b_df)
            
    # Holdout Bins (5 bins also computed due to N=61)
    for cfg in candidate_configs:
        c_name = cfg["candidate_name"]
        s_name = cfg["short_name"]
        for method in ["Uncalibrated", "Sigmoid", "Isotonic"]:
            probs = test_preds_df[f"{c_name}__{method}"].values
            _, _, b_df = compute_ece(y_test.values, probs, n_bins=5, strategy="uniform")
            b_df["model"] = s_name
            b_df["calibration_method"] = method
            b_df["dataset_split"] = "holdout_test_5bins"
            bin_details_records.append(b_df)

    all_bins_df = pd.concat(bin_details_records, ignore_index=True)

    # 9. Save Artifacts
    artifacts_dir = os.path.join("artifacts", "heart", "calibration")
    eval_dir = os.path.join("evaluation", "heart")
    os.makedirs(artifacts_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    print(f"\n[Step 9] Saving structured artifacts to {artifacts_dir}...")

    # Full combined results
    combined_results_df = pd.concat([train_summary_df, holdout_summary_df], ignore_index=True)
    results_csv_path = os.path.join(artifacts_dir, "calibration_results.csv")
    combined_results_df.to_csv(results_csv_path, index=False)
    print(f"Saved: {results_csv_path}")

    # Holdout comparison summary (matching liver platform convention)
    eval_csv_path = os.path.join(eval_dir, "heart_calibration_comparison.csv")
    platform_eval_df = holdout_summary_df[["model", "calibration_method", "brier_score", "log_loss", "roc_auc", "pr_auc"]].copy()
    platform_eval_df.rename(columns={
        "model": "Model",
        "calibration_method": "Calibration_Method",
        "brier_score": "Brier_Score",
        "log_loss": "Log_Loss",
        "roc_auc": "ROC_AUC",
        "pr_auc": "PR_AUC"
    }, inplace=True)
    platform_eval_df.to_csv(eval_csv_path, index=False)
    print(f"Saved: {eval_csv_path}")

    # Artifact calibration summary
    cal_summary_csv = os.path.join(artifacts_dir, "calibration_summary.csv")
    platform_eval_df.to_csv(cal_summary_csv, index=False)
    print(f"Saved: {cal_summary_csv}")

    # OOF and Test predictions
    oof_csv_path = os.path.join(artifacts_dir, "oof_calibrated_predictions.csv")
    oof_probs_df.to_csv(oof_csv_path, index=False)
    print(f"Saved: {oof_csv_path}")

    test_pred_csv_path = os.path.join(artifacts_dir, "test_calibrated_predictions.csv")
    test_preds_df.to_csv(test_pred_csv_path, index=False)
    print(f"Saved: {test_pred_csv_path}")

    # Threshold shift analysis
    shift_csv_path = os.path.join(artifacts_dir, "threshold_shift_analysis.csv")
    shift_df.to_csv(shift_csv_path, index=False)
    print(f"Saved: {shift_csv_path}")

    # Binning details
    bin_csv_path = os.path.join(artifacts_dir, "calibration_bins_detail.csv")
    all_bins_df.to_csv(bin_csv_path, index=False)
    print(f"Saved: {bin_csv_path}")

    # Metrics JSON
    metrics_json_path = os.path.join(artifacts_dir, "calibration_metrics.json")
    metrics_dict = {
        "train_oof_cv": train_summary_df.to_dict(orient="records"),
        "holdout_test": holdout_summary_df.to_dict(orient="records"),
        "threshold_shift": shift_df.to_dict(orient="records"),
    }
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"Saved: {metrics_json_path}")

    # Metadata JSON
    metadata_json_path = os.path.join(artifacts_dir, "calibration_metadata.json")
    metadata_dict = {
        "disease": "Heart Disease (Cleveland)",
        "phase": "Phase 7: Probability Calibration",
        "random_state": 42,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "cv_folds_development": 5,
        "calibrator_cv_internal": 5,
        "calibrator_methods": ["Uncalibrated", "Sigmoid", "Isotonic"],
        "models_evaluated": [cfg["candidate_name"] for cfg in candidate_configs],
        "dataset_integrity": {
            "md5": md5_hash,
            "sha256": sha_hash,
            "file_size": file_size,
        },
        "phase6_thresholds_referenced": PHASE6_OPERATING_POINTS,
        "leakage_safeguards": [
            "Calibration fitting strictly isolated to training cohort",
            "Holdout test set untouched until descriptive evaluation",
            "Cross-fitting used inside CV folds to prevent target leakage",
            "Isotonic overfitting risk explicitly documented"
        ]
    }
    with open(metadata_json_path, "w", encoding="utf-8") as f:
        json.dump(metadata_dict, f, indent=2)
    print(f"Saved: {metadata_json_path}")

    # 10. Generate Publication-Quality Plots
    print("\n[Step 10] Generating reliability diagrams and distribution visualizations...")
    sns.set_theme(style="whitegrid")
    
    # Plot 1: Reliability Diagrams on Training CV (OOF)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
    methods = ["Uncalibrated", "Sigmoid", "Isotonic"]
    colors = {"Uncalibrated": "#7f8c8d", "Sigmoid": "#2980b9", "Isotonic": "#27ae60"}
    markers = {"Uncalibrated": "o", "Sigmoid": "s", "Isotonic": "^"}

    for idx, cfg in enumerate(candidate_configs):
        ax = axes[idx]
        c_name = cfg["candidate_name"]
        s_name = cfg["short_name"]
        
        # Perfect calibration diagonal
        ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration (Ideal)", alpha=0.7)
        
        for method in methods:
            probs = oof_probs_df[f"{c_name}__{method}"].values
            prob_true, prob_pred = calibration_curve(y_train.values, probs, n_bins=10, strategy="uniform")
            
            # Find Brier and ECE for legend
            row = train_summary_df[(train_summary_df["model"] == s_name) & (train_summary_df["calibration_method"] == method)].iloc[0]
            label = f"{method} (Brier={row['brier_score']:.3f}, ECE={row['ece']:.3f})"
            ax.plot(prob_pred, prob_true, marker=markers[method], color=colors[method], linewidth=2, label=label)
            
        ax.set_title(f"{s_name}\n(Training Cohort CV, N=242)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Mean Predicted Probability", fontsize=11)
        if idx == 0:
            ax.set_ylabel("Observed Positive Frequency", fontsize=11)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.legend(loc="upper left", fontsize=9, frameon=True)
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Heart Disease Phase 7 — Reliability Diagrams (Development CV, N=242)", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    cv_rel_path = os.path.join(artifacts_dir, "reliability_diagrams_train_cv.png")
    plt.savefig(cv_rel_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {cv_rel_path}")

    # Plot 2: Reliability Diagrams on Locked Hold-Out (Descriptive, N=61)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
    for idx, cfg in enumerate(candidate_configs):
        ax = axes[idx]
        c_name = cfg["candidate_name"]
        s_name = cfg["short_name"]
        
        ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration (Ideal)", alpha=0.7)
        
        for method in methods:
            probs = test_preds_df[f"{c_name}__{method}"].values
            prob_true, prob_pred = calibration_curve(y_test.values, probs, n_bins=5, strategy="uniform")
            
            row = holdout_summary_df[(holdout_summary_df["model"] == s_name) & (holdout_summary_df["calibration_method"] == method)].iloc[0]
            label = f"{method} (Brier={row['brier_score']:.3f}, LL={row['log_loss']:.3f})"
            ax.plot(prob_pred, prob_true, marker=markers[method], color=colors[method], linewidth=2, label=label)
            
        ax.set_title(f"{s_name}\n(Locked Hold-Out, N=61)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Mean Predicted Probability", fontsize=11)
        if idx == 0:
            ax.set_ylabel("Observed Positive Frequency", fontsize=11)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.legend(loc="upper left", fontsize=9, frameon=True)
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Heart Disease Phase 7 — Descriptive Reliability Diagrams (Hold-Out, N=61)", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    holdout_rel_path = os.path.join(artifacts_dir, "reliability_diagrams_holdout.png")
    plt.savefig(holdout_rel_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {holdout_rel_path}")

    # Plot 3: Probability Distribution Comparison (Hold-Out)
    fig, axes = plt.subplots(3, 3, figsize=(16, 12), sharex=True, sharey=True)
    for row_idx, cfg in enumerate(candidate_configs):
        c_name = cfg["candidate_name"]
        s_name = cfg["short_name"]
        
        for col_idx, method in enumerate(methods):
            ax = axes[row_idx, col_idx]
            probs = test_preds_df[f"{c_name}__{method}"].values
            
            # Histogram split by true class
            sns.histplot(
                probs[y_test.values == 0],
                bins=np.linspace(0, 1, 11),
                color="#3498db",
                label="Control (y=0)",
                alpha=0.6,
                ax=ax,
                stat="count"
            )
            sns.histplot(
                probs[y_test.values == 1],
                bins=np.linspace(0, 1, 11),
                color="#e74c3c",
                label="Heart Disease (y=1)",
                alpha=0.6,
                ax=ax,
                stat="count"
            )
            
            if row_idx == 0:
                ax.set_title(f"{method} Calibration", fontsize=12, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(f"{s_name}\nCount", fontsize=11, fontweight="bold")
            else:
                ax.set_ylabel("")
            if row_idx == 2:
                ax.set_xlabel("Predicted Probability", fontsize=11)
            ax.set_xlim([0, 1])
            ax.legend(loc="upper center", fontsize=9)
            ax.grid(True, linestyle=":", alpha=0.5)

    plt.suptitle("Heart Disease Phase 7 — Probability Distributions by True Class (Hold-Out, N=61)", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()
    dist_path = os.path.join(artifacts_dir, "probability_distribution_comparison.png")
    plt.savefig(dist_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {dist_path}")

    # 11. Final Dataset Integrity Verification
    print("\n[Step 11] Performing post-execution dataset integrity verification...")
    post_valid, post_md5, post_sha, post_size = verify_raw_dataset_integrity()
    assert post_valid
    assert post_md5 == md5_hash
    assert post_sha == sha_hash
    assert post_size == file_size
    print(f"Post-execution dataset integrity verified: Raw data remains 100% untouched.")

    print("\n" + "=" * 70)
    print("HEART DISEASE — PHASE 7 CALIBRATION ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
