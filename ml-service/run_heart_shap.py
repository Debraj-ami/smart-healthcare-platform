"""
Heart Disease SHAP Explainability Runner (Phase 8).

Executes reproducible, leakage-free SHAP explainability analysis for the
Cleveland Heart Disease finalist models:
1. Validates raw dataset cryptographic integrity (MD5, SHA-256).
2. Loads modeling dataset and performs stratified 80/20 train/test split.
3. Fits preprocessor and finalized models on the 242-sample training cohort.
4. Initializes model-specific SHAP explainers (LinearExplainer for LR, TreeExplainer for RF/XGB).
5. Verifies exact mathematical additivity / efficiency on test observations.
6. Computes global feature importance rankings and feature attribution distributions.
7. Generates local explanation attributions for representative holdout patients.
8. Preserves explicit mappings between raw input clinical variables and transformed features.
9. Exports structured CSV, JSON, and publication-quality visualization artifacts.
10. Confirms post-execution dataset integrity.
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
import shap

from app.heart_preprocessing import (
    verify_raw_dataset_integrity,
    load_heart_raw_data,
    create_modeling_dataset,
    split_heart_data,
)
from app.heart_calibration import get_heart_finalist_configs
from app.heart_shap import (
    get_heart_feature_mapping,
    fit_model_and_preprocessor_for_shap,
    create_heart_shap_explainer,
    compute_shap_explanations,
    compute_global_feature_importance,
    explain_single_observation,
    generate_representative_local_explanations,
)


def main():
    print("=" * 70)
    print("HEART DISEASE — PHASE 8: SHAP EXPLAINABILITY ANALYSIS")
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
    print("\n[Step 2] Loading dataset and performing stratified 80/20 train/test split...")
    df_raw = load_heart_raw_data()
    X, y = create_modeling_dataset(df_raw)
    X_train, X_test, y_train, y_test = split_heart_data(X, y, test_size=0.20, random_state=42)
    print(f"Training cohort: {len(X_train)} samples ({y_train.sum()} positive, {(y_train == 0).sum()} negative)")
    print(f"Holdout cohort:  {len(X_test)} samples ({y_test.sum()} positive, {(y_test == 0).sum()} negative)")

    # 3. Model Configurations
    candidate_configs = get_heart_finalist_configs()
    primary_cfg = candidate_configs[0]  # Logistic Regression (Robust L1)
    print(f"\n[Step 3] Primary Finalist Architecture: {primary_cfg['candidate_name']}")
    print(f"  - Model Family:     {primary_cfg['family']}")
    print(f"  - Config ID:        {primary_cfg['config_id']}")
    print(f"  - Preprocessing:    RobustScaler(drop='first')")
    print(f"  - SHAP Explainer:   shap.LinearExplainer (log-odds space)")
    print(f"  - Calibration:      Uncalibrated baseline (natively calibrated in Phase 7: ECE=0.0383, Slope=1.0704)")

    # 4. Fit Estimators and Preprocessors
    print("\n[Step 4] Fitting primary model and computing SHAP values via LinearExplainer...")
    est_lr, prep_lr, X_tr_lr, feat_names_lr = fit_model_and_preprocessor_for_shap(
        primary_cfg, X_train, y_train
    )
    explainer_lr = create_heart_shap_explainer(est_lr, X_tr_lr, model_family="LogisticRegression")
    
    rf_cfg = candidate_configs[1]
    est_rf, prep_rf, X_tr_rf, feat_names_rf = fit_model_and_preprocessor_for_shap(
        rf_cfg, X_train, y_train
    )
    explainer_rf = create_heart_shap_explainer(est_rf, X_tr_rf, model_family="RandomForest")

    xgb_cfg = candidate_configs[2]
    est_xgb, prep_xgb, X_tr_xgb, feat_names_xgb = fit_model_and_preprocessor_for_shap(
        xgb_cfg, X_train, y_train
    )
    explainer_xgb = create_heart_shap_explainer(est_xgb, X_tr_xgb, model_family="XGBoost")

    # Feature Mapping Setup with fitted preprocessors
    mapping_lr = get_heart_feature_mapping("LogisticRegression", preprocessor=prep_lr)
    mapping_tree = get_heart_feature_mapping("RandomForest", preprocessor=prep_rf)
    print(f"Extracted {len(mapping_lr)} features for Logistic Regression and {len(mapping_tree)} features for Tree models.")
    
    # Compute on both train and test cohorts
    exp_lr_train = compute_shap_explanations(explainer_lr, prep_lr, est_lr, X_train, "LogisticRegression")
    exp_lr_test = compute_shap_explanations(explainer_lr, prep_lr, est_lr, X_test, "LogisticRegression")

    # Additivity Verification
    lr_reconstructed = exp_lr_test["base_values"] + exp_lr_test["shap_values"].sum(axis=1)
    lr_discrepancy = np.max(np.abs(lr_reconstructed - exp_lr_test["model_outputs"]))
    print(f"LinearExplainer Additivity Check (Log-Odds Space): Max Discrepancy = {lr_discrepancy:.2e}")
    assert lr_discrepancy < 1e-10, "LinearExplainer additivity check failed!"

    # 6. Compute SHAP for Tree Finalists (Random Forest & XGBoost)
    print("\n[Step 6] Computing SHAP values for Random Forest and XGBoost finalists...")
    exp_rf_test = compute_shap_explanations(explainer_rf, prep_rf, est_rf, X_test, "RandomForest")
    rf_reconstructed = exp_rf_test["base_values"] + exp_rf_test["shap_values"].sum(axis=1)
    rf_discrepancy = np.max(np.abs(rf_reconstructed - exp_rf_test["model_outputs"]))
    print(f"RandomForest TreeExplainer Additivity Check (Probability Space): Max Discrepancy = {rf_discrepancy:.2e}")
    assert rf_discrepancy < 1e-10, "RandomForest TreeExplainer additivity check failed!"

    exp_xgb_test = compute_shap_explanations(explainer_xgb, prep_xgb, est_xgb, X_test, "XGBoost")
    xgb_reconstructed = exp_xgb_test["base_values"] + exp_xgb_test["shap_values"].sum(axis=1)
    xgb_discrepancy = np.max(np.abs(xgb_reconstructed - exp_xgb_test["model_outputs"]))
    print(f"XGBoost TreeExplainer Additivity Check (Log-Odds Margin): Max Discrepancy = {xgb_discrepancy:.2e}")
    assert xgb_discrepancy < 1e-4, "XGBoost TreeExplainer additivity check failed!"

    # 7. Global Feature Importance Analysis
    print("\n[Step 7] Computing global feature importance rankings...")
    df_imp_lr = compute_global_feature_importance(
        exp_lr_test["shap_values"], feat_names_lr, mapping_lr
    )
    print("\n--- TOP 10 GLOBAL MODEL CONTRIBUTORS (LOGISTIC REGRESSION) ---")
    print(df_imp_lr[["global_rank", "display_name", "original_feature", "mean_abs_shap", "pct_importance"]].head(10).to_string(index=False))

    df_imp_rf = compute_global_feature_importance(
        exp_rf_test["shap_values"], feat_names_rf, mapping_tree
    )
    df_imp_xgb = compute_global_feature_importance(
        exp_xgb_test["shap_values"], feat_names_xgb, mapping_tree
    )

    # 8. Local Explanation Analysis on Representative Holdout Cases
    print("\n[Step 8] Generating local explanations for representative holdout patients...")
    df_local_lr = generate_representative_local_explanations(
        explainer_lr, prep_lr, est_lr, X_test, y_test, model_family="LogisticRegression"
    )
    print("\n--- REPRESENTATIVE LOCAL PATIENT EXPLANATIONS (HOLD-OUT TEST SET) ---")
    local_display_cols = ["case_type", "sample_index", "true_label", "model_estimated_probability", "top_positive_contributors", "top_negative_contributors"]
    print(df_local_lr[local_display_cols].to_string(index=False))

    # 9. Save Artifacts
    artifacts_dir = os.path.join("artifacts", "heart", "shap")
    os.makedirs(artifacts_dir, exist_ok=True)
    print(f"\n[Step 9] Saving structured artifacts to {artifacts_dir}...")

    # Save feature mapping
    mapping_csv_path = os.path.join(artifacts_dir, "shap_feature_mapping.csv")
    mapping_lr.to_csv(mapping_csv_path, index=False)
    print(f"Saved: {mapping_csv_path}")

    # Save global feature importance
    imp_csv_path = os.path.join(artifacts_dir, "shap_feature_importance.csv")
    df_imp_lr.to_csv(imp_csv_path, index=False)
    print(f"Saved: {imp_csv_path}")

    # Save multi-model global summary
    global_summary_rows = []
    for rank in range(1, 11):
        lr_row = df_imp_lr[df_imp_lr["global_rank"] == rank].iloc[0]
        rf_row = df_imp_rf[df_imp_rf["global_rank"] == rank].iloc[0]
        xgb_row = df_imp_xgb[df_imp_xgb["global_rank"] == rank].iloc[0]
        global_summary_rows.append({
            "rank": rank,
            "lr_feature": lr_row["display_name"],
            "lr_mean_abs_shap": lr_row["mean_abs_shap"],
            "rf_feature": rf_row["display_name"],
            "rf_mean_abs_shap": rf_row["mean_abs_shap"],
            "xgb_feature": xgb_row["display_name"],
            "xgb_mean_abs_shap": xgb_row["mean_abs_shap"],
        })
    df_global_summary = pd.DataFrame(global_summary_rows)
    summary_csv_path = os.path.join(artifacts_dir, "shap_global_summary.csv")
    df_global_summary.to_csv(summary_csv_path, index=False)
    print(f"Saved: {summary_csv_path}")

    # Save local explanations
    local_csv_path = os.path.join(artifacts_dir, "shap_local_explanations.csv")
    df_local_lr.to_csv(local_csv_path, index=False)
    print(f"Saved: {local_csv_path}")

    # Save metadata JSON
    metadata_json_path = os.path.join(artifacts_dir, "shap_metadata.json")
    metadata_dict = {
        "disease": "Heart Disease (Cleveland)",
        "phase": "Phase 8: SHAP Explainability & Feature Attributions",
        "primary_model": {
            "name": primary_cfg["candidate_name"],
            "family": primary_cfg["family"],
            "config_id": primary_cfg["config_id"],
            "scaler": primary_cfg["scaler"],
            "penalty": primary_cfg["penalty"],
            "C": primary_cfg["C"],
            "solver": primary_cfg["solver"],
            "calibration": "Uncalibrated Baseline (Natively Calibrated in Phase 7)",
            "output_space": "log-odds (decision_function)",
            "shap_explainer": "shap.LinearExplainer",
            "background_masker": "shap.maskers.Independent(X_train_transformed)",
            "background_samples": len(X_train),
        },
        "comparative_models": [
            {
                "name": rf_cfg["candidate_name"],
                "family": rf_cfg["family"],
                "shap_explainer": "shap.TreeExplainer",
                "output_space": "probability (class 1)",
            },
            {
                "name": xgb_cfg["candidate_name"],
                "family": xgb_cfg["family"],
                "shap_explainer": "shap.TreeExplainer",
                "output_space": "log-odds-margin",
            }
        ],
        "dataset_integrity": {
            "md5": md5_hash,
            "sha256": sha_hash,
            "file_size": file_size,
        },
        "top_5_global_features_lr": df_imp_lr["display_name"].head(5).tolist(),
        "interpretability_safety_rules": [
            "SHAP explains model mathematical output, not biological causation",
            "Model-estimated probabilities do not constitute clinical diagnosis",
            "Individual contributions reflect relative feature weights on this cohort",
            "Transformed one-hot indicators retain explicit reference mapping"
        ]
    }
    with open(metadata_json_path, "w", encoding="utf-8") as f:
        json.dump(metadata_dict, f, indent=2)
    print(f"Saved: {metadata_json_path}")

    # 10. Generate Visualizations
    print("\n[Step 10] Generating publication-quality SHAP plots...")
    sns.set_theme(style="whitegrid")

    # Plot 1: SHAP Global Bar Plot (Primary Model - Logistic Regression)
    plt.figure(figsize=(10, 7))
    top_features = df_imp_lr.head(12)
    colors = sns.color_palette("Blues_r", len(top_features))
    bars = plt.barh(range(len(top_features)), top_features["mean_abs_shap"][::-1], color=colors[::-1], align="center")
    plt.yticks(range(len(top_features)), top_features["display_name"][::-1], fontsize=10)
    plt.xlabel("Mean |SHAP Value| (Average Impact on Model Log-Odds Output)", fontsize=11, fontweight="bold")
    plt.title("Heart Disease — Global Feature Importance (Logistic Regression L1)\nMean Absolute SHAP Value across Test Cohort (N=61)", fontsize=13, fontweight="bold")
    
    # Annotate bars with values
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.02, bar.get_y() + bar.get_height() / 2, f"{width:.3f}", ha="left", va="center", fontsize=9)
        
    plt.xlim(0, max(top_features["mean_abs_shap"]) * 1.15)
    plt.tight_layout()
    bar_plot_path = os.path.join(artifacts_dir, "shap_global_bar_plot.png")
    plt.savefig(bar_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {bar_plot_path}")

    # Plot 2: Model Comparison Feature Importance
    fig, axes = plt.subplots(1, 3, figsize=(20, 7), sharex=False)
    models_to_plot = [
        ("Logistic Regression (Robust L1)", df_imp_lr, "Blues_r", axes[0]),
        ("Random Forest (n=200, leaf=5)", df_imp_rf, "Greens_r", axes[1]),
        ("XGBoost (n=100, depth=2, lr=0.05)", df_imp_xgb, "Oranges_r", axes[2]),
    ]
    for m_name, df_m, pal, ax in models_to_plot:
        top_m = df_m.head(10)
        c_list = sns.color_palette(pal, len(top_m))
        ax.barh(range(len(top_m)), top_m["mean_abs_shap"][::-1], color=c_list[::-1], align="center")
        ax.set_yticks(range(len(top_m)))
        ax.set_yticklabels(top_m["display_name"][::-1], fontsize=10)
        ax.set_xlabel("Mean |SHAP Value|", fontsize=11, fontweight="bold")
        ax.set_title(m_name, fontsize=12, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Heart Disease Phase 8 — Global SHAP Feature Importance across Finalist Models", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    comp_plot_path = os.path.join(artifacts_dir, "shap_model_comparison_importance.png")
    plt.savefig(comp_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {comp_plot_path}")

    # Plot 3: SHAP Beeswarm Summary Plot for Logistic Regression
    fig, ax = plt.subplots(figsize=(11, 8))
    # Create Explanation object for shap.plots.beeswarm
    explanation_obj = shap.Explanation(
        values=exp_lr_test["shap_values"],
        base_values=exp_lr_test["base_values"],
        data=exp_lr_test["X_transformed"],
        feature_names=[mapping_lr[mapping_lr["transformed_feature"] == fn]["display_name"].values[0] for fn in feat_names_lr]
    )
    plt.sca(ax)
    shap.plots.beeswarm(explanation_obj, max_display=12, show=False)
    plt.title("Heart Disease — SHAP Beeswarm Summary Plot (Logistic Regression)\nDirectional Feature Impact on Model Log-Odds Output (Test Cohort, N=61)", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    beeswarm_path = os.path.join(artifacts_dir, "shap_beeswarm_summary_plot.png")
    plt.savefig(beeswarm_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {beeswarm_path}")

    # Plot 4: Local Waterfall Visualizations for Representative Cases
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # Case A: High-Confidence True Positive
    tp_sample_idx = int(df_local_lr[df_local_lr["case_type"] == "High-Confidence True Positive"]["sample_index"].iloc[0])
    test_loc_tp = list(X_test.index).index(tp_sample_idx)
    exp_tp = shap.Explanation(
        values=exp_lr_test["shap_values"][test_loc_tp],
        base_values=exp_lr_test["base_values"][test_loc_tp],
        data=exp_lr_test["X_transformed"][test_loc_tp],
        feature_names=[mapping_lr[mapping_lr["transformed_feature"] == fn]["display_name"].values[0] for fn in feat_names_lr]
    )
    plt.sca(axes[0])
    shap.plots.waterfall(exp_tp, max_display=8, show=False)
    axes[0].set_title(f"True Positive Case (Sample #{tp_sample_idx})\nTrue CAD, Estimated Prob = {df_local_lr[df_local_lr['case_type'] == 'High-Confidence True Positive']['model_estimated_probability'].iloc[0]:.2f}", fontsize=11, fontweight="bold")

    # Case B: High-Confidence True Negative
    tn_sample_idx = int(df_local_lr[df_local_lr["case_type"] == "High-Confidence True Negative"]["sample_index"].iloc[0])
    test_loc_tn = list(X_test.index).index(tn_sample_idx)
    exp_tn = shap.Explanation(
        values=exp_lr_test["shap_values"][test_loc_tn],
        base_values=exp_lr_test["base_values"][test_loc_tn],
        data=exp_lr_test["X_transformed"][test_loc_tn],
        feature_names=[mapping_lr[mapping_lr["transformed_feature"] == fn]["display_name"].values[0] for fn in feat_names_lr]
    )
    plt.sca(axes[1])
    shap.plots.waterfall(exp_tn, max_display=8, show=False)
    axes[1].set_title(f"True Negative Case (Sample #{tn_sample_idx})\nControl, Estimated Prob = {df_local_lr[df_local_lr['case_type'] == 'High-Confidence True Negative']['model_estimated_probability'].iloc[0]:.2f}", fontsize=11, fontweight="bold")

    plt.suptitle("Heart Disease Phase 8 — Local Patient Waterfall Attributions (Holdout Test Set)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    local_plot_path = os.path.join(artifacts_dir, "shap_local_waterfall_examples.png")
    plt.savefig(local_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {local_plot_path}")

    # 11. Final Dataset Integrity Verification
    print("\n[Step 11] Performing post-execution dataset integrity verification...")
    post_valid, post_md5, post_sha, post_size = verify_raw_dataset_integrity()
    assert post_valid
    assert post_md5 == md5_hash
    assert post_sha == sha_hash
    assert post_size == file_size
    print(f"Post-execution dataset integrity verified: Raw data remains 100% untouched.")

    print("\n" + "=" * 70)
    print("HEART DISEASE — PHASE 8 SHAP EXPLAINABILITY ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
