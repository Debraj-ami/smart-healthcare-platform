"""
Heart Disease Threshold Analysis & Operating-Point Evaluation Runner (Phase 6).

Executes threshold exploration across the [0.01, 0.99] spectrum for the 3 Phase 5 finalists:
1. Logistic Regression (Robust L1)
2. Random Forest (n=200, leaf=5)
3. XGBoost (n=100, depth=2, lr=0.05)

Boundaries:
- Strict fit-on-train / OOF threshold selection discipline.
- No model retraining.
- No probability calibration.
- No SHAP explainability.
- No clinical claims or production deployment.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt
import seaborn as sns

from app.heart_preprocessing import verify_raw_dataset_integrity
from app.heart_threshold_analysis import (
    MODEL_COLUMNS,
    DEFAULT_THRESHOLDS,
    validate_prediction_artifacts,
    calculate_threshold_metrics,
    analyze_threshold_grid,
    find_high_sensitivity_regions,
    find_balanced_operating_point,
    summarize_operating_points,
    verify_default_threshold_results,
)


def main():
    print("=" * 75)
    print("HEART DISEASE — PHASE 6: THRESHOLD ANALYSIS & OPERATING-POINT EVALUATION")
    print("=" * 75)

    # 1. Pre-experiment raw dataset integrity verification
    print("\n[Step 1] Verifying raw dataset integrity...")
    valid, md5_pre, sha256_pre, size_pre = verify_raw_dataset_integrity()
    assert valid, f"Raw dataset corrupted! MD5: {md5_pre}, SHA-256: {sha256_pre}"
    print(f"Raw dataset verified: 18,461 bytes | MD5: {md5_pre} | SHA-256: {sha256_pre}")

    # 2. Prediction artifact validation
    print("\n[Step 2] Validating Phase 5 prediction artifacts...")
    oof_path = os.path.join("artifacts", "heart", "experiments", "oof_predictions.csv")
    test_path = os.path.join("artifacts", "heart", "experiments", "test_predictions.csv")

    df_oof, df_test = validate_prediction_artifacts(oof_path, test_path)
    print(f"OOF predictions verified:  {len(df_oof)} rows (131 negative, 111 positive)")
    print(f"Test predictions verified: {len(df_test)} rows (33 negative, 28 positive)")

    # 3. Verify Default Threshold 0.50 Reproduction
    print("\n[Step 3] Verifying default threshold 0.50 reproducibility against Phase 5...")
    def_results = verify_default_threshold_results(df_test)
    for model_name, is_match in def_results.items():
        print(f"  {model_name}: 100% Phase 5 reproduction -> {is_match}")

    # 4. Threshold Grid Analysis on OOF Data
    print("\n[Step 4] Running 99-step threshold scan [0.01, 0.99] on OOF training data...")
    oof_grids = []
    test_grids = []
    high_sens_records = []
    summary_dfs = []

    y_oof = df_oof["true_label"].values
    y_test = df_test["true_label"].values

    selected_thresholds = {}

    for key, info in MODEL_COLUMNS.items():
        name = info["name"]
        oof_col = info["oof_col"]
        test_col = info["test_prob_col"]

        # Grid evaluation
        oof_grid = analyze_threshold_grid(y_oof, df_oof[oof_col].values, model_name=name, cohort_type="OOF")
        test_grid = analyze_threshold_grid(y_test, df_test[test_col].values, model_name=name, cohort_type="Hold-Out Test")

        oof_grids.append(oof_grid)
        test_grids.append(test_grid)

        # High-Sensitivity operating region (OOF Sensitivity >= 0.90)
        high_sens_df = find_high_sensitivity_regions(oof_grid, min_sensitivity=0.90)
        if len(high_sens_df) > 0:
            # Select upper boundary of high-sensitivity region (highest threshold maintaining >=90% sensitivity)
            cand_hs = high_sens_df.iloc[-1]
            t_hs = cand_hs["threshold"]
            print(f"\n[{name}] High-Sensitivity Region (Sens >= 0.90):")
            print(f"  Threshold span: [{high_sens_df['threshold'].min():.2f}, {t_hs:.2f}]")
            print(f"  Candidate Operating Point (t={t_hs:.2f}):")
            print(f"    OOF Sensitivity: {cand_hs['sensitivity']:.4f} ({int(cand_hs['tp'])}/111, FN={int(cand_hs['fn'])})")
            print(f"    OOF Specificity: {cand_hs['specificity']:.4f} ({int(cand_hs['tn'])}/131, FP={int(cand_hs['fp'])})")
            print(f"    OOF Precision:   {cand_hs['precision']:.4f}")
            print(f"    OOF NPV:         {cand_hs['npv']:.4f}")
            print(f"    OOF F1-Score:    {cand_hs['f1']:.4f}")
        else:
            print(f"\n[{name}] No threshold achieved >=0.90 sensitivity.")
            t_hs = 0.50

        # Balanced operating point (Max Youden's J on OOF)
        cand_bal = find_balanced_operating_point(oof_grid)
        t_bal = cand_bal["threshold"]
        print(f"  Balanced Operating Point (Max Youden's J, t={t_bal:.2f}):")
        print(f"    OOF Sensitivity: {cand_bal['sensitivity']:.4f}")
        print(f"    OOF Specificity: {cand_bal['specificity']:.4f}")
        print(f"    OOF Youden J:    {cand_bal['youden_j']:.4f}")
        print(f"    OOF Accuracy:    {cand_bal['accuracy']:.4f}")
        print(f"    OOF F1-Score:    {cand_bal['f1']:.4f}")

        selected_thresholds[name] = {"high_sens": t_hs, "balanced": t_bal}

        # Collect high-sensitivity candidates
        for _, row in high_sens_df.iterrows():
            rec = row.to_dict()
            rec["is_candidate_upper_bound"] = (row["threshold"] == t_hs)
            high_sens_records.append(rec)

        # Operating points comparison
        model_summary = summarize_operating_points(
            oof_grid_df=oof_grid,
            test_grid_df=test_grid,
            model_name=name,
            high_sens_threshold=t_hs,
            balanced_threshold=t_bal,
        )
        summary_dfs.append(model_summary)

    df_oof_all = pd.concat(oof_grids, ignore_index=True)
    df_test_all = pd.concat(test_grids, ignore_index=True)
    df_high_sens = pd.DataFrame(high_sens_records)
    df_summary = pd.concat(summary_dfs, ignore_index=True)

    # 5. Descriptive Hold-Out Test Evaluation
    print("\n[Step 5] Descriptive Hold-Out Test Set Evaluation of Candidate Operating Points:")
    print(df_summary[["model_name", "operating_point", "threshold", "test_sensitivity", "test_specificity", "test_precision", "test_npv", "test_f1", "test_fn", "test_fp"]].to_string())

    # 6. Save Structured Artifacts
    artifact_dir = os.path.join("artifacts", "heart", "threshold")
    eval_dir = os.path.join("evaluation", "heart")
    os.makedirs(artifact_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    print(f"\n[Step 6] Saving threshold artifacts to {artifact_dir}...")
    oof_csv_path = os.path.join(artifact_dir, "threshold_metrics_oof.csv")
    df_oof_all.to_csv(oof_csv_path, index=False)
    print(f"Saved: {oof_csv_path}")

    test_csv_path = os.path.join(artifact_dir, "threshold_metrics_test.csv")
    df_test_all.to_csv(test_csv_path, index=False)
    print(f"Saved: {test_csv_path}")

    high_sens_path = os.path.join(artifact_dir, "high_sensitivity_candidates.csv")
    df_high_sens.to_csv(high_sens_path, index=False)
    print(f"Saved: {high_sens_path}")

    summary_path = os.path.join(artifact_dir, "threshold_summary.csv")
    df_summary.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")

    # Copy to evaluation/heart/ (matching platform conventions)
    eval_summary_path = os.path.join(eval_dir, "heart_threshold_analysis.csv")
    df_summary.to_csv(eval_summary_path, index=False)
    print(f"Saved: {eval_summary_path}")

    # Default threshold comparison artifact
    def_comp_df = df_summary[df_summary["operating_point"] == "Default (Reference)"].copy()
    def_comp_path = os.path.join(artifact_dir, "default_threshold_comparison.csv")
    def_comp_df.to_csv(def_comp_path, index=False)
    print(f"Saved: {def_comp_path}")

    # Metadata JSON
    metadata = {
        "dataset": "processed.cleveland.data",
        "n_train_oof": len(df_oof),
        "n_test_holdout": len(df_test),
        "threshold_range": "[0.01, 0.99]",
        "num_thresholds": len(DEFAULT_THRESHOLDS),
        "models_analyzed": [info["name"] for info in MODEL_COLUMNS.values()],
        "selected_operating_points": selected_thresholds,
        "default_0.50_reproduction": def_results,
        "md5_checksum": md5_pre,
        "sha256_checksum": sha256_pre,
    }
    meta_path = os.path.join(artifact_dir, "threshold_analysis_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved: {meta_path}")

    # 7. Visualization Generation
    print("\n[Step 7] Generating threshold trade-off visualizations...")
    sns.set_theme(style="whitegrid", font_scale=1.05)
    colors = {"Logistic Regression (Robust L1)": "#1f77b4", "Random Forest (n=200, leaf=5)": "#2ca02c", "XGBoost (n=100, depth=2, lr=0.05)": "#ff7f0e"}

    # Plot 1: Sensitivity and Specificity vs Threshold
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for i, (key, info) in enumerate(MODEL_COLUMNS.items()):
        name = info["name"]
        sub = df_oof_all[df_oof_all["model_name"] == name]
        axes[i].plot(sub["threshold"], sub["sensitivity"], label="Sensitivity", color="#d62728", lw=2.2)
        axes[i].plot(sub["threshold"], sub["specificity"], label="Specificity", color="#1f77b4", lw=2.2)
        axes[i].axhline(0.90, color="gray", linestyle="--", alpha=0.7, label="Sensitivity = 90% Target")
        axes[i].axvline(selected_thresholds[name]["high_sens"], color="green", linestyle=":", lw=2.0, label=f"High-Sens (t={selected_thresholds[name]['high_sens']:.2f})")
        axes[i].axvline(0.50, color="black", linestyle="--", alpha=0.5, label="Default (t=0.50)")
        axes[i].set_title(f"{name}\n(OOF Training Data, N=242)")
        axes[i].set_xlabel("Decision Threshold (t)")
        axes[i].set_xlim([0.0, 1.0])
        axes[i].set_ylim([0.0, 1.02])
        if i == 0:
            axes[i].set_ylabel("Metric Value")
            axes[i].legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    p1 = os.path.join(artifact_dir, "sensitivity_specificity_vs_threshold.png")
    plt.savefig(p1, dpi=300)
    plt.close()
    print(f"Saved: {p1}")

    # Plot 2: Precision and NPV vs Threshold
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for i, (key, info) in enumerate(MODEL_COLUMNS.items()):
        name = info["name"]
        sub = df_oof_all[df_oof_all["model_name"] == name]
        axes[i].plot(sub["threshold"], sub["precision"], label="Precision (PPV)", color="#9467bd", lw=2.2)
        axes[i].plot(sub["threshold"], sub["npv"], label="NPV", color="#8c564b", lw=2.2)
        axes[i].axhline(111 / 242, color="gray", linestyle="--", alpha=0.7, label=f"Prevalence Baseline ({111/242:.2f})")
        axes[i].axvline(selected_thresholds[name]["high_sens"], color="green", linestyle=":", lw=2.0, label=f"High-Sens (t={selected_thresholds[name]['high_sens']:.2f})")
        axes[i].set_title(f"{name}\n(OOF Training Data, N=242)")
        axes[i].set_xlabel("Decision Threshold (t)")
        axes[i].set_xlim([0.0, 1.0])
        axes[i].set_ylim([0.2, 1.02])
        if i == 0:
            axes[i].set_ylabel("Metric Value")
            axes[i].legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    p2 = os.path.join(artifact_dir, "precision_npv_vs_threshold.png")
    plt.savefig(p2, dpi=300)
    plt.close()
    print(f"Saved: {p2}")

    # Plot 3: F1 and Accuracy vs Threshold
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for i, (key, info) in enumerate(MODEL_COLUMNS.items()):
        name = info["name"]
        sub = df_oof_all[df_oof_all["model_name"] == name]
        axes[i].plot(sub["threshold"], sub["f1"], label="F1-Score", color="#2ca02c", lw=2.2)
        axes[i].plot(sub["threshold"], sub["accuracy"], label="Accuracy", color="#17becf", lw=2.2)
        axes[i].axvline(selected_thresholds[name]["balanced"], color="blue", linestyle=":", lw=2.0, label=f"Balanced (t={selected_thresholds[name]['balanced']:.2f})")
        axes[i].axvline(0.50, color="black", linestyle="--", alpha=0.5, label="Default (t=0.50)")
        axes[i].set_title(f"{name}\n(OOF Training Data, N=242)")
        axes[i].set_xlabel("Decision Threshold (t)")
        axes[i].set_xlim([0.0, 1.0])
        axes[i].set_ylim([0.3, 0.95])
        if i == 0:
            axes[i].set_ylabel("Metric Value")
            axes[i].legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    p3 = os.path.join(artifact_dir, "f1_accuracy_vs_threshold.png")
    plt.savefig(p3, dpi=300)
    plt.close()
    print(f"Saved: {p3}")

    # Plot 4: False Positive and False Negative Counts vs Threshold
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, (key, info) in enumerate(MODEL_COLUMNS.items()):
        name = info["name"]
        sub = df_oof_all[df_oof_all["model_name"] == name]
        axes[i].plot(sub["threshold"], sub["fn"], label="False Negatives (Missed CAD)", color="#d62728", lw=2.4)
        axes[i].plot(sub["threshold"], sub["fp"], label="False Positives (False Alarms)", color="#ff7f0e", lw=2.4)
        axes[i].axvline(selected_thresholds[name]["high_sens"], color="green", linestyle=":", lw=2.0, label=f"High-Sens (t={selected_thresholds[name]['high_sens']:.2f})")
        axes[i].axvline(0.50, color="black", linestyle="--", alpha=0.5, label="Default (t=0.50)")
        axes[i].set_title(f"{name}\nError Counts (OOF N=242)")
        axes[i].set_xlabel("Decision Threshold (t)")
        axes[i].set_xlim([0.0, 1.0])
        axes[i].set_ylabel("Patient Count")
        if i == 0:
            axes[i].legend(loc="upper center", fontsize=9)
    plt.tight_layout()
    p4 = os.path.join(artifact_dir, "false_positive_negative_vs_threshold.png")
    plt.savefig(p4, dpi=300)
    plt.close()
    print(f"Saved: {p4}")

    # Plot 5: Youden's J Statistic vs Threshold
    plt.figure(figsize=(9, 6))
    for key, info in MODEL_COLUMNS.items():
        name = info["name"]
        sub = df_oof_all[df_oof_all["model_name"] == name]
        t_bal = selected_thresholds[name]["balanced"]
        y_max = sub[np.isclose(sub["threshold"], t_bal, atol=1e-3)]["youden_j"].values[0]
        plt.plot(sub["threshold"], sub["youden_j"], label=f"{name} (Peak={y_max:.3f} at t={t_bal:.2f})", lw=2.2, color=colors[name])
        plt.scatter([t_bal], [y_max], color=colors[name], s=60, zorder=5)
    plt.axvline(0.50, color="black", linestyle="--", alpha=0.5, label="Default Cutoff (t=0.50)")
    plt.xlabel("Decision Threshold (t)")
    plt.ylabel("Youden's J Statistic (Sensitivity + Specificity - 1)")
    plt.title("Youden's J Statistic across Threshold Spectrum\n(OOF Training Data, N=242)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 0.75])
    plt.legend(loc="lower center", fontsize=9.5)
    plt.tight_layout()
    p5 = os.path.join(artifact_dir, "youden_j_vs_threshold.png")
    plt.savefig(p5, dpi=300)
    plt.close()
    print(f"Saved: {p5}")

    # 8. Post-experiment raw dataset integrity verification
    print("\n[Step 8] Verifying post-experiment raw dataset integrity...")
    valid_post, md5_post, sha256_post, size_post = verify_raw_dataset_integrity()
    assert valid_post, f"Raw dataset corrupted post-experiment! MD5: {md5_post}"
    assert md5_post == md5_pre and sha256_post == sha256_pre, "Checksum mismatch post-experiment!"
    print(f"Raw dataset verified unchanged: MD5 {md5_post} | SHA-256 {sha256_post}")

    print("\n" + "=" * 75)
    print("PHASE 6 THRESHOLD ANALYSIS COMPLETED SUCCESSFULLY")
    print("=" * 75)


if __name__ == "__main__":
    main()
