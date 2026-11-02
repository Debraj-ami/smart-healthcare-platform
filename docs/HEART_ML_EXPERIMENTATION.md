# Heart Disease — Machine Learning Experimentation & Model Comparison (Phase 5)

**Document Version**: 1.0.0  
**Phase**: Phase 5 — Machine Learning Experimentation & Model Comparison  
**Parent Specification**: [`docs/HEART_PREPROCESSING.md`](file:///d:/smart-healthcare-platform/docs/HEART_PREPROCESSING.md)  
**Implementation Modules**:
- [`ml-service/app/heart_experimentation.py`](file:///d:/smart-healthcare-platform/ml-service/app/heart_experimentation.py)
- [`ml-service/run_heart_experiments.py`](file:///d:/smart-healthcare-platform/ml-service/run_heart_experiments.py)  
**Automated Test Suite**: [`ml-service/tests/test_heart_experimentation.py`](file:///d:/smart-healthcare-platform/ml-service/tests/test_heart_experimentation.py) (10/10 tests passing, 38/38 total project tests passing)  
**Source Dataset**: `datasets/raw/heart/processed.cleveland.data` (SHA-256: `a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8`)  
**Status**: Complete & Experimentally Validated  

---

## 1. Primary Objective & Experimental Philosophy

The primary objective of Phase 5 is to experimentally benchmark candidate machine learning algorithms for binary coronary artery disease (CAD) screening:
- $y = 0 \iff \text{num} = 0$ (Absence of significant CAD, $<50\%$ stenosis)
- $y = 1 \iff \text{num} \ge 1$ (Presence of significant CAD, $\ge 50\%$ stenosis in $\ge 1$ major vessel)

### Absolute Phase Boundaries
This phase constitutes a controlled **empirical screening benchmark**, NOT a clinical deployment or calibration phase:
- **No Clinical Deployment**: No production model bundles (`heart_model_bundle.joblib`) are generated.
- **No Clinical Threshold Selection**: The default threshold $0.5$ is utilized strictly as an evaluation convention for classification metrics requiring discrete labels. True operating thresholds are deferred to **Phase 6 (Threshold Analysis)**.
- **No Probability Calibration**: Raw predicted probabilities represent empirical screening scores rather than calibrated physiological risk. Systematic calibration is deferred to **Phase 7 (Probability Calibration)**.
- **No SHAP Explainability**: Complex game-theoretic attribution is deferred to **Phase 8 (SHAP Explainability)**. Model-native feature importance and coefficients are reported for scientific exploration only.
- **No System Integration**: No changes to Spring Boot, React, Chatbot, or Liver services.

---

## 2. Dataset & Cryptographic Verification

All experimentation utilizes strictly the single source dataset:
`datasets/raw/heart/processed.cleveland.data`

### Immutability Assurance
- **Observed File Size**: Exactly 18,461 bytes.
- **Pre-Experiment MD5**: `2d91a8ff69cfd9616aa47b59d6f843db`
- **Pre-Experiment SHA-256**: `a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8`
- **Post-Experiment Check**: Identical hashes re-verified post-execution. The raw file is strictly read-only.
- **Dimensions**: 303 patient records, 14 attributes.
- **Class Balance**: 164 Negative (54.13%) vs. 139 Positive (45.87%). Imbalance ratio: 1.18:1 (mild, no synthetic oversampling applied).

---

## 3. Data Partitioning & Cross-Validation Strategy

### 3.1 Locked Hold-Out Test Split
- Partitioning function: `split_heart_data(X, y, test_size=0.20, random_state=42)`
- **Training Partition ($X_{\text{train}}, y_{\text{train}}$)**: Exactly 242 observations (131 negative, 111 positive; 45.87% prevalence).
- **Hold-Out Test Partition ($X_{\text{test}}, y_{\text{test}}$)**: Exactly 61 observations (33 negative, 28 positive; 45.90% prevalence).
- **Isolation Policy**: The 61 test samples were completely locked during all CV training, hyperparameter exploration, and model selection. They were evaluated strictly **once** at the conclusion of experimentation.

### 3.2 25-Fold Repeated Stratified Cross-Validation
- Resampling: `RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)` $\rightarrow$ **25 validation folds**.
- **Fit-on-Train Discipline**: Preprocessing pipelines (`ColumnTransformer`) were instantiated and fitted strictly on the training fold of each split. Fold validation samples and hold-out test samples were strictly transformed using the fitted training statistics.
- Zero data leakage occurred across fold boundaries.

---

## 4. Candidate Model Configurations & Search Spaces

A total of **44 controlled configurations** were systematically evaluated across three model families:

### Family A: Logistic Regression (20 Configurations)
- **Preprocessors**:
  - `StandardScaler`: Median imputation $\rightarrow$ Z-score standardization.
  - `RobustScaler`: Median imputation $\rightarrow$ Interquartile range scaling (outlier-resilient).
  - Categorical variables: OneHotEncoder with `drop='first'`.
- **Penalties**: L1 (Lasso, feature selection) vs. L2 (Ridge).
- **Regularization Strengths**: $C \in \{0.01, 0.1, 1.0, 10.0, 100.0\}$.
- **Solver**: `liblinear` (exact coordinate descent for L1/L2 binary classification).

### Family B: Random Forest (12 Configurations)
- **Preprocessor**: `build_tree_preprocessor(drop=None)` (median numerical imputation without scaling; mode imputation for categoricals).
- **Estimator Count**: $n_{\text{estimators}} \in \{200, 500\}$.
- **Tree Depth**: $\text{max\_depth} \in \{\text{None}, 5, 10\}$.
- **Leaf Regularization**: $\text{min\_samples\_leaf} \in \{1, 2, 5\}$.
- **Class Weighting**: $\text{class\_weight} \in \{\text{None}, \text{'balanced'}\}$.

### Family C: XGBoost (12 Configurations)
- **Preprocessor**: `build_tree_preprocessor(drop=None)`.
- **Estimator Count**: $n_{\text{estimators}} \in \{100, 200, 300\}$.
- **Tree Depth**: $\text{max\_depth} \in \{2, 3, 4\}$ (shallow trees to prevent overfitting on 242 samples).
- **Learning Rate**: $\eta \in \{0.03, 0.05, 0.1\}$.
- **Subsampling**: $\text{subsample} \in \{0.8, 1.0\}$; $\text{colsample\_bytree} \in \{0.8, 1.0\}$.
- **Objective / Metric**: `binary:logistic`, `eval_metric='logloss'`.

---

## 5. Cross-Validation Results & Performance Comparison

All metrics are reported as **Mean $\pm$ Standard Deviation** across the 25 validation folds ($N=242$ training cohort):

| Model Family | Configuration ID | ROC-AUC (CV) | PR-AUC (CV) | Accuracy (CV) | Sensitivity (CV) | Specificity (CV) | F1-Score (CV) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | `LR_robust_l1_C1.0` | **0.9003 $\pm$ 0.0403** | 0.8916 $\pm$ 0.0491 | **0.8239 $\pm$ 0.0569** | 0.7851 $\pm$ 0.0714 | 0.8564 $\pm$ 0.0826 | **0.8038 $\pm$ 0.0622** |
| Logistic Regression | `LR_standard_l1_C1.0` | 0.8994 $\pm$ 0.0401 | 0.8914 $\pm$ 0.0487 | 0.8247 $\pm$ 0.0573 | 0.7842 $\pm$ 0.0709 | 0.8588 $\pm$ 0.0818 | 0.8032 $\pm$ 0.0625 |
| Logistic Regression | `LR_standard_l2_C1.0` | 0.8952 $\pm$ 0.0413 | 0.8865 $\pm$ 0.0503 | 0.8248 $\pm$ 0.0583 | 0.7851 $\pm$ 0.0722 | 0.8580 $\pm$ 0.0831 | 0.8036 $\pm$ 0.0637 |
| **Random Forest** | `RF_n200_dNone_leaf5_cwNone` | **0.8969 $\pm$ 0.0399** | **0.8961 $\pm$ 0.0394** | 0.8190 $\pm$ 0.0488 | 0.7706 $\pm$ 0.0826 | **0.8595 $\pm$ 0.0628** | 0.7948 $\pm$ 0.0586 |
| Random Forest | `RF_n200_d5_leaf5_cwNone` | 0.8949 $\pm$ 0.0409 | 0.8951 $\pm$ 0.0401 | 0.8190 $\pm$ 0.0488 | 0.7715 $\pm$ 0.0819 | 0.8587 $\pm$ 0.0631 | 0.7952 $\pm$ 0.0581 |
| Random Forest | `RF_n200_d5_leaf2_cwbalanced` | 0.8887 $\pm$ 0.0431 | 0.8864 $\pm$ 0.0425 | 0.8140 $\pm$ 0.0502 | 0.7932 $\pm$ 0.0792 | 0.8318 $\pm$ 0.0664 | 0.7963 $\pm$ 0.0594 |
| **XGBoost** | `XGB_n100_d2_lr0.05_sub0.8_col0.8`| **0.8869 $\pm$ 0.0420** | 0.8865 $\pm$ 0.0412 | 0.8107 $\pm$ 0.0519 | 0.7725 $\pm$ 0.0782 | 0.8426 $\pm$ 0.0784 | 0.7887 $\pm$ 0.0573 |
| XGBoost | `XGB_n100_d3_lr0.03_sub0.8_col0.8`| 0.8818 $\pm$ 0.0425 | 0.8845 $\pm$ 0.0421 | 0.8033 $\pm$ 0.0512 | 0.7518 $\pm$ 0.0811 | 0.8468 $\pm$ 0.0772 | 0.7749 $\pm$ 0.0584 |
| XGBoost | `XGB_n200_d2_lr0.05_sub0.8_col0.8`| 0.8816 $\pm$ 0.0442 | 0.8814 $\pm$ 0.0440 | 0.8066 $\pm$ 0.0526 | 0.7634 $\pm$ 0.0801 | 0.8431 $\pm$ 0.0795 | 0.7796 $\pm$ 0.0591 |

---

## 6. Multiple-Comparison & Statistical Discipline

1. **Overlap of Confidence Intervals**: The standard deviations across the 25 validation folds ($\pm 0.040$ to $\pm 0.044$ for ROC-AUC) exceed the metric differences between top models ($\Delta \text{ROC-AUC} \approx 0.013$). Therefore, no single model family can be declared strictly superior with statistical significance.
2. **Impact of Regularization**:
   - For Logistic Regression, $C=1.0$ with L1 penalty offered the best balance of sparsity and discrimination. Severe L1 penalty ($C=0.01$) shrank all coefficients to zero.
   - For Random Forest, constraining leaf size (`min_samples_leaf=5`) provided better variance control than unrestricted leaves (`min_samples_leaf=1`), reducing overfitting.
   - For XGBoost, shallow depth ($\text{max\_depth}=2$) consistently outperformed deeper trees ($\text{max\_depth}=4$), confirming that on a 242-sample dataset, depth-2 trees guard against high variance.

---

## 7. Final Candidate Selection & Out-Of-Fold Predictions

Three representative candidate models were selected for downstream research:
1. **Candidate 1 (Logistic Regression)**: `LR_robust_l1_C1.0` (Sparse, interpretable linear model with robust scaling).
2. **Candidate 2 (Random Forest)**: `RF_n200_dNone_leaf5_cwNone` (Non-linear bagging ensemble with leaf regularization).
3. **Candidate 3 (XGBoost)**: `XGB_n100_d2_lr0.05_sub0.8_col0.8` (Shallow boosting ensemble with feature and instance subsampling).

### Out-Of-Fold (OOF) Training Predictions
Out-of-fold predicted probabilities across the 5 repeats were saved to `artifacts/heart/experiments/oof_predictions.csv` (242 samples). These predictions are leak-free and will serve as the validation foundation for Phase 6 threshold curves and Phase 7 probability calibration curves.

---

## 8. Final Evaluation on Locked Hold-Out Test Set (N=61)

Each candidate pipeline was fit on all 242 training observations and evaluated **strictly once** on the untouched 61-sample test set (33 negative, 28 positive):

| Candidate Identifier | Architecture | Test ROC-AUC | Test PR-AUC | Test Accuracy | Test Sensitivity | Test Specificity | Test Precision | Test F1-Score | Confusion Matrix (TN / FP / FN / TP) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Candidate 1** | Logistic Regression (Robust L1) | 0.9513 | 0.9274 | 0.8689 | 0.8929 (25/28) | 0.8485 (28/33) | 0.8333 (25/30) | 0.8621 | TN=28, FP=5, FN=3, TP=25 |
| **Candidate 2** | Random Forest ($n=200, \text{leaf}=5$) | 0.9513 | 0.9401 | 0.8852 | 0.9286 (26/28) | 0.8485 (28/33) | 0.8387 (26/31) | 0.8814 | TN=28, FP=5, FN=2, TP=26 |
| **Candidate 3** | XGBoost ($n=100, d=2, \eta=0.05$) | **0.9589** | **0.9543** | **0.9016** | **0.9286 (26/28)** | **0.8788 (29/33)** | **0.8667 (26/30)** | **0.8966** | **TN=29, FP=4, FN=2, TP=26** |

*Note*: Test performance was evaluated under the default cutoff of 0.5 solely as a baseline reference point. Clinical threshold determination to maximize sensitivity and eliminate false negatives will be performed in Phase 6.

---

## 9. Model-Native Feature Importance Analysis

Feature importances were extracted by mapping pipeline output columns to clinical feature names using `get_heart_feature_names()`:

### 9.1 Logistic Regression Coefficients (L1 Penalty)
- **Strongest Positive Associations with CAD Risk**:
  1. `thal_7.0` (Reversible defect on thallium scan): $+1.102$
  2. `ca` (Number of fluoroscopy major vessels colored): $+1.077$
  3. `cp_4.0` (Asymptomatic chest pain presentation): $+0.763$
  4. `sex` (Male gender): $+0.748$
  5. `oldpeak` (ST depression induced by exercise): $+0.542$
- **Protective Association**:
  - `thalach` (Maximum heart rate achieved): $-0.460$
- **Sparsity / Feature Elimination**: L1 regularization with $C=1.0$ effectively zeroed coefficients for `fbs` (fasting blood sugar $>120$ mg/dl), `restecg_1.0`, and `slope_2.0`.

### 9.2 Tree Ensemble Importances (Gini & Split Gain)
- **Random Forest Top Features**: `ca` (0.163), `thal_7.0` (0.144), `oldpeak` (0.134), `thalach` (0.124), `cp_4.0` (0.098).
- **XGBoost Top Features**: `thal_7.0` (0.244), `ca` (0.187), `cp_4.0` (0.160), `exang` (0.091), `oldpeak` (0.063).

*Methodological Guardrail*: These coefficients and importances reflect empirical associations within this single retrospective cohort and must **never** be interpreted as biological or medical causation.

---

## 10. Generated Experimentation Artifacts

All experiment artifacts are structured and saved within `ml-service/artifacts/heart/experiments/`:

1. `cv_results_all.csv`: Full metric records for all 44 configurations evaluated across 25 validation folds.
2. `candidate_summary.csv`: Comprehensive side-by-side table of CV metrics and hold-out test results.
3. `ml-service/evaluation/heart/model_comparison.csv`: Module comparison file matching platform standards.
4. `oof_predictions.csv`: 242-sample out-of-fold predictions for all three candidate finalists.
5. `test_predictions.csv`: 61-sample final test set predictions and class labels.
6. `final_holdout_metrics.json`: Detailed JSON representation of hold-out performance.
7. `feature_importance.csv`: Complete mapped feature importances and regression coefficients.
8. `experiment_metadata.json`: Environmental metadata, checksums, and execution parameters.
9. `confusion_matrices.png`: Side-by-side confusion matrix heatmaps for all finalists.
10. `roc_curves.png`: High-resolution ROC curves on the hold-out test set.
11. `precision_recall_curves.png`: High-resolution PR curves with prevalence baseline.
12. `cv_comparison_plot.png`: Cross-validation bar plot with fold-to-fold error bars.
13. `feature_importance_plot.png`: Top-10 feature plots across all three architectures.

---

## 11. Study Limitations & Clinical Guardrails

1. **Cohort Sample Size**: Total cohort comprises 303 observations. Cross-validation standard deviations are $\pm 0.04$, meaning minor performance differences between models must not be overinterpreted.
2. **Single-Center Retrospective Bias**: Sourced exclusively from the Cleveland Clinic Foundation (1984). Referral bias (patients referred for cardiac catheterization) is present.
3. **Absence of Probability Calibration**: Uncalibrated probabilities reflect raw model outputs, not true physiological event probabilities. Phase 7 will evaluate Brier scores and reliability diagrams.
4. **Clinical Threshold Requirement**: The standard 0.5 threshold yields 2 to 3 false negatives on the test set. In cardiovascular screening, missing an active CAD patient carries significant risk. Phase 6 must determine an operating point achieving $\ge 90\%$ sensitivity.
5. **No Production Deployment**: These models are research prototypes. No bundle has been created or deployed.

---

## 12. Handoff to Phase 6 (Threshold Analysis)

Phase 5 experimentation has concluded successfully with all artifacts and benchmarks generated.

**Handoff Deliverables to Phase 6**:
- Primary Candidates: `LR_robust_l1_C1.0`, `RF_n200_dNone_leaf5_cwNone`, `XGB_n100_d2_lr0.05_sub0.8_col0.8`.
- Out-of-Fold Dataset: `ml-service/artifacts/heart/experiments/oof_predictions.csv` (242 training observations).
- Objectives for Phase 6:
  - Generate full threshold scanning curves (thresholds from 0.01 to 0.99).
  - Target screening operating point with Sensitivity $\ge 90\%$.
  - Quantify trade-offs in Specificity and False Positive burden.
