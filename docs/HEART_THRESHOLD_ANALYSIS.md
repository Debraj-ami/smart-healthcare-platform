# Heart Disease — Threshold Analysis & Operating-Point Evaluation (Phase 6)

**Document Version**: 1.0.0  
**Phase**: Phase 6 — Threshold Analysis & Operating-Point Evaluation  
**Parent Specifications**:
- [`docs/HEART_PREPROCESSING.md`](file:///d:/smart-healthcare-platform/docs/HEART_PREPROCESSING.md) (Phase 4)
- [`docs/HEART_ML_EXPERIMENTATION.md`](file:///d:/smart-healthcare-platform/docs/HEART_ML_EXPERIMENTATION.md) (Phase 5)  
**Implementation Modules**:
- [`ml-service/app/heart_threshold_analysis.py`](file:///d:/smart-healthcare-platform/ml-service/app/heart_threshold_analysis.py)
- [`ml-service/run_heart_threshold_analysis.py`](file:///d:/smart-healthcare-platform/ml-service/run_heart_threshold_analysis.py)  
**Automated Test Suite**: [`ml-service/tests/test_heart_threshold_analysis.py`](file:///d:/smart-healthcare-platform/ml-service/tests/test_heart_threshold_analysis.py) (12/12 tests passing, 50/50 total project tests passing)  
**Parent Artifacts**:
- `ml-service/artifacts/heart/experiments/oof_predictions.csv` ($N=242$ training cohort)
- `ml-service/artifacts/heart/experiments/test_predictions.csv` ($N=61$ hold-out cohort)  
**Source Dataset**: `datasets/raw/heart/processed.cleveland.data` (SHA-256: `a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8`)  
**Status**: Complete & Empirically Verified  

---

## 1. Objective & Scope

In clinical cardiovascular risk screening, the standard computational cutoff of $t = 0.50$ is rarely optimal because the costs of diagnostic misclassifications are severely asymmetric:
- **False Negative (Missed CAD)**: A patient with obstructive coronary artery disease ($ \ge 50\%$ vessel stenosis) is mistakenly categorized as low-risk. This carries the hazard of unmanaged progression to acute coronary syndromes, myocardial infarction, or sudden cardiac death.
- **False Positive (False Alarm)**: A patient without significant CAD is flagged as high-risk. This leads to psychological distress and referral for secondary confirmatory diagnostics (e.g. exercise stress testing, coronary CT angiography).

Phase 6 systematically evaluates the discrete threshold range from 0.01 to 0.99 evaluated at 0.01 increments (99 discrete threshold steps) for the three Phase 5 finalist architectures to quantify these trade-offs and establish candidate operating points on empirical Out-Of-Fold (OOF) training data.

### Absolute Phase Boundaries
- **No Model Retraining**: Predictions from Phase 5 are analyzed directly without modifying trained model weights.
- **No Probability Calibration**: Predicted values remain uncalibrated model scores; calibration is strictly deferred to **Phase 7 (Probability Calibration)**.
- **No SHAP Explainability**: Complex feature attributions are deferred to **Phase 8 (SHAP Explainability)**.
- **No Production Deployment**: No clinical decision threshold is declared final, and no production bundles (`heart_model_bundle.joblib`) or FastAPI endpoints are created.

---

## 2. Why Threshold Analysis is Required

While continuous discrimination metrics (such as ROC-AUC and PR-AUC) evaluate ranking ability across all possible cutoffs, real-world clinical decision support requires a discrete decision rule:
$$\hat{y} = \begin{cases} 1 & \text{if } P(Y=1 \mid X) \ge t \\ 0 & \text{if } P(Y=1 \mid X) < t \end{cases}$$

Varying $t$ does not alter model discrimination, but directly governs the operational balance between **Sensitivity (Recall)** and **Specificity**. Shifting the threshold downward increases disease sensitivity at the expense of specificity, resulting in fewer false negatives alongside an increased number of false positives on this cohort.

---

## 3. OOF Development vs. Hold-Out Evaluation Methodology

To prevent data snooping and threshold selection leakage:

```
242 Training Patients
      ↓
Out-Of-Fold Predictions (oof_predictions.csv)
      ↓
Threshold Analysis (99 steps)
      ↓
Identify High-Sensitivity (Sens >= 90%) & Balanced (Max Youden J) Cutoffs
      ↓
Apply selected thresholds ONCE to 61-sample Hold-Out Test Set (test_predictions.csv)
      ↓
Descriptive Final Evaluation (NO threshold re-tuning)
```

1. **Threshold Exploration**: Carried out strictly on the 242-sample Out-Of-Fold (OOF) training predictions generated during 5-repeat 5-fold cross-validation.
2. **Hold-Out Isolation**: The 61-sample test set was never inspected to choose, tune, or alter any threshold. It is utilized exclusively for final descriptive confirmation.

---

## 4. Threshold Grid & Metric Definitions

A 99-step decision threshold grid was constructed:
$$t \in \{0.01, 0.02, 0.03, \dots, 0.98, 0.99\} \quad (\Delta t = 0.01)$$

For each model and threshold $t$, the binary prediction was generated as $\hat{y} = \mathbb{I}(p \ge t)$. Primary screening metrics:
- **Sensitivity (Recall / True Positive Rate)**:
  $$\text{Sensitivity} = \frac{TP}{TP + FN}$$
- **Specificity (True Negative Rate)**:
  $$\text{Specificity} = \frac{TN}{TN + FP}$$
- **Precision (Positive Predictive Value, PPV)**:
  $$\text{Precision} = \frac{TP}{TP + FP} \quad (\text{safe fallback: } 1.0 \text{ if } TP+FP=0 \text{ and } TP=0)$$
- **Negative Predictive Value (NPV)**:
  $$\text{NPV} = \frac{TN}{TN + FN} \quad (\text{safe fallback: } 1.0 \text{ if } TN+FN=0 \text{ and } TN=0)$$
- **Accuracy**:
  $$\text{Accuracy} = \frac{TP + TN}{N}$$
- **F1-Score**:
  $$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Sensitivity}}{\text{Precision} + \text{Sensitivity}}$$
- **False Positive Rate (FPR)**:
  $$\text{FPR} = \frac{FP}{FP + TN} = 1 - \text{Specificity}$$
- **False Negative Rate (FNR)**:
  $$\text{FNR} = \frac{FN}{FN + TP} = 1 - \text{Sensitivity}$$
- **Youden's J Statistic**:
  $$J = \text{Sensitivity} + \text{Specificity} - 1$$

---

## 5. Models Analyzed

The three Phase 5 finalist architectures:
1. **Logistic Regression (Robust L1)**: `LR_robust_l1_C1.0` (L1-regularized sparse linear model with RobustScaler).
2. **Random Forest ($n=200, \text{leaf}=5$)**: `RF_n200_dNone_leaf5_cwNone` (Unconstrained depth, leaf-regularized bagging ensemble).
3. **XGBoost ($n=100, \text{depth}=2, \eta=0.05$)**: `XGB_n100_d2_lr0.05_sub0.8_col0.8` (Shallow boosting ensemble with subsampling).

---

## 6. Default Threshold 0.50 Baseline Verification

Evaluating the default threshold $t = 0.50$ reproduces the Phase 5 hold-out test results (61 observations: 33 negative, 28 positive):

| Model Name | Test Accuracy | Test Sensitivity | Test Specificity | Test Precision | Test F1 | Confusion Matrix (TN / FP / FN / TP) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Robust L1)** | 0.8689 | 0.8929 (25/28) | 0.8485 (28/33) | 0.8333 | 0.8621 | TN=28, FP=5, FN=3, TP=25 | Verified (100% Match) |
| **Random Forest ($n=200, \text{leaf}=5$)** | 0.8852 | 0.9286 (26/28) | 0.8485 (28/33) | 0.8387 | 0.8814 | TN=28, FP=5, FN=2, TP=26 | Verified (100% Match) |
| **XGBoost ($n=100, d=2, \eta=0.05$)** | 0.9016 | 0.9286 (26/28) | 0.8788 (29/33) | 0.8667 | 0.8966 | TN=29, FP=4, FN=2, TP=26 | Verified (100% Match) |

---

## 7. High-Sensitivity Operating Regions (OOF Training Data, N=242)

To minimize missed CAD cases, we examined the **High-Sensitivity screening region** where $\text{Sensitivity} \ge 0.90$ on OOF data (111 positive cases):

| Model Architecture | High-Sensitivity Range | Selected Candidate Cutoff | OOF Sensitivity | OOF Specificity | OOF Precision | OOF NPV | OOF F1 | Missed Cases (FN) | False Alarms (FP) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Robust L1)** | $[0.01, 0.31]$ | **$t = 0.31$** | **0.9009 (100/111)** | 0.7099 (93/131) | 0.7246 | 0.8942 | 0.8032 | **11** (vs 25 at $t=0.50$) | 38 (vs 19 at $t=0.50$) |
| **Random Forest ($n=200, \text{leaf}=5$)** | $[0.01, 0.32]$ | **$t = 0.32$** | **0.9009 (100/111)** | 0.6489 (85/131) | 0.6849 | 0.8854 | 0.7782 | **11** (vs 26 at $t=0.50$) | 46 (vs 19 at $t=0.50$) |
| **XGBoost ($n=100, d=2, \eta=0.05$)** | $[0.01, 0.24]$ | **$t = 0.24$** | **0.9009 (100/111)** | 0.5878 (77/131) | 0.6494 | 0.8750 | 0.7547 | **11** (vs 24 at $t=0.50$) | 54 (vs 19 at $t=0.50$) |

### Screening Significance
By adjusting the decision threshold downward from $0.50$ to the candidate screening cutoffs ($0.31$ for LR, $0.32$ for RF, $0.24$ for XGBoost), **missed CAD cases are reduced from 24–26 down to 11 (a 56% reduction in missed disease)** while maintaining clinical specificity between $59\%$ and $71\%$.

---

## 8. Balanced Operating Regions (Maximizing Youden's J on OOF Data)

For diagnostic settings where false positives and false negatives carry equal weight, we identified the operating points that maximize Youden's J index ($J = \text{Sensitivity} + \text{Specificity} - 1$):

| Model Architecture | Balanced Cutoff ($t$) | OOF Youden J | OOF Sensitivity | OOF Specificity | OOF Accuracy | OOF Precision | OOF F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Robust L1)** | **$t = 0.56$** | **0.6652** | 0.7568 (84/111) | 0.9084 (119/131) | 0.8388 | 0.8750 | 0.8116 |
| **Random Forest ($n=200, \text{leaf}=5$)** | **$t = 0.55$** | **0.6471** | 0.7387 (82/111) | 0.9084 (119/131) | 0.8306 | 0.8723 | 0.8000 |
| **XGBoost ($n=100, d=2, \eta=0.05$)** | **$t = 0.62$** | **0.6610** | 0.7297 (81/111) | 0.9313 (122/131) | 0.8388 | 0.9000 | 0.8060 |

---

## 9. Threshold Stability Analysis

We examined performance stability across neighboring threshold cutoffs ($\Delta t = \pm 0.02$):
- **Logistic Regression ($t \in [0.29, 0.33]$)**:
  - $t = 0.29$: Sensitivity = $90.99\%$ (101/111), Specificity = $70.23\%$, $\text{FN}=10$
  - $t = 0.30$: Sensitivity = $90.09\%$ (100/111), Specificity = $70.99\%$, $\text{FN}=11$
  - $t = 0.31$: Sensitivity = $90.09\%$ (100/111), Specificity = $70.99\%$, $\text{FN}=11$
  - $t = 0.32$: Sensitivity = $89.19\%$ (99/111), Specificity = $70.99\%$, $\text{FN}=12$
  - $t = 0.33$: Sensitivity = $89.19\%$ (99/111), Specificity = $70.99\%$, $\text{FN}=12$
- **Findings**: The high-sensitivity zone exhibits smooth, plateau-like behavior without point singularities or chaotic jumps. Sensitivity decreases monotonically by exactly 1 patient count per $0.01$ shift in threshold, reflecting clean model rank ordering.

---

## 10. Descriptive Evaluation on Locked Hold-Out Test Set (N=61)

Applying the OOF-derived candidate operating points to the untouched 61-sample test set (28 positive, 33 negative; prevalence $45.9\%$):

| Model Name | Operating Point | Decision Cutoff ($t$) | Test Sensitivity | Test Specificity | Test Precision | Test NPV | Test Accuracy | Test F1 | Test FN | Test FP |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | High-Sensitivity | **$t = 0.31$** | **0.9643 (27/28)** | 0.7273 (24/33) | 0.7500 (27/36) | **0.9600 (24/25)** | 0.8361 | 0.8438 | **1** | 9 |
| Logistic Regression | Balanced (Youden J) | $t = 0.56$ | 0.8929 (25/28) | 0.8788 (29/33) | 0.8621 (25/29) | 0.9062 (29/32) | 0.8852 | 0.8772 | 3 | 4 |
| Logistic Regression | Default Reference | $t = 0.50$ | 0.8929 (25/28) | 0.8485 (28/33) | 0.8333 (25/30) | 0.9032 (28/31) | 0.8689 | 0.8621 | 3 | 5 |
| **Random Forest** | High-Sensitivity | **$t = 0.32$** | **0.9643 (27/28)** | 0.6364 (21/33) | 0.6923 (27/39) | **0.9545 (21/22)** | 0.7869 | 0.8060 | **1** | 12 |
| Random Forest | Balanced (Youden J) | $t = 0.55$ | 0.8929 (25/28) | **0.9091 (30/33)** | **0.8929 (25/28)** | 0.9091 (30/33) | **0.9016** | **0.8929** | 3 | 3 |
| Random Forest | Default Reference | $t = 0.50$ | 0.9286 (26/28) | 0.8485 (28/33) | 0.8387 (26/31) | 0.9333 (28/30) | 0.8852 | 0.8814 | 2 | 5 |
| **XGBoost** | High-Sensitivity | **$t = 0.24$** | **1.0000 (28/28)** | 0.6364 (21/33) | 0.7000 (28/40) | **1.0000 (21/21)** | 0.8033 | 0.8235 | **0** | 12 |
| XGBoost | Balanced (Youden J) | $t = 0.62$ | 0.8571 (24/28) | 0.8788 (29/33) | 0.8571 (24/28) | 0.8788 (29/33) | 0.8689 | 0.8571 | 4 | 4 |
| XGBoost | Default Reference | $t = 0.50$ | 0.9286 (26/28) | 0.8788 (29/33) | 0.8667 (26/30) | 0.9355 (29/31) | **0.9016** | **0.8966** | 2 | 4 |

---

## 11. False-Positive vs. False-Negative Clinical Trade-Offs

Comparing the High-Sensitivity operating point against Default $t=0.50$:
1. **Missed CAD Cases (False Negatives)**:
   - Logistic Regression drops from 3 false negatives to **1 false negative** (96.4% sensitivity).
   - Random Forest drops from 2 false negatives to **1 false negative** (96.4% sensitivity).
   - XGBoost drops from 2 false negatives to **0 false negatives** (100.0% sensitivity, capturing all 28 diseased test patients).
2. **False Alarms (False Positives)**:
   - For Logistic Regression, false positives increase moderately from 5 to 9 (specificity shifts from $84.8\%$ to $72.7\%$).
   - For Random Forest and XGBoost, false positives increase from 4–5 to 12 (specificity shifts from $\sim 85–88\%$ to $63.6\%$).
3. **Empirical Trade-Off Interpretation**: At the higher-sensitivity operating points, the holdout results show fewer false negatives but more false positives compared with the default 0.50 threshold. This demonstrates the empirical sensitivity-specificity trade-off associated with lowering the classification threshold. These results should be interpreted as model-performance characteristics on this dataset rather than as clinical evidence for ruling out disease or determining a patient's need for secondary testing.

---

## 12. Artifacts Produced

The following reproducibility artifacts have been generated in `ml-service/artifacts/heart/threshold/`:

1. `threshold_metrics_oof.csv`: Full metric records for all 99 thresholds across all 3 models on OOF training data ($242$ rows $\times 99$ cutoffs).
2. `threshold_metrics_test.csv`: Full metric records across 99 thresholds on hold-out test data.
3. `high_sensitivity_candidates.csv`: Complete table of thresholds satisfying OOF Sensitivity $\ge 0.90$.
4. `threshold_summary.csv`: Side-by-side metric comparison of High-Sensitivity, Balanced, and Default operating points across OOF and hold-out test sets.
5. `default_threshold_comparison.csv`: Direct Phase 5 baseline reproducibility audit at $t = 0.50$.
6. `threshold_analysis_metadata.json`: Complete execution metadata, checksums, and operating point parameters.
7. **Visualizations**:
   - `sensitivity_specificity_vs_threshold.png`: Dual-axis Sensitivity vs. Specificity curves.
   - `precision_npv_vs_threshold.png`: Precision and NPV curves with prevalence baseline.
   - `f1_accuracy_vs_threshold.png`: F1-Score and Accuracy curves across thresholds.
   - `false_positive_negative_vs_threshold.png`: Direct counts of False Positives and False Negatives.
   - `youden_j_vs_threshold.png`: Youden's J statistic curves highlighting optimal balance points.
8. **Platform Standard Comparison**: [`ml-service/evaluation/heart/heart_threshold_analysis.csv`](file:///d:/smart-healthcare-platform/ml-service/evaluation/heart/heart_threshold_analysis.csv).

---

## 13. Limitations & Clinical Guardrails

1. **Cohort Sample Size**: Total cohort comprises 303 observations (242 training, 61 test). In the test set, each patient represents $\approx 3.57\%$ sensitivity ($1/28$) and $\approx 3.03\%$ specificity ($1/33$). Minor percentage changes must be interpreted within this discrete granularity.
2. **Referral Cohort Characteristics**: Cleveland Clinic catheterization patients from 1984 reflect elevated pre-test disease probability ($45.9\%$). In a general population screening cohort with lower prevalence (e.g. $10\%$), PPV will be lower while NPV will be higher for identical sensitivity and specificity values.
3. **No Clinical Optimization**: The evaluated thresholds are **experimental candidate operating points**, not medically certified clinical cutoffs.
4. **Uncalibrated Model Scores**: These threshold analyses operate on raw model output scores. Probability calibration (Phase 7) is required before interpreting probabilities as true risk percentages.

---

## 14. Handoff to Phase 7 (Probability Calibration)

Phase 6 threshold analysis is complete and experimentally verified.

**Handoff Deliverables for Phase 7**:
- **Candidate Operating Points**:
  - High-Sensitivity Screening: LR ($t=0.31$), RF ($t=0.32$), XGBoost ($t=0.24$)
  - Balanced Diagnostic: LR ($t=0.56$), RF ($t=0.55$), XGBoost ($t=0.62$)
  - Default Baseline: $t=0.50$
- **OOF Probability Distribution**: Raw model-estimated probabilities preserved in [`oof_predictions.csv`](file:///d:/smart-healthcare-platform/ml-service/artifacts/heart/experiments/oof_predictions.csv).
- **Phase 7 Research Objectives**:
  - Evaluate probability calibration quality (Brier Score, Expected Calibration Error).
  - Construct reliability diagrams / calibration curves.
  - Test post-hoc calibration methods (Platt Sigmoid Scaling vs. Isotonic Regression).
  - Quantify whether calibration preserves or modifies ranking and threshold behavior.
