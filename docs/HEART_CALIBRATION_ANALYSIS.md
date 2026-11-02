# Heart Disease — Probability Calibration Analysis (Phase 7)

**Document Version**: 1.0.0  
**Phase**: Phase 7 — Probability Calibration Analysis  
**Parent Specifications**:
- [`docs/HEART_PREPROCESSING.md`](file:///d:/smart-healthcare-platform/docs/HEART_PREPROCESSING.md) (Phase 4)
- [`docs/HEART_ML_EXPERIMENTATION.md`](file:///d:/smart-healthcare-platform/docs/HEART_ML_EXPERIMENTATION.md) (Phase 5)
- [`docs/HEART_THRESHOLD_ANALYSIS.md`](file:///d:/smart-healthcare-platform/docs/HEART_THRESHOLD_ANALYSIS.md) (Phase 6)  
**Implementation Modules**:
- [`ml-service/app/heart_calibration.py`](file:///d:/smart-healthcare-platform/ml-service/app/heart_calibration.py)
- [`ml-service/run_heart_calibration.py`](file:///d:/smart-healthcare-platform/ml-service/run_heart_calibration.py)  
**Interactive Notebooks**:
- [`notebooks/06_heart_calibration_analysis.ipynb`](file:///d:/smart-healthcare-platform/notebooks/06_heart_calibration_analysis.ipynb)
- [`notebooks/07_heart_calibration_analysis.ipynb`](file:///d:/smart-healthcare-platform/notebooks/07_heart_calibration_analysis.ipynb)  
**Automated Test Suite**: [`ml-service/tests/test_heart_calibration.py`](file:///d:/smart-healthcare-platform/ml-service/tests/test_heart_calibration.py) (15/15 tests passing, 65/65 total project tests passing)  
**Artifact Directory**: `ml-service/artifacts/heart/calibration/`  
**Platform Evaluation Summary**: [`ml-service/evaluation/heart/heart_calibration_comparison.csv`](file:///d:/smart-healthcare-platform/ml-service/evaluation/heart/heart_calibration_comparison.csv)  
**Source Dataset**: `datasets/raw/heart/processed.cleveland.data` (SHA-256: `a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8`, MD5: `2d91a8ff69cfd9616aa47b59d6f843db`, 18,461 bytes)  
**Status**: Complete & Empirically Verified  

---

## 1. Objective & Scope

In clinical cardiovascular risk screening, classification models output continuous numerical values $\hat{p} \in [0, 1]$. While Phase 5 established the discrimination capability of the candidate models (ranking patients with disease above patients without disease), **Phase 7 evaluates empirical probability calibration**: determining whether the numerical probability values produced by the finalist models correspond reasonably to observed outcome frequencies.

For example, among patients whose model-estimated probability is around $0.80$, the observed positive frequency in that probability region should ideally be close to $80\%$.

### Absolute Phase Boundaries
- **No Model Retraining of Search Spaces**: Evaluates the exact three finalist configurations selected in Phase 5.
- **No Clinical Diagnosis Claims**: Model-estimated probabilities represent empirical statistical associations on this dataset cohort, not true individual patient biological risk.
- **No Holdout Contamination**: The 61-sample holdout test set is never used to select or fit calibrators.
- **No SHAP Explainability**: Feature attribution analysis is deferred to **Phase 8 (SHAP Explainability & Feature Attributions)**.
- **No Backend/Frontend Integration**: Production FastAPI endpoints, Spring Boot services, and React frontends remain untouched until later phases.

---

## 2. Why Probability Calibration is Needed

Standard machine learning classifiers are optimized for discrimination or surrogate loss functions, not necessarily for calibrated probabilities:
1. **Tree Ensembles (Random Forest)**: Predict probabilities by averaging binary decision votes across trees. This structure pulls predicted probabilities away from $0.0$ and $1.0$ toward the cohort base rate (near $0.50$), producing probabilities that understate extreme certainty and over-compress moderate certainty.
2. **Gradient Boosted Trees (XGBoost)**: While log-loss training encourages calibration, regularized leaf weights and shallow trees can distort probabilities in tails.
3. **Linear Models (Logistic Regression)**: While theoretically producing log-odds, L1 regularization (Lasso) shrinks coefficients toward zero, which can compress probability dispersion.

In clinical decision support, calibrated probabilities are essential because clinicians, risk stratifiers, and shared decision-making protocols rely on the numerical probability scale to establish triage categories (e.g., low, intermediate, high risk) rather than arbitrary ranking indices.

---

## 3. Difference Between Discrimination and Calibration

| Dimension | Discrimination | Calibration |
| :--- | :--- | :--- |
| **Question Answered** | *Can the model separate positive cases from negative controls?* | *Does a model-estimated probability of $p$ match an observed positive frequency of $p$?* |
| **Primary Focus** | Relative ranking order among patients. | Accuracy and reliability of the absolute probability values. |
| **Key Metrics** | ROC-AUC, PR-AUC, Sensitivity, Specificity. | Brier Score, Log Loss, Expected Calibration Error (ECE), Calibration Slope/Intercept. |
| **Invariance** | Invariant under any strictly monotonic transformation. | Highly sensitive to scaling, shifting, or compression of probabilities. |
| **Clinical Role** | Identifies who has higher risk than whom. | Informs whether estimated risk matches real-world disease incidence. |

---

## 4. Dataset Cohort Split & Provenance

The modeling cohort is derived from the 303-row Cleveland dataset, with binary target mapping:
$$y = \begin{cases} 0 & \text{if } \text{num} = 0 \text{ (control, } < 50\% \text{ stenosis)} \\ 1 & \text{if } \text{num} \ge 1 \text{ (coronary artery disease, } \ge 50\% \text{ stenosis)} \end{cases}$$

Splitting adheres strictly to the locked Phase 4 stratified partition:
- **Training Cohort ($80\%$)**: $N = 242$ patients ($111$ positive cases [45.87%], $131$ negative controls [54.13%]).
- **Hold-out Test Cohort ($20\%$)**: $N = 61$ patients ($28$ positive cases [45.90%], $33$ negative controls [54.10%]).

```
                       Cleveland Dataset (N=303)
                                  │
                 Stratified 80/20 Train/Test Split
                                  │
         ┌────────────────────────┴────────────────────────┐
         ▼                                                 ▼
Training Cohort (N=242)                           Locked Hold-Out (N=61)
         │                                                 │
 5-Fold Stratified CV                                      │
  - Preprocessing inside folds                             │
  - Sigmoid / Isotonic cross-fitting                       │
  - OOF Calibration evaluation                             │
         │                                                 │
         ├────────────────────────────────────────┐        │
         ▼                                        ▼        │
Fit Final Calibrators (N=242)            Compare OOF vs    │
         │                               Phase 6 Cutoffs   │
         ▼                                        │        ▼
   Fitted Pipelines ───────────────────────► Strict Descriptive Evaluation
                                             (NO method selection on test)
```

---

## 5. Leakage Prevention Protocol

Data leakage during probability calibration is a frequent point of failure in clinical machine learning. Phase 7 implements four strict leakage safeguards:

1. **Strict Hold-Out Quarantine**: The 61-sample holdout test set is excluded from calibration training, cross-validation, hyperparameter selection, and algorithm comparison.
2. **Cross-Fitting Inside Training CV**: On the 242-sample training cohort, 5-fold Stratified Cross-Validation is employed. In each fold, base preprocessors and classifiers are trained on 4/5 of the data, and calibrators are fitted strictly on fold training data (`CalibratedClassifierCV` internal cross-validation). Out-of-fold predictions on validation folds are aggregated.
3. **No Target Contamination in Preprocessing**: Imputation medians, standard scaling, and one-hot encoders are fitted exclusively on training instances.
4. **Isotonic Overfitting Monitoring**: Non-parametric calibration models are evaluated using out-of-fold cross-validation log-loss to detect overfitting before any model selection is made.

---

## 6. Uncalibrated Baseline Models

We evaluate the three finalized Phase 5 architectures:
1. **Logistic Regression (Robust L1)**: `scaler='robust'`, `penalty='l1'`, `C=1.0`, `solver='liblinear'`.
2. **Random Forest (n=200, leaf=5)**: `n_estimators=200`, `min_samples_leaf=5`, unscaled tree preprocessor.
3. **XGBoost (n=100, depth=2, lr=0.05)**: `n_estimators=100`, `max_depth=2`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`.

---

## 7. Sigmoid (Platt) Calibration

Sigmoid calibration fits a univariate logistic model over the classifier's continuous decision values or log-odds:
$$\hat{P}(Y=1 \mid f(X)) = \frac{1}{1 + \exp(A \cdot f(X) + B)}$$
where parameters $A$ and $B$ are estimated via maximum likelihood.

**Properties on Small Cohorts**:
- Highly parametric and rigid (two parameters $A$ and $B$).
- Monotonic transformation that preserves ranking order.
- Exceptionally resistant to overfitting, making it well-suited for moderate-to-small sample sizes ($N=242$).
- Effectively stretches probabilities when models over-compress predictions near the center.

---

## 8. Isotonic Calibration

Isotonic regression fits a non-parametric piecewise constant monotonic step function:
$$\min_{\hat{m}_1 \le \hat{m}_2 \le \dots \le \hat{m}_n} \sum_{i=1}^n (y_i - \hat{m}_i)^2$$
using the Pool Adjacent Violators Algorithm (PAVA).

**Risks on Small Cohorts ($N=242$)**:
- **Non-Parametric Flexibility**: Makes no parametric shape assumptions.
- **Step-Boundary Overfitting**: In small cohorts ($N=242$), PAVA can create abrupt probability plateaus and extreme boundary steps based on idiosyncratic training clusters.
- When evaluated out-of-fold, step overconfidence leads to heavy logarithmic loss penalties if an out-of-fold sample falls into a misjudged probability bin.

---

## 9. Evaluation Metrics

### A. Brier Score (`brier_score_loss`)
Measures the mean squared error between model-estimated probabilities $p_i \in [0, 1]$ and true binary outcomes $y_i \in \{0, 1\}$:
$$\text{Brier Score} = \frac{1}{N} \sum_{i=1}^N (p_i - y_i)^2$$
- Range: $[0, 1]$. Lower is better ($0.0$ is perfect; $0.25$ represents an uninformative $0.50$ baseline).

### B. Log Loss (`log_loss`)
Cross-entropy loss heavily penalizes confident but erroneous probability assertions:
$$\text{Log Loss} = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \ln(p_i) + (1 - y_i) \ln(1 - p_i) \right]$$
- Range: $[0, \infty)$. Lower is better. Sensitive to boundary miscalibration.

### C. Expected Calibration Error (ECE)
Partitions predicted probabilities into $M=10$ uniform bins $B_1, \dots, B_M \subset [0, 1]$:
$$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
where $\text{acc}(B_m)$ is the observed positive frequency in bin $m$, and $\text{conf}(B_m)$ is the mean predicted probability.
- Lower is better.

### D. Calibration Intercept and Slope
Regresses true labels against the logit-transformed probabilities:
$$\text{logit}(P(Y=1)) = \alpha + \beta \cdot \text{logit}(p)$$
- **Calibration Intercept ($\alpha$)**: Ideal $= 0.0$. Values $< 0$ indicate systematic overestimation of risk; values $> 0$ indicate underestimation.
- **Calibration Slope ($\beta$)**: Ideal $= 1.0$. Values $> 1.0$ indicate that predicted probabilities are too compressed towards the center; values $< 1.0$ indicate that probabilities are too extreme.

---

## 10. Calibration Binning Strategy & Reliability Analysis

To balance statistical granularity with small-sample stability ($N_{\text{train}}=242$, $N_{\text{holdout}}=61$):
- **Development Cohort ($N=242$)**: Evaluated using $M=10$ uniform bins with bin width $0.10$.
- **Hold-out Cohort ($N=61$)**: Evaluated using $M=5$ uniform bins with bin width $0.20$ to maintain sufficient observations per bin.

### Bin Allocation Breakdown (Training Cohort OOF, $N=242$)
For Uncalibrated Logistic Regression:
- Bin [0.0, 0.1]: $72$ samples, Mean Pred $= 0.0384$, Observed Freq $= 0.0556$, Gap $= 0.0172$
- Bin [0.1, 0.2]: $20$ samples, Mean Pred $= 0.1449$, Observed Freq $= 0.1000$, Gap $= 0.0449$
- Bin [0.2, 0.3]: $14$ samples, Mean Pred $= 0.2450$, Observed Freq $= 0.2143$, Gap $= 0.0307$
- Bin [0.3, 0.4]: $15$ samples, Mean Pred $= 0.3541$, Observed Freq $= 0.4000$, Gap $= 0.0459$
- Bin [0.4, 0.5]: $8$ samples, Mean Pred $= 0.4578$, Observed Freq $= 0.3750$, Gap $= 0.0828$
- Bin [0.5, 0.6]: $14$ samples, Mean Pred $= 0.5562$, Observed Freq $= 0.6429$, Gap $= 0.0867$
- Bin [0.6, 0.7]: $11$ samples, Mean Pred $= 0.6559$, Observed Freq $= 0.7273$, Gap $= 0.0714$
- Bin [0.7, 0.8]: $13$ samples, Mean Pred $= 0.7533$, Observed Freq $= 0.6923$, Gap $= 0.0610$
- Bin [0.8, 0.9]: $23$ samples, Mean Pred $= 0.8532$, Observed Freq $= 0.9130$, Gap $= 0.0598$
- Bin [0.9, 1.0]: $52$ samples, Mean Pred $= 0.9575$, Observed Freq $= 0.9423$, Gap $= 0.0152$

---

## 11. Empirical Results: Training Cohort Cross-Validation (N=242)

The table below summarizes cross-validated out-of-fold probability performance across all 242 training instances:

| Model | Method | Brier Score | Log Loss | ECE | Cal. Intercept | Cal. Slope | ROC-AUC | PR-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Uncalibrated | **0.1188** | **0.3848** | **0.0383** | **-0.0358** | **1.0704** | 0.9076 | 0.8943 |
| Logistic Regression | Sigmoid | 0.1221 | 0.3933 | 0.0686 | 0.1302 | 1.3396 | 0.9083 | 0.8976 |
| Logistic Regression | Isotonic | 0.1153 | 0.4985 | 0.0779 | 0.0129 | 0.8471 | 0.9159 | 0.9032 |
| **Random Forest** | Uncalibrated | 0.1328 | 0.4208 | 0.0900 | 0.0645 | 1.5774 | 0.8987 | 0.8951 |
| **Random Forest** | **Sigmoid** | **0.1285** | **0.4053** | **0.0634** | **0.0264** | **1.1996** | 0.8975 | 0.8939 |
| Random Forest | Isotonic | 0.1292 | 0.6543 | 0.0719 | -0.0899 | 0.5818 | 0.8903 | 0.8839 |
| **XGBoost** | Uncalibrated | **0.1313** | **0.4131** | 0.0659 | **-0.0146** | **1.0877** | 0.8890 | 0.8882 |
| XGBoost | Sigmoid | 0.1342 | 0.4262 | **0.0643** | 0.0169 | 1.2164 | 0.8863 | 0.8743 |
| XGBoost | Isotonic | 0.1357 | 0.5466 | 0.0719 | -0.1353 | 0.7332 | 0.8835 | 0.8806 |

### Analysis of Development Findings:
1. **Logistic Regression**:
   - The uncalibrated baseline is remarkably well-calibrated natively, achieving the lowest ECE ($0.0383$), calibration slope ($1.0704$, very close to ideal $1.00$), and log loss ($0.3848$).
   - Post-hoc calibration yields minimal improvement.
2. **Random Forest**:
   - The uncalibrated ensemble shows significant probability compression (calibration slope $= 1.5774$, ECE $= 0.0900$).
   - **Sigmoid calibration delivers a distinct, consistent improvement**: Brier score drops from $0.1328$ to $0.1285$, Log Loss drops from $0.4208$ to $0.4053$, ECE improves from $0.0900$ to $0.0634$, and the slope adjusts to $1.1996$.
3. **Isotonic Overfitting on Small Cohorts**:
   - Across all three models, isotonic regression suffers severe log-loss inflation on CV:
     - Logistic Regression: Log loss increases from $0.3848 \rightarrow 0.4985$ (+29.5%).
     - Random Forest: Log loss increases from $0.4208 \rightarrow 0.6543$ (+55.5%).
     - XGBoost: Log loss increases from $0.4131 \rightarrow 0.5466$ (+32.3%).
   - This empirical result validates the methodological requirement: **isotonic regression overfits step boundaries on small cohorts ($N=242$)**.

---

## 12. Descriptive Hold-Out Evaluation (N=61)

Following the freeze of the development methodology on the training cohort, models were refitted on the full 242-sample training cohort and evaluated **descriptively** on the locked 61-sample holdout test set ($33$ controls, $28$ positive cases):

| Model | Method | Brier Score | Log Loss | ECE (5 Bins) | Cal. Intercept | Cal. Slope | ROC-AUC | PR-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Uncalibrated | 0.0886 | 0.3019 | 0.0808 | -0.5893 | 1.2867 | 0.9513 | 0.9274 |
| Logistic Regression | Sigmoid | 0.0903 | 0.3120 | 0.1113 | -0.5118 | 1.6115 | 0.9535 | 0.9332 |
| Logistic Regression | Isotonic | 0.0912 | 0.2993 | 0.0934 | -0.6739 | 1.0608 | 0.9513 | 0.9296 |
| **Random Forest** | Uncalibrated | 0.1045 | 0.3545 | 0.1257 | -0.3821 | 2.3021 | 0.9513 | 0.9401 |
| **Random Forest** | **Sigmoid** | **0.0923** | **0.3162** | **0.1150** | -0.4297 | 1.7394 | 0.9513 | 0.9429 |
| Random Forest | Isotonic | 0.0898 | 0.3019 | 0.0715 | -0.4016 | 1.0889 | 0.9513 | 0.9272 |
| **XGBoost** | Uncalibrated | 0.0861 | 0.2927 | 0.0969 | -0.6039 | 1.6505 | 0.9589 | 0.9543 |
| XGBoost | Sigmoid | 0.0861 | 0.3073 | 0.0871 | -0.5266 | 1.6814 | 0.9578 | 0.9503 |
| XGBoost | Isotonic | 0.0809 | 0.2705 | 0.0640 | -0.4860 | 1.2131 | 0.9610 | 0.9559 |

> [!NOTE]
> In accordance with strict methodological standards, holdout results are presented purely for descriptive confirmation and are not used to choose the calibration algorithm.

---

## 13. Relationship to Phase 6 Threshold Analysis

Phase 6 established operational decision thresholds on raw uncalibrated probabilities:
- **Logistic Regression**: High-Sensitivity $t=0.31$, Default $t=0.50$, Balanced Youden $t=0.56$.
- **Random Forest**: High-Sensitivity $t=0.32$, Default $t=0.50$, Balanced Youden $t=0.55$.
- **XGBoost**: High-Sensitivity $t=0.24$, Default $t=0.50$, Balanced Youden $t=0.62$.

### Shift in Operating Points Post-Calibration
Probability calibration warps the continuous numerical mapping from raw scores to probabilities. While patient ranking remains essentially intact (ROC-AUC remains virtually identical), **the numerical probability cutoff corresponding to a specific clinical operating goal shifts**:

| Model | Calibration Method | Phase 6 Raw High-Sens Cutoff | Calibrated High-Sens Cutoff | Calibrated Sensitivity | Calibrated Specificity | Phase 6 Raw Youden Cutoff | Calibrated Youden Cutoff | Calibrated Youden J |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Sigmoid | 0.31 | **0.32** | 0.9009 | 0.7023 | 0.56 | **0.45** | 0.7247 |
| Logistic Regression | Isotonic | 0.31 | **0.33** | 0.9099 | 0.7557 | 0.56 | **0.45** | 0.7282 |
| **Random Forest** | **Sigmoid** | 0.32 | **0.24** | 0.9099 | 0.6107 | 0.55 | **0.64** | 0.6686 |
| Random Forest | Isotonic | 0.32 | **0.26** | 0.9009 | 0.5649 | 0.55 | **0.58** | 0.6776 |
| **XGBoost** | Sigmoid | 0.24 | **0.24** | 0.9009 | 0.5954 | 0.62 | **0.55** | 0.6499 |
| XGBoost | Isotonic | 0.24 | **0.27** | 0.9009 | 0.5802 | 0.62 | **0.62** | 0.6583 |

### Critical Takeaways:
1. **Thresholds are Scale-Dependent**: Decision cutoffs cannot be transferred directly between uncalibrated and calibrated spaces. For Random Forest, the high-sensitivity threshold shifted from $0.32 \rightarrow 0.24$, while the balanced Youden threshold shifted from $0.55 \rightarrow 0.64$.
2. **Discrimination Invariance**: Probability calibration does not inherently improve or degrade discrimination; it simply aligns numerical probabilities with empirical frequencies.
3. **No Phase 6 Invalidation**: Phase 6 artifacts remain completely valid and intact; Phase 7 demonstrates how the numerical decision scale transforms under calibration.

---

## 14. Reliability Diagrams & Visualizations

Phase 7 produced three publication-quality visualizations saved to `ml-service/artifacts/heart/calibration/`:
1. `reliability_diagrams_train_cv.png`: 3-panel comparison across models on the training cohort ($N=242$), plotting mean predicted probabilities against observed positive frequencies with $10$ uniform bins and reference ideal diagonals.
2. `reliability_diagrams_holdout.png`: 3-panel descriptive reliability curves on the locked holdout ($N=61$) across Uncalibrated, Sigmoid, and Isotonic methods.
3. `probability_distribution_comparison.png`: $3 \times 3$ grid of histograms displaying predicted probability distributions segmented by ground truth (Control vs. Heart Disease) across all models and calibration techniques.

---

## 15. Limitations & Caveats

1. **Cohort Sample Size Constraints ($N=242$ Training, $N=61$ Holdout)**:
   - In small cohorts, empirical bin frequencies exhibit wider sampling variance.
   - Non-parametric isotonic regression suffers from step-boundary overfitting, confirmed by cross-validated log loss penalties.
2. **Dataset-Specific Calibration**:
   - Calibration is dependent on cohort disease prevalence ($45.87\%$ in Cleveland). In populations with different baseline cardiovascular disease prevalence, calibration intercepts will require recalibration.
3. **Absence of True Individual Risk**:
   - Model-estimated probabilities represent empirical group frequencies under this protocol, not true individual patient risk or guaranteed biological certainty.

---

## 16. Final Empirical Conclusion

1. **Logistic Regression (Robust L1)**:
   - The uncalibrated baseline is already exceptionally well-calibrated natively on this dataset (Brier $= 0.1188$, Log Loss $= 0.3848$, ECE $= 0.0383$, Slope $= 1.0704$).
   - Sigmoid calibration maintains comparable performance; Isotonic calibration introduces step-boundary log-loss inflation.
2. **Random Forest (n=200, leaf=5)**:
   - Uncalibrated probabilities exhibit pronounced center-compression (slope $= 1.5774$).
   - **Sigmoid (Platt) calibration provides clear empirical benefits**: reduces Brier score ($0.1328 \rightarrow 0.1285$), lowers log loss ($0.4208 \rightarrow 0.4053$), improves ECE ($0.0900 \rightarrow 0.0634$), and adjusts calibration slope ($1.5774 \rightarrow 1.1996$).
3. **XGBoost (n=100, depth=2, lr=0.05)**:
   - Uncalibrated shallow trees demonstrate solid native calibration (slope $= 1.0877$, ECE $= 0.0659$).
   - Sigmoid calibration maintains near-equivalent metrics; Isotonic calibration elevates cross-validated log loss.
4. **Methodological Recommendation for Platform Implementation**:
   - **Sigmoid (Platt) calibration** is the robust, generalizable calibration methodology for tree ensembles on this small cohort, preventing the severe cross-validation log-loss inflation observed with isotonic regression.

---

## 17. Handoff to Phase 8

Phase 7 outputs establish calibrated probability pipelines for downstream consumption:
- **Phase 8 (SHAP Explainability & Feature Attributions)**: Will compute TreeSHAP and KernelSHAP attributions on the finalist models, explaining individual predictions and validating clinical feature contributions (e.g., chest pain type, ST-segment depression, fluoroscopy vessels, maximum heart rate).
- **Subsequent Phases**: Production model bundling, FastAPI inference schemas, and clinical user interface integration will ingest calibrated probabilities and threshold mappings.
