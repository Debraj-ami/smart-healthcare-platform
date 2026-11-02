# Heart Disease — SHAP Explainability & Feature Attributions (Phase 8)

**Document Version**: 1.0.0  
**Phase**: Phase 8 — SHAP Explainability & Feature Attributions  
**Parent Specifications**:
- [`docs/HEART_PREPROCESSING.md`](file:///d:/smart-healthcare-platform/docs/HEART_PREPROCESSING.md) (Phase 4)
- [`docs/HEART_ML_EXPERIMENTATION.md`](file:///d:/smart-healthcare-platform/docs/HEART_ML_EXPERIMENTATION.md) (Phase 5)
- [`docs/HEART_THRESHOLD_ANALYSIS.md`](file:///d:/smart-healthcare-platform/docs/HEART_THRESHOLD_ANALYSIS.md) (Phase 6)
- [`docs/HEART_CALIBRATION_ANALYSIS.md`](file:///d:/smart-healthcare-platform/docs/HEART_CALIBRATION_ANALYSIS.md) (Phase 7)  
**Implementation Modules**:
- [`ml-service/app/heart_shap.py`](file:///d:/smart-healthcare-platform/ml-service/app/heart_shap.py)
- [`ml-service/run_heart_shap.py`](file:///d:/smart-healthcare-platform/ml-service/run_heart_shap.py)  
**Interactive Notebook**:
- [`notebooks/08_heart_shap_explainability.ipynb`](file:///d:/smart-healthcare-platform/notebooks/08_heart_shap_explainability.ipynb)  
**Automated Test Suite**: [`ml-service/tests/test_heart_shap.py`](file:///d:/smart-healthcare-platform/ml-service/tests/test_heart_shap.py) (15/15 tests passing, 80/80 total project tests passing)  
**Artifact Directory**: `ml-service/artifacts/heart/shap/`  
**Source Dataset**: `datasets/raw/heart/processed.cleveland.data` (SHA-256: `a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8`, MD5: `2d91a8ff69cfd9616aa47b59d6f843db`, 18,461 bytes)  
**Status**: Complete & Empirically Verified  

---

## 1. Objective & Scope

In clinical cardiovascular risk screening, classification models cannot operate as uninterpretable black boxes. Clinical decision-makers, auditors, and patients require auditable, mathematically grounded answers to:
> **"Which input clinical features contributed to the model's prediction, and in what direction?"**

Phase 8 implements **SHAP (SHapley Additive exPlanations)** based on cooperative game theory to explain model behavior for the finalized Heart Disease architectures.

### Absolute Boundaries & Safety Language
- **Model Attribution Only**: SHAP values quantify the mathematical contribution of an input feature toward the **model's numerical prediction output**.
- **No Biological Causation**: Feature contributions must **never** be interpreted as medical causation. A high positive contribution from a feature (e.g., fluoroscopy vessels) indicates that the model assigned increased weight to this observation on this dataset; it does not prove biological disease etiology.
- **No Diagnostic Claims**: Model-estimated probabilities and feature attributions do not constitute a clinical diagnosis or medical proof of coronary artery disease.
- **Strict Separation of Concerns**: No modifications were made to Phase 5, Phase 6, or Phase 7 artifacts. No production FastAPI bundling or frontend deployment is implemented in this phase.

---

## 2. Why SHAP is Needed in Clinical Decision Support

Standard global feature importance metrics (such as tree Gini impurity or magnitude of linear coefficients) suffer from critical limitations in healthcare:
1. **Lack of Local Granularity**: Global metrics tell us which features are important on average across the training cohort, but cannot explain why a specific individual patient was assigned a high or low estimated risk.
2. **Inconsistent Attribution under Feature Correlation**: Gini impurity is biased toward continuous features with high cardinality and fails to account for interaction effects.
3. **Clinical Auditability**: Regulatory and clinical governance frameworks require that individual alerts can be audited down to individual input measurements.

SHAP resolves these challenges by assigning an additive attribution $\phi_j(x)$ to each feature $j$ for each observation $x$, satisfying game-theoretic axioms of **Efficiency**, **Symmetry**, **Dummy**, and **Additivity**.

---

## 3. The SHAP Theoretical Concept

For a predictive model $f(x)$, the Shapley value of feature $j$ represents the average marginal contribution of feature $j$ across all possible feature subsets $S \subseteq F \setminus \{j\}$:
$$\phi_j(x) = \sum_{S \subseteq F \setminus \{j\}} \frac{|S|! (|F| - |S| - 1)!}{|F|!} \left[ f_x(S \cup \{j\}) - f_x(S) \right]$$

### The Efficiency / Additivity Axiom
The foundational guarantee of SHAP is that individual attributions sum exactly to the difference between the model output $f(x)$ and the baseline expected value $\mathbb{E}[f(X)]$:
$$\mathbb{E}[f(X)] + \sum_{j=1}^p \phi_j(x) = f(x)$$
Where:
- $\mathbb{E}[f(X)] = \phi_0$ is the **base value** (the model's expected output over the reference training distribution).
- $\phi_j(x) > 0$: The feature **contributes positively**, shifting the model output higher (toward disease).
- $\phi_j(x) < 0$: The feature **contributes negatively**, pulling the model output lower (toward control).

---

## 4. Global vs. Local Explainability

| Aspect | Global Explainability | Local Explainability |
| :--- | :--- | :--- |
| **Question** | *Which features most strongly drive the model's predictions overall?* | *Why did the model assign this specific probability to this patient?* |
| **Mathematical Basis** | Mean absolute SHAP value: $\frac{1}{N} \sum_{i=1}^N \|\phi_{i, j}\|$ | Individual attribution vector: $\vec{\phi}(x) = (\phi_1(x), \dots, \phi_p(x))$ |
| **Visual Representation** | Bar plots and beeswarm summary distributions. | Waterfall and force attributions. |
| **Clinical Role** | Model auditability, regulatory review, sanity check against cardiology consensus. | Patient-level clinical decision support, triage justification, risk alert transparency. |

---

## 5. Final Model Configuration

The cumulative empirical evidence from Phases 5–7 establishes the primary finalized model architecture:

| Property | Primary Finalist Configuration |
| :--- | :--- |
| **Model Family** | **Logistic Regression (Sparse L1 / Lasso)** |
| **Config ID** | `LR_robust_l1_C1.0` |
| **Phase 5 Evidence** | Highest 25-fold Repeated Stratified CV ROC-AUC ($0.9076 \pm 0.0506$), holdout ROC-AUC of $0.9513$, sparse feature selection (10 non-zero features). |
| **Phase 6 Evidence** | Established high-sensitivity screening operating point ($t=0.31$, Sensitivity $= 90.1\%$, Specificity $= 64.9\%$). |
| **Phase 7 Evidence** | Logistic Regression showed comparatively good empirical calibration on this dataset ($\text{Brier} = 0.1188$, $\text{Log Loss} = 0.3848$, $\text{ECE} = 0.0383$, Calibration Slope $= 1.0704$), and post-hoc calibration did not demonstrate a consistent improvement. |
| **Preprocessing** | `RobustScaler(drop='first')` on numericals, OneHotEncoder on categoricals with reference dropping. |
| **Calibration** | **Uncalibrated Baseline** (comparatively good empirical calibration on this dataset; post-hoc calibration did not demonstrate a consistent improvement). |
| **SHAP Explainer** | `shap.LinearExplainer` (exact analytical Shapley values). |
| **Output Space** | **Log-odds space** (`decision_function`). |

*Comparative evaluations are also maintained for **Random Forest** (`shap.TreeExplainer` in probability space) and **XGBoost** (`shap.TreeExplainer` in log-odds margin space).*

---

## 6. Model Output Space Being Explained

To maintain technical precision and avoid methodological ambiguity, Phase 8 explicitly specifies what SHAP explains for each model:

1. **Logistic Regression**:
   - The classifier's raw output is the linear log-odds decision function:
     $$\text{decision\_function}(x) = \beta_0 + \sum_{j=1}^{18} \beta_j x_j$$
   - `LinearExplainer` explains this **log-odds space**. Additivity holds exactly:
     $$\text{base\_value} + \sum_{j=1}^{18} \phi_j(x) = \text{decision\_function}(x)$$
   - The predicted probability is obtained via the standard logistic sigmoid:
     $$P(Y=1 \mid x) = \frac{1}{1 + \exp(-\text{decision\_function}(x))}$$
   - Because the logistic sigmoid is strictly monotonic, positive log-odds contributions directly increase predicted risk, while negative contributions decrease predicted risk.
2. **Random Forest**:
   - `TreeExplainer` directly explains the **positive-class probability space**:
     $$\text{base\_value} + \sum_{j=1}^{22} \phi_{j, 1}(x) = P(Y=1 \mid x)$$
3. **XGBoost**:
   - `TreeExplainer` explains the **raw margin (log-odds)** of the gradient boosted decision trees:
     $$\text{base\_value} + \sum_{j=1}^{22} \phi_j(x) = \text{margin}(x)$$

---

## 7. Feature Preprocessing & Traceability Mapping

Because Heart Disease preprocessing involves robust scaling, median imputation, and categorical one-hot encoding, maintaining traceability between transformed columns and original clinical variables is essential:

### 18 Transformed Features (Logistic Regression Pipeline)

| Index | Transformed Column | Original Variable | Clinical Display Name | Group | Clinical Description |
| :---: | :--- | :--- | :--- | :--- | :--- |
| 1 | `num__age` | `age` | Age | Numerical | Patient age in years (robustly scaled) |
| 2 | `num__trestbps` | `trestbps` | Resting Blood Pressure | Numerical | Resting BP in mm Hg on admission |
| 3 | `num__chol` | `chol` | Serum Cholesterol | Numerical | Serum cholesterol in mg/dl |
| 4 | `num__thalach` | `thalach` | Maximum Heart Rate | Numerical | Max heart rate achieved during stress test |
| 5 | `num__oldpeak` | `oldpeak` | ST Depression (Oldpeak) | Numerical | Exercise-induced ST depression relative to rest |
| 6 | `bin__sex` | `sex` | Sex (Male) | Binary | Biological sex (1 = male, 0 = female) |
| 7 | `bin__fbs` | `fbs` | Fasting Blood Sugar > 120 | Binary | Fasting blood sugar > 120 mg/dl |
| 8 | `bin__exang` | `exang` | Exercise-Induced Angina | Binary | Exercise-induced angina (1 = yes, 0 = no) |
| 9 | `cat__cp_2.0` | `cp` | Chest Pain: Atypical Angina | Categorical | Chest pain type 2 (ref: cp=1 typical) |
| 10 | `cat__cp_3.0` | `cp` | Chest Pain: Non-Anginal | Categorical | Chest pain type 3: non-anginal pain |
| 11 | `cat__cp_4.0` | `cp` | Chest Pain: Asymptomatic | Categorical | Chest pain type 4: asymptomatic ischaemia |
| 12 | `cat__restecg_1.0` | `restecg` | Resting ECG: ST-T Abnormality | Categorical | ST-T wave inversion / elevation |
| 13 | `cat__restecg_2.0` | `restecg` | Resting ECG: LV Hypertrophy | Categorical | Left ventricular hypertrophy by Estes' |
| 14 | `cat__slope_2.0` | `slope` | ST Slope: Flat | Categorical | ST slope flat (ref: slope=1 upsloping) |
| 15 | `cat__slope_3.0` | `slope` | ST Slope: Downsloping | Categorical | ST slope downsloping during peak exercise |
| 16 | `cat__thal_6.0` | `thal` | Thallium: Fixed Defect | Categorical | Fixed defect (previous infarction) |
| 17 | `cat__thal_7.0` | `thal` | Thallium: Reversible Defect | Categorical | Reversible defect (active ischemia) |
| 18 | `count__ca` | `ca` | Fluoroscopy Vessels (ca) | Count | Number of major vessels (0-3) colored |

---

## 8. Global SHAP Results (Logistic Regression Finalist)

Evaluated across the 61 holdout test observations ($33$ controls, $28$ positive cases):

| Rank | Transformed Feature | Original Feature | Clinical Display Name | Mean \|SHAP Value\| | Relative % | Direction of Impact |
| :---: | :--- | :--- | :--- | :---: | :---: | :--- |
| **1** | `count__ca` | `ca` | **Fluoroscopy Vessels (ca)** | **1.0232** | **23.94%** | Higher vessel-count values contribute positively to the model output |
| **2** | `cat__thal_7.0` | `thal` | **Thallium: Reversible Defect** | **0.6065** | **14.19%** | Presence of this encoded category contributes positively to the model output |
| **3** | `cat__cp_4.0` | `cp` | **Chest Pain: Asymptomatic** | **0.5578** | **13.05%** | Presence of asymptomatic category contributes positively to the model output |
| **4** | `bin__sex` | `sex` | **Sex (Male)** | **0.5094** | **11.92%** | Male sex contributes positively to the model output for this model |
| **5** | `cat__slope_2.0` | `slope` | **ST Slope: Flat** | **0.4248** | **9.94%** | Flat ST segment slope contributes positively to the model output |
| **6** | `num__thalach` | `thalach` | **Maximum Heart Rate** | **0.2747** | **6.43%** | Lower maximum heart rate values contribute positively to the model output (pushing toward positive class) |
| **7** | `bin__exang` | `exang` | **Exercise-Induced Angina** | **0.2023** | **4.73%** | Presence of exercise-induced angina contributes positively to the model output |
| **8** | `num__trestbps` | `trestbps` | **Resting Blood Pressure** | **0.1729** | **4.05%** | Higher resting blood pressure values contribute positively to the model output |
| **9** | `num__oldpeak` | `oldpeak` | **ST Depression (Oldpeak)** | **0.1676** | **3.92%** | Higher ST depression values contribute positively to the model output |
| **10** | `cat__restecg_2.0` | `restecg` | **Resting ECG: LV Hypertrophy** | **0.1164** | **2.72%** | Presence of LV hypertrophy contributes positively to the model output |
| 11 | `cat__cp_3.0` | `cp` | Chest Pain: Non-Anginal | 0.0838 | 1.96% | Contributes negatively to the model output relative to the reference category |
| 12 | `num__chol` | `chol` | Serum Cholesterol | 0.0617 | 1.44% | Minor positive contribution to the model output |
| 13 | `bin__fbs` | `fbs` | Fasting Blood Sugar > 120 | 0.0245 | 0.57% | Minor positive contribution to the model output |
| 14 | `num__age` | `age` | Age | 0.0000 | 0.00% | L1 penalty shrunk coefficient to 0 (zero contribution) |
| 15 | `cat__cp_2.0` | `cp` | Chest Pain: Atypical Angina | 0.0000 | 0.00% | L1 penalty shrunk coefficient to 0 (zero contribution) |
| 16 | `cat__restecg_1.0` | `restecg` | Resting ECG: ST-T Abnormality | 0.0000 | 0.00% | L1 penalty shrunk coefficient to 0 (zero contribution) |
| 17 | `cat__slope_3.0` | `slope` | ST Slope: Downsloping | 0.0000 | 0.00% | L1 penalty shrunk coefficient to 0 (zero contribution) |
| 18 | `cat__thal_6.0` | `thal` | Thallium: Fixed Defect | 0.0000 | 0.00% | L1 penalty shrunk coefficient to 0 (zero contribution) |

### Contextual Analysis of Global Findings:
- The top 4 features account for **63.1%** of total model attribution:
  1. **Fluoroscopy Vessels (`ca`)**: The single strongest driver (23.94%). Reflects the statistical association between fluoroscopy vessel count and the model output on this dataset.
  2. **Reversible Thallium Defect (`thal_7.0`)**: 14.19%. Reflects the positive model attribution associated with this thallium category on this dataset.
  3. **Asymptomatic Chest Pain (`cp_4.0`)**: 13.05%. Reflects the positive model attribution associated with the asymptomatic category on this cohort.
  4. **Male Sex (`sex`)**: 11.92%. Reflects the positive model attribution associated with male sex on this cohort.

---

## 9. Local Explanation Analysis on Representative Holdout Patients

We audited real patient observations from the locked holdout test set ($N=61$) to verify local interpretability:

| Case Description | Sample Index | True Label | Model Probability | Log-Odds Output | Primary Positive Contributors | Primary Negative Contributors |
| :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **High-Confidence True Positive** | #62 | 1 (Positive) | **0.9963** | +5.60 | Fluoroscopy Vessels (+2.68); Thallium Reversible Defect (+0.78); Asymptomatic Chest Pain (+0.58) | Serum Cholesterol (-0.09); Resting Blood Pressure (-0.04) |
| **High-Confidence True Negative** | #94 | 0 (Negative) | **0.0259** | -3.63 | Resting ECG: LV Hypertrophy (+0.12); Resting Blood Pressure (+0.06) | Female Sex (-0.79); Fluoroscopy Vessels (-0.68); Non-Asymptomatic CP (-0.54) |
| **Borderline Case** | #271 | 0 (Negative) | **0.5110** | +0.04 | Asymptomatic Chest Pain (+0.58); Resting Blood Pressure (+0.40); Male Sex (+0.37) | Fluoroscopy Vessels (-0.68); Normal Thallium (-0.51); ST Slope: Flat (-0.39) |
| **False Positive Discordance** | #92 | 0 (Negative) | **0.9651** | +3.32 | Fluoroscopy Vessels (+2.68); Thallium Reversible Defect (+0.78); ST Slope: Flat (+0.46) | Non-Asymptomatic CP (-0.54); Non-Anginal CP (-0.20); No Exang (-0.15) |
| **False Negative Discordance** | #66 | 1 (Positive) | **0.2653** | -1.02 | ST Slope: Flat (+0.46); Male Sex (+0.37); ST Depression (+0.30) | Fluoroscopy Vessels (-0.68); Non-Asymptomatic CP (-0.54); Normal Thallium (-0.51) |

### Key Observations from Local Audits:
- **True Positive (Patient #62)**: Patient had 3 vessels colored on fluoroscopy and a reversible thallium defect, contributing positively to the log-odds by $+3.46$ log-odds units to yield a model-estimated probability of $\hat{p} = 0.9963$.
- **True Negative (Patient #94)**: Patient had zero fluoroscopy vessels, normal thallium scan, and female sex, pulling the model log-odds down to $-3.63$ to yield a model-estimated probability of $\hat{p} = 0.0259$.
- **Discordant False Positive (Patient #92)**: Patient had 2 vessels colored on fluoroscopy and a reversible thallium defect, leading the model to assign an elevated model-estimated probability ($0.9651$) despite the negative target label ($y=0$). This illustrates how the model responds to specific feature patterns in this dataset.
- **Discordant False Negative (Patient #66)**: Patient had zero fluoroscopy vessels, non-asymptomatic chest pain, and normal thallium scan, leading the model to assign a low model-estimated probability ($0.2653$) despite the positive target label ($y=1$).

---

## 10. Multi-Model Architecture Comparison

Comparing global feature importance across model families:

| Rank | Logistic Regression (Robust L1) | Random Forest (n=200, leaf=5) | XGBoost (n=100, depth=2, lr=0.05) |
| :---: | :--- | :--- | :--- |
| **1** | Fluoroscopy Vessels (ca) (1.023) | Thallium: Normal (0.093) | Thallium: Normal (0.505) |
| **2** | Thallium: Reversible Defect (0.607) | Fluoroscopy Vessels (ca) (0.076) | Chest Pain: Asymptomatic (0.428) |
| **3** | Chest Pain: Asymptomatic (0.558) | Thallium: Reversible Defect (0.061) | Fluoroscopy Vessels (ca) (0.404) |
| **4** | Sex (Male) (0.509) | ST Depression (Oldpeak) (0.059) | Maximum Heart Rate (0.339) |
| **5** | ST Slope: Flat (0.425) | Maximum Heart Rate (0.046) | ST Depression (Oldpeak) (0.327) |

*Consensus finding: Across all three model families, `ca` (fluoroscopy vessels), `thal` (thallium defect), `cp` (chest pain asymptomatic), and `thalach` (maximum heart rate) consistently emerge as the top 4 most influential clinical variables.*

---

## 11. Relationship to Phase 7 Probability Calibration

The conceptual pipeline is structured as follows:

```
Raw Patient Inputs (13 Variables)
        ↓
Feature Preprocessing (ColumnTransformer)
        ↓
Transformed Features (18 Features)
        ↓
Base Classifier (Logistic Regression L1) ──► SHAP Explainer (LinearExplainer in Log-Odds)
        ↓
Model Log-Odds Decision Value [decision_function]
        ↓
Calibration Layer (Standard Logistic Sigmoid)
        ↓
Model-Estimated Probability p in [0, 1]
```

### Explaining the Model Layer
- For the primary model (Logistic Regression), **SHAP explains the classifier's log-odds decision output**.
- In Phase 7, Logistic Regression showed comparatively good empirical calibration on this dataset ($\text{ECE} = 0.0383$, $\text{Slope} = 1.0704$, $\text{Brier} = 0.1188$), and post-hoc calibration did not demonstrate a consistent improvement.
- Therefore, the log-odds decision function maps directly to model-estimated probabilities via the standard sigmoid:
  $$P(Y=1 \mid x) = \frac{1}{1 + \exp\left(-\left(\text{base\_value} + \sum_{j=1}^{18} \phi_j(x)\right)\right)}$$
- This ensures that SHAP explanations directly reflect the feature contributions driving the model's estimated probability on this dataset without introducing additional post-hoc distortion.

---

## 12. Artifacts Produced

The following artifacts have been generated in [`ml-service/artifacts/heart/shap/`](file:///d:/smart-healthcare-platform/ml-service/artifacts/heart/shap/):
1. `shap_feature_importance.csv`: Ranked mean absolute SHAP values and relative importance percentages.
2. `shap_feature_mapping.csv`: Complete dictionary mapping raw input clinical variables to transformed feature columns.
3. `shap_global_summary.csv`: Multi-model global comparison across LR, RF, and XGBoost.
4. `shap_local_explanations.csv`: Structured feature contributions for representative holdout patients.
5. `shap_metadata.json`: Machine-readable metadata recording model parameters, background masker configurations, and safety rules.
6. `shap_global_bar_plot.png`: Publication-quality horizontal bar chart of global feature attributions.
7. `shap_beeswarm_summary_plot.png`: Directional beeswarm plot illustrating value-to-impact distributions.
8. `shap_model_comparison_importance.png`: 3-panel comparative bar plot across all three finalist architectures.
9. `shap_local_waterfall_examples.png`: Local waterfall plots displaying feature contributions for representative patients.

---

## 13. Limitations & Safety Boundaries

1. **Association vs. Causation**: SHAP values describe statistical contributions within the model. They must never be interpreted as clinical evidence that modifying a feature (e.g., lowering blood pressure during an acute episode) will alter disease status.
2. **Cohort Dependency**: Attributions reflect the Cleveland Clinic 1984 cohort distribution ($N=303$, 45.9% prevalence). In populations with different demographic structures, feature attributions may shift.
3. **Reference Category Interpretation**: In one-hot encoded variables with reference dropping (`drop='first'`), individual SHAP values represent contributions relative to the omitted baseline category (e.g., typical angina for chest pain).
4. **No Direct Clinical Diagnosis**: Explanations are strictly clinical decision support aids and must be presented alongside appropriate non-diagnostic disclaimers.

---

## 14. Handoff to Production & Service Integration

Phase 8 completes the offline research and validation sequence for Heart Disease:
- **Phase 0–4**: Dataset Provenance, Functional Specification, Preprocessing.
- **Phase 5–7**: Machine Learning Experimentation, Threshold Analysis, Probability Calibration.
- **Phase 8**: Explainable AI / SHAP Attributions.

**Deliverables Ready for Subsequent Phases**:
- The finalized model pipeline (`Logistic Regression Robust L1`) and fitted preprocessor.
- The SHAP `LinearExplainer` and background training distribution.
- The human-readable feature dictionary mapping internal transformed features to clinical UI labels.
- In subsequent deployment phases, these components will be packaged into `heart_model_bundle.joblib` and served via FastAPI endpoints (`/heart/predict` and `/heart/explain`).
