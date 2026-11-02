# Heart Disease — Feature Engineering & Data Design (Phase 3)

**Document Version**: 1.0.0  
**Phase**: Phase 3 — Feature Engineering & Data Design  
**Parent Architecture**: [`docs/HEART_ARCHITECTURE.md`](file:///d:/smart-healthcare-platform/docs/HEART_ARCHITECTURE.md)  
**Upstream Artifacts**: [`docs/HEART_DATASET_ANALYSIS.md`](file:///d:/smart-healthcare-platform/docs/HEART_DATASET_ANALYSIS.md), [`docs/HEART_USER_JOURNEYS.md`](file:///d:/smart-healthcare-platform/docs/HEART_USER_JOURNEYS.md)  
**Source Dataset**: `datasets/raw/heart/processed.cleveland.data` (SHA-256: `a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8`)  
**Status**: Complete & Verified  

---

## 1. Scope & Objectives

This specification defines the modeling data contract, feature semantics, transformation strategies, leakage prevention rules, and experimental validation design for the **Heart Disease** risk prediction pipeline.

### Operational Boundaries
- **No Model Training**: No machine learning models (Logistic Regression, Random Forest, XGBoost) are trained in Phase 3.
- **No Model Selection or Tuning**: No hyperparameters or algorithms are selected.
- **No Threshold Optimization**: Risk thresholds and operating points are deferred to Phase 6.
- **No Probability Calibration**: Calibration curve fitting is deferred to Phase 7.
- **No SHAP Artifacts**: Explainer generation is deferred to Phase 8.
- **No Raw Data Modification**: The source file `datasets/raw/heart/processed.cleveland.data` is treated as strictly read-only.
- **Independence of Other Diseases**: Liver Disease implementation remains untouched; Diabetes and Parkinson's work is not started.

---

## 2. Raw Dataset Contract

The Cleveland Heart Disease dataset is stored as a headerless comma-separated ASCII file at `datasets/raw/heart/processed.cleveland.data`.

### Verified Raw Contract Summary
- **Record Count**: Exactly 303 observations.
- **Attribute Count**: Exactly 14 columns (13 predictive attributes + 1 angiographic target).
- **Delimiter**: Comma (`,`), exactly 14 tokens per line, 0 malformed lines.
- **Missing Value Sentinel**: String literal `'?'` occurring exclusively in `ca` (4 instances) and `thal` (2 instances). Total affected rows: 6 (1.98%).
- **Standard NaNs**: Exactly 0.
- **Exact Duplicate Rows**: Exactly 0.
- **Pandas Inferred Types**: 11 `float64`, 2 `object` (`ca`, `thal`), 1 `int64` (`num`).

---

## 3. Target Definition & Formulation

### 3.1 Raw Target Semantics (`num`)
The raw target attribute `num` represents the angiographic disease status of the patient determined by coronary arteriography:
- `num = 0`: Absence of significant coronary artery disease (< 50% diameter stenosis in all major vessels). Cohort count: 164 (54.13%).
- `num = 1`: Significant stenosis (>= 50% diameter narrowing) in 1 major coronary vessel. Cohort count: 55 (18.15%).
- `num = 2`: Significant stenosis in 2 major coronary vessels. Cohort count: 36 (11.88%).
- `num = 3`: Significant stenosis in 3 major coronary vessels. Cohort count: 35 (11.55%).
- `num = 4`: Significant stenosis in 4 major coronary vessels. Cohort count: 13 (4.29%).

> [!IMPORTANT]
> **Clinical Terminology Guardrail**: The discrete numbers `1, 2, 3, 4` in coronary arteriography indicate the **anatomical count of diseased major vessels** (single-vessel vs. multi-vessel CAD), **NOT** progressive histological "stages" (such as cancer staging). The platform documentation must never refer to these values as disease "stages".

### 3.2 Binary Screening Formulation
For clinical risk screening, the platform formulates a binary classification task:
$$y = \begin{cases} 0, & \text{if } num = 0 \quad (\text{Absence of significant CAD}) \\ 1, & \text{if } num \ge 1 \quad (\text{Presence of significant CAD}) \end{cases}$$

- **Target 0 (Negative / Absence)**: 164 observations (54.13%).
- **Target 1 (Positive / Presence)**: 139 observations (45.87%).
- **Cohort Balance Ratio**: 1.18 : 1 (favorable balance, no severe class imbalance mitigation needed).
- **Integrity Rule**: The binary target $y$ is constructed exclusively within derived modeling DataFrames and pipeline transformers. The raw `num` column in `datasets/raw/heart/` remains strictly unmodified.

---

## 4. Definitive Heart Feature Dictionary

The 13 predictive clinical features are documented below with their exact raw representations, domains, semantic classifications, and transformation requirements:

| Raw Name | Semantic Meaning | Raw Representation | Expected Domain | Inferred Type | Modeling Type | Missing Behavior | Transformation Required | Encoding Required | Scaling Required |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| **`age`** | Patient age in years | Float | 18–120 (Obs: 29–77) | `float64` | Continuous Numerical | None (0) | No | No | Yes (LR) |
| **`sex`** | Biological sex | Float (`0.0`, `1.0`) | `{0.0, 1.0}` | `float64` | Binary Categorical | None (0) | Cast to integer | No (Passthrough) | No |
| **`cp`** | Chest pain presentation | Float (`1.0`–`4.0`) | `{1.0, 2.0, 3.0, 4.0}` | `float64` | Nominal Categorical | None (0) | No | Yes (One-Hot) | No |
| **`trestbps`** | Resting blood pressure on admission | Float (mm Hg) | 60–250 (Obs: 94–200) | `float64` | Continuous Numerical | None (0) | No | No | Yes (LR) |
| **`chol`** | Serum cholesterol | Float (mg/dl) | 80–600 (Obs: 126–564) | `float64` | Continuous Numerical | None (0) | No | No | Yes (LR) |
| **`fbs`** | Fasting blood sugar > 120 mg/dl | Float (`0.0`, `1.0`) | `{0.0, 1.0}` | `float64` | Binary Categorical | None (0) | Cast to integer | No (Passthrough) | No |
| **`restecg`** | Resting electrocardiogram findings | Float (`0.0`–`2.0`) | `{0.0, 1.0, 2.0}` | `float64` | Nominal Categorical | None (0) | No | Yes (One-Hot) | No |
| **`thalach`** | Maximum heart rate during exercise | Float (bpm) | 50–250 (Obs: 71–202) | `float64` | Continuous Numerical | None (0) | No | No | Yes (LR) |
| **`exang`** | Exercise-induced angina | Float (`0.0`, `1.0`) | `{0.0, 1.0}` | `float64` | Binary Categorical | None (0) | Cast to integer | No (Passthrough) | No |
| **`oldpeak`** | ST depression (exercise relative to rest) | Float (mm) | 0.0–10.0 (Obs: 0.0–6.2) | `float64` | Continuous Numerical | None (0) | No | No | Yes (LR) |
| **`slope`** | Peak exercise ST segment slope | Float (`1.0`–`3.0`) | `{1.0, 2.0, 3.0}` | `float64` | Coded Categorical / Ordinal | None (0) | No | Yes (One-Hot / Ordinal) | No |
| **`ca`** | Major vessels (0–3) colored by fluoroscopy | String / Object (`'0.0'`–`'3.0'`, `'?'`) | `{0, 1, 2, 3}` | `object` | Discrete Count Numerical | 4 missing (`'?'`) | Replace `'?'` with NaN; Impute | Evaluate (Numeric vs One-Hot) | Optional |
| **`thal`** | Thallium perfusion scintigraphy | String / Object (`'3.0'`, `'6.0'`, `'7.0'`, `'?'`) | `{3, 6, 7}` | `object` | Nominal Categorical | 2 missing (`'?'`) | Replace `'?'` with NaN; Impute | Yes (One-Hot) | No |

---

## 5. Feature-Type Classification & Modeling Rationale

A critical error in tabular clinical machine learning is assuming that any feature stored as a number is continuous. The Cleveland dataset encodes discrete states as `float64` in text. We systematically classify features into four modeling families:

### 1. Continuous Numerical (5 features)
- **`age`**, **`trestbps`**, **`chol`**, **`thalach`**, **`oldpeak`**
- *Rationale*: These represent true biological or physical continuous scales. Arithmetic distances, standard deviations, and ratios are mathematically meaningful.
- *Modeling Treatment*: Subject to scaling for distance-sensitive and linear algorithms; retained unscaled for decision tree ensembles.

### 2. Binary Categorical (3 features)
- **`sex`** (`0` = Female, `1` = Male)
- **`fbs`** (`0` = <=120 mg/dl, `1` = >120 mg/dl)
- **`exang`** (`0` = No, `1` = Yes)
- *Rationale*: Two mutually exclusive states representing demographics or threshold indicators.
- *Modeling Treatment*: Maintained directly as integer binary indicators $\{0, 1\}$. One-hot encoding is redundant and adds unnecessary collinear dimensions.

### 3. Nominal Categorical (3 features)
- **`cp`** (Chest pain type: 1 = Typical Angina, 2 = Atypical Angina, 3 = Non-Anginal, 4 = Asymptomatic)
- **`restecg`** (Resting ECG: 0 = Normal, 1 = ST-T abnormality, 2 = LV hypertrophy)
- **`thal`** (Thallium scintigraphy: 3 = Normal, 6 = Fixed Defect, 7 = Reversible Defect)
- *Rationale*: Discrete qualitative diagnostic categories. The numbers are arbitrary classification codes. Treating `cp` as a continuous integer falsely assumes that Asymptomatic ($4$) is "four times" Typical Angina ($1$), and that Atypical Angina ($2$) lies linearly between them.
- *Modeling Treatment*: Must be One-Hot Encoded.

### 4. Special Case Features
- **`ca`** (Fluoroscopy vessel count, 0–3): Non-negative count feature. Evaluated as discrete count vs. nominal categorical (see Section 9).
- **`slope`** (ST slope, 1–3): Clinically ordered physiological trajectory. Evaluated as nominal one-hot vs. ordinal integer (see Section 10).

---

## 6. Missing-Value Handling Design

### 6.1 Missingness Characterization
- Missing values exist exclusively as `'?'` characters in two specialized diagnostic columns:
  - `ca`: 4 missing observations (1.32% of records; rows 166, 192, 287, 302).
  - `thal`: 2 missing observations (0.66% of records; rows 87, 266).
- Total missingness in cohort: 6 values across 6 distinct rows (1.98% of cohort).

### 6.2 Ingestion & Sentinel Mapping
Upon DataFrame ingestion from the raw file, all occurrences of `'?'` must be converted to standard `np.nan` prior to pipeline transformation:
```python
df.replace('?', np.nan, inplace=True)
```

### 6.3 Candidate Imputation Strategies
Because missingness is very low (< 2%), listwise row deletion would discard valid patient data, while complex synthetic imputation runs the risk of overfitting. We design two candidate strategies:

| Strategy | `ca` (Count Feature) | `thal` (Nominal Feature) | Justification |
| :--- | :--- | :--- | :--- |
| **Strategy A (Standard Baseline)** | `SimpleImputer(strategy='median')` | `SimpleImputer(strategy='most_frequent')` | Median preserves the discrete integer count nature of `ca` (median = 0); mode preserves the valid category for `thal` (mode = 3.0 / Normal). Simple, robust, zero variance leakage. |
| **Strategy B (Iterative / Model-Based)** | `IterativeImputer` (MICE) | `SimpleImputer(strategy='most_frequent')` | Leverages multivariate correlations with age and resting vitals. Must be evaluated empirically in Phase 4. |

### 6.4 Leakage Prevention Protocol
Imputers must call `.fit()` **strictly and exclusively** on the training fold during cross-validation, and call `.transform()` on validation/test folds:
$$\hat{\mu}_{\text{train}} = \text{median}(X_{\text{train}}), \quad X_{\text{val}}^{\text{imputed}} = \text{impute}(X_{\text{val}}, \hat{\mu}_{\text{train}})$$
Test set observations must never inform the imputed values.

---

## 7. Categorical Encoding Design

### 7.1 Candidate Encoders for Nominal Features (`cp`, `restecg`, `thal`)

We evaluate `OneHotEncoder(handle_unknown='ignore')` under two configurations:

```
Configuration 1: drop='first' (Reference Dummy Encoding)
  - cp: 3 dummy columns (drops category 1 as reference)
  - restecg: 2 dummy columns (drops category 0 as reference)
  - thal: 2 dummy columns (drops category 3 as reference)

Configuration 2: drop=None (Full One-Hot Encoding)
  - cp: 4 binary indicator columns
  - restecg: 3 binary indicator columns
  - thal: 3 binary indicator columns
```

### 7.2 Model-Specific Implications of Dummy Encoding

| Dimension | `drop='first'` | `drop=None` (Full One-Hot) |
| :--- | :--- | :--- |
| **Logistic Regression** | **Preferred**: Eliminates exact multicollinearity (dummy variable trap). Yields stable coefficient estimation and well-conditioned matrix inversion. | Causes rank deficiency if unregularized; L2 regularization accommodates full dummy, but coefficients become dependent on regularization strength. |
| **Random Forest** | Suboptimal: Asymmetric representation. When the dropped category is active, all dummy indicators are 0, forcing trees to split on negation. | **Preferred**: Trees can isolate any category with a single split on its dedicated binary indicator column. |
| **XGBoost** | Suboptimal: Asymmetric sparsity patterns for the reference class. | **Preferred**: Symmetric gradient calculations across all levels. |
| **SHAP Attribution** | Attributions reflect contrast relative to the omitted baseline category, complicating intuitive patient explanations. | **Preferred**: Every category receives a direct attribution score (e.g. `cp_Asymptomatic = +0.14`), greatly simplifying clinical explanation. |

*Phase 4 Decision*: The pipeline architecture should parameterize `drop` so that Logistic Regression can use `drop='first'` while tree ensembles can evaluate `drop=None`.

---

## 8. Binary Feature Design (`sex`, `fbs`, `exang`)

- **Analysis**:
  - `sex`: `0.0` (Female, 32.01%), `1.0` (Male, 67.99%).
  - `fbs`: `0.0` (Fasting Blood Sugar <= 120, 85.15%), `1.0` (Fasting Blood Sugar > 120, 14.85%).
  - `exang`: `0.0` (Exercise Angina Absent, 67.33%), `1.0` (Exercise Angina Present, 32.67%).
- **Design Decision**: These features are cast to integer binary indicators $\{0, 1\}$ and passed through without one-hot encoding:
  - Preserves natural zero-baseline semantics.
  - Prevents expanding the feature space with collinear inverse columns (e.g., `sex_0` and `sex_1`).
  - Directly interpretable by linear coefficients and tree split criteria.

---

## 9. `ca` Feature Representation Decision

`ca` represents the number of major coronary vessels (0, 1, 2, or 3) colored by fluoroscopy.

### Candidate Treatments
1. **Approach A: Discrete Count Numerical (Continuous / Count)**:
   - Preserves natural clinical hierarchy: having 3 vessels with stenosis conveys strictly higher anatomical obstruction than 1 vessel.
   - Requires only 1 feature column.
   - Imputed via median (median = 0).
2. **Approach B: Nominal One-Hot Encoded (Categories 0, 1, 2, 3)**:
   - Does not assume constant linear step increase in risk per additional vessel.
   - Expands `ca` into 4 dummy columns (or 3 with `drop='first'`).
   - Risk of sparse splits on category `3` (N = 20, 6.6%).

*Design Recommendation*: Approach A (Count Numerical) serves as the primary baseline representation because vessel count has strong monotonic clinical meaning. Approach B will be benchmarked as an alternative during Phase 5 experimentation.

---

## 10. `slope` Feature Representation Decision

`slope` represents the slope of the peak exercise ST segment:
- `1.0`: Upsloping (normal hemodynamic response during stress)
- `2.0`: Flat (equivocal / mild ischemia)
- `3.0`: Downsloping (severe myocardial ischemia)

### Why Integer Codes Must Not Be Assumed Linear
Although the clinical severity increases monotonically ($1 < 2 < 3$), the numerical distance between Upsloping ($1$) and Flat ($2$) is not necessarily equal to the distance between Flat ($2$) and Downsloping ($3$) in terms of log-odds of CAD.

### Candidate Treatments
1. **Approach A: Nominal One-Hot Encoding**:
   - Encoded as 3 levels: `slope_upsloping`, `slope_flat`, `slope_downsloping`.
   - Allows the estimator to assign independent non-linear weights to flat vs. downsloping segments.
2. **Approach B: Ordinal Integer Encoding (1, 2, 3)**:
   - Preserves order with a single column.
   - Imposes linear step constraint on linear models.

*Design Recommendation*: Approach A (Nominal One-Hot) is designated as the primary baseline candidate to avoid unverified linear distance assumptions.

---

## 11. Numerical Transformation & Scaling Design

Features evaluated: `age`, `trestbps`, `chol`, `thalach`, `oldpeak`.

### 11.1 Algorithm Sensitivity
- **Logistic Regression**: Strictly requires scaling. Gradient descent and L1/L2 regularization penalize coefficients based on magnitude; unscaled features with large variance (e.g. `chol` ranging from 126 to 564) would artificially dominate features with small variance (e.g. `oldpeak` ranging from 0.0 to 6.2).
- **Random Forest & XGBoost**: Completely invariant to monotonic scaling because tree splits depend strictly on ordering, not metric distance.

### 11.2 Candidate Scalers
1. **`StandardScaler`**:
   $$z = \frac{x - \mu}{\sigma}$$
   Centering at zero with unit variance. Assumes approximately normal distribution. Sensitive to extreme values.
2. **`RobustScaler`**:
   $$z = \frac{x - \text{median}}{\text{IQR}}$$
   Centering at median, scaled by Interquartile Range ($Q_3 - Q_1$). Highly resistant to outlier distortion in skewed distributions like `chol` (skewness = +1.14) and `oldpeak` (skewness = +1.27).

*Phase 4 Plan*: The pipeline will support interchangeable scaler injection so that both `StandardScaler` and `RobustScaler` can be compared during Phase 5.

---

## 12. Outlier Policy

### 12.1 Observed High Values
During Phase 0, the following extreme clinical values were identified:
- `chol = 564 mg/dl` (patient index 48; next highest is 417 mg/dl).
- `trestbps = 200 mm Hg` (patient index 126).
- `oldpeak = 6.2 mm` (patient index 91).

### 12.2 Policy Rules
1. **Prohibition of Blind Deletion**: Extreme biological observations must **NOT** be deleted or trimmed simply because they lie outside $1.5 \times \text{IQR}$ or 3 standard deviations.
2. **Clinical Plausibility**:
   - Serum cholesterol of 564 mg/dl is clinically valid and characteristic of severe familial hypercholesterolemia, a profound risk factor for coronary disease.
   - Blood pressure of 200 mm Hg represents acute hypertensive crisis, commonly encountered in cardiac referral cohorts.
   - Deleting these observations would induce severe selection bias by removing the highest-risk patients.
3. **Algorithmic Mitigation**: Instead of data truncation, algorithmic robustness (e.g., `RobustScaler` for linear models, or decision tree ensembles which are immune to outliers) will be used.

---

## 13. User Input to Model Input Contract

The end-to-end field mapping from user interface to model feature vector:

| # | User-Facing Field (React) | Structured DTO (Spring Boot) | ML Pydantic Schema (FastAPI) | Pipeline Stage | Model Input Vector Column |
| :-: | :--- | :--- | :--- | :--- | :--- |
| 1 | "Age" (Years: 18–120) | `Integer age` | `age: int` | Scaler (`Standard` / `Robust`) | `num__age` |
| 2 | "Sex" ("Male" / "Female") | `String gender` | `sex: float` (`1.0` / `0.0`) | Passthrough (Binary) | `bin__sex` |
| 3 | "Chest Pain Type" (4 choices) | `Integer chestPainType` (1–4) | `cp: float` (1.0–4.0) | OneHotEncoder | `cat__cp_1`, `cat__cp_2`, `cat__cp_3`, `cat__cp_4` |
| 4 | "Resting Blood Pressure" (mm Hg) | `Double restingBp` | `trestbps: float` | Scaler (`Standard` / `Robust`) | `num__trestbps` |
| 5 | "Serum Cholesterol" (mg/dL) | `Double cholesterol` | `chol: float` | Scaler (`Standard` / `Robust`) | `num__chol` |
| 6 | "Fasting Blood Sugar > 120" | `Boolean fastingBs` | `fbs: float` (`1.0` / `0.0`) | Passthrough (Binary) | `bin__fbs` |
| 7 | "Resting ECG" (3 choices) | `Integer restingEcg` (0–2) | `restecg: float` (0.0–2.0) | OneHotEncoder | `cat__restecg_0`, `cat__restecg_1`, `cat__restecg_2` |
| 8 | "Max Heart Rate" (bpm) | `Double maxHeartRate` | `thalach: float` | Scaler (`Standard` / `Robust`) | `num__thalach` |
| 9 | "Exercise Induced Angina" | `Boolean exerciseAngina` | `exang: float` (`1.0` / `0.0`) | Passthrough (Binary) | `bin__exang` |
| 10 | "ST Depression (Oldpeak)" (mm) | `Double oldpeak` | `oldpeak: float` | Scaler (`Standard` / `Robust`) | `num__oldpeak` |
| 11 | "ST Segment Slope" (3 choices) | `Integer stSlope` (1–3) | `slope: float` (1.0–3.0) | OneHotEncoder | `cat__slope_1`, `cat__slope_2`, `cat__slope_3` |
| 12 | "Fluoroscopy Vessels" (0–3) | `Integer majorVessels` (0–3) | `ca: float` (0.0–3.0) | Median Imputer + Scaler | `num__ca` |
| 13 | "Thallium Stress Test" (3 choices) | `Integer thalliumStress` (3, 6, 7) | `thal: float` (3.0, 6.0, 7.0) | Mode Imputer + OneHot | `cat__thal_3`, `cat__thal_6`, `cat__thal_7` |

---

## 14. Data Leakage Prevention Rules

To guarantee strict scientific integrity:

1. **Isolation of Test Set**: An 80/20 stratified hold-out test set must be set aside **prior** to computing any transformation parameters.
2. **Strict Fit-on-Train Discipline**:
   - `SimpleImputer.fit()` must be executed exclusively on training folds.
   - `StandardScaler.fit()` (mean, standard deviation) must be computed exclusively on training folds.
   - `OneHotEncoder.fit()` (category discovery) must be executed exclusively on training folds.
3. **No Target Leakage**: Target labels ($y$) must never be used as inputs or feature priors.
4. **Pipeline Encapsulation**: All transformations must be bundled inside `sklearn.pipeline.Pipeline` or `sklearn.compose.ColumnTransformer`. No standalone pre-processed CSVs may be used to train models.

---

## 15. Train / Test Split Strategy

- **Hold-Out Split**:
  - Sample Size: 80% Train (242 samples) / 20% Test (61 samples).
  - Split Type: `StratifiedShuffleSplit` or `train_test_split(..., stratify=y, random_state=42)`.
  - Target Preservation: Preserves ~54% Negative / 46% Positive across both sets.
- **Cross-Validation Scheme (for Phase 5 Experimentation)**:
  - **Repeated Stratified K-Fold**: 5 folds, 5 repeats (`n_splits=5, n_repeats=5, random_state=42`).
  - Ensures that every training fold undergoes complete internal preprocessing fit, preventing validation fold contamination.

---

## 16. Model Compatibility Matrix

| Requirement / Property | Logistic Regression (L1/L2) | Random Forest | XGBoost |
| :--- | :--- | :--- | :--- |
| **Numerical Feature Scaling** | **Mandatory** (`StandardScaler` or `RobustScaler`). | Invariant (No scaling needed). | Invariant (No scaling needed). |
| **Categorical Encoding** | **Mandatory One-Hot** (Requires `drop='first'` to prevent singular matrix). | **One-Hot** (`drop=None` preferred for symmetric tree splits). | **One-Hot** (`drop=None` preferred). |
| **Missing-Value Handling** | **Required** (Cannot process NaN). | **Required** (Scikit-Learn RF requires imputation). | Native handling supported, but explicit imputation preferred for consistency. |
| **Outlier Sensitivity** | High (Extreme outliers pull decision boundary; requires regularization or `RobustScaler`). | Low (Monotonic tree partitioning is robust to extreme outliers). | Moderate (Gradients can be affected by extreme outliers). |
| **Interpretability** | Linear log-odds coefficients ($\beta_i$). | Feature importances (Gini / MDI) + SHAP TreeExplainer. | Gain / Weight importances + SHAP TreeExplainer. |
| **Collinearity Sensitivity** | High (Correlated features inflate variance of coefficients). | Low (Random subspace sampling mitigates collinearity). | Low (Selects single split among correlated features). |

---

## 17. SHAP Compatibility & Feature Naming Design

When SHAP explainability is implemented in Phase 8, the explainer operates on the post-transformed feature matrix ($X_{\text{transformed}}$).

### Feature Expansion & Naming
- Applying `ColumnTransformer` expands the 13 raw predictors into approximately 20–23 numeric feature columns (depending on one-hot `drop` strategy).
- Phase 4 must generate a definitive feature name mapping via `ColumnTransformer.get_feature_names_out()`:
  ```python
  transformed_names = preprocessor.get_feature_names_out()
  ```
- Phase 8 will map transformed column names back to clean clinical labels:
  - `cat__cp_4.0` $\rightarrow$ *"Chest Pain: Asymptomatic"*
  - `num__chol` $\rightarrow$ *"Serum Cholesterol (mg/dL)"*
  - `cat__thal_7.0` $\rightarrow$ *"Thallium Defect: Reversible Defect"*

---

## 18. Derived Dataset Policy

- **Reproducibility Over Static Files**: The platform prioritizes declarative code pipelines over intermediate static CSVs.
- **Derived Data Directory**: If a cleaned/imputed dataset is generated for exploratory visualization in `datasets/processed/heart/`, it must:
  1. Be clearly named `heart_derived_exploratory.csv`.
  2. Contain an accompanying metadata file documenting the exact transformation script and hash.
  3. Never overwrite or replace `datasets/raw/heart/processed.cleveland.data`.
  4. Never contain test-set-derived transformations.

---

## 19. Feature Engineering Policy

- **No Medical Formulas**: The platform strictly prohibits inventing ad-hoc cardiovascular risk scores (e.g. combining blood pressure and cholesterol into a custom ratio) without established clinical literature backing.
- **No Synthetic Rows**: No SMOTE, GAN, or synthetic patient vectors will be generated prior to empirical evidence of necessity.
- **Simplicity First**: A clean, leakage-free transformation of the 13 clinically verified predictors is scientifically preferable to complex artificial feature combinations.

---

## 20. Phase 4 Handoff Requirements

Phase 4 (**Heart Preprocessing & Dataset Preparation**) must implement:
1. `load_heart_raw_data()`: Ingests `datasets/raw/heart/processed.cleveland.data`, assigns standard 14 column names, and converts `'?'` to `np.nan`.
2. `create_binary_target()`: Derives $y = (num \ge 1).astype(int)$ without altering the raw DataFrame.
3. `create_train_test_split()`: Generates stratified 80/20 train/test splits with fixed seed (`random_state=42`).
4. `build_heart_preprocessor()`: Constructs the production `ColumnTransformer` bundling:
   - Median/Mode imputation.
   - Scaler candidates (`StandardScaler`, `RobustScaler`).
   - One-hot encoder (`OneHotEncoder(handle_unknown='ignore')`).
5. Preprocessing test suite: Verifies zero data leakage, correct transformed output shapes, and valid feature naming.
6. **Strict Handoff Boundary**: Phase 4 will **NOT** train final classifiers or select production models.

---

## 21. Verification Checklist

- [x] Specification document created at [`docs/HEART_FEATURE_DATA_DESIGN.md`](file:///d:/smart-healthcare-platform/docs/HEART_FEATURE_DATA_DESIGN.md).
- [x] All 13 predictors comprehensively detailed in feature dictionary.
- [x] Target definition strictly separates raw multi-vessel count from binary screening target.
- [x] Prohibition of "disease stage" terminology strictly enforced.
- [x] Missing value strategy explicitly requires fit-on-train discipline.
- [x] `ca` and `slope` representations documented with alternatives for empirical validation.
- [x] Outlier retention policy defined (no arbitrary truncation).
- [x] Model compatibility matrix completed across LR, RF, and XGBoost.
- [x] SHAP naming and feature expansion considerations addressed.
- [x] Raw dataset checksum re-verified and identical (18,461 bytes).
- [x] Liver Disease implementation 100% untouched.
- [x] No ML models trained, no code modified, no services restarted.
