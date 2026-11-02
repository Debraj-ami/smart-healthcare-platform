# Data Preprocessing Specification

## 1. Purpose

This document defines the dataset-specific preprocessing standards, target formulation, data leakage prevention protocols, validation splitting strategies, and performance evaluation considerations for the four disease modules of the Smart Healthcare Platform:

- **Heart Disease**
- **Diabetes**
- **Parkinson's Disease**
- **Liver Disease**

The primary objective is to establish a rigorous, reproducible, and leakage-safe machine learning engineering framework prior to model training and experimentation.

---

## 2. General Preprocessing Principles

1. **Raw Data Immutability**: Raw datasets stored in `datasets/raw/` must remain completely unmodified and treated as read-only source files.
2. **Reproducibility**: All data cleaning, encoding, imputation, and transformation operations must be deterministic, scriptable, and reproducible across environments.
3. **Explicit Handling of Missing Values**: Missing values, whether encoded as nulls, sentinels (e.g., `'?'`), or hidden physiological zeros, must be identified and handled through documented strategies.
4. **Appropriate Categorical Encoding**: Categorical features must be encoded appropriately (e.g., binary mapping or one-hot encoding) without introducing artificial ordinality where none exists.
5. **Strict Fit-on-Train Discipline**: All transformations (imputers, scalers, encoders) must be fitted exclusively on the training dataset or training folds.
6. **No Test-Set Contamination**: Test-set or validation-fold statistics (means, medians, frequencies, standard deviations) must never inform or influence preprocessing parameters.
7. **Zero Target Leakage**: Target labels must never be used to derive or inform feature transformations.
8. **Balanced Evaluation Metrics**: Model evaluation must not rely solely on simple accuracy, especially in the presence of class imbalance; domain-appropriate discrimination and calibration metrics must be used.
9. **Modularity**: Dataset-specific preprocessing must be modular and tailored to the unique physiological and structural characteristics of each disease dataset.
10. **Scientific Precision & Non-Causality**: Exploratory observations (such as extreme laboratory values, statistical associations, or feature correlations) describe empirical patterns within the cohorts and do not by themselves establish clinical causation. The project must avoid unverified medical claims that were not established by the dataset inspection.

---

## 3. Heart Disease

### Inspection Findings
- **Sample Size**: 303 patient observations.
- **Dimensionality**: 13 input features and 1 target column (`num`).
- **Sentinel Missing Values**: Sentinel strings (`'?'`) exist in `ca` (4 occurrences) and `thal` (2 occurrences), causing pandas to infer these columns as generic `object` types.
- **Target Distribution**: The raw target `num` contains discrete integer values from `0` to `4`:
  - `0`: 164 records (54.13%) — Absence of significant coronary artery disease (< 50% narrowing).
  - `1`: 55 records (18.15%) — Stage 1 presence.
  - `2`: 36 records (11.88%) — Stage 2 presence.
  - `3`: 35 records (11.55%) — Stage 3 presence.
  - `4`: 13 records (4.29%) — Stage 4 presence.

### Preprocessing & Modeling Formulation
- **Binary Formulation**: The platform will evaluate the clinically established binary classification formulation:
  - `0`: Absence of heart disease (`num == 0`)
  - `1`: Presence of heart disease (`num >= 1`)
- **Preservation of Raw Target**: The original multi-stage target values in the raw dataset must not be overwritten or modified.
- **Missing Value Handling**: Sentinel `'?'` characters in `ca` and `thal` must be converted to standard nulls during pipeline ingestion and imputed using an appropriate strategy (e.g., median, mode, or iterative imputation). The final strategy will be evaluated empirically during model experiments.
- **Clinical Variable Semantics & Representation**: The raw dataset represents several clinical variables as floating-point numbers, but these encompass distinct semantic types:
  - `sex`: Binary (`1.0` = male, `0.0` = female).
  - `cp`: Nominal categorical (4 chest pain types: typical angina, atypical angina, non-anginal pain, asymptomatic).
  - `fbs`: Binary fasting blood sugar indicator (`1.0` = true, `0.0` = false).
  - `restecg`: Nominal categorical (normal, ST-T abnormality, left ventricular hypertrophy).
  - `exang`: Binary exercise-induced angina (`1.0` = yes, `0.0` = no).
  - `slope`: Ordinal/categorical peak exercise ST segment slope (upsloping, flat, downsloping).
  - `thal`: Nominal categorical thallium scintigraphy defect (normal, fixed defect, reversible defect).
  - `ca`: Count-like numerical feature (number of major vessels colored by fluoroscopy, 0–3).
  The final preprocessing representation must be selected according to the specific semantic meaning of each feature rather than treating every coded value as continuous or every feature as nominal. The raw dataset remains unmodified.
- **Validation Split Strategy**: Standard **Stratified K-Fold Cross-Validation** must be used to preserve the ~54% / 46% class balance across all folds.

---

## 4. Diabetes

### Inspection Findings
- **Sample Size**: 768 patient observations.
- **Dimensionality**: 8 input features and 1 target column (`Outcome`).
- **Class Distribution**: 500 negative records (`Outcome = 0`, 65.10%) and 268 positive records (`Outcome = 1`, 34.90%).
- **Hidden Missing Values (Zero Values)**: Standard missing-value checks report 0 nulls, but physiologically impossible zero values exist in five vital features:
  - `Insulin`: 374 zero values (48.70% missing)
  - `SkinThickness`: 227 zero values (29.56% missing)
  - `BloodPressure`: 35 zero values (4.56% missing)
  - `BMI`: 11 zero values (1.43% missing)
  - `Glucose`: 5 zero values (0.65% missing)

### Preprocessing & Modeling Formulation
- **Handling Hidden Missingness**: A living individual cannot have zero blood glucose, blood pressure, skinfold thickness, or BMI. These zero values represent missing observations and must be treated as such in the preprocessing pipeline.
- **Raw Data Integrity**: Zero values must NOT be overwritten or replaced directly in the raw `datasets/raw/diabetes/diabetes.csv` file.
- **Investigation of Imputation Approaches**: Given that nearly half the dataset lacks `Insulin` and nearly a third lacks `SkinThickness`, imputation techniques (e.g., median imputation, KNN imputation, or missing indicator flags) must be evaluated systematically within the pipeline.
- **Validation Split Strategy**: **Stratified K-Fold Cross-Validation** must be employed to maintain the 65.10% / 34.90% class proportion across training and evaluation splits.

---

## 5. Parkinson's Disease

### Inspection Findings
- **Sample Size**: 195 sustained vowel phonation recordings.
- **Dimensionality**: 22 continuous acoustic features, 1 identifier column (`name`), and 1 binary target (`status`).
- **Class Distribution**:
  - Record-Level (195 rows): 147 PD (`status = 1`, 75.38%) vs. 48 Healthy (`status = 0`, 24.62%).
  - Subject-Level (32 individuals): 24 PD patients (75.00%) vs. 8 Healthy individuals (25.00%).
- **Subject Identifier & Repeated Observations**:
  - The `name` column encodes subject identities (e.g., `phon_R01_S01_1` belongs to subject `phon_R01_S01`).
  - There are exactly **32 distinct subjects**; 29 subjects contributed 6 recordings each, and 3 subjects contributed 7 recordings each.

### Critical Leakage Prevention & Splitting Protocol
- **Prohibition of Identifier Usage**: The `name` column is an administrative identifier and must never be fed into any machine learning model as a predictive feature.
- **Strict Prohibition of Random Row-Level Splitting**: Standard random train/test splitting or random K-fold cross-validation **must NOT be used**. Because acoustic vocal anatomy is highly individual, random splitting would place voice recordings from the same subject into both training and test sets. This creates severe data leakage, causing models to memorize individual speaker identities rather than generalizable dysphonia patterns.
- **Mandatory Group-Aware Splitting**: Validation and testing **must enforce Subject-Aware Group Splitting** (e.g., `GroupKFold` or `StratifiedGroupKFold` grouped by extracted `subject_id`). Under no circumstances may observations from the same subject appear in both training and test folds.
- **Evaluation Integrity**: Subject-level leakage prevention is mandatory for credible, generalizable Parkinson's risk assessment.

---

## 6. Liver Disease

### Inspection Findings
- **Sample Size**: 583 patient records.
- **Dimensionality**: 10 input features (9 numerical, 1 categorical) and 1 target variable (`Selector`).
- **Target Distribution**:
  - Class `1` (Liver Disease Patient): 416 observations (71.36%)
  - Class `2` (Non-Liver Control): 167 observations (28.64%)
- **Missing Values**: Exactly 4 missing values in `A_G_Ratio` (Albumin-to-Globulin Ratio, 0.69%).
- **Duplicate Records**: Exactly 13 duplicate rows (2.23% of records; 26 rows involved in duplicate pairs).
- **Categorical Feature**: `Gender` has 441 Males (75.64%) and 142 Females (24.36%).

### Preprocessing & Modeling Formulation
- **Duplicate Investigation**: Exact duplicate observations must be investigated and resolved prior to final cross-validation splitting to eliminate cross-fold replication leakage.
- **Missing Ratio Handling**: Missing values in `A_G_Ratio` must be handled within the preprocessing pipeline (e.g., median imputation or formulaic derivation from `Albumin` and `Total_Proteins`).
- **Categorical Encoding**: Binary encoding must be applied to `Gender` (`Male` / `Female`).
- **Raw Data Immutability**: The raw dataset files must remain untouched.
- **Experimental Selection**: Specific techniques (such as SMOTE, RobustScaler, or specific imputation algorithms) must **not be prescribed prematurely**; they will be benchmarked and evaluated during empirical training experiments.
- **Validation Split Strategy**: **Stratified K-Fold Cross-Validation** must be used to preserve the 71.36% / 28.64% class distribution.

---

## 7. Scaling Strategy

Feature scaling requirements depend directly on the mathematical characteristics of the candidate algorithms:

- **Candidate Models for the Platform**:
  1. **Logistic Regression** (Linear model — highly sensitive to feature magnitudes; requires scaling).
  2. **Random Forest** (Tree ensemble — monotonic decision boundaries; invariant to monotonic feature scale).
  3. **XGBoost** (Gradient boosted trees — invariant to numerical scale).

- **Standardization & Robustness**:
  - For algorithms sensitive to scale, standard scaling (`StandardScaler`) or robust quantile scaling (`RobustScaler`) will be evaluated.
  - Extreme biological values (e.g., acute bilirubin or liver enzyme spikes) must be accommodated without distorting normal clinical ranges.
- **Pipeline Encapsulation**: All scaling transformations must be embedded inside `scikit-learn` pipelines so that scaling parameters (mean, variance, medians) are computed strictly on training folds and applied downstream without leakage.

---

## 8. Class Imbalance

Several platform datasets exhibit notable class imbalance:
- **Liver Disease**: ~2.49 : 1 ratio (71.36% positive).
- **Parkinson's Disease**: ~3.06 : 1 record ratio (75.38% positive) / 3.0 : 1 subject ratio.
- **Diabetes**: ~1.87 : 1 ratio (65.10% negative).

### Evaluation Implications
- **Inadequacy of Accuracy**: Standard accuracy is deceptive in imbalanced medical cohorts (e.g., predicting all patients as positive on the Liver dataset yields a baseline accuracy of 71.36% without learning clinical patterns).
- **Candidate Evaluation Metrics**:
  - **ROC-AUC** (Area Under the Receiver Operating Characteristic Curve)
  - **PR-AUC / Average Precision** (Area Under the Precision-Recall Curve — critical for imbalanced cohorts)
  - **Recall / Sensitivity** (Ability to identify true disease cases, vital for medical screening)
  - **Specificity** (Ability to identify true healthy controls)
  - **F1-Score / Balanced F-Beta**
  - **Confusion Matrix**
  - **Brier Score & Calibration Curves** (To verify that predicted risk probabilities reflect true event likelihoods)
- **Experimental Objectivity**: No model will be declared superior prior to empirical benchmarking across these comprehensive metrics.

---

## 9. Data Leakage Prevention

To maintain strict scientific and clinical integrity, the following rules must be enforced across all pipelines:

1. **Transformations Fitted Exclusively on Training Folds**: Imputers, scalers, and encoders must only call `.fit()` or `.fit_transform()` on training folds.
2. **Strict Isolation of Hold-Out Test Sets**: Test sets must be isolated prior to any processing and used solely for final evaluation via `.transform()`.
3. **No Target Contamination**: Target values must never be used in feature imputation or transformation.
4. **No Premature Resampling**: Resampling methods (e.g., SMOTE, random undersampling) must NEVER be applied to the entire dataset before splitting. Resampling is only permissible within training folds during cross-validation.
5. **Subject-Level Group Isolation**: For Parkinson's Disease, all recordings from a single patient must remain strictly within either the training fold or the testing fold.
6. **Exclusion of Administrative Identifiers**: Identifiers (e.g., `name` in Parkinson's) must be dropped before feature matrix construction.

---

## 10. Evaluation Strategy

The standard evaluation protocol across all four disease prediction modules consists of:

- **Primary Validation Scheme**:
  - **Heart Disease, Diabetes, Liver Disease**: Repeated Stratified K-Fold Cross-Validation (e.g., 5-fold or 10-fold CV).
  - **Parkinson's Disease**: Group-Aware Cross-Validation (`GroupKFold` or `StratifiedGroupKFold`) grouped by subject.
- **Independent Hold-Out Test Split**: Where sample size permits, a hold-out test split (stratified by target, and group-aware for Parkinson's) will be set aside for unbiased final assessment.
- **Metric Suite**: Simultaneous reporting of Precision, Recall, Specificity, F1-Score, ROC-AUC, PR-AUC, and Confusion Matrices.
- **Probability Calibration**: Calibration methods (such as Platt scaling / sigmoid calibration or Isotonic Regression) must themselves be fitted strictly using appropriate training or validation data (e.g., via internal cross-validation on the training set). The final hold-out test set must remain completely untouched until final evaluation to ensure unbiased performance verification.

---

## 11. Model Experimentation

For each of the four disease modules, model development will systematically benchmark three baseline candidate algorithms:

1. **Logistic Regression** (Interpretable linear baseline with L1/L2 regularization)
2. **Random Forest** (Nonlinear bagging ensemble robust to outliers and variance)
3. **XGBoost** (Gradient boosting ensemble optimized for tabular clinical data)

Each candidate model will be evaluated through its dedicated, leakage-safe pipeline. No algorithm will be assumed optimal a priori; model selection will be governed entirely by empirical validation results.

---

## 12. Research Reproducibility

To ensure complete experimental reproducibility:
- **Seed Management**: Fixed pseudorandom seeds (`random_state`) must be documented and used across all stochastic splitters and estimators.
- **Explicit Versioning**: Dataset sources, dates, row counts, and checksums must be recorded.
- **Declarative Pipeline Specifications**: Feature names, imputation settings, encoder mappings, and hyperparameter grids must be fully serialized.
- **Artifact Tracking**: Serialized trained pipelines (`joblib` / `pickle`) and structured experiment evaluation metrics will be versioned alongside code.

---

## 13. Important Principle

> **"Preprocessing decisions must be derived from dataset inspection, implemented without data leakage, and validated experimentally rather than assumed in advance."**
