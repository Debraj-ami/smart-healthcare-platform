# Heart Disease — Dataset Understanding, Provenance & Data Foundation (Phase 0)

**Document Version**: 1.0.0  
**Phase**: Phase 0 — Disease-Specific Setup, Dataset Understanding, Provenance and Data Foundation  
**Disease Module**: Coronary Heart Disease Risk Assessment  
**Reference Notebook**: [`notebooks/04_heart_dataset_analysis.ipynb`](file:///d:/smart-healthcare-platform/notebooks/04_heart_dataset_analysis.ipynb)  
**Status**: Completed & Verified  

---

## 1. Executive Summary

This document establishes the data foundation for the Heart Disease risk prediction module of the Smart Healthcare Platform. The analysis operates directly on the immutable source dataset located at `datasets/raw/heart/processed.cleveland.data`.

### Key Metrics Summary
- **Cohort Size**: 303 patient observations.
- **Attributes**: 14 (13 predictive clinical/demographic features, 1 angiographic target).
- **Missing Data**: 6 total missing values across 2 features (`ca`: 4, `thal`: 2) represented via `'?'` sentinels (1.98% of rows). Zero standard IEEE NaNs.
- **Duplicate Records**: 0 exact duplicates (0.0%).
- **Raw Target (`num`)**: Multiclass integer (0: 164 [54.13%], 1: 55 [18.15%], 2: 36 [11.88%], 3: 35 [11.55%], 4: 13 [4.29%]).
- **Exploratory Binary Screening Target**: 164 Negative (`num=0`, 54.13%) vs. 139 Positive (`num>=1`, 45.87%). Imbalance ratio: 1.18 : 1.
- **Source Integrity**: SHA-256 and MD5 cryptographic hashes verified before and after analysis — 100% identical.

---

## 2. Dataset Provenance

- **Dataset Name**: Heart Disease Database (Cleveland Database)
- **Source Repository**: UCI Machine Learning Repository
- **Creators / Principal Investigators**:
  1. **Robert Detrano, M.D., Ph.D.** — V.A. Medical Center, Long Beach and Cleveland Clinic Foundation
  2. **Andras Janosi, M.D.** — Hungarian Institute of Cardiology, Budapest
  3. **William Steinbrunn, M.D.** — University Hospital, Zurich, Switzerland
  4. **Matthias Pfisterer, M.D.** — University Hospital, Basel, Switzerland
- **Donor**: David W. Aha
- **Original Filename**: `processed.cleveland.data`
- **Local Repository Path**: `datasets/raw/heart/processed.cleveland.data`
- **File Format**: Headerless, comma-delimited ASCII text
- **Cryptographic Hashes**:
  - **MD5**: `2d91a8ff69cfd9616aa47b59d6f843db`
  - **SHA-256**: `a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8`
  - **File Size**: 18,461 bytes

### Known Dataset Limitations
1. **Sample Size**: Modest cohort (N = 303), demanding parsimonious model architectures and cross-validation to prevent overfitting.
2. **Demographic Bias**: 67.99% male (206) vs. 32.01% female (97), reflecting late-1980s referral and diagnostic patterns for coronary catheterization.
3. **Single Center Origin**: Collected from the Cleveland Clinic Foundation (Ohio, USA); generalization to multi-ethnic populations requires explicit caution.
4. **Specialized Diagnostic Tests**: High-predictive features (`ca` from fluoroscopy, `thal` from nuclear scintigraphy) require hospital-level invasive or radionuclide diagnostics.

---

## 3. Schema & Feature Dictionary

The raw file does not contain header lines. The standard 14 clinical attributes are mapped as follows:

| Column Name | Clinical Semantic Description | Raw Type | Semantic Classification | Allowed / Observed Domain | Missingness |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`age`** | Patient age in years | `float64` | Continuous Numerical | 29.0 – 77.0 | 0 (0.0%) |
| **`sex`** | Biological sex | `float64` | Binary Categorical | 0.0 (Female, 32.01%), 1.0 (Male, 67.99%) | 0 (0.0%) |
| **`cp`** | Chest pain presentation | `float64` | Nominal Categorical (4 levels) | 1.0 (Typical angina: 23), 2.0 (Atypical angina: 50), 3.0 (Non-anginal: 86), 4.0 (Asymptomatic: 144) | 0 (0.0%) |
| **`trestbps`** | Resting blood pressure on hospital admission | `float64` | Continuous Numerical | 94.0 – 200.0 mm Hg | 0 (0.0%) |
| **`chol`** | Serum cholesterol level | `float64` | Continuous Numerical | 126.0 – 564.0 mg/dl | 0 (0.0%) |
| **`fbs`** | Fasting blood sugar > 120 mg/dl | `float64` | Binary Categorical | 0.0 (False: 258, 85.15%), 1.0 (True: 45, 14.85%) | 0 (0.0%) |
| **`restecg`** | Resting electrocardiogram findings | `float64` | Nominal Categorical (3 levels) | 0.0 (Normal: 151, 49.83%), 1.0 (ST-T abnormality: 4, 1.32%), 2.0 (LV hypertrophy: 148, 48.84%) | 0 (0.0%) |
| **`thalach`** | Maximum heart rate achieved during exercise | `float64` | Continuous Numerical | 71.0 – 202.0 bpm | 0 (0.0%) |
| **`exang`** | Exercise-induced angina | `float64` | Binary Categorical | 0.0 (No: 204, 67.33%), 1.0 (Yes: 99, 32.67%) | 0 (0.0%) |
| **`oldpeak`** | ST depression induced by exercise relative to rest | `float64` | Continuous Numerical | 0.0 – 6.2 mm | 0 (0.0%) |
| **`slope`** | Peak exercise ST segment slope | `float64` | Ordinal Categorical (3 levels) | 1.0 (Upsloping: 142, 46.86%), 2.0 (Flat: 140, 46.20%), 3.0 (Downsloping: 21, 6.93%) | 0 (0.0%) |
| **`ca`** | Number of major vessels (0–3) colored by fluoroscopy | `object` | Discrete Count Numerical | '0.0' (176), '1.0' (65), '2.0' (38), '3.0' (20), '?' (4) | 4 (1.32%) |
| **`thal`** | Thallium myocardial perfusion scintigraphy | `object` | Nominal Categorical (3 levels) | '3.0' (Normal: 166), '6.0' (Fixed: 18), '7.0' (Reversible: 117), '?' (2) | 2 (0.66%) |
| **`num`** | Angiographic coronary artery disease status | `int64` | Target Variable (Multi-stage) | 0 (164), 1 (55), 2 (36), 3 (35), 4 (13) | 0 (0.0%) |

---

## 4. Feature-Type Classification & Rationale

1. **Continuous Numerical (5 features)**: `age`, `trestbps`, `chol`, `thalach`, `oldpeak`
   - *Rationale*: Physical, physiological, or biochemical continuous measurements where distance, variance, and ordering are mathematically meaningful.
2. **Discrete Count Numerical (1 feature)**: `ca`
   - *Rationale*: Non-negative integer count (0 to 3) representing anatomically fluoroscoped vessels.
3. **Binary Categorical (3 features)**: `sex`, `fbs`, `exang`
   - *Rationale*: Two mutually exclusive states representing demographic classification or physiological threshold criteria.
4. **Nominal Categorical (3 features)**: `cp`, `restecg`, `thal`
   - *Rationale*: Qualitative diagnostic categories. Despite being coded as integers/floats in raw text, arithmetic distance does not correspond to clinical distance (e.g. chest pain type 4 is not "4 times" chest pain type 1). One-hot encoding is mathematically necessary.
5. **Ordinal Categorical (1 feature)**: `slope`
   - *Rationale*: Clinically established physiological hierarchy (1: normal upsloping < 2: equivocal flat < 3: ischemic downsloping).

---

## 5. Statistical Profile of Numerical Variables

Computed on a temporary clean numerical view without modifying the raw dataset:

| Feature | Count | Mean | Std Dev | Min | 25% | Median | 75% | Max | IQR | Skewness |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`age`** | 303.0 | 54.44 | 9.04 | 29.0 | 48.0 | 56.0 | 61.0 | 77.0 | 13.0 | -0.21 |
| **`trestbps`** | 303.0 | 131.69 | 17.60 | 94.0 | 120.0 | 130.0 | 140.0 | 200.0 | 20.0 | +0.71 |
| **`chol`** | 303.0 | 246.69 | 51.78 | 126.0 | 211.0 | 241.0 | 275.0 | 564.0 | 64.0 | +1.14 |
| **`thalach`** | 303.0 | 149.61 | 22.88 | 71.0 | 133.5 | 153.0 | 166.0 | 202.0 | 32.5 | -0.54 |
| **`oldpeak`** | 303.0 | 1.04 | 1.16 | 0.0 | 0.0 | 0.8 | 1.6 | 6.2 | 1.6 | +1.27 |
| **`ca`** | 299.0 | 0.67 | 0.94 | 0.0 | 0.0 | 0.0 | 1.0 | 3.0 | 1.0 | +1.19 |

### Observations on Extreme Values
- **Cholesterol (`chol`)**: Maximum value of 564 mg/dl (patient 48; next highest is 417 mg/dl). This represents severe familial or clinical hypercholesterolemia. Because this reflects genuine pathology, it must NOT be arbitrarily trimmed.
- **Blood Pressure (`trestbps`)**: Reaches 200 mm Hg (severe stage 2/crisis hypertension), completely plausible in hospital admission settings.
- **ST Depression (`oldpeak`)**: Right-skewed with 99 observations (32.7%) at exactly 0.0 mm (no exercise-induced depression).

---

## 6. Missing Value & Quality Audit

1. **Sentinels**: Missing observations are exclusively encoded as `'?'`.
   - `ca`: 4 missing values (indices 166, 192, 287, 302).
   - `thal`: 2 missing values (indices 87, 266).
   - Total rows affected: 6 (1.98%).
2. **Missing Value Resolution Policy**: Handled inside scikit-learn transformers fitted strictly on training folds during subsequent phases. Raw file remains untouched.
3. **Duplicates**: 0 duplicate rows.
4. **Rare Category Warning**: `restecg = 1.0` appears in only 4 observations (1.32%). Stratified splits must ensure stability across folds.

---

## 7. Target Formulation

### Raw Multi-Stage Target (`num`)
- `0`: 164 observations (54.13%) — Negative / Absence of CAD (< 50% stenosis)
- `1`: 55 observations (18.15%) — Stage 1 CAD
- `2`: 36 observations (11.88%) — Stage 2 CAD
- `3`: 35 observations (11.55%) — Stage 3 CAD
- `4`: 13 observations (4.29%) — Stage 4 CAD

### Exploratory Binary Screening Formulation
- **Class 0 (Negative)**: `num == 0` (164 records, 54.13%)
- **Class 1 (Positive)**: `num >= 1` (139 records, 45.87%)
- **Cohort Balance Ratio**: 1.18 : 1 (favorable balance for binary classification; no severe imbalance mitigation required).
- *Boundary Statement*: The binary target is an exploratory planning formulation for the screening pipeline. NO model training is conducted in this phase.

---

## 8. Preliminary Preprocessing Architecture (Phase 1 / Phase 5 Design)

```
[Raw Ingestion] (datasets/raw/heart/processed.cleveland.data)
      │
      ▼
[Replace '?' with NaN]
      │
      ▼
[Stratified Hold-Out Split] (80% Train, 20% Test; random_state=42)
      │
      ├───────────────────────────────┐
      ▼                               ▼
[Train Fold Ingestion]        [Isolated Hold-out Test]
      │                               │
      ▼                               │
[ColumnTransformer (.fit())]          │
  ├─ Num: SimpleImputer + Scaler      │
  ├─ Cat: SimpleImputer + OneHot      │
  └─ Bin: Passthrough                 │
      │                               │
      ├──────────────────────┐        │
      ▼                      ▼        ▼
[Cross-Validation]      [.transform() on Test]
Repeated Stratified       Unbiased Hold-out
5-Fold CV                Evaluation
```

1. **Imputation**: Median imputation for numerical/count features (`ca`); most frequent / mode imputation for categorical features (`thal`).
2. **Categorical Encoding**: `OneHotEncoder(handle_unknown='ignore', drop='first')` for `cp`, `restecg`, `thal`.
3. **Scaling**: Evaluate `StandardScaler` and `RobustScaler` for continuous features (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`).
4. **Model Compatibility**:
   - **Logistic Regression**: Strictly requires scaled numericals and one-hot encoded nominals.
   - **Random Forest**: Invariant to scale, handles nonlinear interactions, robust to outliers.
   - **XGBoost**: Handles sparsity, invariant to scale, requires numeric representation of categories.
5. **Data Leakage Prohibition**: Imputers, encoders, and scalers must be fitted strictly on training folds. The hold-out test set must remain unvisited until final evaluation.

---

## 9. Verification & Sign-off

- **Notebook Execution**: `notebooks/04_heart_dataset_analysis.ipynb` executed end-to-end with 0 errors.
- **Raw File Checksums**:
  - Pre-Analysis MD5: `2d91a8ff69cfd9616aa47b59d6f843db`
  - Post-Analysis MD5: `2d91a8ff69cfd9616aa47b59d6f843db` (Match confirmed)
  - Pre-Analysis SHA-256: `a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8`
  - Post-Analysis SHA-256: `a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8` (Match confirmed)
- **Model Training**: 0 models trained.
- **Liver Disease Files**: 100% untouched.
