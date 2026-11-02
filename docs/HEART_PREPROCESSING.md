
# Heart Disease — Preprocessing Pipeline & Dataset Preparation (Phase 4)

**Document Version**: 1.0.0  
**Phase**: Phase 4 — Preprocessing Pipeline Construction & Dataset Preparation  
**Parent Specification**: [`docs/HEART_FEATURE_DATA_DESIGN.md`](file:///d:/smart-healthcare-platform/docs/HEART_FEATURE_DATA_DESIGN.md)  
**Implementation Module**: [`ml-service/app/heart_preprocessing.py`](file:///d:/smart-healthcare-platform/ml-service/app/heart_preprocessing.py)  
**Automated Test Suite**: [`ml-service/tests/test_heart_preprocessing.py`](file:///d:/smart-healthcare-platform/ml-service/tests/test_heart_preprocessing.py)  
**Source Dataset**: `datasets/raw/heart/processed.cleveland.data` (SHA-256: `a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8`)  
**Status**: Complete & Verified (18/18 Automated Unit Tests Passing)  

---

## 1. Source Dataset & Integrity Assurance

The single, immutable source of truth for Heart Disease modeling is the Cleveland Clinic dataset stored at `datasets/raw/heart/processed.cleveland.data`.

### Cryptographic Checksum Verification
- **Byte Count**: Exactly 18,461 bytes.
- **MD5 Hash**: `2d91a8ff69cfd9616aa47b59d6f843db`
- **SHA-256 Hash**: `a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8`
- **Immutability Policy**: The raw file was read strictly in read-only mode. Pre- and post-test verification confirms that the raw dataset is 100% byte-for-byte unchanged.

---

## 2. Raw Schema & Ingestion

The reusable loader function `load_heart_raw_data()` in [`ml-service/app/heart_preprocessing.py`](file:///d:/smart-healthcare-platform/ml-service/app/heart_preprocessing.py) implements the canonical ingestion process:

1. **Headerless Parsing**: Ingests raw data with `header=None` and assigns the 14 standard column names:
   `age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`, `num`.
2. **Dimension Enforcement**: Enforces exactly 303 rows and 14 columns.
3. **Sentinel Conversion**: Maps all string sentinel `'?'` occurrences directly to `np.nan`.
4. **Type Normalization**: Casts numeric predictor columns to `float64` and target `num` to `int64`.
5. **Domain Validation**: Validates that all non-null values fall strictly within clinically established discrete sets:
   - `sex` $\in \{0.0, 1.0\}$
   - `cp` $\in \{1.0, 2.0, 3.0, 4.0\}$
   - `fbs` $\in \{0.0, 1.0\}$
   - `restecg` $\in \{0.0, 1.0, 2.0\}$
   - `exang` $\in \{0.0, 1.0\}$
   - `slope` $\in \{1.0, 2.0, 3.0\}$
   - `ca` $\in \{0.0, 1.0, 2.0, 3.0\}$
   - `thal` $\in \{3.0, 6.0, 7.0\}$
   - `num` $\in \{0, 1, 2, 3, 4\}$
6. **Continuous Range Checks**: Verifies that physiological variables satisfy strict positivity constraints (`age > 0`, `trestbps > 0`, `chol > 0`, `thalach > 0`, `oldpeak >= 0`).

---

## 3. Target Construction & Separation

Function: `create_modeling_dataset(df)`

- **Predictor Matrix ($X$)**: Contains exactly the 13 clinical attributes. The raw target `num` is strictly excluded from $X$.
- **Binary Modeling Target ($y$)**:
  $$y = (num \ge 1).astype(int)$$
  - `y = 0` (Absence of significant CAD): **164 observations (54.13%)**.
  - `y = 1` (Presence of significant CAD): **139 observations (45.87%)**.
  - **Cohort Imbalance Ratio**: **1.18 : 1** (Negative : Positive).
- **Clinical Terminology Guardrail**: The raw discrete values `1, 2, 3, 4` indicate the **anatomical count of diseased major vessels** (single-vessel vs. multi-vessel CAD), **NOT** progressive histological "stages".

---

## 4. Train / Test Split Strategy

Function: `split_heart_data(X, y, test_size=0.20, random_state=42)`

- **Hold-out Split**: Stratified 80% training / 20% testing split.
  - **Training Set ($X_{\text{train}}, y_{\text{train}}$)**: Exactly 242 observations (131 negative, 111 positive).
  - **Test Set ($X_{\text{test}}, y_{\text{test}}$)**: Exactly 61 observations (33 negative, 28 positive).
- **Stratification Precision**: Class proportions are identical across splits (~54.1% negative / 45.9% positive).
- **Isolation Boundary**: Splitting occurs **before** any learned preprocessing transformer is instantiated or fitted.

---

## 5. Feature Groups

The 13 predictive features are partitioned into explicit functional groups:

| Group Name | Feature Identifiers | Preprocessing Pipeline Strategy |
| :--- | :--- | :--- |
| **NUMERICAL** | `age`, `trestbps`, `chol`, `thalach`, `oldpeak` | Median imputation $\rightarrow$ Scaler (`Standard` or `Robust` for LR; No scaler for trees). |
| **BINARY** | `sex`, `fbs`, `exang` | Mode imputation $\rightarrow$ Passthrough as $\{0, 1\}$. |
| **CATEGORICAL** | `cp`, `restecg`, `slope`, `thal` | Mode imputation $\rightarrow$ `OneHotEncoder(handle_unknown='ignore')`. |
| **COUNT / DISCRETE** | `ca` | Median imputation $\rightarrow$ Scaled for LR; Unscaled count for trees. |

---

## 6. Missing-Value Handling Architecture

- **Observed Missingness**: Exactly 6 missing values across 2 features:
  - `ca`: 4 missing values (1.32%) at indices `[166, 192, 287, 302]`.
  - `thal`: 2 missing values (0.66%) at indices `[87, 266]`.
- **Imputation Strategy**:
  - `ca` (Count): `SimpleImputer(strategy='median')` — preserves the integer nature of vessel count (training median = 0.0).
  - `thal` (Nominal): `SimpleImputer(strategy='most_frequent')` — imputes the predominant modal category (training mode = 3.0 / Normal).
- **Leakage Prevention**: Imputers call `.fit()` strictly on training data/folds. Test-set values never influence imputation statistics.

---

## 7. Numerical Preprocessing & Scaling Options

- **Algorithm-Specific Requirements**:
  - **Logistic Regression**: High sensitivity to scale. Without scaling, cholesterol (range 126–564) would artificially dominate ST depression (range 0.0–6.2).
  - **Tree Models (Random Forest, XGBoost)**: Completely invariant to monotonic scaling. Unnecessary scaling is omitted to preserve natural numerical interpretability.
- **Configurable Scalers for Linear Models**:
  - `scaler='standard'`: `StandardScaler()` (centers to zero mean, unit variance).
  - `scaler='robust'`: `RobustScaler()` (centers to median, scales by interquartile range; highly resilient to skewed features like `chol` with skew +1.14 and `oldpeak` with skew +1.27).

---

## 8. Categorical Encoding Architecture

- **Transformer**: `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`
- **Configuration Options**:
  - `drop='first'`: Recommended for Logistic Regression to eliminate collinearity and prevent matrix singularity.
  - `drop=None`: Recommended for Random Forest, XGBoost, and SHAP to ensure symmetric splits and direct per-category attribution.
- **Unseen Category Resilience**: Tested and verified. Unseen categories in production or test sets are encoded as all-zero dummy vectors without raising runtime errors.

---

## 9. Treatment of Specific Features

### 9.1 Binary Features (`sex`, `fbs`, `exang`)
Passed through directly as $\{0, 1\}$ integer indicators. This avoids redundant inverse indicator columns (e.g. `sex_0` and `sex_1`) and keeps the feature space compact.

### 9.2 `ca` (Vessel Count: 0–3)
Treated as a discrete count numerical feature with median imputation. Preserves the natural clinical monotonic relationship (more stenosed vessels indicates greater coronary involvement) using a single parameter.

### 9.3 `slope` (ST Segment Slope: 1, 2, 3)
Treated as a nominal categorical feature encoded via `OneHotEncoder`. Avoids the unverified clinical assumption that the step distance from Upsloping ($1$) to Flat ($2$) is identical to Flat ($2$) to Downsloping ($3$).

---

## 10. Reusable Preprocessing Builders

Module: [`ml-service/app/heart_preprocessing.py`](file:///d:/smart-healthcare-platform/ml-service/app/heart_preprocessing.py)

### 1. `build_logistic_regression_preprocessor(scaler='standard', drop='first')`
```python
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), NUMERICAL_FEATURES),
        ('bin', Pipeline([('imputer', SimpleImputer(strategy='most_frequent'))]), BINARY_FEATURES),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', drop=drop, sparse_output=False))]), CATEGORICAL_FEATURES),
        ('count', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), COUNT_FEATURES),
    ],
    remainder='drop',
    verbose_feature_names_out=True,
)
```

### 2. `build_tree_preprocessor(drop=None)`
```python
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median'))]), NUMERICAL_FEATURES),
        ('bin', Pipeline([('imputer', SimpleImputer(strategy='most_frequent'))]), BINARY_FEATURES),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', drop=drop, sparse_output=False))]), CATEGORICAL_FEATURES),
        ('count', Pipeline([('imputer', SimpleImputer(strategy='median'))]), COUNT_FEATURES),
    ],
    remainder='drop',
    verbose_feature_names_out=True,
)
```

### 3. `build_heart_preprocessor(model_type='logistic', scaler='standard', drop='first')`
Unified factory routing to the appropriate preprocessor based on model architecture.

---

## 11. Feature Name Generation

Function: `get_heart_feature_names(preprocessor)`

When fitted on `X_train`, the preprocessors produce deterministic output feature names:
- **Logistic Regression (`drop='first'`)**: Exactly **19 features** (`num__age`, `num__trestbps`, `num__chol`, `num__thalach`, `num__oldpeak`, `bin__sex`, `bin__fbs`, `bin__exang`, `cat__cp_2.0`, `cat__cp_3.0`, `cat__cp_4.0`, `cat__restecg_1.0`, `cat__restecg_2.0`, `cat__slope_2.0`, `cat__slope_3.0`, `cat__thal_6.0`, `cat__thal_7.0`, `count__ca`).
- **Tree Models (`drop=None`)**: Exactly **22 features** (includes reference categories `cat__cp_1.0`, `cat__restecg_0.0`, `cat__slope_1.0`, `cat__thal_3.0`).

---

## 12. Automated Data Leakage Verification

All 6 mandatory data leakage tests were implemented and verified in [`ml-service/tests/test_heart_preprocessing.py`](file:///d:/smart-healthcare-platform/ml-service/tests/test_heart_preprocessing.py):

| Test ID | Leakage Guardrail | Test Mechanism | Result |
| :---: | :--- | :--- | :---: |
| **TEST 1** | Imputation Leakage | Modifying test set with extreme outliers (age = 999) does not alter training-fitted median statistics. | **PASS** |
| **TEST 2** | Scaler Leakage | Injecting extreme values into test set (chol = 10,000) does not change training scaler mean/variance. | **PASS** |
| **TEST 3** | Categorical Encoder Leakage | Introducing novel categories in test set does not alter training encoder category arrays. | **PASS** |
| **TEST 4** | Target Leakage | Confirms target column `num` is never present in $X$ or transformed feature vectors. | **PASS** |
| **TEST 5** | Deterministic Transform | Repeated `.transform()` calls on test data produce bit-for-bit identical numerical arrays. | **PASS** |
| **TEST 6** | Nested CV Independence | In 5-fold cross-validation, preprocessing is fitted independently on each fold; fold training statistics vary naturally. | **PASS** |

---

## 13. Test Coverage & Verification Results

Execution Command: `pytest tests/test_heart_preprocessing.py -v`

```
tests/test_heart_preprocessing.py::test_raw_dataset_cryptographic_integrity PASSED [  5%]
tests/test_heart_preprocessing.py::test_load_heart_raw_data_schema PASSED [ 11%]
tests/test_heart_preprocessing.py::test_missing_value_sentinel_conversion PASSED [ 16%]
tests/test_heart_preprocessing.py::test_category_and_range_validation PASSED [ 22%]
tests/test_heart_preprocessing.py::test_extreme_values_retained PASSED   [ 27%]
tests/test_heart_preprocessing.py::test_target_construction PASSED       [ 33%]
tests/test_heart_preprocessing.py::test_stratified_train_test_split PASSED [ 38%]
tests/test_heart_preprocessing.py::test_logistic_regression_preprocessor_execution PASSED [ 44%]
tests/test_heart_preprocessing.py::test_tree_preprocessor_execution PASSED [ 50%]
tests/test_heart_preprocessing.py::test_unified_factory PASSED           [ 55%]
tests/test_heart_preprocessing.py::test_handle_unseen_categories PASSED  [ 61%]
tests/test_heart_preprocessing.py::test_leakage_1_test_values_do_not_influence_imputation PASSED [ 66%]
tests/test_heart_preprocessing.py::test_leakage_2_test_values_do_not_influence_scaling PASSED [ 72%]
tests/test_heart_preprocessing.py::test_leakage_3_test_categories_do_not_influence_encoder_fitting PASSED [ 77%]
tests/test_heart_preprocessing.py::test_leakage_4_target_not_passed_to_preprocessing PASSED [ 83%]
tests/test_heart_preprocessing.py::test_leakage_5_test_data_uses_train_fitted_transformers PASSED [ 88%]
tests/test_heart_preprocessing.py::test_leakage_6_cv_pipelines_fit_independently PASSED [ 94%]
tests/test_heart_preprocessing.py::test_raw_dataset_remains_unmodified_after_all_operations PASSED [100%]

======================== 18 passed in 3.74s ========================
```

- **Full Suite Integration**: Ran `pytest tests/ -v`: **All 28 tests passed** (18 Heart preprocessing + 10 Liver service), proving zero regression to Liver ML functionality.

---

## 14. Reproducibility & Environment Parameters

- **Python Version**: `3.12.1`
- **scikit-learn Version**: `1.6.1`
- **pandas Version**: `2.2.3`
- **numpy Version**: `2.2.3`
- **Random Seed**: Fixed at `42` (`random_state=42`) across train/test splitting and cross-validation fold generation.
- **Raw File Integrity**: Post-test hash confirmed identical to pre-test hash.

---

## 15. Phase 5 Handoff Requirements

Phase 5 (**Heart ML Experimentation & Model Comparison**) is provided with:
1. **`load_heart_raw_data()`**: Verified ingestion returning clean DataFrame.
2. **`create_modeling_dataset()`**: Separating $X$ (13 predictors) and $y$ (binary target).
3. **`split_heart_data()`**: Providing isolated $X_{\text{train}}, X_{\text{test}}, y_{\text{train}}, y_{\text{test}}$.
4. **`build_heart_preprocessor()`**: Preprocessing factories ready to embed inside `Pipeline([('preprocessor', ...), ('model', ...)])`.
5. **Experimentation Mandate for Phase 5**:
   - Benchmark candidate algorithms: **Logistic Regression** (L1/L2), **Random Forest**, and **XGBoost**.
   - Compare `StandardScaler` vs. `RobustScaler` for Logistic Regression.
   - Run Repeated Stratified K-Fold Cross-Validation.
   - Evaluate ROC-AUC, PR-AUC, Sensitivity, Specificity, F1-Score, and Brier Score.
   - Select winning baseline model empirically without premature assumptions.
