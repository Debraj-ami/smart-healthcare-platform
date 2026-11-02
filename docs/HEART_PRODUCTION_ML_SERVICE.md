# Heart Disease — Production ML Service Specification
## Stage 1: Production Artifact & FastAPI Microservice

---

## 1. Executive Summary & North Star Alignment

The Smart Healthcare Platform is architected around four core platform objectives:

1. **Personalized baseline/profile for each patient:** Establishing individual clinical norms and longitudinal health baselines.
2. **Context-aware health monitoring and trend/deviation analysis:** Detecting subtle shifts, lifestyle factors, and physiological trajectory changes over time.
3. **Disease-specific AI/ML risk prediction with interpretable outputs:** Deploying rigorously validated, calibrated, and explainable models tailored to specific disease etiologies.
4. **Proactive alerts/decision support based on patient context and model results:** Generating timely, context-sensitive clinical notifications and recommendations for healthcare providers.

### Stage 1 Scope & Boundary Definition
This stage implements **Objective #3** for the Cleveland Heart Disease module by delivering a production-grade, serialized ML artifact and an inference endpoint inside the platform's unified FastAPI ML service (`smart-healthcare-ml-service`).

```
+-----------------------------------------------------------------------------------+
|                           SMART HEALTHCARE PLATFORM                               |
+-----------------------------------------------------------------------------------+
|  [Objective #1]           [Objective #2]                    [Objective #4]        |
|  Personalized Baseline    Context-Aware Monitoring          Proactive Alerts      |
|  (Spring Boot / DB)       (Longitudinal Trends)             (Clinical Support)    |
+-----------------------------------------------------------------------------------+
                                      |
                           Orchestration & Context
                                      v
+-----------------------------------------------------------------------------------+
|  [Objective #3] DISEASE-SPECIFIC ML INFERENCE ENGINE (This Stage)                |
|  FastAPI Microservice (`ml-service`)                                              |
|                                                                                   |
|  +---------------------------+       +-----------------------------------------+  |
|  | Liver Screening Engine    |       | Heart Screening Engine                  |  |
|  | - Logistic Regression Iso |       | - Logistic Regression (Robust L1, C=1.0)|  |
|  | - Tree / Linear SHAP      |       | - Configured Threshold (t = 0.31)       |  |
|  | - Calibrated Probabilities|       | - SHAP LinearExplainer (Log-Odds)       |  |
|  +---------------------------+       +-----------------------------------------+  |
+-----------------------------------------------------------------------------------+
```

### Strict Non-Clinical Scope Boundary
The FastAPI ML service is **strictly a statistical inference and decision-support microservice**. It:
- **Does NOT** diagnose coronary artery disease.
- **Does NOT** replace clinicians or medical judgment.
- **Does NOT** calculate personalized patient baselines (responsibility of Spring Boot / Platform layer).
- **Does NOT** perform context-aware longitudinal trend analysis (responsibility of Spring Boot / Platform layer).
- **Does NOT** issue definitive clinical orders or therapeutic prescriptions.

The model output is explicitly designated as **model-estimated probability**, the classification cutoff is the **configured model classification threshold**, and the local attribution values explain **model mathematical behavior in log-odds space**, never medical etiology or clinical causation.

---

## 2. Production Model Architecture & Lifecycle Lineage

The production model configuration is the definitive culmination of Phases 0 through 8:

| Phase | Description | Production Choice & Rationale |
| :--- | :--- | :--- |
| **Phase 0** | Dataset Provenance | Cleveland Clinic Foundation Heart Disease Dataset ($N=303$, 13 clinical predictors). Cryptographic integrity verified (MD5: `2d91a8ff69cfd9616aa47b59d6f843db`). |
| **Phase 1** | User Journeys | Primary screening workflow; high-sensitivity screening profile prioritized to minimize false negatives. |
| **Phase 2** | Architecture | Clean separation between disease-specific ML engines and platform orchestration. |
| **Phase 3** | Feature Design | 13 predictor features; target `num` partitioned into binary screening target ($y = 1$ if $\text{num} \ge 1$, $0$ otherwise). |
| **Phase 4** | Preprocessing | Leakage-free `ColumnTransformer`: numerical median imputation + `RobustScaler`; categorical mode imputation + `OneHotEncoder(drop='first', handle_unknown='ignore')`. |
| **Phase 5** | Experimentation | Finalist model: `LogisticRegression(penalty='l1', C=1.0, solver='liblinear', random_state=42)`. Achieved holdout ROC-AUC of 0.9513, PR-AUC of 0.9274. |
| **Phase 6** | Threshold Analysis | Selected operating threshold $t = 0.31$ (high-sensitivity policy targeting $\ge 90\%$ sensitivity on training OOF; achieved $92.9\%$ on holdout test set). |
| **Phase 7** | Probability Calibration | Empirically well-calibrated natively (Brier = 0.0886, Log Loss = 0.3019 on holdout; ECE = 0.0383, calibration slope = 1.0704). Post-hoc calibration did not demonstrate consistent improvement; uncalibrated baseline retained. |
| **Phase 8** | SHAP Explainability | `shap.LinearExplainer` initialized with training background distribution in log-odds space; strict additivity efficiency confirmed ($\text{discrepancy} < 10^{-10}$). |

---

## 3. Serialized Model Artifact Structure

The production artifact is packaged into a self-contained serialized bundle located at:
`ml-service/artifacts/heart/heart_model_bundle.joblib`

Accompanied by machine-readable metadata at:
`ml-service/artifacts/heart/production_metadata.json`

### Bundle Components (`heart_model_bundle.joblib`)
- `preprocessor`: Fitted scikit-learn `ColumnTransformer` (fitted strictly on $X_{\text{train}}$, $N=242$).
- `classifier`: Fitted `LogisticRegression` estimator (penalty='l1', C=1.0, solver='liblinear', random_state=42).
- `shap_explainer`: Fitted `shap.LinearExplainer` with `shap.maskers.Independent` over the 242-sample training background.
- `threshold`: Float scalar `0.31`.
- `threshold_description`: Operational description of the sensitivity-focused screening cutoff.
- `calibration_method`: String documenting the uncalibrated baseline's empirical calibration status.
- `output_space`: `"log-odds"`.
- `feature_names_raw`: List of 13 raw input column names.
- `feature_names_transformed`: List of 18 transformed column names.
- `feature_descriptions`: Structured clinical dictionary mapping transformed columns to human-readable names and biological groupings.
- `expected_input_schema`: Specification of input types and bounds.

### Lifecycle Invariant
The FastAPI application loads this bundle into memory **once upon startup** via the `lifespan` handler. **At no point during inference does the service retrain, refit, re-calibrate, or recalculate background distributions.**

---

## 4. Preprocessing Pipeline & Feature Schema

The preprocessing pipeline guarantees deterministic transformation without data leakage:

```
Raw Input (13 features)
  ├── Numerical (5 features: age, trestbps, chol, thalach, oldpeak)
  │     ├── SimpleImputer(strategy='median')
  │     └── RobustScaler(quantile_range=(25.0, 75.0))
  ├── Binary (3 features: sex, fbs, exang)
  │     └── SimpleImputer(strategy='most_frequent')
  ├── Categorical (4 features: cp, restecg, slope, thal)
  │     ├── SimpleImputer(strategy='most_frequent')
  │     └── OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
  └── Count (1 feature: ca)
        └── SimpleImputer(strategy='median')
```

### Transformed Feature Vector (18 Features)
1. `num__age`: Standardized patient age in years.
2. `num__trestbps`: Standardized resting blood pressure (mm Hg).
3. `num__chol`: Standardized serum cholesterol (mg/dl).
4. `num__thalach`: Standardized maximum heart rate achieved (bpm).
5. `num__oldpeak`: Standardized ST depression relative to rest.
6. `bin__sex`: Biological sex (1 = male, 0 = female).
7. `bin__fbs`: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false).
8. `bin__exang`: Exercise-induced angina (1 = yes, 0 = no).
9. `cat__cp_2.0`: Chest pain type 2 (atypical angina).
10. `cat__cp_3.0`: Chest pain type 3 (non-anginal pain).
11. `cat__cp_4.0`: Chest pain type 4 (asymptomatic ischemia).
12. `cat__restecg_1.0`: Resting ECG ST-T wave abnormality.
13. `cat__restecg_2.0`: Resting ECG left ventricular hypertrophy.
14. `cat__slope_2.0`: ST slope flat.
15. `cat__slope_3.0`: ST slope downsloping.
16. `cat__thal_6.0`: Thallium fixed defect (prior infarction).
17. `cat__thal_7.0`: Thallium reversible defect (exercise-induced ischemia).
18. `count__ca`: Number of fluoroscopy-colored major vessels (0 to 3).

---

## 5. Probability Estimation & Threshold Behavior

### Probability Calculation
Inference applies the fitted Logistic Regression model:
1. Model log-odds output (decision function):
   $$z = \mathbf{w}^T \mathbf{x} + b$$
2. Model-estimated probability via standard logistic sigmoid:
   $$P(Y = 1 \mid \mathbf{x}) = \sigma(z) = \frac{1}{1 + e^{-z}}$$

### Operating Classification Threshold ($t = 0.31$)
The binary classification prediction is evaluated against the configured operating threshold:
$$\hat{y} = \begin{cases} 1 & \text{if } P(Y = 1 \mid \mathbf{x}) \ge 0.31 \\ 0 & \text{if } P(Y = 1 \mid \mathbf{x}) < 0.31 \end{cases}$$

- **Operational Role**: High-sensitivity preliminary screening cutoff established in Phase 6 to minimize false negative risk in an initial risk identification workflow.
- **Safety Precaution**: The threshold is an operational model parameter, **not a clinical diagnostic criterion**. Both the probability and threshold are returned to the client to ensure full transparency.

---

## 6. SHAP Explainability Architecture

### Mathematical Space & Efficiency
Local feature explanations are produced via `shap.LinearExplainer` evaluated in **log-odds space** against the 242-sample training background:
$$z(\mathbf{x}) = \phi_0 + \sum_{i=1}^{18} \phi_i(\mathbf{x})$$
where:
- $\phi_0$ is the reference base value (expected log-odds across the background).
- $\phi_i(\mathbf{x})$ is the attribution of transformed feature $i$.
- $\sum_{i} \phi_i$ exactly reconstructs $z(\mathbf{x}) - \phi_0$ ($\text{discrepancy} < 10^{-10}$).

### Clinical Mapping & Partitioning
Attributions are mapped through `HEART_FEATURE_DESCRIPTIONS` to expose:
- `feature`: Original clinical field (e.g. `ca`, `thal`, `cp`).
- `transformed_feature`: Internal pipeline column (e.g. `cat__thal_7.0`).
- `display_name`: Human-readable clinical label (e.g. `Thallium: Reversible Defect`).
- `raw_value`: The actual value submitted for this patient.
- `contribution`: The directional log-odds attribution ($\phi_i$).
- `abs_contribution`: Absolute value used for importance ranking ($|\phi_i|$).
- `direction`:
  - `INCREASES_RISK_ESTIMATE` ($\phi_i > 0.005$)
  - `DECREASES_RISK_ESTIMATE` ($\phi_i < -0.005$)
  - `NEUTRAL` ($-0.005 \le \phi_i \le 0.005$)

### Failure Isolation
If SHAP attribution encounters an unexpected runtime error, **the model prediction and probability are still returned**. The explanation payload gracefully marks `available=False` with the corresponding error message. **Under no circumstances are fake or synthetic explanation values generated.**

---

## 7. API Specification & Endpoints

### 1. `GET /health`
Inspects microservice availability and loaded model bundles.

**Response (HTTP 200)**:
```json
{
  "status": "UP",
  "service": "smart-healthcare-ml-service",
  "version": "1.0.0",
  "modelsLoaded": [
    "Logistic Regression (Isotonic Calibrated)",
    "Logistic Regression (Robust L1)"
  ]
}
```

### 2. `POST /predict/heart` (Alias: `POST /api/predict/heart`)
Generates model-estimated probability, binary classification decision, and local SHAP explanations for Cleveland Heart clinical inputs.

#### Request Schema (`HeartPredictionRequest`)
| Field | Type | Bounds / Allowed Values | Description |
| :--- | :--- | :--- | :--- |
| `age` | float | $[1.0, 120.0]$ | Patient age in years. |
| `sex` | float | $\{0.0, 1.0\}$ | Biological sex ($1.0 = \text{Male}, 0.0 = \text{Female}$). |
| `cp` | float | $\{1.0, 2.0, 3.0, 4.0\}$ | Chest pain type (1: typical, 2: atypical, 3: non-anginal, 4: asymptomatic). |
| `trestbps` | float | $[50.0, 300.0]$ | Resting blood pressure in mm Hg on admission. |
| `chol` | float | $[50.0, 800.0]$ | Serum cholesterol in mg/dl. |
| `fbs` | float | $\{0.0, 1.0\}$ | Fasting blood sugar $> 120$ mg/dl ($1.0 = \text{true}, 0.0 = \text{false}$). |
| `restecg` | float | $\{0.0, 1.0, 2.0\}$ | Resting ECG (0: normal, 1: ST-T abnormality, 2: LV hypertrophy). |
| `thalach` | float | $[50.0, 250.0]$ | Maximum heart rate achieved during exercise stress test. |
| `exang` | float | $\{0.0, 1.0\}$ | Exercise-induced angina ($1.0 = \text{yes}, 0.0 = \text{no}$). |
| `oldpeak` | float | $[0.0, 15.0]$ | ST depression induced by exercise relative to rest. |
| `slope` | float | $\{1.0, 2.0, 3.0\}$ | Slope of peak exercise ST segment (1: upsloping, 2: flat, 3: downsloping). |
| `ca` | float \| null | $\{0.0, 1.0, 2.0, 3.0, \text{null}\}$ | Number of fluoroscopy-colored major vessels; nullable per Phase 4 contract. |
| `thal` | float \| null | $\{3.0, 6.0, 7.0, \text{null}\}$ | Thallium defect (3: normal, 6: fixed, 7: reversible); nullable per Phase 4 contract. |

*Validation Guard: Any request containing the target variable `num` is strictly rejected with HTTP 422.*

#### Response Schema (`HeartPredictionResponse`)
```json
{
  "disease": "HEART_DISEASE",
  "model": {
    "name": "Logistic Regression (Robust L1)",
    "version": "1.0.0",
    "config_id": "LR_robust_l1_C1.0",
    "output_space": "log_odds",
    "calibration_status": "Uncalibrated Baseline (Empirically Calibrated)"
  },
  "probability": 0.8872,
  "prediction": 1,
  "threshold": 0.31,
  "threshold_description": "Configured model classification threshold (t=0.31) targeting >= 90% sensitivity established in Phase 6.",
  "explanation": {
    "available": true,
    "output_space": "log_odds",
    "base_value": -0.2514,
    "model_output": 2.0625,
    "positive_contributors": [
      {
        "feature": "thal",
        "transformed_feature": "cat__thal_7.0",
        "display_name": "Thallium: Reversible Defect",
        "raw_value": 7.0,
        "contribution": 0.9421,
        "abs_contribution": 0.9421,
        "direction": "INCREASES_RISK_ESTIMATE"
      },
      {
        "feature": "ca",
        "transformed_feature": "count__ca",
        "display_name": "Fluoroscopy Vessels (ca)",
        "raw_value": 2.0,
        "contribution": 0.8124,
        "abs_contribution": 0.8124,
        "direction": "INCREASES_RISK_ESTIMATE"
      }
    ],
    "negative_contributors": [
      {
        "feature": "thalach",
        "transformed_feature": "num__thalach",
        "display_name": "Maximum Heart Rate",
        "raw_value": 112.0,
        "contribution": -0.3142,
        "abs_contribution": 0.3142,
        "direction": "DECREASES_RISK_ESTIMATE"
      }
    ],
    "all_contributions": [ ... ],
    "error": null
  },
  "input_summary": { ... },
  "disclaimer": "This is a research-based risk screening tool developed for decision support experimentation. The model output is a statistical probability estimate and does NOT constitute a clinical diagnosis, prognosis, or definitive proof of coronary artery disease. The configured classification threshold (t=0.31) is an operational sensitivity-focused screening cutoff, not a medical diagnostic threshold. SHAP feature attributions explain model mathematical behavior in log-odds space and do not establish biological or medical causation. Clinical decisions must always be made by a qualified healthcare professional."
}
```

---

## 8. Validation, Error Handling & Safety Boundaries

| Scenario | HTTP Status | Response Handling |
| :--- | :--- | :--- |
| **Missing required input** (e.g. omitted `age`) | `422 Unprocessable Entity` | Clean Pydantic error details listing missing required field. |
| **Target field submitted** (`num`) | `422 Unprocessable Entity` | Explicit rejection message: `"Target variable 'num' is not accepted in prediction requests."` |
| **Invalid categorical value** (e.g. `cp = 5.0`) | `422 Unprocessable Entity` | Explicit validation error stating allowed discrete values. |
| **NaN or Infinite value** | `422 Unprocessable Entity` | Sanitized error response conforming to RFC 7159 JSON without crashing JSON encoder. |
| **Out-of-range clinical values** (e.g. `age = 150`) | `422 Unprocessable Entity` | Validation error specifying physical boundary ranges. |
| **Model bundle missing/corrupted** | `503 Service Unavailable` | Descriptive message indicating artifact unavailability. |
| **SHAP computation failure** | `200 OK` | Valid prediction returned; `explanation.available = false`, `explanation.error` populated; zero fabricated attributions. |

---

## 9. Verification & Quality Assurance Suite

The implementation was validated against a 100% automated test suite (`pytest -v tests/`):
- **18 Heart Service Tests** (`tests/test_heart_service.py`):
  - Application imports & dual-route registration (`/predict/heart` & `/api/predict/heart`).
  - Pre-warmed model bundle loading & single-load caching.
  - High-risk and low-risk clinical profile inference correctness.
  - Imputation of nullable `ca` and `thal` per Phase 4 contract.
  - Mathematical agreement between FastAPI endpoint and direct serialized bundle inference.
  - SHAP output space verification (`log_odds`), additivity efficiency, and clinical label mapping.
  - Positive/negative contribution sorting by absolute magnitude.
  - Rejection of invalid categories, out-of-range values, missing fields, NaNs, and infinities.
  - Strict rejection of the target variable `num`.
  - Graceful handling of artifact failure (503) and SHAP failure (isolated fallback).
  - Verification that model `fit()` is never invoked during inference.
- **80 Regression Tests** across all existing modules:
  - 18 Preprocessing Tests (`test_heart_preprocessing.py`)
  - 10 ML Experimentation Tests (`test_heart_experimentation.py`)
  - 12 Threshold Analysis Tests (`test_heart_threshold_analysis.py`)
  - 15 Probability Calibration Tests (`test_heart_calibration.py`)
  - 15 SHAP Explainability Tests (`test_heart_shap.py`)
  - 10 Liver Service Tests (`test_liver_service.py`)
- **Total Suite**: **98 passed, 0 failed**.

---

## 10. Future Spring Boot Orchestration Integration

When the platform integrates this service into Spring Boot:
1. **Spring Boot Feign / WebClient**: Will post patient observations to `/api/predict/heart`.
2. **Context Association**: Spring Boot will combine the returned `probability` and `explanation` with:
   - Longitudinal baseline metrics (Objective #1).
   - Dynamic sensor streams and trajectory deviations (Objective #2).
   - Provider alerting rules and proactive decision-support workflows (Objective #4).
3. **No Redundant Modeling**: The ML model remains isolated in the Python FastAPI microservice, preserving clear architectural separation of concerns.
