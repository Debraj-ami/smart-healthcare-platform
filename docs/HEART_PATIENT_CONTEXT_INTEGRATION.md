# Heart Disease — Patient Context & Personalization Integration (Stage 3)

## 1. Purpose

This document details the architectural integration connecting Heart Disease machine learning assessments (established in Stages 1 and 2) with the platform's existing **patient context and personalization infrastructure** (Core Objectives #1 and #2).

The integration enables:
- Resolving verified patient profiles when `patientId` is provided in an assessment request.
- Associating and persisting disease assessment records in the existing `disease_assessments` relational schema.
- Chronologically retrieving historical Heart Disease assessments for longitudinal context tracking.
- Inspecting established personal baselines without manufacturing synthetic medical observations.
- Maintaining pure stateless inference when `patientId` is omitted.

---

## 2. Existing Context Architecture Discovered

During pre-implementation inspection of the Spring Boot backend (`backend/src/main/java/com/smarthealth/platform/`), the following existing architecture was identified:

1. **Patient Representation**:
   - Patient identity is modeled by the [`User`](file:///d:/smart-healthcare-platform/backend/src/main/java/com/smarthealth/platform/entity/User.java) entity with `role = UserRole.PATIENT`. Key demographic attributes include `dateOfBirth` (allowing exact chronological age calculation) and `gender`.
   - Detailed clinical metadata (blood group, height, weight, lifestyle, medical history) is stored in the 1-to-1 [`PatientProfile`](file:///d:/smart-healthcare-platform/backend/src/main/java/com/smarthealth/platform/entity/PatientProfile.java) entity and accessible via `GET /api/patients/{id}`.

2. **Health Observations (`health_observations`)**:
   - Longitudinal biometric time-series data is stored in [`HealthObservation`](file:///d:/smart-healthcare-platform/backend/src/main/java/com/smarthealth/platform/entity/HealthObservation.java) (`patient_id`, `metric_type`, `metric_value`, `unit`, `logged_at`, `activity_level`, `activity_minutes`).
   - Managed by `HealthObservationRepository` and exposed via `/api/observations`.

3. **Personal Baselines (`personal_baselines`)**:
   - Handled by [`PersonalBaselineService`](file:///d:/smart-healthcare-platform/backend/src/main/java/com/smarthealth/platform/service/PersonalBaselineService.java).
   - Computes statistical baseline values: mean ($\mu$) and variability/sample standard deviation ($\sigma$) across logged observations for a specific `(patientId, metricType)`.
   - Requires at least 1 observation; when count = 1, variability = 0.0.
   - Exposed via `GET /api/baselines/patient/{patientId}` and `POST /api/baselines/patient/{patientId}/{metricType}/calculate`.

4. **Trend & Deviation Analysis (`trend_analyses`)**:
   - Handled by [`TrendAnalysisService`](file:///d:/smart-healthcare-platform/backend/src/main/java/com/smarthealth/platform/service/TrendAnalysisService.java).
   - Computes standardized deviation $Z = (current - \mu) / \sigma$, trend direction (`INCREASING`, `DECREASING`, `STABLE`), persistence steps, and snapshot statistics.
   - Designed exclusively for periodic observation metrics; does not apply to discrete ML screening probabilities.

5. **Disease Assessments (`disease_assessments`)**:
   - Modeled by [`DiseaseAssessment`](file:///d:/smart-healthcare-platform/backend/src/main/java/com/smarthealth/platform/entity/DiseaseAssessment.java) (`id`, `patient_id` FK -> `users.id`, `disease_type`, `risk_score`, `risk_level`, `input_features`, `assessed_at`).
   - `DiseaseType.HEART_DISEASE` was already declared in the schema enum.
   - Stored calibrated/estimated probability in `risk_score`, JSON-serialized SHAP in `input_features`, and `risk_level` as `null` (since arbitrary categorization tiers are not clinically validated).

---

## 3. Heart Integration Point

The integration point is embedded directly in [`HeartAssessmentService`](file:///d:/smart-healthcare-platform/backend/src/main/java/com/smarthealth/platform/service/HeartAssessmentService.java) and exposed via [`HeartPredictController`](file:///d:/smart-healthcare-platform/backend/src/main/java/com/smarthealth/platform/controller/HeartPredictController.java):

```
Client / Frontend (React)
           │
           ▼
[POST /api/assessments/heart/predict] (Spring Boot Controller)
           │
           ▼
[HeartAssessmentService.predict(request)]
    ├── 1. Resolve Patient: UserService.getUserEntity(patientId) [if patientId != null]
    ├── 2. Query Prior Heart Assessments: DiseaseAssessmentRepository.findByPatientIdAndDiseaseType...
    ├── 3. Query Personal Baselines: PersonalBaselineRepository.findByPatientId(patientId)
    ├── 4. Execute ML Inference: MlServiceClient.predictHeart(request)
    │           │ (HTTP POST /api/predict/heart)
    │           ▼
    │       [FastAPI Python Microservice] -> Logistic Regression (Robust L1, t=0.31) + SHAP
    │           │
    │           ◀ (probability, prediction, threshold, SHAP attributions)
    ├── 5. Persist Assessment: DiseaseAssessmentRepository.save(...)
    ├── 6. Assemble Patient Context: Previous count, last assessedAt, last probability, active baselines
    └── 7. Return HeartPredictResponse with populated patientContext DTO
```

---

## 4. Patient Association

- When `patientId` is present in [`HeartPredictRequest`](file:///d:/smart-healthcare-platform/backend/src/main/java/com/smarthealth/platform/dto/HeartPredictRequest.java):
  - `UserService.getUserEntity(patientId)` resolves the corresponding `User` record.
  - If the user does not exist in the database, a `ResourceNotFoundException` is raised, triggering HTTP 404 with structured error payload. The ML service is **not** called.
  - If the patient is valid, demographic attributes (`gender`, chronological `ageYears` from `dateOfBirth`) are linked.
- When `patientId` is `null`:
  - The assessment executes completely statelessly.
  - No database records are read or written.
  - `patientContext`, `assessmentId`, and `assessedAt` remain `null` in the response.

---

## 5. Assessment Persistence

- Assessments containing a valid `patientId` are persisted to the existing `disease_assessments` table.
- Record attributes:
  - `patient`: Foreign key to `users` table.
  - `disease_type`: `DiseaseType.HEART_DISEASE`.
  - `risk_score`: Model-estimated probability (e.g., `0.8872`).
  - `risk_level`: `null`. Unvalidated categorical risk bands (such as LOW/MODERATE/HIGH) are **not** fabricated.
  - `input_features`: Compact JSON document containing `probability`, `prediction`, `threshold`, `shapBaseValue`, `modelOutput`, `allContributions`, and `inputSummary`.
  - `assessed_at`: Managed timestamp (`@PrePersist`).
- The generated `assessmentId` and `assessedAt` timestamp are returned in the top-level response and in historical lookups.

---

## 6. Previous Heart Assessment Handling

- `DiseaseAssessmentRepository` was extended with:
  ```java
  List<DiseaseAssessment> findByPatientIdAndDiseaseTypeOrderByAssessedAtDesc(Long patientId, DiseaseType diseaseType);
  ```
- Before saving a new assessment, prior assessments for `(patientId, DiseaseType.HEART_DISEASE)` are queried:
  - `previousAssessmentCount`: Total number of prior Heart assessments on record.
  - `lastAssessedAt`: Timestamp of the most recent prior assessment (or `null` if first assessment).
  - `lastProbability`: Model-estimated probability of the most recent prior assessment (or `null` if first assessment).
- Dedicated endpoint for historical lookups:
  - `GET /api/assessments/heart/patient/{patientId}`
  - Returns chronological list of recorded Heart assessments (`List<DiseaseAssessmentResponse>`) ordered descending by assessment date.

---

## 7. Baseline Integration

- Leverages the existing `PersonalBaselineRepository.findByPatientId(patientId)`.
- If the patient has established baselines (for instance, computed from periodic blood pressure or heart rate observations):
  - `hasEstablishedBaselines`: `true`
  - `establishedBaselineMetrics`: List of metric type strings (e.g., `["BLOOD_PRESSURE_SYSTOLIC"]`).
- If no baselines exist:
  - `hasEstablishedBaselines`: `false`
  - `establishedBaselineMetrics`: `[]` (empty list)
- **Strict Boundary**: The 13 Heart screening features (e.g., `trestbps`, `chol`, `thalach`) are inputs for ML inference and are **not** automatically converted into longitudinal observations or personal baselines. No synthetic baselines are ever manufactured.

---

## 8. Trend Integration

- Platform trend analysis (`TrendAnalysisService`) evaluates longitudinal time-series data for generic biometrics.
- Disease assessments represent discrete screening evaluations, not continuous biometric streams.
- Therefore, probability changes between discrete assessments are **not** fed into the $Z$-score trend engine, and probability shifts are **never** interpreted as clinical progression.
- Context transparency notes explicitly inform callers of this distinction.

---

## 9. Data Flow & Separation of Concerns

```
                     ┌───────────────────────────────────────┐
                     │            React Frontend             │
                     └──────────────────┬────────────────────┘
                                        │ POST /api/assessments/heart/predict
                                        ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│ Spring Boot Backend (Orchestration & Context Layer)                               │
│                                                                                   │
│  [UserService]            [PersonalBaselineRepo]       [DiseaseAssessmentRepo]   │
│         │ (validate)                │ (check baselines)           │ (query prior) │
│         ▼                           ▼                             ▼               │
│  ┌─────────────────────────────────────────────────────────────────────────────┐  │
│  │ HeartAssessmentService                                                      │  │
│  │   - Prepares features                                                       │  │
│  │   - Calls ML microservice                                                   │  │
│  │   - Persists assessment                                                     │  │
│  │   - Injects patientContext DTO                                              │  │
│  └──────────────────────────────────────┬──────────────────────────────────────┘  │
└─────────────────────────────────────────┼─────────────────────────────────────────┘
                                          │ POST /api/predict/heart
                                          ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│ FastAPI ML Microservice (Inference & Explainability Only)                          │
│                                                                                   │
│  - Accepts 13 numeric predictors                                                  │
│  - Executes Scikit-Learn Preprocessing Pipeline                                    │
│  - Executes Logistic Regression (Robust L1, C=1.0)                                │
│  - Applies Configured Classification Threshold (t=0.31)                           │
│  - Computes Tree/Linear SHAP Feature Contributions in Log-Odds Space               │
│  - NO database access, NO patient identity knowledge, NO state                     │
└───────────────────────────────────────────────────────────────────────────────────┘
```

---

## 10. Separation Between ML and Patient Context

| Concern | Owner | Implementation |
|---|---|---|
| Model weights & inference | FastAPI ML Service | `ml-service/artifacts/heart/heart_model_bundle.joblib` |
| Feature preprocessing & scaling | FastAPI ML Service | RobustScaler + OneHotEncoder in pipeline bundle |
| Threshold evaluation ($t = 0.31$) | FastAPI ML Service | `ml-service/app/heart_service.py` |
| Local SHAP attributions | FastAPI ML Service | Log-odds space feature attribution engine |
| Patient identity & authentication | Spring Boot | `UserService`, `User`, `PatientProfile` |
| Personal baseline calculation | Spring Boot | `PersonalBaselineService`, `PersonalBaselineRepository` |
| Longitudinal trend monitoring | Spring Boot | `TrendAnalysisService`, `TrendAnalysisRepository` |
| Assessment record persistence | Spring Boot | `DiseaseAssessmentRepository`, `disease_assessments` |
| Historical assessment query | Spring Boot | `GET /api/assessments/heart/patient/{id}` |

---

## 11. What Is Currently Supported

1. **Patient-linked Heart inference**: Passing `patientId` automatically resolves the patient and populates `patientContext`.
2. **Stateless Heart inference**: Omitting `patientId` performs pure ML inference without persistence or context overhead.
3. **Previous Heart assessment retrieval**: Exposing prior assessment count, previous probability, and timestamp in prediction responses, plus a dedicated historical assessment endpoint (`GET /api/assessments/heart/patient/{id}`).
4. **Baseline awareness**: Accurately reporting whether a patient has established biometric baselines and listing existing baseline metric types.
5. **SHAP & probability fidelity**: ML probability, binary prediction against $t=0.31$, and local SHAP explanations remain identical regardless of whether `patientId` is present.
6. **Graceful error handling**: Non-existent patients return HTTP 404 before calling the ML service; ML service outages return HTTP 503.

---

## 12. What Is Intentionally Not Supported

1. **Automatic conversion of ML inputs to observations**: Heart screening inputs (`trestbps`, `chol`, `thalach`) are not automatically inserted into `health_observations`. An assessment snapshot is not equivalent to a continuous longitudinal observation series.
2. **Fabricated baseline metrics**: If a patient lacks historical biometric observations for cholesterol or blood pressure, no baseline is manufactured.
3. **Artificial probability trends**: Heart screening probabilities are not fed into the $Z$-score trend engine, and probability delta is not presented as clinical progression.
4. **Categorical risk labeling**: `risk_level` remains `null`. Arbitrary risk tiers (LOW/MODERATE/HIGH) are not forced onto empirical model probabilities.
5. **Direct ML database access**: FastAPI does not connect to MySQL or query patient entities.

---

## 13. Limitations

- **Cross-Sectional Dataset Origin**: The production model was trained on the Cleveland Heart Disease dataset (cross-sectional snapshot). It predicts immediate screening likelihood, not future incidence time-to-event.
- **Biometric Calibration Independence**: The model outputs reflect empirical uncalibrated baseline probabilities (demonstrated in Phase 7 to possess strong empirical alignment, Brier score = 0.1068). They should not be conflated with longitudinal clinical risk trajectory.
- **Baseline Metric Coverage**: Personal baselines depend on data actively logged in `health_observations`. If patients only take occasional screening assessments without logging vitals, baseline metrics remain empty.

---

## 14. Verification & Testing

### Backend Spring Boot Tests (`backend/`)
- Suite: [`HeartPredictControllerTests`](file:///d:/smart-healthcare-platform/backend/src/test/java/com/smarthealth/platform/HeartPredictControllerTests.java) + Platform Suite
- Tests passing: **54 passed** (0 failures, 0 errors, 0 skipped).
- Covered scenarios:
  1. Valid Heart request accepts all 13 predictor fields.
  2. All 13 fields serialized and forwarded to ML service correctly.
  3. Target variable `num` strictly rejected with HTTP 400.
  4. Nullable fields `ca=null` and `thal=null` succeed per Phase 4 contract.
  5. ML service response mapped completely (probability, prediction, threshold 0.31, model metadata, SHAP).
  6. Isolated SHAP failure (`explanation.available=false`) preserved without fabricating values.
  7. ML service unavailability (503) handled gracefully.
  8. ML service 500 error handled safely.
  9. Malformed ML responses rejected safely.
  10. Invalid categorical values, out-of-range bounds, and missing required fields rejected (400).
  11. Patient persistence in `disease_assessments` table when `patientId` is provided.
  12. First assessment for patient correctly initializes `previousAssessmentCount = 0`, `lastProbability = null`, `hasEstablishedBaselines = false`.
  13. Subsequent assessment for patient accurately reflects `previousAssessmentCount = 1`, `lastProbability = 0.8872`, and previous `lastAssessedAt`.
  14. Patient with established baselines reports `hasEstablishedBaselines = true` and `establishedBaselineMetrics` without fabricating unrecorded metrics.
  15. Stateless request without `patientId` returns `patientContext = null` and skips persistence.
  16. Unknown `patientId` safely returns HTTP 404 without invoking ML service.
  17. Dedicated `GET /api/assessments/heart/patient/{id}` retrieves historical assessments in reverse chronological order.
  18. Historical assessment endpoint returns 404 for unknown patient ID.

### Python ML Service Tests (`ml-service/`)
- Suite: `pytest -v tests/`
- Tests passing: **98 passed** (0 failures).
- Preprocessing, experimentation, threshold analysis, calibration, SHAP, and FastAPI inference remain 100% verified.

### Platform Cross-Platform Total
- Total automated tests passing: **152 passed** (54 Spring Boot + 98 FastAPI/ML).

---

## 15. Core Objective Alignment

| Core Objective | Heart Stage 1 & 2 Status | Heart Stage 3 Status (This Stage) |
|---|---|---|
| **#1. Personalized baseline / profile** | Not connected | Patient resolved via `UserService`; demographic age/gender linked; existing platform baselines inspected and reflected without synthetic fabrication. |
| **#2. Context-aware monitoring & trends** | Not connected | Historical Heart assessments retrieved in reverse chronological order; previous probability tracked; clear architectural boundary established between observation trends and screening assessments. |
| **#3. Disease-specific ML risk prediction** | Complete (FastAPI + Spring Boot) | Fully operational: Logistic Regression (Robust L1), threshold $t=0.31$, SHAP explanations in log-odds space. Unchanged. |
| **#4. Proactive alerts & decision support** | Future Stage | Unlocked: Structured historical context and baseline awareness now available for future alert rule evaluation. |

---

## 16. Explicit Architectural Guardrails

- **FastAPI remains ML-only**: Pure stateless HTTP inference microservice with zero database or user profile dependencies.
- **Spring Boot remains platform & context orchestrator**: Owns security, persistence, patient resolution, baselines, and historical aggregation.
- **Heart ML probability**: Remains a model-estimated probability from the trained Logistic Regression model.
- **Classification threshold (0.31)**: Preserved exactly as established in Phase 6 for high sensitivity ($\ge 90\%$).
- **SHAP explainability**: Local feature contributions evaluated strictly in log-odds space.
- **No clinical diagnosis**: Outputs represent statistical screening and decision support, not definitive diagnoses.
- **No synthetic history**: Zero artificial baselines, trends, or observations are manufactured.
