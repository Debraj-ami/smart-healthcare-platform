# Heart Disease — User Journey & Functional Specification (Phase 1)

**Document Version**: 1.0.0  
**Phase**: Phase 1 — Disease-Specific User Journey and Functional Specification  
**Disease Module**: Coronary Heart Disease Risk Assessment  
**Upstream Artifacts**: [`docs/HEART_DATASET_ANALYSIS.md`](file:///d:/smart-healthcare-platform/docs/HEART_DATASET_ANALYSIS.md), [`notebooks/04_heart_dataset_analysis.ipynb`](file:///d:/smart-healthcare-platform/notebooks/04_heart_dataset_analysis.ipynb)  
**Status**: Complete & Verified  

---

## 1. Purpose

This document defines the functional specification, user journeys, data contracts, and operational boundaries for the **Heart Disease** prediction and monitoring module of the Smart Healthcare Platform.

The objective is to establish an end-to-end clinical screening workflow that bridges the patient-facing React interface, the Java Spring Boot orchestration layer, the Python FastAPI ML inference service, and the Conversational Chatbot assistant without violating clinical safety, diagnostic boundaries, or architectural separation of concerns.

---

## 1.1 Core Product Principle: Patient-Centric Health Entry

The Smart Healthcare Platform is **not** intended to be:
> *"Patient enters 13 clinical values → model predicts Heart Disease → end."*

The Heart ML model is only **one analytical component** of the healthcare platform. The platform must remain deeply useful even when a patient does **not** possess all 13 clinical, laboratory, or diagnostic test values required by the statistical model.

The platform provides **three distinct Heart pathways**:

### Path 1 — Symptoms ("I'm experiencing symptoms")
- **Purpose**: Conversational symptom interaction, safety assessment, contextual follow-up, and emergency triage.
- **Route**: Conversational Chatbot (`/chat`).
- **Behavior**: Patient describes symptoms (e.g. *"I have shortness of breath"* or *"chest discomfort"*). Emergency safety screening occurs first. Targeted follow-ups preserve symptom context. **No forced ML prediction** and no demanding of lab values (cholesterol, ECG, thal) unless the user explicitly requests a clinical screening assessment and safety allows it.

### Path 2 — Health Monitoring ("Monitor my health")
- **Purpose**: Track real-world physiological measurements and contextual longitudinal changes over time.
- **Route**: Health Monitoring module (`/monitoring`).
- **Behavior**: Leverages available patient observations (blood pressure, resting heart rate, activity, etc.) to establish personal baselines, calculate deviations, and highlight trends without requiring full 13-feature clinical panels. Does not invent clinical diagnoses from trends.

### Path 3 — Clinical Data Assessment ("I have clinical results")
- **Purpose**: Execute the validated 13-feature Heart ML screening model when structured laboratory/diagnostic test values are available.
- **Route**: Heart Disease Assessment Modal (`/assessments`).
- **Behavior**: Uses the validated production Logistic Regression model, calibrated probability, and SHAP feature attributions. If required clinical data (e.g. cholesterol, ECG) are missing, the system **never invents or silently imputes** values; it clearly communicates that the disease-specific model cannot run yet and directs the user to Path 1 or Path 2.

---

## 2. Scope

### In-Scope (Phase 1 Specification)
- Definition of patient and clinician user journeys for Heart Disease.
- Specification of the conceptual input contract based on the verified 14-attribute Cleveland schema.
- Distinction between user-facing presentation and model-ready representations.
- Conceptual validation rules (type, range, categorical integrity).
- Result structure design (calibrated probability, SHAP local attributions, screening guidance).
- Safety boundaries (screening vs. diagnosis, acute emergency redirection).
- Personalization boundaries (population risk vs. longitudinal individual baseline deviation).
- Chatbot boundaries within the Heart Disease domain.
- Architectural alignment with existing platform infrastructure (`health_observations`, `personal_baselines`, `trend_analyses`, `disease_assessments`).

### Explicitly Out-of-Scope
- Machine learning model training, hyperparameter tuning, or fitting (Phase 5).
- Risk threshold selection or unverified risk-tier categorization (Phase 6).
- Probability calibration fitting (Phase 7).
- SHAP background explainer generation (Phase 8).
- Production API endpoint or database schema implementation (Phase 2 / Phase 3).
- Real-time IoT or wearable hardware integrations (Future Scope).
- Medical diagnosis claims or direct clinical prescribing.
- Diabetes and Parkinson's Disease implementation.

---

## 3. Disease-Specific User Journeys

### Journey A: New Patient Assessment (First-Time Heart Risk Screening)

This journey accommodates a patient who possesses clinical lab reports, vitals, or exercise stress test results and seeks a cardiovascular risk assessment.

```
Patient Logs In (React)
          │
          ▼
Selects "Heart Disease Assessment"
          │
          ▼
Enters 13 Clinical Attributes (Form / Modal)
          │
          ▼
Client-Side Validation (Range & Type Checks)
          │
          ▼
POST /api/v1/assessments/heart (Spring Boot)
          │
          ▼
Server-Side Request Validation
          │
          ▼
POST /predict/heart (FastAPI ML Service)
          │
          ▼
Pipeline Preprocessing + Model Inference + SHAP Explanation
          │
          ▼
Return Calibrated Risk Probability + Feature Attributions
          │
          ▼
Persist Record (MySQL `disease_assessments`)
          │
          ▼
Render Results on Patient Dashboard
  ├─ Estimated Risk Probability (0.0 to 1.0)
  ├─ Key Contributing Factors (SHAP Bar / Waterfalls)
  └─ Screening-Oriented Educational Guidance
```

1. **Access Assessment**: The authenticated patient navigates to the Assessments module and selects "Heart Disease".
2. **Clinical Data Input**: The user enters their clinical parameters (e.g., resting blood pressure, cholesterol, chest pain category, exercise test metrics).
3. **Multi-Tier Validation**: Client-side validation guards against malformed values; Spring Boot validates the payload against strict physiological bounds before forwarding to the ML service.
4. **Model Inference**: The Python ML service transforms inputs using the trained pipeline, computes the calibrated probability of significant coronary artery disease, and derives SHAP feature attributions.
5. **Persistence**: Spring Boot records the assessment in `disease_assessments` (storing the genuine probability in `riskScore`, serialized SHAP in `inputFeatures`, and leaving unverified risk tiers as `null`).
6. **Presentation**: The patient receives their calibrated probability, visual breakdown of top risk drivers, and educational screening guidance with an explicit recommendation to consult a physician.
7. **Diagnostic Disclaimers**: The interface explicitly displays: *"This assessment is an algorithmic risk screening tool and does not constitute a clinical medical diagnosis."*

---

### Journey B: Returning Patient (Longitudinal Tracking & Contextual Assessment)

This journey accommodates patients who have previously logged assessments or vital observations over time.

```
Returning Patient
          │
          ▼
Submits New Heart Assessment Parameters
          │
          ▼
Spring Boot Orchestrator
  ├─ 1. Dispatches New Features to ML Service -> Obtains Current Calibrated Risk
  └─ 2. Queries Historical `health_observations` & `personal_baselines`
          │
          ▼
Context & Trend Synthesis
  ├─ Evaluates Vital Trends (e.g., Resting Blood Pressure Trajectory)
  ├─ Measures Baseline Deviation (e.g., Cholesterol Shift vs. Personal Mean)
  └─ Compares with Prior Heart Assessments
          │
          ▼
Combined Dashboard Presentation
  ├─ Current Assessment Risk Probability (ML-driven)
  ├─ Longitudinal Risk Trajectory (Prior vs. Current Assessment)
  ├─ Personal Baseline Variance (Vital-specific Context)
  └─ Context-Aware Guidance
```

1. **Longitudinal Context Retrieval**: Upon submitting new parameters, the platform accesses the patient's historical records.
2. **Independent Risk Inference**: The ML model calculates the current risk probability strictly from the current clinical feature set, preserving model invariance.
3. **Personalized Context Layer**: The backend retrieves existing baselines for continuous metrics (e.g., `trestbps`, `chol`, `thalach`) from `personal_baselines` and computes trend slopes from `trend_analyses`.
4. **Separation of Signals**: The UI presents both signals distinctly:
   - **Population Risk Estimate**: *"The model estimates a 42% probability of coronary artery disease based on current clinical indicators."*
   - **Personal Longitudinal Context**: *"Your resting blood pressure has decreased by 12 mm Hg compared to your 6-month personal baseline."*
5. **Non-Diagnostic Trend Policy**: Longitudinal improvements or worsening do NOT alter the raw model output; they provide clinical context alongside the model probability.

---

### Journey C: Continuous Monitoring & Metric Ingestion (Conceptual Foundation)

This journey details how individual cardiovascular vitals participate in the platform's longitudinal monitoring architecture (`health_observations` → `personal_baselines` → `trend_analyses` → `alerts`).

```
Periodic Vital Measurement (e.g., Blood Pressure, Resting Heart Rate)
                               │
                               ▼
            POST /api/v1/monitoring/metrics (Spring Boot)
                               │
                               ▼
                 Persist in `health_observations`
                               │
                               ▼
                  `PersonalBaselineService`
            (Updates Running Mean, Variance, Count)
                               │
                               ▼
                   `TrendAnalysisService`
        (Determines Slope, Direction: IMPROVING / WORSENING)
                               │
                               ▼
            Threshold / Sustained Deviation Detection
                               │
                               ▼
        System Alert Generated (e.g., Sustained Hypertension)
                               │
                               ▼
       Informs Next Scheduled Heart Disease Risk Assessment
```

- **Monitored Cardiovascular Metrics**:
  - `systolic_blood_pressure` / `diastolic_blood_pressure` (corresponds to `trestbps`)
  - `resting_heart_rate`
  - `total_cholesterol` (corresponds to `chol`)
  - `fasting_blood_glucose` (corresponds to `fbs`)
  - `activity_minutes` / `exercise_frequency`
- **Baseline Computation**: Vitals logged over time update the patient's individual running baseline without triggering an entire ML model execution on partial data.
- **Alert Generation**: Sustained high blood pressure or extreme deviations trigger notifications advising clinical evaluation or suggesting a comprehensive Heart Disease assessment.
- **Future Scope Boundary**: Automated continuous sensor/wearable polling is designated for future phases. In this phase, metric ingestion occurs via standard authenticated REST requests.

---

### Journey D: Healthcare Provider / Clinician Review

This journey allows authorized medical practitioners to review a patient's cardiovascular profile during clinical consultations.

1. **Provider View**: Clinician accesses the patient's cardiovascular summary.
2. **Review Assessment Timeline**: Chronological history of all Heart Disease assessments, showing changes in estimated probability over time.
3. **Explainability Inspection**: Clinicians can inspect the SHAP feature importance plot for any historical assessment to understand which specific physiological factors drove the model's output (e.g., high fluoroscopy vessel count vs. thallium reversible defect).
4. **Longitudinal Corroboration**: Clinicians review longitudinal trends in blood pressure, heart rate, and cholesterol alongside the model outputs.
5. **Decision Support**: The platform serves exclusively as an assistive screening tool; diagnostic and therapeutic decisions remain under clinician authority.

---

### Journey E: Conversational Chatbot — Heart Domain

The platform's grounded conversational chatbot interacts with the Heart Disease domain under strict architectural and safety boundaries.

```
User Message: "I have high cholesterol and get chest tightness when climbing stairs."
                                   │
                                   ▼
                 Emergency Safety Gate (SafetyChecker)
      (Detects Red Flags: crushing chest pain, arm radiation, syncope)
         ├─ RED FLAG DETECTED ──► Immediate Emergency 911 Redirection
         └─ NO ACUTE RED FLAG
                   │
                   ▼
      Deterministic Symptom & Context Extraction
  (Extracts: chest tightness, exertion trigger, high cholesterol history)
                   │
                   ▼
       Clinical Knowledge Grounding (Cardiovascular Domain)
                   │
                   ▼
       Context Integration (Prior Heart Assessments in DB)
                   │
                   ▼
       Grounded LLM Response Synthesis & Response Validation
                   │
                   ▼
  Natural Conversational Guidance:
  - Explains relationship between exertion symptoms and cardiac evaluation
  - Cites historical assessment probability if already completed
  - Recommends structured Heart Disease Assessment or doctor visit
  - DOES NOT INVENT OR PREDICT A RISK PROBABILITY
```

#### Permitted Chatbot Capabilities
- Answering educational questions regarding coronary disease, risk factors, and diagnostic tests (e.g., explaining what an angiogram or thallium scan measures).
- Gathering structured cardiovascular symptoms (chest pain character, exertion relation, duration) to assist the patient in preparing for an assessment.
- Explaining the results of an **already completed** Heart Disease ML assessment, translating SHAP feature impacts into plain language.
- Guiding the patient to schedule a clinical appointment or take the structured assessment.

#### Strict Chatbot Prohibitions
- **NO Independent Prediction**: The chatbot must NEVER generate, calculate, or hallucinate a numerical risk probability or percentage on its own.
- **NO Diagnostic Pronouncements**: The chatbot must never state *"You have coronary artery disease."*
- **Single Source of Truth**: The Python ML service remains the sole authority for risk calculations.

---

## 4. Assessment Input Contract Concept

Based on the verified 14-attribute Cleveland schema from Phase 0, the conceptual input contract for the Heart Disease assessment is defined below:

| Feature Name | Clinical Label | Source / Origin | Required | Expected Representation | Conceptual Validation Range | Clinical Significance |
| :--- | :--- | :--- | :---: | :--- | :--- | :--- |
| **`age`** | Patient Age | User / Profile | Yes | Integer years | 18 – 120 years (Cohort: 29–77) | Primary demographic risk factor for atherosclerosis. |
| **`sex`** | Biological Sex | User / Profile | Yes | Coded / String (`Male`, `Female`) | `Male` (1.0), `Female` (0.0) | Gender disparity in CAD presentation and age of onset. |
| **`cp`** | Chest Pain Type | User Input | Yes | Categorical Selection (4 levels) | `1`: Typical Angina<br>`2`: Atypical Angina<br>`3`: Non-anginal Pain<br>`4`: Asymptomatic | Key symptomatic discriminator of myocardial ischemia. |
| **`trestbps`** | Resting Blood Pressure | User / Vitals | Yes | Continuous (mm Hg) | 60 – 250 mm Hg (Cohort: 94–200) | Major hemodynamic marker for vascular resistance and hypertension. |
| **`chol`** | Serum Cholesterol | Lab Report | Yes | Continuous (mg/dl) | 80 – 600 mg/dl (Cohort: 126–564) | Lipid biomarker; elevated levels accelerate plaque formation. |
| **`fbs`** | Fasting Blood Sugar > 120 mg/dl | Lab Report | Yes | Binary (`True` / `False`) | `1.0`: >120 mg/dl<br>`0.0`: <=120 mg/dl | Surrogate marker for pre-diabetes / diabetes comorbidity. |
| **`restecg`** | Resting Electrocardiogram | Diagnostic Test | Yes | Categorical Selection (3 levels) | `0`: Normal<br>`1`: ST-T wave abnormality<br>`2`: LV hypertrophy | Identifies resting electrical conduction and structural heart changes. |
| **`thalach`** | Maximum Heart Rate Achieved | Stress Test | Yes | Continuous (bpm) | 50 – 250 bpm (Cohort: 71–202) | Marker of chronotropic reserve during exercise stress testing. |
| **`exang`** | Exercise-Induced Angina | Clinical History | Yes | Binary (`Yes` / `No`) | `1.0`: Yes<br>`0.0`: No | Powerful physiological sign of demand-ischemia mismatch. |
| **`oldpeak`** | ST Depression (Exercise vs Rest) | Stress Test ECG | Yes | Continuous (mm) | 0.0 – 10.0 mm (Cohort: 0.0–6.2) | Quantitative electrocardiographic measure of myocardial ischemia. |
| **`slope`** | Peak Exercise ST Segment Slope | Stress Test ECG | Yes | Categorical Selection (3 levels) | `1`: Upsloping<br>`2`: Flat<br>`3`: Downsloping | Direction of ST deviation during maximum cardiac exertion. |
| **`ca`** | Fluoroscopy Major Vessels (0–3) | Catheterization | Yes* | Integer Count | 0, 1, 2, 3 vessels colored (or `null` if missing in raw data) | Anatomical count of calcified/stenosed major coronary vessels. |
| **`thal`** | Thallium Scintigraphy Defect | Nuclear Stress Test | Yes* | Categorical Selection (3 levels) | `3`: Normal<br>`6`: Fixed Defect<br>`7`: Reversible Defect (or `null`) | Nuclear perfusion imaging demonstrating fixed infarct vs reversible ischemia. |

*\*Note: While `ca` and `thal` contain missing entries in the historical cohort, user assessment submissions will require complete entry or trigger standardized pipeline imputation established during Phase 5.*

---

## 5. User-Facing vs. Model-Ready Representation

To provide an optimal patient experience while maintaining rigorous machine learning compatibility, the presentation layer must decouple clinical terminology from backend numerical codes:

| Field | User-Facing Form Option | Intermediate DTO (Spring Boot) | Model Pipeline Representation (Python) |
| :--- | :--- | :--- | :--- |
| **`sex`** | Radio: "Male" / "Female" | `String gender` ("Male", "Female") | `float64` (`1.0`, `0.0`) |
| **`cp`** | Dropdown:<br>• "Typical Angina (substernal pressure with exertion)"<br>• "Atypical Angina (atypical chest pain)"<br>• "Non-Anginal Pain (unrelated to exertion)"<br>• "Asymptomatic (no chest pain)" | `Integer chestPainType` (1, 2, 3, 4) | One-hot encoded vector via pipeline transformer |
| **`fbs`** | Radio: "Yes (> 120 mg/dL)" / "No (<= 120 mg/dL)" | `Boolean fastingBloodSugarOver120` | `float64` (`1.0`, `0.0`) |
| **`restecg`** | Dropdown:<br>• "Normal"<br>• "ST-T Wave Abnormality (T-wave inversions)"<br>• "Left Ventricular Hypertrophy (Estes' criteria)" | `Integer restingEcg` (0, 1, 2) | One-hot encoded vector via pipeline transformer |
| **`exang`** | Radio: "Yes" / "No" | `Boolean exerciseInducedAngina` | `float64` (`1.0`, `0.0`) |
| **`slope`** | Dropdown:<br>• "Upsloping (normal response to exercise)"<br>• "Flat (horizontal ST segment)"<br>• "Downsloping (severe ischemia)" | `Integer stSlope` (1, 2, 3) | Coded ordinal / one-hot encoded vector |
| **`ca`** | Select: "0", "1", "2", "3" vessels | `Integer majorVesselsCount` (0–3) | Numeric count `float64` |
| **`thal`** | Dropdown:<br>• "Normal (normal blood flow)"<br>• "Fixed Defect (prior infarction / scar tissue)"<br>• "Reversible Defect (transient ischemia during stress)" | `Integer thalliumStressTest` (3, 6, 7) | One-hot encoded vector via pipeline transformer |

*Design Rule: Under no circumstances should legacy numerical codes (e.g. `thal=7`, `cp=4`) be displayed to patients without clear descriptive clinical context.*

---

## 6. Conceptual Validation Rules

To prevent malformed inputs from contaminating inference pipelines, multi-layer validation is enforced:

### 1. Structural & Presence Validation
- All required clinical parameters must be supplied. Missing fields result in HTTP `400 Bad Request` with explicit field-level error messages.
- Request payloads must adhere to JSON schemas.

### 2. Numerical Boundary Validation
- `age`: Must be an integer between 18 and 120.
- `trestbps`: Must be a numeric value between 60.0 and 250.0 mm Hg.
- `chol`: Must be a numeric value between 80.0 and 600.0 mg/dl.
- `thalach`: Must be a numeric value between 50.0 and 250.0 bpm.
- `oldpeak`: Must be a numeric value between 0.0 and 10.0 mm.

### 3. Categorical & Domain Integrity
- `sex`: Must be one of `['Male', 'Female']`.
- `cp`: Must be an integer in `[1, 2, 3, 4]`.
- `fbs`: Must be a boolean or binary `{0, 1}`.
- `restecg`: Must be an integer in `[0, 1, 2]`.
- `exang`: Must be a boolean or binary `{0, 1}`.
- `slope`: Must be an integer in `[1, 2, 3]`.
- `ca`: Must be an integer in `[0, 1, 2, 3]`.
- `thal`: Must be an integer in `[3, 6, 7]`.

---

## 7. Risk Result Structure Design

Following platform architecture, the output of the Heart Disease assessment is designed as follows:

```json
{
  "assessmentId": 1042,
  "patientId": 45,
  "disease": "HEART_DISEASE",
  "assessedAt": "2026-09-30T10:15:30Z",
  "probability": 0.384,
  "riskLevel": null,
  "model": {
    "name": "Heart_Disease_Classifier",
    "version": "1.0.0",
    "calibrationMethod": "isotonic"
  },
  "explanation": {
    "baseValue": 0.458,
    "outputSpace": "probability",
    "features": [
      {
        "feature": "thal",
        "displayName": "Thallium Defect",
        "value": "Reversible Defect",
        "shapValue": 0.142,
        "impact": "INCREASES_RISK"
      },
      {
        "feature": "ca",
        "displayName": "Fluoroscopy Vessels",
        "value": 2,
        "shapValue": 0.089,
        "impact": "INCREASES_RISK"
      },
      {
        "feature": "thalach",
        "displayName": "Max Heart Rate",
        "value": 172,
        "shapValue": -0.065,
        "impact": "DECREASES_RISK"
      }
    ]
  },
  "guidance": "Your assessed probability of significant coronary artery disease is 38.4%. Key factors contributing to this estimate include your thallium stress test findings and fluoroscopy vessel count, counterbalanced by good exercise heart rate response. Please discuss these findings with a cardiologist."
}
```

### Critical Design Constraints:
- **Primary Metric**: The genuine calibrated probability (`probability`) is the primary quantitative output.
- **No Arbitrary Risk Tiers**: In Phase 1, `riskLevel` remains `null`. Arbitrary thresholds (such as `<0.33` = LOW, `0.33-0.66` = MODERATE, `>0.66` = HIGH) are NOT defined or forced prior to Phase 6 empirical threshold analysis.
- **Explainability**: Local SHAP values explain *why* the model produced the probability by displaying feature attributions.

---

## 8. Safety & Diagnostic Boundaries

1. **Non-Diagnostic Nature**: The platform is strictly an assistive screening and risk-estimation platform. It does NOT diagnose coronary heart disease, prescribe therapy, or substitute for formal coronary angiography or clinical consultation.
2. **Acute Emergency Protocol (Red Flags)**:
   - If a patient reports acute cardiovascular warning signs (e.g., *crushing or radiating chest pain, severe shortness of breath, unexplained fainting, diaphoresis*), the system must immediately trigger an emergency alert modal directing the patient to emergency medical services (911 or local emergency facility).
   - Emergency gating takes precedence over routine assessment forms or conversational flows.
3. **Mandatory Disclaimers**: Every assessment view, exportable report, and conversational explanation must carry a visible disclaimer stating that results are estimates derived from statistical models and clinical history.

---

## 9. Personalization & Baseline Boundaries

To maintain scientific validity, the platform strictly separates two distinct analytical domains:

| Dimension | Population Risk (ML Model) | Personal Context & Trajectory (Baseline Engine) |
| :--- | :--- | :--- |
| **Question Answered** | *"What is the probability of coronary disease given this cross-sectional snapshot of clinical variables?"* | *"How does the patient's current physiological parameter compare to their own historical normal range?"* |
| **Mathematical Basis** | Supervised statistical machine learning trained on the clinical cohort. | Rolling temporal statistics (mean, variance, slope) calculated over the individual's history. |
| **Data Requirements** | Full 13-feature clinical panel. | Individual metric streams over time (e.g. blood pressure log). |
| **Separation Protocol** | Baseline trends do NOT modify or corrupt the ML model's weights or probability. | Model risk does NOT overwrite historical metric variance. |
| **Reporting Standard** | Displayed as a distinct statistical probability (e.g. 38.4%). | Displayed as contextual trend indicators (e.g. -12 mm Hg over 90 days). |

---

## 10. Future IoT / Wearable Monitoring Relationship

- **Conceptual Mapping**: While active IoT hardware connections are out-of-scope for Phase 1, the Heart Disease module is architected to seamlessly consume continuous vitals in future phases.
- **Data Conduit**: Wearable sensors (e.g., smartwatches, continuous blood pressure cuffs) will feed into the existing `POST /api/v1/monitoring/metrics` endpoint, populating `health_observations`.
- **Pre-Population**: When a patient initiates a formal Heart Disease assessment, available vitals from recent observations (such as latest resting blood pressure or resting heart rate) can be pre-filled into the form for user confirmation.
- **No Premature Implementation**: No custom Bluetooth, WebBluetooth, or proprietary device drivers are implemented in this phase.

---

## 11. Verification Checklist

- [x] Phase 1 specification document exists at [`docs/HEART_USER_JOURNEYS.md`](file:///d:/smart-healthcare-platform/docs/HEART_USER_JOURNEYS.md).
- [x] All 5 user journeys (New Patient, Returning Patient, Continuous Monitoring, Provider Review, Chatbot) defined.
- [x] Clear architectural separation between ML probability and personal baseline/trend.
- [x] Chatbot strictly bounded to conversational guidance; prohibited from independent prediction.
- [x] Conceptual input contract covers the verified 13 predictive features from Phase 0.
- [x] Explicit distinction between user-facing presentation and model representations.
- [x] No arbitrary risk tiers (LOW/MODERATE/HIGH) invented.
- [x] No ML models trained, no code changed, no services restarted.
- [x] Liver Disease implementation remains 100% untouched.
- [x] Diabetes and Parkinson's modules remain untouched.
