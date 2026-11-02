# Database Design

## Overview
MySQL is the primary relational database for the platform, storing user identities, clinical metrics, risk assessment history, and monitoring data.

---

## Core Entities (High-Level Schema)

### 1. `users`
- Stores user accounts and demographic details.
- Attributes: `id`, `email`, `password_hash`, `first_name`, `last_name`, `date_of_birth`, `gender`, `role` (PATIENT, DOCTOR, ADMIN), `created_at`, `updated_at`.

### 2. `patient_profiles`
- Extended clinical profile for patients.
- Attributes: `id`, `user_id` (FK -> `users.id`), `blood_group`, `height_cm`, `weight_kg`, `lifestyle_context`, `medical_history`.

### 3. `disease_assessments`
- Generic and disease-specific records of assessments conducted.
- Attributes: `id`, `patient_id` (FK -> `users.id`), `disease_type` (`HEART_DISEASE`, `DIABETES`, `PARKINSONS`, `LIVER_DISEASE`), `risk_score`, `risk_level` (LOW, MODERATE, HIGH), `input_features` (JSON or normalized tables), `assessed_at`.

### 4. `health_observations`
- Time-series monitoring logs for periodic patient measurements.
- Attributes: `id`, `patient_id` (FK -> `users.id`), `metric_type` (e.g., blood pressure, glucose, tremor score, liver enzyme panel), `metric_value`, `unit`, `logged_at`.

### 5. `chatbot_conversations`
- History and context for interactions with the conversational assistant.
- Attributes: `id`, `user_id` (FK -> `users.id`), `session_id`, `message_role` (USER, ASSISTANT), `content`, `timestamp`.




### 6. `personal_baselines`

- Stores individualized baseline information calculated from historical
  patient observations.

- Attributes:
  `id`, `patient_id`, `metric_type`, `baseline_value`,
  `variability`, `observation_count`, `calculated_at`, `updated_at`.

### 7. `trend_analyses`

- Stores context-aware analysis of changes in patient health metrics
  relative to the patient's personal baseline.

- Attributes:
  `id`, `patient_id`, `metric_type`, `baseline_id`,
  `current_value`, `deviation_score`, `trend_direction`,
  `persistence`, `analyzed_at`.

### 8. `alerts`

- Stores alerts generated from validated disease assessment or
  context/trend analysis.

- Attributes:
  `id`, `patient_id`, `assessment_id`, `trend_analysis_id`,
  `alert_type`, `severity`, `message`, `status`, `created_at`.

---
## Database Directory Structure (`database/`)
- `database/schema/`: Schema definitions and DDL scripts.
- `database/seeds/`: Initial reference data or seed scripts.