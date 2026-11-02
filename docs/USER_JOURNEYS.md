# User Journeys

This document defines the major user journeys supported by the Smart
Healthcare Platform for multi-disease risk prediction and monitoring.

Supported diseases:
- Heart Disease (see disease-specific specification: [HEART_USER_JOURNEYS.md](HEART_USER_JOURNEYS.md))
- Diabetes
- Parkinson's Disease
- Liver Disease

---

## 1. Patient User Journeys

### 1.1 New User — Symptom-Based Assessment

This journey supports a user who has symptoms but does not yet have
structured medical measurements or reports.

1. **Registration & Profile Setup**
   - User creates an account and provides basic health/profile information.

2. **Start Health Assessment**
   - User describes their health concern or symptoms using natural language.

3. **Conversational Follow-up**
   - The chatbot asks relevant follow-up questions to obtain additional
     information such as duration, severity, frequency, associated symptoms,
     and relevant medical history.

4. **Structured Symptom Information**
   - The chatbot converts the conversation into structured information
     that can be processed by the backend.

5. **Safety & Assessment**
   - The system evaluates whether sufficient information is available for
     further assessment and identifies situations requiring professional
     medical attention.

6. **Relevant Risk Assessment**
   - When applicable and sufficient structured data is available, the system
     sends relevant features to the appropriate disease-specific ML service.

7. **Result & Explanation**
   - The user receives the available risk assessment, contributing factors,
     relevant deviations, and appropriate guidance.

8. **Save Assessment**
   - The assessment and relevant observations are stored for future reference.

---

### 1.2 Structured Multi-Disease Risk Assessment

This journey supports users who already have health measurements,
laboratory values, or relevant medical information.

1. **Login & Profile Access**
   - User logs into the React application.

2. **Select Assessment**
   - User selects Heart Disease, Diabetes, Parkinson's Disease, or Liver Disease.

3. **Submit Health Data**
   - User enters the required disease-specific health parameters.

4. **Input Validation**
   - The system validates the submitted values before prediction.

5. **Disease-Specific ML Prediction**
   - The relevant ML model processes the validated features.

6. **Risk Result**
   - The system returns the prediction and probability/risk information.

7. **Explainability**
   - SHAP-based explanations identify important contributing features.

8. **Save Result**
   - The submitted observations and prediction are stored in the user's
     longitudinal health history.

---

### 1.3 Returning User — Personalized Monitoring

This journey supports users who have accumulated historical health data.

1. **Submit New Observation**
   - User records a new health measurement or observation.

2. **Retrieve Historical Data**
   - The system retrieves relevant previous observations.

3. **Personal Baseline**
   - The system estimates the user's individual baseline from historical data.

4. **Deviation Analysis**
   - The new observation is compared with the user's baseline.

5. **Trend Analysis**
   - The system evaluates changes over time, including direction,
     magnitude, and persistence where sufficient historical data exists.

6. **Disease Risk Assessment**
   - Relevant disease-specific ML models are used when appropriate.

7. **Combined Result**
   - The system presents disease risk information together with
     personalized deviations and trends.

8. **Alert / Guidance**
   - Meaningful changes may generate an alert or appropriate guidance.

9. **Store New Observation**
   - The new observation and resulting analysis are added to the
     longitudinal record.

---

### 1.4 Continuous Monitoring — IoT / Wearable

This journey represents the future continuous-monitoring extension.

1. **Connect Health Device**
   - User connects a supported wearable or IoT source.

2. **Automatic Data Collection**
   - Physiological or activity data is periodically received.

3. **Data Validation**
   - Incoming observations are validated and processed.

4. **Update Personal Baseline**
   - Historical observations are used to maintain the user's baseline.

5. **Context & Trend Analysis**
   - The system evaluates deviations and trends.

6. **Risk / Alert Assessment**
   - Significant changes may trigger appropriate alerts or further assessment.

7. **Dashboard Update**
   - New observations, trends, and alerts become available on the user's
     dashboard.

---

## 2. Healthcare Provider / Clinician User Journeys

### 2.1 Reviewing Patient Risk Profiles

1. **Provider Authentication**
   - Clinician accesses the provider interface.

2. **Patient List**
   - Clinician views patients available to them.

3. **Patient Profile**
   - Clinician opens an individual patient's health profile.

4. **Review Historical Data**
   - Clinician reviews observations, predictions, trends, and alerts.

5. **Inspect Assessment Details**
   - Clinician reviews submitted parameters, model results,
     contributing factors, and relevant personalized deviations.

6. **Clinical Notes / Care Information**
   - Where supported, the clinician can record notes or recommendations.

---

## 3. Conversational Assistant Journey

### 3.1 General Health Question

1. User asks a general health-related question.
2. Chatbot interprets the user's request.
3. Chatbot provides an informational response.
4. The response avoids presenting itself as a definitive medical diagnosis.

### 3.2 Symptom-Based Conversation

1. User describes symptoms in natural language.
2. Chatbot identifies relevant information from the conversation.
3. Chatbot asks appropriate follow-up questions.
4. Structured symptom information is generated.
5. The structured information is passed to the appropriate assessment
   workflow when applicable.
6. The chatbot presents the resulting information in understandable language.
7. The system provides appropriate guidance based on the available assessment.

### 3.3 Context-Aware Conversation

1. With appropriate authorization, the chatbot accesses relevant patient
   context.
2. The chatbot uses available historical observations and assessment results
   to understand the user's question.
3. The chatbot explains existing results and trends.
4. The chatbot does not independently generate disease predictions that are
   outside the validated ML/assessment services.

---

## 4. Core Data Flow

### Patient Measurement Flow

User
  ↓
React Frontend
  ↓
Spring Boot Backend
  ↓
FastAPI ML Service
  ↓
Disease-Specific ML Model
  ↓
SHAP / Explainability
  ↓
Personalization & Context Analysis
  ↓
Spring Boot
  ↓
MySQL
  ↓
React Dashboard

### Symptom Conversation Flow

User
  ↓
React
  ↓
Chatbot Service
  ↓
LLM / Transformer
  ↓
Structured Symptoms
  ↓
Spring Boot
  ↓
Assessment / ML Services
  ↓
Result + Explanation
  ↓
Chatbot
  ↓
User

### Longitudinal Monitoring Flow

New Observation
  ↓
Historical Observations
  ↓
Personal Baseline
  ↓
Context + Trend Analysis
  ↓
Disease Risk Assessment
  ↓
Explanation / Alert
  ↓
Dashboard + Longitudinal Record