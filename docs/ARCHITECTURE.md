# System Architecture

## Overview
The Smart Healthcare Platform is a modular system for multi-disease risk prediction (Heart Disease, Diabetes, Parkinson's Disease, Liver Disease) and health monitoring.

---

## Technical Layers

### 1. Presentation Layer (`frontend/`)
- **Technology**: React
- **Responsibilities**:
  - Patient and provider portal interfaces.
  - Interactive risk assessment input forms for each disease.
  - Longitudinal monitoring dashboards, trend visualizers, and chatbot interface.

### 2. Core Application & Orchestration Layer (`backend/`)
- **Technology**: Java Spring Boot
- **Responsibilities**:
  - Authentication, authorization, and user management.
  - Orchestrating requests between the frontend, MySQL database, Python ML service, and Chatbot service.
  - Business rules, validation, and historical monitoring data management.

### 3. Machine Learning Inference Layer (`ml-service/`)
- **Technology**: Python FastAPI
- **Responsibilities**:
  - Serving trained machine learning models for:
    - Heart Disease prediction
    - Diabetes prediction
    - Parkinson's Disease prediction
    - Liver Disease prediction
  - Providing inference endpoints and returning risk probabilities and classification outputs.

### 4. Conversational Assistant Layer (`chatbot-service/`)
- **Technology**: Python (FastAPI / LLM service)
- **Responsibilities**:
  - Personalized and context-aware conversational health assistant.
  - Interfacing with LLM endpoints and incorporating user health context.

### 5. Data Persistence Layer (`database/`)
- **Technology**: MySQL
- **Responsibilities**:
  - Primary relational store for user profiles, clinical records, prediction history, and monitoring metrics.

---

## High-Level Interaction Flow

```
[ React Frontend ]
        │
        ▼ (HTTP / REST)
[ Java Spring Boot Backend ] ──────────────► [ MySQL Database ]
        │                       (JPA / JDBC)
        ├──────────────────────► [ Python FastAPI ML Service ] (Prediction Endpoints)
        │
        └──────────────────────► [ Python Chatbot Service ] (Conversational LLM)
```
---

## Personalization and Explainability Flow

[ MySQL / Historical Health Data ]
                │
                ▼
[ Personal Baseline Engine ]
                │
                ▼
[ Context + Trend Analysis ]
                │
                ▼
[ Disease Risk Assessment ]
                │
                ▼
[ SHAP Explainability ]
                │
                ▼
[ Spring Boot Backend ]
                │
                ▼
[ React Dashboard ]

---

## Detailed ML Prediction Flow

[ React Frontend ]
        │
        │ Health Parameters
        ▼
[ Spring Boot Backend ]
        │
        │ Validated Prediction Request
        ▼
[ FastAPI ML Service ]
        │
        ├──► Preprocessing
        │
        ├──► Disease-Specific Model
        │
        └──► SHAP Explanation
                │
                ▼
        [ Prediction Result ]
                │
                ▼
        [ Spring Boot ]
                │
        ├──► Store Result in MySQL
        │
        └──► Return Result to React

---

## Conversational Symptom Flow

[ User ]
    │
    ▼
[ React Chat Interface ]
    │
    ▼
[ Chatbot Service ]
    │
    ▼
[ LLM / Transformer ]
    │
    ├──► Understand User Message
    │
    ├──► Extract Symptoms
    │
    └──► Ask Follow-up Questions
             │
             ▼
     [ Structured Information ]
             │
             ▼
     [ Spring Boot Backend ]
             │
             ▼
     [ Assessment / ML Service ]
             │
             ▼
     [ Result + Explanation ]
             │
             ▼
     [ Chatbot Service ]
             │
             ▼
           [ User ]

---

## Service Responsibilities

### React Frontend

Responsible for:
- User interaction
- Data entry
- Visualization
- Dashboard
- Chat interface

### Spring Boot Backend

Responsible for:
- Authentication and authorization
- User and profile management
- Data validation
- Database access
- Business logic
- Service orchestration
- Prediction history
- Alerts

### FastAPI ML Service

Responsible for:
- ML preprocessing
- Disease-specific inference
- Risk probability/classification
- SHAP-based model explanation

### Chatbot Service

Responsible for:
- Natural-language conversation
- Symptom extraction
- Follow-up questions
- Conversational context
- Explaining structured system results

### MySQL

Responsible for persistent storage of:
- Users
- Health profiles
- Health observations
- Prediction records
- Personal baseline information
- Trend information
- Alerts
- Chat sessions and messages