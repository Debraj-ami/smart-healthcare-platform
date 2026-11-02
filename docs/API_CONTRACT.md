# API Contract

This document provides high-level API contracts connecting the services: React frontend, Java Spring Boot backend, Python FastAPI ML service, and Python Chatbot service.

---

## 1. Backend Service (Java Spring Boot)

Base URL: `/api/v1`

### Authentication & Users
- `POST /api/v1/auth/register`: Register new user.
- `POST /api/v1/auth/login`: Authenticate and obtain JWT token.
- `GET /api/v1/users/me`: Retrieve current authenticated profile.

### Multi-Disease Risk Assessment
- `POST /api/v1/assessments/{disease_type}`:
  - Supported `disease_type`: `heart`, `diabetes`, `parkinsons`, `liver`
  - Forwards feature payload to the ML service, saves record to MySQL, returns risk result.
- `GET /api/v1/assessments/history`: Get historical assessments for authenticated patient.
- `GET /api/v1/assessments/{id}`: Get detailed assessment report.

### Health Monitoring
- `POST /api/v1/monitoring/metrics`: Log patient vital / metric.
- `GET /api/v1/monitoring/metrics`: Retrieve historical metrics for trend visualization.

### Chatbot Gateway
- `POST /api/v1/chat/message`: Send user message and patient context to the Chatbot service.

---

## 2. ML Inference Service (Python FastAPI)

Base URL: `http://localhost:8000`

### Health Check
- `GET /health`: Service and model status check.

### Multi-Disease Inference Endpoints
- `POST /predict/heart`:
  - Request: Heart health features (e.g., age, sex, chest pain type, resting BP, cholesterol, etc.)
  - Response: `{ "risk_score": float, "risk_level": "LOW"|"MODERATE"|"HIGH", "probabilities": {...} }`
- `POST /predict/diabetes`:
  - Request: Diabetes indicators (e.g., glucose, insulin, BMI, age, etc.)
  - Response: `{ "risk_score": float, "risk_level": "LOW"|"MODERATE"|"HIGH", "probabilities": {...} }`
- `POST /predict/parkinsons`:
  - Request: Biomedical voice/motor measurements.
  - Response: `{ "risk_score": float, "risk_level": "LOW"|"MODERATE"|"HIGH", "probabilities": {...} }`
- `POST /predict/liver`:
  - Request: Liver biomarker panel (e.g., bilirubin, enzymes, total proteins, etc.)
  - Response: `{ "risk_score": float, "risk_level": "LOW"|"MODERATE"|"HIGH", "probabilities": {...} }`

---

## 3. Chatbot / LLM Service (Python FastAPI)

Base URL: `http://localhost:8001`

### Chat Endpoint
- `POST /chat`:
  - Request:
    ```json
    {
      "session_id": "string",
      "user_message": "string",
      "patient_context": {
        "recent_assessments": [...],
        "known_conditions": [...]
      }
    }
    ```
  - Response:
    ```json
    {
      "reply": "string",
      "disclaimer": "This assistant does not provide official medical diagnoses."
    }
    ```
