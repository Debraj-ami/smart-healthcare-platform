# Smart Healthcare Platform

A real-world AI-based personalized and context-aware smart healthcare platform for multi-disease risk prediction and monitoring.

## Supported Diseases
1. **Heart Disease**
2. **Diabetes**
3. **Parkinson's Disease**
4. **Liver Disease**

---

## Technical Architecture & Services

The platform consists of four major technical layers, with MySQL as the primary database:

| Directory | Layer / Service | Technology | Description |
| :--- | :--- | :--- | :--- |
| `frontend/` | Web Client | React | User interface for patients and healthcare providers for risk assessment and monitoring. |
| `backend/` | Application Backend | Java Spring Boot | Core business logic, authentication, data aggregation, and orchestration between services. |
| `ml-service/` | ML Inference Service | Python FastAPI | Machine learning service serving predictive models for multi-disease risk scoring. |
| `chatbot-service/` | Conversational Assistant | Python (FastAPI / LLM) | Context-aware healthcare assistant for user interaction and health insights. |
| `database/` | Primary Database | MySQL | Relational data persistence (schemas, migrations, seed scripts). |
| `datasets/` | Data Storage | Data Files | Raw and processed datasets for disease prediction. |
| `notebooks/` | Exploration & Training | Jupyter Notebooks | Model research, exploratory data analysis (EDA), and experimentation. |
| `docs/` | Documentation | Markdown | Architectural specifications, database designs, API contracts, and user journeys. |

---

## Documentation

- [User Journeys](file:///d:/smart-healthcare-platform/docs/USER_JOURNEYS.md)
- [System Architecture](file:///d:/smart-healthcare-platform/docs/ARCHITECTURE.md)
- [Database Design](file:///d:/smart-healthcare-platform/docs/DATABASE_DESIGN.md)
- [API Contract](file:///d:/smart-healthcare-platform/docs/API_CONTRACT.md)
