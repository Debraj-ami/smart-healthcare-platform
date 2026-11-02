# Smart Healthcare Platform — Backend Foundation (Phase 10)

Spring Boot 3.4.x backend foundation for the Smart Healthcare Platform ("An AI-Based Personalized and Context-Aware Smart Healthcare Platform for Multi-Disease Risk Prediction and Monitoring").

---

## 1. Architectural Overview & Boundaries

The Spring Boot backend functions as the **core application and data persistence layer**.

```
[ React Frontend (Phase 11) ]
           │
           ▼ (HTTP / REST)
[ Spring Boot Backend (Phase 10) ] ─────────► [ MySQL Database ]
           │                                 (smart_healthcare_db)
           ├────────────────────────► [ ML FastAPI Service (Port 8000) ]
           │
           └────────────────────────► [ Chatbot FastAPI Service (Port 8001) ]
```

### Safety & Clinical Role Boundaries
- **Non-Diagnostic**: The backend stores and retrieves risk assessments, health observations, and conversation sessions. It does **not** generate clinical diagnoses, compute ML probabilities, or alter model predictions.
- **Orchestration**: The backend acts as the gateway between the frontend user interface and specialized downstream microservices (ML inference and Conversational NLP).

---

## 2. Prerequisites & Technology Stack

- **Java**: 25 (LTS, OpenJDK 25.0.4.1+)
- **Build Tool**: Maven 3.9+ (or included Maven wrapper `./mvnw` / `mvnw.cmd`)
- **Framework**: Spring Boot 3.4.3
- **ORM / Persistence**: Spring Data JPA / Hibernate
- **Database**: MySQL 8.0+
- **Documentation**: Springdoc OpenAPI / Swagger UI (v2.8.5)
- **Monitoring**: Spring Boot Actuator

---

## 3. Database Setup (MySQL)

### Database Name
`smart_healthcare_db`

### DDL Schema Initialization
The complete, reproducible SQL DDL schema is provided at:
```
database/schema/schema.sql
```

To create the database and tables directly in MySQL:
```bash
mysql -u root -p < database/schema/schema.sql
```

Or allow Spring Boot's JPA automatic schema generation (`update` mode) to manage the schema on startup.

---

## 4. Environment Variables Configuration

### Required Variables for MySQL
| Variable | Description | Default Value |
|---|---|---|
| `SERVER_PORT` | HTTP server port | `8080` |
| `DB_HOST` | MySQL host address | `localhost` |
| `DB_PORT` | MySQL port | `3306` |
| `DB_NAME` | Database name | `smart_healthcare_db` |
| `DB_USERNAME` | MySQL user | `root` |
| `DB_PASSWORD` | MySQL password | *(none - must be set)* |
| `JPA_HIBERNATE_DDL_AUTO` | Hibernate DDL strategy | `update` |
| `JPA_SHOW_SQL` | Log SQL queries | `false` |

### Setting Environment Variables in Windows PowerShell

**For the current PowerShell session:**
```powershell
$env:DB_HOST = "localhost"
$env:DB_PORT = "3306"
$env:DB_NAME = "smart_healthcare_db"
$env:DB_USERNAME = "root"
$env:DB_PASSWORD = "your_actual_password"
```

**Or persistently for the current Windows user (so all new PowerShell terminals have it):**
```powershell
[System.Environment]::SetEnvironmentVariable("DB_HOST", "localhost", "User")
[System.Environment]::SetEnvironmentVariable("DB_PORT", "3306", "User")
[System.Environment]::SetEnvironmentVariable("DB_NAME", "smart_healthcare_db", "User")
[System.Environment]::SetEnvironmentVariable("DB_USERNAME", "root", "User")
[System.Environment]::SetEnvironmentVariable("DB_PASSWORD", "your_actual_password", "User")
```

> **Security Note**: Never commit actual credentials or `.env` files with production secrets into version control.

---

## 5. Building and Running

### Run Backend Locally
Using the Maven wrapper from the `backend/` directory:

**Windows (PowerShell/CMD):**
```powershell
.\mvnw.cmd spring-boot:run
```

**Linux/macOS:**
```bash
./mvnw spring-boot:run
```

The server will start at `http://localhost:8080`.

---

## 6. API Documentation (Swagger / OpenAPI)

Once the application is running, the interactive Swagger UI is accessible at:
```
http://localhost:8080/swagger-ui/index.html
```

OpenAPI v3 JSON specification:
```
http://localhost:8080/v3/api-docs
```

---

## 7. Initial REST Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | Health and readiness check |
| `POST` | `/api/users` | Register a new user (`PATIENT`, `DOCTOR`, `ADMIN`) |
| `GET` | `/api/users/{id}` | Retrieve user profile details by ID |
| `GET` | `/api/patients/{id}` | Retrieve patient clinical profile |
| `POST` | `/api/patients/{id}/profile` | Create or update patient clinical profile |
| `POST` | `/api/observations` | Record a time-series health observation |
| `GET` | `/api/observations/patient/{patientId}` | Retrieve observations history for a patient |
| `POST` | `/api/assessments` | Persist a disease risk assessment result |
| `GET` | `/api/assessments/patient/{patientId}` | Retrieve disease assessment history for a patient |
| `POST` | `/api/chat/conversations` | Persist chatbot conversation messages |
| `GET` | `/api/chat/conversations/{sessionId}` | Retrieve messages for a chat session |
| `GET` | `/api/alerts/patient/{patientId}` | Retrieve clinical alerts for a patient |
| `POST` | `/api/baselines/patient/{patientId}/{metricType}/calculate` | Explicitly calculate/recalculate baseline from observations |
| `GET` | `/api/baselines/patient/{patientId}` | Retrieve all computed baselines for a patient |
| `GET` | `/api/baselines/patient/{patientId}/{metricType}` | Retrieve baseline for specific patient and metric |
| `POST` | `/api/trends/patient/{patientId}/{metricType}/analyze` | Execute context-aware trend, deviation score, and persistence analysis |
| `GET` | `/api/trends/patient/{patientId}` | Retrieve historical trend analyses with context for a patient |
| `GET` | `/api/trends/patient/{patientId}/{metricType}` | Retrieve historical trend analyses for specific patient and metric |

---

## 8. Statistical Personalization & Context-Aware Monitoring (Objectives 1 & 2)

> [!IMPORTANT]
> **Clinical Boundary Disclaimer:**
> "The context-aware layer provides descriptive statistical personalization using patient history, temporal context, optional activity context, and individual baseline. It does not itself diagnose disease or produce a clinical severity score."
> 
> The platform strictly separates personalization from disease diagnosis:
> ```
>                    Patient Data
>                         │
>             ┌───────────┴───────────┐
>             ▼                       ▼
>      Disease ML Model       Personalization Engine
>             │                       ├── Baseline Mean & Variability
>             ▼                       ├── Standardized Deviation Score
>      Disease Risk Prob              ├── Trend Direction & Persistence
>                                     ├── Historical Context (Window of 3)
>                                     ├── Temporal Context (4-tier)
>                                     └── Activity Context (Optional)
>                         │
>                         ▼
>                  Combined Result (for later UI)
>                         │
>                         ▼
>                  SHAP Explanation
> ```

### 8.1 Personal Baseline (Objective 1)
Individualized baselines are computed strictly within a patient-metric isolation boundary (`patient_id` + `metric_type`).

- **Arithmetic Mean (Baseline Value):**
  $$\mu = \frac{1}{n} \sum_{i=1}^n x_i$$
- **Variability (Population Standard Deviation):**
  $$\sigma = \sqrt{\frac{1}{n} \sum_{i=1}^n (x_i - \mu)^2} \quad (n > 1); \quad \sigma = 0.0 \quad (n = 1)$$
- **Automatic Incremental Update:** Whenever a new observation is saved via `POST /api/observations`, the patient's baseline for that specific metric is automatically recalculated.

### 8.2 Context-Aware Deviation & Trend Analysis (Objective 2)

#### Standardized Deviation & Absolute Deviation
- **Standardized Deviation Score:**
  $$\text{deviation\_score} = \begin{cases} \frac{x_{\text{curr}} - \mu}{\sigma}, & \sigma > 0 \\ 0.0, & \sigma = 0 \end{cases}$$
- **Absolute Deviation:**
  $$\text{absolute\_deviation} = x_{\text{curr}} - \mu$$

#### Trend Direction & Persistence
- **Trend Direction:**
  - `INCREASING`: Latest consecutive observations are moving strictly upward ($x_n > x_{n-1}$).
  - `DECREASING`: Latest consecutive observations are moving strictly downward ($x_n < x_{n-1}$).
  - `STABLE`: Equal recent values ($x_n = x_{n-1}$) or single observation ($n = 1$).
- **Persistence:** Number of consecutive observations from the latest entry that continuously support the active trend direction.
- **Sequential Value Changes:** Tracks `previousValue`, `valueChangeFromPrevious` ($x_n - x_{n-1}$), and `percentageChangeFromPrevious` ($\frac{x_n - x_{n-1}}{x_{n-1}} \times 100$ guarded against division by zero).

#### Temporal Context
Derived deterministically from the observation timestamp (`loggedAt`):
- `observationDate`: `YYYY-MM-DD`
- `observationTime`: `HH:mm:ss`
- `hourOfDay`: `0`–`23`
- `dayOfWeek`: Day name in uppercase (e.g. `WEDNESDAY`)
- **4-Tier Time of Day:**
  - `00:00:00` – `05:59:59` $\to$ `NIGHT`
  - `06:00:00` – `11:59:59` $\to$ `MORNING`
  - `12:00:00` – `17:59:59` $\to$ `AFTERNOON`
  - `18:00:00` – `23:59:59` $\to$ `EVENING`

#### Patient-History Context
Computed across historical observations of the same `(patientId, metricType)`:
- `observationCount`: Total number of observations recorded
- `historicalMinimum`: Minimum value observed
- `historicalMaximum`: Maximum value observed
- `historicalMean`: Arithmetic mean of all observations
- `recentMean`: Mean of the latest 3 observations (or all available if $< 3$)
- `differenceFromHistoricalMean`: $x_{\text{curr}} - \mu_{\text{historical}}$
- `differenceFromRecentMean`: $x_{\text{curr}} - \mu_{\text{recent}}$

#### Activity Context (Optional)
Prepares the platform for future IoT/wearable integration (Phase 13) without fabricating wearable data:
- `activityLevel`: `UNKNOWN` (default when omitted), `RESTING`, `LIGHT`, `MODERATE`, `VIGOROUS`
- `activityMinutes`: Optional integer ($\ge 0$)

---

## 9. Running Automated Tests

The automated test suite uses an in-memory H2 database configured in MySQL compatibility mode. Tests execute completely independently without requiring a pre-existing MySQL instance or credentials.

To run tests:
```powershell
.\mvnw.cmd clean test
```

Test coverage includes (25 automated test cases):
- Spring Boot Application context loading
- Service health endpoint verification
- User registration, duplicate email handling, validation failure tests, and lookup
- Patient profile creation and retrieval
- Health observation recording and querying
- Personal baseline single and multiple observation calculations (arithmetic mean and population standard deviation)
- Patient and metric baseline isolation
- Incremental baseline recalculation upon new observation posting
- Standardized personal deviation score calculation (zero and non-zero variability)
- Monotonic increasing and decreasing trend identification with persistence tracking
- Stable trend classification on flat and single-observation data
- Disease assessment persistence and querying
- Chatbot conversational history logging and retrieval
- Clinical alert querying
