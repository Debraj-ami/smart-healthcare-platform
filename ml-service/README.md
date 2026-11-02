# Smart Healthcare Platform — ML Inference Service

Production microservice for disease risk screening and SHAP explainability.

## Technology Stack
- **Framework**: FastAPI (Python 3.12+)
- **Server**: Uvicorn
- **ML Runtime**: scikit-learn, SHAP, NumPy, Pandas, Joblib
- **Port**: `8001` (default)

## Architecture & Candidate Model
- **Disease**: Liver Disease
- **Model Candidate**: Logistic Regression with 5-fold Isotonic Calibration
- **Training Protocol**: Trained strictly on 456 samples (`X_train`) from the Indian Liver Patient Dataset (ILPD). Preprocessing involves median imputation and standardization for continuous markers, plus binary one-hot encoding for gender.
- **Holdout Performance**:
  - Brier Score: 0.1576
  - Log Loss: 0.4472
  - ROC-AUC: 0.8122
  - PR-AUC: 0.9273
- **Explainability**: Local analytical Shapley values generated via `shap.LinearExplainer` in log-odds space on the scaled feature manifold.

## Endpoints

### 1. Health Check
- `GET /health`
- Returns service status, version, and loaded models.

### 2. Liver Disease Prediction
- `POST /api/predict/liver`
- Accepts 10 clinical features:
  - `age` (1–120)
  - `gender` ("Male" | "Female")
  - `totalBilirubin` (>0)
  - `directBilirubin` (>=0)
  - `alkphos` or `alkalinePhosphatase` (>0)
  - `sgpt` (>0)
  - `sgot` (>0)
  - `totalProteins` (>0)
  - `albumin` (>0)
  - `agRatio` (>0)
- Returns estimated probability, ranked SHAP local feature contributions, input summary, and research disclaimer.

## Running Locally
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

## Running Automated Tests
```bash
pytest tests/ -v
```
