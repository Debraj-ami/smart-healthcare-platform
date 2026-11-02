from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from app.model import load_liver_bundle, predict_liver
from app.heart_service import load_heart_bundle, predict_heart
from app.schemas import (
    HealthResponse,
    LiverPredictionRequest,
    LiverPredictionResponse,
    HeartPredictionRequest,
    HeartPredictionResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm model bundles on application startup
    try:
        load_liver_bundle()
        print("ML Service: Liver model bundle loaded successfully into memory.")
    except Exception as e:
        print(f"ML Service Warning: Failed to pre-load liver bundle on startup: {e}")

    try:
        load_heart_bundle()
        print("ML Service: Heart model bundle loaded successfully into memory.")
    except Exception as e:
        print(f"ML Service Warning: Failed to pre-load heart bundle on startup: {e}")

    yield


app = FastAPI(
    title="Smart Healthcare Platform — ML Inference Service",
    description="Production ML inference microservice for disease risk screening and SHAP explainability.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for Spring Boot orchestration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


from fastapi.encoders import jsonable_encoder


def _sanitize_for_json(obj):
    import math
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return str(obj)
        return obj
    elif isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    return obj


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc: RequestValidationError):
    """
    Sanitizes validation errors to ensure compliance with RFC 7159 JSON specifications
    when invalid inputs contain NaN, infinite floating-point values, or custom exceptions.
    """
    encoded_errors = jsonable_encoder(exc.errors())
    sanitized = _sanitize_for_json(encoded_errors)
    return JSONResponse(status_code=422, content={"detail": sanitized})




@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """
    Health check endpoint returning microservice readiness and loaded models.
    Distinguishes service availability and individual disease model artifact availability.
    """
    loaded = []
    liver_ok = False
    heart_ok = False

    try:
        liver_bundle = load_liver_bundle()
        loaded.append(liver_bundle.get("model_name", "Liver Disease Model"))
        liver_ok = True
    except Exception:
        pass

    try:
        heart_bundle = load_heart_bundle()
        loaded.append(heart_bundle.get("model_name", "Heart Disease Model"))
        heart_ok = True
    except Exception:
        pass

    if liver_ok and heart_ok:
        status_text = "UP"
    elif liver_ok or heart_ok:
        status_text = "DEGRADED"
    else:
        status_text = "DOWN"

    return HealthResponse(
        status=status_text,
        service="smart-healthcare-ml-service",
        version="1.0.0",
        modelsLoaded=loaded,
    )


@app.post(
    "/api/predict/liver",
    response_model=LiverPredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Predictions"],
)
def predict_liver_disease(request: LiverPredictionRequest):
    """
    Generate calibrated disease probability and local SHAP feature contributions
    for the submitted liver clinical measurements.
    """
    try:
        response = predict_liver(request)
        return response
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(ve),
        )
    except FileNotFoundError as fe:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model artifact unavailable: {str(fe)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}",
        )


@app.post(
    "/predict/heart",
    response_model=HeartPredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Predictions"],
)
@app.post(
    "/api/predict/heart",
    response_model=HeartPredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Predictions"],
)
def predict_heart_disease(request: HeartPredictionRequest):
    """
    Generate model-estimated disease probability, threshold-based classification decision,
    and local SHAP feature contributions for the submitted Heart clinical measurements.
    """
    try:
        response = predict_heart(request)
        return response
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(ve),
        )
    except FileNotFoundError as fe:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model artifact unavailable: {str(fe)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8001, reload=True)
