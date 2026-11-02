"""
FastAPI ML Service Heart Disease Tests (Productionization Stage 1).

Validates:
1. Application imports and route registration (/health, /predict/heart, /api/predict/heart).
2. Heart model artifact loading, schema consistency, and single-load caching.
3. Inference correctness: probability range [0, 1], binary prediction (0/1), threshold (0.31).
4. Direct reproduction against serialized model.
5. SHAP explanation structure, log-odds space, additivity efficiency, and clinical feature mapping.
6. Positive and negative contributor sorting and classification.
7. Handling of nullable features (ca, thal) per Phase 4 preprocessing contract.
8. Rejection of missing required inputs, invalid categories, out-of-range numerics, NaNs, and infinities.
9. Strict rejection of target variable 'num'.
10. Safe handling of artifact failure (503 Service Unavailable).
11. Safe handling of SHAP explanation failure without fabricating values.
12. Confirmation that no retraining occurs during inference.
13. Health endpoint multi-model reporting.
"""

import pytest
import math
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app.main import app
from app.heart_service import load_heart_bundle, predict_heart, request_to_heart_dataframe
from app.schemas import HeartPredictionRequest

client = TestClient(app)

SAMPLE_HIGH_RISK_HEART = {
    "age": 63.0,
    "sex": 1.0,
    "cp": 4.0,
    "trestbps": 140.0,
    "chol": 260.0,
    "fbs": 0.0,
    "restecg": 1.0,
    "thalach": 112.0,
    "exang": 1.0,
    "oldpeak": 3.0,
    "slope": 2.0,
    "ca": 2.0,
    "thal": 7.0,
}

SAMPLE_LOW_RISK_HEART = {
    "age": 37.0,
    "sex": 0.0,
    "cp": 3.0,
    "trestbps": 120.0,
    "chol": 215.0,
    "fbs": 0.0,
    "restecg": 0.0,
    "thalach": 170.0,
    "exang": 0.0,
    "oldpeak": 0.0,
    "slope": 1.0,
    "ca": 0.0,
    "thal": 3.0,
}

SAMPLE_NULLABLE_HEART = {
    "age": 52.0,
    "sex": 1.0,
    "cp": 2.0,
    "trestbps": 128.0,
    "chol": 205.0,
    "fbs": 1.0,
    "restecg": 0.0,
    "thalach": 160.0,
    "exang": 0.0,
    "oldpeak": 1.0,
    "slope": 1.0,
    "ca": None,
    "thal": None,
}


# =====================================================================
# 1. APPLICATION & ROUTING TESTS
# =====================================================================

def test_fastapi_app_imports_and_heart_routes_registered():
    """Verify application imports and Heart routes are properly registered."""
    routes = [route.path for route in app.routes]
    assert "/health" in routes
    assert "/predict/heart" in routes
    assert "/api/predict/heart" in routes


def test_health_endpoint_reports_both_liver_and_heart():
    """Verify health endpoint reports status UP and lists both models."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "UP"
    assert data["service"] == "smart-healthcare-ml-service"
    assert len(data["modelsLoaded"]) >= 2
    assert any("Logistic Regression (Robust L1)" in m for m in data["modelsLoaded"])
    assert any("Liver" in m or "Logistic Regression" in m for m in data["modelsLoaded"])


# =====================================================================
# 2. ARTIFACT LOADING & INFERENCE
# =====================================================================

def test_heart_model_artifact_loads_with_required_components():
    """Verify Heart model bundle loads with all required components."""
    bundle = load_heart_bundle()
    assert "preprocessor" in bundle
    assert "classifier" in bundle
    assert "shap_explainer" in bundle
    assert "threshold" in bundle
    assert bundle["threshold"] == 0.31
    assert bundle["model_name"] == "Logistic Regression (Robust L1)"
    assert bundle["model_version"] == "1.0.0"


def test_valid_heart_prediction_high_risk():
    """Verify successful inference for high-risk clinical profile."""
    for endpoint in ["/predict/heart", "/api/predict/heart"]:
        response = client.post(endpoint, json=SAMPLE_HIGH_RISK_HEART)
        assert response.status_code == 200
        data = response.json()

        assert data["disease"] == "HEART_DISEASE"
        assert data["model"]["name"] == "Logistic Regression (Robust L1)"
        assert data["model"]["version"] == "1.0.0"
        assert data["model"]["config_id"] == "LR_robust_l1_C1.0"
        assert data["model"]["output_space"] == "log_odds"

        # Probability
        assert isinstance(data["probability"], float)
        assert 0.0 <= data["probability"] <= 1.0
        assert data["probability"] > 0.5  # High-risk profile

        # Threshold and binary prediction
        assert data["threshold"] == 0.31
        assert "threshold_description" in data
        assert data["prediction"] == 1  # 1 because probability > 0.31

        # Explanation
        assert data["explanation"]["available"] is True
        assert data["explanation"]["output_space"] == "log_odds"
        assert isinstance(data["explanation"]["base_value"], float)
        assert isinstance(data["explanation"]["model_output"], float)

        # Disclaimers
        assert "research-based" in data["disclaimer"].lower()
        assert "not constitute a clinical diagnosis" in data["disclaimer"].lower()


def test_valid_heart_prediction_low_risk():
    """Verify successful inference for low-risk profile."""
    response = client.post("/predict/heart", json=SAMPLE_LOW_RISK_HEART)
    assert response.status_code == 200
    data = response.json()

    assert data["probability"] < 0.31
    assert data["prediction"] == 0
    assert data["threshold"] == 0.31


def test_heart_prediction_with_nullable_ca_and_thal():
    """Verify that ca=None and thal=None are properly imputed by Phase 4 preprocessor."""
    response = client.post("/predict/heart", json=SAMPLE_NULLABLE_HEART)
    assert response.status_code == 200
    data = response.json()
    assert 0.0 <= data["probability"] <= 1.0
    assert data["prediction"] in [0, 1]
    assert data["explanation"]["available"] is True


def test_prediction_matches_serialized_model_direct_inference():
    """Verify that FastAPI response exactly matches direct calculation from model bundle."""
    bundle = load_heart_bundle()
    clf = bundle["classifier"]
    preprocessor = bundle["preprocessor"]

    req = HeartPredictionRequest(**SAMPLE_HIGH_RISK_HEART)
    df_sample = request_to_heart_dataframe(req)
    X_trans = preprocessor.transform(df_sample)

    expected_decision = float(clf.decision_function(X_trans)[0])
    expected_prob = float(clf.predict_proba(X_trans)[0, 1])

    response = client.post("/predict/heart", json=SAMPLE_HIGH_RISK_HEART)
    assert response.status_code == 200
    data = response.json()

    assert math.isclose(data["probability"], round(expected_prob, 4), abs_tol=1e-4)
    assert math.isclose(data["explanation"]["model_output"], round(expected_decision, 4), abs_tol=1e-4)


# =====================================================================
# 3. SHAP EXPLANATION & FEATURE MAPPING
# =====================================================================

def test_shap_explanation_structure_and_additivity():
    """Verify SHAP explanation structure and mathematical additivity in log-odds space."""
    response = client.post("/predict/heart", json=SAMPLE_HIGH_RISK_HEART)
    assert response.status_code == 200
    exp = response.json()["explanation"]

    assert exp["output_space"] == "log_odds"
    base_val = exp["base_value"]
    model_output = exp["model_output"]
    all_contribs = exp["all_contributions"]

    assert len(all_contribs) == 18  # 18 transformed features in Robust L1 pipeline

    # Verify mathematical additivity: base_value + sum(contributions) == model_output
    sum_contribs = sum(item["contribution"] for item in all_contribs)
    reconstructed = base_val + sum_contribs
    assert math.isclose(reconstructed, model_output, abs_tol=0.01)


def test_positive_and_negative_contributions():
    """Verify that positive and negative contributors are partitioned and sorted by magnitude."""
    response = client.post("/predict/heart", json=SAMPLE_HIGH_RISK_HEART)
    assert response.status_code == 200
    exp = response.json()["explanation"]

    pos = exp["positive_contributors"]
    neg = exp["negative_contributors"]

    assert len(pos) > 0
    for item in pos:
        assert item["contribution"] > 0
        assert item["direction"] == "INCREASES_RISK_ESTIMATE"

    assert len(neg) > 0
    for item in neg:
        assert item["contribution"] < 0
        assert item["direction"] == "DECREASES_RISK_ESTIMATE"

    # Sorted descending by absolute magnitude
    pos_mags = [item["abs_contribution"] for item in pos]
    neg_mags = [item["abs_contribution"] for item in neg]
    assert pos_mags == sorted(pos_mags, reverse=True)
    assert neg_mags == sorted(neg_mags, reverse=True)


def test_feature_mapping_and_clinical_labels():
    """Verify transformed features are mapped through clinical feature dictionary."""
    response = client.post("/predict/heart", json=SAMPLE_HIGH_RISK_HEART)
    assert response.status_code == 200
    all_contribs = response.json()["explanation"]["all_contributions"]

    for item in all_contribs:
        assert "feature" in item
        assert "transformed_feature" in item
        assert "display_name" in item
        assert "contribution" in item
        assert "abs_contribution" in item
        assert "direction" in item
        # Ensure display name is not just the raw transformed string
        assert item["display_name"] != ""
        assert not item["display_name"].startswith("cat__")


# =====================================================================
# 4. INPUT VALIDATION & SAFETY GUARDS
# =====================================================================

def test_missing_required_input_rejected():
    """Verify 422 error when a required field is omitted."""
    incomplete = SAMPLE_HIGH_RISK_HEART.copy()
    del incomplete["age"]
    response = client.post("/predict/heart", json=incomplete)
    assert response.status_code == 422


def test_invalid_categorical_values_rejected():
    """Verify 422 error on invalid categorical inputs."""
    # Sex: only 0 or 1
    bad_sex = SAMPLE_HIGH_RISK_HEART.copy()
    bad_sex["sex"] = 2.0
    assert client.post("/predict/heart", json=bad_sex).status_code == 422

    # Chest pain: only 1, 2, 3, 4
    bad_cp = SAMPLE_HIGH_RISK_HEART.copy()
    bad_cp["cp"] = 5.0
    assert client.post("/predict/heart", json=bad_cp).status_code == 422

    # Fasting blood sugar: only 0 or 1
    bad_fbs = SAMPLE_HIGH_RISK_HEART.copy()
    bad_fbs["fbs"] = 3.0
    assert client.post("/predict/heart", json=bad_fbs).status_code == 422

    # Resting ECG: only 0, 1, 2
    bad_ecg = SAMPLE_HIGH_RISK_HEART.copy()
    bad_ecg["restecg"] = 4.0
    assert client.post("/predict/heart", json=bad_ecg).status_code == 422

    # Exercise angina: only 0 or 1
    bad_exang = SAMPLE_HIGH_RISK_HEART.copy()
    bad_exang["exang"] = 2.0
    assert client.post("/predict/heart", json=bad_exang).status_code == 422

    # Slope: only 1, 2, 3
    bad_slope = SAMPLE_HIGH_RISK_HEART.copy()
    bad_slope["slope"] = 0.0
    assert client.post("/predict/heart", json=bad_slope).status_code == 422

    # Fluoroscopy vessels: only None, 0, 1, 2, 3
    bad_ca = SAMPLE_HIGH_RISK_HEART.copy()
    bad_ca["ca"] = 4.0
    assert client.post("/predict/heart", json=bad_ca).status_code == 422

    # Thallium: only None, 3, 6, 7
    bad_thal = SAMPLE_HIGH_RISK_HEART.copy()
    bad_thal["thal"] = 5.0
    assert client.post("/predict/heart", json=bad_thal).status_code == 422


def test_nan_and_infinite_values_rejected():
    """Verify that NaN and Infinite inputs are strictly rejected with 422."""
    # Test via raw HTTP content with NaN/Infinity tokens
    raw_nan_payload = '{"age": 63.0, "sex": 1.0, "cp": 4.0, "trestbps": NaN, "chol": 260.0, "fbs": 0.0, "restecg": 1.0, "thalach": 112.0, "exang": 1.0, "oldpeak": 3.0, "slope": 2.0, "ca": 2.0, "thal": 7.0}'
    res_nan = client.post("/predict/heart", content=raw_nan_payload, headers={"Content-Type": "application/json"})
    assert res_nan.status_code == 422

    raw_inf_payload = '{"age": 63.0, "sex": 1.0, "cp": 4.0, "trestbps": 140.0, "chol": Infinity, "fbs": 0.0, "restecg": 1.0, "thalach": 112.0, "exang": 1.0, "oldpeak": 3.0, "slope": 2.0, "ca": 2.0, "thal": 7.0}'
    res_inf = client.post("/predict/heart", content=raw_inf_payload, headers={"Content-Type": "application/json"})
    assert res_inf.status_code == 422

    # Direct Pydantic schema validation test
    with pytest.raises(ValueError):
        bad = SAMPLE_HIGH_RISK_HEART.copy()
        bad["trestbps"] = float("nan")
        HeartPredictionRequest(**bad)

    with pytest.raises(ValueError):
        bad = SAMPLE_HIGH_RISK_HEART.copy()
        bad["chol"] = float("inf")
        HeartPredictionRequest(**bad)



def test_target_variable_num_rejected():
    """Verify that target variable 'num' is strictly rejected."""
    payload_with_target = SAMPLE_HIGH_RISK_HEART.copy()
    payload_with_target["num"] = 1
    response = client.post("/predict/heart", json=payload_with_target)
    assert response.status_code == 422
    assert "num" in response.text.lower()


def test_out_of_bounds_numerical_values_rejected():
    """Verify that extreme out-of-range clinical inputs are rejected with 422."""
    bad_age = SAMPLE_HIGH_RISK_HEART.copy()
    bad_age["age"] = 150.0
    assert client.post("/predict/heart", json=bad_age).status_code == 422

    bad_bp = SAMPLE_HIGH_RISK_HEART.copy()
    bad_bp["trestbps"] = 10.0
    assert client.post("/predict/heart", json=bad_bp).status_code == 422


# =====================================================================
# 5. ERROR RESILIENCE & INTEGRITY
# =====================================================================

def test_model_artifact_failure_handled_safely():
    """Verify 503 Service Unavailable when model bundle cannot be loaded."""
    with patch("app.heart_service.load_heart_bundle", side_effect=FileNotFoundError("Simulated missing bundle")):
        response = client.post("/predict/heart", json=SAMPLE_HIGH_RISK_HEART)
        assert response.status_code == 503
        assert "Model artifact unavailable" in response.json()["detail"]


def test_shap_failure_does_not_fabricate_explanations():
    """
    Verify that if SHAP calculation fails, prediction succeeds and explanation
    reports failure cleanly without fabricating numbers.
    """
    bundle = load_heart_bundle()
    failing_explainer = MagicMock(side_effect=RuntimeError("Simulated SHAP failure"))

    with patch.dict(bundle, {"shap_explainer": failing_explainer}):
        response = client.post("/predict/heart", json=SAMPLE_HIGH_RISK_HEART)
        assert response.status_code == 200
        data = response.json()

        # Prediction and probability succeed
        assert isinstance(data["probability"], float)
        assert data["probability"] > 0.5
        assert data["prediction"] == 1

        # Explanation gracefully reports failure
        exp = data["explanation"]
        assert exp["available"] is False
        assert "Simulated SHAP failure" in exp["error"]
        assert exp["positive_contributors"] == []
        assert exp["negative_contributors"] == []
        assert exp["all_contributions"] == []


def test_no_model_retraining_occurs_during_inference():
    """Verify that fit() is never called on classifier or preprocessor during inference."""
    bundle = load_heart_bundle()
    clf = bundle["classifier"]
    preprocessor = bundle["preprocessor"]

    with patch.object(clf, "fit") as mock_clf_fit, patch.object(preprocessor, "fit") as mock_prep_fit:
        response = client.post("/predict/heart", json=SAMPLE_HIGH_RISK_HEART)
        assert response.status_code == 200
        mock_clf_fit.assert_not_called()
        mock_prep_fit.assert_not_called()
