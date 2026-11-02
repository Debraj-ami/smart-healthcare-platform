import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

SAMPLE_LIVER_INPUT = {
    "age": 62,
    "gender": "Male",
    "totalBilirubin": 10.9,
    "directBilirubin": 5.5,
    "alkphos": 699,
    "sgpt": 64,
    "sgot": 100,
    "totalProteins": 7.5,
    "albumin": 3.2,
    "agRatio": 0.74,
}

SAMPLE_HEALTHY_INPUT = {
    "age": 30,
    "gender": "Female",
    "totalBilirubin": 0.7,
    "directBilirubin": 0.2,
    "alkphos": 160,
    "sgpt": 20,
    "sgot": 22,
    "totalProteins": 7.2,
    "albumin": 4.1,
    "agRatio": 1.2,
}


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "UP"
    assert data["service"] == "smart-healthcare-ml-service"
    assert "version" in data
    assert len(data["modelsLoaded"]) >= 1


def test_valid_liver_prediction():
    response = client.post("/api/predict/liver", json=SAMPLE_LIVER_INPUT)
    assert response.status_code == 200
    data = response.json()

    assert data["disease"] == "LIVER_DISEASE"
    assert "model" in data
    assert data["model"]["name"] == "Logistic Regression (Isotonic Calibrated)"
    assert data["model"]["version"] == "1.0.0"

    # Probability must be a valid float between 0 and 1
    assert isinstance(data["probability"], float)
    assert 0.0 <= data["probability"] <= 1.0

    # Explanation structure
    assert "explanation" in data
    assert "features" in data["explanation"]
    assert len(data["explanation"]["features"]) == 10

    # Input summary
    assert data["inputSummary"]["age"] == 62
    assert data["inputSummary"]["gender"] == "Male"
    assert data["inputSummary"]["totalBilirubin"] == 10.9

    # Research disclaimer
    assert "disclaimer" in data
    assert "research-based" in data["disclaimer"].lower()


def test_shap_explanation_structure():
    response = client.post("/api/predict/liver", json=SAMPLE_LIVER_INPUT)
    assert response.status_code == 200
    data = response.json()

    explanation = data["explanation"]
    assert "baseValue" in explanation
    assert explanation["outputSpace"] == "log-odds"

    features = explanation["features"]
    assert len(features) == 10

    # Ensure required fields in each contribution item
    for item in features:
        assert "feature" in item
        assert "transformedFeature" in item
        assert "value" in item
        assert "contribution" in item
        assert item["direction"] in ["INCREASES_RISK", "DECREASES_RISK", "NEUTRAL"]

    # Verify features are ranked by absolute magnitude descending
    magnitudes = [abs(f["contribution"]) for f in features]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_deterministic_prediction():
    res1 = client.post("/api/predict/liver", json=SAMPLE_LIVER_INPUT).json()
    res2 = client.post("/api/predict/liver", json=SAMPLE_LIVER_INPUT).json()

    assert res1["probability"] == res2["probability"]
    assert res1["explanation"]["baseValue"] == res2["explanation"]["baseValue"]
    assert [f["contribution"] for f in res1["explanation"]["features"]] == [
        f["contribution"] for f in res2["explanation"]["features"]
    ]


def test_distinct_inputs_produce_different_predictions():
    res_high = client.post("/api/predict/liver", json=SAMPLE_LIVER_INPUT).json()
    res_low = client.post("/api/predict/liver", json=SAMPLE_HEALTHY_INPUT).json()

    assert res_high["probability"] != res_low["probability"]
    # The high clinical indicator sample should have higher probability
    assert res_high["probability"] > res_low["probability"]


def test_missing_required_field():
    bad_input = SAMPLE_LIVER_INPUT.copy()
    del bad_input["age"]
    response = client.post("/api/predict/liver", json=bad_input)
    assert response.status_code == 422


def test_invalid_gender():
    bad_input = SAMPLE_LIVER_INPUT.copy()
    bad_input["gender"] = "UnknownGender"
    response = client.post("/api/predict/liver", json=bad_input)
    assert response.status_code == 422


def test_invalid_numeric_values():
    # Negative total bilirubin
    bad_input = SAMPLE_LIVER_INPUT.copy()
    bad_input["totalBilirubin"] = -2.5
    response = client.post("/api/predict/liver", json=bad_input)
    assert response.status_code == 422

    # String where number expected
    bad_input2 = SAMPLE_LIVER_INPUT.copy()
    bad_input2["age"] = "not_a_number"
    response2 = client.post("/api/predict/liver", json=bad_input2)
    assert response2.status_code == 422


def test_boundary_validation():
    # Age boundary
    boundary_input = SAMPLE_LIVER_INPUT.copy()
    boundary_input["age"] = 0
    response = client.post("/api/predict/liver", json=boundary_input)
    assert response.status_code == 422

    boundary_input["age"] = 121
    response = client.post("/api/predict/liver", json=boundary_input)
    assert response.status_code == 422


def test_alias_alkaline_phosphatase():
    alias_input = SAMPLE_LIVER_INPUT.copy()
    del alias_input["alkphos"]
    alias_input["alkalinePhosphatase"] = 699
    response = client.post("/api/predict/liver", json=alias_input)
    assert response.status_code == 200
    assert response.json()["inputSummary"]["alkalinePhosphatase"] == 699
