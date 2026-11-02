import json
import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from app.main import app, session_manager, extraction_engine
from app.backend_client import set_backend_client
from app.llm_client import MockLLMClient

client = TestClient(app)


def get_mock_patient_4_data():
    """Mock authentic Patient 4 data matching real database records."""
    assessment_10 = {
        "id": 10,
        "patientId": 4,
        "diseaseType": "HEART_DISEASE",
        "riskScore": 0.1127,
        "riskLevel": None,
        "assessedAt": "2026-10-02T00:04:35",
        "inputFeatures": json.dumps({
            "probability": 0.1127,
            "prediction": 0,
            "threshold": 0.31,
            "shapBaseValue": -0.0879,
            "modelOutput": -2.0631,
            "explanationAvailable": True,
            "allContributions": [
                {
                    "feature": "ca",
                    "transformedFeature": "count__ca",
                    "displayName": "Fluoroscopy Vessels (ca)",
                    "rawValue": 0.0,
                    "contribution": -0.6792,
                    "absContribution": 0.6792,
                    "direction": "DECREASES_RISK_ESTIMATE"
                },
                {
                    "feature": "cp",
                    "transformedFeature": "cat__cp_4.0",
                    "displayName": "Chest Pain: Asymptomatic",
                    "rawValue": 1.0,
                    "contribution": -0.5414,
                    "absContribution": 0.5414,
                    "direction": "DECREASES_RISK_ESTIMATE"
                },
                {
                    "feature": "sex",
                    "transformedFeature": "bin__sex",
                    "displayName": "Sex (Male)",
                    "rawValue": 1.0,
                    "contribution": 0.3705,
                    "absContribution": 0.3705,
                    "direction": "INCREASES_RISK_ESTIMATE"
                }
            ],
            "positiveContributors": [
                {
                    "feature": "sex",
                    "transformedFeature": "bin__sex",
                    "displayName": "Sex (Male)",
                    "rawValue": 1.0,
                    "contribution": 0.3705,
                    "absContribution": 0.3705,
                    "direction": "INCREASES_RISK_ESTIMATE"
                }
            ],
            "negativeContributors": [
                {
                    "feature": "ca",
                    "transformedFeature": "count__ca",
                    "displayName": "Fluoroscopy Vessels (ca)",
                    "rawValue": 0.0,
                    "contribution": -0.6792,
                    "absContribution": 0.6792,
                    "direction": "DECREASES_RISK_ESTIMATE"
                },
                {
                    "feature": "cp",
                    "transformedFeature": "cat__cp_4.0",
                    "displayName": "Chest Pain: Asymptomatic",
                    "rawValue": 1.0,
                    "contribution": -0.5414,
                    "absContribution": 0.5414,
                    "direction": "DECREASES_RISK_ESTIMATE"
                }
            ],
            "inputSummary": {
                "age": 58.0,
                "sex": 1.0,
                "cp": 1.0,
                "trestbps": 130.0,
                "chol": 240.0,
                "fbs": 0.0,
                "restecg": 0.0,
                "thalach": 150.0,
                "exang": 0.0,
                "oldpeak": 1.0,
                "slope": 1.0,
                "ca": 0.0,
                "thal": 3.0
            }
        })
    }

    assessment_9 = {
        "id": 9,
        "patientId": 4,
        "diseaseType": "HEART_DISEASE",
        "riskScore": 0.1850,
        "riskLevel": None,
        "assessedAt": "2026-09-15T10:00:00",
        "inputFeatures": json.dumps({
            "probability": 0.1850,
            "prediction": 0,
            "threshold": 0.31
        })
    }

    glucose_trend = {
        "id": 16,
        "patientId": 4,
        "metricType": "GLUCOSE",
        "currentValue": 110.0,
        "baselineValue": 120.0,
        "variability": 14.14,
        "deviationFromBaseline": -10.0,
        "previousValue": 130.0,
        "valueChangeFromPrevious": -20.0,
        "percentageChangeFromPrevious": -15.38,
        "trendDirection": "DECREASING",
        "persistence": 3
    }

    glucose_baseline = {
        "id": 4,
        "patientId": 4,
        "metricType": "GLUCOSE",
        "baselineValue": 120.0,
        "variability": 14.14,
        "observationCount": 6
    }

    glucose_obs = [
        {"id": 19, "patientId": 4, "metricType": "GLUCOSE", "metricValue": 110.0, "unit": "mg/dL", "loggedAt": "2026-09-23T12:56:50"},
        {"id": 18, "patientId": 4, "metricType": "GLUCOSE", "metricValue": 130.0, "unit": "mg/dL", "loggedAt": "2026-09-23T12:56:03"}
    ]

    return {
        "assessments": [assessment_10, assessment_9],
        "trends": [glucose_trend],
        "baselines": [glucose_baseline],
        "observations": glucose_obs
    }


@pytest.fixture(autouse=True)
def setup_context_test():
    session_manager._sessions.clear()
    extraction_engine.llm_client = MockLLMClient()
    yield
    set_backend_client(None)
    session_manager._sessions.clear()


def test_test_a_monitoring_retrieval():
    """TEST A: 'What does my recent health monitoring show?' retrieves real monitoring."""
    data = get_mock_patient_4_data()
    mock_backend = MagicMock()
    mock_backend.get_patient_health_context.return_value = {
        "patient_id": 4,
        "has_history": True,
        "trends": data["trends"],
        "baselines": data["baselines"],
        "recent_observations": data["observations"]
    }
    set_backend_client(mock_backend)

    resp = client.post("/api/chat", json={
        "message": "What does my recent health monitoring show?",
        "session_id": "sess_test_a",
        "user_id": "4"
    })
    assert resp.status_code == 200
    msg = resp.json()["assistant_message"]
    assert "Glucose" in msg or "GLUCOSE" in msg
    assert "110" in msg
    assert "120" in msg
    assert "falling" in msg.lower() or "decreasing" in msg.lower()


def test_test_b_glucose_retrieval():
    """TEST B: 'What does my recent blood glucose monitoring show?' retrieves glucose context."""
    data = get_mock_patient_4_data()
    mock_backend = MagicMock()
    mock_backend.get_patient_health_context.return_value = {
        "patient_id": 4,
        "has_history": True,
        "trends": data["trends"],
        "baselines": data["baselines"],
        "recent_observations": data["observations"]
    }
    set_backend_client(mock_backend)

    resp = client.post("/api/chat", json={
        "message": "What does my recent blood glucose monitoring show?",
        "session_id": "sess_test_b",
        "user_id": "4"
    })
    assert resp.status_code == 200
    msg = resp.json()["assistant_message"]
    assert "110" in msg
    assert "120" in msg
    assert "Glucose" in msg or "GLUCOSE" in msg


def test_test_c_baseline_comparison():
    """TEST C: 'How does my latest glucose compare with my personal baseline?' retrieves deviation."""
    data = get_mock_patient_4_data()
    mock_backend = MagicMock()
    mock_backend.get_patient_health_context.return_value = {
        "patient_id": 4,
        "has_history": True,
        "trends": data["trends"],
        "baselines": data["baselines"],
        "recent_observations": data["observations"]
    }
    set_backend_client(mock_backend)

    resp = client.post("/api/chat", json={
        "message": "How does my latest glucose compare with my personal baseline?",
        "session_id": "sess_test_c",
        "user_id": "4"
    })
    assert resp.status_code == 200
    msg = resp.json()["assistant_message"]
    assert "120" in msg
    assert "personal baseline" in msg.lower()


def test_test_d_heart_result_meaning():
    """TEST D: 'What does my Heart assessment mean?' retrieves real latest Heart assessment."""
    data = get_mock_patient_4_data()
    mock_backend = MagicMock()
    mock_backend.get_patient_heart_assessments.return_value = data["assessments"]
    mock_backend.get_patient_heart_assessment_by_id.return_value = data["assessments"][0]
    set_backend_client(mock_backend)

    resp = client.post("/api/chat", json={
        "message": "What does my Heart assessment mean?",
        "session_id": "sess_test_d",
        "user_id": "4",
        "context": {
            "source": "assessment",
            "disease": "HEART",
            "assessmentId": 10
        }
    })
    assert resp.status_code == 200
    msg = resp.json()["assistant_message"]
    assert "Assessment #10" in msg
    assert "11.3%" in msg or "11.27%" in msg or "0.1127" in msg
    assert "0.31" in msg
    assert "screening output" in msg.lower() or "not a definitive medical diagnosis" in msg.lower()


def test_test_e_shap_explanation():
    """TEST E: 'Why did the model give me this result?' retrieves real SHAP explanation."""
    data = get_mock_patient_4_data()
    mock_backend = MagicMock()
    mock_backend.get_patient_heart_assessments.return_value = data["assessments"]
    mock_backend.get_patient_heart_assessment_by_id.return_value = data["assessments"][0]
    set_backend_client(mock_backend)

    resp = client.post("/api/chat", json={
        "message": "Why did the model give me this result?",
        "session_id": "sess_test_e",
        "user_id": "4",
        "context": {
            "source": "assessment",
            "disease": "HEART",
            "assessmentId": 10
        }
    })
    assert resp.status_code == 200
    msg = resp.json()["assistant_message"]
    assert "Fluoroscopy Vessels (ca)" in msg or "ca" in msg
    assert "Chest Pain" in msg or "cp" in msg
    assert "Sex" in msg or "sex" in msg
    assert "SHAP" in msg or "Contributors" in msg


def test_test_f_safety_non_diagnostic():
    """TEST F: 'Does this mean I definitely have heart disease?' qualifies model output safely."""
    data = get_mock_patient_4_data()
    mock_backend = MagicMock()
    mock_backend.get_patient_heart_assessments.return_value = data["assessments"]
    mock_backend.get_patient_heart_assessment_by_id.return_value = data["assessments"][0]
    set_backend_client(mock_backend)

    resp = client.post("/api/chat", json={
        "message": "Does this mean I definitely have heart disease?",
        "session_id": "sess_test_f",
        "user_id": "4",
        "context": {
            "source": "assessment",
            "disease": "HEART",
            "assessmentId": 10
        }
    })
    assert resp.status_code == 200
    msg = resp.json()["assistant_message"]
    assert "not a diagnosis" in msg.lower() or "not a medical diagnosis" in msg.lower()
    assert "physician" in msg.lower() or "cardiologist" in msg.lower()


def test_test_g_previous_assessment_comparison():
    """TEST G: 'How does this compare with my previous assessment?' compares with historical records."""
    data = get_mock_patient_4_data()
    mock_backend = MagicMock()
    mock_backend.get_patient_heart_assessments.return_value = data["assessments"]
    mock_backend.get_patient_heart_assessment_by_id.return_value = data["assessments"][0]
    set_backend_client(mock_backend)

    resp = client.post("/api/chat", json={
        "message": "How does this compare with my previous assessment?",
        "session_id": "sess_test_g",
        "user_id": "4",
        "context": {
            "source": "assessment",
            "disease": "HEART",
            "assessmentId": 10
        }
    })
    assert resp.status_code == 200
    msg = resp.json()["assistant_message"]
    assert "#10" in msg
    assert "#9" in msg
    assert "not equate to clinical disease progression" in msg.lower() or "progression" in msg.lower()


def test_test_h_combined_cross_source_guidance():
    """TEST H: 'Considering my monitoring and Heart assessment, what should I discuss with a doctor?' aggregates sources."""
    data = get_mock_patient_4_data()
    mock_backend = MagicMock()
    mock_backend.get_patient_heart_assessments.return_value = data["assessments"]
    mock_backend.get_patient_heart_assessment_by_id.return_value = data["assessments"][0]
    mock_backend.get_patient_health_context.return_value = {
        "patient_id": 4,
        "has_history": True,
        "trends": data["trends"],
        "baselines": data["baselines"],
        "recent_observations": data["observations"]
    }
    set_backend_client(mock_backend)

    resp = client.post("/api/chat", json={
        "message": "Considering my monitoring and Heart assessment, what should I discuss with a doctor?",
        "session_id": "sess_test_h",
        "user_id": "4",
        "context": {
            "source": "assessment",
            "disease": "HEART",
            "assessmentId": 10
        }
    })
    assert resp.status_code == 200
    msg = resp.json()["assistant_message"]
    # Contains assessment data
    assert "Assessment #10" in msg
    assert "11.3%" in msg or "11.27%" in msg
    # Contains monitoring data
    assert "Glucose" in msg or "GLUCOSE" in msg
    assert "110" in msg
    # Contains physician discussion questions
    assert "Doctor" in msg or "doctor" in msg or "provider" in msg
    # Strictly non-causal
    assert "separate pieces of information" in msg.lower()


def test_test_i_context_continuity_multi_turn():
    """TEST I: Context continuity: opening Heart Assessment #10 maintains context across subsequent turns."""
    data = get_mock_patient_4_data()
    mock_backend = MagicMock()
    mock_backend.get_patient_heart_assessments.return_value = data["assessments"]
    mock_backend.get_patient_heart_assessment_by_id.return_value = data["assessments"][0]
    mock_backend.get_patient_health_context.return_value = {
        "patient_id": 4,
        "has_history": True,
        "trends": data["trends"],
        "baselines": data["baselines"],
        "recent_observations": data["observations"]
    }
    set_backend_client(mock_backend)

    sess_id = "sess_continuity_test_i"

    # Turn 1: Opens with assessment context
    resp1 = client.post("/api/chat", json={
        "message": "What does this result mean?",
        "session_id": sess_id,
        "user_id": "4",
        "context": {
            "source": "assessment",
            "disease": "HEART",
            "assessmentId": 10
        }
    })
    assert resp1.status_code == 200
    assert "Assessment #10" in resp1.json()["assistant_message"]
    assert "11.3%" in resp1.json()["assistant_message"]

    # Turn 2: Follow-up SHAP question WITHOUT repeating context
    resp2 = client.post("/api/chat", json={
        "message": "What influenced it?",
        "session_id": sess_id,
        "user_id": "4"
    })
    assert resp2.status_code == 200
    msg2 = resp2.json()["assistant_message"]
    assert "Assessment #10" in msg2
    assert ("ca" in msg2 or "Fluoroscopy" in msg2 or "Chest Pain" in msg2 or "SHAP" in msg2)

    # Turn 3: Doctor discussion question WITHOUT repeating context
    resp3 = client.post("/api/chat", json={
        "message": "What should I discuss with my doctor?",
        "session_id": sess_id,
        "user_id": "4"
    })
    assert resp3.status_code == 200
    msg3 = resp3.json()["assistant_message"]
    assert "Assessment #10" in msg3
    assert "Doctor" in msg3 or "doctor" in msg3


def test_monitoring_no_data_behavior():
    """Section 11: If a metric (e.g. HEART_RATE) has zero observations, honestly report no data."""
    mock_backend = MagicMock()
    mock_backend.get_patient_health_context.return_value = {
        "patient_id": 4,
        "has_history": False,
        "trends": [],
        "baselines": [],
        "recent_observations": []
    }
    set_backend_client(mock_backend)

    resp = client.post("/api/chat", json={
        "message": "How is my heart rate doing?",
        "session_id": "sess_no_hr",
        "user_id": "4"
    })
    assert resp.status_code == 200
    msg = resp.json()["assistant_message"]
    assert "no historical readings" in msg.lower() or "no historical records" in msg.lower()
    # Must NOT claim heart rate is normal
    assert "your heart rate is normal" not in msg.lower()


def test_patient_isolation_enforcement():
    """Section 36: Patient 1 cannot access Patient 4's Assessment #10."""
    data = get_mock_patient_4_data()
    mock_backend = MagicMock()
    # Patient 1 only has assessment #11, NOT #10
    mock_backend.get_patient_heart_assessments.side_effect = lambda pid: (
        [{"id": 11, "patientId": 1, "riskScore": 0.1034}] if pid == 1 else data["assessments"]
    )
    mock_backend.get_patient_heart_assessment_by_id.side_effect = lambda pid, aid: (
        None if pid == 1 and aid == 10 else data["assessments"][0]
    )
    set_backend_client(mock_backend)

    # Patient 1 tries to access Assessment #10
    resp = client.post("/api/chat", json={
        "message": "What does this result mean?",
        "session_id": "sess_patient_1_attempt",
        "user_id": "1",
        "context": {
            "source": "assessment",
            "disease": "HEART",
            "assessmentId": 10
        }
    })
    assert resp.status_code == 200
    msg = resp.json()["assistant_message"]
    # Assessment #10 was rejected for Patient 1; Patient 1 sees their own record (#11) or safe response, never #10
    assert "#10" not in msg
    assert "11.3%" not in msg
    assert "0.1127" not in msg
