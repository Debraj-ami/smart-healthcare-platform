import os
import sys
import json
import pytest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from app.main import app, session_manager, extraction_engine
from app.schemas import (
    IntentType,
    SafetyStatus,
    NextActionType,
    ChatResponse
)
from app.llm_client import MockLLMClient
from app.backend_client import (
    set_backend_client,
    BackendClient,
    BackendServiceUnavailableError,
    BackendClientError
)
from app.heart_assessment import (
    HeartAssessmentState,
    HeartExtractor,
    format_heart_prediction_response,
    format_heart_history_response,
    format_heart_explanation_response,
    HEART_REQUIRED_FIELDS
)

client = TestClient(app)


@pytest.fixture(autouse=True)
def clean_sessions_and_backend():
    """Reset session manager, LLM client, and backend client between tests."""
    session_manager._sessions.clear()
    extraction_engine.llm_client = MockLLMClient()
    set_backend_client(None)
    yield
    session_manager._sessions.clear()
    extraction_engine.llm_client = MockLLMClient()
    set_backend_client(None)


def get_mock_heart_prediction_response(prob=0.1911, prediction=0, threshold=0.31):
    """Deterministic mock response mirroring Spring Boot /api/assessments/heart/predict."""
    return {
        "assessmentId": 201,
        "patientId": 1,
        "disease": "HEART_DISEASE",
        "probability": prob,
        "prediction": prediction,
        "threshold": threshold,
        "thresholdDescription": "Operating threshold optimized on Cleveland Heart Disease validation set.",
        "modelName": "Logistic Regression (Robust L1)",
        "modelVersion": "1.0.0",
        "configId": "heart_logistic_regression_production",
        "outputSpace": "log_odds",
        "calibrationStatus": "CALIBRATED_ISOTONIC",
        "explanationAvailable": True,
        "explanationOutputSpace": "log_odds",
        "shapBaseValue": -0.42,
        "modelOutput": -1.44,
        "positiveContributors": [
            {
                "feature": "trestbps",
                "transformedFeature": "num__trestbps",
                "displayName": "Resting Blood Pressure",
                "rawValue": 145.0,
                "contribution": 0.35,
                "absContribution": 0.35,
                "direction": "INCREASES_RISK"
            },
            {
                "feature": "cp",
                "transformedFeature": "cat__cp_4.0",
                "displayName": "Chest Pain Type",
                "rawValue": 4,
                "contribution": 0.28,
                "absContribution": 0.28,
                "direction": "INCREASES_RISK"
            }
        ],
        "negativeContributors": [
            {
                "feature": "thalach",
                "transformedFeature": "num__thalach",
                "displayName": "Maximum Heart Rate",
                "rawValue": 160.0,
                "contribution": -0.42,
                "absContribution": 0.42,
                "direction": "DECREASES_RISK"
            }
        ],
        "allContributions": [
            {
                "feature": "thalach",
                "transformedFeature": "num__thalach",
                "displayName": "Maximum Heart Rate",
                "rawValue": 160.0,
                "contribution": -0.42,
                "absContribution": 0.42,
                "direction": "DECREASES_RISK"
            },
            {
                "feature": "trestbps",
                "transformedFeature": "num__trestbps",
                "displayName": "Resting Blood Pressure",
                "rawValue": 145.0,
                "contribution": 0.35,
                "absContribution": 0.35,
                "direction": "INCREASES_RISK"
            }
        ],
        "inputSummary": {
            "age": 55, "sex": 1, "cp": 4, "trestbps": 145, "chol": 230,
            "fbs": 0, "restecg": 0, "thalach": 160, "exang": 0,
            "oldpeak": 1.2, "slope": 1, "ca": 0, "thal": 3
        },
        "disclaimer": "This platform provides AI-assisted health screening and risk estimation."
    }


def get_mock_heart_history():
    """Mock list returned by Spring Boot GET /api/assessments/heart/patient/{id}."""
    return [
        {
            "id": 201,
            "patientId": 1,
            "diseaseType": "HEART_DISEASE",
            "riskScore": 0.1911,
            "riskCategory": "LOW",
            "assessedAt": "2026-10-01T10:00:00",
            "inputFeatures": json.dumps({
                "probability": 0.1911,
                "prediction": 0,
                "threshold": 0.31,
                "modelName": "Logistic Regression (Robust L1)",
                "modelVersion": "1.0.0",
                "shapBaseValue": -0.42,
                "modelOutput": -1.44,
                "explanationAvailable": True,
                "positiveContributors": [
                    {
                        "feature": "trestbps",
                        "transformedFeature": "num__trestbps",
                        "displayName": "Resting Blood Pressure",
                        "rawValue": 145.0,
                        "contribution": 0.35
                    },
                    {
                        "feature": "cp",
                        "transformedFeature": "cat__cp_4.0",
                        "displayName": "Chest Pain Type",
                        "rawValue": 4,
                        "contribution": 0.28
                    }
                ],
                "negativeContributors": [
                    {
                        "feature": "thalach",
                        "transformedFeature": "num__thalach",
                        "displayName": "Maximum Heart Rate",
                        "rawValue": 160.0,
                        "contribution": -0.42
                    }
                ]
            })
        },
        {
            "id": 195,
            "patientId": 1,
            "diseaseType": "HEART_DISEASE",
            "riskScore": 0.254,
            "riskCategory": "LOW",
            "assessedAt": "2026-09-15T14:30:00",
            "inputFeatures": json.dumps({
                "probability": 0.254,
                "prediction": 0,
                "threshold": 0.31
            })
        }
    ]


# ============================================================
# A. Intent & Routing Tests
# ============================================================
def test_heart_risk_assessment_intent():
    """Asking for heart disease risk assessment triggers heart slot filling."""
    resp = client.post("/api/chat", json={
        "message": "Can you assess my heart disease risk?",
        "session_id": "sess_heart_intent_1",
        "user_id": "1"
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "Heart Disease risk assessment" in data["assistant_message"]
    assert data["next_action"] == NextActionType.ASK_FOLLOW_UP.value
    assert data["assessment_handoff"] is not None
    assert "heart_disease" in data["assessment_handoff"]["disease_candidates"]


def test_heart_history_intent():
    """Asking for past heart assessments routes to history handler and calls backend."""
    mock_backend = MagicMock()
    mock_backend.get_patient_heart_assessments.return_value = get_mock_heart_history()
    set_backend_client(mock_backend)

    resp = client.post("/api/chat", json={
        "message": "Have I taken a heart assessment before?",
        "session_id": "sess_heart_hist_1",
        "user_id": "1"
    })
    assert resp.status_code == 200
    data = resp.json()
    assert mock_backend.get_patient_heart_assessments.called
    assert "2 Heart Disease screening assessments" in data["assistant_message"]
    assert "19.1%" in data["assistant_message"]
    assert "25.4%" in data["assistant_message"]


def test_heart_explanation_intent():
    """Asking what the heart assessment showed routes to explanation handler."""
    mock_backend = MagicMock()
    mock_backend.get_patient_heart_assessments.return_value = get_mock_heart_history()
    set_backend_client(mock_backend)

    resp = client.post("/api/chat", json={
        "message": "What did my last heart assessment show?",
        "session_id": "sess_heart_exp_1",
        "user_id": "1"
    })
    assert resp.status_code == 200
    data = resp.json()
    assert mock_backend.get_patient_heart_assessments.called
    assert "Heart Assessment #201" in data["assistant_message"]
    assert "19.1%" in data["assistant_message"]
    assert "Decision threshold: t = 0.31" in data["assistant_message"]


def test_heart_shap_explanation_intent():
    """Asking what factors influenced the score explains SHAP contributions."""
    mock_backend = MagicMock()
    mock_backend.get_patient_heart_assessments.return_value = get_mock_heart_history()
    set_backend_client(mock_backend)

    resp = client.post("/api/chat", json={
        "message": "What factors influenced my heart score?",
        "session_id": "sess_heart_shap_1",
        "user_id": "1"
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "Key Model Contributors (SHAP)" in data["assistant_message"]
    assert "Resting Blood Pressure" in data["assistant_message"]
    assert "Maximum Heart Rate" in data["assistant_message"]
    assert "model-output contributions" in data["assistant_message"].lower()


# ============================================================
# B. Slot Extraction & Validation Tests
# ============================================================
def test_extractor_valid_values():
    """All 13 fields extract and validate correctly."""
    extractor = HeartExtractor()
    msg = (
        "age 55, male, chest pain 4, resting bp 140, cholesterol 240, "
        "fbs 0, restecg 0, max heart rate 150, exercise angina no, "
        "oldpeak 1.5, slope 2, 0 vessels, thal normal"
    )
    extracted, nulls, errors = extractor.extract_and_validate(msg)
    assert not errors
    assert extracted["age"] == 55
    assert extracted["sex"] == 1
    assert extracted["cp"] == 4
    assert extracted["trestbps"] == 140.0
    assert extracted["chol"] == 240.0
    assert extracted["fbs"] == 0
    assert extracted["restecg"] == 0
    assert extracted["thalach"] == 150.0
    assert extracted["exang"] == 0
    assert extracted["oldpeak"] == 1.5
    assert extracted["slope"] == 2
    assert extracted["ca"] == 0
    assert extracted["thal"] == 3


def test_extractor_invalid_values_rejected():
    """Out-of-bounds or invalid categorical values are rejected with explicit errors."""
    extractor = HeartExtractor()

    _, _, err1 = extractor.extract_and_validate("age -5")
    assert any("Age must be an integer between 1 and 120" in e for e in err1)

    val2, err2 = extractor.validate_field("sex", 7)
    assert val2 is None and "Sex must be 0 (Female) or 1 (Male)" in err2

    val3, err3 = extractor.validate_field("cp", 9)
    assert val3 is None and "Chest pain type (cp) must be 1" in err3

    val4, err4 = extractor.validate_field("restecg", 8)
    assert val4 is None and "Resting ECG must be 0" in err4

    val5, err5 = extractor.validate_field("ca", 7)
    assert val5 is None and "Number of major vessels (ca) must be between 0 and 3" in err5

    val6, err6 = extractor.validate_field("thal", 4)
    assert val6 is None and "Thallium scintigraphy (thal) must be 3" in err6


def test_extractor_nullable_ca_and_thal():
    """Explicitly unknown or not provided ca and thal are recognized as null."""
    extractor = HeartExtractor()
    extracted, nulls, errors = extractor.extract_and_validate(
        "I don't know the number of vessels and thal is unknown"
    )
    assert not errors
    assert "ca" in nulls
    assert "thal" in nulls


def test_no_inferred_numbers_from_ambiguous_text():
    """Ambiguous descriptions like 'my BP is high' do not fabricate a number."""
    extractor = HeartExtractor()
    extracted, nulls, errors = extractor.extract_and_validate("My BP is high and cholesterol is bad")
    assert "trestbps" not in extracted
    assert "chol" not in extracted


# ============================================================
# C. Multi-Turn Conversation & Confirmation Before Prediction
# ============================================================
def test_multi_turn_flow_and_confirmation():
    """
    Test multi-turn flow:
    Turn 1: Request assessment -> asks for age & sex
    Turn 2: Provide age & sex -> asks for next missing
    Turn 3: Provide remaining clinical features -> asks for confirmation
    Turn 4: User corrects cholesterol -> re-asks confirmation with updated value
    Turn 5: User confirms ("Yes proceed") -> backend called and prediction returned
    """
    mock_backend = MagicMock()
    mock_backend.predict_heart.return_value = get_mock_heart_prediction_response()
    set_backend_client(mock_backend)

    sess_id = "sess_multi_turn_heart_1"

    # Turn 1: Initial request
    r1 = client.post("/api/chat", json={
        "message": "Can you assess my heart disease risk?",
        "session_id": sess_id,
        "user_id": "1"
    })
    d1 = r1.json()
    assert "Heart Disease risk assessment" in d1["assistant_message"]
    assert d1["next_action"] == NextActionType.ASK_FOLLOW_UP.value

    # Turn 2: Age and sex
    r2 = client.post("/api/chat", json={
        "message": "My age is 55 and I am male",
        "session_id": sess_id,
        "user_id": "1"
    })
    d2 = r2.json()
    assert d2["next_action"] == NextActionType.ASK_FOLLOW_UP.value
    assert "Chest Pain Type" in d2["assistant_message"]

    # Turn 3: Remaining 11 features
    remaining = (
        "Chest pain is asymptomatic (cp 4), resting BP 145, cholesterol 230, "
        "fasting blood sugar is normal (fbs 0), resting ecg normal (restecg 0), "
        "max heart rate 160, no exercise angina (exang 0), "
        "st depression 1.2, slope flat (slope 2), 0 vessels (ca 0), thal normal (thal 3)"
    )
    r3 = client.post("/api/chat", json={
        "message": remaining,
        "session_id": sess_id,
        "user_id": "1"
    })
    d3 = r3.json()
    # Must ask for confirmation, NOT immediately predict
    assert not mock_backend.predict_heart.called
    assert "Here is what I have" in d3["assistant_message"]
    assert "Would you like me to run the Heart Disease risk assessment?" in d3["assistant_message"]
    assert "Serum Cholesterol (mg/dL): 230" in d3["assistant_message"]

    # Turn 4: User corrects cholesterol
    r4 = client.post("/api/chat", json={
        "message": "Actually my cholesterol is 210",
        "session_id": sess_id,
        "user_id": "1"
    })
    d4 = r4.json()
    assert not mock_backend.predict_heart.called
    assert "Serum Cholesterol (mg/dL): 210" in d4["assistant_message"]
    assert "Would you like me to run the Heart Disease risk assessment?" in d4["assistant_message"]

    # Turn 5: User confirms!
    r5 = client.post("/api/chat", json={
        "message": "Yes, please run it",
        "session_id": sess_id,
        "user_id": "1"
    })
    d5 = r5.json()
    assert mock_backend.predict_heart.called
    call_args = mock_backend.predict_heart.call_args[0][0]
    assert call_args["age"] == 55
    assert call_args["sex"] == 1
    assert call_args["chol"] == 210.0
    assert "num" not in call_args  # Target variable must NEVER be sent
    assert "19.1%" in d5["assistant_message"]
    assert d5["assessment_handoff"]["assessment_result"]["probability"] == 0.1911


# ============================================================
# D. Backend Error Handling & Safety
# ============================================================
def test_backend_service_unavailable():
    """When Spring Boot fails or is unavailable, safe error message is returned with preserved inputs."""
    mock_backend = MagicMock()
    mock_backend.predict_heart.side_effect = BackendServiceUnavailableError("Service down")
    set_backend_client(mock_backend)

    sess_id = "sess_backend_err_1"
    all_inputs = (
        "age 55, male, cp 4, resting bp 145, cholesterol 210, "
        "fbs 0, restecg 0, max heart rate 160, exang 0, "
        "oldpeak 1.2, slope 2, ca 0, thal 3"
    )
    # Collect inputs
    client.post("/api/chat", json={"message": all_inputs, "session_id": sess_id, "user_id": "1"})
    # Confirm
    r = client.post("/api/chat", json={"message": "Yes, run it", "session_id": sess_id, "user_id": "1"})
    assert r.status_code == 200
    data = r.json()
    assert "service is currently unavailable" in data["assistant_message"]
    assert "No result was generated" in data["assistant_message"]


# ============================================================
# E. Probability & SHAP Grounding
# ============================================================
def test_probability_grounding():
    """Response strictly formats the backend probability and operating threshold."""
    res = get_mock_heart_prediction_response(prob=0.1911, prediction=0, threshold=0.31)
    text = format_heart_prediction_response(res)
    assert "Model-estimated probability: 19.1% (0.1911)" in text
    assert "Decision threshold: 31% (t = 0.31)" in text
    assert "The probability is the model's estimated output for this assessment" in text
    assert "not an automated medical diagnostic tool" in text


def test_shap_categorical_feature_description():
    """SHAP explanation distinguishes encoded one-hot categories from continuous values."""
    res = get_mock_heart_prediction_response()
    text = format_heart_prediction_response(res)
    assert "Chest Pain Type [encoded category: cp 4.0]: +0.2800" in text
    assert "These values show how individual model features contributed to this model's output" in text
    assert "not biological causation" in text


# ============================================================
# F. Patient Context & Isolation
# ============================================================
def test_unauthenticated_patient_history_rejected():
    """Asking for history without authenticated patient returns guidance message."""
    resp = client.post("/api/chat", json={
        "message": "What did my last heart assessment show?",
        "session_id": "sess_anon_1",
        "user_id": "anonymous"
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "require an authenticated patient context" in data["assistant_message"]


def test_patient_isolation_calls_correct_id():
    """Patient 4 queries history with patientId=4, Patient 9 queries with patientId=9."""
    mock_backend = MagicMock()
    mock_backend.get_patient_heart_assessments.return_value = []
    set_backend_client(mock_backend)

    client.post("/api/chat", json={
        "message": "Show me my previous heart assessment",
        "session_id": "sess_p4",
        "user_id": "4"
    })
    mock_backend.get_patient_heart_assessments.assert_called_with(4)

    client.post("/api/chat", json={
        "message": "Show me my previous heart assessment",
        "session_id": "sess_p9",
        "user_id": "9"
    })
    mock_backend.get_patient_heart_assessments.assert_called_with(9)


# ============================================================
# G. Emergency Safety Routing Wins Over Heart Assessment
# ============================================================
def test_emergency_chest_pain_overrides_heart_assessment():
    """Severe crushing chest pain must trigger emergency routing immediately, not Heart ML."""
    resp = client.post("/api/chat", json={
        "message": "I have severe crushing chest pain. Can you check my heart risk?",
        "session_id": "sess_emerg_1",
        "user_id": "1"
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["safety_status"] == SafetyStatus.URGENT_MEDICAL_EVALUATION.value
    assert data["next_action"] == NextActionType.EMERGENCY_ROUTING.value
    assert "emergency" in data["assistant_message"].lower()
    # Assessment must NOT have started
    sess = session_manager.get("sess_emerg_1")
    assert sess is None or sess.heart_assessment is None


def test_emergency_negation_handling():
    """'I have no chest pain' does not trigger emergency routing."""
    resp = client.post("/api/chat", json={
        "message": "I have no chest pain, but can you assess my heart disease risk?",
        "session_id": "sess_neg_1",
        "user_id": "1"
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["safety_status"] != SafetyStatus.URGENT_MEDICAL_EVALUATION.value
    assert data["next_action"] != NextActionType.EMERGENCY_ROUTING.value
    assert "Heart Disease risk assessment" in data["assistant_message"]


# ============================================================
# H. General Education vs Patient Assessment
# ============================================================
def test_general_heart_education_question():
    """'What is heart disease?' provides educational info without launching assessment."""
    resp = client.post("/api/chat", json={
        "message": "What is heart disease?",
        "session_id": "sess_edu_1",
        "user_id": "1"
    })
    assert resp.status_code == 200
    data = resp.json()
    sess = session_manager.get("sess_edu_1")
    assert sess is None or sess.heart_assessment is None
    assert "Heart Disease risk assessment" not in data["assistant_message"]


# ============================================================
# I. Patient-Centric Health Entry & Missing Clinical Data UX
# ============================================================
def test_missing_cholesterol_disclaimed_explains_alternatives():
    """When a patient states they don't have cholesterol, the system does not invent a value and offers alternatives."""
    # First, user initiates assessment inquiry
    resp = client.post("/api/chat", json={
        "message": "I want to check my heart risk",
        "session_id": "sess_miss_1",
        "user_id": "1"
    })
    assert resp.status_code == 200

    # User explicitly disclaims having cholesterol
    resp2 = client.post("/api/chat", json={
        "message": "I don't have my cholesterol",
        "session_id": "sess_miss_1",
        "user_id": "1"
    })
    assert resp2.status_code == 200
    data2 = resp2.json()
    msg = data2["assistant_message"]
    # Must explain that the screening model requires cholesterol and won't estimate or invent it
    assert "screening model requires" in msg.lower()
    assert "won't estimate" in msg.lower() and "invent" in msg.lower()
    # Must offer alternative pathways: symptom conversation or health monitoring
    assert "symptom" in msg.lower()
    assert "monitor" in msg.lower()
    # ML model must NOT have been called
    sess = session_manager.get("sess_miss_1")
    assert sess is not None
    assert sess.heart_assessment.completed_assessment is None
    assert sess.heart_assessment.status != "CONFIRMED"


def test_missing_cholesterol_and_ecg_disclaimed():
    """User disclaims multiple lab/clinical values ('I don't have cholesterol or ECG')."""
    resp = client.post("/api/chat", json={
        "message": "Can you check my heart risk? I don't have my cholesterol or ECG.",
        "session_id": "sess_miss_2",
        "user_id": "1"
    })
    assert resp.status_code == 200
    data = resp.json()
    msg = data["assistant_message"]
    assert "screening model requires" in msg.lower()
    assert "cholesterol" in msg.lower() or "ecg" in msg.lower()
    assert "symptom" in msg.lower()
    assert "monitor" in msg.lower()


def test_ambiguous_bp_measurement_not_invented_as_numeric():
    """'My BP is high' must prompt for measured resting blood pressure without guessing a number."""
    # Start assessment flow
    resp = client.post("/api/chat", json={
        "message": "I want to assess my heart risk",
        "session_id": "sess_bp_1",
        "user_id": "1"
    })
    assert resp.status_code == 200

    # Provide vague BP
    resp2 = client.post("/api/chat", json={
        "message": "My BP is high",
        "session_id": "sess_bp_1",
        "user_id": "1"
    })
    assert resp2.status_code == 200
    data2 = resp2.json()
    msg = data2["assistant_message"]
    assert "resting blood pressure" in msg.lower()
    # Does NOT invent a numeric value
    sess = session_manager.get("sess_bp_1")
    assert sess is not None
    assert sess.heart_assessment.values.get("trestbps") is None


def test_pure_symptom_shortness_of_breath_does_not_force_clinical_assessment():
    """'I have shortness of breath' triggers symptom follow-up/safety, NOT asking for cholesterol/ECG."""
    resp = client.post("/api/chat", json={
        "message": "I have shortness of breath",
        "session_id": "sess_symp_1",
        "user_id": "1"
    })
    assert resp.status_code == 200
    data = resp.json()
    msg = data["assistant_message"]
    # Must NOT demand clinical inputs like cholesterol, ECG, thal, fluoroscopy
    assert "cholesterol" not in msg.lower()
    assert "fluoroscopy" not in msg.lower()
    assert "st depression" not in msg.lower()
    sess = session_manager.get("sess_symp_1")
    assert sess is None or sess.heart_assessment is None


def test_symptom_context_preservation_across_turns():
    """Shortness of breath followed by additional symptom preserves symptom context."""
    resp1 = client.post("/api/chat", json={
        "message": "I have shortness of breath",
        "session_id": "sess_symp_ctx",
        "user_id": "1"
    })
    assert resp1.status_code == 200

    resp2 = client.post("/api/chat", json={
        "message": "I also have pain on the left side",
        "session_id": "sess_symp_ctx",
        "user_id": "1"
    })
    assert resp2.status_code == 200
    sess = session_manager.get("sess_symp_ctx")
    assert sess is not None
    # Both symptoms are tracked or context is preserved without launching 13-feature Heart ML
    assert sess.heart_assessment is None

