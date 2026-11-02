import pytest
import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from app.main import app, session_manager, extraction_engine
from app.schemas import (
    IntentType,
    SafetyStatus,
    NextActionType
)
from app.llm_client import MockLLMClient
from app.backend_client import (
    set_backend_client,
    BackendClient,
    BackendServiceUnavailableError
)
from app.liver_assessment import LiverAssessmentState

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


def get_mock_ml_response():
    """Deterministic mock response mirroring Spring Boot / ML FastAPI LiverPredictResponse."""
    return {
        "assessmentId": 101,
        "patientId": 4,
        "assessedAt": "2026-09-25T12:00:00",
        "disease": "LIVER_DISEASE",
        "probability": 0.616,
        "modelName": "Logistic Regression",
        "modelVersion": "1.0.0",
        "calibrationMethod": "Isotonic 5-Fold Calibration",
        "shapBaseValue": -0.85,
        "shapOutputSpace": "log-odds",
        "shapFeatures": [
            {
                "feature": "Total Bilirubin",
                "transformedFeature": "total_bilirubin",
                "value": 10.9,
                "contribution": 0.52,
                "direction": "INCREASES_RISK"
            },
            {
                "feature": "Direct Bilirubin",
                "transformedFeature": "direct_bilirubin",
                "value": 5.5,
                "contribution": 0.38,
                "direction": "INCREASES_RISK"
            },
            {
                "feature": "Alkaline Phosphatase",
                "transformedFeature": "alkaline_phosphatase",
                "value": 699.0,
                "contribution": 0.25,
                "direction": "INCREASES_RISK"
            },
            {
                "feature": "Albumin",
                "transformedFeature": "albumin",
                "value": 3.0,
                "contribution": -0.15,
                "direction": "DECREASES_RISK"
            }
        ],
        "inputSummary": {
            "age": 62,
            "gender": "Male",
            "totalBilirubin": 10.9,
            "directBilirubin": 5.5,
            "alkalinePhosphatase": 699.0,
            "sgpt": 64.0,
            "sgot": 100.0,
            "totalProteins": 6.8,
            "albumin": 3.0,
            "agRatio": 0.8
        },
        "disclaimer": "This is a research and screening tool."
    }


# ============================================================
# A. User supplies age / gender
# ============================================================
def test_user_supplies_age_and_gender():
    """
    Test A: Verifies that conversational age and gender are correctly extracted,
    stored in the liver assessment state, and targeted follow-up asks for remaining fields.
    """
    session_id = "test-session-a"
    payload = {
        "user_id": "4",
        "session_id": session_id,
        "message": "I am 62 years old and male. I want to check my liver risk."
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()

    session = session_manager.get_or_create(session_id, "4")
    assert session.liver_assessment is not None
    assert session.liver_assessment.values["age"] == 62
    assert session.liver_assessment.values["gender"] == "Male"

    # Must ask for missing laboratory values
    assert data["next_action"] == NextActionType.ASK_FOLLOW_UP
    assert "Total Bilirubin" in data["assistant_message"]
    assert "Direct Bilirubin" in data["assistant_message"]
    assert "Alkaline Phosphatase" in data["assistant_message"]
    assert "Age" not in data["follow_up_question"]


# ============================================================
# B. User supplies multiple lab values in one message
# ============================================================
def test_user_supplies_multiple_lab_values():
    """
    Test B: Verifies that multiple laboratory parameters in a single message
    are all correctly parsed and saved into the assessment state simultaneously.
    """
    session_id = "test-session-b"
    payload = {
        "user_id": "4",
        "session_id": session_id,
        "message": "Age 62, male, bilirubin 10.9, direct bilirubin 5.5, SGPT 64, SGOT 100."
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200

    session = session_manager.get_or_create(session_id, "4")
    assert session.liver_assessment is not None
    values = session.liver_assessment.values

    assert values["age"] == 62
    assert values["gender"] == "Male"
    assert values["total_bilirubin"] == 10.9
    assert values["direct_bilirubin"] == 5.5
    assert values["sgpt"] == 64.0
    assert values["sgot"] == 100.0

    # Remaining missing fields must be exactly 4: alkphos, total_proteins, albumin, a_g_ratio
    missing = session.liver_assessment.get_missing_fields()
    assert set(missing) == {"alkaline_phosphatase", "total_proteins", "albumin", "a_g_ratio"}


# ============================================================
# C. Values persist across turns
# ============================================================
def test_values_persist_across_turns():
    """
    Test C: Verifies multi-turn state preservation. Values supplied in Turn 1
    are NOT lost when additional values are supplied in Turn 2 and Turn 3.
    """
    session_id = "test-session-c"

    # Turn 1: Demographic fields
    client.post("/api/chat", json={
        "user_id": "4",
        "session_id": session_id,
        "message": "I am 62 years old and male. Please assess my liver risk."
    })

    # Turn 2: Bilirubin fields
    client.post("/api/chat", json={
        "user_id": "4",
        "session_id": session_id,
        "message": "My total bilirubin is 10.9 and direct bilirubin is 5.5."
    })

    # Turn 3: Enzymes
    client.post("/api/chat", json={
        "user_id": "4",
        "session_id": session_id,
        "message": "Alkaline phosphatase is 699, SGPT is 64, SGOT is 100."
    })

    session = session_manager.get_or_create(session_id, "4")
    values = session.liver_assessment.values

    assert values["age"] == 62
    assert values["gender"] == "Male"
    assert values["total_bilirubin"] == 10.9
    assert values["direct_bilirubin"] == 5.5
    assert values["alkaline_phosphatase"] == 699.0
    assert values["sgpt"] == 64.0
    assert values["sgot"] == 100.0


# ============================================================
# D. Missing values trigger targeted follow-up
# ============================================================
def test_missing_values_targeted_follow_up():
    """
    Test D: Confirms the chatbot asks specifically for unsupplied parameters,
    without re-requesting already supplied parameters.
    """
    session_id = "test-session-d"

    resp = client.post("/api/chat", json={
        "user_id": "4",
        "session_id": session_id,
        "message": "Total proteins 6.8, albumin 3.0, A/G ratio 0.8."
    })
    data = resp.json()
    assert data["next_action"] == NextActionType.ASK_FOLLOW_UP

    msg = data["assistant_message"]
    # Recorded values are acknowledged
    assert "Total Proteins" in msg
    assert "Albumin" in msg
    assert "A/G Ratio" in msg

    # Follow-up asks for remaining items
    assert "Total Bilirubin" in msg
    assert "Age" in msg
    assert "SGPT" in msg


# ============================================================
# E. Invalid numeric input is rejected
# ============================================================
def test_invalid_numeric_input_rejected():
    """
    Test E: Verifies out-of-bounds or malformed values are rejected with
    explicit validation errors, and are not silently coerced or stored.
    """
    session_id = "test-session-e"

    resp = client.post("/api/chat", json={
        "user_id": "4",
        "session_id": session_id,
        "message": "I want to test liver. My age is 250 and my bilirubin is -5.0."
    })
    data = resp.json()
    assert data["next_action"] == NextActionType.ASK_FOLLOW_UP

    session = session_manager.get_or_create(session_id, "4")
    # Invalid values must not be accepted into state
    assert session.liver_assessment.values.get("age") is None
    assert session.liver_assessment.values.get("total_bilirubin") is None

    # Explicit validation feedback provided in response
    msg = data["assistant_message"]
    assert "Age must be an integer between 1 and 120" in msg or "between 1 and 120" in msg
    assert "Total Bilirubin must be greater than 0" in msg or "greater than 0" in msg


# ============================================================
# F, G, H, I. Complete 10-field input -> ML prediction call & SHAP
# ============================================================
def test_complete_ten_field_prediction_success():
    """
    Tests F, G, H, I:
    - F: 10 complete inputs trigger prediction call to Spring Boot
    - G: Successful ML response returned to chatbot
    - H: Real probability (61.6%) passed through unchanged
    - I: SHAP feature contributions explained without causal claims
    """
    session_id = "test-session-fghi"

    mock_backend = MagicMock(spec=BackendClient)
    mock_backend.predict_liver.return_value = get_mock_ml_response()
    set_backend_client(mock_backend)

    # Provide all 10 fields across two natural turns
    client.post("/api/chat", json={
        "user_id": "4",
        "session_id": session_id,
        "message": "I am 62 years old and male. Bilirubin 10.9, direct bilirubin 5.5, SGPT 64, SGOT 100."
    })

    resp2 = client.post("/api/chat", json={
        "user_id": "4",
        "session_id": session_id,
        "message": "Alkaline phosphatase is 699, total proteins 6.8, albumin 3.0, A/G ratio 0.8."
    })

    assert resp2.status_code == 200
    data = resp2.json()

    # F: Verify BackendClient was called with the exact 10 fields + patientId
    assert mock_backend.predict_liver.call_count == 1
    call_args = mock_backend.predict_liver.call_args[0][0]
    assert call_args["patientId"] == 4
    assert call_args["age"] == 62
    assert call_args["gender"] == "Male"
    assert call_args["totalBilirubin"] == 10.9
    assert call_args["directBilirubin"] == 5.5
    assert call_args["alkalinePhosphatase"] == 699.0
    assert call_args["sgpt"] == 64.0
    assert call_args["sgot"] == 100.0
    assert call_args["totalProteins"] == 6.8
    assert call_args["albumin"] == 3.0
    assert call_args["agRatio"] == 0.8

    # G: Assessment result returned in structured payload
    handoff = data["assessment_handoff"]
    assert handoff is not None
    assert handoff["assessment_result"] is not None
    assert handoff["assessment_result"]["assessmentId"] == 101

    # H: Real probability passed through unchanged (61.6%)
    assert handoff["assessment_result"]["probability"] == 0.616
    assert "61.6%" in data["assistant_message"]

    # I: SHAP explanation present and avoids causal language
    msg = data["assistant_message"]
    assert "Total Bilirubin" in msg
    assert "Direct Bilirubin" in msg
    assert "higher screening risk" in msg or "SHAP Analysis" in msg
    # Must NOT make definitive diagnosis or causal claims
    assert "caused your disease" not in msg
    assert "proves you have" not in msg
    assert "NOT a definitive medical diagnosis" in msg
    assert "healthcare professional" in msg.lower() or "physician" in msg.lower()


# ============================================================
# J. ML / Spring Boot failure produces no fabricated prediction
# ============================================================
def test_backend_failure_produces_no_fabricated_prediction():
    """
    Test J: When Spring Boot or ML service is unavailable:
    - Application returns safe HTTP 200 (does not crash)
    - Informs user of unavailability
    - NEVER invents or fabricates a probability or risk score
    """
    session_id = "test-session-j"

    mock_backend = MagicMock(spec=BackendClient)
    mock_backend.predict_liver.side_effect = BackendServiceUnavailableError("Connection refused on port 8080")
    set_backend_client(mock_backend)

    # Provide all 10 required fields in one turn
    resp = client.post("/api/chat", json={
        "user_id": "4",
        "session_id": session_id,
        "message": (
            "Age 62, male, bilirubin 10.9, direct bilirubin 5.5, alkaline phosphatase 699, "
            "SGPT 64, SGOT 100, total proteins 6.8, albumin 3.0, A/G ratio 0.8."
        )
    })

    assert resp.status_code == 200
    data = resp.json()
    msg = data["assistant_message"]

    # Controlled error handling
    assert "unavailable" in msg.lower()
    # No fabricated probability
    assert "61.6%" not in msg
    assert "probability:" not in msg.lower()
    assert "risk score:" not in msg.lower()
    assert data["assessment_handoff"]["assessment_result"] is None


# ============================================================
# K. Emergency routing takes precedence over liver assessment
# ============================================================
def test_emergency_routing_takes_precedence_over_liver_flow():
    """
    Test K: Emergency red-flag symptoms must immediately route to 911 / emergency,
    even if the session is currently in the middle of a liver disease assessment.
    """
    session_id = "test-session-k"

    mock_backend = MagicMock(spec=BackendClient)
    set_backend_client(mock_backend)

    # Turn 1: Start liver assessment
    client.post("/api/chat", json={
        "user_id": "4",
        "session_id": session_id,
        "message": "I am 62 years old and male. Bilirubin is 10.9."
    })

    # Turn 2: Emergency message
    resp_urgent = client.post("/api/chat", json={
        "user_id": "4",
        "session_id": session_id,
        "message": "I have sudden severe crushing chest pain radiating to my left arm and severe shortness of breath."
    })

    assert resp_urgent.status_code == 200
    data = resp_urgent.json()

    assert data["safety_status"] == SafetyStatus.URGENT_MEDICAL_EVALUATION
    assert data["next_action"] == NextActionType.EMERGENCY_ROUTING
    assert "EMERGENCY" in data["assistant_message"]

    # Model prediction was NEVER called
    assert mock_backend.predict_liver.call_count == 0


# ============================================================
# L. Session completion and follow-up without rerunning ML
# ============================================================
def test_session_completion_and_educational_followup():
    """
    Test L: After prediction completes, user can ask follow-up questions without
    automatically re-executing the ML model unless clinical inputs change.
    """
    session_id = "test-session-l"

    mock_backend = MagicMock(spec=BackendClient)
    mock_backend.predict_liver.return_value = get_mock_ml_response()
    set_backend_client(mock_backend)

    # 1. Complete prediction
    client.post("/api/chat", json={
        "user_id": "4",
        "session_id": session_id,
        "message": (
            "Age 62, male, bilirubin 10.9, direct bilirubin 5.5, alkaline phosphatase 699, "
            "SGPT 64, SGOT 100, total proteins 6.8, albumin 3.0, A/G ratio 0.8."
        )
    })
    assert mock_backend.predict_liver.call_count == 1

    # 2. User asks educational question
    resp_followup = client.post("/api/chat", json={
        "user_id": "4",
        "session_id": session_id,
        "message": "What is blood pressure?"
    })

    assert resp_followup.status_code == 200
    data_followup = resp_followup.json()
    assert data_followup["intent"] == IntentType.GENERAL_HEALTH_QUESTION
    assert "blood pressure" in data_followup["assistant_message"].lower()

    # Prediction service was NOT re-executed
    assert mock_backend.predict_liver.call_count == 1


# ============================================================
# End-to-end multi-turn: Symptom collection -> Liver prediction
# ============================================================
def test_end_to_end_symptom_to_liver_prediction_flow():
    """
    Verifies full lifecycle:
    1. User reports right upper abdominal pain -> symptom follow-up.
    2. User provides severity -> symptom handoff contract prepared with 'liver_disease'.
    3. User provides 10 clinical & lab values -> backend ML called, calibrated probability & SHAP returned.
    """
    session_id = "test-session-e2e"

    mock_backend = MagicMock(spec=BackendClient)
    mock_backend.predict_liver.return_value = get_mock_ml_response()
    set_backend_client(mock_backend)

    # Turn 1: Symptom, location, duration
    resp1 = client.post("/api/chat", json={
        "user_id": "4",
        "session_id": session_id,
        "message": "I have had pain below my right ribs for two days."
    })
    assert resp1.status_code == 200
    assert resp1.json()["next_action"] == NextActionType.ASK_FOLLOW_UP

    # Turn 2: Severity and frequency
    resp2 = client.post("/api/chat", json={
        "user_id": "4",
        "session_id": session_id,
        "message": "It is moderate pain, and it comes and goes."
    })
    assert resp2.status_code == 200
    assert resp2.json()["next_action"] == NextActionType.ASK_FOLLOW_UP

    # Turn 3: Contextual questions answered -> handoff with candidate liver_disease
    resp3 = client.post("/api/chat", json={
        "user_id": "4",
        "session_id": session_id,
        "message": "No fever or yellowing of eyes."
    })
    assert resp3.status_code == 200
    data3 = resp3.json()
    assert data3["next_action"] == NextActionType.REQUEST_DISEASE_ASSESSMENT
    assert "liver_disease" in data3["assessment_handoff"]["disease_candidates"]

    # Turn 4: User conversationally provides the 10 required liver clinical values
    resp4 = client.post("/api/chat", json={
        "user_id": "4",
        "session_id": session_id,
        "message": (
            "I am 62 years old and male. My bilirubin is 10.9, direct bilirubin 5.5, "
            "alkaline phosphatase 699, SGPT 64, SGOT 100, total proteins 6.8, albumin 3.0, A/G ratio 0.8."
        )
    })
    assert resp4.status_code == 200
    data4 = resp4.json()

    # Backend was invoked with the 10 fields and patientId
    assert mock_backend.predict_liver.call_count == 1
    call_payload = mock_backend.predict_liver.call_args[0][0]
    assert call_payload["patientId"] == 4
    assert call_payload["age"] == 62
    assert call_payload["gender"] == "Male"
    assert call_payload["totalBilirubin"] == 10.9
    assert call_payload["directBilirubin"] == 5.5
    assert call_payload["alkalinePhosphatase"] == 699.0
    assert call_payload["sgpt"] == 64.0
    assert call_payload["sgot"] == 100.0
    assert call_payload["totalProteins"] == 6.8
    assert call_payload["albumin"] == 3.0
    assert call_payload["agRatio"] == 0.8

    # Safe response received with genuine probability and SHAP explanation
    assert "61.6%" in data4["assistant_message"]
    assert "Assessment ID: 101" in data4["assistant_message"]
    assert "Total Bilirubin" in data4["assistant_message"]
    assert "NOT a definitive medical diagnosis" in data4["assistant_message"]
    assert data4["assessment_handoff"]["assessment_result"]["probability"] == 0.616


# ============================================================
# 11. Measurement Object Validation & Fallback Tests (Req 8A-8E)
# ============================================================
def test_measurement_objects_validate_successfully():
    """Requirement 8A: LLM measurement objects validate successfully."""
    from app.schemas import StructuredHealthInformation, Measurement

    raw_info = {
        "measurements": [
            {"name": "total_bilirubin", "value": 10.9, "unit": "mg/dL"},
            {"name": "direct_bilirubin", "value": 5.5, "unit": "mg/dL"},
            {"name": "sgpt", "value": 64.0, "unit": "U/L"}
        ]
    }
    validated = StructuredHealthInformation.model_validate(raw_info)
    assert len(validated.measurements) == 3
    assert all(isinstance(m, Measurement) for m in validated.measurements)
    assert validated.measurements[0].name == "total_bilirubin"
    assert validated.measurements[0].value == 10.9
    assert validated.measurements[0].unit == "mg/dL"


def test_measurement_strings_trigger_fallback_without_crashing():
    """Requirement 8B: LLM measurement strings trigger fallback to deterministic extraction without crashing."""
    from app.extraction import ExtractionEngine
    from app.llm_client import MockLLMClient, LLMResponseError
    import json

    # Canned LLM response containing invalid string measurements instead of dict objects
    invalid_llm_json = json.dumps({
        "assistant_message": "Checking your liver labs.",
        "intent": "DISEASE_RISK_ASSESSMENT",
        "extracted_info": {
            "symptoms": [],
            "location": None,
            "duration": None,
            "severity": None,
            "onset": None,
            "frequency": None,
            "associated_symptoms": [],
            "medical_history": [],
            "medications": [],
            "known_conditions": [],
            "age": None,
            "sex": None,
            "measurements": [
                "Total Bilirubin (10.9 mg/dL)",
                "Direct Bilirubin (5.5 mg/dL)"
            ]
        }
    })

    client_with_bad_output = MockLLMClient(canned_response=invalid_llm_json)
    engine = ExtractionEngine(llm_client=client_with_bad_output)

    # Calling extract_with_llm raises LLMResponseError due to Pydantic schema validation
    with pytest.raises(LLMResponseError) as exc_info:
        engine.extract_with_llm("Total Bilirubin is 10.9 mg/dL")
    assert "failed schema validation" in str(exc_info.value)

    # Chat endpoint catches LLMResponseError and safely falls back without crashing
    session_id = "test-session-fallback-strings"
    resp = client.post("/api/chat", json={
        "user_id": "4",
        "session_id": session_id,
        "message": "Total Bilirubin is 10.9 mg/dL and Direct Bilirubin is 5.5 mg/dL."
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["next_action"] == NextActionType.ASK_FOLLOW_UP
    # Values extracted via deterministic fallback
    collected = data["assessment_handoff"]["structured_inputs"]
    assert collected["total_bilirubin"] == 10.9
    assert collected["direct_bilirubin"] == 5.5


def test_fallback_extracts_single_lab_value():
    """Requirement 8C: Fallback extracts 'Total Bilirubin is 10.9 mg/dL'."""
    from app.extraction import ExtractionEngine

    engine = ExtractionEngine()
    info = engine.extract_health_information("Total Bilirubin is 10.9 mg/dL")

    assert len(info.measurements) == 1
    m = info.measurements[0]
    assert m.name == "total_bilirubin"
    assert m.value == 10.9
    assert m.unit == "mg/dL"


def test_fallback_extracts_multiple_lab_values_in_one_message():
    """Requirement 8D: Fallback extracts multiple lab values in one message."""
    from app.extraction import ExtractionEngine

    engine = ExtractionEngine()
    msg = "My bilirubin is 10.9, direct bilirubin 5.5, alkaline phosphatase 699, SGPT 64, and SGOT 100."
    info = engine.extract_health_information(msg)

    meas_map = {m.name: (m.value, m.unit) for m in info.measurements}
    assert meas_map["total_bilirubin"] == (10.9, "mg/dL")
    assert meas_map["direct_bilirubin"] == (5.5, "mg/dL")
    assert meas_map["alkaline_phosphatase"] == (699.0, "IU/L")
    assert meas_map["sgpt"] == (64.0, "U/L")
    assert meas_map["sgot"] == (100.0, "U/L")


def test_multiturn_liver_state_remains_intact_across_fallback():
    """Requirement 8E: Existing multi-turn Liver state remains intact when a turn uses fallback."""
    session_id = "test-session-intact-state"

    # Turn 1: User supplies age and gender
    resp1 = client.post("/api/chat", json={
        "user_id": "4",
        "session_id": session_id,
        "message": "I am 62 years old and male. I want to check my liver health."
    })
    assert resp1.status_code == 200
    data1 = resp1.json()
    collected1 = data1["assessment_handoff"]["structured_inputs"]
    assert collected1["age"] == 62
    assert collected1["gender"] == "Male"

    # Turn 2: User supplies lab values
    resp2 = client.post("/api/chat", json={
        "user_id": "4",
        "session_id": session_id,
        "message": "My total bilirubin is 10.9 and direct bilirubin is 5.5 mg/dL."
    })
    assert resp2.status_code == 200
    data2 = resp2.json()
    collected2 = data2["assessment_handoff"]["structured_inputs"]

    # Both previously collected (age, gender) and newly collected fields are present
    assert collected2["age"] == 62
    assert collected2["gender"] == "Male"
    assert collected2["total_bilirubin"] == 10.9
    assert collected2["direct_bilirubin"] == 5.5
    # Missing fields correctly calculated
    missing = data2["assessment_handoff"]["assessment_state"]["missing_fields"]
    assert "alkaline_phosphatase" in missing
    assert "sgpt" in missing
    assert "total_bilirubin" not in missing


def test_llm_valid_measurement_objects_consumed_by_liver_assessment(monkeypatch):
    """Requirement 5: Normalized measurement values from successful LLM extraction are consumed by Liver assessment."""
    from app.llm_client import MockLLMClient
    from app.main import extraction_engine
    import json

    valid_llm_json = json.dumps({
        "assistant_message": "I noted your liver laboratory measurements.",
        "intent": "DISEASE_RISK_ASSESSMENT",
        "extracted_info": {
            "symptoms": [],
            "location": None,
            "duration": None,
            "severity": None,
            "onset": None,
            "frequency": None,
            "associated_symptoms": [],
            "medical_history": [],
            "medications": [],
            "known_conditions": [],
            "age": 62,
            "sex": "Male",
            "measurements": [
                {"name": "total_bilirubin", "value": 10.9, "unit": "mg/dL"},
                {"name": "direct_bilirubin", "value": 5.5, "unit": "mg/dL"},
                {"name": "alkaline_phosphatase", "value": 699.0, "unit": "IU/L"}
            ]
        }
    })

    client_with_good_output = MockLLMClient(canned_response=valid_llm_json)
    monkeypatch.setattr(extraction_engine, "llm_client", client_with_good_output)

    session_id = "test-session-llm-measurements"
    resp = client.post("/api/chat", json={
        "user_id": "4",
        "session_id": session_id,
        "message": "I am 62 years old and male. Bilirubin 10.9, direct 5.5, alk phos 699."
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["next_action"] == NextActionType.ASK_FOLLOW_UP

    collected = data["assessment_handoff"]["structured_inputs"]
    assert collected["age"] == 62
    assert collected["gender"] == "Male"
    assert collected["total_bilirubin"] == 10.9
    assert collected["direct_bilirubin"] == 5.5
    assert collected["alkaline_phosphatase"] == 699.0


