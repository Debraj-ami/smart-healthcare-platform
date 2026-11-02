import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from app.main import app, session_manager, extraction_engine
from app.schemas import (
    IntentType,
    SafetyStatus,
    NextActionType
)

from app.llm_client import BaseLLMClient, MockLLMClient
from app.fast_path import FastPathDetector
from app.liver_assessment import LiverExtractor

client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_state():
    """Ensure in-memory sessions and extraction engine are reset between tests."""
    session_manager._sessions.clear()
    extraction_engine.llm_client = MockLLMClient()
    yield
    session_manager._sessions.clear()
    extraction_engine.llm_client = MockLLMClient()


# ============================================================
# 1. Pure Lab Input Does NOT Invoke LLM (Fast Path Verification)
# ============================================================
def test_pure_lab_input_bypasses_llm():
    """
    Input: "My Total Bilirubin is 10.9 mg/dL and Albumin is 3.2 g/dL."
    Verifies:
      - Does NOT invoke the LLM (mock LLM call_count == 0)
      - session.last_routing is FAST_PATH
      - Measurements extracted into session.health_info.measurements
      - Liver assessment state updated and follow-up asks for missing fields
    """
    mock_llm = MagicMock(spec=BaseLLMClient)
    extraction_engine.llm_client = mock_llm

    payload = {
        "user_id": "test-user-fp-1",
        "session_id": "test-session-fp-1",
        "message": "My Total Bilirubin is 10.9 mg/dL and Albumin is 3.2 g/dL."
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()

    # CRITICAL: LLM was never called
    assert mock_llm.generate_chat_response.call_count == 0

    session = session_manager.get_or_create("test-session-fp-1", "test-user-fp-1")
    assert session.last_routing == "FAST_PATH"
    assert session.liver_assessment is not None
    assert session.liver_assessment.values["total_bilirubin"] == 10.9
    assert session.liver_assessment.values["albumin"] == 3.2

    # Check Pydantic Measurement structure
    bili_m = next((m for m in session.health_info.measurements if m.name == "total_bilirubin"), None)
    alb_m = next((m for m in session.health_info.measurements if m.name == "albumin"), None)
    assert bili_m is not None
    assert bili_m.value == 10.9
    assert bili_m.unit == "mg/dL"
    assert alb_m is not None
    assert alb_m.value == 3.2
    assert alb_m.unit == "g/dL"

    # Missing fields prompt
    assert data["next_action"] == NextActionType.ASK_FOLLOW_UP
    assert "Direct Bilirubin" in data["assistant_message"]


def test_fast_path_aliases():
    """
    Verifies common aliases (t bili, d bili, alk_phos, alt, ast) bypass the LLM.
    """
    mock_llm = MagicMock(spec=BaseLLMClient)
    extraction_engine.llm_client = mock_llm

    # Turn with t_bili and alt
    payload = {
        "user_id": "test-user-alias",
        "session_id": "test-session-alias",
        "message": "t_bili 10.9, ALT 64"
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200

    assert mock_llm.generate_chat_response.call_count == 0
    session = session_manager.get_or_create("test-session-alias", "test-user-alias")
    assert session.last_routing == "FAST_PATH"
    assert session.liver_assessment.values["total_bilirubin"] == 10.9
    assert session.liver_assessment.values["sgpt"] == 64.0


# ============================================================
# 2. General Health Questions DO Use LLM Path
# ============================================================
def test_general_health_question_uses_llm_path():
    """
    Input: "What is bilirubin?" or "Can you tell me what bilirubin means?"
    Verifies:
      - Does NOT trigger fast-path
      - session.last_routing is LLM_PATH
      - Returns educational health info without fast-path lab extraction
    """
    payload = {
        "user_id": "test-user-gen",
        "session_id": "test-session-gen",
        "message": "What is bilirubin?"
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()

    session = session_manager.get_or_create("test-session-gen", "test-user-gen")
    assert session.last_routing == "LLM_PATH"
    assert data["intent"] == IntentType.GENERAL_HEALTH_QUESTION
    assert data["next_action"] == NextActionType.PROVIDE_GENERAL_INFO
    assert len(session.health_info.measurements) == 0


def test_question_about_lab_meaning_uses_llm_path():
    """
    Input: "Can you tell me what bilirubin means?"
    """
    payload = {
        "user_id": "test-user-gen2",
        "session_id": "test-session-gen2",
        "message": "Can you tell me what bilirubin means?"
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200

    session = session_manager.get_or_create("test-session-gen2", "test-user-gen2")
    assert session.last_routing == "LLM_PATH"


# ============================================================
# 3. Symptom-First Messages DO Use Symptom Flow (Path B Protection)
# ============================================================
def test_symptom_first_message_protected_from_fast_path():
    """
    Inputs:
      - "I have pain under my right ribs for two days."
      - "I'm worried about my liver."
      - "Do I have liver disease?"
      - "I have been feeling tired and my doctor mentioned liver problems."
    Verifies:
      - All route through LLM_PATH (Path B / Conversational)
      - None are hijacked by fast path
    """
    test_messages = [
        "I have pain under my right ribs for two days.",
        "I'm worried about my liver.",
        "Do I have liver disease?",
        "I have been feeling tired and my doctor mentioned liver problems."
    ]

    for idx, msg in enumerate(test_messages):
        sid = f"test-sym-session-{idx}"
        uid = f"test-sym-user-{idx}"
        payload = {"user_id": uid, "session_id": sid, "message": msg}
        response = client.post("/api/chat", json=payload)
        assert response.status_code == 200

        session = session_manager.get_or_create(sid, uid)
        assert session.last_routing == "LLM_PATH", f"Message '{msg}' incorrectly routed to {session.last_routing}"


# ============================================================
# 4. Emergency Messages Handled BEFORE Fast-Path Extraction
# ============================================================
def test_emergency_handled_before_fast_path():
    """
    Input: Emergency message containing a lab-like mention or chest pain.
    "I have crushing chest pain, difficulty breathing, and my bilirubin is 10.9"
    Verifies:
      - Evaluated and intercepted by safety red-flag check immediately
      - Returns URGENT_MEDICAL_EVALUATION
      - Does not execute fast path
    """
    payload = {
        "user_id": "test-user-emerg",
        "session_id": "test-session-emerg",
        "message": "I have crushing chest pain, difficulty breathing, and my bilirubin is 10.9"
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["safety_status"] == SafetyStatus.URGENT_MEDICAL_EVALUATION
    assert data["next_action"] == NextActionType.EMERGENCY_ROUTING
    assert "emergency services" in data["assistant_message"].lower()


# ============================================================
# 5. Explicit Report Requests Still Enter Path C
# ============================================================
def test_explicit_report_request_enters_path_c():
    """
    Input: "I have my liver blood test results."
    Verifies:
      - Does not match fast path (no numerical values yet)
      - Routes to LLM_PATH
      - Initiates liver assessment state in Path C asking user to supply values
    """
    payload = {
        "user_id": "test-user-report",
        "session_id": "test-session-report",
        "message": "I have my liver blood test results."
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()

    session = session_manager.get_or_create("test-session-report", "test-user-report")
    assert session.last_routing == "LLM_PATH"
    assert session.liver_assessment is not None
    assert session.liver_assessment.status == "IN_PROGRESS"
    assert data["next_action"] == NextActionType.ASK_FOLLOW_UP


# ============================================================
# 6. Multi-Turn State Persistence Across Fast-Path Turns
# ============================================================
def test_multi_turn_fast_path_state_persistence():
    """
    Turn 1: "Age 62, male."
    Turn 2: "Bilirubin 10.9 and direct bilirubin 5.5."
    Turn 3: "SGPT 64, SGOT 100."
    Turn 4: "Albumin 3.2 and A/G ratio 0.74."

    Verifies:
      - Turn 1 extracts age 62, male
      - Turn 2 activates FAST_PATH, inherits age and gender, adds bilirubin & direct bilirubin
      - Turn 3 activates FAST_PATH, preserves all previous values, adds sgpt & sgot
      - Turn 4 activates FAST_PATH, preserves all previous values, adds albumin & a_g_ratio
    """
    uid = "test-user-multiturn"
    sid = "test-session-multiturn"

    # Turn 1: Demographics
    r1 = client.post("/api/chat", json={"user_id": uid, "session_id": sid, "message": "Age 62, male."})
    assert r1.status_code == 200
    session = session_manager.get_or_create(sid, uid)
    assert session.health_info.age == 62
    assert session.health_info.sex == "Male"

    # Turn 2: Bilirubin values (Fast Path)
    r2 = client.post("/api/chat", json={
        "user_id": uid, "session_id": sid,
        "message": "Bilirubin 10.9 and direct bilirubin 5.5."
    })
    assert r2.status_code == 200
    assert session.last_routing == "FAST_PATH"
    assert session.liver_assessment.values["age"] == 62
    assert session.liver_assessment.values["gender"] == "Male"
    assert session.liver_assessment.values["total_bilirubin"] == 10.9
    assert session.liver_assessment.values["direct_bilirubin"] == 5.5

    # Turn 3: Enzymes (Fast Path)
    r3 = client.post("/api/chat", json={
        "user_id": uid, "session_id": sid,
        "message": "SGPT 64, SGOT 100."
    })
    assert r3.status_code == 200
    assert session.last_routing == "FAST_PATH"
    # Verify previous values are intact
    assert session.liver_assessment.values["age"] == 62
    assert session.liver_assessment.values["gender"] == "Male"
    assert session.liver_assessment.values["total_bilirubin"] == 10.9
    assert session.liver_assessment.values["direct_bilirubin"] == 5.5
    # Verify new values
    assert session.liver_assessment.values["sgpt"] == 64.0
    assert session.liver_assessment.values["sgot"] == 100.0

    # Turn 4: Proteins & Ratio (Fast Path)
    r4 = client.post("/api/chat", json={
        "user_id": uid, "session_id": sid,
        "message": "Albumin 3.2 and A/G ratio 0.74."
    })
    assert r4.status_code == 200
    assert session.last_routing == "FAST_PATH"
    # Verify all 8 fields collected so far are intact
    vals = session.liver_assessment.values
    assert vals["age"] == 62
    assert vals["gender"] == "Male"
    assert vals["total_bilirubin"] == 10.9
    assert vals["direct_bilirubin"] == 5.5
    assert vals["sgpt"] == 64.0
    assert vals["sgot"] == 100.0
    assert vals["albumin"] == 3.2
    assert vals["a_g_ratio"] == 0.74

    # Missing fields should be only alkaline_phosphatase and total_proteins
    missing = session.liver_assessment.get_missing_fields()
    assert set(missing) == {"alkaline_phosphatase", "total_proteins"}


# ============================================================
# 7. Multiple Lab Values in One Message
# ============================================================
def test_multiple_lab_values_extracted_correctly():
    """
    Input: "Bilirubin 10.9, SGPT 64, SGOT 100."
    Verifies:
      - 3 lab values extracted in a single fast-path pass
      - Appropriate units assigned
      - Saved into session.health_info.measurements
    """
    uid = "test-user-multi-val"
    sid = "test-session-multi-val"

    response = client.post("/api/chat", json={
        "user_id": uid,
        "session_id": sid,
        "message": "Bilirubin 10.9, SGPT 64, SGOT 100."
    })
    assert response.status_code == 200

    session = session_manager.get_or_create(sid, uid)
    assert session.last_routing == "FAST_PATH"

    meas = {m.name: (m.value, m.unit) for m in session.health_info.measurements}
    assert meas["total_bilirubin"] == (10.9, "mg/dL")
    assert meas["sgpt"] == (64.0, "U/L")
    assert meas["sgot"] == (100.0, "U/L")


# ============================================================
# 8. Malformed/Ambiguous Numerical Text Falls Back Safely
# ============================================================
def test_malformed_and_ambiguous_text_falls_back_safely():
    """
    Inputs:
      - "Bilirubin high"
      - "My bilirubin was about 10 or 20, not sure"
      - "SGPT is maybe around eighty"
      - "Bilirubin 10-20"
    Verifies:
      - Fall back safely to LLM_PATH
      - Do NOT silently become accepted medical data in session.health_info.measurements
    """
    malformed_inputs = [
        "Bilirubin high",
        "My bilirubin was about 10 or 20, not sure",
        "SGPT is maybe around eighty",
        "Bilirubin 10-20"
    ]

    for idx, text in enumerate(malformed_inputs):
        sid = f"test-malform-session-{idx}"
        uid = f"test-malform-user-{idx}"
        response = client.post("/api/chat", json={
            "user_id": uid,
            "session_id": sid,
            "message": text
        })
        assert response.status_code == 200

        session = session_manager.get_or_create(sid, uid)
        assert session.last_routing == "LLM_PATH"
        # Must not silently create fraudulent measurements
        assert len(session.health_info.measurements) == 0


# ============================================================
# 9. Fast-Path Completion to Spring Boot Prediction
# ============================================================
def test_fast_path_completes_assessment_with_spring_boot():
    """
    When all 10 fields are supplied via fast path, verifies:
      - predict_liver is called on the backend client
      - ChatResponse contains the assessment handoff and results
      - session.last_routing is FAST_PATH
    """
    uid = "test-user-fp-complete"
    sid = "test-session-fp-complete"

    # Pre-populate session with demographics
    session = session_manager.get_or_create(sid, uid)
    session.health_info.age = 50
    session.health_info.sex = "Female"

    mock_prediction = {
        "assessmentId": 999,
        "patientId": 1,
        "disease": "LIVER_DISEASE",
        "probability": 0.825,
        "modelName": "Logistic Regression",
        "modelVersion": "1.0",
        "calibrationMethod": "Isotonic Calibration",
        "shapFeatures": [
            {"feature": "Total Bilirubin", "value": "10.9", "contribution": 0.35, "direction": "INCREASES_RISK"}
        ]
    }

    all_labs_msg = (
        "Total Bilirubin 10.9, Direct Bilirubin 5.5, Alk Phos 699, "
        "SGPT 64, SGOT 100, Total Proteins 6.8, Albumin 3.2, A/G Ratio 0.74"
    )

    with patch("app.main.get_backend_client") as mock_get_client:
        mock_client_instance = MagicMock()
        mock_client_instance.predict_liver.return_value = mock_prediction
        mock_get_client.return_value = mock_client_instance

        response = client.post("/api/chat", json={
            "user_id": uid,
            "session_id": sid,
            "message": all_labs_msg
        })
        assert response.status_code == 200
        data = response.json()

        assert session.last_routing == "FAST_PATH"
        assert session.liver_assessment.status == "COMPLETED"
        assert mock_client_instance.predict_liver.call_count == 1
        assert "82.5%" in data["assistant_message"]
        assert data["assessment_handoff"] is not None
        assert data["assessment_handoff"]["assessment_state"]["status"] == "COMPLETED"
