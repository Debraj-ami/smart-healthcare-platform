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
from app.backend_client import BackendServiceUnavailableError

client = TestClient(app)

SAMPLE_TRENDS_GLUCOSE = [
    {
        "id": 101,
        "patientId": 42,
        "metricType": "GLUCOSE",
        "currentValue": 150.0,
        "previousValue": 120.0,
        "baselineValue": 122.5,
        "variability": 15.0,
        "deviationScore": 1.83,
        "deviationFromBaseline": 27.5,
        "absoluteDeviation": 27.5,
        "valueChangeFromPrevious": 30.0,
        "percentageChangeFromPrevious": 25.0,
        "trendDirection": "RISING",
        "persistence": 2,
        "historicalContext": {
            "observationCount": 4,
            "historicalMinimum": 100.0,
            "historicalMaximum": 150.0,
            "historicalMean": 120.0,
            "recentMean": 126.67,
            "diffHistoricalMean": 30.0,
            "diffRecentMean": 23.33
        },
        "analyzedAt": "2026-09-25T14:30:00"
    }
]

SAMPLE_OBSERVATIONS_GLUCOSE = [
    {
        "id": 1,
        "patientId": 42,
        "metricType": "GLUCOSE",
        "metricValue": 120.0,
        "unit": "mg/dL",
        "loggedAt": "2026-09-24T10:00:00"
    },
    {
        "id": 2,
        "patientId": 42,
        "metricType": "GLUCOSE",
        "metricValue": 150.0,
        "unit": "mg/dL",
        "loggedAt": "2026-09-25T10:00:00"
    }
]


@pytest.fixture(autouse=True)
def reset_state():
    """Ensure in-memory sessions and extraction engine are reset between tests."""
    session_manager._sessions.clear()
    extraction_engine.llm_client = MockLLMClient()
    yield
    session_manager._sessions.clear()
    extraction_engine.llm_client = MockLLMClient()


# ============================================================
# Test A: Personal-history question requests patient context
# ============================================================
def test_personal_history_queries_request_patient_context():
    """
    Verifies that explicit historical/personal health context questions
    activate PATIENT_CONTEXT routing and call backend_client.get_patient_health_context.
    """
    test_queries = [
        ("How has my health been recently?", None),
        ("How have my readings changed?", None),
        ("Has my glucose changed from my usual level?", "GLUCOSE"),
        ("What has changed in my recent measurements?", None),
        ("How am I doing compared with my previous readings?", None)
    ]

    for idx, (query, expected_metric) in enumerate(test_queries):
        sid = f"test-ctx-session-{idx}"
        uid = "42"

        with patch("app.main.get_backend_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.get_patient_health_context.return_value = {
                "patient_id": 42,
                "has_history": True,
                "trends": SAMPLE_TRENDS_GLUCOSE,
                "baselines": [],
                "recent_observations": SAMPLE_OBSERVATIONS_GLUCOSE
            }
            mock_get_client.return_value = mock_client

            response = client.post("/api/chat", json={
                "user_id": uid,
                "session_id": sid,
                "message": query
            })
            assert response.status_code == 200
            data = response.json()

            # Verify backend context was queried
            assert mock_client.get_patient_health_context.call_count == 1
            call_args = mock_client.get_patient_health_context.call_args
            assert call_args[0][0] == 42
            if expected_metric:
                assert call_args[1].get("metric_type") == expected_metric

            session = session_manager.get_or_create(sid, uid)
            assert session.last_routing == "PATIENT_CONTEXT"
            assert data["intent"] == IntentType.MONITORING_QUERY
            # Response contains real values from backend
            assert "150.0" in data["assistant_message"]
            assert "122.5" in data["assistant_message"]


# ============================================================
# Test B: General health question does NOT request patient context
# ============================================================
def test_general_health_question_does_not_request_patient_context():
    """
    Verifies that educational health questions route to LLM_PATH without calling context client.
    """
    gen_queries = [
        "What is glucose?",
        "What is bilirubin?",
        "What does high blood pressure mean?"
    ]

    for idx, query in enumerate(gen_queries):
        sid = f"test-gen-ctx-session-{idx}"
        uid = "42"

        with patch("app.main.get_backend_client") as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client

            response = client.post("/api/chat", json={
                "user_id": uid,
                "session_id": sid,
                "message": query
            })
            assert response.status_code == 200
            data = response.json()

            # Patient context client must NOT be called
            assert mock_client.get_patient_health_context.call_count == 0

            session = session_manager.get_or_create(sid, uid)
            assert session.last_routing == "LLM_PATH"
            assert data["intent"] == IntentType.GENERAL_HEALTH_QUESTION


# ============================================================
# Test C: Symptom-only message remains in symptom workflow
# ============================================================
def test_symptom_only_message_remains_in_symptom_workflow():
    """
    Verifies that subjective symptom messages route to Path B symptom screening without querying context.
    """
    symptom_queries = [
        "I have been feeling tired for three days.",
        "I have pain under my right ribs."
    ]

    for idx, query in enumerate(symptom_queries):
        sid = f"test-sym-ctx-session-{idx}"
        uid = "42"

        with patch("app.main.get_backend_client") as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client

            response = client.post("/api/chat", json={
                "user_id": uid,
                "session_id": sid,
                "message": query
            })
            assert response.status_code == 200
            data = response.json()

            assert mock_client.get_patient_health_context.call_count == 0
            session = session_manager.get_or_create(sid, uid)
            assert session.last_routing == "LLM_PATH"
            assert data["intent"] == IntentType.SYMPTOM_ASSESSMENT


# ============================================================
# Test D: Structured lab message still uses FAST_PATH
# ============================================================
def test_structured_lab_message_still_uses_fast_path():
    """
    Verifies Phase 2A fast path retains priority for structured lab inputs.
    """
    sid = "test-lab-fast-session"
    uid = "42"

    with patch("app.main.get_backend_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        response = client.post("/api/chat", json={
            "user_id": uid,
            "session_id": sid,
            "message": "My Total Bilirubin is 10.9 mg/dL and Albumin is 3.2 g/dL."
        })
        assert response.status_code == 200

        # Context retrieval must NOT be invoked
        assert mock_client.get_patient_health_context.call_count == 0
        session = session_manager.get_or_create(sid, uid)
        assert session.last_routing == "FAST_PATH"


# ============================================================
# Test E: Retrieved context is passed to the LLM as verified context
# ============================================================
def test_retrieved_context_passed_to_llm_with_guardrails():
    """
    Verifies that when a real/custom LLM is configured, it receives the exact
    verified context (currentValue, baseline, trendDirection) and strict guardrails.
    """
    mock_llm = MagicMock(spec=BaseLLMClient)
    mock_llm.generate_chat_response.return_value = (
        "Your blood glucose reading is currently 150.0 mg/dL, showing an upward trend compared to your 122.5 mg/dL baseline."
    )
    extraction_engine.llm_client = mock_llm

    sid = "test-llm-pass-session"
    uid = "42"

    with patch("app.main.get_backend_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.get_patient_health_context.return_value = {
            "patient_id": 42,
            "has_history": True,
            "trends": SAMPLE_TRENDS_GLUCOSE,
            "baselines": [],
            "recent_observations": SAMPLE_OBSERVATIONS_GLUCOSE
        }
        mock_get_client.return_value = mock_client

        response = client.post("/api/chat", json={
            "user_id": uid,
            "session_id": sid,
            "message": "Has my glucose changed from my usual level?"
        })
        assert response.status_code == 200

        assert mock_llm.generate_chat_response.call_count == 1
        call_kwargs = mock_llm.generate_chat_response.call_args[1]

        prompt_text = call_kwargs["messages"][0]["content"]
        system_prompt = call_kwargs["system_prompt"]

        # Context contains real numbers
        assert "150.0" in prompt_text
        assert "122.5" in prompt_text
        assert "RISING" in prompt_text
        assert "GLUCOSE" in prompt_text

        # System prompt contains strict medical constraints
        assert "NEVER invent" in system_prompt
        assert "NEVER diagnose" in system_prompt
        assert "distinguish observed statistical trend" in system_prompt.lower() or "distinguish observed statistical trends" in system_prompt.lower()


# ============================================================
# Test F & G: Missing patient history is handled safely without fabrication
# ============================================================
def test_missing_patient_history_handled_safely_no_fabrication():
    """
    When the patient has no records in Spring Boot:
      - Responds safely that no history is recorded yet
      - Does NOT fabricate default values (e.g. 0, 100, etc.)
      - LLM is not passed fabricated numbers
    """
    mock_llm = MagicMock(spec=BaseLLMClient)
    extraction_engine.llm_client = mock_llm

    sid = "test-missing-history-session"
    uid = "99"

    with patch("app.main.get_backend_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.get_patient_health_context.return_value = {
            "patient_id": 99,
            "has_history": False,
            "trends": [],
            "baselines": [],
            "recent_observations": []
        }
        mock_get_client.return_value = mock_client

        response = client.post("/api/chat", json={
            "user_id": uid,
            "session_id": sid,
            "message": "How have my readings changed?"
        })
        assert response.status_code == 200
        data = response.json()

        # LLM was NOT invoked with fabricated data
        assert mock_llm.generate_chat_response.call_count == 0

        # Transparent response stating no history available
        assert "no historical readings" in data["assistant_message"].lower()
        session = session_manager.get_or_create(sid, uid)
        assert session.last_routing == "PATIENT_CONTEXT"


# ============================================================
# Test H: Spring Boot / context retrieval failure is handled safely
# ============================================================
def test_backend_failure_handled_safely():
    """
    When Spring Boot is unavailable or times out:
      - Does not crash with HTTP 500
      - Returns status 200 with safe controlled explanation
      - Conversation is preserved
    """
    sid = "test-backend-fail-session"
    uid = "42"

    with patch("app.main.get_backend_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.get_patient_health_context.side_effect = BackendServiceUnavailableError("Connection refused on port 8080")
        mock_get_client.return_value = mock_client

        response = client.post("/api/chat", json={
            "user_id": uid,
            "session_id": sid,
            "message": "How has my health been recently?"
        })
        assert response.status_code == 200
        data = response.json()

        assert "unable to retrieve your historical health records" in data["assistant_message"].lower()
        session = session_manager.get_or_create(sid, uid)
        assert session.last_routing == "PATIENT_CONTEXT"
        assert len(session.history) == 2  # user + assistant turns preserved


# ============================================================
# Test I: Multi-turn conversation state remains intact
# ============================================================
def test_multiturn_conversation_state_with_patient_context():
    """
    Multi-turn flow:
      Turn 1: Symptom complaint ("I have had a mild headache for two days.") -> Path B
      Turn 2: Historical context question ("How have my readings changed?") -> PATIENT_CONTEXT
      Turn 3: Educational question ("What is glucose?") -> Path A
    Verifies all turns are preserved sequentially in ConversationState.
    """
    sid = "test-multiturn-ctx-session"
    uid = "42"

    with patch("app.main.get_backend_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.get_patient_health_context.return_value = {
            "patient_id": 42,
            "has_history": True,
            "trends": SAMPLE_TRENDS_GLUCOSE,
            "baselines": [],
            "recent_observations": SAMPLE_OBSERVATIONS_GLUCOSE
        }
        mock_get_client.return_value = mock_client

        # Turn 1: Symptom
        r1 = client.post("/api/chat", json={"user_id": uid, "session_id": sid, "message": "I have had a mild headache for two days."})
        assert r1.status_code == 200
        session = session_manager.get_or_create(sid, uid)
        assert session.last_routing == "LLM_PATH"
        assert len(session.health_info.symptoms) > 0

        # Turn 2: Patient Context
        r2 = client.post("/api/chat", json={"user_id": uid, "session_id": sid, "message": "How have my readings changed?"})
        assert r2.status_code == 200
        assert session.last_routing == "PATIENT_CONTEXT"
        # Symptom info from turn 1 must still be intact
        assert len(session.health_info.symptoms) > 0

        # Turn 3: Educational Question
        r3 = client.post("/api/chat", json={"user_id": uid, "session_id": sid, "message": "What is glucose?"})
        assert r3.status_code == 200
        assert session.last_routing == "LLM_PATH"

        # All 6 messages (3 user + 3 assistant) preserved
        assert len(session.history) == 6
        assert session.history[0].content == "I have had a mild headache for two days."
        assert session.history[2].content == "How have my readings changed?"
        assert session.history[4].content == "What is glucose?"
