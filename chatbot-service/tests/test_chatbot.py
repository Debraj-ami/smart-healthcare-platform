import pytest
import json
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
import sys
import os

# Add chatbot-service to sys.path so app can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import app, session_manager, extraction_engine
from app.schemas import (
    IntentType,
    SafetyStatus,
    NextActionType
)
from app.llm_client import (
    BaseLLMClient,
    MockLLMClient,
    OpenAICompatibleLLMClient,
    LLMUnavailableError,
    LLMResponseError,
    get_llm_client
)
from app.extraction import ExtractionEngine
from app.safety import SafetyChecker


client = TestClient(app)


@pytest.fixture(autouse=True)
def clean_sessions():
    """Ensure in-memory sessions and extraction engine are reset between tests."""
    session_manager._sessions.clear()
    extraction_engine.llm_client = MockLLMClient()
    yield
    session_manager._sessions.clear()
    extraction_engine.llm_client = MockLLMClient()


# ============================================================
# 1. General Health Question Test
# ============================================================
def test_general_health_question():
    """
    Input: "What is blood pressure?" / "What does high bilirubin mean?"
    Verifies:
      - intent is GENERAL_HEALTH_QUESTION
      - next_action is PROVIDE_GENERAL_INFO
      - educational information provided without diagnosing or prescribing treatment
    """
    payload = {
        "user_id": "test-user-3",
        "session_id": "test-session-3",
        "message": "What is blood pressure?"
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["intent"] == IntentType.GENERAL_HEALTH_QUESTION
    assert data["next_action"] == NextActionType.PROVIDE_GENERAL_INFO
    assert "mmHg" in data["assistant_message"] or "arteries" in data["assistant_message"]
    assert data["safety_status"] == SafetyStatus.NORMAL_CONVERSATION
    assert data["assessment_handoff"] is None


def test_general_health_question_bilirubin():
    """Input: "What does high bilirubin mean?" verifies cautious educational response without diagnosis."""
    payload = {
        "user_id": "test-user-bili",
        "session_id": "test-session-bili",
        "message": "What does high bilirubin mean?"
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["intent"] == IntentType.GENERAL_HEALTH_QUESTION
    assert data["next_action"] == NextActionType.PROVIDE_GENERAL_INFO
    assert "bilirubin" in data["assistant_message"].lower()
    # Must NOT diagnose the patient
    assert "you have" not in data["assistant_message"].lower()


# ============================================================
# 2. Normal Conversation Test
# ============================================================
def test_normal_conversation():
    """
    Input: "I take metformin for my diabetes."
    Verifies: standard non-emergency conversational intent, updates context safely.
    """
    payload = {
        "user_id": "test-user-norm",
        "session_id": "test-session-norm",
        "message": "I take metformin for my diabetes."
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["intent"] == IntentType.MEDICAL_HISTORY_UPDATE
    assert data["safety_status"] == SafetyStatus.NORMAL_CONVERSATION
    assert data["next_action"] == NextActionType.PROVIDE_GENERAL_INFO
    assert "Metformin" in data["structured_health_information"]["medications"]


# ============================================================
# 3. Structured Symptom Extraction Test (Never Invents Missing Values)
# ============================================================
def test_basic_symptom_extraction():
    """
    Input: "I've had pain below my right ribs for two days."
    Verifies:
      - symptom name: pain
      - location: right upper abdomen
      - duration: two days / 2 days
      - unstated fields (severity, frequency) remain strictly None/null
    """
    payload = {
        "user_id": "test-user-1",
        "session_id": "test-session-1",
        "message": "I've had pain below my right ribs for two days."
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["session_id"] == "test-session-1"
    assert data["intent"] == IntentType.SYMPTOM_ASSESSMENT

    health_info = data["structured_health_information"]
    assert len(health_info["symptoms"]) >= 1
    assert health_info["symptoms"][0]["name"] == "pain"
    assert "right" in health_info["location"]
    assert "days" in health_info["duration"]
    # Critical requirement: Never invent missing values
    assert health_info["severity"] is None
    assert health_info["frequency"] is None
    assert health_info["age"] is None


def test_structured_extraction_with_mocked_llm_json():
    """
    Tests that when a real/mocked LLM returns structured JSON:
      - JSON is parsed and validated by Pydantic
      - Extracted symptoms match
      - Missing values remain null (not invented)
    """
    canned_llm_json = json.dumps({
        "assistant_message": "I understand you have intermittent pain in your right ribs for 2 days.",
        "intent": "SYMPTOM_ASSESSMENT",
        "extracted_info": {
            "symptoms": [
                {
                    "name": "pain",
                    "location": "right upper abdomen",
                    "duration": "2 days",
                    "severity": None,
                    "onset": None,
                    "frequency": "intermittent",
                    "associated_symptoms": []
                }
            ],
            "location": "right upper abdomen",
            "duration": "2 days",
            "severity": None,
            "onset": None,
            "frequency": "intermittent",
            "associated_symptoms": [],
            "medical_history": [],
            "medications": [],
            "known_conditions": [],
            "age": None,
            "sex": None
        }
    })

    mock_client = MockLLMClient(canned_response=canned_llm_json)
    extraction_engine.llm_client = mock_client

    payload = {
        "user_id": "test-user-llm-json",
        "session_id": "test-session-llm-json",
        "message": "I have pain below my right ribs for two days. It comes and goes."
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()

    health_info = data["structured_health_information"]
    assert health_info["symptoms"][0]["name"] == "pain"
    assert health_info["location"] == "right upper abdomen"
    assert health_info["duration"] == "2 days"
    assert health_info["frequency"] == "intermittent"
    assert health_info["severity"] is None  # Not invented


# ============================================================
# 4. Missing Information -> Follow-Up Question Test
# ============================================================
def test_missing_information_follow_up():
    """
    Input: "I have pain in my abdomen."
    Verifies: System asks a targeted follow-up question instead of inventing missing values.
    """
    payload = {
        "user_id": "test-user-7",
        "session_id": "test-session-7",
        "message": "I have pain in my abdomen."
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["next_action"] == NextActionType.ASK_FOLLOW_UP
    assert data["follow_up_question"] is not None
    # Missing fields remain None
    assert data["structured_health_information"]["duration"] is None
    assert data["structured_health_information"]["severity"] is None


# ============================================================
# 5. Multi-Turn Session Context Test
# ============================================================
def test_follow_up_response_and_state_accumulation():
    """
    Turn 1: "I have pain below my right ribs."
    Turn 2: "It started two days ago."
    Verifies: Conversation state accumulates both location and duration across turns.
    """
    session_id = "test-session-multiturn"
    user_id = "test-user-2"

    # Turn 1
    resp1 = client.post("/api/chat", json={
        "user_id": user_id,
        "session_id": session_id,
        "message": "I have pain below my right ribs."
    })
    assert resp1.status_code == 200
    data1 = resp1.json()
    assert data1["structured_health_information"]["location"] == "right upper abdomen"
    assert data1["structured_health_information"]["duration"] is None

    # Turn 2
    resp2 = client.post("/api/chat", json={
        "user_id": user_id,
        "session_id": session_id,
        "message": "It started two days ago."
    })
    assert resp2.status_code == 200
    data2 = resp2.json()

    # Both location and duration must be retained
    assert data2["structured_health_information"]["location"] == "right upper abdomen"
    assert data2["structured_health_information"]["duration"] is not None
    assert "days" in data2["structured_health_information"]["duration"]
    assert data2["intent"] == IntentType.FOLLOW_UP_RESPONSE


# ============================================================
# 6. Structured Symptom Accumulation Test Across Multiple Turns
# ============================================================
def test_multi_turn_symptom_accumulation_chain():
    """
    Turn 1: Reports symptom & location
    Turn 2: Reports duration
    Turn 3: Reports severity
    Verifies sequential accumulation of attributes into state.
    """
    session_id = "test-session-accumulate"
    user_id = "test-user-accumulate"

    # Turn 1: Symptom & Location
    r1 = client.post("/api/chat", json={
        "user_id": user_id,
        "session_id": session_id,
        "message": "I am experiencing pain below my right ribs."
    })
    assert r1.status_code == 200
    d1 = r1.json()
    assert d1["structured_health_information"]["location"] == "right upper abdomen"
    assert d1["structured_health_information"]["duration"] is None

    # Turn 2: Duration
    r2 = client.post("/api/chat", json={
        "user_id": user_id,
        "session_id": session_id,
        "message": "For about 3 days."
    })
    assert r2.status_code == 200
    d2 = r2.json()
    assert d2["structured_health_information"]["location"] == "right upper abdomen"
    assert d2["structured_health_information"]["duration"] == "3 days"
    assert d2["structured_health_information"]["severity"] is None

    # Turn 3: Severity
    r3 = client.post("/api/chat", json={
        "user_id": user_id,
        "session_id": session_id,
        "message": "It is moderate pain, around 5/10."
    })
    assert r3.status_code == 200
    d3 = r3.json()
    assert d3["structured_health_information"]["location"] == "right upper abdomen"
    assert d3["structured_health_information"]["duration"] == "3 days"
    assert d3["structured_health_information"]["severity"] in ("moderate", "5/10")


# ============================================================
# 7. Urgent/Red-Flag Message -> Emergency Routing BEFORE LLM
# ============================================================
def test_urgent_symptom_routing():
    """
    Input: "I have sudden severe crushing chest pain radiating to my left arm and severe shortness of breath."
    Verifies:
      - safety_status = URGENT_MEDICAL_EVALUATION
      - next_action = EMERGENCY_ROUTING
      - emergency message urgently directs to emergency care
      - NO disease diagnosis is claimed
      - LLM is NOT called to downgrade the safety status
    """
    # Create a mock LLM that would try to downgrade if called
    downgrade_mock = MagicMock(spec=BaseLLMClient)
    downgrade_mock.generate_chat_response.return_value = json.dumps({
        "assistant_message": "You are totally fine, just rest.",
        "intent": "GENERAL_HEALTH_QUESTION",
        "extracted_info": {}
    })
    extraction_engine.llm_client = downgrade_mock

    payload = {
        "user_id": "test-user-emergency",
        "session_id": "test-session-emergency",
        "message": "I have sudden severe crushing chest pain radiating to my left arm and severe shortness of breath."
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["safety_status"] == SafetyStatus.URGENT_MEDICAL_EVALUATION
    assert data["next_action"] == NextActionType.EMERGENCY_ROUTING
    assert "EMERGENCY" in data["assistant_message"]
    assert "911" in data["assistant_message"] or "emergency" in data["assistant_message"].lower()

    # Verify LLM was NEVER called because safety check ran first
    assert downgrade_mock.generate_chat_response.call_count == 0


# ============================================================
# 8. LLM Unavailable -> Safe Fallback (No Crash)
# ============================================================
def test_llm_unavailable_safe_fallback():
    """
    When the LLM provider fails with a network error or timeout:
      - Application does NOT crash with 500
      - Returns status 200 with safe controlled fallback message
      - Informs user of temporary unavailability
    """
    failing_client = MagicMock(spec=BaseLLMClient)
    failing_client.generate_chat_response.side_effect = LLMUnavailableError("Connection refused on port 11434")
    extraction_engine.llm_client = failing_client

    payload = {
        "user_id": "test-user-fallback",
        "session_id": "test-session-fallback",
        "message": "I have a question about my health."
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "service is temporarily unavailable" in data["assistant_message"].lower()
    assert data["safety_status"] == SafetyStatus.NORMAL_CONVERSATION
    assert data["next_action"] == NextActionType.PROVIDE_GENERAL_INFO


# ============================================================
# 9. Invalid LLM Structured Response -> Controlled Handling
# ============================================================
def test_invalid_llm_response_controlled_handling():
    """
    When the LLM returns invalid JSON or unparseable text:
      - Application does NOT crash with 500
      - Falls back safely to deterministic extraction without throwing unhandled exceptions
    """
    broken_client = MagicMock(spec=BaseLLMClient)
    broken_client.generate_chat_response.return_value = "This is not JSON at all! {broken: true,"
    extraction_engine.llm_client = broken_client

    payload = {
        "user_id": "test-user-broken",
        "session_id": "test-session-broken",
        "message": "I have pain below my right ribs for two days."
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()

    # Rule-based fallback still successfully extracted the key details!
    assert data["structured_health_information"]["location"] == "right upper abdomen"
    assert data["structured_health_information"]["symptoms"][0]["name"] == "pain"


# ============================================================
# 10. No Fabricated Probability Test
# ============================================================
def test_no_fabricated_probability():
    """
    Verifies that the chatbot never invents a disease risk score or probability.
    """
    payload = {
        "user_id": "test-user-prob",
        "session_id": "test-session-prob",
        "message": "What is my probability of having liver disease?"
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()

    msg = data["assistant_message"].lower()
    assert "probability: " not in msg
    assert "risk score: " not in msg
    assert "85%" not in msg
    assert "70%" not in msg
    # Confirms probability calculation requires laboratory values
    assert "laboratory" in msg or "clinical parameters" in msg or "no disease probability" in msg


# ============================================================
# 11. No Fabricated Diagnosis Test
# ============================================================
def test_no_fabricated_diagnosis():
    """
    Input: "I have had yellowing of my eyes and fatigue for a week."
    Verifies:
      - Chatbot does not state a definitive diagnosis
      - Missing values remain null/empty
    """
    payload = {
        "user_id": "test-user-diag",
        "session_id": "test-session-diag",
        "message": "I have had yellowing of my eyes and fatigue for a week."
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()

    msg = data["assistant_message"].lower()
    assert "you have liver disease" not in msg
    assert "you are diagnosed with" not in msg
    assert "diagnosis:" not in msg


# ============================================================
# 12. Disease Assessment Handoff When Enough Information Exists
# ============================================================
def test_disease_assessment_handoff_when_enough_information():
    """
    When sufficient descriptive context (symptom, location, duration, severity, frequency)
    has been collected:
      - next_action is REQUEST_DISEASE_ASSESSMENT
      - assessment_handoff contains disease_candidates (e.g. liver_disease)
      - structured_inputs contains validated clinical data
      - No fake probability is invented
    """
    session_id = "test-session-handoff"
    user_id = "test-user-handoff"

    # Turn 1: Symptom, location, duration
    client.post("/api/chat", json={
        "user_id": user_id,
        "session_id": session_id,
        "message": "I have had pain below my right ribs for two days."
    })

    # Turn 2: Severity
    client.post("/api/chat", json={
        "user_id": user_id,
        "session_id": session_id,
        "message": "It is moderate pain."
    })

    # Turn 3: Frequency
    client.post("/api/chat", json={
        "user_id": user_id,
        "session_id": session_id,
        "message": "The pain is intermittent, it comes and goes."
    })

    # Turn 4: Contextual liver questions answered
    resp = client.post("/api/chat", json={
        "user_id": user_id,
        "session_id": session_id,
        "message": "No fever or yellowing of eyes."
    })

    assert resp.status_code == 200
    data = resp.json()

    assert data["next_action"] == NextActionType.REQUEST_DISEASE_ASSESSMENT
    assert data["assessment_handoff"] is not None
    assert "liver_disease" in data["assessment_handoff"]["disease_candidates"]
    assert data["assessment_handoff"]["structured_inputs"]["location"] == "right upper abdomen"


# ============================================================
# 13. Disease Risk Assessment Direct Request Test
# ============================================================
def test_disease_risk_assessment_request():
    """
    Input: "I want to check my diabetes risk."
    Verifies: intent is DISEASE_RISK_ASSESSMENT, next_action is REQUEST_DISEASE_ASSESSMENT.
    """
    payload = {
        "user_id": "test-user-4",
        "session_id": "test-session-4",
        "message": "I want to check my diabetes risk."
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["intent"] == IntentType.DISEASE_RISK_ASSESSMENT
    assert data["next_action"] == NextActionType.REQUEST_DISEASE_ASSESSMENT
    assert data["assessment_handoff"] is not None
    assert "diabetes" in data["assessment_handoff"]["disease_candidates"]


# ============================================================
# 14. Unknown Input Safe Handling Test
# ============================================================
def test_unknown_input_handling():
    """
    Input: "xyz 123 random words with no health meaning"
    Verifies: handled safely without crash, intent is UNKNOWN, next_action is CLARIFY.
    """
    payload = {
        "user_id": "test-user-5",
        "session_id": "test-session-5",
        "message": "xyz 123 blabla qwerty"
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["intent"] == IntentType.UNKNOWN
    assert data["next_action"] == NextActionType.CLARIFY
    assert "clarify" in data["assistant_message"].lower()


# ============================================================
# 15. Health Liveness Endpoint Test
# ============================================================
def test_health_endpoint():
    """Verifies GET /health returns 200 and healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "chatbot-service"
    assert "version" in data
    assert "llm_mode" in data


# ============================================================
# 16. Ollama Configuration Recognition Test (Offline / Finite)
# ============================================================
def test_ollama_configuration_recognized(monkeypatch):
    """
    Verifies that the Ollama configuration in .env is recognized and correctly
    configures the LLM client without requiring an active Ollama server.
    Also verifies that the mock provider remains selectable when configured.
    """
    # 1. Verify that get_llm_client recognizes the active Ollama environment configuration
    client_instance = get_llm_client()
    assert isinstance(client_instance, OpenAICompatibleLLMClient)
    assert not isinstance(client_instance, MockLLMClient)
    assert client_instance.base_url == "http://localhost:11434/v1"
    assert client_instance.model_name == "llama3.2:3b"
    assert client_instance.api_key == "ollama"
    assert client_instance.timeout == 60.0

    # 2. Verify ExtractionEngine initialized with Ollama client reflects real LLM provider
    engine = ExtractionEngine(llm_client=client_instance)
    assert engine.mode_label == "real LLM provider"

    # 3. Verify mock provider remains available when LLM_PROVIDER is set to mock
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    mock_instance = get_llm_client()
    assert isinstance(mock_instance, MockLLMClient)
    mock_engine = ExtractionEngine(llm_client=mock_instance)
    assert mock_engine.mode_label == "development/testing extraction -- not a medical NLP model"


# ============================================================
# 17. LLM Request Timeout Configuration Test
# ============================================================
def test_llm_timeout_configuration(monkeypatch):
    """
    Verifies that the LLM client timeout defaults to 60s, is configurable via
    LLM_TIMEOUT_SECONDS, and can be overridden via explicit constructor argument.
    """
    # 1. Configurable via LLM_TIMEOUT_SECONDS environment variable
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "90")
    custom_client = OpenAICompatibleLLMClient(
        api_key="test-key",
        base_url="http://localhost:11434/v1",
        model_name="llama3.2:3b"
    )
    assert custom_client.timeout == 90.0

    # 2. Explicit constructor argument takes precedence over environment variable
    explicit_client = OpenAICompatibleLLMClient(
        api_key="test-key",
        base_url="http://localhost:11434/v1",
        model_name="llama3.2:3b",
        timeout=45.0
    )
    assert explicit_client.timeout == 45.0

    # 3. Fallback to 60.0s when environment variable is unset or invalid
    monkeypatch.delenv("LLM_TIMEOUT_SECONDS", raising=False)
    default_client = OpenAICompatibleLLMClient(
        api_key="test-key",
        base_url="http://localhost:11434/v1",
        model_name="llama3.2:3b"
    )
    assert default_client.timeout == 60.0

    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "invalid_value")
    fallback_client = OpenAICompatibleLLMClient(
        api_key="test-key",
        base_url="http://localhost:11434/v1",
        model_name="llama3.2:3b"
    )
    assert fallback_client.timeout == 60.0
