import json
import pytest
from unittest.mock import MagicMock, patch
import httpx
from fastapi.testclient import TestClient

from app.main import app, session_manager, extraction_engine
from app.schemas import (
    IntentType,
    SafetyStatus,
    NextActionType,
    Symptom,
    StructuredHealthInformation
)
from app.llm_client import (
    OpenAICompatibleLLMClient,
    MockLLMClient,
    LLMUnavailableError,
    LLMResponseError
)
from app.extraction import ExtractionEngine
from app.patient_context import (
    build_deterministic_context_explanation,
    explain_patient_context
)

client = TestClient(app)


@pytest.fixture(autouse=True)
def clean_state():
    """Reset session state and extraction engine between tests."""
    session_manager._sessions.clear()
    extraction_engine.llm_client = MockLLMClient()
    yield
    session_manager._sessions.clear()
    extraction_engine.llm_client = MockLLMClient()


# ==============================================================================
# Test A & B: Educational questions do NOT extract symptoms
# ==============================================================================
def test_educational_question_bilirubin_no_jaundice_symptom():
    """
    User asks: 'What is bilirubin?'
    Expected:
      - Intent: GENERAL_HEALTH_QUESTION
      - No user-reported 'jaundice' or other clinical symptom in structured_health_information.
    """
    response = client.post("/api/chat", json={
        "user_id": "test-edu-1",
        "session_id": "test-session-edu-1",
        "message": "What is bilirubin?"
    })
    assert response.status_code == 200
    data = response.json()

    assert data["intent"] == IntentType.GENERAL_HEALTH_QUESTION
    assert data["next_action"] == NextActionType.PROVIDE_GENERAL_INFO
    symptoms = data["structured_health_information"]["symptoms"]
    assert len(symptoms) == 0, f"Expected 0 symptoms, but found: {symptoms}"
    # Verify 'jaundice' is not present
    sym_names = [s["name"].lower() for s in symptoms]
    assert "jaundice" not in sym_names


def test_educational_question_jaundice_no_user_symptom():
    """
    User asks: 'What is jaundice?' or 'Can jaundice occur with high bilirubin?'
    Expected:
      - Intent: GENERAL_HEALTH_QUESTION
      - 'jaundice' is not attributed as an active patient symptom.
    """
    queries = [
        "What is jaundice?",
        "Can jaundice occur with high bilirubin?",
        "Explain bilirubin.",
        "What does bilirubin mean?"
    ]

    for idx, query in enumerate(queries):
        response = client.post("/api/chat", json={
            "user_id": f"test-edu-{idx}",
            "session_id": f"test-session-edu-{idx}",
            "message": query
        })
        assert response.status_code == 200
        data = response.json()
        assert data["intent"] == IntentType.GENERAL_HEALTH_QUESTION
        symptoms = data["structured_health_information"]["symptoms"]
        assert len(symptoms) == 0, f"Query '{query}' produced unexpected symptoms: {symptoms}"


def test_educational_question_with_mocked_llm_hallucinating_jaundice():
    """
    Simulates real LLM behavior where Ollama returned 'jaundice' symptom for 'What is bilirubin?'.
    Verifies that the extraction engine filters out the hallucinated symptom.
    """
    mock_llm_output = json.dumps({
        "assistant_message": "Bilirubin is a compound formed by the breakdown of hemoglobin.",
        "intent": "GENERAL_HEALTH_QUESTION",
        "extracted_info": {
            "symptoms": [
                {"name": "jaundice", "location": None, "duration": None, "severity": None, "onset": None, "frequency": None, "associated_symptoms": []}
            ],
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
            "measurements": []
        }
    })

    extraction_engine.llm_client = MockLLMClient(canned_response=mock_llm_output)

    response = client.post("/api/chat", json={
        "user_id": "test-edu-llm-1",
        "session_id": "test-session-edu-llm-1",
        "message": "What is bilirubin?"
    })
    assert response.status_code == 200
    data = response.json()

    # The hallucinated jaundice symptom must be filtered out
    symptoms = data["structured_health_information"]["symptoms"]
    assert len(symptoms) == 0, f"Hallucinated symptom was not filtered out: {symptoms}"


# ==============================================================================
# Test C & D: Actual symptom reports ARE recognized
# ==============================================================================
def test_actual_symptom_i_have_jaundice():
    """
    User reports: 'I have jaundice.'
    Expected:
      - Intent: SYMPTOM_ASSESSMENT
      - Symptom recognized: 'jaundice'
    """
    response = client.post("/api/chat", json={
        "user_id": "test-sym-1",
        "session_id": "test-session-sym-1",
        "message": "I have jaundice."
    })
    assert response.status_code == 200
    data = response.json()

    assert data["intent"] in (IntentType.SYMPTOM_ASSESSMENT, IntentType.FOLLOW_UP_RESPONSE)
    symptoms = data["structured_health_information"]["symptoms"]
    assert len(symptoms) >= 1
    sym_names = [s["name"].lower() for s in symptoms]
    assert "jaundice" in sym_names


def test_actual_symptom_diagnosed_with_jaundice():
    """
    User reports: 'I was diagnosed with jaundice.'
    Expected:
      - Jaundice recognized as clinical entity.
    """
    response = client.post("/api/chat", json={
        "user_id": "test-sym-diag",
        "session_id": "test-session-sym-diag",
        "message": "I was diagnosed with jaundice."
    })
    assert response.status_code == 200
    data = response.json()
    symptoms = data["structured_health_information"]["symptoms"]
    assert len(symptoms) >= 1
    assert "jaundice" in [s["name"].lower() for s in symptoms]


def test_actual_symptom_my_eyes_are_yellow():
    """
    User reports: 'My eyes are yellow.' / 'I have yellowing of my eyes.'
    Expected:
      - Recognized as jaundice/yellowing symptom.
    """
    variations = [
        "My eyes are yellow.",
        "I have yellowing of my eyes.",
        "I am experiencing jaundice."
    ]

    for idx, var in enumerate(variations):
        response = client.post("/api/chat", json={
            "user_id": f"test-yellow-{idx}",
            "session_id": f"test-session-yellow-{idx}",
            "message": var
        })
        assert response.status_code == 200
        data = response.json()
        symptoms = data["structured_health_information"]["symptoms"]
        assert len(symptoms) >= 1, f"Failed to extract symptom from: '{var}'"
        sym_names = [s["name"].lower() for s in symptoms]
        assert "jaundice" in sym_names


# ==============================================================================
# Test E: Emergency safety remains BEFORE LLM invocation
# ==============================================================================
def test_emergency_safety_runs_before_llm():
    """
    User message: 'I have severe crushing chest pain and severe shortness of breath.'
    Expected:
      - URGENT_MEDICAL_EVALUATION
      - EMERGENCY_ROUTING
      - LLM client is NEVER called
    """
    mock_llm = MagicMock()
    extraction_engine.llm_client = mock_llm

    response = client.post("/api/chat", json={
        "user_id": "test-emg-1",
        "session_id": "test-session-emg-1",
        "message": "I have severe crushing chest pain and severe shortness of breath."
    })
    assert response.status_code == 200
    data = response.json()

    assert data["safety_status"] == SafetyStatus.URGENT_MEDICAL_EVALUATION
    assert data["next_action"] == NextActionType.EMERGENCY_ROUTING
    # Ensure LLM was not touched
    assert mock_llm.generate_chat_response.call_count == 0
    assert mock_llm.generate_response.call_count == 0


# ==============================================================================
# Test F: Patient context wording: personal baseline/range, NOT normal range
# ==============================================================================
def test_patient_context_wording_personal_baseline_not_normal_range():
    """
    Verifies that patient context explanations describe baselines as 'personal baseline'
    or 'personal baseline/range', and NEVER as 'normal range' or 'within normal limits'.
    """
    sample_context = {
        "patient_id": 42,
        "has_history": True,
        "trends": [
            {
                "id": 101,
                "patientId": 42,
                "metricType": "GLUCOSE",
                "currentValue": 125.0,
                "previousValue": 122.0,
                "baselineValue": 122.5,
                "variability": 15.0,
                "deviationScore": 0.17,
                "deviationFromBaseline": 2.5,
                "absoluteDeviation": 2.5,
                "valueChangeFromPrevious": 3.0,
                "percentageChangeFromPrevious": 2.5,
                "trendDirection": "STABLE",
                "persistence": 3,
                "historicalContext": {"observationCount": 5}
            }
        ],
        "baselines": [],
        "recent_observations": []
    }

    # Test deterministic explanation
    explanation = build_deterministic_context_explanation(sample_context, requested_metric="GLUCOSE")
    exp_lower = explanation.lower()

    # Must refer to personal baseline
    assert "personal baseline" in exp_lower
    # Must NOT describe personal baseline as a clinical normal range
    assert "within the normal range" not in exp_lower
    assert "clinical normal range" not in exp_lower or "not a medical diagnosis or clinical normal range" in exp_lower


# ==============================================================================
# Test G: LLM Observability and Error Classification
# ==============================================================================
def test_llm_read_timeout_error_distinction():
    """
    Verifies that an httpx.ReadTimeout raises LLMUnavailableError with
    detailed diagnostic info including timeout duration and read timeout label.
    """
    client_instance = OpenAICompatibleLLMClient(
        api_key="test-key",
        base_url="http://localhost:11434/v1",
        model_name="llama3.2:3b",
        timeout=5.0
    )

    with patch("httpx.Client.post") as mock_post:
        mock_post.side_effect = httpx.ReadTimeout("The read operation timed out")

        with pytest.raises(LLMUnavailableError) as exc_info:
            client_instance.generate_chat_response(messages=[{"role": "user", "content": "hello"}])

        err_msg = str(exc_info.value)
        assert "did not respond within" in err_msg
        assert "read timeout" in err_msg.lower()


def test_llm_connect_timeout_error_distinction():
    """
    Verifies that an httpx.ConnectTimeout raises LLMUnavailableError with
    connect timeout label.
    """
    client_instance = OpenAICompatibleLLMClient(
        api_key="test-key",
        base_url="http://localhost:11434/v1",
        model_name="llama3.2:3b",
        timeout=5.0
    )

    with patch("httpx.Client.post") as mock_post:
        mock_post.side_effect = httpx.ConnectTimeout("Connect timed out")

        with pytest.raises(LLMUnavailableError) as exc_info:
            client_instance.generate_chat_response(messages=[{"role": "user", "content": "hello"}])

        err_msg = str(exc_info.value)
        assert "connection timed out" in err_msg.lower()


def test_llm_http_500_upstream_error_distinction():
    """
    Verifies that an HTTP 500 status raises LLMUnavailableError with status code.
    """
    client_instance = OpenAICompatibleLLMClient(
        api_key="test-key",
        base_url="http://localhost:11434/v1",
        model_name="llama3.2:3b"
    )

    mock_resp = MagicMock()
    mock_resp.status_code = 502
    mock_resp.text = "Bad Gateway"

    with patch("httpx.Client.post", return_value=mock_resp):
        with pytest.raises(LLMUnavailableError) as exc_info:
            client_instance.generate_chat_response(messages=[{"role": "user", "content": "hello"}])

        assert "HTTP 502" in str(exc_info.value)


def test_llm_http_400_client_error_distinction():
    """
    Verifies that an HTTP 400 status raises LLMResponseError with status code.
    """
    client_instance = OpenAICompatibleLLMClient(
        api_key="test-key",
        base_url="http://localhost:11434/v1",
        model_name="llama3.2:3b"
    )

    mock_resp = MagicMock()
    mock_resp.status_code = 404
    mock_resp.text = "Model not found"

    with patch("httpx.Client.post", return_value=mock_resp):
        with pytest.raises(LLMResponseError) as exc_info:
            client_instance.generate_chat_response(messages=[{"role": "user", "content": "hello"}])

        assert "HTTP 404" in str(exc_info.value)
