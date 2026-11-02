import json
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app.main import app, session_manager, extraction_engine
from app.schemas import IntentType, SafetyStatus, NextActionType, StructuredHealthInformation, Symptom
from app.llm_client import BaseLLMClient, LLMUnavailableError
from app.extraction import StreamingJSONExtractor

client = TestClient(app)


def parse_sse_events(response_text: str):
    """Utility to parse SSE blocks into a list of (event_type, parsed_data) tuples."""
    events = []
    blocks = response_text.strip().split("\n\n")
    for block in blocks:
        if not block.strip():
            continue
        event_type = "message"
        data_str = ""
        for line in block.split("\n"):
            line = line.strip()
            if line.startswith("event:"):
                event_type = line.replace("event:", "").strip()
            elif line.startswith("data:"):
                data_str = line.replace("data:", "").strip()
        if data_str:
            try:
                parsed = json.loads(data_str)
                events.append((event_type, parsed))
            except json.JSONDecodeError:
                events.append((event_type, data_str))
    return events


# 1. Streaming endpoint exists and responds correctly
def test_streaming_endpoint_exists_and_responds():
    response = client.post(
        "/api/chat/stream",
        json={"user_id": "test-user-1", "session_id": "sess-stream-1", "message": "Hello doctor"}
    )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")
    events = parse_sse_events(response.text)
    assert len(events) >= 1
    # Check complete event is present
    complete_events = [data for ev_type, data in events if ev_type == "complete"]
    assert len(complete_events) == 1
    assert complete_events[0]["session_id"] == "sess-stream-1"


# 2. First stream event contains assistant text rather than raw JSON
def test_first_stream_event_contains_text_not_raw_json():
    # Set up LLM to return JSON chunks
    canned_json = json.dumps({
        "assistant_message": "Headaches can be caused by dehydration or stress.",
        "intent": "SYMPTOM_ASSESSMENT",
        "extracted_info": {
            "symptoms": [{"name": "headache", "duration": "today", "severity": "mild"}],
            "duration": "today",
            "severity": "mild"
        },
        "follow_up_question": "Are you experiencing any nausea?"
    })

    class MockStreamingLLM(BaseLLMClient):
        def generate_chat_response(self, messages, system_prompt=None, temperature=0.2, response_format=None):
            return canned_json

        def stream_chat_response(self, messages, system_prompt=None, temperature=0.2, response_format=None):
            # Yield in small chunks simulating SSE tokens
            chunk_size = 8
            for i in range(0, len(canned_json), chunk_size):
                yield canned_json[i:i + chunk_size]

    original_client = extraction_engine.llm_client
    try:
        extraction_engine.llm_client = MockStreamingLLM()
        response = client.post(
            "/api/chat/stream",
            json={"user_id": "test-user-2", "session_id": "sess-stream-2", "message": "I have a headache today"}
        )
        assert response.status_code == 200
        events = parse_sse_events(response.text)
        token_events = [data for ev_type, data in events if ev_type == "token"]
        assert len(token_events) > 0

        # Verify that no token event contains JSON scaffolding like '{"assistant_message":'
        for tok_data in token_events:
            text = tok_data.get("text", "")
            assert '{"assistant_message"' not in text
            assert '"intent":' not in text
            assert '"extracted_info":' not in text

        # Concatenated token text matches assistant_message
        full_streamed_text = "".join(t["text"] for t in token_events)
        assert "Headaches can be caused by dehydration or stress." in full_streamed_text
    finally:
        extraction_engine.llm_client = original_client


# 3. Final event contains structured health information
def test_final_event_contains_structured_health_information():
    canned_json = json.dumps({
        "assistant_message": "I note that you have had a headache for 2 days.",
        "intent": "SYMPTOM_ASSESSMENT",
        "extracted_info": {
            "symptoms": [{"name": "headache", "duration": "2 days", "severity": "moderate"}],
            "duration": "2 days",
            "severity": "moderate"
        },
        "follow_up_question": "Does light bother your eyes?"
    })

    class MockStreamingLLM(BaseLLMClient):
        def generate_chat_response(self, messages, system_prompt=None, temperature=0.2, response_format=None):
            return canned_json

        def stream_chat_response(self, messages, system_prompt=None, temperature=0.2, response_format=None):
            yield canned_json

    original_client = extraction_engine.llm_client
    try:
        extraction_engine.llm_client = MockStreamingLLM()
        response = client.post(
            "/api/chat/stream",
            json={"user_id": "test-user-3", "session_id": "sess-stream-3", "message": "I have a moderate headache for 2 days"}
        )
        assert response.status_code == 200
        events = parse_sse_events(response.text)
        complete_events = [data for ev_type, data in events if ev_type == "complete"]
        assert len(complete_events) == 1
        data = complete_events[0]
        assert data["intent"] == "SYMPTOM_ASSESSMENT"
        assert "structured_health_information" in data
        assert len(data["structured_health_information"]["symptoms"]) == 1
        assert data["structured_health_information"]["symptoms"][0]["name"] == "headache"
    finally:
        extraction_engine.llm_client = original_client


# 4. Existing /api/chat contract remains valid
def test_existing_chat_contract_remains_valid():
    response = client.post(
        "/api/chat",
        json={"user_id": "test-user-4", "session_id": "sess-chat-4", "message": "What is blood pressure?"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    assert "assistant_message" in data
    assert "intent" in data
    assert "structured_health_information" in data
    assert "safety_status" in data
    assert "next_action" in data
    assert data["session_id"] == "sess-chat-4"


# 5. Emergency safety still bypasses LLM streaming
def test_emergency_safety_bypasses_llm_streaming():
    call_count = {"count": 0}

    class SpyingLLM(BaseLLMClient):
        def generate_chat_response(self, messages, system_prompt=None, temperature=0.2, response_format=None):
            call_count["count"] += 1
            return "Should not be called"

        def stream_chat_response(self, messages, system_prompt=None, temperature=0.2, response_format=None):
            call_count["count"] += 1
            yield "Should not be called"

    original_client = extraction_engine.llm_client
    try:
        extraction_engine.llm_client = SpyingLLM()
        response = client.post(
            "/api/chat/stream",
            json={"user_id": "test-user-5", "session_id": "sess-stream-5", "message": "I have severe crushing chest pain radiating to my arm and shortness of breath"}
        )
        assert response.status_code == 200
        assert call_count["count"] == 0  # LLM was NEVER called

        events = parse_sse_events(response.text)
        complete_events = [data for ev_type, data in events if ev_type == "complete"]
        assert len(complete_events) == 1
        assert complete_events[0]["safety_status"] == SafetyStatus.URGENT_MEDICAL_EVALUATION.value
        assert complete_events[0]["next_action"] == NextActionType.EMERGENCY_ROUTING.value
        assert "911" in complete_events[0]["assistant_message"] or "emergency" in complete_events[0]["assistant_message"].lower()
    finally:
        extraction_engine.llm_client = original_client


# 6. Educational bilirubin query still produces no personal jaundice symptom
def test_educational_bilirubin_query_no_personal_jaundice():
    canned_json = json.dumps({
        "assistant_message": "Bilirubin is a yellowish compound produced during normal breakdown of red blood cells. High levels can cause jaundice.",
        "intent": "GENERAL_HEALTH_QUESTION",
        "extracted_info": {
            "symptoms": [{"name": "jaundice"}],  # Hallucinated symptom from educational explanation
            "duration": None,
            "severity": None
        },
        "follow_up_question": None
    })

    class EducationalMockLLM(BaseLLMClient):
        def generate_chat_response(self, messages, system_prompt=None, temperature=0.2, response_format=None):
            return canned_json

        def stream_chat_response(self, messages, system_prompt=None, temperature=0.2, response_format=None):
            yield canned_json

    original_client = extraction_engine.llm_client
    try:
        extraction_engine.llm_client = EducationalMockLLM()
        response = client.post(
            "/api/chat/stream",
            json={"user_id": "test-user-6", "session_id": "sess-stream-6", "message": "What is bilirubin and why is it measured?"}
        )
        assert response.status_code == 200
        events = parse_sse_events(response.text)
        complete_events = [data for ev_type, data in events if ev_type == "complete"]
        assert len(complete_events) == 1
        data = complete_events[0]
        # Jaundice MUST be filtered out
        symptoms = data["structured_health_information"]["symptoms"]
        assert len(symptoms) == 0
    finally:
        extraction_engine.llm_client = original_client


# 7. Personal symptom extraction remains functional
def test_personal_symptom_extraction_remains_functional():
    canned_json = json.dumps({
        "assistant_message": "I understand that your skin and eyes have turned yellow.",
        "intent": "SYMPTOM_ASSESSMENT",
        "extracted_info": {
            "symptoms": [{"name": "jaundice", "location": "skin and eyes", "duration": "3 days"}],
            "duration": "3 days"
        },
        "follow_up_question": "Do you also notice dark urine or abdominal pain?"
    })

    class PersonalSymptomLLM(BaseLLMClient):
        def generate_chat_response(self, messages, system_prompt=None, temperature=0.2, response_format=None):
            return canned_json

        def stream_chat_response(self, messages, system_prompt=None, temperature=0.2, response_format=None):
            yield canned_json

    original_client = extraction_engine.llm_client
    try:
        extraction_engine.llm_client = PersonalSymptomLLM()
        response = client.post(
            "/api/chat/stream",
            json={"user_id": "test-user-7", "session_id": "sess-stream-7", "message": "I have jaundice for 3 days and my eyes are yellow."}
        )
        assert response.status_code == 200
        events = parse_sse_events(response.text)
        complete_events = [data for ev_type, data in events if ev_type == "complete"]
        assert len(complete_events) == 1
        symptoms = complete_events[0]["structured_health_information"]["symptoms"]
        assert len(symptoms) == 1
        assert symptoms[0]["name"] == "jaundice"
    finally:
        extraction_engine.llm_client = original_client


# 8. Long message is not truncated
def test_long_message_not_truncated():
    captured_messages = []

    class LongCaptureLLM(BaseLLMClient):
        def generate_chat_response(self, messages, system_prompt=None, temperature=0.2, response_format=None):
            captured_messages.extend(messages)
            return json.dumps({
                "assistant_message": "I have received your detailed health story.",
                "intent": "SYMPTOM_ASSESSMENT",
                "extracted_info": {"symptoms": [{"name": "fatigue"}]}
            })

        def stream_chat_response(self, messages, system_prompt=None, temperature=0.2, response_format=None):
            captured_messages.extend(messages)
            yield json.dumps({
                "assistant_message": "I have received your detailed health story.",
                "intent": "SYMPTOM_ASSESSMENT",
                "extracted_info": {"symptoms": [{"name": "fatigue"}]}
            })

    long_story = (
        "I am writing to explain everything that has happened to me over the past few weeks. "
        "About three weeks ago, I started feeling unusually exhausted after normal workdays. "
        "Initially I thought it was just lack of sleep or work-related stress, but even after getting 8 to 9 hours of sleep, "
        "the profound fatigue remained unchanged. A week later, I developed a dull ache on the right side of my abdomen, "
        "just underneath my ribs. The ache comes and goes, but seems slightly more noticeable after fatty meals. "
        "Additionally, my family noticed that the whites of my eyes seemed slightly yellow, and my urine has appeared "
        "dark amber like strong tea. I have not had a high fever, but I feel slightly feverish in the evenings. "
        "I do not have any known chronic medical conditions, and I currently do not take any prescribed medications, "
        "only an occasional paracetamol. I would like guidance on what might be causing these symptoms and what tests I should discuss with a doctor."
    )
    assert len(long_story.split()) >= 150

    original_client = extraction_engine.llm_client
    try:
        extraction_engine.llm_client = LongCaptureLLM()
        response = client.post(
            "/api/chat/stream",
            json={"user_id": "test-user-8", "session_id": "sess-stream-8", "message": long_story}
        )
        assert response.status_code == 200
        # Verify the entire user prompt was forwarded without arbitrary truncation
        prompt_content = captured_messages[0]["content"]
        assert long_story in prompt_content
    finally:
        extraction_engine.llm_client = original_client


# 9. Liver assessment authority remains unchanged
def test_liver_assessment_authority_remains_unchanged():
    # User provides structured lab data -> fast-path triggers without LLM
    response = client.post(
        "/api/chat/stream",
        json={"user_id": "test-user-9", "session_id": "sess-stream-9", "message": "Total Bilirubin 2.5, Direct Bilirubin 1.1, Alkphos 290, SGPT 55, SGOT 60, Total Proteins 6.8, Albumin 3.5, A/G Ratio 1.0, Age 45, Gender Male"}
    )
    assert response.status_code == 200
    events = parse_sse_events(response.text)
    complete_events = [data for ev_type, data in events if ev_type == "complete"]
    assert len(complete_events) == 1
    data = complete_events[0]
    assert data["intent"] == IntentType.DISEASE_RISK_ASSESSMENT.value
    assert data["assessment_handoff"] is not None


# 10. Patient context authority remains unchanged
def test_patient_context_authority_remains_unchanged():
    with patch("app.main.get_backend_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.get_patient_health_context.return_value = {
            "patient_id": 10,
            "has_history": True,
            "trends": [{"metric_type": "blood_glucose", "direction": "RISING", "summary": "Rising trend"}],
            "baselines": [],
            "recent_observations": []
        }
        mock_get_client.return_value = mock_client

        response = client.post(
            "/api/chat/stream",
            json={"user_id": "10", "session_id": "sess-stream-10", "message": "How have my glucose readings changed over time?"}
        )
        assert response.status_code == 200
        events = parse_sse_events(response.text)
        complete_events = [data for ev_type, data in events if ev_type == "complete"]
        assert len(complete_events) == 1
        assert complete_events[0]["intent"] == IntentType.MONITORING_QUERY.value
        assert "Rising trend" in complete_events[0]["assistant_message"] or "glucose" in complete_events[0]["assistant_message"].lower()


# 11. Stream failure does not fabricate a response
def test_stream_failure_does_not_fabricate_response():
    class FailingLLM(BaseLLMClient):
        def generate_chat_response(self, messages, system_prompt=None, temperature=0.2, response_format=None):
            raise LLMUnavailableError("Ollama connection timed out")

        def stream_chat_response(self, messages, system_prompt=None, temperature=0.2, response_format=None):
            raise LLMUnavailableError("Ollama connection timed out")

    original_client = extraction_engine.llm_client
    try:
        extraction_engine.llm_client = FailingLLM()
        response = client.post(
            "/api/chat/stream",
            json={"user_id": "test-user-11", "session_id": "sess-stream-11", "message": "I feel unwell today"}
        )
        assert response.status_code == 200
        events = parse_sse_events(response.text)
        complete_events = [data for ev_type, data in events if ev_type == "complete"]
        assert len(complete_events) == 1
        # Controlled graceful error message, NOT fabricated diagnosis
        msg = complete_events[0]["assistant_message"]
        assert "temporarily unavailable" in msg
    finally:
        extraction_engine.llm_client = original_client


# 12. No second LLM call is introduced
def test_no_second_llm_call_introduced():
    call_count = {"count": 0}

    class SingleCallLLM(BaseLLMClient):
        def generate_chat_response(self, messages, system_prompt=None, temperature=0.2, response_format=None):
            call_count["count"] += 1
            return json.dumps({
                "assistant_message": "I have processed your health message.",
                "intent": "SYMPTOM_ASSESSMENT",
                "extracted_info": {"symptoms": []}
            })

        def stream_chat_response(self, messages, system_prompt=None, temperature=0.2, response_format=None):
            call_count["count"] += 1
            yield json.dumps({
                "assistant_message": "I have processed your health message.",
                "intent": "SYMPTOM_ASSESSMENT",
                "extracted_info": {"symptoms": []}
            })

    original_client = extraction_engine.llm_client
    try:
        extraction_engine.llm_client = SingleCallLLM()
        response = client.post(
            "/api/chat/stream",
            json={"user_id": "test-user-12", "session_id": "sess-stream-12", "message": "I feel nauseous today"}
        )
        assert response.status_code == 200
        assert call_count["count"] == 1  # Exactly ONE call, no 2nd or 3rd call
    finally:
        extraction_engine.llm_client = original_client
