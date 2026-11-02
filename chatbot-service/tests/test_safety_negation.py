import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app.main import app, extraction_engine
from app.schemas import SafetyStatus, NextActionType
from app.safety import SafetyChecker
from app.llm_client import BaseLLMClient

client = TestClient(app)
safety_checker = SafetyChecker()


# ==========================================
# 1. POSITIVE EMERGENCY TESTS
# ==========================================

@pytest.mark.parametrize("message", [
    "I have severe chest pain.",
    "I am experiencing severe chest pain.",
    "I suddenly developed severe chest pain.",
    "I have severe difficulty breathing.",
    "I cannot breathe.",
    "Crushing chest pain radiating to left arm."
])
def test_positive_emergency_symptoms_trigger_safety(message):
    status, triggers, emergency_msg = safety_checker.evaluate(message)
    assert status == SafetyStatus.URGENT_MEDICAL_EVALUATION
    assert len(triggers) > 0
    assert emergency_msg is not None

    # Verify end-to-end API response
    response = client.post(
        "/api/chat",
        json={"user_id": "test-user-pos", "session_id": "test-sess-pos", "message": message}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["safety_status"] == SafetyStatus.URGENT_MEDICAL_EVALUATION.value
    assert data["next_action"] == NextActionType.EMERGENCY_ROUTING.value


# ==========================================
# 2. NEGATED EMERGENCY TESTS
# ==========================================

@pytest.mark.parametrize("message", [
    "I do not have severe chest pain.",
    "I don't have severe chest pain.",
    "I have not noticed severe chest pain.",
    "I have not experienced chest pain.",
    "No chest pain.",
    "I have no difficulty breathing.",
    "I am not having difficulty breathing.",
    "I have not noticed severe chest pain or difficulty breathing.",
    "I have no chest pain and no difficulty breathing.",
    "I have neither severe chest pain nor difficulty breathing.",
    "Patient denies severe chest pain.",
    "Negative for severe chest pain.",
    "I am free of severe chest pain.",
    "Chest pain: none",
    "Chest pain: negative",
    "Chest pain is absent"
])
def test_negated_emergency_symptoms_do_not_trigger_safety(message):
    status, triggers, emergency_msg = safety_checker.evaluate(message)
    assert status == SafetyStatus.NORMAL_CONVERSATION
    assert len(triggers) == 0
    assert emergency_msg is None

    # Verify end-to-end API response does NOT route to emergency
    response = client.post(
        "/api/chat",
        json={"user_id": "test-user-neg", "session_id": f"test-sess-neg-{abs(hash(message))}", "message": message}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["safety_status"] != SafetyStatus.URGENT_MEDICAL_EVALUATION.value
    assert data["next_action"] != NextActionType.EMERGENCY_ROUTING.value


# ==========================================
# 3. MIXED POSITIVE & NEGATIVE TESTS
# ==========================================

def test_mixed_no_chest_pain_but_severe_difficulty_breathing():
    """
    Negation in the first clause must NOT cancel the positive emergency symptom in the second clause.
    """
    msg = "I have no chest pain, but I suddenly developed severe difficulty breathing."
    status, triggers, emergency_msg = safety_checker.evaluate(msg)
    assert status == SafetyStatus.URGENT_MEDICAL_EVALUATION
    assert any("respiratory" in t.lower() for t in triggers)


def test_mixed_no_difficulty_breathing_but_severe_crushing_chest_pain():
    """
    Negation of respiratory distress must NOT cancel crushing chest pain.
    """
    msg = "I do not have difficulty breathing, but I now have severe crushing chest pain."
    status, triggers, emergency_msg = safety_checker.evaluate(msg)
    assert status == SafetyStatus.URGENT_MEDICAL_EVALUATION
    assert any("coronary" in t.lower() or "chest" in t.lower() for t in triggers)


def test_mixed_severe_chest_pain_but_no_shortness_of_breath():
    """
    Positive chest pain must trigger emergency even if shortness of breath is negated.
    """
    msg = "I have severe chest pain but no shortness of breath."
    status, triggers, emergency_msg = safety_checker.evaluate(msg)
    assert status == SafetyStatus.URGENT_MEDICAL_EVALUATION
    assert any("coronary" in t.lower() or "chest" in t.lower() for t in triggers)


def test_mixed_no_chest_pain_and_no_difficulty_breathing():
    """
    Both symptoms are explicitly negated -> must NOT trigger emergency.
    """
    msg = "I have no chest pain and no difficulty breathing."
    status, triggers, emergency_msg = safety_checker.evaluate(msg)
    assert status == SafetyStatus.NORMAL_CONVERSATION
    assert len(triggers) == 0


# ==========================================
# 4. LONG MESSAGE RUNTIME REGRESSION TEST
# ==========================================

def test_long_message_with_negated_red_flags_passes_safety_gate():
    """
    Reproduces the exact 300-500 word user narrative that failed in production:
    'I have not noticed severe chest pain or difficulty breathing.'
    Must pass safety evaluation, proceed to normal conversation, and reach the LLM.
    """
    long_narrative = (
        "I am writing to explain everything that has happened to me over the past few weeks. "
        "About three weeks ago, I started feeling unusually exhausted after normal workdays. "
        "Initially I thought it was just lack of sleep or work-related stress, but even after getting 8 to 9 hours of sleep, "
        "the profound fatigue remained unchanged. A week later, I developed a dull ache on the right side of my abdomen, "
        "just underneath my ribs. The ache comes and goes, but seems slightly more noticeable after fatty meals. "
        "Additionally, my family noticed that the whites of my eyes seemed slightly yellow, and my urine has appeared "
        "dark amber like strong tea. I have not had a high fever, but I feel slightly feverish in the evenings. "
        "I have not noticed severe chest pain or difficulty breathing. "
        "I do not have any known chronic medical conditions, and I currently do not take any prescribed medications, "
        "only an occasional paracetamol. I would like guidance on what might be causing these symptoms and what tests I should discuss with a doctor."
    )

    # 1. Deterministic safety evaluation check
    status, triggers, emergency_msg = safety_checker.evaluate(long_narrative)
    assert status == SafetyStatus.NORMAL_CONVERSATION
    assert len(triggers) == 0
    assert emergency_msg is None

    # 2. End-to-end API check (non-streaming)
    response = client.post(
        "/api/chat",
        json={"user_id": "long-user-1", "session_id": "long-sess-1", "message": long_narrative}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["safety_status"] == SafetyStatus.NORMAL_CONVERSATION.value
    assert data["next_action"] != NextActionType.EMERGENCY_ROUTING.value
    assert "911" not in data["assistant_message"]

    # 3. End-to-end streaming check
    stream_response = client.post(
        "/api/chat/stream",
        json={"user_id": "long-user-1", "session_id": "long-sess-stream-1", "message": long_narrative}
    )
    assert stream_response.status_code == 200
    # Must NOT contain emergency warning in SSE text
    assert "EMERGENCY WARNING" not in stream_response.text


# ==========================================
# 5. SAFETY PRIORITY BEFORE LLM TEST
# ==========================================

def test_genuine_emergency_bypasses_llm_invocation():
    """
    Proves that a genuinely positive emergency message triggers emergency routing
    BEFORE the LLM is invoked, ensuring LLM is never called to downgrade emergency acuity.
    """
    call_tracker = {"llm_called": False}

    class SpyingLLMClient(BaseLLMClient):
        def generate_chat_response(self, messages, system_prompt=None, temperature=0.2, response_format=None):
            call_tracker["llm_called"] = True
            return "Should never be reached"

        def stream_chat_response(self, messages, system_prompt=None, temperature=0.2, response_format=None):
            call_tracker["llm_called"] = True
            yield "Should never be reached"

    original_client = extraction_engine.llm_client
    try:
        extraction_engine.llm_client = SpyingLLMClient()
        response = client.post(
            "/api/chat",
            json={"user_id": "emer-user", "session_id": "emer-sess", "message": "I suddenly developed crushing chest pain radiating to my left arm."}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["safety_status"] == SafetyStatus.URGENT_MEDICAL_EVALUATION.value
        assert data["next_action"] == NextActionType.EMERGENCY_ROUTING.value
        assert call_tracker["llm_called"] is False
    finally:
        extraction_engine.llm_client = original_client
