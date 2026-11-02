"""
Tests for Phase 9 Step 3.7 Runtime Issues & Regressions:
1. Positive facts persist across turns.
2. Negative facts are added without deleting previous positive facts.
3. Explicit "I haven't noticed any yellowing of my eyes or skin" extracts jaundice/yellowing as NEGATIVE.
4. Explicit "My urine looks normal" extracts normal urine as NEGATIVE.
5. Negative findings are never converted into positive symptoms.
6. Unmentioned symptoms remain UNKNOWN, not negative.
7. Current-turn extracted facts are returned in ChatResponse.
8. Previous-turn entities are not incorrectly reused as current-turn entities.
9. Ollama timeout/failure still produces deterministic grounded fallback.
10. Fallback does not fabricate diagnosis, probability, tests, or medical history.
11. The fallback uses the current turn's negative findings.
12. Existing safety/negation tests continue passing.
"""

from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient

from app.main import app, extraction_engine, session_manager
from app.llm_client import BaseLLMClient, MockLLMClient, LLMUnavailableError
from app.conversation import ConversationState
from app.schemas import SafetyStatus, NextActionType

client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_state():
    session_manager._sessions.clear()
    extraction_engine.llm_client = MockLLMClient()
    yield
    session_manager._sessions.clear()
    extraction_engine.llm_client = MockLLMClient()


# ==============================================================================
# 1. Positive facts persist across turns
# ==============================================================================
def test_positive_facts_persist_across_turns():
    session_id = "sess-reg-1"
    user_id = "u-reg-1"

    # Turn 1: Seed positive symptoms
    resp1 = client.post("/api/chat", json={
        "user_id": user_id, "session_id": session_id,
        "message": "I have nausea, vomiting, and dull upper-right abdominal pain."
    })
    assert resp1.status_code == 200
    sess = session_manager.get_or_create(session_id, user_id)
    sym_names1 = [s.name.lower() for s in sess.health_info.symptoms]
    assert "nausea" in sym_names1
    assert "vomiting" in sym_names1
    assert "pain" in sym_names1

    # Turn 2: Provide duration without repeating symptoms
    resp2 = client.post("/api/chat", json={
        "user_id": user_id, "session_id": session_id,
        "message": "The discomfort started two days ago."
    })
    assert resp2.status_code == 200
    sess = session_manager.get_or_create(session_id, user_id)
    sym_names2 = [s.name.lower() for s in sess.health_info.symptoms]
    assert "nausea" in sym_names2
    assert "vomiting" in sym_names2
    assert "pain" in sym_names2
    assert sess.health_info.duration is not None


# ==============================================================================
# 2. Negative facts are added without deleting previous positive facts
# ==============================================================================
def test_negative_facts_added_without_deleting_previous_positive_facts():
    sess = ConversationState(session_id="sess-reg-2", user_id="u-reg-2")

    # Turn 1
    info1 = extraction_engine.extract_health_information("I have nausea and vomiting.")
    sess.merge_health_info(info1)
    assert len(sess.health_info.symptoms) == 2
    assert len(sess.health_info.negative_symptoms) == 0

    # Turn 2
    info2 = extraction_engine.extract_health_information(
        "I haven't noticed any yellowing of my eyes or skin, and my urine looks normal."
    )
    sess.merge_health_info(info2)

    # Positive symptoms persist
    sym_names = [s.name.lower() for s in sess.health_info.symptoms]
    assert "nausea" in sym_names
    assert "vomiting" in sym_names

    # Negative findings accumulated
    neg_lower = [n.lower() for n in sess.health_info.negative_symptoms]
    assert any("jaundice" in n or "yellow" in n for n in neg_lower)
    assert any("urine" in n for n in neg_lower)


# ==============================================================================
# 3. Explicit "I haven't noticed any yellowing..." extracts jaundice as NEGATIVE
# ==============================================================================
def test_extract_negative_jaundice_yellowing():
    info = extraction_engine.extract_health_information(
        "I haven't noticed any yellowing of my eyes or skin."
    )
    neg_lower = [n.lower() for n in info.negative_symptoms]
    assert any("jaundice" in n or "yellow" in n for n in neg_lower)
    assert len(info.symptoms) == 0


# ==============================================================================
# 4. Explicit "My urine looks normal" extracts normal urine as NEGATIVE
# ==============================================================================
def test_extract_negative_normal_urine():
    info = extraction_engine.extract_health_information("My urine looks normal.")
    neg_lower = [n.lower() for n in info.negative_symptoms]
    assert any("urine" in n for n in neg_lower)
    assert len(info.symptoms) == 0


# ==============================================================================
# 5. Negative findings are never converted into positive symptoms
# ==============================================================================
def test_negative_findings_never_converted_to_positive():
    info = extraction_engine.extract_health_information(
        "No fever, no vomiting, no yellowing of eyes, no dark urine, no abdominal pain, and no chest pain."
    )
    assert len(info.symptoms) == 0
    neg_lower = [n.lower() for n in info.negative_symptoms]
    assert any("fever" in n for n in neg_lower)
    assert any("vomit" in n for n in neg_lower)
    assert any("yellow" in n or "jaundice" in n for n in neg_lower)
    assert any("urine" in n for n in neg_lower)
    assert any("abdominal" in n for n in neg_lower)
    assert any("chest" in n for n in neg_lower)


# ==============================================================================
# 6. Unmentioned symptoms remain UNKNOWN, not negative
# ==============================================================================
def test_unmentioned_symptoms_remain_unknown():
    info = extraction_engine.extract_health_information(
        "I have dull pain in my right upper abdomen."
    )
    assert len(info.negative_symptoms) == 0
    neg_lower = [n.lower() for n in info.negative_symptoms]
    assert "fever" not in neg_lower
    assert "jaundice" not in neg_lower
    assert "vomiting" not in neg_lower
    assert "nausea" not in neg_lower


# ==============================================================================
# 7. Current-turn extracted facts are returned in ChatResponse
# ==============================================================================
def test_current_turn_extracted_facts_returned_in_chat_response():
    session_id = "sess-reg-7"
    user_id = "u-reg-7"

    # Turn 1
    client.post("/api/chat", json={
        "user_id": user_id, "session_id": session_id,
        "message": "I have nausea and vomiting."
    })

    # Turn 2
    resp2 = client.post("/api/chat", json={
        "user_id": user_id, "session_id": session_id,
        "message": "I haven't noticed any yellowing of my eyes or skin, and my urine looks normal."
    })
    assert resp2.status_code == 200
    data2 = resp2.json()

    # Current turn extraction must contain Turn 2's negative findings and NO positive symptoms
    assert "current_turn_extraction" in data2
    curr = data2["current_turn_extraction"]
    assert len(curr["symptoms"]) == 0
    assert any("jaundice" in n.lower() or "yellow" in n.lower() for n in curr["negative_symptoms"])
    assert any("urine" in n.lower() for n in curr["negative_symptoms"])

    # Cumulative state still contains Turn 1's symptoms
    cum = data2["structured_health_information"]
    cum_syms = [s["name"] for s in cum["symptoms"]]
    assert "nausea" in cum_syms
    assert "vomiting" in cum_syms


# ==============================================================================
# 8. Previous-turn entities are not incorrectly reused as current-turn entities
# ==============================================================================
def test_previous_turn_entities_not_reused_as_current_turn_entities():
    session_id = "sess-reg-8"
    user_id = "u-reg-8"

    # Turn 1: Symptoms reported
    client.post("/api/chat", json={
        "user_id": user_id, "session_id": session_id,
        "message": "I have nausea and vomiting."
    })

    # Turn 2: Non-symptom message
    resp2 = client.post("/api/chat", json={
        "user_id": user_id, "session_id": session_id,
        "message": "Thank you for the explanation."
    })
    assert resp2.status_code == 200
    data2 = resp2.json()

    # Turn 2 current_turn_extraction must NOT duplicate Turn 1's symptoms
    curr = data2["current_turn_extraction"]
    assert curr is None or (len(curr["symptoms"]) == 0 and len(curr["negative_symptoms"]) == 0)


# ==============================================================================
# 9. Ollama timeout/failure still produces deterministic grounded fallback
# ==============================================================================
def test_ollama_failure_produces_deterministic_grounded_fallback():
    session_id = "sess-reg-9"
    user_id = "u-reg-9"

    failing_llm = MagicMock(spec=BaseLLMClient)
    failing_llm.generate_chat_response.side_effect = LLMUnavailableError("Ollama connection timed out")
    extraction_engine.llm_client = failing_llm

    resp = client.post("/api/chat", json={
        "user_id": user_id, "session_id": session_id,
        "message": (
            "I have dull pain in my right upper abdomen for two days after oily food. "
            "What could cause this and what tests would help a doctor?"
        )
    })
    assert resp.status_code == 200
    data = resp.json()
    msg = data["assistant_message"].lower()

    # Grounded fallback returned, NOT a bare unhelpful error
    assert "hepatobiliary" in msg or "gallbladder" in msg
    assert "ultrasound" in msg
    assert msg != "the conversational ai service is temporarily unavailable."


# ==============================================================================
# 10. Fallback does not fabricate diagnosis, probability, tests, or history
# ==============================================================================
def test_fallback_does_not_fabricate_diagnosis_or_probability():
    session_id = "sess-reg-10"
    user_id = "u-reg-10"

    failing_llm = MagicMock(spec=BaseLLMClient)
    failing_llm.generate_chat_response.side_effect = LLMUnavailableError("Ollama 503 Service Unavailable")
    extraction_engine.llm_client = failing_llm

    resp = client.post("/api/chat", json={
        "user_id": user_id, "session_id": session_id,
        "message": "I haven't noticed any yellowing of my eyes or skin, and my urine looks normal. Does that information change what a doctor might consider?"
    })
    assert resp.status_code == 200
    data = resp.json()
    msg = data["assistant_message"].lower()

    # No fabricated diagnosis or probability
    assert "you have" not in msg
    assert "%" not in msg
    assert "probability" not in msg
    info = data["structured_health_information"]
    assert len(info["known_conditions"]) == 0
    assert len(info["previous_tests"]) == 0


# ==============================================================================
# 11. The fallback uses the current turn's negative findings
# ==============================================================================
def test_fallback_uses_current_turn_negative_findings():
    session_id = "sess-reg-11"
    user_id = "u-reg-11"

    # Turn 1
    client.post("/api/chat", json={
        "user_id": user_id, "session_id": session_id,
        "message": "I have dull pain in my right upper abdomen for two days after oily meals."
    })

    # Turn 2 with failing LLM
    failing_llm = MagicMock(spec=BaseLLMClient)
    failing_llm.generate_chat_response.side_effect = LLMUnavailableError("Ollama connection timed out")
    extraction_engine.llm_client = failing_llm

    resp = client.post("/api/chat", json={
        "user_id": user_id, "session_id": session_id,
        "message": "I haven't noticed any yellowing of my eyes or skin, and my urine looks normal. Does that information change what a doctor might consider?"
    })
    assert resp.status_code == 200
    msg = resp.json()["assistant_message"].lower()

    # Acknowledges negative findings
    assert "jaundice" in msg or "yellow" in msg
    assert "urine" in msg
    assert "negative" in msg or "absence" in msg or "denied" in msg
    # Explains clinical context
    assert "rule" in msg or "distinguish" in msg or "context" in msg or "pattern" in msg


# ==============================================================================
# 12. Existing safety and negation tests continue passing
# ==============================================================================
def test_existing_safety_negation_intact():
    # Emergency triggers safety
    resp_emerg = client.post("/api/chat", json={
        "user_id": "u-safe-1", "session_id": "s-safe-1",
        "message": "I have severe crushing chest pain and severe difficulty breathing!"
    })
    assert resp_emerg.status_code == 200
    assert resp_emerg.json()["safety_status"] == SafetyStatus.URGENT_MEDICAL_EVALUATION

    # Negated emergency does NOT trigger safety
    resp_neg = client.post("/api/chat", json={
        "user_id": "u-safe-2", "session_id": "s-safe-2",
        "message": "I have a headache, but I do not have severe chest pain or difficulty breathing."
    })
    assert resp_neg.status_code == 200
    assert resp_neg.json()["safety_status"] == SafetyStatus.NORMAL_CONVERSATION
