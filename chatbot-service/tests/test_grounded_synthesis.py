"""
Tests for Phase 9 Step 3.7: Grounded LLM Synthesis for Natural Clinical Conversation.

Verifies:
1. LLM prompt receives the original user message.
2. LLM prompt receives verified structured facts.
3. LLM prompt receives only relevant clinical grounding (no unrelated domains).
4. Negative findings filter out contradicted components and are explicitly passed in prompt.
5. Unmentioned symptoms are NOT treated as negative findings.
6. Grounding is explicitly labeled as educational background, NOT patient diagnoses.
7. Mock LLM natural response is returned through ChatResponse (not hardcoded paragraph).
8. Mock LLM failure returns a safe deterministic fallback without fabricating medical facts.
9. Emergency red flags bypass grounded synthesis; LLM is not called.
10. Diabetes grounding provided for thirst and frequent urination.
11. Parkinson's neurological grounding provided for tremor, stiffness, bradykinesia.
12. Cardiovascular grounding provided for exertional chest discomfort.
13. Streaming Answer-First emits token and complete events with progressive tokens.
14. Follow-up question is limited to at most one approved by FollowUpEngine.
15. Patient context queries remain authoritative on their own path.
16. Acceptance Test: Message A vs Message B produce different grounded LLM prompts and components (SAME DOMAIN != SAME ANSWER).
"""

import json
from unittest.mock import MagicMock, patch, call
import pytest
from fastapi.testclient import TestClient

from app.main import app, extraction_engine, session_manager
from app.llm_client import BaseLLMClient, MockLLMClient
from app.clinical_knowledge import clinical_knowledge_engine
from app.grounded_synthesis import (
    grounded_synthesizer,
    build_grounding_prompt,
    format_verified_facts,
    format_clinical_grounding,
    SYSTEM_GROUNDED_SYNTHESIS_PROMPT
)
from app.schemas import StructuredHealthInformation, Symptom

client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_state():
    session_manager._sessions.clear()
    extraction_engine.llm_client = MockLLMClient()
    yield
    session_manager._sessions.clear()
    extraction_engine.llm_client = MockLLMClient()


# ==============================================================================
# TEST 1 — LLM receives actual user message
# ==============================================================================
def test_grounded_synthesis_llm_receives_actual_user_message():
    session_id = "sess-gtest-1"
    user_id = "user-gt-1"
    raw_message = (
        "I've been feeling weak and tired for the last two days. I have reduced appetite, "
        "nausea, vomiting twice, a feverish feeling, and dull pain in the upper-right side of my abdomen. "
        "The pain gets worse after heavy or oily meals. What could potentially cause these symptoms and what tests help?"
    )

    mock_llm = MagicMock(spec=BaseLLMClient)
    mock_llm.generate_chat_response.return_value = (
        "Based on your description, the combination of upper-right abdominal pain and digestive symptoms "
        "warrants clinical evaluation. A physician may assess hepatobiliary or gastrointestinal possibilities."
    )
    extraction_engine.llm_client = mock_llm

    resp = client.post("/api/chat", json={
        "user_id": user_id, "session_id": session_id, "message": raw_message
    })
    assert resp.status_code == 200
    assert mock_llm.generate_chat_response.called

    # Inspect call arguments
    call_args = mock_llm.generate_chat_response.call_args
    prompt_sent = call_args[1]["messages"][0]["content"]

    # Verify original user message is explicitly included in the prompt
    assert 'USER MESSAGE:\n"' + raw_message + '"' in prompt_sent


# ==============================================================================
# TEST 2 — LLM receives verified facts
# ==============================================================================
def test_grounded_synthesis_llm_receives_verified_facts():
    session_id = "sess-gtest-2"
    user_id = "user-gt-2"
    message = (
        "I have dull pain in my right upper abdomen for 2 days after oily food. "
        "What could cause this and what tests would help a doctor?"
    )

    mock_llm = MagicMock(spec=BaseLLMClient)
    mock_llm.generate_chat_response.return_value = (
        "A clinician might evaluate the right upper abdominal discomfort you noted. "
        "Common evaluations include an abdominal ultrasound."
    )
    extraction_engine.llm_client = mock_llm

    resp = client.post("/api/chat", json={
        "user_id": user_id, "session_id": session_id, "message": message
    })
    assert resp.status_code == 200

    prompt_sent = mock_llm.generate_chat_response.call_args[1]["messages"][0]["content"]

    # Verify structured facts section exists and contains extracted facts
    assert "VERIFIED USER FACTS" in prompt_sent
    assert "pain" in prompt_sent
    assert "right upper abdomen" in prompt_sent
    assert "two days" in prompt_sent or "2 days" in prompt_sent or "dull" in prompt_sent
    assert "oily" in prompt_sent or "heavy meals" in prompt_sent


# ==============================================================================
# TEST 3 — LLM receives only relevant clinical grounding
# ==============================================================================
def test_grounded_synthesis_llm_receives_only_relevant_grounding():
    session_id = "sess-gtest-3"
    user_id = "user-gt-3"
    message = "I have upper-right abdominal pain after oily meals. What could cause this and what tests might help?"

    mock_llm = MagicMock(spec=BaseLLMClient)
    mock_llm.generate_chat_response.return_value = (
        "Discomfort in the upper-right quadrant following rich meals may point to biliary tract irritation. "
        "An abdominal ultrasound is often recommended by physicians."
    )
    extraction_engine.llm_client = mock_llm

    resp = client.post("/api/chat", json={
        "user_id": user_id, "session_id": session_id, "message": message
    })
    assert resp.status_code == 200

    prompt_sent = mock_llm.generate_chat_response.call_args[1]["messages"][0]["content"]

    # Verify RUQ / hepatobiliary grounding is present
    assert "Hepatobiliary" in prompt_sent or "biliary" in prompt_sent
    assert "ultrasound" in prompt_sent

    # Verify unrelated domains are NOT present in grounding
    assert "Parkinson" not in prompt_sent
    assert "substantia nigra" not in prompt_sent
    assert "HbA1c" not in prompt_sent
    assert "troponin" not in prompt_sent


# ==============================================================================
# TEST 4 — Negative findings filter components & appear in prompt
# ==============================================================================
def test_grounded_synthesis_negative_findings_preserved_and_components_filtered():
    session_id = "sess-gtest-4"
    user_id = "user-gt-4"
    message = (
        "I have upper-right abdominal pain. I do not have nausea, vomiting, fever, or jaundice. "
        "What kinds of causes might a doctor consider?"
    )

    mock_llm = MagicMock(spec=BaseLLMClient)
    mock_llm.generate_chat_response.return_value = (
        "Noting the absence of nausea, fever, and jaundice is helpful. "
        "A physician may still explore localized biliary or gastrointestinal causes."
    )
    extraction_engine.llm_client = mock_llm

    resp = client.post("/api/chat", json={
        "user_id": user_id, "session_id": session_id, "message": message
    })
    assert resp.status_code == 200

    prompt_sent = mock_llm.generate_chat_response.call_args[1]["messages"][0]["content"]

    # Verify negative findings are clearly communicated in the prompt
    assert "Explicitly Denied Symptoms / Negative Findings" in prompt_sent
    assert "nausea" in prompt_sent
    assert "fever" in prompt_sent

    # Verify nausea/vomiting-specific knowledge component was excluded
    assert "Upper gastrointestinal involvement" not in prompt_sent


# ==============================================================================
# TEST 5 — Not mentioned vs negative
# ==============================================================================
def test_grounded_synthesis_not_mentioned_vs_negative():
    session_id = "sess-gtest-5"
    user_id = "user-gt-5"
    message = "I have upper-right abdominal pain. What could cause this and what tests would help a doctor?"

    mock_llm = MagicMock(spec=BaseLLMClient)
    mock_llm.generate_chat_response.return_value = "A doctor might assess upper abdominal pain."
    extraction_engine.llm_client = mock_llm

    resp = client.post("/api/chat", json={
        "user_id": user_id, "session_id": session_id, "message": message
    })
    assert resp.status_code == 200

    prompt_sent = mock_llm.generate_chat_response.call_args[1]["messages"][0]["content"]

    # The user did NOT deny nausea or fever; they were simply not mentioned
    assert "Explicitly Denied Symptoms / Negative Findings: None reported" in prompt_sent


# ==============================================================================
# TEST 6 — Grounding is educational, NOT patient diagnosis
# ==============================================================================
def test_grounded_synthesis_prompt_labels_grounding_not_diagnosis():
    session_id = "sess-gtest-6"
    user_id = "user-gt-6"
    message = "What causes pain in the right upper abdomen and what tests are done?"

    mock_llm = MagicMock(spec=BaseLLMClient)
    mock_llm.generate_chat_response.return_value = "Educational discussion of possibilities."
    extraction_engine.llm_client = mock_llm

    resp = client.post("/api/chat", json={
        "user_id": user_id, "session_id": session_id, "message": message
    })
    assert resp.status_code == 200

    prompt_sent = mock_llm.generate_chat_response.call_args[1]["messages"][0]["content"]
    system_prompt = mock_llm.generate_chat_response.call_args[1]["system_prompt"]

    # Explicit educational / non-diagnostic labeling
    assert "CLINICAL GROUNDING (Educational background for clinicians, NOT diagnoses)" in prompt_sent
    assert "Educational Categories to Discuss (NOT diagnoses)" in prompt_sent
    assert "You are NOT a diagnostic system" in system_prompt
    assert "Do NOT turn knowledge possibilities into a definitive patient diagnosis" in system_prompt


# ==============================================================================
# TEST 7 & STEP 24 — LLM natural output is returned through ChatResponse
# ==============================================================================
def test_grounded_synthesis_natural_response_returned():
    session_id = "sess-gtest-7"
    user_id = "user-gt-7"
    message = (
        "I have been feeling weak and tired for two days, with nausea, vomiting twice, and dull pain "
        "in the upper-right abdomen worse after oily meals. What could potentially cause these symptoms "
        "and what tests would generally help a doctor understand what is happening?"
    )

    natural_mock_response = (
        "From what you've described, the combination of upper-right abdominal discomfort and vomiting deserves "
        "a clinician's assessment. The fact that your symptoms become worse after heavier meals is useful context, "
        "and a clinician may consider whether the pattern points toward a biliary or gastrointestinal source. "
        "They may decide whether blood tests or abdominal imaging are appropriate."
    )

    mock_llm = MagicMock(spec=BaseLLMClient)
    mock_llm.generate_chat_response.return_value = natural_mock_response
    extraction_engine.llm_client = mock_llm

    resp = client.post("/api/chat", json={
        "user_id": user_id, "session_id": session_id, "message": message
    })
    assert resp.status_code == 200
    data = resp.json()

    # The exact natural synthesized response is returned
    assert data["assistant_message"] == natural_mock_response
    # Verify it does NOT return the old hardcoded raw paragraph
    assert "While symptoms alone cannot determine a diagnosis" not in data["assistant_message"]


# ==============================================================================
# TEST 8 — LLM failure returns safe fallback without fabricating medical answer
# ==============================================================================
def test_grounded_synthesis_llm_failure_safe_fallback():
    session_id = "sess-gtest-8"
    user_id = "user-gt-8"
    message = "I have upper-right abdominal pain and nausea. What could cause this and what tests help?"

    failing_llm = MagicMock(spec=BaseLLMClient)
    failing_llm.generate_chat_response.side_effect = RuntimeError("Ollama connection timed out!")
    extraction_engine.llm_client = failing_llm

    resp = client.post("/api/chat", json={
        "user_id": user_id, "session_id": session_id, "message": message
    })
    assert resp.status_code == 200
    data = resp.json()
    msg = data["assistant_message"].lower()

    # Safe deterministic knowledge fallback is returned
    assert "hepatobiliary" in msg or "gallbladder" in msg
    assert "ultrasound" in msg
    assert "symptoms alone cannot determine a diagnosis" in msg


# ==============================================================================
# TEST 9 — Emergency bypass: LLM must NOT be called
# ==============================================================================
def test_grounded_synthesis_emergency_bypass_llm_not_called():
    session_id = "sess-gtest-9"
    user_id = "user-gt-9"
    emergency_msg = "I have severe crushing chest pain and severe difficulty breathing! What causes this?"

    failing_llm = MagicMock(spec=BaseLLMClient)
    failing_llm.generate_chat_response.side_effect = AssertionError("LLM must NEVER be called on emergency!")
    extraction_engine.llm_client = failing_llm

    resp = client.post("/api/chat", json={
        "user_id": user_id, "session_id": session_id, "message": emergency_msg
    })
    assert resp.status_code == 200
    data = resp.json()

    # Emergency safety gate intercepted before grounded synthesis
    failing_llm.generate_chat_response.assert_not_called()
    assert data["safety_status"] == "URGENT_MEDICAL_EVALUATION"


# ==============================================================================
# TEST 10 — Diabetes / Glycemic domain grounding
# ==============================================================================
def test_grounded_synthesis_diabetes_domain():
    session_id = "sess-gtest-10"
    user_id = "user-gt-10"
    message = "I have had increased thirst and frequent urination for a week. What could cause this and what tests help?"

    mock_llm = MagicMock(spec=BaseLLMClient)
    mock_llm.generate_chat_response.return_value = "Evaluation for glycemic regulation is typically considered."
    extraction_engine.llm_client = mock_llm

    resp = client.post("/api/chat", json={
        "user_id": user_id, "session_id": session_id, "message": message
    })
    assert resp.status_code == 200

    prompt_sent = mock_llm.generate_chat_response.call_args[1]["messages"][0]["content"]

    # Verify metabolic/glycemic grounding
    assert "Endocrine / Metabolic" in prompt_sent or "Glycemic" in prompt_sent
    assert "Fasting Plasma Glucose" in prompt_sent or "HbA1c" in prompt_sent


# ==============================================================================
# TEST 11 — Parkinson's / Neurological domain grounding
# ==============================================================================
def test_grounded_synthesis_parkinsons_domain():
    session_id = "sess-gtest-11"
    user_id = "user-gt-11"
    message = "I have noticed a resting hand tremor, muscle stiffness, and slowness of movement. What could cause this?"

    mock_llm = MagicMock(spec=BaseLLMClient)
    mock_llm.generate_chat_response.return_value = "Neurological evaluation is appropriate for movement changes."
    extraction_engine.llm_client = mock_llm

    resp = client.post("/api/chat", json={
        "user_id": user_id, "session_id": session_id, "message": message
    })
    assert resp.status_code == 200

    prompt_sent = mock_llm.generate_chat_response.call_args[1]["messages"][0]["content"]

    # Verify neurological grounding
    assert "Neurological" in prompt_sent or "Movement" in prompt_sent
    assert "rigidity" in prompt_sent or "bradykinesia" in prompt_sent
    assert "neurologist" in prompt_sent.lower() or "updrs" in prompt_sent.lower()


# ==============================================================================
# TEST 12 — Cardiovascular domain grounding
# ==============================================================================
def test_grounded_synthesis_chest_cardiovascular_domain():
    session_id = "sess-gtest-12"
    user_id = "user-gt-12"
    message = "I have chest tightness during exertion. What could potentially cause this and what tests would a doctor do?"

    mock_llm = MagicMock(spec=BaseLLMClient)
    mock_llm.generate_chat_response.return_value = "Exertional chest discomfort warrants prompt medical evaluation."
    extraction_engine.llm_client = mock_llm

    resp = client.post("/api/chat", json={
        "user_id": user_id, "session_id": session_id, "message": message
    })
    assert resp.status_code == 200

    prompt_sent = mock_llm.generate_chat_response.call_args[1]["messages"][0]["content"]

    # Verify cardiovascular grounding
    assert "Cardiovascular" in prompt_sent
    assert "ECG" in prompt_sent or "troponin" in prompt_sent


# ==============================================================================
# TEST 13 — Streaming Answer-First uses LLM streaming
# ==============================================================================
def test_grounded_synthesis_streaming():
    session_id = "sess-gtest-13"
    user_id = "user-gt-13"
    message = "I have upper-right abdominal pain after oily meals. What kinds of causes might a doctor consider?"

    mock_llm = MagicMock(spec=BaseLLMClient)
    mock_llm.stream_chat_response.return_value = iter([
        "Upper right abdominal ",
        "discomfort following meals ",
        "may involve biliary irritation."
    ])
    extraction_engine.llm_client = mock_llm

    resp = client.post("/api/chat/stream", json={
        "user_id": user_id, "session_id": session_id, "message": message
    })
    assert resp.status_code == 200
    assert mock_llm.stream_chat_response.called

    body = resp.text
    assert "event: token" in body
    assert "event: complete" in body
    assert "Upper right abdominal" in body
    assert "biliary irritation" in body


# ==============================================================================
# TEST 14 — Follow-up: At most one approved follow-up question
# ==============================================================================
def test_grounded_synthesis_single_approved_follow_up_question():
    session_id = "sess-gtest-14"
    user_id = "user-gt-14"
    message = "I have had dull pain in my right upper abdomen for two days. What could cause this?"

    mock_llm = MagicMock(spec=BaseLLMClient)
    mock_llm.generate_chat_response.return_value = "A clinician may evaluate your abdominal pain."
    extraction_engine.llm_client = mock_llm

    resp = client.post("/api/chat", json={
        "user_id": user_id, "session_id": session_id, "message": message
    })
    assert resp.status_code == 200

    prompt_sent = mock_llm.generate_chat_response.call_args[1]["messages"][0]["content"]

    # Prompt contains approved single follow-up question
    assert "APPROVED FOLLOW-UP QUESTION" in prompt_sent
    # Verified follow-up field is singular
    data = resp.json()
    assert data["follow_up_question"] is None or isinstance(data["follow_up_question"], str)


# ==============================================================================
# TEST 15 — Patient context remains authoritative
# ==============================================================================
def test_grounded_synthesis_patient_context_remains_authoritative():
    session_id = "sess-gtest-15"
    user_id = "42"

    with patch("app.main.get_backend_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.get_patient_health_context.return_value = {
            "patient_id": 42,
            "has_history": True,
            "trends": [],
            "baselines": [],
            "recent_observations": []
        }
        mock_get_client.return_value = mock_client

        resp = client.post("/api/chat", json={
            "user_id": user_id,
            "session_id": session_id,
            "message": "How has my health been recently?"
        })
        assert resp.status_code == 200
        data = resp.json()

        # Patient context routing took place, not generic clinical knowledge
        sess = session_manager.get_or_create(session_id, user_id)
        assert sess.last_routing == "PATIENT_CONTEXT"
        assert mock_client.get_patient_health_context.called
        assert data["intent"] == "MONITORING_QUERY"


# ==============================================================================
# STEP 23 ACCEPTANCE TEST — SAME DOMAIN != SAME ANSWER
# ==============================================================================
def test_grounded_synthesis_message_a_vs_message_b_differ():
    """
    STEP 23: Message A (positive symptoms, oily meal trigger, nausea, vomiting)
    vs Message B (negative symptoms: no nausea, vomiting, fever, jaundice, no meal relation).
    Asserts:
    - Both select hepatobiliary_ruq domain.
    - Grounding differs between A and B.
    - Meal-related component present in A, absent in B.
    - Nausea/vomiting component present in A, absent in B.
    - Negative findings explicitly present in B's prompt.
    - Original messages passed to LLM are different.
    """
    msg_a = (
        "I've been feeling weak and tired for the last two days. I have reduced appetite, "
        "nausea, vomiting twice, a feverish feeling, and dull pain in the upper-right side of my abdomen. "
        "The pain gets worse after heavy or oily meals."
    )
    msg_b = (
        "I've had a dull pain in the upper-right side of my abdomen for two days. "
        "I don't have nausea, vomiting, fever, or jaundice, and the pain isn't related to meals."
    )

    mock_llm = MagicMock(spec=BaseLLMClient)
    mock_llm.generate_chat_response.side_effect = [
        "Synthesized response for Patient A",
        "Synthesized response for Patient B"
    ]
    extraction_engine.llm_client = mock_llm

    # Call A
    resp_a = client.post("/api/chat", json={
        "user_id": "user-acc-a", "session_id": "sess-acc-a",
        "message": msg_a + " What could potentially cause this and what tests would help?"
    })
    assert resp_a.status_code == 200

    # Call B
    resp_b = client.post("/api/chat", json={
        "user_id": "user-acc-b", "session_id": "sess-acc-b",
        "message": msg_b + " What could potentially cause this and what tests would help?"
    })
    assert resp_b.status_code == 200

    prompt_a = mock_llm.generate_chat_response.call_args_list[0][1]["messages"][0]["content"]
    prompt_b = mock_llm.generate_chat_response.call_args_list[1][1]["messages"][0]["content"]

    # 1. Both share the hepatobiliary domain
    assert "Hepatobiliary" in prompt_a
    assert "Hepatobiliary" in prompt_b

    # 2. Original messages are different
    assert msg_a in prompt_a
    assert msg_b in prompt_b
    assert prompt_a != prompt_b

    # 3. Grounding components differ
    # Patient A has meal triggers and nausea/vomiting
    assert "Biliary tract / gallbladder conditions" in prompt_a
    assert "Upper gastrointestinal involvement" in prompt_a

    # Patient B excludes meal triggers and nausea/vomiting
    assert "Upper gastrointestinal involvement" not in prompt_b
    assert "Biliary tract / gallbladder conditions" not in prompt_b

    # 4. Negative findings preserved for B
    assert "Explicitly Denied Symptoms / Negative Findings" in prompt_b
    assert "nausea" in prompt_b
    assert "fever" in prompt_b
    assert "jaundice" in prompt_b
