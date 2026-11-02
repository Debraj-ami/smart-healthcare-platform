"""
Regression Test Suite for Real-World 4-Turn Dialogue, Conversational Progression,
Multi-Part Query Handling, and Diagnostic Uncertainty.

Tests the live conversational flow:
  Turn 1: Complex multi-symptom narrative with exertional triggers, rest relief,
          functional baseline changes, explicit denials (no chest pain, no fainting),
          and speculative musing (tired/stressed).
  Turn 2: Multi-part query asking BOTH "What have I told you" AND "What do you still need to know".
  Turn 3: Diagnosis uncertainty question ("Do you think this is definitely a heart problem?").
  Turn 4: Progressive next-step question ("What should I do next?") that builds forward
          without repeating Turn 3's differential paragraphs.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app, session_manager, extraction_engine
from app.llm_client import MockLLMClient
from app.backend_client import set_backend_client
from app.schemas import IntentType, SafetyStatus, NextActionType, UserGoal, AnswerMode

client = TestClient(app)

@pytest.fixture(autouse=True)
def clean_state():
    session_manager._sessions.clear()
    extraction_engine.llm_client = MockLLMClient()
    set_backend_client(None)
    yield
    session_manager._sessions.clear()
    extraction_engine.llm_client = MockLLMClient()
    set_backend_client(None)


TURN_1 = (
    "Hi, I haven't been feeling well for the last couple of weeks. I've noticed that I get "
    "short of breath when I'm working or walking for a while, and sometimes my heart starts "
    "beating faster than usual. It usually gets better after I stop and sit down for a few "
    "minutes. I didn't have this problem before, and earlier I could work for several hours "
    "without needing to stop, but now I have to take breaks. I don't have chest pain and "
    "I haven't fainted, but I'm worried because this is new for me. I'm not sure whether "
    "it's because I'm tired, stressed, or something related to my heart. I don't know what "
    "information would be useful, so can you help me understand what I should do and what "
    "I should tell a doctor?"
)

TURN_2 = (
    "What are the important things I've told you about my symptoms, and what information "
    "do you still need to know?"
)

TURN_3 = "Do you think this is definitely a heart problem?"

TURN_4 = "What should I do next?"


def test_four_turn_exact_live_conversation_progression():
    """
    Executes the exact 4-turn clinical dialogue sequentially through /api/chat.
    Verifies state accumulation, non-repetition, non-diagnosis, and progression.
    """
    session_id = "test-live-four-turn-regression"
    user_id = "patient-four-turn"

    # =========================================================================
    # TURN 1
    # =========================================================================
    res1 = client.post("/api/chat", json={
        "user_id": user_id,
        "session_id": session_id,
        "message": TURN_1
    })
    assert res1.status_code == 200
    d1 = res1.json()
    assert d1["session_id"] == session_id
    assert d1["safety_status"] == SafetyStatus.NORMAL_CONVERSATION

    session = session_manager.get_or_create(session_id, user_id)
    info1 = session.health_info

    # 1. No accidental name extraction greeting
    assert "Hello, Not" not in d1["assistant_message"]
    assert "Nice to meet you, Not" not in d1["assistant_message"]

    # 2. Symptoms extracted cleanly
    symptom_names = [s.name.lower() for s in info1.symptoms]
    assert "shortness of breath" in symptom_names
    assert "palpitations" in symptom_names

    # 3. Chest pain explicitly negative, NOT positive
    assert "pain" not in symptom_names
    assert any("chest" in n.lower() for n in info1.negative_symptoms)
    assert any("faint" in n.lower() for n in info1.negative_symptoms)

    # 4. Triggers, relieving factors, and functional baseline change
    assert any("exertion" in t.lower() or "work" in t.lower() or "walk" in t.lower() for t in info1.triggers)
    assert any("rest" in rf.lower() for rf in info1.relieving_factors)
    assert info1.functional_change is not None
    assert "work" in info1.functional_change.lower() or "break" in info1.functional_change.lower()

    # 5. Speculative tiredness/stress is NOT converted into patient fact
    assert "fatigue" not in symptom_names
    assert "exhaustion" not in symptom_names
    assert "stress" not in symptom_names
    assert "anxiety" not in symptom_names

    # 6. Overall timeline preserved
    assert info1.duration is not None
    assert "week" in info1.duration.lower()

    # =========================================================================
    # TURN 2: Multi-part query (Knowns + Unknowns)
    # =========================================================================
    res2 = client.post("/api/chat", json={
        "user_id": user_id,
        "session_id": session_id,
        "message": TURN_2
    })
    assert res2.status_code == 200
    d2 = res2.json()
    msg2 = d2["assistant_message"]

    # Answers BOTH parts of the query
    assert "### A. USER-REPORTED INFORMATION" in msg2
    assert "### B. NOT YET PROVIDED / STILL UNKNOWN" in msg2

    # Section A contains known symptoms and context
    assert "shortness of breath" in msg2.lower()
    assert "palpitations" in msg2.lower()
    assert "rest" in msg2.lower()
    assert "chest pain" in msg2.lower()

    # Section B explicitly lists clinically relevant unknowns
    assert "Episode Duration:" in msg2 or "duration" in msg2.lower()
    assert "Vital Signs" in msg2 or "blood pressure" in msg2.lower() or "heart rate" in msg2.lower()
    assert "Current Medications: not provided" in msg2

    # Since chest pain was explicitly denied, it must NOT demand "Pain Severity"
    assert "Pain Severity: not explicitly specified" not in msg2

    # =========================================================================
    # TURN 3: Diagnostic Uncertainty Question
    # =========================================================================
    res3 = client.post("/api/chat", json={
        "user_id": user_id,
        "session_id": session_id,
        "message": TURN_3
    })
    assert res3.status_code == 200
    d3 = res3.json()
    msg3 = d3["assistant_message"]

    # 1. Intent correctly handled as symptom assessment/evaluation
    assert d3["intent"] == IntentType.SYMPTOM_ASSESSMENT

    # 2. Honest communication of diagnostic uncertainty without diagnosing
    assert "cannot" in msg3.lower() or "not possible" in msg3.lower()
    assert "definitely" in msg3.lower()
    assert "heart problem" in msg3.lower() or "cardiac" in msg3.lower()

    # 3. No diagnostic declaration
    assert "you have heart disease" not in msg3.lower()
    assert "you are diagnosed" not in msg3.lower()

    # 4. Explains difference between symptoms and clinical diagnosis
    assert "Symptoms vs. Diagnosis" in msg3 or "diagnosis" in msg3.lower()
    assert "cardiovascular" in msg3.lower() or "respiratory" in msg3.lower()

    # 5. Explains necessary clinical evaluations (physical exam, 12-lead ECG, blood tests)
    assert "ecg" in msg3.lower() or "electrocardiogram" in msg3.lower()
    assert "physical examination" in msg3.lower() or "physician" in msg3.lower()

    # 6. Safety escalation present
    assert "emergency" in msg3.lower()

    # 7. Guidance tracking
    assert "diagnostic_uncertainty_explained" in session.delivered_guidance

    # =========================================================================
    # TURN 4: Next-Steps Request (Progression, No Repetition)
    # =========================================================================
    res4 = client.post("/api/chat", json={
        "user_id": user_id,
        "session_id": session_id,
        "message": TURN_4
    })
    assert res4.status_code == 200
    d4 = res4.json()
    msg4 = d4["assistant_message"]

    # 1. Intent correctly identified
    assert d4["intent"] == IntentType.SYMPTOM_ASSESSMENT

    # 2. Does NOT repeat Turn 3's differential diagnostic paragraphs
    assert "Symptoms vs. Diagnosis" not in msg4
    assert "distinct categories, including cardiovascular conditions" not in msg4

    # 3. Progresses into actionable, practical preparation
    assert "Schedule an In-Person Medical Evaluation" in msg4 or "Schedule" in msg4
    assert "Episode Log" in msg4 or "Symptom Diary" in msg4 or "diary" in msg4.lower() or "log" in msg4.lower()
    assert "Pace Your Activities" in msg4 or "pace" in msg4.lower()
    assert "Emergency" in msg4

    # 4. Final state integrity: all 4 user messages + 4 assistant replies preserved
    assert len(session.history) == 8


def test_missing_information_standalone_query():
    """
    Tests standalone question: 'What information do you still need to know?'
    Verifies that the planner identifies clinically relevant unknowns.
    """
    session_id = "test-missing-info-query"
    user_id = "patient-missing-info"

    client.post("/api/chat", json={
        "user_id": user_id,
        "session_id": session_id,
        "message": "I've been feeling short of breath when working for the past week."
    })

    res = client.post("/api/chat", json={
        "user_id": user_id,
        "session_id": session_id,
        "message": "What information do you still need to know?"
    })
    assert res.status_code == 200
    msg = res.json()["assistant_message"]

    assert "### B. NOT YET PROVIDED / STILL UNKNOWN" in msg
    assert "Current Medications: not provided" in msg
    assert "Vital Signs" in msg or "not measured" in msg


def test_diagnosis_question_avoids_definitive_claim():
    """
    Tests 'Is this definitely heart disease?' without prior context.
    Must communicate uncertainty and refuse to diagnose.
    """
    res = client.post("/api/chat", json={
        "user_id": "user-diag-check",
        "session_id": "sess-diag-check",
        "message": "Is this definitely heart disease?"
    })
    assert res.status_code == 200
    msg = res.json()["assistant_message"]

    assert "cannot" in msg.lower() or "not possible" in msg.lower()
    assert "you have heart disease" not in msg.lower()
    assert "not a medical diagnosis" in msg or "clinical evaluation" in msg.lower() or "cannot be confirmed" in msg.lower()


def test_next_steps_standalone_query():
    """
    Tests 'What should I do next?' after reporting exertional symptoms.
    Must output actionable next steps (doctor evaluation, diary, pacing, emergency).
    """
    session_id = "test-next-steps-query"
    user_id = "patient-next-steps"

    client.post("/api/chat", json={
        "user_id": user_id,
        "session_id": session_id,
        "message": "My heart beats fast when I work, but settles when I rest."
    })

    res = client.post("/api/chat", json={
        "user_id": user_id,
        "session_id": session_id,
        "message": "What should I do next?"
    })
    assert res.status_code == 200
    msg = res.json()["assistant_message"]

    assert "medical evaluation" in msg.lower() or "appointment" in msg.lower()
    assert "emergency" in msg.lower()
