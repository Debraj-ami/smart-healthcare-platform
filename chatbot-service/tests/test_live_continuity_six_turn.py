"""
Live end-to-end integration regression test for the exact 6-turn patient conversation
and follow-up summary query.

Covers:
A. Exact six-turn live-path reproduction via TestClient(/api/chat).
B. Conversation / session ID remains identical across all turns.
C. State is reused and accumulated across all turns.
D. Turn 6 does not trigger greeting ('Hello, Not!').
E. Previous symptom facts (shortness of breath, palpitations) remain available.
F. Conversation synopsis is available and populated.
G. Final response uses prior context.
H. Patient-reported facts (triggers, relieving factors, functional change) are preserved.
I. LLM interpretations cannot become patient facts.
J. Unsupported 'fatigue/exhaustion' or 'stress/anxiety' are never presented as patient facts.
K. Patient isolation between different sessions.
L. Emergency handling remains intact.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app, session_manager, extraction_engine
from app.llm_client import MockLLMClient
from app.backend_client import set_backend_client
from app.schemas import IntentType, SafetyStatus, NextActionType

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

MESSAGES = [
    "I'm not feeling well. I'm getting shortness of breath while working.",
    "It started a few days ago. It gets worse when I have to work continuously, but it gets better when I sit down and rest.",
    "I don't have chest pain, but sometimes I feel my heart beating faster when this happens.",
    "It's mostly just fast, not irregular. It usually settles down after a few minutes when I rest. I don't feel faint.",
    "This is new for me. I could normally work for several hours without getting breathless, but recently I have to stop and rest.",
    "I'm not having the breathing problem right now. It only happens when I'm working, and it goes away when I rest. What should I do?"
]


def test_exact_six_turn_live_path_and_grounding():
    session_id = "test-live-six-turn-continuity"
    user_id = "patient-live-user"

    responses = []
    for idx, msg in enumerate(MESSAGES):
        payload = {
            "user_id": user_id,
            "session_id": session_id,
            "message": msg
        }
        res = client.post("/api/chat", json=payload)
        assert res.status_code == 200, f"Turn {idx+1} failed with status {res.status_code}"
        data = res.json()
        assert data["session_id"] == session_id, f"Turn {idx+1} returned mismatched session_id"
        responses.append(data)

    # Turn 6 assertions
    turn_6 = responses[5]
    reply_6 = turn_6["assistant_message"]

    # D. Final turn does not trigger greeting / reset
    assert "Hello, Not!" not in reply_6, f"Turn 6 was reset with 'Hello, Not!': {reply_6}"
    assert "Nice to meet you" not in reply_6
    assert "I'm here to assist with health inquiries" not in reply_6

    # Verify session state from session manager
    session = session_manager.get_or_create(session_id, user_id)
    assert session.session_id == session_id

    # C. State is reused and accumulated across all six turns
    assert len(session.history) == 12  # 6 user + 6 assistant

    # E. Previous symptom facts remain available
    symptom_names = [s.name.lower() for s in session.health_info.symptoms]
    assert "shortness of breath" in symptom_names
    assert "palpitations" in symptom_names
    # Chest pain should be in negative symptoms, NOT positive symptoms
    assert "pain" not in symptom_names
    assert any("chest" in n.lower() for n in session.health_info.negative_symptoms)

    # H. Patient-reported facts are preserved
    assert any("exertion" in t.lower() or "work" in t.lower() for t in session.health_info.triggers)
    assert any("rest" in rf.lower() for rf in session.health_info.relieving_factors)
    assert session.health_info.functional_change is not None

    # F. Conversation synopsis is available
    synopsis = session.build_conversation_synopsis()
    assert "shortness of breath" in synopsis.lower()
    assert "rest" in synopsis.lower()

    # J. Unsupported 'fatigue/exhaustion' or 'stress/anxiety' not presented as patient-reported facts
    assert "fatigue" not in symptom_names
    assert "exhaustion" not in symptom_names
    assert "anxiety" not in symptom_names
    assert "stress" not in symptom_names

    # Check reply text doesn't claim patient reported fatigue or anxiety
    reply_lower = reply_6.lower()
    assert "your symptoms are related to fatigue" not in reply_lower
    assert "your symptoms seem to be related to fatigue" not in reply_lower
    assert "your symptoms are related to exhaustion" not in reply_lower

    # Follow-up test: "What have I told you so far about my breathing problem?"
    summary_msg = "What have I told you so far about my breathing problem?"
    res_summary = client.post("/api/chat", json={
        "user_id": user_id,
        "session_id": session_id,
        "message": summary_msg
    })
    assert res_summary.status_code == 200
    summary_reply = res_summary.json()["assistant_message"].lower()

    # Verify the summary accurately references what the patient told
    assert "shortness of breath" in summary_reply or "breathing" in summary_reply
    assert "work" in summary_reply
    assert "rest" in summary_reply
    # Does not claim patient reported fatigue/stress
    assert "you reported fatigue" not in summary_reply
    assert "you reported stress" not in summary_reply


def test_patient_isolation_under_same_sequence():
    """K. Patient isolation remains intact."""
    sess_a = "patient-a-session"
    sess_b = "patient-b-session"

    # Patient A sends exertional breathing message
    client.post("/api/chat", json={
        "user_id": "user-a",
        "session_id": sess_a,
        "message": "I have shortness of breath while working."
    })

    # Patient B asks general question
    res_b = client.post("/api/chat", json={
        "user_id": "user-b",
        "session_id": sess_b,
        "message": "What is normal resting heart rate?"
    })

    state_a = session_manager.get_or_create(sess_a, "user-a")
    state_b = session_manager.get_or_create(sess_b, "user-b")

    assert any(s.name == "shortness of breath" for s in state_a.health_info.symptoms)
    assert not any(s.name == "shortness of breath" for s in state_b.health_info.symptoms)


def test_emergency_handling_remains_intact():
    """L. Emergency handling remains intact and deterministic."""
    res = client.post("/api/chat", json={
        "user_id": "user-emerg",
        "session_id": "sess-emerg",
        "message": "I have crushing chest pain radiating to my left arm and I can't breathe."
    })
    assert res.status_code == 200
    data = res.json()
    assert (
        data["safety_status"] == SafetyStatus.URGENT_MEDICAL_EVALUATION.value
        or data["next_action"] == NextActionType.EMERGENCY_ROUTING.value
        or "911" in data["assistant_message"]
        or "emergency" in data["assistant_message"].lower()
    )
