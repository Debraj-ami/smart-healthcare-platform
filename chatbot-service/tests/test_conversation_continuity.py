"""
Regression test: Six-turn conversation continuity.

Reproduces the observed failure where a patient's multi-turn conversation
produces a generic/reset response at Turn 6 when the user asks for a
doctor-oriented summary.

The test validates:
1. Structured session state (health_info) accumulates correctly across all 6 turns.
2. Answered topics are tracked and prevent redundant questions.
3. A Turn 6 summary/doctor-handoff request produces a contextually rich
   response that references symptoms, duration, triggers, and negatives
   from earlier turns — NOT a generic reset.
"""

import os
import sys
import pytest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from app.main import app, session_manager, extraction_engine
from app.schemas import IntentType, SafetyStatus, NextActionType
from app.llm_client import MockLLMClient
from app.backend_client import set_backend_client


client = TestClient(app)

SESSION_ID = "continuity-regression-session"
USER_ID = "patient_42"


@pytest.fixture(autouse=True)
def clean_state():
    """Reset all session state between tests."""
    session_manager._sessions.clear()
    extraction_engine.llm_client = MockLLMClient()
    set_backend_client(None)
    yield
    session_manager._sessions.clear()
    extraction_engine.llm_client = MockLLMClient()
    set_backend_client(None)


def send_message(message: str, session_id: str = SESSION_ID, user_id: str = USER_ID) -> dict:
    """Helper to send a chat message and return the response dict."""
    resp = client.post("/api/chat", json={
        "message": message,
        "session_id": session_id,
        "user_id": user_id
    })
    assert resp.status_code == 200, f"HTTP {resp.status_code}: {resp.text}"
    return resp.json()


# ---------------------------------------------------------------------------
# Test 1: Session state accumulates correctly across 6 turns
# ---------------------------------------------------------------------------
class TestSixTurnStateAccumulation:
    """Validates that health_info fields accumulate across 6 conversational turns."""

    def test_turn1_symptoms_recorded(self):
        """Turn 1: chest discomfort and fatigue are extracted and recorded."""
        r = send_message(
            "I'm not feeling well. I'm getting some discomfort in my chest area "
            "and I've been more tired than usual lately."
        )
        session = session_manager.get(SESSION_ID)
        assert session is not None, "Session must exist after Turn 1"
        info = session.health_info

        # At least one symptom should be extracted (pain/discomfort or fatigue)
        sym_names = [s.name.lower() for s in info.symptoms]
        has_chest = any("chest" in n or "discomfort" in n or "pain" in n for n in sym_names)
        has_fatigue = any("tired" in n or "fatigue" in n for n in sym_names)
        assert has_chest or has_fatigue, (
            f"Turn 1 must extract chest discomfort or fatigue. Got symptoms: {sym_names}"
        )

    def test_turn2_duration_and_trigger_recorded(self):
        """Turn 2: duration and meal trigger are merged into existing state."""
        send_message(
            "I'm not feeling well. I'm getting some discomfort in my chest area "
            "and I've been more tired than usual lately."
        )
        r = send_message("It started about 3 days ago. The tiredness is worse after meals.")
        session = session_manager.get(SESSION_ID)
        info = session.health_info

        # Duration should be recorded
        has_duration = bool(info.duration) or any(s.duration for s in info.symptoms)
        assert has_duration, "Turn 2 must record duration ('3 days')"

        # Trigger should be recorded
        has_trigger = bool(info.triggers)
        has_trigger_topic = "triggers" in session.answered_topics or "diet_food" in session.answered_topics
        assert has_trigger or has_trigger_topic, (
            f"Turn 2 must record meal trigger. triggers={info.triggers}, "
            f"answered_topics={session.answered_topics}"
        )

    def test_turn3_negatives_and_new_symptom(self):
        """Turn 3: negative symptoms (no dizziness/fainting) and new symptom (rapid heartbeat)."""
        send_message(
            "I'm not feeling well. I'm getting some discomfort in my chest area "
            "and I've been more tired than usual lately."
        )
        send_message("It started about 3 days ago. The tiredness is worse after meals.")
        r = send_message(
            "No, I haven't had any dizziness or fainting. "
            "But I did notice my heart beating faster sometimes."
        )
        session = session_manager.get(SESSION_ID)
        info = session.health_info

        # Negative symptoms must be tracked
        neg_lower = [n.lower() for n in info.negative_symptoms]
        has_neg_dizziness = any("dizz" in n for n in neg_lower)
        has_neg_fainting = any("faint" in n or "consciousness" in n for n in neg_lower)
        assert has_neg_dizziness or has_neg_fainting, (
            f"Turn 3 must record negative dizziness/fainting. negatives={info.negative_symptoms}"
        )

        # Answered topics should include dizziness_syncope
        assert "dizziness_syncope" in session.answered_topics, (
            f"Turn 3 must mark dizziness_syncope answered. topics={session.answered_topics}"
        )

    def test_turn4_demographics(self):
        """Turn 4: age and sex are recorded."""
        send_message(
            "I'm not feeling well. I'm getting some discomfort in my chest area "
            "and I've been more tired than usual lately."
        )
        send_message("It started about 3 days ago. The tiredness is worse after meals.")
        send_message(
            "No, I haven't had any dizziness or fainting. "
            "But I did notice my heart beating faster sometimes."
        )
        r = send_message("I'm 52 years old, male. I don't take any medications.")
        session = session_manager.get(SESSION_ID)
        info = session.health_info

        assert info.age == 52 or info.age == "52", (
            f"Turn 4 must record age=52. Got age={info.age}"
        )
        assert info.sex is not None, "Turn 4 must record sex"

    def test_turn5_answer_first_preserves_state(self):
        """Turn 5: causes question does not clear accumulated state."""
        send_message(
            "I'm not feeling well. I'm getting some discomfort in my chest area "
            "and I've been more tired than usual lately."
        )
        send_message("It started about 3 days ago. The tiredness is worse after meals.")
        send_message(
            "No, I haven't had any dizziness or fainting. "
            "But I did notice my heart beating faster sometimes."
        )
        send_message("I'm 52 years old, male. I don't take any medications.")
        r = send_message("What could be causing these symptoms?")
        session = session_manager.get(SESSION_ID)
        info = session.health_info

        # All previous state must still be present
        assert len(info.symptoms) >= 1, "Turn 5 must preserve symptoms from earlier turns"
        assert info.age is not None, "Turn 5 must preserve age from Turn 4"

    def test_turn6_summary_references_early_context(self):
        """
        Turn 6: summary request must produce a response referencing
        information from earlier turns (chest, fatigue, duration, negatives).
        This is the PRIMARY regression test for the observed failure.
        """
        send_message(
            "I'm not feeling well. I'm getting some discomfort in my chest area "
            "and I've been more tired than usual lately."
        )
        send_message("It started about 3 days ago. The tiredness is worse after meals.")
        send_message(
            "No, I haven't had any dizziness or fainting. "
            "But I did notice my heart beating faster sometimes."
        )
        send_message("I'm 52 years old, male. I don't take any medications.")
        send_message("What could be causing these symptoms?")
        r = send_message(
            "Based on everything I've told you, what information would be most useful for a doctor?"
        )

        msg = r["assistant_message"].lower()

        # The response MUST reference early-turn information
        # At minimum, it should mention symptoms or key clinical context
        has_symptom_ref = any(k in msg for k in (
            "chest", "discomfort", "fatigue", "tired", "pain"
        ))
        has_duration_ref = any(k in msg for k in (
            "3 day", "three day", "duration", "days"
        ))
        has_demographic_ref = any(k in msg for k in (
            "52", "male", "age"
        ))
        has_negative_ref = any(k in msg for k in (
            "no dizziness", "denied", "negative", "fainting", "no fainting"
        ))

        # Must have at least symptom reference + one other contextual reference
        contextual_refs = sum([has_symptom_ref, has_duration_ref, has_demographic_ref, has_negative_ref])
        assert contextual_refs >= 2, (
            f"Turn 6 summary must reference at least 2 contextual elements from earlier turns. "
            f"symptom={has_symptom_ref}, duration={has_duration_ref}, "
            f"demographic={has_demographic_ref}, negative={has_negative_ref}. "
            f"Response: {r['assistant_message'][:500]}"
        )

        # Must NOT be a generic/reset response
        generic_markers = [
            "how can i help you",
            "what symptoms are you experiencing",
            "could you please clarify",
            "i am a conversational health assistant"
        ]
        is_generic = any(gm in msg for gm in generic_markers)
        assert not is_generic, (
            f"Turn 6 must NOT produce a generic reset response. "
            f"Response: {r['assistant_message'][:500]}"
        )


# ---------------------------------------------------------------------------
# Test 2: Session history count is correct across turns
# ---------------------------------------------------------------------------
class TestSessionHistoryIntegrity:
    """Validates that the raw session history accumulates messages correctly."""

    def test_history_message_count_after_6_turns(self):
        """After 6 user turns, session history should have 12 messages (6 user + 6 assistant)."""
        messages = [
            "I'm not feeling well. I'm getting some discomfort in my chest area and I've been more tired than usual lately.",
            "It started about 3 days ago. The tiredness is worse after meals.",
            "No, I haven't had any dizziness or fainting. But I did notice my heart beating faster sometimes.",
            "I'm 52 years old, male. I don't take any medications.",
            "What could be causing these symptoms?",
            "Based on everything I've told you, what information would be most useful for a doctor?"
        ]
        for msg in messages:
            send_message(msg)

        session = session_manager.get(SESSION_ID)
        assert session is not None
        # Each turn adds 1 user + 1 assistant = 2 messages per turn
        assert len(session.history) == 12, (
            f"Expected 12 history entries (6 turns × 2). Got {len(session.history)}"
        )

    def test_early_turns_present_in_history(self):
        """Turn 1 and Turn 2 messages must be present in raw session history."""
        messages = [
            "I'm not feeling well. I'm getting some discomfort in my chest area and I've been more tired than usual lately.",
            "It started about 3 days ago. The tiredness is worse after meals.",
            "No, I haven't had any dizziness or fainting. But I did notice my heart beating faster sometimes.",
            "I'm 52 years old, male. I don't take any medications.",
            "What could be causing these symptoms?",
            "Based on everything I've told you, what information would be most useful for a doctor?"
        ]
        for msg in messages:
            send_message(msg)

        session = session_manager.get(SESSION_ID)
        user_messages = [m.content for m in session.history if m.role == "user"]
        assert "I'm not feeling well" in user_messages[0], (
            "Turn 1 user message must be first in history"
        )
        assert "3 days ago" in user_messages[1], (
            "Turn 2 user message must be second in history"
        )


# ---------------------------------------------------------------------------
# Test 3: Answered topics prevent redundant questions
# ---------------------------------------------------------------------------
class TestAnsweredTopicProtection:
    """Validates that answered topics are preserved and prevent re-asking."""

    def test_answered_topics_accumulate(self):
        """Topics from all turns accumulate in session.answered_topics."""
        send_message(
            "I'm not feeling well. I'm getting some discomfort in my chest area "
            "and I've been more tired than usual lately."
        )
        send_message("It started about 3 days ago. The tiredness is worse after meals.")
        send_message(
            "No, I haven't had any dizziness or fainting. "
            "But I did notice my heart beating faster sometimes."
        )
        send_message("I'm 52 years old, male. I don't take any medications.")

        session = session_manager.get(SESSION_ID)

        # Duration topic from Turn 2
        assert "duration" in session.answered_topics, (
            f"Duration must be in answered topics. Got: {session.answered_topics}"
        )
        # Dizziness/syncope negative from Turn 3
        assert "dizziness_syncope" in session.answered_topics, (
            f"Dizziness/syncope must be in answered topics. Got: {session.answered_topics}"
        )
        # Medications from Turn 4
        assert "medications" in session.answered_topics, (
            f"Medications must be in answered topics. Got: {session.answered_topics}"
        )

    def test_no_redundant_follow_up_on_answered_topic(self):
        """Follow-up questions must not ask about topics already answered."""
        send_message(
            "I'm not feeling well. I'm getting some discomfort in my chest area "
            "and I've been more tired than usual lately."
        )
        send_message("It started about 3 days ago. The tiredness is worse after meals.")
        send_message(
            "No, I haven't had any dizziness or fainting. "
            "But I did notice my heart beating faster sometimes."
        )
        r = send_message("I'm 52 years old, male. I don't take any medications.")

        follow_up = r.get("follow_up_question")
        if follow_up:
            follow_up_lower = follow_up.lower()
            # Should NOT ask about duration (answered Turn 2)
            assert "how long" not in follow_up_lower or "duration" not in follow_up_lower, (
                f"Follow-up must not re-ask about duration. Got: {follow_up}"
            )
            # Should NOT ask about dizziness/fainting (answered Turn 3)
            assert "dizz" not in follow_up_lower and "faint" not in follow_up_lower, (
                f"Follow-up must not re-ask about dizziness/fainting. Got: {follow_up}"
            )


# ---------------------------------------------------------------------------
# Test 4: Conversation synopsis bridges history truncation
# ---------------------------------------------------------------------------
class TestConversationSynopsis:
    """Validates that the conversation synopsis captures full multi-turn context."""

    def test_synopsis_available_after_multiple_turns(self):
        """After 5 turns, the session should be able to produce a synopsis."""
        send_message(
            "I'm not feeling well. I'm getting some discomfort in my chest area "
            "and I've been more tired than usual lately."
        )
        send_message("It started about 3 days ago. The tiredness is worse after meals.")
        send_message(
            "No, I haven't had any dizziness or fainting. "
            "But I did notice my heart beating faster sometimes."
        )
        send_message("I'm 52 years old, male. I don't take any medications.")
        send_message("What could be causing these symptoms?")

        session = session_manager.get(SESSION_ID)

        # The session should be able to produce a synopsis that includes early-turn data
        synopsis = session.build_conversation_synopsis()
        synopsis_lower = synopsis.lower()

        assert any(k in synopsis_lower for k in ("chest", "discomfort", "pain")), (
            f"Synopsis must include chest symptom from Turn 1. Got: {synopsis[:300]}"
        )
        assert any(k in synopsis_lower for k in ("3 day", "duration", "days")), (
            f"Synopsis must include duration from Turn 2. Got: {synopsis[:300]}"
        )

    def test_synopsis_includes_negatives(self):
        """Synopsis should capture explicit negative findings."""
        send_message(
            "I'm not feeling well. I'm getting some discomfort in my chest area "
            "and I've been more tired than usual lately."
        )
        send_message("It started about 3 days ago. The tiredness is worse after meals.")
        send_message(
            "No, I haven't had any dizziness or fainting. "
            "But I did notice my heart beating faster sometimes."
        )

        session = session_manager.get(SESSION_ID)
        synopsis = session.build_conversation_synopsis()
        synopsis_lower = synopsis.lower()

        assert any(k in synopsis_lower for k in ("denied", "no dizziness", "negative", "dizziness")), (
            f"Synopsis must include denied dizziness. Got: {synopsis[:300]}"
        )
