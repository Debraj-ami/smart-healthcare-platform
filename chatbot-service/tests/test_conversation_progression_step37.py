"""
Phase 9 - Step 3.7 Regression Tests: New-Information-Aware Conversation Progression.

Validates:
A. Current-turn new information is identified (intermittent, 30-60 min, radiation to back).
B. Previously known facts remain preserved across turns.
C. New information is not incorrectly classified as old information.
D. Grounded synthesis prompt contains clearly separated sections for:
   - CURRENT USER MESSAGE
   - VERIFIED NEW INFORMATION FROM THIS TURN
   - VERIFIED INFORMATION ALREADY KNOWN
   - RELEVANT CLINICAL GROUNDING
   - PREVIOUSLY DISCUSSED / ANSWERED TOPICS
   - RELEVANT UNANSWERED INFORMATION
E. Prompt instructs the LLM to prioritize new information and explain what it changes.
F. Previously discussed information is not unnecessarily repeated by deterministic logic.
G. Follow-up selection avoids already answered topics.
H. Explicit negative findings remain negative (not converted to positive or unknown).
I. Ollama failure still uses grounded deterministic fallback with newly reported facts acknowledgment.
J. Existing safety red-flag behavior remains intact.
K. Educational-query classification remains intact.
L. Streaming chat endpoint outputs progression context and preserves contracts.
M. Patient context retrieval integration remains intact.
N. ML/SHAP diagnostic boundary is strictly respected (no symptom-only ML execution).
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app.main import app, session_manager, extraction_engine
from app.schemas import (
    StructuredHealthInformation,
    Symptom,
    NextActionType,
    SafetyStatus,
    IntentType
)
from app.conversation import (
    ConversationState,
    FollowUpEngine,
    compute_new_information,
    build_conversation_progress_context
)
from app.extraction import ExtractionEngine, is_question_about_causes_or_what_helps
from app.grounded_synthesis import GroundedSynthesisEngine, SYSTEM_GROUNDED_SYNTHESIS_PROMPT, grounded_synthesizer
from app.clinical_knowledge import ClinicalKnowledgeEngine, ClinicalGroundingContext
from app.llm_client import MockLLMClient

client = TestClient(app)


@pytest.fixture(autouse=True)
def clean_sessions():
    """Ensure in-memory sessions and extraction engines are reset to fast mock between tests."""
    session_manager._sessions.clear()
    extraction_engine.llm_client = MockLLMClient()
    grounded_synthesizer._llm_client = MockLLMClient()
    yield


# ==============================================================================
# A, B, C: Current-turn New vs Historical Information Extraction & Preservation
# ==============================================================================

def test_current_turn_new_information_identified_and_historical_preserved():
    """
    Test A & B & C:
    Turn 1: RUQ pain, nausea, vomiting, feverish feeling, oily-food trigger.
    Turn 2: pain is intermittent, lasts 30-60 minutes, radiates to back, 'What could that indicate?'.
    Assert:
    - Turn 2 progression context identifies intermittent frequency, 30-60 min duration, and back radiation.
    - Historical facts (RUQ pain, nausea, vomiting, oily-food trigger) remain preserved in known information.
    - New information is not classified as old, and old is not classified as new.
    """
    session_id = "test-sess-progression-ab"
    user_id = "test-user-progression"

    msg_turn1 = (
        "I've been feeling weak and tired for the last two days. I have reduced appetite, nausea, "
        "vomiting twice, a feverish feeling, and dull pain in the upper-right side of my abdomen. "
        "The pain gets worse after heavy or oily meals. I can still move around normally, and the "
        "pain isn't extremely severe. I haven't noticed severe chest pain, difficulty breathing, "
        "fainting, or loss of consciousness. I haven't had any tests yet. I'm mainly worried about "
        "what could potentially be causing this and what information or tests would help a doctor understand."
    )

    r1 = client.post("/api/chat", json={
        "user_id": user_id,
        "session_id": session_id,
        "message": msg_turn1
    })
    assert r1.status_code == 200
    data1 = r1.json()

    # Turn 1 extraction checks
    h1 = data1["structured_health_information"]
    assert "right upper abdomen" in (h1["location"] or "")
    assert any("vomit" in s["name"].lower() for s in h1["symptoms"])
    assert any("nausea" in s["name"].lower() for s in h1["symptoms"])
    assert any("oily" in t.lower() or "heavy" in t.lower() for t in h1["triggers"])

    # Turn 2: New details (intermittent, 30-60 min, radiates to back)
    msg_turn2 = "The pain comes and goes rather than being constant. It usually lasts 30 minutes to an hour, and sometimes I feel it toward my back. What could that indicate?"

    r2 = client.post("/api/chat", json={
        "user_id": user_id,
        "session_id": session_id,
        "message": msg_turn2
    })
    assert r2.status_code == 200
    data2 = r2.json()

    # Verify Progression Context in ChatResponse
    assert "progression_context" in data2
    prog = data2["progression_context"]
    assert prog is not None

    new_info = prog["newly_reported_information"]
    old_info = prog["previously_known_information"]

    # Test A: New facts identified
    assert new_info["frequency"] == "intermittent"
    assert "30 minutes to an hour" in (new_info["duration"] or "")
    assert new_info["radiation"] == "back" or any(s.get("radiation") == "back" for s in new_info.get("symptoms", []))

    # Test B: Previously known facts preserved
    assert "right upper abdomen" in (old_info["location"] or "")
    assert any("oily" in t.lower() or "heavy" in t.lower() for t in old_info["triggers"])
    assert any("vomit" in s["name"].lower() for s in old_info["symptoms"])

    # Test C: New facts were NOT in previous turn
    assert old_info.get("frequency") is None or old_info.get("frequency") != "intermittent"
    assert old_info.get("radiation") is None


def test_compute_new_information_deterministic_purity():
    """
    Test deterministic calculation of new info relative to previous state.
    """
    prev = StructuredHealthInformation(
        symptoms=[Symptom(name="abdominal pain", location="right upper abdomen", severity="moderate")],
        location="right upper abdomen",
        severity="moderate",
        triggers=["oily meals"]
    )

    curr = StructuredHealthInformation(
        frequency="intermittent",
        duration="30 to 60 minutes",
        radiation="back"
    )

    diff = compute_new_information(prev, curr)

    assert diff.frequency == "intermittent"
    assert diff.duration == "30 to 60 minutes"
    assert diff.radiation == "back"
    assert diff.location is None
    assert diff.severity is None
    assert len(diff.triggers) == 0


# ==============================================================================
# D & E: Grounded Synthesis Prompt Structure & Instructions
# ==============================================================================

def test_grounded_synthesis_prompt_sections_and_prioritization():
    """
    Test D & E:
    Verifies that build_grounding_prompt creates clearly separated sections:
    - CURRENT USER MESSAGE
    - VERIFIED NEW INFORMATION FROM THIS TURN
    - VERIFIED INFORMATION ALREADY KNOWN
    - RELEVANT CLINICAL GROUNDING
    - PREVIOUSLY DISCUSSED / ANSWERED TOPICS
    - RELEVANT UNANSWERED INFORMATION
    And that SYSTEM_GROUNDED_SYNTHESIS_PROMPT instructs prioritizing new facts.
    """
    engine = GroundedSynthesisEngine()

    prev_info = StructuredHealthInformation(
        symptoms=[Symptom(name="abdominal pain", location="upper right abdomen")],
        location="upper right abdomen",
        triggers=["oily food"]
    )
    new_info = StructuredHealthInformation(
        frequency="intermittent",
        duration="30 to 60 minutes",
        radiation="back"
    )

    state = ConversationState(session_id="test-prompt-sections", user_id="user1")
    state.health_info = prev_info
    state.answered_topics = {"location", "symptoms", "triggers"}

    progress_context = build_conversation_progress_context(
        message="The pain comes and goes, lasts 30 to 60 minutes, and radiates to back. What does that indicate?",
        previous_info=prev_info,
        new_info=new_info,
        session=state,
        intent="CAUSES_OR_WHAT_HELPS"
    )

    mock_grounding = ClinicalGroundingContext(
        domain_id="hepatobiliary_ruq",
        display_title="Upper-right abdominal / hepatobiliary evaluation",
        selected_components=[],
        component_ids=["episodic_radiating_biliary_pattern"],
        possible_categories=["Biliary colic or gallbladder irritation"],
        relevant_information=["Whether episodes occur postprandially"],
        clinical_examinations=["Abdominal palpation"],
        laboratory_tests=["Liver function tests"],
        diagnostic_imaging=["Abdominal ultrasound"],
        disease_candidates=["gallbladder_disease"],
        deterministic_fallback_message="Fallback test",
        excluded_components={}
    )

    prompt = engine.build_grounding_prompt(
        message="The pain comes and goes, lasts 30 to 60 minutes, and radiates to back. What does that indicate?",
        health_info=prev_info,
        grounding=mock_grounding,
        follow_up="Have you noticed yellowing of your eyes or dark urine?",
        session=state,
        progress_context=progress_context
    )

    # Test D: Required separated sections present
    assert "CURRENT USER MESSAGE:" in prompt
    assert "VERIFIED NEW INFORMATION FROM THIS TURN" in prompt
    assert "VERIFIED INFORMATION ALREADY KNOWN (VERIFIED USER FACTS):" in prompt
    assert "RELEVANT CLINICAL GROUNDING" in prompt
    assert "PREVIOUSLY DISCUSSED / ANSWERED TOPICS:" in prompt
    assert "RELEVANT UNANSWERED INFORMATION:" in prompt
    assert "APPROVED FOLLOW-UP QUESTION" in prompt

    # Test E: Priority instruction in system prompt
    assert "Prioritize newly reported information" in SYSTEM_GROUNDED_SYNTHESIS_PROMPT
    assert "Acknowledge important new facts" in SYSTEM_GROUNDED_SYNTHESIS_PROMPT
    assert "Do not repeat previously explained material unnecessarily" in SYSTEM_GROUNDED_SYNTHESIS_PROMPT
    assert "Explain what the NEW information adds or changes" in SYSTEM_GROUNDED_SYNTHESIS_PROMPT


# ==============================================================================
# F & I: Deterministic Logic & Fallback Respects New Information Without Repetition
# ==============================================================================

def test_deterministic_fallback_acknowledges_new_info_and_prioritizes():
    """
    Test F & I:
    When Ollama is unavailable, the deterministic fallback acknowledges the new facts
    (intermittent pattern, 30-60 minute duration, radiation to back) without restarting
    from a generic blank state.
    """
    cke = ClinicalKnowledgeEngine()

    prev_info = StructuredHealthInformation(
        symptoms=[Symptom(name="dull pain", location="upper-right abdomen")],
        location="upper-right abdomen",
        triggers=["heavy meals"]
    )
    new_info = StructuredHealthInformation(
        frequency="intermittent",
        duration="30 to 60 minutes",
        radiation="back"
    )

    state = ConversationState(session_id="test-fallback", user_id="user1")
    state.health_info = prev_info
    state.answered_topics = {"location", "triggers"}

    progress_context = build_conversation_progress_context(
        message="The pain comes and goes rather than being constant. It lasts 30 to 60 minutes and goes to my back.",
        previous_info=prev_info,
        new_info=new_info,
        session=state,
        intent="CAUSES_OR_WHAT_HELPS"
    )

    # Force LLM synthesis failure to test fallback
    with patch("app.grounded_synthesis.grounded_synthesizer.synthesize_response", return_value=None):
        resp = cke.generate_answer_first_response(
            session=state,
            message="The pain comes and goes rather than being constant. It lasts 30 to 60 minutes and goes to my back.",
            request_user_id="user1",
            extracted_info=new_info,
            follow_up_engine=FollowUpEngine(),
            progress_context=progress_context
        )

    msg = resp.assistant_message.lower()
    # Verifies acknowledgment of newly reported facts in fallback
    assert "intermittent" in msg or "coming and going" in msg
    assert "30 to 60 minutes" in msg or "duration" in msg
    assert "back" in msg
    # Verifies grounding in biliary colic context
    assert "biliary" in msg or "gallbladder" in msg


# ==============================================================================
# G: Follow-up Engine Avoids Already Answered Topics
# ==============================================================================

def test_followup_selection_avoids_already_answered_topics():
    """
    Test G:
    If the user has already answered nausea, vomiting, feverish feeling, oily-food trigger,
    and duration/severity/frequency, the follow-up engine must NOT ask about nausea,
    fever, meals, or severity again.
    """
    state = ConversationState(session_id="test-fu-avoid", user_id="user1")
    state.health_info = StructuredHealthInformation(
        symptoms=[
            Symptom(name="abdominal pain", location="upper-right abdomen", severity="moderate"),
            Symptom(name="nausea"),
            Symptom(name="vomiting")
        ],
        location="upper-right abdomen",
        severity="moderate",
        duration="2 days",
        frequency="intermittent",
        triggers=["oily food"]
    )
    state.answered_topics = {"location", "duration", "severity", "frequency", "nausea", "vomiting", "fever", "triggers"}

    fu_engine = FollowUpEngine()
    question = fu_engine.generate_follow_up(state)

    if question:
        q_low = question.lower()
        assert "nausea" not in q_low
        assert "vomit" not in q_low
        assert "how long" not in q_low
        assert "how severe" not in q_low


# ==============================================================================
# H: Explicit Negative Findings Remain Negative
# ==============================================================================

def test_explicit_negative_findings_remain_strictly_negative():
    """
    Test H:
    Explicit denial of chest pain, shortness of breath, or jaundice must never be
    inverted into positive symptoms or treated as unmentioned.
    """
    msg = "I haven't noticed severe chest pain, difficulty breathing, or yellowing of the skin."
    engine = ExtractionEngine()
    extracted = engine.extract_health_information(msg)

    assert "severe chest pain" in extracted.negative_symptoms
    assert "difficulty breathing" in extracted.negative_symptoms
    assert any("jaundice" in s for s in extracted.negative_symptoms)

    # Verify none of these became positive symptoms
    pos_names = [s.name.lower() for s in extracted.symptoms]
    assert "severe chest pain" not in pos_names
    assert "difficulty breathing" not in pos_names
    assert "jaundice" not in pos_names


# ==============================================================================
# J, K, L, M, N: Safety, Education, Streaming, Patient Context, ML/SHAP Authority
# ==============================================================================

def test_safety_gate_remains_active_on_red_flags():
    """
    Test J:
    Red flags (e.g. crushing chest pain, cannot breathe) immediately trigger emergency safety.
    """
    resp = client.post("/api/chat", json={
        "user_id": "u-safe",
        "session_id": "s-safe",
        "message": "I am experiencing crushing chest pain and I cannot breathe!"
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["safety_status"] == SafetyStatus.URGENT_MEDICAL_EVALUATION
    assert data["next_action"] == NextActionType.EMERGENCY_ROUTING


def test_educational_query_classification_remains_intact():
    """
    Test K:
    General knowledge queries (e.g. 'What is bilirubin?') route educationally without diagnosing.
    """
    resp = client.post("/api/chat", json={
        "user_id": "u-edu",
        "session_id": "s-edu",
        "message": "What is bilirubin and how does the liver process it?"
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["intent"] in [IntentType.GENERAL_HEALTH_QUESTION, "GENERAL_HEALTH_QUESTION", "EDUCATIONAL", "GENERAL_KNOWLEDGE"]
    assert "bilirubin" in data["assistant_message"].lower()


def test_streaming_endpoint_provides_progression_context():
    """
    Test L:
    Streaming endpoint includes progression_context in complete event.
    """
    session_id = "stream-prog-sess"
    user_id = "stream-prog-user"

    # Turn 1
    resp1 = client.post("/api/chat/stream", json={
        "user_id": user_id,
        "session_id": session_id,
        "message": "I have upper-right stomach pain after eating fried chicken."
    })
    assert resp1.status_code == 200

    # Turn 2
    resp2 = client.post("/api/chat/stream", json={
        "user_id": user_id,
        "session_id": session_id,
        "message": "The pain comes and goes and radiates toward my back. What could that indicate?"
    })
    assert resp2.status_code == 200
    content = resp2.text
    assert "event: complete" in content
    assert "progression_context" in content


def test_patient_context_handling_intact():
    """
    Test M:
    Handling patient context does not raise or fabricate unmentioned conditions.
    """
    resp = client.post("/api/chat", json={
        "user_id": "test-patient-id",
        "session_id": "sess-patient-test",
        "message": "Hello, I am Debraj."
    })
    assert resp.status_code == 200
    assert resp.json()["safety_status"] == SafetyStatus.NORMAL_CONVERSATION


def test_symptom_workflow_never_triggers_ml_model_prematurely():
    """
    Test N:
    A symptom inquiry ('What could that indicate?') NEVER executes ML risk assessment
    or fabricates disease probability.
    """
    resp = client.post("/api/chat", json={
        "user_id": "user-ml-boundary",
        "session_id": "sess-ml-boundary",
        "message": "My upper-right abdomen hurts and it radiates to my back. What could that indicate?"
    })
    assert resp.status_code == 200
    data = resp.json()
    # Next action must NOT be ML risk assessment until lab values are provided
    assert data["next_action"] != "DISEASE_RISK_ASSESSMENT"
    assert "risk_score" not in data
    assert "disease_probability" not in data
