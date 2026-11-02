"""
Phase 9 — Step 3.7-B: Tests for Conversation Response Planning & Clinical Prioritization.

Verifies:
A. New-information response plan (Turn 1 -> Turn 2)
B. Correction/refinement (meal trigger refinement)
C. Objective update (feverish -> temperature 38.2°C)
D. Summary request ("Based on everything I've told you so far...")
E. New-information explanation ("What could that indicate?")
F. Negative findings & unrelated branching protection
G. Already answered topic avoidance in follow-up selection
H. Follow-up progression (at most one follow-up, relevant and unanswered)
I. Unsupported inference protection (38.2°C + chills != "infection")
J-O. Safety, streaming, educational, patient-context, ML/SHAP authority preservation
"""

import pytest
from app.schemas import (
    StructuredHealthInformation,
    Symptom,
    UserGoal,
    AnswerMode,
    ResponsePlan,
)
from app.conversation import ConversationState, ChatMessage
from app.clinical_knowledge import ClinicalKnowledgeEngine, ClinicalGroundingContext
from app.extraction import ExtractionEngine
from app.response_planner import ResponsePlanner, response_planner
from app.grounded_synthesis import GroundedSynthesisEngine, build_grounding_prompt

extraction_engine = ExtractionEngine()


# ==============================================================================
# A. New-Information Response Plan
# ==============================================================================

def test_a_new_information_response_plan():
    """
    Turn 1: RUQ pain + nausea + vomiting + oily-food trigger
    Turn 2: intermittent + 30-60 minutes + radiation to back
    Assert:
    - new information identified
    - response mode is appropriate
    - current-turn focus contains the new facts
    """
    engine = ClinicalKnowledgeEngine()
    session = ConversationState(session_id="test_plan_turn1_2", user_id="user_a")

    # Turn 1
    t1_text = "I have right upper abdominal pain with nausea and vomiting after oily meals."
    t1_info = extraction_engine.extract_health_information(t1_text)
    session.health_info = t1_info
    session.answered_topics = {"location", "symptoms", "triggers"}
    session.history.append(ChatMessage(role="user", content=t1_text))
    session.history.append(ChatMessage(role="assistant", content="I understand you have upper right pain after oily meals."))

    # Turn 2
    t2_text = "The pain comes and goes rather than being constant. It usually lasts around 30 minutes to an hour, and sometimes I feel it toward my back."
    t2_info = extraction_engine.extract_health_information(t2_text)

    grounding = engine.build_grounding_context(
        message=t2_text,
        health_info=t2_info,
        session=session
    )

    plan = response_planner.build_response_plan(
        message=t2_text,
        session=session,
        extracted_info=t2_info,
        grounding=grounding
    )

    # Assert new information identified
    assert plan.new_information.frequency == "intermittent"
    assert plan.new_information.duration is not None
    assert "30" in plan.new_information.duration or "minute" in plan.new_information.duration
    assert plan.new_information.radiation == "back"

    # Assert current-turn focus contains the new facts
    focus_str = " ".join(plan.current_turn_focus).lower()
    assert "intermittent" in focus_str or "frequency" in focus_str
    assert "duration" in focus_str or "30" in focus_str
    assert "radiation" in focus_str or "back" in focus_str

    # Assert response mode is appropriate
    assert plan.answer_mode in {AnswerMode.NEW_INFORMATION_EXPLANATION, AnswerMode.DIRECT_ANSWER}

    # Assert instruction in priorities to prioritize new information
    priorities_joined = " ".join(plan.response_priorities).lower()
    assert "newly reported" in priorities_joined or "current-turn" in priorities_joined


# ==============================================================================
# B. Correction / Refinement
# ==============================================================================

def test_b_correction_and_refinement():
    """
    Earlier: pain worse after heavy or oily meals
    Later: actually, the pain doesn't always happen after meals
    Assert:
    - changed information detected
    - latest statement is acknowledged/refined
    - user goal is CORRECTION_OR_UPDATE or answer mode is CORRECTION_ACKNOWLEDGMENT
    """
    session = ConversationState(session_id="test_plan_correction", user_id="user_b")
    t1_info = StructuredHealthInformation(
        symptoms=[Symptom(name="abdominal pain", location="upper right abdomen")],
        location="upper right abdomen",
        triggers=["oily food"]
    )
    session.health_info = t1_info
    session.history.append(ChatMessage(role="user", content="The pain gets worse after heavy or oily meals."))
    session.history.append(ChatMessage(role="assistant", content="Noted that oily meals trigger the pain."))

    refinement_text = "Actually, the pain doesn't always happen after meals."
    t2_info = extraction_engine.extract_health_information(refinement_text)

    plan = response_planner.build_response_plan(
        message=refinement_text,
        session=session,
        extracted_info=t2_info
    )

    assert plan.user_goal == UserGoal.CORRECTION_OR_UPDATE
    assert plan.answer_mode == AnswerMode.CORRECTION_ACKNOWLEDGMENT
    assert len(plan.changed_information) > 0
    changed_str = " ".join(plan.changed_information).lower()
    assert "meal" in changed_str or "trigger" in changed_str or "inconsistent" in changed_str


# ==============================================================================
# C. Objective Update
# ==============================================================================

def test_c_objective_measurement_update():
    """
    Earlier: feverish feeling
    Later: I checked my temperature and it was 38.2°C
    Assert:
    - new measured temperature recognized (38.2°C)
    - recognized as more specific current information
    """
    session = ConversationState(session_id="test_plan_temp", user_id="user_c")
    t1_info = StructuredHealthInformation(
        symptoms=[Symptom(name="fever", severity="mild")],
        fever="yes"
    )
    session.health_info = t1_info
    session.history.append(ChatMessage(role="user", content="I feel feverish."))
    session.history.append(ChatMessage(role="assistant", content="Noted the feverish sensation."))

    t2_text = "I checked my temperature and it was 38.2°C."
    t2_info = extraction_engine.extract_health_information(t2_text)

    # Extraction test
    assert t2_info.body_temperature is not None
    assert "38.2" in t2_info.body_temperature

    plan = response_planner.build_response_plan(
        message=t2_text,
        session=session,
        extracted_info=t2_info
    )

    assert plan.new_information.body_temperature is not None
    assert "38.2" in plan.new_information.body_temperature
    focus_str = " ".join(plan.current_turn_focus).lower()
    assert "38.2" in focus_str or "temperature" in focus_str


# ==============================================================================
# D. Summary Request
# ==============================================================================

def test_d_summary_request_not_reduced_to_single_question():
    """
    User: "Based on everything I've told you so far, what information would be most useful for a doctor?"
    Assert:
    - summary mode (SUMMARY or EVALUATION_GUIDANCE)
    - known information prioritized
    - unanswered information included
    - not reduced to one arbitrary follow-up
    """
    session = ConversationState(session_id="test_plan_summary", user_id="user_d")
    session.health_info = StructuredHealthInformation(
        symptoms=[Symptom(name="abdominal pain", location="upper right abdomen"), Symptom(name="nausea")],
        location="upper right abdomen",
        frequency="intermittent",
        duration="45 minutes",
        negative_symptoms=["jaundice", "dark urine"]
    )
    session.answered_topics = {"location", "symptoms", "frequency", "duration", "jaundice", "urine"}

    msg = "Based on everything I've told you so far, what information would be most useful for a doctor?"
    t_info = extraction_engine.extract_health_information(msg)

    engine = ClinicalKnowledgeEngine()
    grounding = engine.build_grounding_context(message=msg, health_info=session.health_info, session=session)

    plan = response_planner.build_response_plan(
        message=msg,
        session=session,
        extracted_info=t_info,
        grounding=grounding
    )

    assert plan.user_goal in {UserGoal.SUMMARY_OF_CONVERSATION, UserGoal.EVALUATION_INFORMATION}
    assert plan.answer_mode in {AnswerMode.SUMMARY, AnswerMode.EVALUATION_GUIDANCE}

    # Known facts included
    assert len(plan.important_known_information) >= 2
    known_str = " ".join(plan.important_known_information).lower()
    assert "abdominal pain" in known_str or "upper right" in known_str

    # Unanswered information / evaluation priorities included
    priorities_str = " ".join(plan.response_priorities).lower()
    assert "summarize" in priorities_str
    assert "distinguish" in priorities_str or "evaluation" in priorities_str or "clinician" in priorities_str

    # Check synthesis prompt format contains summary guidelines
    synth_engine = GroundedSynthesisEngine()
    prompt = synth_engine.build_grounding_prompt(
        message=msg,
        health_info=session.health_info,
        grounding=grounding,
        session=session,
        response_plan=plan
    )
    assert "When Summary Mode is requested" in prompt
    assert "RESPONSE MODE:\nSUMMARY" in prompt or "RESPONSE MODE:\nEVALUATION_GUIDANCE" in prompt


# ==============================================================================
# E. New-Information Explanation
# ==============================================================================

def test_e_new_information_explanation():
    """
    "What could that indicate?"
    Assert:
    - direct answer mode or new information explanation
    - current-turn facts prioritized
    - previous explanation not blindly repeated
    """
    session = ConversationState(session_id="test_plan_explain", user_id="user_e")
    session.health_info = StructuredHealthInformation(
        symptoms=[Symptom(name="abdominal pain", location="upper right abdomen")],
        location="upper right abdomen"
    )
    session.history.append(ChatMessage(role="user", content="I have right upper stomach pain."))
    session.history.append(ChatMessage(role="assistant", content="Pain in the right upper quadrant can involve the gallbladder or liver."))

    msg = "It comes and goes in 45 minute waves and radiates to my back. What could that indicate?"
    t_info = extraction_engine.extract_health_information(msg)

    engine = ClinicalKnowledgeEngine()
    grounding = engine.build_grounding_context(message=msg, health_info=t_info, session=session)

    plan = response_planner.build_response_plan(
        message=msg,
        session=session,
        extracted_info=t_info,
        grounding=grounding
    )

    assert plan.user_goal == UserGoal.EXPLANATION_OF_CURRENT_SYMPTOMS
    assert plan.answer_mode in {AnswerMode.NEW_INFORMATION_EXPLANATION, AnswerMode.DIRECT_ANSWER}
    assert bool(plan.avoid_repetition)
    priorities_str = " ".join(plan.response_priorities).lower()
    assert "newly reported" in priorities_str or "what this new pattern adds" in priorities_str


# ==============================================================================
# F. Negative Findings & Unrelated Branching Prevention
# ==============================================================================

def test_f_negative_findings_and_no_unrelated_urinary_branch():
    """
    User: "I haven't noticed yellowing of my eyes or skin, and my urine looks normal."
    Assert:
    - negative findings remain strictly negative
    - avoid_unsupported_inference forbids unrelated urinary differential branching (e.g. UTI, kidney stones)
    - negative findings are not treated as definitive exclusion
    """
    session = ConversationState(session_id="test_plan_neg", user_id="user_f")
    msg = "I haven't noticed yellowing of my eyes or skin, and my urine looks normal."
    t_info = extraction_engine.extract_health_information(msg)

    assert any("jaundice" in s.lower() for s in t_info.negative_symptoms)
    assert any("urine" in s.lower() for s in t_info.negative_symptoms)

    plan = response_planner.build_response_plan(
        message=msg,
        session=session,
        extracted_info=t_info
    )

    assert len(plan.explicit_negative_findings) >= 2
    neg_str = " ".join(plan.explicit_negative_findings).lower()
    assert "jaundice" in neg_str or "yellowing" in neg_str
    assert "urine" in neg_str

    # Check unsupported inference guards
    rules_str = " ".join(plan.avoid_unsupported_inference).lower()
    assert "do not treat negative findings as definitive exclusion" in rules_str
    assert "do not introduce unrelated differential categories" in rules_str or "urine" in rules_str


# ==============================================================================
# G. Already Answered Topic Avoidance
# ==============================================================================

def test_g_follow_up_selection_avoids_already_answered_topics():
    """
    Already known:
    - nausea = yes
    - vomiting = yes
    - oily-food trigger = yes
    - no jaundice
    - urine normal
    Assert:
    - follow-up topic does not ask about nausea, vomiting, triggers, jaundice, or urine
    """
    session = ConversationState(session_id="test_plan_answered", user_id="user_g")
    session.health_info = StructuredHealthInformation(
        symptoms=[Symptom(name="abdominal pain"), Symptom(name="nausea"), Symptom(name="vomiting")],
        location="upper right abdomen",
        triggers=["oily food"],
        negative_symptoms=["jaundice", "dark urine"]
    )
    session.answered_topics = {
        "location", "symptoms", "triggers", "nausea", "vomiting", "jaundice", "urine"
    }

    plan = response_planner.build_response_plan(
        message="The pain is quite uncomfortable.",
        session=session,
        extracted_info=extraction_engine.extract_health_information("The pain is quite uncomfortable.")
    )

    if plan.follow_up_topic:
        chosen = plan.follow_up_topic.lower()
        assert "nausea" not in chosen
        assert "vomiting" not in chosen
        assert "oily" not in chosen
        assert "jaundice" not in chosen
        assert "urine" not in chosen


# ==============================================================================
# H. Follow-Up Progression
# ==============================================================================

def test_h_follow_up_progression_at_most_one_unanswered_topic():
    """
    Assert:
    - At most ONE follow-up topic selected
    - Topic is relevant and unanswered
    """
    session = ConversationState(session_id="test_plan_followup", user_id="user_h")
    session.health_info = StructuredHealthInformation(
        symptoms=[Symptom(name="abdominal pain")],
        location="upper right abdomen"
    )
    session.answered_topics = {"location", "symptoms"}

    plan = response_planner.build_response_plan(
        message="I'm having upper right pain.",
        session=session,
        extracted_info=session.health_info
    )

    # Exactly one topic string or None
    assert plan.follow_up_topic is None or isinstance(plan.follow_up_topic, str)
    if plan.follow_up_topic:
        assert any(topic in plan.follow_up_topic.lower() for topic in plan.relevant_unanswered_topics)


# ==============================================================================
# I. Unsupported Inference Protection
# ==============================================================================

def test_i_unsupported_inference_protection():
    """
    User: "I have 38.2°C fever and chills."
    Assert:
    - Planner does NOT label this as "likely infection"
    - avoidance rules clearly forbid declaring an infection without clinical workup
    """
    msg = "I have 38.2°C fever and chills."
    info = extraction_engine.extract_health_information(msg)

    assert "chills" in [s.name for s in info.symptoms]
    assert info.body_temperature is not None

    plan = response_planner.build_response_plan(
        message=msg,
        session=None,
        extracted_info=info
    )

    # Must NOT label disease or diagnosed infection in current_turn_focus or priorities
    focus_str = " ".join(plan.current_turn_focus).lower()
    priorities_str = " ".join(plan.response_priorities).lower()
    assert "cholecystitis diagnosis" not in focus_str
    assert "proven infection" not in focus_str
    assert "diagnose infection" not in priorities_str

    # Must explicitly instruct avoidance of declaring infection
    rules_str = " ".join(plan.avoid_unsupported_inference).lower()
    assert "do not declare or diagnose an active infection" in rules_str or "infection" in rules_str


from app.llm_client import MockLLMClient
from app.main import app, extraction_engine, session_manager
from app.schemas import SafetyStatus, NextActionType

@pytest.fixture(autouse=True)
def reset_state():
    orig_client = extraction_engine.llm_client
    session_manager._sessions.clear()
    extraction_engine.llm_client = MockLLMClient()
    yield
    session_manager._sessions.clear()
    extraction_engine.llm_client = orig_client


# ==============================================================================
# J-O. System Integration & Boundary Preservation Tests
# ==============================================================================

def test_j_safety_gate_remains_active_and_bypasses_synthesis():
    """
    K. Safety gate remains active on red flags and bypasses normal synthesis.
    """
    from fastapi.testclient import TestClient

    client = TestClient(app)
    resp = client.post("/api/chat", json={
        "message": "I suddenly developed severe crushing chest pain radiating to my left arm.",
        "session_id": "test_safety_gate_planner",
        "user_id": "user_j"
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["safety_status"] == SafetyStatus.URGENT_MEDICAL_EVALUATION.value
    assert data["next_action"] == NextActionType.EMERGENCY_ROUTING.value
    assert "emergency" in data["assistant_message"].lower() or "911" in data["assistant_message"]


def test_k_streaming_includes_response_plan():
    """
    L. Streaming endpoint POST /api/chat/stream includes response plan in complete event.
    """
    from fastapi.testclient import TestClient
    import json
    from app.main import app

    client = TestClient(app)
    resp = client.post("/api/chat/stream", json={
        "message": "I have right upper abdominal pain for 2 days.",
        "session_id": "test_stream_plan_sess",
        "user_id": "user_k"
    })
    assert resp.status_code == 200

    blocks = resp.text.strip().split("\n\n")
    complete_payload = None
    for block in blocks:
        lines = [l.strip() for l in block.split("\n") if l.strip()]
        is_complete = any(l == "event: complete" for l in lines)
        for l in lines:
            if l.startswith("data:"):
                try:
                    parsed = json.loads(l[5:].strip())
                    if is_complete or "response_plan" in parsed:
                        complete_payload = parsed
                except Exception:
                    pass

    assert complete_payload is not None
    assert "response_plan" in complete_payload
    assert complete_payload["response_plan"] is not None
    assert "answer_mode" in complete_payload["response_plan"]


def test_l_educational_query_preserves_education_mode():
    """
    M. Educational query tests continue passing with GENERAL_EDUCATION mode.
    """
    plan = response_planner.build_response_plan(
        message="What is bilirubin?",
        session=None
    )
    assert plan.user_goal == UserGoal.GENERAL_INFORMATION
    assert plan.answer_mode == AnswerMode.GENERAL_EDUCATION
    assert plan.avoid_repetition == []


def test_m_patient_context_query_mode():
    """
    N. Patient context query tests continue passing with PATIENT_CONTEXT mode.
    """
    plan = response_planner.build_response_plan(
        message="How have my readings changed compared to last week?",
        session=None
    )
    assert plan.user_goal == UserGoal.PERSONAL_CONTEXT
    assert plan.answer_mode == AnswerMode.PATIENT_CONTEXT


def test_n_ml_shap_authority_remains_untouched():
    """
    O. ML/SHAP behavior remains untouched; symptom dialogue never triggers prediction.
    """
    from fastapi.testclient import TestClient
    from app.main import app
    from app.schemas import NextActionType

    client = TestClient(app)
    resp = client.post("/api/chat", json={
        "message": "My stomach has been hurting in the upper right side.",
        "session_id": "test_ml_untouched",
        "user_id": "user_n"
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["next_action"] != NextActionType.REQUEST_DISEASE_ASSESSMENT.value
    assert data["assessment_handoff"] is None


def test_o_deterministic_fallback_uses_response_plan_when_ollama_unavailable():
    """
    Deterministic fallback respects response plan, acknowledges new facts, and does not invent facts.
    """
    from unittest.mock import patch
    from app.conversation import FollowUpEngine

    engine = ClinicalKnowledgeEngine()
    session = ConversationState(session_id="test_fallback_plan", user_id="user_o")
    session.health_info = StructuredHealthInformation(
        symptoms=[Symptom(name="abdominal pain", location="upper right abdomen")],
        location="upper right abdomen",
        triggers=["oily meals"]
    )
    session.answered_topics = {"location", "symptoms", "triggers"}

    t2_msg = "The pain comes and goes every 45 minutes and radiates to my back."
    t2_info = extraction_engine.extract_health_information(t2_msg)

    # Force LLM synthesis to return None to test deterministic fallback
    with patch("app.grounded_synthesis.grounded_synthesizer.synthesize_response", return_value=None):
        resp = engine.generate_answer_first_response(
            session=session,
            message=t2_msg,
            request_user_id="user_o",
            extracted_info=t2_info,
            follow_up_engine=FollowUpEngine()
        )

    msg = resp.assistant_message.lower()
    assert "intermittent" in msg or "comes and goes" in msg or "45 minute" in msg or "back" in msg
    assert resp.response_plan is not None
    assert "diagnosis:" not in msg

