"""
Unit and integration tests for Phase 9 Step 3.5:
Conversation State, Follow-Up Quality, and Extraction Completeness.

Verifies:
A. Multi-turn state merge
B. Previously answered question detection
C. Explicit negative symptom persistence
D. User correction of unsupported inference
E. No invented severity
F. No invented frequency
G. No assistant-generated condition becoming a patient condition
H. Complete long-message extraction
I. Follow-up progression
J. Answer-first behavior
K. Educational query filtering
L. Casual conversation scope & GK redirection
M. Context preserved across 4+ turns (Section 14 multi-turn benchmark)
"""

import json
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from app.main import app, session_manager, extraction_engine
from app.schemas import (
    IntentType,
    SafetyStatus,
    NextActionType,
    StructuredHealthInformation,
    Symptom
)
from app.llm_client import MockLLMClient, BaseLLMClient


client = TestClient(app)


@pytest.fixture(autouse=True)
def clean_sessions():
    """Ensure in-memory sessions and extraction engine are reset between tests."""
    session_manager._sessions.clear()
    extraction_engine.llm_client = MockLLMClient()
    yield
    session_manager._sessions.clear()
    extraction_engine.llm_client = MockLLMClient()


# ==============================================================================
# A. Multi-Turn State Merge Across Turns
# ==============================================================================
def test_multiturn_state_merge():
    """Verifies that subsequent turns enrich rather than replace prior verified findings."""
    session_id = "test-multiturn-merge"
    user_id = "user-merge-1"

    # Turn 1: Primary symptom and duration
    r1 = client.post("/api/chat", json={
        "user_id": user_id,
        "session_id": session_id,
        "message": "I have had dull pain in my right upper abdomen for two days."
    })
    assert r1.status_code == 200
    info1 = r1.json()["structured_health_information"]
    assert info1["location"] == "right upper abdomen"
    assert "days" in info1["duration"]
    assert info1["severity"] == "dull"

    # Turn 2: New concurrent symptom and triggers
    r2 = client.post("/api/chat", json={
        "user_id": user_id,
        "session_id": session_id,
        "message": "I also feel nauseous and it gets worse after oily meals."
    })
    assert r2.status_code == 200
    info2 = r2.json()["structured_health_information"]

    # Must preserve location, duration, severity from Turn 1, and add nausea + triggers from Turn 2
    assert info2["location"] == "right upper abdomen"
    assert "days" in info2["duration"]
    assert info2["severity"] == "dull"
    s_names = [s["name"] for s in info2["symptoms"]]
    assert "pain" in s_names
    assert "nausea" in s_names
    assert any("oily" in tr for tr in info2["triggers"])


# ==============================================================================
# B. Previously Answered Question Detection & Prevention
# ==============================================================================
def test_previously_answered_question_prevention():
    """
    Verifies that the follow-up engine strictly tracks answered topics and NEVER
    asks the user about topics they have already covered.
    """
    session_id = "test-answered-prevention"
    user_id = "user-prev-1"

    # Turn 1: User provides food and fluids
    client.post("/api/chat", json={
        "user_id": user_id,
        "session_id": session_id,
        "message": "I have upper-right abdominal pain for two days. I drink 2 liters of water daily, but eating oily meals makes it worse."
    })
    session = session_manager.get_or_create(session_id, user_id)
    assert "fluid_intake" in session.answered_topics
    assert "diet_food" in session.answered_topics
    assert "triggers" in session.answered_topics

    # Turn 2: User answers about bowel and urine
    resp2 = client.post("/api/chat", json={
        "user_id": user_id,
        "session_id": session_id,
        "message": "My urine is somewhat darker and I have no blood in stool."
    })
    assert "urine" in session.answered_topics
    assert "stool" in session.answered_topics

    msg = resp2.json()["assistant_message"].lower()
    follow_up = (resp2.json().get("follow_up_question") or "").lower()

    # The assistant must NOT re-ask about food, water, urine, or stool
    for forbidden in ["fluid intake", "how much water", "urine output", "changes in your stool", "eating habits"]:
        assert forbidden not in follow_up
        assert forbidden not in msg


# ==============================================================================
# C. Explicit Negative Symptom Persistence Across Turns
# ==============================================================================
def test_explicit_negative_symptom_persistence():
    """
    Verifies explicit negative findings (denied red flags and symptoms) are
    preserved and never converted into positive symptoms or erased by later turns.
    """
    session_id = "test-negative-persistence"
    user_id = "user-neg-1"

    # Turn 1: User explicitly denies chest pain, breathing difficulty, dizziness, and fainting
    client.post("/api/chat", json={
        "user_id": user_id,
        "session_id": session_id,
        "message": "I have abdominal discomfort for 2 days. I have not noticed severe chest pain or difficulty breathing. No dizziness or fainting."
    })
    session = session_manager.get_or_create(session_id, user_id)
    negatives = [n.lower() for n in session.health_info.negative_symptoms]
    assert any("chest pain" in n for n in negatives)
    assert any("breathing" in n for n in negatives)
    assert any("dizziness" in n for n in negatives)
    assert any("fainting" in n for n in negatives)

    # Positive symptoms must NOT contain the negated items
    pos_names = [s.name.lower() for s in session.health_info.symptoms]
    assert "chest pain" not in pos_names
    assert "difficulty breathing" not in pos_names
    assert "dizziness" not in pos_names
    assert "fainting" not in pos_names

    # Turn 2: Later message without mentioning those negatives
    r2 = client.post("/api/chat", json={
        "user_id": user_id,
        "session_id": session_id,
        "message": "It is somewhat worse after heavy meals."
    })
    info2 = r2.json()["structured_health_information"]
    negatives2 = [n.lower() for n in info2["negative_symptoms"]]

    # Negatives must still persist completely!
    assert any("chest pain" in n for n in negatives2)
    assert any("breathing" in n for n in negatives2)
    assert any("dizziness" in n for n in negatives2)
    assert any("fainting" in n for n in negatives2)


# ==============================================================================
# D. User Correction of Unsupported Inference
# ==============================================================================
def test_user_correction_overrides_unsupported_inference():
    """
    If an earlier turn or assistant inference assumed mild-to-moderate severity,
    and user says 'I did not say the pain was mild-to-moderate', the unsupported
    severity is wiped immediately.
    """
    session_id = "test-user-correction"
    user_id = "user-corr-1"
    session = session_manager.get_or_create(session_id, user_id)

    # Simulate an earlier turn where severity was set
    session.health_info.location = "right upper abdomen"
    session.health_info.severity = "mild-to-moderate"
    session.health_info.symptoms.append(Symptom(name="pain", severity="mild-to-moderate"))

    # User explicitly corrects the assumption
    resp = client.post("/api/chat", json={
        "user_id": user_id,
        "session_id": session_id,
        "message": "I did not say the pain was mild-to-moderate."
    })
    assert resp.status_code == 200
    info = resp.json()["structured_health_information"]

    # Unsupported severity must be removed/null
    assert info["severity"] is None
    assert info["symptoms"][0]["severity"] is None
    assert "severity" in session.corrected_facts


# ==============================================================================
# E. No Invented Severity (User-Source-of-Truth)
# ==============================================================================
def test_no_invented_severity():
    """Verifies severity remains null when user does not explicitly state it."""
    payload = {
        "user_id": "user-no-sev",
        "session_id": "sess-no-sev",
        "message": "I have upper-right abdominal pain."
    }
    resp = client.post("/api/chat", json=payload)
    assert resp.status_code == 200
    info = resp.json()["structured_health_information"]
    assert info["location"] == "right upper abdomen"
    assert info["severity"] is None
    assert info["symptoms"][0]["severity"] is None


def test_bounded_user_severity():
    """Verifies that non-standard user severity boundaries like 'not extremely severe' are preserved."""
    payload = {
        "user_id": "user-bound-sev",
        "session_id": "sess-bound-sev",
        "message": "I have pain in my abdomen. The pain is not extremely severe."
    }
    resp = client.post("/api/chat", json=payload)
    assert resp.status_code == 200
    info = resp.json()["structured_health_information"]
    assert info["severity"] == "not extremely severe"


# ==============================================================================
# F. No Invented Frequency (User-Source-of-Truth)
# ==============================================================================
def test_no_invented_frequency():
    """Verifies frequency remains null when user does not explicitly state it."""
    payload = {
        "user_id": "user-no-freq",
        "session_id": "sess-no-freq",
        "message": "I have pain below my right ribs."
    }
    resp = client.post("/api/chat", json=payload)
    assert resp.status_code == 200
    info = resp.json()["structured_health_information"]
    assert info["frequency"] is None
    assert info["symptoms"][0]["frequency"] is None


# ==============================================================================
# G. No Assistant-Generated Condition Becomes a Patient Condition
# ==============================================================================
def test_assistant_possibility_does_not_become_patient_condition():
    """
    If the LLM discusses possible causes like GERD or infection,
    they must NEVER be converted into known_conditions.
    """
    canned_llm_json = json.dumps({
        "assistant_message": "Possible causes could include gastrointestinal reflux (GERD) or an infection.",
        "intent": "SYMPTOM_ASSESSMENT",
        "extracted_info": {
            "symptoms": [{"name": "pain", "location": "right upper abdomen", "duration": None, "severity": None, "onset": None, "frequency": None, "associated_symptoms": []}],
            "location": "right upper abdomen",
            "duration": None,
            "severity": None,
            "onset": None,
            "frequency": None,
            "associated_symptoms": [],
            "medical_history": [],
            "medications": [],
            # Malicious/hallucinated leak of discussion into known_conditions:
            "known_conditions": ["GERD", "Infection"],
            "age": None,
            "sex": None,
            "measurements": []
        }
    })
    extraction_engine.llm_client = MockLLMClient(canned_response=canned_llm_json)

    resp = client.post("/api/chat", json={
        "user_id": "user-leak-check",
        "session_id": "sess-leak-check",
        "message": "I have pain in my right side."
    })
    assert resp.status_code == 200
    info = resp.json()["structured_health_information"]

    # Neither GERD nor Infection should leak into patient known conditions
    assert "GERD" not in info["known_conditions"]
    assert "Infection" not in info["known_conditions"]
    assert len(info["known_conditions"]) == 0


# ==============================================================================
# H. Complete Long-Message Extraction
# ==============================================================================
def test_complete_long_message_extraction():
    """
    Verifies that a complex narrative retains all explicit findings:
    symptoms, duration, triggers, relieving factors, negative findings, fluid, history, and test status.
    """
    long_message = (
        "I have been weak and tired for two days. I have reduced appetite, nausea, vomiting twice, "
        "feverish feeling, and dull upper-right abdominal pain. It gets worse after oily meals. "
        "Rest helps somewhat and water helps somewhat. I have been drinking 2 to 2.5 liters of water. "
        "My urine is somewhat darker but I am urinating normally. No blood in stool and no black/tarry stool. "
        "I have not noticed severe chest pain or difficulty breathing. No dizziness and no fainting. "
        "I had previous stomach problems several months ago and have had no recent medical tests."
    )
    resp = client.post("/api/chat", json={
        "user_id": "user-long-narrative",
        "session_id": "sess-long-narrative",
        "message": long_message
    })
    assert resp.status_code == 200
    info = resp.json()["structured_health_information"]

    # 1. Symptoms
    sym_names = [s["name"] for s in info["symptoms"]]
    assert "pain" in sym_names
    assert "nausea" in sym_names
    assert "vomiting" in sym_names
    assert "fever" in sym_names
    assert "fatigue" in sym_names
    assert "weakness" in sym_names
    assert "reduced appetite" in sym_names

    # 2. Location & Duration & Severity
    assert info["location"] == "right upper abdomen"
    assert "days" in info["duration"]
    assert info["severity"] == "dull"

    # 3. Triggers & Relieving Factors
    assert any("oily" in tr for tr in info["triggers"])
    assert any("rest" in rf for rf in info["relieving_factors"])
    assert any("water" in rf for rf in info["relieving_factors"])

    # 4. Negative Symptoms
    negatives = [n.lower() for n in info["negative_symptoms"]]
    assert any("chest pain" in n for n in negatives)
    assert any("breathing" in n for n in negatives)
    assert any("dizziness" in n for n in negatives)
    assert any("fainting" in n for n in negatives)
    assert any("blood in stool" in n for n in negatives)
    assert any("black/tarry stool" in n or "black or tarry" in n for n in negatives)

    # 5. Medical History & Previous Tests
    assert any("stomach" in h for h in info["medical_history"])
    assert any("no recent" in t.lower() for t in info["previous_tests"])

    # 6. Measurements (fluid intake)
    fluid_meas = next((m for m in info["measurements"] if m["name"] == "fluid_intake"), None)
    assert fluid_meas is not None
    assert fluid_meas["value"] == 2.25


# ==============================================================================
# I. Follow-Up Progression
# ==============================================================================
def test_follow_up_progression():
    """Verifies that follow-up questions systematically progress without cycling."""
    session_id = "sess-progression"
    user_id = "user-prog"

    # Turn 1: Symptom & Location provided -> Asks duration
    r1 = client.post("/api/chat", json={
        "user_id": user_id, "session_id": session_id,
        "message": "I have pain in my right ribs."
    })
    assert "duration" in r1.json()["follow_up_question"].lower() or "how long" in r1.json()["follow_up_question"].lower()

    # Turn 2: Duration provided -> Asks severity
    r2 = client.post("/api/chat", json={
        "user_id": user_id, "session_id": session_id,
        "message": "It started 2 days ago."
    })
    assert "severity" in r2.json()["follow_up_question"].lower() or "scale" in r2.json()["follow_up_question"].lower()

    # Turn 3: Severity provided -> Asks frequency
    r3 = client.post("/api/chat", json={
        "user_id": user_id, "session_id": session_id,
        "message": "It is moderate pain."
    })
    assert "constant" in r3.json()["follow_up_question"].lower() or "come and go" in r3.json()["follow_up_question"].lower()


# ==============================================================================
# J. Answer-First Behavior
# ==============================================================================
def test_answer_first_behavior():
    """
    When user asks 'What could potentially cause these symptoms and what information or tests would help a doctor?',
    assistant must answer the question first before offering follow-up.
    """
    session_id = "sess-answer-first"
    user_id = "user-af-1"

    # Seed initial context
    client.post("/api/chat", json={
        "user_id": user_id, "session_id": session_id,
        "message": "I have had dull upper-right abdominal pain and nausea for two days after oily meals."
    })

    # Explicit inquiry
    resp = client.post("/api/chat", json={
        "user_id": user_id, "session_id": session_id,
        "message": "What could potentially cause these symptoms and what information or tests would help a doctor?"
    })
    assert resp.status_code == 200
    msg = resp.json()["assistant_message"].lower()

    # 1. Acknowledges uncertainty & broad categories cautiously
    assert "cannot determine a diagnosis" in msg or "symptoms alone" in msg
    assert "hepatobiliary" in msg or "gallbladder" in msg or "liver" in msg
    assert "gastrointestinal" in msg or "gastritis" in msg or "digestive" in msg

    # 2. Explains information and tests (ultrasound, lab tests, palpation)
    assert "ultrasound" in msg
    assert "liver function" in msg or "lft" in msg or "blood test" in msg
    assert "physical" in msg or "palpation" in msg or "examination" in msg

    # 3. Avoids demanding food details again
    assert "tell me more about your eating habits" not in msg


def test_answer_first_exact_runtime_phrases_and_llm_bypass():
    """
    Verifies that real runtime Answer-First phrases (A, B, C, D) are detected by
    is_question_about_causes_or_what_helps() and bypass LLM generation, routing directly
    to ANSWER_FIRST with authoritative clinical content.
    """
    from app.extraction import is_question_about_causes_or_what_helps

    msg_a = "I am mainly concerned about what could potentially cause these symptoms and what information or tests would generally help a doctor understand what is happening."
    msg_b = "What kinds of causes might a doctor consider for this combination of symptoms, and what examinations or tests might help distinguish between them?"
    msg_c = "I already answered those questions. What other information would help?"
    msg_d = "Please answer that first, and then tell me if there is any important information I have not provided yet."
    full_turn = (
        "I have already explained that the discomfort is worse after heavy and oily meals. "
        "What I really want to understand is what kinds of causes a doctor might consider for this combination of symptoms, "
        "and what examinations or tests might help distinguish between them. "
        "Please answer that first, and then tell me if there is any important information I have not provided yet."
    )

    # 1. Detector unit checks
    assert is_question_about_causes_or_what_helps(msg_a) is True
    assert is_question_about_causes_or_what_helps(msg_b) is True
    assert is_question_about_causes_or_what_helps(msg_c) is True
    assert is_question_about_causes_or_what_helps(msg_d) is True
    assert is_question_about_causes_or_what_helps(full_turn) is True

    # 2. Integration check: Even if LLM would output irrelevant question, Answer-First deterministic path takes precedence
    session_id = "sess-af-bypass"
    user_id = "user-af-bypass"

    # Turn 1: Seed symptoms with normal extraction
    client.post("/api/chat", json={
        "user_id": user_id, "session_id": session_id,
        "message": "I have had dull pain in my right upper abdomen for two days after oily food."
    })

    # Set mock LLM with an unhelpful response to verify Answer-First completely bypasses it
    bad_llm_output = {
        "intent": "SYMPTOM_ASSESSMENT",
        "assistant_message": "Would you be willing to share more about your typical daily food intake and any recent stressors or changes you've experienced?",
        "follow_up_question": "Can you tell me about your daily food intake?",
        "extracted_info": {}
    }
    extraction_engine.llm_client = MockLLMClient(canned_response=json.dumps(bad_llm_output))

    # Turn 2: User asks runtime phrase B
    resp_b = client.post("/api/chat", json={
        "user_id": user_id, "session_id": session_id,
        "message": msg_b
    })
    assert resp_b.status_code == 200
    sess = session_manager.get_or_create(session_id, user_id)
    assert sess.last_routing == "ANSWER_FIRST"
    data_b = resp_b.json()
    msg_out = data_b["assistant_message"].lower()

    # Authoritative deterministic content
    assert "hepatobiliary" in msg_out or "gallbladder" in msg_out
    assert "ultrasound" in msg_out
    # Must NOT contain the bad LLM hallucination
    assert "recent stressors" not in msg_out
    assert "typical daily food intake" not in msg_out

    # Turn 3: Streaming check with full runtime turn
    stream_resp = client.post("/api/chat/stream", json={
        "user_id": user_id, "session_id": session_id,
        "message": full_turn
    })
    assert stream_resp.status_code == 200
    assert sess.last_routing == "ANSWER_FIRST"
    stream_text = stream_resp.text
    assert "hepatobiliary" in stream_text.lower() or "gallbladder" in stream_text.lower()
    assert "recent stressors" not in stream_text


# ==============================================================================
# K. Educational Query Filtering
# ==============================================================================
def test_educational_query_filtering():
    """Verifies definitional queries do not extract false personal symptoms."""
    for query in ["What is bilirubin?", "What is jaundice?", "Explain liver function tests."]:
        resp = client.post("/api/chat", json={
            "user_id": "user-edu", "session_id": f"sess-edu-{hash(query)}",
            "message": query
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["intent"] == IntentType.GENERAL_HEALTH_QUESTION
        assert len(data["structured_health_information"]["symptoms"]) == 0
        assert len(data["structured_health_information"]["known_conditions"]) == 0


# ==============================================================================
# L. Casual Conversation Scope & GK Redirection
# ==============================================================================
def test_casual_conversation_greeting():
    """Verifies natural friendly greeting without symptom interrogation."""
    resp = client.post("/api/chat", json={
        "user_id": "user-casual-1", "session_id": "sess-casual-1",
        "message": "Hii"
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "SmartHealth" in data["assistant_message"] or "Hello" in data["assistant_message"]
    assert len(data["structured_health_information"]["symptoms"]) == 0


def test_casual_conversation_introduction():
    """Verifies friendly self-introduction acknowledgment."""
    resp = client.post("/api/chat", json={
        "user_id": "user-casual-2", "session_id": "sess-casual-2",
        "message": "My name is Debraj and I'm a final-year CSE student."
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "Debraj" in data["assistant_message"] or "Hello" in data["assistant_message"]
    assert len(data["structured_health_information"]["symptoms"]) == 0


def test_out_of_scope_gk_redirect():
    """Verifies concise, natural healthcare boundary redirection on GK trivia queries."""
    resp = client.post("/api/chat", json={
        "user_id": "user-gk", "session_id": "sess-gk",
        "message": "Who is the Prime Minister of India?"
    })
    assert resp.status_code == 200
    data = resp.json()
    expected_redirect = "I'm primarily designed for health-related questions, so general GK isn't my main area. If you have a health question, symptom, report, or wellness concern, I'd be happy to help."
    assert data["assistant_message"] == expected_redirect
    assert len(data["structured_health_information"]["symptoms"]) == 0


# ==============================================================================
# M. Section 14 Benchmark: Context Preserved Across 4+ Turns
# ==============================================================================
def test_section_14_multiturn_benchmark():
    """
    Executes the exact multi-turn conversation from Section 14:
    Turn 1: Narrative symptom description with food trigger
    Turn 2: Water intake, dark urine, normal urination, stool negation
    Turn 3: Energy progression, fatigue, negative dizziness & fainting
    Turn 4: User asks 'I already answered those questions. What other information would help?'

    Verifies:
      - Does NOT re-ask about food, fluid, urine, stool, energy, or dizziness/fainting
      - Retains all positive & negative information across turns
      - Answers the user's request (Answer-First)
      - Selects only genuinely missing clinical information if asking a question
    """
    session_id = "sess-sec-14-benchmark"
    user_id = "user-debraj-14"

    # Turn 1
    t1_msg = "I have been weak and tired for two days. I have reduced appetite, nausea, vomiting twice, feverish feeling, and dull upper-right abdominal pain. It gets worse after oily meals."
    r1 = client.post("/api/chat", json={"user_id": user_id, "session_id": session_id, "message": t1_msg})
    assert r1.status_code == 200
    s1 = session_manager.get_or_create(session_id, user_id)
    assert "diet_food" in s1.answered_topics
    assert "energy_fatigue" in s1.answered_topics
    assert s1.health_info.severity == "dull"

    # Turn 2
    t2_msg = "I have been drinking 2 to 2.5 liters of water. My urine is somewhat darker but I'm urinating normally. No blood in stool and no black/tarry stool."
    r2 = client.post("/api/chat", json={"user_id": user_id, "session_id": session_id, "message": t2_msg})
    assert r2.status_code == 200
    s2 = session_manager.get_or_create(session_id, user_id)
    assert "fluid_intake" in s2.answered_topics
    assert "urine" in s2.answered_topics
    assert "stool" in s2.answered_topics
    neg2 = [n.lower() for n in s2.health_info.negative_symptoms]
    assert any("blood in stool" in n for n in neg2)
    assert any("black" in n for n in neg2)

    # Turn 3
    t3_msg = "My energy is lower today and the fatigue is worse than yesterday. I have no dizziness or fainting."
    r3 = client.post("/api/chat", json={"user_id": user_id, "session_id": session_id, "message": t3_msg})
    assert r3.status_code == 200
    s3 = session_manager.get_or_create(session_id, user_id)
    assert "dizziness_syncope" in s3.answered_topics
    neg3 = [n.lower() for n in s3.health_info.negative_symptoms]
    assert any("dizziness" in n for n in neg3)
    assert any("fainting" in n for n in neg3)

    # Turn 4
    t4_msg = "I already answered those questions. What other information would help?"
    r4 = client.post("/api/chat", json={"user_id": user_id, "session_id": session_id, "message": t4_msg})
    assert r4.status_code == 200
    d4 = r4.json()
    msg4 = d4["assistant_message"].lower()
    follow_up4 = (d4.get("follow_up_question") or "").lower()

    # 1. Answer-First: Directly explains what information and tests help a clinician
    assert "cannot determine a diagnosis" in msg4 or "symptoms alone" in msg4
    assert "hepatobiliary" in msg4 or "gallbladder" in msg4 or "liver" in msg4
    assert "ultrasound" in msg4 or "liver function" in msg4 or "lft" in msg4 or "blood" in msg4

    # 2. Strict prevention: NEVER re-ask about food, fluid, urine, stool, energy, dizziness, fainting
    for forbidden in [
        "fluid intake", "how much water", "urine output", "changes in your stool",
        "oily meals", "eating habits", "energy levels", "dizziness", "fainting"
    ]:
        assert forbidden not in follow_up4

    # 3. Retain complete multi-turn state
    final_info = d4["structured_health_information"]
    assert final_info["location"] == "right upper abdomen"
    assert "days" in final_info["duration"]
    assert final_info["severity"] == "dull"
    final_syms = [s["name"] for s in final_info["symptoms"]]
    assert "pain" in final_syms
    assert "nausea" in final_syms
    assert "vomiting" in final_syms
    assert "fatigue" in final_syms
    assert "weakness" in final_syms


# ==============================================================================
# N. Complete Two-Section Summary Benchmark
# ==============================================================================
def test_complete_two_section_summary():
    """
    Verifies that when asked 'Summarize everything I have told you',
    the assistant produces two clearly separated sections:
    A. USER-REPORTED INFORMATION
    B. NOT YET PROVIDED / STILL UNKNOWN
    with negative findings preserved and 'not provided' for unstated items.
    """
    session_id = "sess-summary-benchmark"
    user_id = "user-sum-1"

    # Populate session
    client.post("/api/chat", json={
        "user_id": user_id, "session_id": session_id,
        "message": "I have had dull pain in my right upper abdomen for two days. Worse after oily food. No blood in stool, and no dizziness."
    })

    resp = client.post("/api/chat", json={
        "user_id": user_id, "session_id": session_id,
        "message": "Summarize everything I have told you."
    })
    assert resp.status_code == 200
    msg = resp.json()["assistant_message"]

    # Must contain both distinct sections
    assert "### A. USER-REPORTED INFORMATION" in msg
    assert "### B. NOT YET PROVIDED / STILL UNKNOWN" in msg

    # Preserves user facts and negatives
    assert "Symptom:" in msg or "Reported Symptoms" in msg
    assert "right upper abdomen" in msg
    assert "dull" in msg
    assert "blood in stool" in msg or "Denied" in msg
    assert "dizziness" in msg

    # Section B identifies missing items with 'not provided', not 'no medications'
    assert "Current Medications: not provided" in msg
    assert "no medications" not in msg.lower()
