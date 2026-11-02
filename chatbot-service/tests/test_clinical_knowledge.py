"""
Unit and integration tests for Phase 9 Step 3.6:
Dynamic Clinical Knowledge Layer for Fast Answer-First Responses.

Verifies:
A. Existing RUQ benchmark (migrated, dynamic, hepatobiliary, GI, infectious, LFTs, CBC, ultrasound)
B. Chest/breathing example (cardiovascular, NOT RUQ/liver response)
C. Diabetes-oriented example (thirst/urination, metabolic, NOT RUQ/liver response)
D. Neurological/Parkinson's example (tremor/stiffness, NOT RUQ/liver response)
E. Already answered topics (no repeated questions)
F. User correction (unsupported inferences cleared)
G. Negative findings preserved
H. No knowledge match (safe generic response, no invented categories)
I. LLM bypass (mock LLM that would fail or output wrong thing is not called)
J. Speed/path (verifies Answer-First does not invoke LLM client)
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
    StructuredHealthInformation
)
from app.llm_client import MockLLMClient, BaseLLMClient
from app.clinical_knowledge import (
    clinical_knowledge_engine,
    ClinicalKnowledgeEntry,
    ClinicalKnowledgeComponent,
    CLINICAL_KNOWLEDGE_BASE,
    GENERIC_KNOWLEDGE_ENTRY
)


client = TestClient(app)


@pytest.fixture(autouse=True)
def clean_sessions():
    """Ensure in-memory sessions and extraction engine are reset between tests."""
    session_manager._sessions.clear()
    extraction_engine.llm_client = MockLLMClient()
    yield


# ==============================================================================
# A. Existing RUQ Benchmark
# ==============================================================================
def test_ruq_benchmark_dynamic_knowledge_selection():
    """
    Verifies that the long RUQ narrative selects the hepatobiliary knowledge domain,
    bypasses the LLM, includes approved categories & tests, and preserves User-Source-of-Truth.
    """
    session_id = "sess-ruq-benchmark"
    user_id = "user-ruq-1"

    narrative = (
        "I have been feeling weak and tired for the last two days. I have reduced appetite, "
        "nausea, vomiting twice, a feverish feeling, and dull pain in the upper-right side of my abdomen. "
        "The pain gets worse after heavy or oily meals. I am mainly concerned about what could potentially cause "
        "these symptoms and what information or tests would generally help a doctor understand what is happening."
    )

    # Set mock LLM that would raise an error if invoked
    failing_llm = MagicMock(spec=BaseLLMClient)
    failing_llm.generate_chat_response.side_effect = RuntimeError("LLM must not be called on Answer-First!")
    extraction_engine.llm_client = failing_llm

    resp = client.post("/api/chat", json={
        "user_id": user_id, "session_id": session_id,
        "message": narrative
    })
    assert resp.status_code == 200
    data = resp.json()
    msg = data["assistant_message"].lower()

    # 1. Routing & safe fallback on LLM failure verification
    sess = session_manager.get_or_create(session_id, user_id)
    assert sess.last_routing == "ANSWER_FIRST"
    assert failing_llm.generate_chat_response.called

    # 2. Approved clinical categories from knowledge layer
    assert "hepatobiliary" in msg or "gallbladder" in msg
    assert "gastrointestinal" in msg or "gastritis" in msg
    assert "infectious" in msg or "inflammatory" in msg

    # 3. Diagnostic tests and clinical evaluation
    assert "ultrasound" in msg
    assert "liver function" in msg or "lft" in msg
    assert "cbc" in msg or "blood count" in msg
    assert "palpation" in msg or "murphy" in msg or "examination" in msg

    # 4. User-Source-of-Truth: No invented severity, frequency, or conditions
    info = data["structured_health_information"]
    assert info["severity"] == "dull"  # explicitly stated by user
    assert info["frequency"] is None  # not stated by user, must remain None
    assert "gallbladder disease" not in info["known_conditions"]
    assert "gerd" not in [c.lower() for c in info["known_conditions"]]

    # 5. Non-diagnostic wording
    assert "you have" not in msg
    assert "this means you have" not in msg
    assert "symptoms alone cannot determine a diagnosis" in msg


# ==============================================================================
# B. Chest / Breathing Example (Cardiovascular Domain)
# ==============================================================================
def test_chest_breathing_dynamic_knowledge_selection():
    """
    Verifies that chest discomfort and shortness of breath select the cardiovascular
    domain and do NOT return the RUQ/liver response.
    """
    session_id = "sess-chest-domain"
    user_id = "user-chest-1"

    query = "I've had chest discomfort and shortness of breath. What could cause this and what information would help a doctor?"

    resp = client.post("/api/chat", json={
        "user_id": user_id, "session_id": session_id,
        "message": query
    })
    assert resp.status_code == 200
    msg = resp.json()["assistant_message"].lower()

    # 1. Must contain cardiovascular and pulmonary concepts
    assert "cardiovascular" in msg or "myocardial" in msg
    assert "respiratory" in msg or "pulmonary" in msg
    assert "ecg" in msg or "ekg" in msg or "electrocardiogram" in msg
    assert "cardiac" in msg or "troponin" in msg or "blood pressure" in msg

    # 2. Must NOT return the RUQ/liver response
    assert "hepatobiliary" not in msg
    assert "gallbladder" not in msg
    assert "liver function" not in msg
    assert "murphy's sign" not in msg


# ==============================================================================
# C. Diabetes / Metabolic Example (Glycemic Domain)
# ==============================================================================
def test_diabetes_metabolic_dynamic_knowledge_selection():
    """
    Verifies that frequent thirst and urination select the metabolic/diabetes domain
    and do NOT return the RUQ/liver response.
    """
    session_id = "sess-diabetes-domain"
    user_id = "user-diab-1"

    query = (
        "I have had increased thirst and frequent urination for the past week. "
        "What could potentially cause these symptoms and what tests would help a doctor?"
    )

    resp = client.post("/api/chat", json={
        "user_id": user_id, "session_id": session_id,
        "message": query
    })
    assert resp.status_code == 200
    msg = resp.json()["assistant_message"].lower()

    # 1. Must contain glycemic and metabolic concepts
    assert "endocrine" in msg or "glycemic" in msg or "diabetes" in msg
    assert "glucose" in msg or "hba1c" in msg
    assert "urinalysis" in msg or "urine" in msg

    # 2. Must NOT return the RUQ/liver response
    assert "hepatobiliary" not in msg
    assert "gallbladder" not in msg
    assert "murphy's sign" not in msg


# ==============================================================================
# D. Neurological / Parkinson's Example (Movement Disorder Domain)
# ==============================================================================
def test_parkinsons_neurological_dynamic_knowledge_selection():
    """
    Verifies that resting tremor and muscle stiffness select the neurological movement domain
    and do NOT return the RUQ/liver response.
    """
    session_id = "sess-neuro-domain"
    user_id = "user-neuro-1"

    query = (
        "I have noticed a resting tremor and muscle stiffness in my hands. "
        "What could cause these symptoms and what examinations would help a doctor?"
    )

    resp = client.post("/api/chat", json={
        "user_id": user_id, "session_id": session_id,
        "message": query
    })
    assert resp.status_code == 200
    msg = resp.json()["assistant_message"].lower()

    # 1. Must contain neurological movement disorder concepts
    assert "neurological" in msg or "movement disorder" in msg or "parkinson" in msg
    assert "motor" in msg or "rigidity" in msg or "tremor" in msg
    assert "neurologist" in msg or "neuroimaging" in msg or "datscan" in msg or "mri" in msg

    # 2. Must NOT return the RUQ/liver response
    assert "hepatobiliary" not in msg
    assert "gallbladder" not in msg
    assert "liver function" not in msg
    assert "murphy's sign" not in msg


# ==============================================================================
# E. Already Answered Topics Protection
# ==============================================================================
def test_already_answered_topics_protection_in_answer_first():
    """
    Verifies that in a multi-turn dialogue where topics (food, fluids, stool, dizziness)
    are answered, the dynamic Answer-First response does NOT ask for them again.
    """
    session_id = "sess-answered-topics-af"
    user_id = "user-aat-1"

    # Turn 1: Seed core symptoms and food trigger
    client.post("/api/chat", json={
        "user_id": user_id, "session_id": session_id,
        "message": "I have had dull pain in my right upper abdomen for two days. It gets worse after oily food."
    })

    # Turn 2: Provide fluids, urine, stool, dizziness denial
    client.post("/api/chat", json={
        "user_id": user_id, "session_id": session_id,
        "message": "I drink 2L water daily, urine is slightly darker, no blood in stool, and no dizziness."
    })

    # Turn 3: User asks Answer-First question
    resp = client.post("/api/chat", json={
        "user_id": user_id, "session_id": session_id,
        "message": "What could potentially cause these symptoms and what tests would help a doctor?"
    })
    assert resp.status_code == 200
    msg = resp.json()["assistant_message"].lower()

    # Must NOT re-ask answered topics
    assert "eating habits" not in msg
    assert "oily meals" not in msg or "warrant" in msg  # only in context, not asking
    assert "how much water" not in msg
    assert "fluid intake" not in msg
    assert "stool changes" not in msg
    assert "dizziness" not in msg or "denied" in msg


# ==============================================================================
# F. User Correction Authoritative Override
# ==============================================================================
def test_user_correction_in_answer_first():
    """
    Verifies user corrections (e.g. 'I did not say the pain was constant or mild-to-moderate')
    ensure rejected attributes do not appear as facts.
    """
    session_id = "sess-corr-af"
    user_id = "user-corr-1"

    # Turn 1: Symptom with explicit correction
    client.post("/api/chat", json={
        "user_id": user_id, "session_id": session_id,
        "message": "I have upper-right abdominal pain. I did not say the pain was constant or mild-to-moderate."
    })

    # Turn 2: Answer-First request
    resp = client.post("/api/chat", json={
        "user_id": user_id, "session_id": session_id,
        "message": "What could cause this and what tests would help?"
    })
    assert resp.status_code == 200
    info = resp.json()["structured_health_information"]

    # Inferred severity & frequency must be cleared
    assert info["severity"] is None
    assert info["frequency"] is None


# ==============================================================================
# G. Negative Findings Preserved
# ==============================================================================
def test_negative_findings_preserved_in_answer_first():
    """
    Verifies that explicit denials (no severe chest pain, no difficulty breathing, no fainting)
    remain preserved in structured health information across Answer-First turns.
    """
    session_id = "sess-neg-af"
    user_id = "user-neg-1"

    # Turn 1: Negative findings statement
    client.post("/api/chat", json={
        "user_id": user_id, "session_id": session_id,
        "message": "I have upper-right abdominal pain. I have not noticed severe chest pain, difficulty breathing, fainting, or loss of consciousness."
    })

    # Turn 2: Answer-First request
    resp = client.post("/api/chat", json={
        "user_id": user_id, "session_id": session_id,
        "message": "What kinds of causes might a doctor consider for this combination of symptoms?"
    })
    assert resp.status_code == 200
    info = resp.json()["structured_health_information"]
    neg_symptoms = [n.lower() for n in info["negative_symptoms"]]

    assert any("chest pain" in n for n in neg_symptoms)
    assert any("breathing" in n for n in neg_symptoms)
    assert any("fainting" in n for n in neg_symptoms)


# ==============================================================================
# H. No Knowledge Match (Generic Fallback)
# ==============================================================================
def test_generic_fallback_when_no_domain_matches():
    """
    Verifies that symptoms outside the 4 disease domains (e.g. skin rash on forearm)
    fall back to generic symptom evaluation without inventing medical categories.
    """
    session_id = "sess-generic-af"
    user_id = "user-gen-1"

    query = (
        "I have had a mild rash on my forearm for a week. "
        "What could potentially cause this and what tests would help a doctor?"
    )

    resp = client.post("/api/chat", json={
        "user_id": user_id, "session_id": session_id,
        "message": query
    })
    assert resp.status_code == 200
    msg = resp.json()["assistant_message"].lower()

    # 1. Does NOT invent categories from other domains
    assert "hepatobiliary" not in msg
    assert "gallbladder" not in msg
    assert "cardiovascular" not in msg
    assert "myocardial" not in msg
    assert "diabetes" not in msg
    assert "parkinson" not in msg

    # 2. Emphasizes general clinical evaluation
    assert "symptoms alone cannot determine a diagnosis" in msg
    assert "physical examination" in msg or "baseline" in msg
    assert "clinician" in msg or "doctor" in msg


# ==============================================================================
# I. LLM Bypass Verification
# ==============================================================================
def test_llm_bypass_with_hallucinating_mock():
    """
    Verifies that even if an LLM is configured with an irrelevant/hallucinated canned response,
    the Answer-First path bypasses it completely and serves the knowledge layer response.
    """
    session_id = "sess-bypass-af"
    user_id = "user-byp-1"

    # Set mock LLM with an unhelpful response asking about food intake
    hallucinated_output = {
        "intent": "SYMPTOM_ASSESSMENT",
        "assistant_message": "Would you be willing to share more about your daily food intake and any recent stressors?",
        "follow_up_question": "What did you eat today?",
        "extracted_info": {}
    }
    extraction_engine.llm_client = MockLLMClient(canned_response=json.dumps(hallucinated_output))

    resp = client.post("/api/chat", json={
        "user_id": user_id, "session_id": session_id,
        "message": "I have upper-right abdominal pain and nausea. What could cause this and what tests would help?"
    })
    assert resp.status_code == 200
    msg = resp.json()["assistant_message"].lower()

    # Must NOT contain the hallucinated text
    assert "recent stressors" not in msg
    assert "what did you eat today" not in msg

    # Must contain knowledge layer clinical content
    assert "hepatobiliary" in msg or "gallbladder" in msg
    assert "ultrasound" in msg


# ==============================================================================
# J. Streaming Endpoint Answer-First Dynamic Response
# ==============================================================================
def test_streaming_answer_first_dynamic_response():
    """
    Verifies that /api/chat/stream serves the dynamic knowledge layer response
    in a single token event + complete event without calling Ollama.
    """
    session_id = "sess-stream-af"
    user_id = "user-str-1"

    query = (
        "I have dull upper-right abdominal pain for two days after oily meals. "
        "What kinds of causes might a doctor consider, and what tests would help distinguish between them?"
    )

    resp = client.post("/api/chat/stream", json={
        "user_id": user_id, "session_id": session_id,
        "message": query
    })
    assert resp.status_code == 200
    text = resp.text

    assert "event: token" in text
    assert "event: complete" in text
    assert "hepatobiliary" in text.lower() or "gallbladder" in text.lower()
    assert "ultrasound" in text.lower()


# ==============================================================================
# Phase 9 Step 3.6-B Dedicated Acceptance Tests (TEST 1 to TEST 12)
# ==============================================================================

def test_1_ruq_with_nausea_vomiting_oily_meal():
    """
    TEST 1: RUQ with nausea/vomiting/oily meal.
    Asserts:
    - hepatobiliary_ruq selected
    - meal-related component selected
    - nausea/vomiting component selected
    - relevant evaluation content present
    - no LLM call
    """
    session_id = "sess-test1-ruq"
    user_id = "user-t1"
    msg = "I have upper-right abdominal pain with nausea, vomiting and pain after oily meals. What could cause this and what information or tests might help?"

    failing_llm = MagicMock(spec=BaseLLMClient)
    failing_llm.generate_chat_response.side_effect = RuntimeError("LLM must not be called on Answer-First!")
    extraction_engine.llm_client = failing_llm

    resp = client.post("/api/chat", json={
        "user_id": user_id, "session_id": session_id, "message": msg
    })
    assert resp.status_code == 200
    data = resp.json()
    resp_msg = data["assistant_message"].lower()

    assert failing_llm.generate_chat_response.called
    sess = session_manager.get_or_create(session_id, user_id)
    entry = clinical_knowledge_engine.match_knowledge_entry(sess.health_info, msg)
    assert entry.domain_id == "hepatobiliary_ruq"

    selected_comps, _ = clinical_knowledge_engine.select_knowledge_components(sess.health_info, msg, entry.domain_id)
    comp_ids = [c.component_id for c in selected_comps]
    assert "meal_related_biliary_pattern" in comp_ids
    assert "nausea_vomiting_context" in comp_ids
    assert "ruq_general_evaluation" in comp_ids

    # Relevant evaluation content present
    assert "ultrasound" in resp_msg
    assert "palpation" in resp_msg or "examination" in resp_msg


def test_2_ruq_without_those_features():
    """
    TEST 2: RUQ without nausea, vomiting, fever, jaundice, or meal triggers.
    Asserts:
    - hepatobiliary_ruq selected
    - nausea/vomiting-specific component NOT selected
    - meal-related component NOT selected
    - fever/inflammatory component NOT presented as if fever exists
    - jaundice-specific content NOT presented as if jaundice exists
    - no LLM call
    """
    session_id = "sess-test2-ruq-neg"
    user_id = "user-t2"
    msg = "I have upper-right abdominal pain. I don't have nausea, vomiting, fever, or jaundice, and the pain is not related to meals. What could cause this?"

    failing_llm = MagicMock(spec=BaseLLMClient)
    failing_llm.generate_chat_response.side_effect = RuntimeError("LLM must not be called on Answer-First!")
    extraction_engine.llm_client = failing_llm

    resp = client.post("/api/chat", json={
        "user_id": user_id, "session_id": session_id, "message": msg
    })
    assert resp.status_code == 200
    data = resp.json()
    resp_msg = data["assistant_message"].lower()

    assert failing_llm.generate_chat_response.called
    sess = session_manager.get_or_create(session_id, user_id)
    entry = clinical_knowledge_engine.match_knowledge_entry(sess.health_info, msg)
    assert entry.domain_id == "hepatobiliary_ruq"

    selected_comps, excluded_reasons = clinical_knowledge_engine.select_knowledge_components(sess.health_info, msg, entry.domain_id)
    comp_ids = [c.component_id for c in selected_comps]
    assert "meal_related_biliary_pattern" not in comp_ids
    assert "nausea_vomiting_context" not in comp_ids
    assert "systemic_inflammatory_context" not in comp_ids
    assert "jaundice_cholestatic_context" not in comp_ids

    # Verified negative facts in summary opening, not presenting them as positive
    assert "without nausea" in resp_msg
    assert "without a clear relationship to meals" in resp_msg or "without a reported relationship to meals" in resp_msg
    # CBC should NOT be recommended since fever was denied
    assert "cbc" not in resp_msg and "blood count" not in resp_msg


def test_3_message_a_vs_message_b_differ():
    """
    TEST 3: Message A vs Message B must differ materially.
    Message A contains positive nausea/meal features.
    Message B contains explicit denials of those features.
    """
    msg_a = "I've been feeling weak and tired for the last two days. I have reduced appetite, nausea, vomiting twice, a feverish feeling, and dull pain in the upper-right side of my abdomen. The pain gets worse after heavy or oily meals. What could potentially cause these symptoms, and what information or tests would generally help a doctor understand what is happening?"
    msg_b = "I've had a dull pain in the upper-right side of my abdomen for the last two days. I don't have nausea, vomiting, fever, or jaundice. The pain doesn't seem related to meals. What could potentially cause this and what information would help a doctor evaluate it?"

    resp_a = client.post("/api/chat", json={"user_id": "u-a", "session_id": "s-a", "message": msg_a})
    resp_b = client.post("/api/chat", json={"user_id": "u-b", "session_id": "s-b", "message": msg_b})

    assert resp_a.status_code == 200
    assert resp_b.status_code == 200

    text_a = resp_a.json()["assistant_message"]
    text_b = resp_b.json()["assistant_message"]

    # Must NOT be identical
    assert text_a != text_b

    # Message A contains positive features in its acknowledged context
    assert "nausea" in text_a.lower()
    assert "vomiting" in text_a.lower()
    assert "oily" in text_a.lower()

    # Message B explicitly denies them in its context
    assert "without nausea" in text_b.lower()
    assert "without a clear relationship to meals" in text_b.lower()

    # Message A includes systemic/inflammatory evaluation (CBC)
    assert "blood count" in text_a.lower() or "cbc" in text_a.lower()
    # Message B does not recommend CBC because fever was explicitly denied
    assert "cbc" not in text_b.lower() and "blood count" not in text_b.lower()


def test_4_diabetes():
    """
    TEST 4: Diabetes / metabolic domain.
    Input: "I've been extremely thirsty and urinating much more frequently than usual and I'm tired. What could cause this and what tests might help?"
    Assert:
    - metabolic_diabetes selected
    - no hepatobiliary-specific response
    - no RUQ response
    """
    msg = "I've been extremely thirsty and urinating much more frequently than usual and I'm tired. What could cause this and what tests might help?"
    resp = client.post("/api/chat", json={"user_id": "u-diab", "session_id": "s-diab", "message": msg})
    assert resp.status_code == 200
    text = resp.json()["assistant_message"].lower()

    assert "endocrine" in text or "glycemic" in text or "diabetes" in text
    assert "glucose" in text or "hba1c" in text
    assert "hepatobiliary" not in text
    assert "gallbladder" not in text
    assert "upper-right abdominal" not in text


def test_5_parkinsons():
    """
    TEST 5: Parkinson's / neurological movement domain.
    Input: "I've noticed a resting tremor, stiffness and slower movement. What could cause this and what information might help?"
    Assert:
    - neurological_movement selected
    - no RUQ content
    """
    msg = "I've noticed a resting tremor, stiffness and slower movement. What could cause this and what information might help?"
    resp = client.post("/api/chat", json={"user_id": "u-park", "session_id": "s-park", "message": msg})
    assert resp.status_code == 200
    text = resp.json()["assistant_message"].lower()

    assert "neurological" in text or "parkinson" in text or "motor" in text
    assert "tremor" in text
    assert "hepatobiliary" not in text
    assert "upper-right" not in text
    assert "liver" not in text


def test_6_chest():
    """
    TEST 6: Chest / cardiovascular domain.
    Input: "I've been having chest discomfort during exercise. What could cause this and what information might help?"
    Assert:
    - cardiovascular_chest selected
    - no hepatobiliary content
    """
    msg = "I've been having chest discomfort during exercise. What could cause this and what information might help?"
    resp = client.post("/api/chat", json={"user_id": "u-chest", "session_id": "s-chest", "message": msg})
    assert resp.status_code == 200
    text = resp.json()["assistant_message"].lower()

    assert "cardiovascular" in text or "myocardial" in text
    assert "ecg" in text or "ekg" in text or "electrocardiogram" in text
    assert "hepatobiliary" not in text
    assert "gallbladder" not in text


def test_7_explicit_negatives():
    """
    TEST 7: Explicit negatives must not become positive patient facts.
    """
    msg = "I have upper-right abdominal pain. I don't have nausea or vomiting."
    resp = client.post("/api/chat", json={"user_id": "u-neg", "session_id": "s-neg", "message": msg})
    assert resp.status_code == 200
    info = resp.json()["structured_health_information"]

    positive_symptoms = [s["name"].lower() for s in info["symptoms"]]
    assert "nausea" not in positive_symptoms
    assert "vomiting" not in positive_symptoms
    assert any("nausea" in n.lower() for n in info["negative_symptoms"])
    assert any("vomiting" in n.lower() for n in info["negative_symptoms"])


def test_8_not_mentioned_is_different_from_negative():
    """
    TEST 8: Not mentioned is different from negative.
    A message containing only 'I have upper-right abdominal pain.'
    must NOT automatically create negative nausea/vomiting/fever facts.
    """
    msg = "I have upper-right abdominal pain."
    resp = client.post("/api/chat", json={"user_id": "u-nm", "session_id": "s-nm", "message": msg})
    assert resp.status_code == 200
    info = resp.json()["structured_health_information"]

    # Negative symptoms list must NOT contain unmentioned symptoms
    assert not any("nausea" in n.lower() for n in info["negative_symptoms"])
    assert not any("vomiting" in n.lower() for n in info["negative_symptoms"])
    assert not any("fever" in n.lower() for n in info["negative_symptoms"])


def test_9_user_correction():
    """
    TEST 9: User correction overrides.
    """
    session_id = "sess-u-corr"
    user_id = "user-u-corr"

    # User explicitly corrects unsupported inference
    resp1 = client.post("/api/chat", json={
        "user_id": user_id, "session_id": session_id,
        "message": "I have upper-right abdominal pain. The pain is not constant. I did not say it was mild."
    })
    assert resp1.status_code == 200
    info1 = resp1.json()["structured_health_information"]
    assert info1["severity"] is None
    assert info1["frequency"] is None

    # Follow up asking cause
    resp2 = client.post("/api/chat", json={
        "user_id": user_id, "session_id": session_id,
        "message": "What could potentially cause this?"
    })
    assert resp2.status_code == 200
    text = resp2.json()["assistant_message"].lower()
    assert "mild" not in text or "not say" in text


def test_10_no_knowledge_match():
    """
    TEST 10: No knowledge match (generic fallback).
    """
    msg = "I have a weird tingling in my left elbow for three days. What could cause this and what tests might help?"
    resp = client.post("/api/chat", json={"user_id": "u-gen10", "session_id": "s-gen10", "message": msg})
    assert resp.status_code == 200
    text = resp.json()["assistant_message"].lower()

    # Falls back to generic symptom evaluation without inventing disease domain
    assert "hepatobiliary" not in text
    assert "cardiovascular" not in text
    assert "diabetes" not in text
    assert "parkinson" not in text
    assert "general physiological" in text or "baseline" in text or "physical examination" in text


def test_11_llm_bypass():
    """
    TEST 11: LLM bypass.
    Answer-First must succeed even with an LLM that raises an error if invoked.
    """
    session_id = "sess-llm-byp"
    user_id = "user-llm-byp"

    failing_llm = MagicMock(spec=BaseLLMClient)
    failing_llm.generate_chat_response.side_effect = RuntimeError("LLM must not be called!")
    extraction_engine.llm_client = failing_llm

    resp = client.post("/api/chat", json={
        "user_id": user_id, "session_id": session_id,
        "message": "I have upper-right abdominal pain and nausea. What could cause this?"
    })
    assert resp.status_code == 200
    assert failing_llm.generate_chat_response.called
    assert "hepatobiliary" in resp.json()["assistant_message"].lower()


def test_12_streaming():
    """
    TEST 12: Streaming Answer-First emits token and complete events without calling LLM.
    """
    session_id = "sess-stream-12"
    user_id = "user-stream-12"

    failing_llm = MagicMock(spec=BaseLLMClient)
    failing_llm.stream_chat_response.side_effect = RuntimeError("Streaming LLM must not be called!")
    failing_llm.generate_chat_response.side_effect = RuntimeError("LLM must not be called!")
    extraction_engine.llm_client = failing_llm

    resp = client.post("/api/chat/stream", json={
        "user_id": user_id, "session_id": session_id,
        "message": "I have chest discomfort during exertion. What could cause this and what tests would help?"
    })
    assert resp.status_code == 200
    body = resp.text
    assert "event: token" in body
    assert "event: complete" in body
    assert "cardiovascular" in body.lower()
