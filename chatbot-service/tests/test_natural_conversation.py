"""
Unit and integration tests for Step 2 of Phase 9:
LLM-First Natural Healthcare Conversation.

Verifies:
1. Normal general health question uses LLM response.
2. Symptom message uses LLM-generated conversational response rather than deterministic questionnaire.
3. Long multi-sentence healthcare narrative reaches LLM path and returns natural conversational guidance.
4. Follow-up turn uses existing conversation history.
5. Educational question does not create false personal symptoms.
6. Personal symptom statement extracts legitimate symptoms.
7. Emergency safety check runs before LLM call and takes absolute priority.
8. Existing liver assessment handoff remains available.
9. Existing patient-context routing remains available.
10. Authoritative ML probability / SHAP values cannot be replaced or fabricated by LLM.
"""

import json
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from app.main import app, session_manager, extraction_engine
from app.schemas import (
    IntentType,
    SafetyStatus,
    NextActionType,
    Symptom,
    StructuredHealthInformation
)
from app.llm_client import MockLLMClient, BaseLLMClient


client = TestClient(app)


@pytest.fixture(autouse=True)
def clean_state():
    """Reset session state and extraction engine between tests."""
    session_manager._sessions.clear()
    extraction_engine.llm_client = MockLLMClient()
    yield
    session_manager._sessions.clear()
    extraction_engine.llm_client = MockLLMClient()


# ==============================================================================
# 1. Normal General Health Question Uses LLM Response
# ==============================================================================
def test_normal_general_health_question_uses_llm_response():
    """Verifies general health question receives conversational LLM output."""
    llm_payload = {
        "assistant_message": "Bilirubin is a compound formed during the breakdown of red blood cells. In typical health, the liver processes it and passes it through bile.",
        "intent": "GENERAL_HEALTH_QUESTION",
        "follow_up_question": None,
        "extracted_info": {
            "symptoms": [],
            "location": None,
            "duration": None,
            "severity": None,
            "onset": None,
            "frequency": None,
            "associated_symptoms": [],
            "medical_history": [],
            "medications": [],
            "known_conditions": [],
            "age": None,
            "sex": None,
            "measurements": []
        }
    }
    extraction_engine.llm_client = MockLLMClient(canned_response=json.dumps(llm_payload))

    response = client.post("/api/chat", json={
        "user_id": "user-nat-1",
        "session_id": "sess-nat-1",
        "message": "What is bilirubin?"
    })
    assert response.status_code == 200
    data = response.json()

    assert data["assistant_message"] == llm_payload["assistant_message"]
    assert data["intent"] == IntentType.GENERAL_HEALTH_QUESTION
    assert data["next_action"] == NextActionType.PROVIDE_GENERAL_INFO
    assert len(data["structured_health_information"]["symptoms"]) == 0


# ==============================================================================
# 2. Symptom Message Uses LLM-Generated Conversational Response Path
# ==============================================================================
def test_symptom_message_uses_llm_conversational_response():
    """
    Symptom conversations must use the LLM's conversational answer rather than
    overwriting it with a rigid deterministic questionnaire template.
    """
    conversational_text = (
        "Vomiting with fever can occur with several conditions, including gastrointestinal infections "
        "and systemic illnesses, though these symptoms alone cannot determine a specific diagnosis. "
        "To help evaluate what might be happening, could you tell me how many times you have vomited and whether you can keep fluids down?"
    )
    llm_payload = {
        "assistant_message": conversational_text,
        "intent": "SYMPTOM_ASSESSMENT",
        "follow_up_question": "Are you able to keep any fluids down?",
        "extracted_info": {
            "symptoms": [
                {"name": "vomiting", "location": None, "duration": "since morning", "severity": None, "onset": None, "frequency": None, "associated_symptoms": ["fever"]},
                {"name": "fever", "location": None, "duration": None, "severity": None, "onset": None, "frequency": None, "associated_symptoms": []}
            ],
            "location": None,
            "duration": "since morning",
            "severity": None,
            "onset": None,
            "frequency": None,
            "associated_symptoms": [],
            "medical_history": [],
            "medications": [],
            "known_conditions": [],
            "age": None,
            "sex": None,
            "measurements": []
        }
    }
    extraction_engine.llm_client = MockLLMClient(canned_response=json.dumps(llm_payload))

    response = client.post("/api/chat", json={
        "user_id": "user-nat-2",
        "session_id": "sess-nat-2",
        "message": "I've been vomiting since this morning and feel feverish."
    })
    assert response.status_code == 200
    data = response.json()

    # Must be the LLM's natural conversational response
    assert data["assistant_message"] == conversational_text
    # Must NOT be the rigid deterministic template
    assert "Thank you for sharing that information. To help understand your symptoms more clearly" not in data["assistant_message"]
    # Follow-up and structured health information are preserved
    assert data["follow_up_question"] == "Are you able to keep any fluids down?"
    symptom_names = [s["name"] for s in data["structured_health_information"]["symptoms"]]
    assert "vomiting" in symptom_names
    assert "fever" in symptom_names


# ==============================================================================
# 3. Long Multi-Sentence Symptom Paragraph Uses LLM Conversational Path
# ==============================================================================
def test_long_multisentence_symptom_narrative_uses_llm_conversational_path():
    """
    Verifies that a realistic long paragraph (multi-sentence detailed narrative)
    reaches the LLM path without truncation, extracts structured data, and responds
    with natural conversational synthesis instead of a questionnaire.
    """
    long_narrative = (
        "I have been vomiting since early this morning around 6 AM and now I am starting to feel feverish and warm to touch. "
        "I also have noticeable aching pain located right under my right ribs that feels tender when I move or breathe deeply. "
        "I have not been able to eat anything all day because the nausea is overwhelming, and I am feeling increasingly weak and lightheaded. "
        "I am particularly concerned because something similar happened about six months ago after a heavy meal, but this episode feels more intense and is not settling on its own. "
        "I am currently resting in bed and trying to sip water, but even small sips make me feel sick to my stomach."
    )

    llm_synthesis = (
        "Thank you for sharing those details. Experiencing vomiting since this morning alongside feverish symptoms, "
        "weakness, and localized discomfort under your right ribs can stem from several potential medical categories, "
        "including acute gastrointestinal infection or irritation involving the gallbladder or biliary tract. "
        "While these symptoms warrant careful attention, symptoms alone cannot provide a definitive diagnosis. "
        "Given that you are having difficulty keeping fluids down and feeling weak, staying hydrated is crucial. "
        "How frequently are you vomiting right now, and have you noticed any yellowing of your eyes or dark urine?"
    )

    llm_payload = {
        "assistant_message": llm_synthesis,
        "intent": "SYMPTOM_ASSESSMENT",
        "follow_up_question": "Have you noticed any yellowing of your eyes or dark urine?",
        "extracted_info": {
            "symptoms": [
                {"name": "vomiting", "location": None, "duration": "since morning", "severity": "intense", "onset": "sudden", "frequency": None, "associated_symptoms": ["fever", "pain", "fatigue"]},
                {"name": "fever", "location": None, "duration": None, "severity": None, "onset": None, "frequency": None, "associated_symptoms": []},
                {"name": "pain", "location": "right upper abdomen", "duration": "since morning", "severity": "moderate", "onset": None, "frequency": None, "associated_symptoms": []},
                {"name": "fatigue", "location": None, "duration": None, "severity": None, "onset": None, "frequency": None, "associated_symptoms": []}
            ],
            "location": "right upper abdomen",
            "duration": "since morning",
            "severity": "moderate",
            "onset": "sudden",
            "frequency": None,
            "associated_symptoms": ["fever", "pain", "fatigue"],
            "medical_history": ["similar episode six months ago"],
            "medications": [],
            "known_conditions": [],
            "age": None,
            "sex": None,
            "measurements": []
        }
    }

    mock_client = MockLLMClient(canned_response=json.dumps(llm_payload))
    extraction_engine.llm_client = mock_client

    response = client.post("/api/chat", json={
        "user_id": "user-long-1",
        "session_id": "sess-long-1",
        "message": long_narrative
    })
    assert response.status_code == 200
    data = response.json()

    # Verifies routing
    session = session_manager.get_or_create("sess-long-1", "user-long-1")
    assert session.last_routing == "LLM_PATH"

    # Verifies conversational response synthesis
    assert data["assistant_message"] == llm_synthesis
    assert "Thank you for sharing that information. To help understand your symptoms more clearly" not in data["assistant_message"]

    # Verifies structured symptoms extracted from long narrative
    extracted = data["structured_health_information"]
    symptom_names = [s["name"] for s in extracted["symptoms"]]
    assert "vomiting" in symptom_names
    assert "pain" in symptom_names
    assert extracted["location"] == "right upper abdomen"


# ==============================================================================
# 4. Follow-Up Turn Uses Conversation History
# ==============================================================================
def test_follow_up_turn_uses_conversation_history():
    """
    Turn 1: 'I've been having headaches for three days.'
    Turn 2: 'They are worse in the evening.'
    Verifies that the second turn prompt contains Turn 1 context from session history.
    """
    session_id = "sess-history-test"
    user_id = "user-history-test"

    # Turn 1
    t1_payload = {
        "assistant_message": "Headaches lasting three days can be related to tension, hydration, stress, or other factors. Could you share where the pain is located and what makes it better or worse?",
        "intent": "SYMPTOM_ASSESSMENT",
        "follow_up_question": "Where is the pain located and when is it most noticeable?",
        "extracted_info": {
            "symptoms": [{"name": "headache", "location": "head", "duration": "3 days", "severity": None, "onset": None, "frequency": None, "associated_symptoms": []}],
            "location": "head",
            "duration": "3 days",
            "severity": None,
            "onset": None,
            "frequency": None,
            "associated_symptoms": [],
            "medical_history": [],
            "medications": [],
            "known_conditions": [],
            "age": None,
            "sex": None,
            "measurements": []
        }
    }
    extraction_engine.llm_client = MockLLMClient(canned_response=json.dumps(t1_payload))

    resp1 = client.post("/api/chat", json={
        "user_id": user_id,
        "session_id": session_id,
        "message": "I've been having headaches for three days."
    })
    assert resp1.status_code == 200

    # Turn 2: Spy on the prompt passed to generate_chat_response
    t2_payload = {
        "assistant_message": "Noticing that your headaches worsen in the evening often points to eye strain, tension, or fatigue built up during the day. Do you also experience sensitivity to light or nausea?",
        "intent": "FOLLOW_UP_RESPONSE",
        "follow_up_question": "Do you also experience sensitivity to light or nausea?",
        "extracted_info": {
            "symptoms": [{"name": "headache", "location": "head", "duration": "3 days", "severity": "worse in evening", "onset": None, "frequency": "evening", "associated_symptoms": []}],
            "location": "head",
            "duration": "3 days",
            "severity": "worse in evening",
            "onset": None,
            "frequency": "evening",
            "associated_symptoms": [],
            "medical_history": [],
            "medications": [],
            "known_conditions": [],
            "age": None,
            "sex": None,
            "measurements": []
        }
    }

    captured_prompts = []
    real_generate = extraction_engine.llm_client.generate_chat_response

    class SpyingLLMClient(BaseLLMClient):
        def generate_chat_response(self, messages, system_prompt=None, temperature=0.7, max_tokens=None, response_format=None):
            captured_prompts.append(messages[0]["content"])
            return json.dumps(t2_payload)

    extraction_engine.llm_client = SpyingLLMClient()

    resp2 = client.post("/api/chat", json={
        "user_id": user_id,
        "session_id": session_id,
        "message": "They are worse in the evening."
    })
    assert resp2.status_code == 200

    # Verify that the captured prompt for Turn 2 contains Turn 1 conversation history
    assert len(captured_prompts) == 1
    t2_prompt = captured_prompts[0]
    assert "[RECENT CONVERSATION HISTORY]" in t2_prompt
    assert "I've been having headaches for three days" in t2_prompt
    assert "User Message: They are worse in the evening." in t2_prompt


# ==============================================================================
# 5. Educational Question Does Not Create False Personal Symptoms
# ==============================================================================
def test_educational_query_with_assistant_message_mentioning_jaundice():
    """
    Direct reproduction of the runtime bug:
    The user asks an educational question about bilirubin.
    The LLM generates an assistant message mentioning jaundice and includes jaundice
    in extracted_info.
    Verifies that the extraction engine and pipeline completely filter it out,
    ensuring assistant text never becomes a patient symptom.
    """
    llm_runtime_payload = {
        "assistant_message": (
            "Bilrubin is a yellow pigment produced during the breakdown of hemoglobin in the liver. "
            "It is measured in a blood test to assess liver function, detect liver disease, "
            "and monitor conditions like jaundice or hemolytic anemias."
        ),
        "intent": "GENERAL_HEALTH_QUESTION",
        "follow_up_question": None,
        "extracted_info": {
            "symptoms": [
                {"name": "jaundice", "location": None, "duration": None, "severity": None, "onset": None, "frequency": None, "associated_symptoms": []}
            ],
            "location": None,
            "duration": None,
            "severity": None,
            "onset": None,
            "frequency": None,
            "associated_symptoms": [],
            "medical_history": [],
            "medications": [],
            "known_conditions": [],
            "age": None,
            "sex": None,
            "measurements": []
        }
    }
    extraction_engine.llm_client = MockLLMClient(canned_response=json.dumps(llm_runtime_payload))

    response = client.post("/api/chat", json={
        "user_id": "user-edu-repro",
        "session_id": "sess-edu-repro",
        "message": "What is bilirubin and why is it measured in a blood test?"
    })
    assert response.status_code == 200
    data = response.json()

    # The assistant message is preserved and informative
    assert "jaundice" in data["assistant_message"].lower()
    # But jaundice is NEVER extracted as a patient symptom
    symptoms = data["structured_health_information"]["symptoms"]
    assert len(symptoms) == 0, f"Expected 0 symptoms, but found: {symptoms}"


@pytest.mark.parametrize("query", [
    "What is bilirubin?",
    "What is bilirubin and why is it measured?",
    "What is bilirubin and why is it measured in a blood test?",
    "What is jaundice?",
    "Can high bilirubin cause jaundice?",
    "Why do doctors test bilirubin?",
    "Why do doctors test for jaundice?",
    "Why is jaundice checked?",
    "What causes jaundice?",
    "How do you test for jaundice?",
    "Is jaundice dangerous?",
    "When is bilirubin checked?",
    "Can jaundice happen with liver disease?",
    "Explain what jaundice and bilirubin mean."
])
def test_educational_question_no_false_personal_symptoms(query):
    """Educational inquiries mentioning clinical terms must not extract personal symptoms."""
    response = client.post("/api/chat", json={
        "user_id": "user-edu-safe",
        "session_id": f"sess-edu-{hash(query)}",
        "message": query
    })
    assert response.status_code == 200
    data = response.json()

    symptoms = data["structured_health_information"]["symptoms"]
    assert len(symptoms) == 0, f"False personal symptoms extracted from query: '{query}'"


# ==============================================================================
# 6. Personal Symptom Statement Extracts Legitimate Symptoms
# ==============================================================================
@pytest.mark.parametrize("query,expected_symptoms", [
    ("I have jaundice and I feel weak.", ["jaundice"]),
    ("My eyes are yellow and my urine is dark.", ["jaundice"]),
    ("I was diagnosed with jaundice yesterday.", ["jaundice"]),
    ("I am experiencing jaundice.", ["jaundice"]),
    ("Since yesterday my eyes have become yellow.", ["jaundice"]),
    ("I have vomiting and fever.", ["vomiting", "fever"])
])
def test_personal_symptom_statement_extracts_symptom(query, expected_symptoms):
    """Personal statements describing symptoms must be extracted."""
    response = client.post("/api/chat", json={
        "user_id": "user-sym-test",
        "session_id": f"sess-sym-{hash(query)}",
        "message": query
    })
    assert response.status_code == 200
    data = response.json()

    symptom_names = [s["name"] for s in data["structured_health_information"]["symptoms"]]
    for exp in expected_symptoms:
        assert exp in symptom_names, f"Expected symptom '{exp}' not found in {symptom_names} for query: '{query}'"


# ==============================================================================
# 7. Emergency Safety Still Runs Before LLM
# ==============================================================================
def test_emergency_safety_runs_before_llm():
    """Red-flag emergency input triggers immediate urgent guidance without LLM call."""
    # Configure mock LLM to throw an exception if called
    failing_llm = MagicMock()
    failing_llm.generate_chat_response.side_effect = AssertionError("LLM should not be called on emergency!")
    extraction_engine.llm_client = failing_llm

    emergency_msg = "I have severe crushing chest pain and severe shortness of breath"
    response = client.post("/api/chat", json={
        "user_id": "user-emerg-1",
        "session_id": "sess-emerg-1",
        "message": emergency_msg
    })
    assert response.status_code == 200
    data = response.json()

    assert data["safety_status"] == SafetyStatus.URGENT_MEDICAL_EVALUATION
    assert data["next_action"] == NextActionType.EMERGENCY_ROUTING
    assert "emergency" in data["assistant_message"].lower() or "immediate" in data["assistant_message"].lower()
    # LLM was never invoked
    failing_llm.generate_chat_response.assert_not_called()


# ==============================================================================
# 8. Existing Liver Assessment Handoff Remains Available
# ==============================================================================
def test_existing_liver_assessment_handoff_remains_available():
    """
    When sufficient liver laboratory parameters are provided, the specialized
    liver assessment handoff remains functional and invokes the workflow.
    """
    response = client.post("/api/chat", json={
        "user_id": "user-liver-handoff",
        "session_id": "sess-liver-handoff",
        "message": "My total bilirubin is 10.9 mg/dL and albumin is 3.2 g/dL."
    })
    assert response.status_code == 200
    data = response.json()

    assert data["assessment_handoff"] is not None
    assert "liver_disease" in data["assessment_handoff"]["disease_candidates"]
    collected = data["assessment_handoff"]["structured_inputs"]
    assert collected["total_bilirubin"] == 10.9
    assert collected["albumin"] == 3.2


# ==============================================================================
# 9. Existing Patient-Context Routing Remains Available
# ==============================================================================
def test_existing_patient_context_routing_remains_available():
    """Context questions regarding prior readings continue routing to patient context."""
    with patch("app.main.get_backend_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.get_patient_health_context.return_value = {
            "patient_id": 4,
            "has_history": True,
            "trends": [
                {
                    "metricType": "GLUCOSE",
                    "currentValue": 115.0,
                    "previousValue": 112.0,
                    "baselineValue": 112.0,
                    "trendDirection": "STABLE"
                }
            ],
            "baselines": [],
            "recent_observations": []
        }
        mock_get_client.return_value = mock_client

        response = client.post("/api/chat", json={
            "user_id": "4",
            "session_id": "sess-context-1",
            "message": "How have my readings changed recently?"
        })
        assert response.status_code == 200
        data = response.json()

        assert data["intent"] == IntentType.MONITORING_QUERY
        assert mock_client.get_patient_health_context.call_count == 1


# ==============================================================================
# 10. LLM-Generated Response Does Not Replace Authoritative ML Probability / SHAP
# ==============================================================================
def test_llm_response_does_not_replace_authoritative_ml_probability():
    """
    Verifies that the authoritative ML pipeline remains the sole authority for
    probability and SHAP data; the LLM cannot fabricate or override ML predictions.
    """
    from app.backend_client import BackendClient, set_backend_client

    mock_backend = MagicMock(spec=BackendClient)
    mock_backend.predict_liver.return_value = {
        "assessmentId": 101,
        "probability": 0.616,
        "riskLevel": "ELEVATED",
        "primaryContributors": ["Total Bilirubin", "SGPT", "Direct Bilirubin"],
        "shapValues": {"totalBilirubin": 0.28, "sgpt": 0.15},
        "modelConfidence": "HIGH",
        "missingRequiredFields": []
    }
    set_backend_client(mock_backend)

    # Supply all 10 liver fields to complete assessment
    response = client.post("/api/chat", json={
        "user_id": "4",
        "session_id": "sess-ml-auth",
        "message": "Age 45, male. Total bilirubin 10.9, direct bilirubin 5.5, alkaline phosphatase 290, sgpt 64, sgot 100, total proteins 6.8, albumin 3.2, A/G ratio 0.9."
    })
    assert response.status_code == 200
    data = response.json()

    handoff = data["assessment_handoff"]
    assert handoff is not None
    assert handoff["assessment_result"] is not None
    # Must be the exact authoritative ML probability (0.616), not an LLM hallucination
    assert handoff["assessment_result"]["probability"] == 0.616
    assert handoff["assessment_result"]["riskLevel"] == "ELEVATED"
