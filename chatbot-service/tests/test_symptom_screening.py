"""
Automated unit and integration tests for PATH B: Symptom-First Screening Guidance.

Verifies:
1. Symptom-only conversation completes without requesting Liver laboratory values.
2. Right upper abdominal pain produces a safe, structured screening summary.
3. No definitive medical diagnosis language is present.
4. No fabricated numerical disease probability is present.
5. Emergency / red-flag safety behavior takes absolute precedence.
6. Multi-turn symptom state accumulates and remains intact.
7. Users explicitly requesting Liver report intake still enter Path C.
"""

import pytest
import re
from fastapi.testclient import TestClient
from app.main import app, session_manager
from app.schemas import NextActionType, SafetyStatus

client = TestClient(app)


def test_symptom_only_conversation_completes_without_requesting_liver_lab_values():
    """
    Requirement 1 & 5: A patient with only symptoms completes a conversational
    screening dialogue WITHOUT the assistant ever requesting bilirubin, SGPT, SGOT,
    alkaline phosphatase, albumin, or other laboratory values.
    """
    session_id = "test-session-symptom-only-1"
    user_id = "patient-101"

    # Turn 1: Patient reports symptoms without any laboratory reports
    resp1 = client.post("/api/chat", json={
        "user_id": user_id,
        "session_id": session_id,
        "message": "I have had an uncomfortable pain below my right ribs for two days."
    })
    assert resp1.status_code == 200
    data1 = resp1.json()
    assert data1["next_action"] == NextActionType.ASK_FOLLOW_UP
    # Must NOT demand lab values
    for forbidden in ["bilirubin", "sgpt", "sgot", "alkaline phosphatase", "albumin", "a/g ratio"]:
        assert forbidden not in data1["assistant_message"].lower()

    # Turn 2: Patient reports severity and frequency
    resp2 = client.post("/api/chat", json={
        "user_id": user_id,
        "session_id": session_id,
        "message": "It is moderate pain, around 5 out of 10, and it comes and goes."
    })
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2["next_action"] == NextActionType.ASK_FOLLOW_UP
    for forbidden in ["bilirubin", "sgpt", "sgot", "alkaline phosphatase", "albumin", "a/g ratio"]:
        assert forbidden not in data2["assistant_message"].lower()

    # Turn 3: Patient answers contextual question (no fever or jaundice)
    resp3 = client.post("/api/chat", json={
        "user_id": user_id,
        "session_id": session_id,
        "message": "No fever or yellowing of my eyes."
    })
    assert resp3.status_code == 200
    data3 = resp3.json()

    # Symptom screening completes
    msg = data3["assistant_message"]
    # Must NOT require laboratory values to complete
    assert "To proceed with the Liver Disease screening assessment, I still need" not in msg
    assert "Total Bilirubin (mg/dL)" not in msg

    # State verification: liver_assessment was NOT forced on the patient
    session = session_manager.get_or_create(session_id, user_id)
    assert session.liver_assessment is None


def test_right_upper_abdominal_pain_screening_summary_content():
    """
    Requirement 2: A right-upper-abdominal-pain example produces a rich,
    screening-oriented response covering reported symptoms, clinical associations,
    monitoring guidance, clinician recommendation, emergency warning, and disclaimer.
    """
    session_id = "test-session-ruq-screening"
    user_id = "patient-102"

    # Multi-turn symptom collection
    client.post("/api/chat", json={
        "user_id": user_id,
        "session_id": session_id,
        "message": "I have had pain below my right ribs for two days."
    })
    client.post("/api/chat", json={
        "user_id": user_id,
        "session_id": session_id,
        "message": "It is moderate pain, and it comes and goes."
    })
    resp = client.post("/api/chat", json={
        "user_id": user_id,
        "session_id": session_id,
        "message": "No fever or yellowing of eyes."
    })
    assert resp.status_code == 200
    data = resp.json()
    msg = data["assistant_message"]

    # 1. Summary of reported findings
    assert "pain" in msg.lower()
    assert "right ribs" in msg.lower() or "right upper abdomen" in msg.lower()
    assert "2 days" in msg.lower() or "two days" in msg.lower()

    # 2. Educational clinical associations (gallbladder, liver, GI tract)
    assert "gallbladder" in msg.lower()
    assert "liver" in msg.lower()
    assert "digestive" in msg.lower() or "gastrointestinal" in msg.lower()

    # 3. Important associated symptoms to monitor
    assert "jaundice" in msg.lower() or "yellowing" in msg.lower()
    assert "dark" in msg.lower() and "urine" in msg.lower()
    assert "fever" in msg.lower()

    # 4. Appropriate professional evaluation recommendation
    assert "physician" in msg.lower() or "healthcare professional" in msg.lower()
    assert "physical examination" in msg.lower() or "ultrasound" in msg.lower()

    # 5. Clear emergency / red-flag instructions
    assert "emergency" in msg.lower()

    # 6. Explicit disclaimer that this is NOT a diagnosis
    assert "NOT a medical diagnosis" in msg


def test_screening_summary_avoids_definitive_diagnosis_language():
    """
    Requirement 3 & 10: Ensures no definitive medical diagnostic claims exist
    in the symptom screening output.
    """
    session_id = "test-session-no-diagnosis"
    user_id = "patient-103"

    client.post("/api/chat", json={
        "user_id": user_id,
        "session_id": session_id,
        "message": "I have had pain below my right ribs for two days."
    })
    client.post("/api/chat", json={
        "user_id": user_id,
        "session_id": session_id,
        "message": "It is moderate pain, and it comes and goes."
    })
    resp = client.post("/api/chat", json={
        "user_id": user_id,
        "session_id": session_id,
        "message": "No fever or yellowing of eyes."
    })
    msg = resp.json()["assistant_message"]

    forbidden_diagnostic_claims = [
        "you have liver disease",
        "you have hepatitis",
        "this proves",
        "confirmed diagnosis",
        "i diagnose",
        "you are diagnosed with",
        "you have gallstones"
    ]
    for claim in forbidden_diagnostic_claims:
        assert claim not in msg.lower()

    # Verifies cautious screening language is used
    assert "cannot confirm a specific cause" in msg.lower() or "cannot confirm an underlying cause" in msg.lower() or "not a medical diagnosis" in msg.lower()


def test_screening_summary_contains_no_fabricated_disease_probability():
    """
    Requirement 4 & 9: Ensures symptom-only screening generates NO numerical
    disease probability or fabricated risk score.
    """
    session_id = "test-session-no-prob"
    user_id = "patient-104"

    client.post("/api/chat", json={
        "user_id": user_id,
        "session_id": session_id,
        "message": "I have had pain below my right ribs for two days."
    })
    client.post("/api/chat", json={
        "user_id": user_id,
        "session_id": session_id,
        "message": "It is moderate pain, and it comes and goes."
    })
    resp = client.post("/api/chat", json={
        "user_id": user_id,
        "session_id": session_id,
        "message": "No fever or yellowing of eyes."
    })
    data = resp.json()
    msg = data["assistant_message"]

    # No percentage probabilities (e.g., "75%", "0.85")
    assert not re.search(r"\b\d{1,2}(?:\.\d+)?%", msg)
    assert "probability:" not in msg.lower()
    assert "risk score:" not in msg.lower()

    # Assessment handoff result must be None (no fake calculation)
    if data["assessment_handoff"]:
        assert data["assessment_handoff"]["assessment_result"] is None


def test_emergency_red_flag_priority_over_symptom_workflow():
    """
    Requirement 5 & 7: Emergency red-flag symptoms take immediate precedence
    over standard symptom follow-up or screening guidance.
    """
    session_id = "test-session-emergency"
    user_id = "patient-105"

    resp = client.post("/api/chat", json={
        "user_id": user_id,
        "session_id": session_id,
        "message": "I have pain under my ribs, but also sudden severe crushing chest pain and I cannot breathe."
    })
    assert resp.status_code == 200
    data = resp.json()

    assert data["safety_status"] == SafetyStatus.URGENT_MEDICAL_EVALUATION
    assert data["next_action"] == NextActionType.EMERGENCY_ROUTING
    assert "EMERGENCY WARNING" in data["assistant_message"]
    assert "911" in data["assistant_message"]


def test_explicit_liver_report_request_enters_path_c():
    """
    Requirement 6 & 7: When a user explicitly requests a liver report / blood test
    assessment, they enter the existing Path C workflow.
    """
    session_id = "test-session-explicit-path-c"
    user_id = "patient-106"

    resp = client.post("/api/chat", json={
        "user_id": user_id,
        "session_id": session_id,
        "message": "I have my liver blood test results. I am 62 years old and male."
    })
    assert resp.status_code == 200
    data = resp.json()

    # Enters liver assessment workflow
    assert data["next_action"] == NextActionType.ASK_FOLLOW_UP
    session = session_manager.get_or_create(session_id, user_id)
    assert session.liver_assessment is not None
    assert session.liver_assessment.values["age"] == 62
    assert session.liver_assessment.values["gender"] == "Male"
    # Prompts for missing lab values
    assert "Total Bilirubin" in data["assistant_message"]


def test_multiturn_symptom_state_remains_intact():
    """
    Requirement 8: Multi-turn symptom state accumulates and remains intact
    across all conversational turns.
    """
    session_id = "test-session-state-intact"
    user_id = "patient-107"

    # Turn 1: Symptom & duration
    client.post("/api/chat", json={
        "user_id": user_id,
        "session_id": session_id,
        "message": "I have had stomach pain since yesterday."
    })
    session = session_manager.get_or_create(session_id, user_id)
    assert len(session.health_info.symptoms) >= 1
    assert session.health_info.symptoms[0].name.lower() == "pain"
    assert session.health_info.duration == "since yesterday"

    # Turn 2: Severity & frequency
    client.post("/api/chat", json={
        "user_id": user_id,
        "session_id": session_id,
        "message": "The pain is moderate, around 5 out of 10, and it is constant."
    })
    assert session.health_info.severity == "moderate"
    assert session.health_info.frequency == "constant"
    # Prior duration remains intact
    assert session.health_info.duration == "since yesterday"
