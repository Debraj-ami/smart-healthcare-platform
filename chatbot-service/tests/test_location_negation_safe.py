"""
Regression tests for Phase 9: Positive Location Extraction + Negation-Safe Location Handling.
Tests all required cases A through M and generic pain preservation.
"""

import pytest
from app.extraction import ExtractionEngine, extract_location, extract_negative_symptoms, is_symptom_negated
from app.response_validator import ResponseValidator


@pytest.fixture
def engine():
    return ExtractionEngine()


@pytest.fixture
def validator():
    return ResponseValidator()


def test_case_a_exact_runtime_test_1(engine):
    """
    A. Exact Runtime Test 1:
    User message with dull pain in upper right side of abdomen and denied severe chest pain.
    Must extract RUQ location, NOT chest.
    Positive symptoms must include pain.
    Negative symptoms must include severe chest pain and difficulty breathing.
    """
    msg = (
        "I've been feeling weak and feverish for the past two days. "
        "I have a dull pain in the upper right side of my abdomen, especially after eating heavy or oily food. "
        "I've also had nausea, occasional vomiting, reduced appetite, and I feel tired. "
        "I haven't noticed severe chest pain or difficulty breathing. "
        "What could potentially cause these symptoms, and what information or tests would help a doctor evaluate them?"
    )
    info = engine.extract_health_information(msg)

    # Location must be RUQ concept, NOT chest
    assert info.location == "right upper abdomen"
    assert info.location != "chest"

    # Pain must remain a positive symptom
    symptom_names = [s.name.lower() for s in info.symptoms]
    assert "pain" in symptom_names

    # Associated symptoms present
    assert any("nausea" in s for s in symptom_names)
    assert any("vomiting" in s for s in symptom_names)
    assert any("fever" in s for s in symptom_names)

    # Negative findings
    assert "severe chest pain" in info.negative_symptoms
    assert "difficulty breathing" in info.negative_symptoms


def test_case_b_upper_right_space(engine):
    """B. 'I have pain in my upper right abdomen.' -> Expected positive RUQ location."""
    msg = "I have pain in my upper right abdomen."
    info = engine.extract_health_information(msg)
    assert info.location == "right upper abdomen"
    assert any(s.name == "pain" for s in info.symptoms)


def test_case_c_upper_right_hyphen(engine):
    """C. 'I have pain in my upper-right abdomen.' -> Expected positive RUQ location."""
    msg = "I have pain in my upper-right abdomen."
    info = engine.extract_health_information(msg)
    assert info.location == "right upper abdomen"
    assert any(s.name == "pain" for s in info.symptoms)


def test_case_d_right_upper_quadrant(engine):
    """D. 'I have pain in my right upper quadrant.' -> Expected positive RUQ location."""
    msg = "I have pain in my right upper quadrant."
    info = engine.extract_health_information(msg)
    assert info.location == "right upper abdomen"
    assert any(s.name == "pain" for s in info.symptoms)


def test_case_e_denied_severe_chest_pain_no_location(engine):
    """E. 'I haven't noticed severe chest pain or difficulty breathing.' -> Expected: no positive chest location."""
    msg = "I haven't noticed severe chest pain or difficulty breathing."
    info = engine.extract_health_information(msg)
    assert info.location is None
    assert info.location != "chest"
    assert "severe chest pain" in info.negative_symptoms
    assert "difficulty breathing" in info.negative_symptoms


def test_case_f_contrastive_chest_denied_ruq_positive(engine):
    """F. 'I don't have chest pain, but I have pain in my upper right abdomen.' -> Expected: negative chest pain, positive RUQ location."""
    msg = "I don't have chest pain, but I have pain in my upper right abdomen."
    info = engine.extract_health_information(msg)
    assert info.location == "right upper abdomen"
    assert "chest pain" in info.negative_symptoms
    assert any(s.name == "pain" for s in info.symptoms)


def test_case_g_contrastive_abdomen_denied_chest_positive(engine):
    """G. 'I don't have abdominal pain, but I have chest pain.' -> Expected: negative abdominal pain, positive chest location."""
    msg = "I don't have abdominal pain, but I have chest pain."
    info = engine.extract_health_information(msg)
    assert info.location == "chest"
    assert "abdominal pain" in info.negative_symptoms
    assert any(s.name == "pain" for s in info.symptoms)


def test_case_h_abstract_question_no_location(engine):
    """H. 'Could chest pain be important?' -> Expected: no positive chest location, no negative chest location."""
    msg = "Could chest pain be important?"
    info = engine.extract_health_information(msg)
    assert info.location is None
    assert "chest pain" not in info.negative_symptoms
    assert "severe chest pain" not in info.negative_symptoms


def test_case_i_general_statement_no_location(engine):
    """I. 'Chest pain can have many causes.' -> Expected: no patient location."""
    msg = "Chest pain can have many causes."
    info = engine.extract_health_information(msg)
    assert info.location is None
    assert info.location != "chest"


def test_case_j_denied_pain_in_chest_abdomen_positive(engine):
    """J. 'I haven't noticed pain in my chest, but I have pain in my abdomen.' -> Expected: negative chest pain, positive abdomen location."""
    msg = "I haven't noticed pain in my chest, but I have pain in my abdomen."
    info = engine.extract_health_information(msg)
    assert info.location == "abdomen"
    assert "chest pain" in info.negative_symptoms
    assert any(s.name == "pain" for s in info.symptoms)


def test_case_k_contrastive_correction_ruq_positive_chest_rejected(engine):
    """K. 'My pain is in the upper right side of my abdomen, not my chest.' -> Expected: positive RUQ/upper-right abdomen, chest must NOT become positive."""
    msg = "My pain is in the upper right side of my abdomen, not my chest."
    info = engine.extract_health_information(msg)
    assert info.location == "right upper abdomen"
    assert info.location != "chest"
    assert any(s.name == "pain" for s in info.symptoms)


def test_case_l_denied_severe_chest_pain_only(engine):
    """L. 'I don't have severe chest pain.' -> Expected: negative chest pain, no positive chest location."""
    msg = "I don't have severe chest pain."
    info = engine.extract_health_information(msg)
    assert info.location is None
    assert info.location != "chest"
    assert "severe chest pain" in info.negative_symptoms


def test_case_m_positive_severe_chest_pain(engine):
    """M. 'I have severe chest pain.' -> Expected: positive chest location."""
    msg = "I have severe chest pain."
    info = engine.extract_health_information(msg)
    assert info.location == "chest"
    assert any(s.name == "pain" for s in info.symptoms)


def test_severe_chest_pain_denial_does_not_drop_generic_pain(engine):
    """
    Regression test proving that 'severe chest pain' denial
    does not cause generic 'pain' elsewhere in the same user message to be discarded.
    """
    msg = "I've had dull pain for two days. I haven't noticed severe chest pain."
    info = engine.extract_health_information(msg)
    assert any(s.name == "pain" for s in info.symptoms)
    assert "severe chest pain" in info.negative_symptoms
    assert info.location is None  # no positive chest location


def test_validator_chest_safeguard(engine, validator):
    """
    Validator safeguard: asserts violation if response claims patient has chest pain
    when chest was only denied or not reported.
    """
    msg = "I haven't noticed severe chest pain or difficulty breathing."
    facts = engine.extract_health_information(msg)
    supported = validator.extract_supported_facts(current_turn_extraction=facts)

    hallucinated_response = "You've described dull chest pain lasting two days."
    res = validator.validate(hallucinated_response, supported)
    assert res.is_valid is False
    assert any("LOCATION" in v for v in res.violations)
