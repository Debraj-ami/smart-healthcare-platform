"""
Tests for Grounded Response Validation (Phase 9 Step 3.7-C Correction).

Verifies that the ResponseValidator deterministically blocks unsupported
patient-specific facts or unsupported clinical inferences, while preserving
general educational clinical statements and valid patient facts.
"""

import pytest
from app.schemas import (
    StructuredHealthInformation,
    Symptom,
    Measurement,
)
from app.conversation import ConversationState
from app.extraction import ExtractionEngine
from app.clinical_knowledge import ClinicalKnowledgeEngine
from app.response_validator import ResponseValidator, response_validator

extraction_engine = ExtractionEngine()


# ==============================================================================
# Required Test Cases (A through O)
# ==============================================================================

def test_a_verified_overall_duration():
    """
    A. Verified overall duration:
    User: "I've had this pain for two days."
    Generated: "Your symptoms have been present for two days."
    PASS
    """
    facts = extraction_engine.extract_health_information("I've had this pain for two days.")
    gen = "Your symptoms have been present for two days."
    supported = response_validator.extract_supported_facts(current_turn_extraction=facts)

    res = response_validator.validate(gen, supported)
    assert res.is_valid is True
    assert len(res.violations) == 0


def test_b_unknown_episode_duration():
    """
    B. Unknown episode duration:
    User: "I've had this pain for two days." (no episode duration specified)
    Generated: "Your pain lasts 30 minutes."
    FAIL
    """
    facts = extraction_engine.extract_health_information("I've had this pain for two days.")
    gen = "Your pain lasts 30 minutes."
    supported = response_validator.extract_supported_facts(current_turn_extraction=facts)

    res = response_validator.validate(gen, supported)
    assert res.is_valid is False
    assert any("EPISODE_DURATION" in v for v in res.violations)

    repaired = response_validator.validate_and_repair(gen, current_turn_extraction=facts)
    assert "lasts 30 minutes" not in repaired.lower()


def test_c_unknown_frequency():
    """
    C. Unknown frequency:
    User: "I have dull pain in the upper-right abdomen."
    Generated: "The episodic nature of your pain..."
    FAIL
    """
    facts = extraction_engine.extract_health_information("I have dull pain in the upper-right abdomen.")
    gen = "The episodic nature of your pain is important to note."
    supported = response_validator.extract_supported_facts(current_turn_extraction=facts)

    res = response_validator.validate(gen, supported)
    assert res.is_valid is False
    assert any("FREQUENCY" in v for v in res.violations)

    repaired = response_validator.validate_and_repair(gen, current_turn_extraction=facts)
    assert "episodic nature" not in repaired.lower()


def test_d_unknown_frequency_alternate_wording():
    """
    D. Unknown frequency alternate wording:
    User: "I have dull pain in the upper-right abdomen."
    Generated: "The intermittent pattern you described..."
    FAIL
    """
    facts = extraction_engine.extract_health_information("I have dull pain in the upper-right abdomen.")
    gen = "The intermittent pattern you described is noteworthy."
    supported = response_validator.extract_supported_facts(current_turn_extraction=facts)

    res = response_validator.validate(gen, supported)
    assert res.is_valid is False
    assert any("FREQUENCY" in v for v in res.violations)

    repaired = response_validator.validate_and_repair(gen, current_turn_extraction=facts)
    assert "intermittent pattern" not in repaired.lower()


def test_e_unknown_radiation():
    """
    E. Unknown radiation:
    User: "I have dull pain in the upper-right abdomen."
    Generated: "Your pain radiates to your back."
    FAIL
    """
    facts = extraction_engine.extract_health_information("I have dull pain in the upper-right abdomen.")
    gen = "Your pain radiates to your back."
    supported = response_validator.extract_supported_facts(current_turn_extraction=facts)

    res = response_validator.validate(gen, supported)
    assert res.is_valid is False
    assert any("RADIATION" in v for v in res.violations)

    repaired = response_validator.validate_and_repair(gen, current_turn_extraction=facts)
    assert "your pain radiates to your back" not in repaired.lower()


def test_f_verified_trigger():
    """
    F. Verified trigger:
    User: "The pain gets worse after heavy or oily meals."
    Generated: "Your pain worsens after oily meals."
    PASS
    """
    facts = extraction_engine.extract_health_information("The pain gets worse after heavy or oily meals.")
    gen = "Your pain worsens after oily meals."
    supported = response_validator.extract_supported_facts(current_turn_extraction=facts)

    res = response_validator.validate(gen, supported)
    assert res.is_valid is True
    assert len(res.violations) == 0


def test_g_general_clinical_statement():
    """
    G. General clinical statement:
    Generated: "Biliary conditions can sometimes cause episodic pain."
    PASS (general clinical education is allowed)
    """
    facts = extraction_engine.extract_health_information("I have stomach pain.")
    gen = "Biliary conditions can sometimes cause episodic pain."
    supported = response_validator.extract_supported_facts(current_turn_extraction=facts)

    res = response_validator.validate(gen, supported)
    assert res.is_valid is True
    assert len(res.violations) == 0


def test_h_patient_specific_unsupported_clinical_inference():
    """
    H. Patient-specific unsupported clinical inference:
    User: "I don't have chest pain."
    Generated: "The absence of chest pain makes a cardiac problem unlikely."
    FAIL
    """
    facts = extraction_engine.extract_health_information("I don't have chest pain.")
    gen = "The absence of chest pain makes a cardiac problem unlikely."
    supported = response_validator.extract_supported_facts(current_turn_extraction=facts)

    res = response_validator.validate(gen, supported)
    assert res.is_valid is False
    assert any("INFERENCE" in v for v in res.violations)

    repaired = response_validator.validate_and_repair(gen, current_turn_extraction=facts)
    assert "makes a cardiac problem unlikely" not in repaired.lower()


def test_i_verified_negative():
    """
    I. Verified negative:
    User: "I haven't noticed severe chest pain."
    Generated: "You haven't reported severe chest pain."
    PASS
    """
    facts = extraction_engine.extract_health_information("I haven't noticed severe chest pain.")
    gen = "You haven't reported severe chest pain."
    supported = response_validator.extract_supported_facts(current_turn_extraction=facts)

    res = response_validator.validate(gen, supported)
    assert res.is_valid is True
    assert len(res.violations) == 0


def test_j_unknown_jaundice():
    """
    J. Unknown jaundice:
    User: "I have abdominal pain." (never mentioned jaundice)
    Generated: "You don't have jaundice."
    FAIL (unknown symptom treated as verified negative)
    """
    facts = extraction_engine.extract_health_information("I have abdominal pain.")
    gen = "You don't have jaundice."
    supported = response_validator.extract_supported_facts(current_turn_extraction=facts)

    res = response_validator.validate(gen, supported)
    assert res.is_valid is False
    assert any("NEGATIVE" in v for v in res.violations)

    repaired = response_validator.validate_and_repair(gen, current_turn_extraction=facts)
    assert "you don't have jaundice" not in repaired.lower()


def test_k_unknown_temperature():
    """
    K. Unknown temperature:
    User: "I have a feverish feeling." (no measured numerical temperature)
    Generated: "Your temperature is elevated."
    FAIL
    """
    facts = extraction_engine.extract_health_information("I have a feverish feeling.")
    gen = "Your temperature is elevated."
    supported = response_validator.extract_supported_facts(current_turn_extraction=facts)

    res = response_validator.validate(gen, supported)
    assert res.is_valid is False
    assert any("TEMPERATURE" in v for v in res.violations)

    repaired = response_validator.validate_and_repair(gen, current_turn_extraction=facts)
    assert "your temperature is elevated" not in repaired.lower()


def test_l_subjective_feverish_feeling():
    """
    L. Subjective feverish feeling:
    User: "I have a feverish feeling."
    Generated: "You described feeling feverish."
    PASS
    """
    facts = extraction_engine.extract_health_information("I have a feverish feeling.")
    gen = "You described feeling feverish."
    supported = response_validator.extract_supported_facts(current_turn_extraction=facts)

    res = response_validator.validate(gen, supported)
    assert res.is_valid is True
    assert len(res.violations) == 0


def test_m_later_verified_frequency():
    """
    M. Later verified frequency:
    Turn 1: User says: "I have dull pain in the upper-right abdomen."
    Turn 2: User says: "The pain comes and goes."
    Generated: "Your pain comes and goes."
    PASS (also: "The intermittent pattern you described..." becomes valid)
    """
    session = ConversationState(session_id="test_val_m", user_id="u_m")
    t1_info = extraction_engine.extract_health_information("I have dull pain in the upper-right abdomen.")
    session.merge_health_info(t1_info)

    t2_info = extraction_engine.extract_health_information("The pain comes and goes.")
    session.merge_health_info(t2_info)

    supported = response_validator.extract_supported_facts(cumulative_session=session)

    # 1. "Your pain comes and goes."
    res1 = response_validator.validate("Your pain comes and goes.", supported)
    assert res1.is_valid is True

    # 2. "The intermittent pattern you described is helpful context."
    res2 = response_validator.validate("The intermittent pattern you described is helpful context.", supported)
    assert res2.is_valid is True


def test_n_later_verified_episode_duration():
    """
    N. Later verified episode duration:
    Turn 1: User says: "I have pain in the upper-right abdomen."
    Turn 2: User says: "Each episode lasts about 30 minutes."
    Generated: "Each episode lasts about 30 minutes."
    PASS
    """
    session = ConversationState(session_id="test_val_n", user_id="u_n")
    t1_info = extraction_engine.extract_health_information("I have pain in the upper-right abdomen.")
    session.merge_health_info(t1_info)

    t2_info = extraction_engine.extract_health_information("Each episode lasts about 30 minutes.")
    session.merge_health_info(t2_info)

    supported = response_validator.extract_supported_facts(cumulative_session=session)
    gen = "Each episode lasts about 30 minutes."

    res = response_validator.validate(gen, supported)
    assert res.is_valid is True
    assert len(res.violations) == 0


def test_o_overall_duration_vs_episode_duration():
    """
    O. Overall duration vs episode duration:
    User says: "I've had this pain for two days."
    Generated 1: "You've had the pain for two days." -> PASS
    Generated 2: "Each pain episode lasts two days." -> FAIL
    """
    facts = extraction_engine.extract_health_information("I've had this pain for two days.")
    supported = response_validator.extract_supported_facts(current_turn_extraction=facts)

    # Generated 1: Overall duration
    gen1 = "You've had the pain for two days."
    res1 = response_validator.validate(gen1, supported)
    assert res1.is_valid is True

    # Generated 2: Conflating overall duration with episode duration
    gen2 = "Each pain episode lasts two days."
    res2 = response_validator.validate(gen2, supported)
    assert res2.is_valid is False
    assert any("EPISODE_DURATION" in v for v in res2.violations)


# ==============================================================================
# Runtime Regression Tests (Sections 8 & 13)
# ==============================================================================

def test_runtime_regression_exact_multisymptom_sentence():
    """
    Regression Test: Exact Runtime Failure Sentence.
    User message:
    "I've been feeling weak and tired for the last two days. I have reduced appetite,
     nausea, vomiting twice, a feverish feeling, and dull pain in the upper-right
     side of my abdomen. The pain gets worse after heavy or oily meals. I can still
     move around normally, and the pain isn't extremely severe. I haven't noticed
     severe chest pain, difficulty breathing, fainting, or loss of consciousness.
     I haven't had any tests yet. I'm mainly worried about what could potentially
     be causing these symptoms and what information or tests would help a doctor
     understand what's going on."

    Generated response containing:
    "The episodic nature of your symptoms, with pain lasting 30-60 minutes, and the
     absence of severe chest pain or difficulty breathing, suggests that your symptoms
     may be related to a localized issue in the upper digestive tract."

    The validator MUST:
    1. Reject the response as invalid (detecting frequency, episode duration, and inference violations).
    2. Safely repair or replace it without inventing unverified facts.
    """
    user_msg = (
        "I've been feeling weak and tired for the last two days. I have reduced appetite, "
        "nausea, vomiting twice, a feverish feeling, and dull pain in the upper-right side of my abdomen. "
        "The pain gets worse after heavy or oily meals. I can still move around normally, and the pain isn't "
        "extremely severe. I haven't noticed severe chest pain, difficulty breathing, fainting, or loss of consciousness. "
        "I haven't had any tests yet. I'm mainly worried about what could potentially be causing these symptoms and what "
        "information or tests would help a doctor understand what's going on."
    )
    extracted = extraction_engine.extract_health_information(user_msg)
    session = ConversationState(session_id="test_reg_exact", user_id="u_reg_exact")
    session.merge_health_info(extracted, user_message=user_msg)

    llm_generated = (
        "The episodic nature of your symptoms, with pain lasting 30-60 minutes, and the absence "
        "of severe chest pain or difficulty breathing, suggests that your symptoms may be related "
        "to a localized issue in the upper digestive tract."
    )

    supported = response_validator.extract_supported_facts(cumulative_session=session, current_turn_extraction=extracted)
    res = response_validator.validate(llm_generated, supported)

    # 1. Must be rejected
    assert res.is_valid is False
    assert any("FREQUENCY" in v for v in res.violations)
    assert any("EPISODE_DURATION" in v for v in res.violations)
    assert any("INFERENCE" in v for v in res.violations)

    # 2. Must be safely repaired
    repaired = response_validator.validate_and_repair(
        generated_response=llm_generated,
        current_turn_extraction=extracted,
        cumulative_session=session
    )

    repaired_lower = repaired.lower()
    # Must NOT contain the unverified attributions
    assert "the episodic nature" not in repaired_lower
    assert "30-60 minutes" not in repaired_lower
    assert "suggests that your symptoms may be related to a localized issue" not in repaired_lower
    assert "suggests a localized" not in repaired_lower

    # Repaired text must pass validation
    post_check = response_validator.validate(repaired, supported)
    assert post_check.is_valid is True


def test_runtime_regression_negative_inference_digestive():
    """
    Regression Test: Unsupported Negative Inference.
    "The absence of severe chest pain or difficulty breathing suggests a localized digestive issue."
    The validator MUST reject and repair this unsupported clinical inference.
    """
    user_msg = "I have dull pain in the upper-right abdomen. I haven't noticed severe chest pain or difficulty breathing."
    extracted = extraction_engine.extract_health_information(user_msg)
    supported = response_validator.extract_supported_facts(current_turn_extraction=extracted)

    gen = "The absence of severe chest pain or difficulty breathing suggests a localized digestive issue."

    res = response_validator.validate(gen, supported)
    assert res.is_valid is False
    assert any("INFERENCE" in v for v in res.violations)

    repaired = response_validator.validate_and_repair(gen, current_turn_extraction=extracted)
    assert "suggests a localized digestive issue" not in repaired.lower()
    # Must provide safe factual negative finding reporting
    assert "severe chest pain" in repaired.lower() or "difficulty breathing" in repaired.lower()

    post_check = response_validator.validate(repaired, supported)
    assert post_check.is_valid is True


# ==============================================================================
# Critical Failure Investigation Regressions (Step 3.7-C Correction)
# ==============================================================================

def test_runtime_regression_unsupported_duration_and_radiation_cannot_self_support():
    """
    Section 9: Regression test that cannot self-support.
    Explicitly sets verified facts:
    frequency = None
    episode_duration = None
    radiation = None
    overall_duration = "two days"
    triggers = ["heavy/oily meals"]
    location = "upper-right abdomen"

    Then provide this exact generated response:
    "Based on your symptoms, the pain worsens after oily meals. The symptoms you described,
     with a 30-60 minute duration, and the back radiation to the right shoulder or upper back,
     are also relevant to gallbladder evaluation."

    The validator MUST NOT return a final response containing:
    - 30-60 minute duration as a patient fact
    - back radiation as a patient fact
    - shoulder radiation as a patient fact
    """
    from app.response_validator import SupportedPatientFacts
    supp = SupportedPatientFacts(
        symptoms={"pain"},
        location="upper-right abdomen",
        overall_duration="two days",
        triggers=["heavy/oily meals", "oily meals"],
        frequency=None,
        episode_duration=None,
        radiation=None
    )

    gen = (
        "Based on your symptoms, the pain worsens after oily meals. The symptoms you described, "
        "with a 30-60 minute duration, and the back radiation to the right shoulder or upper back, "
        "are also relevant to gallbladder evaluation."
    )

    # 1. Validation must fail on both episode duration and radiation
    res = response_validator.validate(gen, supp)
    assert res.is_valid is False
    assert any("EPISODE_DURATION" in v for v in res.violations)
    assert any("RADIATION" in v for v in res.violations)

    # 2. Repair must cleanly remove unverified attributions while preserving verified oily meals worsening
    repaired = response_validator.repair(gen, res.violations, supp)
    repaired_lower = repaired.lower()
    assert "30-60 minute" not in repaired_lower
    assert "back radiation to the right shoulder" not in repaired_lower
    assert "oily meals" in repaired_lower or "heavy" in repaired_lower

    # 3. Post-validation on repaired text must pass against the same verified facts
    post_check = response_validator.validate(repaired, supp)
    assert post_check.is_valid is True

    # 4. Also test: "The episodic nature of your symptoms..." must be removed/repaired
    gen_episodic = "The episodic nature of your symptoms is a key factor."
    res_ep = response_validator.validate(gen_episodic, supp)
    assert res_ep.is_valid is False
    repaired_ep = response_validator.repair(gen_episodic, res_ep.violations, supp)
    assert "episodic nature" not in repaired_ep.lower()


def test_section_10_general_knowledge_vs_patient_attribution():
    """
    Section 10: General clinical education vs patient attribution.
    Verified facts: frequency = None, radiation = None.
    Generated:
    "Some biliary conditions can cause episodic pain, and pain can sometimes radiate toward the back."
    -> MUST PASS.

    Generated:
    "Your episodic pain radiates to your back."
    -> MUST FAIL.
    """
    from app.response_validator import SupportedPatientFacts
    supp = SupportedPatientFacts(
        symptoms={"pain"},
        frequency=None,
        radiation=None
    )

    # General clinical education (allowed)
    gen_general = "Some biliary conditions can cause episodic pain, and pain can sometimes radiate toward the back."
    res_general = response_validator.validate(gen_general, supp)
    assert res_general.is_valid is True
    assert len(res_general.violations) == 0

    # Direct patient attribution of unverified attributes (blocked)
    gen_patient = "Your episodic pain radiates to your back."
    res_patient = response_validator.validate(gen_patient, supp)
    assert res_patient.is_valid is False
    assert any("FREQUENCY" in v for v in res_patient.violations)
    assert any("RADIATION" in v for v in res_patient.violations)


def test_section_11_legitimate_multi_turn_later_information():
    """
    Section 11: Multi-turn legitimate information retention.
    Turn 1: User: "I've had dull pain in my upper-right abdomen for two days."
    Turn 2: User: "The pain comes and goes. Each episode lasts around 30 minutes and sometimes goes toward my back."
    Then: "Your pain comes and goes and sometimes radiates toward your back." -> MUST PASS.
    """
    session = ConversationState(session_id="test_turn11_ret", user_id="u11")

    # Turn 1
    msg1 = "I've had dull pain in my upper-right abdomen for two days."
    info1 = extraction_engine.extract_health_information(msg1)
    session.merge_health_info(info1, user_message=msg1)

    supp1 = response_validator.extract_supported_facts(cumulative_session=session)
    assert supp1.frequency is None
    assert supp1.episode_duration is None
    assert supp1.radiation is None

    # Turn 2
    msg2 = "The pain comes and goes. Each episode lasts around 30 minutes and sometimes goes toward my back."
    info2 = extraction_engine.extract_health_information(msg2)
    session.merge_health_info(info2, user_message=msg2)

    supp2 = response_validator.extract_supported_facts(cumulative_session=session)
    assert supp2.frequency == "intermittent"
    assert supp2.episode_duration is not None and "30 minute" in supp2.episode_duration
    assert supp2.radiation is not None and "back" in supp2.radiation

    # Statement attributing the now-verified facts must pass
    gen = "Your pain comes and goes and sometimes radiates toward your back."
    res = response_validator.validate(gen, supp2)
    assert res.is_valid is True
    assert len(res.violations) == 0


def test_section_12_session_isolation_guarantee():
    """
    Section 12: Session Isolation.
    Ensures that health facts established in Session A do NOT leak into a fresh Session B.
    """
    from app.conversation import SessionManager
    sm = SessionManager()

    # Session A: user shares episodic pain lasting 30 minutes with radiation
    sess_a = sm.get_or_create("session_isolated_A", "user_A")
    info_a = extraction_engine.extract_health_information(
        "The pain comes and goes, each episode lasts 30 minutes, and radiates to my back."
    )
    sess_a.merge_health_info(info_a)

    supp_a = response_validator.extract_supported_facts(cumulative_session=sess_a)
    assert supp_a.frequency == "intermittent"
    assert supp_a.episode_duration is not None
    assert supp_a.radiation is not None

    # Session B: fresh session with new ID
    sess_b = sm.get_or_create("session_isolated_B", "user_B")
    supp_b = response_validator.extract_supported_facts(cumulative_session=sess_b)
    assert supp_b.frequency is None
    assert supp_b.episode_duration is None
    assert supp_b.radiation is None
    assert len(supp_b.symptoms) == 0


def test_section_13_clinical_language_safety_assertions():
    """
    Section 13: Clinical language safety and blocking unsupported diagnostic claims.
    """
    from app.response_validator import SupportedPatientFacts
    supp = SupportedPatientFacts(
        symptoms={"pain"},
        location="upper-right abdomen"
    )

    # 1. Definite diagnosis confirmation claim
    gen1 = "The symptoms confirm biliary colic."
    res1 = response_validator.validate(gen1, supp)
    assert res1.is_valid is False
    assert any("INFERENCE" in v for v in res1.violations)

    # 2. Definite condition indicator claim
    gen2 = "The pattern indicates gallstones."
    # "indicates" without hedging
    gen2_claim = "The absence of breathing difficulty means this is probably gastrointestinal."
    res2 = response_validator.validate(gen2_claim, supp)
    assert res2.is_valid is False
    assert any("INFERENCE" in v for v in res2.violations)

