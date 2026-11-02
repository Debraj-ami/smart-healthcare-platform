import pytest
from fastapi.testclient import TestClient
from app.main import app, extraction_engine, session_manager, grounded_synthesizer
from app.extraction import extract_negative_symptoms, ExtractionEngine
from app.llm_client import MockLLMClient
from app.conversation import is_symptom_negated

client = TestClient(app)
extractor = ExtractionEngine()


@pytest.fixture(autouse=True)
def reset_state():
    session_manager._sessions.clear()
    extraction_engine.llm_client = MockLLMClient()
    grounded_synthesizer.llm_client = MockLLMClient()
    yield
    session_manager._sessions.clear()
    extraction_engine.llm_client = MockLLMClient()
    grounded_synthesizer.llm_client = MockLLMClient()


class TestPaleStoolsNegativeExtraction:
    """
    Tests for Phase 9 targeted negative finding extraction:
    Pale Stools + Precise User Wording + Negation Scope Preservation.
    """

    def test_exact_runtime_test3_message_extracts_all_three_negatives(self):
        """
        Tests the exact message from Runtime Test 3:
        'I haven't noticed any yellowing of my eyes or skin, and my urine looks normal. I also haven't noticed unusually pale stools. Does that information change what a doctor might consider?'
        Must extract:
        1. yellowing of eyes/skin / jaundice
        2. urine looks normal
        3. unusually pale stools are not noticed
        And 0 positive symptoms.
        """
        msg = (
            "I haven't noticed any yellowing of my eyes or skin, and my urine looks normal. "
            "I also haven't noticed unusually pale stools. "
            "Does that information change what a doctor might consider?"
        )
        negatives = extract_negative_symptoms(msg)

        # 1. yellowing of eyes/skin
        assert any("jaundice" in n.lower() or "yellow" in n.lower() for n in negatives), (
            f"Missing jaundice negative in {negatives}"
        )
        # 2. urine looks normal
        assert any("urine" in n.lower() for n in negatives), (
            f"Missing urine negative in {negatives}"
        )
        # 3. unusually pale stools
        assert any("pale" in n.lower() and "stool" in n.lower() for n in negatives), (
            f"Missing pale stools negative in {negatives}"
        )
        # Verify user source wording preservation
        assert any("unusually pale stools" in n.lower() for n in negatives), (
            f"Expected user wording 'unusually pale stools' in {negatives}"
        )

        # Verify full deterministic extraction produces no positive symptoms
        info = extractor.extract_health_information(msg)
        assert len(info.symptoms) == 0, f"Expected 0 positive symptoms, got: {info.symptoms}"
        assert len(info.negative_symptoms) == 3

    @pytest.mark.parametrize(
        "variant,expected_phrase",
        [
            ("I haven't noticed unusually pale stools.", "unusually pale stools"),
            ("I haven't noticed pale stools.", "pale stools"),
            ("I don't have pale stools.", "pale stools"),
            ("My stools aren't unusually pale.", "unusually pale stools"),
            ("I haven't seen pale stools.", "pale stools"),
            ("I haven't noticed any unusually light-colored stools.", "unusually light-colored stools"),
            ("I haven't noticed pale or clay-colored stools.", "pale or clay-colored stools"),
            ("I have not noticed unusually pale bowel movements.", "unusually pale bowel movements"),
        ]
    )
    def test_all_eight_pale_stool_natural_variants(self, variant, expected_phrase):
        """Validates all 8 natural language variants of pale stool negative findings."""
        negatives = extract_negative_symptoms(variant)
        assert len(negatives) >= 1, f"Failed to extract negative finding from: {variant}"
        matched = [n for n in negatives if "pale" in n.lower() or "clay" in n.lower() or "light" in n.lower()]
        assert len(matched) >= 1, f"Expected pale stool negative in {negatives}"
        assert expected_phrase in matched[0].lower(), (
            f"Expected '{expected_phrase}' in '{matched[0]}'"
        )
        assert "no pale stools" in matched[0].lower(), (
            f"Expected normalized concept '(no pale stools)' in '{matched[0]}'"
        )

    def test_coordinated_negation_scope(self):
        """
        Coordinated denial across 'or':
        'I haven't noticed jaundice or pale stools.'
        Must extract both jaundice and pale stools as explicit negatives.
        """
        msg = "I haven't noticed jaundice or pale stools."
        negatives = extract_negative_symptoms(msg)
        assert any("jaundice" in n.lower() for n in negatives)
        assert any("pale" in n.lower() for n in negatives)

    def test_contrastive_conjunction_pale_stools_positive(self):
        """
        Contrastive negation:
        'I haven't noticed jaundice, but I have noticed pale stools.'
        Must extract:
        - jaundice = negative
        - pale stools = POSITIVE symptom (NOT negative)
        """
        msg = "I haven't noticed jaundice, but I have noticed pale stools."
        negatives = extract_negative_symptoms(msg)
        assert any("jaundice" in n.lower() for n in negatives)
        assert not any("pale" in n.lower() for n in negatives), (
            f"Pale stools must NOT be negative when affirmed after 'but': {negatives}"
        )

        info = extractor.extract_health_information(msg)
        sym_names = [s.name.lower() for s in info.symptoms]
        assert "pale stools" in sym_names, f"Expected 'pale stools' in positive symptoms: {sym_names}"

    def test_contrastive_conjunction_pale_stools_negative_dark_urine_positive(self):
        """
        Contrastive negation:
        'I don't have pale stools, but my urine is dark.'
        Must extract:
        - pale stools = explicit negative
        - dark urine = present / positive finding
        """
        msg = "I don't have pale stools, but my urine is dark."
        negatives = extract_negative_symptoms(msg)
        assert any("pale" in n.lower() for n in negatives)
        assert not any("urine" in n.lower() for n in negatives)

        info = extractor.extract_health_information(msg)
        sym_names = [s.name.lower() for s in info.symptoms]
        assert "dark urine" in sym_names, f"Expected 'dark urine' in positive symptoms: {sym_names}"

    @pytest.mark.parametrize(
        "non_negative_message",
        [
            "Have you noticed pale stools?",
            "I am worried about pale stools.",
            "Could pale stools be important?",
            "Should I look for pale stools?",
            "Some people with gallbladder problems have pale stools.",
            "I have upper-right abdominal pain.",
        ]
    )
    def test_questions_and_non_findings_remain_unknown(self, non_negative_message):
        """Questions, concerns, and general knowledge statements must NEVER become negative findings."""
        negatives = extract_negative_symptoms(non_negative_message)
        pale_negatives = [n for n in negatives if "pale" in n.lower() or "clay" in n.lower()]
        assert len(pale_negatives) == 0, (
            f"Incorrectly extracted pale stool negative from '{non_negative_message}': {pale_negatives}"
        )

    def test_end_to_end_multiturn_api_flow(self):
        """
        End-to-end integration test through the FastAPI endpoint simulating Runtime Test 3:
        Turn 1: User reports abdominal pain, nausea, vomiting, feverish feeling.
        Turn 2: User responds with:
          'I haven't noticed any yellowing of my eyes or skin, and my urine looks normal. I also haven't noticed unusually pale stools. Does that information change what a doctor might consider?'
        Verifies:
        - current_turn_extraction contains exactly the 3 negative findings and NO positive symptoms.
        - cumulative structured_health_information retains Turn 1 symptoms AND incorporates all 3 negative findings.
        - HTTP 200 response with safe clinical guidance.
        """
        session_id = "test-session-pale-stools-phase9"
        user_id = "test-user-pale-stools"

        # Turn 1
        t1_msg = (
            "I've been feeling weak and tired for two days with nausea, vomiting twice, "
            "a feverish feeling, and dull pain in the upper-right abdomen."
        )
        resp1 = client.post("/api/chat", json={
            "user_id": user_id,
            "session_id": session_id,
            "message": t1_msg
        })
        assert resp1.status_code == 200
        d1 = resp1.json()
        assert len(d1["structured_health_information"]["symptoms"]) > 0

        # Turn 2: Exact prompt from Runtime Test 3
        t2_msg = (
            "I haven't noticed any yellowing of my eyes or skin, and my urine looks normal. "
            "I also haven't noticed unusually pale stools. "
            "Does that information change what a doctor might consider?"
        )
        resp2 = client.post("/api/chat", json={
            "user_id": user_id,
            "session_id": session_id,
            "message": t2_msg
        })
        assert resp2.status_code == 200
        d2 = resp2.json()

        # Turn 2 current_turn_extraction
        curr = d2.get("current_turn_extraction")
        assert curr is not None
        assert len(curr["symptoms"]) == 0, f"Expected 0 symptoms in Turn 2, got: {curr['symptoms']}"
        curr_negs = [n.lower() for n in curr["negative_symptoms"]]
        assert any("jaundice" in n or "yellow" in n for n in curr_negs)
        assert any("urine" in n for n in curr_negs)
        assert any("unusually pale stools" in n for n in curr_negs)

        # Turn 2 cumulative state
        cum = d2.get("structured_health_information")
        assert cum is not None
        cum_negs = [n.lower() for n in cum["negative_symptoms"]]
        assert any("jaundice" in n or "yellow" in n for n in cum_negs)
        assert any("urine" in n for n in cum_negs)
        assert any("unusually pale stools" in n for n in cum_negs)

        # Assistant response exists and is non-empty
        assert len(d2["assistant_message"]) > 0
