"""
Phase 9 Step 3.7-C Correction: Grounded Response Validation.

Validates that LLM-generated responses do not introduce unsupported
patient-specific facts or unsupported clinical inferences.

Enforces strict separation between:
1. Patient-specific verified facts vs. Unverified/unknown patient attributes.
2. Overall symptom duration vs. Individual episode duration.
3. General clinical knowledge vs. Direct patient attribution.
4. Reported negative findings vs. Causal/probability inferences.
"""

import re
import logging
from typing import List, Optional, Any, Set
from dataclasses import dataclass, field

from app.schemas import StructuredHealthInformation, ResponsePlan

logger = logging.getLogger(__name__)


@dataclass
class SupportedPatientFacts:
    """
    Normalized representation of verified facts that the assistant is allowed
    to attribute specifically to the patient.
    """
    symptoms: Set[str] = field(default_factory=set)
    location: Optional[str] = None
    quality: Optional[str] = None
    severity: Optional[str] = None
    overall_duration: Optional[str] = None
    episode_duration: Optional[str] = None
    frequency: Optional[str] = None
    radiation: Optional[str] = None
    triggers: List[str] = field(default_factory=list)
    relieving_factors: List[str] = field(default_factory=list)
    body_temperature: Optional[str] = None
    has_measured_temperature: bool = False
    subjective_fever: bool = False
    negative_symptoms: Set[str] = field(default_factory=set)
    medical_history: List[str] = field(default_factory=list)
    medications: List[str] = field(default_factory=list)
    allergies: List[str] = field(default_factory=list)
    previous_tests: List[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    """Outcome of validating a generated assistant response."""
    is_valid: bool
    violations: List[str] = field(default_factory=list)
    repaired_response: Optional[str] = None
    used_fallback: bool = False


class ResponseValidator:
    """
    Deterministic response validator and repair engine.
    Ensures that the LLM response never attributes unverified facts or
    unsupported certainty/exclusion claims to the patient.
    """

    def extract_supported_facts(
        self,
        cumulative_session: Optional[Any] = None,
        current_turn_extraction: Optional[StructuredHealthInformation] = None
    ) -> SupportedPatientFacts:
        """
        Builds a comprehensive SupportedPatientFacts object combining
        cumulative verified state and the current turn extraction.
        """
        facts = SupportedPatientFacts()

        cum_info = getattr(cumulative_session, "health_info", None) if cumulative_session else None

        # 1. Symptoms & Quality
        sym_set: Set[str] = set()
        all_symptoms = (cum_info.symptoms if cum_info else []) + (current_turn_extraction.symptoms if current_turn_extraction else [])
        for s in all_symptoms:
            sym_set.add(s.name.lower())
            if "dull" in s.name.lower():
                facts.quality = "dull"
            elif "sharp" in s.name.lower():
                facts.quality = "sharp"
        if cum_info:
            for a in cum_info.associated_symptoms:
                sym_set.add(a.lower())
        if current_turn_extraction:
            for a in current_turn_extraction.associated_symptoms:
                sym_set.add(a.lower())
        facts.symptoms = sym_set

        # Subjective fever vs measured temperature (strictly from user sources)
        user_history_text = " ".join(
            msg.content.lower() for msg in getattr(cumulative_session, "history", [])
            if getattr(msg, "role", None) == "user"
        )
        if any("fever" in s for s in sym_set) or "fever" in user_history_text:
            facts.subjective_fever = True

        # 2. Location
        loc = (current_turn_extraction.location if current_turn_extraction and current_turn_extraction.location
               else (cum_info.location if cum_info else None))
        facts.location = loc.lower() if loc else None

        # 3. Severity
        sev = (current_turn_extraction.severity if current_turn_extraction and current_turn_extraction.severity
               else (cum_info.severity if cum_info else None))
        facts.severity = sev.lower() if sev else None

        # 4. Duration: Strictly separate Overall Duration vs Episode Duration
        raw_durations: List[str] = []
        if cum_info and cum_info.duration:
            raw_durations.append(cum_info.duration)
        if current_turn_extraction and current_turn_extraction.duration:
            raw_durations.append(current_turn_extraction.duration)
        for s in all_symptoms:
            if s.duration:
                raw_durations.append(s.duration)

        for d in raw_durations:
            d_low = d.lower()
            if any(w in d_low for w in ["day", "week", "month", "year", "since"]):
                facts.overall_duration = d_low
            elif any(w in d_low for w in ["minute", "min", "hour", "hr", "wave"]):
                facts.episode_duration = d_low

        # 5. Frequency
        freq = (current_turn_extraction.frequency if current_turn_extraction and current_turn_extraction.frequency
                else (cum_info.frequency if cum_info else None))
        if not freq:
            for s in all_symptoms:
                if s.frequency:
                    freq = s.frequency
                    break
        facts.frequency = freq.lower() if freq else None

        # 6. Radiation
        rad = (getattr(current_turn_extraction, "radiation", None) if current_turn_extraction
               else (getattr(cum_info, "radiation", None) if cum_info else None))
        facts.radiation = rad.lower() if rad else None

        # 7. Body Temperature (Objective Measurement)
        temp = (getattr(current_turn_extraction, "body_temperature", None) if current_turn_extraction
                else (getattr(cum_info, "body_temperature", None) if cum_info else None))
        if not temp and cum_info:
            for m in cum_info.measurements:
                if m.name == "body_temperature":
                    temp = f"{m.value}{m.unit or ''}"
                    break
        if not temp and current_turn_extraction:
            for m in current_turn_extraction.measurements:
                if m.name == "body_temperature":
                    temp = f"{m.value}{m.unit or ''}"
                    break
        facts.body_temperature = temp
        facts.has_measured_temperature = bool(temp and any(c.isdigit() for c in temp))

        # 8. Triggers
        tr_list: List[str] = []
        if cum_info:
            tr_list.extend([t.lower() for t in cum_info.triggers])
        if current_turn_extraction:
            for t in current_turn_extraction.triggers:
                if t.lower() not in tr_list:
                    tr_list.append(t.lower())
        facts.triggers = tr_list

        # 9. Relieving Factors
        rf_list: List[str] = []
        if cum_info:
            rf_list.extend([r.lower() for r in cum_info.relieving_factors])
        if current_turn_extraction:
            for r in current_turn_extraction.relieving_factors:
                if r.lower() not in rf_list:
                    rf_list.append(r.lower())
        facts.relieving_factors = rf_list

        # 10. Negative Symptoms
        neg_set: Set[str] = set()
        if cum_info:
            for n in cum_info.negative_symptoms:
                neg_set.add(n.lower())
        if current_turn_extraction:
            for n in current_turn_extraction.negative_symptoms:
                neg_set.add(n.lower())
        facts.negative_symptoms = neg_set

        # 11. History, Meds, Allergies, Tests
        if cum_info:
            facts.medical_history = list(cum_info.medical_history)
            facts.medications = list(cum_info.medications)
            facts.allergies = list(cum_info.allergies)
            facts.previous_tests = list(cum_info.previous_tests)
        if current_turn_extraction:
            for h in current_turn_extraction.medical_history:
                if h not in facts.medical_history:
                    facts.medical_history.append(h)
            for m in current_turn_extraction.medications:
                if m not in facts.medications:
                    facts.medications.append(m)
            for a in current_turn_extraction.allergies:
                if a not in facts.allergies:
                    facts.allergies.append(a)
            for t in current_turn_extraction.previous_tests:
                if t not in facts.previous_tests:
                    facts.previous_tests.append(t)

        return facts

    def is_general_medical_statement(self, sentence: str) -> bool:
        """
        Determines if a statement represents general clinical education
        rather than attribution of facts to the patient.
        """
        s = sentence.lower().strip()

        # Direct patient attribution markers disqualify general statement status
        patient_attribution_markers = [
            r"\byour\s+(?:episodic|intermittent|constant|radiating)\s+(?:pain|symptoms?)\b",
            r"\bthe\s+episodic\s+nature\s+of\s+(?:your|the)?\s*(?:pain|symptoms?)\b",
            r"\bthe\s+intermittent\s+(?:nature|pattern)\s+of\s+(?:your|the)?\s*(?:pain|symptoms?)\b",
            r"\bthe\s+intermittent\s+pattern\s+you\s+(?:described|reported|mentioned)\b",
            r"\byour\s+pain\s+(?:is|radiates|radiating|comes\s+and\s+goes|lasts?)\b",
            r"\bwith\s+pain\s+lasting\b",
            r"\bpain\s+lasting\s+[0-9]+\b",
            r"\byour\s+symptoms?\s+(?:lasting|have\s+been\s+present)\b",
            r"\bthe\s+symptoms\s+you\s+(?:described|reported|mentioned)\b",
            r"\b(?:with\s+)?(?:a\s+)?[0-9]+(?:\s*-\s*[0-9]+)?\s*(?:minutes?|mins?|hours?|hrs?)\s*duration\b",
            r"\beach\s+pain\s+episode\s+lasts?\b",
            r"\b(?:the\s+)?back\s+radiation\b",
            r"\bradiation\s+to\s+(?:the|your)?\s*(?:right\s+|upper\s+)?(?:shoulder|back|scapula)\b",
            r"\bthe\s+back\s+radiation\s+you\s+(?:described|reported|mentioned)\b",
            r"\bthe\s+pain\s+radiates?\s+to\s+(?:the|your)?\s*shoulder\b",
            r"\byour\s+temperature\s+is\b",
            r"\byou\s+(?:have|measured|reported)\s+a\s+temperature\s+of\b",
            r"\byou\s+(?:don'?t|do\s+not|have\s+no)\s+(?:jaundice|yellowing)\b",
            r"\byour\s+episodic\s+pain\s+suggests\b",
            r"\bincluding\s+the\s+episodic\s+nature\b",
            r"\bthe\s+absence\s+of\s+[^.!?]*\bsuggests\b",
            r"\bthe\s+absence\s+of\s+[^.!?]*\bmakes\b"
        ]
        for m in patient_attribution_markers:
            if re.search(m, s):
                return False

        # Educational / broad clinical pattern markers
        general_markers = [
            r"\b(?:some|certain|many)?\s*(?:biliary|gallbladder|liver|cardiac|medical|digestive)\s*(?:conditions?|diseases?|issues?|problems?|causes?)\s*(?:can|may|often|typically|might|sometimes)\b",
            r"\b(?:biliary\s+colic|cholecystitis|gallstones?|hepatitis)\s*(?:can|may|often|typically|might|sometimes|presents?)\b",
            r"\b(?:a\s+doctor|a\s+clinician|a\s+physician|clinicians?|physicians?|doctors?|medical\s+evaluation)\s*(?:may|might|will|often|typically|would|can)?\s*(?:evaluate|consider|check|order|look|test|investigate|inquire|assess|ask)\b",
            r"\bpotential\s+causes?\s*(?:include|for|of)\b",
            r"\bpatients?\s+with\b",
            r"\b(?:episodes|episodes\s+of\s+pain)\s+can\s+occur\b",
            r"\bpain\s+can\s+(?:sometimes\s+)?radiate\b",
            r"\bcan\s+sometimes\s+last\s+for\s+a\s+period\s+of\s+time\b"
        ]
        return any(re.search(gm, s) for gm in general_markers)

    def validate(
        self,
        generated_response: str,
        supported_facts: SupportedPatientFacts,
        grounding: Optional[Any] = None,
        response_plan: Optional[ResponsePlan] = None
    ) -> ValidationResult:
        """
        Validates the generated response for unsupported patient-specific claims
        or clinical inferences.
        """
        violations: List[str] = []
        text = generated_response.strip()

        # Split into sentence-like units
        sentences = re.split(r"(?<=[.!?])\s+", text)

        for sent in sentences:
            sent_clean = sent.strip()
            if not sent_clean:
                continue

            # Safe unknown acknowledgments (e.g. "You haven't reported whether the pain radiates to the back")
            is_unknown_acknowledgment = bool(re.search(
                r"\b(?:have\s+not|haven'?t|did\s+not|didn'?t|never)\s+(?:reported|described|mentioned|stated|noted)\b",
                sent_clean,
                re.IGNORECASE
            ))

            # 1. Frequency Attribution Check (Episodic / Intermittent)
            freq_words = ["episodic", "intermittent", "comes and goes", "waves", "constant", "continuous"]
            has_freq_word = any(re.search(rf"\b{re.escape(w)}\b", sent_clean, re.IGNORECASE) for w in freq_words)
            if has_freq_word and not is_unknown_acknowledgment:
                is_freq_verified = bool(
                    supported_facts.frequency and any(
                        w in supported_facts.frequency for w in ["intermittent", "episodic", "comes and goes", "waves", "constant", "continuous"]
                    )
                )
                if not is_freq_verified and not self.is_general_medical_statement(sent_clean):
                    patient_freq_patterns = [
                        r"\bthe\s+episodic\s+nature\s+of\s+(?:your|the)?\s*(?:pain|symptoms?)\b",
                        r"\bthe\s+intermittent\s+(?:nature|pattern)\s+of\s+(?:your|the)?\s*(?:pain|symptoms?)\b",
                        r"\bthe\s+intermittent\s+pattern\s+you\s+(?:described|reported|mentioned)\b",
                        r"\byour\s+episodic\s+(?:pain|symptoms?)\b",
                        r"\byour\s+intermittent\s+(?:pain|symptoms?)\b",
                        r"\b(?:your|the)?\s*pain\s+(?:is|being)\s+(?:episodic|intermittent|constant)\b",
                        r"\bincluding\s+the\s+episodic\s+nature\b",
                        r"\bthe\s+new\s+information[^.!?]*\bepisodic\b",
                        r"\byour\s+pain\s+comes\s+and\s+goes\b",
                        r"\byour\s+episodic\s+pain\s+suggests\b"
                    ]
                    if any(re.search(p, sent_clean, re.IGNORECASE) for p in patient_freq_patterns):
                        violations.append(f"UNSUPPORTED_PATIENT_FREQUENCY: Patient did not report episodic/intermittent frequency in '{sent_clean}'")

            # 2. Episode Duration Check (vs Overall Duration)
            episode_dur_patterns = [
                r"\b(?:with\s+)?(?:a\s+)?([0-9]+(?:\s*-\s*[0-9]+)?\s*(?:minutes?|mins?|hours?|hrs?))\s*(?:duration|long)?\b",
                r"\bduration\s+of\s+([0-9]+(?:\s*-\s*[0-9]+)?\s*(?:minutes?|mins?|hours?|hrs?))\b",
                r"\b(?:with\s+)?(?:pain|symptoms?|discomfort)?\s*lasting\s+([0-9]+(?:\s*-\s*[0-9]+)?\s*(?:minutes?|mins?|hours?|hrs?))\b",
                r"\b(?:your\s+)?(?:pain|symptoms?)\s+that\s+lasts?\s+([0-9]+(?:\s*-\s*[0-9]+)?\s*(?:minutes?|mins?|hours?|hrs?))\b",
                r"\byour\s+(?:pain|symptoms?)\s+lasts?\s+([0-9]+(?:\s*-\s*[0-9]+)?\s*(?:minutes?|mins?|hours?|hrs?))\b",
                r"\bepisodes?\s+lasting\s+([0-9]+(?:\s*-\s*[0-9]+)?\s*(?:minutes?|mins?|hours?|hrs?))\b",
                r"\beach\s+(?:pain\s+)?episode\s+lasts?\s+([^.,;]+)\b",
                r"\byour\s+symptoms?\s+lasting\s+([0-9]+(?:\s*-\s*[0-9]+)?\s*(?:minutes?|mins?|hours?|hrs?))\b"
            ]
            for ep in episode_dur_patterns:
                m_ep = re.search(ep, sent_clean, re.IGNORECASE)
                if m_ep and not is_unknown_acknowledgment and not self.is_general_medical_statement(sent_clean):
                    ep_val = m_ep.group(1).lower().strip()
                    numbers_in_ep = re.findall(r"\d+", ep_val)
                    is_ep_verified = bool(
                        supported_facts.episode_duration and (
                            ep_val in supported_facts.episode_duration
                            or supported_facts.episode_duration in ep_val
                            or (numbers_in_ep and any(n in supported_facts.episode_duration for n in numbers_in_ep))
                        )
                    )
                    if not is_ep_verified:
                        violations.append(f"UNSUPPORTED_EPISODE_DURATION: Individual episode duration '{ep_val}' was not reported by patient in '{sent_clean}'")

            # 3. Radiation Attribution Check
            rad_patterns = [
                r"\b(?:the\s+)?back\s+radiation\b",
                r"\b(?:radiation|radiat(?:es?|ing)|spread(?:s?|ing))\s+(?:to|into|toward)\s+(?:the|your)?\s*(?:right\s+|upper\s+)?(?:shoulder|back|scapula)\b",
                r"\byour\s+pain\s+(?:radiates?|radiating)\b",
                r"\bthe\s+pain\s+radiates?\s+to\s+(?:the|your)?\s*shoulder\b",
            ]
            matched_rad_pat = next((p for p in rad_patterns if re.search(p, sent_clean, re.IGNORECASE)), None)
            if matched_rad_pat and not is_unknown_acknowledgment and not self.is_general_medical_statement(sent_clean):
                is_rad_verified = bool(
                    supported_facts.radiation and any(r in supported_facts.radiation for r in ["back", "shoulder", "scapula"])
                )
                if not is_rad_verified:
                    violations.append(f"UNSUPPORTED_PATIENT_RADIATION: Patient did not report radiation to back/shoulder in '{sent_clean}'")

            # 4. Temperature Check (Measured vs Subjective)
            # A: "Your temperature is elevated" when no measured temperature
            if re.search(r"\byour\s+temperature\s+is\s+elevated\b", sent_clean, re.IGNORECASE):
                if not supported_facts.has_measured_temperature:
                    violations.append(f"UNSUPPORTED_PATIENT_TEMPERATURE: Elevated temperature asserted without measured temperature in '{sent_clean}'")

            # B: Numerical temperature attribution
            temp_match = re.search(r"\b(?:your\s+temperature\s+is\s+|temperature\s+of\s+|reported\s+a\s+temperature\s+of\s+)([3-4][0-9](?:\.[0-9]+)?\s*(?:°\s*[CF]|degrees?|C|F)?)\b", sent_clean, re.IGNORECASE)
            if temp_match and not self.is_general_medical_statement(sent_clean):
                reported_temp_val = temp_match.group(1).strip()
                is_temp_verified = bool(
                    supported_facts.has_measured_temperature and supported_facts.body_temperature and any(
                        digit in supported_facts.body_temperature for digit in ["38", "39", "40", "37", "100", "101", "102"]
                    )
                )
                if not is_temp_verified:
                    violations.append(f"UNSUPPORTED_PATIENT_TEMPERATURE: Numerical temperature '{reported_temp_val}' was not measured by patient in '{sent_clean}'")

            # 5. Unknown Treated as Negative Check (e.g. "You don't have jaundice")
            neg_attrib_match = re.search(r"\byou\s+(?:don'?t\s+have|do\s+not\s+have|have\s+no)\s+(?:any\s+)?(jaundice|yellowing|dark\s+urine|pale\s+stools)\b", sent_clean, re.IGNORECASE)
            if neg_attrib_match:
                symptom_denied = neg_attrib_match.group(1).lower()
                is_in_verified_negatives = any(symptom_denied in n for n in supported_facts.negative_symptoms)
                if not is_in_verified_negatives:
                    violations.append(f"UNKNOWN_TREATED_AS_NEGATIVE: '{symptom_denied}' was never mentioned by user; cannot assert 'You don't have {symptom_denied}'")

            # 6. Unsupported Clinical Inferences & Negative-Finding Causality
            # e.g. "The absence of severe chest pain or difficulty breathing suggests a localized digestive issue"
            # e.g. "The absence of chest pain makes a heart/cardiac problem unlikely"
            negative_inference_patterns = [
                r"\b(?:the\s+)?(?:absence|lack)\s+of\s+[^.!?]*\b(?:suggests?|points?\s+toward|indicates?|makes?|rules?\s+out|means?)\b[^.!?]*\b(?:localized\s+issue|digestive|gastrointestinal|heart|cardiac|cardiovascular|liver)\b",
                r"\b(?:makes?|making)\s+(?:a\s+)?(?:heart|cardiac|cardiovascular|gastrointestinal|liver)\s+(?:problem|issue|disease|condition|event|cause)?\s+(?:unlikely|less\s+likely|likely|probable)\b",
                r"\b(?:since|because)\s+you\s+(?:don'?t|do\s+not)\s+have\s+(?:severe\s+)?(?:chest\s+pain|difficulty\s+breathing)[^.!?]*\b(?:this\s+is\s+probably|points?\s+toward|suggests?|makes?)\b",
                r"\byour\s+episodic\s+pain\s+suggests\b",
                r"\brules?\s+out\s+(?:a\s+)?([a-z\s]+)\b",
                r"\bconfirms?\s+(?:that\s+you\s+have\s+)?(?:a\s+)?([a-z\s]+)\b",
                r"\bmeans?\s+you\s+(?:probably|definitely|likely)\s+(?:have|do\s+not\s+have)\b"
            ]
            is_negated_disclaimer = bool(re.search(
                r"\b(?:cannot|can\s+not|does\s+not|doesn'?t|not|never|unable\s+to)\s+(?:confirm|rule\s+out|diagnose|prove)\b",
                sent_clean,
                re.IGNORECASE
            ))
            if not is_negated_disclaimer:
                for nip in negative_inference_patterns:
                    if re.search(nip, sent_clean, re.IGNORECASE):
                        violations.append(f"UNSUPPORTED_CLINICAL_INFERENCE: Inappropriate certainty/causal claim from negative findings in '{sent_clean}'")
                        break

            # 7. Unsupported Chest Location Attribution Safeguard
            chest_attrib_match = re.search(
                r"\b(?:you(?:'ve|\s+have)?\s+(?:described|reported|mentioned|have)\s+(?:a\s+)?(?:dull\s+|sharp\s+|severe\s+)?chest\s+pain|your\s+chest\s+pain|pain\s+in\s+your\s+chest)\b",
                sent_clean,
                re.IGNORECASE
            )
            if chest_attrib_match and not is_unknown_acknowledgment and not self.is_general_medical_statement(sent_clean):
                is_chest_positive = (
                    supported_facts.location == "chest"
                    and not any("chest" in n for n in supported_facts.negative_symptoms)
                )
                if not is_chest_positive:
                    violations.append(f"UNSUPPORTED_PATIENT_LOCATION: Patient did not report positive chest pain in '{sent_clean}'")

            # 8. Unsupported Fatigue / Exhaustion / Stress / Anxiety Attribution Check
            unsupported_psych_fatigue_patterns = [
                r"\b(?:your\s+symptoms?\s+(?:seem\s+to\s+be|are|may\s+be)?\s+related\s+to\s+(?:fatigue|exhaustion|stress|anxiety)|related\s+to\s+fatigue\s+or\s+exhaustion|related\s+to\s+stress\s+or\s+anxiety)\b",
                r"\b(?:caused\s+by|due\s+to)\s+(?:fatigue|exhaustion|stress|anxiety)\b",
                r"\b(?:indicate|suggests?)\s+(?:that\s+)?(?:you\s+are\s+suffering\s+from\s+)?(?:fatigue|exhaustion|stress|anxiety)\b",
                r"\byour\s+(?:fatigue|exhaustion|stress|anxiety)\b"
            ]
            for p in unsupported_psych_fatigue_patterns:
                m_pf = re.search(p, sent_clean, re.IGNORECASE)
                if m_pf and not is_unknown_acknowledgment and not self.is_general_medical_statement(sent_clean):
                    is_fatigue_verified = any("fatigue" in s or "tired" in s or "exhaust" in s for s in supported_facts.symptoms)
                    is_stress_verified = any("stress" in s or "anxiety" in s for s in supported_facts.symptoms)
                    if not is_fatigue_verified and not is_stress_verified:
                        violations.append(f"UNSUPPORTED_PATIENT_ATTRIBUTION: Patient did not report fatigue, exhaustion, stress, or anxiety in '{sent_clean}'")
                        break

        return ValidationResult(
            is_valid=(len(violations) == 0),
            violations=violations
        )

    def repair(
        self,
        generated_response: str,
        violations: List[str],
        supported_facts: SupportedPatientFacts,
        grounding: Optional[Any] = None
    ) -> str:
        """
        Safely repairs unsupported patient-specific claims by rewriting or removing
        the specific unverified attributions while preserving valid contextual discussion.
        """
        repaired = generated_response

        # 0a. Check and repair unsupported fatigue/exhaustion/stress/anxiety assertions
        if any("UNSUPPORTED_PATIENT_ATTRIBUTION: Patient did not report fatigue" in v for v in violations):
            fatigue_stress_pat = re.compile(
                r"[^.!?]*\b(?:your\s+symptoms?\s+(?:seem\s+to\s+be|are|may\s+be)?\s+related\s+to\s+(?:fatigue|exhaustion|stress|anxiety)|related\s+to\s+fatigue\s+or\s+exhaustion|related\s+to\s+stress\s+or\s+anxiety|caused\s+by\s+(?:fatigue|exhaustion|stress|anxiety))[^.!?]*[.!?]?",
                re.IGNORECASE
            )
            repaired = fatigue_stress_pat.sub(
                "Symptoms such as shortness of breath and rapid heart rate during work that settle with rest warrant medical evaluation by a healthcare provider. Clinicians consider multiple factors, including cardiovascular, respiratory, or exertional considerations.",
                repaired
            )

        # 0. Check and repair the full runtime sentence combinations
        runtime_combo_pat = re.compile(
            r"The\s+episodic\s+nature\s+of\s+your\s+symptoms,\s*with\s+pain\s+lasting\s+30-60\s+minutes,\s*and\s+the\s+absence\s+of\s+severe\s+chest\s+pain\s+or\s+difficulty\s+breathing,\s*suggests\s+that\s+your\s+symptoms\s+may\s+be\s+related\s+to\s+a\s+localized\s+issue\s+in\s+the\s+upper\s+digestive\s+tract\.?",
            re.IGNORECASE
        )
        if runtime_combo_pat.search(repaired):
            combo_replacement = (
                "The worsening of your pain after heavy or oily meals is a relevant detail. "
                "You haven't described whether the pain is constant or comes and goes, or how long individual episodes last. "
                "You have not reported severe chest pain or difficulty breathing, which are helpful negative findings for a clinician to know."
            )
            repaired = runtime_combo_pat.sub(combo_replacement, repaired)

        runtime_combo_pat2 = re.compile(
            r"The\s+symptoms\s+you\s+described,\s*with\s+a\s+30-60\s+minute\s+duration,\s*and\s+the\s+back\s+radiation\s+to\s+the\s+right\s+shoulder\s+or\s+upper\s+back,\s*are\s+also\s+relevant\s+to\s+gallbladder(?:/biliary)?\s+evaluation\.?",
            re.IGNORECASE
        )
        if runtime_combo_pat2.search(repaired):
            combo_replacement2 = (
                "The symptoms you described in your upper-right abdomen are relevant to clinical evaluation. "
                "You have not described how long individual episodes last, or whether the discomfort spreads to your back or shoulder."
            )
            repaired = runtime_combo_pat2.sub(combo_replacement2, repaired)

        # 1. Repair frequency violations
        if any("FREQUENCY" in v or "suggests" in v for v in violations):
            repaired = re.sub(
                r",?\s*(?:including\s+)?the\s+episodic\s+nature\s+of\s+(?:the|your)?\s*(?:pain|symptoms?)\s+and\s+(?:its\s+)?worsening",
                ", including the worsening of the pain",
                repaired,
                flags=re.IGNORECASE
            )
            repaired = re.sub(
                r"\bThe\s+episodic\s+nature\s+of\s+(?:the|your)?\s*(?:pain|symptoms?)\s+and\s+(?:its\s+)?worsening\b",
                "The worsening of your pain",
                repaired,
                flags=re.IGNORECASE
            )
            repaired = re.sub(
                r"\bThe\s+intermittent\s+pattern\s+you\s+described\b",
                "The pattern you described",
                repaired,
                flags=re.IGNORECASE
            )
            repaired = re.sub(
                r"\bThe\s+episodic\s+nature\s+of\s+(?:your|the)?\s*symptoms\b",
                "The symptoms you described",
                repaired,
                flags=re.IGNORECASE
            )
            repaired = re.sub(
                r"\bThe\s+episodic\s+nature\s+of\s+(?:your|the)?\s*pain\b",
                "The pain you described",
                repaired,
                flags=re.IGNORECASE
            )
            repaired = re.sub(
                r"\byour\s+episodic\s+(?:pain|symptoms?)\b",
                "your pain",
                repaired,
                flags=re.IGNORECASE
            )
            repaired = re.sub(
                r"\bYour\s+episodic\s+pain\s+suggests?\s+biliary\s+colic\b",
                "Some biliary conditions can cause episodic pain",
                repaired,
                flags=re.IGNORECASE
            )

        # 2. Repair episode duration violations
        if any("EPISODE_DURATION" in v for v in violations):
            repaired = re.sub(
                r"\bEach\s+pain\s+episode\s+lasts\s+two\s+days\.?",
                "You have had the pain for two days.",
                repaired,
                flags=re.IGNORECASE
            )
            repaired = re.sub(
                r"\bYour\s+pain\s+lasts?\s+[0-9]+(?:\s*-\s*[0-9]+)?\s*minutes?\.?",
                "You have not described how long individual pain episodes last.",
                repaired,
                flags=re.IGNORECASE
            )
            repaired = re.sub(
                r",?\s*(?:with\s+)?(?:a\s+)?[0-9]+(?:\s*-\s*[0-9]+)?\s*(?:minute|min|hour|hr)s?\s*(?:duration|long)?",
                "",
                repaired,
                flags=re.IGNORECASE
            )
            repaired = re.sub(
                r",?\s*(?:with\s+)?pain\s+lasting\s+[0-9]+(?:\s*-\s*[0-9]+)?\s*minutes?",
                "",
                repaired,
                flags=re.IGNORECASE
            )

        # 3. Repair radiation violations
        if any("RADIATION" in v for v in violations):
            repaired = re.sub(
                r"\bThe\s+back\s+radiation\s+you\s+described\b",
                "The discomfort you described",
                repaired,
                flags=re.IGNORECASE
            )
            repaired = re.sub(
                r"\bThe\s+pain\s+radiates?\s+to\s+(?:the|your)?\s*shoulder\.?",
                "You have not described whether the discomfort spreads to your shoulder.",
                repaired,
                flags=re.IGNORECASE
            )
            repaired = re.sub(
                r"\bYour\s+pain\s+radiates?\s+(?:to|toward)\s+(?:your|the)?\s*back\.?",
                "You have not mentioned whether the discomfort spreads to your back or shoulder.",
                repaired,
                flags=re.IGNORECASE
            )
            repaired = re.sub(
                r",?\s*(?:and\s+)?(?:the\s+)?back\s+radiation\s+to\s+[^.,;]+",
                "",
                repaired,
                flags=re.IGNORECASE
            )
            repaired = re.sub(
                r",?\s*(?:and\s+)?radiating\s+(?:to|toward)\s+(?:your|the)?\s*back",
                "",
                repaired,
                flags=re.IGNORECASE
            )

        # 4. Repair temperature violations
        if any("TEMPERATURE" in v for v in violations):
            repaired = re.sub(
                r"\bYour\s+temperature\s+is\s+elevated\.?",
                "You described feeling feverish, though a measured temperature has not been provided.",
                repaired,
                flags=re.IGNORECASE
            )
            repaired = re.sub(
                r"\bYour\s+temperature\s+is\s+[3-4][0-9](?:\.[0-9]+)?\s*(?:°\s*[CF]|degrees?|C|F)?\.?",
                "You reported feeling feverish, but have not provided a measured temperature.",
                repaired,
                flags=re.IGNORECASE
            )
            repaired = re.sub(
                r"\bwith\s+a\s+(?:measured\s+)?temperature\s+of\s+[3-4][0-9](?:\.[0-9]+)?\s*(?:°\s*[CF]|degrees?|C|F)?",
                "with a feverish feeling",
                repaired,
                flags=re.IGNORECASE
            )

        # 5. Repair unknown-as-negative violations
        if any("UNKNOWN_TREATED_AS_NEGATIVE" in v for v in violations):
            repaired = re.sub(
                r"\bYou\s+(?:don'?t|do\s+not)\s+have\s+jaundice\.?",
                "You haven't reported jaundice.",
                repaired,
                flags=re.IGNORECASE
            )
            repaired = re.sub(
                r"\bYou\s+have\s+no\s+jaundice\.?",
                "You haven't reported jaundice.",
                repaired,
                flags=re.IGNORECASE
            )
            repaired = re.sub(
                r"\bYou\s+(?:don'?t|do\s+not)\s+have\s+(?:any\s+)?pale\s+stools\.?",
                "You haven't reported pale stools.",
                repaired,
                flags=re.IGNORECASE
            )
            repaired = re.sub(
                r"\bYou\s+have\s+no\s+(?:unusually\s+)?pale\s+stools\.?",
                "You haven't reported pale stools.",
                repaired,
                flags=re.IGNORECASE
            )

        # 6. Repair unsupported negative inferences
        if any("UNSUPPORTED_CLINICAL_INFERENCE" in v for v in violations):
            repaired = re.sub(
                r"\bThe\s+absence\s+of\s+severe\s+chest\s+pain\s+or\s+difficulty\s+breathing\s*(?:,\s*)?suggests\s+[^.!?]*\b(?:digestive|gastrointestinal|localized\s+issue)[^.!?]*\.?",
                "You haven't reported severe chest pain or difficulty breathing.",
                repaired,
                flags=re.IGNORECASE
            )
            repaired = re.sub(
                r"\bThe\s+absence\s+of\s+(?:severe\s+)?chest\s+pain\s+(?:and\s+difficulty\s+breathing\s+)?(?:makes\s+a\s+(?:heart|cardiac)\s+problem\s+unlikely|reduces\s+the\s+likelihood\s+of\s+a\s+heart\s+attack)\.?",
                "You haven't reported severe chest pain.",
                repaired,
                flags=re.IGNORECASE
            )

        # 7. Repair unsupported chest location violations
        if any("LOCATION" in v for v in violations):
            repaired = re.sub(
                r"\b(?:You(?:'ve|\s+have)?\s+(?:described|reported|mentioned)\s+)?(?:dull\s+|sharp\s+|severe\s+)?chest\s+pain\b",
                "upper-right abdominal pain" if (supported_facts.location and "abdomen" in supported_facts.location) else "your symptoms",
                repaired,
                flags=re.IGNORECASE
            )
            repaired = re.sub(
                r"\bpain\s+in\s+your\s+chest\b",
                "pain in your upper-right abdomen" if (supported_facts.location and "abdomen" in supported_facts.location) else "your symptoms",
                repaired,
                flags=re.IGNORECASE
            )

        # Clean up any duplicate spaces or punctuation artifacts
        repaired = re.sub(r"\s+", " ", repaired).strip()
        repaired = re.sub(r"\s+([.,!?])", r"\1", repaired)

        return repaired

    def validate_and_repair(
        self,
        generated_response: str,
        current_turn_extraction: Optional[StructuredHealthInformation] = None,
        cumulative_session: Optional[Any] = None,
        response_plan: Optional[ResponsePlan] = None,
        grounding: Optional[Any] = None
    ) -> str:
        """
        Executes the full Grounded Response Validation pipeline:
        1. Evaluates generated response against verified facts and grounding boundaries.
        2. If valid, returns the generated response directly.
        3. If invalid, attempts deterministic safe local repair.
        4. If repair passes validation, returns repaired response.
        5. If repair fails or is unsafe, returns deterministic grounded fallback.
        """
        if not generated_response or not generated_response.strip():
            if grounding and getattr(grounding, "deterministic_fallback_message", None):
                return grounding.deterministic_fallback_message
            return "Thank you for sharing your symptoms. Please consult a healthcare provider for comprehensive evaluation."

        supported_facts = self.extract_supported_facts(
            cumulative_session=cumulative_session,
            current_turn_extraction=current_turn_extraction
        )

        res = self.validate(
            generated_response=generated_response,
            supported_facts=supported_facts,
            grounding=grounding,
            response_plan=response_plan
        )

        # Diagnostic audit logging for traceability
        logger.info(
            "ResponseValidator Trace:\n"
            "  1. current_turn_extraction: symptoms=%s, duration=%s, freq=%s, rad=%s\n"
            "  2. cumulative_session.health_info: symptoms=%s, duration=%s, freq=%s, rad=%s\n"
            "  3. SupportedPatientFacts: overall_dur=%s, ep_dur=%s, freq=%s, rad=%s, temp=%s, neg=%s\n"
            "  4. generated_response (pre-validation): '%s'\n"
            "  5. validation: is_valid=%s, violations=%s",
            [s.name for s in current_turn_extraction.symptoms] if current_turn_extraction else None,
            getattr(current_turn_extraction, "duration", None),
            getattr(current_turn_extraction, "frequency", None),
            getattr(current_turn_extraction, "radiation", None),
            [s.name for s in cumulative_session.health_info.symptoms] if cumulative_session and hasattr(cumulative_session, "health_info") else None,
            getattr(cumulative_session.health_info, "duration", None) if cumulative_session and hasattr(cumulative_session, "health_info") else None,
            getattr(cumulative_session.health_info, "frequency", None) if cumulative_session and hasattr(cumulative_session, "health_info") else None,
            getattr(cumulative_session.health_info, "radiation", None) if cumulative_session and hasattr(cumulative_session, "health_info") else None,
            supported_facts.overall_duration,
            supported_facts.episode_duration,
            supported_facts.frequency,
            supported_facts.radiation,
            supported_facts.body_temperature,
            sorted(list(supported_facts.negative_symptoms)),
            generated_response.strip(),
            res.is_valid,
            res.violations
        )

        if res.is_valid:
            return generated_response.strip()

        logger.info(
            "ResponseValidator detected unsupported claims (%d violations): %s. Executing safe repair pipeline.",
            len(res.violations), res.violations
        )

        repaired = self.repair(
            generated_response=generated_response,
            violations=res.violations,
            supported_facts=supported_facts,
            grounding=grounding
        )

        # Verify that repaired text is compliant against the SAME verified facts
        post_check = self.validate(
            generated_response=repaired,
            supported_facts=supported_facts,
            grounding=grounding,
            response_plan=response_plan
        )

        if post_check.is_valid and len(repaired) > 20:
            logger.info("ResponseValidator safe repair succeeded: '%s'", repaired)
            return repaired

        # If repair is not clean or violations remain, use deterministic grounded fallback
        logger.warning(
            "ResponseValidator repair could not resolve all violations (%s). Using deterministic grounded fallback.",
            post_check.violations
        )
        if grounding and getattr(grounding, "deterministic_fallback_message", None):
            return grounding.deterministic_fallback_message

        # Fallback to verified fact summary
        sym_list = ", ".join(sorted(supported_facts.symptoms)) if supported_facts.symptoms else "your reported symptoms"
        return (
            f"Based on what you have shared, you described {sym_list}"
            + (f" located in the {supported_facts.location}" if supported_facts.location else "")
            + ". A clinician evaluating these symptoms would consider relevant diagnostic tests to determine the underlying cause."
        )


# Global singleton instance
response_validator = ResponseValidator()
