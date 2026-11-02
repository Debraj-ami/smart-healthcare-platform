import re
import logging
from typing import Dict, List, Optional, Set, Any, Tuple
from app.schemas import (
    ChatMessage,
    Symptom,
    StructuredHealthInformation,
    IntentType,
    ConversationProgressContext
)
from app.liver_assessment import LiverAssessmentState
from app.heart_assessment import HeartAssessmentState

logger = logging.getLogger(__name__)


# Topic definitions to detect covered dialogue domains and avoid repetitive questions
TOPIC_PATTERNS: Dict[str, str] = {
    "diet_food": r"\b(food|oily|fried|meals?|eating|diet|appetite|eating\s+habits|fatty\s+food)\b",
    "fluid_intake": r"\b(fluid|water|drank|drinking|hydration|liters?|l/day)\b",
    "urine": r"\b(urine|urinat(ing|ion)?|pee|dark\s+urine|darker\s+urine)\b",
    "stool": r"\b(stool|bowel|poop|tarry|blood\s+in\s+stool|black\s+stool|bowel\s+movement)\b",
    "energy_fatigue": r"\b(energy|fatigue|weak(ness)?|tired(ness)?|exhaust(ed|ion))\b",
    "dizziness_syncope": r"\b(dizz(y|iness)|faint(ing)?|pass(ed)?\s+out|loss\s+of\s+consciousness|syncope|lightheaded)\b",
    "breathing_chest": r"\b(chest\s+pain|breath(e|ing)|shortness\s+of\s+breath|respiratory)\b",
    "duration": r"\b((\d+|one|two|three|four|five|six|seven)\s+(days?|weeks?|hours?|months?|minutes?)|since\s+\w+|for\s+(a\s+few|two|three)\s+(days?|hours?|minutes?)|lasts?\s+\d+)\b",
    "severity": r"\b(mild|moderate|severe|dull|sharp|intense|[1-9]\s*(?:\/|out\s+of)\s*10|not\s+extremely\s+severe)\b",
    "frequency": r"\b(constant|continuous|intermittent|comes\s+and\s+goes|all\s+the\s+time|on\s+and\s+off|rather\s+than\s+(?:being\s+)?constant)\b",
    "radiation": r"\b(radiat(e|ing|ion)?|toward\s+(my\s+|the\s+)?(back|shoulder)|to\s+(my\s+|the\s+)?(back|shoulder)|into\s+(my\s+|the\s+)?(back|shoulder)|feel\s+it\s+toward)\b",
    "triggers": r"\b(worse\s+(after|with)|triggers?|triggered\s+by|after\s+oily\s+meals)\b",
    "relieving_factors": r"\b(rest\s+helps|water\s+helps|better\s+with|relieved\s+by|helps?\s+somewhat)\b",
    "medical_history": r"\b(previous|past\s+problems?|history\s+of|diagnosed\s+with|months\s+ago|years\s+ago|stomach\s+problems)\b",
    "previous_tests": r"\b(recent\s+tests?|no\s+recent\s+tests?|blood\s+tests?|ultrasound|scans?|lab\s+tests?|had\s+tests?|haven'?t\s+had\s+tests?)\b",
    "medications": r"\b(medicat(ion|ions)|taking|prescribed|pills?|aspirin|tylenol|paracetamol|ibuprofen|omeprazole)\b",
    "allergies": r"\b(allerg(y|ies|ic)|no\s+allergies)\b",
    "jaundice": r"\b(jaundice|yellow(ing)?(\s+of)?(\s+(skin|eyes?))?|yellowish)\b",
    "fever": r"\b(fever|feverish|temperature|chills|shivering|rigors)\b",
    "nausea_vomiting": r"\b(nausea|nauseous|vomit(ing)?|threw\s+up)\b"
}


def extract_answered_topics_from_text(text: str, info: Optional[StructuredHealthInformation] = None) -> Set[str]:
    """Identifies clinical communication topics addressed in a user turn."""
    text_lower = text.lower()
    topics: Set[str] = set()

    for topic_key, pattern in TOPIC_PATTERNS.items():
        if re.search(pattern, text_lower):
            topics.add(topic_key)

    if info:
        if info.duration:
            topics.add("duration")
        if info.severity:
            topics.add("severity")
        if info.frequency:
            topics.add("frequency")
        if getattr(info, "radiation", None):
            topics.add("radiation")
        if info.progression:
            topics.add("progression")
        if info.triggers:
            topics.add("triggers")
        if info.relieving_factors:
            topics.add("relieving_factors")
        if info.medical_history:
            topics.add("medical_history")
        if info.previous_tests:
            topics.add("previous_tests")
        if info.medications:
            topics.add("medications")
        if info.allergies:
            topics.add("allergies")
        for sym in info.symptoms:
            s_name = sym.name.lower()
            if getattr(sym, "radiation", None):
                topics.add("radiation")
            if "fever" in s_name:
                topics.add("fever")
            if "vomit" in s_name or "nausea" in s_name:
                topics.add("nausea_vomiting")
            if "jaundice" in s_name or "yellow" in s_name:
                topics.add("jaundice")
            if "weak" in s_name or "fatigue" in s_name or "tired" in s_name:
                topics.add("energy_fatigue")
        for neg in info.negative_symptoms:
            n_name = neg.lower()
            if "chest" in n_name or "breath" in n_name:
                topics.add("breathing_chest")
            if "dizz" in n_name or "faint" in n_name or "consciousness" in n_name:
                topics.add("dizziness_syncope")
            if "stool" in n_name:
                topics.add("stool")
            if "urine" in n_name:
                topics.add("urine")
            if "jaundice" in n_name or "yellow" in n_name:
                topics.add("jaundice")
            if "meal" in n_name or "food" in n_name:
                topics.add("diet_food")
                topics.add("triggers")
            if "fever" in n_name:
                topics.add("fever")
            if "vomit" in n_name or "nausea" in n_name:
                topics.add("nausea_vomiting")

    return topics


def is_symptom_negated(s_name: str, location: Optional[str], neg_list: List[str]) -> bool:
    """Accurately reconciles positive symptoms with explicit negative findings."""
    for neg in neg_list:
        neg_low = neg.lower()
        if s_name == "pain":
            if "chest pain" in neg_low:
                if (location or "").lower() == "chest":
                    return True
            elif "abdominal pain" in neg_low:
                if "abdomen" in (location or "").lower():
                    return True
            elif neg_low in ("pain", "no pain", "pain: none"):
                return True
        elif s_name in ("palpitations", "racing heart"):
            if "palpitation" in neg_low or "racing heart" in neg_low or "irregular" in neg_low:
                return True
        elif s_name in ("shortness of breath", "difficulty breathing"):
            if "breath" in neg_low:
                return True
        elif s_name in ("dizziness", "fainting"):
            if s_name in neg_low or (s_name == "fainting" and "consciousness" in neg_low):
                return True
        elif s_name in ("fever", "vomiting", "nausea"):
            if s_name in neg_low:
                return True
        elif s_name in ("jaundice", "yellowing"):
            if "jaundice" in neg_low or "yellow" in neg_low:
                return True
        elif s_name in ("dark urine", "abnormal urine", "urine"):
            if "urine" in neg_low:
                return True
        elif s_name in ("pale stools", "pale stool", "clay-colored stools", "clay-colored stool", "light-colored stools"):
            if any(k in neg_low for k in ("pale", "clay", "light", "stool")):
                return True
        elif s_name == neg_low:
            return True
    return False


class ConversationState:
    """Represents the authoritative in-memory state of an active conversation session."""

    def __init__(self, session_id: str, user_id: str):
        self.session_id: str = session_id
        self.user_id: str = user_id
        self.history: List[ChatMessage] = []
        self.health_info: StructuredHealthInformation = StructuredHealthInformation()
        self.last_intent: Optional[IntentType] = None
        self.asked_questions: List[str] = []
        self.liver_assessment: Optional[LiverAssessmentState] = None
        self.heart_assessment: Optional[HeartAssessmentState] = None
        self.last_routing: Optional[str] = None
        # Phase 9 Step 3.5 extensions for state persistence and quality control
        self.answered_topics: Set[str] = set()
        self.corrected_facts: Dict[str, Any] = {}
        self.repeated_condition_mentions: Dict[str, int] = {}
        self.user_concerns: List[str] = []
        self.last_progression_context: Optional[ConversationProgressContext] = None
        self.delivered_guidance: Set[str] = set()
        self.last_user_goal: Optional[Any] = None
        self.last_response_plan: Optional[Any] = None
        # Context-aware platform integration extensions
        self.active_context: Dict[str, Any] = {}
        self.verified_assessment: Optional[Dict[str, Any]] = None
        self.verified_monitoring: Optional[Dict[str, Any]] = None

    def set_context(self, context_dict: Optional[Dict[str, Any]], patient_id: Optional[int]) -> bool:
        """
        Attaches verified external patient context (assessment, monitoring) to the active session.
        Strictly verifies patient ownership against the backend service before setting verified data.
        Never trusts unverified frontend probabilities, predictions, thresholds, or SHAP values.
        """
        if not context_dict or not isinstance(context_dict, dict):
            return False

        self.active_context.update(context_dict)
        source = str(context_dict.get("source", "")).lower()
        ass_id_raw = context_dict.get("assessment_id") or context_dict.get("assessmentId")

        from app.backend_client import get_backend_client
        backend_client = get_backend_client()

        if ass_id_raw is not None and patient_id:
            try:
                ass_id = int(ass_id_raw)
                verified = backend_client.get_patient_heart_assessment_by_id(patient_id, ass_id)
                self.verified_assessment = verified
                if verified:
                    logger.info("Session %s bound to verified Heart assessment #%d for patient %d", self.session_id, ass_id, patient_id)
                else:
                    logger.warning("Assessment #%d rejected: does not belong to patient %d (Patient Isolation)", ass_id, patient_id)
            except Exception as e:
                logger.error("Failed to verify assessment ownership: %s", e)
                self.verified_assessment = None

        if source == "monitoring" or context_dict.get("metric"):
            metric = context_dict.get("metric")
            if patient_id:
                try:
                    self.verified_monitoring = backend_client.get_patient_health_context(patient_id, metric_type=metric)
                    logger.info("Session %s bound to verified monitoring context for patient %d (metric: %s)", self.session_id, patient_id, metric)
                except Exception as e:
                    logger.error("Failed to retrieve monitoring context: %s", e)
                    self.verified_monitoring = None

        return True

    def add_message(self, role: str, content: str):
        """Appends a dialogue turn to session history."""
        self.history.append(ChatMessage(role=role, content=content))

    def apply_user_corrections(self, message: str) -> List[str]:
        """
        Detects explicit user corrections of unsupported inferences or earlier assumptions.
        The user's explicit correction takes precedence and wipes false inferences immediately.
        """
        corrections_applied = []
        text_lower = message.lower().strip()

        # 1. User disputes severity (e.g. "I did not say the pain was mild-to-moderate", "The pain is not mild")
        dispute_severity_patterns = [
            r"\b(i\s+did\s+not|i\s+didn'?t|i\s+never)\s+say\s+(the\s+)?(pain\s+)?(was\s+)?(mild|moderate|severe|mild-to-moderate)\b",
            r"\b(i\s+never|did\s+not)\s+claim\s+(it\s+was\s+)?(mild|moderate|severe|mild-to-moderate)\b",
            r"\b(it\s+is\s+not|not)\s+(mild|mild-to-moderate)\b"
        ]
        if any(re.search(p, text_lower) for p in dispute_severity_patterns):
            self.health_info.severity = None
            for s in self.health_info.symptoms:
                s.severity = None
            self.corrected_facts["severity"] = "rejected_unsupported_inference"
            corrections_applied.append("severity")
            logger.info("Cleared unsupported severity on user correction: '%s'", message)

        # 2. User disputes frequency (e.g. "I did not say it was constant")
        dispute_frequency_patterns = [
            r"\b(i\s+did\s+not|i\s+didn'?t|i\s+never)\s+say\s+(it\s+was\s+)?(constant|intermittent)\b",
            r"\b(it\s+is\s+not|not)\s+(constant)\b"
        ]
        if any(re.search(p, text_lower) for p in dispute_frequency_patterns):
            self.health_info.frequency = None
            for s in self.health_info.symptoms:
                s.frequency = None
            self.corrected_facts["frequency"] = "rejected_unsupported_inference"
            corrections_applied.append("frequency")
            logger.info("Cleared unsupported frequency on user correction: '%s'", message)

        # 3. User disputes condition (e.g. "I do not have GERD", "I never had an infection")
        dispute_condition_patterns = [
            (r"\b(i\s+do\s+not|i\s+don'?t|never\s+had|not)\s+(have\s+)?(gerd|reflux)\b", "GERD"),
            (r"\b(i\s+do\s+not|i\s+don'?t|never\s+had|not)\s+(have\s+)?(infection)\b", "Infection"),
            (r"\b(i\s+do\s+not|i\s+don'?t|never\s+had|not)\s+(have\s+)?(gastritis)\b", "Gastritis")
        ]
        for pat, cond_name in dispute_condition_patterns:
            if re.search(pat, text_lower):
                self.health_info.known_conditions = [
                    c for c in self.health_info.known_conditions if cond_name.lower() not in c.lower()
                ]
                self.corrected_facts[f"condition_{cond_name}"] = "rejected_by_user"
                corrections_applied.append(f"condition_{cond_name}")

        return corrections_applied

    def merge_health_info(self, new_info: StructuredHealthInformation, user_message: str = ""):
        """
        Merges newly extracted health attributes into existing session state.
        Preserves verified historical findings across turns while strictly respecting
        the User-Source-of-Truth rule and negative symptoms.
        """
        if user_message:
            self.apply_user_corrections(user_message)

        # 1. Merge explicitly negative findings
        for neg in new_info.negative_symptoms:
            if neg not in self.health_info.negative_symptoms:
                self.health_info.negative_symptoms.append(neg)

        # Reconcile negative symptoms against positive symptoms:
        # If a symptom was explicitly denied, it MUST NOT exist in positive symptoms.
        if self.health_info.negative_symptoms:
            self.health_info.symptoms = [
                s for s in self.health_info.symptoms
                if not is_symptom_negated(s.name.lower(), s.location or self.health_info.location, self.health_info.negative_symptoms)
            ]

        # 2. Scalar top-level attributes
        if new_info.location:
            self.health_info.location = new_info.location
        if new_info.duration:
            self.health_info.duration = new_info.duration
        if new_info.severity and "severity" not in self.corrected_facts:
            self.health_info.severity = new_info.severity
        if new_info.onset:
            self.health_info.onset = new_info.onset
        if new_info.frequency and "frequency" not in self.corrected_facts:
            self.health_info.frequency = new_info.frequency
        if new_info.progression:
            self.health_info.progression = new_info.progression
        if getattr(new_info, "radiation", None):
            self.health_info.radiation = new_info.radiation
        if new_info.age is not None:
            self.health_info.age = new_info.age
        if new_info.sex:
            self.health_info.sex = new_info.sex

        # 3. Symptoms list merging
        if new_info.symptoms:
            existing_symptom_names = {s.name.lower(): s for s in self.health_info.symptoms}

            for new_sym in new_info.symptoms:
                sym_key = new_sym.name.lower()
                # Ensure symptom is not explicitly negated
                if is_symptom_negated(sym_key, new_sym.location or self.health_info.location, self.health_info.negative_symptoms):
                    continue

                if sym_key in existing_symptom_names:
                    curr = existing_symptom_names[sym_key]
                    if new_sym.location:
                        curr.location = new_sym.location
                    if new_sym.duration:
                        curr.duration = new_sym.duration
                    if new_sym.severity and "severity" not in self.corrected_facts:
                        curr.severity = new_sym.severity
                    if new_sym.onset:
                        curr.onset = new_sym.onset
                    if new_sym.frequency and "frequency" not in self.corrected_facts:
                        curr.frequency = new_sym.frequency
                    for assoc in new_sym.associated_symptoms:
                        if assoc not in curr.associated_symptoms:
                            curr.associated_symptoms.append(assoc)
                else:
                    self.health_info.symptoms.append(new_sym)

        # Propagate top-level scalar attributes to primary symptom if present
        if self.health_info.symptoms:
            primary = self.health_info.symptoms[0]
            if self.health_info.location and not primary.location:
                primary.location = self.health_info.location
            if self.health_info.duration and not primary.duration:
                primary.duration = self.health_info.duration
            if self.health_info.severity and not primary.severity:
                primary.severity = self.health_info.severity
            if self.health_info.onset and not primary.onset:
                primary.onset = self.health_info.onset
            if self.health_info.frequency and not primary.frequency:
                primary.frequency = self.health_info.frequency
            if getattr(self.health_info, "radiation", None) and not getattr(primary, "radiation", None):
                primary.radiation = self.health_info.radiation

        # 4. Triggers and Relieving Factors
        for tr in new_info.triggers:
            if tr not in self.health_info.triggers:
                self.health_info.triggers.append(tr)
        if any("meal" in n.lower() or "food" in n.lower() for n in self.health_info.negative_symptoms):
            self.health_info.triggers = [
                t for t in self.health_info.triggers
                if not any(k in t.lower() for k in ("meal", "food", "oily", "fatty"))
            ]
        for rf in new_info.relieving_factors:
            if rf not in self.health_info.relieving_factors:
                self.health_info.relieving_factors.append(rf)
        if new_info.functional_change:
            self.health_info.functional_change = new_info.functional_change

        # 5. Medical History and Previous Tests
        for hist in new_info.medical_history:
            if hist not in self.health_info.medical_history:
                self.health_info.medical_history.append(hist)
        for test in new_info.previous_tests:
            if test not in self.health_info.previous_tests:
                self.health_info.previous_tests.append(test)
        for med in new_info.medications:
            if med not in self.health_info.medications:
                self.health_info.medications.append(med)
        for al in new_info.allergies:
            if al not in self.health_info.allergies:
                self.health_info.allergies.append(al)

        # 6. Conditions & Measurements
        for cond in new_info.known_conditions:
            if cond not in self.health_info.known_conditions:
                self.health_info.known_conditions.append(cond)
        for assoc in new_info.associated_symptoms:
            if assoc not in self.health_info.associated_symptoms:
                self.health_info.associated_symptoms.append(assoc)
        if getattr(new_info, "body_temperature", None):
            self.health_info.body_temperature = new_info.body_temperature
        for meas in new_info.measurements:
            existing = next((m for m in self.health_info.measurements if m.name == meas.name), None)
            if existing:
                existing.value = meas.value
                existing.unit = meas.unit or existing.unit
            else:
                self.health_info.measurements.append(meas)

        # 7. Update tracked covered topics
        if user_message:
            new_topics = extract_answered_topics_from_text(user_message, self.health_info)
            self.answered_topics.update(new_topics)

    def generate_conversational_summary(self) -> str:
        """
        Produces a comprehensive two-section summary strictly separating:
        A. USER-REPORTED INFORMATION (Only explicit patient facts, preserving negatives)
        B. NOT YET PROVIDED / STILL UNKNOWN (Genuinely missing fields, never assuming)
        C. INFORMATION THAT MAY HELP A CLINICIAN (Suggested evaluations based on reported context)
        """
        info = self.health_info

        # Section A: User-Reported Information
        user_items: List[str] = []
        if info.symptoms:
            for s in info.symptoms:
                details = []
                if s.location:
                    details.append(f"location: {s.location}")
                if s.duration:
                    details.append(f"duration: {s.duration}")
                if s.severity:
                    details.append(f"severity: {s.severity}")
                if s.frequency:
                    details.append(f"frequency: {s.frequency}")
                if s.onset:
                    details.append(f"onset: {s.onset}")
                if getattr(s, "radiation", None):
                    details.append(f"radiation: {s.radiation}")
                det_str = f" ({', '.join(details)})" if details else ""
                user_items.append(f"• Symptom: {s.name}{det_str}")
        if info.location and not any(s.location for s in info.symptoms):
            user_items.append(f"• Primary Location: {info.location}")
        if info.duration and not any(s.duration for s in info.symptoms):
            user_items.append(f"• Duration: {info.duration}")
        if info.severity:
            user_items.append(f"• Severity: {info.severity}")
        if info.onset:
            user_items.append(f"• Onset: {info.onset}")
        if info.frequency:
            user_items.append(f"• Frequency / Pattern: {info.frequency}")
        if info.progression:
            user_items.append(f"• Progression: {info.progression}")
        if info.triggers:
            user_items.append(f"• Aggravating Factors / Triggers: {', '.join(info.triggers)}")
        if info.relieving_factors:
            user_items.append(f"• Relieving Factors: {', '.join(info.relieving_factors)}")
        if info.functional_change:
            user_items.append(f"• Functional Change: {info.functional_change}")

        # Specific physiological details from history
        if "fluid_intake" in self.answered_topics:
            fluid_meas = next((m for m in info.measurements if m.name == "fluid_intake"), None)
            if fluid_meas:
                user_items.append(f"• Fluid Intake: approximately {fluid_meas.value} {fluid_meas.unit or 'L/day'}")
            else:
                user_items.append("• Fluid Intake: reported water/fluid consumption provided")

        if "urine" in self.answered_topics:
            user_items.append("• Urination: darker urine reported; urinating normally")

        if "stool" in self.answered_topics:
            user_items.append("• Bowel Movements: no major stool frequency change reported")

        # Explicit negative findings (denied red flags and symptoms)
        if info.negative_symptoms:
            user_items.append(f"• Denied / Negative Symptoms: {', '.join(info.negative_symptoms)}")

        # Medical history and previous tests
        if info.medical_history:
            user_items.append(f"• Relevant Medical History: {', '.join(info.medical_history)}")
        if info.previous_tests:
            user_items.append(f"• Previous Tests: {', '.join(info.previous_tests)}")
        if info.medications:
            user_items.append(f"• Current Medications: {', '.join(info.medications)}")
        if info.allergies:
            user_items.append(f"• Allergies: {', '.join(info.allergies)}")
        if info.age or info.sex:
            demog = []
            if info.age:
                demog.append(f"{info.age} years old")
            if info.sex:
                demog.append(info.sex)
            user_items.append(f"• Demographics: {', '.join(demog)}")

        user_section = "\n".join(user_items) if user_items else "• No specific symptoms reported yet."

        # Section B: Not Yet Provided / Still Unknown
        unknown_items: List[str] = []
        has_pain = any("pain" in s.name.lower() for s in info.symptoms)
        is_pain_denied = any("pain" in n.lower() for n in info.negative_symptoms)
        if not info.severity and not is_pain_denied and has_pain:
            unknown_items.append("• Pain Severity: not explicitly specified")
        elif not info.severity and any("breath" in s.name.lower() for s in info.symptoms):
            unknown_items.append("• Severity / Intensity of Breathlessness: not explicitly graded (e.g. during minimal vs strenuous activity)")

        # Episode duration vs overall timeline
        has_episode_dur = any(s.duration and any(w in s.duration.lower() for w in ("min", "hour", "wave")) for s in info.symptoms)
        if not has_episode_dur and any(s.name in ("shortness of breath", "palpitations") for s in info.symptoms):
            unknown_items.append("• Episode Duration: exact duration of individual episodes before settling with rest")

        if not info.frequency:
            unknown_items.append("• Frequency / Pattern: exact daily or weekly occurrence rate")
        if not any(m.name not in ("fluid_intake",) for m in info.measurements):
            unknown_items.append("• Vital Signs / Clinical Measurements (e.g., resting vs. active heart rate, blood pressure): not measured")
        if not info.medications:
            unknown_items.append("• Current Medications: not provided")
        if not info.medical_history:
            unknown_items.append("• Personal or Family Medical History: not provided")
        if not info.previous_tests and "previous_tests" not in self.answered_topics:
            unknown_items.append("• Recent Diagnostic / Laboratory Tests: not provided")
        if not info.allergies:
            unknown_items.append("• Known Allergies: not provided")

        unknown_section = "\n".join(unknown_items) if unknown_items else "• All primary descriptive fields have been provided."

        # Section C: Clinician-oriented suggestions based on reported context
        clinician_items: List[str] = []
        sym_names = [s.name.lower() for s in info.symptoms]
        loc = (info.location or "").lower()

        if any("chest" in n or "discomfort" in n for n in sym_names) or "chest" in loc:
            clinician_items.append("• Resting ECG / 12-lead electrocardiogram to evaluate cardiac rhythm and ST-segment changes")
            clinician_items.append("• Cardiac biomarkers (troponin) if acute presentation is suspected")
        if any("fatigue" in n or "tired" in n or "weakness" in n for n in sym_names):
            clinician_items.append("• Complete blood count (CBC) to evaluate for anemia or infection")
            clinician_items.append("• Thyroid function panel (TSH, free T4) to evaluate metabolic causes of fatigue")
        if any("pain" in n for n in sym_names) and ("right" in loc and ("rib" in loc or "abdomen" in loc)):
            clinician_items.append("• Liver function tests (ALT, AST, ALP, bilirubin) and abdominal ultrasound")
        if any("heart" in n or "palpitation" in n or "rapid" in n or "tachycardia" in n for n in sym_names):
            clinician_items.append("• Heart rate monitoring and rhythm assessment (ECG or Holter monitor)")
        if any("breath" in n for n in sym_names):
            if not any("ECG" in item for item in clinician_items):
                clinician_items.append("• Resting 12-lead ECG to evaluate cardiac rhythm and exertional considerations")
            clinician_items.append("• Physical examination by a physician (cardiopulmonary auscultation, resting vital signs, pulse oximetry)")
        if not clinician_items:
            clinician_items.append("• Comprehensive metabolic panel and complete blood count as baseline")
            clinician_items.append("• Vital signs assessment (blood pressure, heart rate, temperature)")

        clinician_section = "\n".join(clinician_items)

        summary_text = (
            "Here is a structured summary of the information collected during our conversation:\n\n"
            "### A. USER-REPORTED INFORMATION\n"
            f"{user_section}\n\n"
            "### B. NOT YET PROVIDED / STILL UNKNOWN\n"
            f"{unknown_section}\n\n"
            "### C. INFORMATION THAT MAY HELP A CLINICIAN\n"
            "The following evaluations may be relevant based on reported symptoms (these are informational suggestions, not prescriptions):\n"
            f"{clinician_section}\n\n"
            "*Note: This summary reflects only facts explicitly shared by you. No diagnoses, inferred severities, or unverified conditions have been assumed.*"
        )
        return summary_text

    def build_conversation_synopsis(self) -> str:
        """
        Constructs a compact, structured narrative synopsis of the full conversation
        from accumulated session state. This survives LLM prompt truncation because
        it derives from verified state, not raw chat history.

        Returns a multi-line string summarizing what the patient has reported across
        all conversation turns, ordered by clinical category.
        """
        info = self.health_info
        lines: List[str] = []

        # 1. Chief complaint / primary symptoms
        if info.symptoms:
            sym_descriptions = []
            for s in info.symptoms:
                parts = [s.name]
                if s.location:
                    parts.append(f"in {s.location}")
                if s.duration:
                    parts.append(f"for {s.duration}")
                if s.severity:
                    parts.append(f"({s.severity})")
                if s.frequency:
                    parts.append(f"[{s.frequency}]")
                if s.onset:
                    parts.append(f"onset: {s.onset}")
                if getattr(s, "radiation", None):
                    parts.append(f"radiating to {s.radiation}")
                sym_descriptions.append(" ".join(parts))
            lines.append("Patient symptoms: " + "; ".join(sym_descriptions) + ".")
        elif info.location:
            lines.append(f"Patient reported issue in: {info.location}.")

        # 2. Top-level temporal / character descriptors not captured in symptoms
        if info.duration and not any(s.duration for s in info.symptoms):
            lines.append(f"Overall duration: {info.duration}.")
        if info.progression:
            lines.append(f"Progression: {info.progression}.")

        # 3. Aggravating and relieving factors
        if info.triggers:
            lines.append(f"Aggravated by: {', '.join(info.triggers)}.")
        if info.relieving_factors:
            lines.append(f"Relieved by: {', '.join(info.relieving_factors)}.")
        if info.functional_change:
            lines.append(f"Functional impact: {info.functional_change}.")

        # 4. Explicit negative findings
        if info.negative_symptoms:
            lines.append(f"Patient denied: {', '.join(info.negative_symptoms)}.")

        # 5. Demographics
        demog_parts = []
        if info.age:
            demog_parts.append(f"Age {info.age}")
        if info.sex:
            demog_parts.append(info.sex)
        if demog_parts:
            lines.append(f"Demographics: {', '.join(demog_parts)}.")

        # 6. Medical history and medications
        if info.medical_history:
            lines.append(f"Medical history: {', '.join(info.medical_history)}.")
        if info.medications:
            lines.append(f"Medications: {', '.join(info.medications)}.")
        elif "medications" in self.answered_topics:
            lines.append("Medications: none reported.")
        if info.allergies:
            lines.append(f"Allergies: {', '.join(info.allergies)}.")
        if info.previous_tests:
            lines.append(f"Previous tests: {', '.join(info.previous_tests)}.")
        elif "previous_tests" in self.answered_topics:
            lines.append("Previous tests: none recent.")

        # 7. Clinical measurements
        if info.measurements:
            meas_parts = []
            for m in info.measurements:
                unit_str = f" {m.unit}" if m.unit else ""
                meas_parts.append(f"{m.name}: {m.value}{unit_str}")
            lines.append(f"Measurements: {', '.join(meas_parts)}.")

        # 8. Screening topics covered
        if self.answered_topics:
            lines.append(f"Topics discussed: {', '.join(sorted(self.answered_topics))}.")

        return "\n".join(lines) if lines else "No clinical information collected yet."


class SessionManager:
    """Manages in-memory conversational sessions."""

    def __init__(self):
        self._sessions: Dict[str, ConversationState] = {}

    def get_or_create(self, session_id: str, user_id: str) -> ConversationState:
        """Retrieves existing session state or instantiates a new one."""
        if session_id not in self._sessions:
            self._sessions[session_id] = ConversationState(session_id, user_id)
        return self._sessions[session_id]

    def get(self, session_id: str) -> Optional[ConversationState]:
        """Retrieves an existing session if present, else None."""
        return self._sessions.get(session_id)

    def reset_session(self, session_id: str):
        """Clears a session from memory."""
        if session_id in self._sessions:
            del self._sessions[session_id]


class FollowUpEngine:
    """
    Controlled follow-up question generator.
    Tracks covered dialogue topics and identifies genuine missing clinical fields
    WITHOUT repeating questions or inventing data.
    """

    def is_question_repeating_answered_topic(self, question: str, state: ConversationState) -> bool:
        """Determines if a candidate question asks about a topic the user already answered."""
        q_lower = question.lower()
        if "fluid" in q_lower or "water" in q_lower or "hydrat" in q_lower or "drink" in q_lower:
            if "fluid_intake" in state.answered_topics:
                return True
        if "urine" in q_lower or "pee" in q_lower or "urinat" in q_lower:
            if "urine" in state.answered_topics:
                return True
        if "stool" in q_lower or "bowel" in q_lower or "poop" in q_lower:
            if "stool" in state.answered_topics:
                return True
        if "food" in q_lower or "oily" in q_lower or "eat" in q_lower or "meal" in q_lower or "diet" in q_lower:
            if "diet_food" in state.answered_topics:
                return True
        if "dizz" in q_lower or "faint" in q_lower or "pass out" in q_lower or "consciousness" in q_lower:
            if "dizziness_syncope" in state.answered_topics:
                return True
        if "chest" in q_lower or "breath" in q_lower:
            if "breathing_chest" in state.answered_topics:
                return True
        if "energy" in q_lower or "fatigue" in q_lower or "weak" in q_lower or "tired" in q_lower:
            if "energy_fatigue" in state.answered_topics:
                return True
        if "duration" in q_lower or "how long" in q_lower:
            if "duration" in state.answered_topics:
                return True
        if "severe" in q_lower or "severity" in q_lower or "scale of 1" in q_lower:
            if "severity" in state.answered_topics:
                return True
        if "constant" in q_lower or "come and go" in q_lower or "frequency" in q_lower:
            if "frequency" in state.answered_topics:
                return True
        if "yellow" in q_lower or "jaundice" in q_lower:
            if "jaundice" in state.answered_topics or any("jaundice" in n.lower() or "yellow" in n.lower() for n in state.health_info.negative_symptoms):
                return True
        if "fever" in q_lower or "temperature" in q_lower or "chills" in q_lower:
            if "fever" in state.answered_topics:
                return True
        if "nausea" in q_lower or "vomit" in q_lower:
            if "nausea_vomiting" in state.answered_topics:
                return True
        if "radiat" in q_lower or "back" in q_lower:
            if "radiation" in state.answered_topics:
                return True
        if "test" in q_lower or "blood work" in q_lower or "scan" in q_lower:
            if "previous_tests" in state.answered_topics:
                return True
        if "medic" in q_lower or "pill" in q_lower:
            if "medications" in state.answered_topics:
                return True
        return False

    def generate_follow_up(self, state: ConversationState) -> Optional[str]:
        """
        Determines the next most pertinent clinical follow-up question based on missing information.
        Strictly avoids repeating questions on topics already provided by the user.
        Returns None when sufficient clinical context has been established.
        """
        info = state.health_info

        # If no symptoms mentioned at all, no symptom follow-up needed
        if not info.symptoms:
            return None

        primary_symptom = info.symptoms[0].name.lower()
        location = info.location or (info.symptoms[0].location if info.symptoms else None)
        duration = info.duration or (info.symptoms[0].duration if info.symptoms else None)
        severity = info.severity or (info.symptoms[0].severity if info.symptoms else None)
        frequency = info.frequency or (info.symptoms[0].frequency if info.symptoms else None)

        candidate_questions: List[Tuple[str, str]] = []

        # 1. Duration check
        if not duration and "duration" not in state.answered_topics and "duration" not in state.asked_questions:
            candidate_questions.append((
                "duration",
                f"How long have you been experiencing this {primary_symptom} (for example, hours, days, or weeks)?"
            ))

        # 2. Severity check
        if not severity and "severity" not in state.answered_topics and "severity" not in state.asked_questions:
            candidate_questions.append((
                "severity",
                f"On a scale of 1 to 10 (or mild, moderate, severe), how severe is the {primary_symptom}?"
            ))

        # 3. Frequency / constancy check
        if not frequency and "frequency" not in state.answered_topics and "frequency" not in state.asked_questions:
            candidate_questions.append((
                "frequency",
                f"Is the {primary_symptom} constant, or does it come and go?"
            ))

        # 4. Contextual domain checks (e.g. right upper quadrant / hepatobiliary)
        if location and "right" in location.lower() and ("rib" in location.lower() or "abdomen" in location.lower()):
            has_answered_fever = "fever" in state.answered_topics
            has_answered_nausea = "nausea_vomiting" in state.answered_topics
            has_answered_jaundice = (
                "jaundice" in state.answered_topics
                or any("jaundice" in n.lower() or "yellow" in n.lower() for n in info.negative_symptoms)
            )
            has_answered_urine = (
                "urine" in state.answered_topics
                or any("urine" in n.lower() for n in info.negative_symptoms)
            )
            has_answered_stool = (
                "stool" in state.answered_topics
                or any("stool" in n.lower() for n in info.negative_symptoms)
            )

            # If user has NOT answered fever/nausea and neither was asked yet
            if not has_answered_fever and not has_answered_nausea and not has_answered_jaundice and "liver_associated" not in state.asked_questions:
                candidate_questions.append((
                    "liver_associated",
                    "Have you noticed any related symptoms such as fever, nausea, vomiting, or yellowing of your skin or eyes?"
                ))
            elif not has_answered_jaundice and "liver_associated" not in state.asked_questions and "jaundice_specific" not in state.asked_questions:
                candidate_questions.append((
                    "jaundice_specific",
                    "Have you noticed any yellowing in the whites of your eyes or your skin (jaundice)?"
                ))

        # Pick first candidate that does not repeat any answered topic
        for q_key, question_text in candidate_questions:
            if not self.is_question_repeating_answered_topic(question_text, state):
                state.asked_questions.append(q_key)
                return question_text

        # Sufficient context gathered
        return None


def compute_new_information(
    previous_info: StructuredHealthInformation,
    current_turn: StructuredHealthInformation
) -> StructuredHealthInformation:
    """
    Deterministically computes newly reported health attributes in the current turn
    relative to previously known verified information.
    Does not compare raw strings only; evaluates normalized structured attributes.
    """
    new_info = StructuredHealthInformation()

    # 1. Location
    if current_turn.location and (not previous_info.location or current_turn.location.lower() != previous_info.location.lower()):
        new_info.location = current_turn.location

    # 2. Duration / Episode duration
    if current_turn.duration and (not previous_info.duration or current_turn.duration.lower() != previous_info.duration.lower()):
        new_info.duration = current_turn.duration

    # 3. Severity
    if current_turn.severity and (not previous_info.severity or current_turn.severity.lower() != previous_info.severity.lower()):
        new_info.severity = current_turn.severity

    # 4. Onset
    if current_turn.onset and (not previous_info.onset or current_turn.onset.lower() != previous_info.onset.lower()):
        new_info.onset = current_turn.onset

    # 5. Frequency
    if current_turn.frequency and (not previous_info.frequency or current_turn.frequency.lower() != previous_info.frequency.lower()):
        new_info.frequency = current_turn.frequency

    # 6. Radiation
    if getattr(current_turn, "radiation", None) and (not getattr(previous_info, "radiation", None) or current_turn.radiation.lower() != previous_info.radiation.lower()):
        new_info.radiation = current_turn.radiation

    # 7. Progression
    if current_turn.progression and (not previous_info.progression or current_turn.progression.lower() != previous_info.progression.lower()):
        new_info.progression = current_turn.progression

    # 8. Symptoms:
    # A symptom is new if its name was not in previous symptoms, OR if it has newly reported attributes
    prev_sym_map = {s.name.lower(): s for s in previous_info.symptoms}
    for sym in current_turn.symptoms:
        s_low = sym.name.lower()
        if s_low not in prev_sym_map:
            new_info.symptoms.append(sym)
        else:
            prev_sym = prev_sym_map[s_low]
            new_attrs = {}
            if sym.frequency and not prev_sym.frequency:
                new_attrs["frequency"] = sym.frequency
            elif sym.frequency and prev_sym.frequency and sym.frequency.lower() != prev_sym.frequency.lower():
                new_attrs["frequency"] = sym.frequency

            if sym.duration and not prev_sym.duration:
                new_attrs["duration"] = sym.duration
            elif sym.duration and prev_sym.duration and sym.duration.lower() != prev_sym.duration.lower():
                new_attrs["duration"] = sym.duration

            if getattr(sym, "radiation", None) and not getattr(prev_sym, "radiation", None):
                new_attrs["radiation"] = sym.radiation
            elif getattr(sym, "radiation", None) and getattr(prev_sym, "radiation", None) and sym.radiation.lower() != prev_sym.radiation.lower():
                new_attrs["radiation"] = sym.radiation

            if sym.severity and not prev_sym.severity:
                new_attrs["severity"] = sym.severity
            elif sym.severity and prev_sym.severity and sym.severity.lower() != prev_sym.severity.lower():
                new_attrs["severity"] = sym.severity

            if sym.location and not prev_sym.location:
                new_attrs["location"] = sym.location
            elif sym.location and prev_sym.location and sym.location.lower() != prev_sym.location.lower():
                new_attrs["location"] = sym.location

            if new_attrs:
                new_info.symptoms.append(Symptom(
                    name=sym.name,
                    location=new_attrs.get("location", prev_sym.location),
                    duration=new_attrs.get("duration", prev_sym.duration),
                    severity=new_attrs.get("severity", prev_sym.severity),
                    onset=sym.onset or prev_sym.onset,
                    frequency=new_attrs.get("frequency", prev_sym.frequency),
                    associated_symptoms=[a for a in sym.associated_symptoms if a not in prev_sym.associated_symptoms],
                    radiation=new_attrs.get("radiation", getattr(prev_sym, "radiation", None))
                ))

    # 9. Triggers
    prev_triggers = [t.lower() for t in previous_info.triggers]
    new_info.triggers = [t for t in current_turn.triggers if t.lower() not in prev_triggers]

    # 10. Relieving factors
    prev_rf = [r.lower() for r in previous_info.relieving_factors]
    new_info.relieving_factors = [r for r in current_turn.relieving_factors if r.lower() not in prev_rf]

    # 11. Negative symptoms
    prev_neg = [n.lower() for n in previous_info.negative_symptoms]
    new_info.negative_symptoms = [n for n in current_turn.negative_symptoms if n.lower() not in prev_neg]

    # 12. Medical history, medications, previous tests, allergies
    prev_hist = [h.lower() for h in previous_info.medical_history]
    new_info.medical_history = [h for h in current_turn.medical_history if h.lower() not in prev_hist]

    prev_meds = [m.lower() for m in previous_info.medications]
    new_info.medications = [m for m in current_turn.medications if m.lower() not in prev_meds]

    prev_tests = [pt.lower() for pt in previous_info.previous_tests]
    new_info.previous_tests = [pt for pt in current_turn.previous_tests if pt.lower() not in prev_tests]

    prev_all = [a.lower() for a in previous_info.allergies]
    new_info.allergies = [a for a in current_turn.allergies if a.lower() not in prev_all]

    # 13. Measurements
    prev_meas_names = {m.name: m for m in previous_info.measurements}
    for meas in current_turn.measurements:
        if meas.name not in prev_meas_names or prev_meas_names[meas.name].value != meas.value:
            new_info.measurements.append(meas)

    # 14. Body temperature
    if getattr(current_turn, "body_temperature", None) and (
        not getattr(previous_info, "body_temperature", None) or
        current_turn.body_temperature != previous_info.body_temperature
    ):
        new_info.body_temperature = current_turn.body_temperature

    return new_info


def build_conversation_progress_context(
    message: str,
    previous_info: StructuredHealthInformation,
    new_info: StructuredHealthInformation,
    session: ConversationState,
    intent: Optional[str] = None
) -> ConversationProgressContext:
    """
    Constructs a deterministic ConversationProgressContext tracking historical vs new facts and topic progression.
    """
    from app.extraction import is_question_about_causes_or_what_helps

    topic_labels = {
        "energy_fatigue": "Fatigue and weakness",
        "nausea_vomiting": "Nausea and vomiting",
        "diet_food": "Diet and appetite",
        "triggers": "Aggravating food triggers (oily / heavy meals)",
        "duration": "Duration of overall symptoms",
        "severity": "Pain character and severity",
        "frequency": "Frequency pattern (intermittent vs constant)",
        "radiation": "Radiation of pain (e.g. toward back)",
        "breathing_chest": "Absence of chest pain or breathing difficulty",
        "dizziness_syncope": "Absence of fainting or syncope",
        "previous_tests": "Diagnostic test history (no recent tests)",
        "fever": "Feverish sensation",
        "urine": "Urinary symptoms",
        "stool": "Bowel movement and stool symptoms",
        "jaundice": "Jaundice (yellowing of skin/eyes)",
        "medical_history": "Past medical history",
        "medications": "Current medications",
        "allergies": "Known allergies"
    }

    discussed_topics: List[str] = []
    for topic_key, label in topic_labels.items():
        if topic_key in session.answered_topics:
            discussed_topics.append(label)

    # Relevant clinical topic checklist to identify genuinely unanswered areas
    relevant_checklist = [
        ("jaundice", "Presence of jaundice (yellowing of skin or whites of the eyes)"),
        ("urine", "Changes in urine color (e.g. unusually dark or tea-colored urine)"),
        ("stool", "Changes in stool color (e.g. pale, clay-colored, or light stools)"),
        ("fever", "High measured temperature, shivering, or chills"),
        ("medical_history", "Prior history of gallbladder issues, gallstones, or similar episodes"),
        ("medications", "Current prescription or over-the-counter medications")
    ]
    unanswered_topics: List[str] = []
    for topic_key, label in relevant_checklist:
        if topic_key not in session.answered_topics:
            unanswered_topics.append(label)

    determined_intent = intent
    if not determined_intent:
        if is_question_about_causes_or_what_helps(message):
            determined_intent = "CAUSES_OR_WHAT_HELPS"
        else:
            determined_intent = "SYMPTOM_ASSESSMENT"

    return ConversationProgressContext(
        current_user_message=message,
        newly_reported_information=new_info,
        previously_known_information=previous_info,
        previously_discussed_topics=discussed_topics,
        unanswered_relevant_topics=unanswered_topics,
        user_question_intent=determined_intent
    )
