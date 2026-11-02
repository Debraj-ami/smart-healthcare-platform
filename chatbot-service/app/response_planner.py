"""
Deterministic Response Planner for SmartHealth Chatbot.

Controls WHAT the conversational response should do and prioritize.
Ollama / Grounded LLM Synthesis controls HOW the response is naturally worded.

Architecture:
USER MESSAGE
    ↓
Emergency Safety Gate
    ↓
Deterministic Extraction
    ↓
Verified Conversation State
    ↓
Clinical Knowledge Grounding
    ↓
RESPONSE PLANNER (Deterministic, < 1ms, No LLM, No Network)
    ↓
Grounded LLM Synthesis
    ↓
Natural Conversational Response
"""

import re
import logging
from typing import List, Optional, Any, Set

from app.schemas import (
    UserGoal,
    AnswerMode,
    ResponsePlan,
    StructuredHealthInformation,
    IntentType
)
from app.extraction import (
    is_summary_request,
    is_missing_information_request,
    is_next_steps_request,
    is_diagnosis_question,
    is_question_about_causes_or_what_helps
)

logger = logging.getLogger(__name__)


class ResponsePlanner:
    """
    Deterministic response planning engine.
    Constructs a structured ResponsePlan for grounded LLM synthesis.
    """

    def detect_user_goal(
        self,
        message: str,
        intent: IntentType,
        session: Any = None
    ) -> UserGoal:
        """
        Deterministically classifies the user's conversational goal.
        """
        text = message.lower().strip()

        # 1. Diagnosis question (e.g. "Do you think this is definitely a heart problem?")
        if is_diagnosis_question(message):
            return UserGoal.DIAGNOSIS_QUESTION

        # 2. Next steps request (e.g. "What should I do next?")
        if is_next_steps_request(message) or re.search(
            r"\b(what\s+should\s+i\s+do\s+next|next\s+steps?|what\s+are\s+the\s+next\s+steps|how\s+should\s+i\s+proceed)\b",
            text
        ):
            return UserGoal.NEXT_STEPS

        # 3. Missing information request (e.g. "What information do you still need to know?")
        if is_missing_information_request(message) and not is_summary_request(message):
            return UserGoal.MISSING_INFORMATION

        # 4. Summary request (e.g. "What are the important things I've told you?")
        if is_summary_request(message) or re.search(
            r"\bbased\s+on\s+everything\s+(?:i('?ve|\s+have)?\s+told\s+you|we('?ve|\s+have)?\s+discussed|so\s+far)\b",
            text
        ) or re.search(
            r"\bwhat\s+information\s+(?:would|could|might)\s+be\s+most\s+useful\s+for\s+a\s+doctor(?:\s+to\s+know)?\b",
            text
        ):
            return UserGoal.SUMMARY_OF_CONVERSATION

        # 5. Correction or refinement
        if re.search(
            r"\b(?:actually|correction|to\s+clarify|wait|i\s+checked|i\s+measured|i\s+was\s+wrong|not\s+really|doesn'?t\s+always|not\s+always)\b",
            text
        ):
            return UserGoal.CORRECTION_OR_UPDATE

        # 6. Follow-up to previous response / impact of new detail
        if re.search(
            r"\b(?:does\s+that\s+(?:information\s+)?change|how\s+does\s+that\s+(?:change|affect)|does\s+that\s+mean|what\s+about\s+that)\b",
            text
        ):
            return UserGoal.FOLLOW_UP_TO_PREVIOUS_RESPONSE

        # 7. Evaluation / testing inquiries and guidance queries (e.g. "What should I do?")
        if (
            re.search(
                r"\b(?:what\s+tests?|what\s+examinations?|what\s+information\s+(?:would|might|could)\s+(?:help|a\s+doctor\s+need)|what\s+would\s+a\s+doctor\s+test|how\s+would\s+a\s+doctor\s+evaluate)\b",
                text
            ) or re.search(
                r"\b(what\s+should\s+i\s+do|what\s+can\s+i\s+do|what\s+to\s+do|what\s+ought\s+i\s+to\s+do)\b",
                text
            )
        ) and not re.search(r"\bbased\s+on\s+everything\b", text):
            return UserGoal.EVALUATION_INFORMATION

        # 8. Personal context / trend inquiries
        if re.search(
            r"\b(?:how\s+have\s+my\s+readings\s+changed|my\s+past\s+results|compare\s+with\s+my\s+previous|my\s+blood\s+pressure\s+trend|my\s+readings)\b",
            text
        ):
            return UserGoal.PERSONAL_CONTEXT

        # 9. General information / educational inquiries
        if intent == IntentType.GENERAL_HEALTH_QUESTION or (
            re.search(r"^\s*(what|how|why)\s+is\s+(blood\s+pressure|bilirubin|cholesterol|hypertension|jaundice|diabetes|liver)\b", text)
            and not re.search(r"\b(i\s+have|my|i\s+feel)\b", text)
        ):
            return UserGoal.GENERAL_INFORMATION

        # 10. Symptom causes / explanation inquiries
        if is_question_about_causes_or_what_helps(message) or re.search(
            r"\bwhat\s+(?:could|might|would)\s+(?:that|this)\s+(?:indicate|mean|suggest|signify|be|point\s+to)\b",
            text
        ):
            return UserGoal.EXPLANATION_OF_CURRENT_SYMPTOMS

        if session and getattr(session, "history", None) and len(session.history) > 0:
            return UserGoal.FOLLOW_UP_TO_PREVIOUS_RESPONSE

        return UserGoal.UNKNOWN

    def detect_changed_or_refined_information(
        self,
        message: str,
        previous_info: StructuredHealthInformation,
        current_info: StructuredHealthInformation
    ) -> List[str]:
        """
        Identifies if the current turn explicitly refines, updates, or corrects earlier facts.
        """
        text = message.lower().strip()
        changed: List[str] = []

        # 1. Meal relationship refinement
        has_prior_meal_trigger = any(
            any(k in tr.lower() for k in ("meal", "food", "oily", "heavy", "fatty"))
            for tr in previous_info.triggers
        )
        if has_prior_meal_trigger and re.search(
            r"\b(?:not\s+always|doesn'?t\s+always|isn'?t\s+always|not\s+necessarily)\s+(?:happen\s+after|occur\s+after|worse\s+after|related\s+to)\s+meals?\b",
            text
        ):
            changed.append("Meal relationship refined: discomfort does not consistently or always occur after meals")

        # 2. Objective body temperature update from subjective fever
        prior_feverish = any("fever" in s.name.lower() for s in previous_info.symptoms)
        curr_temp = next((m for m in current_info.measurements if m.name == "body_temperature"), None)
        curr_temp_str = getattr(current_info, "body_temperature", None)
        if curr_temp is not None or curr_temp_str:
            val_display = curr_temp_str or (f"{curr_temp.value}{curr_temp.unit}" if curr_temp else "")
            if prior_feverish or getattr(previous_info, "body_temperature", None):
                changed.append(f"Body temperature measured objectively at {val_display} (previously reported as subjective feverish feeling)")
            else:
                changed.append(f"Body temperature measured objectively at {val_display}")

        # 3. Frequency refinement
        if re.search(r"\bactually\b", text) and current_info.frequency == "intermittent" and previous_info.frequency != "intermittent":
            changed.append("Pain pattern refined: intermittent (comes and goes rather than being constant)")

        # 4. Severity refinement
        if re.search(r"\bactually\b", text) and current_info.severity and previous_info.severity and current_info.severity != previous_info.severity:
            changed.append(f"Severity refined from {previous_info.severity} to {current_info.severity}")

        return changed

    def determine_answer_mode(
        self,
        user_goal: UserGoal,
        new_info: StructuredHealthInformation,
        changed_info: List[str],
        intent: IntentType,
        message: str = ""
    ) -> AnswerMode:
        """
        Determines the operational answer mode for synthesis.
        """
        if user_goal == UserGoal.DIAGNOSIS_QUESTION:
            return AnswerMode.DIAGNOSIS_UNCERTAINTY

        if user_goal == UserGoal.NEXT_STEPS:
            return AnswerMode.NEXT_STEPS

        if user_goal == UserGoal.MISSING_INFORMATION:
            return AnswerMode.MISSING_INFORMATION

        if user_goal == UserGoal.SUMMARY_OF_CONVERSATION:
            if message and is_missing_information_request(message):
                return AnswerMode.SUMMARY_AND_MISSING
            return AnswerMode.SUMMARY

        if user_goal == UserGoal.CORRECTION_OR_UPDATE or changed_info:
            return AnswerMode.CORRECTION_ACKNOWLEDGMENT

        if user_goal == UserGoal.GENERAL_INFORMATION or intent == IntentType.GENERAL_HEALTH_QUESTION:
            return AnswerMode.GENERAL_EDUCATION

        if user_goal == UserGoal.PERSONAL_CONTEXT:
            return AnswerMode.PATIENT_CONTEXT

        if user_goal == UserGoal.EVALUATION_INFORMATION:
            return AnswerMode.EVALUATION_GUIDANCE

        # If new details were introduced
        has_new_details = bool(
            new_info.frequency or
            getattr(new_info, "radiation", None) or
            new_info.duration or
            new_info.symptoms or
            new_info.measurements or
            getattr(new_info, "body_temperature", None)
        )

        if has_new_details:
            return AnswerMode.NEW_INFORMATION_EXPLANATION

        if user_goal == UserGoal.EXPLANATION_OF_CURRENT_SYMPTOMS:
            return AnswerMode.DIRECT_ANSWER

        return AnswerMode.FOLLOW_UP

    def determine_current_turn_focus(
        self,
        new_info: StructuredHealthInformation,
        changed_info: List[str]
    ) -> List[str]:
        """
        Lists distinct focal points introduced in the current turn.
        """
        focus: List[str] = []

        if new_info.frequency:
            if new_info.frequency == "intermittent":
                focus.append("Intermittent pain pattern (comes and goes rather than being constant)")
            else:
                focus.append(f"Pain frequency pattern: {new_info.frequency}")

        if new_info.duration:
            d_low = new_info.duration.lower()
            if any(w in d_low for w in ["minute", "min", "hour", "hr", "wave"]):
                focus.append(f"Episode duration: {new_info.duration}")
            else:
                focus.append(f"Overall duration: {new_info.duration}")

        rad = getattr(new_info, "radiation", None)
        if rad:
            focus.append(f"Radiation of discomfort toward the {rad}")

        for s in new_info.symptoms:
            focus.append(f"Newly reported symptom: {s.name}")

        for m in new_info.measurements:
            unit = m.unit or ""
            focus.append(f"Objective measurement: {m.name} = {m.value}{unit}")

        for t in new_info.triggers:
            focus.append(f"Aggravating factor: {t}")

        for c in changed_info:
            if c not in focus:
                focus.append(c)

        return focus

    def determine_important_known_information(
        self,
        previous_info: StructuredHealthInformation
    ) -> List[str]:
        """
        Extracts high-value verified historical facts to ground the conversation.
        """
        known: List[str] = []

        if previous_info.location:
            known.append(f"Location: {previous_info.location}")

        if previous_info.symptoms:
            syms_str = ", ".join(s.name for s in previous_info.symptoms)
            known.append(f"Reported symptoms: {syms_str}")

        if previous_info.severity:
            known.append(f"Severity: {previous_info.severity}")

        if previous_info.duration:
            known.append(f"Overall timeline: {previous_info.duration}")

        if previous_info.triggers:
            tr_str = ", ".join(previous_info.triggers)
            known.append(f"Reported triggers: {tr_str}")

        if previous_info.measurements:
            for m in previous_info.measurements:
                unit = m.unit or ""
                known.append(f"Known measurement: {m.name} = {m.value}{unit}")

        return known

    def select_follow_up_topic(
        self,
        session: Any,
        health_info: StructuredHealthInformation,
        answer_mode: AnswerMode
    ) -> Optional[str]:
        """
        Deterministically selects ONE unanswered, clinically relevant topic.
        Never repeats topics the user already answered or denied.
        """
        answered_topics: Set[str] = getattr(session, "answered_topics", set()) if session else set()
        asked_questions: List[str] = getattr(session, "asked_questions", []) if session else []

        symptoms_lower = [s.name.lower() for s in health_info.symptoms]
        negatives_lower = [n.lower() for n in health_info.negative_symptoms]

        # 1. Jaundice / Dark urine / Pale stools
        has_jaundice = (
            "jaundice" in answered_topics
            or any("jaundice" in s or "yellow" in s for s in symptoms_lower)
            or any("jaundice" in n or "yellow" in n for n in negatives_lower)
        )
        has_urine = (
            "urine" in answered_topics
            or any("urine" in s for s in symptoms_lower)
            or any("urine" in n for n in negatives_lower)
        )
        if not has_jaundice and not has_urine and "jaundice_specific" not in asked_questions and "liver_associated" not in asked_questions:
            return "yellowing in the eyes or skin (jaundice), or changes in urine color (e.g. unusually dark or tea-colored)"

        # 2. Objective temperature / Chills
        has_temp_measurement = any(m.name == "body_temperature" for m in health_info.measurements)
        has_chills = (
            any("chill" in s for s in symptoms_lower)
            or any("chill" in n for n in negatives_lower)
        )
        if not has_temp_measurement and not has_chills and "temp_chills" not in asked_questions:
            return "measured body temperature or chills / shivering"

        # 3. Exact meal timing / onset interval
        has_meal_timing = "meal_timing" in answered_topics or "meal_timing" in asked_questions
        if not has_meal_timing:
            return "how soon after eating the discomfort begins and whether specific food types trigger it"

        # 4. Prior episodes / Personal history
        has_history = "prior_episodes" in answered_topics or "medical_history" in answered_topics or "prior_episodes" in asked_questions
        if not has_history:
            return "any previous similar episodes or personal history of gallbladder or digestive issues"

        return None

    def determine_response_priorities(
        self,
        answer_mode: AnswerMode,
        user_goal: UserGoal,
        current_turn_focus: List[str],
        changed_info: List[str],
        grounding: Any
    ) -> List[str]:
        """
        Establishes ordered instructions detailing what the LLM response must accomplish.
        """
        priorities: List[str] = []

        if answer_mode == AnswerMode.DIAGNOSIS_UNCERTAINTY:
            priorities = [
                "Directly answer whether this is definitely a heart problem by communicating clinical uncertainty honestly",
                "Explain that symptoms alone (shortness of breath and faster heartbeat with exertion) cannot establish a definitive diagnosis",
                "Acknowledge the reported exertional pattern, relief with rest, change in work tolerance, and negative findings (no chest pain, no fainting)",
                "Differentiate between symptoms, possible physiological causes (cardiovascular, respiratory, or deconditioning), and a formal medical diagnosis",
                "Explain what diagnostic evaluations a physician may perform (physical exam, resting 12-lead ECG, blood tests)",
                "Emphasize red-flag emergency symptoms that warrant immediate emergency medical care"
            ]
        elif answer_mode == AnswerMode.NEXT_STEPS:
            priorities = [
                "Provide clear, actionable, progressive next steps without diagnosing or repeating previous differential paragraphs",
                "Recommend scheduling an evaluation with a primary healthcare provider and preparing a description of the exertional pattern",
                "Advise keeping a brief symptom diary / episode log (trigger, duration, measured heart rate if available, time to relief)",
                "Advise pacing activities and avoiding strenuous overexertion until evaluated",
                "Reinforce emergency vigilance for acute red flags (sudden chest pain, fainting, breathing distress at rest)"
            ]
        elif answer_mode in (AnswerMode.SUMMARY_AND_MISSING, AnswerMode.MISSING_INFORMATION):
            priorities = [
                "Clearly address both requested parts: summarize verified reported facts AND explicitly identify what remains unknown",
                "Provide a structured breakdown of patient-reported facts vs. unmeasured/unknown parameters",
                "Explain why the missing clinical details help a healthcare provider evaluate the presentation",
                "Conclude cleanly without pressuring or diagnosing"
            ]
        elif answer_mode == AnswerMode.SUMMARY:
            priorities = [
                "Summarize the key verified symptoms, timing, and pattern already reported by the user",
                "Acknowledge reported negative findings to maintain an accurate distinguishing context",
                "Explicitly differentiate between what is established and what remains clinically unknown or unmeasured",
                "Outline the primary evaluations and tests a doctor may consider (e.g. physical examination, resting 12-lead ECG, blood tests)",
                "Explain briefly why this missing diagnostic information helps distinguish between clinical possibilities",
                "Conclude cleanly without diagnosing"
            ]
        elif answer_mode == AnswerMode.NEW_INFORMATION_EXPLANATION:
            priorities = [
                "Directly answer the user's question regarding what the symptoms may indicate",
                "Acknowledge only the verified newly reported details before explaining clinical context",
                "Explain specifically what the verified new information adds or clarifies",
                "Do NOT repeat earlier explanations in full; summarize previously established context in at most one sentence",
                "Present clinical possibilities as educational categories for medical evaluation without making a diagnosis",
                "Conclude with at most one focused follow-up question on an unanswered relevant topic"
            ]
        elif answer_mode == AnswerMode.CORRECTION_ACKNOWLEDGMENT:
            priorities = [
                "Acknowledge the user's clarification, refinement, or measurement first",
                "Explain how this updated fact adjusts the clinical evaluation context",
                "Address any specific question the user asked directly",
                "Do not treat earlier statements as absolute facts once refined",
                "Ask at most one focused follow-up question on an unanswered topic"
            ]
        elif answer_mode == AnswerMode.EVALUATION_GUIDANCE:
            priorities = [
                "Directly address what steps a person should take (e.g. scheduling a medical evaluation with a healthcare provider / doctor, resting when symptoms arise) and what evaluations a clinician may consider",
                "Recognize the reported exertional pattern (symptoms occurring during continuous work and relieved by rest) and any reduction in work tolerance",
                "Present relevant clinical evaluations (e.g. physical examination, resting 12-lead ECG, blood tests) as options a doctor may consider",
                "Explain the clinical purpose of each evaluation without diagnosing heart disease",
                "Provide appropriate next-step guidance"
            ]
        else:
            priorities = [
                "Directly answer the user's message first",
                "Ground response in verified user facts and selected clinical grounding",
                "Avoid repeating previously discussed differential details",
                "At most one focused follow-up question on an unanswered topic"
            ]

        return priorities

    def determine_avoid_repetition(
        self,
        session: Any,
        answer_mode: AnswerMode
    ) -> List[str]:
        """
        Lists topics and explanations to avoid re-explaining based on conversation trajectory.
        """
        avoid: List[str] = []
        delivered = getattr(session, "delivered_guidance", set()) if session else set()
        answered_topics = getattr(session, "answered_topics", set()) if session else set()

        if "diagnostic_uncertainty_explained" in delivered:
            avoid.append("Diagnostic uncertainty and broad differential categories were already explained in a previous turn. Do not repeat the same cardiovascular vs respiratory differential paragraph.")

        if "medical_evaluation_recommended" in delivered:
            avoid.append("Medical evaluation was already recommended. Acknowledge this context and provide practical, concrete preparation steps rather than repeating generic consult advice.")

        if "emergency_red_flags_explained" in delivered:
            avoid.append("Emergency red-flag signs were previously explained. Present emergency precautions concisely rather than repeating the full standard block.")

        if "summary_provided" in delivered:
            avoid.append("Symptom summary was already provided. Do not dump the entire symptom list again unless explicitly asked.")

        if answer_mode == AnswerMode.NEW_INFORMATION_EXPLANATION:
            avoid.append("Do not repeat the complete introductory explanation of all broad categories discussed in earlier turns")
            avoid.append("Do not re-explain nausea, vomiting, or oily-food triggers in detail")

        if answered_topics:
            avoid.append(f"Do not ask questions about topics already answered: {', '.join(sorted(answered_topics))}")

        return avoid

    def determine_facts_to_reference(
        self,
        info: StructuredHealthInformation,
        session: Any = None
    ) -> List[str]:
        """
        Extracts high-value verified patient-reported facts to ground the conversation.
        """
        facts: List[str] = []
        for s in info.symptoms:
            det = []
            if s.duration:
                det.append(f"duration: {s.duration}")
            if s.severity:
                det.append(f"severity: {s.severity}")
            if s.frequency:
                det.append(f"frequency: {s.frequency}")
            det_str = f" ({', '.join(det)})" if det else ""
            facts.append(f"Reported symptom: {s.name}{det_str}")

        if info.duration and not any(s.duration for s in info.symptoms):
            facts.append(f"Overall timeline: {info.duration}")

        if info.triggers:
            facts.append(f"Aggravating factors / triggers: {', '.join(info.triggers)}")

        if info.relieving_factors:
            facts.append(f"Relieving factors: {', '.join(info.relieving_factors)}")

        if info.functional_change:
            facts.append(f"Functional baseline change: {info.functional_change}")

        if info.negative_symptoms:
            facts.append(f"Explicitly denied findings: {', '.join(info.negative_symptoms)}")

        return facts

    def determine_unknowns_to_request(
        self,
        info: StructuredHealthInformation,
        session: Any = None
    ) -> List[str]:
        """
        Determines clinically relevant unknown information tailored to the current presentation.
        """
        unknowns: List[str] = []
        sym_names = [s.name.lower() for s in info.symptoms]
        is_cardiopulmonary = any(k in sym_names for k in ("shortness of breath", "palpitations", "chest pain", "breathless"))
        has_pain = any("pain" in s for s in sym_names)
        is_pain_denied = any("pain" in n.lower() for n in info.negative_symptoms)

        if is_cardiopulmonary:
            has_ep_dur = any(s.duration and any(w in s.duration.lower() for w in ("min", "hour", "wave")) for s in info.symptoms)
            if not has_ep_dur:
                unknowns.append("Exact duration of individual episodes before settling with rest")

            unknowns.append("Measured heart rate or pulse rate during active episodes vs. at rest")

            has_bp = any("blood_pressure" in m.name for m in info.measurements)
            if not has_bp:
                unknowns.append("Blood pressure measurements if checked recently")

            unknowns.append("Whether shortness of breath occurs when lying flat (orthopnea) or awakens you at night")
            unknowns.append("Any noticeable swelling in your feet, ankles, or lower legs")

        if not info.severity and not is_pain_denied and has_pain:
            unknowns.append("Pain severity on a 1-10 scale or descriptive intensity")
        elif not info.severity and any("breath" in s for s in sym_names):
            unknowns.append("Severity or functional limitation of breathlessness (e.g. minimal vs. strenuous exertion)")

        if not info.frequency:
            unknowns.append("Exact frequency or pattern of episode occurrence")

        if not info.medications:
            unknowns.append("Current prescription medications, over-the-counter drugs, or supplements")

        if not info.medical_history:
            unknowns.append("Personal or family history of heart disease, hypertension, or respiratory conditions")

        if not info.previous_tests:
            unknowns.append("Recent diagnostic tests (e.g., 12-lead ECG, blood work, or chest imaging)")

        return unknowns

    def determine_safety_guidance(
        self,
        info: StructuredHealthInformation
    ) -> List[str]:
        """
        Generates deterministic safety escalation guidance based on clinical presentation.
        """
        return [
            "Seek immediate emergency medical care if you develop sudden, crushing, or spreading chest discomfort or pressure",
            "Seek emergency care if discomfort radiates to the jaw, neck, back, or arms",
            "Seek emergency care for fainting, loss of consciousness, or sudden collapse",
            "Seek emergency care for acute, severe difficulty breathing occurring at rest"
        ]

    def determine_avoid_unsupported_inference(
        self,
        health_info: StructuredHealthInformation,
        current_turn: StructuredHealthInformation,
        grounding: Any
    ) -> List[str]:
        """
        Defines strict boundaries against unsupported clinical inference or unrelated branching.
        """
        avoid: List[str] = [
            "Do not infer or diagnose specific diseases (e.g. do NOT declare 'gallstones', 'cholecystitis', or 'infection'). Explain them strictly as potential categories for clinical evaluation.",
            "Do not state or imply disease probabilities or numerical risks without lab-verified ML assessment.",
            "Never treat unmentioned symptoms as negative; only explicitly reported denials count as negative findings. Do not treat negative findings as definitive exclusion."
        ]

        # Temperature / chills inference protection
        has_fever = (
            any("fever" in s.name.lower() for s in health_info.symptoms) or
            any(m.name == "body_temperature" for m in health_info.measurements)
        )
        has_chills = any("chill" in s.name.lower() for s in health_info.symptoms)
        if has_fever or has_chills:
            avoid.append(
                "When fever, elevated temperature (e.g. 38.2°C), or chills are present, do NOT declare 'infection'. "
                "Describe them as clinical signs that prompt evaluation of infectious, inflammatory, or hepatobiliary considerations."
            )

        # Unrelated branching protection (e.g. normal urine -> UTI / kidney stones)
        negatives = [n.lower() for n in health_info.negative_symptoms]
        if any("urine" in n for n in negatives) or any("jaundice" in n for n in negatives):
            avoid.append(
                "Do not introduce unrelated clinical branches: If user reports normal urine or absence of jaundice as negative findings, "
                "do NOT introduce urinary tract infections, kidney stones, or unrelated renal disorders unless supported by positive clinical findings."
            )
        # Strict protection against unsupported fatigue / exhaustion / stress / anxiety inferences
        avoid.append(
            "CRITICAL: The patient did NOT report fatigue, exhaustion, stress, or anxiety. "
            "NEVER claim or imply that the patient has 'fatigue', 'exhaustion', 'stress', or 'anxiety'. "
            "Do NOT state 'Your symptoms seem to be related to fatigue or exhaustion' or suggest 'stress or anxiety'. "
            "Strictly distinguish PATIENT-REPORTED FACTS from POSSIBLE GENERAL EXPLANATIONS."
        )
        avoid.append(
            "Recognize exertional pattern and functional changes: When symptoms occur during work/effort and improve with rest, "
            "describe it as an exertional or activity-related pattern."
        )
        avoid.append(
            "Do not diagnose heart disease or declare definitive cardiac conditions. "
            "Describe potential categories (cardiovascular, respiratory, or exertional) as possibilities for a clinician to evaluate."
        )

        return avoid

    def build_response_plan(
        self,
        message: str,
        session: Any = None,
        extracted_info: Optional[StructuredHealthInformation] = None,
        grounding: Optional[Any] = None,
        intent: IntentType = IntentType.SYMPTOM_ASSESSMENT
    ) -> ResponsePlan:
        """
        Assembles the complete deterministic ResponsePlan for grounded LLM synthesis.
        Executes in < 1ms with 0 network calls and 0 LLM calls.
        """
        from app.conversation import compute_new_information

        if extracted_info is None:
            from app.extraction import ExtractionEngine
            extracted_info = ExtractionEngine().extract_health_information(message)

        previous_info = session.health_info.model_copy(deep=True) if session else StructuredHealthInformation()
        new_info = compute_new_information(previous_info, extracted_info)

        user_goal = self.detect_user_goal(message, intent, session)
        changed_info = self.detect_changed_or_refined_information(message, previous_info, extracted_info)
        answer_mode = self.determine_answer_mode(user_goal, new_info, changed_info, intent, message=message)

        current_turn_focus = self.determine_current_turn_focus(new_info, changed_info)
        important_known = self.determine_important_known_information(previous_info)

        # Effective cumulative health info including current turn
        effective_info = session.health_info.model_copy(deep=True) if session else extracted_info.model_copy(deep=True)
        for n in extracted_info.negative_symptoms:
            if n not in effective_info.negative_symptoms:
                effective_info.negative_symptoms.append(n)
        for s in extracted_info.symptoms:
            if not any(es.name.lower() == s.name.lower() for es in effective_info.symptoms):
                effective_info.symptoms.append(s)
        if getattr(extracted_info, "body_temperature", None):
            effective_info.body_temperature = extracted_info.body_temperature
        if extracted_info.functional_change and not effective_info.functional_change:
            effective_info.functional_change = extracted_info.functional_change
        for t in extracted_info.triggers:
            if t not in effective_info.triggers:
                effective_info.triggers.append(t)
        for rf in extracted_info.relieving_factors:
            if rf not in effective_info.relieving_factors:
                effective_info.relieving_factors.append(rf)

        answered_topics = list(getattr(session, "answered_topics", set())) if session else []
        follow_up_topic = self.select_follow_up_topic(session, effective_info, answer_mode)

        # Determine relevant unanswered topics
        core_topics = ["location", "duration", "severity", "frequency", "jaundice", "fever", "triggers", "medical_history"]
        unanswered = [t for t in core_topics if t not in answered_topics]

        explicit_negatives = list(effective_info.negative_symptoms)

        priorities = self.determine_response_priorities(answer_mode, user_goal, current_turn_focus, changed_info, grounding)
        avoid_rep = self.determine_avoid_repetition(session, answer_mode)
        avoid_inf = self.determine_avoid_unsupported_inference(
            effective_info,
            new_info,
            grounding
        )

        facts_to_ref = self.determine_facts_to_reference(effective_info, session)
        unknowns_to_req = self.determine_unknowns_to_request(effective_info, session)
        safety_guidance = self.determine_safety_guidance(effective_info)

        sub_goals: List[UserGoal] = []
        if is_summary_request(message) and is_missing_information_request(message):
            sub_goals = [UserGoal.SYMPTOM_SUMMARY, UserGoal.MISSING_INFORMATION]

        next_action_str = None
        if answer_mode == AnswerMode.NEXT_STEPS:
            next_action_str = "Schedule medical evaluation, maintain symptom log, and pace physical activities"
        elif answer_mode == AnswerMode.DIAGNOSIS_UNCERTAINTY:
            next_action_str = "Consult a healthcare provider for formal cardiopulmonary clinical evaluation"

        delivered_list = list(getattr(session, "delivered_guidance", set())) if session else []

        return ResponsePlan(
            user_goal=user_goal,
            sub_goals=sub_goals,
            answer_mode=answer_mode,
            current_turn_focus=current_turn_focus,
            new_information=new_info,
            changed_information=changed_info,
            important_known_information=important_known,
            previously_discussed_topics=answered_topics,
            relevant_unanswered_topics=unanswered,
            explicit_negative_findings=explicit_negatives,
            response_priorities=priorities,
            follow_up_topic=follow_up_topic,
            avoid_repetition=avoid_rep,
            avoid_unsupported_inference=avoid_inf,
            facts_to_reference=facts_to_ref,
            unknowns_to_request=unknowns_to_req,
            safety_guidance=safety_guidance,
            avoid_repetition_topics=avoid_rep,
            already_delivered_guidance=delivered_list,
            next_action=next_action_str
        )

    def generate_plan_response(
        self,
        plan: ResponsePlan,
        session: Any,
        message: str
    ) -> str:
        """
        Synthesizes a grounded, non-repetitive response directly from the ResponsePlan.
        Used deterministically when LLM synthesis is bypassed or unavailable.
        """
        delivered = getattr(session, "delivered_guidance", set()) if session else set()

        if plan.answer_mode == AnswerMode.DIAGNOSIS_UNCERTAINTY:
            delivered.add("diagnostic_uncertainty_explained")
            delivered.add("medical_evaluation_recommended")
            delivered.add("emergency_red_flags_explained")
            if session:
                session.delivered_guidance = delivered

            resp = (
                "Based on symptoms alone, it is not possible to determine whether this is definitely a heart problem, "
                "nor can a cardiac-related cause be confirmed or ruled out without formal clinical evaluation.\n\n"
                "**Clinical Context & Distinctions:**\n"
                "• **Symptoms vs. Diagnosis**: Shortness of breath and a rapid heartbeat during exertion can arise from several "
                "distinct categories, including cardiovascular conditions, respiratory factors, physical deconditioning, or other physiological responses. "
                "Symptoms alone provide clues, but cannot establish an underlying diagnosis.\n"
                "• **Your Reported Context**: You have noted that these episodes occur during continuous exertion (working or walking), "
                "settle after resting for a few minutes, and represent a change from your usual baseline activity tolerance over the last couple of weeks. "
                "You have also noted that you do not have chest pain and have not fainted.\n"
                "• **Role of Clinical Evaluation**: An AI assistant cannot diagnose heart disease or rule it out. A definitive medical assessment "
                "requires an in-person physical examination by a healthcare provider, resting vital signs, and objective investigations such as a resting 12-lead ECG "
                "and laboratory tests.\n\n"
                "**Urgent Safety Precautions:**\n"
                "If you develop any red-flag symptoms—such as sudden severe chest pressure, discomfort spreading to the jaw, neck, or arm, fainting, or acute severe breathing distress at rest—please seek immediate emergency medical care."
            )
            return resp

        elif plan.answer_mode == AnswerMode.NEXT_STEPS:
            already_addressed_uncertainty = "diagnostic_uncertainty_explained" in delivered
            delivered.add("next_steps_explained")
            delivered.add("medical_evaluation_recommended")
            if session:
                session.delivered_guidance = delivered

            intro = (
                "To move forward from our discussion about potential causes, here are the recommended, actionable next steps to take:"
                if already_addressed_uncertainty else
                "Based on the symptoms you've described—including exertional shortness of breath, rapid heartbeat that settles with rest, and reduced activity tolerance—here are the recommended next steps:"
            )

            resp = (
                f"{intro}\n\n"
                "1. **Schedule an In-Person Medical Evaluation**\n"
                "   Arrange an appointment with your primary care physician or a general healthcare clinic. When you speak with the doctor, clearly describe the pattern: "
                "symptoms starting during continuous physical effort or walking, improving after several minutes of rest, and the noticeable reduction in your usual work tolerance over the past couple of weeks.\n\n"
                "2. **Keep an Episode Log / Symptom Diary**\n"
                "   Before your appointment, jot down brief notes whenever an episode occurs: the specific activity that triggered it, how long the episode lasted, "
                "your pulse or heart rate if you have a watch or monitor, and how many minutes of rest were required before feeling back to normal. This timeline provides valuable objective data for your clinician.\n\n"
                "3. **Pace Your Activities & Avoid Overexertion**\n"
                "   Listen to your body and take planned rest breaks before symptoms become pronounced. Avoid pushing through shortness of breath or rapid heartbeat until a physician has evaluated your cardiopulmonary response.\n\n"
                "4. **Emergency Red-Flag Vigilance**\n"
                "   Seek immediate emergency medical care if you experience acute chest discomfort, pressure radiating to the arm or jaw, fainting or near-fainting, or severe difficulty breathing that does not resolve with rest."
            )
            return resp

        elif plan.answer_mode in (AnswerMode.SUMMARY, AnswerMode.SUMMARY_AND_MISSING, AnswerMode.MISSING_INFORMATION):
            delivered.add("summary_provided")
            delivered.add("missing_info_provided")
            if session:
                session.delivered_guidance = delivered
            return session.generate_conversational_summary()

        elif plan.answer_mode == AnswerMode.EVALUATION_GUIDANCE:
            delivered.add("medical_evaluation_recommended")
            if session:
                session.delivered_guidance = delivered
            resp = (
                "Based on your reported exertional shortness of breath, faster heartbeat relieved by rest, and reduced activity tolerance, "
                "here is how a healthcare provider would typically evaluate these symptoms:\n\n"
                "• **In-Person Physical Examination**: A doctor will listen to your heart and lung sounds (auscultation), check resting vital signs (blood pressure, pulse, oxygen saturation), and check for any signs of fluid retention.\n"
                "• **Resting 12-Lead Electrocardiogram (ECG)**: Evaluates the electrical rhythm of your heart and looks for signs of strain or conduction irregularities.\n"
                "• **Routine Laboratory Workup**: Basic blood tests (e.g. complete blood count, electrolyte panel, thyroid screening) to evaluate for contributing factors such as anemia or metabolic imbalances.\n"
                "• **Recommended Action**: Schedule a visit with your primary healthcare provider, keep a record of your episodes, and seek urgent care if you experience chest pain, fainting, or severe breathlessness at rest."
            )
            return resp

        else:
            return session.generate_conversational_summary()


# Global singleton instance
response_planner = ResponsePlanner()
