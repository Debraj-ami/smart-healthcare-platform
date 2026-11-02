"""
Grounded LLM Synthesis Layer for SmartHealth Chatbot.
Synthesizes natural, context-aware conversational responses from verified patient facts
and retrieved clinical knowledge grounding without allowing unrestricted medical generation.

Architecture:
USER MESSAGE -> Safety -> Extraction -> Verified Facts -> Clinical Knowledge Grounding -> LLM Synthesis -> USER

Responsibilities:
1. Formats compact verified patient facts (distinguishing explicit negatives from unmentioned).
2. Formats approved educational clinical grounding components.
3. Injects strict clinical boundaries (no diagnosis, no prescription, clinician evaluation).
4. Invokes the configured LLM client (OpenAI-compatible / Ollama / mock) via non-streaming or streaming.
5. Handles timeouts, network errors, or malformed responses gracefully with safe fallbacks.
"""

import time
import json
import logging
from typing import Dict, List, Optional, Any, Iterator, Tuple

from app.schemas import (
    StructuredHealthInformation,
    ChatMessage,
    ChatResponse,
    IntentType,
    SafetyStatus,
    NextActionType,
    AssessmentHandoffContract
)
from app.llm_client import BaseLLMClient, MockLLMClient, LLMError, LLMUnavailableError, LLMResponseError
from app.clinical_knowledge import ClinicalKnowledgeComponent, ClinicalGroundingContext

logger = logging.getLogger(__name__)

SYSTEM_GROUNDED_SYNTHESIS_PROMPT = """You are the conversational clinical-information assistant for SmartHealth.
Your role is to explain health information conversationally, empathetically, and safely as an ongoing dialogue.
You are NOT a diagnostic system.

You MUST follow the structured Response Plan and these clinical guardrails:
1. ANSWER THE USER'S ACTUAL QUESTION AND FOLLOW THE PLAN:
   - Answer the user's actual question directly and first.
   - Follow the designated Response Mode and Response Priorities in the plan.
   - When Summary Mode is requested, summarize the known information, distinguish what is unknown, outline useful clinician tests, and explain why they help.

2. PRIORITIZE NEW INFORMATION AND ACKNOWLEDGE CHANGES:
   - Prioritize newly reported information: Acknowledge important new facts verified from the user's report before explaining.
   - Acknowledge changed or refined information (e.g. updated meal relationship or measured temperature) instead of treating earlier statements as absolute facts.
   - Explain what the NEW information adds or changes.
   - Do not repeat previously explained material unnecessarily. If earlier possibilities remain relevant, summarize them briefly rather than restarting the whole explanation.

3. USE VERIFIED PATIENT FACTS AS SOURCE OF TRUTH:
   - Base your answer on the patient's actual reported facts (distinguishing newly reported from already known).
   - Never invent or assume symptoms, duration, severity, frequency, radiation, or medical history.
   - Do not infer facts that were not reported.
   - NEVER claim or state that the patient has 'fatigue', 'exhaustion', 'stress', or 'anxiety' unless explicitly reported by the patient. Do NOT say 'Your symptoms seem to be related to fatigue or exhaustion' or suggest 'stress or anxiety'.
   - Strictly distinguish PATIENT-REPORTED FACTS from POSSIBLE GENERAL EXPLANATIONS.
   - Distinguish explicit negative findings (symptoms the user specifically denied) from unmentioned symptoms.
   - If the user explicitly denied a symptom (e.g. "no nausea"), acknowledge that denial and never describe that symptom as present.
   - Respect explicit negative findings. Never treat negative findings as definitive exclusion claims.
   - Never treat unmentioned information as negative.

4. STAY STRICTLY WITHIN CLINICAL GROUNDING:
   - Ground your clinical explanations in the provided educational categories, distinguishing factors, and potential clinical evaluations.
   - Do NOT turn knowledge possibilities into a definitive patient diagnosis (never say "You have X" or "This means you have X").
   - Do not convert possibilities into diagnoses; do not treat symptoms as proof of a disease.
   - Do not infer unsupported causes (e.g. temperature/fever and chills warrant clinical evaluation of potential infectious or inflammatory etiologies, but do NOT declare an infection).
   - Do not introduce unrelated medical branches (e.g. do not discuss urinary tract infection or kidney stones simply because normal urine was mentioned as a negative finding).
   - Frame categories cautiously using phrases like "Clinicians often evaluate several broad possibilities, such as...", "A doctor may consider...", or "Potential areas for medical evaluation include...".

5. NON-PRESCRIPTIVE EVALUATION GUIDANCE:
   - Frame diagnostic tests and examinations cautiously ("A qualified clinician may consider...", "A doctor might recommend...").
   - Never state that a test or prescription is definitely required.
   - Do not prescribe medications or recommend specific drug treatments.

6. FOCUSED, NON-REPETITIVE FOLLOW-UP:
   - Ask at most one useful follow-up question.
   - The follow-up should target an unanswered relevant topic from the Response Plan.
   - Do not ask something that the user already answered.
   - Do not ask generic repetitive questions.
   - If no follow-up is provided, do NOT ask any follow-up questions.
"""


def format_verified_facts(health_info: StructuredHealthInformation) -> str:
    """Formats verified user facts compactly, strictly distinguishing positive from explicit negatives."""
    lines = []

    # 1. Symptoms
    if health_info.symptoms:
        sym_desc = []
        for s in health_info.symptoms:
            details = []
            if s.severity:
                details.append(f"severity: {s.severity}")
            if s.location:
                details.append(f"location: {s.location}")
            if s.duration:
                details.append(f"duration: {s.duration}")
            if s.frequency:
                details.append(f"frequency: {s.frequency}")
            if getattr(s, "radiation", None):
                details.append(f"radiation: {s.radiation}")
            det_str = f" ({', '.join(details)})" if details else ""
            sym_desc.append(f"• {s.name}{det_str}")
        lines.append("Reported Symptoms:\n" + "\n".join(sym_desc))
    elif health_info.location:
        lines.append(f"Reported Area: {health_info.location}")

    if health_info.duration and not any(s.duration for s in health_info.symptoms):
        lines.append(f"Overall Duration: {health_info.duration}")

    if health_info.severity and not any(s.severity for s in health_info.symptoms):
        lines.append(f"Overall Severity: {health_info.severity}")

    if health_info.frequency and not any(s.frequency for s in health_info.symptoms):
        lines.append(f"Frequency / Pattern: {health_info.frequency}")

    if getattr(health_info, "radiation", None) and not any(getattr(s, "radiation", None) for s in health_info.symptoms):
        lines.append(f"Radiation: {health_info.radiation}")

    if health_info.progression:
        lines.append(f"Progression: {health_info.progression}")

    if getattr(health_info, "functional_change", None):
        lines.append(f"Functional Impact: {health_info.functional_change}")

    # 2. Aggravating Triggers
    if health_info.triggers:
        lines.append("Aggravating Triggers:\n" + "\n".join(f"• {t}" for t in health_info.triggers))

    # 3. Relieving Factors
    if health_info.relieving_factors:
        lines.append("Relieving Factors:\n" + "\n".join(f"• {rf}" for rf in health_info.relieving_factors))

    # 4. Explicit Negative Findings (Denials)
    if health_info.negative_symptoms:
        lines.append(
            "Explicitly Denied Symptoms / Negative Findings (Patient explicitly does NOT have):\n" +
            "\n".join(f"• {n}" for n in health_info.negative_symptoms)
        )
    else:
        lines.append("Explicitly Denied Symptoms / Negative Findings: None reported (unmentioned symptoms must NOT be assumed negative)")

    # 5. Medical History & Medications
    if health_info.known_conditions:
        lines.append("Confirmed Medical Conditions:\n" + "\n".join(f"• {c}" for c in health_info.known_conditions))
    if health_info.medications:
        lines.append("Current Medications:\n" + "\n".join(f"• {m}" for m in health_info.medications))
    if health_info.previous_tests:
        lines.append("Previous Diagnostic Tests:\n" + "\n".join(f"• {pt}" for pt in health_info.previous_tests))

    return "\n\n".join(lines) if lines else "None reported."


def format_clinical_grounding(grounding: ClinicalGroundingContext) -> str:
    """Formats educational grounding context for the LLM."""
    lines = [f"Clinical Domain: {grounding.display_title}"]

    if grounding.possible_categories:
        lines.append("Educational Categories to Discuss (NOT diagnoses):\n" + "\n".join(f"• {c}" for c in grounding.possible_categories))

    if grounding.relevant_information:
        lines.append("Key Distinguishing Information Clinicians Consider:\n" + "\n".join(f"• {info}" for info in grounding.relevant_information))

    evals = []
    if grounding.clinical_examinations:
        evals.append("Clinical Examination: " + grounding.clinical_examinations[0])
    if grounding.laboratory_tests:
        evals.append("Laboratory Tests: " + grounding.laboratory_tests[0])
    if grounding.diagnostic_imaging:
        evals.append("Diagnostic Imaging: " + grounding.diagnostic_imaging[0])

    if evals:
        lines.append("Potential Clinical Evaluations a Clinician May Consider:\n" + "\n".join(f"• {e}" for e in evals))

    return "\n\n".join(lines)


def build_grounding_prompt(
    message: str,
    health_info: StructuredHealthInformation,
    grounding: ClinicalGroundingContext,
    session: Any = None,
    follow_up: Optional[str] = None,
    progress_context: Optional[Any] = None,
    response_plan: Optional[Any] = None
) -> str:
    """
    Constructs the compact grounding prompt combining clearly separated sections:
    1. CURRENT USER MESSAGE
    2. USER'S GOAL
    3. RESPONSE MODE
    4. VERIFIED NEW INFORMATION FROM THIS TURN
    5. CHANGED / CORRECTED INFORMATION
    6. IMPORTANT VERIFIED INFORMATION ALREADY KNOWN
    7. EXPLICIT NEGATIVE FINDINGS
    8. PREVIOUSLY DISCUSSED / ANSWERED TOPICS
    9. RELEVANT UNANSWERED TOPICS (RELEVANT UNANSWERED INFORMATION)
    10. RELEVANT CLINICAL GROUNDING
    11. RESPONSE PRIORITIES
    12. FOLLOW-UP TOPIC
    13. SAFETY / NON-DIAGNOSIS RULES
    """
    if progress_context is None and session is not None and getattr(session, "last_progression_context", None) is not None:
        progress_context = session.last_progression_context

    if response_plan is None:
        if progress_context is not None and getattr(progress_context, "response_plan", None) is not None:
            response_plan = progress_context.response_plan
        elif session is not None and getattr(session, "last_response_plan", None) is not None:
            response_plan = session.last_response_plan
        else:
            try:
                from app.response_planner import response_planner
                response_plan = response_planner.build_response_plan(
                    message=message,
                    session=session,
                    extracted_info=health_info,
                    grounding=grounding
                )
            except Exception as e:
                logger.debug("Could not construct fallback response plan: %s", e)
                response_plan = None

    if response_plan is not None:
        new_facts_str = format_verified_facts(response_plan.new_information)
        if response_plan.important_known_information:
            known_facts_str = "\n".join(f"• {k}" for k in response_plan.important_known_information)
        elif progress_context is not None:
            known_facts_str = format_verified_facts(progress_context.previously_known_information)
        else:
            known_facts_str = "None (initial turn or not previously tracked)."
        discussed_topics = response_plan.previously_discussed_topics
        unanswered_topics = response_plan.relevant_unanswered_topics
        user_goal_val = response_plan.user_goal.value if hasattr(response_plan.user_goal, "value") else str(response_plan.user_goal)
        answer_mode_val = response_plan.answer_mode.value if hasattr(response_plan.answer_mode, "value") else str(response_plan.answer_mode)
        changed_str = "\n".join(f"• {c}" for c in response_plan.changed_information) if response_plan.changed_information else "None reported."
        negatives_str = "\n".join(f"• {n}" for n in response_plan.explicit_negative_findings) if response_plan.explicit_negative_findings else "None reported."
        priorities_str = "\n".join(f"{i + 1}. {p}" for i, p in enumerate(response_plan.response_priorities))
        follow_up_topic_str = response_plan.follow_up_topic or "None (screening context sufficient or summary requested)"
        avoid_inference_str = "\n".join(f"• {a}" for a in response_plan.avoid_unsupported_inference)
    elif progress_context is not None:
        new_facts_str = format_verified_facts(progress_context.newly_reported_information)
        known_facts_str = format_verified_facts(progress_context.previously_known_information)
        discussed_topics = progress_context.previously_discussed_topics
        unanswered_topics = progress_context.unanswered_relevant_topics
        user_goal_val = "EXPLANATION_OF_CURRENT_SYMPTOMS"
        answer_mode_val = "DIRECT_ANSWER"
        changed_str = "None reported."
        negatives_str = "\n".join(f"• {n}" for n in health_info.negative_symptoms) if health_info.negative_symptoms else "None reported."
        priorities_str = "1. Answer the user's question directly\n2. Prioritize newly reported information"
        follow_up_topic_str = follow_up or "None"
        avoid_inference_str = "• Do not diagnose or infer unsupported causes\n• Respect explicit negative findings"
    else:
        new_facts_str = format_verified_facts(health_info)
        known_facts_str = "None (initial turn or not previously tracked)."
        discussed_topics = []
        unanswered_topics = []
        user_goal_val = "EXPLANATION_OF_CURRENT_SYMPTOMS"
        answer_mode_val = "DIRECT_ANSWER"
        changed_str = "None reported."
        negatives_str = "\n".join(f"• {n}" for n in health_info.negative_symptoms) if health_info.negative_symptoms else "None reported."
        priorities_str = "1. Answer the user's question directly\n2. Ground in verified patient facts"
        follow_up_topic_str = follow_up or "None"
        avoid_inference_str = "• Do not diagnose or infer unsupported causes\n• Respect explicit negative findings"

    formatted_grounding = format_clinical_grounding(grounding)

    discussed_str = "\n".join(f"• {t}" for t in discussed_topics) if discussed_topics else "None (initial turn or no previous topics)."
    unanswered_str = "\n".join(f"• {u}" for u in unanswered_topics) if unanswered_topics else "None (all primary screening topics collected)."

    # Determine if this is a summary/doctor-oriented query needing full context
    import re as _re
    is_full_context = (
        user_goal_val in ("SUMMARY_OF_CONVERSATION", "EVALUATION_INFORMATION") or
        bool(_re.search(
            r"\b(based\s+on\s+everything|what\s+(information|details?)\s+(would|could|might)\s+.*(doctor|clinician|physician)|"
            r"summarize|summary\s+for\s+(my\s+)?doctor|tell\s+my\s+doctor|prepare\s+for\s+(my\s+)?appointment)\b",
            message.lower()
        ))
    )

    history_lines = []
    if session and hasattr(session, "history"):
        if is_full_context:
            # Full history for summary / doctor queries
            prior_messages = [m for m in session.history if m.content != message]
        else:
            # Standard 2-message window for regular turns
            prior_messages = [m for m in session.history if m.content != message][-2:]
        for m in prior_messages:
            history_lines.append(f"{m.role.capitalize()}: {m.content}")

    # Inject conversation synopsis for multi-turn conversations
    synopsis_str = ""
    if session and hasattr(session, "build_conversation_synopsis") and hasattr(session, "history") and len(session.history) > 4:
        synopsis = session.build_conversation_synopsis()
        if synopsis and synopsis != "No clinical information collected yet.":
            synopsis_str = f"CONVERSATION SYNOPSIS (Full patient narrative from all prior turns):\n{synopsis}"

    prompt_parts = [
        f'CURRENT USER MESSAGE:\n"{message}"',
        f"USER'S GOAL:\n{user_goal_val}",
        f"RESPONSE MODE:\n{answer_mode_val}",
        f"NEW INFORMATION FROM THIS TURN / VERIFIED NEW INFORMATION FROM THIS TURN:\n{new_facts_str}",
        f"CHANGED / CORRECTED INFORMATION:\n{changed_str}",
        f"IMPORTANT VERIFIED INFORMATION ALREADY KNOWN (VERIFIED USER FACTS):\n{known_facts_str}",
        f"EXPLICIT NEGATIVE FINDINGS:\n{negatives_str}",
        f"PREVIOUSLY DISCUSSED TOPICS / PREVIOUSLY DISCUSSED / ANSWERED TOPICS:\n{discussed_str}",
        f"RELEVANT UNANSWERED TOPICS / RELEVANT UNANSWERED INFORMATION:\n{unanswered_str}",
        f"RELEVANT CLINICAL GROUNDING (Educational background for clinicians, NOT diagnoses):\n{formatted_grounding}",
        f"RESPONSE PRIORITIES:\n{priorities_str}",
        f"FOLLOW-UP TOPIC:\n{follow_up_topic_str}",
        f"SAFETY / NON-DIAGNOSIS RULES:\n{avoid_inference_str}"
    ]

    if synopsis_str:
        prompt_parts.append(synopsis_str)

    if history_lines:
        prompt_parts.append("RECENT CONVERSATION CONTEXT:\n" + "\n".join(history_lines))

    if follow_up:
        prompt_parts.append(
            f"APPROVED FOLLOW-UP QUESTION (You may naturally phrase this single question at the end targeting the follow-up topic):\n{follow_up}"
        )
    else:
        prompt_parts.append("NO FOLLOW-UP QUESTION: All necessary screening context is collected. Do NOT ask any follow-up questions.")

    prompt_parts.append(
        "SYNTHESIS INSTRUCTIONS:\n"
        "- Answer the user's actual question first.\n"
        "- Follow the Response Plan and prioritize current-turn changes and newly reported information.\n"
        "- When Summary Mode is requested, summarize the known information, distinguish what is unknown, outline useful clinician tests, and explain why they help.\n"
        "- Prioritize newly reported information: Acknowledge important new facts before explaining.\n"
        "- Do not repeat previously explained material unnecessarily. If previous possibilities remain relevant, summarize them briefly rather than restarting the whole explanation.\n"
        "- Explain what the NEW information adds or changes.\n"
        "- Do not infer facts that were not reported.\n"
        "- Do not convert possibilities into diagnoses; do not treat symptoms as proof of a disease.\n"
        "- Respect explicit negative findings. Never treat negative findings as definitive exclusion claims.\n"
        "- Do not introduce unrelated medical branches.\n"
        "- Never treat unmentioned information as negative.\n"
        "- Ask at most one useful follow-up question. The follow-up should target an unanswered relevant topic from the plan. Do not ask something that the user already answered. Do not ask generic repetitive questions."
    )

    return "\n\n".join(prompt_parts)


class GroundedSynthesisEngine:
    """
    Orchestrates grounded natural language response generation using Ollama / LLM client.
    Guarantees that responses are constrained by verified user facts and selected clinical grounding.
    """

    build_grounding_prompt = staticmethod(build_grounding_prompt)

    def __init__(self, llm_client: Optional[BaseLLMClient] = None):
        self._llm_client = llm_client

    def _get_client(self, fallback_client: Optional[BaseLLMClient] = None) -> Optional[BaseLLMClient]:
        if fallback_client is not None:
            return fallback_client
        if self._llm_client is not None:
            return self._llm_client
        try:
            from app.extraction import extraction_engine
            return extraction_engine.llm_client
        except Exception:
            return None

    def synthesize_response(
        self,
        session: Any,
        message: str,
        grounding: ClinicalGroundingContext,
        follow_up: Optional[str] = None,
        llm_client: Optional[BaseLLMClient] = None,
        progress_context: Optional[Any] = None,
        response_plan: Optional[Any] = None
    ) -> Optional[str]:
        """
        Generates a non-streaming grounded clinical response via the LLM client.
        Falls back safely to None if the model is unavailable or encounters an error.
        """
        client = self._get_client(llm_client)
        if not client:
            logger.info("No LLM client configured for grounded synthesis. Serving deterministic fallback.")
            return None

        # If running default MockLLMClient without canned response, return None to trigger deterministic fallback
        if isinstance(client, MockLLMClient) and (
            not getattr(client, "canned_response", None)
            or not client.canned_response.strip()
        ):
            return None

        prompt = build_grounding_prompt(
            message=message,
            health_info=session.health_info if session else StructuredHealthInformation(),
            grounding=grounding,
            session=session,
            follow_up=follow_up,
            progress_context=progress_context,
            response_plan=response_plan
        )

        start_time = time.perf_counter()
        try:
            response_text = client.generate_chat_response(
                messages=[{"role": "user", "content": prompt}],
                system_prompt=SYSTEM_GROUNDED_SYNTHESIS_PROMPT,
                temperature=0.2
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            clean = response_text.strip()
            if len(clean) < 10:
                logger.warning("Grounded synthesis returned empty/short output in %.2fms", elapsed_ms)
                return None

            # Validate response: Reject raw extraction JSON payloads or default extraction mock strings
            if (clean.startswith("{") and clean.endswith("}")) or '"intent":' in clean or '"extracted_info"' in clean:
                logger.warning("Grounded synthesis received raw JSON instead of conversational synthesis. Falling back.")
                return None
            if "[Mock LLM Response: development/testing extraction" in clean:
                logger.warning("Grounded synthesis received default extraction mock. Falling back to deterministic grounding.")
                return None

            logger.info(
                "Grounded Synthesis Success: session_id=%s, domain='%s', components=%s, latency_ms=%.2f",
                session.session_id if session else "unknown",
                grounding.domain_id,
                grounding.component_ids,
                elapsed_ms
            )
            return clean

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.warning(
                "Grounded synthesis LLM invocation failed after %.2fms: %s. Using deterministic fallback.",
                elapsed_ms, e
            )
            return None

    def stream_synthesize_response(
        self,
        session: Any,
        message: str,
        grounding: ClinicalGroundingContext,
        follow_up: Optional[str] = None,
        llm_client: Optional[BaseLLMClient] = None,
        progress_context: Optional[Any] = None,
        response_plan: Optional[Any] = None
    ) -> Iterator[str]:
        """
        Streams natural grounded response tokens from the LLM client.
        Falls back to yielding deterministic fallback chunks if the LLM fails.
        """
        client = self._get_client(llm_client)

        # Default MockLLMClient or raw extraction JSON fallback handling
        if not client or (
            isinstance(client, MockLLMClient) and (
                not getattr(client, "canned_response", None)
                or not client.canned_response.strip()
                or (client.canned_response.strip().startswith("{") and client.canned_response.strip().endswith("}"))
                or '"intent":' in client.canned_response
                or '"extracted_info"' in client.canned_response
                or "[Mock LLM Response: development/testing extraction" in client.canned_response
            )
        ):
            for chunk in self._chunk_text(grounding.deterministic_fallback_message):
                yield chunk
            return

        prompt = build_grounding_prompt(
            message=message,
            health_info=session.health_info if session else StructuredHealthInformation(),
            grounding=grounding,
            session=session,
            follow_up=follow_up,
            progress_context=progress_context,
            response_plan=response_plan
        )

        start_time = time.perf_counter()
        streamed_chars = 0
        first_token_time: Optional[float] = None

        try:
            for chunk in client.stream_chat_response(
                messages=[{"role": "user", "content": prompt}],
                system_prompt=SYSTEM_GROUNDED_SYNTHESIS_PROMPT,
                temperature=0.2
            ):
                if first_token_time is None and chunk.strip():
                    first_token_time = (time.perf_counter() - start_time) * 1000
                streamed_chars += len(chunk)
                yield chunk

            total_ms = (time.perf_counter() - start_time) * 1000
            if streamed_chars < 10:
                logger.warning("Grounded stream yielded insufficient content (%d chars). Yielding fallback.", streamed_chars)
                for chunk in self._chunk_text(grounding.deterministic_fallback_message):
                    yield chunk
            else:
                logger.info(
                    "Grounded Synthesis Stream Complete: session_id=%s, domain='%s', ttft_ms=%.2f, total_ms=%.2f",
                    session.session_id if session else "unknown",
                    grounding.domain_id,
                    first_token_time or 0.0,
                    total_ms
                )

        except Exception as e:
            total_ms = (time.perf_counter() - start_time) * 1000
            logger.warning(
                "Grounded synthesis streaming failed after %.2fms: %s. Yielding deterministic fallback.",
                total_ms, e
            )
            for chunk in self._chunk_text(grounding.deterministic_fallback_message):
                yield chunk

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = 25) -> Iterator[str]:
        """Yields text in small word-aligned or fixed chunks for streaming."""
        for i in range(0, len(text), chunk_size):
            yield text[i:i + chunk_size]


# Global singleton instance
grounded_synthesizer = GroundedSynthesisEngine()
