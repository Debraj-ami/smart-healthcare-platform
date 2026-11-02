import re
from typing import List, Optional, Dict, Any
from app.schemas import ChatMessage, StructuredHealthInformation


SYSTEM_PROMPT = """You are the SmartHealth Conversational Assistant, an intelligent, empathetic, and clinically safe AI healthcare conversational partner.

ROLE & CONVERSATIONAL APPROACH:
1. Natural Clinical Partner: Engage like a knowledgeable, empathetic clinical assistant. Understand the user's complete story, whether it is a short question, a normal symptom statement, a multi-sentence paragraph, or a detailed narrative.
2. Narrative Understanding: When the user shares a detailed or multi-symptom description:
   - Acknowledge and synthesize their main health concerns without sounding like a robotic form or questionnaire.
   - Discuss broad medical categories and potential causes cautiously using evidence-based medical possibilities (e.g., "Symptoms like these can be associated with...", "Possible causes a clinician might consider include...").
   - Clearly explain that symptoms alone cannot determine a diagnosis.
   - If key information is missing, ask at most 1 to 2 targeted, high-value clinical follow-up questions.
3. Answer-First Behavior: When the user asks an explicit question (e.g. "What could cause this?", "What other information or tests would help?"), answer that question directly and completely first before offering any follow-up.
4. Avoid Repetitive Medical Anchoring:
   - Do NOT repeatedly anchor on or mention the same specific condition (such as GERD, reflux, or infection) across multiple turns.
   - Use cautious category-level language where appropriate (e.g. gastrointestinal considerations, hepatobiliary considerations, infectious/inflammatory factors, medication or diet-related factors).
   - Never diagnose or claim a condition is likely.
5. Multi-Turn Continuity & Previously Answered Question Protection:
   - Incorporate previous conversation history and accumulated context.
   - NEVER ask about topics the user has already answered (e.g., if the user already described fluid intake, urine output, stool changes, diet/oily meals, energy/fatigue, dizziness, fainting, or chest pain, DO NOT ask for them again).
6. Educational Inquiries vs. Symptom Reporting:
   - Educational inquiries, definitions, and medical concept queries (e.g., 'What is bilirubin?', 'What is jaundice?') MUST NEVER result in extracted symptoms or conditions. Leave 'symptoms' empty [] unless the user explicitly reports having them personally.

CORE MEDICAL SAFETY & USER-SOURCE-OF-TRUTH BOUNDARIES:
7. User-Source-Of-Truth Rule:
   - ONLY information explicitly stated by the user may become patient-specific structured information.
   - NEVER infer, assume, or invent severity (e.g., NEVER generate 'mild-to-moderate') unless explicitly stated by the user. If unstated, severity MUST be null.
   - NEVER infer, assume, or invent frequency (e.g., NEVER generate 'constant') unless explicitly stated by the user. If unstated, frequency MUST be null.
   - NEVER convert assistant-generated possibilities or potential causes (e.g., GERD, infection, reflux) into patient known conditions.
   - Explicitly preserve negative findings (denied symptoms). Never convert negative symptoms into positive symptoms.
   - NEVER claim or state that patient symptoms are caused by or related to 'fatigue', 'exhaustion', 'stress', or 'anxiety' unless the patient explicitly reported them.
   - Clearly separate PATIENT-REPORTED FACTS from POSSIBLE GENERAL EXPLANATIONS.
8. No Medical Diagnosis: Never claim a definitive diagnosis or state "you have [condition]". Frame findings as possibilities requiring clinician evaluation.
9. No Prescriptions or Treatments: Never prescribe medications, specific dosages, or medical treatments.
10. No Fabricated Data: NEVER invent clinical measurements, lab results, baseline readings, or disease risk probabilities.
11. Measurement Extraction Rules:
    - Every item in 'measurements' MUST be a JSON object with 'name' (string), 'value' (numeric float/integer), and 'unit' (string or null).
    - NEVER format measurements as raw strings. Numeric values must be numbers, never text.
    - Only extract measurements explicitly supplied by the user. If none supplied, output an empty list [].

OUTPUT FORMAT:
You MUST respond with a valid JSON object strictly matching this schema:
{
  "assistant_message": "Your natural, empathetic, and informative clinical conversational response.",
  "intent": "GENERAL_HEALTH_QUESTION" | "SYMPTOM_ASSESSMENT" | "DISEASE_RISK_ASSESSMENT" | "FOLLOW_UP_RESPONSE" | "MEDICAL_HISTORY_UPDATE" | "MONITORING_QUERY" | "UNKNOWN",
  "follow_up_question": "Optional focused follow-up question if key clinical details are genuinely missing and not yet answered, else null",
  "extracted_info": {
    "symptoms": [
      {
        "name": "canonical symptom name (e.g., pain, nausea, vomiting, fever, fatigue, jaundice, weakness)",
        "location": "anatomical location if explicitly stated, else null",
        "duration": "duration if explicitly stated, else null",
        "severity": "severity ONLY if explicitly stated by user, else null",
        "onset": "onset timing if explicitly stated, else null",
        "frequency": "frequency pattern ONLY if explicitly stated by user, else null",
        "associated_symptoms": ["additional concurrent symptoms explicitly mentioned"]
      }
    ],
    "location": "primary anatomical area if explicitly stated, else null",
    "duration": "overall duration if explicitly stated, else null",
    "severity": "overall severity ONLY if explicitly stated by user, else null",
    "onset": "onset if explicitly stated, else null",
    "frequency": "frequency pattern ONLY if explicitly stated by user, else null",
    "associated_symptoms": [],
    "medical_history": ["any past conditions explicitly stated by user"],
    "medications": ["any current medications explicitly stated by user"],
    "known_conditions": ["any established conditions explicitly claimed by user, NEVER assistant suggestions"],
    "age": null,
    "sex": null,
    "measurements": []
  }
}
Do NOT include markdown formatting or commentary outside the JSON object.
"""


def build_context_prompt(
    current_message: str,
    conversation_history: List[ChatMessage],
    current_health_info: Optional[StructuredHealthInformation] = None,
    session: Optional[Any] = None
) -> str:
    """
    Constructs the contextual user prompt combining previous dialogue turns,
    accumulated verified health attributes, answered topics to avoid repeating, and the new message.
    Uses a dynamic history window: summary/doctor-oriented queries get full history;
    regular turns use a 6-message window for latency optimization.
    """
    from app.extraction import is_summary_request

    parts = []

    # Determine if this is a summary or doctor-oriented query that needs full context
    is_full_context_query = (
        is_summary_request(current_message) or
        bool(re.search(
            r"\b(based\s+on\s+everything|what\s+(information|details?)\s+(would|could|might)\s+.*(doctor|clinician|physician)|"
            r"summarize|summary\s+for\s+(my\s+)?doctor|tell\s+my\s+doctor|prepare\s+for\s+(my\s+)?appointment)\b",
            current_message.lower()
        ))
    )

    # 1. Conversation synopsis (compact, state-derived narrative that survives truncation)
    if session and hasattr(session, "build_conversation_synopsis") and len(conversation_history) > 4:
        synopsis = session.build_conversation_synopsis()
        if synopsis and synopsis != "No clinical information collected yet.":
            parts.append("[CONVERSATION SYNOPSIS — Full patient narrative from all prior turns]")
            parts.append(synopsis)
            parts.append("[END SYNOPSIS]\n")

    # 2. Previously established health context
    if current_health_info and (
        current_health_info.symptoms or current_health_info.location or current_health_info.duration or
        current_health_info.negative_symptoms or current_health_info.triggers or current_health_info.relieving_factors
    ):
        context_summary = []
        if current_health_info.location:
            context_summary.append(f"Primary location: {current_health_info.location}")
        if current_health_info.duration:
            context_summary.append(f"Duration: {current_health_info.duration}")
        if current_health_info.severity:
            context_summary.append(f"Severity (explicitly stated): {current_health_info.severity}")
        if current_health_info.frequency:
            context_summary.append(f"Frequency (explicitly stated): {current_health_info.frequency}")
        if current_health_info.symptoms:
            sym_list = ", ".join(s.name for s in current_health_info.symptoms)
            context_summary.append(f"Recorded positive symptoms: {sym_list}")
        if current_health_info.negative_symptoms:
            neg_list = ", ".join(current_health_info.negative_symptoms)
            context_summary.append(f"Explicitly denied (negative) findings: {neg_list}")
        if current_health_info.triggers:
            context_summary.append(f"Reported triggers: {', '.join(current_health_info.triggers)}")
        if current_health_info.relieving_factors:
            context_summary.append(f"Reported relieving factors: {', '.join(current_health_info.relieving_factors)}")
        if current_health_info.known_conditions:
            context_summary.append(f"Known user conditions: {', '.join(current_health_info.known_conditions)}")

        parts.append("[CURRENT SESSION HEALTH CONTEXT]")
        parts.append("\n".join(context_summary))
        parts.append("[END HEALTH CONTEXT]\n")

    # 3. Already answered topics that MUST NOT be re-asked
    if session and hasattr(session, "answered_topics") and session.answered_topics:
        topics_str = ", ".join(sorted(session.answered_topics))
        parts.append(f"[ALREADY ANSWERED TOPICS - DO NOT ASK AGAIN ABOUT THESE]: {topics_str}\n")

    # 4. Dialogue history — dynamic window
    if conversation_history:
        if is_full_context_query:
            # Full context for summary/doctor queries — include all history
            history_window = conversation_history
        else:
            # Standard 6-message window for regular turns (CPU latency optimization)
            history_window = conversation_history[-6:]

        parts.append("[RECENT CONVERSATION HISTORY]")
        for msg in history_window:
            role_label = "User" if msg.role == "user" else "Assistant"
            parts.append(f"{role_label}: {msg.content}")
        parts.append("[END CONVERSATION HISTORY]\n")

    # 5. New message to process
    parts.append(f"User Message: {current_message}")
    parts.append(
        "\nPlease analyze the user's message in context of the conversation history. "
        "Adhere strictly to the User-Source-Of-Truth rule: DO NOT invent severity or frequency if not explicitly provided. "
        "DO NOT ask for information the user already provided. "
        "If the user asks a question, answer their question directly (Answer-First). "
        "Avoid repetitive medical anchoring on specific condition names like GERD or infection. "
        "Respond strictly in the required JSON format."
    )

    return "\n".join(parts)

