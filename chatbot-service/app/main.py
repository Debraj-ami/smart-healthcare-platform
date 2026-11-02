import os
import re
import json
import logging
from typing import Dict, Any, Optional, List, Tuple, Set
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import (
    ChatRequest,
    ChatResponse,
    IntentType,
    SafetyStatus,
    NextActionType,
    AssessmentHandoffContract,
    StructuredHealthInformation,
    Measurement,
    AnswerMode
)
from app.safety import SafetyChecker
from app.llm_client import (
    load_env_file,
    get_llm_client,
    LLMUnavailableError,
    LLMResponseError,
    BaseLLMClient,
    MockLLMClient
)
from app.extraction import (
    ExtractionEngine,
    is_out_of_scope_gk,
    is_casual_greeting_or_intro,
    is_summary_request,
    is_missing_information_request,
    is_next_steps_request,
    is_diagnosis_question,
    is_question_about_causes_or_what_helps
)
from app.conversation import SessionManager, FollowUpEngine
from app.liver_assessment import (
    LiverAssessmentState,
    LiverExtractor,
    format_liver_prediction_response,
    FIELD_LABELS,
    normalize_liver_field_name,
    LIVER_LAB_UNITS
)
from app.heart_assessment import (
    HeartAssessmentState,
    HeartExtractor,
    heart_extractor,
    format_heart_prediction_response,
    format_heart_history_response,
    format_heart_explanation_response,
    HEART_FIELD_LABELS,
    HEART_REQUIRED_FIELDS,
    HEART_CATEGORICAL_MAPS,
    normalize_heart_field_name
)
from app.backend_client import (
    get_backend_client,
    BackendServiceUnavailableError,
    BackendClientError
)
from app.symptom_screening import generate_symptom_screening_summary
from app.fast_path import FastPathDetector
from app.patient_context import is_patient_context_query, explain_patient_context
from app.clinical_knowledge import clinical_knowledge_engine
from app.grounded_synthesis import grounded_synthesizer

# Ensure .env runtime configuration is loaded
load_env_file()

logger = logging.getLogger(__name__)

# Initialize FastAPI application
app = FastAPI(
    title="Smart Healthcare Platform — Chatbot Service",
    description="Conversational NLP and clinical information-extraction service.",
    version="1.0.0"
)

# Enable CORS for frontend/gateway integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize core services
safety_checker = SafetyChecker()
llm_client: BaseLLMClient = get_llm_client()
extraction_engine = ExtractionEngine(llm_client=llm_client)
session_manager = SessionManager()
follow_up_engine = FollowUpEngine()
liver_extractor = LiverExtractor()
fast_path_detector = FastPathDetector(liver_extractor=liver_extractor)


@app.get("/health", tags=["System"])
def health_check():
    """Liveness and health check endpoint."""
    return {
        "status": "healthy",
        "service": "chatbot-service",
        "version": "1.0.0",
        "llm_mode": extraction_engine.mode_label
    }


def handle_liver_assessment(
    session: Any,
    request_user_id: str,
    extracted_fields: Dict[str, Any],
    validation_errors: List[str]
) -> Tuple[str, NextActionType, Optional[str], Optional[AssessmentHandoffContract]]:
    """
    Centralized Liver Disease assessment handler.
    Merges laboratory inputs and demographics, tracks validation errors and missing fields,
    and initiates Spring Boot ML prediction once all required inputs are present.
    Shared identically between Fast Path and conversational Path C.
    """
    # 1. Initialize LiverAssessmentState if not already present
    if session.liver_assessment is None:
        patient_id = None
        try:
            patient_id = int(request_user_id)
        except ValueError:
            digits = re.findall(r'\d+', request_user_id)
            patient_id = int(digits[0]) if digits else 1

        session.liver_assessment = LiverAssessmentState(
            session_id=session.session_id,
            patient_id=patient_id
        )

    # 2. Inherit previously collected demographic fields and measurements from session health info
    if session.health_info.age and "age" not in session.liver_assessment.values:
        val, err = liver_extractor.validate_field("age", session.health_info.age)
        if val is not None and not err:
            session.liver_assessment.values["age"] = val
    if session.health_info.sex and "gender" not in session.liver_assessment.values:
        val, err = liver_extractor.validate_field("gender", session.health_info.sex)
        if val is not None and not err:
            session.liver_assessment.values["gender"] = val
    for m in session.health_info.measurements:
        canonical = normalize_liver_field_name(m.name)
        if canonical and canonical not in session.liver_assessment.values:
            val, err = liver_extractor.validate_field(canonical, m.value)
            if val is not None and not err:
                session.liver_assessment.values[canonical] = val

    # 3. Update validation errors and newly extracted fields
    session.liver_assessment.validation_errors = validation_errors

    if extracted_fields:
        for k, v in extracted_fields.items():
            session.liver_assessment.values[k] = v
            if k == "age":
                session.health_info.age = v
            elif k == "gender":
                session.health_info.sex = v
            elif k in LIVER_LAB_UNITS:
                existing = next((m for m in session.health_info.measurements if m.name == k), None)
                if existing:
                    existing.value = float(v)
                else:
                    session.health_info.measurements.append(Measurement(
                        name=k,
                        value=float(v),
                        unit=LIVER_LAB_UNITS.get(k)
                    ))
        # If user updates inputs after a completed assessment, allow re-prediction
        if session.liver_assessment.status == "COMPLETED":
            session.liver_assessment.status = "IN_PROGRESS"
            session.liver_assessment.completed_assessment = None

    # 4. Formulate response based on completeness
    if session.liver_assessment.validation_errors:
        next_action = NextActionType.ASK_FOLLOW_UP
        follow_up_question = session.liver_assessment.validation_errors[0]
        assistant_message = session.liver_assessment.generate_missing_followup()
        assessment_handoff = AssessmentHandoffContract(
            next_action=NextActionType.REQUEST_DISEASE_ASSESSMENT,
            disease_candidates=["liver_disease"],
            structured_inputs=session.liver_assessment.get_collected_fields(),
            assessment_state={
                "status": "VALIDATION_ERROR",
                "validation_errors": session.liver_assessment.validation_errors,
                "collected_fields": session.liver_assessment.get_collected_fields(),
                "missing_fields": session.liver_assessment.get_missing_fields()
            }
        )
        return assistant_message, next_action, follow_up_question, assessment_handoff

    elif not session.liver_assessment.is_complete():
        next_action = NextActionType.ASK_FOLLOW_UP
        missing = session.liver_assessment.get_missing_fields()
        follow_up_question = f"Please provide: {FIELD_LABELS.get(missing[0], missing[0])}"
        assistant_message = session.liver_assessment.generate_missing_followup()
        assessment_handoff = AssessmentHandoffContract(
            next_action=NextActionType.REQUEST_DISEASE_ASSESSMENT,
            disease_candidates=["liver_disease"],
            structured_inputs=session.liver_assessment.get_collected_fields(),
            assessment_state={
                "status": "IN_PROGRESS",
                "collected_fields": session.liver_assessment.get_collected_fields(),
                "missing_fields": missing
            }
        )
        return assistant_message, next_action, follow_up_question, assessment_handoff

    else:
        # All 10 fields are present and valid -> call Spring Boot Liver prediction API
        patient_id = session.liver_assessment.patient_id or 1
        payload = session.liver_assessment.to_spring_payload(patient_id=patient_id)
        backend_client = get_backend_client()
        try:
            result = backend_client.predict_liver(payload)
            session.liver_assessment.completed_assessment = result
            session.liver_assessment.status = "COMPLETED"
            session.liver_assessment.last_error = None
            assistant_message = format_liver_prediction_response(result)
            next_action = NextActionType.REQUEST_DISEASE_ASSESSMENT
            follow_up_question = None
            assessment_handoff = AssessmentHandoffContract(
                next_action=NextActionType.REQUEST_DISEASE_ASSESSMENT,
                disease_candidates=["liver_disease"],
                structured_inputs=payload,
                assessment_result=result,
                assessment_state={
                    "status": "COMPLETED",
                    "assessment_id": result.get("assessmentId"),
                    "collected_fields": session.liver_assessment.get_collected_fields(),
                    "missing_fields": []
                }
            )
        except (BackendServiceUnavailableError, BackendClientError) as e:
            session.liver_assessment.status = "ERROR"
            session.liver_assessment.last_error = str(e)
            assistant_message = (
                "I have all 10 clinical and laboratory values required for the Liver Disease screening assessment. "
                "However, the backend assessment service is currently unavailable. "
                "Your laboratory values have been preserved in your session. "
                "Please try again in a few moments, or share these laboratory values directly with your healthcare provider."
            )
            next_action = NextActionType.REQUEST_DISEASE_ASSESSMENT
            follow_up_question = None
            assessment_handoff = AssessmentHandoffContract(
                next_action=NextActionType.REQUEST_DISEASE_ASSESSMENT,
                disease_candidates=["liver_disease"],
                structured_inputs=payload,
                assessment_result=None,
                assessment_state={
                    "status": "ERROR",
                    "error": str(e),
                    "collected_fields": session.liver_assessment.get_collected_fields(),
                    "missing_fields": []
                }
            )
        return assistant_message, next_action, follow_up_question, assessment_handoff


def format_heart_meaning_response(assessment: Dict[str, Any]) -> str:
    """
    Explains the model prediction and estimated probability in safe, grounded clinical language.
    Does NOT diagnose disease. Uses authoritative values from Spring Boot.
    """
    prob_val = assessment.get("riskScore") if assessment.get("riskScore") is not None else assessment.get("probability", 0.0)
    prob = float(prob_val)
    raw_inputs = assessment.get("inputFeatures")
    parsed_json: Dict[str, Any] = {}
    if raw_inputs and isinstance(raw_inputs, str):
        try:
            parsed_json = json.loads(raw_inputs)
        except Exception:
            parsed_json = {}

    threshold = float(parsed_json.get("threshold") or assessment.get("threshold") or 0.31)
    pred_raw = parsed_json.get("prediction") if parsed_json.get("prediction") is not None else assessment.get("prediction")
    pred_val = int(pred_raw) if pred_raw is not None else (1 if prob >= threshold else 0)
    pred_label = "Positive (Elevated risk score)" if pred_val == 1 else "Negative (Below screening cutoff)"
    ass_id = assessment.get("id") or assessment.get("assessmentId") or "Current"

    lines = [
        f"Here is the detailed explanation for Heart Assessment #{ass_id}:",
        f"• Model-estimated probability: {prob * 100:.1f}% ({prob:.4f})",
        f"• Decision threshold: t = {threshold:.2f}",
        f"• Binary model prediction: {pred_label}",
        "",
        "Understanding What This Result Means:",
        f"The machine-learning model calculated an estimated probability of approximately {prob * 100:.1f}% for cardiovascular risk based on the clinical parameters provided.",
        f"Because this estimated value is {'at or above' if pred_val == 1 else 'below'} the workflow operating cutoff of {threshold * 100:.0f}%, the recorded model screening prediction was {pred_label.split(' ')[0]}.",
        "",
        "Clinical Context & Safety Boundary:",
        "This result is an automated statistical risk screening output, NOT a definitive medical diagnosis.",
        "A negative screening prediction does not guarantee the complete absence of cardiovascular conditions, nor does an elevated score prove that heart disease is present. Only a licensed physician or cardiologist can provide a diagnosis through comprehensive clinical examination and diagnostic testing (such as ECG, cardiac imaging, or lab tests).",
        "",
        "Would you like me to explain the SHAP factor contributions that influenced this model output, compare this with previous assessments, or discuss questions for your doctor?"
    ]
    return "\n".join(lines)


def format_heart_safety_response(assessment: Optional[Dict[str, Any]]) -> str:
    """
    Addresses diagnostic certainty questions (e.g. 'Does this mean I have heart disease?').
    Strictly clarifies that model prediction is not a diagnosis.
    """
    ass_part = ""
    if assessment:
        prob_val = assessment.get("riskScore") if assessment.get("riskScore") is not None else assessment.get("probability", 0.0)
        prob = float(prob_val)
        raw_inputs = assessment.get("inputFeatures")
        parsed_json: Dict[str, Any] = {}
        if raw_inputs and isinstance(raw_inputs, str):
            try:
                parsed_json = json.loads(raw_inputs)
            except Exception:
                parsed_json = {}
        threshold = float(parsed_json.get("threshold") or assessment.get("threshold") or 0.31)
        pred_raw = parsed_json.get("prediction") if parsed_json.get("prediction") is not None else assessment.get("prediction")
        pred_val = int(pred_raw) if pred_raw is not None else (1 if prob >= threshold else 0)
        pred_label = "Positive" if pred_val == 1 else "Negative"
        ass_id = assessment.get("id") or assessment.get("assessmentId") or "Current"
        ass_part = f" In your recorded Assessment #{ass_id}, the model-estimated output was approximately {prob * 100:.1f}%. The workflow's operating threshold was {threshold * 100:.0f}%, so the recorded model prediction was {pred_label}."

    return (
        f"No.{ass_part} This is a model screening output and is not a diagnosis.\n\n"
        "Here is why this distinction is critical:\n"
        "1. Screening vs. Diagnosis: Machine learning models evaluate statistical associations across recorded features, but cannot account for all clinical variables, physical symptoms, or biological nuances.\n"
        "2. No Absolute Guarantee: A negative screening result does not guarantee that you are free of heart disease, nor does an elevated score prove that disease exists.\n"
        "3. Physician Evaluation Required: Only a qualified healthcare provider or cardiologist can diagnose cardiovascular conditions through direct medical evaluation, personal and family medical history, and clinical diagnostic tests such as electrocardiograms (ECG), stress tests, or echocardiograms.\n\n"
        "If you ever experience symptoms such as severe chest pressure, radiating pain to your arm or jaw, sudden shortness of breath, or fainting, seek emergency medical care immediately."
    )


def format_heart_comparison_response(
    current_assessment: Optional[Dict[str, Any]],
    all_assessments: List[Dict[str, Any]]
) -> str:
    """
    Compares the current Heart assessment with previous historical assessments.
    Strictly avoids equating model probability changes with clinical disease progression.
    """
    if not all_assessments:
        return "No historical Heart assessments were found in your patient records to compare."

    sorted_ass = sorted(
        all_assessments,
        key=lambda a: a.get("assessedAt") or "",
        reverse=True
    )

    curr = current_assessment or sorted_ass[0]
    curr_id = curr.get("id") or curr.get("assessmentId")
    curr_prob = float(curr.get("riskScore") if curr.get("riskScore") is not None else curr.get("probability", 0.0))
    curr_date = curr.get("assessedAt", "N/A")

    prior_assessments = [a for a in sorted_ass if (a.get("id") or a.get("assessmentId")) != curr_id]

    if not prior_assessments:
        return (
            f"You currently have one recorded Heart assessment on file (#{curr_id} on {curr_date}, model output: {curr_prob * 100:.1f}%). "
            "There are no earlier assessments recorded in your profile to compare against."
        )

    prev = prior_assessments[0]
    prev_id = prev.get("id") or prev.get("assessmentId")
    prev_prob = float(prev.get("riskScore") if prev.get("riskScore") is not None else prev.get("probability", 0.0))
    prev_date = prev.get("assessedAt", "N/A")

    diff_prob = curr_prob - prev_prob
    diff_sign = f"+{diff_prob * 100:.1f}%" if diff_prob > 0 else f"{diff_prob * 100:.1f}%"

    lines = [
        "Comparison of Heart Disease Screening Assessments:",
        f"• Current Assessment (#{curr_id}): {curr_prob * 100:.1f}% (recorded on {curr_date})",
        f"• Previous Assessment (#{prev_id}): {prev_prob * 100:.1f}% (recorded on {prev_date})",
        f"• Model Output Difference: {diff_sign}",
        "",
        "Important Clinical Context:",
        "Differences in model-estimated outputs between assessments reflect changes in the entered clinical features and model calculations at those points in time. "
        "The model-estimated outputs differed between the two assessments, but this does NOT equate to clinical disease progression or improvement. Please review your overall cardiovascular health with your physician."
    ]

    if len(prior_assessments) > 1:
        lines.append("\nOther historical assessments on record:")
        for extra in prior_assessments[1:3]:
            eid = extra.get("id") or extra.get("assessmentId")
            eprob = float(extra.get("riskScore") if extra.get("riskScore") is not None else extra.get("probability", 0.0))
            edate = extra.get("assessedAt", "N/A")
            lines.append(f"• #{eid}: {eprob * 100:.1f}% on {edate}")

    return "\n".join(lines)


def format_cross_source_guidance_response(
    patient_id: int,
    assessment: Optional[Dict[str, Any]],
    monitoring_data: Dict[str, Any],
    health_info: Optional[StructuredHealthInformation] = None
) -> str:
    """
    Synthesizes multi-source context (Heart assessment + monitoring trends + symptoms)
    for grounded, non-diagnostic physician discussion support.
    Strictly maintains non-causality between separate data sources.
    """
    lines = [
        "Summary for Your Healthcare Provider Discussion:",
        ""
    ]

    # 1. Heart Assessment Section
    if assessment:
        prob_val = assessment.get("riskScore") if assessment.get("riskScore") is not None else assessment.get("probability", 0.0)
        prob = float(prob_val)
        raw_inputs = assessment.get("inputFeatures")
        parsed_json: Dict[str, Any] = {}
        if raw_inputs and isinstance(raw_inputs, str):
            try:
                parsed_json = json.loads(raw_inputs)
            except Exception:
                parsed_json = {}
        threshold = float(parsed_json.get("threshold") or assessment.get("threshold") or 0.31)
        pred_raw = parsed_json.get("prediction") if parsed_json.get("prediction") is not None else assessment.get("prediction")
        pred_val = int(pred_raw) if pred_raw is not None else (1 if prob >= threshold else 0)
        pred_label = "Positive (Elevated)" if pred_val == 1 else "Negative (Below cutoff)"
        ass_id = assessment.get("id") or assessment.get("assessmentId") or "N/A"
        ass_date = assessment.get("assessedAt", "N/A")

        lines.extend([
            f"1. Heart Disease Screening Output (Assessment #{ass_id}, {ass_date}):",
            f"   • Model-estimated probability: {prob * 100:.1f}% (Operating cutoff: {threshold * 100:.0f}%)",
            f"   • Recorded model prediction: {pred_label}",
        ])
        pos_contribs = parsed_json.get("positiveContributors") or []
        neg_contribs = parsed_json.get("negativeContributors") or []
        if pos_contribs:
            top_pos = [c.get("displayName") or c.get("feature") for c in pos_contribs[:2] if c.get("displayName") or c.get("feature")]
            if top_pos:
                lines.append(f"   • Key factors contributing toward model output: {', '.join(top_pos)}")
        if neg_contribs:
            top_neg = [c.get("displayName") or c.get("feature") for c in neg_contribs[:2] if c.get("displayName") or c.get("feature")]
            if top_neg:
                lines.append(f"   • Key factors contributing away from model output: {', '.join(top_neg)}")
        lines.append("")

    # 2. Monitoring Context Section
    trends = monitoring_data.get("trends", [])
    baselines = monitoring_data.get("baselines", [])
    observations = monitoring_data.get("recent_observations", [])

    lines.append("2. Recent Health Monitoring & Baselines:")
    latest_trends: Dict[str, Dict[str, Any]] = {}
    for t in trends:
        m = t.get("metricType")
        if m and m not in latest_trends:
            latest_trends[m] = t

    if latest_trends:
        for m_name, t in latest_trends.items():
            curr = t.get("currentValue")
            base = t.get("baselineValue")
            direction = t.get("trendDirection", "STABLE")
            signed_dev = t.get("deviationFromBaseline")
            unit = "mg/dL" if "GLUCOSE" in m_name else "bpm" if "HEART" in m_name else "mmHg" if "PRESSURE" in m_name else ""
            unit_str = f" {unit}" if unit else ""
            dev_str = f" ({signed_dev:+.1f}{unit_str} from baseline)" if signed_dev is not None else ""
            lines.append(f"   • {m_name.replace('_', ' ').title()}: Latest reading {curr}{unit_str}, personal baseline {base}{unit_str}{dev_str}, trend: {direction.lower()}")
    elif observations:
        latest_obs = observations[-1]
        lines.append(f"   • Latest logged observation: {latest_obs.get('metricType')} = {latest_obs.get('metricValue')} {latest_obs.get('unit', '')}")
    else:
        lines.append("   • No longitudinal monitoring observations currently recorded in profile.")
    lines.append("")

    # 3. Reported Symptoms (if any from conversation)
    if health_info and (health_info.symptoms or health_info.duration):
        sym_names = [s.name for s in health_info.symptoms]
        if sym_names:
            lines.append(f"3. Reported Symptoms in Conversation:\n   • Reported: {', '.join(sym_names)}" + (f" (duration: {health_info.duration})" if health_info.duration else ""))
            lines.append("")

    # 4. Recommended Physician Questions
    lines.extend([
        "Key Questions Worth Discussing With Your Doctor:",
        "• How do my clinical features (blood pressure, cholesterol, resting markers) compare with my personalized cardiovascular targets?",
        "• Does my health monitoring trend warrant any dietary, lifestyle, or clinical medication adjustments?",
        "• Would any formal diagnostic testing (such as a 12-lead ECG, stress test, or lab lipid panel) be beneficial at this time?",
        "",
        "Clinical & Safety Disclaimer:",
        "Your monitoring observations and your Heart assessment output are separate pieces of information that you can discuss with your healthcare professional. Statistical trends in your health monitoring do not prove or cause heart disease risk. This platform provides screening and decision support, not medical diagnosis or treatment. Always discuss these results with your healthcare provider for clinical interpretation."
    ])

    return "\n".join(lines)


def is_heart_history_or_explanation_query(message: str, session: Optional[Any] = None) -> Tuple[bool, str]:
    """
    Detects if the user is asking about prior Heart assessments, assessment count,
    assessment results, or SHAP/model explanations for an existing Heart assessment.
    Returns (is_match, query_type) where query_type is:
      'CROSS_SOURCE', 'SAFETY_INQUIRY', 'SHAP', 'COMPARISON', 'EXPLANATION', or 'HISTORY'.
    """
    text_lower = message.strip().lower()

    # Reject general educational inquiries like "what is heart disease", "what is chest pain", "what causes a heart attack"
    is_general_edu = bool(re.search(
        r"^(what\s+is|what\s+are|define|explain|tell\s+me\s+about|what\s+causes)\s+(heart\s+disease|chest\s+pain|angina|heart\s+attack|coronary\s+artery)\b",
        text_lower
    )) and not any(k in text_lower for k in ("my", "i", "mine", "score", "result", "assessment", "why did", "why was", "previous", "last", "this result", "this assessment", "doctor", "discuss"))

    if is_general_edu:
        return False, ""

    has_heart = any(k in text_lower for k in ("heart", "cardiac", "cardiovascular"))
    has_recent_assessment = bool(
        session and getattr(session, "heart_assessment", None)
        and session.heart_assessment.status == "COMPLETED"
    )
    has_assessment_context = bool(
        session and (
            getattr(session, "verified_assessment", None) is not None
            or (getattr(session, "active_context", None) and (
                session.active_context.get("source") == "assessment"
                or session.active_context.get("assessment_id")
                or session.active_context.get("assessmentId")
                or session.active_context.get("disease") == "HEART"
            ))
        )
    )

    # 1. Combined Cross-Source / Doctor Discussion
    is_cross_source = bool(re.search(
        r"\b(considering\s+my\s+.*(monitoring|assessment)|what\s+should\s+i\s+discuss\s+with\s+(a\s+|my\s+)?doctor|how\s+does\s+this\s+compare\s+with\s+my\s+.*monitoring|monitoring\s+and\s+.*assessment|assessment\s+and\s+.*monitoring)\b",
        text_lower
    ))
    if is_cross_source and (has_heart or has_assessment_context or (session and getattr(session, "active_context", None))):
        return True, "CROSS_SOURCE"

    if not has_heart and not has_recent_assessment and not has_assessment_context:
        return False, ""

    # 2. Safety / Diagnostic certainty inquiry
    is_safety_diag = bool(re.search(
        r"\b(does\s+this\s+mean\s+i\s+(definitely\s+)?have\s+heart\s+disease|do\s+i\s+have\s+heart\s+disease|am\s+i\s+diagnosed\s+with\s+heart\s+disease)\b",
        text_lower
    ))
    if is_safety_diag:
        return True, "SAFETY_INQUIRY"

    # 3. SHAP / factor / contributor inquiry
    is_shap = bool(re.search(
        r"\b(factors?|contribut(ed|ing|ion|or|ors)?|influenc(ed|ing)?|shap|why\s+did\s+(the\s+)?model|why\s+did\s+i\s+get\s+this\s+result|why\s+did\s+it\s+give\s+me|what\s+contributed|what\s+influenced)\b",
        text_lower
    ))
    if is_shap:
        return True, "SHAP"

    # 4. Comparison inquiry
    is_comparison = bool(re.search(
        r"\b(how\s+does\s+this\s+compare\s+with\s+my\s+previous|compare\s+(with|to)\s+my\s+previous|how\s+does\s+this\s+compare\s+with\s+earlier|previous\s+vs\s+current)\b",
        text_lower
    ))
    if is_comparison:
        return True, "COMPARISON"

    # 5. Assessment Meaning / Explanation inquiry
    is_explanation = bool(re.search(
        r"\b(what\s+does\s+this\s+result\s+mean|what\s+does\s+my\s+.*(result|assessment|score)\s+mean|what\s+did\s+my\s+.*(show|say)|why\s+was\s+my\s+.*(assessment|score|result)|why\s+is\s+my\s+.*(assessment|score|result)|explain\s+(this\s+|my\s+).*(assessment|result|score)|interpret\s+(this\s+|my\s+).*(assessment|result)|what\s+does\s+this\s+mean)\b",
        text_lower
    ))
    if is_explanation:
        return True, "EXPLANATION"

    # 6. History inquiry
    is_history = bool(re.search(
        r"\b(have\s+i\s+taken|how\s+many\s+.*assessments?|previous\s+.*assessment|last\s+.*assessment|prior\s+.*assessment|history\s+of\s+.*assessment|my\s+previous\s+.*(result|score)|show\s+(me\s+)?my\s+.*assessment|what\s+was\s+my\s+.*(result|score|probability|assessment)|past\s+heart\s+assessment)\b",
        text_lower
    ))
    if is_history:
        return True, "HISTORY"

    return False, ""


def handle_heart_history_or_explanation(
    message: str,
    session: Any,
    request_user_id: str,
    heart_query_type: str
) -> str:
    """
    Centralized handler for Heart assessment inquiries, explanations, SHAP factor attributions,
    safety checks, comparisons, and cross-source doctor guidance.
    """
    patient_id = None
    if request_user_id and request_user_id not in ("anonymous", "guest", "default_user", ""):
        try:
            patient_id = int(request_user_id)
        except ValueError:
            digits = re.findall(r'\d+', request_user_id)
            patient_id = int(digits[0]) if digits else None

    if not patient_id:
        return (
            "Patient-specific Heart assessment records and explanations require an authenticated patient context. "
            "Please sign in with your patient account to view your prior assessment history or SHAP model explanations."
        )

    backend_client = get_backend_client()
    try:
        assessments = backend_client.get_patient_heart_assessments(patient_id)
        target_assessment = getattr(session, "verified_assessment", None)
        if not target_assessment and assessments:
            target_assessment = assessments[0]

        if heart_query_type == "CROSS_SOURCE":
            monitoring_data = backend_client.get_patient_health_context(patient_id)
            return format_cross_source_guidance_response(
                patient_id=patient_id,
                assessment=target_assessment,
                monitoring_data=monitoring_data,
                health_info=session.health_info
            )
        elif heart_query_type == "SAFETY_INQUIRY":
            return format_heart_safety_response(target_assessment)
        elif heart_query_type == "COMPARISON":
            return format_heart_comparison_response(target_assessment, assessments)
        elif heart_query_type == "SHAP":
            if target_assessment:
                return format_heart_explanation_response(target_assessment)
            elif assessments:
                return format_heart_explanation_response(assessments[0])
            elif session.heart_assessment and session.heart_assessment.completed_assessment:
                return format_heart_explanation_response(session.heart_assessment.completed_assessment)
            else:
                return (
                    "No prior Heart Disease assessments were found for your patient profile. "
                    "Would you like me to guide you through a new Heart Disease risk assessment?"
                )
        elif heart_query_type == "EXPLANATION":
            if target_assessment:
                return format_heart_meaning_response(target_assessment)
            elif assessments:
                return format_heart_meaning_response(assessments[0])
            else:
                return "No prior Heart Disease assessments were found for your patient profile."
        elif heart_query_type == "HISTORY":
            return format_heart_history_response(assessments)
        else:
            return format_heart_history_response(assessments)
    except (BackendServiceUnavailableError, BackendClientError) as e:
        logger.warning("Spring Boot heart history/explanation retrieval failed: %s", e)
        return (
            "I am currently unable to retrieve your historical Heart assessment records from the backend service. "
            "Please try again in a few moments."
        )


def handle_heart_assessment(
    session: Any,
    request_user_id: str,
    extracted_fields: Dict[str, Any],
    explicit_nulls: Set[str],
    validation_errors: List[str],
    user_message: str = ""
) -> Tuple[str, NextActionType, Optional[str], Optional[AssessmentHandoffContract]]:
    """
    Centralized Heart Disease assessment handler.
    Merges 13 clinical inputs, tracks validation errors and missing fields,
    implements a user confirmation step before prediction, and invokes
    Spring Boot POST /api/assessments/heart/predict upon confirmation.
    """
    # 1. Initialize HeartAssessmentState if not already present
    if session.heart_assessment is None:
        patient_id = None
        try:
            patient_id = int(request_user_id)
        except ValueError:
            digits = re.findall(r'\d+', request_user_id)
            patient_id = int(digits[0]) if digits else 1

        session.heart_assessment = HeartAssessmentState(
            session_id=session.session_id,
            patient_id=patient_id
        )

    # 2. Inherit previously collected demographic fields from session health info
    if session.health_info.age and "age" not in session.heart_assessment.values:
        val, err = heart_extractor.validate_field("age", session.health_info.age)
        if val is not None and not err:
            session.heart_assessment.values["age"] = val
    if session.health_info.sex and "sex" not in session.heart_assessment.values:
        val, err = heart_extractor.validate_field("sex", session.health_info.sex)
        if val is not None and not err:
            session.heart_assessment.values["sex"] = val

    # 3. Handle explicit user cancellation / reset
    text_lower = user_message.strip().lower()
    if any(k in text_lower for k in ("cancel assessment", "stop assessment", "reset assessment", "cancel heart")):
        session.heart_assessment.status = "CANCELLED"
        session.heart_assessment.confirmation_pending = False
        return (
            "Your Heart Disease screening assessment has been cancelled. "
            "Let me know if you would like to start over or discuss other health inquiries.",
            NextActionType.PROVIDE_GENERAL_INFO,
            None,
            None
        )

    # 3a. Check for ambiguous qualitative measurements like "My BP is high" without numeric values
    ambiguous_metric = heart_extractor.detect_ambiguous_measurement(user_message)
    if ambiguous_metric:
        label = HEART_FIELD_LABELS.get(ambiguous_metric, ambiguous_metric)
        unit = "mm Hg" if ambiguous_metric == "trestbps" else "bpm"
        msg = (
            f"What is your measured {label.lower()} value (in {unit})? "
            "The Heart screening model requires an exact numeric measurement. "
            "If you don't have an exact measurement right now, that's okay — we can continue with your health monitoring or discuss any symptoms without running the Heart screening model."
        )
        return msg, NextActionType.ASK_FOLLOW_UP, f"What is your measured {label}?", None

    # 3b. Check if user explicitly states they do not have one or more required clinical values
    disclaimed_fields = heart_extractor.detect_disclaimed_required_fields(user_message)
    if disclaimed_fields:
        if "general_labs" in disclaimed_fields:
            field_desc = "clinical and laboratory test results (such as serum cholesterol, resting ECG, and resting blood pressure)"
        else:
            names = [HEART_FIELD_LABELS.get(f, f) for f in disclaimed_fields]
            field_desc = " and ".join(names)

        msg = (
            f"The current Heart Disease screening model requires {field_desc} to compute an accurate risk score. "
            "I won't estimate, invent, or use assumed population averages for missing medical values. "
            "If you have a validated measurement or lab report later, you can complete the clinical assessment at any time.\n\n"
            "In the meantime, you can still use Smart Health for:\n"
            "• Symptom conversations — describe any symptoms you are experiencing for safety screening and clinical guidance.\n"
            "• Health monitoring — track daily observations like blood pressure, heart rate, personal baselines, and trend deviations.\n\n"
            "Would you like to discuss any symptoms you are having, or review your health monitoring measurements?"
        )
        return msg, NextActionType.PROVIDE_GENERAL_INFO, None, None

    # 4. Check if we are currently awaiting confirmation
    is_awaiting_conf = (
        session.heart_assessment.status == "AWAITING_CONFIRMATION"
        or session.heart_assessment.confirmation_pending
    )

    is_user_confirming = any(k in text_lower for k in (
        "yes", "confirm", "proceed", "run", "go ahead", "please do", "predict", "sure", "ok", "run it", "yes proceed"
    )) and not any(k in text_lower for k in ("no", "don't", "dont", "cancel", "wait", "actually"))

    # 5. Merge new extracted fields, nulls, and validation errors
    if validation_errors:
        session.heart_assessment.validation_errors = validation_errors

    if explicit_nulls:
        session.heart_assessment.explicit_nulls.update(explicit_nulls)
        for null_field in explicit_nulls:
            if null_field in session.heart_assessment.values:
                del session.heart_assessment.values[null_field]

    if extracted_fields:
        for k, v in extracted_fields.items():
            session.heart_assessment.values[k] = v
            if k in session.heart_assessment.explicit_nulls:
                session.heart_assessment.explicit_nulls.remove(k)
            if k == "age":
                session.health_info.age = v
            elif k == "sex":
                session.health_info.sex = "Male" if v == 1 else "Female"

        # If user corrected a value while awaiting confirmation or completed, clear completed and re-evaluate
        if session.heart_assessment.status == "COMPLETED":
            session.heart_assessment.status = "IN_PROGRESS"
            session.heart_assessment.completed_assessment = None
            session.heart_assessment.confirmation_pending = False

    # 6. If awaiting confirmation and user confirmed -> Run prediction!
    if is_awaiting_conf and is_user_confirming and session.heart_assessment.is_all_collected():
        patient_id = session.heart_assessment.patient_id or 1
        payload = session.heart_assessment.to_spring_payload(patient_id=patient_id)
        backend_client = get_backend_client()
        try:
            result = backend_client.predict_heart(payload)
            session.heart_assessment.completed_assessment = result
            session.heart_assessment.status = "COMPLETED"
            session.heart_assessment.confirmation_pending = False
            session.heart_assessment.last_error = None
            assistant_message = format_heart_prediction_response(result)
            next_action = NextActionType.REQUEST_DISEASE_ASSESSMENT
            follow_up_question = None
            assessment_handoff = AssessmentHandoffContract(
                next_action=NextActionType.REQUEST_DISEASE_ASSESSMENT,
                disease_candidates=["heart_disease"],
                structured_inputs=payload,
                assessment_result=result,
                assessment_state={
                    "status": "COMPLETED",
                    "assessment_id": result.get("assessmentId"),
                    "collected_fields": session.heart_assessment.get_collected_fields(),
                    "missing_fields": []
                }
            )
            return assistant_message, next_action, follow_up_question, assessment_handoff
        except (BackendServiceUnavailableError, BackendClientError) as e:
            session.heart_assessment.status = "ERROR"
            session.heart_assessment.last_error = str(e)
            assistant_message = (
                "I have all 13 clinical features required for the Heart Disease assessment. "
                "However, the assessment service is currently unavailable. No result was generated. "
                "Your inputs have been preserved in your session. Please try again shortly."
            )
            next_action = NextActionType.REQUEST_DISEASE_ASSESSMENT
            follow_up_question = None
            assessment_handoff = AssessmentHandoffContract(
                next_action=NextActionType.REQUEST_DISEASE_ASSESSMENT,
                disease_candidates=["heart_disease"],
                structured_inputs=payload,
                assessment_result=None,
                assessment_state={
                    "status": "ERROR",
                    "error": str(e),
                    "collected_fields": session.heart_assessment.get_collected_fields(),
                    "missing_fields": []
                }
            )
            return assistant_message, next_action, follow_up_question, assessment_handoff

    # 7. If there are validation errors -> show errors and prompt
    if session.heart_assessment.validation_errors:
        next_action = NextActionType.ASK_FOLLOW_UP
        follow_up_question = session.heart_assessment.validation_errors[0]
        assistant_message = session.heart_assessment.generate_missing_followup()
        assessment_handoff = AssessmentHandoffContract(
            next_action=NextActionType.REQUEST_DISEASE_ASSESSMENT,
            disease_candidates=["heart_disease"],
            structured_inputs=session.heart_assessment.get_collected_fields(),
            assessment_state={
                "status": "VALIDATION_ERROR",
                "validation_errors": session.heart_assessment.validation_errors,
                "collected_fields": session.heart_assessment.get_collected_fields(),
                "missing_fields": session.heart_assessment.get_missing_fields()
            }
        )
        return assistant_message, next_action, follow_up_question, assessment_handoff

    # 8. If missing fields remain -> prompt for missing
    if not session.heart_assessment.is_all_collected():
        session.heart_assessment.status = "IN_PROGRESS"
        session.heart_assessment.confirmation_pending = False
        next_action = NextActionType.ASK_FOLLOW_UP
        missing = session.heart_assessment.get_missing_fields()
        follow_up_question = f"Please provide: {HEART_FIELD_LABELS.get(missing[0], missing[0])}"
        assistant_message = session.heart_assessment.generate_missing_followup()
        assessment_handoff = AssessmentHandoffContract(
            next_action=NextActionType.REQUEST_DISEASE_ASSESSMENT,
            disease_candidates=["heart_disease"],
            structured_inputs=session.heart_assessment.get_collected_fields(),
            assessment_state={
                "status": "IN_PROGRESS",
                "collected_fields": session.heart_assessment.get_collected_fields(),
                "missing_fields": missing
            }
        )
        return assistant_message, next_action, follow_up_question, assessment_handoff

    # 9. All 13 fields are collected! Request confirmation before prediction (Part 6)
    session.heart_assessment.status = "AWAITING_CONFIRMATION"
    session.heart_assessment.confirmation_pending = True
    next_action = NextActionType.ASK_FOLLOW_UP
    follow_up_question = "Would you like me to run the Heart Disease risk assessment?"
    assistant_message = session.heart_assessment.generate_confirmation_summary()
    assessment_handoff = AssessmentHandoffContract(
        next_action=NextActionType.REQUEST_DISEASE_ASSESSMENT,
        disease_candidates=["heart_disease"],
        structured_inputs=session.heart_assessment.get_collected_fields(),
        assessment_state={
            "status": "AWAITING_CONFIRMATION",
            "collected_fields": session.heart_assessment.get_collected_fields(),
            "missing_fields": []
        }
    )
    return assistant_message, next_action, follow_up_question, assessment_handoff


@app.post("/api/chat", response_model=ChatResponse, tags=["Chat"])
def process_chat(request: ChatRequest) -> ChatResponse:
    """
    Main conversational endpoint:
    Processes natural-language health messages, extracts structured clinical information,
    applies safety red-flag checks, conducts follow-up triage, and prepares assessment handoffs.
    """
    message = request.message.strip()
    if not message:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message content cannot be empty."
        )

    # 1. Conservative Safety Red-Flag Check (Deterministic - runs BEFORE any LLM invocation)
    safety_status, triggers, emergency_msg = safety_checker.evaluate(message)
    if safety_status == SafetyStatus.URGENT_MEDICAL_EVALUATION and emergency_msg:
        # Emergency situation detected: route to urgent care without ordinary assessment
        # The LLM is NEVER allowed to downgrade urgency or fabricate a diagnosis
        return ChatResponse(
            session_id=request.session_id,
            assistant_message=emergency_msg,
            intent=IntentType.SYMPTOM_ASSESSMENT,
            structured_health_information=StructuredHealthInformation(),
            follow_up_question=None,
            safety_status=SafetyStatus.URGENT_MEDICAL_EVALUATION,
            next_action=NextActionType.EMERGENCY_ROUTING,
            assessment_handoff=None
        )

    # 2. Retrieve or create session state
    session = session_manager.get_or_create(request.session_id, request.user_id)
    has_prior_context = len(session.history) > 0

    patient_id = None
    if request.user_id and request.user_id not in ("anonymous", "guest", "default_user", ""):
        try:
            patient_id = int(request.user_id)
        except ValueError:
            digits = re.findall(r'\d+', request.user_id)
            patient_id = int(digits[0]) if digits else None

    # Apply client launch context if provided
    if request.context:
        session.set_context(request.context, patient_id)

    # 2a. Out-of-Scope General Knowledge (GK) redirect
    if is_out_of_scope_gk(message):
        session.add_message(role="user", content=message)
        gk_resp = (
            "I'm primarily designed for health-related questions, so general GK isn't my main area. "
            "If you have a health question, symptom, report, or wellness concern, I'd be happy to help."
        )
        session.add_message(role="assistant", content=gk_resp)
        return ChatResponse(
            session_id=request.session_id,
            assistant_message=gk_resp,
            intent=IntentType.GENERAL_HEALTH_QUESTION,
            structured_health_information=session.health_info,
            follow_up_question=None,
            safety_status=SafetyStatus.NORMAL_CONVERSATION,
            next_action=NextActionType.PROVIDE_GENERAL_INFO,
            assessment_handoff=None
        )

    # 2b. Natural Greeting and Casual Introduction (only for new session without prior clinical context)
    if not has_prior_context and not bool(session.health_info.symptoms):
        is_casual, casual_type = is_casual_greeting_or_intro(message)
        if is_casual:
            session.add_message(role="user", content=message)
            if casual_type == "greeting":
                casual_msg = (
                    "Hello! I'm your SmartHealth conversational assistant. How can I help you today? "
                    "Feel free to ask a health question, describe any symptoms you are experiencing, or enter laboratory reports."
                )
            elif casual_type and casual_type.startswith("intro:"):
                name = casual_type.split(":", 1)[1]
                casual_msg = (
                    f"Hello, {name}! Nice to meet you. I'm here to assist with health inquiries, symptom screening, "
                    "and laboratory risk assessments. How can I help you today?"
                )
            else:
                casual_msg = (
                    "Hello! I'm here to assist with health inquiries, symptom screening, "
                    "and laboratory risk assessments. How can I help you today?"
                )
            session.add_message(role="assistant", content=casual_msg)
            return ChatResponse(
                session_id=request.session_id,
                assistant_message=casual_msg,
                intent=IntentType.GENERAL_HEALTH_QUESTION,
                structured_health_information=session.health_info,
                follow_up_question=None,
                safety_status=SafetyStatus.NORMAL_CONVERSATION,
                next_action=NextActionType.PROVIDE_GENERAL_INFO,
                assessment_handoff=None
            )

    # 2c. Two-Section Conversational Summary or Missing Information Request
    if is_summary_request(message) or is_missing_information_request(message):
        session.add_message(role="user", content=message)
        from app.response_planner import response_planner
        plan = response_planner.build_response_plan(
            message=message,
            session=session,
            intent=IntentType.SYMPTOM_ASSESSMENT
        )
        session.last_response_plan = plan
        summary_msg = response_planner.generate_plan_response(plan, session, message)
        session.add_message(role="assistant", content=summary_msg)
        return ChatResponse(
            session_id=request.session_id,
            assistant_message=summary_msg,
            intent=IntentType.SYMPTOM_ASSESSMENT,
            structured_health_information=session.health_info,
            follow_up_question=None,
            safety_status=SafetyStatus.NORMAL_CONVERSATION,
            next_action=NextActionType.PROVIDE_GENERAL_INFO,
            assessment_handoff=None
        )

    # 2d. Apply explicit user corrections to session state
    session.apply_user_corrections(message)

    # 2e. Historical Heart Assessment, SHAP Explanation, Safety, or Cross-Source Guidance Query
    is_heart_hist_exp, heart_query_type = is_heart_history_or_explanation_query(message, session)
    if is_heart_hist_exp:
        logger.info("HEART_HISTORY_OR_EXPLANATION activated for session %s, type=%s", request.session_id, heart_query_type)
        session.last_routing = "HEART_HISTORY"
        session.add_message(role="user", content=message)
        session.last_intent = IntentType.MONITORING_QUERY

        assistant_message = handle_heart_history_or_explanation(
            message=message,
            session=session,
            request_user_id=request.user_id,
            heart_query_type=heart_query_type
        )

        session.add_message(role="assistant", content=assistant_message)
        return ChatResponse(
            session_id=request.session_id,
            assistant_message=assistant_message,
            intent=IntentType.MONITORING_QUERY,
            structured_health_information=session.health_info,
            follow_up_question=None,
            safety_status=SafetyStatus.NORMAL_CONVERSATION,
            next_action=NextActionType.PROVIDE_GENERAL_INFO,
            assessment_handoff=None
        )

    # 3. Deterministic Fast-Path Detection for structured lab inputs
    is_fast_path, fast_fields, fast_errors = fast_path_detector.detect(message)
    if is_fast_path:
        logger.info("FAST_PATH activated for session %s: %s", request.session_id, list(fast_fields.keys()))
        session.last_routing = "FAST_PATH"
        session.add_message(role="user", content=message)
        session.last_intent = IntentType.DISEASE_RISK_ASSESSMENT

        assistant_msg, next_act, follow_up_q, handoff = handle_liver_assessment(
            session=session,
            request_user_id=request.user_id,
            extracted_fields=fast_fields,
            validation_errors=fast_errors
        )
        session.add_message(role="assistant", content=assistant_msg)

        return ChatResponse(
            session_id=request.session_id,
            assistant_message=assistant_msg,
            intent=IntentType.DISEASE_RISK_ASSESSMENT,
            structured_health_information=session.health_info,
            follow_up_question=follow_up_q,
            safety_status=SafetyStatus.NORMAL_CONVERSATION,
            next_action=next_act,
            assessment_handoff=handoff
        )

    # 4. Returning-Patient Historical Health Context Detection (Phase 2B)
    is_context_query, detected_metric = is_patient_context_query(message, session=session)
    if is_context_query:
        logger.info("PATIENT_CONTEXT activated for session %s, metric=%s", request.session_id, detected_metric)
        session.last_routing = "PATIENT_CONTEXT"
        session.add_message(role="user", content=message)
        session.last_intent = IntentType.MONITORING_QUERY

        if not patient_id:
            assistant_message = (
                "Personal health trends and baseline comparisons require an authenticated patient profile. "
                "Please log in to your patient account to view your verified historical measurements."
            )
        else:
            backend_client = get_backend_client()
            try:
                context_data = backend_client.get_patient_health_context(patient_id, metric_type=detected_metric)
                assistant_message = explain_patient_context(
                    context_data=context_data,
                    user_message=message,
                    llm_client=extraction_engine.llm_client,
                    requested_metric=detected_metric
                )
            except (BackendServiceUnavailableError, BackendClientError) as e:
                logger.warning("Spring Boot context retrieval failed: %s", e)
                assistant_message = (
                    "I am currently unable to retrieve your historical health records from the health data service. "
                    "Your active conversation session is preserved. Please try again in a few moments, or check your profile directly."
                )

        session.add_message(role="assistant", content=assistant_message)
        return ChatResponse(
            session_id=request.session_id,
            assistant_message=assistant_message,
            intent=IntentType.MONITORING_QUERY,
            structured_health_information=session.health_info,
            follow_up_question=None,
            safety_status=SafetyStatus.NORMAL_CONVERSATION,
            next_action=NextActionType.PROVIDE_GENERAL_INFO,
            assessment_handoff=None
        )

        return ChatResponse(
            session_id=request.session_id,
            assistant_message=assistant_message,
            intent=IntentType.MONITORING_QUERY,
            structured_health_information=session.health_info,
            follow_up_question=None,
            safety_status=SafetyStatus.NORMAL_CONVERSATION,
            next_action=NextActionType.PROVIDE_GENERAL_INFO,
            assessment_handoff=None
        )

    # 4b. Answer-First Path with Grounded LLM Synthesis
    if is_question_about_causes_or_what_helps(message):
        session.last_routing = "ANSWER_FIRST"
        extracted_info = extraction_engine.extract_health_information(message)
        return clinical_knowledge_engine.generate_answer_first_response(
            session=session,
            message=message,
            request_user_id=request.user_id,
            extracted_info=extracted_info,
            follow_up_engine=follow_up_engine,
            llm_client=extraction_engine.llm_client
        )

    # 5. Standard LLM / Conversational Path
    session.last_routing = "LLM_PATH"
    llm_assistant_msg: Optional[str] = None
    llm_follow_up: Optional[str] = None
    try:
        res = extraction_engine.extract_with_llm(message, session)
        if isinstance(res, tuple) and len(res) == 4:
            llm_assistant_msg, intent, extracted_info, llm_follow_up = res
        elif isinstance(res, tuple) and len(res) == 3:
            llm_assistant_msg, intent, extracted_info = res
            llm_follow_up = None
        else:
            llm_assistant_msg = None
            intent = IntentType.UNKNOWN
            extracted_info = StructuredHealthInformation()
            llm_follow_up = None
    except LLMUnavailableError as e:
        logger.warning("LLM service unavailable for session %s: %s", request.session_id, e)
        # Preserve deterministic extraction from current user message
        extracted_info = extraction_engine.extract_health_information(message)
        # Update conversation state with newly extracted facts
        session.add_message(role="user", content=message)
        session.merge_health_info(extracted_info, user_message=message)
        session.last_intent = extraction_engine.classify_intent(message, has_prior_context=has_prior_context)
        # Build clinical grounding from updated verified facts
        grounding = clinical_knowledge_engine.build_grounding_context(
            session.health_info, message, session, follow_up_engine
        )
        # Generate useful deterministic fallback response
        assistant_message = (
            "The conversational AI service is temporarily unavailable for natural synthesis; "
            "providing verified clinical information based on your reported context.\n\n"
            + grounding.deterministic_fallback_message
        )
        session.add_message(role="assistant", content=assistant_message)
        follow_up = follow_up_engine.generate_follow_up(session)
        return ChatResponse(
            session_id=session.session_id,
            assistant_message=assistant_message,
            intent=session.last_intent,
            structured_health_information=session.health_info,
            current_turn_extraction=extracted_info,
            follow_up_question=follow_up,
            safety_status=SafetyStatus.NORMAL_CONVERSATION,
            next_action=NextActionType.ASK_FOLLOW_UP if follow_up else NextActionType.PROVIDE_GENERAL_INFO,
            assessment_handoff=None
        )
    except LLMResponseError as e:
        logger.warning("LLM response error for session %s: %s. Falling back to deterministic extraction.", request.session_id, e)
        # Controlled fallback on invalid structured LLM output without crashing
        intent = extraction_engine.classify_intent(message, has_prior_context=has_prior_context)
        extracted_info = extraction_engine.extract_health_information(message)
        llm_assistant_msg = None
        llm_follow_up = None

    return formulate_chat_turn(
        session=session,
        message=message,
        request_user_id=request.user_id,
        intent=intent,
        extracted_info=extracted_info,
        llm_assistant_msg=llm_assistant_msg,
        llm_follow_up=llm_follow_up,
        safety_status=SafetyStatus.NORMAL_CONVERSATION
    )


def formulate_chat_turn(
    session: Any,
    message: str,
    request_user_id: str,
    intent: IntentType,
    extracted_info: StructuredHealthInformation,
    llm_assistant_msg: Optional[str],
    llm_follow_up: Optional[str],
    safety_status: SafetyStatus
) -> ChatResponse:
    """
    Central response orchestration function.
    Merges session state, evaluates liver assessment triggers, symptom screening,
    and returns a standardized ChatResponse.
    Shared between non-streaming /api/chat and streaming /api/chat/stream.
    """
    # 1. Update session history and merge health state
    session.add_message(role="user", content=message)
    session.merge_health_info(extracted_info, user_message=message)
    session.last_intent = intent

    # Deterministic Response Planner
    from app.response_planner import response_planner
    plan = response_planner.build_response_plan(
        message=message,
        session=session,
        extracted_info=extracted_info,
        grounding=None,
        intent=intent
    )
    session.last_response_plan = plan

    # 2. Response Orchestration & Decision Flow
    assistant_message = ""
    follow_up_question = None
    next_action = NextActionType.PROVIDE_GENERAL_INFO
    assessment_handoff = None

    text_lower = message.lower()

    # Detect if message contains concrete extracted liver inputs or validation errors
    extracted_fields, validation_errors = liver_extractor.extract_and_validate(
        message,
        measurements=extracted_info.measurements,
        age=extracted_info.age,
        sex=extracted_info.sex
    )
    # Distinguish actual laboratory metrics from demographic or symptom inputs
    has_actual_lab_inputs = any(k in extracted_fields for k in LIVER_LAB_UNITS)
    has_lab_validation_errors = any(
        any(lab in err.lower() for lab in ("bilirubin", "phosphatase", "sgpt", "sgot", "protein", "albumin", "ratio"))
        for err in validation_errors
    )
    is_explicit_liver_report = bool(re.search(
        r"\b(i\s+have|enter|submit|my)\s+(my\s+)?(liver\s+)?(blood\s+|lab\s+|test\s+)?(test|report|results?|panel|values?)\b"
        r"|\b(check|test|evaluate|assess)\s+(my\s+)?(liver\s+)?(health|risk|report|test)\b"
        r"|\b(liver\s+report|liver\s+test|liver\s+panel)\b",
        text_lower
    ))
    is_liver_intent = (
        intent == IntentType.DISEASE_RISK_ASSESSMENT
        and any(k in text_lower for k in ("liver", "hepatic"))
    )
    is_liver_in_progress = (
        session.liver_assessment is not None
        and session.liver_assessment.status == "IN_PROGRESS"
    )
    is_liver_followup = (
        session.liver_assessment is not None
        and intent == IntentType.FOLLOW_UP_RESPONSE
        and (has_actual_lab_inputs or any(k in text_lower for k in ("yes", "run", "check", "predict", "proceed")))
    )

    should_handle_liver = (
        is_liver_in_progress
        or is_liver_intent
        or is_explicit_liver_report
        or has_actual_lab_inputs
        or has_lab_validation_errors
        or is_liver_followup
    )

    # Detect if message contains concrete extracted heart inputs or validation errors
    heart_extracted_fields, heart_explicit_nulls, heart_validation_errors = heart_extractor.extract_and_validate(
        message,
        age=extracted_info.age,
        sex=extracted_info.sex
    )
    has_heart_clinical_inputs = any(k in heart_extracted_fields for k in (
        "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
    )) or len(heart_explicit_nulls) > 0
    has_heart_validation_errors = len(heart_validation_errors) > 0
    is_explicit_heart_request = bool(re.search(
        r"\b(assess|check|test|screen|calculate|evaluate)\s+(my\s+)?(heart|cardiac|cardiovascular)\s*(disease|risk|health|assessment)?\b"
        r"|\b(heart|cardiac)\s*(disease\s+)?(risk\s+)?(assessment|screening|test|check)\b",
        text_lower
    ))
    is_heart_intent = (
        intent == IntentType.DISEASE_RISK_ASSESSMENT
        and any(k in text_lower for k in ("heart", "cardiac", "cardiovascular"))
    )
    is_heart_in_progress = (
        session.heart_assessment is not None
        and session.heart_assessment.status in ("IN_PROGRESS", "AWAITING_CONFIRMATION", "VALIDATION_ERROR")
    )
    is_heart_followup = (
        session.heart_assessment is not None
        and session.heart_assessment.status in ("IN_PROGRESS", "AWAITING_CONFIRMATION", "VALIDATION_ERROR")
        and (
            intent == IntentType.FOLLOW_UP_RESPONSE
            or has_heart_clinical_inputs
            or "age" in heart_extracted_fields
            or "sex" in heart_extracted_fields
            or any(k in text_lower for k in ("yes", "confirm", "proceed", "run", "sure", "cancel", "no", "stop"))
        )
    )

    is_general_edu = bool(re.search(
        r"^(what\s+is|what\s+are|define|explain|tell\s+me\s+about|what\s+causes)\s+(heart\s+disease|chest\s+pain|angina|heart\s+attack|coronary\s+artery)\b",
        text_lower
    )) and not any(k in text_lower for k in ("my", "i", "mine", "score", "result", "assessment", "why did", "why was", "previous", "last"))

    has_heart_clinical_validation_errors = any(
        any(k in err.lower() for k in ("chest pain", "blood pressure", "cholesterol", "sugar", "ecg", "heart rate", "angina", "st depression", "slope", "vessel", "thal"))
        for err in heart_validation_errors
    ) or (is_heart_in_progress and len(heart_validation_errors) > 0)

    should_handle_heart = (
        not is_general_edu
        and not (should_handle_liver and not (is_heart_in_progress or is_heart_intent or is_explicit_heart_request or has_heart_clinical_inputs))
        and (
            is_heart_in_progress
            or is_heart_intent
            or is_explicit_heart_request
            or has_heart_clinical_inputs
            or has_heart_clinical_validation_errors
            or is_heart_followup
        )
    )

    is_user_asking_causes_or_help = is_question_about_causes_or_what_helps(message)
    if is_user_asking_causes_or_help:
        intent = IntentType.SYMPTOM_ASSESSMENT

    if should_handle_heart:
        assistant_message, next_action, follow_up_question, assessment_handoff = handle_heart_assessment(
            session=session,
            request_user_id=request_user_id,
            extracted_fields=heart_extracted_fields,
            explicit_nulls=heart_explicit_nulls,
            validation_errors=heart_validation_errors,
            user_message=message
        )

    # General health questions take precedence if user is not supplying liver lab values or heart inputs
    elif intent == IntentType.GENERAL_HEALTH_QUESTION and not has_actual_lab_inputs and not has_heart_clinical_inputs and not is_user_asking_causes_or_help:
        next_action = NextActionType.PROVIDE_GENERAL_INFO
        if llm_assistant_msg and len(llm_assistant_msg.strip()) > 10:
            assistant_message = llm_assistant_msg
        else:
            # Educational health responses (cautious, no diagnosis, no treatment)
            if "blood pressure" in text_lower:
                assistant_message = (
                    "Blood pressure is the force of circulating blood against the walls of the body's arteries. "
                    "It is measured in millimeters of mercury (mmHg) and recorded as two numbers: systolic pressure "
                    "(pressure when the heart beats) over diastolic pressure (pressure when the heart rests between beats). "
                    "A normal reading for most adults is typically below 120/80 mmHg."
                )
            elif "cholesterol" in text_lower:
                assistant_message = (
                    "Cholesterol is a waxy, fat-like substance produced by the liver and found in certain foods. "
                    "While the body needs cholesterol to build cells, elevated low-density lipoprotein (LDL) cholesterol "
                    "can accumulate in blood vessels, increasing the risk of cardiovascular complications."
                )
            elif "bilirubin" in text_lower:
                assistant_message = (
                    "Bilirubin is a yellowish substance produced during the normal breakdown of red blood cells. "
                    "It travels through the liver and is eventually excreted. Elevated levels can sometimes indicate "
                    "issues with liver function, bile ducts, or red blood cell breakdown, but an accurate evaluation "
                    "requires comprehensive laboratory testing and consultation with a physician."
                )
            else:
                assistant_message = (
                    "I have processed your general health inquiry. While I can provide general wellness and health education, "
                    "please consult a qualified healthcare provider for specific clinical recommendations."
                )

    elif should_handle_liver:
        assistant_message, next_action, follow_up_question, assessment_handoff = handle_liver_assessment(
            session=session,
            request_user_id=request_user_id,
            extracted_fields=extracted_fields,
            validation_errors=validation_errors
        )

    elif intent in (IntentType.SYMPTOM_ASSESSMENT, IntentType.FOLLOW_UP_RESPONSE) or is_user_asking_causes_or_help:
        follow_up = follow_up_engine.generate_follow_up(session)

        # Identify candidate assessment domains based on collected symptoms
        disease_candidates = []
        info = session.health_info
        loc = (info.location or "").lower()
        sym_names = [s.name.lower() for s in info.symptoms]

        if ("right" in loc and ("rib" in loc or "abdomen" in loc)) or "jaundice" in sym_names or "dark urine" in sym_names:
            disease_candidates.append("liver_disease")
        if "chest" in loc or "shortness of breath" in sym_names:
            disease_candidates.append("heart_disease")
        if "tremor" in sym_names or "stiffness" in sym_names:
            disease_candidates.append("parkinsons_disease")

        if not disease_candidates:
            disease_candidates.append("general_risk_evaluation")

        structured_inputs = {
            "symptoms": [s.model_dump() for s in info.symptoms],
            "location": info.location,
            "duration": info.duration,
            "severity": info.severity,
            "onset": info.onset,
            "frequency": info.frequency,
            "known_conditions": info.known_conditions,
            "medications": info.medications,
            "measurements": [m.model_dump() for m in info.measurements],
            "negative_symptoms": info.negative_symptoms,
            "triggers": info.triggers,
            "relieving_factors": info.relieving_factors,
            "progression": info.progression,
            "previous_tests": info.previous_tests,
            "radiation": getattr(info, "radiation", None)
        }

        if is_user_asking_causes_or_help:
            return clinical_knowledge_engine.generate_answer_first_response(
                session=session,
                message=message,
                request_user_id=request_user_id,
                extracted_info=extracted_info,
                follow_up_engine=follow_up_engine,
                llm_client=extraction_engine.llm_client
            )

        elif llm_assistant_msg and len(llm_assistant_msg.strip()) > 10:
            assistant_message = llm_assistant_msg
            if llm_follow_up and len(llm_follow_up.strip()) > 3 and not follow_up_engine.is_question_repeating_answered_topic(llm_follow_up, session):
                follow_up_question = llm_follow_up.strip()
                next_action = NextActionType.ASK_FOLLOW_UP
            elif follow_up:
                follow_up_question = follow_up
                next_action = NextActionType.ASK_FOLLOW_UP
            else:
                follow_up_question = None
                next_action = NextActionType.REQUEST_DISEASE_ASSESSMENT

            # Formulate handoff contract when symptom collection is sufficiently complete
            if not follow_up:
                assessment_handoff = AssessmentHandoffContract(
                    next_action=NextActionType.REQUEST_DISEASE_ASSESSMENT,
                    disease_candidates=disease_candidates,
                    structured_inputs=structured_inputs
                )
        else:
            # Deterministic fallback when LLM conversational response is not available (e.g. MockLLMClient unit tests)
            if follow_up and plan.answer_mode not in (AnswerMode.DIAGNOSIS_UNCERTAINTY, AnswerMode.NEXT_STEPS):
                follow_up_question = follow_up
                next_action = NextActionType.ASK_FOLLOW_UP
                assistant_message = (
                    f"Thank you for sharing that information. To help understand your symptoms more clearly: {follow_up}"
                )
            elif plan.answer_mode in (AnswerMode.NEXT_STEPS, AnswerMode.DIAGNOSIS_UNCERTAINTY, AnswerMode.EVALUATION_GUIDANCE):
                assistant_message = response_planner.generate_plan_response(plan, session, message)
                next_action = NextActionType.PROVIDE_GENERAL_INFO
            else:
                # Sufficient symptom context collected -> formulate assessment handoff
                next_action = NextActionType.REQUEST_DISEASE_ASSESSMENT
                assessment_handoff = AssessmentHandoffContract(
                    next_action=NextActionType.REQUEST_DISEASE_ASSESSMENT,
                    disease_candidates=disease_candidates,
                    structured_inputs=structured_inputs
                )
                assistant_message = generate_symptom_screening_summary(session)

    elif intent == IntentType.DISEASE_RISK_ASSESSMENT:
        next_action = NextActionType.REQUEST_DISEASE_ASSESSMENT
        text_lower = message.lower()
        target_domain = "general"
        if "diabetes" in text_lower:
            target_domain = "diabetes"
        elif "liver" in text_lower:
            target_domain = "liver_disease"
        elif "heart" in text_lower:
            target_domain = "heart_disease"
        elif "parkinson" in text_lower:
            target_domain = "parkinsons_disease"

        assessment_handoff = AssessmentHandoffContract(
            next_action=NextActionType.REQUEST_DISEASE_ASSESSMENT,
            disease_candidates=[target_domain],
            structured_inputs={
                "requested_domain": target_domain,
                "known_conditions": session.health_info.known_conditions,
                "symptoms": [s.model_dump() for s in session.health_info.symptoms]
            }
        )

        assistant_message = (
            f"Understood. You would like to evaluate your {target_domain.replace('_', ' ').title()} risk. "
            "Our disease assessment engine requires specific clinical parameters and laboratory measurements "
            "to perform an evidence-based risk calculation. "
            "No disease probability has been calculated yet, as laboratory values have not been supplied."
        )

    elif intent == IntentType.MEDICAL_HISTORY_UPDATE:
        next_action = NextActionType.PROVIDE_GENERAL_INFO
        assistant_message = (
            "Thank you. Your medical history and medication information have been updated in your active session context."
        )

    elif intent == IntentType.MONITORING_QUERY:
        next_action = NextActionType.PROVIDE_GENERAL_INFO
        patient_id = None
        try:
            patient_id = int(request_user_id)
        except ValueError:
            digits = re.findall(r'\d+', request_user_id)
            patient_id = int(digits[0]) if digits else 1

        backend_client = get_backend_client()
        try:
            context_data = backend_client.get_patient_health_context(patient_id)
            assistant_message = explain_patient_context(
                context_data=context_data,
                user_message=message,
                llm_client=extraction_engine.llm_client
            )
        except (BackendServiceUnavailableError, BackendClientError) as e:
            logger.warning("Spring Boot context retrieval failed in LLM path: %s", e)
            assistant_message = (
                "I am currently unable to retrieve your historical health records from the health data service. "
                "Your active conversation session is preserved. Please try again in a few moments, or check your profile directly."
            )

    else:
        # Unknown intent
        next_action = NextActionType.CLARIFY
        assistant_message = (
            "I am a conversational health assistant designed to assist with structured symptom collection, "
            "health education, and risk assessment preparation. "
            "Could you please clarify your health question or symptom in a few more words?"
        )

    # Validate assistant message before persisting or returning
    from app.response_validator import response_validator
    if assistant_message:
        assistant_message = response_validator.validate_and_repair(
            generated_response=assistant_message,
            current_turn_extraction=extracted_info,
            cumulative_session=session,
            response_plan=getattr(session, "last_response_plan", None),
            grounding=None
        )

    # 3. Record assistant turn in session history
    session.add_message(role="assistant", content=assistant_message)

    return ChatResponse(
        session_id=session.session_id,
        assistant_message=assistant_message,
        intent=intent,
        structured_health_information=session.health_info,
        current_turn_extraction=extracted_info,
        follow_up_question=follow_up_question,
        safety_status=safety_status,
        next_action=next_action,
        assessment_handoff=assessment_handoff,
        progression_context=session.last_progression_context,
        response_plan=getattr(session, "last_response_plan", None)
    )


@app.post("/api/chat/stream", tags=["Chat"])
def stream_chat(request: ChatRequest) -> StreamingResponse:
    """
    Streaming conversational endpoint using Server-Sent Events (SSE).
    Progressively streams assistant message tokens in real-time while hiding internal JSON scaffolding.
    Upon stream completion, delivers the full ChatResponse containing structured health information,
    intent classification, safety status, and handoff contracts.
    """
    message = request.message.strip()
    if not message:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message content cannot be empty."
        )

    # 1. Conservative Safety Red-Flag Check (Deterministic - runs BEFORE any LLM invocation)
    safety_status, triggers, emergency_msg = safety_checker.evaluate(message)
    if safety_status == SafetyStatus.URGENT_MEDICAL_EVALUATION and emergency_msg:
        def emergency_stream():
            yield f"event: token\ndata: {json.dumps({'text': emergency_msg})}\n\n"
            complete_data = ChatResponse(
                session_id=request.session_id,
                assistant_message=emergency_msg,
                intent=IntentType.SYMPTOM_ASSESSMENT,
                structured_health_information=StructuredHealthInformation(),
                follow_up_question=None,
                safety_status=SafetyStatus.URGENT_MEDICAL_EVALUATION,
                next_action=NextActionType.EMERGENCY_ROUTING,
                assessment_handoff=None
            ).model_dump()
            yield f"event: complete\ndata: {json.dumps(complete_data)}\n\n"

        return StreamingResponse(
            emergency_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
        )

    # 2. Retrieve or create session state
    session = session_manager.get_or_create(request.session_id, request.user_id)
    has_prior_context = len(session.history) > 0

    patient_id = None
    if request.user_id and request.user_id not in ("anonymous", "guest", "default_user", ""):
        try:
            patient_id = int(request.user_id)
        except ValueError:
            digits = re.findall(r'\d+', request.user_id)
            patient_id = int(digits[0]) if digits else None

    # Apply client launch context if provided
    if request.context:
        session.set_context(request.context, patient_id)

    # 2a. Out-of-Scope General Knowledge (GK) redirect
    if is_out_of_scope_gk(message):
        session.add_message(role="user", content=message)
        gk_resp = (
            "I'm primarily designed for health-related questions, so general GK isn't my main area. "
            "If you have a health question, symptom, report, or wellness concern, I'd be happy to help."
        )
        session.add_message(role="assistant", content=gk_resp)

        def gk_stream():
            yield f"event: token\ndata: {json.dumps({'text': gk_resp})}\n\n"
            complete_data = ChatResponse(
                session_id=request.session_id,
                assistant_message=gk_resp,
                intent=IntentType.GENERAL_HEALTH_QUESTION,
                structured_health_information=session.health_info,
                follow_up_question=None,
                safety_status=SafetyStatus.NORMAL_CONVERSATION,
                next_action=NextActionType.PROVIDE_GENERAL_INFO,
                assessment_handoff=None
            ).model_dump()
            yield f"event: complete\ndata: {json.dumps(complete_data)}\n\n"

        return StreamingResponse(
            gk_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
        )

    # 2b. Natural Greeting and Casual Introduction (only for new session without prior clinical context)
    if not has_prior_context and not bool(session.health_info.symptoms):
        is_casual, casual_type = is_casual_greeting_or_intro(message)
        if is_casual:
            session.add_message(role="user", content=message)
            if casual_type == "greeting":
                casual_msg = (
                    "Hello! I'm your SmartHealth conversational assistant. How can I help you today? "
                    "Feel free to ask a health question, describe any symptoms you are experiencing, or enter laboratory reports."
                )
            elif casual_type and casual_type.startswith("intro:"):
                name = casual_type.split(":", 1)[1]
                casual_msg = (
                    f"Hello, {name}! Nice to meet you. I'm here to assist with health inquiries, symptom screening, "
                    "and laboratory risk assessments. How can I help you today?"
                )
            else:
                casual_msg = (
                    "Hello! I'm here to assist with health inquiries, symptom screening, "
                    "and laboratory risk assessments. How can I help you today?"
                )
            session.add_message(role="assistant", content=casual_msg)

            def casual_stream():
                yield f"event: token\ndata: {json.dumps({'text': casual_msg})}\n\n"
                complete_data = ChatResponse(
                    session_id=request.session_id,
                    assistant_message=casual_msg,
                    intent=IntentType.GENERAL_HEALTH_QUESTION,
                    structured_health_information=session.health_info,
                    follow_up_question=None,
                    safety_status=SafetyStatus.NORMAL_CONVERSATION,
                    next_action=NextActionType.PROVIDE_GENERAL_INFO,
                    assessment_handoff=None
                ).model_dump()
                yield f"event: complete\ndata: {json.dumps(complete_data)}\n\n"

            return StreamingResponse(
                casual_stream(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
            )

    # 2c. Two-Section Conversational Summary or Missing Information Request
    if is_summary_request(message) or is_missing_information_request(message):
        session.add_message(role="user", content=message)
        from app.response_planner import response_planner
        plan = response_planner.build_response_plan(
            message=message,
            session=session,
            intent=IntentType.SYMPTOM_ASSESSMENT
        )
        session.last_response_plan = plan
        summary_msg = response_planner.generate_plan_response(plan, session, message)
        session.add_message(role="assistant", content=summary_msg)

        def summary_stream():
            yield f"event: token\ndata: {json.dumps({'text': summary_msg})}\n\n"
            complete_data = ChatResponse(
                session_id=request.session_id,
                assistant_message=summary_msg,
                intent=IntentType.SYMPTOM_ASSESSMENT,
                structured_health_information=session.health_info,
                follow_up_question=None,
                safety_status=SafetyStatus.NORMAL_CONVERSATION,
                next_action=NextActionType.PROVIDE_GENERAL_INFO,
                assessment_handoff=None
            ).model_dump()
            yield f"event: complete\ndata: {json.dumps(complete_data)}\n\n"

        return StreamingResponse(
            summary_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
        )

    # 2d. Apply explicit user corrections to session state
    session.apply_user_corrections(message)

    # 2e. Historical Heart Assessment, SHAP Explanation, Safety, or Cross-Source Guidance Query
    is_heart_hist_exp, heart_query_type = is_heart_history_or_explanation_query(message, session)
    if is_heart_hist_exp:
        session.last_routing = "HEART_HISTORY"
        session.add_message(role="user", content=message)
        session.last_intent = IntentType.MONITORING_QUERY

        assistant_message = handle_heart_history_or_explanation(
            message=message,
            session=session,
            request_user_id=request.user_id,
            heart_query_type=heart_query_type
        )

        session.add_message(role="assistant", content=assistant_message)

        def heart_hist_stream():
            yield f"event: token\ndata: {json.dumps({'text': assistant_message})}\n\n"
            complete_data = ChatResponse(
                session_id=request.session_id,
                assistant_message=assistant_message,
                intent=IntentType.MONITORING_QUERY,
                structured_health_information=session.health_info,
                follow_up_question=None,
                safety_status=SafetyStatus.NORMAL_CONVERSATION,
                next_action=NextActionType.PROVIDE_GENERAL_INFO,
                assessment_handoff=None
            ).model_dump()
            yield f"event: complete\ndata: {json.dumps(complete_data)}\n\n"

        return StreamingResponse(
            heart_hist_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
        )

    # 3. Deterministic Fast-Path Detection for structured lab inputs
    is_fast_path, fast_fields, fast_errors = fast_path_detector.detect(message)
    if is_fast_path:
        session.last_routing = "FAST_PATH"
        session.add_message(role="user", content=message)
        session.last_intent = IntentType.DISEASE_RISK_ASSESSMENT

        assistant_msg, next_act, follow_up_q, handoff = handle_liver_assessment(
            session=session,
            request_user_id=request.user_id,
            extracted_fields=fast_fields,
            validation_errors=fast_errors
        )
        session.add_message(role="assistant", content=assistant_msg)

        def fast_path_stream():
            yield f"event: token\ndata: {json.dumps({'text': assistant_msg})}\n\n"
            complete_data = ChatResponse(
                session_id=request.session_id,
                assistant_message=assistant_msg,
                intent=IntentType.DISEASE_RISK_ASSESSMENT,
                structured_health_information=session.health_info,
                follow_up_question=follow_up_q,
                safety_status=SafetyStatus.NORMAL_CONVERSATION,
                next_action=next_act,
                assessment_handoff=handoff
            ).model_dump()
            yield f"event: complete\ndata: {json.dumps(complete_data)}\n\n"

        return StreamingResponse(
            fast_path_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
        )

    # 4. Returning-Patient Historical Health Context Detection
    is_context_query, detected_metric = is_patient_context_query(message, session=session)
    if is_context_query:
        session.last_routing = "PATIENT_CONTEXT"
        session.add_message(role="user", content=message)
        session.last_intent = IntentType.MONITORING_QUERY

        if not patient_id:
            assistant_message = (
                "Personal health trends and baseline comparisons require an authenticated patient profile. "
                "Please log in to your patient account to view your verified historical measurements."
            )
        else:
            backend_client = get_backend_client()
            try:
                context_data = backend_client.get_patient_health_context(patient_id, metric_type=detected_metric)
                assistant_message = explain_patient_context(
                    context_data=context_data,
                    user_message=message,
                    llm_client=extraction_engine.llm_client,
                    requested_metric=detected_metric
                )
            except (BackendServiceUnavailableError, BackendClientError) as e:
                logger.warning("Spring Boot context retrieval failed: %s", e)
                assistant_message = (
                    "I am currently unable to retrieve your historical health records from the health data service. "
                    "Your active conversation session is preserved. Please try again in a few moments, or check your profile directly."
                )

        session.add_message(role="assistant", content=assistant_message)

        def context_stream():
            yield f"event: token\ndata: {json.dumps({'text': assistant_message})}\n\n"
            complete_data = ChatResponse(
                session_id=request.session_id,
                assistant_message=assistant_message,
                intent=IntentType.MONITORING_QUERY,
                structured_health_information=session.health_info,
                follow_up_question=None,
                safety_status=SafetyStatus.NORMAL_CONVERSATION,
                next_action=NextActionType.PROVIDE_GENERAL_INFO,
                assessment_handoff=None
            ).model_dump()
            yield f"event: complete\ndata: {json.dumps(complete_data)}\n\n"

        return StreamingResponse(
            context_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
        )

    # 4b. Answer-First Streaming with Grounded LLM Synthesis
    if is_question_about_causes_or_what_helps(message):
        session.last_routing = "ANSWER_FIRST"
        extracted_info = extraction_engine.extract_health_information(message)
        previous_info = session.health_info.model_copy(deep=True)
        from app.conversation import compute_new_information, build_conversation_progress_context
        new_info = compute_new_information(previous_info, extracted_info)
        progress_context = build_conversation_progress_context(
            message=message,
            previous_info=previous_info,
            new_info=new_info,
            session=session,
            intent="CAUSES_OR_WHAT_HELPS"
        )
        session.last_progression_context = progress_context

        # Deterministic Response Planner
        from app.response_planner import response_planner
        plan = response_planner.build_response_plan(
            message=message,
            session=session,
            extracted_info=extracted_info,
            grounding=None,
            intent=IntentType.SYMPTOM_ASSESSMENT
        )
        session.last_response_plan = plan
        if progress_context is not None:
            progress_context.response_plan = plan

        session.add_message(role="user", content=message)
        session.merge_health_info(extracted_info, user_message=message)
        session.last_intent = IntentType.SYMPTOM_ASSESSMENT

        info = session.health_info
        grounding = clinical_knowledge_engine.build_grounding_context(
            info, message, session, follow_up_engine, progress_context=progress_context, response_plan=plan
        )
        follow_up = follow_up_engine.generate_follow_up(session)

        def answer_first_stream():
            streamed_text = ""
            for chunk in grounded_synthesizer.stream_synthesize_response(
                session=session,
                message=message,
                grounding=grounding,
                follow_up=follow_up,
                llm_client=extraction_engine.llm_client,
                progress_context=progress_context,
                response_plan=plan
            ):
                streamed_text += chunk
                yield f"event: token\ndata: {json.dumps({'text': chunk})}\n\n"

            if not streamed_text.strip():
                streamed_text = grounding.deterministic_fallback_message

            from app.response_validator import response_validator
            validated_text = response_validator.validate_and_repair(
                generated_response=streamed_text,
                current_turn_extraction=extracted_info,
                cumulative_session=session,
                response_plan=plan,
                grounding=grounding
            )

            session.add_message(role="assistant", content=validated_text)

            structured_inputs = {
                "symptoms": [s.model_dump() for s in info.symptoms],
                "location": info.location,
                "duration": info.duration,
                "severity": info.severity,
                "onset": info.onset,
                "frequency": info.frequency,
                "known_conditions": info.known_conditions,
                "medications": info.medications,
                "measurements": [m.model_dump() for m in info.measurements],
                "negative_symptoms": info.negative_symptoms,
                "triggers": info.triggers,
                "relieving_factors": info.relieving_factors,
                "progression": info.progression,
                "previous_tests": info.previous_tests,
                "radiation": getattr(info, "radiation", None)
            }

            if follow_up:
                follow_up_q = follow_up
                next_act = NextActionType.ASK_FOLLOW_UP
                handoff = None
            else:
                follow_up_q = None
                next_act = NextActionType.REQUEST_DISEASE_ASSESSMENT
                handoff = AssessmentHandoffContract(
                    next_action=NextActionType.REQUEST_DISEASE_ASSESSMENT,
                    disease_candidates=list(grounding.disease_candidates),
                    structured_inputs=structured_inputs
                )

            complete_resp = ChatResponse(
                session_id=session.session_id,
                assistant_message=validated_text,
                intent=IntentType.SYMPTOM_ASSESSMENT,
                structured_health_information=session.health_info,
                current_turn_extraction=extracted_info,
                follow_up_question=follow_up_q,
                safety_status=SafetyStatus.NORMAL_CONVERSATION,
                next_action=next_act,
                assessment_handoff=handoff,
                progression_context=progress_context,
                response_plan=plan
            )
            yield f"event: complete\ndata: {json.dumps(complete_resp.model_dump())}\n\n"

        return StreamingResponse(
            answer_first_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
        )

    # 5. Standard LLM Streaming Generation
    def stream_generator():
        session.last_routing = "LLM_STREAM"
        streamed_assistant_msg = ""
        final_extraction = None

        try:
            for event_type, data in extraction_engine.stream_extract_with_llm(message, session):
                if event_type == "token":
                    streamed_assistant_msg += data
                    yield f"event: token\ndata: {json.dumps({'text': data})}\n\n"
                elif event_type == "complete":
                    final_extraction = data
        except LLMUnavailableError as e:
            logger.warning("LLM service unavailable during stream for session %s: %s", request.session_id, e)
            extracted_info = extraction_engine.extract_health_information(message)
            session.add_message(role="user", content=message)
            session.merge_health_info(extracted_info, user_message=message)
            session.last_intent = extraction_engine.classify_intent(message, has_prior_context=has_prior_context)
            grounding = clinical_knowledge_engine.build_grounding_context(
                session.health_info, message, session, follow_up_engine
            )
            fallback_msg = (
                "The conversational AI service is temporarily unavailable for natural synthesis; "
                "providing verified clinical information based on your reported context.\n\n"
                + grounding.deterministic_fallback_message
            )
            for chunk in grounded_synthesizer._chunk_text(fallback_msg):
                yield f"event: token\ndata: {json.dumps({'text': chunk})}\n\n"
            session.add_message(role="assistant", content=fallback_msg)
            follow_up = follow_up_engine.generate_follow_up(session)
            complete_data = ChatResponse(
                session_id=session.session_id,
                assistant_message=fallback_msg,
                intent=session.last_intent,
                structured_health_information=session.health_info,
                current_turn_extraction=extracted_info,
                follow_up_question=follow_up,
                safety_status=SafetyStatus.NORMAL_CONVERSATION,
                next_action=NextActionType.ASK_FOLLOW_UP if follow_up else NextActionType.PROVIDE_GENERAL_INFO,
                assessment_handoff=None
            ).model_dump()
            yield f"event: complete\ndata: {json.dumps(complete_data)}\n\n"
            return
        except Exception as e:
            logger.error("Error during streaming generation for session %s: %s", request.session_id, e)
            yield f"event: error\ndata: {json.dumps({'detail': 'Streaming generation was interrupted.'})}\n\n"
            return

        if final_extraction:
            llm_assistant_msg, intent, extracted_info, llm_follow_up = final_extraction
        else:
            intent = extraction_engine.classify_intent(message, has_prior_context=has_prior_context)
            extracted_info = extraction_engine.extract_health_information(message)
            llm_assistant_msg = streamed_assistant_msg or None
            llm_follow_up = None

        response = formulate_chat_turn(
            session=session,
            message=message,
            request_user_id=request.user_id,
            intent=intent,
            extracted_info=extracted_info,
            llm_assistant_msg=llm_assistant_msg,
            llm_follow_up=llm_follow_up,
            safety_status=SafetyStatus.NORMAL_CONVERSATION
        )

        if not streamed_assistant_msg and response.assistant_message:
            yield f"event: token\ndata: {json.dumps({'text': response.assistant_message})}\n\n"

        yield f"event: complete\ndata: {json.dumps(response.model_dump())}\n\n"

    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
    )
