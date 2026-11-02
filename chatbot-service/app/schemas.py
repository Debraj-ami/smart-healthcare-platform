from enum import Enum
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field


class IntentType(str, Enum):
    """Controlled set of conversational intents for the healthcare platform."""
    GENERAL_HEALTH_QUESTION = "GENERAL_HEALTH_QUESTION"
    SYMPTOM_ASSESSMENT = "SYMPTOM_ASSESSMENT"
    DISEASE_RISK_ASSESSMENT = "DISEASE_RISK_ASSESSMENT"
    FOLLOW_UP_RESPONSE = "FOLLOW_UP_RESPONSE"
    MEDICAL_HISTORY_UPDATE = "MEDICAL_HISTORY_UPDATE"
    MONITORING_QUERY = "MONITORING_QUERY"
    UNKNOWN = "UNKNOWN"


class SafetyStatus(str, Enum):
    """Urgency level determined by safety routing."""
    NORMAL_CONVERSATION = "NORMAL_CONVERSATION"
    URGENT_MEDICAL_EVALUATION = "URGENT_MEDICAL_EVALUATION"


class NextActionType(str, Enum):
    """Workflow action indicated by conversation orchestrator."""
    ASK_FOLLOW_UP = "ASK_FOLLOW_UP"
    REQUEST_DISEASE_ASSESSMENT = "REQUEST_DISEASE_ASSESSMENT"
    PROVIDE_GENERAL_INFO = "PROVIDE_GENERAL_INFO"
    EMERGENCY_ROUTING = "EMERGENCY_ROUTING"
    CLARIFY = "CLARIFY"


class UserGoal(str, Enum):
    """Categorized conversational intent or objective of the user's turn."""
    GENERAL_INFORMATION = "GENERAL_INFORMATION"
    EXPLANATION_OF_CURRENT_SYMPTOMS = "EXPLANATION_OF_CURRENT_SYMPTOMS"
    EVALUATION_INFORMATION = "EVALUATION_INFORMATION"
    SUMMARY_OF_CONVERSATION = "SUMMARY_OF_CONVERSATION"
    SYMPTOM_SUMMARY = "SYMPTOM_SUMMARY"
    MISSING_INFORMATION = "MISSING_INFORMATION"
    NEXT_STEPS = "NEXT_STEPS"
    DIAGNOSIS_QUESTION = "DIAGNOSIS_QUESTION"
    FOLLOW_UP_TO_PREVIOUS_RESPONSE = "FOLLOW_UP_TO_PREVIOUS_RESPONSE"
    CORRECTION_OR_UPDATE = "CORRECTION_OR_UPDATE"
    PERSONAL_CONTEXT = "PERSONAL_CONTEXT"
    MONITORING = "MONITORING"
    BASELINE = "BASELINE"
    TREND = "TREND"
    ALERTS = "ALERTS"
    PATIENT_HISTORY = "PATIENT_HISTORY"
    CLARIFICATION = "CLARIFICATION"
    UNKNOWN = "UNKNOWN"


class AnswerMode(str, Enum):
    """Architectural mode guiding synthesis prioritization and structure."""
    DIRECT_ANSWER = "DIRECT_ANSWER"
    NEW_INFORMATION_EXPLANATION = "NEW_INFORMATION_EXPLANATION"
    SUMMARY = "SUMMARY"
    MISSING_INFORMATION = "MISSING_INFORMATION"
    SUMMARY_AND_MISSING = "SUMMARY_AND_MISSING"
    NEXT_STEPS = "NEXT_STEPS"
    DIAGNOSIS_UNCERTAINTY = "DIAGNOSIS_UNCERTAINTY"
    EVALUATION_GUIDANCE = "EVALUATION_GUIDANCE"
    CORRECTION_ACKNOWLEDGMENT = "CORRECTION_ACKNOWLEDGMENT"
    FOLLOW_UP = "FOLLOW_UP"
    GENERAL_EDUCATION = "GENERAL_EDUCATION"
    PATIENT_CONTEXT = "PATIENT_CONTEXT"


class Symptom(BaseModel):
    """Granular representation of a single extracted symptom."""
    name: str = Field(..., description="Canonical or extracted symptom name, e.g., 'pain'")
    location: Optional[str] = Field(None, description="Anatomical location, e.g., 'right upper abdomen'")
    duration: Optional[str] = Field(None, description="Duration of symptom, e.g., '2 days'")
    severity: Optional[str] = Field(None, description="Reported severity, e.g., 'moderate', '7/10'")
    onset: Optional[str] = Field(None, description="Onset timing, e.g., 'sudden', 'gradual'")
    frequency: Optional[str] = Field(None, description="Frequency pattern, e.g., 'constant', 'intermittent'")
    associated_symptoms: List[str] = Field(default_factory=list, description="Concurrent symptoms reported")
    radiation: Optional[str] = Field(None, description="Radiation or spreading location, e.g. 'back', 'right shoulder'")


class Measurement(BaseModel):
    """Extracted biometric or clinical measurement."""
    name: str = Field(..., description="Measurement name, e.g., 'blood_pressure', 'glucose'")
    value: float = Field(..., description="Numerical value")
    unit: Optional[str] = Field(None, description="Measurement unit, e.g., 'mmHg', 'mg/dL'")


class StructuredHealthInformation(BaseModel):
    """
    Structured representation of information extracted from natural language.
    Unknown information remains null or empty. Values are never invented.
    """
    symptoms: List[Symptom] = Field(default_factory=list)
    location: Optional[str] = Field(None, description="Primary anatomical area mentioned")
    duration: Optional[str] = Field(None, description="Overall duration mentioned")
    severity: Optional[str] = Field(None, description="Overall severity descriptor")
    onset: Optional[str] = Field(None, description="Onset description")
    frequency: Optional[str] = Field(None, description="Frequency pattern")
    associated_symptoms: List[str] = Field(default_factory=list)
    medical_history: List[str] = Field(default_factory=list)
    medications: List[str] = Field(default_factory=list)
    age: Optional[int] = Field(None, description="Patient age if explicitly provided")
    sex: Optional[str] = Field(None, description="Patient biological sex if explicitly provided")
    known_conditions: List[str] = Field(default_factory=list)
    measurements: List[Measurement] = Field(default_factory=list)
    negative_symptoms: List[str] = Field(default_factory=list, description="Explicitly denied or negative symptoms")
    triggers: List[str] = Field(default_factory=list, description="Explicit aggravating factors or triggers (e.g. oily meals)")
    relieving_factors: List[str] = Field(default_factory=list, description="Explicit relieving factors (e.g. rest, water)")
    progression: Optional[str] = Field(None, description="Reported symptom progression over time")
    previous_tests: List[str] = Field(default_factory=list, description="Previous medical tests reported, or 'none'")
    allergies: List[str] = Field(default_factory=list, description="Explicitly reported allergies")
    radiation: Optional[str] = Field(None, description="Radiation or spreading location (e.g. 'back', 'right shoulder')")
    body_temperature: Optional[str] = Field(None, description="Reported body temperature (e.g. '38.2°C')")
    functional_change: Optional[str] = Field(None, description="Reported change in functional capacity, exercise tolerance, or ability to work")
    llm_interpretations: List[str] = Field(default_factory=list, description="General educational possibilities or explanations suggested by assistant, strictly separated from patient-reported facts")


class ResponsePlan(BaseModel):
    """
    Structured deterministic plan controlling what the conversational response must do.
    Constructed before grounded synthesis to guide LLM wording without making clinical inferences.
    """
    user_goal: UserGoal = Field(default=UserGoal.UNKNOWN)
    sub_goals: List[str] = Field(default_factory=list, description="Specific sub-intents or secondary requested components")
    answer_mode: AnswerMode = Field(default=AnswerMode.DIRECT_ANSWER)
    current_turn_focus: List[str] = Field(default_factory=list)
    new_information: StructuredHealthInformation = Field(default_factory=StructuredHealthInformation)
    changed_information: List[str] = Field(default_factory=list)
    important_known_information: List[str] = Field(default_factory=list)
    facts_to_reference: List[str] = Field(default_factory=list, description="Explicit facts the response must ground on")
    unknowns_to_request: List[str] = Field(default_factory=list, description="Explicit clinically relevant unknown items to request or identify")
    previously_discussed_topics: List[str] = Field(default_factory=list)
    relevant_unanswered_topics: List[str] = Field(default_factory=list)
    explicit_negative_findings: List[str] = Field(default_factory=list)
    response_priorities: List[str] = Field(default_factory=list)
    safety_guidance: Optional[Union[str, List[str]]] = Field(default=None, description="Deterministic safety escalation or precaution instruction")
    platform_data_to_use: Optional[str] = Field(default=None, description="Real platform records/assessments to reference")
    medical_education_needed: List[str] = Field(default_factory=list, description="Approved general educational concepts")
    next_action: Optional[str] = Field(default=None, description="Recommended next action or clinical handoff")
    questions_to_ask: List[str] = Field(default_factory=list, description="Specific clinical questions to ask user")
    follow_up_topic: Optional[str] = Field(default=None)
    avoid_repetition: List[str] = Field(default_factory=list)
    avoid_repetition_topics: List[str] = Field(default_factory=list, description="Guidance concepts already communicated")
    already_delivered_guidance: List[str] = Field(default_factory=list, description="Historical guidance delivered in session")
    avoid_unsupported_inference: List[str] = Field(default_factory=list)


class ConversationProgressContext(BaseModel):
    """
    Compact deterministic representation of dialogue progression across turns.
    Differentiates newly reported facts from historical context and tracks topic coverage.
    """
    current_user_message: str
    newly_reported_information: StructuredHealthInformation
    previously_known_information: StructuredHealthInformation
    previously_discussed_topics: List[str] = Field(default_factory=list)
    unanswered_relevant_topics: List[str] = Field(default_factory=list)
    user_question_intent: Optional[str] = None
    response_plan: Optional[ResponsePlan] = None


class AssessmentHandoffContract(BaseModel):
    """
    Controlled handoff payload prepared for disease-specific ML engines and real predictions.
    Does NOT assert a diagnosis.
    """
    next_action: NextActionType = NextActionType.REQUEST_DISEASE_ASSESSMENT
    disease_candidates: List[str] = Field(
        default_factory=list,
        description="Relevant disease modules based on collected symptoms (e.g., 'liver_disease', 'diabetes')"
    )
    structured_inputs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Structured clinical fields extracted and validated for backend engine consumption"
    )
    assessment_result: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Completed assessment result from backend ML service if executed"
    )
    assessment_state: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Active assessment workflow state (status, collected and missing fields)"
    )


class ChatMessage(BaseModel):
    """Individual dialogue message in a conversation thread."""
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Text content of the message")
    timestamp: Optional[str] = Field(None, description="ISO timestamp")


class ChatRequest(BaseModel):
    """Incoming request payload from user/client."""
    user_id: str = Field(..., description="Unique user identifier")
    session_id: str = Field(..., description="Unique conversational session identifier")
    message: str = Field(..., description="Natural language message from user")
    conversation_history: Optional[List[ChatMessage]] = Field(
        None,
        description="Optional list of prior messages in the active dialogue"
    )
    context: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional client context such as source ('assessment', 'monitoring'), disease, assessment_id, metric"
    )


class ChatResponse(BaseModel):
    """Structured response payload returned by the Chatbot Service."""
    session_id: str
    assistant_message: str
    intent: IntentType
    structured_health_information: StructuredHealthInformation
    current_turn_extraction: Optional[StructuredHealthInformation] = None
    follow_up_question: Optional[str] = None
    safety_status: SafetyStatus
    next_action: NextActionType
    assessment_handoff: Optional[AssessmentHandoffContract] = None
    progression_context: Optional[ConversationProgressContext] = None
    response_plan: Optional[ResponsePlan] = None
