export type SafetyStatus =
  | 'NORMAL_CONVERSATION'
  | 'URGENT_MEDICAL_EVALUATION'
  | 'SAFE'
  | 'AMBIGUOUS_SYMPTOM';

export type IntentType =
  | 'GENERAL_HEALTH_QUESTION'
  | 'SYMPTOM_ASSESSMENT'
  | 'DISEASE_RISK_ASSESSMENT'
  | 'FOLLOW_UP_RESPONSE'
  | 'MEDICAL_HISTORY_UPDATE'
  | 'MONITORING_QUERY'
  | 'UNKNOWN'
  | 'EMERGENCY_TRIAGE'
  | 'OUT_OF_SCOPE';

export type NextActionType =
  | 'ASK_FOLLOW_UP'
  | 'REQUEST_DISEASE_ASSESSMENT'
  | 'PROVIDE_GENERAL_INFO'
  | 'EMERGENCY_ROUTING'
  | 'CLARIFY'
  | 'ANSWER_QUESTION'
  | 'PROMPT_FOR_ASSESSMENT'
  | 'URGENT_TRIAGE_DISPATCH';

export interface ExtractedSymptom {
  name: string;
  symptom_name?: string; // backwards compatibility
  location?: string;
  body_location?: string; // backwards compatibility
  duration?: string;
  severity?: string;
  onset?: string;
  frequency?: string;
  associated_symptoms?: string[];
}

export interface ExtractedMeasurement {
  name: string;
  value: number;
  unit?: string;
}

export interface StructuredHealthInformation {
  symptoms?: ExtractedSymptom[];
  location?: string;
  duration?: string;
  severity?: string;
  onset?: string;
  frequency?: string;
  associated_symptoms?: string[];
  medical_history?: string[];
  medications?: string[];
  age?: number;
  sex?: string;
  known_conditions?: string[];
  measurements?: ExtractedMeasurement[];
  metrics?: Record<string, any>;
  lifestyle?: Record<string, any>;
  negative_symptoms?: string[];
  triggers?: string[];
  relieving_factors?: string[];
  progression?: string;
  previous_tests?: string[];
  allergies?: string[];
}

export interface AssessmentHandoffContract {
  next_action?: NextActionType;
  disease_candidates?: string[];
  structured_inputs?: Record<string, any>;
  assessment_result?: Record<string, any> | null;
  assessment_state?: Record<string, any> | null;
  target_engine?: string;
  ready_for_inference?: boolean;
  extracted_features?: Record<string, any>;
  missing_features?: string[];
}

export interface ChatAssistantContextData {
  source?: 'assessment' | 'monitoring' | 'dashboard' | string;
  disease?: string;
  assessmentId?: number | string;
  metric?: string;
  [key: string]: any;
}

export interface ChatRequest {
  message: string;
  session_id?: string;
  user_id?: string;
  conversation_history?: Array<{
    role: string;
    content: string;
    timestamp?: string;
  }>;
  current_route?: string;
  context?: ChatAssistantContextData;
  metadata?: Record<string, any>;
}

export interface ChatResponse {
  session_id: string;
  assistant_message: string;
  message?: string; // backwards compatibility fallback
  intent: IntentType;
  structured_health_information: StructuredHealthInformation;
  current_turn_extraction?: StructuredHealthInformation | null;
  structured_data?: StructuredHealthInformation; // backwards compatibility fallback
  follow_up_question?: string | null;
  safety_status: SafetyStatus;
  next_action: NextActionType;
  assessment_handoff?: AssessmentHandoffContract | null;
  handoff_contract?: AssessmentHandoffContract | null; // backwards compatibility fallback
}

export interface ChatMessage {
  id: string;
  sender: 'user' | 'assistant';
  text: string;
  timestamp: string;
  safetyStatus?: SafetyStatus;
  intent?: IntentType;
  structuredData?: StructuredHealthInformation;
  currentTurnExtraction?: StructuredHealthInformation | null;
  assessmentHandoff?: AssessmentHandoffContract | null;
}

