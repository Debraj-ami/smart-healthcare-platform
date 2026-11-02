# SmartHealth Conversational Assistant & NLP Layer

## 1. Overview & Purpose
The **Chatbot Service** is the conversational natural-language understanding (NLU) and clinical information-extraction layer of the Smart Healthcare Platform. It provides an intuitive, empathetic chat interface for users to:
- Report subjective health symptoms in natural language.
- Ask general health and wellness educational questions.
- Maintain multi-turn conversational context over session threads.
- Extract structured clinical information (`StructuredHealthInformation`) without manual form-filling.
- Coordinate seamless assessment handoffs (`REQUEST_DISEASE_ASSESSMENT`) to backend machine learning services (such as the Liver Disease ML pipeline).

> **CRITICAL ARCHITECTURAL BOUNDARY**:  
> The chatbot is a conversational screening and structured extraction interface. It is **NOT** a doctor, does **NOT** independently diagnose medical conditions, does **NOT** prescribe treatments, and **NEVER** fabricates numerical disease probabilities or lab values. All disease predictions must be computed by the dedicated ML services.

---

## 2. Architecture Diagram

```
                 +-----------------------------------------+
                 |            React Chat UI                |
                 +-----------------------------------------+
                                      |
                                      | POST /api/chat
                                      v
                 +-----------------------------------------+
                 |          FastAPI Chatbot Service        |
                 +-----------------------------------------+
                                      |
                                      v
                 +-----------------------------------------+
                 |   Deterministic Safety / Red-Flag Check |
                 +-----------------------------------------+
                    /                                   \
  [Emergency Detected]                             [Safe / Normal]
                   /                                       \
                  v                                         v
   +-----------------------------+         +-------------------------------+
   | Immediate Emergency Routing |         |  In-Memory Session Manager    |
   | (911 / Emergency Room Alert)|         |  (Maintains Dialogue State)   |
   +-----------------------------+         +-------------------------------+
                                                            |
                                                            v
                                           +-------------------------------+
                                           |      Configurable LLM         |
                                           | (Ollama / OpenAI-Compatible)  |
                                           +-------------------------------+
                                              /                         \
                                             v                           v
                              +--------------------+      +--------------------+
                              |  Natural Language  |      | Structured Symptom |
                              |      Response      |      |     Extraction     |
                              +--------------------+      +--------------------+
                                             \                           /
                                              v                         v
                                           +-------------------------------+
                                           | Deterministic Backend Checks  |
                                           | (Validation & Follow-Up Logic)|
                                           +-------------------------------+
                                                            |
                                             +--------------+--------------+
                                             |                             |
                                    [Missing Details]            [Sufficient Context]
                                             |                             |
                                             v                             v
                                 +----------------------+     +-------------------------+
                                 | Targeted Follow-Up   |     |   Assessment Handoff    |
                                 | Question (Duration,  |     |   Contract Prepared     |
                                 | Severity, Frequency) |     |   (Candidates: Liver,   |
                                 +----------------------+     |   Heart, Diabetes, etc.)|
                                                              +-------------------------+
                                                                           |
                                                                           v
                                                              +-------------------------+
                                                              |  Spring Boot / ML       |
                                                              |  Production Pipeline    |
                                                              +-------------------------+
```

---

## 3. Strict Separation of Responsibilities

To ensure patient safety, reliability, and architectural clarity, the platform strictly separates responsibilities:

| Component | Allowed Responsibilities | Forbidden Responsibilities |
| :--- | :--- | :--- |
| **LLM Engine** | • Natural language comprehension<br>• Maintaining conversational tone and empathy<br>• Extracting mentioned symptoms and attributes<br>• Asking natural clarifying follow-ups<br>• Answering general health education questions cautiously<br>• Explaining verified ML results already calculated by the backend | • **NEVER** invent disease probabilities or risk scores<br>• **NEVER** declare a medical diagnosis<br>• **NEVER** prescribe medications or treatments<br>• **NEVER** fabricate patient vitals or lab values<br>• **NEVER** downgrade red-flag medical emergencies |
| **Deterministic Backend** | • Pre-LLM safety & red-flag pattern screening<br>• Session state persistence and turn history<br>• Pydantic schema validation of extracted data<br>• Decision logic for assessment handoff completeness<br>• Guardrails preventing fabricated values | • Does not replace natural dialogue flow |
| **ML Inference Service** | • Calibrated disease probability calculation (e.g. Isotonic Logistic Regression)<br>• SHAP local and global feature attribution<br>• Feature preprocessing and standardization | • Does not conduct open conversational dialogue |

---

## 4. Safety Flow & Emergency Protocol

Patient safety is guaranteed through a **deterministic rule-based triage layer (`SafetyChecker`) that executes BEFORE any LLM invocation**.

```
User Message
    │
    ▼
[Deterministic Safety Checker]
    │
    ├─────► [Red Flag Detected?] ──YES──► [Immediate Emergency Response]
    │                                     • Next action: EMERGENCY_ROUTING
    │                                     • Safety status: URGENT_MEDICAL_EVALUATION
    │                                     • Directs to 911 / nearest emergency department
    │                                     • LLM is NEVER called; cannot downgrade urgency
    │
    └─────► [No Red Flag] ────────NO───► [Proceed to Session & LLM Processing]
```

### Monitored Emergency Patterns:
1. **Acute Coronary Syndrome**: Crushing chest pain, radiation to left arm, jaw, neck, or back.
2. **Severe Respiratory Distress**: Inability to breathe, gasping, cyanosis (turning blue).
3. **Altered Mental Status / Syncope**: Loss of consciousness, unresponsiveness.
4. **Acute Stroke Signs**: Sudden facial droop, slurred speech, one-sided weakness, thunderclap headache.
5. **Severe Hemorrhage**: Uncontrolled bleeding, vomiting or coughing up large amounts of blood.

---

## 5. Structured Symptom Extraction (No Hallucinated Data)

The LLM transforms unstructured user statements into structured `Pydantic` schemas:

### Example:
**User Input**:
> *"I have pain below my right ribs for two days. It comes and goes."*

**Extracted JSON (`StructuredHealthInformation`)**:
```json
{
  "symptoms": [
    {
      "name": "pain",
      "location": "right upper abdomen",
      "duration": "2 days",
      "severity": null,
      "onset": null,
      "frequency": "intermittent",
      "associated_symptoms": []
    }
  ],
  "location": "right upper abdomen",
  "duration": "2 days",
  "severity": null,
  "frequency": "intermittent"
}
```

### Strict Anti-Hallucination Policy:
- If severity (e.g., mild, moderate, 8/10) is **not** stated, `severity` remains `null`.
- If frequency (e.g., constant, intermittent) is **not** stated, `frequency` remains `null`.
- Unreported lab values, ages, and medications remain strictly `null` or empty lists.

---

## 6. Multi-Turn Session State & Follow-Up Engine

The service maintains state per `session_id` using `SessionManager` and `ConversationState`:
1. **Accumulation**: Turn 1 extracts symptom and location. Turn 2 extracts duration. Turn 3 extracts severity. Attributes are non-destructively merged across turns.
2. **Deterministic Follow-Up Check**: `FollowUpEngine` inspects what clinical descriptors are still missing (Duration $\rightarrow$ Severity $\rightarrow$ Frequency $\rightarrow$ Anatomical Specifics).
3. **No Redundancy**: Avoids asking repetitive questions for attributes already provided.

---

## 7. Assessment Handoff (`REQUEST_DISEASE_ASSESSMENT`)

When sufficient descriptive context is gathered, the chatbot triggers an assessment handoff contract:

```json
{
  "next_action": "REQUEST_DISEASE_ASSESSMENT",
  "disease_candidates": ["liver_disease"],
  "structured_inputs": {
    "symptoms": [{"name": "pain", "location": "right upper abdomen", "duration": "2 days", "severity": "moderate", "frequency": "intermittent"}],
    "location": "right upper abdomen",
    "duration": "2 days",
    "severity": "moderate",
    "frequency": "intermittent"
  }
}
```

The chatbot informs the patient that symptoms have been summarized and can be submitted to the clinical ML engine once biometric/laboratory values (e.g., Bilirubin, SGPT, Albumin) are entered.

---

## 8. General Health Educational Questions

For messages such as:
> *"What does high bilirubin mean?"*
> *"What is blood pressure?"*

The assistant provides concise, educational explanations:
- Explains the biological marker cautiously.
- Explicitly states that accurate interpretation requires clinical context.
- Does **not** diagnose the user or jump to conclusions.
- Recommends professional medical follow-up.

---

## 9. Error Handling & Gateway Resilience

If the LLM provider fails, times out, or returns a 5xx error:
- The service does **NOT** crash.
- It returns an HTTP 200 response with a controlled, safe fallback message:
  > *"The conversational AI service is temporarily unavailable. You can still use the structured health assessment features."*
- Deterministic emergency safety screening remains 100% active even if the LLM is completely offline.

---

## 10. Provider Configuration & Setup

Configuration is handled dynamically via environment variables. **No API keys are hardcoded.**

### Option A: Local Ollama (Recommended for Local Development)
Ollama provides a local, private OpenAI-compatible endpoint without external API costs or cloud dependencies.

1. Install and run [Ollama](https://ollama.com):
   ```bash
   ollama run llama3
   ```
2. Configure `.env` in `chatbot-service/`:
   ```bash
   LLM_PROVIDER=ollama
   LLM_BASE_URL=http://localhost:11434/v1
   LLM_API_KEY=ollama
   LLM_MODEL=llama3
   ```

### Option B: OpenAI-Compatible Remote Provider (vLLM, OpenAI, Azure)
Configure `.env`:
```bash
LLM_PROVIDER=openai_compatible
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-your-secret-api-key
LLM_MODEL=gpt-4o-mini
```

### Option C: Deterministic Mock Mode (Default for Testing / CI)
```bash
LLM_PROVIDER=mock
```
Uses `MockLLMClient` with zero external calls, ideal for automated test suites.

---

## 11. Automated Testing

Run the test suite using `pytest`:
```bash
cd chatbot-service
pytest
```

The automated test suite verifies 17 test cases without requiring external LLM connections or active localhost servers:
1. `test_general_health_question`: General inquiries return educational info without diagnosis.
2. `test_general_health_question_bilirubin`: Cautious explanation of clinical markers.
3. `test_normal_conversation`: Non-emergency medical history updates.
4. `test_basic_symptom_extraction`: Extraction of symptom, location, duration; unmentioned fields remain null.
5. `test_structured_extraction_with_mocked_llm_json`: Schema validation of LLM JSON output.
6. `test_missing_information_follow_up`: Asks targeted follow-ups when details are missing.
7. `test_follow_up_response_and_state_accumulation`: Multi-turn state accumulation.
8. `test_multi_turn_symptom_accumulation_chain`: Multi-turn symptom attribute merging.
9. `test_urgent_symptom_routing`: Red-flag emergencies routed before LLM; LLM cannot downgrade.
10. `test_llm_unavailable_safe_fallback`: Graceful fallback when LLM gateway fails.
11. `test_invalid_llm_response_controlled_handling`: Graceful handling of malformed LLM responses.
12. `test_no_fabricated_probability`: Guarantees chatbot never fabricates numerical risk scores.
13. `test_no_fabricated_diagnosis`: Guarantees chatbot never fabricates clinical diagnoses.
14. `test_disease_assessment_handoff_when_enough_information`: Handoff contract created upon complete symptom profile.
15. `test_disease_risk_assessment_request`: Direct user request for risk assessment.
16. `test_unknown_input_handling`: Clarification request for unclassifiable text.
17. `test_health_endpoint`: Liveness check returns HTTP 200.

---

## 12. Current Status & Production Limitations

### Implemented Functionality:
- Real configurable LLM client supporting Ollama and OpenAI-compatible endpoints.
- Structured JSON prompt engineering with Pydantic validation.
- Deterministic emergency red-flag safety screening running ahead of LLM.
- In-memory multi-turn session tracking and non-destructive attribute merging.
- Assessment handoff contract generation linking to production ML models.

### Production Limitations & Future Enhancements:
1. **Persistent Session Storage**: Active sessions currently reside in-memory; production deployment should persist session state in Redis or MySQL.
2. **Rate Limiting & Authentication**: The service relies on the Spring Boot API Gateway for user authentication and rate limiting.
3. **Domain Vocabulary Expansion**: Additional disease candidate mappings can be added as diabetes and cardiology ML models are trained and calibrated in subsequent phases.

---

## 13. Dynamic Clinical Knowledge Layer (Phase 9 Step 3.6 & 3.6-B)

### Purpose & Architecture
When users ask clinical inquiry questions such as *"What could potentially cause these symptoms and what tests would help a doctor?"*, the **Clinical Knowledge Layer** (`app/clinical_knowledge.py`) performs deterministic, context-aware retrieval of approved clinical components:
- Fine-grained `ClinicalKnowledgeComponent` items (e.g. meal-related biliary pattern, nausea/vomiting context, glycemic regulation).
- Context-aware filtering based on positive extracted symptoms and explicit denials (`negative_symptoms`).
- Decouples clinical knowledge storage and retrieval from response generation.

---

## 14. Grounded LLM Synthesis for Natural Clinical Conversation (Phase 9 Step 3.7)

### Architecture
In Phase 9 Step 3.7, the Answer-First architecture was upgraded from rigid, hardcoded medical text assembly to **Grounded LLM Synthesis**:

```
USER MESSAGE
    │
    ▼
Emergency Safety Gate (Deterministic pre-triage; bypasses LLM immediately if red flags)
    │
    ▼
Deterministic Extraction (Preserves user facts, distinguishing 'not mentioned' from 'negative')
    │
    ▼
Verified User Facts (Authoritative source of truth for patient situation)
    │
    ▼
Clinical Knowledge Retrieval (ClinicalKnowledgeEngine selects relevant components & exclusions)
    │
    ▼
Relevant Grounding Context (Compact structured educational background)
    │
    ▼
GROUNDED LLM SYNTHESIS (GroundedSynthesisEngine via Ollama / OpenAI-compatible LLM)
    │
    ▼
Natural Conversational Response
    │
    ▼
USER
```

### Core Principles & Clear Separation of Responsibilities

1. **User Facts are Authoritative (Patient Source of Truth)**:
   - Extracted deterministically from the user's actual messages.
   - Distinguishes explicit negative findings (e.g. *"no nausea"*, *"denies chest pain"*) from unmentioned symptoms.
   - The LLM is strictly prohibited from inventing symptoms, test results, severity, frequency, or medical history.

2. **Clinical Knowledge is Grounding (Educational Background)**:
   - Retrieved deterministically by `ClinicalKnowledgeEngine`.
   - Never discovers medical knowledge from scratch; only provides approved, relevant knowledge components for the matched clinical domain.
   - Contradicted components (e.g. meal-provocation when the user denied food triggers) are excluded prior to prompt construction.
   - Structured metadata labels all categories as educational background for clinicians, **never** patient diagnoses.

3. **LLM is Language Synthesis**:
   - The LLM receives the original user message, verified facts, approved grounding, and strict clinical guardrails.
   - Synthesizes a natural, empathetic, professional explanation directly answering the user's inquiry.
   - Never diagnoses, never prescribes drugs, and frames all evaluations as options a qualified clinician may consider.
   - Asks at most ONE approved follow-up question (guided by `FollowUpEngine`).

4. **Deterministic Authoritative Systems**:
   - **Emergency Safety Gate**: Executes BEFORE LLM synthesis. Severe chest pain / acute respiratory distress immediately routes to emergency guidance without calling Ollama.
   - **Machine Learning & SHAP**: Hand-off to dedicated ML services (e.g. Liver Disease ML model) remains completely authoritative. The LLM cannot alter risk probabilities, ML predictions, or SHAP values.
   - **Patient History**: Database context queries (`PATIENT_CONTEXT`) remain authoritative; the LLM merely phrases verified database metrics.

5. **Safe Fallback & Resilience**:
   - If the LLM times out or connection fails, `GroundedSynthesisEngine` safely falls back to deterministic clinical grounding output.
   - No fabricated natural response is generated on failure.

> **NOTE ON RAG**:  
> Full vector retrieval-augmented generation (vector databases, dense embeddings, external medical search APIs) is **NOT** implemented yet. The current architecture uses high-precision deterministic clinical component retrieval coupled with grounded synthesis, ensuring sub-millisecond retrieval predictability with conversational naturalness.

---

## 15. Grounded Response Validation (Phase 9 Step 3.7-C)

### Why It Exists
While Grounded LLM Synthesis naturally articulates responses, LLMs can inadvertently hallucinate patient-specific details that were never reported (e.g. attributing *"episodic nature of your pain"* when the user only reported *"pain"* without frequency, or claiming *"your temperature is 38.2°C"* when only a subjective *"feverish feeling"* was shared).

**Grounded Response Validation (`ResponseValidator`)** serves as a deterministic safety guard between LLM generation and the user:
```
USER MESSAGE
    │
    ▼
Emergency Safety Gate
    │
    ▼
Deterministic Extraction
    │
    ▼
Verified User Facts
    │
    ▼
Clinical Knowledge Grounding
    │
    ▼
Response Planner
    │
    ▼
Grounded LLM Synthesis
    │
    ▼
FINAL RESPONSE VALIDATION (ResponseValidator: <1ms, 0 Network, 0 LLM Calls)
    │
    ▼
USER (Authoritative Validated Response)
```

### Supported Facts vs. Unknown Facts
- **Supported Facts**: Only health attributes explicitly provided by the user in the current turn or recorded in cumulative session state are allowed to be attributed to the patient (e.g. location, quality, reported triggers, explicitly measured vitals).
- **Unknown is NOT Negative**: If a symptom or finding (such as jaundice, abnormal urine, or radiation) was never mentioned by the patient, the system strictly treats it as **UNKNOWN / NOT REPORTED**. The assistant must **never** state *"You don't have jaundice"* when the patient simply never mentioned jaundice. Neutral phrases such as *"You haven't reported jaundice"* or inquiring about it are permitted.

### Patient Facts vs. General Clinical Knowledge
The validator sharply separates general medical background from patient-specific attribution:
- **Allowed (General Clinical Knowledge)**:
  - *"Some biliary conditions can cause episodic upper abdominal pain."*
  - *"Gallbladder issues can cause pain that radiates toward the back or right shoulder."*
  - *"Clinicians evaluating these symptoms would check whether jaundice is present."*
- **Blocked / Repaired (Unsupported Patient Claims)**:
  - *"The episodic nature of your pain..."* (when frequency was unstated).
  - *"Your pain radiates to your back."* (when radiation was unstated).
  - *"Your temperature is 38.2°C."* (when no numerical measurement was given).
  - *"You do not have jaundice."* (when jaundice was never discussed).

### Unsupported Inference Protection
The validator blocks affirmative probability and certainty assertions derived from negative findings or symptom presence:
- Prohibits definitive claims such as *"makes a heart problem unlikely"*, *"rules out infection"*, *"confirms gallbladder disease"*, or *"means you probably have..."*.
- Enforces safe, non-diagnostic framing: acknowledging the absence of red flags without making medical exclusion conclusions.

### User Correction Handling
When the user subsequently updates or clarifies information (e.g. Turn 1: *"I have stomach pain"*, Turn 2: *"The pain comes and goes rather than being constant"*), the cumulative conversation state adopts the verified frequency. Once verified, statements referencing episodic or intermittent pain become valid and are permitted.

### Streaming Validation Behavior
In streaming endpoints (`POST /api/chat/stream`), tokens stream smoothly to maintain responsive interactive UX. As streaming finishes, the entire accumulated response is validated and safely repaired by `ResponseValidator`. The final SSE `complete` event payload contains the authoritative, validated `assistant_message`, which the frontend applies to guarantee factual consistency.

### Safe Repair & Fallback Strategy
1. **Validation**: The validator inspects the message against normalized `SupportedPatientFacts`.
2. **Safe Local Repair**: When an unverified patient claim is detected, the validator deterministically rewrites or removes the unsupported attribution while preserving valid context (e.g. converting *"The episodic nature of your pain and its worsening after heavy meals"* to *"The worsening of your pain after heavy meals"*).
3. **Deterministic Fallback**: If safe local repair cannot resolve all violations, the system safely falls back to the deterministic grounded output from `ClinicalKnowledgeEngine`, guaranteeing that no fabricated patient facts reach the user.

---

## 12. Advanced Conversational Progression & Response Planning Architecture

### Core Objectives
1. **Genuine Context-Awareness**: The assistant maintains multi-turn state across long dialogues, distinguishing between explicit patient-reported facts, explicit denials (negative findings), and unmeasured/unknown parameters.
2. **Whole-Question Answering**: Multi-part queries (e.g. *"What are the important things I've told you about my symptoms, and what information do you still need to know?"*) are decomposed by the planner into sub-goals and answered in full: summarizing verified facts while explicitly detailing clinically relevant missing information.
3. **Conversational Trajectory & Repetition Control**: Tracks delivered guidance concepts (`session.delivered_guidance`) across turns. When a user transitions from diagnostic uncertainty questions (*"Do you think this is definitely a heart problem?"*) to actionable inquiries (*"What should I do next?"*), the assistant acknowledges previous discussion and moves directly forward into concrete next steps (appointment scheduling, symptom logs, pacing, emergency red flags) without repeating redundant differential diagnostic text.
4. **Diagnostic Boundary & Uncertainty Communication**: Directly addresses user inquiries regarding disease certainty without diagnosing. Clearly differentiates symptoms from broad physiological categories (cardiovascular, respiratory, deconditioning) and formal clinical diagnoses, explaining necessary medical evaluations (resting 12-lead ECG, physical examination, blood work).
5. **No Medical Hardcoding or Fabricated Fallbacks**: All reasoning flows from structured state, verified clinical knowledge, and deterministic planning. No fake laboratory values, invented probabilities, or hardcoded medical response paragraphs are used.



