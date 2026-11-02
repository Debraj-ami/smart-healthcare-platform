# Chatbot User Flows & Concrete Dialogue Scenarios
## Smart Healthcare Platform — Multi-Turn Conversational Examples

This document outlines the concrete, turn-by-turn dialogue flows across all key user journeys, edge cases, and failure scenarios.

---

## Flow A: General Health Question (Path A)

**Scenario:** A curious patient or family member asks for an educational explanation of a health condition without personal symptom complaints or laboratory reports.

### Turn 1
* **User:** *"What is fatty liver disease and how does it develop?"*
* **Safety Evaluation:** Normal conversation (no emergency red flags).
* **Intent Classification:** `GENERAL_HEALTH_QUESTION`
* **Fast-Path Check:** No structured lab values detected; routed to conversational LLM.
* **Assistant Response:**
  > "Fatty liver disease (steatosis) occurs when excess fat accumulates in liver cells. It is generally categorized into non-alcoholic fatty liver disease (NAFLD/MASLD), which is closely linked to metabolic factors like insulin resistance, elevated triglycerides, and obesity, and alcohol-related liver disease.
  >
  > In its early stages, it often causes no noticeable symptoms. Over time, persistent fat accumulation can trigger inflammation (steatohepatitis) or fibrosis.
  >
  > *Please note: This is general medical information and not a personal clinical evaluation. If you have concerns about your liver health, a physician can order appropriate screening tests such as an ultrasound or blood panel.*"
* **Next Action:** `PROVIDE_GENERAL_INFO`
* **Assessment Handoff:** None.

---

## Flow B: New Patient with Symptoms and No Reports (Path B)

**Scenario:** A patient experiences abdominal discomfort and nausea. They have no laboratory reports or physical test results. The chatbot progressively collects symptom attributes without ever demanding lab numbers.

### Turn 1
* **User:** *"I've been feeling this uncomfortable pain in my stomach and feeling sick."*
* **Safety Evaluation:** Normal conversation (moderate discomfort, no acute red flags).
* **Intent Classification:** `SYMPTOM_ASSESSMENT`
* **Extracted Attributes:** `symptoms: ["pain", "nausea"]`, `location: "stomach"`.
* **Missing Attributes:** Specific anatomical quadrant, duration, severity, constancy.
* **Assistant Response:**
  > "I'm sorry to hear you're feeling unwell. To help me understand what you're experiencing, could you tell me where in your abdomen the discomfort is located, and how long you've had it?"
* **Next Action:** `ASK_FOLLOW_UP`

### Turn 2
* **User:** *"It's on my right side just under my rib cage, started about three days ago."*
* **Safety Evaluation:** Normal conversation.
* **Intent Classification:** `FOLLOW_UP_RESPONSE`
* **Extracted Attributes:** `location: "right upper abdomen"`, `duration: "3 days"`.
* **Missing Attributes:** Severity (scale of 1-10), frequency (constant vs. intermittent).
* **Assistant Response:**
  > "Thank you for clarifying. Is this pain steady and constant, or does it come and go? Also, how severe would you rate it on a scale from 1 to 10?"
* **Next Action:** `ASK_FOLLOW_UP`

### Turn 3
* **User:** *"It comes and goes, especially after meals. Maybe a 5 or 6 out of 10."*
* **Safety Evaluation:** Normal conversation.
* **Intent Classification:** `FOLLOW_UP_RESPONSE`
* **Extracted Attributes:** `frequency: "intermittent"`, `severity: "moderate (5-6/10)"`, `trigger: "post-prandial"`.
* **Missing Attributes:** Any systemic symptoms (fever, jaundice, dark urine).
* **Assistant Response:**
  > "Have you noticed any yellowing of your eyes or skin, dark tea-colored urine, fever, or chills alongside this pain?"
* **Next Action:** `ASK_FOLLOW_UP`

### Turn 4
* **User:** *"No fever, and no yellowing of eyes or skin."*
* **Safety Evaluation:** Normal conversation.
* **Symptom Context Completeness:** Sufficient descriptive context established.
* **Assistant Response:**
  > "Thank you for providing those details. 
  > 
  > **Summary of your reported symptoms:**
  > • Moderate intermittent pain in the right upper abdomen (under the right ribs) for 3 days
  > • Associated nausea, aggravated after eating
  > • No jaundice, fever, or changes in urine color
  >
  > **Screening Guidance:**
  > Intermittent pain in the right upper quadrant after meals is commonly associated with gallbladder or hepatobiliary issues (such as gallstones or biliary colic) or upper gastrointestinal inflammation. 
  >
  > Because these symptoms warrant an in-person physical exam and potentially an abdominal ultrasound or liver function blood test, I recommend scheduling an appointment with your primary care physician or a gastroenterologist for an accurate evaluation.
  >
  > *If you develop a high fever, severe unmanageable pain, or yellowing of the skin/eyes, seek urgent medical care immediately.*"
* **Next Action:** `PROVIDE_GENERAL_INFO` (Symptom summary complete, no lab values forced).

---

## Flow C: Patient Providing Liver Report Values (Path C)

**Scenario:** The user explicitly brings laboratory report results for screening.

### Turn 1
* **User:** *"I got my annual physical blood work back and I want to check my liver health. I am 62 years old and male."*
* **Safety Evaluation:** Normal conversation.
* **Intent Classification:** `DISEASE_RISK_ASSESSMENT`
* **Extracted Attributes:** `age: 62`, `gender: "Male"`.
* **Session Liver State:** Initialized; `collected: {age: 62, gender: 'Male'}`, `missing: 8 lab metrics`.
* **Assistant Response:**
  > "I can help you review your liver panel values using our production screening model.
  >
  > **Recorded values so far:**
  > • Age: 62 years
  > • Gender: Male
  >
  > **To run the screening model, I still need your laboratory metrics:**
  > • Total Bilirubin (mg/dL)
  > • Direct Bilirubin (mg/dL)
  > • Alkaline Phosphatase (IU/L)
  > • SGPT / ALT (U/L)
  > • SGOT / AST (U/L)
  > • Total Proteins (g/dL)
  > • Albumin (g/dL)
  > • A/G Ratio
  >
  > You can paste or type these values whenever you have them ready."
* **Next Action:** `ASK_FOLLOW_UP`

### Turn 2
* **User:** *"My bilirubin is 10.9 and direct bilirubin is 5.5. Alkaline phosphatase is 699."*
* **Fast-Path Detection:** Lab metrics detected (`total_bilirubin: 10.9`, `direct_bilirubin: 5.5`, `alkaline_phosphatase: 699.0`).
* **Session Liver State:** Updated; `collected: 5 fields`, `missing: 5 fields`.
* **Assistant Response:**
  > "Recorded values so far:
  > • Age: 62 years
  > • Gender: Male
  > • Total Bilirubin: 10.9 mg/dL
  > • Direct Bilirubin: 5.5 mg/dL
  > • Alkaline Phosphatase: 699.0 IU/L
  >
  > To complete the screening, I still need:
  > • SGPT / ALT (U/L)
  > • SGOT / AST (U/L)
  > • Total Proteins (g/dL)
  > • Albumin (g/dL)
  > • A/G Ratio"
* **Next Action:** `ASK_FOLLOW_UP`

### Turn 3
* **User:** *"SGPT is 64, SGOT 100, total proteins 6.8, albumin 3.0, and A/G ratio is 0.8."*
* **Fast-Path Detection:** Remaining 5 lab metrics extracted and validated.
* **Session Liver State:** 10 of 10 fields complete. Validation errors: 0.
* **Backend Execution:** Chatbot calls Spring Boot `POST /api/assessments/liver/predict` → ML FastAPI → Isotonic calibrated probability & SHAP attribution → Spring Boot saves `DiseaseAssessment (ID: 101)`.
* **Assistant Response:**
  > "I have all 10 clinical and laboratory values required for the Liver Disease screening assessment. The calculation has completed.
  >
  > **Assessment Results:**
  > • **Assessment ID:** 101
  > • **Screening Model:** Logistic Regression (Isotonically Calibrated)
  > • **Estimated Risk Probability:** **61.6%**
  >
  > **Key Contributing Factors (SHAP Explanation):**
  > • **Total Bilirubin (10.9 mg/dL)** had the strongest positive influence increasing the estimated risk score.
  > • **Alkaline Phosphatase (699.0 IU/L)** also contributed substantially toward the elevated risk estimate.
  > • **Direct Bilirubin (5.5 mg/dL)** contributed moderately toward the prediction.
  >
  > **Important Medical Notice:**
  > This probability is a statistical screening estimation derived from historical clinical cohorts, **NOT a definitive medical diagnosis**. It does not prove the presence of liver disease. Elevated liver enzymes and bilirubin require thorough clinical interpretation, correlation with your medical history, and follow-up imaging by a qualified physician."
* **Next Action:** `REQUEST_DISEASE_ASSESSMENT` (Complete, saved in profile).

---

## Flow D: Returning Patient Asking About a Metric Trend (Path D)

**Scenario:** An authenticated patient (`patientId: 4`) returns to the portal and asks about their historical lab trends.

### Turn 1
* **User:** *"How are my liver enzymes trending compared to my last checkup?"*
* **Safety Evaluation:** Normal conversation.
* **Intent Classification:** `MONITORING_QUERY`
* **Backend Interaction:** Chatbot calls Spring Boot `GET /api/v1/observations/patient/4` and `GET /api/v1/baselines/patient/4`.
* **Spring Boot Response:** Retrieves 3 historical observations over 6 months:
  - May 2026: SGPT 110 U/L, SGOT 135 U/L (baseline mean: SGPT 45 U/L)
  - July 2026: SGPT 85 U/L, SGOT 98 U/L
  - September 2026: SGPT 58 U/L, SGOT 64 U/L
* **Assistant Response:**
  > "Based on your health records over the past 4 months, your liver transaminase enzymes show a consistent downward (improving) trend:
  >
  > • **SGPT (ALT):** Decreased from 110 U/L in May to 85 U/L in July, and most recently to **58 U/L** in September (approaching your personal baseline mean of 45 U/L).
  > • **SGOT (AST):** Decreased from 135 U/L to **64 U/L**.
  >
  > While both enzymes are still slightly elevated above your personal baseline, the steady decrease indicates positive progress.
  >
  > You can view the full longitudinal charts and deviation graphs in your **Health Monitoring** dashboard tab."
* **Next Action:** `PROVIDE_GENERAL_INFO`

---

## Flow E: Emergency / Red-Flag Situation

**Scenario:** A patient reports severe, acute symptoms indicative of a life-threatening cardiovascular emergency.

### Turn 1
* **User:** *"I'm having sudden crushing chest pain that radiates down my left arm, and I can hardly breathe."*
* **Safety Evaluation:** **URGENT_MEDICAL_EVALUATION** (Triggered: Acute Coronary Syndrome & Severe Respiratory Distress).
* **Pre-LLM Action:** Deterministic safety filter halts all conversational and diagnostic processing immediately.
* **Assistant Response:**
  > "🚨 **EMERGENCY MEDICAL WARNING**
  >
  > Your message indicates symptoms that may be life-threatening (**severe crushing chest pain radiating to the arm and breathing difficulty**).
  >
  > **DO NOT WAIT for this chatbot to respond or attempt an online health assessment.**
  >
  > **Immediate Action Required:**
  > 1. **Call 911 (or your local emergency services) immediately.**
  > 2. If you are alone, unlock your front door so emergency responders can enter.
  > 3. Sit or lie down in a comfortable position and try to remain calm while emergency assistance is on the way."
* **Next Action:** `EMERGENCY_ROUTING`
* **Assessment Handoff:** Disabled. Session locked to emergency protocol.

---

## Flow F: Non-Health / Casual Conversational Input

**Scenario:** User sends greetings, small talk, or an off-topic query.

### Turn 1
* **User:** *"Good morning! Can you help me plan my dinner tonight?"*
* **Safety Evaluation:** Normal conversation.
* **Intent Classification:** `UNKNOWN` / `GENERAL_CONVERSATION`
* **Assistant Response:**
  > "Good morning! While I specialize in health screening, symptom exploration, and medical report reviews, I can certainly share general healthy dietary principles.
  >
  > For a balanced, liver- and heart-friendly meal, consider:
  > • A lean protein source (such as grilled salmon, chicken breast, or lentils)
  > • A generous serving of leafy greens or cruciferous vegetables
  > • Whole grains like quinoa or brown rice
  > • Healthy fats such as olive oil or avocado
  >
  > If you have specific dietary restrictions or health symptoms you'd like to discuss, feel free to let me know!"
* **Next Action:** `PROVIDE_GENERAL_INFO`

---

## Flow G: LLM Service Unavailable / Timeout

**Scenario:** The local Ollama server is offline, restarting, or the request exceeded the 60-second processing window.

### Turn 1
* **User:** *"Can you explain what an A/G ratio of 0.8 means?"*
* **System Event:** Client catches `LLMUnavailableError` (HTTP connection refused or read timeout).
* **Fail-Safe Mechanism:** No crash; safe system fallback engaged.
* **Assistant Response:**
  > "The conversational AI service is temporarily slow or unavailable.
  >
  > While conversational answers are temporarily paused:
  > • You can still use our direct disease assessment calculators in the **Assessments** tab.
  > • Your previous session history and health records remain safe.
  >
  > Please try sending your message again in a moment."
* **Next Action:** `PROVIDE_GENERAL_INFO`
* **Safety Status:** `NORMAL_CONVERSATION` (Never invents speculative medical text when offline).

---

## Flow H: Structured Extraction Failure / Malformed LLM Output

**Scenario:** The LLM produces invalid schema formatting (e.g. outputting strings inside `measurements` instead of dictionaries).

### Turn 1
* **User:** *"Here are my labs: Total Bilirubin is 10.9 mg/dL and Direct Bilirubin is 5.5 mg/dL."*
* **System Event:** LLM response contains:
  `"measurements": ["Total Bilirubin (10.9 mg/dL)", "Direct Bilirubin (5.5 mg/dL)"]`
* **Pydantic Validation:** Raises `LLMResponseError` due to schema constraint failure.
* **Fail-Safe Mechanism:** System catches `LLMResponseError`, logs warning, and silently activates deterministic fallback extractor (`extract_health_information`).
* **Deterministic Fallback Result:**
  - Extracts `Measurement(name='total_bilirubin', value=10.9, unit='mg/dL')`
  - Extracts `Measurement(name='direct_bilirubin', value=5.5, unit='mg/dL')`
* **Assistant Response:**
  > "Recorded values so far:
  > • Total Bilirubin: 10.9 mg/dL
  > • Direct Bilirubin: 5.5 mg/dL
  >
  > To proceed with the Liver Disease screening assessment, I still need:
  > • Age
  > • Gender (Male or Female)
  > • Alkaline Phosphatase (IU/L)
  > • SGPT / ALT (U/L)
  > • SGOT / AST (U/L)
  > • Total Proteins (g/dL)
  > • Albumin (g/dL)
  > • A/G Ratio"
* **User Experience:** Seamless; the user experiences **zero disruption** or error popups, and their laboratory parameters are accurately recorded.
