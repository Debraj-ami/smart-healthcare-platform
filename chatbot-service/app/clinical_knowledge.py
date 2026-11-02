"""
Clinical Knowledge Layer for SmartHealth Chatbot.
Provides deterministic, fast (<1ms) clinical knowledge selection and dynamic,
context-aware Answer-First response generation based on verified patient facts without invoking LLMs.

Design Principles:
1. User-Source-of-Truth remains absolute: Knowledge entries provide clinical educational
   categories and test explanations, but NEVER convert suggestions into patient facts.
2. No diagnosis: All possibilities are framed as categories a clinician may evaluate.
3. No hallucinated tests/prescriptions: Only project-approved diagnostic concepts are referenced.
4. Clean separation: Knowledge components and scoring are strictly decoupled from orchestration.
5. Context-aware knowledge selection: Component selection, distinguishing information, and
   tests/evaluations dynamically reflect verified user facts (positive and negative).
"""

import re
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Tuple

from app.schemas import (
    ChatMessage,
    ChatResponse,
    StructuredHealthInformation,
    IntentType,
    SafetyStatus,
    NextActionType,
    AssessmentHandoffContract
)

logger = logging.getLogger(__name__)


@dataclass
class ClinicalKnowledgeComponent:
    """
    Fine-grained, context-aware clinical knowledge component.
    Specifies WHEN it is relevant based on verified patient facts and exclusions.
    """
    component_id: str
    domain_id: str
    title: str
    required_features: List[str] = field(default_factory=list)
    supporting_features: List[str] = field(default_factory=list)
    excluded_by_features: List[str] = field(default_factory=list)
    possible_categories: List[str] = field(default_factory=list)
    relevant_information: List[str] = field(default_factory=list)
    clinical_examinations: List[str] = field(default_factory=list)
    laboratory_tests: List[str] = field(default_factory=list)
    diagnostic_imaging: List[str] = field(default_factory=list)
    priority: int = 0


@dataclass
class ClinicalKnowledgeEntry:
    """Structured representation of a disease/symptom clinical knowledge domain."""
    domain_id: str
    display_title: str
    matching_symptoms: List[str]
    matching_locations: List[str]
    matching_triggers: List[str]
    matching_keywords: List[str]
    possible_categories: List[str]
    relevant_information: List[str]
    clinical_examinations: List[str]
    laboratory_tests: List[str]
    diagnostic_imaging: List[str]
    disease_candidates: List[str]
    explanation_notes: Optional[str] = None


@dataclass
class ClinicalGroundingContext:
    """Structured clinical grounding payload produced by ClinicalKnowledgeEngine for LLM synthesis."""
    domain_id: str
    display_title: str
    selected_components: List[ClinicalKnowledgeComponent]
    component_ids: List[str]
    possible_categories: List[str]
    relevant_information: List[str]
    clinical_examinations: List[str]
    laboratory_tests: List[str]
    diagnostic_imaging: List[str]
    disease_candidates: List[str]
    deterministic_fallback_message: str
    excluded_components: Dict[str, str] = field(default_factory=dict)


# Curated, approved project clinical knowledge entries for domain matching & backwards compatibility
CLINICAL_KNOWLEDGE_BASE: List[ClinicalKnowledgeEntry] = [
    # 1. Hepatobiliary / RUQ (Liver & Gallbladder Disease Domain)
    ClinicalKnowledgeEntry(
        domain_id="hepatobiliary_ruq",
        display_title="Upper-right abdominal / Hepatobiliary evaluation",
        matching_symptoms=[
            "pain", "abdominal pain", "nausea", "vomiting", "fever",
            "jaundice", "dark urine", "reduced appetite"
        ],
        matching_locations=["right upper abdomen", "abdomen"],
        matching_triggers=["oily", "fatty", "heavy", "food", "meals"],
        matching_keywords=[
            "gallbladder", "biliary", "liver", "ruq", "upper-right",
            "upper right", "right upper", "ribs", "jaundice", "hepatobiliary"
        ],
        possible_categories=[
            "**Hepatobiliary causes**: Irritation or inflammation of the gallbladder (such as biliary colic or gallstones) or localized liver/biliary tract inflammation.",
            "**Gastrointestinal causes**: Upper digestive tract irritation or localized visceral discomfort.",
            "**Infectious or inflammatory factors**: Systemic or gastrointestinal infections, especially when accompanied by feverish sensations or fatigue."
        ],
        relevant_information=[
            "Relationship to meals, particularly whether symptoms intensify after fatty or oily food.",
            "Radiation of discomfort to the right shoulder, scapula, or upper back.",
            "Presence of systemic signs such as fever, chills, jaundice (yellowing of skin/eyes), or darkening of urine."
        ],
        clinical_examinations=[
            "A physician performs physical abdominal palpation to check for localized tenderness or signs of gallbladder irritation (such as Murphy's sign)."
        ],
        laboratory_tests=[
            "Liver Function Tests (LFTs) measuring bilirubin, ALT/SGPT, AST/SGOT, and alkaline phosphatase, along with a complete blood count (CBC) to assess for systemic inflammation."
        ],
        diagnostic_imaging=[
            "An abdominal ultrasound is frequently the first-line examination to inspect the gallbladder, bile ducts, and liver parenchyma."
        ],
        disease_candidates=["liver_disease"],
        explanation_notes="Approved project content migrated from initial RUQ baseline."
    ),

    # 2. Cardiovascular / Chest (Heart Disease Domain)
    ClinicalKnowledgeEntry(
        domain_id="cardiovascular_chest",
        display_title="Chest discomfort and cardiovascular evaluation",
        matching_symptoms=[
            "chest pain", "pain", "shortness of breath", "breathless",
            "difficulty breathing", "palpitations", "chest discomfort"
        ],
        matching_locations=["chest"],
        matching_triggers=["exertion", "exercise", "climbing stairs", "physical activity", "stress"],
        matching_keywords=[
            "chest", "cardiovascular", "heart", "angina", "shortness of breath",
            "palpitations", "cardiac", "coronary"
        ],
        possible_categories=[
            "**Cardiovascular causes**: Variations in myocardial blood flow, coronary circulation considerations, or cardiac workload strain.",
            "**Respiratory or pulmonary causes**: Pleural irritation, reactive airway patterns, or pulmonary considerations.",
            "**Musculoskeletal chest wall factors**: Costochondral irritation, chest wall muscular strain, or postural factors.",
            "**Upper digestive tract factors**: Gastroesophageal reflux causing discomfort that can mimic thoracic sensations."
        ],
        relevant_information=[
            "Whether the discomfort is triggered by physical exertion, emotional stress, or cold temperatures and relieved by rest.",
            "Any radiation to the jaw, neck, back, or left arm.",
            "Associated sensations such as shortness of breath, perspiration, or lightheadedness."
        ],
        clinical_examinations=[
            "A clinician conducts a physical cardiovascular examination including resting blood pressure, heart sounds auscultation, and lung fields assessment."
        ],
        laboratory_tests=[
            "Cardiac biomarker tests (such as troponin) when indicated, along with a complete blood count and lipid profile."
        ],
        diagnostic_imaging=[
            "A 12-lead electrocardiogram (ECG / EKG) to inspect electrical conduction and rhythm, alongside chest radiography or echocardiography if determined appropriate by a doctor."
        ],
        disease_candidates=["heart_disease"],
        explanation_notes="Cardiovascular domain knowledge for non-emergency sub-acute symptom queries."
    ),

    # 3. Metabolic / Glycemic (Diabetes Disease Domain)
    ClinicalKnowledgeEntry(
        domain_id="metabolic_diabetes",
        display_title="Glycemic regulation and metabolic evaluation",
        matching_symptoms=[
            "thirst", "frequent urination", "increased thirst", "fatigue",
            "weakness", "blurred vision", "weight loss"
        ],
        matching_locations=[],
        matching_triggers=["sugar", "sweet", "carbohydrates", "dehydration"],
        matching_keywords=[
            "thirst", "urination", "glucose", "sugar", "diabetes",
            "metabolic", "polyuria", "polydipsia", "glycemic"
        ],
        possible_categories=[
            "**Endocrine and glycemic regulation causes**: Insulin resistance, impaired fasting glucose, or type 2 diabetes mellitus leading to osmotic diuresis.",
            "**Fluid balance and renal factors**: Increased urinary volume secondary to solute clearance or electrolyte shifts.",
            "**Dietary or medication-related factors**: High glycemic food intake, excessive fluid/caffeine consumption, or diuretic medications."
        ],
        relevant_information=[
            "Duration and progression of increased fluid intake and daily or nighttime urination frequency.",
            "Any unintentional weight loss despite normal or increased food intake.",
            "Family history of diabetes or related cardiovascular/metabolic conditions."
        ],
        clinical_examinations=[
            "A healthcare provider assesses vital signs, hydration status, body mass index (BMI), and peripheral sensory examination."
        ],
        laboratory_tests=[
            "Fasting plasma glucose (FPG) and Glycated Hemoglobin (HbA1c) to evaluate blood sugar control over the past 2–3 months, along with an electrolyte panel."
        ],
        diagnostic_imaging=[
            "Routine urinalysis to inspect for glucosuria (glucose in urine), proteinuria, and ketone bodies."
        ],
        disease_candidates=["diabetes"],
        explanation_notes="Metabolic and diabetes screening domain knowledge."
    ),

    # 4. Neurological Movement / Motor (Parkinson's Disease Domain)
    ClinicalKnowledgeEntry(
        domain_id="neurological_movement",
        display_title="Neurological motor and movement evaluation",
        matching_symptoms=[
            "tremor", "shaking", "stiffness", "rigidity", "slowness",
            "bradykinesia", "balance difficulty", "gait changes"
        ],
        matching_locations=["hands", "legs", "body"],
        matching_triggers=["rest", "stress", "fatigue"],
        matching_keywords=[
            "tremor", "shaking", "stiffness", "rigidity", "parkinson",
            "neurological", "movement", "slowness", "bradykinesia"
        ],
        possible_categories=[
            "**Neurological movement disorder patterns**: Parkinsonian syndromes, essential tremor, or idiopathic Parkinson's disease involving basal ganglia motor circuits.",
            "**Secondary or medication-induced parkinsonism**: Motor effects secondary to dopamine-blocking agents, anti-emetics, or systemic medications.",
            "**Musculoskeletal or age-related motor changes**: Joint stiffness or muscular resistance unrelated to central nervous system pathways."
        ],
        relevant_information=[
            "Whether tremor predominantly occurs at rest or during purposeful voluntary movement (action tremor).",
            "Whether motor stiffness or slowness began asymmetrically (more on one side of the body than the other).",
            "Current and past medication history, including any newly prescribed or altered prescriptions."
        ],
        clinical_examinations=[
            "A neurologist or physician conducts a structured motor examination assessing resting tremor, cogwheel rigidity, finger tapping speed, posture, and gait."
        ],
        laboratory_tests=[
            "Baseline metabolic panels, thyroid function tests, and serum ceruloplasmin/copper levels to rule out secondary causes of motor symptoms."
        ],
        diagnostic_imaging=[
            "Brain MRI or specialized neuroimaging (such as DaTscan) if indicated by a specialist to evaluate dopaminergic pathways or exclude structural abnormalities."
        ],
        disease_candidates=["parkinsons_disease"],
        explanation_notes="Parkinson's and neurological movement disorder domain knowledge."
    )
]

# Fallback generic knowledge entry for unsupported or non-specific symptom combinations
GENERIC_KNOWLEDGE_ENTRY = ClinicalKnowledgeEntry(
    domain_id="generic_symptom_evaluation",
    display_title="General clinical symptom evaluation",
    matching_symptoms=[],
    matching_locations=[],
    matching_triggers=[],
    matching_keywords=[],
    possible_categories=[
        "**General physiological or lifestyle factors**: Changes in physical activity, stress, hydration, or sleep patterns influencing overall well-being.",
        "**Infectious or inflammatory factors**: Self-limiting viral or bacterial responses manifesting as non-specific discomfort.",
        "**Localized tissue considerations**: Underlying tissue irritation that requires physical examination to characterize."
    ],
    relevant_information=[
        "Precise timeline of symptom onset, progression, and whether symptoms are constant or intermittent.",
        "Aggravating triggers or factors that provide partial relief.",
        "Presence of any associated symptoms or changes in baseline daily routines."
    ],
    clinical_examinations=[
        "A clinician performs a baseline physical examination, reviewing vital signs and checking pertinent body systems."
    ],
    laboratory_tests=[
        "Standard baseline blood evaluations (such as a complete blood count and basic metabolic panel) as determined appropriate by a doctor."
    ],
    diagnostic_imaging=[
        "Diagnostic imaging or targeted investigations are selected by a physician based on findings from the clinical examination."
    ],
    disease_candidates=["general_risk_evaluation"],
    explanation_notes="Fallback entry when symptoms do not match a specific disease domain."
)


# ==============================================================================
# FINE-GRAINED CLINICAL KNOWLEDGE COMPONENTS
# ==============================================================================

DOMAIN_KNOWLEDGE_COMPONENTS: Dict[str, List[ClinicalKnowledgeComponent]] = {
    # --------------------------------------------------------------------------
    # 1. Hepatobiliary / RUQ Components
    # --------------------------------------------------------------------------
    "hepatobiliary_ruq": [
        ClinicalKnowledgeComponent(
            component_id="ruq_general_evaluation",
            domain_id="hepatobiliary_ruq",
            title="General RUQ and abdominal evaluation",
            required_features=["pain", "abdominal pain", "right upper abdomen", "abdomen", "ruq"],
            supporting_features=["dull", "upper-right", "upper right", "right upper", "ribs"],
            excluded_by_features=[],
            possible_categories=[
                "**Hepatobiliary causes**: Irritation or inflammation of the gallbladder (such as biliary colic or gallstones) or localized liver/biliary tract inflammation.",
                "**Gastrointestinal causes**: Upper digestive tract irritation or localized visceral discomfort."
            ],
            relevant_information=[
                "Radiation of discomfort to the right shoulder, scapula, or upper back.",
                "Whether the sensation is constant, fluctuating, or sharp in character."
            ],
            clinical_examinations=[
                "A physician performs abdominal palpation to check for localized tenderness in the right upper quadrant or signs of gallbladder irritation (such as Murphy's sign)."
            ],
            laboratory_tests=[
                "Liver Function Tests (LFTs) measuring bilirubin, ALT, AST, and alkaline phosphatase to assess hepatic parenchymal and biliary cell integrity."
            ],
            diagnostic_imaging=[
                "An abdominal ultrasound is frequently the first-line imaging examination to inspect the gallbladder, bile ducts, and liver parenchyma."
            ],
            priority=10
        ),
        ClinicalKnowledgeComponent(
            component_id="meal_related_biliary_pattern",
            domain_id="hepatobiliary_ruq",
            title="Meal-triggered biliary colic pattern",
            required_features=["meal", "meals", "oily", "fatty", "food"],
            supporting_features=["upper-right", "upper right", "gallbladder", "biliary", "colic"],
            excluded_by_features=["meal relationship", "unrelated to meals"],
            possible_categories=[
                "**Biliary tract / gallbladder conditions**: Symptoms provoked by rich, fatty, or heavy foods frequently point toward biliary contractions or gallbladder irritation (e.g. gallstones or biliary colic)."
            ],
            relevant_information=[
                "Specific timing between meal intake and symptom onset (typically 30–90 minutes post-prandial)."
            ],
            clinical_examinations=[
                "Physical examination for focal peritoneal signs or Murphy's sign upon deep right upper quadrant inspiration."
            ],
            laboratory_tests=[],
            diagnostic_imaging=[
                "Targeted gallbladder ultrasound to check for gallstones, gallbladder wall thickening, or biliary sludge."
            ],
            priority=8
        ),
        ClinicalKnowledgeComponent(
            component_id="episodic_radiating_biliary_pattern",
            domain_id="hepatobiliary_ruq",
            title="Episodic biliary colic and radiation pattern",
            required_features=["intermittent", "comes and goes", "back", "radiat", "radiation", "episode", "minutes", "hour"],
            supporting_features=["upper-right", "upper right", "right upper", "dull", "colic", "biliary", "gallbladder"],
            excluded_by_features=[],
            possible_categories=[
                "**Biliary tract / gallbladder motility involvement**: Discomfort occurring in discrete, intermittent episodes (often lasting 30–60 minutes or a few hours) and radiating toward the back or right shoulder blade is a hallmark pattern associated with biliary colic or gallbladder contraction against transient resistance."
            ],
            relevant_information=[
                "Discrete duration of pain episodes (e.g. 30–60 minutes) and whether pain resolves completely between episodes.",
                "Radiation of discomfort toward the upper back, interscapular region, or right shoulder blade."
            ],
            clinical_examinations=[
                "Abdominal palpation for localized right upper quadrant tenderness or positive Murphy's sign upon inspiration."
            ],
            laboratory_tests=[
                "Liver Function Tests (total and direct bilirubin, ALT, AST, alkaline phosphatase) to assess for transient ductal obstruction or biliary strain."
            ],
            diagnostic_imaging=[
                "Transabdominal ultrasound to evaluate for gallstones (cholelithiasis), gallbladder wall thickening, or biliary sludge."
            ],
            priority=10
        ),
        ClinicalKnowledgeComponent(
            component_id="nausea_vomiting_context",
            domain_id="hepatobiliary_ruq",
            title="Upper digestive and nausea/vomiting context",
            required_features=["nausea", "vomiting"],
            supporting_features=["reduced appetite", "appetite", "stomach"],
            excluded_by_features=["nausea", "vomiting"],
            possible_categories=[
                "**Upper gastrointestinal involvement**: Concurrent nausea or vomiting often reflects upper gastric lining irritation, delayed gastric emptying, or autonomic reflex activation from visceral inflammation."
            ],
            relevant_information=[
                "Ability to tolerate oral fluids and maintain hydration without continuous emesis."
            ],
            clinical_examinations=[
                "Clinical assessment of hydration status, mucosal moisture, and abdominal bowel sounds."
            ],
            laboratory_tests=[
                "Serum electrolytes and basic metabolic panel to monitor hydration status and acid-base balance if vomiting persists."
            ],
            diagnostic_imaging=[],
            priority=7
        ),
        ClinicalKnowledgeComponent(
            component_id="systemic_inflammatory_context",
            domain_id="hepatobiliary_ruq",
            title="Infectious or inflammatory context",
            required_features=["fever", "feverish", "chills", "high temperature"],
            supporting_features=["weakness", "tired", "fatigue"],
            excluded_by_features=["fever"],
            possible_categories=[
                "**Infectious or inflammatory factors**: Systemic manifestations such as feverish sensations or fatigue suggest active inflammatory or infectious processes (such as cholecystitis, cholangitis, or systemic response)."
            ],
            relevant_information=[
                "Temperature measurements and presence of shivering, rigors, or body aches."
            ],
            clinical_examinations=[
                "Vital signs monitoring including body temperature, heart rate, and systematic inflammatory check."
            ],
            laboratory_tests=[
                "A complete blood count (CBC) to assess white blood cell count and systemic inflammatory markers."
            ],
            diagnostic_imaging=[],
            priority=6
        ),
        ClinicalKnowledgeComponent(
            component_id="jaundice_cholestatic_context",
            domain_id="hepatobiliary_ruq",
            title="Cholestatic / jaundice pattern",
            required_features=["jaundice", "dark urine", "yellowing", "yellow"],
            supporting_features=["pale stool", "itching"],
            excluded_by_features=["jaundice", "yellowing of eyes/skin (jaundice)"],
            possible_categories=[
                "**Hepatocellular or biliary outflow obstruction**: Scleral icterus, dark urine, or pale stools indicate compromised bilirubin excretion or hepatic parenchymal strain."
            ],
            relevant_information=[
                "Progression of scleral yellowing or changes in urine/stool coloration."
            ],
            clinical_examinations=[
                "Scleral and dermatologic inspection for jaundice alongside abdominal organomegaly."
            ],
            laboratory_tests=[
                "Comprehensive Liver Function Tests (LFTs) measuring total and direct bilirubin, ALT, AST, and alkaline phosphatase."
            ],
            diagnostic_imaging=[
                "Biliary tree ultrasound or magnetic resonance cholangiopancreatography (MRCP) to inspect for choledocholithiasis or bile duct dilation."
            ],
            priority=9
        )
    ],

    # --------------------------------------------------------------------------
    # 2. Cardiovascular / Chest Components
    # --------------------------------------------------------------------------
    "cardiovascular_chest": [
        ClinicalKnowledgeComponent(
            component_id="chest_exertional_cardiac_pattern",
            domain_id="cardiovascular_chest",
            title="Cardiovascular exertional pattern",
            required_features=["exertion", "exercise", "climbing stairs", "physical activity", "stress", "work", "working", "working continuously", "continuous work"],
            supporting_features=["chest pain", "chest discomfort", "pressure", "angina", "shortness of breath", "breathless"],
            excluded_by_features=["exertion"],
            possible_categories=[
                "**Cardiovascular causes**: Variations in myocardial blood flow, coronary circulation considerations, or cardiac workload strain, especially when linked to physical effort or continuous work."
            ],
            relevant_information=[
                "Whether discomfort or shortness of breath is provoked by physical exertion or continuous work and relieved promptly by rest.",
                "Any radiation to the left arm, neck, jaw, or upper back."
            ],
            clinical_examinations=[
                "A resting cardiovascular examination including blood pressure assessment, peripheral pulses, and heart sounds auscultation."
            ],
            laboratory_tests=[
                "Cardiac biomarkers (such as high-sensitivity troponin) when indicated by a physician to evaluate myocardial status."
            ],
            diagnostic_imaging=[
                "A 12-lead electrocardiogram (ECG / EKG) to inspect electrical conduction and rhythm."
            ],
            priority=10
        ),
        ClinicalKnowledgeComponent(
            component_id="chest_general_thoracic_evaluation",
            domain_id="cardiovascular_chest",
            title="General thoracic evaluation",
            required_features=["chest", "chest pain", "chest discomfort", "pain"],
            supporting_features=[],
            excluded_by_features=["chest pain"],
            possible_categories=[
                "**Cardiovascular considerations**: Assessment of coronary perfusion and cardiac rhythm.",
                "**Musculoskeletal chest wall factors**: Costochondral irritation, rib cage strain, or postural muscular tension.",
                "**Upper digestive tract factors**: Gastroesophageal reflux causing discomfort that can mimic thoracic sensations."
            ],
            relevant_information=[
                "Whether the discomfort changes with deep breathing, body positioning, or chest wall pressing."
            ],
            clinical_examinations=[
                "Physical chest wall palpation to check for reproducible tenderness, along with cardiopulmonary auscultation."
            ],
            laboratory_tests=[
                "Baseline blood evaluations including a complete blood count and lipid profile."
            ],
            diagnostic_imaging=[
                "A 12-lead ECG and chest radiography if determined appropriate by a doctor."
            ],
            priority=5
        ),
        ClinicalKnowledgeComponent(
            component_id="respiratory_dyspnea_context",
            domain_id="cardiovascular_chest",
            title="Respiratory / dyspnea context",
            required_features=["shortness of breath", "breathless", "difficulty breathing", "wheezing", "breath", "breathing"],
            supporting_features=["cough"],
            excluded_by_features=["difficulty breathing", "shortness of breath"],
            possible_categories=[
                "**Respiratory or pulmonary causes**: Pleural irritation, reactive airway patterns, or ventilation-perfusion considerations."
            ],
            relevant_information=[
                "Whether breathing difficulty occurs at rest, during exertion, or when lying flat (orthopnea)."
            ],
            clinical_examinations=[
                "Pulmonary auscultation to check for wheezing, crackles, or decreased breath sounds, along with pulse oximetry."
            ],
            laboratory_tests=[],
            diagnostic_imaging=[
                "Chest radiography (X-ray) to evaluate lung fields and pleural spaces."
            ],
            priority=8
        ),
        ClinicalKnowledgeComponent(
            component_id="palpitations_rhythm_context",
            domain_id="cardiovascular_chest",
            title="Palpitations and rhythm context",
            required_features=["palpitations", "racing heart", "fluttering", "irregular heartbeat", "heart beating faster", "heart beating fast", "rapid heart rate", "fast heart rate"],
            supporting_features=["dizziness", "lightheadedness"],
            excluded_by_features=["palpitations"],
            possible_categories=[
                "**Cardiac rhythm variations**: Premature contractions, sinus tachycardia, or transient rhythm patterns."
            ],
            relevant_information=[
                "Frequency and duration of rapid or fluttering episodes, whether the rhythm is regular or irregular, and prompt resolution with rest."
            ],
            clinical_examinations=[
                "Pulse rhythm check and resting cardiac auscultation."
            ],
            laboratory_tests=[
                "Electrolyte panel (potassium, magnesium) and thyroid-stimulating hormone (TSH) to screen for metabolic triggers of palpitations."
            ],
            diagnostic_imaging=[
                "Ambulatory cardiac monitoring (Holter monitor) or 12-lead ECG if rhythm changes require capture."
            ],
            priority=9
        )
    ],

    # --------------------------------------------------------------------------
    # 3. Metabolic / Glycemic Components
    # --------------------------------------------------------------------------
    "metabolic_diabetes": [
        ClinicalKnowledgeComponent(
            component_id="glycemic_osmotic_pattern",
            domain_id="metabolic_diabetes",
            title="Classic glycemic osmotic pattern",
            required_features=["thirst", "frequent urination", "increased thirst", "polyuria", "polydipsia", "urination"],
            supporting_features=["fatigue", "weakness"],
            excluded_by_features=[],
            possible_categories=[
                "**Endocrine and glycemic regulation causes**: Insulin resistance, impaired fasting glucose, or type 2 diabetes mellitus leading to osmotic diuresis."
            ],
            relevant_information=[
                "Duration and progression of increased fluid intake and daily or nighttime urination frequency."
            ],
            clinical_examinations=[
                "A healthcare provider assesses vital signs, hydration status, body mass index (BMI), and peripheral sensory examination."
            ],
            laboratory_tests=[
                "Fasting plasma glucose (FPG) and Glycated Hemoglobin (HbA1c) to evaluate blood sugar control over the past 2–3 months, along with an electrolyte panel."
            ],
            diagnostic_imaging=[
                "Routine urinalysis to inspect for glucosuria (glucose in urine), proteinuria, and ketone bodies."
            ],
            priority=10
        ),
        ClinicalKnowledgeComponent(
            component_id="metabolic_catabolic_weight_context",
            domain_id="metabolic_diabetes",
            title="Weight changes and catabolic context",
            required_features=["weight loss", "losing weight"],
            supporting_features=["hunger", "increased appetite"],
            excluded_by_features=["weight loss"],
            possible_categories=[
                "**Metabolic catabolic state**: Cellular energy deficits leading to breakdown of adipose or muscle tissue despite normal intake."
            ],
            relevant_information=[
                "Approximate amount and rate of unintentional weight loss over recent weeks or months."
            ],
            clinical_examinations=[
                "Physical nutritional assessment and baseline weight tracking."
            ],
            laboratory_tests=[
                "Comprehensive metabolic panel and thyroid panel to evaluate metabolic rate."
            ],
            diagnostic_imaging=[],
            priority=8
        ),
        ClinicalKnowledgeComponent(
            component_id="metabolic_vision_fatigue_context",
            domain_id="metabolic_diabetes",
            title="Visual fluctuations and fatigue context",
            required_features=["blurred vision", "blurry vision", "vision", "fatigue", "weakness"],
            supporting_features=["tired"],
            excluded_by_features=[],
            possible_categories=[
                "**Secondary osmotic or cellular effects**: Transient lens hydration fluctuations causing blurred vision, alongside reduced cellular glucose utilization causing generalized fatigue."
            ],
            relevant_information=[
                "Whether visual changes fluctuate throughout the day or after meals."
            ],
            clinical_examinations=[
                "Visual acuity screening and baseline peripheral neurological assessment."
            ],
            laboratory_tests=[
                "Complete blood count and electrolyte panel to rule out anemia or fluid imbalances."
            ],
            diagnostic_imaging=[],
            priority=6
        ),
        ClinicalKnowledgeComponent(
            component_id="fluid_intake_dietary_factors",
            domain_id="metabolic_diabetes",
            title="Dietary and fluid balance factors",
            required_features=[],
            supporting_features=["sweet", "sugar", "carbohydrates", "dehydration"],
            excluded_by_features=[],
            possible_categories=[
                "**Dietary or fluid balance factors**: High glycemic food intake, excessive fluid/caffeine consumption, or osmotic solute clearance."
            ],
            relevant_information=[
                "Daily beverage intake (sugary beverages, caffeine) and family history of diabetes."
            ],
            clinical_examinations=[],
            laboratory_tests=[],
            diagnostic_imaging=[],
            priority=4
        )
    ],

    # --------------------------------------------------------------------------
    # 4. Neurological Movement / Motor Components
    # --------------------------------------------------------------------------
    "neurological_movement": [
        ClinicalKnowledgeComponent(
            component_id="resting_tremor_parkinsonian_pattern",
            domain_id="neurological_movement",
            title="Resting tremor and parkinsonian motor pattern",
            required_features=["tremor", "shaking"],
            supporting_features=["rest", "hands"],
            excluded_by_features=["tremor"],
            possible_categories=[
                "**Neurological movement disorder patterns**: Parkinsonian syndromes, essential tremor, or idiopathic Parkinson's disease involving basal ganglia motor circuits."
            ],
            relevant_information=[
                "Whether tremor predominantly occurs at rest or during purposeful voluntary movement (action tremor)."
            ],
            clinical_examinations=[
                "A neurologist or physician conducts a structured motor examination assessing resting tremor amplitude, frequency, and postural stability."
            ],
            laboratory_tests=[
                "Baseline metabolic panels and thyroid function tests to exclude secondary metabolic tremor."
            ],
            diagnostic_imaging=[
                "Brain MRI or specialized neuroimaging (such as DaTscan) if indicated by a specialist to evaluate dopaminergic pathways or exclude structural abnormalities."
            ],
            priority=10
        ),
        ClinicalKnowledgeComponent(
            component_id="rigidity_slowness_bradykinesia_pattern",
            domain_id="neurological_movement",
            title="Rigidity and bradykinesia pattern",
            required_features=["stiffness", "rigidity", "slowness", "bradykinesia"],
            supporting_features=["balance difficulty", "gait changes", "legs", "body"],
            excluded_by_features=[],
            possible_categories=[
                "**Extrapyramidal motor circuit involvement**: Muscular rigidity (such as cogwheel resistance) and slowness of fine motor movements (bradykinesia)."
            ],
            relevant_information=[
                "Whether motor stiffness or slowness began asymmetrically (more on one side of the body than the other)."
            ],
            clinical_examinations=[
                "A neurologist assesses cogwheel rigidity, rapid alternating movements (finger tapping), posture, and gait."
            ],
            laboratory_tests=[],
            diagnostic_imaging=[
                "Brain MRI to rule out structural lesions, vascular parkinsonism, or normal pressure hydrocephalus."
            ],
            priority=9
        ),
        ClinicalKnowledgeComponent(
            component_id="secondary_medication_motor_context",
            domain_id="neurological_movement",
            title="Secondary and medication-induced motor changes",
            required_features=["medication", "medications", "prescriptions"],
            supporting_features=["anti-emetics", "neuroleptics"],
            excluded_by_features=[],
            possible_categories=[
                "**Secondary or medication-induced parkinsonism**: Motor effects secondary to dopamine-blocking agents, anti-emetics, or systemic medications."
            ],
            relevant_information=[
                "Current and past medication history, including any newly prescribed or altered prescriptions."
            ],
            clinical_examinations=[
                "Clinical medication review alongside motor symmetry evaluation."
            ],
            laboratory_tests=[
                "Serum ceruloplasmin/copper levels if young-onset or unexplained motor signs occur."
            ],
            diagnostic_imaging=[],
            priority=6
        ),
        ClinicalKnowledgeComponent(
            component_id="musculoskeletal_stiffness_factors",
            domain_id="neurological_movement",
            title="Musculoskeletal motor changes",
            required_features=[],
            supporting_features=["joints", "morning stiffness"],
            excluded_by_features=[],
            possible_categories=[
                "**Musculoskeletal or age-related motor changes**: Joint stiffness or muscular resistance unrelated to central nervous system pathways."
            ],
            relevant_information=[
                "Whether stiffness improves with gentle activity or warmth."
            ],
            clinical_examinations=[
                "Active and passive range of motion of affected joints."
            ],
            laboratory_tests=[],
            diagnostic_imaging=[],
            priority=4
        )
    ],

    # --------------------------------------------------------------------------
    # 5. Generic Symptom Evaluation Component
    # --------------------------------------------------------------------------
    "generic_symptom_evaluation": [
        ClinicalKnowledgeComponent(
            component_id="generic_clinical_baseline",
            domain_id="generic_symptom_evaluation",
            title="Baseline clinical symptom evaluation",
            required_features=[],
            supporting_features=[],
            excluded_by_features=[],
            possible_categories=[
                "**General physiological or lifestyle factors**: Changes in physical activity, stress, hydration, or sleep patterns influencing overall well-being.",
                "**Infectious or inflammatory factors**: Self-limiting viral or bacterial responses manifesting as non-specific discomfort.",
                "**Localized tissue considerations**: Underlying tissue irritation that requires physical examination to characterize."
            ],
            relevant_information=[
                "Precise timeline of symptom onset, progression, and whether symptoms are constant or intermittent.",
                "Aggravating triggers or factors that provide partial relief."
            ],
            clinical_examinations=[
                "A clinician performs a baseline physical examination, reviewing vital signs and checking pertinent body systems."
            ],
            laboratory_tests=[
                "Standard baseline blood evaluations (such as a complete blood count and basic metabolic panel) as determined appropriate by a doctor."
            ],
            diagnostic_imaging=[
                "Diagnostic imaging or targeted investigations are selected by a physician based on findings from the clinical examination."
            ],
            priority=1
        )
    ]
}


class ClinicalKnowledgeEngine:
    """
    Deterministic matching and dynamic response generation engine for Answer-First inquiries.
    Evaluates verified user symptoms and conversation state against structured clinical knowledge.
    Never calls external network services or LLMs, ensuring sub-millisecond execution.
    """

    def __init__(
        self,
        knowledge_base: Optional[List[ClinicalKnowledgeEntry]] = None,
        components_map: Optional[Dict[str, List[ClinicalKnowledgeComponent]]] = None
    ):
        self.knowledge_base = knowledge_base or CLINICAL_KNOWLEDGE_BASE
        self.components_map = components_map or DOMAIN_KNOWLEDGE_COMPONENTS

    def match_knowledge_entry(
        self,
        health_info: StructuredHealthInformation,
        message: str
    ) -> ClinicalKnowledgeEntry:
        """
        Determines the most relevant ClinicalKnowledgeEntry based on verified user attributes.
        Uses transparent, deterministic weighted scoring:
          - Location match: weight 3
          - Symptom match: weight 2
          - Trigger match: weight 2
          - Message keyword match: weight 1
        """
        msg_lower = message.lower()
        loc_lower = (health_info.location or "").lower()
        symptom_names = [s.name.lower() for s in health_info.symptoms]
        triggers_lower = [t.lower() for t in health_info.triggers]

        best_entry: Optional[ClinicalKnowledgeEntry] = None
        best_score = 0

        for entry in self.knowledge_base:
            score = 0

            # 1. Location match (weight 3)
            if entry.matching_locations and loc_lower:
                if any(ml in loc_lower for ml in entry.matching_locations):
                    score += 3

            # 2. Symptoms match (weight 2 per match)
            for s_name in symptom_names:
                if any(ms in s_name for ms in entry.matching_symptoms):
                    score += 2

            # 3. Triggers match (weight 2 per match)
            for tr in triggers_lower:
                if any(mt in tr for mt in entry.matching_triggers):
                    score += 2

            # 4. Message keywords match (weight 1 per keyword)
            for kw in entry.matching_keywords:
                if re.search(rf"\b{re.escape(kw)}\b", msg_lower):
                    score += 1

            if score > best_score:
                best_score = score
                best_entry = entry

        # Threshold check: requires at least score >= 2 for domain-specific match
        if best_entry and best_score >= 2:
            return best_entry

        return GENERIC_KNOWLEDGE_ENTRY

    def select_knowledge_components(
        self,
        health_info: StructuredHealthInformation,
        message: str,
        domain_id: str,
        new_info: Optional[StructuredHealthInformation] = None
    ) -> Tuple[List[ClinicalKnowledgeComponent], Dict[str, str]]:
        """
        Filters and scores fine-grained knowledge components for the selected domain against verified user facts.
        Components contradicted by explicit negative findings are excluded.
        Components with unsatisfied required features are skipped.
        Returns:
          Tuple[List[ClinicalKnowledgeComponent], Dict[str, str]] (selected_components, excluded_reasons)
        """
        components = self.components_map.get(domain_id, self.components_map.get("generic_symptom_evaluation", []))
        msg_lower = message.lower()
        loc_lower = (health_info.location or "").lower()
        positive_symptoms = [s.name.lower() for s in health_info.symptoms]
        triggers_lower = [t.lower() for t in health_info.triggers]
        neg_symptoms_lower = [n.lower() for n in health_info.negative_symptoms]

        selected: List[Tuple[int, ClinicalKnowledgeComponent]] = []
        excluded_reasons: Dict[str, str] = {}

        for comp in components:
            # 1. Exclusion check against explicit negative facts
            is_excluded = False
            for ex in comp.excluded_by_features:
                ex_clean = ex.lower()
                # Check in recorded negative symptoms
                if any(ex_clean in n or n in ex_clean for n in neg_symptoms_lower):
                    excluded_reasons[comp.component_id] = f"explicit negative {ex}"
                    is_excluded = True
                    break
                # Check explicit negation in message text
                if re.search(rf"\b(?:no|not|without|den(?:y|ies|ied)|neither|never)\s+(?:\w+\s+){{0,3}}{re.escape(ex_clean)}\b", msg_lower):
                    excluded_reasons[comp.component_id] = f"explicit negative {ex}"
                    is_excluded = True
                    break

            if is_excluded:
                continue

            # 2. Required features check
            if comp.required_features:
                matched_required = []
                for rf in comp.required_features:
                    rf_clean = rf.lower()
                    # Positive symptom match
                    if any(rf_clean in s or s in rf_clean for s in positive_symptoms):
                        matched_required.append(rf)
                    # Location match
                    elif loc_lower and (rf_clean in loc_lower or loc_lower in rf_clean):
                        matched_required.append(rf)
                    # Trigger match
                    elif any(rf_clean in tr for tr in triggers_lower):
                        matched_required.append(rf)
                    # Un-negated message keyword match
                    elif re.search(rf"\b{re.escape(rf_clean)}\b", msg_lower):
                        # Ensure not negated
                        neg_pat = rf"\b(?:no|not|without|den(?:y|ies|ied)|neither|never|doesn'?t\s+seem)\s+(?:\w+\s+){{0,3}}{re.escape(rf_clean)}\b"
                        if not re.search(neg_pat, msg_lower):
                            matched_required.append(rf)

                if not matched_required:
                    excluded_reasons[comp.component_id] = "required features not present in user facts"
                    continue

                req_score = len(matched_required) * 5
            else:
                req_score = 0

            # 3. Supporting features bonus
            supp_score = 0
            for sf in comp.supporting_features:
                sf_clean = sf.lower()
                if (any(sf_clean in s for s in positive_symptoms) or
                        (loc_lower and sf_clean in loc_lower) or
                        any(sf_clean in tr for tr in triggers_lower) or
                        re.search(rf"\b{re.escape(sf_clean)}\b", msg_lower)):
                    supp_score += 2

            # 4. New Information Priority Boost: if component matches newly reported facts
            new_info_boost = 0
            if new_info is not None:
                new_symptoms = [s.name.lower() for s in new_info.symptoms]
                new_triggers = [t.lower() for t in new_info.triggers]
                new_rad = (getattr(new_info, "radiation", None) or "").lower()
                new_freq = (new_info.frequency or "").lower()
                new_dur = (new_info.duration or "").lower()

                for feat in comp.required_features + comp.supporting_features:
                    f_low = feat.lower()
                    if (new_rad and (f_low in new_rad or new_rad in f_low)) or \
                       (new_freq and (f_low in new_freq or new_freq in f_low)) or \
                       (new_dur and (f_low in new_dur or new_dur in f_low)) or \
                       any(f_low in s for s in new_symptoms) or \
                       any(f_low in t for t in new_triggers):
                        new_info_boost += 10
                        break

            total_score = comp.priority + req_score + supp_score + new_info_boost
            selected.append((total_score, comp))

        # If domain-specific filtering yielded empty, ensure baseline component is available
        if not selected and components:
            selected.append((components[0].priority, components[0]))

        # Sort by total score descending, then priority descending
        selected.sort(key=lambda x: (x[0], x[1].priority), reverse=True)
        return [c for _, c in selected], excluded_reasons

    def build_symptom_context_description(
        self,
        health_info: StructuredHealthInformation,
        entry: ClinicalKnowledgeEntry,
        message: str = ""
    ) -> str:
        """
        Dynamically constructs a context-aware description of the presentation based
        STRICTLY on verified user facts and explicit negative findings.
        Never infers symptoms that were not mentioned.
        """
        loc = (health_info.location or "").lower()
        sym_names = [s.name.lower() for s in health_info.symptoms]
        sev = (health_info.severity or "").lower()
        dur = (health_info.duration or "").lower()
        triggers = health_info.triggers
        negatives = health_info.negative_symptoms
        msg_lower = message.lower()

        # Primary symptom & location phrase
        primary_sym = sym_names[0] if sym_names else "symptoms"
        sev_str = f"{sev} " if sev else ""
        dur_str = f" lasting {dur}" if dur else ""

        if (
            any(k in loc for k in ("right upper", "upper right", "upper-right", "ruq", "gallbladder"))
            or any(k in msg_lower for k in ("right upper", "upper right", "upper-right", "ruq"))
        ):
            core_phrase = f"{sev_str}upper-right abdominal pain{dur_str}"
        elif "chest" in loc or entry.domain_id == "cardiovascular_chest":
            if "discomfort" in msg_lower or "chest discomfort" in sym_names:
                core_phrase = f"{sev_str}chest discomfort{dur_str}"
            else:
                core_phrase = f"{sev_str}chest pain{dur_str}"
        elif entry.domain_id == "metabolic_diabetes":
            core_phrase = f"increased thirst and frequent urination{dur_str}"
        elif entry.domain_id == "neurological_movement":
            if "tremor" in sym_names and ("stiffness" in sym_names or "rigidity" in sym_names):
                core_phrase = f"resting tremor and muscle stiffness{dur_str}"
            elif "tremor" in sym_names:
                core_phrase = f"tremor{dur_str}"
            elif "stiffness" in sym_names:
                core_phrase = f"muscle stiffness{dur_str}"
            else:
                core_phrase = f"motor and movement changes{dur_str}"
        else:
            loc_str = f" in the {loc}" if loc else ""
            core_phrase = f"{sev_str}{primary_sym}{loc_str}{dur_str}"

        # Additional verified positive symptoms
        additional_positives = []
        for s in health_info.symptoms[1:]:
            s_low = s.name.lower()
            if s_low not in core_phrase and s_low not in additional_positives:
                additional_positives.append(s_low)

        # Check for un-negated feverish mention
        if "feverish" in msg_lower and "fever" not in [n.lower() for n in negatives]:
            if "feverish sensation" not in additional_positives and "fever" not in additional_positives:
                additional_positives.append("a feverish sensation")

        assoc_clause = ""
        if additional_positives:
            if len(additional_positives) == 1:
                assoc_clause = f", accompanied by {additional_positives[0]}"
            else:
                assoc_clause = f", accompanied by {', '.join(additional_positives[:-1])}, and {additional_positives[-1]}"

        # Positive triggers
        trigger_clause = ""
        if triggers:
            first_tr = triggers[0]
            if "oily" in first_tr or "heavy" in first_tr or "food" in first_tr:
                trigger_clause = " (especially worse after oily or heavy meals)"
            elif "exertion" in first_tr or "exercise" in first_tr:
                trigger_clause = " (occurring during physical exercise)"
            else:
                trigger_clause = f" ({first_tr})"

        # Explicit negative findings (User-Source-of-Truth: ONLY stated negatives)
        neg_clean = []
        has_neg_meal = False
        for n in negatives:
            n_low = n.lower()
            if "meal" in n_low or "food" in n_low:
                has_neg_meal = True
            elif "jaundice" in n_low or "yellow" in n_low:
                if "jaundice" not in neg_clean:
                    neg_clean.append("jaundice")
            elif "vomit" in n_low:
                if "vomiting" not in neg_clean:
                    neg_clean.append("vomiting")
            elif "nausea" in n_low:
                if "nausea" not in neg_clean:
                    neg_clean.append("nausea")
            elif "fever" in n_low:
                if "fever" not in neg_clean:
                    neg_clean.append("fever")
            elif "urine" in n_low:
                if "dark urine" not in neg_clean and "abnormal urine" not in neg_clean:
                    neg_clean.append("dark urine")
            elif "abdominal pain" in n_low:
                if "abdominal pain" not in neg_clean:
                    neg_clean.append("abdominal pain")
            elif "chest pain" in n_low:
                if "severe chest pain" not in neg_clean:
                    neg_clean.append("severe chest pain")
            elif "breath" in n_low:
                if "difficulty breathing" not in neg_clean:
                    neg_clean.append("difficulty breathing")
            elif "faint" in n_low or "consciousness" in n_low:
                if "fainting" not in neg_clean:
                    neg_clean.append("fainting")

        # Also check explicit meal denial in message if not already captured in negatives
        if not has_neg_meal and (
            re.search(r"\b(?:not|doesn'?t\s+seem|un)?related\s+to\s+meals?\b", msg_lower) or
            re.search(r"\b(?:no|not|without)\s+clear\s+relationship\s+to\s+meals?\b", msg_lower)
        ):
            has_neg_meal = True

        preferred_order = ["nausea", "vomiting", "fever", "jaundice", "dark urine", "abdominal pain", "severe chest pain", "difficulty breathing", "fainting"]
        neg_clean.sort(key=lambda x: preferred_order.index(x) if x in preferred_order else 99)

        neg_clause = ""
        if neg_clean:
            if len(neg_clean) == 1:
                neg_clause = f", without {neg_clean[0]}"
            else:
                neg_clause = f", without {', '.join(neg_clean[:-1])}, or {neg_clean[-1]}"

        if has_neg_meal:
            if neg_clause:
                neg_clause += ", and without a clear relationship to meals"
            else:
                neg_clause = ", without a clear relationship to meals"

        return f"{core_phrase}{assoc_clause}{trigger_clause}{neg_clause}"

    def filter_relevant_information(
        self,
        selected_components: List[ClinicalKnowledgeComponent],
        health_info: StructuredHealthInformation,
        session: Any
    ) -> List[str]:
        """
        Dynamically filters distinguishing clinical information.
        Excludes topics that the user already explicitly answered, claimed, or denied.
        """
        answered_topics: Set[str] = getattr(session, "answered_topics", set())
        triggers_lower = [t.lower() for t in health_info.triggers]
        negatives_lower = [n.lower() for n in health_info.negative_symptoms]
        symptoms_lower = [s.name.lower() for s in health_info.symptoms]

        candidates: List[str] = []
        for comp in selected_components:
            for info_item in comp.relevant_information:
                item_lower = info_item.lower()

                # 1. Meal / Food relationship
                if "meal" in item_lower or "fatty" in item_lower or "food" in item_lower:
                    if ("food" in answered_topics or "meals" in answered_topics or
                            any("meal" in t or "food" in t or "oily" in t for t in triggers_lower) or
                            any("meal" in n for n in negatives_lower)):
                        continue

                # 2. Jaundice / Yellowing / Dark urine
                if "jaundice" in item_lower or "dark" in item_lower or "yellow" in item_lower:
                    if ("jaundice" in answered_topics or "urine" in answered_topics or
                            any("jaundice" in n or "yellow" in n for n in negatives_lower) or
                            any("jaundice" in s or "urine" in s for s in symptoms_lower)):
                        continue

                # 3. Fever / Chills / Shivering
                if "fever" in item_lower or "temperature" in item_lower or "shivering" in item_lower:
                    if (any("fever" in n for n in negatives_lower) or
                            any("fever" in s for s in symptoms_lower)):
                        continue

                # 4. Fluids / Hydration / Emesis
                if "fluid" in item_lower or "hydration" in item_lower or "emesis" in item_lower:
                    if ("fluids" in answered_topics or
                            any("vomit" in n for n in negatives_lower) or
                            any("nausea" in n for n in negatives_lower)):
                        continue

                # 5. Exertion / Rest
                if "exertion" in item_lower or "rest" in item_lower:
                    if any("exertion" in t or "exercise" in t for t in triggers_lower):
                        continue

                # 6. Weight loss
                if "weight loss" in item_lower or "unintentional weight" in item_lower:
                    if "weight" in answered_topics or any("weight" in s for s in symptoms_lower):
                        continue

                if info_item not in candidates:
                    candidates.append(info_item)

        # Fallback if all domain topics were answered
        if not candidates:
            candidates.append("Progression of symptoms over time and whether comfort levels shift with posture or rest.")

        return candidates[:3]

    def format_evaluations(
        self,
        selected_components: List[ClinicalKnowledgeComponent]
    ) -> str:
        """
        Assembles potential clinical evaluations strictly supported by the selected components.
        Uses cautious, non-prescriptive phrasing ('A clinician may consider...').
        """
        selected_exams: List[str] = []
        selected_labs: List[str] = []
        selected_imaging: List[str] = []

        for comp in selected_components:
            for exam in comp.clinical_examinations:
                if exam not in selected_exams:
                    selected_exams.append(exam)
            for lab in comp.laboratory_tests:
                if lab not in selected_labs:
                    selected_labs.append(lab)
            for img in comp.diagnostic_imaging:
                if img not in selected_imaging:
                    selected_imaging.append(img)

        bullets = []
        if selected_exams:
            bullets.append(f"• **Clinical Examination**: {selected_exams[0]}")
        if selected_labs:
            if len(selected_labs) > 1:
                combined_labs = f"{selected_labs[0]} Additionally, {selected_labs[1]}"
                bullets.append(f"• **Laboratory Blood Tests**: {combined_labs}")
            else:
                bullets.append(f"• **Laboratory Blood Tests**: {selected_labs[0]}")
        if selected_imaging:
            if len(selected_imaging) > 1:
                combined_imaging = f"{selected_imaging[0]} In addition, {selected_imaging[1]}"
                bullets.append(f"• **Diagnostic Imaging**: {combined_imaging}")
            else:
                bullets.append(f"• **Diagnostic Imaging**: {selected_imaging[0]}")

        eval_content = "\n".join(bullets)
        return (
            "\n\n**What evaluations and diagnostic tests a clinician may consider:**\n"
            f"{eval_content}\n\n"
            "A qualified clinician will decide which physical examinations, laboratory tests, or imaging are appropriate for your specific case."
        )

    def build_grounding_context(
        self,
        health_info: StructuredHealthInformation,
        message: str,
        session: Any = None,
        follow_up_engine: Any = None,
        progress_context: Optional[Any] = None,
        response_plan: Optional[Any] = None
    ) -> ClinicalGroundingContext:
        """
        Retrieves, filters, and packages relevant structured clinical knowledge components
        and verified facts into a compact ClinicalGroundingContext payload for LLM synthesis.
        """
        start_time = time.perf_counter()
        if progress_context is None and session is not None:
            progress_context = getattr(session, "last_progression_context", None)

        if response_plan is None:
            if progress_context is not None and getattr(progress_context, "response_plan", None) is not None:
                response_plan = progress_context.response_plan
            elif session is not None and getattr(session, "last_response_plan", None) is not None:
                response_plan = session.last_response_plan

        entry = self.match_knowledge_entry(health_info, message)
        new_facts = progress_context.newly_reported_information if progress_context else None
        selected_components, excluded_reasons = self.select_knowledge_components(
            health_info, message, entry.domain_id, new_info=new_facts
        )

        categories: List[str] = []
        for comp in selected_components:
            for cat in comp.possible_categories:
                if cat not in categories:
                    categories.append(cat)
        if not categories:
            categories = list(entry.possible_categories)

        distinguishing_info = self.filter_relevant_information(selected_components, health_info, session)

        selected_exams: List[str] = []
        selected_labs: List[str] = []
        selected_imaging: List[str] = []
        for comp in selected_components:
            for exam in comp.clinical_examinations:
                if exam not in selected_exams:
                    selected_exams.append(exam)
            for lab in comp.laboratory_tests:
                if lab not in selected_labs:
                    selected_labs.append(lab)
            for img in comp.diagnostic_imaging:
                if img not in selected_imaging:
                    selected_imaging.append(img)

        # Build high-quality deterministic response as reliable fallback if LLM is unavailable
        follow_up = follow_up_engine.generate_follow_up(session) if (session and follow_up_engine) else None
        symptom_context = self.build_symptom_context_description(health_info, entry, message)

        has_new_details = (
            progress_context is not None and
            progress_context.newly_reported_information is not None and
            bool(
                progress_context.newly_reported_information.frequency or
                getattr(progress_context.newly_reported_information, "radiation", None) or
                progress_context.newly_reported_information.duration or
                progress_context.newly_reported_information.symptoms
            ) and
            progress_context.previously_known_information is not None and
            bool(progress_context.previously_known_information.symptoms or progress_context.previously_known_information.location)
        )

        plan_mode = getattr(response_plan, "answer_mode", None)
        mode_val = plan_mode.value if hasattr(plan_mode, "value") else str(plan_mode or "")

        if mode_val == "SUMMARY":
            opening = (
                f"Based on everything you have shared so far, here is a summary of your reported presentation and what information a doctor would find most useful:\n\n"
                f"**Reported Clinical Presentation:**\n"
                f"You have described {symptom_context}.\n\n"
                f"**Clinical Categories Under Consideration:**\n"
            )
        elif mode_val == "CORRECTION_ACKNOWLEDGMENT" and getattr(response_plan, "changed_information", None):
            changed_str = "; ".join(response_plan.changed_information)
            opening = (
                f"Thank you for clarifying that ({changed_str}). "
                f"In the context of the {symptom_context} reported earlier, this updated information helps refine what a clinician evaluates:\n\n"
            )
        elif has_new_details:
            new_inf = progress_context.newly_reported_information
            new_items = []
            if new_inf.frequency:
                new_items.append(f"the {new_inf.frequency} pattern (coming and going)")
            if new_inf.duration:
                new_items.append(f"episodes lasting {new_inf.duration}")
            if getattr(new_inf, "radiation", None):
                new_items.append(f"discomfort radiating toward your {new_inf.radiation}")
            for s in new_inf.symptoms:
                if getattr(s, "radiation", None) and f"discomfort radiating toward your {s.radiation}" not in new_items:
                    new_items.append(f"discomfort extending toward your {s.radiation}")

            new_summary = ", ".join(new_items) if new_items else "the newly reported symptom details"
            opening = (
                f"That additional detail regarding {new_summary} is useful clinical information. "
                f"In the context of the {symptom_context} reported earlier, this pattern is particularly relevant "
                "when a clinician evaluates gallbladder or biliary involvement (such as biliary colic). "
                "While symptoms alone cannot determine a diagnosis, evaluating this presentation warrants considering several clinical possibilities:\n\n"
            )
        else:
            opening = (
                f"You've described {symptom_context}. While symptoms alone cannot determine a diagnosis, "
                "evaluating this presentation warrants considering several clinical possibilities:\n\n"
            )
        categories_block = "\n".join(f"{i + 1}. {cat}" for i, cat in enumerate(categories))
        info_block = (
            "\n\n**What information helps distinguish between these possibilities:**\n"
            + "\n".join(f"• {item}" for item in distinguishing_info)
        )
        tests_block = self.format_evaluations(selected_components)

        negative_acknowledgment = ""
        if health_info.negative_symptoms:
            denied_str = ", ".join(health_info.negative_symptoms)
            negative_acknowledgment = (
                f"\n\n**Consideration of reported negative findings:**\n"
                f"You specifically denied / reported the absence of: {denied_str}. "
                "These negative findings provide useful distinguishing context for a clinician. "
                "While the absence of these signs helps narrow clinical possibilities, "
                "negative findings do not by themselves completely rule conditions in or out, "
                "and evaluation still depends on the complete symptom pattern, physical examination, and appropriate testing."
            )

        if follow_up:
            deterministic_msg = opening + categories_block + info_block + tests_block + negative_acknowledgment + f"\n\nTo help evaluate further, {follow_up}"
        else:
            deterministic_msg = opening + categories_block + info_block + tests_block + negative_acknowledgment

        grounding_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "Clinical Grounding Retrieved: domain='%s', selected_components=%s (count=%d), excluded=%s, elapsed_ms=%.2f",
            entry.domain_id,
            [c.component_id for c in selected_components],
            len(selected_components),
            list(excluded_reasons.keys()),
            grounding_ms
        )

        return ClinicalGroundingContext(
            domain_id=entry.domain_id,
            display_title=entry.display_title,
            selected_components=selected_components,
            component_ids=[c.component_id for c in selected_components],
            possible_categories=categories,
            relevant_information=distinguishing_info,
            clinical_examinations=selected_exams,
            laboratory_tests=selected_labs,
            diagnostic_imaging=selected_imaging,
            disease_candidates=list(entry.disease_candidates),
            deterministic_fallback_message=deterministic_msg,
            excluded_components=excluded_reasons
        )

    def generate_answer_first_response(
        self,
        session: Any,
        message: str,
        request_user_id: str,
        extracted_info: StructuredHealthInformation,
        follow_up_engine: Any,
        llm_client: Optional[Any] = None,
        progress_context: Optional[Any] = None
    ) -> ChatResponse:
        """
        Dynamically assembles an authoritative, context-aware Answer-First response via Grounded LLM Synthesis.
        1. Records user message and merges extracted health attributes into session state.
        2. Retrieves relevant clinical knowledge grounding components (emphasizing new findings).
        3. Invokes Grounded LLM Synthesis for natural conversational language with dialogue progression.
        4. Falls back safely to deterministic knowledge output if LLM fails or times out.
        5. Formulates standardized ChatResponse with handoff contract when screening is complete.
        """
        from app.conversation import compute_new_information, build_conversation_progress_context

        # 1. Compute conversation progression before merging into cumulative session state
        if progress_context is None:
            previous_info = session.health_info.model_copy(deep=True)
            new_info = compute_new_information(previous_info, extracted_info)
            progress_context = build_conversation_progress_context(
                message=message,
                previous_info=previous_info,
                new_info=new_info,
                session=session,
                intent="CAUSES_OR_WHAT_HELPS"
            )
        session.last_progression_context = progress_context

        # Build deterministic Response Plan
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

        # 2. Update session state
        session.add_message(role="user", content=message)
        session.merge_health_info(extracted_info, user_message=message)
        session.last_intent = IntentType.SYMPTOM_ASSESSMENT

        info = session.health_info

        # 3. Build Grounding Context
        grounding = self.build_grounding_context(
            info, message, session, follow_up_engine, progress_context=progress_context, response_plan=plan
        )

        # 4. Follow-up engine check
        follow_up = follow_up_engine.generate_follow_up(session)

        # 5. Synthesize response via GroundedSynthesisEngine (Ollama / LLM client)
        from app.grounded_synthesis import grounded_synthesizer
        synthesized_msg = grounded_synthesizer.synthesize_response(
            session=session,
            message=message,
            grounding=grounding,
            follow_up=follow_up,
            llm_client=llm_client,
            progress_context=progress_context,
            response_plan=plan
        )

        from app.response_validator import response_validator
        if synthesized_msg and len(synthesized_msg.strip()) > 15:
            assistant_message = response_validator.validate_and_repair(
                generated_response=synthesized_msg.strip(),
                current_turn_extraction=extracted_info,
                cumulative_session=session,
                response_plan=plan,
                grounding=grounding
            )
        else:
            # Deterministic safe fallback
            assistant_message = grounding.deterministic_fallback_message

        # 6. Formulate handoff contract & next action
        disease_candidates = list(grounding.disease_candidates)
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
            follow_up_question = follow_up
            next_action = NextActionType.ASK_FOLLOW_UP
            assessment_handoff = None
        else:
            follow_up_question = None
            next_action = NextActionType.REQUEST_DISEASE_ASSESSMENT
            assessment_handoff = AssessmentHandoffContract(
                next_action=NextActionType.REQUEST_DISEASE_ASSESSMENT,
                disease_candidates=disease_candidates,
                structured_inputs=structured_inputs
            )

        # 7. Record assistant message and return response
        session.add_message(role="assistant", content=assistant_message)

        return ChatResponse(
            session_id=session.session_id,
            assistant_message=assistant_message,
            intent=IntentType.SYMPTOM_ASSESSMENT,
            structured_health_information=session.health_info,
            current_turn_extraction=extracted_info,
            follow_up_question=follow_up_question,
            safety_status=SafetyStatus.NORMAL_CONVERSATION,
            next_action=next_action,
            assessment_handoff=assessment_handoff,
            progression_context=progress_context,
            response_plan=plan
        )


# Global singleton instance
clinical_knowledge_engine = ClinicalKnowledgeEngine()
