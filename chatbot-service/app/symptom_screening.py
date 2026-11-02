import logging
from typing import Optional, List
from app.conversation import ConversationState

logger = logging.getLogger(__name__)


def generate_symptom_screening_summary(session: ConversationState) -> str:
    """
    Generates a comprehensive, safe, screening-oriented summary once sufficient
    symptom context has been collected.

    Strict clinical guidelines:
    - Never asserts a medical diagnosis (no 'You have...', 'This proves...').
    - Never fabricates or provides unvalidated numerical disease probabilities.
    - Does NOT force or require laboratory value collection.
    - Explicitly states that symptoms can stem from multiple causes.
    - Recommends appropriate in-person healthcare professional evaluation.
    - Identifies warning signs and emergency instructions.
    - Informs the user that if they possess laboratory reports, they may share them optionally.
    """
    info = session.health_info
    loc = (info.location or "").lower()
    sym_names = [s.name.lower() for s in info.symptoms]

    # 1. Summary of reported findings
    summary_items: List[str] = []
    if info.symptoms:
        s_names = ", ".join([s.name for s in info.symptoms])
        summary_items.append(f"• Reported Symptoms: {s_names}")
    if info.location:
        summary_items.append(f"• Location: {info.location}")
    if info.duration:
        summary_items.append(f"• Duration: {info.duration}")
    if info.severity:
        summary_items.append(f"• Severity: {info.severity}")
    if info.frequency:
        summary_items.append(f"• Frequency / Pattern: {info.frequency}")
    if info.progression:
        summary_items.append(f"• Progression: {info.progression}")
    if info.triggers:
        summary_items.append(f"• Aggravating Factors / Triggers: {', '.join(info.triggers)}")
    if info.relieving_factors:
        summary_items.append(f"• Relieving Factors: {', '.join(info.relieving_factors)}")
    if info.negative_symptoms:
        summary_items.append(f"• Denied / Negative Symptoms: {', '.join(info.negative_symptoms)}")
    if info.previous_tests:
        summary_items.append(f"• Previous Tests: {', '.join(info.previous_tests)}")
    if info.known_conditions:
        summary_items.append(f"• Known Conditions: {', '.join(info.known_conditions)}")
    if info.medications:
        summary_items.append(f"• Medications: {', '.join(info.medications)}")
    if info.age or info.sex:
        demog = []
        if info.age:
            demog.append(f"{info.age} years old")
        if info.sex:
            demog.append(info.sex)
        summary_items.append(f"• Demographics: {', '.join(demog)}")

    summary_block = "\n".join(summary_items) if summary_items else "• Symptoms reported during conversation"

    # 2. Identify clinical anatomical / physiological associations
    is_hepatobiliary = (
        ("right" in loc and ("rib" in loc or "abdomen" in loc or "side" in loc or "quadrant" in loc or "upper" in loc))
        or any(k in sym_names for k in ("jaundice", "yellowing", "dark urine", "pale stool"))
    )
    is_cardiovascular = (
        "chest" in loc or any(k in sym_names for k in ("shortness of breath", "palpitations"))
    )
    is_neurological = (
        any(k in sym_names for k in ("tremor", "stiffness", "rigidity", "slowness"))
    )

    if is_hepatobiliary:
        association_text = (
            "Symptoms localized to the right upper abdomen (below the rib cage) can occur with several conditions. "
            "In clinical practice, physicians commonly evaluate whether such discomfort relates to the gallbladder "
            "(such as biliary colic or gallstones), the liver or biliary system (such as hepatic inflammation or bile duct changes), "
            "or the upper digestive tract (such as gastritis or peptic irritation). "
            "Because symptoms alone cannot confirm a specific cause, further in-person clinical evaluation is appropriate."
        )
        monitoring_text = (
            "Important symptoms to monitor: yellowing of the skin or eyes (jaundice), dark tea-colored urine, "
            "pale clay-colored stools, persistent vomiting, high fever, or chills."
        )
    elif is_cardiovascular:
        association_text = (
            "Chest discomfort and breathing symptoms can involve cardiovascular, respiratory, or musculoskeletal structures. "
            "Because symptoms in the thoracic region can stem from multiple distinct causes ranging from benign musculoskeletal "
            "strain to conditions requiring clinical care, a prompt medical evaluation is essential."
        )
        monitoring_text = (
            "Important symptoms to monitor: discomfort radiating to the arm, jaw, neck, or back, increasing shortness of breath, "
            "cold sweats, dizziness, or lightheadedness."
        )
    elif is_neurological:
        association_text = (
            "Tremor, movement changes, or muscle stiffness can involve neurological motor pathways or musculoskeletal factors. "
            "A structured clinical and neurological examination is necessary to assess the underlying pattern."
        )
        monitoring_text = (
            "Important symptoms to monitor: progression to other limbs, balance or gait instability, sudden weakness, or speech changes."
        )
    else:
        association_text = (
            "These reported symptoms can occur across a variety of localized or general health conditions. "
            "Many common health issues present with overlapping features, so symptoms alone cannot confirm an underlying cause."
        )
        monitoring_text = (
            "Important symptoms to monitor: sudden worsening of intensity, unexplained fever, unmanageable pain, or new accompanying symptoms."
        )

    response = (
        "Thank you for providing your symptom details. Here is a summary of what you shared:\n\n"
        f"{summary_block}\n\n"
        "**Clinical Screening Context:**\n"
        f"{association_text}\n\n"
        "**Important Signs to Monitor:**\n"
        f"{monitoring_text}\n\n"
        "**Recommended Next Steps:**\n"
        "• We recommend scheduling an appointment with a qualified healthcare professional (such as your primary care physician) "
        "for an in-person physical examination and any diagnostic workup (such as ultrasound imaging or blood tests) they deem appropriate.\n"
        "• *Note: You do not need laboratory reports to use this chatbot. However, if you already have or later receive formal laboratory "
        "test results (such as a Liver Function Test panel), you can share those values here anytime to run our evidence-based risk screening model.*\n\n"
        "**Emergency Guidance:**\n"
        "If you experience sudden severe unmanageable pain, high fever, difficulty breathing, fainting, or signs of acute distress, "
        "please seek immediate emergency medical care.\n\n"
        "*Disclaimer: This conversational screening summary is for informational and educational purposes only and is NOT a medical diagnosis. "
        "It does not replace professional clinical evaluation or personalized medical care.*"
    )
    return response
