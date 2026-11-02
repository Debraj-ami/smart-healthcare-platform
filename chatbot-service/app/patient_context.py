import re
import logging
from typing import Dict, Any, List, Tuple, Optional
from app.llm_client import BaseLLMClient, MockLLMClient, LLMResponseError, LLMUnavailableError

logger = logging.getLogger(__name__)

METRIC_ALIASES: Dict[str, List[str]] = {
    "GLUCOSE": [r"\bglucose\b", r"\bblood\s+sugar\b", r"\bsugar\b"],
    "HEART_RATE": [r"\bheart\s+rate\b", r"\bpulse\b", r"\bbpm\b"],
    "BLOOD_PRESSURE_SYSTOLIC": [r"\bbp\b", r"\bblood\s+pressure\b", r"\bsystolic\b"],
    "CHOLESTEROL": [r"\bcholesterol\b", r"\bldl\b", r"\bhdl\b", r"\blipids?\b"],
    "WEIGHT": [r"\bweight\b", r"\bbmi\b"],
    "TOTAL_BILIRUBIN": [r"\bbilirubin\b", r"\bt_bili\b"]
}

PATIENT_CONTEXT_SYSTEM_PROMPT = (
    "You are a clinical assistant explaining verified historical patient health trends.\n"
    "Strict Medical Constraints:\n"
    "1. Use ONLY the supplied verified patient health context.\n"
    "2. NEVER invent, extrapolate, or assume values not present in the context.\n"
    "3. NEVER diagnose any disease or fabricate clinical risk probabilities.\n"
    "4. NEVER claim causal mechanisms (e.g. do not guess why a reading changed).\n"
    "5. Clearly distinguish observed statistical trends from a medical diagnosis.\n"
    "6. Clearly state when specific metrics or historical data are unavailable in the record.\n"
    "7. Always advise consulting a qualified healthcare professional for medical interpretation.\n"
    "8. Personal Baseline vs. Normal Range: The user's baseline and variability reflect their personal historical measurements, NOT a clinical normal range. NEVER refer to a user's personal baseline as a 'normal range', 'clinically normal', or 'within normal limits'. Always describe readings close to baseline as 'within your personal baseline/range' or 'consistent with your personal historical baseline'."
)


def is_patient_context_query(message: str, session: Optional[Any] = None) -> Tuple[bool, Optional[str]]:
    """
    Deterministically detects if user is asking about their personal/historical health context.
    Returns:
        (is_context_query, detected_metric_type)
    """
    text = message.strip()
    if not text:
        return False, None
    text_lower = text.lower()

    # If session has active monitoring context, inherit metric
    active_metric = None
    if session and getattr(session, "active_context", None):
        active_metric = session.active_context.get("metric")

    # 1. Negative Filter: Generic educational inquiries without personal reference
    educational_prefix = r"^\s*(what\s+is|what\s+are|what\s+does\s+.+\s+mean|explain\b|define\b|tell\s+me\s+about\b)"
    if re.search(educational_prefix, text_lower):
        has_personal = bool(re.search(r"\b(my|mine|i|me|am\s+i|this\s+trend|this\s+result|our)\b", text_lower))
        has_measurement_ref = bool(re.search(
            r"\b(reading|readings|measurement|measurements|number|numbers|result|results|trend|trends|baseline|usual|history|past|previous|recent|monitoring)\b",
            text_lower
        ))
        if not (has_personal and has_measurement_ref) and not (active_metric and "trend" in text_lower):
            return False, None

    # 2. Negative Filter: Symptom-first complaints without personal reading/trend request
    symptom_keywords = r"\b(pain|aching|hurts?|sore|discomfort|tender|burning|swelling|cramp|nausea|vomit|fever|fatigue|tired|weakness|exhaustion|breathless|shortness\s+of\s+breath|cough)\b"
    if re.search(symptom_keywords, text_lower):
        explicit_measurement_query = bool(re.search(
            r"\b(my\s+)?(reading|readings|measurement|measurements|trend|trends|baseline|usual\s+level|historical|previous\s+reading|past\s+reading|monitoring)\b",
            text_lower
        ))
        if not explicit_measurement_query:
            return False, None

    # 3. Detect target metric if explicitly requested
    detected_metric: Optional[str] = None
    for metric_name, patterns in METRIC_ALIASES.items():
        if any(re.search(p, text_lower) for p in patterns):
            detected_metric = metric_name
            break

    if not detected_metric and active_metric:
        detected_metric = active_metric

    # 4. Positive indicators for personal health context queries
    context_patterns = [
        # Health monitoring broad queries
        r"\b(what\s+does|how\s+does)\s+my\s+(recent\s+)?(health\s+)?monitoring\s+(show|look|say)\b",
        r"\bwhat\s+does\s+my\s+recent\s+.*monitoring\s+show\b",
        r"\bhow\s+have\s+my\s+(health\s+)?readings\s+been\s+recently\b",
        r"\bwhat\s+has\s+changed\s+in\s+my\s+health\s+measurements\b",
        # Broad personal health inquiries: "How has my health been recently?" / "How is my health doing lately?"
        r"\bhow\s+(has|is)\s+my\s+health\s+(been|doing)\s+(recently|lately)\b",
        r"\bhow\s+am\s+i\s+doing\s+compared\b",
        # Specific metric doing/changes: "How is my glucose doing?", "How has my blood glucose changed?"
        r"\bhow\s+is\s+my\s+(glucose|sugar|blood\s+sugar|heart\s+rate|blood\s+pressure|bp|cholesterol|weight)\s+doing\b",
        r"\bhow\s+has\s+my\s+(glucose|sugar|blood\s+sugar|heart\s+rate|blood\s+pressure|bp|cholesterol|weight)\s+changed\b",
        r"\bis\s+my\s+(glucose|sugar|blood\s+sugar|heart\s+rate|blood\s+pressure|bp|cholesterol|weight)\s+(going\s+up|going\s+down|rising|falling|increasing|decreasing|trending|up\s+or\s+down)\b",
        # Readings & measurements changes: "How have my readings changed?", "Has my glucose changed from my usual level?"
        r"\b(how|what|has|have|did|are)\b.*\b(my|previous|past|recent|usual|baseline)\b.*\b(reading|readings|measurement|measurements|observation|observations|value|values|number|numbers|level|levels|trend|trends|glucose|sugar|blood\s+pressure|heart\s+rate|cholesterol|weight|monitoring)\b.*\b(change|changed|changing|differ|different|doing|trend|trends|compared|progress|lately|recently|usual\s+level|show)\b",
        # Recent changes: "What has changed in my recent measurements?"
        r"\bwhat\s+(has\s+)?changed\s+in\s+my\s+(recent\s+)?(measurements|readings|numbers|results|observations|levels|monitoring)\b",
        # Baseline inquiries: "What is my personal baseline?", "How does my latest glucose compare with my personal baseline?"
        r"\bwhat\s+is\s+my\s+(personal\s+)?baseline\b",
        r"\bhow\s+does\s+my\s+.*compare\s+(with|to)\s+(my\s+)?(personal\s+)?baseline\b",
        r"\bcompared\s+(with|to)\s+(my\s+)?(previous|past|last|baseline|usual|personal\s+baseline)\b",
        r"\b(from|to|vs|versus)\s+(my\s+)?(usual|normal)\s+level\b",
        r"\b(how\s+do\s+my\s+numbers\s+compare\s+to\s+my\s+baseline)\b",
        r"\b(from|to|vs|versus)\s+(my\s+)?baseline\b",
        # Historical queries: "What are my previous readings?", "Show my trends"
        r"\b(what\s+are|show|view|check|tell\s+me)\s+(my\s+)?(previous|past|recent|historical)\s+(readings?|measurements?|records?|results?|trends?|numbers?)\b",
        r"\b(how\s+are|what\s+are)\s+my\s+(trends?|readings?|numbers?)\b",
        r"\bmy\s+(readings?|measurements?|numbers?|levels?)\s+(have\s+)?(changed|differ|look|trending)\b",
        r"\bhas\s+my\s+(glucose|sugar|blood\s+pressure|heart\s+rate|cholesterol|weight)\s+changed\b",
        # Trend meaning inquiry when in monitoring context
        r"\bwhat\s+does\s+this\s+trend\s+mean\b",
        r"\bexplain\s+this\s+trend\b"
    ]

    for p in context_patterns:
        if re.search(p, text_lower):
            return True, detected_metric

    return False, None


def format_patient_context_summary(
    context_data: Dict[str, Any],
    requested_metric: Optional[str] = None
) -> Tuple[bool, str]:
    """
    Converts raw Spring Boot trends, baselines, and observations into a structured, verified summary.
    Returns:
        (has_data, formatted_text_summary)
    """
    trends: List[Dict[str, Any]] = context_data.get("trends", [])
    baselines: List[Dict[str, Any]] = context_data.get("baselines", [])
    observations: List[Dict[str, Any]] = context_data.get("recent_observations", [])

    if not trends and not baselines and not observations:
        return False, "No historical readings or baselines found for this patient."

    # Filter by requested metric if specified
    if requested_metric:
        req_norm = requested_metric.upper()
        matched_trends = [t for t in trends if t.get("metricType", "").upper() == req_norm]
        matched_baselines = [b for b in baselines if b.get("metricType", "").upper() == req_norm]
        matched_obs = [o for o in observations if o.get("metricType", "").upper() == req_norm]

        if matched_trends or matched_baselines or matched_obs:
            trends = matched_trends
            baselines = matched_baselines
            observations = matched_obs
        else:
            return False, f"No historical records found for {requested_metric.title()}."

    sections = [
        f"Verified Patient Health Context (Patient ID: {context_data.get('patient_id', 'N/A')}):"
    ]

    # 1. Trend analysis snapshots
    # Group by latest trend per metric
    latest_trends: Dict[str, Dict[str, Any]] = {}
    for t in trends:
        m_type = t.get("metricType", "UNKNOWN")
        if m_type not in latest_trends:
            latest_trends[m_type] = t

    if latest_trends:
        sections.append("\nTrend Analysis & Baseline Comparisons:")
        for m_type, t in latest_trends.items():
            current_val = t.get("currentValue")
            prev_val = t.get("previousValue")
            baseline_val = t.get("baselineValue")
            variability = t.get("variability")
            dev_score = t.get("deviationScore")
            signed_dev = t.get("deviationFromBaseline")
            abs_dev = t.get("absoluteDeviation")
            change_prev = t.get("valueChangeFromPrevious")
            pct_change = t.get("percentageChangeFromPrevious")
            direction = t.get("trendDirection", "UNKNOWN")
            persistence = t.get("persistence")
            hist = t.get("historicalContext") or {}

            lines = [f"• Metric: {m_type}"]
            if current_val is not None:
                lines.append(f"  - Latest Reading: {current_val}")
            if prev_val is not None:
                chg_str = f"{change_prev:+.2f}" if change_prev is not None else "N/A"
                pct_str = f"{pct_change:+.1f}%" if pct_change is not None else ""
                lines.append(f"  - Previous Reading: {prev_val} (Change: {chg_str} {pct_str})")
            if baseline_val is not None:
                var_str = f" ±{variability:.2f}" if variability is not None else ""
                lines.append(f"  - Personal Baseline: {baseline_val}{var_str} (personal historical baseline, not a clinical normal range)")
            if signed_dev is not None:
                abs_val = abs_dev if abs_dev is not None else abs(signed_dev)
                lines.append(f"  - Deviation from Baseline: {signed_dev:+.2f} (Absolute: {abs_val:.2f})")
            if dev_score is not None:
                lines.append(f"  - Standardized Deviation Score: {dev_score:+.2f}")
            if direction:
                persist_str = f" (Persistence: {persistence} intervals)" if persistence else ""
                lines.append(f"  - Trend Direction: {direction}{persist_str}")
            if hist.get("observationCount"):
                lines.append(
                    f"  - History: Min {hist.get('historicalMinimum')}, Max {hist.get('historicalMaximum')}, "
                    f"Mean {hist.get('historicalMean')}, Count: {hist.get('observationCount')}"
                )
            sections.append("\n".join(lines))

    # 2. Standalone baselines if not already covered
    if baselines and not latest_trends:
        sections.append("\nPersonal Baselines:")
        for b in baselines:
            m_type = b.get("metricType", "UNKNOWN")
            b_val = b.get("baselineValue")
            var = b.get("variability")
            cnt = b.get("observationCount")
            var_str = f" ±{var:.2f}" if var is not None else ""
            sections.append(f"• {m_type}: Personal baseline {b_val}{var_str} based on {cnt} observations (personal historical baseline)")

    # 3. Recent observation log
    if observations:
        sections.append("\nRecent Observations Logged:")
        for obs in observations[-5:]:
            m_type = obs.get("metricType")
            val = obs.get("metricValue")
            unit = obs.get("unit", "")
            logged_at = obs.get("loggedAt", "N/A")
            sections.append(f"• {m_type}: {val} {unit} (at {logged_at})")

    return True, "\n".join(sections)


def build_deterministic_context_explanation(
    context_data: Dict[str, Any],
    requested_metric: Optional[str] = None
) -> str:
    """
    Constructs a medically safe, non-diagnostic natural language explanation
    directly from verified Spring Boot context data.
    Used for MockLLMClient and graceful fallback.
    """
    trends = context_data.get("trends", [])
    baselines = context_data.get("baselines", [])
    observations = context_data.get("recent_observations", [])

    if requested_metric:
        req_norm = requested_metric.upper()
        trends = [t for t in trends if t.get("metricType", "").upper() == req_norm]
        baselines = [b for b in baselines if b.get("metricType", "").upper() == req_norm]
        observations = [o for o in observations if o.get("metricType", "").upper() == req_norm]

    if not trends and not baselines and not observations:
        target = requested_metric.title() if requested_metric else "your measurements"
        return (
            f"I checked your health profile, but there are no historical readings, baseline measurements, "
            f"or trend analyses recorded for {target} yet. "
            "Once you record observations, I will be able to analyze changes and compare them against your personal baseline."
        )

    paragraphs = ["Based on your verified health records from your profile:"]

    # Deduplicate trends by metric (latest first)
    latest_trends: Dict[str, Dict[str, Any]] = {}
    for t in trends:
        m = t.get("metricType", "Metric")
        if m not in latest_trends:
            latest_trends[m] = t

    for metric_name, t in latest_trends.items():
        curr = t.get("currentValue")
        prev = t.get("previousValue")
        base = t.get("baselineValue")
        var = t.get("variability")
        chg = t.get("valueChangeFromPrevious")
        pct = t.get("percentageChangeFromPrevious")
        direction = t.get("trendDirection", "STABLE")
        persistence = t.get("persistence", 1)
        signed_dev = t.get("deviationFromBaseline")

        unit = "mg/dL" if "GLUCOSE" in metric_name else "bpm" if "HEART" in metric_name else "mmHg" if "PRESSURE" in metric_name else ""
        unit_str = f" {unit}" if unit else ""

        desc = f"• {metric_name.replace('_', ' ').title()}: Your latest reading is {curr}{unit_str}"
        if prev is not None:
            chg_sign = f"+{chg:.1f}" if chg and chg > 0 else f"{chg:.1f}" if chg is not None else ""
            pct_sign = f"+{pct:.1f}%" if pct and pct > 0 else f"{pct:.1f}%" if pct is not None else ""
            desc += f", compared to your previous reading of {prev}{unit_str} ({chg_sign}{unit_str}, {pct_sign})"

        if base is not None:
            var_desc = f" (variability: ±{var:.1f}{unit_str})" if var else ""
            desc += f". Your personal baseline is {base:.1f}{unit_str}{var_desc}"
            if signed_dev is not None:
                if var is not None and abs(signed_dev) <= var:
                    desc += f", which is within your personal baseline/range"
                else:
                    dev_sign = "above" if signed_dev > 0 else "below" if signed_dev < 0 else "at"
                    desc += f", placing your current reading {abs(signed_dev):.1f}{unit_str} {dev_sign} your personal baseline"

        dir_desc = (
            "rising" if direction in ("RISING", "INCREASING")
            else "falling" if direction in ("FALLING", "DECREASING")
            else "stable"
        )
        desc += f", with a {dir_desc} trend observed across {persistence} consecutive intervals."
        paragraphs.append(desc)

    paragraphs.extend([
        "",
        "Clinical Context & Medical Disclaimer:",
        "These measurements and statistical deviations reflect observed changes in your personal history relative to your personal baseline, and are NOT a medical diagnosis or clinical normal range. "
        "Please review these trends with your healthcare provider for clinical interpretation and personalized care advice."
    ])

    return "\n".join(paragraphs)


def explain_patient_context(
    context_data: Dict[str, Any],
    user_message: str,
    llm_client: BaseLLMClient,
    requested_metric: Optional[str] = None
) -> str:
    """
    Orchestrates the natural-language explanation of verified patient health context.
    Passes verified context to LLM with strict medical guardrails, or uses deterministic fallback.
    """
    has_data, formatted_summary = format_patient_context_summary(context_data, requested_metric=requested_metric)

    if not has_data:
        target = requested_metric.title() if requested_metric else "your measurements"
        return (
            f"I checked your health profile, but there are no historical readings, baseline measurements, "
            f"or trend analyses recorded for {target} yet. "
            "Once you record observations, I will be able to analyze changes and compare them against your personal baseline."
        )

    # If running default MockLLMClient without canned response, use deterministic explanation
    if isinstance(llm_client, MockLLMClient) and (
        not getattr(llm_client, "canned_response", None)
        or not llm_client.canned_response.strip().startswith("{")
    ):
        return build_deterministic_context_explanation(context_data, requested_metric=requested_metric)

    user_prompt = (
        f'User Question: "{user_message}"\n\n'
        f"{formatted_summary}\n\n"
        "Please explain the patient's verified measurements, changes from previous readings, "
        "baseline comparison, and trend direction in clear, empathetic, and medically cautious natural language. "
        "Follow all strict medical guardrails. "
        "CRITICAL WORDING RULE: Describe readings close to baseline as 'within your personal baseline/range' "
        "or 'consistent with your personal historical baseline'. NEVER refer to personal baseline as a 'normal range' or 'within normal limits'."
    )

    try:
        explanation = llm_client.generate_chat_response(
            messages=[{"role": "user", "content": user_prompt}],
            system_prompt=PATIENT_CONTEXT_SYSTEM_PROMPT,
            temperature=0.2
        )
        if explanation and len(explanation.strip()) > 15:
            return explanation.strip()
    except (LLMUnavailableError, LLMResponseError) as e:
        logger.warning("LLM explanation failed (%s), falling back to deterministic explanation.", e)

    return build_deterministic_context_explanation(context_data, requested_metric=requested_metric)
