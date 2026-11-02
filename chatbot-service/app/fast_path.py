import re
import logging
from typing import Dict, Any, List, Tuple, Optional
from app.liver_assessment import LiverExtractor, LIVER_LAB_UNITS

logger = logging.getLogger(__name__)


class FastPathDetector:
    """
    Deterministic fast-path detector for structured medical/laboratory inputs.
    Identifies clear clinical laboratory measurements that can safely bypass the LLM,
    while ensuring symptom screening (Path B), general health questions (Path A),
    and emergency scenarios always retain their appropriate routing.
    """

    # Educational inquiries and conversational questions -> MUST go to LLM / Path A
    QUESTION_PATTERNS = [
        r"^\s*(what|how|why|can|could|when|does|is|are|tell\s+me|explain|define)\b",
        r"\bwhat\s+(is|are|does)\b",
        r"\bwhat\s+does\s+.+\s+mean\b",
        r"\bcan\s+you\s+(tell|explain|help)\b",
        r"\bexplain\b",
        r"\bmeaning\s+of\b",
        r"\bwhy\s+is\b",
        r"\btell\s+me\s+about\b",
        r"\bdo\s+i\s+have\b",
        r"\bam\s+i\s+at\s+risk\b",
        r"\bworried\b",
        r"\bshould\s+i\b",
        r"\?"
    ]

    # Subjective symptom complaints -> MUST go to LLM / Path B
    SYMPTOM_PATTERNS = [
        r"\b(pain|aching|hurts?|sore|discomfort|tender|burning|swelling|cramp)\b",
        r"\b(nausea|vomit(ing)?|fever|fatigue|tired(ness)?|weakness|exhaustion)\b",
        r"\b(yellow(ing)?|jaundice|dark\s+urine|pale\s+stool|itching|pruritus)\b",
        r"\b(shortness\s+of\s+breath|breathless|cough|chest\s+pain)\b",
        r"\b(feeling\s+tired|feel\s+sick|feel\s+bad|feeling\s+unwell)\b",
        r"\b(under\s+my\s+right\s+ribs?|right\s+rib)\b"
    ]

    # Ambiguous or non-committal phrasing -> MUST NOT silently become medical data
    AMBIGUITY_PATTERNS = [
        r"\b(about|approx|approximately|around|maybe|not\s+sure|guess|roughly)\b",
        r"\b\d+(?:\.\d+)?\s*(?:to|-)\s*\d+(?:\.\d+)?\b",
        r"\b(?:or|somewhere)\s+around\b"
    ]

    def __init__(self, liver_extractor: Optional[LiverExtractor] = None):
        self.liver_extractor = liver_extractor or LiverExtractor()

    def detect(self, message: str) -> Tuple[bool, Dict[str, Any], List[str]]:
        """
        Determines whether message qualifies for deterministic fast path.
        Returns:
            (is_match, valid_extracted_fields, validation_errors)
        """
        text = message.strip()
        if not text:
            return False, {}, []

        text_lower = text.lower()

        # 1. Reject questions and conversational inquiries
        for p in self.QUESTION_PATTERNS:
            if re.search(p, text_lower):
                return False, {}, []

        # 2. Reject natural language symptoms (protects Path B)
        for p in self.SYMPTOM_PATTERNS:
            if re.search(p, text_lower):
                return False, {}, []

        # 3. Reject ambiguous numeric phrasing (protects against guessing/converting estimates)
        for p in self.AMBIGUITY_PATTERNS:
            if re.search(p, text_lower):
                return False, {}, []

        # 4. Deterministic extraction and validation using existing LiverExtractor
        valid_fields, errors = self.liver_extractor.extract_and_validate(text)

        # Check if actual laboratory measurements were extracted
        has_lab_measurements = any(k in valid_fields for k in LIVER_LAB_UNITS)

        # Fast path activates ONLY when at least one valid laboratory measurement is present
        if has_lab_measurements:
            return True, valid_fields, errors

        return False, {}, []
