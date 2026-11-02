import re
from typing import Tuple, List, Optional
from app.schemas import SafetyStatus


# Comprehensive red-flag trigger patterns for critical medical emergencies
RED_FLAG_PATTERNS = [
    # 1. Cardiovascular / Acute Coronary Syndrome
    {
        "category": "acute_coronary_or_severe_cardiac",
        "keywords": [
            r"\bcrushing\s+(chest\s+)?(pain|pressure)\b",
            r"\bsevere\s+chest\s+(pain|pressure|tightness)\b",
            r"\bchest\s+pain\s+(radiating|spreading)\s+to\s+(left\s+arm|jaw|neck|back)\b",
            r"\bchest\s+pain\s+and\s+(shortness\s+of\s+breath|difficulty\s+breathing|sweating|nausea)\b"
        ],
        "description": "Severe chest pain or suspected acute coronary syndrome"
    },
    # 2. Severe Respiratory Distress
    {
        "category": "severe_respiratory_distress",
        "keywords": [
            r"\b(severe|can'?t|cannot)\s+(breathe|breath|breathing)\b",
            r"\b(severe|acute)\s+(difficulty|trouble|distress)\s+(in\s+|with\s+)?(breathing|breath)\b",
            r"\bsevere\s+shortness\s+of\s+breath\b",
            r"\bgasping\s+for\s+air\b",
            r"\bturning\s+blue\b",
            r"\bsuffocating\b"
        ],
        "description": "Severe respiratory distress or inability to breathe"
    },
    # 3. Altered Mental Status / Syncope
    {
        "category": "loss_of_consciousness_or_syncope",
        "keywords": [
            r"\bloss\s+of\s+consciousness\b",
            r"\bpassed\s+out\b",
            r"\bfainted\s+and\s+(can'?t|not)\s+(wake|wake\s+up|respond)\b",
            r"\bunconscious\b",
            r"\bunresponsive\b"
        ],
        "description": "Loss of consciousness, syncope, or unresponsiveness"
    },
    # 4. Acute Neurological / Suspected Stroke
    {
        "category": "acute_neurological_deficit",
        "keywords": [
            r"\bsudden\s+(facial|face)\s+droop(ing)?\b",
            r"\bsudden\s+slurred\s+speech\b",
            r"\b(sudden|severe)\s+(weakness|numbness)\s+(on\s+one\s+side|in\s+one\s+arm|in\s+one\s+leg)\b",
            r"\bsudden\s+inability\s+to\s+(speak|talk|move|walk)\b",
            r"\bworst\s+headache\s+of\s+(my\s+)?life\b",
            r"\bthunderclap\s+headache\b"
        ],
        "description": "Sudden severe neurological deficit or suspected stroke"
    },
    # 5. Severe Hemorrhage
    {
        "category": "severe_hemorrhage",
        "keywords": [
            r"\b(uncontrolled|severe|massive)\s+bleeding\b",
            r"\bvomiting\s+(large\s+amounts\s+of\s+)?blood\b",
            r"\bcoughing\s+up\s+(large\s+amounts\s+of\s+)?blood\b",
            r"\bhemorrhag(e|ing)\b"
        ],
        "description": "Severe uncontrolled bleeding or hematemesis"
    }
]

# Negation cues commonly preceding symptom descriptions in natural clinical narratives
NEGATION_CUE_PATTERN = re.compile(
    r"\b("
    r"no|not|non|never|neither|nor|without|free\s+of|"
    r"den(y|ies|ied|ying)|negative\s+for|rules?\s+out|ruled\s+out|"
    r"do\s+not|don'?t|does\s+not|doesn'?t|did\s+not|didn'?t|"
    r"have\s+not|haven'?t|has\s+not|hasn'?t|had\s+not|hadn'?t|"
    r"is\s+not|isn'?t|are\s+not|aren'?t|was\s+not|wasn'?t|were\s+not|weren'?t|"
    r"am\s+not|ain'?t"
    r")\b",
    re.IGNORECASE
)

# Clause and contrastive boundaries that terminate backward negation scope
CLAUSE_BOUNDARY_PATTERN = re.compile(
    r"([.;!?\n\r]|[,;]?\s*\b(but|however|yet|nevertheless|nonetheless|although|though|except|whereas|still|otherwise|until)\b)",
    re.IGNORECASE
)

# Post-negation cues directly following symptom terms (e.g. "chest pain: none", "chest pain is absent")
POST_NEGATION_PATTERN = re.compile(
    r"^\s*([:=-]?\s*(is\s+|was\s+)?(none|negative|absent|denied|ruled\s+out|zero))\b",
    re.IGNORECASE
)


class SafetyChecker:
    """
    Rule-based safety triage layer.
    Identifies high-acuity symptom combinations and triggers emergency routing.
    Distinguishes active positive emergency symptoms from negated or denied symptoms.
    Does NOT diagnose disease.
    """

    def __init__(self):
        # Compile regular expressions for performance and safety
        self.compiled_rules = []
        for group in RED_FLAG_PATTERNS:
            compiled_patterns = [re.compile(p, re.IGNORECASE) for p in group["keywords"]]
            self.compiled_rules.append({
                "category": group["category"],
                "description": group["description"],
                "patterns": compiled_patterns
            })

    @staticmethod
    def is_match_negated(text: str, match_start: int, match_end: int) -> bool:
        """
        Determines whether a specific emergency pattern match in text is negated.
        Evaluates:
        1. Post-negation directly after match (e.g., 'chest pain: none', 'chest pain is absent').
        2. Pre-negation within the local clause (e.g., 'I do not have severe chest pain', 'No chest pain').
        3. Coordinated negation across 'or' / 'nor' (e.g., 'not noticed severe chest pain or difficulty breathing').
        Prevents broad/global cancellation by stopping backward scope at clause boundaries ('but', ';', '.', etc.).
        """
        # 1. Check post-negation directly after match
        following_text = text[match_end:match_end + 30]
        if POST_NEGATION_PATTERN.search(following_text):
            return True

        # 2. Check pre-negation before match within the same clause
        preceding_text = text[:match_start]

        # Find the nearest clause boundary before match_start
        boundary_matches = list(CLAUSE_BOUNDARY_PATTERN.finditer(preceding_text))
        if boundary_matches:
            last_boundary = boundary_matches[-1]
            clause_text = preceding_text[last_boundary.end():]
        else:
            clause_text = preceding_text

        # Search for negation cues within this local clause
        neg_matches = list(NEGATION_CUE_PATTERN.finditer(clause_text))
        if neg_matches:
            last_neg = neg_matches[-1]
            text_between = clause_text[last_neg.end():]
            words_between = text_between.strip().split()
            # Direct negation scope within a clause (up to 6 words)
            if len(words_between) <= 6:
                return True
            # Coordinated negation scope within clause (e.g. "have not noticed severe chest pain or difficulty breathing")
            if re.search(r"\b(or|nor)\b", text_between, re.IGNORECASE) and len(words_between) <= 12:
                return True

        # 3. Check coordinated negation across clause boundary if the clause starts with 'or' / 'nor'
        # e.g., "I have not noticed severe chest pain, or difficulty breathing."
        coord_match = re.search(r"\b(or|nor)\b\s*([a-zA-Z\s]{0,25})$", preceding_text, re.IGNORECASE)
        if coord_match:
            text_before_or = preceding_text[:coord_match.start()]
            prev_boundaries = list(CLAUSE_BOUNDARY_PATTERN.finditer(text_before_or))
            prev_clause = text_before_or[prev_boundaries[-1].end():] if prev_boundaries else text_before_or
            prev_neg_matches = list(NEGATION_CUE_PATTERN.finditer(prev_clause))
            if prev_neg_matches:
                last_prev_neg = prev_neg_matches[-1]
                words_prev = prev_clause[last_prev_neg.end():].strip().split()
                if len(words_prev) <= 8:
                    return True

        return False

    def evaluate(self, message: str) -> Tuple[SafetyStatus, List[str], Optional[str]]:
        """
        Evaluates input text against critical emergency red flags, distinguishing positive
        symptoms from negated or denied symptoms.
        Returns:
            (SafetyStatus, list_of_triggered_categories, emergency_message_or_None)
        """
        triggers = []
        for rule in self.compiled_rules:
            category_matched = False
            for pattern in rule["patterns"]:
                for match in pattern.finditer(message):
                    if not self.is_match_negated(message, match.start(), match.end()):
                        triggers.append(rule["description"])
                        category_matched = True
                        break  # Found positive match for this pattern
                if category_matched:
                    break  # Match one per category

        if triggers:
            emergency_message = (
                "EMERGENCY WARNING: Your message indicates potentially critical symptoms "
                f"({', '.join(triggers)}). "
                "This conversational service cannot provide medical treatment or emergency care. "
                "Please immediately call emergency services (e.g., 911 or your local emergency number) "
                "or proceed to the nearest emergency department without delay."
            )
            return SafetyStatus.URGENT_MEDICAL_EVALUATION, triggers, emergency_message

        return SafetyStatus.NORMAL_CONVERSATION, [], None
