import re
import json
import logging
from typing import Optional, List, Dict, Any, Tuple, Iterator, Set
from app.schemas import (
    IntentType,
    Symptom,
    Measurement,
    StructuredHealthInformation
)
from app.llm_client import BaseLLMClient, MockLLMClient, LLMResponseError, LLMUnavailableError

logger = logging.getLogger(__name__)


def extract_pale_stool_user_wording(text: str) -> str:
    """Extracts precise user-reported phrasing for pale or discolored stools."""
    patterns = [
        (r"\bunusually\s+pale\s+stools?\b", "unusually pale stools"),
        (r"\bstools?\s+(?:are\s+not|aren'?t|look\s+not|do\s+not\s+look|don'?t\s+look|have\s+not\s+been|haven'?t\s+been)\s+unusually\s+pale\b", "unusually pale stools"),
        (r"\bunusually\s+light(?:-|\s+)colored\s+stools?\b", "unusually light-colored stools"),
        (r"\bpale\s+or\s+clay(?:-|\s+)colored\s+stools?\b", "pale or clay-colored stools"),
        (r"\bunusually\s+pale\s+bowel\s+movements?\b", "unusually pale bowel movements"),
        (r"\bclay(?:-|\s+)colored\s+stools?\b", "clay-colored stools"),
        (r"\blight(?:-|\s+)colored\s+stools?\b", "light-colored stools"),
        (r"\bpale\s+bowel\s+movements?\b", "pale bowel movements"),
        (r"\bpale\s+stools?\b", "pale stools")
    ]
    for pat, label in patterns:
        if re.search(pat, text, re.IGNORECASE):
            return label
    return "pale stools"


def extract_pale_stools_denial(text: str) -> Optional[str]:
    """
    Detects explicit patient negative findings for pale / clay-colored stools.
    Guards against questions, general statements, concerns, and contrastive positive affirmations.
    Preserves exact user wording combined with normalized clinical concept.
    """
    text_clean = text.lower().strip()

    # Guard: If pale stools is stated as positive after contrastive conjunction (e.g. 'haven't noticed jaundice, but I have noticed pale stools')
    contrast_pos = re.search(
        r"\b(?:but|however|except|yet)\s+(?:i\s+)?(?:have\s+(?:noticed|seen)|do\s+have|notice|have|noticed|seen)\s+[^.;!?\n]*?\b(?:pale|clay|light-colored)\s+(?:stools?|bowel\s+movements?)\b",
        text_clean
    )
    if contrast_pos:
        return None

    sentences = re.split(r"[\n.!?]+", text_clean)

    # 1. Subject negation: e.g. "my stools aren't unusually pale"
    subj_neg_match = re.search(
        r"\b(?:my\s+)?(?:stools?|bowel\s+movements?)\s+(?:are\s+not|aren'?t|do\s+not\s+look|don'?t\s+look|have\s+not\s+been|haven'?t\s+been)\s+(?:unusually\s+)?(?:pale|clay(?:-|\s+)colored|light(?:-|\s+)colored)\b",
        text_clean
    )
    if subj_neg_match:
        matched_text = subj_neg_match.group(0)
        for sent in sentences:
            if matched_text in sent:
                if re.match(r"^\s*(?:could|should|can|would|is|are|do|does|have\s+you)\b", sent.strip()) and not re.search(r"\b(?:i\s+haven'?t|i\s+don'?t|i\s+do\s+not|not\s+noticed)\b", sent):
                    return None
        wording = extract_pale_stool_user_wording(text_clean)
        return f"{wording} (no pale stools)"

    # 2. Direct pre-negation and serial clause negation
    cues = r"(?:no|not|neither|nor|without|den(?:y|ies|ied)|never|haven'?t\s+noticed|not\s+noticed|haven'?t\s+seen|not\s+seen|haven'?t|don'?t\s+have|do\s+not\s+have|don'?t|do\s+not|negative\s+for)"
    pale_terms_pat = r"(?:(?:unusually\s+)?(?:pale|clay(?:-|\s+)colored|light(?:-|\s+)colored)(?:\s+or\s+(?:clay(?:-|\s+)colored|pale))?\s+(?:stools?|bowel\s+movements?|poop))"

    # Direct pre-negation (cue within 4 words without contrastive conjunctions)
    direct_pat = rf"\b{cues}\s+(?:(?:(?!but|however|except|yet)\w+)\s+){{0,4}}{pale_terms_pat}\b"
    m_direct = re.search(direct_pat, text_clean)
    if m_direct:
        for sent in sentences:
            if re.search(pale_terms_pat, sent):
                if re.match(r"^\s*(?:could|should|can|would|have\s+you|did\s+you|is\s+there|are\s+there)\b", sent.strip()) and not re.search(r"\b(?:i\s+haven'?t|i\s+have\s+not|i\s+don'?t|i\s+do\s+not|not\s+noticed)\b", sent):
                    return None
        wording = extract_pale_stool_user_wording(text_clean)
        return f"{wording} (no pale stools)"

    # Serial clause negation (e.g. "haven't noticed jaundice or pale stools")
    clause_pat = rf"\b{cues}\s+[^.;!?\n]*?\b{pale_terms_pat}\b"
    m_clause = re.search(clause_pat, text_clean)
    if m_clause:
        matched_str = m_clause.group(0)
        cue_to_term = matched_str[:re.search(pale_terms_pat, matched_str).start()]
        if not re.search(r"\b(but|however|except|yet)\b", cue_to_term):
            for sent in sentences:
                if re.search(pale_terms_pat, sent):
                    if re.match(r"^\s*(?:could|should|can|would|have\s+you|did\s+you|is\s+there|are\s+there)\b", sent.strip()) and not re.search(r"\b(?:i\s+haven'?t|i\s+have\s+not|i\s+don'?t|i\s+do\s+not|not\s+noticed)\b", sent):
                        return None
            wording = extract_pale_stool_user_wording(text_clean)
            return f"{wording} (no pale stools)"

    return None


def extract_negative_symptoms(message: str) -> List[str]:
    """
    Extracts explicit negative or denied clinical symptoms and red flags.
    Never converts negative symptoms into positive symptoms.
    Supports direct pre-negation, coordinated negation across 'or'/'nor'/'and no', and post-negation.
    """
    text = message.lower().strip()
    negatives: List[str] = []

    def has_negated_term(term_pattern: str) -> bool:
        # 1. Direct pre-negation: cue followed directly by term (up to 4 words without contrastive conjunctions)
        direct_pat = rf"\b(?:no|not|neither|nor|without|den(?:y|ies|ied)|never|haven'?t\s+noticed|not\s+noticed|haven'?t\s+seen|not\s+seen|haven'?t|don'?t\s+have|do\s+not\s+have|don'?t|do\s+not|negative\s+for)\s+(?:(?!but|however|except|yet)\w+\s+){{0,4}}{term_pattern}\b"
        if re.search(direct_pat, text):
            return True
        # 2. Negated serial clause (e.g. "not noticed A, B, C, or D" or "don't have A, B, or C" bounded before contrastive conjunctions like "but")
        clause_pat = rf"\b(?:no|not|neither|without|den(?:y|ies|ied)|not\s+noticed|haven'?t\s+noticed|haven'?t|don'?t\s+have|do\s+not\s+have|don'?t|do\s+not|negative\s+for)\s+[^.;!?\n]*?\b{term_pattern}\b"
        m = re.search(clause_pat, text)
        if m:
            matched_str = m.group(0)
            if not re.search(r"\b(but|however|except|yet)\b", matched_str):
                return True
        # 3. Post-negation: term followed by none / negative / absent
        post_pat = rf"\b{term_pattern}\s*[:=-]?\s*(?:is\s+|was\s+)?(?:none|negative|absent|denied|zero)\b"
        if re.search(post_pat, text):
            return True
        return False

    # 1. Cardiovascular / Chest Pain
    if (
        has_negated_term(r"(?:severe\s+)?chest\s+pain")
        or has_negated_term(r"(?:severe\s+)?pain\s+in\s+(?:the\s+|my\s+)?chest")
        or has_negated_term(r"pain\s+(?:in|around)\s+(?:the\s+|my\s+)?chest")
    ):
        negatives.append("severe chest pain" if ("severe" in text and "severe chest pain" in text) else "chest pain")

    # 2. Respiratory Distress / Shortness of Breath
    if has_negated_term(r"(?:severe\s+)?(?:difficulty\s+breathing|shortness\s+of\s+breath|trouble\s+breathing)"):
        negatives.append("difficulty breathing")

    # 3. Dizziness / Lightheadedness
    if has_negated_term(r"dizz(?:y|iness)"):
        negatives.append("dizziness")

    # 4. Fainting / Syncope / Loss of consciousness
    if has_negated_term(r"(?:faint(?:ed|ing)?|loss\s+of\s+consciousness|passing\s+out|syncope)"):
        negatives.append("fainting / loss of consciousness")

    # 5. Blood in stool
    if has_negated_term(r"(?:obvious\s+)?blood\s+in\s+stool"):
        negatives.append("blood in stool")

    # 6. Black / tarry stool
    if has_negated_term(r"(?:black|tarry|black\/tarry)\s+stool"):
        negatives.append("black/tarry stool")

    # 7. Fever / Chills
    if has_negated_term(r"fever"):
        negatives.append("fever")
    if has_negated_term(r"chills?"):
        negatives.append("chills")

    # 8. Vomiting
    if has_negated_term(r"vomit(?:ing)?"):
        negatives.append("vomiting")

    # 9. Nausea
    if has_negated_term(r"nausea"):
        negatives.append("nausea")

    # 10. Jaundice / Yellowing
    if has_negated_term(r"(?:yellowing(?:\s+of\s+(?:eyes?|skin))?|jaundice)"):
        negatives.append("yellowing of eyes/skin (jaundice)")

    # 11. Stool frequency changes
    if has_negated_term(r"(?:major\s+)?stool(?:-|\s+)frequency\s+change"):
        negatives.append("stool frequency changes")

    # 12. Meal relationship / food trigger denial
    if (has_negated_term(r"(?:related\s+to\s+meals?|meal\s+relationship|food\s+triggers?|worse\s+(?:after|with)\s+meals?)") or
            re.search(r"\b(?:not|doesn'?t\s+seem|un)?related\s+to\s+meals?\b", text) or
            re.search(r"\b(?:not|no|never)\s+(?:worse\s+(?:after|with)|triggered\s+by)\s+(?:meals?|food|oily\s+food)\b", text)):
        negatives.append("meal relationship")

    # 13. Urine changes / normal urine / dark urine
    if (
        re.search(r"\b(?:my\s+)?urine\s+(?:looks?|is|appears)\s+normal\b", text)
        or re.search(r"\bnormal\s+urine\b", text)
        or has_negated_term(r"(?:dark\s+|abnormal\s+|changes?\s+in\s+)?urine")
        or re.search(r"\burine\s*[:=-]?\s*normal\b", text)
    ):
        negatives.append("normal urine (no dark urine)")

    # 14. Abdominal pain denial
    if (
        has_negated_term(r"(?:severe\s+)?(?:abdominal|stomach|belly)\s+pain")
        or has_negated_term(r"pain\s+in\s+(?:the\s+|my\s+)?(?:abdomen|stomach|belly|right\s+upper\s+abdomen)")
        or re.search(r"\bno\s+(?:abdominal|stomach|belly)\s+pain\b", text)
    ):
        negatives.append("abdominal pain")

    # 15. Pale / clay-colored / light stools denial (preserving user wording + clinical concept)
    pale_stool_neg = extract_pale_stools_denial(text)
    if pale_stool_neg and pale_stool_neg not in negatives:
        negatives.append(pale_stool_neg)

    return negatives


def extract_triggers(message: str) -> List[str]:
    """Extracts reported triggers or aggravating factors."""
    text = message.lower().strip()
    triggers: List[str] = []
    # Guard: If meal relationship is explicitly negated, do not extract meal triggers
    neg_meals = bool(
        re.search(r"\b(?:not|doesn'?t\s+seem|un)?related\s+to\s+meals?\b", text) or
        re.search(r"\b(?:no|not|without)\s+(?:clear\s+)?(?:relationship\s+to\s+meals?|meal\s+trigger)\b", text) or
        re.search(r"\b(?:not|no|never)\s+(?:worse\s+(?:after|with)|triggered\s+by)\s+(?:meals?|food|oily\s+food)\b", text) or
        re.search(r"\b(?:not\s+always|doesn'?t\s+always|isn'?t\s+always|not\s+necessarily)\s+(?:happen\s+after|occur\s+after|worse\s+after|related\s+to)\s+meals?\b", text)
    )
    if not neg_meals:
        if (re.search(r"\b(worse\s+(after|with)|after\s+eating)\s+(oily|fried|heavy|fatty)\s+(meals?|foods?)\b", text) or
                "oily meals" in text or "fried food" in text):
            triggers.append("worse after oily/fried/heavy meals")
        elif re.search(r"\bworse\s+(after|with)\s+([^,.;\n]+)", text):
            m = re.search(r"\bworse\s+(after|with)\s+([^,.;\n]+)", text)
            if m:
                triggers.append(f"worse with {m.group(2).strip()}")

    # Exertional / Work triggers
    if re.search(r"\b(worse\s+when\s+(?:i\s+have\s+to\s+)?work(?:ing)?(?:\s+continuously)?|happens?\s+(?:only\s+)?when\s+(?:i'?m\s+)?working|while\s+working|during\s+work|continuous\s+work|get\s+.*when\s+(?:i'?m\s+)?(?:working|walking)|when\s+(?:i'?m\s+)?(?:working|walking)\s+for\s+a\s+while)\b", text):
        if not re.search(r"\b(?:not|without|never)\s+(?:worse\s+when\s+working|happens?\s+when\s+working)\b", text):
            triggers.append("exertion / working or walking")
    elif re.search(r"\b(worse\s+with\s+(?:exertion|exercise|physical\s+activity))\b", text):
        triggers.append("exertion")

    return triggers


def extract_relieving_factors(message: str) -> List[str]:
    """Extracts reported relieving factors."""
    text = message.lower().strip()
    factors: List[str] = []
    if re.search(r"\b(better\s+when\s+(?:i\s+)?(?:sit\s+down\s+and\s+)?rest|settles\s+down\s+.*when\s+(?:i\s+)?rest|goes\s+away\s+when\s+(?:i\s+)?rest|relieved\s+by\s+rest|rest\s+helps?|better\s+after\s+(?:i\s+)?(?:stop\s+and\s+sit\s+down|sit\s+down|rest)|stop\s+and\s+sit\s+down)\b", text):
        factors.append("rest")
    if re.search(r"\bwater\s+helps?\s*(somewhat)?\b", text):
        factors.append("water/fluid intake helps somewhat")
    return factors


def extract_functional_change(message: str) -> Optional[str]:
    """Extracts reported changes in work or exercise capacity."""
    text = message.lower().strip()
    if re.search(r"\b(could\s+normally\s+work\s+.*without\s+getting\s+breathless|normally\s+work\s+.*recently\s+have\s+to\s+stop|recently\s+have\s+to\s+stop\s+and\s+rest|reduction\s+in\s+(?:ability\s+to\s+work|work\s+tolerance)|reduced\s+work\s+tolerance|this\s+is\s+new\s+for\s+me\b.*work|earlier\s+i\s+could\s+work\s+.*without\s+needing\s+to\s+stop|now\s+i\s+have\s+to\s+take\s+breaks|used\s+to\s+work\s+.*now\s+(?:i\s+)?(?:need|have\s+to\s+take)\s+breaks)\b", text):
        return "reduced work tolerance / need to stop and take breaks sooner than normal baseline"
    return None


def extract_progression(message: str) -> Optional[str]:
    """Extracts progression statements over time."""
    text = message.lower().strip()
    if re.search(r"\b(somewhat\s+)?worse\s+today\s+than\s+yesterday\b", text):
        return "somewhat worse today than yesterday"
    if re.search(r"\benergy\s+is\s+lower\s+today\b", text):
        return "energy lower today"
    return None


def extract_radiation(message: str) -> Optional[str]:
    """Extracts reported radiation or spreading location of pain."""
    text = message.lower().strip()
    patterns = [
        r"\b(?:radiat(?:es?|ing)?|spread(?:s|ing)?|extend(?:s|ing)?|travel(?:s|ing)?|shoot(?:s|ing)?|feel\s+it)\s+(?:to|towards?|into|in)\s+(?:the\s+|my\s+)?(upper\s+back|lower\s+back|back|right\s+shoulder|scapula|shoulder\s+blade|shoulder|jaw|neck|left\s+arm|arm)\b",
        r"\b(?:toward|towards|to)\s+(?:my\s+|the\s+)?(upper\s+back|lower\s+back|back|right\s+shoulder|scapula|shoulder\s+blade|shoulder|neck|jaw)\b",
        r"\bradiat(?:es?|ing)?\s+to\s+(?:the\s+|my\s+)?([a-z\s]+)"
    ]
    for p in patterns:
        m = re.search(p, text)
        if m:
            val = m.group(1).strip()
            if val and val not in ("normal", "none", "me"):
                return val
    return None



def extract_medical_history(message: str) -> List[str]:
    """Extracts past medical history explicitly stated by user."""
    text = message.lower().strip()
    history: List[str] = []
    if re.search(r"\bprevious\s+stomach\s+problems(\s+several\s+months\s+ago)?\b", text):
        history.append("previous stomach problems several months ago")
    return history


def extract_previous_tests(message: str) -> List[str]:
    """Extracts reported diagnostic test history."""
    text = message.lower().strip()
    tests: List[str] = []
    if (re.search(r"\b(no|not\s+had|haven'?t\s+had)\s+(?:any\s+)?recent\s+(?:medical\s+)?tests?\b", text) or
            re.search(r"\bno\s+recent\s+tests?\b", text)):
        tests.append("No recent medical tests reported")
    return tests


def is_out_of_scope_gk(message: str) -> bool:
    """Detects general knowledge / trivia queries outside healthcare scope."""
    text = message.lower().strip()
    gk_patterns = [
        r"\b(who\s+is|who\s+was)\s+(the\s+)?(prime\s+minister|pm|president|king|queen|governor|mayor|chief\s+minister|cm)\b",
        r"\b(prime\s+minister|president)\s+of\b",
        r"\b(capital|currency|population|area)\s+of\s+[a-z]+",
        r"\bwho\s+won\s+(the\s+)?(world\s+cup|match|game|election|trophy|oscar)\b",
        r"\b(weather|temperature)\s+in\s+[a-z]+",
        r"\btell\s+me\s+a\s+joke\b",
        r"\bwho\s+is\s+(narendra\s+modi|joe\s+biden|donald\s+trump|elon\s+musk)\b"
    ]
    return any(bool(re.search(p, text)) for p in gk_patterns)


def is_casual_greeting_or_intro(message: str) -> Tuple[bool, Optional[str]]:
    """Detects natural greetings and personal introductions."""
    text = message.lower().strip()
    # Check if there are any symptoms, health complaints, questions, or medical context
    has_medical_context = bool(re.search(
        r"\b(breath(e|ing|less(ness)?)?|pain|hurts?|ache|aching|discomfort|nausea|vomit(ing)?|"
        r"fever|feverish|fatigue|tired|weak(ness)?|dizz(y|iness)|faint(ing)?|cough(ing)?|"
        r"blood|rash|jaundice|yellow(ing)?|pressure|sugar|glucose|heart|pulse|palpitation|"
        r"swelling|problem|symptom|ill|sick|doctor|help|advice|what\s+should\s+i\s+do|"
        r"what\s+can\s+i\s+do|what\s+to\s+do|worse|better|rest|liver|kidney|disease|"
        r"condition|worried|concern(ed)?|scared|anxious)\b",
        text
    ))
    if has_medical_context:
        return False, None

    greeting_match = re.match(r"^(hi+|hello+|hey+|good\s+(morning|afternoon|evening)|howdy)[!.,? ]*$", text)
    if greeting_match:
        return True, "greeting"

    non_names = {
        "not", "a", "an", "the", "just", "feeling", "experiencing", "having",
        "getting", "sorry", "here", "still", "now", "so", "very", "really",
        "also", "currently", "new", "fine", "okay", "good", "bad", "sick", "ill",
        "worried", "wondering", "looking", "asking", "trying"
    }

    intro_match = re.search(r"\bmy\s+name\s+is\s+([a-zA-Z]+)\b", text)
    if intro_match:
        name = intro_match.group(1)
        if name.lower() not in non_names and len(name) > 1:
            return True, f"intro:{name.title()}"

    i_am_match = re.match(
        r"^(?:hello|hi|hey)?[,\s]*(?:i'?m|i\s+am)\s+([a-zA-Z]+)(?:[,\s]+(?:a|an|the)?\s*(?:[a-zA-Z-]+\s+)*(?:student|engineer|developer|software\s+engineer|designer|doctor|nurse|teacher|worker|programmer|analyst))?[.! ]*$",
        text
    )
    if i_am_match:
        name = i_am_match.group(1)
        if name.lower() not in non_names and len(name) > 1:
            return True, f"intro:{name.title()}"

    return False, None


def is_summary_request(message: str) -> bool:
    """Detects when user explicitly requests a summary of what they shared."""
    text = message.lower().strip()
    summary_patterns = [
        r"\b(summariz(e|ing)|give\s+me\s+a\s+summary|summary\s+of)\s+(everything|all|what|my\s+symptoms|what\s+i(\s+have)?\s+told)\b",
        r"\bsummarize\s+everything\b",
        r"\bwhat\s+have\s+i\s+told\s+you(\s+so\s+far)?\b",
        r"\bwhat\s+are\s+the\s+important\s+things\s+i('?ve|\s+have)?\s+told\s+you\b",
        r"\bwhat\s+(?:did|have)\s+i\s+(?:tell|told)\s+you\b",
        r"\bwhat\s+do\s+you\s+(?:know|remember)\s+about\s+my\s+symptoms\b",
        r"\bbased\s+on\s+everything\s+(?:i('?ve|\s+have)?\s+told\s+you|we('?ve|\s+have)?\s+discussed|so\s+far)\b",
        r"\bwhat\s+information\s+(?:would|could|might)\s+be\s+most\s+useful\s+for\s+a\s+doctor(?:\s+to\s+know)?\b"
    ]
    return any(bool(re.search(p, text)) for p in summary_patterns)


def is_missing_information_request(message: str) -> bool:
    """Detects when user explicitly asks what information is still missing, unknown, or needed."""
    text = message.lower().strip()
    patterns = [
        r"\bwhat\s+information\s+do\s+you\s+still\s+need(?:\s+to\s+know)?\b",
        r"\bwhat\s+(?:else\s+)?do\s+you\s+still\s+need(?:\s+to\s+know)?\b",
        r"\bwhat\s+(?:information\s+)?don'?t\s+you\s+know(?:\s+yet)?\b",
        r"\bwhat\s+(?:is\s+)?still\s+unknown\b",
        r"\bwhat\s+(?:information\s+)?is\s+missing\b",
        r"\bwhat\s+other\s+details?\s+do\s+you\s+need\b",
        r"\bwhat\s+do\s+you\s+still\s+need\s+from\s+me\b"
    ]
    return any(bool(re.search(p, text)) for p in patterns)


def is_next_steps_request(message: str) -> bool:
    """Detects when user explicitly asks what to do next."""
    text = message.lower().strip()
    patterns = [
        r"\bwhat\s+should\s+i\s+do\s+next\b",
        r"\bwhat\s+should\s+i\s+do\s+now\b",
        r"\bwhat\s+are\s+(?:the\s+)?next\s+steps?\b",
        r"\bwhat\s+to\s+do\s+next\b",
        r"\bhow\s+should\s+i\s+proceed\b",
        r"\bwhere\s+do\s+we\s+go\s+from\s+here\b",
        r"\bwhat\s+steps\s+should\s+i\s+take\b"
    ]
    return any(bool(re.search(p, text)) for p in patterns)


def is_diagnosis_question(message: str) -> bool:
    """Detects when user asks whether symptoms definitely indicate a specific disease (e.g. heart problem)."""
    text = message.lower().strip()
    patterns = [
        r"\b(?:do\s+you\s+think\s+)?(?:is\s+)?this\s+(?:is\s+)?definitely\s+(?:a\s+)?(?:heart|cardiac|liver|serious)\s+(?:problem|disease|condition|attack)\b",
        r"\bdo\s+you\s+think\s+(?:this\s+is\s+)?(?:definitely|certainly)\b",
        r"\bcould\s+this\s+definitely\s+be\b",
        r"\bam\s+i\s+definitely\s+having\s+a\b",
        r"\bis\s+it\s+definitely\s+a\b",
        r"\bdo\s+i\s+have\s+heart\s+disease\b",
        r"\bdo\s+i\s+have\s+liver\s+disease\b",
        r"\bare\s+you\s+diagnosing\s+me\b"
    ]
    return any(bool(re.search(p, text)) for p in patterns)


def is_question_about_causes_or_what_helps(message: str) -> bool:
    """Detects user questions asking what could cause symptoms or what information/tests would help."""
    text = message.lower().strip()
    patterns = [
        r"\bwhat\s+(?:other\s+)?information\s+(?:would|might|could)\s+help\b",
        r"\bwhat\s+(?:information|examinations?|tests?)(?:\s+or\s+(?:examinations?|tests?|information))*\s+(?:would|might|could)\s+(?:help|help\s+distinguish)\b",
        r"\bwhat\s+(?:kinds\s+of\s+)?causes?\b",
        r"\bwhat\s+(?:could|might)\s+(?:potentially\s+)?cause\b",
        r"\bwhat\s+causes\b",
        r"\bwhat\s+(?:information\s+or\s+)?(?:examinations?\s+or\s+)?tests?\s+(?:would|might|could)\s+(?:generally\s+)?help\s+(?:a\s+)?doctor\b",
        r"\bdistinguish\s+between\s+(?:them|causes|possibilities)\b",
        r"\bwhat\s+other\s+tests?\b",
        r"\bi\s+already\s+answered\s+those\s+questions\b",
        r"\b(?:please\s+)?answer\s+(?:that|this)\s+first\b",
        r"\b(?:important\s+)?information\s+(?:i\s+have\s+)?not\s+provided\b",
        r"\b(?:what\s+)?(?:a\s+)?doctors?\s+(?:might|could|would)\s+consider\b",
        r"\b(?:change|affect)\s+(?:what|how)\s+(?:a\s+)?doctors?\s+(?:might|could|would)\s+consider\b",
        r"\bdoes\s+that\s+(?:information\s+)?change\b",
        r"\bwhat\s+(?:could|might|would)\s+(?:that|this)\s+(?:indicate|mean|suggest|signify|be)\b",
        r"\bwhat\s+does\s+(?:that|this)\s+(?:indicate|mean|suggest|signify)\b",
        r"\bcould\s+(?:that|this)\s+(?:indicate|mean|suggest)\b",
        r"\bwhat\s+(?:could|might)\s+(?:that|this)\s+(?:point\s+to|be)\b"
    ]
    return any(bool(re.search(p, text)) for p in patterns)


def is_symptom_negated(s_name: str, location: Optional[str], neg_list: List[str]) -> bool:
    """Accurately reconciles positive symptoms with explicit negative findings."""
    for neg in neg_list:
        neg_low = neg.lower()
        if s_name == "pain":
            if "chest pain" in neg_low:
                if (location or "").lower() == "chest":
                    return True
            elif "abdominal pain" in neg_low:
                if "abdomen" in (location or "").lower():
                    return True
            elif neg_low in ("pain", "no pain", "pain: none"):
                return True
        elif s_name in ("shortness of breath", "difficulty breathing"):
            if "breath" in neg_low:
                return True
        elif s_name in ("dizziness", "fainting"):
            if s_name in neg_low or (s_name == "fainting" and "consciousness" in neg_low):
                return True
        elif s_name in ("fever", "vomiting", "nausea"):
            if s_name in neg_low:
                return True
        elif s_name in ("jaundice", "yellowing"):
            if "jaundice" in neg_low or "yellow" in neg_low:
                return True
        elif s_name in ("dark urine", "abnormal urine", "urine"):
            if "urine" in neg_low:
                return True
        elif s_name in ("pale stools", "pale stool", "clay-colored stools", "clay-colored stool", "light-colored stools"):
            if any(k in neg_low for k in ("pale", "clay", "light", "stool")):
                return True
        elif s_name == neg_low:
            return True
    return False


LOCATION_PATTERNS = {
    "right upper abdomen": [
        r"\b(?:upper(?:-|\s+)right|right\s+upper)\s+(?:side\s+(?:of\s+)?(?:the\s+|my\s+)?)?(?:abdomen|abdominal(?:\s+(?:area|region))?|quadrant|belly|stomach)\b",
        r"\b(?:ruq|right\s+upper\s+quadrant|upper(?:-|\s+)right\s+quadrant)\b",
        r"\b(?:below|under)\s+(?:my\s+|the\s+)?right\s+ribs?\b",
        r"\bright\s+rib\s+area\b",
    ],
    "chest": [r"\bchest\b", r"\bbreastbone\b", r"\bsternum\b"],
    "head": [r"\bhead\b", r"\btemples?\b", r"\bforehead\b"],
    "abdomen": [r"\babdomen\b", r"\bbelly\b", r"\bstomach\b"],
    "back": [r"\bback\b", r"\blower\s+back\b", r"\bupper\s+back\b"],
    "hands": [r"\bhands?\b", r"\bfingers?\b"],
    "legs": [r"\blegs?\b", r"\bfeet\b", r"\bfoot\b", r"\bankles?\b"]
}


def is_location_mention_positive(text: str, match_start: int, match_end: int) -> bool:
    """
    Determines if a location match in text represents a positive patient finding,
    as opposed to being negated, inside a question, or part of a general statement.
    """
    # 1. Find sentence boundaries
    sent_start = 0
    for i in range(match_start - 1, -1, -1):
        if text[i] in ".!?;\n":
            sent_start = i + 1
            break

    sent_end = len(text)
    for i in range(match_end, len(text)):
        if text[i] in ".!?;\n":
            sent_end = i
            break

    sentence = text[sent_start:sent_end].strip()

    # 2. Check abstract question without personal patient attribution
    has_personal = bool(re.search(r"\b(i|i've|i'm|my|me|mine|we|our|myself)\b", sentence, re.I))
    is_question = (
        sentence.endswith("?")
        or (sent_end < len(text) and text[sent_end] == "?")
        or bool(re.match(r"^\s*(?:could|can|is|are|does|do|would|should|what\s+if|why|what\s+causes?)\b", sentence, re.I))
    )
    if not has_personal and is_question:
        return False

    # 3. Check general medical statement without personal patient attribution
    general_pat = r"\b(?:can\s+(?:have|be|cause|lead|indicate|point)|is\s+(?:a\s+)?(?:common|possible|defined|frequent|typical|caused\s+by|associated\s+with)|causes?\s+(?:of|include)|refers\s+to|means\b|often\s+(?:indicates?|points?|means?)|may\s+(?:indicate|mean|cause|be))\b"
    if not has_personal and re.search(general_pat, sentence, re.I):
        return False

    # 4. Check clause boundaries within sentence before match
    text_before_match = text[sent_start:match_start]
    contrast_pat = r"\b(?:but|however|except|yet|although|though|whereas|instead\s+of|rather\s+than)\b"
    clause_start = sent_start
    for cb in re.finditer(contrast_pat, text_before_match, re.I):
        clause_start = sent_start + cb.end()

    # Contrastive negative boundary (e.g. "... in abdomen, not my chest")
    contrast_neg_pat = r",\s*(?:not|never)\b"
    for cnb in re.finditer(contrast_neg_pat, text_before_match, re.I):
        clause_start = sent_start + cnb.start() + 1

    clause_before = text[clause_start:match_start]

    # 5. Check pre-negation cues in clause_before
    neg_cues = r"\b(?:no|not|neither|nor|without|never|haven'?t\s+noticed|not\s+noticed|haven'?t\s+seen|not\s+seen|haven'?t\s+had|not\s+had|haven'?t|don'?t\s+have|do\s+not\s+have|don'?t|do\s+not|didn'?t\s+have|didn'?t|did\s+not\s+have|did\s+not|deny|denies|denied|free\s+of|zero)\b"
    if re.search(neg_cues, clause_before, re.I):
        return False

    # 6. Check post-negation cues in clause_after
    clause_after = text[match_end:sent_end]
    post_neg = r"^\s*(?:pain)?\s*[:=-]?\s*(?:is\s+|was\s+)?(?:none|negative|absent|denied|zero|ruled\s+out)\b"
    if re.search(post_neg, clause_after, re.I):
        return False

    return True


def extract_location(message: str) -> Optional[str]:
    """
    Extracts positive anatomical location from message.
    Ensures negative findings, questions, and general medical statements
    never establish a positive patient location.
    """
    text = message.lower().strip()
    for loc_name, patterns in LOCATION_PATTERNS.items():
        for p in patterns:
            for m in re.finditer(p, text):
                if is_location_mention_positive(text, m.start(), m.end()):
                    return loc_name
    return None


def validate_user_source_of_truth(
    extracted_info: StructuredHealthInformation,
    message: str
) -> StructuredHealthInformation:
    """
    Enforces the User-Source-of-Truth rule:
    1. Removes severity unless explicitly stated in the user's message.
    2. Removes frequency unless explicitly stated in the user's message.
    3. Removes conditions from known_conditions unless explicitly claimed by user.
    4. Extracts negative findings and ensures negative symptoms are never in positive symptoms.
    5. Preserves triggers, relieving factors, progression, medical history, and previous tests.
    """
    text_lower = message.lower().strip()

    # 1. Severity verification against user message
    if extracted_info.severity:
        if bool(re.search(r"\b(did\s+not\s+say|never\s+said|didn'?t\s+say)\b.*?\b(mild|moderate|severe|dull|sharp)\b", text_lower)):
            extracted_info.severity = None
            for s in extracted_info.symptoms:
                s.severity = None
        else:
            sev = extracted_info.severity.lower()
            has_sev_words = bool(re.search(
                r"\b(mild|moderate|severe|dull|sharp|intense|slight|minor|[1-9]\s*(?:\/|out\s+of)\s*10|not\s+extremely\s+severe)\b",
                text_lower
            ))
            # Ensure 'severe' did not only appear inside a negated phrase
            if "severe" in sev and not bool(re.search(r"\b(severe\s+pain|pain\s+is\s+severe)\b", text_lower)):
                neg_severe = bool(re.search(r"\b(no|not\s+noticed|den(y|ies|ied)|without)\s+severe\b", text_lower))
                if neg_severe:
                    has_sev_words = False

            if not has_sev_words:
                extracted_info.severity = None
                for s in extracted_info.symptoms:
                    s.severity = None
            elif "not extremely severe" in text_lower:
                extracted_info.severity = "not extremely severe"
                for s in extracted_info.symptoms:
                    s.severity = "not extremely severe"

    # 2. Frequency verification against user message
    if extracted_info.frequency:
        if bool(re.search(r"\b(did\s+not\s+say|never\s+said|didn'?t\s+say)\b.*?\b(constant|continuous|intermittent)\b", text_lower)):
            extracted_info.frequency = None
            for s in extracted_info.symptoms:
                s.frequency = None
        else:
            has_freq_words = bool(re.search(
                r"\b(constant|continuous|intermittent|comes\s+and\s+goes|all\s+the\s+time|on\s+and\s+off|twice|three\s+times|\d+\s+times)\b",
                text_lower
            ))
            if not has_freq_words:
                extracted_info.frequency = None
                for s in extracted_info.symptoms:
                    s.frequency = None

    # 3. Known conditions verification (prevent assistant suggestions like GERD/infection from becoming patient facts)
    if extracted_info.known_conditions:
        allowed_conditions = []
        for cond in extracted_info.known_conditions:
            cond_lower = cond.lower()
            has_personal_claim = bool(re.search(
                rf"\b(diagnosed\s+with|history\s+of|i\s+have\s+a\s+history\s+of|my\s+{re.escape(cond_lower)}|i\s+have\s+{re.escape(cond_lower)})\b",
                text_lower
            ))
            if has_personal_claim:
                allowed_conditions.append(cond)
        extracted_info.known_conditions = allowed_conditions

    # 4. Extract explicit negative findings from text
    negatives = extract_negative_symptoms(message)
    for neg in negatives:
        if neg not in extracted_info.negative_symptoms:
            extracted_info.negative_symptoms.append(neg)

    # Filter out negated symptoms from extracted positive symptoms
    if extracted_info.negative_symptoms:
        extracted_info.symptoms = [
            s for s in extracted_info.symptoms
            if not is_symptom_negated(s.name.lower(), s.location or extracted_info.location, extracted_info.negative_symptoms)
        ]

    # 5. Triggers, relieving factors, progression, medical history, previous tests
    for tr in extract_triggers(message):
        if tr not in extracted_info.triggers:
            extracted_info.triggers.append(tr)
    if any("meal" in n.lower() or "food" in n.lower() for n in extracted_info.negative_symptoms):
        extracted_info.triggers = [
            t for t in extracted_info.triggers
            if not any(k in t.lower() for k in ("meal", "food", "oily", "fatty"))
        ]
    for rf in extract_relieving_factors(message):
        if rf not in extracted_info.relieving_factors:
            extracted_info.relieving_factors.append(rf)
    prog = extract_progression(message)
    if prog and not extracted_info.progression:
        extracted_info.progression = prog
    for mh in extract_medical_history(message):
        if mh not in extracted_info.medical_history:
            extracted_info.medical_history.append(mh)
    for pt in extract_previous_tests(message):
        if pt not in extracted_info.previous_tests:
            extracted_info.previous_tests.append(pt)
    func_change = extract_functional_change(message)
    if func_change and not extracted_info.functional_change:
        extracted_info.functional_change = func_change

    return extracted_info


class StreamingJSONExtractor:
    """
    Incrementally extracts assistant_message tokens from an LLM stream producing JSON.
    Hides internal JSON scaffolding so the user only sees natural conversational text.
    Accumulates the entire response to parse structured attributes when generation completes.
    """
    def __init__(self):
        self.accumulated = ""
        self.in_message = False
        self.message_finished = False
        self.escaped = False
        self.is_json: Optional[bool] = None
        self.emitted_text: List[str] = []

    def process_chunk(self, chunk: str) -> List[str]:
        emitted_this_chunk: List[str] = []
        self.accumulated += chunk

        if self.is_json is None:
            stripped = self.accumulated.strip()
            if stripped:
                if stripped.startswith("{"):
                    self.is_json = True
                else:
                    self.is_json = False

        # Mode A: Non-JSON stream (plain text completion)
        if self.is_json is False:
            self.emitted_text.append(chunk)
            return [chunk]

        # Mode B: JSON stream
        escape_map = {'n': '\n', 't': '\t', 'r': '\r', '"': '"', '\\': '\\', '/': '/'}

        if not self.in_message and not self.message_finished:
            marker = '"assistant_message":'
            pos = self.accumulated.find(marker)
            if pos != -1:
                sub = self.accumulated[pos + len(marker):]
                quote_pos = sub.find('"')
                if quote_pos != -1:
                    self.in_message = True
                    start_idx = pos + len(marker) + quote_pos + 1
                    val_part = self.accumulated[start_idx:]
                    for ch in val_part:
                        if self.escaped:
                            emitted_this_chunk.append(escape_map.get(ch, ch))
                            self.escaped = False
                        elif ch == '\\':
                            self.escaped = True
                        elif ch == '"':
                            self.in_message = False
                            self.message_finished = True
                            break
                        else:
                            emitted_this_chunk.append(ch)
        elif self.in_message and not self.message_finished:
            for ch in chunk:
                if self.escaped:
                    emitted_this_chunk.append(escape_map.get(ch, ch))
                    self.escaped = False
                elif ch == '\\':
                    self.escaped = True
                elif ch == '"':
                    self.in_message = False
                    self.message_finished = True
                    break
                else:
                    emitted_this_chunk.append(ch)

        if emitted_this_chunk:
            text = "".join(emitted_this_chunk)
            self.emitted_text.append(text)
            return [text]
        return []

    def get_full_assistant_message(self) -> str:
        return "".join(self.emitted_text)


class ExtractionEngine:
    """
    Information extraction and intent classification engine.
    Supports real configurable LLM-driven extraction with fallback to deterministic rule-based parsing.
    """

    def __init__(self, llm_client: Optional[BaseLLMClient] = None):
        self.llm_client = llm_client or MockLLMClient()
        self.mode_label = (
            "real LLM provider" if not isinstance(self.llm_client, MockLLMClient)
            else "development/testing extraction -- not a medical NLP model"
        )

    @staticmethod
    def parse_llm_json(raw_text: str) -> Dict[str, Any]:
        """Extracts and parses JSON object from LLM completion string."""
        text = raw_text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            json_str = text[first_brace:last_brace + 1]
            return json.loads(json_str)

        return json.loads(text)

    @staticmethod
    def _is_educational_inquiry(message: str) -> bool:
        """
        Determines whether a message is an educational / definitional inquiry
        without active personal symptom reporting.
        """
        text = message.lower().strip()
        educational_pattern = (
            r"\b(what\s+is|what\s+are|what\s+does|what\s+causes?|what\s+would\s+cause|"
            r"why\s+is|why\s+are|why\s+do|why\s+does|why\s+would|"
            r"how\s+is|how\s+are|how\s+does|how\s+do|how\s+can|how\s+to|"
            r"can\s+.+\s+(occur|happen|cause|lead|be|indicate)|"
            r"could\s+.+\s+(occur|happen|cause|lead|be|indicate)|"
            r"explain\b|define\b|definition\b|tell\s+me\s+about\b|information\s+on\b|"
            r"is\s+.+\s+(dangerous|normal|serious|curable|contagious|a\s+symptom|related|common)|"
            r"when\s+is|when\s+do|when\s+should|"
            r"why\s+.+\s+(measured|tested|checked|examined))\b"
        )
        personal_attribution_pattern = (
            r"\b(i\s+have|i'?m\s+having|i\s+feel|i\s+experience|i\s+am\s+experiencing|"
            r"diagnosed(\s+with)?|my\b|mine\b|suffering(\s+from)?|developed|started|"
            r"i\s+am\s+having|i'?ve\s+had|i'?ve\s+been|for\s+me\b)\b"
        )
        has_educational = bool(re.search(educational_pattern, text))
        has_question_mark = "?" in text and not bool(re.search(r"\b(i\s+have|my|i\s+feel)\b", text))
        has_personal = bool(re.search(personal_attribution_pattern, text))
        return (has_educational or has_question_mark) and not has_personal

    @classmethod
    def _filter_symptoms(
        cls,
        symptoms: List[Symptom],
        message: str,
        intent: Optional[IntentType] = None
    ) -> List[Symptom]:
        """
        Filters out symptoms that are either:
        1. Hallucinated by the LLM without any mention in the user's message.
        2. Merely mentioned as concepts in an educational question without personal attribution.
        """
        if not symptoms:
            return []

        text = message.lower().strip()
        is_educational = (
            (intent == IntentType.GENERAL_HEALTH_QUESTION)
            or cls._is_educational_inquiry(message)
        )
        has_personal = bool(re.search(
            r"\b(i\s+have|i'?m\s+having|i\s+feel|i\s+experience|i\s+am\s+experiencing|"
            r"diagnosed(\s+with)?|my\b|mine\b|suffering(\s+from)?|developed|started|"
            r"i\s+am\s+having|i'?ve\s+had|i'?ve\s+been|for\s+me\b)\b",
            text
        ))

        if is_educational and not has_personal:
            logger.debug("Filtered all symptoms from educational query without personal attribution: '%s'", message)
            return []

        symptom_mention_patterns = {
            "jaundice": r"\b(jaundice|yellow(ing)?(\s+(of\s+)?(skin|eyes?))?|eyes?\s+(are\s+)?yellow(ish)?|icterus)\b",
            "pain": r"\b(pain|aching|hurts?|soreness?)\b",
            "nausea": r"\b(nausea|nauseous|sick\s+to\s+(my\s+)?stomach)\b",
            "vomiting": r"\bvomit(ing)?\b",
            "fever": r"\b(fever|feverish|high\s+temperature)\b",
            "fatigue": r"\b(fatigue|exhaustion|tired(ness)?)\b",
            "weakness": r"\bweak(ness)?\b",
            "reduced appetite": r"\b(reduced|poor|loss\s+of)\s+appetite\b",
            "dark urine": r"\b(?:dark(?:er)?\s+urine|urine\s+(?:is\s+|looks?\s+|turned\s+)?dark(?:er)?)\b",
            "pale stools": r"\b(?:(?:unusually\s+)?(?:pale|clay(?:-|\s+)colored|light(?:-|\s+)colored)(?:\s+or\s+(?:clay(?:-|\s+)colored|pale))?\s+(?:stools?|bowel\s+movements?|poop)|(?:stools?|bowel\s+movements?)\s+(?:are\s+|look\s+|have\s+been\s+)?(?:unusually\s+)?(?:pale|clay(?:-|\s+)colored|light(?:-|\s+)colored))\b",
            "itching": r"\b(itching|pruritus|itchy)\b",
            "tremor": r"\b(tremor|shaking)\b",
            "stiffness": r"\b(stiffness|rigidity)\b",
            "shortness of breath": r"\b(?:shortness|short)\s+of\s+breath|difficulty\s+breathing|breathless(?:ness)?\b",
            "palpitations": r"\b(palpitations?|racing\s+heart|heart\s+(?:starts?\s+)?beating\s+(faster|fast|rapidly)|heart\s+beats?\s+faster|rapid\s+heart\s*rate|fast\s+heart\s*rate|fluttering)\b",
            "cough": r"\bcough(ing)?\b",
            "headache": r"\bheadache\b",
            "dizziness": r"\bdizz(y|iness)\b",
            "fainting": r"\bfaint(ing)?\b"
        }

        filtered: List[Symptom] = []
        for s in symptoms:
            canonical = s.name.lower().strip()
            pattern = symptom_mention_patterns.get(canonical, rf"\b{re.escape(canonical)}\b")

            is_mentioned = bool(re.search(pattern, text))
            if not is_mentioned:
                logger.debug("Filtered unmentioned symptom '%s' from message: '%s'", s.name, message)
                continue

            filtered.append(s)

        return filtered

    def extract_with_llm(
        self, message: str, session: Any = None
    ) -> Tuple[Optional[str], IntentType, StructuredHealthInformation, Optional[str]]:
        """
        Extracts structured health information and classifies intent using the LLM.
        Falls back to rule-based extraction if using default mock or on malformed output.
        """
        if isinstance(self.llm_client, MockLLMClient) and (
            not getattr(self.llm_client, "canned_response", None)
            or not self.llm_client.canned_response.strip().startswith("{")
        ):
            has_prior = len(session.history) > 0 if session else False
            intent = self.classify_intent(message, has_prior_context=has_prior)
            extracted = self.extract_health_information(message)
            return None, intent, extracted, None

        from app.prompts import SYSTEM_PROMPT, build_context_prompt

        history = session.history if session else []
        health_info = session.health_info if session else None
        user_prompt = build_context_prompt(message, history, health_info, session=session)

        raw_output = self.llm_client.generate_chat_response(
            messages=[{"role": "user", "content": user_prompt}],
            system_prompt=SYSTEM_PROMPT,
            temperature=0.2,
            response_format={"type": "json_object"}
        )

        try:
            parsed = self.parse_llm_json(raw_output)
        except Exception as e:
            raise LLMResponseError(f"Failed to parse LLM structured output as JSON: {e}") from e

        raw_intent = parsed.get("intent", "UNKNOWN")
        try:
            intent = IntentType(raw_intent)
        except ValueError:
            has_prior = len(history) > 0
            intent = self.classify_intent(message, has_prior_context=has_prior)

        raw_info = parsed.get("extracted_info") or {}
        try:
            extracted_health_info = StructuredHealthInformation.model_validate(raw_info)
        except Exception as e:
            raise LLMResponseError(f"LLM extracted_info failed schema validation: {e}") from e

        extracted_health_info.symptoms = self._filter_symptoms(
            extracted_health_info.symptoms, message, intent
        )

        is_educational = (
            intent == IntentType.GENERAL_HEALTH_QUESTION
            or self._is_educational_inquiry(message)
        )
        has_personal = bool(re.search(
            r"\b(i\s+have|i'?m\s+having|i\s+feel|i\s+experience|i\s+am\s+experiencing|"
            r"diagnosed(\s+with)?|my\b|mine\b|suffering(\s+from)?|developed|started|"
            r"i\s+am\s+having|i'?ve\s+had|i'?ve\s+been|for\s+me\b)\b",
            message.lower().strip()
        ))
        if is_educational and not has_personal:
            extracted_health_info.symptoms = []
            extracted_health_info.known_conditions = []
            extracted_health_info.medical_history = []

        # Strict User-Source-Of-Truth validation across all fields
        extracted_health_info = validate_user_source_of_truth(extracted_health_info, message)

        assistant_message = parsed.get("assistant_message")
        follow_up_question = parsed.get("follow_up_question")

        return assistant_message, intent, extracted_health_info, follow_up_question

    def stream_extract_with_llm(
        self, message: str, session: Any = None
    ) -> Iterator[Tuple[str, Any]]:
        """
        Streams natural assistant tokens as they are generated by the LLM, hiding JSON scaffolding,
        and finally yields the complete parsed extraction tuple.
        """
        if isinstance(self.llm_client, MockLLMClient) and (
            not getattr(self.llm_client, "canned_response", None)
            or not self.llm_client.canned_response.strip().startswith("{")
        ):
            has_prior = len(session.history) > 0 if session else False
            intent = self.classify_intent(message, has_prior_context=has_prior)
            extracted = self.extract_health_information(message)
            yield ("complete", (None, intent, extracted, None))
            return

        from app.prompts import SYSTEM_PROMPT, build_context_prompt

        history = session.history if session else []
        health_info = session.health_info if session else None
        user_prompt = build_context_prompt(message, history, health_info, session=session)

        extractor = StreamingJSONExtractor()

        for chunk in self.llm_client.stream_chat_response(
            messages=[{"role": "user", "content": user_prompt}],
            system_prompt=SYSTEM_PROMPT,
            temperature=0.2,
            response_format={"type": "json_object"}
        ):
            tokens = extractor.process_chunk(chunk)
            for tok in tokens:
                yield ("token", tok)

        raw_output = extractor.accumulated
        try:
            parsed = self.parse_llm_json(raw_output)
        except Exception as e:
            logger.warning("Failed to parse accumulated LLM stream output as JSON: %s. Using fallback.", e)
            parsed = {"assistant_message": extractor.get_full_assistant_message()}

        raw_intent = parsed.get("intent", "UNKNOWN")
        try:
            intent = IntentType(raw_intent)
        except ValueError:
            has_prior = len(history) > 0
            intent = self.classify_intent(message, has_prior_context=has_prior)

        raw_info = parsed.get("extracted_info") or {}
        try:
            extracted_health_info = StructuredHealthInformation.model_validate(raw_info)
        except Exception:
            extracted_health_info = StructuredHealthInformation()

        extracted_health_info.symptoms = self._filter_symptoms(
            extracted_health_info.symptoms, message, intent
        )

        is_educational = (
            intent == IntentType.GENERAL_HEALTH_QUESTION
            or self._is_educational_inquiry(message)
        )
        has_personal = bool(re.search(
            r"\b(i\s+have|i'?m\s+having|i\s+feel|i\s+experience|i\s+am\s+experiencing|"
            r"diagnosed(\s+with)?|my\b|mine\b|suffering(\s+from)?|developed|started|"
            r"i\s+am\s+having|i'?ve\s+had|i'?ve\s+been|for\s+me\b)\b",
            message.lower().strip()
        ))
        if is_educational and not has_personal:
            extracted_health_info.symptoms = []
            extracted_health_info.known_conditions = []
            extracted_health_info.medical_history = []

        # Strict User-Source-Of-Truth validation across all fields
        extracted_health_info = validate_user_source_of_truth(extracted_health_info, message)

        assistant_message = parsed.get("assistant_message") or extractor.get_full_assistant_message()
        follow_up_question = parsed.get("follow_up_question")

        yield ("complete", (assistant_message, intent, extracted_health_info, follow_up_question))

    def classify_intent(self, message: str, has_prior_context: bool = False) -> IntentType:
        """
        Classifies user message into controlled intent categories.
        """
        text = message.lower().strip()

        # 1. Disease Risk Assessment Intent
        risk_patterns = [
            r"\b(check|assess|evaluate|calculate|predict|what is my)\s+(my\s+)?(diabetes|liver|heart|parkinson'?s?|disease)?\s*risk\b",
            r"\b(check|assess)\s+(my\s+)?(liver|heart|diabetes|parkinson'?s?)\b",
            r"\bam i at risk\b",
            r"\brisk\s+assessment\b",
            r"\b(what is my\s+)?(probability|chance)\s+of\s+(having\s+|getting\s+)?(diabetes|liver|heart|parkinson'?s?|disease)\b",
            r"\bprobability\s+of\s+(having|getting)\b",
            r"\b(i\s+have|enter|submit|my)\s+(my\s+)?(liver\s+)?(blood\s+|lab\s+|test\s+)?(test|report|results?|panel|values?)\b",
            r"\b(liver|blood)\s+(test|report|results?|panel)\b",
            r"\b(check|test)\s+liver\b"
        ]
        for p in risk_patterns:
            if re.search(p, text):
                return IntentType.DISEASE_RISK_ASSESSMENT

        # 2. Context-aware diagnosis inquiry or next-steps or missing-info in active session
        if is_diagnosis_question(message) or is_next_steps_request(message) or is_missing_information_request(message) or is_summary_request(message):
            return IntentType.SYMPTOM_ASSESSMENT

        # 3. Longitudinal Monitoring Query Intent
        monitoring_patterns = [
            r"\b(my\s+previous|last|past|historical)\s+(results?|readings?|records?|tests?)\b",
            r"\b(track|show|view)\s+(my\s+)?(trend|history|readings)\b",
            r"\b(has\s+my|have\s+my)\s+.*\s+(changed|trend)\b",
            r"\bwhat\s+is\s+my\s+baseline\b"
        ]
        for p in monitoring_patterns:
            if re.search(p, text):
                return IntentType.MONITORING_QUERY

        # 4. Medical History Update Intent
        history_patterns = [
            r"\b(i\s+take|i\s+am\s+taking|prescribed|my\s+medication\s+is)\b",
            r"\b(my\s+medical\s+history|diagnosed\s+with|history\s+of)\b",
            r"\bi\s+have\s+(hypertension|asthma|high\s+cholesterol|hepatitis)\b"
        ]
        for p in history_patterns:
            if re.search(p, text):
                return IntentType.MEDICAL_HISTORY_UPDATE

        # 5. General Health Question Intent
        gen_patterns = [
            r"^\s*(what|how|why|can|when|does|is|are)\s+",
            r"\bwhat\s+is\s+(blood\s+pressure|cholesterol|liver|diabetes|hypertension|glucose|bilirubin|heart\s+disease|angina)\b",
            r"\bwhat\s+does\s+.+\s+mean\b",
            r"\bexplain\b",
            r"\bdefine\b",
            r"\btell\s+me\s+about\b"
        ]
        if any(re.search(p, text) for p in gen_patterns) and not re.search(r"\b(i\s+have|my|i'?m\s+having|hurts?|pain|short\s+of\s+breath|shortness)\b", text):
            return IntentType.GENERAL_HEALTH_QUESTION

        # 6. Follow-Up Response (if context exists and message is an answer to a question or ongoing symptom report)
        if has_prior_context:
            if re.search(r"\b(what\s+should\s+i\s+do|what\s+can\s+i\s+do|what\s+to\s+do|what\s+ought\s+i\s+to\s+do|next\s+steps?)\b", text):
                return IntentType.SYMPTOM_ASSESSMENT

            follow_up_patterns = [
                r"\b(it\s+started|began|since|ago|duration|for\s+\d+|started)\b",
                r"\b(\d+\s*(days?|weeks?|months?|hours?|minutes?|mins?))\b",
                r"\b(mild|moderate|severe|\d+\s*/\s*10)\b",
                r"\b(constant|intermittent|comes\s+and\s+goes)\b",
                r"\b(worse|better|evening|morning|night|afternoon|settles\s+down|rest)\b",
                r"\b(fast|regular|irregular|faint|breathless|stop\s+and\s+rest|new\s+for\s+me)\b",
                r"^(yes|no|none|never|sometimes)\b"
            ]
            for p in follow_up_patterns:
                if re.search(p, text):
                    return IntentType.FOLLOW_UP_RESPONSE

        # 7. Symptom Assessment Intent
        symptom_patterns = [
            r"\b(pain|ache|hurts?|sore|discomfort|tender|swelling|cramp|burning)\b",
            r"\b(nausea|vomit(ing)?|fever|fatigue|tired(ness)?|weakness)\b",
            r"\b(yellow(ing)?|jaundice|dark\s+urine|pale\s+stool|itching)\b",
            r"\b(tremor|shaking|stiffness|slowness|rigidity)\b",
            r"\b(shortness\s+of\s+breath|short\s+of\s+breath|cough|breathless|breathing)\b",
            r"\b(heart\s+beating|palpitations?|racing\s+heart|fluttering)\b",
            r"\b(what\s+should\s+i\s+do|what\s+can\s+i\s+do|what\s+to\s+do|next\s+steps?)\b",
            r"\bi\s+(have|feel|experience|suffer\s+from)\b"
        ]
        for p in symptom_patterns:
            if re.search(p, text):
                return IntentType.SYMPTOM_ASSESSMENT

        return IntentType.UNKNOWN

    def extract_health_information(self, message: str) -> StructuredHealthInformation:
        """
        Parses natural language text into StructuredHealthInformation.
        Adheres strictly to the User-Source-Of-Truth rule:
        Never invents severity, frequency, duration, or conditions.
        """
        text = message.lower().strip()

        extracted_symptoms: List[Symptom] = []
        overall_location: Optional[str] = None
        overall_duration: Optional[str] = None
        overall_severity: Optional[str] = None
        overall_onset: Optional[str] = None
        overall_frequency: Optional[str] = None
        medications: List[str] = []
        conditions: List[str] = []
        measurements: List[Measurement] = []
        age: Optional[int] = None
        sex: Optional[str] = None

        # 1. Negative symptoms extraction
        negative_symptoms = extract_negative_symptoms(message)

        # 2. Location Detection (Negation-safe, question-safe, general-statement-safe)
        overall_location = extract_location(message)

        # 3. Duration Detection
        duration_patterns = [
            r"(\d+\s*(?:to|-)\s*\d+)\s*(?:minutes?|mins?|hours?|days?|weeks?)",
            r"(?:(?:usually\s+)?lasts?\s+)?(?:about\s+|around\s+|approx(?:\.|imately)?\s+)?((\d+|one|two|three|four|five|six|seven|eight|nine|ten|\d+\s*(?:to|-)\s*\d+)\s*(?:minutes?|mins?)\s+to\s+(?:an?\s+)?hour)",
            r"(?:for\s+)?(?:the\s+last\s+|the\s+past\s+)(?:a\s+couple\s+of|couple\s+of|a\s+few|\d+|one|two|three|four|five|six)\s+(?:days?|weeks?|months?|years?)",
            r"for\s+(?:the\s+last\s+|the\s+past\s+)?(a\s+couple\s+of|couple\s+of|a\s+few|two|three|\d+)\s+(days?|weeks?|months?|years?)",
            r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(days?|weeks?|months?|years?)",
            r"since\s+(yesterday|last\s+night|this\s+morning|\w+day)",
            r"for\s+(a\s+couple\s+of|a\s+few|two|three)\s+(days?|weeks?|hours?|minutes?)",
            r"(?:started\s+)?(a\s+few|a\s+couple\s+of|\d+|one|two|three|four|five)\s+(days?|weeks?|months?|hours?|minutes?)\s+ago",
            r"\ba\s+few\s+days\b",
            r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(minutes?|mins?|hours?)"
        ]
        for p in duration_patterns:
            match = re.search(p, text)
            if match:
                if p == duration_patterns[1]:
                    overall_duration = match.group(1).strip()
                else:
                    overall_duration = match.group(0).strip()
                break

        # 4. Severity Detection (User-Source-Of-Truth: ONLY if explicitly stated)
        if re.search(r"\b(did\s+not\s+say|never\s+said|didn'?t\s+say)\b.*?\b(mild|moderate|severe|dull|sharp)\b", text):
            overall_severity = None
        elif re.search(r"\bnot\s+extremely\s+severe\b", text):
            overall_severity = "not extremely severe"
        elif re.search(r"\bdull\b", text):
            overall_severity = "dull"
        elif re.search(r"\bsharp\b", text):
            overall_severity = "sharp"
        else:
            severity_patterns = [
                (r"\b(mild|slight|minor)\b", "mild"),
                (r"\b(moderate|medium)\b", "moderate"),
                (r"\b(severe|intense|unbearable|excruciating)\b", "severe"),
                (r"\b([1-9]|10)\s*(out\s+of|\/)\s*10\b", lambda m: m.group(0))
            ]
            for p, val in severity_patterns:
                m = re.search(p, text)
                if m:
                    # Guard: If 'severe' matched, ensure it's not merely part of a negated finding
                    matched_val = val(m) if callable(val) else val
                    if matched_val == "severe":
                        is_neg_severe = bool(re.search(r"\b(no|not\s+noticed|den(y|ies|ied)|without)\s+severe\b", text))
                        if is_neg_severe:
                            continue
                    overall_severity = matched_val
                    break

        # 5. Onset Detection
        if re.search(r"\bsudden(ly)?\b", text):
            overall_onset = "sudden"
        elif re.search(r"\bgradual(ly)?\b", text):
            overall_onset = "gradual"
        elif re.search(r"\bstarted\s+two\s+days\s+ago\b", text):
            overall_onset = "2 days ago"

        # 6. Frequency Detection (User-Source-Of-Truth: ONLY if explicitly stated)
        is_intermittent_negated = bool(re.search(r"\b(?:not|rather\s+than|instead\s+of)\s+(?:comes?\s+and\s+goes?|intermittent)\b", text))
        is_constant_negated = bool(re.search(r"\b(?:not|rather\s+than|instead\s+of)\s+(?:being\s+)?(?:constant|continuous)\b", text))

        has_intermittent = bool(re.search(r"\b(intermittent|comes\s+and\s+goes|periodic|on\s+and\s+off)\b", text))
        has_constant = bool(re.search(r"\b(constant|continuous|all\s+the\s+time|non-?stop)\b", text))

        if re.search(r"\b(did\s+not\s+say|never\s+said|didn'?t\s+say)\b.*?\b(constant|continuous|intermittent)\b", text):
            overall_frequency = None
        elif has_intermittent and not is_intermittent_negated:
            overall_frequency = "intermittent"
        elif has_constant and not is_constant_negated:
            overall_frequency = "constant"

        # 7. Symptom Detection
        symptom_keywords = [
            ("pain", r"\b(pain|aching|hurts?|soreness)\b"),
            ("nausea", r"\b(nausea|nauseous)\b"),
            ("vomiting", r"\bvomit(ing)?\b"),
            ("fever", r"\b(fever|feverish|high\s+temperature)\b"),
            ("chills", r"\b(chills?|rigors?|shivering)\b"),
            ("fatigue", r"\b(fatigue|exhaustion|tired(ness)?)\b"),
            ("weakness", r"\bweak(ness)?\b"),
            ("reduced appetite", r"\b(reduced|poor|loss\s+of)\s+appetite\b"),
            ("jaundice", r"\b(jaundice|yellow(ing)?(\s+(of\s+)?(skin|eyes?))?|eyes?\s+(are\s+)?yellow(ish)?)\b"),
            ("dark urine", r"\b(?:dark(?:er)?\s+urine|urine\s+(?:is\s+|looks?\s+|turned\s+)?dark(?:er)?)\b"),
            ("pale stools", r"\b(?:(?:unusually\s+)?(?:pale|clay(?:-|\s+)colored|light(?:-|\s+)colored)(?:\s+or\s+(?:clay(?:-|\s+)colored|pale))?\s+(?:stools?|bowel\s+movements?|poop))\b"),
            ("itching", r"\bitching|pruritus\b"),
            ("tremor", r"\btremor|shaking\b"),
            ("stiffness", r"\bstiffness|rigidity\b"),
            ("shortness of breath", r"\b(?:shortness|short)\s+of\s+breath|difficulty\s+breathing|breathless(?:ness)?\b"),
            ("palpitations", r"\b(palpitations?|racing\s+heart|heart\s+(?:starts?\s+)?beating\s+(faster|fast|rapidly)|heart\s+beats?\s+faster|rapid\s+heart\s*rate|fast\s+heart\s*rate|fluttering)\b"),
            ("cough", r"\bcough(ing)?\b"),
            ("headache", r"\bheadache\b"),
            ("dizziness", r"\bdizz(y|iness)\b"),
            ("fainting", r"\bfaint(ing)?\b")
        ]

        found_symptoms = []
        neg_lower_list = [n.lower() for n in negative_symptoms]

        for s_name, s_pat in symptom_keywords:
            if re.search(s_pat, text):
                if s_name == "pain":
                    # Check if pain is only mentioned in a negated context (e.g. "don't have chest pain")
                    # If all mentions of pain/hurts/aching/soreness are negated, do not add pain
                    pain_matches = list(re.finditer(s_pat, text))
                    unnegated_found = False
                    for pm in pain_matches:
                        p_start = pm.start()
                        prefix = text[max(0, p_start - 50):p_start]
                        # If preceded by a negation cue within the clause
                        if re.search(r"\b(?:no|not|neither|nor|without|den(?:y|ies|ied)|never|haven'?t\s+noticed|not\s+noticed|haven'?t\s+seen|not\s+seen|haven'?t|don'?t\s+have|do\s+not\s+have|don'?t|do\s+not)\s+(?:(?!but|however|except|yet)\w+\s+){0,4}$", prefix):
                            continue
                        unnegated_found = True
                        break
                    if not unnegated_found:
                        continue
                if s_name == "fatigue":
                    # Guard against speculative tiredness mentioned as a user question/theory rather than an experienced symptom
                    # e.g., "not sure whether it's because I'm tired, stressed, or something related to my heart"
                    if re.search(r"\b(?:not\s+sure\s+(?:whether|if)|wondering\s+if|could\s+it\s+be|maybe|perhaps)\s+.*?\b(?:tired|stress(?:ed)?)\b", text):
                        if not re.search(r"\b(?:i\s+feel|i\s+have|i\s+am|experiencing)\s+(?:constant\s+|extreme\s+|severe\s+)?(?:fatigue|exhaustion|tired)\b", text):
                            continue
                # Ensure symptom is not explicitly negated
                if not is_symptom_negated(s_name, overall_location, negative_symptoms):
                    found_symptoms.append(s_name)

        # 7. Radiation Detection
        radiation = extract_radiation(message)

        if found_symptoms:
            primary_name = found_symptoms[0]
            assoc = found_symptoms[1:]

            primary_symptom = Symptom(
                name=primary_name,
                location=overall_location,
                duration=overall_duration,
                severity=overall_severity,
                onset=overall_onset,
                frequency=overall_frequency,
                associated_symptoms=assoc,
                radiation=radiation
            )
            extracted_symptoms.append(primary_symptom)

            for extra in assoc:
                # Specialized frequency for vomiting if "twice" explicitly reported
                extra_freq = "twice" if extra == "vomiting" and ("twice" in text or "2 times" in text) else None
                extracted_symptoms.append(Symptom(name=extra, frequency=extra_freq))

            extracted_symptoms = self._filter_symptoms(extracted_symptoms, message, None)

        # 8. Demographics (Age / Sex)
        age_match = re.search(r"\b(\d{1,2})\s*(years?\s+old|yo|y\.o\.)\b", text)
        if age_match:
            try:
                age = int(age_match.group(1))
            except ValueError:
                age = None

        if re.search(r"\b(male|man|boy)\b", text) and not re.search(r"\bfemale\b", text):
            sex = "Male"
        elif re.search(r"\b(female|woman|girl)\b", text):
            sex = "Female"

        # 9. Medications
        med_list = ["metformin", "lisinopril", "insulin", "atorvastatin", "aspirin", "omeprazole"]
        for med in med_list:
            if re.search(rf"\b{med}\b", text):
                medications.append(med.capitalize())

        # 10. Known Conditions (Only when user explicitly claims them)
        cond_list = ["diabetes", "hypertension", "hepatitis", "fatty liver", "asthma"]
        for cond in cond_list:
            if re.search(rf"\b(i\s+have|diagnosed\s+with|history\s+of)\s+{cond}\b", text):
                conditions.append(cond.title())

        # 11. Measurements
        bp_match = re.search(r"\b(bp|blood\s+pressure)\s*(is|of)?\s*(\d{2,3})\s*/\s*(\d{2,3})\b", text)
        if bp_match:
            measurements.append(Measurement(
                name="blood_pressure_systolic",
                value=float(bp_match.group(3)),
                unit="mmHg"
            ))
            measurements.append(Measurement(
                name="blood_pressure_diastolic",
                value=float(bp_match.group(4)),
                unit="mmHg"
            ))

        # Fluid intake measurement (e.g. "drinking 2 to 2.5 liters of water")
        fluid_match = re.search(r"\b(?:drinking|drink)\s+([0-9]+(?:\.[0-9]+)?)\s*(?:to|-)\s*([0-9]+(?:\.[0-9]+)?)\s*(?:liters?|l)\b", text)
        if fluid_match:
            val = (float(fluid_match.group(1)) + float(fluid_match.group(2))) / 2.0
            measurements.append(Measurement(name="fluid_intake", value=val, unit="L/day"))
        else:
            fluid_single = re.search(r"\b(?:drinking|drink)\s+([0-9]+(?:\.[0-9]+)?)\s*(?:liters?|l)\b", text)
            if fluid_single:
                measurements.append(Measurement(name="fluid_intake", value=float(fluid_single.group(1)), unit="L/day"))

        # Body temperature measurement (e.g. "38.2°C", "temperature was 38.2 C", "temp 101.5 F")
        temp_match = (
            re.search(r"\b(?:temperature|temp(?:\.|\b))\s*(?:is|was|of|at)?\s*([3-4][0-9](?:\.[0-9]+)?)\s*(?:°\s*[cf]|degrees?\s*(?:celsius|fahrenheit|c|f)|c|f)?\b", text) or
            re.search(r"\b([3-4][0-9]\.[0-9]+)\s*(?:°\s*[cf]|degrees?\s*(?:celsius|fahrenheit|c|f)|c\b)", text)
        )
        
        body_temp_str: Optional[str] = None
        if temp_match:
            temp_val = float(temp_match.group(1))
            unit = "°F" if temp_val > 50 else "°C"
            body_temp_str = f"{temp_val}{unit}"
            measurements.append(Measurement(
                name="body_temperature",
                value=temp_val,
                unit=unit
            ))

        # Liver laboratory measurements
        from app.liver_assessment import LiverExtractor, LIVER_LAB_UNITS
        liver_raw = LiverExtractor().extract_from_text(text)
        for lab_key, unit in LIVER_LAB_UNITS.items():
            if lab_key in liver_raw:
                measurements.append(Measurement(
                    name=lab_key,
                    value=float(liver_raw[lab_key]),
                    unit=unit
                ))

        if age is None and "age" in liver_raw:
            val_age, err_age = LiverExtractor.validate_field("age", liver_raw["age"])
            if val_age is not None and not err_age:
                age = val_age
        if sex is None and "gender" in liver_raw:
            val_sex, err_sex = LiverExtractor.validate_field("gender", liver_raw["gender"])
            if val_sex is not None and not err_sex:
                sex = val_sex

        triggers = extract_triggers(message)
        relieving = extract_relieving_factors(message)
        progression = extract_progression(message)
        history = extract_medical_history(message)
        previous_tests = extract_previous_tests(message)
        functional_change = extract_functional_change(message)

        return StructuredHealthInformation(
            symptoms=extracted_symptoms,
            location=overall_location,
            duration=overall_duration,
            severity=overall_severity,
            onset=overall_onset,
            frequency=overall_frequency,
            associated_symptoms=[s.name for s in extracted_symptoms[1:]],
            medical_history=history,
            medications=medications,
            age=age,
            sex=sex,
            known_conditions=conditions,
            measurements=measurements,
            negative_symptoms=negative_symptoms,
            triggers=triggers,
            relieving_factors=relieving,
            progression=progression,
            previous_tests=previous_tests,
            radiation=radiation,
            body_temperature=body_temp_str,
            functional_change=functional_change
        )






