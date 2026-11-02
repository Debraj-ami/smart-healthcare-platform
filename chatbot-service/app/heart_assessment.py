import re
import json
import logging
from typing import Dict, Any, Optional, List, Tuple, Set
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# The 13 required Cleveland Heart Disease predictor features
HEART_REQUIRED_FIELDS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

# Nullable features in the Cleveland protocol
HEART_NULLABLE_FIELDS = {"ca", "thal"}

HEART_FIELD_LABELS = {
    "age": "Age",
    "sex": "Biological Sex (Female or Male)",
    "cp": "Chest Pain Type (cp)",
    "trestbps": "Resting Blood Pressure (mm Hg)",
    "chol": "Serum Cholesterol (mg/dL)",
    "fbs": "Fasting Blood Sugar (fbs)",
    "restecg": "Resting ECG (restecg)",
    "thalach": "Maximum Heart Rate (thalach, bpm)",
    "exang": "Exercise-Induced Angina (exang)",
    "oldpeak": "ST Depression (oldpeak, mm)",
    "slope": "ST Segment Slope (slope)",
    "ca": "Major Vessels Colored (ca, 0-3)",
    "thal": "Thallium Scintigraphy (thal)",
}

# Categorical mappings
HEART_CATEGORICAL_MAPS = {
    "sex": {0: "Female", 1: "Male"},
    "cp": {
        1: "Typical Angina",
        2: "Atypical Angina",
        3: "Non-Anginal Pain",
        4: "Asymptomatic",
    },
    "fbs": {0: "<= 120 mg/dL", 1: "> 120 mg/dL"},
    "restecg": {
        0: "Normal",
        1: "ST-T Wave Abnormality",
        2: "Left Ventricular Hypertrophy (LVH)",
    },
    "exang": {0: "No", 1: "Yes"},
    "slope": {1: "Upsloping", 2: "Flat", 3: "Downsloping"},
    "ca": {
        0: "0 major vessels",
        1: "1 major vessel",
        2: "2 major vessels",
        3: "3 major vessels",
    },
    "thal": {
        3: "Normal (3.0)",
        6: "Fixed Defect (6.0)",
        7: "Reversible Defect (7.0)",
    },
}

HEART_FIELD_BOUNDS = {
    "age": (1, 120, "Age must be an integer between 1 and 120 years"),
    "trestbps": (50.0, 250.0, "Resting Blood Pressure must be between 50 and 250 mm Hg"),
    "chol": (50.0, 600.0, "Serum Cholesterol must be between 50 and 600 mg/dL"),
    "thalach": (50.0, 250.0, "Maximum Heart Rate (thalach) must be between 50 and 250 bpm"),
    "oldpeak": (0.0, 10.0, "ST Depression (oldpeak) must be between 0.0 and 10.0 mm"),
}


def normalize_heart_field_name(name: Optional[str]) -> Optional[str]:
    """Maps aliases and conversational variations to canonical heart field names."""
    if not name:
        return None
    raw = str(name).strip().lower().replace(" ", "_").replace("/", "_").replace("-", "_").replace(".", "_")

    if raw in ("age", "patient_age"):
        return "age"
    if raw in ("sex", "gender", "biological_sex"):
        return "sex"
    if "chest_pain" in raw or raw in ("cp", "pain_type", "chest_pain_type"):
        return "cp"
    if "resting_bp" in raw or "blood_pressure" in raw or raw in ("trestbps", "rbp", "resting_blood_pressure", "bp"):
        return "trestbps"
    if "cholesterol" in raw or raw in ("chol", "serum_cholesterol"):
        return "chol"
    if "fasting_blood_sugar" in raw or "fasting_sugar" in raw or raw in ("fbs", "blood_sugar"):
        return "fbs"
    if "resting_ecg" in raw or "ecg" in raw or raw in ("restecg", "electrocardiogram"):
        return "restecg"
    if "max_heart_rate" in raw or "maximum_heart_rate" in raw or raw in ("thalach", "max_hr", "heart_rate", "mhr"):
        return "thalach"
    if "exercise_angina" in raw or "exercise_induced_angina" in raw or raw in ("exang", "exercise_pain"):
        return "exang"
    if "st_depression" in raw or "depression" in raw or raw in ("oldpeak", "st_dep"):
        return "oldpeak"
    if "st_slope" in raw or "segment_slope" in raw or raw in ("slope", "st_segment_slope"):
        return "slope"
    if "major_vessels" in raw or "vessels" in raw or raw in ("ca", "num_vessels", "colored_vessels"):
        return "ca"
    if "thallium" in raw or "scintigraphy" in raw or raw in ("thal", "thallium_defect"):
        return "thal"

    return None


class HeartAssessmentState(BaseModel):
    """
    Tracks conversational Heart Disease assessment inputs across turns.
    Preserves collected fields, tracks nullable explicit nulls, validation errors,
    and manages confirmation before calling the Spring Boot predict API.
    """
    session_id: str
    patient_id: Optional[int] = None
    status: str = "IN_PROGRESS"  # "IN_PROGRESS", "AWAITING_CONFIRMATION", "COMPLETED", "ERROR", "CANCELLED"
    values: Dict[str, Any] = Field(default_factory=dict)
    explicit_nulls: Set[str] = Field(default_factory=set)  # Tracks ca/thal explicitly declared unknown
    validation_errors: List[str] = Field(default_factory=list)
    confirmation_pending: bool = False
    completed_assessment: Optional[Dict[str, Any]] = None
    last_error: Optional[str] = None

    def get_missing_fields(self) -> List[str]:
        """Returns list of required fields that have neither a concrete value nor explicit null."""
        missing = []
        for f in HEART_REQUIRED_FIELDS:
            if f in self.values and self.values[f] is not None:
                continue
            if f in self.explicit_nulls:
                continue
            missing.append(f)
        return missing

    def get_collected_fields(self) -> Dict[str, Any]:
        """Returns map of already collected fields, including explicitly provided nulls."""
        res: Dict[str, Any] = {}
        for f in HEART_REQUIRED_FIELDS:
            if f in self.values and self.values[f] is not None:
                res[f] = self.values[f]
            elif f in self.explicit_nulls:
                res[f] = None
        return res

    def is_all_collected(self) -> bool:
        """True if all 13 fields are accounted for (value present or explicit null) with no active errors."""
        return len(self.get_missing_fields()) == 0 and len(self.validation_errors) == 0

    def to_spring_payload(self, patient_id: Optional[int] = None) -> Dict[str, Any]:
        """Converts collected fields to Spring Boot HeartPredictRequest payload. Never sends 'num'."""
        p_id = patient_id or self.patient_id or 1
        return {
            "patientId": int(p_id),
            "age": int(self.values["age"]),
            "sex": int(self.values["sex"]),
            "cp": int(self.values["cp"]),
            "trestbps": float(self.values["trestbps"]),
            "chol": float(self.values["chol"]),
            "fbs": int(self.values["fbs"]),
            "restecg": int(self.values["restecg"]),
            "thalach": float(self.values["thalach"]),
            "exang": int(self.values["exang"]),
            "oldpeak": float(self.values["oldpeak"]),
            "slope": int(self.values["slope"]),
            "ca": int(self.values["ca"]) if ("ca" in self.values and self.values["ca"] is not None) else None,
            "thal": int(self.values["thal"]) if ("thal" in self.values and self.values["thal"] is not None) else None,
        }

    def generate_missing_followup(self) -> str:
        """Constructs conversational, non-repetitive clinical follow-up for missing or invalid inputs."""
        sections = []

        if self.validation_errors:
            sections.append("I noticed an issue with the value provided:")
            for err in self.validation_errors:
                sections.append(f"• {err}")
            sections.append("")

        collected = self.get_collected_fields()
        if collected:
            sections.append("Recorded values so far:")
            for k, v in collected.items():
                display_val = self._format_field_display(k, v)
                sections.append(f"• {HEART_FIELD_LABELS.get(k, k)}: {display_val}")
            sections.append("")

        missing = self.get_missing_fields()
        if missing:
            if not collected:
                sections.append("I can help with a Heart Disease risk assessment. I need some clinical inputs. We can go through them one at a time.\nTo begin, please provide:")
            else:
                sections.append("To proceed with the Heart Disease risk assessment, I need:")
            # Ask for next 2-3 fields to avoid overwhelming the user
            for m in missing[:3]:
                sections.append(f"• {HEART_FIELD_LABELS.get(m, m)}")
            if len(missing) > 3:
                sections.append(f"(plus {len(missing) - 3} other clinical parameters)")
            sections.append("\nYou can provide these values whenever you are ready.")

        return "\n".join(sections)

    def generate_confirmation_summary(self) -> str:
        """Presents a concise summary of the 13 collected inputs and requests confirmation before prediction."""
        lines = [
            "All 13 clinical features for the Heart Disease assessment have been recorded.",
            "",
            "Here is what I have:"
        ]
        collected = self.get_collected_fields()
        for k in HEART_REQUIRED_FIELDS:
            v = collected.get(k)
            display_val = self._format_field_display(k, v)
            lines.append(f"• {HEART_FIELD_LABELS.get(k, k)}: {display_val}")

        lines.extend([
            "",
            "Would you like me to run the Heart Disease risk assessment?",
            "(You can say 'Yes, proceed' to run, or tell me if any value needs to be corrected, e.g., 'Actually my cholesterol is 210'.)"
        ])
        return "\n".join(lines)

    @staticmethod
    def _format_field_display(field: str, val: Any) -> str:
        if val is None:
            return "Not provided / Unknown"
        if field == "age":
            return f"{val} years"
        if field == "trestbps":
            return f"{val} mm Hg"
        if field == "chol":
            return f"{val} mg/dL"
        if field == "thalach":
            return f"{val} bpm"
        if field == "oldpeak":
            return f"{val} mm"
        if field in HEART_CATEGORICAL_MAPS:
            cat_map = HEART_CATEGORICAL_MAPS[field]
            try:
                int_v = int(val)
                return cat_map.get(int_v, str(val))
            except (ValueError, TypeError):
                return str(val)
        return str(val)


class HeartExtractor:
    """
    Extracts and validates Heart-specific clinical parameters from natural language.
    Never invents, guesses, or uses fabricated defaults.
    """

    @staticmethod
    def validate_field(field_name: str, raw_value: Any) -> Tuple[Optional[Any], Optional[str]]:
        """
        Validates individual field format and domain constraints.
        Returns (validated_value, error_message).
        """
        if raw_value is None:
            if field_name in HEART_NULLABLE_FIELDS:
                return None, None
            return None, f"{HEART_FIELD_LABELS.get(field_name, field_name)} cannot be empty."

        # Handle explicit string nulls for ca / thal
        if field_name in HEART_NULLABLE_FIELDS and isinstance(raw_value, str):
            s_val = raw_value.strip().lower()
            if s_val in ("unknown", "none", "not provided", "not_provided", "n/a", "null", "dont know", "don't know", "unsure"):
                return None, None

        # 1. Age
        if field_name == "age":
            try:
                age_val = int(raw_value)
            except (ValueError, TypeError):
                return None, f"Age must be a valid integer number (received: '{raw_value}')."
            min_v, max_v, err_msg = HEART_FIELD_BOUNDS["age"]
            if not (min_v <= age_val <= max_v):
                return None, f"{err_msg} (received: {age_val})."
            return age_val, None

        # 2. Sex: 0 = female, 1 = male
        if field_name == "sex":
            s_str = str(raw_value).strip().lower()
            if s_str in ("0", "female", "woman", "f"):
                return 0, None
            if s_str in ("1", "male", "man", "m"):
                return 1, None
            return None, f"Sex must be 0 (Female) or 1 (Male) (received: '{raw_value}')."

        # 3. Chest Pain Type (cp): 1 = typical angina, 2 = atypical angina, 3 = non-anginal pain, 4 = asymptomatic
        if field_name == "cp":
            try:
                cp_val = int(float(raw_value))
                if cp_val in (1, 2, 3, 4):
                    return cp_val, None
                return None, f"Chest pain type (cp) must be 1 (Typical Angina), 2 (Atypical Angina), 3 (Non-Anginal Pain), or 4 (Asymptomatic) (received: {cp_val})."
            except (ValueError, TypeError):
                cp_str = str(raw_value).strip().lower()
                if "typical" in cp_str and "atypical" not in cp_str:
                    return 1, None
                if "atypical" in cp_str:
                    return 2, None
                if "non_anginal" in cp_str or "non anginal" in cp_str:
                    return 3, None
                if "asymptomatic" in cp_str or "no pain" in cp_str:
                    return 4, None
                return None, f"Chest pain type (cp) must be 1, 2, 3, or 4 (received: '{raw_value}')."

        # 4. Fasting Blood Sugar (fbs): 0 = <= 120, 1 = > 120
        if field_name == "fbs":
            try:
                fbs_val = int(float(raw_value))
                if fbs_val in (0, 1):
                    return fbs_val, None
            except (ValueError, TypeError):
                pass
            fbs_str = str(raw_value).strip().lower()
            if any(k in fbs_str for k in ("> 120", ">120", "above 120", "high", "greater than 120", "elevated")):
                return 1, None
            if any(k in fbs_str for k in ("<= 120", "<=120", "< 120", "<120", "below 120", "normal", "less than or equal to 120")):
                return 0, None
            return None, f"Fasting blood sugar (fbs) must be 0 (<= 120 mg/dL) or 1 (> 120 mg/dL) (received: '{raw_value}')."

        # 5. Resting ECG (restecg): 0 = normal, 1 = ST-T wave abnormality, 2 = LVH
        if field_name == "restecg":
            try:
                r_val = int(float(raw_value))
                if r_val in (0, 1, 2):
                    return r_val, None
                return None, f"Resting ECG must be 0 (Normal), 1 (ST-T Wave Abnormality), or 2 (LVH) (received: {r_val})."
            except (ValueError, TypeError):
                r_str = str(raw_value).strip().lower()
                if "normal" in r_str:
                    return 0, None
                if "st_t" in r_str or "st-t" in r_str or "st t" in r_str or "wave abnormality" in r_str:
                    return 1, None
                if "lvh" in r_str or "hypertrophy" in r_str or "ventricular" in r_str:
                    return 2, None
                return None, f"Resting ECG must be 0 (Normal), 1 (ST-T Wave Abnormality), or 2 (LVH) (received: '{raw_value}')."

        # 6. Exercise-Induced Angina (exang): 0 = no, 1 = yes
        if field_name == "exang":
            s_str = str(raw_value).strip().lower()
            if s_str in ("0", "no", "n", "false", "none"):
                return 0, None
            if s_str in ("1", "yes", "y", "true"):
                return 1, None
            return None, f"Exercise-induced angina (exang) must be 0 (No) or 1 (Yes) (received: '{raw_value}')."

        # 7. ST Slope (slope): 1 = upsloping, 2 = flat, 3 = downsloping
        if field_name == "slope":
            try:
                sl_val = int(float(raw_value))
                if sl_val in (1, 2, 3):
                    return sl_val, None
                return None, f"ST segment slope must be 1 (Upsloping), 2 (Flat), or 3 (Downsloping) (received: {sl_val})."
            except (ValueError, TypeError):
                sl_str = str(raw_value).strip().lower()
                if "upsloping" in sl_str or "up" in sl_str:
                    return 1, None
                if "flat" in sl_str or "horizontal" in sl_str:
                    return 2, None
                if "downsloping" in sl_str or "down" in sl_str:
                    return 3, None
                return None, f"ST segment slope must be 1 (Upsloping), 2 (Flat), or 3 (Downsloping) (received: '{raw_value}')."

        # 8. Major Vessels (ca): 0-3, nullable
        if field_name == "ca":
            try:
                ca_val = int(float(raw_value))
                if 0 <= ca_val <= 3:
                    return ca_val, None
                return None, f"Number of major vessels (ca) must be between 0 and 3, or 'unknown' (received: {ca_val})."
            except (ValueError, TypeError):
                ca_str = str(raw_value).strip().lower()
                if ca_str in ("unknown", "not provided", "none", "null", "don't know", "dont know"):
                    return None, None
                return None, f"Major vessels (ca) must be between 0 and 3, or 'unknown' (received: '{raw_value}')."

        # 9. Thallium (thal): 3 = normal, 6 = fixed defect, 7 = reversible defect, nullable
        if field_name == "thal":
            try:
                th_val = int(float(raw_value))
                if th_val in (3, 6, 7):
                    return th_val, None
                return None, f"Thallium scintigraphy (thal) must be 3 (Normal), 6 (Fixed Defect), 7 (Reversible Defect), or 'unknown' (received: {th_val})."
            except (ValueError, TypeError):
                th_str = str(raw_value).strip().lower()
                if "normal" in th_str:
                    return 3, None
                if "fixed" in th_str:
                    return 6, None
                if "reversible" in th_str:
                    return 7, None
                if th_str in ("unknown", "not provided", "null", "don't know", "dont know"):
                    return None, None
                return None, f"Thallium scintigraphy (thal) must be 3 (Normal), 6 (Fixed Defect), 7 (Reversible Defect), or 'unknown' (received: '{raw_value}')."

        # Continuous numeric fields: trestbps, chol, thalach, oldpeak
        if field_name in HEART_FIELD_BOUNDS:
            try:
                val = float(raw_value)
            except (ValueError, TypeError):
                label = HEART_FIELD_LABELS.get(field_name, field_name)
                return None, f"{label} must be a valid numerical value (received: '{raw_value}')."

            min_v, max_v, err_msg = HEART_FIELD_BOUNDS[field_name]
            if not (min_v <= val <= max_v):
                return None, f"{err_msg} (received: {val})."

            return val, None

        return raw_value, None

    def extract_from_text(self, text: str) -> Tuple[Dict[str, Any], Set[str]]:
        """
        Parses text for mentions of the 13 Heart parameters.
        Returns:
            (extracted_dict, explicit_nulls_set)
        """
        text_lower = text.lower()
        extracted: Dict[str, Any] = {}
        explicit_nulls: Set[str] = set()

        # 1. Age
        m_age = re.search(r'\bage\s*(?:is|[:=])?\s*(\d{1,3})\b', text_lower)
        if m_age:
            extracted['age'] = int(m_age.group(1))
        else:
            m_age2 = re.search(r'\b(\d{1,3})\s*(?:years?\s+old|yo|y\.o\.|yr\s+old|-year-old)\b', text_lower)
            if m_age2:
                extracted['age'] = int(m_age2.group(1))
            else:
                m_age3 = re.search(r'\b(?:i am|i\'?m)\s+(\d{1,3})\b(?:\s*(?:years?\s+old|and\s+male|and\s+female|,?\s*male|,?\s*female))?', text_lower)
                if m_age3 and any(k in text_lower for k in ['year', 'old', 'male', 'female', 'age']):
                    extracted['age'] = int(m_age3.group(1))

        # Check explicit negative/invalid age mention like "age: -5" or "age is -5"
        m_age_invalid = re.search(r'\bage\s*(?:is|[:=])?\s*(-?\d+)\b', text_lower)
        if m_age_invalid and 'age' not in extracted:
            extracted['age'] = int(m_age_invalid.group(1))

        # 2. Sex: Female = 0, Male = 1
        if re.search(r'\b(female|woman)\b', text_lower):
            extracted['sex'] = 0
        elif re.search(r'\b(male|man)\b', text_lower):
            extracted['sex'] = 1
        elif re.search(r'\bsex\s*(?:is|[:=])?\s*(0|1)\b', text_lower):
            m_s = re.search(r'\bsex\s*(?:is|[:=])?\s*(0|1)\b', text_lower)
            if m_s:
                extracted['sex'] = int(m_s.group(1))
        elif re.search(r'\bsex\s*(?:is|[:=])?\s*(\d+)\b', text_lower):
            m_s = re.search(r'\bsex\s*(?:is|[:=])?\s*(\d+)\b', text_lower)
            if m_s:
                extracted['sex'] = int(m_s.group(1))

        # 3. Chest Pain (cp): 1=typical, 2=atypical, 3=non-anginal, 4=asymptomatic
        m_cp_code = re.search(r'\b(?:cp|chest\s+pain(?:\s+type)?)\s*(?:is|[:=])?\s*(\d+)\b', text_lower)
        if m_cp_code:
            extracted['cp'] = int(m_cp_code.group(1))
        elif "typical angina" in text_lower and "atypical" not in text_lower:
            extracted['cp'] = 1
        elif "atypical angina" in text_lower:
            extracted['cp'] = 2
        elif "non-anginal" in text_lower or "non anginal" in text_lower:
            extracted['cp'] = 3
        elif "asymptomatic" in text_lower or "no chest pain" in text_lower or "chest pain: none" in text_lower:
            extracted['cp'] = 4

        # 4. Resting Blood Pressure (trestbps): 50-250 mm Hg
        m_bp = re.search(
            r'\b(?:resting\s+(?:blood\s+pressure|bp)|trestbps|resting_bp|rbp)\s*(?:is|[:=])?\s*(-?\d+(?:\.\d+)?)\b',
            text_lower
        )
        if not m_bp:
            m_bp = re.search(r'\bblood\s+pressure\s*(?:is|[:=])?\s*(-?\d+(?:\.\d+)?)\b(?!\s*\/\s*\d+)', text_lower)
        if not m_bp:
            m_bp = re.search(r'\bbp\s*(?:is|[:=])?\s*(-?\d+(?:\.\d+)?)\b(?!\s*(?:high|low|\/\s*\d+))', text_lower)
        if m_bp:
            extracted['trestbps'] = float(m_bp.group(1))

        # 5. Serum Cholesterol (chol): 50-600 mg/dL
        m_chol = re.search(
            r'\b(?:serum\s+cholesterol|cholesterol|chol)\s*(?:is|[:=])?\s*(-?\d+(?:\.\d+)?)\b',
            text_lower
        )
        if m_chol:
            extracted['chol'] = float(m_chol.group(1))

        # 6. Fasting Blood Sugar (fbs): 0 (<=120), 1 (>120)
        m_fbs_num = re.search(r'\bfbs\s*(?:is|[:=])?\s*(\d+)\b', text_lower)
        if m_fbs_num:
            extracted['fbs'] = int(m_fbs_num.group(1))
        elif any(k in text_lower for k in ("fasting blood sugar > 120", "fbs > 120", "fasting blood sugar above 120", "fbs above 120", "blood sugar > 120")):
            extracted['fbs'] = 1
        elif any(k in text_lower for k in ("fasting blood sugar <= 120", "fbs <= 120", "fasting blood sugar below 120", "fbs below 120", "blood sugar <= 120", "fbs normal")):
            extracted['fbs'] = 0

        # 7. Resting ECG (restecg): 0=normal, 1=ST-T, 2=LVH
        m_ecg_num = re.search(r'\b(?:restecg|resting\s+ecg|ecg)\s*(?:is|[:=])?\s*(\d+)\b', text_lower)
        if m_ecg_num:
            extracted['restecg'] = int(m_ecg_num.group(1))
        elif "st-t wave" in text_lower or "st t wave" in text_lower or "st-t abnormality" in text_lower:
            extracted['restecg'] = 1
        elif "left ventricular hypertrophy" in text_lower or "lvh" in text_lower:
            extracted['restecg'] = 2
        elif "normal ecg" in text_lower or "resting ecg is normal" in text_lower or "normal resting ecg" in text_lower:
            extracted['restecg'] = 0

        # 8. Maximum Heart Rate (thalach): 50-250 bpm
        m_hr = re.search(
            r'\b(?:maximum\s+heart\s+rate|max\s+heart\s+rate|max\s+hr|thalach|mhr)\s*(?:is|[:=])?\s*(-?\d+(?:\.\d+)?)\b',
            text_lower
        )
        if m_hr:
            extracted['thalach'] = float(m_hr.group(1))

        # 9. Exercise-Induced Angina (exang): 0=no, 1=yes
        m_exang_match = re.search(r'\b(?:exercise\s*(?:induced)?\s*angina|exang)\s*(?:is|[:=])?\s*(yes|no|1|0)\b', text_lower)
        if m_exang_match:
            val_str = m_exang_match.group(1)
            extracted['exang'] = 1 if val_str in ('1', 'yes') else 0
        elif any(k in text_lower for k in ("exercise induced angina: yes", "exercise-induced angina: yes", "exercise angina: yes", "exercise induced angina is yes", "exang: yes", "exercise angina yes")):
            extracted['exang'] = 1
        elif any(k in text_lower for k in ("exercise induced angina: no", "exercise-induced angina: no", "exercise angina: no", "no exercise-induced angina", "no exercise angina", "exang: no", "exercise angina no")):
            extracted['exang'] = 0

        # 10. ST Depression (oldpeak): 0.0 - 10.0 mm
        m_oldpeak = re.search(
            r'\b(?:st\s+depression|oldpeak|depression)\s*(?:is|[:=])?\s*(-?\d+(?:\.\d+)?)\b',
            text_lower
        )
        if m_oldpeak:
            extracted['oldpeak'] = float(m_oldpeak.group(1))

        # 11. ST Slope (slope): 1=upsloping, 2=flat, 3=downsloping
        m_slope_num = re.search(r'\b(?:st\s+slope|slope)\s*(?:is|[:=])?\s*(\d+)\b', text_lower)
        if m_slope_num:
            extracted['slope'] = int(m_slope_num.group(1))
        elif "upsloping" in text_lower:
            extracted['slope'] = 1
        elif "downsloping" in text_lower:
            extracted['slope'] = 3
        elif "slope: flat" in text_lower or "slope is flat" in text_lower or "flat slope" in text_lower or "slope flat" in text_lower:
            extracted['slope'] = 2

        # 12. Major Vessels (ca): 0-3 or nullable unknown
        m_ca_num = re.search(r'\b(?:major\s+vessels?|vessels?|ca)\s*(?:is|[:=])?\s*(-?\d+)\b', text_lower)
        if not m_ca_num:
            m_ca_num = re.search(r'\b(-?\d+)\s*(?:major\s+)?vessels?\b', text_lower)
        if m_ca_num:
            extracted['ca'] = int(m_ca_num.group(1))
        elif any(k in text_lower for k in (
            "vessels unknown", "ca unknown", "vessels not provided", "ca not provided",
            "don't know vessels", "dont know vessels", "don't know ca", "dont know ca",
            "i don't know the number of vessels", "i dont know the number of vessels",
            "ca is unknown", "vessels: unknown"
        )):
            explicit_nulls.add("ca")

        # 13. Thallium (thal): 3=normal, 6=fixed, 7=reversible or nullable unknown
        m_thal_num = re.search(r'\b(?:thallium(?:\s+scintigraphy)?|thal)\s*(?:is|[:=])?\s*(-?\d+)\b', text_lower)
        if m_thal_num:
            extracted['thal'] = int(m_thal_num.group(1))
        elif any(k in text_lower for k in (
            "thal normal", "thallium normal", "thal is normal", "thallium is normal"
        )):
            extracted['thal'] = 3
        elif any(k in text_lower for k in (
            "thal fixed", "thallium fixed", "fixed defect"
        )):
            extracted['thal'] = 6
        elif any(k in text_lower for k in (
            "thal reversible", "thallium reversible", "reversible defect"
        )):
            extracted['thal'] = 7
        elif any(k in text_lower for k in (
            "thal unknown", "thallium unknown", "thal not provided", "thallium not provided",
            "don't know thal", "dont know thal", "i don't know thallium", "thal is unknown"
        )):
            explicit_nulls.add("thal")

        return extracted, explicit_nulls

    def extract_and_validate(
        self,
        text: str,
        age: Optional[Any] = None,
        sex: Optional[Any] = None
    ) -> Tuple[Dict[str, Any], Set[str], List[str]]:
        """
        Parses text and validates extracted candidates.
        Returns:
            (valid_extracted_dict, explicit_nulls_set, validation_errors_list)
        """
        raw, nulls = self.extract_from_text(text)

        if age is not None and "age" not in raw:
            raw["age"] = age
        if sex is not None and "sex" not in raw:
            raw["sex"] = sex

        valid: Dict[str, Any] = {}
        errors: List[str] = []

        for field, raw_val in raw.items():
            val, err = self.validate_field(field, raw_val)
            if err:
                errors.append(err)
            else:
                valid[field] = val

        return valid, nulls, errors

    def detect_disclaimed_required_fields(self, text: str) -> List[str]:
        """Detects if the user explicitly states that they lack a required non-nullable clinical parameter."""
        text_lower = text.lower()
        disclaimed: List[str] = []

        if re.search(r"\b(don't\s+have|dont\s+have|no|don't\s+know|dont\s+know|haven'?t\s+got|without|lacking)\s+(my\s+)?(cholesterol|chol)\b", text_lower):
            disclaimed.append("chol")
        if re.search(r"\b(don't\s+have|dont\s+have|no|don't\s+know|dont\s+know|haven'?t\s+had|never\s+had)\s+(an?\s+)?(ecg|electrocardiogram|resting\s+ecg|restecg)\b", text_lower):
            disclaimed.append("restecg")
        if re.search(r"\b(don't\s+have|dont\s+have|no|don't\s+know|dont\s+know)\s+(my\s+)?(blood\s+pressure|resting\s+bp|rbp|trestbps)\b", text_lower):
            disclaimed.append("trestbps")
        if re.search(r"\b(don't\s+have|dont\s+have|no|don't\s+know|dont\s+know)\s+(my\s+)?(fasting\s+blood\s+sugar|fbs|blood\s+sugar)\b", text_lower):
            disclaimed.append("fbs")
        if re.search(r"\b(don't\s+have|dont\s+have|no|don't\s+know|dont\s+know)\s+(my\s+)?(max\s+heart\s+rate|maximum\s+heart\s+rate|thalach)\b", text_lower):
            disclaimed.append("thalach")
        if re.search(r"\b(don't\s+have|dont\s+have|no|don't\s+know|dont\s+know)\s+(these\s+|any\s+|the\s+)?(lab\s+results?|tests?|reports?|clinical\s+values?|measurements?|clinical\s+data)\b", text_lower):
            disclaimed.append("general_labs")

        return disclaimed

    def detect_ambiguous_measurement(self, text: str) -> Optional[str]:
        """Detects vague qualitative statements about vital signs (e.g., 'My BP is high') without numeric values."""
        text_lower = text.lower()
        if re.search(r"\b(my\s+)?(bp|blood\s+pressure)\s+is\s+(high|elevated|low|normal|borderline|bad)\b", text_lower):
            if not re.search(r"\b\d{2,3}(?:\s*\/\s*\d{2,3})?\b", text_lower):
                return "trestbps"
        if re.search(r"\b(my\s+)?(heart\s+rate|pulse)\s+is\s+(high|fast|rapid|slow|low|normal)\b", text_lower):
            if not re.search(r"\b\d{2,3}\b", text_lower):
                return "thalach"
        return None


def format_heart_prediction_response(prediction: Dict[str, Any]) -> str:
    """
    Constructs a safe, research-oriented, non-diagnostic response grounded in the real backend Heart ML response.
    Never asserts a diagnosis, never makes clinical causal claims, never prescribes treatments.
    """
    prob = float(prediction.get("probability", 0.0))
    prob_pct = f"{prob * 100:.1f}%"
    threshold = float(prediction.get("threshold", 0.31))
    threshold_pct = f"{threshold * 100:.0f}%"
    pred_val = int(prediction.get("prediction", 1 if prob >= threshold else 0))
    is_positive = pred_val == 1
    assessment_id = prediction.get("assessmentId", "N/A")
    model_name = prediction.get("modelName", "Logistic Regression (Robust L1)")
    model_version = prediction.get("modelVersion", "1.0.0")

    pred_label = "Positive (Elevated Screening Output)" if is_positive else "Negative (Below Operating Threshold)"

    lines = [
        "The Heart Disease screening assessment has completed using the production model.",
        "",
        "Assessment Results:",
        f"• Model-estimated probability: {prob_pct} ({prob:.4f})",
        f"• Decision threshold: {threshold_pct} (t = {threshold:.2f})",
        f"• Model prediction: {pred_label}",
        f"• Model: {model_name} v{model_version} (Assessment #{assessment_id})",
        "",
        "Understanding the Decision Threshold:",
        "The probability is the model's estimated output for this assessment. The threshold is the operating cutoff used to convert that probability into the model's binary prediction. It does not represent a clinical diagnostic cutoff or guarantee individual disease presence or absence.",
        ""
    ]

    # SHAP Explanations
    explanation_available = prediction.get("explanationAvailable", False)
    pos_contribs = prediction.get("positiveContributors", [])
    neg_contribs = prediction.get("negativeContributors", [])
    shap_base = prediction.get("shapBaseValue")
    out_space = prediction.get("explanationOutputSpace", "log_odds")

    if explanation_available and (pos_contribs or neg_contribs):
        lines.append("Model Output Attribution (SHAP Analysis):")
        lines.append("These values show how individual model features contributed to this model's output for this assessment. They represent statistical model contributions, not biological causation, clinical diagnoses, or proof of disease.")
        lines.append("")

        if pos_contribs:
            lines.append("Positive Contributors (pushed the model output toward higher screening score):")
            for c in pos_contribs[:4]:
                name = c.get("displayName") or c.get("feature", "")
                val = c.get("rawValue", "")
                contrib = float(c.get("contribution", 0.0))
                tf = c.get("transformedFeature", "")
                cat_desc = ""
                if tf and tf.startswith("cat__"):
                    cat_name = tf.replace("cat__", "").replace("_", " ")
                    cat_desc = f" [encoded category: {cat_name}]"
                lines.append(f"• {name}{cat_desc}: +{contrib:.4f}")
            lines.append("")

        if neg_contribs:
            lines.append("Negative Contributors (pushed the model output toward lower screening score):")
            for c in neg_contribs[:4]:
                name = c.get("displayName") or c.get("feature", "")
                val = c.get("rawValue", "")
                contrib = float(c.get("contribution", 0.0))
                tf = c.get("transformedFeature", "")
                cat_desc = ""
                if tf and tf.startswith("cat__"):
                    cat_name = tf.replace("cat__", "").replace("_", " ")
                    cat_desc = f" [encoded category: {cat_name}]"
                lines.append(f"• {name}{cat_desc}: {contrib:.4f}")
            lines.append("")
    else:
        lines.append("Explanation Notice: An explanation is not available for this assessment.\n")

    lines.extend([
        "Clinical & Research Safety Notice:",
        "This platform provides AI-assisted health screening and risk estimation. It is not an automated medical diagnostic tool and is not a substitute for professional cardiovascular evaluation, diagnosis, or treatment. Please consult a qualified cardiologist or healthcare professional for clinical guidance."
    ])

    return "\n".join(lines)


def format_heart_history_response(assessments: List[Dict[str, Any]]) -> str:
    """
    Constructs an authentic response summarizing the patient's prior Heart assessments.
    Never hallucinates data, never describes probability changes as disease progression.
    """
    if not assessments:
        return (
            "No Heart assessments have been recorded yet for your patient profile. "
            "If you would like to complete an assessment, I can guide you through collecting the 13 required clinical parameters."
        )

    count = len(assessments)
    # Sort reverse chronological by assessedAt if present
    sorted_ass = sorted(
        assessments,
        key=lambda a: a.get("assessedAt") or "",
        reverse=True
    )
    latest = sorted_ass[0]
    prob = float(latest.get("riskScore", 0.0))

    # Safely inspect inputFeatures JSON if present
    parsed_json: Dict[str, Any] = {}
    raw_inputs = latest.get("inputFeatures")
    if raw_inputs and isinstance(raw_inputs, str):
        try:
            parsed_json = json.loads(raw_inputs)
        except Exception:
            parsed_json = {}

    threshold = float(parsed_json.get("threshold", 0.31))
    pred_val = int(parsed_json.get("prediction", 1 if prob >= threshold else 0))
    pred_label = "Positive (Elevated score)" if pred_val == 1 else "Negative (Below cutoff)"
    assessed_at = latest.get("assessedAt", "N/A")

    lines = [
        f"You have {count} Heart Disease screening assessment{'s' if count != 1 else ''} recorded in the platform.",
        "",
        "Latest Assessment Summary:",
        f"• Assessment ID: #{latest.get('id', 'N/A')}",
        f"• Date: {assessed_at}",
        f"• Model-estimated probability: {prob * 100:.1f}% ({prob:.4f})",
        f"• Decision threshold: t = {threshold:.2f}",
        f"• Model prediction: {pred_label}",
        ""
    ]

    if count > 1:
        lines.append("Previous model-estimated assessment outputs:")
        for ass in sorted_ass[1:4]:
            prev_p = float(ass.get("riskScore", 0.0))
            lines.append(f"• #{ass.get('id')}: {prev_p * 100:.1f}% on {ass.get('assessedAt', 'N/A')}")
        lines.append("\nNote: Variations across assessments represent prior model screening outputs, not clinical disease progression.")
        lines.append("")

    lines.append("Would you like me to explain the SHAP factor contributions for your latest assessment, or run a new assessment?")
    return "\n".join(lines)


def format_heart_explanation_response(assessment: Dict[str, Any]) -> str:
    """
    Explains an existing assessment's probability and SHAP feature attributions grounded strictly in stored data.
    """
    prob_val = assessment.get("riskScore") if assessment.get("riskScore") is not None else assessment.get("probability", 0.0)
    prob = float(prob_val)
    raw_inputs = assessment.get("inputFeatures")
    parsed_json: Dict[str, Any] = {}
    if raw_inputs and isinstance(raw_inputs, str):
        try:
            parsed_json = json.loads(raw_inputs)
        except Exception:
            parsed_json = {}

    threshold = float(parsed_json.get("threshold") or assessment.get("threshold") or 0.31)
    pred_raw = parsed_json.get("prediction") if parsed_json.get("prediction") is not None else assessment.get("prediction")
    pred_val = int(pred_raw) if pred_raw is not None else (1 if prob >= threshold else 0)
    pred_label = "Positive" if pred_val == 1 else "Negative"
    ass_id = assessment.get("id") or assessment.get("assessmentId") or "Latest"

    lines = [
        f"Here is the detailed explanation for Heart Assessment #{ass_id}:",
        f"• Model-estimated probability: {prob * 100:.1f}%",
        f"• Decision threshold: t = {threshold:.2f}",
        f"• Binary model prediction: {pred_label}",
        "",
        "Understanding this Result:",
        "The probability is the model's estimated output for this assessment. The threshold is the operating cutoff used to convert that probability into the model's binary prediction.",
        ""
    ]

    pos_contribs = parsed_json.get("positiveContributors") or assessment.get("positiveContributors") or []
    neg_contribs = parsed_json.get("negativeContributors") or assessment.get("negativeContributors") or []
    all_contribs = parsed_json.get("allContributions") or assessment.get("allContributions") or []
    if not pos_contribs and all_contribs:
        pos_contribs = [c for c in all_contribs if float(c.get("contribution", 0.0)) > 0]
    if not neg_contribs and all_contribs:
        neg_contribs = [c for c in all_contribs if float(c.get("contribution", 0.0)) < 0]

    if pos_contribs or neg_contribs:
        lines.append("Key Model Contributors (SHAP):")
        lines.append("These values show how individual model features contributed to this model's output for this assessment:")
        if pos_contribs:
            lines.append("Risk-Increasing Contributors:")
            for c in pos_contribs[:3]:
                name = c.get("displayName") or c.get("feature", "")
                tf = c.get("transformedFeature", "")
                contrib = float(c.get("contribution", 0.0))
                cat_desc = f" (encoded: {tf.replace('cat__', '')})" if tf.startswith("cat__") else ""
                lines.append(f"• {name}{cat_desc}: +{contrib:.4f}")
        if neg_contribs:
            lines.append("Risk-Decreasing Contributors:")
            for c in neg_contribs[:3]:
                name = c.get("displayName") or c.get("feature", "")
                tf = c.get("transformedFeature", "")
                contrib = float(c.get("contribution", 0.0))
                cat_desc = f" (encoded: {tf.replace('cat__', '')})" if tf.startswith("cat__") else ""
                lines.append(f"• {name}{cat_desc}: {contrib:.4f}")
        lines.append("")
    else:
        lines.append("An explanation is not available for this assessment.\n")

    lines.extend([
        "Medical Notice:",
        "SHAP values reflect model-output contributions, not biological causation or clinical diagnoses. Consult a physician for cardiovascular evaluation."
    ])

    return "\n".join(lines)


# Singleton extractor instance
heart_extractor = HeartExtractor()
