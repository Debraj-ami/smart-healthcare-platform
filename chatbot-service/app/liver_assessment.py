import re
import logging
from typing import Dict, Any, Optional, List, Tuple
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

LIVER_REQUIRED_FIELDS = [
    "age",
    "gender",
    "total_bilirubin",
    "direct_bilirubin",
    "alkaline_phosphatase",
    "sgpt",
    "sgot",
    "total_proteins",
    "albumin",
    "a_g_ratio"
]

FIELD_LABELS = {
    "age": "Age",
    "gender": "Gender (Male or Female)",
    "total_bilirubin": "Total Bilirubin (mg/dL)",
    "direct_bilirubin": "Direct Bilirubin (mg/dL)",
    "alkaline_phosphatase": "Alkaline Phosphatase (IU/L)",
    "sgpt": "SGPT / ALT (U/L)",
    "sgot": "SGOT / AST (U/L)",
    "total_proteins": "Total Proteins (g/dL)",
    "albumin": "Albumin (g/dL)",
    "a_g_ratio": "A/G Ratio"
}

FIELD_BOUNDS = {
    "age": (1, 120, "Age must be an integer between 1 and 120 years"),
    "total_bilirubin": (0.0001, 100.0, "Total Bilirubin must be greater than 0 and at most 100 mg/dL"),
    "direct_bilirubin": (0.0, 50.0, "Direct Bilirubin must be between 0 and 50 mg/dL"),
    "alkaline_phosphatase": (0.0001, 5000.0, "Alkaline Phosphatase must be greater than 0 and at most 5000 IU/L"),
    "sgpt": (0.0001, 5000.0, "SGPT (ALT) must be greater than 0 and at most 5000 U/L"),
    "sgot": (0.0001, 5000.0, "SGOT (AST) must be greater than 0 and at most 5000 U/L"),
    "total_proteins": (0.0001, 20.0, "Total Proteins must be greater than 0 and at most 20 g/dL"),
    "albumin": (0.0001, 15.0, "Albumin must be greater than 0 and at most 15 g/dL"),
    "a_g_ratio": (0.0001, 10.0, "A/G Ratio must be greater than 0 and at most 10")
}

LIVER_LAB_UNITS: Dict[str, Optional[str]] = {
    "total_bilirubin": "mg/dL",
    "direct_bilirubin": "mg/dL",
    "alkaline_phosphatase": "IU/L",
    "sgpt": "U/L",
    "sgot": "U/L",
    "total_proteins": "g/dL",
    "albumin": "g/dL",
    "a_g_ratio": None,
}


def normalize_liver_field_name(name: Optional[str]) -> Optional[str]:
    """Maps various aliases / formatting to canonical liver field names."""
    if not name:
        return None
    raw = str(name).strip().lower().replace(" ", "_").replace("/", "_").replace("-", "_").replace(".", "_")

    # Direct Bilirubin (before total bilirubin)
    if "direct_bili" in raw or "conjugated_bili" in raw or raw in ("dbil", "direct"):
        return "direct_bilirubin"

    # Total Bilirubin
    if "total_bili" in raw or "t_bili" in raw or raw in ("bilirubin", "bili", "tbil"):
        return "total_bilirubin"

    # Alkaline Phosphatase
    if "alkaline" in raw or "alk_phos" in raw or raw in ("alp", "alkphos", "alk"):
        return "alkaline_phosphatase"

    # SGPT / ALT
    if "sgpt" in raw or "alt" in raw or "alanine" in raw:
        return "sgpt"

    # SGOT / AST
    if "sgot" in raw or "ast" in raw or "aspartate" in raw:
        return "sgot"

    # Total Proteins
    if "total_protein" in raw or raw in ("tp", "total_proteins", "proteins", "protein"):
        return "total_proteins"

    # A/G Ratio
    if "a_g" in raw or "ag_ratio" in raw or "albumin_globulin" in raw or "a_to_g" in raw:
        return "a_g_ratio"

    # Albumin (after A/G ratio)
    if "albumin" in raw or raw == "alb":
        return "albumin"

    # Age / Gender
    if raw in ("age", "patient_age"):
        return "age"
    if raw in ("gender", "sex"):
        return "gender"

    return None


class LiverAssessmentState(BaseModel):
    """
    Tracks conversational Liver Disease ML assessment inputs across turns.
    Preserves collected fields, tracks missing inputs, and records validation feedback.
    """
    session_id: str
    patient_id: Optional[int] = None
    status: str = "IN_PROGRESS"  # "IN_PROGRESS", "COMPLETED", "ERROR"
    values: Dict[str, Any] = Field(default_factory=dict)
    validation_errors: List[str] = Field(default_factory=list)
    completed_assessment: Optional[Dict[str, Any]] = None
    last_error: Optional[str] = None

    def get_missing_fields(self) -> List[str]:
        """Returns list of required fields that have not yet been provided."""
        return [f for f in LIVER_REQUIRED_FIELDS if self.values.get(f) is None]

    def get_collected_fields(self) -> Dict[str, Any]:
        """Returns map of already collected and validated fields."""
        return {f: self.values[f] for f in LIVER_REQUIRED_FIELDS if self.values.get(f) is not None}

    def is_complete(self) -> bool:
        """True if all 10 fields are present and there are no active validation errors."""
        return len(self.get_missing_fields()) == 0 and len(self.validation_errors) == 0

    def to_spring_payload(self, patient_id: int) -> Dict[str, Any]:
        """Converts collected fields to Spring Boot LiverPredictRequest camelCase payload."""
        return {
            "patientId": int(patient_id),
            "age": int(self.values["age"]),
            "gender": str(self.values["gender"]),
            "totalBilirubin": float(self.values["total_bilirubin"]),
            "directBilirubin": float(self.values["direct_bilirubin"]),
            "alkalinePhosphatase": float(self.values["alkaline_phosphatase"]),
            "sgpt": float(self.values["sgpt"]),
            "sgot": float(self.values["sgot"]),
            "totalProteins": float(self.values["total_proteins"]),
            "albumin": float(self.values["albumin"]),
            "agRatio": float(self.values["a_g_ratio"])
        }

    def generate_missing_followup(self) -> str:
        """Constructs targeted, non-repetitive clinical follow-up for missing or invalid inputs."""
        missing = self.get_missing_fields()
        collected = self.get_collected_fields()

        sections = []

        if self.validation_errors:
            sections.append("I noticed an issue with the value provided:")
            for err in self.validation_errors:
                sections.append(f"• {err}")
            sections.append("")

        if collected:
            sections.append("Recorded values so far:")
            for k, v in collected.items():
                unit = (
                    " mg/dL" if "bilirubin" in k
                    else " IU/L" if "alkaline" in k
                    else " U/L" if k in ("sgpt", "sgot")
                    else " g/dL" if "protein" in k or k == "albumin"
                    else " years" if k == "age"
                    else ""
                )
                sections.append(f"• {FIELD_LABELS.get(k, k)}: {v}{unit}")
            sections.append("")

        if missing:
            sections.append("To proceed with the Liver Disease screening assessment, I still need:")
            for m in missing:
                sections.append(f"• {FIELD_LABELS.get(m, m)}")
            sections.append("\nYou can provide these laboratory values whenever they are available.")

        return "\n".join(sections)


class LiverExtractor:
    """
    Extracts and validates liver-specific clinical parameters from natural language.
    Never invents, guesses, or uses default values for missing data.
    """

    @staticmethod
    def validate_field(field_name: str, raw_value: Any) -> Tuple[Optional[Any], Optional[str]]:
        """
        Validates individual field format and domain constraints.
        Returns (validated_value, error_message).
        """
        if raw_value is None:
            return None, None

        if field_name == "gender":
            g_str = str(raw_value).strip().lower()
            if g_str in ("male", "man", "m"):
                return "Male", None
            if g_str in ("female", "woman", "f"):
                return "Female", None
            return None, f"Gender must be 'Male' or 'Female' (received: '{raw_value}')."

        if field_name == "age":
            try:
                age_val = int(raw_value)
            except (ValueError, TypeError):
                return None, f"Age must be a valid integer number (received: '{raw_value}')."
            min_val, max_val, err_msg = FIELD_BOUNDS["age"]
            if not (min_val <= age_val <= max_val):
                return None, f"{err_msg} (received: {age_val})."
            return age_val, None

        # Numeric lab fields
        try:
            val = float(raw_value)
        except (ValueError, TypeError):
            label = FIELD_LABELS.get(field_name, field_name)
            return None, f"{label} must be a valid numerical value (received: '{raw_value}')."

        bounds = FIELD_BOUNDS.get(field_name)
        if bounds:
            min_val, max_val, err_msg = bounds
            if not (min_val <= val <= max_val):
                return None, f"{err_msg} (received: {val})."

        return val, None

    def extract_from_text(self, text: str) -> Dict[str, Any]:
        """
        Parses raw text to identify mentioned clinical parameters.
        Does not apply validation filtering here; returns raw extracted values.
        """
        text_lower = text.lower()
        extracted: Dict[str, Any] = {}

        # 1. Age
        # Look for explicit patterns: "age 62", "age: 62", "62 years old", "62 yo", "i am 62 and male"
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

        # Check for explicit invalid age mention like "age: 250" or "age is 0"
        m_age_invalid = re.search(r'\bage\s*(?:is|[:=])?\s*(-?\d+)\b', text_lower)
        if m_age_invalid and 'age' not in extracted:
            extracted['age'] = int(m_age_invalid.group(1))

        # 2. Gender
        if re.search(r'\b(female|woman)\b', text_lower):
            extracted['gender'] = 'Female'
        elif re.search(r'\b(male|man)\b', text_lower):
            extracted['gender'] = 'Male'

        # 3. Direct Bilirubin (must be evaluated before total bilirubin)
        m_dbil = re.search(
            r'\b(?:direct\s+bilirubin|conjugated\s+bilirubin|dbil|direct\s+bili|d[\s\-_]bili|direct)\s*(?:is|[:=])?\s*(-?\d+(?:\.\d+)?)\b(?!\s*(?:-|to)\s*\d)',
            text_lower
        )
        if m_dbil:
            extracted['direct_bilirubin'] = float(m_dbil.group(1))

        # 4. Total Bilirubin
        m_tbil = re.search(r'\b(?:total\s+bilirubin|total\s+bili|t[\s\-_]bili|tbil)\s*(?:is|[:=])?\s*(-?\d+(?:\.\d+)?)\b(?!\s*(?:-|to)\s*\d)', text_lower)
        if not m_tbil:
            # Negative lookbehind to avoid catching "direct bilirubin" or "d bili" as total bilirubin
            m_tbil = re.search(r'(?<!direct\s)(?<!conjugated\s)(?<!d\s)(?<!d_)\bbilirubin\s*(?:is|[:=])?\s*(-?\d+(?:\.\d+)?)\b(?!\s*(?:-|to)\s*\d)', text_lower)
        if m_tbil:
            extracted['total_bilirubin'] = float(m_tbil.group(1))

        # 5. Alkaline Phosphatase
        m_alk = re.search(
            r'\b(?:alkaline\s+phosphatase|alkphos|alk[\s\-_]phos|alp)\s*(?:is|[:=])?\s*(-?\d+(?:\.\d+)?)\b(?!\s*(?:-|to)\s*\d)',
            text_lower
        )
        if m_alk:
            extracted['alkaline_phosphatase'] = float(m_alk.group(1))

        # 6. SGPT / ALT
        m_sgpt = re.search(
            r'\b(?:sgpt|alt|alanine\s+aminotransferase|alanine\s+transaminase)\s*(?:is|[:=])?\s*(-?\d+(?:\.\d+)?)\b(?!\s*(?:-|to)\s*\d)',
            text_lower
        )
        if m_sgpt:
            extracted['sgpt'] = float(m_sgpt.group(1))

        # 7. SGOT / AST
        m_sgot = re.search(
            r'\b(?:sgot|ast|aspartate\s+aminotransferase|aspartate\s+transaminase)\s*(?:is|[:=])?\s*(-?\d+(?:\.\d+)?)\b(?!\s*(?:-|to)\s*\d)',
            text_lower
        )
        if m_sgot:
            extracted['sgot'] = float(m_sgot.group(1))

        # 8. Total Proteins
        m_tp = re.search(r'\b(?:total\s+proteins?|total\s+protein|tp)\s*(?:is|[:=])?\s*(-?\d+(?:\.\d+)?)\b(?!\s*(?:-|to)\s*\d)', text_lower)
        if not m_tp:
            m_tp = re.search(r'(?<!total\s)\bproteins?\s*(?:is|[:=])?\s*(-?\d+(?:\.\d+)?)\b(?!\s*(?:-|to)\s*\d)', text_lower)
        if m_tp:
            extracted['total_proteins'] = float(m_tp.group(1))

        # 9. A/G Ratio
        m_ag = re.search(
            r'\b(?:a[\/\-_:]g\s+ratio|albumin[\s\/\-_:]+globulin\s+ratio|ag\s+ratio|a[\/\-_:]g|albumin[\s\/\-_:]+to[\s\/\-_:]+globulin\s+ratio)\s*(?:is|[:=])?\s*(-?\d+(?:\.\d+)?)\b(?!\s*(?:-|to)\s*\d)',
            text_lower
        )
        if m_ag:
            extracted['a_g_ratio'] = float(m_ag.group(1))

        # 10. Albumin (avoiding matching A/G ratio)
        m_alb = re.search(
            r'(?<!ratio\s)(?<!globulin\s)\balbumin\b(?!\s+(?:and|\/|\-|to)?\s*globulin|\s+ratio)\s*(?:is|[:=])?\s*(-?\d+(?:\.\d+)?)\b(?!\s*(?:-|to)\s*\d)',
            text_lower
        )
        if not m_alb:
            m_alb = re.search(r'\balb\s*(?:is|[:=])?\s*(-?\d+(?:\.\d+)?)\b(?!\s*(?:-|to)\s*\d)', text_lower)
        if m_alb:
            extracted['albumin'] = float(m_alb.group(1))

        return extracted

    def extract_and_validate(
        self,
        text: str,
        measurements: Optional[List[Any]] = None,
        age: Optional[Any] = None,
        sex: Optional[Any] = None
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        Parses text and/or structured measurements and runs validation rules on each candidate value.
        Returns:
            (valid_extracted_dict, list_of_validation_error_messages)
        """
        raw = self.extract_from_text(text)

        if age is not None and "age" not in raw:
            raw["age"] = age

        if sex is not None and "gender" not in raw:
            raw["gender"] = sex

        if measurements:
            for m in measurements:
                m_name = None
                m_val = None
                if hasattr(m, "name") and hasattr(m, "value"):
                    m_name = m.name
                    m_val = m.value
                elif isinstance(m, dict):
                    m_name = m.get("name")
                    m_val = m.get("value")

                if m_name is not None and m_val is not None:
                    canonical = normalize_liver_field_name(str(m_name))
                    if canonical:
                        raw[canonical] = m_val

        valid: Dict[str, Any] = {}
        errors: List[str] = []

        for field, raw_val in raw.items():
            val, err = self.validate_field(field, raw_val)
            if err:
                errors.append(err)
            elif val is not None:
                valid[field] = val

        return valid, errors


def format_liver_prediction_response(prediction: Dict[str, Any]) -> str:
    """
    Constructs a safe, non-diagnostic explanation of the real ML prediction.
    Uses real probability and real SHAP attributions.
    Never asserts a diagnosis, never makes causal claims, never prescribes treatments.
    """
    prob = prediction.get("probability", 0.0)
    prob_pct = f"{float(prob) * 100:.1f}%"
    assessment_id = prediction.get("assessmentId", "N/A")
    model_name = prediction.get("modelName", "Logistic Regression")
    calibration = prediction.get("calibrationMethod", "Isotonic Calibration")
    shap_features = prediction.get("shapFeatures", [])

    lines = [
        "I have all the clinical and laboratory values required for the Liver Disease screening assessment. The production model calculation has completed.",
        "",
        "Assessment Results:",
        f"• Assessment ID: {assessment_id}",
        f"• Model: {model_name} ({calibration})",
        f"• Estimated Risk Probability: {prob_pct}",
        ""
    ]

    if shap_features:
        lines.append("Feature Contributions (SHAP Analysis):")
        # Rank by absolute magnitude of contribution
        sorted_features = sorted(
            shap_features,
            key=lambda x: abs(float(x.get("contribution", 0.0))),
            reverse=True
        )[:5]

        for feat in sorted_features:
            name = feat.get("feature", "Feature")
            val = feat.get("value", "")
            direction = feat.get("direction", "")
            if direction == "INCREASES_RISK":
                dir_desc = "contributed toward a higher screening risk estimation"
            elif direction == "DECREASES_RISK":
                dir_desc = "contributed toward a lower screening risk estimation"
            else:
                dir_desc = "had a neutral effect on the prediction"
            lines.append(f"• {name} ({val}): {dir_desc}")
        lines.append("")

    lines.extend([
        "Clinical Context & Medical Disclaimer:",
        "This result is a calibrated statistical screening risk calculation and is NOT a definitive medical diagnosis. A higher statistical risk does not prove disease, nor does a lower score guarantee absence of liver conditions.",
        "Please share these laboratory results and this assessment summary with a qualified healthcare professional or hepatologist for comprehensive clinical interpretation, diagnosis, and appropriate follow-up care."
    ])

    return "\n".join(lines)
