from typing import List, Optional, Union, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator, model_validator


class LiverPredictionRequest(BaseModel):
    age: int = Field(..., ge=1, le=120, description="Age of the patient in years (1-120)")
    gender: Literal["Male", "Female"] = Field(..., description="Gender: Male or Female")
    totalBilirubin: float = Field(..., gt=0, le=100.0, description="Total Bilirubin in mg/dL")
    directBilirubin: float = Field(..., ge=0, le=50.0, description="Direct Bilirubin in mg/dL")
    alkphos: Optional[float] = Field(None, gt=0, le=5000.0, description="Alkaline Phosphatase in IU/L")
    alkalinePhosphatase: Optional[float] = Field(None, gt=0, le=5000.0, description="Alias for Alkaline Phosphatase")
    sgpt: float = Field(..., gt=0, le=5000.0, description="SGPT (ALT) in U/L")
    sgot: float = Field(..., gt=0, le=5000.0, description="SGOT (AST) in U/L")
    totalProteins: float = Field(..., gt=0, le=20.0, description="Total Proteins in g/dL")
    albumin: float = Field(..., gt=0, le=15.0, description="Albumin in g/dL")
    agRatio: float = Field(..., gt=0, le=10.0, description="Albumin and Globulin Ratio")

    @model_validator(mode='before')
    @classmethod
    def reconcile_alkphos_and_gender(cls, values: Any) -> Any:
        if isinstance(values, dict):
            # Reconcile alkaline phosphatase field aliases
            if 'alkphos' not in values or values['alkphos'] is None:
                if 'alkalinePhosphatase' in values and values['alkalinePhosphatase'] is not None:
                    values['alkphos'] = values['alkalinePhosphatase']
            elif 'alkalinePhosphatase' not in values or values['alkalinePhosphatase'] is None:
                values['alkalinePhosphatase'] = values['alkphos']

            if values.get('alkphos') is None and values.get('alkalinePhosphatase') is None:
                raise ValueError("Alkaline Phosphatase ('alkphos' or 'alkalinePhosphatase') is required")

            # Normalize gender case if string provided
            if 'gender' in values and isinstance(values['gender'], str):
                g_str = values['gender'].strip().capitalize()
                if g_str in ['Male', 'Female']:
                    values['gender'] = g_str
                else:
                    raise ValueError("Gender must be 'Male' or 'Female'")
        return values

    def get_effective_alkphos(self) -> float:
        return float(self.alkphos if self.alkphos is not None else self.alkalinePhosphatase)


class FeatureContribution(BaseModel):
    feature: str = Field(..., description="Human-readable feature name")
    transformedFeature: str = Field(..., description="Internal pipeline feature name")
    value: Union[float, int, str] = Field(..., description="Submitted patient measurement value")
    contribution: float = Field(..., description="SHAP attribution value in log-odds space")
    direction: Literal["INCREASES_RISK", "DECREASES_RISK", "NEUTRAL"] = Field(
        ..., description="Direction of contribution towards liver disease risk"
    )


class ExplanationOutput(BaseModel):
    baseValue: float = Field(..., description="Baseline log-odds reference value (expected model output)")
    outputSpace: str = Field(default="log-odds", description="Representation space of SHAP values")
    features: List[FeatureContribution] = Field(..., description="Ranked local feature contributions")


class ModelInfo(BaseModel):
    name: str
    version: str
    calibrationMethod: str


class LiverPredictionResponse(BaseModel):
    disease: str = "LIVER_DISEASE"
    model: ModelInfo
    probability: float = Field(..., ge=0.0, le=1.0, description="Estimated calibrated probability of liver disease")
    explanation: ExplanationOutput
    inputSummary: Dict[str, Any]
    disclaimer: str


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    modelsLoaded: List[str]


# =====================================================================
# HEART DISEASE SCHEMAS (Productionization Stage 1)
# =====================================================================

class HeartPredictionRequest(BaseModel):
    age: float = Field(..., ge=1.0, le=120.0, description="Age of the patient in years (1-120)")
    sex: float = Field(..., description="Biological sex (1.0 = Male, 0.0 = Female)")
    cp: float = Field(..., description="Chest pain type (1.0: typical, 2.0: atypical, 3.0: non-anginal, 4.0: asymptomatic)")
    trestbps: float = Field(..., ge=50.0, le=300.0, description="Resting blood pressure in mm Hg on admission (50-300)")
    chol: float = Field(..., ge=50.0, le=800.0, description="Serum cholesterol in mg/dl (50-800)")
    fbs: float = Field(..., description="Fasting blood sugar > 120 mg/dl (1.0 = true, 0.0 = false)")
    restecg: float = Field(..., description="Resting ECG results (0.0: normal, 1.0: ST-T wave abnormality, 2.0: LV hypertrophy)")
    thalach: float = Field(..., ge=50.0, le=250.0, description="Maximum heart rate achieved during exercise stress test (50-250)")
    exang: float = Field(..., description="Exercise induced angina (1.0 = yes, 0.0 = no)")
    oldpeak: float = Field(..., ge=0.0, le=15.0, description="ST depression induced by exercise relative to rest (0.0-15.0)")
    slope: float = Field(..., description="Slope of peak exercise ST segment (1.0: upsloping, 2.0: flat, 3.0: downsloping)")
    ca: Optional[float] = Field(None, description="Number of major vessels (0.0-3.0) colored by fluoroscopy; None allowed per Phase 4 contract")
    thal: Optional[float] = Field(None, description="Thallium scintigraphy defect (3.0: normal, 6.0: fixed, 7.0: reversible); None allowed per Phase 4 contract")

    @model_validator(mode='before')
    @classmethod
    def validate_request_payload(cls, values: Any) -> Any:
        if isinstance(values, dict):
            # 1. Reject target variable 'num'
            if 'num' in values:
                raise ValueError("Target variable 'num' is not accepted in prediction requests.")

            # 2. Reject NaN and infinite values
            import math
            for k, v in values.items():
                if isinstance(v, (int, float)):
                    if math.isnan(v) or math.isinf(v):
                        raise ValueError(f"Feature '{k}' contains invalid NaN or infinite value.")

            # 3. Validate discrete categorical sets
            if 'sex' in values and values['sex'] not in [0, 1, 0.0, 1.0]:
                raise ValueError("Feature 'sex' must be 0 (Female) or 1 (Male).")
            if 'cp' in values and values['cp'] not in [1, 2, 3, 4, 1.0, 2.0, 3.0, 4.0]:
                raise ValueError("Feature 'cp' must be 1.0, 2.0, 3.0, or 4.0.")
            if 'fbs' in values and values['fbs'] not in [0, 1, 0.0, 1.0]:
                raise ValueError("Feature 'fbs' must be 0 (<=120 mg/dl) or 1 (>120 mg/dl).")
            if 'restecg' in values and values['restecg'] not in [0, 1, 2, 0.0, 1.0, 2.0]:
                raise ValueError("Feature 'restecg' must be 0.0, 1.0, or 2.0.")
            if 'exang' in values and values['exang'] not in [0, 1, 0.0, 1.0]:
                raise ValueError("Feature 'exang' must be 0 (No) or 1 (Yes).")
            if 'slope' in values and values['slope'] not in [1, 2, 3, 1.0, 2.0, 3.0]:
                raise ValueError("Feature 'slope' must be 1.0, 2.0, or 3.0.")
            if 'ca' in values and values['ca'] is not None and values['ca'] not in [0, 1, 2, 3, 0.0, 1.0, 2.0, 3.0]:
                raise ValueError("Feature 'ca' must be None, 0.0, 1.0, 2.0, or 3.0.")
            if 'thal' in values and values['thal'] is not None and values['thal'] not in [3, 6, 7, 3.0, 6.0, 7.0]:
                raise ValueError("Feature 'thal' must be None, 3.0 (normal), 6.0 (fixed), or 7.0 (reversible).")

        return values


class HeartFeatureContribution(BaseModel):
    feature: str = Field(..., description="Original clinical feature identifier (e.g. 'ca', 'thal')")
    transformedFeature: str = Field(..., alias="transformed_feature", description="Internal pipeline feature identifier")
    displayName: str = Field(..., alias="display_name", description="Human-readable clinical display label")
    rawValue: Optional[Union[float, int, str]] = Field(None, alias="raw_value", description="Submitted patient measurement value")
    contribution: float = Field(..., description="SHAP attribution value in log-odds space")
    absContribution: float = Field(..., alias="abs_contribution", description="Absolute SHAP attribution magnitude")
    direction: Literal["INCREASES_RISK_ESTIMATE", "DECREASES_RISK_ESTIMATE", "NEUTRAL"] = Field(
        ..., description="Direction of contribution towards model log-odds output"
    )

    model_config = {"populate_by_name": True}


class HeartExplanationOutput(BaseModel):
    available: bool = Field(True, description="Whether SHAP explanation was successfully computed")
    outputSpace: str = Field("log_odds", alias="output_space", description="Model output space explained by SHAP")
    baseValue: float = Field(..., alias="base_value", description="Expected baseline log-odds reference value")
    modelOutput: float = Field(..., alias="model_output", description="Raw model decision function / log-odds output")
    positiveContributors: List[HeartFeatureContribution] = Field(
        default_factory=list, alias="positive_contributors", description="Top contributors increasing log-odds toward positive class"
    )
    negativeContributors: List[HeartFeatureContribution] = Field(
        default_factory=list, alias="negative_contributors", description="Top contributors decreasing log-odds toward negative class"
    )
    allContributions: List[HeartFeatureContribution] = Field(
        default_factory=list, alias="all_contributions", description="All transformed feature contributions ranked by absolute magnitude"
    )
    error: Optional[str] = Field(None, description="Explanation error message if computation failed")

    model_config = {"populate_by_name": True}


class HeartModelInfo(BaseModel):
    name: str = "Logistic Regression (Robust L1)"
    version: str = "1.0.0"
    configId: str = Field("LR_robust_l1_C1.0", alias="config_id")
    outputSpace: str = Field("log_odds", alias="output_space")
    calibrationStatus: str = Field(
        "Uncalibrated Baseline (Empirically Calibrated)",
        alias="calibration_status",
    )

    model_config = {"populate_by_name": True}


class HeartPredictionResponse(BaseModel):
    disease: str = "HEART_DISEASE"
    model: HeartModelInfo
    probability: float = Field(..., ge=0.0, le=1.0, description="Model-estimated probability of heart disease")
    prediction: int = Field(..., ge=0, le=1, description="Binary classification decision (0: Negative, 1: Positive) using configured threshold")
    threshold: float = Field(..., ge=0.0, le=1.0, description="Configured model classification threshold")
    thresholdDescription: str = Field(..., alias="threshold_description", description="Operational definition and context of the classification threshold")
    explanation: HeartExplanationOutput
    inputSummary: Dict[str, Any] = Field(..., alias="input_summary")
    disclaimer: str

    model_config = {"populate_by_name": True}

