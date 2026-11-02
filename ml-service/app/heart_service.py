"""
Heart Disease Production Inference Service.

Handles:
- Loading and caching the serialized Heart model bundle into memory once at startup
- Running deterministic preprocessed inference with Logistic Regression (Robust L1)
- Evaluating binary classification decision against the configured threshold (t=0.31)
- Computing local SHAP feature attributions in log-odds space using LinearExplainer
- Safe fallback when SHAP fails without fabricating explanation values
- Returning structured, machine-readable explanation data mapped to clinical descriptors
"""

import os
import math
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

from app.schemas import (
    HeartPredictionRequest,
    HeartPredictionResponse,
    HeartModelInfo,
    HeartExplanationOutput,
    HeartFeatureContribution,
)
from app.heart_preprocessing import FEATURE_COLUMNS
from app.heart_shap import HEART_FEATURE_DESCRIPTIONS

HEART_RESEARCH_DISCLAIMER = (
    "This is a research-based risk screening tool developed for decision support experimentation. "
    "The model output is a statistical probability estimate and does NOT constitute a clinical diagnosis, "
    "prognosis, or definitive proof of coronary artery disease. The configured classification threshold "
    "(t=0.31) is an operational sensitivity-focused screening cutoff, not a medical diagnostic threshold. "
    "SHAP feature attributions explain model mathematical behavior in log-odds space and do not establish "
    "biological or medical causation. Clinical decisions must always be made by a qualified healthcare professional."
)

_HEART_BUNDLE: Optional[Dict[str, Any]] = None


def get_heart_artifact_path() -> str:
    """Returns the absolute path to the production Heart model bundle."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, "artifacts", "heart", "heart_model_bundle.joblib")


def load_heart_bundle() -> Dict[str, Any]:
    """
    Loads and caches the serialized Heart model bundle into memory.
    Ensures artifact is loaded once rather than reloading per request.
    """
    global _HEART_BUNDLE
    if _HEART_BUNDLE is None:
        path = get_heart_artifact_path()
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Heart model bundle not found at: {path}. "
                "Run train_heart_production_artifact.py to generate the production bundle."
            )
        bundle = joblib.load(path)
        required_keys = ["preprocessor", "classifier", "shap_explainer", "threshold"]
        for k in required_keys:
            if k not in bundle:
                raise ValueError(f"Corrupted Heart model bundle: missing key '{k}' in {path}.")
        _HEART_BUNDLE = bundle
    return _HEART_BUNDLE


def request_to_heart_dataframe(req: HeartPredictionRequest) -> pd.DataFrame:
    """
    Converts a validated HeartPredictionRequest into a single-row pandas DataFrame
    with exact column order and types conforming to the Phase 4 preprocessing contract.
    """
    row_dict = {
        "age": float(req.age),
        "sex": float(req.sex),
        "cp": float(req.cp),
        "trestbps": float(req.trestbps),
        "chol": float(req.chol),
        "fbs": float(req.fbs),
        "restecg": float(req.restecg),
        "thalach": float(req.thalach),
        "exang": float(req.exang),
        "oldpeak": float(req.oldpeak),
        "slope": float(req.slope),
        "ca": float(req.ca) if req.ca is not None else np.nan,
        "thal": float(req.thal) if req.thal is not None else np.nan,
    }
    df = pd.DataFrame([row_dict], columns=FEATURE_COLUMNS)
    return df


def predict_heart(req: HeartPredictionRequest) -> HeartPredictionResponse:
    """
    Executes inference for a validated Heart prediction request:
    1. Loads cached model bundle.
    2. Maps request to pandas DataFrame with exact feature order.
    3. Transforms features through fitted RobustScaler + OneHotEncoder preprocessor.
    4. Computes model log-odds output (decision_function) and model-estimated probability (sigmoid).
    5. Computes binary classification decision via the configured threshold (t=0.31).
    6. Computes local SHAP attributions in log-odds space using the cached LinearExplainer.
    7. Returns versioned, structured response with full explanation and safety disclaimers.
    """
    bundle = load_heart_bundle()

    preprocessor = bundle["preprocessor"]
    clf = bundle["classifier"]
    explainer = bundle["shap_explainer"]
    threshold = float(bundle.get("threshold", 0.31))
    threshold_desc = bundle.get(
        "threshold_description",
        "Configured model classification threshold (t=0.31) targeting >= 90% sensitivity established in Phase 6.",
    )

    # 1. Transform single sample
    df_sample = request_to_heart_dataframe(req)
    X_trans = preprocessor.transform(df_sample)

    # 2. Probability & Decision Function
    decision_val = float(clf.decision_function(X_trans)[0])
    prob_val = float(clf.predict_proba(X_trans)[0, 1])

    # 3. Binary decision via configured threshold
    binary_pred = 1 if prob_val >= threshold else 0

    # 4. SHAP Local Explanation (isolated with safe failure handling)
    transformed_feature_names = bundle.get("feature_names_transformed")
    if transformed_feature_names is None:
        from app.heart_preprocessing import get_heart_feature_names
        transformed_feature_names = get_heart_feature_names(preprocessor)

    explanation_output = None
    try:
        exp = explainer(X_trans)
        shap_vals = exp.values[0]
        base_val = float(exp.base_values[0]) if isinstance(exp.base_values, (np.ndarray, list)) else float(exp.base_values)

        contributions: List[HeartFeatureContribution] = []
        for fn, s_val in zip(transformed_feature_names, shap_vals):
            c_val = float(s_val)
            meta = HEART_FEATURE_DESCRIPTIONS.get(fn, {
                "original_feature": fn.split("__")[-1] if "__" in fn else fn,
                "display_name": fn,
            })
            orig_feature = meta.get("original_feature", fn)
            disp_name = meta.get("display_name", fn)
            raw_val = getattr(req, orig_feature, None)

            if c_val > 0.005:
                direction = "INCREASES_RISK_ESTIMATE"
            elif c_val < -0.005:
                direction = "DECREASES_RISK_ESTIMATE"
            else:
                direction = "NEUTRAL"

            contributions.append(
                HeartFeatureContribution(
                    feature=orig_feature,
                    transformed_feature=fn,
                    display_name=disp_name,
                    raw_value=raw_val,
                    contribution=round(c_val, 4),
                    abs_contribution=round(abs(c_val), 4),
                    direction=direction,
                )
            )

        # Rank all contributions by absolute attribution magnitude
        contributions.sort(key=lambda x: x.absContribution, reverse=True)

        positive_contributors = [c for c in contributions if c.contribution > 0]
        negative_contributors = [c for c in contributions if c.contribution < 0]

        explanation_output = HeartExplanationOutput(
            available=True,
            output_space="log_odds",
            base_value=round(base_val, 4),
            model_output=round(decision_val, 4),
            positive_contributors=positive_contributors[:5],
            negative_contributors=negative_contributors[:5],
            all_contributions=contributions,
            error=None,
        )
    except Exception as e:
        # Prediction succeeded, but explanation failed: safely report failure without fabricating numbers
        explanation_output = HeartExplanationOutput(
            available=False,
            output_space="log_odds",
            base_value=0.0,
            model_output=round(decision_val, 4),
            positive_contributors=[],
            negative_contributors=[],
            all_contributions=[],
            error=f"SHAP explanation calculation failed: {str(e)}",
        )

    # 5. Input summary
    input_summary = {
        "age": req.age,
        "sex": req.sex,
        "cp": req.cp,
        "trestbps": req.trestbps,
        "chol": req.chol,
        "fbs": req.fbs,
        "restecg": req.restecg,
        "thalach": req.thalach,
        "exang": req.exang,
        "oldpeak": req.oldpeak,
        "slope": req.slope,
        "ca": req.ca,
        "thal": req.thal,
    }

    return HeartPredictionResponse(
        disease="HEART_DISEASE",
        model=HeartModelInfo(
            name=bundle.get("model_name", "Logistic Regression (Robust L1)"),
            version=bundle.get("model_version", "1.0.0"),
            config_id="LR_robust_l1_C1.0",
            output_space="log_odds",
            calibration_status=bundle.get("calibration_method", "Uncalibrated Baseline (Empirically Calibrated)"),
        ),
        probability=round(prob_val, 4),
        prediction=binary_pred,
        threshold=round(threshold, 4),
        threshold_description=threshold_desc,
        explanation=explanation_output,
        input_summary=input_summary,
        disclaimer=HEART_RESEARCH_DISCLAIMER,
    )
