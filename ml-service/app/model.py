import os
import joblib
from app.explanation import compute_liver_shap_explanation
from app.preprocessing import request_to_dataframe
from app.schemas import (
    LiverPredictionRequest,
    LiverPredictionResponse,
    ModelInfo,
)

RESEARCH_DISCLAIMER = (
    "This is a research-based risk screening tool developed for decision support experimentation. "
    "The model output is a statistical probability estimate and does NOT constitute a clinical diagnosis, "
    "prognosis, or treatment recommendation. Local feature contributions explain model behavior and do "
    "not establish biological causation. Clinical findings must always be evaluated by a qualified physician."
)

_BUNDLE = None


def get_artifact_path() -> str:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, 'artifacts', 'liver', 'liver_model_bundle.joblib')


def load_liver_bundle():
    global _BUNDLE
    if _BUNDLE is None:
        path = get_artifact_path()
        if not os.path.exists(path):
            raise FileNotFoundError(f"Liver model bundle not found at: {path}. Run train_production_artifact.py first.")
        _BUNDLE = joblib.load(path)
    return _BUNDLE


def predict_liver(req: LiverPredictionRequest) -> LiverPredictionResponse:
    bundle = load_liver_bundle()

    calibrated_model = bundle['calibrated_model']
    preprocessor = bundle['preprocessor']
    explainer = bundle['shap_explainer']

    # 1. Map request into DataFrame
    df_sample = request_to_dataframe(req)

    # 2. Inference: predict calibrated probability
    probs = calibrated_model.predict_proba(df_sample)[:, 1]
    prob_val = float(probs[0])

    # 3. Explainability: compute local SHAP contributions
    explanation = compute_liver_shap_explanation(explainer, preprocessor, df_sample)

    # 4. Input summary
    input_summary = {
        'age': req.age,
        'gender': req.gender,
        'totalBilirubin': req.totalBilirubin,
        'directBilirubin': req.directBilirubin,
        'alkalinePhosphatase': req.get_effective_alkphos(),
        'sgpt': req.sgpt,
        'sgot': req.sgot,
        'totalProteins': req.totalProteins,
        'albumin': req.albumin,
        'agRatio': req.agRatio,
    }

    return LiverPredictionResponse(
        disease="LIVER_DISEASE",
        model=ModelInfo(
            name=bundle.get('model_name', 'Logistic Regression (Isotonic Calibrated)'),
            version=bundle.get('model_version', '1.0.0'),
            calibrationMethod="Isotonic Regression (5-fold CV)",
        ),
        probability=round(prob_val, 4),
        explanation=explanation,
        inputSummary=input_summary,
        disclaimer=RESEARCH_DISCLAIMER,
    )
