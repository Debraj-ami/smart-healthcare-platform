"""
Heart Disease SHAP Explainability Module (Phase 8).

Provides reproducible, leakage-free SHAP (SHapley Additive exPlanations) for the
Cleveland Heart Disease finalist models.

Strict Phase Boundaries:
- Explains model mathematical behavior and feature contributions.
- No clinical diagnosis or medical causation claims.
- No modification of Phase 5, Phase 6, or Phase 7 artifacts.
- Exact traceability between raw inputs, transformed features, and SHAP values.
- Explicit definition of the model output space being explained (log-odds vs. probability).
"""

import os
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from app.heart_preprocessing import (
    build_logistic_regression_preprocessor,
    build_tree_preprocessor,
    get_heart_feature_names,
)
from app.heart_experimentation import (
    build_pipeline_from_config,
    get_candidate_configurations,
)
from app.heart_calibration import get_heart_finalist_configs

# Clinical Feature Metadata and Human-Readable Mappings
HEART_FEATURE_DESCRIPTIONS = {
    "num__age": {
        "original_feature": "age",
        "display_name": "Age",
        "description": "Patient age in years",
        "group": "numerical",
    },
    "num__trestbps": {
        "original_feature": "trestbps",
        "display_name": "Resting Blood Pressure",
        "description": "Resting blood pressure (mm Hg on admission)",
        "group": "numerical",
    },
    "num__chol": {
        "original_feature": "chol",
        "display_name": "Serum Cholesterol",
        "description": "Serum cholesterol in mg/dl",
        "group": "numerical",
    },
    "num__thalach": {
        "original_feature": "thalach",
        "display_name": "Maximum Heart Rate",
        "description": "Maximum heart rate achieved during exercise stress test",
        "group": "numerical",
    },
    "num__oldpeak": {
        "original_feature": "oldpeak",
        "display_name": "ST Depression (Oldpeak)",
        "description": "ST depression induced by exercise relative to rest",
        "group": "numerical",
    },
    "bin__sex": {
        "original_feature": "sex",
        "display_name": "Sex (Male)",
        "description": "Biological sex (1 = male, 0 = female)",
        "group": "binary",
    },
    "bin__fbs": {
        "original_feature": "fbs",
        "display_name": "Fasting Blood Sugar > 120",
        "description": "Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)",
        "group": "binary",
    },
    "bin__exang": {
        "original_feature": "exang",
        "display_name": "Exercise-Induced Angina",
        "description": "Exercise-induced angina (1 = yes, 0 = no)",
        "group": "binary",
    },
    "cat__cp_1.0": {
        "original_feature": "cp",
        "display_name": "Chest Pain: Typical Angina",
        "description": "Chest pain type 1: typical angina",
        "group": "categorical",
    },
    "cat__cp_2.0": {
        "original_feature": "cp",
        "display_name": "Chest Pain: Atypical Angina",
        "description": "Chest pain type 2: atypical angina",
        "group": "categorical",
    },
    "cat__cp_3.0": {
        "original_feature": "cp",
        "display_name": "Chest Pain: Non-Anginal",
        "description": "Chest pain type 3: non-anginal pain",
        "group": "categorical",
    },
    "cat__cp_4.0": {
        "original_feature": "cp",
        "display_name": "Chest Pain: Asymptomatic",
        "description": "Chest pain type 4: asymptomatic ischaemia",
        "group": "categorical",
    },
    "cat__restecg_0.0": {
        "original_feature": "restecg",
        "display_name": "Resting ECG: Normal",
        "description": "Resting electrocardiogram: normal",
        "group": "categorical",
    },
    "cat__restecg_1.0": {
        "original_feature": "restecg",
        "display_name": "Resting ECG: ST-T Abnormality",
        "description": "Resting ECG: ST-T wave abnormality",
        "group": "categorical",
    },
    "cat__restecg_2.0": {
        "original_feature": "restecg",
        "display_name": "Resting ECG: LV Hypertrophy",
        "description": "Resting ECG: probable or definite left ventricular hypertrophy",
        "group": "categorical",
    },
    "cat__slope_1.0": {
        "original_feature": "slope",
        "display_name": "ST Slope: Upsloping",
        "description": "Slope of peak exercise ST segment: upsloping",
        "group": "categorical",
    },
    "cat__slope_2.0": {
        "original_feature": "slope",
        "display_name": "ST Slope: Flat",
        "description": "Slope of peak exercise ST segment: flat",
        "group": "categorical",
    },
    "cat__slope_3.0": {
        "original_feature": "slope",
        "display_name": "ST Slope: Downsloping",
        "description": "Slope of peak exercise ST segment: downsloping",
        "group": "categorical",
    },
    "cat__thal_3.0": {
        "original_feature": "thal",
        "display_name": "Thallium: Normal",
        "description": "Thallium scintigraphy defect: normal blood flow",
        "group": "categorical",
    },
    "cat__thal_6.0": {
        "original_feature": "thal",
        "display_name": "Thallium: Fixed Defect",
        "description": "Thallium scintigraphy defect: fixed defect (previous infarction)",
        "group": "categorical",
    },
    "cat__thal_7.0": {
        "original_feature": "thal",
        "display_name": "Thallium: Reversible Defect",
        "description": "Thallium scintigraphy defect: reversible defect (ischemia)",
        "group": "categorical",
    },
    "count__ca": {
        "original_feature": "ca",
        "display_name": "Fluoroscopy Vessels (ca)",
        "description": "Number of major vessels (0-3) colored by fluoroscopy",
        "group": "count",
    },
}


def get_heart_feature_mapping(
    model_family: str = "LogisticRegression",
    preprocessor: Optional[ColumnTransformer] = None,
) -> pd.DataFrame:
    """
    Returns a DataFrame documenting the mapping from raw inputs to transformed features
    and their clinical display labels.
    """
    if preprocessor is None:
        from app.heart_preprocessing import load_heart_raw_data, create_modeling_dataset, split_heart_data
        df_raw = load_heart_raw_data()
        X, y = create_modeling_dataset(df_raw)
        X_train, _, _, _ = split_heart_data(X, y, test_size=0.20, random_state=42)
        if model_family == "LogisticRegression":
            preprocessor = build_logistic_regression_preprocessor(scaler="robust", drop="first")
        else:
            preprocessor = build_tree_preprocessor(drop=None)
        preprocessor.fit(X_train)
        
    feature_names = get_heart_feature_names(preprocessor)
    
    records = []
    for idx, fn in enumerate(feature_names, 1):
        meta = HEART_FEATURE_DESCRIPTIONS.get(fn, {
            "original_feature": fn.split("__")[-1],
            "display_name": fn,
            "description": "Transformed feature",
            "group": fn.split("__")[0] if "__" in fn else "unknown",
        })
        records.append({
            "feature_index": idx,
            "transformed_feature": fn,
            "original_feature": meta["original_feature"],
            "display_name": meta["display_name"],
            "feature_group": meta["group"],
            "description": meta["description"],
        })
        
    return pd.DataFrame(records)


def fit_model_and_preprocessor_for_shap(
    config: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Tuple[Any, ColumnTransformer, np.ndarray, List[str]]:
    """
    Fits the preprocessor and standalone estimator on training data for SHAP analysis.
    
    Returns:
      (fitted_estimator, fitted_preprocessor, X_train_transformed, feature_names)
    """
    family = config["family"]
    if family == "LogisticRegression":
        preprocessor = build_logistic_regression_preprocessor(
            scaler=config["scaler"],
            drop="first"
        )
    else:
        preprocessor = build_tree_preprocessor(drop=None)
        
    X_train_trans = preprocessor.fit_transform(X_train)
    feature_names = get_heart_feature_names(preprocessor)
    
    # Instantiate the base estimator with identical parameters
    pipeline = build_pipeline_from_config(config)
    estimator = pipeline.named_steps["classifier"]
    estimator.fit(X_train_trans, y_train)
    
    return estimator, preprocessor, X_train_trans, feature_names


def create_heart_shap_explainer(
    estimator: Any,
    X_train_trans: np.ndarray,
    model_family: str = "LogisticRegression",
) -> Any:
    """
    Constructs the appropriate SHAP explainer for the given model architecture.
    
    - LogisticRegression: shap.LinearExplainer in log-odds decision_function space
      with shap.maskers.Independent over the full training distribution.
    - RandomForest: shap.TreeExplainer in probability space (class 1).
    - XGBoost: shap.TreeExplainer in raw margin (log-odds) space.
    """
    if model_family == "LogisticRegression":
        masker = shap.maskers.Independent(X_train_trans, max_samples=len(X_train_trans))
        explainer = shap.LinearExplainer(estimator, masker)
    elif model_family in ("RandomForest", "XGBoost"):
        explainer = shap.TreeExplainer(estimator)
    else:
        raise ValueError(f"Unsupported model_family for SHAP: {model_family}")
        
    return explainer


def compute_shap_explanations(
    explainer: Any,
    preprocessor: ColumnTransformer,
    estimator: Any,
    X_eval: pd.DataFrame,
    model_family: str = "LogisticRegression",
) -> Dict[str, Any]:
    """
    Computes SHAP values, base values, and model outputs for a dataset.
    
    Returns:
      {
        'shap_values': np.ndarray,
        'base_values': np.ndarray or float,
        'feature_names': List[str],
        'X_transformed': np.ndarray,
        'model_outputs': np.ndarray,
        'predicted_probabilities': np.ndarray,
        'output_space': str,
      }
    """
    X_trans = preprocessor.transform(X_eval)
    feature_names = get_heart_feature_names(preprocessor)
    
    raw_probs = estimator.predict_proba(X_trans)[:, 1]
    
    if model_family == "LogisticRegression":
        explanation = explainer(X_trans)
        shap_vals = explanation.values
        base_vals = explanation.base_values
        raw_outputs = estimator.decision_function(X_trans)
        output_space = "log-odds"
    elif model_family == "RandomForest":
        explanation = explainer(X_trans)
        # TreeExplainer on RF outputs (N, n_features, 2) for binary classification
        shap_vals = explanation.values[:, :, 1]
        base_vals = explanation.base_values[:, 1]
        raw_outputs = raw_probs
        output_space = "probability"
    elif model_family == "XGBoost":
        explanation = explainer(X_trans)
        shap_vals = explanation.values
        base_vals = explanation.base_values
        raw_outputs = estimator.predict(X_trans, output_margin=True)
        output_space = "log-odds-margin"
    else:
        raise ValueError(f"Unknown family: {model_family}")
        
    return {
        "shap_values": shap_vals,
        "base_values": base_vals,
        "feature_names": feature_names,
        "X_transformed": X_trans,
        "model_outputs": raw_outputs,
        "predicted_probabilities": raw_probs,
        "output_space": output_space,
    }


def compute_global_feature_importance(
    shap_values: np.ndarray,
    feature_names: List[str],
    mapping_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Calculates mean absolute SHAP value across all observations for global importance ranking.
    """
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    total_shap = np.sum(mean_abs_shap)
    pct_importance = (mean_abs_shap / total_shap) * 100.0 if total_shap > 0 else np.zeros_like(mean_abs_shap)
    
    records = []
    for fn, val, pct in zip(feature_names, mean_abs_shap, pct_importance):
        display_name = fn
        orig_feature = fn.split("__")[-1]
        group = fn.split("__")[0] if "__" in fn else "unknown"
        desc = "Transformed feature"
        
        if mapping_df is not None:
            m_row = mapping_df[mapping_df["transformed_feature"] == fn]
            if len(m_row) > 0:
                display_name = m_row["display_name"].values[0]
                orig_feature = m_row["original_feature"].values[0]
                group = m_row["feature_group"].values[0]
                desc = m_row["description"].values[0]
                
        records.append({
            "transformed_feature": fn,
            "display_name": display_name,
            "original_feature": orig_feature,
            "feature_group": group,
            "mean_abs_shap": round(float(val), 4),
            "pct_importance": round(float(pct), 2),
            "description": desc,
        })
        
    df_imp = pd.DataFrame(records)
    df_imp.sort_values(by="mean_abs_shap", ascending=False, inplace=True)
    df_imp.reset_index(drop=True, inplace=True)
    df_imp["global_rank"] = np.arange(1, len(df_imp) + 1)
    return df_imp


def explain_single_observation(
    explainer: Any,
    preprocessor: ColumnTransformer,
    estimator: Any,
    raw_sample: pd.Series,
    model_family: str = "LogisticRegression",
    top_n: int = 5,
) -> Dict[str, Any]:
    """
    Computes local feature contributions for an individual patient observation.
    
    Returns structured positive and negative contributors ordered by absolute magnitude.
    """
    df_sample = pd.DataFrame([raw_sample])
    exp_dict = compute_shap_explanations(
        explainer, preprocessor, estimator, df_sample, model_family=model_family
    )
    
    shap_vals = exp_dict["shap_values"][0]
    base_val = float(exp_dict["base_values"][0]) if isinstance(exp_dict["base_values"], np.ndarray) else float(exp_dict["base_values"])
    model_out = float(exp_dict["model_outputs"][0])
    prob = float(exp_dict["predicted_probabilities"][0])
    feature_names = exp_dict["feature_names"]
    
    mapping_df = get_heart_feature_mapping(model_family)
    
    contributions = []
    for fn, s_val in zip(feature_names, shap_vals):
        val = float(s_val)
        m_row = mapping_df[mapping_df["transformed_feature"] == fn]
        d_name = m_row["display_name"].values[0] if len(m_row) > 0 else fn
        orig_name = m_row["original_feature"].values[0] if len(m_row) > 0 else fn
        raw_val = raw_sample.get(orig_name, np.nan)
        
        # Contribution direction
        if val > 0.005:
            direction = "INCREASES_RISK_ESTIMATE"
        elif val < -0.005:
            direction = "DECREASES_RISK_ESTIMATE"
        else:
            direction = "NEUTRAL"
            
        contributions.append({
            "transformed_feature": fn,
            "display_name": d_name,
            "original_feature": orig_name,
            "raw_value": raw_val,
            "shap_contribution": round(val, 4),
            "abs_contribution": round(abs(val), 4),
            "direction": direction,
        })
        
    # Sort contributions by absolute magnitude
    contributions.sort(key=lambda x: x["abs_contribution"], reverse=True)
    
    top_positive = [c for c in contributions if c["shap_contribution"] > 0][:top_n]
    top_negative = [c for c in contributions if c["shap_contribution"] < 0][:top_n]
    
    return {
        "model_family": model_family,
        "output_space": exp_dict["output_space"],
        "base_value": round(base_val, 4),
        "model_output": round(model_out, 4),
        "model_estimated_probability": round(prob, 4),
        "top_positive_contributors": top_positive,
        "top_negative_contributors": top_negative,
        "all_contributions": contributions,
    }


def generate_representative_local_explanations(
    explainer: Any,
    preprocessor: ColumnTransformer,
    estimator: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_family: str = "LogisticRegression",
) -> pd.DataFrame:
    """
    Selects real representative cases from the locked holdout test set:
    - Case 1: True Positive (High-confidence Heart Disease case, true=1, prob > 0.85)
    - Case 2: True Negative (High-confidence Control case, true=0, prob < 0.15)
    - Case 3: Borderline Case (prob ~ 0.50)
    - Case 4: False Positive (Control case where model predicts high prob, true=0, prob > 0.50)
    - Case 5: False Negative (Heart Disease case where model predicts low prob, true=1, prob < 0.50)
    """
    X_trans = preprocessor.transform(X_test)
    test_probs = estimator.predict_proba(X_trans)[:, 1]
    y_test_arr = y_test.values
    test_indices = X_test.index.values
    
    records = []
    
    # 1. High-confidence True Positive
    tp_candidates = np.where((y_test_arr == 1) & (test_probs >= 0.80))[0]
    tp_idx = tp_candidates[np.argmax(test_probs[tp_candidates])] if len(tp_candidates) > 0 else np.argmax(test_probs)
    
    # 2. High-confidence True Negative
    tn_candidates = np.where((y_test_arr == 0) & (test_probs <= 0.20))[0]
    tn_idx = tn_candidates[np.argmin(test_probs[tn_candidates])] if len(tn_candidates) > 0 else np.argmin(test_probs)
    
    # 3. Borderline Case (closest to 0.50)
    border_idx = np.argmin(np.abs(test_probs - 0.50))
    
    # 4. False Positive (if any)
    fp_candidates = np.where((y_test_arr == 0) & (test_probs >= 0.50))[0]
    fp_idx = fp_candidates[np.argmax(test_probs[fp_candidates])] if len(fp_candidates) > 0 else None
    
    # 5. False Negative (if any)
    fn_candidates = np.where((y_test_arr == 1) & (test_probs < 0.50))[0]
    fn_idx = fn_candidates[np.argmin(test_probs[fn_candidates])] if len(fn_candidates) > 0 else None
    
    case_indices = [
        ("High-Confidence True Positive", tp_idx),
        ("High-Confidence True Negative", tn_idx),
        ("Borderline Prediction", border_idx),
    ]
    if fp_idx is not None and fp_idx not in [tp_idx, tn_idx, border_idx]:
        case_indices.append(("False Positive Discordance", fp_idx))
    if fn_idx is not None and fn_idx not in [tp_idx, tn_idx, border_idx]:
        case_indices.append(("False Negative Discordance", fn_idx))
        
    for case_label, c_idx in case_indices:
        raw_row = X_test.iloc[c_idx]
        sample_id = int(test_indices[c_idx])
        true_lbl = int(y_test_arr[c_idx])
        
        explanation = explain_single_observation(
            explainer, preprocessor, estimator, raw_row, model_family=model_family, top_n=3
        )
        
        # Format top 3 positive and negative features for tabular display
        pos_str = "; ".join([f"{c['display_name']} ({c['shap_contribution']:+.3f})" for c in explanation["top_positive_contributors"]])
        neg_str = "; ".join([f"{c['display_name']} ({c['shap_contribution']:+.3f})" for c in explanation["top_negative_contributors"]])
        
        records.append({
            "case_type": case_label,
            "sample_index": sample_id,
            "true_label": true_lbl,
            "model_estimated_probability": explanation["model_estimated_probability"],
            "base_value": explanation["base_value"],
            "model_output": explanation["model_output"],
            "top_positive_contributors": pos_str,
            "top_negative_contributors": neg_str,
            "age": raw_row["age"],
            "sex": "Male" if raw_row["sex"] == 1.0 else "Female",
            "cp": raw_row["cp"],
            "thalach": raw_row["thalach"],
            "oldpeak": raw_row["oldpeak"],
            "ca": raw_row["ca"],
            "thal": raw_row["thal"],
        })
        
    return pd.DataFrame(records)
