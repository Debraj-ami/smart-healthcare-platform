from typing import List
import numpy as np
import pandas as pd
from app.preprocessing import HUMAN_READABLE_NAMES
from app.schemas import ExplanationOutput, FeatureContribution


def compute_liver_shap_explanation(
    explainer,
    preprocessor,
    df_sample: pd.DataFrame
) -> ExplanationOutput:
    """
    Generate local SHAP explanations for a single patient sample using the fitted
    LinearExplainer in log-odds space, matching Phase 8 research methodology.
    """
    # 1. Transform raw sample using fitted ColumnTransformer
    X_sample_transformed = preprocessor.transform(df_sample)
    feature_names = preprocessor.get_feature_names_out().tolist()

    # 2. Compute analytical Shapley values via LinearExplainer
    explanation = explainer(X_sample_transformed)
    shap_vals = explanation.values[0]
    base_val = float(explanation.base_values[0])

    # 3. Construct structured local contributions
    contributions: List[FeatureContribution] = []

    # Map raw value by feature
    raw_val_map = {
        'num__Age': df_sample['Age'].iloc[0],
        'num__Total_Bilirubin': df_sample['Total_Bilirubin'].iloc[0],
        'num__Direct_Bilirubin': df_sample['Direct_Bilirubin'].iloc[0],
        'num__Alkphos': df_sample['Alkphos'].iloc[0],
        'num__Sgpt': df_sample['Sgpt'].iloc[0],
        'num__Sgot': df_sample['Sgot'].iloc[0],
        'num__Total_Proteins': df_sample['Total_Proteins'].iloc[0],
        'num__Albumin': df_sample['Albumin'].iloc[0],
        'num__A_G_Ratio': df_sample['A_G_Ratio'].iloc[0],
        'cat__Gender_Male': df_sample['Gender'].iloc[0],
    }

    for name, shap_val in zip(feature_names, shap_vals):
        val = float(shap_val)
        human_name = HUMAN_READABLE_NAMES.get(name, name)
        raw_val = raw_val_map.get(name, '')

        if val > 0.005:
            direction = "INCREASES_RISK"
        elif val < -0.005:
            direction = "DECREASES_RISK"
        else:
            direction = "NEUTRAL"

        contributions.append(
            FeatureContribution(
                feature=human_name,
                transformedFeature=name,
                value=raw_val,
                contribution=round(val, 4),
                direction=direction,
            )
        )

    # Sort by absolute impact descending so primary drivers are ranked first
    contributions.sort(key=lambda c: abs(c.contribution), reverse=True)

    return ExplanationOutput(
        baseValue=round(base_val, 4),
        outputSpace="log-odds",
        features=contributions,
    )
