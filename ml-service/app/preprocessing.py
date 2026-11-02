import pandas as pd
from app.schemas import LiverPredictionRequest


FEATURE_COLUMNS = [
    'Age',
    'Gender',
    'Total_Bilirubin',
    'Direct_Bilirubin',
    'Alkphos',
    'Sgpt',
    'Sgot',
    'Total_Proteins',
    'Albumin',
    'A_G_Ratio',
]

HUMAN_READABLE_NAMES = {
    'num__Age': 'Age',
    'num__Total_Bilirubin': 'Total Bilirubin',
    'num__Direct_Bilirubin': 'Direct Bilirubin',
    'num__Alkphos': 'Alkaline Phosphatase',
    'num__Sgpt': 'SGPT (ALT)',
    'num__Sgot': 'SGOT (AST)',
    'num__Total_Proteins': 'Total Proteins',
    'num__Albumin': 'Albumin',
    'num__A_G_Ratio': 'A/G Ratio',
    'cat__Gender_Male': 'Gender (Male)',
}


def request_to_dataframe(req: LiverPredictionRequest) -> pd.DataFrame:
    """
    Map human-readable request into a DataFrame with exact column names expected
    by the scikit-learn preprocessor pipeline.
    """
    data = {
        'Age': [req.age],
        'Gender': [req.gender],
        'Total_Bilirubin': [req.totalBilirubin],
        'Direct_Bilirubin': [req.directBilirubin],
        'Alkphos': [req.get_effective_alkphos()],
        'Sgpt': [req.sgpt],
        'Sgot': [req.sgot],
        'Total_Proteins': [req.totalProteins],
        'Albumin': [req.albumin],
        'A_G_Ratio': [req.agRatio],
    }
    return pd.DataFrame(data, columns=FEATURE_COLUMNS)
