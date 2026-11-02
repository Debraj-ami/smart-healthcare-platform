import joblib
import numpy as np

b = joblib.load('artifacts/liver/liver_model_bundle.joblib')

print('=== BUNDLE CONTENTS ===')
for k, v in b.items():
    t = type(v).__name__
    print(f'  {k}: {t}')
print()

cal = b['calibrated_model']
print('=== CALIBRATED MODEL ===')
print(f'  Type: {type(cal).__name__}')
print(f'  Method: {cal.method}')
print(f'  CV: {cal.cv}')
print(f'  Estimator type: {type(cal.estimator).__name__}')
print()

exp = b['shap_explainer']
print('=== SHAP EXPLAINER ===')
print(f'  Type: {type(exp).__name__}')
print(f'  Expected value (base value): {exp.expected_value}')
print()

lr = b['lr_classifier']
print('=== BASE LR CLASSIFIER ===')
print(f'  Coefficients shape: {lr.coef_.shape}')
print(f'  Classes: {lr.classes_}')
print(f'  Intercept: {lr.intercept_}')
print(f'  Coefficients: {lr.coef_}')
print()

pre = b['preprocessor']
print('=== PREPROCESSOR ===')
print(f'  Type: {type(pre).__name__}')
print(f'  Transformers: {[t[0] for t in pre.transformers]}')
print(f'  Feature names out: {pre.get_feature_names_out().tolist()}')
print()

# Simulate the actual inference for the test case
import pandas as pd
from app.preprocessing import request_to_dataframe

class FakeReq:
    age = 45
    gender = 'Male'
    totalBilirubin = 0.9
    directBilirubin = 0.3
    totalProteins = 6.8
    albumin = 3.4
    agRatio = 1.0
    sgpt = 35.0
    sgot = 40.0
    def get_effective_alkphos(self): return 200.0

req = FakeReq()
df = request_to_dataframe(req)
print('=== INFERENCE TRACE ===')
print(f'  Input DataFrame:\n{df}')

X_transformed = pre.transform(df)
print(f'  Transformed shape: {X_transformed.shape}')
print(f'  Transformed values: {X_transformed}')

proba = cal.predict_proba(df)
print(f'  Calibrated probabilities: {proba}')
print(f'  P(disease=1): {proba[0,1]:.4f}')

explanation = exp(X_transformed)
print(f'  SHAP values: {explanation.values[0]}')
print(f'  SHAP base value: {explanation.base_values[0]}')
print(f'  Sum check: base + sum(shap) = {explanation.base_values[0] + explanation.values[0].sum():.4f}')
print()
print('Note: SHAP base + sum is in log-odds space')
print('The probability comes from cal.predict_proba (calibrated isotonic), NOT from logistic sigmoid of SHAP sum')
