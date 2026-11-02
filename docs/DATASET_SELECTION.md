# Dataset Selection

## Overview

This document records the datasets selected for the four disease-specific prediction modules of the Smart Healthcare Platform.

The datasets will initially be used for research, model development, preprocessing, and evaluation. Dataset limitations and population differences must be considered when interpreting model performance.

---

## 1. Heart Disease

### Selected Dataset
UCI Heart Disease Dataset

### Purpose
Used for the heart disease risk prediction module.

### Expected Data
Include:
- demographic features
- clinical features
- cardiovascular/exercise-related features
- target/diagnosis variable

### Planned Use
- preprocessing
- feature engineering
- disease-specific model training
- model comparison
- explainability using SHAP

### Important Considerations
Document missing values and the need for careful train/test splitting and preprocessing without data leakage.

---

## 2. Diabetes

### Selected Dataset
Pima Indians Diabetes Dataset

### Purpose
Used for the diabetes risk prediction module.

### Expected Data
Include:
- glucose
- blood pressure
- BMI
- age
- pregnancy-related information
- diabetes pedigree-related information
- outcome/target

### Planned Use
- preprocessing
- feature engineering
- disease-specific model training
- model comparison
- SHAP explainability

### Important Considerations
Document that the dataset represents a specific population and therefore its results should not automatically be treated as representative of all populations.

---

## 3. Parkinson's Disease

### Selected Dataset
UCI Parkinson's Disease Dataset

### Purpose
Used for the Parkinson's disease prediction module.

### Expected Data
Include biomedical voice/disease-related measurements and the disease status target.

### Planned Use
- preprocessing
- feature engineering
- disease-specific model training
- model comparison
- SHAP explainability

### Important Considerations
The dataset contains repeated measurements from subjects. The implementation must avoid data leakage by using subject-aware splitting where subject identifiers are available.

Do not treat individual recordings from the same person as completely independent samples during evaluation.

---

## 4. Liver Disease

### Selected Dataset
UCI Indian Liver Patient Dataset (ILPD)

### Purpose
Used for the liver disease prediction module.

### Expected Data
Include:
- age
- gender
- bilirubin measurements
- protein measurements
- albumin
- albumin/globulin ratio
- liver enzyme measurements
- disease class/target

### Planned Use
- preprocessing
- feature engineering
- disease-specific model training
- model comparison
- SHAP explainability

### Important Considerations
Document population and dataset-size limitations before using the model for broader/generalized claims.

---

## Dataset Handling Principles

1. Original datasets must be stored under datasets/raw/.
2. Raw datasets must never be modified.
3. Cleaned/preprocessed datasets must be stored under datasets/processed/.
4. Dataset preprocessing must be reproducible.
5. Missing values must be handled explicitly.
6. Categorical features must be encoded appropriately.
7. Numerical features must be scaled when required by the selected model.
8. Train/test splitting must prevent data leakage.
9. Class imbalance must be measured before selecting mitigation techniques.
10. Evaluation must use appropriate classification metrics.
11. Dataset limitations must be documented.
12. External validation datasets will be considered later in the project under Phase 17.

---

## Dataset-to-Project Mapping

| Disease | Dataset | Main Purpose |
|---|---|---|
| Heart Disease | UCI Heart Disease | Heart risk prediction |
| Diabetes | Pima Indians Diabetes | Diabetes risk prediction |
| Parkinson's Disease | UCI Parkinson's Disease | Parkinson's risk prediction |
| Liver Disease | UCI ILPD | Liver disease risk prediction |
