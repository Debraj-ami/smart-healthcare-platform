# Datasets Directory

This directory stores datasets used for training and evaluating machine learning models across four diseases: Heart Disease, Diabetes, Parkinson's Disease, and Liver Disease.

---

## Directory Organization

```
datasets/
├── raw/
│   ├── heart/
│   ├── diabetes/
│   ├── parkinsons/
│   └── liver/
├── processed/
│   ├── heart/
│   ├── diabetes/
│   ├── parkinsons/
│   └── liver/
└── README.md
```

---

## Data Stages

- **`raw/`**: Contains the original, unmodified datasets directly collected from source repositories or clinical benchmarks. Files in this directory should never be edited directly.
- **`processed/`**: Contains datasets generated after preprocessing, cleaning, normalization, imputation, and feature engineering, ready for model training and validation.

---

## Disease Categories

- **`heart/`**: Dedicated to the Heart Disease dataset.
- **`diabetes/`**: Dedicated to the Diabetes dataset.
- **`parkinsons/`**: Dedicated to the Parkinson's Disease dataset.
- **`liver/`**: Dedicated to the Liver Disease dataset.
