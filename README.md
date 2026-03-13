## Master Thesis 2026 By Muna Ahmed Farah

## Description
This repository outlines my thesis work, which aims to predict the treatment response in patients 
recieving Low-Dose Radiation Therapy (LDRT), using datasets consisting of immunological and clinical variables,
across multiple timepoints. This project covers data preprocessing, exploratory data analysis,
outlier detection, machine learning modelling with nested cross-validation, Optuna hyperparameter
optimization, RENT feature selection, and SHAP-analysis to identify predictors of treatment response.

## Repository Structure
```
masterthesis/
├── data/
│   └──                             # Raw data (not tracked by git)
├── notebooks/
│   ├── 1-cleaning.ipynb            # Data cleaning and preprocessing
│   ├── 2-exploring.ipynb           # Exploratory data analysis
│   ├── 3-exploring_pca.ipynb       # PCA and multivariate analysis
│   ├── 4-exploring_outliers.ipynb  # Outlier detection
│   ├── 5-modeling_t1-t2.ipynb      # Modelling: t1-t2 outcome
│   └── 6-modeling_t1-t3.ipynb      # Modelling: t1-t3 outcome
├── src/
│   ├── preprocess.py               # Cleaning, imputation, feature engineering
│   ├── explore.py                  # EDA and outlier detection functions
│   ├── model.py                    # Baseline models
│   ├── model_catboost.py           # CatBoost models
│   ├── results.py                  # Development scripts
│   └── pyod_zyran/                 # PyOD functions from Zryan Mustafa
├── models/                         # Saved model files
├── docs/                           # Reports, figures
├── environment.yml                 # Conda environment
└── README.md
```

## Requirements
Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate mt26
```
key dependencies: Python 3.10, CatBoost, LightGBM (<4.0), scikit-learn, RENT, PyOD, Optuna, SHAP, miceforest
