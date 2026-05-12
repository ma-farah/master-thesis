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
│   └──                                # Raw dataset (ignored by gitlab)
├── notebooks/
│   ├── 1-cleaning.ipynb               # Data cleaning and preprocessing steps
│   ├── 2-exploring.ipynb              # Exploratory data analysis
│   ├── 3-exploring_pca.ipynb          # PCA and Trajectory PCA
│   ├── 4-exploring_outliers.ipynb     # PyOD Outlier detection results
│   ├── 5-modeling_baseline.ipynb      # CatBoost Baseline Model on T2 data
│   └── 6-modeling_elasticnet.ipynb    # Final Elasticnet Model on T2 data
│   ├── 7-modeling_svr.ipynb           # Final SVR Model on T2 data
│   ├── 8-modeling_pls.ipynb           # Final PLSR Model on T2 data
│   └── 9-modeling_hgbr.ipynb          # Final HGBR Model on T2 data
│   ├── 10-comparing_models_t3.ipynb   # Summary of T3 models
│   ├── 10-comparing_models_t4.ipynb   # Summary of T4 models
│   └── 10-comparing_models.ipynb      # Summary of T2 models
│   └── models_T3-T4/                  # Folder for similar notebooks of timepoint 3 and 4 models
|
├── src/
│   ├── explore.py                     # Script for EDA and outlier detection 
│   ├── model_elasticnet.py            # Script for ElasticNet Model
│   ├── model_hgbr.py                  # Script for HGBR Model
│   ├── model_pls.py                   # Script for PLSR Model
│   ├── model_svr.py                   # Script for SVR Model
│   ├── model.py                       # Script for Baseline models and plotting results
│   └── pyod_zryan/                    # PyOD functions developed by Zryan Mustafa 
|
├── models/                            # Saved model output files (ignored by gitlab)
├── docs/                              # History of additonal experiments
├── environment.yml                    # Conda environment
└── README.md
```

## Requirements
Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate mt26
```
key dependencies: Python 3.10, CatBoost, LightGBM (<4.0), scikit-learn, RENT, PyOD, Optuna, SHAP, miceforest
