## Master Thesis Repository
### Title: Machine Learning for Predicting Pain Change in Patients Treated with Low-Dose Radiation Therapy
### Author: Muna Ahmed Farah 
### Year: 2026

## Description
This repository outlines my thesis work conducted at NMBU  (Norwegian University of Life Sciences):

This thesis aimed to predict patient-level pain change following low-dose radiation therapy (LDRT) using
immunological and clinical features from data collected in the prospective IMMO-LDRT01 trial. Four regression models
(ElasticNet, SVR, PLSR, and HGBR) were developed at three post-treatment timepoints (T2, T3, T4) and evaluated 
through nested cross-validation, with Optuna hyperparameter optimization and MRMR feature selection. 

Results show that all models yielded R-squared values close to zero or negative across all timepoints, meaning the models 
were not able to capture the variation in the target and learn predictive signals. Several of the models performed worse than a constant-mean model, which only outputs the mean target-value across the training samples as predictions. These results indicate that the immunological and clinical features available in data do not support patient-level prediction of pain change. 
Alternative modeling formulations and integration of psychosocial and behavioral pain measurements alongside immunological data are suggestions for future work.

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
│   └── models_T3-T4/                  # Folder for timepoint 3 and 4 modeling notebooks
|
├── src/
│   ├── explore.py                     # Script for EDA 
│   ├── model_elasticnet.py            # Script for ElasticNet Model
│   ├── model_hgbr.py                  # Script for HGBR Model
│   ├── model_pls.py                   # Script for PLSR Model
│   ├── model_svr.py                   # Script for SVR Model
│   ├── model.py                       # Script for Baseline models and plotting results
│   └── pyod_zryan/                    # PyOD functions developed by Zryan Mustafa 
|
├── models/                            # Saved model output files (ignored by github)
├── docs/                              # History of additonal experiments
├── environment.yml                    # Conda environment requirements
└── README.md
```

## Enviroment
Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate mt26
```
