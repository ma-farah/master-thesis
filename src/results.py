# Full analysis pipeline — calls functions from preprocess, explore, and model
# run_script.py is kept as an old reference copy

#%% Imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from skrub import TableReport
import scikit_na as na

import preprocess
import explore
import model


#%%########## Step 1 — Load raw data ###########################################

print('Step 1: Loading raw data')
df_im, df_cl = explore.load_data()


#%%########## IMMUNOLOGICAL DATASET ############################################

print('\n' + '#'*60)
print('  IMMUNOLOGICAL DATASET')
print('#'*60)


print("TableReport of raw immunological dataset:")
TableReport(df_im, max_plot_columns=138)

print("Na analysis of raw immunological dataset:")
na.altair.plot_heatmap(df_im)

explore.dataset_overview(df_im, name='Immunological')
explore.patient_timepoint_summary(df_im, name='Immunological')


#%%---------- Step 2 — Clean immunological dataset ----------------------------

print('\nStep 2: Cleaning immunological dataset')
df_im, df_im_bcat, df_im_vis = preprocess.clean_im(df_im)

# TableReport after cleaning (before >25% NaN drop — matches run_script.py line 220)
print('\nTableReport of cleaned immunological dataset (before >25% NaN drop):')
TableReport(df_im_bcat, max_plot_columns=180)

print('\nTableReport of cleaned immunological dataset (after >25% NaN drop):')
TableReport(df_im_vis, max_plot_columns=180)


#%%---------- Step 2a — Pearson correlation ------------------------------------

print('\nStep 2a: EDA — Pearson correlation (immunological)')
_im_id_cols = ['Patient', 'Timepoint', 'Date']

im_pearson_matrix, im_pearson_pairs = explore.pearson_correlation(
    df_im_vis,
    id_cols=_im_id_cols,
    name='Immunological',
    n_top=40,
)


#%%---------- Step 2b — RV2 matrix --------------------------------------------

print('\nStep 2b: RV2 matrix (immunological)')
im_rv2_df = explore.rv2_matrix(
    df_im_vis,
    timepoints=[1, 2, 3, 4, 5],
    id_cols=_im_id_cols,
    name='Immunological',
    # feat_cols=None → uses all non-id columns (immunological is all numeric)
)


#%%---------- Step 2c — PCA per timepoint T1-T5 --------------------------------

print('\nStep 2c: PCA per timepoint T1–T5 (immunological)')
im_pca_store = explore.pca_per_timepoint(
    df_im_vis,
    timepoints=[1, 2, 3, 4, 5],
    id_cols=_im_id_cols,
    name='Immunological',
    ncomp=10,
)


#%%---------- Step 2d — Trajectory PCA T1↔T2, T2↔T3, T1↔T3 ------------------

print('\nStep 2d: Trajectory PCA T1-T2, T2-T3, T1-T3 (immunological)')
_mako5 = sns.color_palette('mako', 5)
_im_pairs = [
    (1, 2, _mako5[2], 'T1 → T2'),
    (2, 3, _mako5[4], 'T2 → T3'),
    (1, 3, _mako5[4], 'T1 → T3'),
]

explore.trajectory_pca_im(
    df_im_vis,
    pairs=_im_pairs,
    id_cols=_im_id_cols,
    ncomp=10,
)


#%%---------- Step 2e — MFA T1-T3 (NaN-native, no imputation) ----------------

print('\nStep 2e: MFA T1-T3 (immunological)')
explore.mfa_im(
    df_im_vis,
    timepoints=[1, 2, 3],
    id_cols=_im_id_cols,
    ncomp=5,
)


#%%---------- Step 3 — Imputation (for PyOD) ----------------------------------

print('\nStep 3: Imputing immunological dataset (miceforest + median)')
df_im_imputed = preprocess.impute_miceforest(
    df_im_vis,
    id_cols=_im_id_cols,
    name='Immunological',
    num_datasets=5,
    iterations=10,
    mean_match_candidates=5,
)

df_im_median = preprocess.impute_median(
    df_im_vis,
    id_cols=_im_id_cols,
    name='Immunological',
)


#%%
# Compare imputed datasets:
# feature_cols = df_im[∼id_cols] 
# Compare column statistics between the two imputed datasets

feature_cols = [c for c in df_im_vis.columns if c not in _im_id_cols]
n_missing = df_im_vis[feature_cols].isna().sum()

stats_cmp = pd.DataFrame({
    'n_missing':   n_missing,
    'mean_mice':   df_im_imputed[feature_cols].mean(),
    'mean_median': df_im_median[feature_cols].mean(),
    'std_mice':    df_im_imputed[feature_cols].std(),
    'std_median':  df_im_median[feature_cols].std(),
})
stats_cmp['mean_diff'] = (stats_cmp['mean_mice'] - stats_cmp['mean_median']).abs()
stats_cmp['std_diff']  = (stats_cmp['std_mice']  - stats_cmp['std_median']).abs()

print(f"\n  Features with any missing values: {(stats_cmp['n_missing'] > 0).sum()}")
print(f"  Max missing values in a single feature: {stats_cmp['n_missing'].max()}")
print(f"\n  Mean absolute difference in column means (MICE vs median): {stats_cmp['mean_diff'].mean():.4f}")
print(f"  Mean absolute difference in column stds  (MICE vs median): {stats_cmp['std_diff'].mean():.4f}")
print(f"\n  Top 10 features with largest mean difference (MICE vs median):")
print(stats_cmp.sort_values('mean_diff', ascending=False).head(10)
      [['n_missing', 'mean_mice', 'mean_median', 'mean_diff']].to_string())



#%%---------- Step 4 — PyOD outlier detection (immunological) -----------------

print('\nStep 4: PyOD outlier detection — immunological dataset (Zryan approach)')
_im_feat_cols = [c for c in df_im_vis.columns if c not in _im_id_cols]

no_od_df_im, outlier_candidates_im = explore.run_pyod_zryan(
    df_im_imputed,
    feature_cols=_im_feat_cols,
    patient_col='Patient',
    timepoint_col='Timepoint',
    contamination=0.05,
    name='Immunological (MICE)',
)

print('\nStep 4b: PyOD — immunological dataset, median imputed')
no_od_df_im_med, outlier_candidates_im_med = explore.run_pyod_zryan(
    df_im_median,
    feature_cols=_im_feat_cols,
    patient_col='Patient',
    timepoint_col='Timepoint',
    contamination=0.05,
    name='Immunological (median)',
)


#%%---------- Step 5 — Outlier removal (after expert review) ------------------

print('\nStep 5: Removing confirmed outlier observations (immunological)')
# Confirmed outliers stored in preprocess.IM_CONFIRMED_OUTLIERS
# = [(221,2), (163,1), (150,1), (159,2), (109,5), (266,4)]
df_im_mod = preprocess.remove_outlier_observations(df_im_vis)
print(f"  df_im_mod : {df_im_mod.shape}")


#%%########## CLINICAL DATASET #################################################

print('\n' + '#'*60)
print('  CLINICAL DATASET')
print('#'*60)


print("TableReport of raw clincial dataset:")
TableReport(df_cl, max_plot_columns=138)

print("Na analysis of raw clinical dataset:")
na.altair.plot_heatmap(df_cl)

explore.dataset_overview(df_cl, name='Clinical', patient_col='Patient',
                         timepoint_col=None)

# tablereport and na not showing up?


#%%---------- Step 6 — Clean clinical dataset ---------------------------------

print('\nStep 6: Cleaning clinical dataset')
df_cl_clean, df_cl_bcat, df_cl_vis = preprocess.clean_cl(df_cl)

# need to remove patients with missing questionarre data: 149 rows!

print('Tablereport of Clinical dataset (before dropping 25% nan)')
TableReport(df_cl_bcat)

print('Tablereport of Clinical dataset (after dropping 25% nan)')
TableReport(df_cl_vis)


#%%---------- Step 5a — Pearson correlation (clinical) ------------------------

# cl id cols????

print('\nStep 5a: EDA — Pearson correlation (clinical, float64 features)')
cl_pearson_matrix, cl_pearson_pairs = explore.pearson_correlation(
    df_cl_vis,
    id_cols=_cl_id_cols,
    name='Clinical',
    n_top=40,
)


#%%---------- Step 5a (phik) — Phik correlation (clinical) -------------------

print('\nStep 5a (phik): Phik correlation (clinical, all feature types)')
cl_phik_matrix, cl_phik_pairs = explore.phik_correlation(
    df_cl_vis,
    id_cols=_cl_id_cols,
    num_cols=_cl_num_cols,
    name='Clinical',
    n_top=40,
)


#%%---------- Step 5b — RV2 matrix (clinical) ---------------------------------

print('\nStep 5b: RV2 matrix (clinical)')
cl_rv2_df = explore.rv2_matrix(
    df_cl_vis,
    timepoints=[1, 2, 3, 4, 5],
    id_cols=_cl_id_cols,
    name='Clinical',
    feat_cols=_cl_num_cols,   # clinical RV2 uses numeric columns only
)


#%%---------- Step 5c — PCA per timepoint T1-T5 (clinical) --------------------

print('\nStep 5c: PCA per timepoint T1–T5 (clinical)')
cl_pca_store = explore.pca_per_timepoint(
    df_cl_vis,
    timepoints=[1, 2, 3, 4, 5],
    id_cols=_cl_id_cols,
    name='Clinical',
    feat_cols=_cl_num_cols,
    ncomp=10,
)


#%%---------- Step 5d — PCA score plots coloured by clinical metadata ---------

print('\nStep 5d: PCA coloured by gender / pain_scale / response_category / diagnosis (clinical)')
_cl_color_configs = [
    ('gender',            'categorical', 'mako'),
    ('pain_scale',        'continuous',  'mako'),
    ('response_category', 'categorical', 'mako'),
    ('diagnosis',         'categorical', 'tab20'),
]

explore.pca_colored(
    cl_pca_store,
    timepoints=[1, 2, 3, 4, 5],
    color_configs=_cl_color_configs,
    name='Clinical',
)


#%%---------- Step 6 — Immunological PCA coloured by clinical variables -------

print('\nStep 6: Immunological PCA T1–T5 coloured by clinical categories (df_cl_vis)')
explore.pca_colored(
    im_pca_store,
    timepoints=[1, 2, 3, 4, 5],
    color_configs=_cl_color_configs,
    name='Immunological colored by Clinical',
    color_source_df=df_cl_vis,
)


#%%---------- Step 5f — Imputation (clinical, for PyOD) ----------------------

print('\nStep 5f: Imputing clinical dataset (miceforest, all features)')
_cl_imp_id_cols = ['Patient', 'Timepoint', 'date', 'measurement_timepoint']

df_cl_imputed = preprocess.impute_miceforest(
    df_cl_vis,
    id_cols=_cl_imp_id_cols,
    name='Clinical',
    num_datasets=5,
    iterations=10,
    mean_match_candidates=0,   # KD-tree mean matching fails for mixed-type data
)


#%%---------- Step 5g — PyOD outlier detection (clinical) — PLACEHOLDER ------

print('\nStep 5g: PyOD outlier detection — clinical dataset (Zryan approach)')
# TODO: run after confirming imputed clinical dataset is correct
# _cl_feat_cols = [c for c in df_cl_imputed.columns if c not in _cl_imp_id_cols]
# no_od_df_cl, outlier_candidates_cl = explore.run_pyod_zryan(
#     df_cl_imputed,
#     feature_cols=_cl_feat_cols,
#     patient_col='Patient',
#     timepoint_col='Timepoint',
#     contamination=0.1,
#     name='Clinical',
# )


#%%---------- Step 7 — df_cl_mod: drop leaky cols + remove no-pain rows ------

print('\nStep 7: Creating df_cl_mod (drop leaky columns + remove NaN pain_scale rows)')
_leaky_cols = [c for c in df_cl_vis.columns
               if any(pat in c for pat in preprocess.CL_LEAKY_PATTERNS)]
print(f"  Dropping leaky/metadata columns ({len(_leaky_cols)}): {_leaky_cols}")
df_cl_mod = df_cl_vis.drop(columns=_leaky_cols).copy()

df_cl_mod = preprocess.remove_no_pain_scale_rows(df_cl_mod)


#%%---------- Step 8 — Target variables ---------------------------------------

print('\nStep 8: Computing regression targets (pain_reduction_pct, pain_scale_t2)')
pain_targets = preprocess.create_target_variables(df_cl_vis, df_cl_mod=df_cl_mod)

print(f"\n  TableReport of df_cl_mod:")
TableReport(df_cl_mod, max_plot_columns=100)


#%%########## BASELINE CATBOOST ################################################

print('\n' + '#'*60)
print('  BASELINE CATBOOST MODEL')
print('#'*60)


#%%---------- Step 9 — Prepare baseline T1 datasets --------------------------

print('\nStep 9: Preparing baseline T1 datasets')
df_im_raw_t1, df_cl_bcat_t1, df_bcat_combined_t1 = model.prepare_baseline_datasets(
    df_im_vis, df_cl_bcat, pain_targets
)

TableReport(df_im_raw_t1, max_plot_columns=180)

TableReport(df_cl_bcat_t1, max_plot_columns=180)

TableReport(df_bcat_combined_t1, max_plot_columns=180)

df_bcat_combined_t1 = (
    df_bcat_combined_t1
    .drop(columns=['pain_scale_t2_im', 'pain_reduction_pct_im'], errors='ignore')
    .rename(columns={
        'pain_scale_t2_cl': 'pain_scale_t2',
        'pain_reduction_pct_cl': 'pain_reduction_pct'
    })
)

# remove outlier patients from clinical dataset as well!

# total 127 patients......?

#%%---------- Step 10 — Run baseline CatBoost (both targets) -----------------

print('\nStep 10: Running baseline CatBoost (pain_reduction_pct + pain_scale_t2)')
baseline_results, baseline_shap = model.run_baseline_catboost(
    df_im_raw_t1,
    df_cl_bcat_t1,
    df_bcat_combined_t1,
)


#%%########## ADVANCED MODELS (placeholders) ###################################

print('\nStep 11: Advanced CatBoost (Nested CV + Optuna) — PLACEHOLDER')
# TODO: implement model.run_advanced_catboost(df_combined)

print('\nStep 12: Advanced HGB (Nested CV + Optuna) — PLACEHOLDER')
# TODO: implement model.run_advanced_hgb(df_combined)
