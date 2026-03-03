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
    # feat_cols=None uses all non-id columns (immunological is all numeric)
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
df_im_vis = preprocess.remove_outlier_observations(df_im_vis)
print(f"  df_im_vis : {df_im_vis.shape}")


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
df_cl_vis = preprocess.clean_cl(df_cl)

print('TableReport of cleaned clinical dataset (all columns, for EDA):')
TableReport(df_cl_vis)


#%%---------- Step 5_stats — Clinical descriptive statistics ------------------

print('\nStep 5_stats: Clinical descriptive statistics and distributions')

cl_t1 = df_cl_vis[df_cl_vis['Timepoint'] == 1].copy()

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
mako3 = sns.color_palette('mako', 3)

# Age
sns.histplot(cl_t1['age_at_start'].dropna(), kde=True, ax=axes[0],
             color=mako3[1], bins=15)
axes[0].set_title('Age distribution (T1)')
axes[0].set_xlabel('Age')
axes[0].set_ylabel('Count')

# Gender
gender_counts = cl_t1['gender'].value_counts()
axes[1].bar(gender_counts.index.astype(str), gender_counts.values,
            color=mako3[:len(gender_counts)])
axes[1].set_title('Gender distribution (T1)')
axes[1].set_ylabel('Count')

# Diagnosis
diag_counts = cl_t1['diagnosis'].value_counts()
axes[2].barh(diag_counts.index.astype(str), diag_counts.values,
             color=mako3[1])
axes[2].set_title('Diagnosis distribution (T1)')
axes[2].set_xlabel('Count')
axes[2].invert_yaxis()

plt.tight_layout()
plt.show()

# Pain scale per timepoint
fig, ax = plt.subplots(figsize=(8, 4))
sns.boxplot(data=df_cl_vis[df_cl_vis['pain_scale'].notna()],
            x='Timepoint', y='pain_scale', ax=ax, palette='mako')
ax.set_title('Pain scale distribution per timepoint (T1–T5)')
ax.set_xlabel('Timepoint')
ax.set_ylabel('Pain scale (1–10)')
plt.tight_layout()
plt.show()


#%%---------- Step 5a — Pearson correlation (clinical) ------------------------

cl_id_cols  = ['Patient', 'Timepoint', 'date', 'measurement_timepoint']
cl_num_cols = [c for c in df_cl_vis.columns
                if c not in cl_id_cols and df_cl_vis[c].dtype == 'float64']

print('\nStep 5a: EDA — Pearson correlation (clinical, float64 features)')
cl_pearson_matrix, cl_pearson_pairs = explore.pearson_correlation(
    df_cl_vis,
    id_cols=cl_id_cols,
    name='Clinical',
    n_top=40,
)


#%%---------- Step 5a (phik) — Phik correlation (clinical) -------------------

print('\nStep 5a (phik): Phik correlation (clinical, all feature types)')
cl_phik_matrix, cl_phik_pairs = explore.phik_correlation(
    df_cl_vis,
    id_cols=cl_id_cols,
    num_cols=cl_num_cols,
    name='Clinical',
    n_top=40,
)


#%%---------- Step 5b — RV2 matrix (clinical) ---------------------------------

print('\nStep 5b: RV2 matrix (clinical)')
cl_rv2_df = explore.rv2_matrix(
    df_cl_vis,
    timepoints=[1, 2, 3, 4, 5],
    id_cols=cl_id_cols,
    name='Clinical',
    feat_cols=cl_num_cols,   # clinical RV2 uses numeric columns only
)


#%%---------- Step 5c — PCA per timepoint T1-T5 (clinical) --------------------

print('\nStep 5c: PCA per timepoint T1–T5 (clinical)')
cl_pca_store = explore.pca_per_timepoint(
    df_cl_vis,
    timepoints=[1, 2, 3, 4, 5],
    id_cols=cl_id_cols,
    name='Clinical',
    feat_cols=cl_num_cols,
    ncomp=10,
)


#%%---------- Step 5d — PCA score plots coloured by clinical metadata ---------

print('\nStep 5d: PCA coloured by gender / pain_scale / response_category / diagnosis (clinical)')
cl_color_configs = [
    ('gender',            'categorical', 'mako'),
    ('pain_scale',        'continuous',  'mako'),
    ('response_category', 'categorical', 'mako'),
    ('diagnosis',         'categorical', 'tab20'),
]

explore.pca_colored(
    cl_pca_store,
    timepoints=[1, 2, 3, 4, 5],
    color_configs=cl_color_configs,
    name='Clinical',
)


#%%---------- Step 6 — Immunological PCA coloured by clinical variables -------

print('\nStep 6: Immunological PCA T1–T5 coloured by clinical categories (df_cl_vis)')
explore.pca_colored(
    im_pca_store,
    timepoints=[1, 2, 3, 4, 5],
    color_configs=cl_color_configs,
    name='Immunological colored by Clinical',
    color_source_df=df_cl_vis,
)



#%%---------- Step 7 — df_cl_mod: modeling-only copy -------------------------

print('\nStep 7: Creating df_cl_mod (modeling copy: >25% NaN drop, pain cols, leaky cols)')

# Take copy of df_cl_vis (contains all columns) 
df_cl_mod = df_cl_vis.copy()

# Drop columns with >25% NaN (reduce features before modeling)
mod_protect = ['Patient', 'Timepoint', 'pain_scale', 'date', 'measurement_timepoint']
df_cl_mod = preprocess.drop_high_nan_columns(
    df_cl_mod, threshold=0.25, exclude_cols=mod_protect,
    check_per_timepoint=True,
)

# Remove rows where pain_scale is NaN (because it will be used as target=)
df_cl_mod = preprocess.remove_no_pain_scale_rows(df_cl_mod)

# Drop pain questionnaire columns (not model features — targets are built separately)
pain_cols = [c for c in df_cl_mod.columns
              if c in set(preprocess.CL_PAIN_QUESTIONNAIRE_COLS)]
df_cl_mod = df_cl_mod.drop(columns=pain_cols)

# Drop other leaky columns (response, improvement_percent, etc.)
leaky_cols = [c for c in df_cl_mod.columns
               if any(pat in c for pat in preprocess.CL_LEAKY_PATTERNS)]
df_cl_mod = df_cl_mod.drop(columns=leaky_cols)

print(f"  df_cl_vis : {df_cl_vis.shape}  (all columns, for EDA)")
print(f"  df_cl_mod : {df_cl_mod.shape}  (modeling only)")
print(f"  Dropped pain cols  : {pain_cols}")
print(f"  Dropped leaky cols : {leaky_cols}")


#%%---------- Step 8 — Construct regression targets from clinical data --------

print('\nStep 8: Constructing regression targets from clinical data')
# use df_cl_vis as reference because we have all columns.

# Primary target: pain_scale reduction (T1 → T2)
# construct_datasets_targets returns only patients with non-NaN values at both
# timepoints and in all computed columns, so no further NaN filtering is needed.
# Targets are built from df_cl_vis (still contains pain columns)
pain_targets = model.construct_datasets_targets(df_cl_vis, 'pain_scale', [1, 2])

# Additional pain questionnaire targets (T1 → T2 differences)
# These serve as alternative regression targets; distributions shown below.
targets_daytime    = model.construct_datasets_targets(df_cl_vis, 'pain_daytime',    [1, 2])
targets_under_load = model.construct_datasets_targets(df_cl_vis, 'pain_under_load', [1, 2])


#%%---------- Step 8b — Target distributions ----------------------------------

print('\nStep 8b: Plotting target distributions')

# Collect all reduction_pct columns across all target DataFrames for plotting
target_frames = {
    'pain_scale':      pain_targets,
    'pain_daytime':    targets_daytime,
    'pain_under_load': targets_under_load,
}

fig, axes = plt.subplots(2, len(target_frames), figsize=(5 * len(target_frames), 8))
colors = sns.color_palette('mako', len(target_frames))

for col_idx, (name, tdf) in enumerate(target_frames.items()):
    prefix  = name.replace('_scale', '')   # matches construct_datasets_targets naming
    red_col = f'{prefix}_reduction'
    pct_col = f'{prefix}_reduction_pct'

    # Absolute reduction
    ax0 = axes[0, col_idx]
    sns.histplot(tdf[red_col].dropna(), kde=True, ax=ax0,
                 color=colors[col_idx], bins=20)
    ax0.set_title(f'{name}\nAbsolute reduction (T1−T2)')
    ax0.set_xlabel('Reduction')

    # Percent reduction
    ax1 = axes[1, col_idx]
    sns.histplot(tdf[pct_col].dropna(), kde=True, ax=ax1,
                 color=colors[col_idx], bins=20)
    ax1.set_title(f'{name}\nPercent reduction (%)')
    ax1.set_xlabel('Reduction (%)')

plt.suptitle('Target Distributions (T1 → T2)', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()


#%%########## BASELINE CATBOOST ################################################

print('\n' + '#'*60)
print('  BASELINE CATBOOST MODEL')
print('#'*60)


#%%---------- Step 9 — Prepare modeling datasets (immunological T1-T2 diffs) --

print('\nStep 9: Creating model datasets (immunological T1−T2 differences)')

# Build one (df_immu_alone, df_combined) pair per target.
# pain_reduction and pain_reduction_pct share pain_targets (same patients/dataset);
# the advanced model (Step 11) uses pain_reduction_pct from that same df_combined.
baseline_targets = {
    'pain_reduction':            pain_targets,
    'pain_daytime_reduction':    targets_daytime,
    'pain_under_load_reduction': targets_under_load,
}

model_datasets = {}
for tgt, tdf in baseline_targets.items():
    df_immu, df_comb = model.create_model_datasets(
        df_cl_mod, df_im_vis, tdf, timepoints=[1, 2]
    )
    model_datasets[tgt] = (df_immu, df_comb)

TableReport(model_datasets['pain_reduction'][0], max_plot_columns=180)
TableReport(model_datasets['pain_reduction'][1], max_plot_columns=180)


#%%---------- Step 10 — Run baseline CatBoost (all targets, no SHAP) ----------

print('\nStep 10: Running baseline CatBoost — pain_reduction, pain_daytime_reduction, pain_under_load_reduction')

baseline_results = {}
for tgt, (df_immu, df_comb) in model_datasets.items():
    res_immu, mdl_immu, X_immu, ypred_immu = model.run_catboost_regressor(
        df_immu, tgt, 'Immunological T1−T2 diff')
    res_comb, mdl_comb, X_comb, ypred_comb = model.run_catboost_regressor(
        df_comb, tgt, 'Combined T1−T2 diff')
    baseline_results[tgt] = {
        'Immunological': (res_immu, mdl_immu, X_immu, ypred_immu),
        'Combined':      (res_comb, mdl_comb, X_comb, ypred_comb),
    }

for tgt, ds_results in baseline_results.items():
    model.print_regression_summary(
        {ds: res[0] for ds, res in ds_results.items()}, tgt)



# SHAP / Heatmaps?


#%%########## ADVANCED MODELS ##################################################

print('\n' + '#'*60)
print('  ADVANCED CATBOOST MODEL (Nested CV + Optuna)')
print('#'*60)

#%%---------- Step 11 — Advanced CatBoost on combined dataset -----------------

print('\nStep 11: Advanced CatBoost (Nested CV + Optuna) — combined dataset only')

# Advanced modeling uses the combined dataset for pain_reduction_pct (primary target).
# pain_reduction_pct is produced alongside pain_reduction in pain_targets, so it
# is already present in model_datasets['pain_reduction'][1].
primary_target = 'pain_reduction_pct'
df_combined_adv = model_datasets['pain_reduction'][1]

adv_results, adv_best_params, adv_model, adv_X, adv_ypred = model.run_advanced_catboost(
    df_combined_adv,
    target_col=primary_target,
)


#%%---------- Step 12 — SHAP analysis on final advanced model -----------------

print('\nStep 12: SHAP analysis on advanced CatBoost final model')

adv_shap = model.plot_shap_regressor(
    adv_model, adv_X, f'Advanced CatBoost — {primary_target}')

# 2D density heatmap for advanced model
y_true_adv = df_combined_adv[primary_target].dropna().reset_index(drop=True)
model.plot_prediction_heatmap(
    y_true_adv, adv_ypred.dropna(), f'Advanced CatBoost — {primary_target}')


print('\nStep 13: Advanced HGB (Nested CV + Optuna) — PLACEHOLDER')
# TODO: implement model.run_advanced_hgb(df_combined)
