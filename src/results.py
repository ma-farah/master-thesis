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
df_im_vis = preprocess.clean_im(df_im)

# TableReport after cleaning 
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


#%%---------- Step 2e — MFA T1-T3  ----------------

print('\nStep 2e: MFA T1-T3 (immunological)')
explore.mfa_im(
    df_im_vis,
    timepoints=[1, 2, 3],
    id_cols=_im_id_cols,
    ncomp=5,
)


#%%---------- Step 3 — Imputation (for PyOD) ----------------------------------

# Drop columns with >25% NaN before imputation
nan_frac = df_im_vis.drop(columns=_im_id_cols).isna().mean()
high_nan_cols = nan_frac[nan_frac > 0.25].index.tolist()
df_im_mod = df_im_vis.drop(columns=high_nan_cols)    # copy for modeling
print(f"  Dropped {len(high_nan_cols)} columns with >25% NaN: {sorted(high_nan_cols)}")


print('\nStep 3: Imputing immunological dataset (miceforest + median)')
df_im_imputed = preprocess.impute_miceforest(
    df_im_mod,
    id_cols=_im_id_cols,
    name='Immunological',
    num_datasets=5,
    iterations=10,
    mean_match_candidates=5,
)

df_im_median = preprocess.impute_median(
    df_im_mod,
    id_cols=_im_id_cols,
    name='Immunological',
)


#%%
# Compare imputed datasets:
# feature_cols = df_im[∼id_cols]
# Compare column statistics between the two imputed datasets

feature_cols = [c for c in df_im_mod.columns if c not in _im_id_cols]
n_missing = df_im_mod[feature_cols].isna().sum()

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
_im_feat_cols = [c for c in df_im_mod.columns if c not in _im_id_cols]

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

df_im_mod = preprocess.remove_outlier_observations(df_im_mod)
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
_mod_protect = ['Patient', 'Timepoint', 'date', 'measurement_timepoint']
_cl_nan_frac = df_cl_mod.drop(columns=_mod_protect).isna().mean()
_cl_high_nan = _cl_nan_frac[_cl_nan_frac > 0.25].index.tolist()
df_cl_mod = df_cl_mod.drop(columns=_cl_high_nan)
print(f"  Dropped {len(_cl_high_nan)} columns with >25% NaN: {sorted(_cl_high_nan)}")

# Drop pain questionnaire columns (not model features — targets are built separately)
pain_cols = [c for c in df_cl_mod.columns
              if c in set(preprocess.CL_PAIN_QUESTIONNAIRE_COLS)]
df_cl_mod = df_cl_mod.drop(columns=pain_cols)


# Drop additional columns: baseline pain_scale, pain_points, complaints_since
extra_cols = [c for c in df_cl_mod.columns
              if c in set(preprocess.CL_EXTRA_MODEL_DROP_COLS)]
df_cl_mod = df_cl_mod.drop(columns=extra_cols)

# Drop other leaky columns (response, improvement_percent, etc.)
leaky_cols = [c for c in df_cl_mod.columns
               if any(pat in c for pat in preprocess.CL_LEAKY_PATTERNS)]
df_cl_mod = df_cl_mod.drop(columns=leaky_cols)

print(f"  df_cl_vis : {df_cl_vis.shape}  (all columns, for EDA)")
print(f"  df_cl_mod : {df_cl_mod.shape}  (modeling only)")
print(f"  Dropped pain cols      : {pain_cols}")
print(f"  Dropped extra cols     : {extra_cols}")
print(f"  Dropped leaky cols     : {leaky_cols}")


#%%---------- Step 8 — Construct regression targets from clinical data --------

print('\nStep 8: Constructing regression targets from clinical data')
# use df_cl_vis as reference because we have all columns.

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

# Build one (df_immu_alone, df_combined) pair per unique target source.
# pain_reduction_pct is computed from the same pain_targets as pain_reduction,
# so we alias it rather than calling create_model_datasets a second time.
# leaky sibling column (e.g. pain_daytime_reduction_pct when running on
# pain_daytime_reduction) is excluded inside run_catboost_regressor 

_unique_targets = {
    'pain_reduction':            pain_targets,
    'pain_daytime_reduction':    targets_daytime,
    'pain_under_load_reduction': targets_under_load,
}

model_datasets = {}
for tgt, tdf in _unique_targets.items():
    df_immu, df_comb = model.create_model_datasets(
        df_cl_mod, df_im_mod, tdf, timepoints=[1, 2]
    )
    model_datasets[tgt] = (df_immu, df_comb)

# pain_reduction_pct lives in the same dataset as pain_reduction
model_datasets['pain_reduction_pct'] = model_datasets['pain_reduction']

TableReport(model_datasets['pain_reduction'][0], max_plot_columns=180)
TableReport(model_datasets['pain_reduction'][1], max_plot_columns=180)
TableReport(model_datasets['pain_daytime_reduction'][1], max_plot_columns=180)
TableReport(model_datasets['pain_under_load_reduction'][1], max_plot_columns=180)


#%%---------- Step 10 — Run baseline CatBoost (all targets, no SHAP) ----------

print('\nStep 10: Running baseline CatBoost — pain_reduction, pain_reduction_pct, pain_daytime_reduction, pain_under_load_reduction')

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



#%%---------- Step 10b — Model dataset diagnostics ---------------------------

print('\nStep 10b: Model dataset diagnostics — target distributions and sample sizes')

for tgt, (df_immu, df_comb) in model_datasets.items():
    y = df_comb[tgt].dropna()
    print(f"\n{'─'*55}")
    print(f"  Target : {tgt}")
    print(f"  n (combined, non-NaN target) : {len(y)}")
    print(f"  Features in combined dataset : {df_comb.shape[1]}")
    print(f"  mean={y.mean():.2f}  std={y.std():.2f}  "
          f"min={y.min():.2f}  max={y.max():.2f}")
    print(f"  skew={y.skew():.2f}  kurt={y.kurt():.2f}")
    print(f"  % zeros (no change) : {(y == 0).mean()*100:.1f}%")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    sns.histplot(y, kde=True, ax=axes[0], color=sns.color_palette('mako', 1)[0], bins=20)
    axes[0].set_title(f'{tgt} — distribution')
    axes[0].set_xlabel(tgt)
    axes[1].boxplot(y.dropna(), vert=False)
    axes[1].set_title(f'{tgt} — boxplot (outliers)')
    axes[1].set_xlabel(tgt)
    plt.tight_layout()
    plt.show()

print(f"\n{'─'*55}")
print("  Feature–target Pearson correlations (top 10, combined dataset):")
for tgt, (df_immu, df_comb) in model_datasets.items():
    id_like = ['Patient', 'Timepoint', 'Date', 'date', 'measurement_timepoint']
    num_cols = [c for c in df_comb.select_dtypes(include='number').columns
                if c not in id_like]
    corrs = df_comb[num_cols].corrwith(df_comb[tgt]).drop(index=tgt, errors='ignore')
    corrs = corrs.dropna().abs().sort_values(ascending=False).head(10)
    print(f"\n  {tgt}:")
    print(corrs.to_string())


#%%########## ADVANCED MODELS — CATBOOST ######################################

print('\n' + '#'*60)
print('  CATBOOST MODEL (Nested CV + Optuna)')
print('#'*60)


#%%---------- Step 11a — CatBoost: pain_reduction_pct ------------------------

print('\nStep 11a: CatBoost (Nested CV + Optuna) — pain_reduction_pct')

cb_pct_results, cb_pct_params, cb_pct_model, cb_pct_X, cb_pct_ypred = \
    model.run_advanced_catboost(
        model_datasets['pain_reduction'][1],
        target_col='pain_reduction_pct',
    )


#%%---------- Step 11b — CatBoost SHAP: pain_reduction_pct -------------------

print('\nStep 11b: SHAP — CatBoost (pain_reduction_pct)')

cb_pct_shap = model.plot_shap_regressor(
    cb_pct_model, cb_pct_X, 'CatBoost — pain_reduction_pct')

y_true_cb_pct = model_datasets['pain_reduction'][1]['pain_reduction_pct'].dropna().reset_index(drop=True)
model.plot_prediction_heatmap(
    y_true_cb_pct, cb_pct_ypred.dropna(), 'CatBoost — pain_reduction_pct')


#%%---------- Step 12a — CatBoost: pain_under_load_reduction -----------------

print('\nStep 12a: CatBoost (Nested CV + Optuna) — pain_under_load_reduction')

cb_ul_results, cb_ul_params, cb_ul_model, cb_ul_X, cb_ul_ypred = \
    model.run_advanced_catboost(
        model_datasets['pain_under_load_reduction'][1],
        target_col='pain_under_load_reduction',
    )


#%%---------- Step 12b — CatBoost SHAP: pain_under_load_reduction ------------

print('\nStep 12b: SHAP — CatBoost (pain_under_load_reduction)')

cb_ul_shap = model.plot_shap_regressor(
    cb_ul_model, cb_ul_X, 'CatBoost — pain_under_load_reduction')

y_true_cb_ul = model_datasets['pain_under_load_reduction'][1]['pain_under_load_reduction'].dropna().reset_index(drop=True)
model.plot_prediction_heatmap(
    y_true_cb_ul, cb_ul_ypred.dropna(), 'CatBoost — pain_under_load_reduction')


#%%########## ADVANCED MODELS — HGB ###########################################

print('\n' + '#'*60)
print('  HGB MODEL (Nested CV + Optuna)')
print('#'*60)


#%%---------- Step 13a — HGB: pain_reduction_pct -----------------------------

print('\nStep 13a: HGB (Nested CV + Optuna) — pain_reduction_pct')

hgb_pct_results, hgb_pct_params, hgb_pct_model, hgb_pct_X, hgb_pct_ypred = \
    model.run_advanced_hgb(
        model_datasets['pain_reduction'][1],
        target_col='pain_reduction_pct',
    )

print('\nStep 13b: SHAP — HGB (pain_reduction_pct)')

hgb_pct_shap = model.plot_shap_regressor(
    hgb_pct_model, hgb_pct_X, 'HGB — pain_reduction_pct')

y_true_hgb_pct = model_datasets['pain_reduction'][1]['pain_reduction_pct'].dropna().reset_index(drop=True)
model.plot_prediction_heatmap(
    y_true_hgb_pct, hgb_pct_ypred.dropna(), 'HGB — pain_reduction_pct')


#%%---------- Step 14a — HGB: pain_under_load_reduction ----------------------

print('\nStep 14a: HGB (Nested CV + Optuna) — pain_under_load_reduction')

hgb_ul_results, hgb_ul_params, hgb_ul_model, hgb_ul_X, hgb_ul_ypred = \
    model.run_advanced_hgb(
        model_datasets['pain_under_load_reduction'][1],
        target_col='pain_under_load_reduction',
    )
print('\nStep 14b: SHAP — HGB (pain_under_load_reduction)')

hgb_ul_shap = model.plot_shap_regressor(
    hgb_ul_model, hgb_ul_X, 'HGB — pain_under_load_reduction')

y_true_hgb_ul = model_datasets['pain_under_load_reduction'][1]['pain_under_load_reduction'].dropna().reset_index(drop=True)

model.plot_prediction_heatmap(
    y_true_hgb_ul, hgb_ul_ypred.dropna(), 'HGB — pain_under_load_reduction')



#%%########## T1 → T3 ANALYSIS ################################################

print('\n' + '#'*60)
print('  T1 → T3 ANALYSIS (immunological T1–T3 differences)')
print('#'*60)


#%%---------- Step 15 — Construct T1→T3 targets and datasets -----------------

print('\nStep 15: Constructing T1→T3 regression targets and model datasets')

pain_targets_t3      = model.construct_datasets_targets(df_cl_vis, 'pain_scale',      [1, 3])
targets_under_load_t3 = model.construct_datasets_targets(df_cl_vis, 'pain_under_load', [1, 3])

t3_unique_targets = {
    'pain_reduction':            pain_targets_t3,
    'pain_under_load_reduction': targets_under_load_t3,
}

model_datasets_t3 = {}
for tgt, tdf in t3_unique_targets.items():
    df_immu_t3, df_comb_t3 = model.create_model_datasets(
        df_cl_mod, df_im_mod, tdf, timepoints=[1, 3]
    )
    model_datasets_t3[tgt] = (df_immu_t3, df_comb_t3)

model_datasets_t3['pain_reduction_pct'] = model_datasets_t3['pain_reduction']


#%%---------- Step 15b — T1→T3 target diagnostics ----------------------------

print('\nStep 15b: T1→T3 target distributions and sample sizes')

t3_plot_targets = ['pain_reduction_pct', 'pain_under_load_reduction']

for tgt in t3_plot_targets:
    df_immu_t3, df_comb_t3 = model_datasets_t3[
        'pain_reduction' if tgt == 'pain_reduction_pct' else tgt
    ]
    y = df_comb_t3[tgt].dropna()
    print(f"\n{'─'*55}")
    print(f"  Target : {tgt}  (T1→T3)")
    print(f"  n (combined, non-NaN target) : {len(y)}")
    print(f"  Features in combined dataset : {df_comb_t3.shape[1]}")
    print(f"  mean={y.mean():.2f}  std={y.std():.2f}  "
          f"min={y.min():.2f}  max={y.max():.2f}")
    print(f"  skew={y.skew():.2f}  kurt={y.kurt():.2f}")
    print(f"  % zeros (no change) : {(y == 0).mean()*100:.1f}%")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    sns.histplot(y, kde=True, ax=axes[0], color=sns.color_palette('mako', 1)[0], bins=20)
    axes[0].set_title(f'{tgt} T1→T3 — distribution')
    axes[0].set_xlabel(tgt)
    axes[1].boxplot(y.dropna(), vert=False)
    axes[1].set_title(f'{tgt} T1→T3 — boxplot')
    axes[1].set_xlabel(tgt)
    plt.tight_layout()
    plt.show()


#%%---------- Step 16a — CatBoost T1→T3: pain_reduction_pct ------------------

print('\nStep 16a: CatBoost T1→T3 (Nested CV + Optuna) — pain_reduction_pct')

cb_t3_pct_results, cb_t3_pct_params, cb_t3_pct_model, cb_t3_pct_X, cb_t3_pct_ypred = \
    model.run_advanced_catboost(
        model_datasets_t3['pain_reduction'][1],
        target_col='pain_reduction_pct',
    )


#%%---------- Step 16b — CatBoost T1→T3 SHAP: pain_reduction_pct ------------

print('\nStep 16b: SHAP — CatBoost T1→T3 (pain_reduction_pct)')

cb_t3_pct_shap = model.plot_shap_regressor(
    cb_t3_pct_model, cb_t3_pct_X, 'CatBoost T1→T3 — pain_reduction_pct')

y_true_cb_t3_pct = model_datasets_t3['pain_reduction'][1]['pain_reduction_pct'].dropna().reset_index(drop=True)
model.plot_prediction_heatmap(
    y_true_cb_t3_pct, cb_t3_pct_ypred.dropna(), 'CatBoost T1→T3 — pain_reduction_pct')


#%%---------- Step 17a — CatBoost T1→T3: pain_under_load_reduction -----------

print('\nStep 17a: CatBoost T1→T3 (Nested CV + Optuna) — pain_under_load_reduction')

cb_t3_ul_results, cb_t3_ul_params, cb_t3_ul_model, cb_t3_ul_X, cb_t3_ul_ypred = \
    model.run_advanced_catboost(
        model_datasets_t3['pain_under_load_reduction'][1],
        target_col='pain_under_load_reduction',
    )


#%%---------- Step 17b — CatBoost T1→T3 SHAP: pain_under_load_reduction -----

print('\nStep 17b: SHAP — CatBoost T1→T3 (pain_under_load_reduction)')

cb_t3_ul_shap = model.plot_shap_regressor(
    cb_t3_ul_model, cb_t3_ul_X, 'CatBoost T1→T3 — pain_under_load_reduction')

y_true_cb_t3_ul = model_datasets_t3['pain_under_load_reduction'][1]['pain_under_load_reduction'].dropna().reset_index(drop=True)
model.plot_prediction_heatmap(
    y_true_cb_t3_ul, cb_t3_ul_ypred.dropna(), 'CatBoost T1→T3 — pain_under_load_reduction')

#%%---------- Step 18a — CatBoost + RENT: pain_reduction_pct (T1→T2) ---------

print('\nStep 18a: CatBoost + RENT (Nested CV + Optuna) — pain_reduction_pct (T1→T2)')

cb_rent_pct_results, cb_rent_pct_params, cb_rent_pct_model, cb_rent_pct_X, cb_rent_pct_ypred, cb_rent_pct_features = \
    model.run_advanced_catboost_rent(
        model_datasets['pain_reduction'][1], target_col='pain_reduction_pct')



#%%---------- Step 18b — CatBoost + RENT SHAP: pain_reduction_pct -----------

print('\nStep 18b: SHAP — CatBoost + RENT (pain_reduction_pct)')

cb_rent_pct_shap = model.plot_shap_regressor(
    cb_rent_pct_model, cb_rent_pct_X, 'CatBoost + RENT — pain_reduction_pct')

y_true_cb_rent_pct = model_datasets['pain_reduction'][1]['pain_reduction_pct'].dropna().reset_index(drop=True)
model.plot_prediction_heatmap(
    y_true_cb_rent_pct, cb_rent_pct_ypred.dropna(), 'CatBoost + RENT — pain_reduction_pct')



#%%---------- Step 19a — CatBoost + RENT: pain_under_load_reduction (T1→T2) --

print('\nStep 19a: CatBoost + RENT (Nested CV + Optuna) — pain_under_load_reduction (T1→T2)')

cb_rent_ul_results, cb_rent_ul_params, cb_rent_ul_model, cb_rent_ul_X, cb_rent_ul_ypred, cb_rent_ul_features = \
    model.run_advanced_catboost_rent(
        model_datasets['pain_under_load_reduction'][1], target_col='pain_under_load_reduction')


#%%---------- Step 19b — CatBoost + RENT SHAP: pain_under_load_reduction -----

print('\nStep 19b: SHAP — CatBoost + RENT (pain_under_load_reduction)')

cb_rent_ul_shap = model.plot_shap_regressor(
    cb_rent_ul_model, cb_rent_ul_X, 'CatBoost + RENT — pain_under_load_reduction')

y_true_cb_rent_ul = model_datasets['pain_under_load_reduction'][1]['pain_under_load_reduction'].dropna().reset_index(drop=True)
model.plot_prediction_heatmap(
    y_true_cb_rent_ul, cb_rent_ul_ypred.dropna(), 'CatBoost + RENT — pain_under_load_reduction')

#%%########## ADVANCED MODELS — ELASTICNET + RENT ##############################

print('\n' + '#'*60)
print('  ELASTICNET + RENT MODEL (Nested CV + Optuna)')
print('#'*60)


#%%---------- Step 20a — ElasticNet + RENT: pain_reduction_pct (T1→T2) -------

print('\nStep 20a: ElasticNet + RENT (Nested CV + Optuna) — pain_reduction_pct (T1→T2)')

en_rent_pct_results, en_rent_pct_params, en_rent_pct_model, en_rent_pct_X, en_rent_pct_ypred, en_rent_pct_features = \
    model.run_advanced_elasticnet_rent(
        model_datasets['pain_reduction'][1], target_col='pain_reduction_pct')

#%%---------- Step 20b — ElasticNet + RENT coefficients: pain_reduction_pct ---

print('\nStep 20b: ElasticNet coefficients — pain_reduction_pct')

en_rent_pct_coef = model.plot_elasticnet_coefficients(
    en_rent_pct_model, list(en_rent_pct_X.columns), 'ElasticNet + RENT — pain_reduction_pct')

y_true_en_rent_pct = model_datasets['pain_reduction'][1]['pain_reduction_pct'].dropna().reset_index(drop=True)
model.plot_prediction_heatmap(
    y_true_en_rent_pct, en_rent_pct_ypred.dropna(), 'ElasticNet + RENT — pain_reduction_pct')

#%%---------- Step 21a — ElasticNet + RENT: pain_under_load_reduction (T1→T2) -

print('\nStep 21a: ElasticNet + RENT (Nested CV + Optuna) — pain_under_load_reduction (T1→T2)')

en_rent_ul_results, en_rent_ul_params, en_rent_ul_model, en_rent_ul_X, en_rent_ul_ypred, en_rent_ul_features = \
    model.run_advanced_elasticnet_rent(
        model_datasets['pain_under_load_reduction'][1], target_col='pain_under_load_reduction')

#%%---------- Step 21b — ElasticNet + RENT coefficients: pain_under_load_reduction -

print('\nStep 21b: ElasticNet coefficients — pain_under_load_reduction')

en_rent_ul_coef = model.plot_elasticnet_coefficients(
    en_rent_ul_model, list(en_rent_ul_X.columns), 'ElasticNet + RENT — pain_under_load_reduction')

y_true_en_rent_ul = model_datasets['pain_under_load_reduction'][1]['pain_under_load_reduction'].dropna().reset_index(drop=True)
model.plot_prediction_heatmap(
    y_true_en_rent_ul, en_rent_ul_ypred.dropna(), 'ElasticNet + RENT — pain_under_load_reduction')

#%%########## ADVANCED MODELS — PLS + RENT #####################################

print('\n' + '#'*60)
print('  PLS + RENT MODEL (Nested CV + Optuna)')
print('#'*60)


#%%---------- Step 22a — PLS + RENT: pain_reduction_pct (T1→T2) ---------------

print('\nStep 22a: PLS + RENT (Nested CV + Optuna) — pain_reduction_pct (T1→T2)')

pls_rent_pct_results, pls_rent_pct_params, pls_rent_pct_model, pls_rent_pct_X, pls_rent_pct_ypred, pls_rent_pct_features = \
    model.run_advanced_pls_rent(
        model_datasets['pain_reduction'][1], target_col='pain_reduction_pct')

#%%---------- Step 22b — PLS + RENT coefficients: pain_reduction_pct ----------

print('\nStep 22b: PLS coefficients — pain_reduction_pct')

pls_rent_pct_coef = model.plot_pls_coefficients(
    pls_rent_pct_model, list(pls_rent_pct_X.columns), 'PLS + RENT — pain_reduction_pct')

y_true_pls_rent_pct = model_datasets['pain_reduction'][1]['pain_reduction_pct'].dropna().reset_index(drop=True)
model.plot_prediction_heatmap(
    y_true_pls_rent_pct, pls_rent_pct_ypred.dropna(), 'PLS + RENT — pain_reduction_pct')

#%%---------- Step 23a — PLS + RENT: pain_under_load_reduction (T1→T2) --------

print('\nStep 23a: PLS + RENT (Nested CV + Optuna) — pain_under_load_reduction (T1→T2)')

pls_rent_ul_results, pls_rent_ul_params, pls_rent_ul_model, pls_rent_ul_X, pls_rent_ul_ypred, pls_rent_ul_features = \
    model.run_advanced_pls_rent(
        model_datasets['pain_under_load_reduction'][1], target_col='pain_under_load_reduction')

#%%---------- Step 23b — PLS + RENT coefficients: pain_under_load_reduction ---

print('\nStep 23b: PLS coefficients — pain_under_load_reduction')

pls_rent_ul_coef = model.plot_pls_coefficients(
    pls_rent_ul_model, list(pls_rent_ul_X.columns), 'PLS + RENT — pain_under_load_reduction')

y_true_pls_rent_ul = model_datasets['pain_under_load_reduction'][1]['pain_under_load_reduction'].dropna().reset_index(drop=True)
model.plot_prediction_heatmap(
    y_true_pls_rent_ul, pls_rent_ul_ypred.dropna(), 'PLS + RENT — pain_under_load_reduction')
