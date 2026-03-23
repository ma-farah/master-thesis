# Modeling functions for merging datasets, create targets, baseline modeling, shap plotting
import time
import optuna
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import HistGradientBoostingRegressor
from catboost import CatBoostRegressor, Pool
import joblib, os
import contextlib, io
import preprocess


# Leaky Columns (ignored during modeling)
cl_leaky_columns= ['response', 'improvement_percent', 'pain_scale', 'pain_under_load',
                    'pain_night', 'pain_daytime', 'pain_at_rest', 'morning_stiffness']

# path to save models
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')


#_________________________________________________________________________________________

def construct_datasets_targets(df1, column_name, timepoints):
    """Computing per-patient regression targets from a clinical column across two timepoints.
    Only patients that :
      - have a non-NaN measurement at T_a
      - have a non-NaN measurement at T_b
      - have a non-NaN computed reduction (i.e. no division-by-zero when value_ta == 0)
    Are included
    Parameters
    ----------
    df1         : pd.DataFrame  Cleaned clinical dataset (df_cl_vis).
                                Must contain 'Patient', 'Timepoint', and column_name.
    column_name : str           Column to build targets from, e.g. 'pain_scale',
                                'pain_daytime', 'pain_under_load'.
    timepoints  : list[int]     [t_a, t_b] — reduction is computed as t_a minus t_b.
                                Typically [1, 2] (baseline → first follow-up).

    Returns
    -------
    targets : pd.DataFrame    (patient ids, and target values)
    """
    t_a, t_b = timepoints[0], timepoints[1]
    col_ta  = f'{column_name}_t{t_a}'
    col_tb  = f'{column_name}_t{t_b}'

    prefix  = column_name.replace('_scale', '')
    col_red = f'{prefix}_reduction'
    col_pct = f'{prefix}_reduction_pct'

    # Extract the column at each timepoint (one row per patient, dropping duplicates)
    ta_vals = (
        df1[df1['Timepoint'] == t_a][['Patient', column_name]]
        .rename(columns={column_name: col_ta})
        .drop_duplicates('Patient')
        .reset_index(drop=True)
    )
    tb_vals = (
        df1[df1['Timepoint'] == t_b][['Patient', column_name]]
        .rename(columns={column_name: col_tb})
        .drop_duplicates('Patient')
        .reset_index(drop=True)
    )

    # Inner join on patient id,
    targets = ta_vals.merge(tb_vals, on='Patient', how='inner')
    targets = targets.dropna(subset=[col_ta, col_tb]).reset_index(drop=True)

    # Calcularing reduction 
    targets[col_red] = targets[col_ta] - targets[col_tb]

    # Percent reduction,  set to NaN when baseline is 0 to avoid division-by-zero
    targets[col_pct] = np.where(
        targets[col_ta] != 0,
        (targets[col_ta] - targets[col_tb]) / targets[col_ta] * 100,
        np.nan,
    )

    # drop any patient whose computed target columns contain nan
    targets = targets.dropna(subset=[col_red, col_pct]).reset_index(drop=True)

    print(f"\n  Target distributions:")
    for c in [col_red, col_pct]:
        s = targets[c]
        print(f"    {c:<42s}  mean={s.mean():.3f}  std={s.std():.3f}"
              f"  [{s.min():.3f}, {s.max():.3f}]")

    return targets



def create_model_datasets(df_cl, df_im, targets, timepoints):
    """Create wide-format modeling datasets from clinical and immunological data.

    Parameters
    ----------
    df_cl      : pd.DataFrame  Cleaned clinical dataset 
                               
    df_im      : pd.DataFrame  Cleaned Immunological dataset 
                          
    targets    : pd.DataFrame  Output from construct_datasets_targets().
                               
    timepoints : list[int]     two timepoints

    Returns
    -------
    df_combined : pd.DataFrame
        One row per patient: immunological T_b−T_a difference features
        + clinical T_a baseline features + target columns.
    """
    t_a, t_b = timepoints[0], timepoints[1]
    id_cols  = {'Patient', 'Timepoint'}

    # Restrict to the two timepoints of interest
    df_im_tp = df_im[df_im['Timepoint'].isin([t_a, t_b])].copy()

    # Identify patients that have measurements at both timepoints
    tp_counts     = df_im_tp.groupby('Patient')['Timepoint'].nunique()
    patients_both = tp_counts[tp_counts == 2].index
    df_im_tp      = df_im_tp[df_im_tp['Patient'].isin(patients_both)]

    # Feature colums
    im_feat_cols = [c for c in df_im_tp.columns if c not in id_cols]

    # Extract timepoint 1 and timpoint 2 separately
    df_im_ta = (
        df_im_tp[df_im_tp['Timepoint'] == t_a][['Patient'] + im_feat_cols]
        .rename(columns={c: f'{c}_t{t_a}' for c in im_feat_cols})
        .reset_index(drop=True)
    )
    df_im_tb = (
        df_im_tp[df_im_tp['Timepoint'] == t_b][['Patient'] + im_feat_cols]
        .rename(columns={c: f'{c}_t{t_b}' for c in im_feat_cols})
        .reset_index(drop=True)
    )

    # Merge, compute difference, drop baseline columns for t1 and t2
    df_im_merged = df_im_ta.merge(df_im_tb, on='Patient', how='inner')
    diff_cols = {}
    for c in im_feat_cols:
        col_name         = f'{c}_t{t_b}_minus_t{t_a}'
        diff_cols[c]     = col_name
        df_im_merged[col_name] = df_im_merged[f'{c}_t{t_b}'] - df_im_merged[f'{c}_t{t_a}']

    # Keep only Patient + difference columns 
    df_im_wide = df_im_merged[['Patient'] + list(diff_cols.values())].copy()

    # clinical features
    
    cl_feat_cols = [c for c in df_cl.columns if c not in id_cols]
    df_cl_t1 = (
        df_cl[df_cl['Timepoint'] == t_a][['Patient'] + cl_feat_cols]
        .drop_duplicates('Patient')
        .reset_index(drop=True)
    )

    print(f"\nTotal Number of Clinical features: {len(cl_feat_cols)}")

    # targets, excluding leaky columns and post-treatment columns
    post_tm_cols = [c for c in targets.columns if c.endswith(f'_t{t_b}')] # drop leaky columns
    target_merge  = ['Patient'] + [c for c in targets.columns
                                   if c != 'Patient' and c not in post_tm_cols]

    # Merging datasets

    # Combined dataset becomes: immu difference features + clinical baseline features + target columns
    df_combined = (
        df_im_wide
        .merge(df_cl_t1, on='Patient', how='inner')
        .merge(targets[target_merge], on='Patient', how='inner')
    )

    baseline_cols = [c for c in target_merge if c.endswith(f'_t{t_a}')]
    drop_cols = set(cl_leaky_columns)
    drop = {c for c in df_combined.columns if c in drop_cols}
    if drop:
        print(f"  Dropping {len(drop)} Columns before modeling: {sorted(drop)}")
        df_combined = df_combined.drop(columns=list(drop), errors='ignore')

    baseline_present = [c for c in baseline_cols if c in df_combined.columns]
    if baseline_present:
        df_combined = df_combined.drop(columns=baseline_present)
        print(f"  Dropped baseline target cols : {baseline_present}")

    print(f"\nModeling datasets ready: (T{t_a}–T{t_b} immunological data + clinical baseline variables:")

    print(f"Shape of Combined Dataset: {df_combined.shape}, "
          f"Number of Patients: {df_combined['Patient'].nunique()}")

    return df_combined


def feature_target_correlation(df_model, target_cols, num_cols, ex_cols=None, n_top=20):
    """Pearson (numeric) + PhiK (categorical) correlation between features and each target.

    Parameters
    ----------
    df_model    : pd.DataFrame   model datasets (combined)
    target_cols : list[str]      target columns to correlate against
    num_cols    : list[str]      numeric/interval columns for phik interval_cols
    ex_cols     : list[str]      extra columns to exclude from features
    n_top       : int            top positive + top negative to show for Pearson (default 20)
    """
    import phik as _phik

    always_exclude = set(list(target_cols) + ['Patient', 'Timepoint'] + (ex_cols or []))

    num_feat_cols = [c for c in df_model.select_dtypes(include='number').columns
                     if c not in always_exclude]
    cat_feat_cols = [c for c in df_model.select_dtypes(include=['category', 'object']).columns
                     if c not in always_exclude]

    print(f"\nFeature–target correlations Combined Datasets: (Pearson: {len(num_feat_cols)} numeric | "
          f"PhiK: {len(cat_feat_cols)} categorical)\n")

    for target in target_cols:
        print(f"{'='*60}")
        print(f"  Target: {target}")
        print(f"{'='*60}")

        # ── Pearson (numeric) ─────────────────────────────────────
        sub = df_model[[target] + num_feat_cols].dropna(subset=[target])
        pearson_records = []
        for col in num_feat_cols:
            vals = sub[[target, col]].dropna()
            if len(vals) < 5:
                continue
            r = stats.pearsonr(vals[col], vals[target]).statistic
            pearson_records.append({'Feature': col, 'r': r})

        pearson_df = (pd.DataFrame(pearson_records)
                      .sort_values('r', ascending=False)
                      .reset_index(drop=True))

        top_pos = pearson_df.head(n_top)
        top_neg = pearson_df.tail(n_top).sort_values('r')

        print(f"\n  Pearson Correlations Combined Dataset - Numeric Features: top {n_top} positive:")
        print(f"  {'Feature':<40}  {'r':>7}")
        print("  " + "-" * 50)
        for _, row in top_pos.iterrows():
            print(f"  {row['Feature']:<40}  {row['r']:>7.3f}")

        print(f"\n  Pearson Correlations Combined Dataset - Numeric Features: top {n_top} negative:")
        print(f"  {'Feature':<40}  {'r':>7}")
        print("  " + "-" * 50)
        for _, row in top_neg.iterrows():
            print(f"  {row['Feature']:<40}  {row['r']:>7.3f}")

        # ── PhiK (categorical) ────────────────────────────────────
        if cat_feat_cols:
            df_phik = df_model[[target] + cat_feat_cols].copy()
            for c in df_phik.select_dtypes(['category', 'object']).columns:
                df_phik[c] = df_phik[c].astype(str).replace('nan', np.nan)

            interval_cols = [c for c in num_cols if c == target]
            phik_matrix   = df_phik.phik_matrix(interval_cols=interval_cols)

            phik_df = (phik_matrix[[target]]
                       .drop(index=target, errors='ignore')
                       .rename(columns={target: 'phik'})
                       .reset_index()
                       .rename(columns={'index': 'Feature'})
                       .sort_values('phik', ascending=False)
                       .reset_index(drop=True))

            print(f"\n  PhiK Correlations Combined Dataset — Categorical Features:")
            print(f"  {'Feature':<40}  {'phik':>6}")
            print("  " + "-" * 50)
            for _, row in phik_df.iterrows():
                print(f"  {row['Feature']:<40}  {row['phik']:>6.3f}")

        print()


# BASELINE CATBOOST MODEL
# ___________________________
def run_baseline_catboost(df_model, target_col, name,
                           n_splits=5, n_repeats=5, random_state=42,
                           target_transformer=None):
    """Baseline CatBoostRegressor with RepeatedKFold cross-validation.

    """
    y = df_model[target_col].copy()
    exclude      = ['Patient', 'Timepoint', target_col, 'pain_reduction',
                    'pain_reduction_pct', 'pain_under_load_reduction',
                    'pain_under_load_reduction_pct']
    feature_cols = [c for c in df_model.columns if c not in exclude]
    X            = df_model[feature_cols].copy()

    valid = y.notna()
    X, y  = X[valid].reset_index(drop=True), y[valid].reset_index(drop=True)

    for col in X.select_dtypes(include=['category', 'object']).columns:
        X[col] = X[col].astype(object).fillna('missing')
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()

    print(f"\n{'='*65}")
    print(f"  CatBoost Regressor Baseline — {name}")
    print(f"  Target : {target_col}")
    print(f"  Samples: {len(X)},  Features: {len(feature_cols)}")
    print(f"  CV     : {n_splits}-fold × {n_repeats} repeats = {n_splits * n_repeats} fits")
    print(f"{'='*65}")

    rkf          = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    fold_results = []
    y_pred_sum   = pd.Series(0.0, index=range(len(X)))
    y_pred_count = pd.Series(0,   index=range(len(X)))

    for fold, (train_idx, test_idx) in enumerate(rkf.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        if target_transformer is not None:
            pt_fold     = clone(target_transformer)
            y_train_fit = pd.Series(
                pt_fold.fit_transform(y_train.values.reshape(-1, 1)).ravel(),
                index=y_train.index)
        else:
            pt_fold, y_train_fit = None, y_train

        model = CatBoostRegressor(
            iterations=1000, loss_function='RMSE',
            random_seed=random_state, task_type='CPU',
            thread_count=-1, verbose=0)
        model.fit(Pool(X_train, y_train_fit, cat_features=cat_cols))

        preds_raw = model.predict(X_test)
        preds     = (pt_fold.inverse_transform(preds_raw.reshape(-1, 1)).ravel()
                     if pt_fold is not None else preds_raw)

        # Average predictions across repeats
        y_pred_sum.iloc[test_idx]   += preds
        y_pred_count.iloc[test_idx] += 1

        mae  = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2   = r2_score(y_test, preds)
        fold_results.append({'Fold': fold + 1, 'MAE': mae, 'MSE': rmse**2, 'RMSE': rmse, 'R2': r2})
        print(f"  Fold {fold+1:>2}: MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}")

    y_pred = y_pred_sum / y_pred_count  # averaged across repeats

    results_df  = pd.DataFrame(fold_results)
    metric_cols = ['MAE', 'MSE', 'RMSE', 'R2']
    mean_row    = {'Fold': 'Mean', **{m: results_df[m].mean() for m in metric_cols}}
    std_row     = {'Fold': 'Std',  **{m: results_df[m].std()  for m in metric_cols}}
    results_df  = pd.concat([results_df, pd.DataFrame([mean_row, std_row])], ignore_index=True)

    n_folds = n_splits * n_repeats
    t_crit  = stats.t.ppf(0.975, df=n_folds - 1)
    print(f"\n  Summary ({n_splits}x{n_repeats} CV, 95% CI):")
    for m in metric_cols:
        mv = results_df.loc[results_df['Fold'] == 'Mean', m].iloc[0]
        sv = results_df.loc[results_df['Fold'] == 'Std',  m].iloc[0]
        ci = t_crit * sv / np.sqrt(n_folds)
        print(f"    {m:<5}: {mv:.3f} ± {sv:.4f}  (95% CI [{mv-ci:.3f}, {mv+ci:.3f}])")

    return results_df, model, X, y_pred



def plot_shap_catboost(model, X):
    """SHAP bar + beeswarm plots for a catboost model."""
    import shap
    print(f"\n=== SHAP Analysis: CatBoost ===")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    shap.summary_plot(shap_values, X, plot_type="bar", show=False, max_display=20)
    plt.title(f"SHAP Feature Importance  CatBoost")
    plt.tight_layout()
    plt.show()

    shap.summary_plot(shap_values, X, show=False, max_display=20)
    plt.title(f"SHAP Beeswarm  CatBoost")
    plt.tight_layout()
    plt.show()

    return shap_values


def plot_shap_elasticnet(model, X, scaler):
    """SHAP bar + beeswarm for a fitted ElasticNet.
    Divides SHAP values by scaler.scale_ to convert from scaled to original units
    """
    import shap
    print(f"\n=== SHAP Analysis: Elasticnet ===")

    # Inverse transform X
    X_original = pd.DataFrame(
        scaler.inverse_transform(X),
        columns=X.columns, index=X.index)

    explainer   = shap.LinearExplainer(model, X, feature_perturbation="correlation_dependent")
    shap_values = explainer.shap_values(X)

    # Convert SHAP values to original units
    shap_values = shap_values / scaler.scale_

    shap.summary_plot(shap_values, X_original, plot_type="bar", show=False, max_display=20)
    plt.title(f"SHAP Feature Importance Elasticnet")
    plt.tight_layout()
    plt.show()
    shap.summary_plot(shap_values, X_original, show=False, max_display=20)
    plt.title(f"SHAP Beeswarm Elasticnet")
    plt.tight_layout()
    plt.show()
    
    return shap_values


def plot_shap_svr(model, X, scaler, n_background=20):
    """KernelExplainer SHAP for SVR (RBF).
    Works in original feature units by inverse transforming X
    and wrapping predict to handle scaling internally.
    """
    import shap

    print(f"\n=== SHAP Analysis: SVR ===")

    # Inverse transform X to original units
    X_original = pd.DataFrame(
        scaler.inverse_transform(X),
        columns=X.columns,
        index=X.index
    )

    # Wrapping predicter
    def predict_fn(X_org):
        X_s = scaler.transform(X_org)
        return model.predict(X_s)

    background  = shap.kmeans(X_original, n_background)
    explainer   = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(X_original, nsamples=100)

    shap.summary_plot(shap_values, X_original, plot_type='bar', show=False, max_display=20)
    plt.title('SHAP Feature Importance  SVR')
    plt.tight_layout()
    plt.show()

    shap.summary_plot(shap_values, X_original, show=False, max_display=20)
    plt.title('SHAP Beeswarm  SVR')
    plt.tight_layout()
    plt.show()

    return shap_values 


def plot_shap_pls(model, X, scaler, n_background=20):
    """KernelExplainer SHAP for PLSRegression.
    """
    import shap
    print(f"\n=== SHAP Analysis: PLS ===")

    # Inverse transform X to original units
    X_original = pd.DataFrame(
        scaler.inverse_transform(X),
        columns=X.columns, index=X.index)

    def predict_fn(X_org):
        X_s = scaler.transform(X_org)
        return model.predict(X_s).ravel()

    background  = shap.kmeans(X_original, n_background)
    explainer   = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(X_original)

    shap.summary_plot(shap_values, X_original, plot_type='bar', show=False, max_display=20)
    plt.title(f'SHAP Feature Importance  PLS')
    plt.tight_layout()
    plt.show()
    shap.summary_plot(shap_values, X_original, show=False, max_display=20)
    plt.title(f'SHAP Beeswarm  PLS')
    plt.tight_layout()
    plt.show()
    
    return shap_values


def plot_feature_frequency(feature_freq, name, top_n=30, n_outer=20, threshold=0.50):
    """Bar plot of feature selection frequency across outer folds.

    Parameters:
        feature_freq : pd.Series  — selection counts per feature 
        name         : str        — plot title suffix
        top_n        : int        — max features to display (default 30)
        n_outer      : int        — total number of outer folds (default 20)
        threshold    : float      — frequency threshold to highlight (default 0.50)
    """
    import matplotlib.pyplot as plt

    threshold_count = threshold * n_outer

    # Filter to features selected at least once, take top_n
    freq_plot = (feature_freq[feature_freq > 0]
                 .nlargest(top_n)
                 .sort_values(ascending=True))  # ascending → highest bar at top

    if freq_plot.empty:
        print("  Warning: No features were selected in any fold!")
        return

    fig, ax = plt.subplots(figsize=(8, max(4, len(freq_plot) * 0.4)))

    colors = ['steelblue' if v >= threshold_count else 'lightsteelblue'
              for v in freq_plot.values]

    bars = ax.barh(freq_plot.index, freq_plot.values, color=colors, edgecolor='white')

    # Threshold line
    ax.axvline(x=threshold_count, color='tomato', linestyle='--', linewidth=1.5,
               label=f'≥{int(threshold*100)}% threshold ({int(threshold_count)}/{n_outer} folds)')

    # Value labels on bars
    for bar, val in zip(bars, freq_plot.values):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                f'{int(val)}/{n_outer}',
                va='center', ha='left', fontsize=9)

    ax.set_xlabel('Number of outer folds selected')
    ax.set_xlim(0, n_outer + 2)
    ax.set_xticks(range(0, n_outer + 1, 4))
    ax.set_title(f'MRMR Feature Selection Frequency — {name}')
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.show()
