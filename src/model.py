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
import re


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



def create_model_datasets(df1, df2, targets, timepoints,
                          single_dataset=False, include_baseline=False):
    """Create wide-format modeling datasets from clinical and/or immunological data.

    Parameters
    ----------
    df1              : pd.DataFrame  First dataset (immunological)
    df2              : pd.DataFrame  Second dataset (clinical) — ignored if single_dataset=True
    targets          : pd.DataFrame  Output from construct_datasets_targets()
    timepoints       : list[int]     One timepoint → raw features; two → difference features
    single_dataset   : bool          If True, merge only df1 with targets
    include_baseline : bool          If True, include the baseline pain value for the target
    target_col       : str           Required when include_baseline=True - name of baseline column

    Returns
    -------
    df_combined : pd.DataFrame  — features + target columns, one row per patient
    """
    _baseline_map = {
        'pain_reduction':                'pain_scale_t1',
        'pain_reduction_pct':            'pain_scale_t1',
        'pain_under_load_reduction':     'pain_under_load_t1',
        'pain_under_load_reduction_pct': 'pain_under_load_t1',
    }

    t_a     = timepoints[0]
    t_b     = timepoints[1] if len(timepoints) == 2 else None
    id_cols = {'Patient', 'Timepoint'}

    feat_cols_1 = [c for c in df1.columns if c not in id_cols]

    def _at(tp):
        return (df1[df1['Timepoint'] == tp][['Patient'] + feat_cols_1]
                .rename(columns={c: f'{c}_t{tp}' for c in feat_cols_1})
                .reset_index(drop=True))

    if t_b is not None:
        df_m = _at(t_a).merge(_at(t_b), on='Patient', how='inner')
        for c in feat_cols_1:
            df_m[f'{c}_t{t_b}_minus_t{t_a}'] = df_m[f'{c}_t{t_b}'] - df_m[f'{c}_t{t_a}']
        df1_wide = df_m[['Patient'] + [f'{c}_t{t_b}_minus_t{t_a}' for c in feat_cols_1]]
        desc     = f"T{t_a}–T{t_b} difference features"
    else:
        df1_wide = _at(t_a)
        desc     = f"T{t_a} features"

    post_tm_cols = [c for c in targets.columns if t_b and c.endswith(f'_t{t_b}')]
    target_merge = ['Patient'] + [c for c in targets.columns
                                  if c != 'Patient' and c not in post_tm_cols]

    if single_dataset:
        df_combined = df1_wide.merge(targets[target_merge], on='Patient', how='inner')
    else:
        feat_cols_2 = [c for c in df2.columns if c not in id_cols]
        df2_ta      = (df2[df2['Timepoint'] == t_a][['Patient'] + feat_cols_2]
                       .drop_duplicates('Patient').reset_index(drop=True))
        df_combined = (df1_wide
                       .merge(df2_ta,               on='Patient', how='inner')
                       .merge(targets[target_merge], on='Patient', how='inner'))
        desc += " + clinical features"

    drop = {c for c in df_combined.columns if c in set(cl_leaky_columns)}
    if drop:
        df_combined = df_combined.drop(columns=list(drop), errors='ignore')

    keep_col = _baseline_map.get(include_baseline)
    baseline_present = [c for c in target_merge
                        if re.search(r'_t\d+$', c)
                        and c in df_combined.columns
                        and c != keep_col]
    if baseline_present:
        df_combined = df_combined.drop(columns=baseline_present)

    if keep_col:
        desc += f" + baseline ({keep_col})"

    print(f"\nModeling dataset ready: {desc}")
    print(f"Shape: {df_combined.shape},  Patients: {df_combined['Patient'].nunique()}")
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
    exclude      = ['Patient', 'Timepoint', target_col,
                    'pain_reduction', 'pain_reduction_pct', 'pain_under_load_reduction',
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
            iterations=500, loss_function='RMSE',
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

    print(f"\n  Summary ({n_splits}x{n_repeats} CV:")
    for m in metric_cols:
        mv = results_df.loc[results_df['Fold'] == 'Mean', m].iloc[0]
        sv = results_df.loc[results_df['Fold'] == 'Std',  m].iloc[0]
        print(f"    {m:<5}: {mv:.3f} ± {sv:.4f}")

    return results_df



def plot_shap_hgbr(model, X):
    """SHAP bar + beeswarm plots for HGBR model."""
    import shap
    print(f"\n=== SHAP Analysis: HGBR ===")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    shap.summary_plot(shap_values, X, plot_type="bar", show=False, max_display=20)
    plt.title(f"SHAP Feature Importance  HGBR")
    plt.tight_layout()
    plt.show()

    shap.summary_plot(shap_values, X, show=False, max_display=20)
    plt.title(f"SHAP Beeswarm  HGBR")
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
def plot_sweep(sweep_dfs, title='Performance Metrics against Selected Features'):
    """
    Example on Multiple Model plots usage:
    plot_sweep({
        'ElasticNet': sweep_df_en,
        'PLSR':       sweep_df_pls,
        'SVR':        sweep_df_svr,
        'HGBR':       sweep_df_hgbr,
    }, title='Performance Metrics against Selected Features')
    """
    if isinstance(sweep_dfs, pd.DataFrame):
        sweep_dfs = {'Model': sweep_dfs}

    # Sample viridis at spread-out positions for bigger contrast between middle colors
    cmap = plt.cm.viridis
    n = len(sweep_dfs)
    positions = [0.0, 0.45, 0.75, 1.0] if n == 4 else list(np.linspace(0, 1, n))
    model_colors = [cmap(p) for p in positions]

    metrics = ['RMSE', 'MAE', 'R2']
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    for ax, metric in zip(axes, metrics):
        for model_idx, (model_name, sweep_df) in enumerate(sweep_dfs.items()):
            color = model_colors[model_idx]
            x     = sweep_df['threshold']
            mean  = sweep_df[f'mean_{metric}']
            ax.plot(x, mean, marker='o', color=color, label=model_name)
        ax.set_ylabel(metric)
        ax.grid(True, linestyle='--', alpha=0.5)

    # Single legend for the whole figure, outside the subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right',
               bbox_to_anchor=(1.0, 1.0), bbox_transform=axes[0].transAxes)

    first_df = next(iter(sweep_dfs.values()))
    x_vals   = first_df['threshold']
    x_labels = first_df['threshold_label']
    axes[-1].set_xlabel('Threshold')
    axes[-1].set_xticks(x_vals)
    axes[-1].set_xticklabels(labels=x_labels, fontsize=8, rotation=45, ha='right')

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_feature_frequency(feature_freq, name, top=20):
    """Bar plot of feature selection frequency across outer folds.
    Plots top n features. 

    Parameters:
        feature_freq : pd.Series  — selection counts per feature
        name         : str        — plot title suffix
    """
    import matplotlib.pyplot as plt
    n_outer = 20  # outer folds

    # All features selected in at least 1 fold, highest bar at top
    freq_plot = feature_freq[feature_freq > 0].sort_values(ascending=True).tail(top)

    if freq_plot.empty:
        print("  Warning: No features were selected in any fold!")
        return

    fig, ax = plt.subplots(figsize=(8, max(4, len(freq_plot) * 0.35)))
    fig.subplots_adjust(top=0.97)
    bars = ax.barh(freq_plot.index, freq_plot.values, color='teal', edgecolor='white')

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


def jaccard_scores(selected_features_per_fold, name=''):
    """Compute pairwise Jaccard similarity across all outer fold feature sets.

    Parameters
    ----------
    selected_features_per_fold : list of lists   feature sets from each outer fold
    name                       : str              label for print output

    Returns
    -------
    jaccard_matrix : pd.DataFrame   n_folds × n_folds pairwise Jaccard matrix
    mean_jaccard   : float          mean pairwise Jaccard (excluding diagonal)
    std_jaccard    : float          std of pairwise Jaccard values
    """
    from itertools import combinations

    n_folds = len(selected_features_per_fold)
    sets    = [set(fold) for fold in selected_features_per_fold]

    matrix = np.ones((n_folds, n_folds))
    values = []

    for i, j in combinations(range(n_folds), 2):
        intersection = len(sets[i] & sets[j])
        union        = len(sets[i] | sets[j])
        jac          = intersection / union if union > 0 else 0.0
        matrix[i, j] = jac
        matrix[j, i] = jac
        values.append(jac)

    labels = [f'F{i+1}' for i in range(n_folds)]
    jaccard_matrix = pd.DataFrame(matrix, index=labels, columns=labels)

    mean_jac = np.mean(values)
    std_jac  = np.std(values)
    min_jac  = np.min(values)
    max_jac  = np.max(values)

    print(f"\n{'='*55}")
    print(f"  Jaccard Scores — {name}")
    print(f"  Folds: {n_folds}   Pairs evaluated: {len(values)}")
    print(f"{'='*55}")
    print(f"  Mean Jaccard : {mean_jac:.3f} ± {std_jac:.3f}")
    print(f"  Min  Jaccard : {min_jac:.3f}")
    print(f"  Max  Jaccard : {max_jac:.3f}")

    # upper triangle mask — keeps only lower triangle (no duplicate pairs)
    mask = np.triu(np.ones((n_folds, n_folds), dtype=bool))

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(jaccard_matrix, mask=mask, annot=True, fmt='.2f',
                cmap='mako', vmin=0, vmax=1,
                linewidths=0.5, ax=ax)
    ax.set_title(f'Jaccard Scores: Outer Fold Feature Selections — {name}\n'
                 f'Mean={mean_jac:.3f} ± {std_jac:.3f}',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()

    return jaccard_matrix


def pairwise_metric_comparison(results_dict, alpha=0.05):
    """Pairwise paired t-test across model metrics RMSE, MAE and R2."""
    from itertools import combinations
    model_names = list(results_dict.keys())
    n_models    = len(model_names)
    metrics     = ['RMSE', 'MAE', 'R2']
    get_folds   = lambda df, m: df[~df['Fold'].isin(['Mean','Std'])][m].astype(float).values

    #  p-value matrices
    pval_matrices = {}
    for metric in metrics:
        pm = pd.DataFrame(np.ones((n_models, n_models)), index=model_names, columns=model_names)
        for m1, m2 in combinations(model_names, 2):
            _, p = stats.ttest_rel(get_folds(results_dict[m1], metric),
                                   get_folds(results_dict[m2], metric))
            pm.loc[m1, m2] = pm.loc[m2, m1] = p
        pval_matrices[metric] = pm

    # ── Print ─────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}\n  Pairwise Paired t-test (α={alpha})\n{'='*60}")
    for m1, m2 in combinations(model_names, 2):
        print(f"\n  {m1} vs {m2}\n  {'Metric':<8} {'p-value':>10}  {'Sig':>6}\n  {'-'*28}")
        for metric in metrics:
            p = pval_matrices[metric].loc[m1, m2]
            print(f"  {metric:<8} {p:>10.4f}  {'*YES' if p < alpha else 'no':>6}")

    # ── Heatmaps ──────────────────────────────────────────────────────────────
    fig, axes   = plt.subplots(1, 3, figsize=(18, 5))
    mask_diag   = np.eye(n_models, dtype=bool)

    for ax, metric in zip(axes, metrics):
        annot = pd.DataFrame([[
            '—' if m1 == m2 else f"{pval_matrices[metric].loc[m1,m2]:.3f}{' *' if pval_matrices[metric].loc[m1,m2] < alpha else ''}"
            for m2 in model_names] for m1 in model_names],
            index=model_names, columns=model_names)

        sns.heatmap(pval_matrices[metric].astype(float), annot=annot, fmt='',
                    cmap='mako', vmin=0, vmax=1, mask=mask_diag,
                    linewidths=1.0, linecolor='white', ax=ax,
                    cbar_kws={'label': 'p-value'})

        for i in range(n_models):
            ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=True, color='lightgrey', lw=0))
            ax.text(i+0.5, i+0.5, '—', ha='center', va='center', fontsize=11, color='grey')

        ax.set_title(f'{metric}  (α={alpha})', fontsize=12, fontweight='bold')
        ax.set_xlabel('Model'); ax.set_ylabel('Model')

    plt.suptitle('Pairwise Model Metrics Comparison: p-values', fontsize=14, fontweight='bold')
    plt.tight_layout(); plt.show()
    return pval_matrices

