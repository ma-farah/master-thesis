# Modeling functions — baseline and advanced regressors
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
# shap is imported lazily inside plot_shap_regressor / plot_shap_pipeline
# to avoid kernel crashes on Windows with restricted execution policies.
import joblib, os
import contextlib, io
import preprocess


# ── Constants ─────────────────────────────────────────────────────────────────

# Leaky Columns (ignored during modeling)
cl_leaky_columns= ['response', 'improvement_percent', 'pain_scale', 'pain_under_load',
                    'pain_night', 'pain_daytime', 'pain_at_rest', 'morning_stiffness']


# path to save models
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')


# ══════════════════════════════════════════════════════════════════════════════

def construct_datasets_targets(df1, column_name, timepoints):
    """Compute per-patient regression targets from a clinical column across two timepoints.

    For column_name and timepoints [t_a, t_b], computes per patient:
      - {col}_t{ta}            : raw baseline value (T_a)
      - {col}_t{tb}            : raw post-treatment value (T_b)  ← leaky, for reference only
      - {col}_reduction        : absolute reduction  = value_ta - value_tb
      - {col}_reduction_pct    : percent reduction   = reduction / value_ta × 100

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
    targets : pd.DataFrame
        One row per eligible patient with columns:
          Patient, {col}_t{ta}, {col}_t{tb}, {prefix}_reduction, {prefix}_reduction_pct
        where prefix = column_name with '_scale' stripped (e.g. 'pain_scale' → 'pain').
        Patients with NaN in any computed target column are excluded.
    """
    t_a, t_b = timepoints[0], timepoints[1]
    col_ta  = f'{column_name}_t{t_a}'
    col_tb  = f'{column_name}_t{t_b}'

    # Strip '_scale' from the prefix so 'pain_scale' becomes 'pain_reduction'
    # and 'pain_reduction_pct' rather than 'pain_scale_reduction[_pct]'.
    prefix  = column_name.replace('_scale', '')
    col_red = f'{prefix}_reduction'
    col_pct = f'{prefix}_reduction_pct'

    # Extract the column at each timepoint (one row per patient, drop duplicates)
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

    # Inner join: keep only patients present at BOTH timepoints with non-NaN values
    targets = ta_vals.merge(tb_vals, on='Patient', how='inner')
    targets = targets.dropna(subset=[col_ta, col_tb]).reset_index(drop=True)

    # Absolute reduction (positive = improvement)
    targets[col_red] = targets[col_ta] - targets[col_tb]

    # Percent reduction: set to NaN when baseline is 0 to avoid division-by-zero
    targets[col_pct] = np.where(
        targets[col_ta] != 0,
        (targets[col_ta] - targets[col_tb]) / targets[col_ta] * 100,
        np.nan,
    )

    # Drop any patient whose computed target columns contain NaN
    targets = targets.dropna(subset=[col_red, col_pct]).reset_index(drop=True)

    print(f"\n  Target distributions:")
    for c in [col_red, col_pct]:
        s = targets[c]
        print(f"    {c:<42s}  mean={s.mean():.3f}  std={s.std():.3f}"
              f"  [{s.min():.3f}, {s.max():.3f}]")

    return targets



def create_model_datasets(df_cl, df_im, targets, timepoints):
    """Create wide-format modeling datasets from clinical and immunological data.

    Patients with NaN in all target columns after merging are excluded 

    Parameters
    ----------
    df_cl      : pd.DataFrame  Cleaned clinical dataset 
                               Must contain 'Patient', 'Timepoint', and clinical features.
    df_im      : pd.DataFrame  Cleaned Immunological dataset 
                               Must contain 'Patient', 'Timepoint', and immu features.
    targets    : pd.DataFrame  Output from construct_datasets_targets().
                               Must contain 'Patient' + target columns.
    timepoints : list[int]     [t_a, t_b] to define the immunological difference direction.
                               Typically [1, 2].

    Returns
    -------
    df_combined : pd.DataFrame
        One row per patient: immunological T_b−T_a difference features
        + clinical T_a baseline features + target columns.
    """
    t_a, t_b = timepoints[0], timepoints[1]
    id_cols  = {'Patient', 'Timepoint'}

    # ── IMMUNOLOGICAL: T_a − T_b differences only (one row per patient) ────────

    # Restrict to the two timepoints of interest
    df_im_tp = df_im[df_im['Timepoint'].isin([t_a, t_b])].copy()

    # Identify patients that have measurements at BOTH timepoints
    tp_counts     = df_im_tp.groupby('Patient')['Timepoint'].nunique()
    patients_both = tp_counts[tp_counts == 2].index
    df_im_tp      = df_im_tp[df_im_tp['Patient'].isin(patients_both)]

    # Feature columns = everything except ID columns
    im_feat_cols = [c for c in df_im_tp.columns if c not in id_cols]

    # Extract T_a and T_b separately; rename columns with timepoint suffix (temporary)
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

    # Merge to align T_a and T_b rows; compute difference; drop raw T_a and T_b columns
    df_im_merged = df_im_ta.merge(df_im_tb, on='Patient', how='inner')
    diff_cols = {}
    for c in im_feat_cols:
        col_name         = f'{c}_t{t_b}_minus_t{t_a}'
        diff_cols[c]     = col_name
        df_im_merged[col_name] = df_im_merged[f'{c}_t{t_b}'] - df_im_merged[f'{c}_t{t_a}']

    # Keep only Patient + difference columns (discard raw T_a and T_b feature columns)
    df_im_wide = df_im_merged[['Patient'] + list(diff_cols.values())].copy()

    # ── CLINICAL: T_a baseline rows only ─────────────────────────────────────────
    
    cl_feat_cols = [c for c in df_cl.columns if c not in id_cols]
    df_cl_t1 = (
        df_cl[df_cl['Timepoint'] == t_a][['Patient'] + cl_feat_cols]
        .drop_duplicates('Patient')
        .reset_index(drop=True)
    )

    print(f"\nTotal Number of Clinical features: {len(cl_feat_cols)}")

    # ── TARGETS: exclude leaky post-treatment columns
    post_tm_cols = [c for c in targets.columns if c.endswith(f'_t{t_b}')] # drop leaky columns
    target_merge  = ['Patient'] + [c for c in targets.columns
                                   if c != 'Patient' and c not in post_tm_cols]

    # ── MERGE into final dataset ──────────────────────────────────────────────

    # Combined: immu difference features + clinical T_a baseline + target columns
    df_combined = (
        df_im_wide
        .merge(df_cl_t1, on='Patient', how='inner')
        .merge(targets[target_merge], on='Patient', how='inner')
    )

    baseline_cols = [c for c in target_merge if c.endswith(f'_t{t_a}')]

    # Drop columns
    drop_cols = set(cl_leaky_columns)
    drop = {c for c in df_combined.columns if c in drop_cols}
    if drop:
        print(f"  Dropping {len(drop)} Columns before modeling: {sorted(drop)}")
        df_combined = df_combined.drop(columns=list(drop), errors='ignore')

    # Drop baseline target values (e.g. pain_scale_t1) — regression-to-mean confound
    baseline_present = [c for c in baseline_cols if c in df_combined.columns]
    if baseline_present:
        df_combined = df_combined.drop(columns=baseline_present)
        print(f"  Dropped baseline target cols : {baseline_present}")

    print(f"\nModeling datasets ready: (T{t_a}–T{t_b} immunological data + clinical baseline variables:")

    print(f"Shape of Combined Dataset: {df_combined.shape}, "
          f"Number of Patients: {df_combined['Patient'].nunique()}")

    return df_combined



# BASELINE CATBOOST MODEL
# ══════════════════════════════════════════════════════════════════════════════

def regression_metrics(y_true, y_pred):
    """Compute standard regression metrics for a single prediction array.

    Parameters
    ----------
    y_true : array-like   Ground-truth target values.
    y_pred : array-like   Model-predicted values, same length as y_true.

    Returns
    -------
    dict with keys 'MAE', 'MSE', 'RMSE', 'R2' (float values).
    """
    # Compute each metric individually so callers can inspect any subset
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, y_pred)
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}



def run_catboost_regressor(df_model, target_col, name,
                           n_splits=5, n_repeats=5, random_state=42,
                           target_transformer=None):
    """Run a baseline CatBoostRegressor with RepeatedKFold cross-validation.
    Parameters uses CPU as task type
    ----------
    df_model     : pd.DataFrame   Dataset containing features and target column.
    target_col   : str            Name of the regression target column.
    name         : str            Label used in printed output and summaries.
    n_splits     : int            Number of CV folds (default 5).
    n_repeats    : int            Number of CV repetitions (default 5).
    random_state : int            Seed for RepeatedKFold and CatBoost (default 42).

    Returns
    -------
    results_df : pd.DataFrame       Per-fold MAE/MSE/RMSE/R2 plus Mean and Std rows.
    model      : CatBoostRegressor  Model trained on the final CV fold.
    X          : pd.DataFrame       Feature matrix used (rows with non-NaN target).
    y_pred     : pd.Series          Out-of-fold predictions aligned to X's index.
    """
    # Build the exclusion set: ID columns + target_col.

    y = df_model[target_col].copy() # target
    # exlude id features and other targets
    exclude = ['Patient', 'Timepoint', target_col, 'pain_reduction', 'pain_reduction_pct', 'pain_under_load_reduction', 'pain_under_load_reduction_pct']
    feature_cols = [c for c in df_model.columns if c not in exclude]
    X = df_model[feature_cols].copy()

    valid = y.notna()
    X, y = X[valid].reset_index(drop=True), y[valid].reset_index(drop=True)

    # Convert category dtype to str so CatBoost treats them as categorical features
    for col in X.select_dtypes(include=['category', 'object']).columns:
        X[col] = X[col].astype(str)
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()

    # Print run header with dataset dimensions and CV configuration
    print(f"\n{'='*65}")
    print(f"  CatBoost Regressor Baseline — {name}")
    print(f"  Target : {target_col}")
    print(f"  Samples: {len(X)},  Features: {len(feature_cols)}")
    print(f"  CV     : {n_splits}-fold × {n_repeats} repeats = {n_splits * n_repeats} fits")
    print(f"{'='*65}")

    # Initialise CV splitter and containers for per-fold results and predictions
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    fold_results = []
    y_pred = pd.Series(np.nan, index=range(len(X)), dtype='float64')

    # Train one CatBoostRegressor per fold, collect metrics, and store predictions
    for fold, (train_idx, test_idx) in enumerate(rkf.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        if target_transformer is not None:
            pt_fold     = clone(target_transformer)
            y_train_fit = pd.Series(
                pt_fold.fit_transform(y_train.values.reshape(-1, 1)).ravel(),
                index=y_train.index,
            )
        else:
            pt_fold     = None
            y_train_fit = y_train

        model = CatBoostRegressor(
            iterations=1000,
            loss_function='RMSE',
            random_seed=random_state,
            task_type='CPU', 
            thread_count=-1, #using all cores in paralell
            verbose=0,
        )
        model.fit(Pool(X_train, y_train_fit, cat_features=cat_cols))

        preds_raw = model.predict(X_test)
        preds = (pt_fold.inverse_transform(preds_raw.reshape(-1, 1)).ravel()
                 if pt_fold is not None else preds_raw)
        y_pred.iloc[test_idx] = preds

        # Metrics computed in original-space (inverse-transformed if transformer provided)
        mae  = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mse  = rmse ** 2
        r2   = r2_score(y_test, preds)
        m = {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}
        fold_results.append({'Fold': fold + 1, **m})
        print(f"  Fold {fold+1:>2}: MAE={m['MAE']:.3f}  MSE={m['MSE']:.3f}  "
              f"RMSE={m['RMSE']:.3f}  R²={m['R2']:.3f}")

    # Append Mean and Std summary rows to the per-fold results DataFrame
    results_df = pd.DataFrame(fold_results)
    metric_cols = ['MAE', 'MSE', 'RMSE', 'R2']
    mean_row = {'Fold': 'Mean', **{m: results_df[m].mean() for m in metric_cols}}
    std_row  = {'Fold': 'Std',  **{m: results_df[m].std()  for m in metric_cols}}
    results_df = pd.concat(
        [results_df, pd.DataFrame([mean_row, std_row])], ignore_index=True)

    # Print the mean ± std ± 95% CI summary for each metric
    n_folds = n_splits * n_repeats
    t_crit  = stats.t.ppf(0.975, df=n_folds - 1)
    print(f"\n  Summary ({n_splits}x{n_repeats} CV, 95% CI):")
    for m in metric_cols:
        mv  = results_df.loc[results_df['Fold'] == 'Mean', m].iloc[0]
        sv  = results_df.loc[results_df['Fold'] == 'Std',  m].iloc[0]
        ci  = t_crit * sv / np.sqrt(n_folds)
        print(f"    {m:<5}: {mv:.3f} ± {sv:.4f}  (95% CI [{mv - ci:.3f}, {mv + ci:.3f}])")

    return results_df, model, X, y_pred



def plot_shap_regressor(model, X, name):
    """SHAP bar + beeswarm plots for a fitted CatBoostRegressor."""
    import shap
    print(f"\n=== SHAP Analysis: {name} ===")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    shap.summary_plot(shap_values, X, plot_type="bar", show=False, max_display=20)
    plt.title(f"SHAP Feature Importance — {name}")
    plt.tight_layout()
    plt.show()

    shap.summary_plot(shap_values, X, show=False, max_display=20)
    plt.title(f"SHAP Beeswarm — {name}")
    plt.tight_layout()
    plt.show()

    return shap_values

def plot_shap_elasticnet(model, X, name, scaler=None):
    """SHAP bar + beeswarm for a fitted ElasticNet.

    If scaler is provided, SHAP values are divided by scaler.scale_ to convert
    from scaled units back to original feature units.
    """
    import shap

    print(f"\n=== SHAP Analysis: {name} ===")
    explainer   = shap.LinearExplainer(model, X, feature_perturbation="correlation_dependent")
    shap_values = explainer.shap_values(X)

    if scaler is not None:
        shap_values = shap_values / scaler.scale_

    shap.summary_plot(shap_values, X, plot_type="bar", show=False, max_display=20)
    plt.title(f"SHAP Feature Importance — {name}")
    plt.tight_layout()
    plt.show()
    shap.summary_plot(shap_values, X, show=False, max_display=20)
    plt.title(f"SHAP Beeswarm — {name}")
    plt.tight_layout()
    plt.show()

    return shap_values

def plot_pls_importance(model, X, name):
    """VIP scores + coefficients plot for a fitted PLSRegression."""
    import matplotlib.pyplot as plt

    # ── VIP scores ────────────────────────────────────────────────────────
    t = model.x_scores_
    w = model.x_weights_
    q = model.y_loadings_
    p, h     = w.shape
    vip      = np.zeros(p)
    s        = np.diag(t.T @ t @ q.T @ q)
    for i in range(p):
        weight = np.array([
            (w[i, j] / np.linalg.norm(w[:, j]))**2 for j in range(h)])
        vip[i] = np.sqrt(p * (s @ weight) / np.sum(s))

    # ── Coefficients for direction ────────────────────────────────────────
    coefs = model.coef_.ravel()

    importance_df = pd.DataFrame({
        'feature':     X.columns,
        'vip':         vip,
        'coefficient': coefs,
        'signed_vip':  vip * np.sign(coefs)   # direction from coefficient
    }).sort_values('vip', ascending=True)

    # Only show VIP > 0.8
    importance_df = importance_df[importance_df['vip'] > 0.8]

    colors = ['#d73027' if s > 0 else '#4575b4'
              for s in importance_df['signed_vip']]

    fig, ax = plt.subplots(figsize=(8, max(4, len(importance_df) * 0.4 + 2)))
    ax.barh(importance_df['feature'], importance_df['signed_vip'],
            color=colors, edgecolor='white', height=0.7)
    ax.axvline(x=0,    color='black', linewidth=0.8)
    ax.axvline(x=1.0,  color='gray',  linewidth=0.8,
               linestyle='--', label='VIP=1.0 threshold')
    ax.axvline(x=-1.0, color='gray',  linewidth=0.8, linestyle='--')
    ax.set_xlabel('VIP Score (signed by coefficient direction)')
    ax.set_title(f'PLS Feature Importance\n{name}\n'
                 f'(+) = higher marker → more pain reduction')
    ax.legend()
    plt.tight_layout()
    plt.show()
    return importance_df



def plot_feature_frequency(feature_freq, name, threshold=0.75, n_outer=20, top_n=30):
    """Bar plot of RENT feature selection frequency across outer folds.
    
    Parameters:
        feature_freq : pd.Series  — selection counts per feature (from run_* functions)
        name         : str        — plot title suffix
        threshold    : float      — frequency threshold used for final model (default 0.75)
        n_outer      : int        — total number of outer folds (default 20)
        top_n        : int        — max features to display (default 30)
    """
    import matplotlib.pyplot as plt

    threshold_count = threshold * n_outer

    # Filter to features selected at least once, take top_n
    freq_plot = (feature_freq[feature_freq > 0]
                 .nlargest(top_n)
                 .sort_values(ascending=True))  # ascending → highest bar at top

    if freq_plot.empty:
        print("   Warning: No features were selected in any fold!")
        return

    # Red = met threshold, blue = did not
    colors = ['#d73027' if v >= threshold_count else '#4575b4'
              for v in freq_plot.values]

    fig, ax = plt.subplots(figsize=(8, max(4, len(freq_plot) * 0.4 + 2)))

    ax.barh(freq_plot.index, freq_plot.values,
            color=colors, edgecolor='white', height=0.7)

    # Threshold line
    ax.axvline(x=threshold_count, color='black', linewidth=1.2,
               linestyle='--',
               label=f'≥{int(threshold*100)}% threshold '
                     f'({int(threshold_count)}/{n_outer} folds)')

    # Annotate count on each bar
    for i, (feat, val) in enumerate(freq_plot.items()):
        ax.text(val + 0.1, i, f'{int(val)}/{n_outer}',
                va='center', fontsize=8, color='black')

    ax.set_xlim(0, n_outer + 2)
    ax.set_xlabel(f'Selection count (out of {n_outer} outer folds)')
    ax.set_title(f'RENT Feature Selection Frequency\n{name}')
    ax.legend(loc='lower right')

    # Color legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#d73027', label=f'Selected in ≥{int(threshold*100)}% folds → final model'),
        Patch(facecolor='#4575b4', label=f'Selected in <{int(threshold*100)}% folds → excluded'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    plt.tight_layout()
    plt.show()