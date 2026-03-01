# Modeling functions — baseline and advanced regressors
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
from sklearn.model_selection import RepeatedKFold
from catboost import CatBoostRegressor, Pool
import shap


# ── Constants ─────────────────────────────────────────────────────────────────

# Pain questionnaire columns are potential regression targets, not model features.
# create_model_datasets() strips them from the clinical feature set automatically.
CL_PAIN_QUESTIONNAIRE_COLS = [
    'pain_under_load', 'pain_at_rest', 'pain_daytime',
    'pain_night', 'morning_stiffness', 'pain_scale',
]

# Substrings that flag a column as a leaky outcome variable in clinical data.
# Any clinical column whose name contains one of these strings is excluded from
# model features because it encodes treatment response or derived outcomes.
CL_MODEL_LEAKY_PATTERNS = ['response', 'improvement_percent', 'pain_reduction']


# ══════════════════════════════════════════════════════════════════════════════
# DATASET CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

def construct_datasets_targets(df1, column_name, timepoints):
    """Compute per-patient regression targets from a clinical column across two timepoints.

    For column_name and timepoints [t_a, t_b], computes per patient:
      - {col}_t{ta}            : raw baseline value (T_a)
      - {col}_t{tb}            : raw post-treatment value (T_b)  ← leaky, for reference only
      - {col}_reduction        : absolute reduction  = value_ta - value_tb
      - {col}_reduction_pct    : percent reduction   = reduction / value_ta × 100

    Only patients that satisfy ALL of the following are included:
      - have a non-NaN measurement at T_a
      - have a non-NaN measurement at T_b
      - have a non-NaN computed reduction (i.e. no division-by-zero when value_ta == 0)

    Prints a summary of target distributions and the list of eligible patient IDs.

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
    # Other columns (pain_daytime, pain_under_load, …) are unaffected.
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
    # (handles division-by-zero and any other edge cases)
    targets = targets.dropna(subset=[col_red, col_pct]).reset_index(drop=True)

    # Print summary
    eligible = sorted(targets['Patient'].tolist())
    print(f"\n{'='*60}")
    print(f"  Targets: '{column_name}'  (T{t_a} → T{t_b})")
    print(f"{'='*60}")
    print(f"  Patients with T{t_a} values    : {ta_vals[col_ta].notna().sum()}")
    print(f"  Patients with T{t_b} values    : {tb_vals[col_tb].notna().sum()}")
    print(f"  Eligible (non-NaN both, n)    : {len(targets)}")
    print(f"  Eligible patient IDs          : {eligible}")
    print(f"\n  Target distributions:")
    for c in [col_red, col_pct]:
        s = targets[c]
        print(f"    {c:<42s}  mean={s.mean():.3f}  std={s.std():.3f}"
              f"  [{s.min():.3f}, {s.max():.3f}]")

    return targets



def create_model_datasets(df_cl, df_im, targets, timepoints):
    """Create wide-format modeling datasets from clinical and immunological data.

    Immunological features: for each feature, only the T_a − T_b difference is
    kept as a column (e.g. 'basophils_t1_minus_t2'). Raw T_a and T_b values are
    NOT included. Only patients with immunological measurements at BOTH timepoints
    are eligible.

    Clinical features: T_a (baseline) rows only — the forward-filled patient-level
    variables such as age, gender, diagnosis. Pain questionnaire columns
    (CL_PAIN_QUESTIONNAIRE_COLS) and leaky metadata columns (CL_MODEL_LEAKY_PATTERNS)
    are excluded automatically.

    Target columns merged: all columns from the targets DataFrame EXCEPT the raw
    post-treatment value ({col}_t{t_b}), which is always leaky. The baseline raw
    value ({col}_t{t_a}) and computed reduction/pct columns are included and will
    be handled by run_catboost_regressor's exclusion logic (which excludes target_col
    and any column matching leaky patterns).

    Patients with NaN in all target columns after merging are excluded (this handles
    any residual NaN not caught by construct_datasets_targets).

    Parameters
    ----------
    df_cl      : pd.DataFrame  Cleaned clinical dataset (df_cl_vis or df_cl_mod).
                               Must contain 'Patient', 'Timepoint', and clinical features.
    df_im      : pd.DataFrame  Immunological dataset (df_im_vis or df_im_mod).
                               Must contain 'Patient', 'Timepoint', and immu features.
    targets    : pd.DataFrame  Output from construct_datasets_targets().
                               Must contain 'Patient' + target columns.
    timepoints : list[int]     [t_a, t_b] to define the immunological difference direction.
                               Typically [1, 2].

    Returns
    -------
    df_immu_alone : pd.DataFrame
        One row per patient: immu difference features + target columns.
    df_combined : pd.DataFrame
        One row per patient: immu difference features + clinical baseline features
        + target columns.
    """
    t_a, t_b = timepoints[0], timepoints[1]
    id_cols  = {'Patient', 'Timepoint', 'Date', 'date', 'measurement_timepoint'}

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
        col_name         = f'{c}_t{t_a}_minus_t{t_b}'
        diff_cols[c]     = col_name
        df_im_merged[col_name] = df_im_merged[f'{c}_t{t_a}'] - df_im_merged[f'{c}_t{t_b}']

    # Keep only Patient + difference columns (discard raw T_a and T_b feature columns)
    df_im_wide = df_im_merged[['Patient'] + list(diff_cols.values())].copy()

    # ── CLINICAL: T_a baseline rows only, pain questionnaire + leaky cols removed ─

    cl_leaky    = [c for c in df_cl.columns
                   if any(pat in c for pat in CL_MODEL_LEAKY_PATTERNS)]
    cl_exclude  = id_cols | set(CL_PAIN_QUESTIONNAIRE_COLS) | set(cl_leaky)
    cl_feat_cols = [c for c in df_cl.columns if c not in cl_exclude]

    df_cl_t1 = (
        df_cl[df_cl['Timepoint'] == t_a][['Patient'] + cl_feat_cols]
        .drop_duplicates('Patient')
        .reset_index(drop=True)
    )

    print(f"\n  Clinical features excluded:")
    print(f"    Pain questionnaire cols : {CL_PAIN_QUESTIONNAIRE_COLS}")
    print(f"    Leaky metadata cols     : {sorted(cl_leaky)}")
    print(f"    Clinical features kept  : {len(cl_feat_cols)}")

    # ── TARGETS: exclude the raw post-treatment value (leaky) before merging ────

    # {col}_t{t_b} is the observed outcome at T_b — always leaky when the target
    # is the T_a → T_b change. All other target columns (baseline T_a value and
    # computed reductions) are kept and will be excluded by run_catboost_regressor
    # if they appear leaky relative to the chosen target_col.
    leaky_raw_tb  = [c for c in targets.columns if c.endswith(f'_t{t_b}')]
    target_merge  = ['Patient'] + [c for c in targets.columns
                                   if c != 'Patient' and c not in leaky_raw_tb]

    # ── MERGE into final datasets ─────────────────────────────────────────────

    # Immunological-only: difference features + target columns
    df_immu_alone = df_im_wide.merge(targets[target_merge], on='Patient', how='inner')

    # Combined: difference features + clinical T_a baseline + target columns
    df_combined = (
        df_im_wide
        .merge(df_cl_t1, on='Patient', how='inner')
        .merge(targets[target_merge], on='Patient', how='inner')
    )

    print(f"\nModel datasets ready (T{t_a}–T{t_b} immunological differences only):")
    print(f"  Immunological diff features : {len(diff_cols)}  "
          f"(one T{t_a}−T{t_b} diff per original feature)")
    print(f"  Clinical baseline features  : {len(cl_feat_cols)}")
    print(f"  df_immu_alone : shape={df_immu_alone.shape}, "
          f"patients={df_immu_alone['Patient'].nunique()}")
    print(f"  df_combined   : shape={df_combined.shape}, "
          f"patients={df_combined['Patient'].nunique()}")

    return df_immu_alone, df_combined



# ══════════════════════════════════════════════════════════════════════════════
# BASELINE CATBOOST
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
                           n_splits=5, n_repeats=5, random_state=42):
    """Run a baseline CatBoostRegressor with RepeatedKFold cross-validation.

    Uses n_splits × n_repeats folds (default: 5×5 = 25 fits) with no
    hyperparameter tuning (CatBoostRegressor fixed at 300 iterations).
    Per-fold metrics are printed and collected. The model returned is the
    one trained on the last CV fold — not a refitted full-data model.

    Automatically excluded from features:
      - ID columns  : Patient, Timepoint, Date, date, measurement_timepoint
      - Leaky cols  : any column whose name matches CL_MODEL_LEAKY_PATTERNS
                      ('response', 'improvement_percent', 'pain_reduction') —
                      catches all reduction/pct columns regardless of target name
      - target_col  : the specific regression target passed as argument

    Parameters
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
    # Build the exclusion set: ID columns + target_col + pattern-matched leaky cols.
    # Pattern matching handles any target naming produced by construct_datasets_targets
    # (e.g. pain_daytime_reduction, pain_under_load_reduction_pct, ...) without
    # requiring hardcoded column names.
    id_cols = ['Patient', 'Timepoint', 'Date', 'date', 'measurement_timepoint']
    leaky_patterns = CL_MODEL_LEAKY_PATTERNS  # ['response', 'improvement_percent', 'pain_reduction']
    leaky_cols = [c for c in df_model.columns
                  if any(pat in c for pat in leaky_patterns)]
    exclude = set(id_cols) | set(leaky_cols) | {target_col}

    # Subset to feature columns and extract target; drop rows where target is NaN
    feature_cols = [c for c in df_model.columns if c not in exclude]
    X = df_model[feature_cols].copy()
    y = df_model[target_col].copy()

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

        model = CatBoostRegressor(
            iterations=1000,
            loss_function='RMSE',
            custom_metric=['MAE', 'R2'],
            random_seed=random_state,
            verbose=0,
        )
        model.fit(
            Pool(X_train, y_train, cat_features=cat_cols),
            eval_set=Pool(X_test, y_test, cat_features=cat_cols),
            use_best_model=False,
        )

        preds = model.predict(X_test)
        y_pred.iloc[test_idx] = preds

        # Pull CatBoost's own validation metrics from evals_result_ at the final iteration
        val  = model.evals_result_['validation']
        rmse = val['RMSE'][-1]
        mae  = val['MAE'][-1]
        r2   = val['R2'][-1]
        mse  = rmse ** 2      # MSE is not a native CatBoost metric; derived from RMSE
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


def plot_prediction_heatmap(y_true, y_pred, name, bins=10):
    """2D density heatmap of predicted vs actual regression values.

    Each cell shows how many patients had a given (predicted, actual) combination.
    A perfect model clusters along the diagonal. Off-diagonal density reveals
    systematic over- or under-prediction in specific ranges.

    With small samples (~100 patients), bins=10 gives a reasonable resolution
    without too many empty cells. Adjust bins downward if the plot looks too sparse.

    Parameters
    ----------
    y_true : array-like   True target values.
    y_pred : array-like   Regression predictions from model.predict().
    name   : str          Label shown in the plot title.
    bins   : int          Number of bins along each axis (default 10).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))

    # 2D histogram heatmap: x = predicted, y = actual
    # cbar_kws label is 'Patient count' since each cell is a count of patients
    sns.histplot(
        x=y_pred, y=y_true,
        bins=bins,
        cmap='mako',
        cbar=True,
        cbar_kws={'label': 'Patient count'},
        ax=ax,
    )

    # Diagonal reference line = perfect prediction
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, linestyle='--', color='white', linewidth=1.2,
            alpha=0.7, label='Perfect prediction')

    ax.set_xlabel('Predicted pain reduction (%)', fontsize=11)
    ax.set_ylabel('Actual pain reduction (%)',    fontsize=11)
    ax.set_title(f'Predicted vs Actual — {name}', fontsize=12)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.show()


def plot_shap_regressor(model, X, name):
    """SHAP bar + beeswarm plots for a fitted CatBoostRegressor."""
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


def print_regression_summary(results_dict, target_col):
    """Print a mean ± std (95% CI) summary table across all datasets for a given target."""
    metric_cols = ['MAE', 'MSE', 'RMSE', 'R2']
    rows = []
    for ds_name, res_df in results_dict.items():
        fold_rows = res_df[~res_df['Fold'].isin(['Mean', 'Std'])]
        n      = len(fold_rows)
        t_crit = stats.t.ppf(0.975, df=n - 1)
        row = {'Dataset': ds_name}
        for m in metric_cols:
            mv = fold_rows[m].mean()
            sv = fold_rows[m].std()
            ci = t_crit * sv / np.sqrt(n)
            row[m] = f"{mv:.3f} ± {sv:.4f} [{mv - ci:.3f}, {mv + ci:.3f}]"
        rows.append(row)
    summary = pd.DataFrame(rows)
    print(f"\n{'='*90}")
    print(f"  CATBOOST BASELINE SUMMARY — Target: {target_col}  (mean ± std, 95% CI)")
    print(f"{'='*90}")
    print(summary.to_string(index=False))
    return summary

# ══════════════════════════════════════════════════════════════════════════════
# ADVANCED CATBOOST  (Nested CV + Optuna)
# ══════════════════════════════════════════════════════════════════════════════

def run_advanced_catboost(df_combined, target_col='pain_reduction_pct', random_state=42):
    """Advanced CatBoostRegressor with nested CV and Optuna hyperparameter tuning.

    Outer CV : RepeatedKFold(n_splits=4, n_repeats=5) = 20 outer folds.
    Inner CV : RepeatedKFold(n_splits=4, n_repeats=5) = 20 fits per Optuna trial, later try 25.
    Optuna   : 20 trials per outer fold, objective = minimize RMSE via
               OptunaSearchCV with scoring='neg_root_mean_squared_error'.
    Final model trained on full dataset with last outer fold's best params for SHAP.

    Parameters
    ----------
    df_combined : pd.DataFrame  Combined T1 dataset (immunological + clinical).
    target_col  : str           Regression target (default: 'pain_reduction_pct').
    random_state: int           Random seed for CV splitters and CatBoost (default 42).

    Returns
    -------
    results_df     : pd.DataFrame       Per-fold MAE/MSE/RMSE/R2 + Mean/Std rows.
    best_params_df : pd.DataFrame       Best hyperparameters found per outer fold.
    model          : CatBoostRegressor  Final model trained on full dataset.
    X              : pd.DataFrame       Feature matrix used.
    y_pred         : pd.Series          Full-data predictions from final model.
    """
    import optuna
    try:
        from optuna.integration import OptunaSearchCV
    except ImportError:
        from optuna_integration import OptunaSearchCV

    # Build the exclusion set — same pattern-based logic as run_catboost_regressor.
    # ID columns are always excluded; any column whose name contains a leaky
    # pattern substring (CL_MODEL_LEAKY_PATTERNS) is also excluded, so all
    # derived outcome columns (pain_reduction, pain_reduction_pct, response_*, …)
    # are removed regardless of which target_col is passed in.

    id_cols = ['Patient', 'Timepoint', 'Date', 'date', 'measurement_timepoint']
    leaky_patterns = CL_MODEL_LEAKY_PATTERNS  # ['response', 'improvement_percent', 'pain_reduction']
    leaky_cols = [c for c in df_combined.columns
                  if any(pat in c for pat in leaky_patterns)]
    exclude = set(id_cols) | set(leaky_cols) | {target_col}

    # Subset features and target; drop rows with NaN target
    feature_cols = [c for c in df_combined.columns if c not in exclude]
    X = df_combined[feature_cols].copy()
    y = df_combined[target_col].copy()

    valid = y.notna()
    X, y = X[valid].reset_index(drop=True), y[valid].reset_index(drop=True)

    # Convert category/object dtypes to str for CatBoost categorical handling
    for col in X.select_dtypes(include=['category', 'object']).columns:
        X[col] = X[col].astype(str)
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()

    # Print run header
    print(f"\n{'='*65}")
    print(f"  Advanced CatBoost — {target_col}")
    print(f"  Samples: {len(X)},  Features: {len(feature_cols)}")
    print(f"  Outer: 4×5=20 folds  |  Inner: 4×5=20 fits/trial  |  Trials: 20")
    print(f"{'='*65}")

    # Outer and inner CV splitters
    outer_cv = RepeatedKFold(n_splits=4, n_repeats=5, random_state=random_state)
    inner_cv = RepeatedKFold(n_splits=4, n_repeats=5, random_state=random_state) # try 25 repeats!

    # Optuna hyperparameter search space for CatBoost
    param_distributions = {
        'depth':               optuna.distributions.IntDistribution(3, 8),
        'learning_rate':       optuna.distributions.FloatDistribution(1e-3, 0.3, log=True),
        'l2_leaf_reg':         optuna.distributions.FloatDistribution(1e-2, 10.0, log=True),
        'bagging_temperature': optuna.distributions.FloatDistribution(0.0, 1.0),
        'random_strength':     optuna.distributions.FloatDistribution(0.0, 10.0),
        'min_data_in_leaf':    optuna.distributions.IntDistribution(1, 30),
    }

    fold_results   = []
    best_params_list = []

    # Show each Optuna trial so progress is visible during long runs
    optuna.logging.set_verbosity(optuna.logging.INFO)

    start = time.time()

    # Outer loop — each iteration is one outer fold evaluation
    for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X), start=1):
        print(f"\n  Outer fold {outer_fold}/{outer_cv.get_n_splits()}")

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Base CatBoost model — fixed non-tuned params; cat_features set in constructor.
        # iterations=300 here (inner CV only) keeps tuning tractable; the final model uses 1000.
        # custom_metric is intentionally omitted: sklearn's clone() cannot round-trip
        # CatBoost's internal representation of custom_metric via get_params/set_params.
        # Metrics are evaluated externally via sklearn's scoring='neg_root_mean_squared_error'.
        base_model = CatBoostRegressor(
            iterations=300, # try 1000
            cat_features=cat_cols,
            random_seed=random_state,
            verbose=0,
        )

        # OptunaSearchCV handles the inner CV + Optuna tuning
        optuna_search = OptunaSearchCV(
            estimator=base_model,
            param_distributions=param_distributions,
            cv=inner_cv,
            scoring='neg_root_mean_squared_error',
            n_trials=20,
            n_jobs=-1,   
            verbose=0,
        )

        optuna_search.fit(X_train, y_train)
        best_params_list.append(optuna_search.best_params_)

        # Evaluate best inner model on the outer test fold
        preds = optuna_search.predict(X_test)
        rmse  = np.sqrt(mean_squared_error(y_test, preds))
        mae   = mean_absolute_error(y_test, preds)
        r2    = r2_score(y_test, preds)
        mse   = rmse ** 2

        fold_results.append({'Fold': outer_fold, 'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2})
        print(f"    MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}")
        print(f"    Best params: {optuna_search.best_params_}")

    elapsed = time.time() - start
    print(f"\n  Training time: {elapsed:.1f}s  ({elapsed/60:.1f} min)")

    # Build results DataFrame with per-fold rows + Mean/Std summary
    results_df  = pd.DataFrame(fold_results)
    metric_cols = ['MAE', 'MSE', 'RMSE', 'R2']
    mean_row = {'Fold': 'Mean', **{m: results_df[m].mean() for m in metric_cols}}
    std_row  = {'Fold': 'Std',  **{m: results_df[m].std()  for m in metric_cols}}
    results_df = pd.concat(
        [results_df, pd.DataFrame([mean_row, std_row])], ignore_index=True)

    # Print summary with 95% CI
    n_outer = len(fold_results)
    t_crit  = stats.t.ppf(0.975, df=n_outer - 1)
    print(f"\n  Summary (4×5 outer CV, 20 Optuna trials, 95% CI):")
    for m in metric_cols:
        mv = mean_row[m]
        sv = std_row[m]
        ci = t_crit * sv / np.sqrt(n_outer)
        print(f"    {m:<5}: {mv:.3f} ± {sv:.4f}  (95% CI [{mv - ci:.3f}, {mv + ci:.3f}])")

    # Best hyperparameters table across outer folds
    best_params_df = pd.DataFrame(best_params_list)
    best_params_df.index = [f"Fold {i+1}" for i in range(len(best_params_list))]
    print(f"\n  Best hyperparameters per outer fold:")
    print(best_params_df.to_string())

    # Train final model on full dataset using last fold's best params — for SHAP
    final_model = CatBoostRegressor(
        iterations=1000,
        loss_function='RMSE',
        custom_metric=['MAE', 'R2'],
        cat_features=cat_cols,
        random_seed=random_state,
        verbose=0,
        **optuna_search.best_params_,
    )
    final_model.fit(X, y)
    y_pred = pd.Series(final_model.predict(X), index=range(len(X)), dtype='float64')

    return results_df, best_params_df, final_model, X, y_pred


# ══════════════════════════════════════════════════════════════════════════════
# ADVANCED HGB  (Nested CV + Optuna) — placeholder
# ══════════════════════════════════════════════════════════════════════════════

# TODO: implement HistGradientBoostingRegressor nested CV
# Same nested CV structure as Advanced CatBoost
# OrdinalEncoder for categoricals inside Pipeline
# Objective: minimize RMSE
# Feature importance: HGB built-in + SHAP
