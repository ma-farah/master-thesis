# Modeling functions — baseline and advanced regressors
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import HistGradientBoostingRegressor
from catboost import CatBoostRegressor, Pool
import shap


# ── Constants ─────────────────────────────────────────────────────────────────

# Substrings that flag a column as a leaky outcome variable.
# Used in run_catboost_regressor and run_advanced_catboost to exclude derived
# outcome columns (pain_reduction, pain_reduction_pct, response_*, …) that enter
# the dataset through the targets merge.  Clinical pain questionnaire cols are
# filtered upstream in results.py Step 7 (df_cl_mod) before reaching these functions.
CL_MODEL_LEAKY_PATTERNS = ['response', 'improvement_percent', '_reduction']


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

    # ── CLINICAL: T_a baseline rows only ─────────────────────────────────────────
    # df_cl (df_cl_mod) is pre-filtered: pain questionnaire cols and leaky metadata
    # cols were already removed in results.py Step 7. Only exclude ID cols here.

    cl_feat_cols = [c for c in df_cl.columns if c not in id_cols]

    df_cl_t1 = (
        df_cl[df_cl['Timepoint'] == t_a][['Patient'] + cl_feat_cols]
        .drop_duplicates('Patient')
        .reset_index(drop=True)
    )

    print(f"\n  Clinical features: {len(cl_feat_cols)} (pain/leaky cols pre-filtered upstream)")

    # ── TARGETS: include baseline (_t{t_a}), exclude leaky post-treatment (_t{t_b}) ──

    # The baseline value (_t{t_a}) is a legitimate predictor — it captures the
    # patient's starting severity for the target being modelled. It is included as
    # a feature. The post-treatment value (_t{t_b}) is always leaky and excluded.
    # Reduction columns (_reduction, _reduction_pct) are merged in but are then
    # excluded from the feature matrix by CL_MODEL_LEAKY_PATTERNS inside each
    # model function; only the specific target_col is retained as the response.
    #
    # Because each call to create_model_datasets is tied to one targets DataFrame
    # (pain_scale targets OR pain_under_load targets), each dataset carries only
    # its own baseline — there is no cross-contamination between targets.
    leaky_tp_cols = [c for c in targets.columns if c.endswith(f'_t{t_b}')]
    target_merge  = ['Patient'] + [c for c in targets.columns
                                   if c != 'Patient' and c not in leaky_tp_cols]

    # ── MERGE into final datasets ─────────────────────────────────────────────

    # Immunological-only: difference features + target columns
    df_immu_alone = df_im_wide.merge(targets[target_merge], on='Patient', how='inner')

    # Combined: difference features + clinical T_a baseline + target columns
    df_combined = (
        df_im_wide
        .merge(df_cl_t1, on='Patient', how='inner')
        .merge(targets[target_merge], on='Patient', how='inner')
    )

    baseline_cols = [c for c in target_merge if c.endswith(f'_t{t_a}')]
    print(f"\nModel datasets ready (T{t_a}–T{t_b} immunological differences only):")
    print(f"  Immunological diff features : {len(diff_cols)}  "
          f"(one T{t_a}−T{t_b} diff per original feature)")
    print(f"  Clinical baseline features  : {len(cl_feat_cols)}")
    print(f"  Target baseline included    : {baseline_cols}")
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
            random_seed=random_state,
            verbose=0,
        )
        model.fit(
            Pool(X_train, y_train, cat_features=cat_cols),
        )

        preds = model.predict(X_test)
        y_pred.iloc[test_idx] = preds

        # Compute metrics from predictions (consistent with advanced model)
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
    ax.plot(lims, lims, linestyle='--', color='red', linewidth=1.2,
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
    """Print a mean ± std summary table across all datasets for a given target."""
    metric_cols = ['MAE', 'MSE', 'RMSE', 'R2']
    rows = []
    for ds_name, res_df in results_dict.items():
        fold_rows = res_df[~res_df['Fold'].isin(['Mean', 'Std'])]
        row = {'Dataset': ds_name}
        for m in metric_cols:
            mv = fold_rows[m].mean()
            sv = fold_rows[m].std()
            row[m] = f"{mv:.3f} ± {sv:.4f}"
        rows.append(row)
    summary = pd.DataFrame(rows)
    print(f"\n{'='*75}")
    print(f"  CATBOOST BASELINE SUMMARY — Target: {target_col}  (mean ± std)")
    print(f"{'='*75}")
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
        'depth':               optuna.distributions.IntDistribution(3, 10),
        'learning_rate':       optuna.distributions.FloatDistribution(1e-3, 0.3, log=True),
        'l2_leaf_reg':         optuna.distributions.FloatDistribution(1, 10.0, log=True),
        'bagging_temperature': optuna.distributions.FloatDistribution(0.0, 1.0),
        'random_strength':     optuna.distributions.FloatDistribution(0.0, 10.0)
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
# ADVANCED CATBOOST + RENT FEATURE SELECTION  (Nested CV + Optuna)
# ══════════════════════════════════════════════════════════════════════════════

def run_advanced_catboost_rent(
    df_combined, target_col='pain_reduction_pct', random_state=42,
    tau_1=0.7, tau_2=0.75, tau_3=0.95,
):
    """CatBoostRegressor with RENT feature selection + nested CV + Optuna.

    RENT (Repeated Elastic Net Technique) is applied inside each outer fold on
    the training split only — no test leakage.  It trains K=100 ElasticNet
    models on random sub-samples and selects features that are:
      τ₁: selected in ≥tau_1 fraction of the K models
      τ₂: sign-consistent in ≥tau_2 fraction of the K models
      τ₃: t-test p-value ≥tau_3 (coefficient distribution significantly ≠ 0)

    Because RENT uses ElasticNet internally (requires numeric, no NaN), a
    preprocessing step (OrdinalEncoder + SimpleImputer) is applied to X_train
    before passing to RENT.  The selected feature names are then used to subset
    the original X_train/X_test (CatBoost still handles NaN and categoricals
    natively).

    Outer CV : RepeatedKFold(n_splits=4, n_repeats=5) = 20 outer folds.
    Inner CV : RepeatedKFold(n_splits=4, n_repeats=5) = 20 fits per trial.
    Optuna   : 20 trials per outer fold, scoring = neg_root_mean_squared_error.

    Parameters
    ----------
    df_combined : pd.DataFrame  Combined T1 dataset (immunological + clinical).
    target_col  : str           Regression target (default: 'pain_reduction_pct').
    random_state: int           Random seed (default 42).
    tau_1       : float         RENT τ₁ cutoff — selection frequency (default 0.7).
    tau_2       : float         RENT τ₂ cutoff — sign consistency (default 0.75).
    tau_3       : float         RENT τ₃ cutoff — t-test threshold (default 0.95).

    Returns
    -------
    results_df              : pd.DataFrame       Per-fold metrics + Mean/Std rows.
    best_params_df          : pd.DataFrame       Best hyperparameters per outer fold.
    final_model             : CatBoostRegressor  Final model on full dataset.
    X_final                 : pd.DataFrame       Feature matrix (RENT-selected features).
    y_pred                  : pd.Series          Full-data predictions from final model.
    selected_features_per_fold : list[list[str]] Selected feature names per outer fold.
    """
    import optuna
    try:
        from optuna.integration import OptunaSearchCV
    except ImportError:
        from optuna_integration import OptunaSearchCV
    import warnings
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OrdinalEncoder
    from RENT import RENT

    warnings.filterwarnings('ignore', category=FutureWarning, module='RENT')
    warnings.filterwarnings('ignore', category=RuntimeWarning, module='RENT')
    warnings.filterwarnings('ignore', message='OptunaSearchCV is experimental')

    # Same exclusion logic as run_advanced_catboost
    id_cols = ['Patient', 'Timepoint', 'Date', 'date', 'measurement_timepoint']
    leaky_cols = [c for c in df_combined.columns
                  if any(pat in c for pat in CL_MODEL_LEAKY_PATTERNS)]
    exclude = set(id_cols) | set(leaky_cols) | {target_col}

    feature_cols = [c for c in df_combined.columns if c not in exclude]
    X = df_combined[feature_cols].copy()
    y = df_combined[target_col].copy()

    valid = y.notna()
    X, y = X[valid].reset_index(drop=True), y[valid].reset_index(drop=True)

    # Convert category/object dtypes to str for CatBoost categorical handling
    for col in X.select_dtypes(include=['category', 'object']).columns:
        X[col] = X[col].astype(str)
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()

    print(f"\n{'='*65}")
    print(f"  CatBoost + RENT — {target_col}")
    print(f"  Samples: {len(X)},  Features: {len(feature_cols)}")
    print(f"  RENT: K=100, τ₁={tau_1}, τ₂={tau_2}, τ₃={tau_3}")
    print(f"  Outer: 4×5=20 folds  |  Inner: 4×5=20 fits/trial  |  Trials: 20")
    print(f"{'='*65}")

    outer_cv = RepeatedKFold(n_splits=4, n_repeats=5, random_state=random_state)
    inner_cv = RepeatedKFold(n_splits=4, n_repeats=5, random_state=random_state)

    param_distributions = {
        'depth':               optuna.distributions.IntDistribution(3, 10),
        'learning_rate':       optuna.distributions.FloatDistribution(1e-3, 0.3, log=True),
        'l2_leaf_reg':         optuna.distributions.FloatDistribution(1, 10.0, log=True),
        'bagging_temperature': optuna.distributions.FloatDistribution(0.0, 1.0),
        'random_strength':     optuna.distributions.FloatDistribution(0.0, 10.0),
    }

    fold_results           = []
    best_params_list       = []
    selected_features_per_fold = []
    optuna.logging.set_verbosity(optuna.logging.INFO)
    start = time.time()

    for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X), start=1):
        print(f"\n  Outer fold {outer_fold}/{outer_cv.get_n_splits()}")

        X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # ── RENT feature selection on training split only ─────────────────────
        # RENT uses ElasticNet internally → needs numeric, no NaN.
        # Step 1: OrdinalEncode categorical (str) columns
        X_train_enc = X_train.copy()
        cat_mask_cols = [c for c in X_train.columns if X_train[c].dtype == object]
        if cat_mask_cols:
            oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            X_train_enc[cat_mask_cols] = oe.fit_transform(X_train[cat_mask_cols])

        # Step 2: Impute NaN with column median
        imputer = SimpleImputer(strategy='median')
        X_train_rent_arr = imputer.fit_transform(X_train_enc.astype(float))
        X_train_rent = pd.DataFrame(X_train_rent_arr, columns=feature_cols)

        # Step 3: Run RENT
        rent_model = RENT.RENT_Regression(
            data=X_train_rent,
            target=y_train.values,
            feat_names=feature_cols,
            C=[0.1, 1, 10],
            l1_ratios=[0.1, 0.5, 0.9],
            autoEnetParSel=True,
            poly='OFF',
            testsize_range=(0.25, 0.25),
            K=100,
            random_state=random_state,
            verbose=0,
        )
        rent_model.train()
        selected_idx = rent_model.select_features(
            tau_1_cutoff=tau_1, tau_2_cutoff=tau_2, tau_3_cutoff=tau_3)

        if len(selected_idx) == 0:
            print(f"    RENT: 0 features selected — using all {len(feature_cols)}")
            selected_cols = feature_cols
        else:
            selected_cols = [feature_cols[i] for i in selected_idx]
            preview = selected_cols[:8]
            suffix  = '...' if len(selected_cols) > 8 else ''
            print(f"    RENT: {len(selected_cols)}/{len(feature_cols)} features — {preview}{suffix}")

        selected_features_per_fold.append(selected_cols)

        # ── Inner CV + Optuna on RENT-selected features ───────────────────────
        X_train_sel = X_train[selected_cols]
        X_test_sel  = X_test[selected_cols]

        # CatBoost categorical features restricted to selected columns
        cat_cols_sel = [c for c in cat_cols if c in selected_cols]

        base_model = CatBoostRegressor(
            iterations=300,
            cat_features=cat_cols_sel,
            random_seed=random_state,
            verbose=0,
        )

        optuna_search = OptunaSearchCV(
            estimator=base_model,
            param_distributions=param_distributions,
            cv=inner_cv,
            scoring='neg_root_mean_squared_error',
            n_trials=20,
            n_jobs=-1,
            verbose=0,
        )

        optuna_search.fit(X_train_sel, y_train)
        best_params_list.append(optuna_search.best_params_)

        preds = optuna_search.predict(X_test_sel)
        rmse  = np.sqrt(mean_squared_error(y_test, preds))
        mae   = mean_absolute_error(y_test, preds)
        r2    = r2_score(y_test, preds)
        mse   = rmse ** 2

        fold_results.append({'Fold': outer_fold, 'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2})
        print(f"    MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}")
        print(f"    Best params: {optuna_search.best_params_}")

    elapsed = time.time() - start
    print(f"\n  Training time: {elapsed:.1f}s  ({elapsed/60:.1f} min)")

    # Results DataFrame + summary
    results_df  = pd.DataFrame(fold_results)
    metric_cols = ['MAE', 'MSE', 'RMSE', 'R2']
    mean_row = {'Fold': 'Mean', **{m: results_df[m].mean() for m in metric_cols}}
    std_row  = {'Fold': 'Std',  **{m: results_df[m].std()  for m in metric_cols}}
    results_df = pd.concat(
        [results_df, pd.DataFrame([mean_row, std_row])], ignore_index=True)

    n_outer = len(fold_results)
    t_crit  = stats.t.ppf(0.975, df=n_outer - 1)
    print(f"\n  Summary (4×5 outer CV + RENT, 20 Optuna trials, 95% CI):")
    for m in metric_cols:
        mv = mean_row[m]; sv = std_row[m]
        ci = t_crit * sv / np.sqrt(n_outer)
        print(f"    {m:<5}: {mv:.3f} ± {sv:.4f}  (95% CI [{mv - ci:.3f}, {mv + ci:.3f}])")

    best_params_df = pd.DataFrame(best_params_list)
    best_params_df.index = [f"Fold {i+1}" for i in range(len(best_params_list))]
    print(f"\n  Best hyperparameters per outer fold:")
    print(best_params_df.to_string())

    # Print feature selection summary across folds
    from collections import Counter
    all_selected = [f for fold_feats in selected_features_per_fold for f in fold_feats]
    freq = Counter(all_selected)
    print(f"\n  RENT feature selection frequency (top 20 across {n_outer} folds):")
    for feat, cnt in freq.most_common(20):
        print(f"    {cnt:>3}/{n_outer}  {feat}")

    # Final model on full dataset using RENT on full X, last fold's best params
    X_enc_full = X.copy()
    cat_mask_cols_full = [c for c in X.columns if X[c].dtype == object]
    if cat_mask_cols_full:
        oe_full = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_enc_full[cat_mask_cols_full] = oe_full.fit_transform(X[cat_mask_cols_full])
    imputer_full = SimpleImputer(strategy='median')
    X_rent_full = pd.DataFrame(
        imputer_full.fit_transform(X_enc_full.astype(float)), columns=feature_cols)

    rent_final = RENT.RENT_Regression(
        data=X_rent_full,
        target=y.values,
        feat_names=feature_cols,
        C=[0.1, 1, 10],
        l1_ratios=[0.1, 0.5, 0.9],
        autoEnetParSel=True,
        poly='OFF',
        testsize_range=(0.25, 0.25),
        K=100,
        random_state=random_state,
        verbose=0,
    )
    rent_final.train()
    final_idx  = rent_final.select_features(
        tau_1_cutoff=tau_1, tau_2_cutoff=tau_2, tau_3_cutoff=tau_3)
    final_cols = [feature_cols[i] for i in final_idx] if len(final_idx) > 0 else feature_cols
    print(f"\n  Final model RENT selected {len(final_cols)}/{len(feature_cols)} features.")

    X_final = X[final_cols]
    cat_cols_final = [c for c in cat_cols if c in final_cols]

    final_model = CatBoostRegressor(
        iterations=1000,
        loss_function='RMSE',
        custom_metric=['MAE', 'R2'],
        cat_features=cat_cols_final,
        random_seed=random_state,
        verbose=0,
        **optuna_search.best_params_,
    )
    final_model.fit(X_final, y)
    y_pred = pd.Series(final_model.predict(X_final), index=range(len(X_final)), dtype='float64')

    return results_df, best_params_df, final_model, X_final, y_pred, selected_features_per_fold


# ══════════════════════════════════════════════════════════════════════════════
# ADVANCED ELASTICNET + RENT  (Nested CV + Optuna)
# ══════════════════════════════════════════════════════════════════════════════

def run_advanced_elasticnet_rent(
    df_combined, target_col='pain_reduction_pct', random_state=42,
    tau_1=0.7, tau_2=0.75, tau_3=0.95,
):
    """ElasticNet with RENT feature selection + nested CV + Optuna.

      1. OrdinalEncoder — applied once before the outer CV loop to convert any
         str/category columns to integers (fixed mapping, no statistical fit).
      2. SimpleImputer (median) — fitted on X_train inside each outer fold;
         also re-fitted on each inner split via the sklearn Pipeline.
      3. StandardScaler — same: fitted on X_train per outer fold for RENT;
         re-fitted per inner split via the Pipeline.

    RENT runs on the fully preprocessed (imputed + scaled) X_train of each
    outer fold.  The Optuna inner CV uses a Pipeline
    (SimpleImputer → StandardScaler → ElasticNet) so preprocessing is fit
    only on inner-train splits.

    Outer CV : RepeatedKFold(n_splits=4, n_repeats=5) = 20 outer folds.
    Inner CV : RepeatedKFold(n_splits=4, n_repeats=5) = 20 fits per trial.
    Optuna   : 20 trials per outer fold, scoring = neg_root_mean_squared_error.
    Tuned    : alpha (1e-4 – 100, log), l1_ratio (0.01 – 1.0).

    Parameters
    ----------
    df_combined : pd.DataFrame  Combined T1 dataset (immunological + clinical).
    target_col  : str           Regression target (default: 'pain_reduction_pct').
    random_state: int           Random seed (default 42).
    tau_1       : float         RENT τ₁ cutoff — selection frequency (default 0.7).
    tau_2       : float         RENT τ₂ cutoff — sign consistency (default 0.75).
    tau_3       : float         RENT τ₃ cutoff — t-test threshold (default 0.95).

    Returns
    -------
    results_df              : pd.DataFrame  Per-fold metrics + Mean/Std rows.
    best_params_df          : pd.DataFrame  Best hyperparameters per outer fold.
    final_pipeline          : Pipeline      Fitted (Imputer→Scaler→ElasticNet) on full data.
    X_final                 : pd.DataFrame  OrdinalEncoded feature matrix (RENT-selected cols).
    y_pred                  : pd.Series     Full-data predictions from final_pipeline.
    selected_features_per_fold : list[list[str]]  Selected feature names per outer fold.
    """
    import optuna
    try:
        from optuna.integration import OptunaSearchCV
    except ImportError:
        from optuna_integration import OptunaSearchCV
    import warnings
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OrdinalEncoder, StandardScaler
    from sklearn.linear_model import ElasticNet
    from sklearn.pipeline import Pipeline
    from sklearn.exceptions import ConvergenceWarning
    from RENT import RENT

    # Suppress known harmless warnings:
    # - FutureWarning: RENT uses deprecated .applymap() (old pandas API, no effect on results)
    # - RuntimeWarning: RENT divide-by-zero in τ₃ when feature std=0 (handled as NaN internally)
    # - ExperimentalWarning: OptunaSearchCV is marked experimental but works correctly
    warnings.filterwarnings('ignore', category=FutureWarning, module='RENT')
    warnings.filterwarnings('ignore', category=RuntimeWarning, module='RENT')
    warnings.filterwarnings('ignore', message='OptunaSearchCV is experimental')

    # Same exclusion logic as other advanced functions
    id_cols = ['Patient', 'Timepoint', 'Date', 'date', 'measurement_timepoint']
    leaky_cols = [c for c in df_combined.columns
                  if any(pat in c for pat in CL_MODEL_LEAKY_PATTERNS)]
    exclude = set(id_cols) | set(leaky_cols) | {target_col}

    feature_cols = [c for c in df_combined.columns if c not in exclude]
    X_raw = df_combined[feature_cols].copy()
    y = df_combined[target_col].copy()

    valid = y.notna()
    X_raw, y = X_raw[valid].reset_index(drop=True), y[valid].reset_index(drop=True)

    # ── OrdinalEncode str/category columns once upfront ───────────────────────
    # This is a fixed integer mapping (gender, diagnosis, …) — not a statistical
    # fit, so doing it before the outer CV loop introduces negligible leakage.
    X_oe = X_raw.copy()
    cat_cols_list = [c for c in X_oe.columns
                     if X_oe[c].dtype == object or str(X_oe[c].dtype) == 'category']
    if cat_cols_list:
        oe_global = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_oe[cat_cols_list] = oe_global.fit_transform(X_oe[cat_cols_list].astype(str))
    # X_oe is now fully numeric (may still contain NaN)

    print(f"\n{'='*65}")
    print(f"  ElasticNet + RENT — {target_col}")
    print(f"  Samples: {len(X_oe)},  Features: {len(feature_cols)}")
    print(f"  RENT: K=100, τ₁={tau_1}, τ₂={tau_2}, τ₃={tau_3}")
    print(f"  Preprocessing: OrdinalEncode → Impute (median) → StandardScale")
    print(f"  Outer: 4×5=20 folds  |  Inner: 4×5=20 fits/trial  |  Trials: 20")
    print(f"{'='*65}")

    outer_cv = RepeatedKFold(n_splits=4, n_repeats=5, random_state=random_state)
    inner_cv = RepeatedKFold(n_splits=4, n_repeats=25, random_state=random_state)

    # Optuna search space: alpha (regularisation strength) and l1_ratio (L1/L2 mix).
    # Lower bound 1e-2 avoids near-zero alpha that causes severe overfitting on
    # small inner-train splits (~45 samples) and wastes Optuna trials.
    param_distributions = {
        'model__alpha':     optuna.distributions.FloatDistribution(1e-2, 100.0, log=True),
        'model__l1_ratio':  optuna.distributions.FloatDistribution(0.01, 1.0),
    }

    fold_results           = []
    best_params_list       = []
    selected_features_per_fold = []
    optuna.logging.set_verbosity(optuna.logging.INFO)
    start = time.time()

    for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X_oe), start=1):
        print(f"\n  Outer fold {outer_fold}/{outer_cv.get_n_splits()}")

        X_oe_train = X_oe.iloc[train_idx].copy()
        X_oe_test  = X_oe.iloc[test_idx].copy()
        y_train    = y.iloc[train_idx]
        y_test     = y.iloc[test_idx]

        # ── Preprocess X_train for RENT (fit on train only) ───────────────────
        imputer_rent = SimpleImputer(strategy='median').fit(X_oe_train)
        X_imp_train  = imputer_rent.transform(X_oe_train)
        scaler_rent  = StandardScaler().fit(X_imp_train)
        X_rent_train = pd.DataFrame(
            scaler_rent.transform(X_imp_train), columns=feature_cols)

        # ── RENT feature selection on preprocessed X_train ───────────────────
        rent_model = RENT.RENT_Regression(
            data=X_rent_train,
            target=y_train.values,
            feat_names=feature_cols,
            C=[0.1, 1, 10],
            l1_ratios=[0.1, 0.5, 0.9],
            autoEnetParSel=True,
            poly='OFF',
            testsize_range=(0.25, 0.25),
            K=100,
            random_state=random_state,
            verbose=0,
        )
        rent_model.train()
        selected_idx = rent_model.select_features(
            tau_1_cutoff=tau_1, tau_2_cutoff=tau_2, tau_3_cutoff=tau_3)

        if len(selected_idx) == 0:
            print(f"    RENT: 0 features selected — using all {len(feature_cols)}")
            selected_cols = feature_cols
        else:
            selected_cols = [feature_cols[i] for i in selected_idx]
            preview = selected_cols[:8]
            suffix  = '...' if len(selected_cols) > 8 else ''
            print(f"    RENT: {len(selected_cols)}/{len(feature_cols)} features — {preview}{suffix}")

        selected_features_per_fold.append(selected_cols)

        # ── Inner CV + Optuna with Pipeline (re-fits preprocessing per split) ─
        # Pipeline ensures Imputer and Scaler are fit only on inner-train data.
        inner_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler',  StandardScaler()),
            ('model',   ElasticNet(max_iter=100000, random_state=random_state)),
        ])

        optuna_search = OptunaSearchCV(
            estimator=inner_pipe,
            param_distributions=param_distributions,
            cv=inner_cv,
            scoring='neg_root_mean_squared_error',
            n_trials=20,
            n_jobs=-1,
            verbose=0,
        )

        import warnings
        from sklearn.exceptions import ConvergenceWarning
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ConvergenceWarning)
            optuna_search.fit(X_oe_train[selected_cols], y_train)
        best_params_list.append(optuna_search.best_params_)

        preds = optuna_search.predict(X_oe_test[selected_cols])
        rmse  = np.sqrt(mean_squared_error(y_test, preds))
        mae   = mean_absolute_error(y_test, preds)
        r2    = r2_score(y_test, preds)
        mse   = rmse ** 2

        fold_results.append({'Fold': outer_fold, 'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2})
        print(f"    MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}")
        print(f"    Best params: {optuna_search.best_params_}")

    elapsed = time.time() - start
    print(f"\n  Training time: {elapsed:.1f}s  ({elapsed/60:.1f} min)")

    # Results DataFrame + summary
    results_df  = pd.DataFrame(fold_results)
    metric_cols = ['MAE', 'MSE', 'RMSE', 'R2']
    mean_row = {'Fold': 'Mean', **{m: results_df[m].mean() for m in metric_cols}}
    std_row  = {'Fold': 'Std',  **{m: results_df[m].std()  for m in metric_cols}}
    results_df = pd.concat(
        [results_df, pd.DataFrame([mean_row, std_row])], ignore_index=True)

    n_outer = len(fold_results)
    t_crit  = stats.t.ppf(0.975, df=n_outer - 1)
    print(f"\n  Summary (4×5 outer CV + RENT, 20 Optuna trials, 95% CI):")
    for m in metric_cols:
        mv = mean_row[m]; sv = std_row[m]
        ci = t_crit * sv / np.sqrt(n_outer)
        print(f"    {m:<5}: {mv:.3f} ± {sv:.4f}  (95% CI [{mv - ci:.3f}, {mv + ci:.3f}])")

    best_params_df = pd.DataFrame(best_params_list)
    best_params_df.index = [f"Fold {i+1}" for i in range(len(best_params_list))]
    print(f"\n  Best hyperparameters per outer fold:")
    print(best_params_df.to_string())

    # Feature selection frequency across folds
    from collections import Counter
    all_selected = [f for fold_feats in selected_features_per_fold for f in fold_feats]
    freq = Counter(all_selected)
    print(f"\n  RENT feature selection frequency (top 20 across {n_outer} folds):")
    for feat, cnt in freq.most_common(20):
        print(f"    {cnt:>3}/{n_outer}  {feat}")

    # ── Final model on full dataset ───────────────────────────────────────────
    # RENT on full OrdinalEncoded X_oe
    imp_full = SimpleImputer(strategy='median').fit(X_oe)
    sca_full = StandardScaler().fit(imp_full.transform(X_oe))
    X_rent_full = pd.DataFrame(
        sca_full.transform(imp_full.transform(X_oe)), columns=feature_cols)

    rent_final = RENT.RENT_Regression(
        data=X_rent_full,
        target=y.values,
        feat_names=feature_cols,
        C=[0.1, 1, 10],
        l1_ratios=[0, 0.1, 0.25, 0.5, 0.75, 0.9, 1],
        autoEnetParSel=True,
        poly='OFF',
        testsize_range=(0.25, 0.25),
        K=100,
        random_state=random_state,
        verbose=0,
    )
    rent_final.train()
    final_idx  = rent_final.select_features(
        tau_1_cutoff=tau_1, tau_2_cutoff=tau_2, tau_3_cutoff=tau_3)
    final_cols = [feature_cols[i] for i in final_idx] if len(final_idx) > 0 else feature_cols
    print(f"\n  Final model RENT selected {len(final_cols)}/{len(feature_cols)} features.")

    X_final = X_oe[final_cols]

    final_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler()),
        ('model',   ElasticNet(
            max_iter=100000,
            random_state=random_state,
            alpha=optuna_search.best_params_['model__alpha'],
            l1_ratio=optuna_search.best_params_['model__l1_ratio'],
        )),
    ])
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        final_pipeline.fit(X_final, y)
    y_pred = pd.Series(final_pipeline.predict(X_final), index=range(len(X_final)), dtype='float64')

    return results_df, best_params_df, final_pipeline, X_final, y_pred, selected_features_per_fold


def plot_elasticnet_coefficients(pipeline, feature_names, name, top_n=20):
    """Bar chart of ElasticNet standardised coefficients (feature importance).

    For a linear model the standardised coefficients are equivalent to SHAP
    values when features are already scaled.  Only non-zero coefficients are
    shown (ElasticNet with L1 penalty sets many to exactly zero).

    Parameters
    ----------
    pipeline     : sklearn Pipeline  Fitted pipeline ending with an ElasticNet step
                                     named 'model'.
    feature_names: list[str]         Column names matching the features the pipeline
                                     was trained on.
    name         : str               Label shown in the plot title.
    top_n        : int               Maximum number of features to display (default 20).
    """
    coef = pipeline.named_steps['model'].coef_
    coef_series = pd.Series(coef, index=feature_names)

    # Keep only non-zero coefficients, sorted by absolute magnitude
    nonzero = coef_series[coef_series != 0].reindex(
        coef_series[coef_series != 0].abs().sort_values(ascending=False).index
    ).head(top_n)

    if len(nonzero) == 0:
        print(f"  {name}: all ElasticNet coefficients are zero — model predicts the mean.")
        return coef_series

    colors = ['#2d8b8b' if v > 0 else '#c45c5c' for v in nonzero.values]

    fig, ax = plt.subplots(figsize=(7, max(3, len(nonzero) * 0.35)))
    ax.barh(range(len(nonzero)), nonzero.values[::-1], color=colors[::-1])
    ax.set_yticks(range(len(nonzero)))
    ax.set_yticklabels(nonzero.index[::-1], fontsize=9)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Standardised coefficient', fontsize=11)
    ax.set_title(f'ElasticNet Coefficients — {name}', fontsize=12)
    plt.tight_layout()
    plt.show()

    print(f"\n  {name}: {len(nonzero)} non-zero coefficients "
          f"(of {len(coef_series)} features, ElasticNet sparsity = "
          f"{(coef == 0).mean()*100:.1f}%)")
    return coef_series


# ══════════════════════════════════════════════════════════════════════════════
# ADVANCED PLS + RENT  (Nested CV + Optuna)
# ══════════════════════════════════════════════════════════════════════════════

class _PLSWrapper(BaseEstimator, RegressorMixin):
    """Thin sklearn-compatible wrapper around PLSRegression.

    PLSRegression.predict() returns shape (n_samples, 1).  This wrapper
    squeezes the output to 1D so sklearn scorers and OptunaSearchCV work
    without modification.  Only `n_components` is exposed as a hyperparameter.
    """
    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X, y):
        from sklearn.cross_decomposition import PLSRegression
        self._pls = PLSRegression(n_components=self.n_components, scale=False)
        self._pls.fit(X, np.asarray(y).ravel())
        return self

    def predict(self, X):
        return self._pls.predict(X).ravel()


def run_advanced_pls_rent(
    df_combined, target_col='pain_reduction_pct', random_state=42,
    tau_1=0.7, tau_2=0.75, tau_3=0.95,
):
    """PLS regression with RENT feature selection + nested CV + Optuna.

    Partial Least Squares (PLS) is specifically suited for high-p, low-n data
    and weak feature-target correlations: it projects features into a small
    number of latent components that maximally covary with the target.

    Preprocessing (same as ElasticNet RENT):
      1. OrdinalEncoder applied once before outer CV (fixed mapping).
      2. SimpleImputer (median) + StandardScaler fitted on X_train per outer
         fold for RENT; re-fitted per inner split inside the sklearn Pipeline.

    Only one hyperparameter is tuned: n_components (1 – 15).  Upper bound of
    15 is conservative for ~45 inner-train samples; higher components would
    only fit noise.

    Outer CV : RepeatedKFold(n_splits=4, n_repeats=5) = 20 outer folds.
    Inner CV : RepeatedKFold(n_splits=4, n_repeats=5) = 20 fits per trial.
    Optuna   : 20 trials per outer fold, scoring = neg_root_mean_squared_error.

    Parameters
    ----------
    df_combined : pd.DataFrame  Combined T1 dataset (immunological + clinical).
    target_col  : str           Regression target (default: 'pain_reduction_pct').
    random_state: int           Random seed (default 42).
    tau_1       : float         RENT τ₁ cutoff (default 0.7).
    tau_2       : float         RENT τ₂ cutoff (default 0.75).
    tau_3       : float         RENT τ₃ cutoff (default 0.95).

    Returns
    -------
    results_df              : pd.DataFrame  Per-fold metrics + Mean/Std rows.
    best_params_df          : pd.DataFrame  Best n_components per outer fold.
    final_pipeline          : Pipeline      Fitted (Imputer→Scaler→PLS) on full data.
    X_final                 : pd.DataFrame  OrdinalEncoded feature matrix (RENT-selected).
    y_pred                  : pd.Series     Full-data predictions from final_pipeline.
    selected_features_per_fold : list[list[str]]  Selected feature names per outer fold.
    """
    import optuna
    import warnings
    try:
        from optuna.integration import OptunaSearchCV
    except ImportError:
        from optuna_integration import OptunaSearchCV
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OrdinalEncoder, StandardScaler
    from sklearn.pipeline import Pipeline
    from RENT import RENT

    warnings.filterwarnings('ignore', category=FutureWarning, module='RENT')
    warnings.filterwarnings('ignore', category=RuntimeWarning, module='RENT')
    warnings.filterwarnings('ignore', message='OptunaSearchCV is experimental')

    # Exclusion logic
    id_cols = ['Patient', 'Timepoint', 'Date', 'date', 'measurement_timepoint']
    leaky_cols = [c for c in df_combined.columns
                  if any(pat in c for pat in CL_MODEL_LEAKY_PATTERNS)]
    exclude = set(id_cols) | set(leaky_cols) | {target_col}

    feature_cols = [c for c in df_combined.columns if c not in exclude]
    X_raw = df_combined[feature_cols].copy()
    y = df_combined[target_col].copy()

    valid = y.notna()
    X_raw, y = X_raw[valid].reset_index(drop=True), y[valid].reset_index(drop=True)

    # OrdinalEncode str/category columns once upfront
    X_oe = X_raw.copy()
    cat_cols_list = [c for c in X_oe.columns
                     if X_oe[c].dtype == object or str(X_oe[c].dtype) == 'category']
    if cat_cols_list:
        oe_global = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_oe[cat_cols_list] = oe_global.fit_transform(X_oe[cat_cols_list].astype(str))

    print(f"\n{'='*65}")
    print(f"  PLS + RENT — {target_col}")
    print(f"  Samples: {len(X_oe)},  Features: {len(feature_cols)}")
    print(f"  RENT: K=100, τ₁={tau_1}, τ₂={tau_2}, τ₃={tau_3}")
    print(f"  Preprocessing: OrdinalEncode → Impute (median) → StandardScale")
    print(f"  Optuna: n_components ∈ [1, min(15, n_features_selected, 0.75×n_train)]")
    print(f"  Outer: 4×5=20 folds  |  Inner: 4×5=20 fits/trial  |  Trials: 20")
    print(f"{'='*65}")

    outer_cv = RepeatedKFold(n_splits=4, n_repeats=5, random_state=random_state)
    inner_cv = RepeatedKFold(n_splits=4, n_repeats=5, random_state=random_state)

    fold_results           = []
    best_params_list       = []
    selected_features_per_fold = []
    optuna.logging.set_verbosity(optuna.logging.INFO)
    start = time.time()

    for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X_oe), start=1):
        print(f"\n  Outer fold {outer_fold}/{outer_cv.get_n_splits()}")

        X_oe_train = X_oe.iloc[train_idx].copy()
        X_oe_test  = X_oe.iloc[test_idx].copy()
        y_train    = y.iloc[train_idx]
        y_test     = y.iloc[test_idx]

        # ── Preprocess X_train for RENT ───────────────────────────────────────
        imputer_rent = SimpleImputer(strategy='median').fit(X_oe_train)
        X_imp_train  = imputer_rent.transform(X_oe_train)
        scaler_rent  = StandardScaler().fit(X_imp_train)
        X_rent_train = pd.DataFrame(
            scaler_rent.transform(X_imp_train), columns=feature_cols)

        # ── RENT feature selection ────────────────────────────────────────────
        rent_model = RENT.RENT_Regression(
            data=X_rent_train,
            target=y_train.values,
            feat_names=feature_cols,
            C=[0.1, 1, 10],
            l1_ratios=[0.1, 0.5, 0.9],
            autoEnetParSel=True,
            poly='OFF',
            testsize_range=(0.25, 0.25),
            K=100,
            random_state=random_state,
            verbose=0,
        )
        rent_model.train()
        selected_idx = rent_model.select_features(
            tau_1_cutoff=tau_1, tau_2_cutoff=tau_2, tau_3_cutoff=tau_3)

        if len(selected_idx) == 0:
            print(f"    RENT: 0 features selected — using all {len(feature_cols)}")
            selected_cols = feature_cols
        else:
            selected_cols = [feature_cols[i] for i in selected_idx]
            preview = selected_cols[:8]
            suffix  = '...' if len(selected_cols) > 8 else ''
            print(f"    RENT: {len(selected_cols)}/{len(feature_cols)} features — {preview}{suffix}")

        selected_features_per_fold.append(selected_cols)

        # ── Inner CV + Optuna: Pipeline(Imputer → Scaler → PLS) ──────────────
        # n_components must be ≤ min(n_features, n_inner_train_samples)
        # Use n_inner_train_samples ≈ 0.75 * len(X_oe_train) (inner split size)
        max_components = max(1, min(
            15,
            len(selected_cols),
            int(0.75 * len(X_oe_train)) - 1,
        ))
        param_distributions = {
            'model__n_components': optuna.distributions.IntDistribution(1, max_components),
        }
        print(f"    n_components ∈ [1, {max_components}]")

        inner_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler',  StandardScaler()),
            ('model',   _PLSWrapper()),
        ])

        optuna_search = OptunaSearchCV(
            estimator=inner_pipe,
            param_distributions=param_distributions,
            cv=inner_cv,
            scoring='neg_root_mean_squared_error',
            n_trials=20,
            n_jobs=-1,
            verbose=0,
        )

        optuna_search.fit(X_oe_train[selected_cols], y_train)
        best_params_list.append(optuna_search.best_params_)

        preds = optuna_search.predict(X_oe_test[selected_cols])
        rmse  = np.sqrt(mean_squared_error(y_test, preds))
        mae   = mean_absolute_error(y_test, preds)
        r2    = r2_score(y_test, preds)
        mse   = rmse ** 2

        fold_results.append({'Fold': outer_fold, 'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2})
        print(f"    MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}")
        print(f"    Best params: {optuna_search.best_params_}")

    elapsed = time.time() - start
    print(f"\n  Training time: {elapsed:.1f}s  ({elapsed/60:.1f} min)")

    # Results DataFrame + summary
    results_df  = pd.DataFrame(fold_results)
    metric_cols = ['MAE', 'MSE', 'RMSE', 'R2']
    mean_row = {'Fold': 'Mean', **{m: results_df[m].mean() for m in metric_cols}}
    std_row  = {'Fold': 'Std',  **{m: results_df[m].std()  for m in metric_cols}}
    results_df = pd.concat(
        [results_df, pd.DataFrame([mean_row, std_row])], ignore_index=True)

    n_outer = len(fold_results)
    t_crit  = stats.t.ppf(0.975, df=n_outer - 1)
    print(f"\n  Summary (4×5 outer CV + RENT, 20 Optuna trials, 95% CI):")
    for m in metric_cols:
        mv = mean_row[m]; sv = std_row[m]
        ci = t_crit * sv / np.sqrt(n_outer)
        print(f"    {m:<5}: {mv:.3f} ± {sv:.4f}  (95% CI [{mv - ci:.3f}, {mv + ci:.3f}])")

    best_params_df = pd.DataFrame(best_params_list)
    best_params_df.index = [f"Fold {i+1}" for i in range(len(best_params_list))]
    print(f"\n  Best hyperparameters per outer fold:")
    print(best_params_df.to_string())

    # Feature selection frequency
    from collections import Counter
    all_selected = [f for fold_feats in selected_features_per_fold for f in fold_feats]
    freq = Counter(all_selected)
    print(f"\n  RENT feature selection frequency (top 20 across {n_outer} folds):")
    for feat, cnt in freq.most_common(20):
        print(f"    {cnt:>3}/{n_outer}  {feat}")

    # ── Final model on full dataset ───────────────────────────────────────────
    imp_full = SimpleImputer(strategy='median').fit(X_oe)
    sca_full = StandardScaler().fit(imp_full.transform(X_oe))
    X_rent_full = pd.DataFrame(
        sca_full.transform(imp_full.transform(X_oe)), columns=feature_cols)

    rent_final = RENT.RENT_Regression(
        data=X_rent_full,
        target=y.values,
        feat_names=feature_cols,
        C=[0.1, 1, 10],
        l1_ratios=[0.1, 0.5, 0.9],
        autoEnetParSel=True,
        poly='OFF',
        testsize_range=(0.25, 0.25),
        K=100,
        random_state=random_state,
        verbose=0,
    )
    rent_final.train()
    final_idx  = rent_final.select_features(
        tau_1_cutoff=tau_1, tau_2_cutoff=tau_2, tau_3_cutoff=tau_3)
    final_cols = [feature_cols[i] for i in final_idx] if len(final_idx) > 0 else feature_cols
    print(f"\n  Final model RENT selected {len(final_cols)}/{len(feature_cols)} features.")

    X_final = X_oe[final_cols]
    best_nc = optuna_search.best_params_['model__n_components']

    final_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler()),
        ('model',   _PLSWrapper(n_components=best_nc)),
    ])
    final_pipeline.fit(X_final, y)
    y_pred = pd.Series(final_pipeline.predict(X_final), index=range(len(X_final)), dtype='float64')

    return results_df, best_params_df, final_pipeline, X_final, y_pred, selected_features_per_fold


def plot_pls_coefficients(pipeline, feature_names, name, top_n=20):
    """Bar chart of PLS regression coefficients (feature importance).

    Plots the standardised regression coefficients from the PLS model
    (stored inside the _PLSWrapper as _pls.coef_).  Positive values
    mean the feature increases the predicted target; negative values
    mean it decreases it.

    Parameters
    ----------
    pipeline     : sklearn Pipeline  Fitted pipeline with a '_PLSWrapper' step named 'model'.
    feature_names: list[str]         Feature column names.
    name         : str               Label for the plot title.
    top_n        : int               Maximum features to display (default 20).
    """
    coef = pipeline.named_steps['model']._pls.coef_.ravel()
    coef_series = pd.Series(coef, index=feature_names)
    top = coef_series.reindex(
        coef_series.abs().sort_values(ascending=False).index).head(top_n)

    colors = ['#2d8b8b' if v > 0 else '#c45c5c' for v in top.values]

    fig, ax = plt.subplots(figsize=(7, max(3, len(top) * 0.35)))
    ax.barh(range(len(top)), top.values[::-1], color=colors[::-1])
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index[::-1], fontsize=9)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('PLS standardised coefficient', fontsize=11)
    ax.set_title(f'PLS Coefficients (top {top_n}) — {name}', fontsize=12)
    plt.tight_layout()
    plt.show()

    nc = pipeline.named_steps['model'].n_components
    print(f"\n  {name}: n_components={nc}, showing top {len(top)} coefficients by magnitude")
    return coef_series


# ══════════════════════════════════════════════════════════════════════════════
# ADVANCED HGB  (Nested CV + Optuna) — placeholder
# ══════════════════════════════════════════════════════════════════════════════

def run_advanced_hgb(df_combined, target_col='pain_reduction_pct', random_state=42):
    """HistGradientBoostingRegressor with nested CV and Optuna hyperparameter tuning.

    Parameters
    ----------
    df_combined  : pd.DataFrame  Combined T1 dataset (immunological + clinical).
    target_col   : str           Regression target (default: 'pain_reduction_pct').
    random_state : int           Random seed (default 42).

    Returns
    -------
    results_df     : pd.DataFrame                  Per-fold metrics + Mean/Std rows.
    best_params_df : pd.DataFrame                  Best params per outer fold.
    model          : HistGradientBoostingRegressor  Final model trained on full data.
    X              : pd.DataFrame                  Feature matrix used.
    y_pred         : pd.Series                     Full-data predictions from final model.
    """
    import optuna
    try:
        from optuna.integration import OptunaSearchCV
    except ImportError:
        from optuna_integration import OptunaSearchCV

    # Same exclusion logic as run_advanced_catboost
    id_cols = ['Patient', 'Timepoint', 'Date', 'date', 'measurement_timepoint']
    leaky_cols = [c for c in df_combined.columns
                  if any(pat in c for pat in CL_MODEL_LEAKY_PATTERNS)]
    exclude = set(id_cols) | set(leaky_cols) | {target_col}

    feature_cols = [c for c in df_combined.columns if c not in exclude]
    X = df_combined[feature_cols].copy()
    y = df_combined[target_col].copy()

    valid = y.notna()
    X, y = X[valid].reset_index(drop=True), y[valid].reset_index(drop=True)

    # HGB handles categoricals natively when dtype is 'category' (sklearn >= 1.2)
    for col in X.select_dtypes(include=['object', 'category']).columns:
        X[col] = X[col].astype('category')

    print(f"\n{'='*65}")
    print(f"  Advanced HGB — {target_col}")
    print(f"  Samples: {len(X)},  Features: {len(feature_cols)}")
    print(f"  Outer: 4×5=20 folds  |  Inner: 4×5=20 fits/trial  |  Trials: 20")
    print(f"{'='*65}")

    outer_cv = RepeatedKFold(n_splits=4, n_repeats=5, random_state=random_state)
    inner_cv = RepeatedKFold(n_splits=4, n_repeats=5, random_state=random_state)

    param_distributions = {
        'max_iter':          optuna.distributions.IntDistribution(100, 1000),
        'max_depth':         optuna.distributions.IntDistribution(3, 10),
        'learning_rate':     optuna.distributions.FloatDistribution(1e-3, 0.3, log=True),
        'min_samples_leaf':  optuna.distributions.IntDistribution(5, 50),
        'l2_regularization': optuna.distributions.FloatDistribution(1e-4, 10.0, log=True),
        'max_leaf_nodes':    optuna.distributions.IntDistribution(15, 63),
    }

    fold_results     = []
    best_params_list = []
    optuna.logging.set_verbosity(optuna.logging.INFO)
    start = time.time()

    for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X), start=1):
        print(f"\n  Outer fold {outer_fold}/{outer_cv.get_n_splits()}")

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        base_model = HistGradientBoostingRegressor(
            categorical_features='from_dtype',
            random_state=random_state,
        )

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

    results_df  = pd.DataFrame(fold_results)
    metric_cols = ['MAE', 'MSE', 'RMSE', 'R2']
    mean_row = {'Fold': 'Mean', **{m: results_df[m].mean() for m in metric_cols}}
    std_row  = {'Fold': 'Std',  **{m: results_df[m].std()  for m in metric_cols}}
    results_df = pd.concat(
        [results_df, pd.DataFrame([mean_row, std_row])], ignore_index=True)

    n_outer = len(fold_results)
    t_crit  = stats.t.ppf(0.975, df=n_outer - 1)
    print(f"\n  Summary (4×5 outer CV, 20 Optuna trials, 95% CI):")
    for m in metric_cols:
        mv = mean_row[m]; sv = std_row[m]
        ci = t_crit * sv / np.sqrt(n_outer)
        print(f"    {m:<5}: {mv:.3f} ± {sv:.4f}  (95% CI [{mv - ci:.3f}, {mv + ci:.3f}])")

    best_params_df = pd.DataFrame(best_params_list)
    best_params_df.index = [f"Fold {i+1}" for i in range(len(best_params_list))]
    print(f"\n  Best hyperparameters per outer fold:")
    print(best_params_df.to_string())

    # Final model on full dataset (last fold's best params) for SHAP
    final_model = HistGradientBoostingRegressor(
        categorical_features='from_dtype',
        random_state=random_state,
        **optuna_search.best_params_,
    )
    final_model.fit(X, y)
    y_pred = pd.Series(final_model.predict(X), index=range(len(X)), dtype='float64')

    return results_df, best_params_df, final_model, X, y_pred
