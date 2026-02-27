# Modeling functions — baseline and advanced regressors
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
from sklearn.model_selection import RepeatedKFold
from catboost import CatBoostRegressor, Pool
import shap


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
      - Leaky cols  : any column whose name contains 'response',
                      'improvement_percent', 'pain_scale', or
                      'pain_reduction_pct'

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
    # Build the exclusion set: ID columns + target + any leaky column names
    
    always_exclude = [
        'Patient',
        'Timepoint',
        'Date',
        'date',
        'measurement_timepoint',
        'pain_scale',          # T1 baseline pain — embedded in both target formulas
        'pain_scale_t2',       # T2 outcome — always leaky
        'pain_scale_reduction',
        'pain_reduction_pct',
    ]

    exclude = set(always_exclude + [target_col])

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

def prepare_baseline_datasets(df_im_vis, df_cl_bcat, pain_targets):
    """Build the three T1 modeling datasets for baseline CatBoost.

    All three datasets receive targets merged in via left join.
    run_catboost_regressor handles leaky column exclusion internally via
    leaky_patterns — no manual dropping needed here.

    Parameters
    ----------
    df_im_vis    : pd.DataFrame   immunological dataset after >25% NaN drop (NOT imputed)
    df_cl_vis   : pd.DataFrame    clinical dataset,  cleaned
    pain_targets : pd.DataFrame   per-patient targets: Patient, pain_scale_t2,  pain_scale_reduction, pain_reduction_pct

    Returns
    -------
    df_im_raw_t1       : immunological T1 + targets
    df_cl_bcat_t1      : clinical T1 (raw) + targets
    df_bcat_combined_t1: inner join of the two above (suffixes _im/_cl for duplicate cols)
    """
    model_patients = set(pain_targets['Patient'].values)

    # Immunological T1 + targets
    df_im_raw_t1 = (
        df_im_vis[
            (df_im_vis['Timepoint'] == 1) &
            (df_im_vis['Patient'].isin(model_patients))
        ]
        .copy()
        .reset_index(drop=True)
    )
    df_im_raw_t1 = df_im_raw_t1.merge(
        pain_targets[['Patient', 'pain_scale_reduction', 'pain_reduction_pct']],
        on='Patient', how='left'
    )

    # Clinical T1 (raw, unparsed) + targets
    df_cl_bcat_t1 = (
        df_cl_bcat[
            (df_cl_bcat['Timepoint'] == 1) &
            (df_cl_bcat['Patient'].isin(model_patients))
        ]
        .copy()
        .reset_index(drop=True)
    )
    df_cl_bcat_t1 = df_cl_bcat_t1.merge(
        pain_targets[['Patient', 'pain_scale_reduction', 'pain_reduction_pct']],
        on='Patient', how='left'
    )

    # Combined T1: inner join on Patient + Timepoint
    # Both sides are already filtered to Timepoint==1; joining on both keys
    # avoids duplicates and ensures exact patient-timepoint matching.
    # Duplicate feature columns get suffixes _im/_cl;
    # run_catboost_regressor excludes leaky cols via leaky_patterns.
    df_bcat_combined_t1 = df_im_raw_t1.merge(
        df_cl_bcat_t1,
        on=['Patient', 'Timepoint'], how='inner',
        suffixes=('_im', '_cl')
    )

    print(f"\nBaseline T1 datasets:")
    print(f"  Immunological : {df_im_raw_t1.shape},  patients: {df_im_raw_t1['Patient'].nunique()}")
    print(f"  Clinical      : {df_cl_bcat_t1.shape},  patients: {df_cl_bcat_t1['Patient'].nunique()}")
    print(f"  Combined      : {df_bcat_combined_t1.shape}, patients: {df_bcat_combined_t1['Patient'].nunique()}")

    return df_im_raw_t1, df_cl_bcat_t1, df_bcat_combined_t1






def run_baseline_catboost(df_im_raw_t1, df_cl_bcat_t1, df_bcat_combined_t1):
    """Run baseline CatBoost on both regression targets across all three datasets.

    Runs pain_reduction_pct (primary) and pain_scale_t2 (secondary).
    Prints SHAP plots for each dataset × target combination.

    Parameters
    ----------
    df_im_raw_t1        : immunological T1 + targets
    df_cl_bcat_t1       : clinical T1 (raw) + targets
    df_bcat_combined_t1 : combined T1 + targets

    Returns
    -------
    results : dict with keys 'pain_reduction_pct, 'pain_reduction_pct' and 'pain_scale_t2',
              each containing a dict: {dataset_name: (results_df, model, X, y_pred)}
    shap_values : dict with the same structure, values are shap_values arrays
    """
    results     = {}
    shap_values = {}

    for target in ['pain_scale_reduction', 'pain_reduction_pct']:
        print(f"\n{'='*70}")
        print(f"  CATBOOST BASELINE REGRESSOR — Target: {target}")
        print(f"{'='*70}")

        res_im,   model_im,   X_im,   ypred_im   = run_catboost_regressor(
            df_im_raw_t1,       target, "Immunological (raw T1)")
        res_cl,   model_cl,   X_cl,   ypred_cl   = run_catboost_regressor(
            df_cl_bcat_t1,      target, "Clinical (raw T1)")
        res_comb, model_comb, X_comb, ypred_comb = run_catboost_regressor(
            df_bcat_combined_t1, target, "Combined (raw T1)")

        print_regression_summary(
            {"Immunological": res_im, "Clinical": res_cl, "Combined": res_comb},
            target
        )

        sv_im   = plot_shap_regressor(model_im,   X_im,   f"Immunological — {target}")
        sv_cl   = plot_shap_regressor(model_cl,   X_cl,   f"Clinical — {target}")
        sv_comb = plot_shap_regressor(model_comb, X_comb, f"Combined — {target}")

        results[target] = {
            'Immunological': (res_im,   model_im,   X_im,   ypred_im),
            'Clinical':      (res_cl,   model_cl,   X_cl,   ypred_cl),
            'Combined':      (res_comb, model_comb, X_comb, ypred_comb),
        }
        shap_values[target] = {
            'Immunological': sv_im,
            'Clinical':      sv_cl,
            'Combined':      sv_comb,
        }

    return results, shap_values



# ══════════════════════════════════════════════════════════════════════════════
# ADVANCED CATBOOST  (Nested CV + Optuna)
# ══════════════════════════════════════════════════════════════════════════════

def prepare_advanced_dataset(df_im_vis, df_cl_mod, pain_targets):
    """Build the single combined T1 dataset for advanced CatBoost modeling.

    Inner join of df_im_vis T1 + df_cl_mod T1 on Patient + Timepoint.
    Merges pain_scale_reduction and pain_reduction_pct targets.
    Duplicate target columns from both sides get _im/_cl suffixes; the _im
    copies are dropped and _cl copies are renamed to clean names.

    Parameters
    ----------
    df_im_vis    : pd.DataFrame  Immunological dataset (NOT imputed, outliers removed).
    df_cl_mod    : pd.DataFrame  Clinical dataset, parsed + cleaned (df_cl_vis copy).
    pain_targets : pd.DataFrame  Per-patient targets: Patient, pain_scale_reduction,
                                 pain_reduction_pct.

    Returns
    -------
    df_combined : pd.DataFrame  Combined T1 dataset ready for advanced modeling.
    """
    model_patients = set(pain_targets['Patient'].values)

    # Immunological T1 rows + targets
    df_im_t1 = (
        df_im_vis[
            (df_im_vis['Timepoint'] == 1) &
            (df_im_vis['Patient'].isin(model_patients))
        ]
        .copy()
        .reset_index(drop=True)
    )
    df_im_t1 = df_im_t1.merge(
        pain_targets[['Patient', 'pain_scale_reduction', 'pain_reduction_pct']],
        on='Patient', how='left'
    )

    # Clinical T1 rows + targets
    df_cl_t1 = (
        df_cl_mod[
            (df_cl_mod['Timepoint'] == 1) &
            (df_cl_mod['Patient'].isin(model_patients))
        ]
        .copy()
        .reset_index(drop=True)
    )
    df_cl_t1 = df_cl_t1.merge(
        pain_targets[['Patient', 'pain_scale_reduction', 'pain_reduction_pct']],
        on='Patient', how='left'
    )

    # Inner join on Patient + Timepoint; duplicate feature cols get _im/_cl suffixes
    df_combined = df_im_t1.merge(
        df_cl_t1,
        on=['Patient', 'Timepoint'], how='inner',
        suffixes=('_im', '_cl')
    )

    # Drop _im copies of target cols; rename _cl copies to clean names
    df_combined = (
        df_combined
        .drop(columns=['pain_scale_reduction_im', 'pain_reduction_pct_im'], errors='ignore')
        .rename(columns={
            'pain_scale_reduction_cl': 'pain_scale_reduction',
            'pain_reduction_pct_cl':   'pain_reduction_pct',
        })
    )

    print(f"\Tuned Modeling Combined T1 dataset: {df_combined.shape}, "
          f"patients: {df_combined['Patient'].nunique()}")

    return df_combined


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

    # Build the exclusion set — identical logic to baseline
    always_exclude = [
        'Patient', 'Timepoint', 'Date', 'date', 'measurement_timepoint',
        'pain_scale',
        'pain_scale_t2',
        'pain_scale_reduction',
        'pain_reduction_pct',
    ]
    exclude = set(always_exclude + [target_col])

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
    inner_cv = RepeatedKFold(n_splits=4, n_repeats=5, random_state=random_state)

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
