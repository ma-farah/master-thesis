# Modeling functions — baseline and advanced regressors
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold
from catboost import CatBoostRegressor, Pool
import shap


# ══════════════════════════════════════════════════════════════════════════════
# BASELINE CATBOOST
# ══════════════════════════════════════════════════════════════════════════════

def regression_metrics(y_true, y_pred):
    """Return dict of MAE, MSE, RMSE, R² for a regression prediction."""
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, y_pred)
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}


def run_catboost_regressor(df_model, target_col, name,
                           n_splits=5, n_repeats=5, random_state=42):
    """5-fold × 5-repeat RepeatedKFold CatBoostRegressor (25 fits). No hyperparameter tuning.
    Returns (results_df, last_trained_model, X_features, y_pred_series).

    Automatically excluded from features:
      - ID columns  : Patient, Timepoint, Date, date, measurement_timepoint
      - Leaky cols  : any column whose name contains 'response', 'improvement_percent',
                      'pain_scale', or 'pain_reduction_pct'
    """
    always_exclude = ['Patient', 'Timepoint', 'Date', 'date', 'measurement_timepoint']
    leaky_patterns = ['response', 'improvement_percent', 'pain_scale', 'pain_reduction_pct']
    exclude = set(always_exclude + [target_col])
    for col in df_model.columns:
        if any(pat in col.lower() for pat in leaky_patterns):
            exclude.add(col)

    feature_cols = [c for c in df_model.columns if c not in exclude]
    X = df_model[feature_cols].copy()
    y = df_model[target_col].copy()

    valid = y.notna()
    X, y = X[valid].reset_index(drop=True), y[valid].reset_index(drop=True)

    for col in X.select_dtypes(include=['category', 'object']).columns:
        X[col] = X[col].astype(str)
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()

    print(f"\n{'='*65}")
    print(f"  CatBoost Regressor Baseline — {name}")
    print(f"  Target : {target_col}")
    print(f"  Samples: {len(X)},  Features: {len(feature_cols)}")
    print(f"  CV     : {n_splits}-fold × {n_repeats} repeats = {n_splits * n_repeats} fits")
    print(f"{'='*65}")

    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    fold_results = []
    y_pred = pd.Series(np.nan, index=range(len(X)), dtype='float64')

    for fold, (train_idx, test_idx) in enumerate(rkf.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = CatBoostRegressor(iterations=300, random_seed=random_state, verbose=0)
        model.fit(
            Pool(X_train, y_train, cat_features=cat_cols),
            eval_set=Pool(X_test, y_test, cat_features=cat_cols),
            use_best_model=False
        )

        preds = model.predict(X_test)
        y_pred.iloc[test_idx] = preds

        m = regression_metrics(y_test, preds)
        fold_results.append({'Fold': fold + 1, **m})
        print(f"  Fold {fold+1:>2}: MAE={m['MAE']:.3f}  MSE={m['MSE']:.3f}  "
              f"RMSE={m['RMSE']:.3f}  R²={m['R2']:.3f}")

    results_df = pd.DataFrame(fold_results)
    metric_cols = ['MAE', 'MSE', 'RMSE', 'R2']
    mean_row = {'Fold': 'Mean', **{m: results_df[m].mean() for m in metric_cols}}
    std_row  = {'Fold': 'Std',  **{m: results_df[m].std()  for m in metric_cols}}
    results_df = pd.concat(
        [results_df, pd.DataFrame([mean_row, std_row])], ignore_index=True)

    print(f"\n  Summary ({n_splits}x{n_repeats} CV):")
    for m in metric_cols:
        mv = results_df.loc[results_df['Fold'] == 'Mean', m].iloc[0]
        sv = results_df.loc[results_df['Fold'] == 'Std',  m].iloc[0]
        print(f"    {m:<5}: {mv:.3f} ± {sv:.4f}")

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
    """Print a mean ± std summary table across all datasets for a given target."""
    metric_cols = ['MAE', 'MSE', 'RMSE', 'R2']
    rows = []
    for ds_name, res_df in results_dict.items():
        fold_rows = res_df[~res_df['Fold'].isin(['Mean', 'Std'])]
        row = {'Dataset': ds_name}
        for m in metric_cols:
            row[m] = f"{fold_rows[m].mean():.3f} ± {fold_rows[m].std():.4f}"
        rows.append(row)
    summary = pd.DataFrame(rows)
    print(f"\n{'='*70}")
    print(f"  CATBOOST BASELINE SUMMARY — Target: {target_col}")
    print(f"{'='*70}")
    print(summary.to_string(index=False))
    return summary

def prepare_baseline_datasets(df_im_vis, df_cl_bcat, pain_targets):
    """Build the three T1 modeling datasets for baseline CatBoost.

    Parameters
    ----------
    df_im_vis    : pd.DataFrame   immunological dataset after >25% NaN drop (NOT imputed)
    df_cl_bcat   : pd.DataFrame   clinical dataset, English names, raw unparsed values
    pain_targets : pd.DataFrame   per-patient targets: Patient, pain_scale_t2, pain_reduction_pct

    Returns
    -------
    df_im_raw_t1       : immunological T1 (features only, no targets)
    df_cl_bcat_t1      : clinical T1 + targets
    df_bcat_combined_t1: combined T1 + targets (target from clinical side only)
    """
    model_patients = set(pain_targets['Patient'].values)

    # Immunological T1 — features only, NO targets
    df_im_raw_t1 = (
        df_im_vis[
            (df_im_vis['Timepoint'] == 1) &
            (df_im_vis['Patient'].isin(model_patients))
        ]
        .copy()
        .reset_index(drop=True)
    )

    # Clinical T1 — with targets
    df_cl_bcat_t1 = (
        df_cl_bcat[
            (df_cl_bcat['Timepoint'] == 1) &
            (df_cl_bcat['Patient'].isin(model_patients))
        ]
        .copy()
        .reset_index(drop=True)
    )
    df_cl_bcat_t1 = df_cl_bcat_t1.merge(
        pain_targets[['Patient', 'pain_scale_t2', 'pain_reduction_pct']],
        on='Patient', how='left'
    )
    df_cl_bcat_t1 = df_cl_bcat_t1.dropna(subset=['pain_reduction_pct']).reset_index(drop=True)

    # Combined: immunological features + clinical features + clinical targets
    # Drop leaky clinical columns before merging
    leaky_cols = ['pain_scale_t2', 'pain_reduction_pct', 
                  'improvement_percent', 'response', 'response_category', 'response_percent']
    df_cl_features_only = df_cl_bcat_t1.drop(columns=leaky_cols, errors='ignore')
    
    df_bcat_combined_t1 = df_im_raw_t1.merge(
        df_cl_features_only,
        on='Patient', how='inner' #and timepoint!
    )
    
    # Add targets from clinical side only
    df_bcat_combined_t1 = df_bcat_combined_t1.merge(
        pain_targets[['Patient', 'pain_scale_t2', 'pain_reduction_pct']],
        on='Patient', how='left'
    )
    df_bcat_combined_t1 = df_bcat_combined_t1.dropna(subset=['pain_reduction_pct']).reset_index(drop=True)

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
    results : dict with keys 'pain_reduction_pct' and 'pain_scale_t2',
              each containing a dict: {dataset_name: (results_df, model, X, y_pred)}
    shap_values : dict with the same structure, values are shap_values arrays
    """
    results     = {}
    shap_values = {}

    for target in ['pain_reduction_pct', 'pain_scale_t2']:
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
# ADVANCED CATBOOST  (Nested CV + Optuna) — placeholder
# ══════════════════════════════════════════════════════════════════════════════

# TODO: implement nested CV + Optuna tuning
# Outer : RepeatedKFold(n_splits=4, n_repeats=5) = 20 folds
# Inner : RepeatedKFold(n_splits=4, n_repeats=5) = 20 fits per Optuna trial
# Optuna: 20 trials, objective = minimize RMSE
# Dataset: df_combined (df_im_mod T1 + df_cl_mod T1, inner join)
# No pre-imputation needed (CatBoost handles numeric NaN natively)
# SHAP analysis on final model


# ══════════════════════════════════════════════════════════════════════════════
# ADVANCED HGB  (Nested CV + Optuna) — placeholder
# ══════════════════════════════════════════════════════════════════════════════

# TODO: implement HistGradientBoostingRegressor nested CV
# Same nested CV structure as Advanced CatBoost
# OrdinalEncoder for categoricals inside Pipeline
# Objective: minimize RMSE
# Feature importance: HGB built-in + SHAP
