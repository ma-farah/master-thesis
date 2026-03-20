
# CatBoost + MRMR feature selection + Optuna hyperparameter tuning
import time
import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OrdinalEncoder
from scipy import stats
from sklearn.model_selection import RepeatedKFold
from collections import Counter
import joblib, os
import contextlib, io

import preprocess


def _prep_for_catboost(df, cat_cols):
    """Convert category columns to object dtype, filling NaN with 'missing'."""
    out = df.copy()
    for col in cat_cols:
        if col in out.columns:
            out[col] = out[col].astype(object).fillna('missing')
    return out

# _____________________________________________________________________________
# CatBoost + Optuna tuning
# ══════════════════════════════════════════════════════════════════════════════

def run_advanced_catboost_mrmr(
    df_combined, feature_list, target_col='pain_reduction', random_state=42,
    K=20, target_transformer=None,
):
    """CatBoostRegressor and Optuna-tuned hyperparameters.

    Using dataset with 11 selected features.

    Pipeline per outer fold
    -----------------------
    1. MRMR on X_train (simputed + encoded) → select top K features
          2. Inner CV (4×5=20) + Optuna (50 trials) tunes CatBoost HPs on raw X_train
    3. Train final fold model on raw X_train → evaluate on raw X_test (no imputation)

    Final model
    -----------
    Features selected in ≥75% of outer folds, median HPs across outer folds.

    Returns
    -------
    results_df, final_model, X_final, y_pred, best_model_params_list,
    feature_freq, patient_err_df
    """
    from catboost import CatBoostRegressor
    import optuna, warnings, statistics

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings('ignore', message='.*joblib.*')

    N_TRIALS_MODEL = 50   

    y            = df_combined[target_col].copy()
    exclude      = {'Patient', 'Timepoint', target_col, 'pain_reduction',
                    'pain_reduction_pct', 'pain_under_load_reduction',
                    'pain_under_load_reduction_pct'}
    feature_cols = [c for c in df_combined.columns if c not in exclude]
    X            = df_combined[feature_cols].copy()

    valid = y.notna()
    X, y  = X[valid].reset_index(drop=True), y[valid].reset_index(drop=True)
    patient_id_map = df_combined.loc[valid, 'Patient'].reset_index(drop=True)

    cat_cols = X.select_dtypes(include=['category', 'object']).columns.tolist()

    print(f"\n{'='*65}")
    print(f"  CatBoost + Optuna + MRMR — {target_col}")
    print(f"  n={len(X)}, p={len(feature_cols)}, K={K}")
    print(f"  Outer 4×5=20 | Inner 4×5=20 | Model trials={N_TRIALS_MODEL}")
    print(f"{'='*65}")

    outer_cv = RepeatedKFold(n_splits=4, n_repeats=5, random_state=random_state)
    inner_cv = RepeatedKFold(n_splits=4, n_repeats=5, random_state=random_state)

    fold_results            = []
    best_model_params_list  = []
    patient_errors          = []
    start = time.time()

    selected_cols = [f for f in feature_list if f in feature_cols]
    cat_cols_inner = [c for c in cat_cols if c in selected_cols]

    for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X), start=1):
        print(f"\n{'─'*65}")
        print(f"  Outer fold {outer_fold}/{outer_cv.get_n_splits()}")
        print(f"{'─'*65}")

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Power-transform target
        if target_transformer is not None:
            pt_fold     = clone(target_transformer)
            y_train_fit = pd.Series(
                pt_fold.fit_transform(y_train.values.reshape(-1, 1)).ravel(),
                index=y_train.index)
        else:
            pt_fold, y_train_fit = None, y_train

        # CatBoost-ready X_train (category → object, NaN preserved)
        X_train_cb = _prep_for_catboost(X_train, cat_cols)

        # ── Step 2: Inner CV — Optuna tunes CatBoost HPs ──────────────────────
        inner_splits = list(inner_cv.split(X_train_cb))

        def _fit_inner(itr, ival, params):
            m = CatBoostRegressor(
                iterations=500, **params, cat_features=cat_cols_inner,
                loss_function='RMSE', random_seed=random_state,
                task_type='CPU', thread_count=1, logging_level='Silent')
            with contextlib.redirect_stderr(io.StringIO()):
                m.fit(X_train_cb.iloc[itr][selected_cols], y_train_fit.iloc[itr])
            return np.sqrt(mean_squared_error(
                y_train_fit.iloc[ival],
                m.predict(X_train_cb.iloc[ival][selected_cols])))

        def model_objective(trial):
            params = dict(
                depth               = trial.suggest_int(  'depth',               3,    10),
                learning_rate       = trial.suggest_float('learning_rate',       1e-3, 0.3,  log=True),
                l2_leaf_reg         = trial.suggest_float('l2_leaf_reg',         1.0,  10.0, log=True),
                bagging_temperature = trial.suggest_float('bagging_temperature', 0.0,  1.0),
            )
            rmses = joblib.Parallel(n_jobs=-1, prefer='threads')(
                joblib.delayed(_fit_inner)(itr, ival, params)
                for itr, ival in inner_splits)
            return np.mean(rmses)

        def _cb(study, trial):
            if trial.state.name == 'COMPLETE':
                print(f"    Trial {trial.number+1:>3}/{N_TRIALS_MODEL}: "
                      f"RMSE={trial.value:.4f}  {trial.params}")

        model_study = optuna.create_study(direction='minimize')
        with contextlib.redirect_stderr(io.StringIO()):
            model_study.optimize(model_objective, n_trials=N_TRIALS_MODEL,
                                 callbacks=[_cb], show_progress_bar=False)

        best_model_params = model_study.best_params
        best_model_params_list.append(best_model_params)
        print(f"  Best Trial: {model_study.best_trial.number}   "
              f"RMSE={model_study.best_value:.4f}  {best_model_params}")

        # ── Step 3: Train on full X_train → evaluate on X_test ────────────────
        X_test_cb = _prep_for_catboost(X_test, cat_cols)

        fold_model = CatBoostRegressor(
            iterations=1000, **best_model_params, cat_features=cat_cols_inner,
            loss_function='RMSE', random_seed=random_state,
            task_type='CPU', thread_count=-1, logging_level='Silent')
        with contextlib.redirect_stderr(io.StringIO()):
            fold_model.fit(X_train_cb[selected_cols], y_train_fit)

        preds_raw = fold_model.predict(X_test_cb[selected_cols])
        preds     = (pt_fold.inverse_transform(preds_raw.reshape(-1, 1)).ravel()
                     if pt_fold is not None else preds_raw)

        mae  = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2   = r2_score(y_test, preds)
        fold_results.append({'Fold': outer_fold, 'MAE': mae, 'MSE': rmse**2, 'RMSE': rmse, 'R2': r2})

        for idx, true_val, pred_val in zip(test_idx, y_test.values, preds):
            patient_errors.append({
                'Patient':   patient_id_map[idx],
                'abs_error': abs(true_val - pred_val),
            })

        print(f"  Outer Fold {outer_fold} |  Features={len(selected_cols)}: {selected_cols}")
        print(f"    MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}")

    print(f"\n  Training time: {(time.time()-start)/60:.1f} min")

    # ── Results summary ────────────────────────────────────────────────────────
    results_df  = pd.DataFrame(fold_results)
    metric_cols = ['MAE', 'MSE', 'RMSE', 'R2']
    mean_row    = {'Fold': 'Mean', **{m: results_df[m].mean() for m in metric_cols}}
    std_row     = {'Fold': 'Std',  **{m: results_df[m].std()  for m in metric_cols}}
    results_df  = pd.concat([results_df, pd.DataFrame([mean_row, std_row])], ignore_index=True)

    n_outer = len(fold_results)
    t_crit  = stats.t.ppf(0.975, df=n_outer - 1)

    print(f"\n{'='*65}\n  SUMMARY — {target_col} \n{'='*65}")
    for m in metric_cols:
        mv, sv = mean_row[m], std_row[m]
        ci = t_crit * sv / np.sqrt(n_outer)
        print(f"    {m:<5}: {mv:.3f} ± {sv:.4f}   (95% CI [{mv-ci:.3f}, {mv+ci:.3f}])")

    # ── Final model ────────────────────────────────────────────────────────────
    X_final        = _prep_for_catboost(X[selected_cols], cat_cols)
    cat_cols_final = [c for c in cat_cols if c in selected_cols]

    if target_transformer is not None:
        pt_final    = clone(target_transformer)
        y_final_fit = pd.Series(
            pt_final.fit_transform(y.values.reshape(-1, 1)).ravel(), index=y.index)
    else:
        pt_final, y_final_fit = None, y

    hp_final = {k: (int(round(statistics.median([p[k] for p in best_model_params_list])))
                    if isinstance(best_model_params_list[0][k], int)
                    else statistics.median([p[k] for p in best_model_params_list]))
                for k in best_model_params_list[0]}
    print(f"  Final model HPs (median): {hp_final}")

    final_model = CatBoostRegressor(
        iterations=1000, loss_function='RMSE', custom_metric=['MAE', 'R2'],
        cat_features=cat_cols_final, random_seed=random_state,
        task_type='CPU', thread_count=-1, logging_level='Silent',
        **hp_final)
    with contextlib.redirect_stderr(io.StringIO()):
        final_model.fit(X_final, y_final_fit)

    y_pred_raw = pd.Series(final_model.predict(X_final), index=range(len(X_final)), dtype='float64')
    y_pred     = (pd.Series(pt_final.inverse_transform(y_pred_raw.values.reshape(-1, 1)).ravel(),
                            index=y_pred_raw.index, dtype='float64')
                  if pt_final is not None else y_pred_raw)

    return (results_df, final_model, X_final, y_pred,
            best_model_params_list)
