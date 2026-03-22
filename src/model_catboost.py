# CatBoost model with Optuna Hyperparameter tuning
import time
import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
from sklearn.model_selection import RepeatedKFold
import joblib
import contextlib, io
import feature_selection
import preprocess


def _prep_for_catboost(df, cat_cols):
    """Convert category columns to object dtype, filling NaN with 'missing'."""
    out = df.copy()
    for col in cat_cols:
        if col in out.columns:
            out[col] = out[col].astype(object).fillna('missing')
    return out


def run_tuned_catboost(
    df, feature_list, target_col='pain_reduction_pct', random_state=42,
    target_transformer=None,
):
    """CatBoostRegressor with Optuna nested CV.

      1. Inner CV (4×5=20) + Optuna (50 trials) tunes CatBoost HPs on X_train.
      2. Train final fold model on X_train → evaluate on X_test (no imputation).
      3. Final model: median HPs across outer folds, trained on full X.

    Returns: results_df, final_model, X_final, y_pred, best_model_params_list, patient_err_df
    """
    from catboost import CatBoostRegressor
    import optuna, warnings, statistics

    for pat in ['.*less than 75% GPU memory.*', '.*joblib.*', '.*loky.*']:
        warnings.filterwarnings('ignore', message=pat)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    N_TRIALS_MODEL = 50

    y = df[target_col].copy()
    valid = y.notna()

    selected_cols = [f for f in feature_list if f in df.columns]
    X = df[selected_cols].copy()
    cat_cols = X.select_dtypes(include=['category', 'object']).columns.tolist()

    X, y = X[valid].reset_index(drop=True), y[valid].reset_index(drop=True)
    patient_id_map = df.loc[valid, 'Patient'].reset_index(drop=True)

    print(f"\n{'='*65}")
    print(f"  CatBoost + Optuna — {target_col}")
    print(f"  n={len(X)}, p={len(selected_cols)}")
    print(f"  Outer 4×5=20   Inner 4×5=20   Model trials={N_TRIALS_MODEL}")
    print(f"{'='*65}")

    outer_cv = RepeatedKFold(n_splits=4, n_repeats=5, random_state=random_state)
    inner_cv = RepeatedKFold(n_splits=4, n_repeats=5, random_state=random_state)

    fold_results = []
    best_model_params_list = []
    patient_errors = []
    start = time.time()

    for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X), start=1):
        print(f"\n{'─'*65}")
        print(f"  Outer fold {outer_fold}/{outer_cv.get_n_splits()}")
        print(f"{'─'*65}")

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        if target_transformer is not None:
            pt_fold     = clone(target_transformer)
            y_train_fit = pd.Series(
                pt_fold.fit_transform(y_train.values.reshape(-1, 1)).ravel(),
                index=y_train.index)
        else:
            pt_fold, y_train_fit = None, y_train

        X_train_cb = _prep_for_catboost(X_train, cat_cols)
        inner_splits = list(inner_cv.split(X_train_cb))

        def _fit_inner(itr, ival, params):
            m = CatBoostRegressor(
                iterations=500, **params, cat_features=cat_cols, loss_function='RMSE',
                random_seed=random_state, task_type='CPU', thread_count=1,
                logging_level='Silent')
            with contextlib.redirect_stderr(io.StringIO()):
                m.fit(X_train_cb.iloc[itr], y_train_fit.iloc[itr])
            return np.sqrt(mean_squared_error(
                y_train_fit.iloc[ival],
                m.predict(X_train_cb.iloc[ival])))

        def model_objective(trial):
            params = dict(
                depth               = trial.suggest_int(  'depth',               3,   8),
                learning_rate       = trial.suggest_float('learning_rate',       1e-3, 0.3,  log=True),
                l2_leaf_reg         = trial.suggest_float('l2_leaf_reg',         1.0,  10.0, log=True),
                bagging_temperature = trial.suggest_float('bagging_temperature', 0.0,  1.0),
            )
            rmses = joblib.Parallel(n_jobs=4, prefer='threads')(
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

        X_test_cb = _prep_for_catboost(X_test, cat_cols)

        fold_model = CatBoostRegressor(
            iterations=1000, **best_model_params, cat_features=cat_cols,
            loss_function='RMSE', random_seed=random_state,
            task_type='CPU', thread_count=-1, logging_level='Silent')
        with contextlib.redirect_stderr(io.StringIO()):
            fold_model.fit(X_train_cb, y_train_fit)

        preds_raw = fold_model.predict(X_test_cb)
        preds = (pt_fold.inverse_transform(preds_raw.reshape(-1, 1)).ravel()
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

        print(f"  Outer Fold {outer_fold} | Features={len(selected_cols)}: {selected_cols}")
        print(f"    MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}")

    print(f"\n  Training time: {(time.time()-start)/60:.1f} min")

    patient_err_df = (pd.DataFrame(patient_errors)
                      .groupby('Patient')['abs_error']
                      .agg(mean_mae='mean', n_folds='count')
                      .sort_values('mean_mae', ascending=False))

    results_df  = pd.DataFrame(fold_results)
    metric_cols = ['MAE', 'MSE', 'RMSE', 'R2']
    mean_row    = {'Fold': 'Mean', **{m: results_df[m].mean() for m in metric_cols}}
    std_row     = {'Fold': 'Std',  **{m: results_df[m].std()  for m in metric_cols}}
    results_df  = pd.concat([results_df, pd.DataFrame([mean_row, std_row])], ignore_index=True)

    n_outer = len(fold_results)
    t_crit  = stats.t.ppf(0.975, df=n_outer - 1)

    print(f"\n{'='*65}\n  SUMMARY — {target_col}\n{'='*65}")
    for m in metric_cols:
        mv, sv = mean_row[m], std_row[m]
        ci = t_crit * sv / np.sqrt(n_outer)
        print(f"    {m:<5}: {mv:.3f} ± {sv:.4f}   (95% CI [{mv-ci:.3f}, {mv+ci:.3f}])")

    X_final = _prep_for_catboost(X, cat_cols)

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
        cat_features=cat_cols, random_seed=random_state,
        task_type='CPU', thread_count=-1, logging_level='Silent',
        **hp_final)
    with contextlib.redirect_stderr(io.StringIO()):
        final_model.fit(X_final, y_final_fit)

    y_pred_raw = pd.Series(final_model.predict(X_final), index=range(len(X_final)), dtype='float64')
    y_pred = (pd.Series(pt_final.inverse_transform(y_pred_raw.values.reshape(-1, 1)).ravel(),
                        index=y_pred_raw.index, dtype='float64')
              if pt_final is not None else y_pred_raw)

    return (results_df, final_model, X_final, y_pred,
            patient_err_df)



def run_tuned_catboost_mrmr(
    df, K=11, target_col='pain_reduction_pct', random_state=42,
    target_transformer=None,
):
    """CatBoostRegressor with MRMR feature selection (K features) inside each outer CV fold.

      1. MRMR on X_train, select K features per outer fold
      2. Inner CV (4×5=20) + Optuna (50 trials) tunes CatBoost hyperparams on X_train
      3. Train final fold model on X_train,  evaluate on X_test
      4. Final model: top K most frequently selected features, median hyper params across outer folds

    Returns: results_df, final_model, X_final, y_pred,
            feature_freq, patient_err_df
    """
    from catboost import CatBoostRegressor
    from feature_engine.selection import MRMR
    from collections import Counter
    import optuna, warnings, statistics
    import feature_selection

    for pat in ['.*less than 75% GPU memory.*', '.*joblib.*', '.*loky.*']:
        warnings.filterwarnings('ignore', message=pat)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    N_TRIALS_MODEL = 50

    y = df[target_col].copy()
    valid = y.notna()

    exclude = {'Patient', 'Timepoint', target_col, 'pain_reduction',
               'pain_reduction_pct', 'pain_under_load_reduction',
               'pain_under_load_reduction_pct'}
    feature_cols = [c for c in df.columns if c not in exclude]
    X = df[feature_cols].copy()
    cat_cols = X.select_dtypes(include=['category', 'object']).columns.tolist()

    X, y = X[valid].reset_index(drop=True), y[valid].reset_index(drop=True)
    patient_id_map = df.loc[valid, 'Patient'].reset_index(drop=True)

    print(f"\n{'='*65}")
    print(f"  CatBoost + MRMR (K={K}) + Optuna — {target_col}")
    print(f"  n={len(X)}, p={len(feature_cols)}")
    print(f"  Outer 4×5=20   Inner 4×5=20   Model trials={N_TRIALS_MODEL}")
    print(f"{'='*65}")

    outer_cv = RepeatedKFold(n_splits=4, n_repeats=5, random_state=random_state)
    inner_cv = RepeatedKFold(n_splits=4, n_repeats=5, random_state=random_state)

    fold_results               = []
    best_model_params_list     = []
    selected_features_per_fold = []  
    patient_errors             = []
    start = time.time()

    for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X), start=1):
        print(f"\n{'─'*65}")
        print(f"  Outer fold {outer_fold}/{outer_cv.get_n_splits()}")
        print(f"{'─'*65}")

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        if target_transformer is not None:
            pt_fold     = clone(target_transformer)
            y_train_fit = pd.Series(
                pt_fold.fit_transform(y_train.values.reshape(-1, 1)).ravel(),
                index=y_train.index)
        else:
            pt_fold, y_train_fit = None, y_train

        #  MRMR feature selection (fitted on X_train only)
        X_train_mrmr = feature_selection.prep_for_mrmr(X_train, cat_cols, random_state)

        mrmr_sel = MRMR(
            method='RFCQ',
            max_features=K,
            scoring='neg_mean_squared_error',
            param_grid={'n_estimators':    [50, 100, 200, 300, 400, 500],
                        'max_depth':        [2, 3, 4, 5, 6, 7],
                        'min_samples_leaf': [3, 5, 8]},
            cv=5, regression=True,
            random_state=random_state, n_jobs=-1)
        mrmr_sel.fit(X_train_mrmr, y_train_fit)
        selected_cols = list(mrmr_sel.transform(X_train_mrmr).columns)
        selected_features_per_fold.append(selected_cols)
        print(f"  {len(selected_cols)} Selected features: {selected_cols}")
        # ─────────────────────────────────────────────────────────────────────

        cat_cols_sel = [c for c in cat_cols if c in selected_cols]
        X_train_cb   = _prep_for_catboost(X_train[selected_cols], cat_cols_sel)
        inner_splits  = list(inner_cv.split(X_train_cb))

        def _fit_inner(itr, ival, params):
            m = CatBoostRegressor(
                iterations=500, **params, cat_features=cat_cols_sel, loss_function='RMSE',
                random_seed=random_state, task_type='CPU', thread_count=1,
                logging_level='Silent')
            with contextlib.redirect_stderr(io.StringIO()):
                m.fit(X_train_cb.iloc[itr], y_train_fit.iloc[itr])
            return np.sqrt(mean_squared_error(
                y_train_fit.iloc[ival],
                m.predict(X_train_cb.iloc[ival])))

        def model_objective(trial):
            params = dict(
                depth               = trial.suggest_int(  'depth',               3,   8),
                learning_rate       = trial.suggest_float('learning_rate',       1e-3, 0.3,  log=True),
                l2_leaf_reg         = trial.suggest_float('l2_leaf_reg',         1.0,  10.0, log=True),
                bagging_temperature = trial.suggest_float('bagging_temperature', 0.0,  1.0),
            )
            rmses = joblib.Parallel(n_jobs=4, prefer='threads')(
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

        X_test_cb = _prep_for_catboost(X_test[selected_cols], cat_cols_sel)

        fold_model = CatBoostRegressor(
            iterations=1000, **best_model_params, cat_features=cat_cols_sel,
            loss_function='RMSE', random_seed=random_state,
            task_type='CPU', thread_count=-1, logging_level='Silent')
        with contextlib.redirect_stderr(io.StringIO()):
            fold_model.fit(X_train_cb, y_train_fit)

        preds_raw = fold_model.predict(X_test_cb)
        preds = (pt_fold.inverse_transform(preds_raw.reshape(-1, 1)).ravel()
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

        print(f"  Outer Fold {outer_fold}   Features={len(selected_cols)}: {selected_cols}")
        print(f"    MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}")

    print(f"\n  Training time: {(time.time()-start)/60:.1f} min")

    patient_err_df = (pd.DataFrame(patient_errors)
                      .groupby('Patient')['abs_error']
                      .agg(mean_mae='mean', n_folds='count')
                      .sort_values('mean_mae', ascending=False))

    results_df  = pd.DataFrame(fold_results)
    metric_cols = ['MAE', 'MSE', 'RMSE', 'R2']
    mean_row    = {'Fold': 'Mean', **{m: results_df[m].mean() for m in metric_cols}}
    std_row     = {'Fold': 'Std',  **{m: results_df[m].std()  for m in metric_cols}}
    results_df  = pd.concat([results_df, pd.DataFrame([mean_row, std_row])], ignore_index=True)

    n_outer = len(fold_results)
    t_crit  = stats.t.ppf(0.975, df=n_outer - 1)

    print(f"\n{'='*65}\n  SUMMARY — {target_col}\n{'='*65}")
    for m in metric_cols:
        mv, sv = mean_row[m], std_row[m]
        ci = t_crit * sv / np.sqrt(n_outer)
        print(f"    {m:<5}: {mv:.3f} ± {sv:.4f}   (95% CI [{mv-ci:.3f}, {mv+ci:.3f}])")

    # Feature frequency list
    freq = Counter(f for fold in selected_features_per_fold for f in fold)
    feature_freq = (
        pd.Series(dict(freq), name='selection_count')
        .reindex(feature_cols, fill_value=0)
        .sort_values(ascending=False))
    feature_freq.index.name = 'feature'

    print(f"\n  Selected Features Across {n_outer} Outer Folds:")
    for feat, cnt in feature_freq[feature_freq > 0].items():
        marker = '  (top K)' if feat in [f for f, _ in freq.most_common(K)] else ''
        print(f"    {feat:<45} {cnt:>2}/{n_outer}{marker}")

    #  Final model
    # use top K=11 most frequently selected features across all outer folds
    final_cols   = [f for f, _ in freq.most_common(K)]
    cat_cols_final = [c for c in cat_cols if c in final_cols]
    print(f"\n  Final model: {len(final_cols)} features (top {K} by frequency): {final_cols}")

    X_final = _prep_for_catboost(X[final_cols], cat_cols_final)

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
    print(f"  Final model hyperparameters: {hp_final}")

    final_model = CatBoostRegressor(
        iterations=1000, loss_function='RMSE', custom_metric=['MAE', 'R2'],
        cat_features=cat_cols_final, random_seed=random_state,
        task_type='CPU', thread_count=-1, logging_level='Silent',
        **hp_final)
    with contextlib.redirect_stderr(io.StringIO()):
        final_model.fit(X_final, y_final_fit)

    y_pred_raw = pd.Series(final_model.predict(X_final), index=range(len(X_final)), dtype='float64')
    y_pred = (pd.Series(pt_final.inverse_transform(y_pred_raw.values.reshape(-1, 1)).ravel(),
                        index=y_pred_raw.index, dtype='float64')
              if pt_final is not None else y_pred_raw)

    return (results_df, final_model, X_final, y_pred,
            feature_freq, patient_err_df)