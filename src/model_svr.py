import time
import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.svm import SVR
import joblib
import contextlib, io
import preprocess


def run_tuned_svr(
    df_combined, feature_list, target_col='pain_reduction', random_state=42,
    target_transformer=None,
):
    """SVR (with RBF kernel) + Optuna nested CV

      1. Inner CV (4×5=20) + Optuna (50 trials) tunes hyp.parameters C, epsilon, gamma
      2. Train final fold model on X_train,  evaluate on X_test
      3. Final model: median HPs across outer folds, trained on full X

    Returns: results_df, final_model, X_final, y_pred, best_model_params_list, patient_err_df
    """
    import optuna, warnings, statistics

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    N_TRIALS = 50

    y = df_combined[target_col].copy()
    valid = y.notna()

    selected_cols = [f for f in feature_list if f in df_combined.columns]
    X = df_combined[selected_cols].copy()

    X, y = X[valid].reset_index(drop=True), y[valid].reset_index(drop=True)
    patient_id_map = df_combined.loc[valid, 'Patient'].reset_index(drop=True)

    cat_cols = X.select_dtypes(include=['category', 'object']).columns.tolist()

    print(f"\n{'='*65}")
    print(f"  SVR (RBF) + Optuna — {target_col}")
    print(f"  n={len(X)}, p={len(selected_cols)}")
    print(f"  Outer 4×5=20 | Inner 4×5=20 | Optuna trials={N_TRIALS}")
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

        X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Target transform — fit on y_train only
        if target_transformer is not None:
            pt_fold = clone(target_transformer)
            y_train_fit = pd.Series(
                pt_fold.fit_transform(y_train.values.reshape(-1, 1)).ravel(),
                index=y_train.index)
        else:
            pt_fold, y_train_fit = None, y_train

        # 1. Encode 
        if cat_cols:
            oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            X_train[cat_cols] = oe.fit_transform(X_train[cat_cols].astype(str))
            X_test[cat_cols]  = oe.transform(X_test[cat_cols].astype(str))
        X_train = X_train.astype(float)
        X_test  = X_test.astype(float)

        # 2. Impute
        X_train_imp, imputer = preprocess.impute_iterative(
            X_train, ex_cols=None, iterations=10,
            random_state=random_state, verbose=False)
        X_train_imp = pd.DataFrame(
            X_train_imp, columns=selected_cols, index=X_train.index)
        X_test_imp = pd.DataFrame(
            imputer.transform(X_test),
            columns=selected_cols, index=X_test.index)

        # 3. Scale 
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train_imp),
            columns=selected_cols, index=X_train.index)
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test_imp),
            columns=selected_cols, index=X_test.index)

        inner_splits = list(inner_cv.split(X_train_scaled))

        def _fit_inner_svr(itr, ival, params):
            m = SVR(kernel='rbf', **params)
            m.fit(X_train_scaled.iloc[itr], y_train_fit.iloc[itr])
            return np.sqrt(mean_squared_error(
                y_train_fit.iloc[ival],
                m.predict(X_train_scaled.iloc[ival])))
        
        # tune models hyperparameters
        def model_objective(trial):
            params = dict(
                C       = trial.suggest_float('C',       1e-2, 1e2, log=True),
                epsilon = trial.suggest_float('epsilon', 1e-3, 1.0, log=True),
                gamma   = trial.suggest_float('gamma',   1e-3, 1e1, log=True),
            )
            rmses = joblib.Parallel(n_jobs=4, prefer='threads')(
                joblib.delayed(_fit_inner_svr)(itr, ival, params)
                for itr, ival in inner_splits)
            return np.mean(rmses)

        def _cb(study, trial):
            if trial.state.name == 'COMPLETE':
                print(f"    Trial {trial.number+1:>3}/{N_TRIALS}: "
                      f"RMSE={trial.value:.4f}  {trial.params}")

        model_study = optuna.create_study(direction='minimize')
        with contextlib.redirect_stderr(io.StringIO()):
            model_study.optimize(model_objective, n_trials=N_TRIALS,
                                 callbacks=[_cb], show_progress_bar=False)

        best_model_params = model_study.best_params
        best_model_params_list.append(best_model_params)
        print(f"  Best Trial: {model_study.best_trial.number}  "
              f"RMSE={model_study.best_value:.4f}  {best_model_params}")

        fold_model = SVR(kernel='rbf', **best_model_params)
        fold_model.fit(X_train_scaled, y_train_fit)

        preds_raw = fold_model.predict(X_test_scaled)
        preds = (pt_fold.inverse_transform(preds_raw.reshape(-1, 1)).ravel()
                 if pt_fold is not None else preds_raw)

        mae  = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2   = r2_score(y_test, preds)
        fold_results.append({
            'Fold': outer_fold, 'MAE': mae, 'MSE': rmse**2, 'RMSE': rmse, 'R2': r2})

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
    results_df  = pd.concat(
        [results_df, pd.DataFrame([mean_row, std_row])], ignore_index=True)

    n_outer = len(fold_results)
    t_crit  = stats.t.ppf(0.975, df=n_outer - 1)

    print(f"\n{'='*65}\n  SUMMARY — {target_col}\n{'='*65}")
    for m in metric_cols:
        mv, sv = mean_row[m], std_row[m]
        ci = t_crit * sv / np.sqrt(n_outer)
        print(f"    {m:<5}: {mv:.3f} ± {sv:.4f}   (95% CI [{mv-ci:.3f}, {mv+ci:.3f}])")

    # Final mode
    X_final = X.copy()
    if cat_cols:
        # encode
        oe_final = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_final[cat_cols] = oe_final.fit_transform(X_final[cat_cols].astype(str))
    X_final = X_final.astype(float)

    # impute
    X_final_imp, _ = preprocess.impute_iterative(
        X_final, ex_cols=None, iterations=10,
        random_state=random_state, verbose=False)
    X_final_imp = pd.DataFrame(X_final_imp, columns=selected_cols, index=X_final.index)

    # scale
    scaler_final = StandardScaler()
    X_final = pd.DataFrame(
        scaler_final.fit_transform(X_final_imp),
        columns=selected_cols, index=X_final_imp.index)
    
    # reverse transform predictions to original space
    if target_transformer is not None:
        pt_final    = clone(target_transformer)
        y_final_fit = pd.Series(
            pt_final.fit_transform(y.values.reshape(-1, 1)).ravel(), index=y.index)
    else:
        pt_final, y_final_fit = None, y

    hp_final = {
        k: statistics.median([p[k] for p in best_model_params_list])
        for k in best_model_params_list[0]}
    print(f"  Final model HPs: {hp_final}")

    final_model = SVR(kernel='rbf', **hp_final)
    final_model.fit(X_final, y_final_fit)

    y_pred_raw = pd.Series(
        final_model.predict(X_final),
        index=range(len(X_final)), dtype='float64')
    y_pred = (pd.Series(
        pt_final.inverse_transform(y_pred_raw.values.reshape(-1, 1)).ravel(),
        index=y_pred_raw.index, dtype='float64')
              if pt_final is not None else y_pred_raw)

    return (results_df, final_model, X_final, y_pred,
          patient_err_df, scaler_final)



def run_tuned_svr_mrmr(
    df_combined, K=11, target_col='pain_reduction', random_state=42,
    target_transformer=None,
):
    """SVR (with RBF kernel) + MRMR feature selection (K features) inside each outer CV fold.

      1. MRMR on X_train → select K features per outer fold
      2. Inner CV (4×5=20) + Optuna (50 trials) tunes hyp.parameters C, epsilon, gamma
      3. Train final fold model on X_train, evaluate on X_test
      4. Final model: top K most frequently selected features, median HPs across outer folds

    Returns: results_df, final_model, X_final, y_pred, best_model_params_list,
             feature_freq, patient_err_df, scaler_final
    """
    import optuna, warnings, statistics
    from feature_engine.selection import MRMR
    from collections import Counter
    import feature_selection

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    N_TRIALS = 50

    y = df_combined[target_col].copy()
    valid = y.notna()

    exclude = {'Patient', 'Timepoint', target_col, 'pain_reduction',
               'pain_reduction_pct', 'pain_under_load_reduction',
               'pain_under_load_reduction_pct'}
    feature_cols = [c for c in df_combined.columns if c not in exclude]
    X = df_combined[feature_cols].copy()

    X, y = X[valid].reset_index(drop=True), y[valid].reset_index(drop=True)
    patient_id_map = df_combined.loc[valid, 'Patient'].reset_index(drop=True)

    cat_cols = X.select_dtypes(include=['category', 'object']).columns.tolist()

    print(f"\n{'='*65}")
    print(f"  SVR (RBF) + MRMR (K={K}) + Optuna — {target_col}")
    print(f"  n={len(X)}, p={len(feature_cols)}")
    print(f"  Outer 4×5=20 | Inner 4×5=20 | Optuna trials={N_TRIALS}")
    print(f"{'='*65}")

    outer_cv = RepeatedKFold(n_splits=4, n_repeats=5, random_state=random_state)
    inner_cv = RepeatedKFold(n_splits=4, n_repeats=5, random_state=random_state)

    fold_results               = []
    best_model_params_list     = []
    selected_features_per_fold = []  # ← NEW
    patient_errors             = []
    start = time.time()

    for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X), start=1):
        print(f"\n{'─'*65}")
        print(f"  Outer fold {outer_fold}/{outer_cv.get_n_splits()}")
        print(f"{'─'*65}")

        X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Target transform — fit on y_train only
        if target_transformer is not None:
            pt_fold = clone(target_transformer)
            y_train_fit = pd.Series(
                pt_fold.fit_transform(y_train.values.reshape(-1, 1)).ravel(),
                index=y_train.index)
        else:
            pt_fold, y_train_fit = None, y_train

        # ── MRMR feature selection — fitted on X_train only ──────────────────
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

        # 1. Encode — fitted on X_train only, applied to X_test
        X_train_sel = X_train[selected_cols].copy()
        X_test_sel  = X_test[selected_cols].copy()

        cats_sel = [c for c in cat_cols if c in selected_cols]
        if cats_sel:
            oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            X_train_sel[cats_sel] = oe.fit_transform(X_train_sel[cats_sel].astype(str))
            X_test_sel[cats_sel]  = oe.transform(X_test_sel[cats_sel].astype(str))

        X_train_sel = X_train_sel.astype(float)
        X_test_sel  = X_test_sel.astype(float)

        # 2. Impute
        X_train_imp, imputer = preprocess.impute_iterative(
            X_train_sel, ex_cols=None, iterations=10,
            random_state=random_state, verbose=False)
        X_train_imp = pd.DataFrame(
            X_train_imp, columns=selected_cols, index=X_train_sel.index)
        X_test_imp = pd.DataFrame(
            imputer.transform(X_test_sel),
            columns=selected_cols, index=X_test_sel.index)

        # 3. Scale
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train_imp),
            columns=selected_cols, index=X_train_sel.index)
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test_imp),
            columns=selected_cols, index=X_test_sel.index)

        inner_splits = list(inner_cv.split(X_train_scaled))

        def _fit_inner_svr(itr, ival, params):
            m = SVR(kernel='rbf', **params)
            m.fit(X_train_scaled.iloc[itr], y_train_fit.iloc[itr])
            return np.sqrt(mean_squared_error(
                y_train_fit.iloc[ival],
                m.predict(X_train_scaled.iloc[ival])))

        # tune models hyperparameters
        def model_objective(trial):
            params = dict(
                C       = trial.suggest_float('C',       1e-2, 1e2, log=True),
                epsilon = trial.suggest_float('epsilon', 1e-3, 1.0, log=True),
                gamma   = trial.suggest_float('gamma',   1e-3, 1e1, log=True),
            )
            rmses = joblib.Parallel(n_jobs=4, prefer='threads')(
                joblib.delayed(_fit_inner_svr)(itr, ival, params)
                for itr, ival in inner_splits)
            return np.mean(rmses)

        def _cb(study, trial):
            if trial.state.name == 'COMPLETE':
                print(f"    Trial {trial.number+1:>3}/{N_TRIALS}: "
                      f"RMSE={trial.value:.4f}  {trial.params}")

        model_study = optuna.create_study(direction='minimize')
        with contextlib.redirect_stderr(io.StringIO()):
            model_study.optimize(model_objective, n_trials=N_TRIALS,
                                 callbacks=[_cb], show_progress_bar=False)

        best_model_params = model_study.best_params
        best_model_params_list.append(best_model_params)
        print(f"  Best Trial: {model_study.best_trial.number}  "
              f"RMSE={model_study.best_value:.4f}  {best_model_params}")

        fold_model = SVR(kernel='rbf', **best_model_params)
        fold_model.fit(X_train_scaled, y_train_fit)

        preds_raw = fold_model.predict(X_test_scaled)
        preds = (pt_fold.inverse_transform(preds_raw.reshape(-1, 1)).ravel()
                 if pt_fold is not None else preds_raw)

        mae  = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2   = r2_score(y_test, preds)
        fold_results.append({
            'Fold': outer_fold, 'MAE': mae, 'MSE': rmse**2, 'RMSE': rmse, 'R2': r2})

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
    results_df  = pd.concat(
        [results_df, pd.DataFrame([mean_row, std_row])], ignore_index=True)

    n_outer = len(fold_results)
    t_crit  = stats.t.ppf(0.975, df=n_outer - 1)

    print(f"\n{'='*65}\n  SUMMARY — {target_col}\n{'='*65}")
    for m in metric_cols:
        mv, sv = mean_row[m], std_row[m]
        ci = t_crit * sv / np.sqrt(n_outer)
        print(f"    {m:<5}: {mv:.3f} ± {sv:.4f}   (95% CI [{mv-ci:.3f}, {mv+ci:.3f}])")

    # ── Feature frequency ─────────────────────────────────────────────────────
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

    # ── Final model ───────────────────────────────────────────────────────────
    # Top K=11 most frequently selected features across all outer folds
    final_cols = [f for f, _ in freq.most_common(K)]
    print(f"\n  Final model: {len(final_cols)} features (top {K} by frequency): {final_cols}")

    X_final = X[final_cols].copy()

    # encode
    cats_final = [c for c in cat_cols if c in final_cols]
    if cats_final:
        oe_final = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_final[cats_final] = oe_final.fit_transform(X_final[cats_final].astype(str))
    X_final = X_final.astype(float)

    # impute
    X_final_imp, _ = preprocess.impute_iterative(
        X_final, ex_cols=None, iterations=10,
        random_state=random_state, verbose=False)
    X_final_imp = pd.DataFrame(X_final_imp, columns=final_cols, index=X_final.index)

    # scale
    scaler_final = StandardScaler()
    X_final = pd.DataFrame(
        scaler_final.fit_transform(X_final_imp),
        columns=final_cols, index=X_final_imp.index)

    # reverse transform predictions to original space
    if target_transformer is not None:
        pt_final    = clone(target_transformer)
        y_final_fit = pd.Series(
            pt_final.fit_transform(y.values.reshape(-1, 1)).ravel(), index=y.index)
    else:
        pt_final, y_final_fit = None, y

    hp_final = {
        k: statistics.median([p[k] for p in best_model_params_list])
        for k in best_model_params_list[0]}
    print(f"  Final model HPs: {hp_final}")

    final_model = SVR(kernel='rbf', **hp_final)
    final_model.fit(X_final, y_final_fit)

    y_pred_raw = pd.Series(
        final_model.predict(X_final),
        index=range(len(X_final)), dtype='float64')
    y_pred = (pd.Series(
        pt_final.inverse_transform(y_pred_raw.values.reshape(-1, 1)).ravel(),
        index=y_pred_raw.index, dtype='float64')
              if pt_final is not None else y_pred_raw)

    return (results_df, final_model, X_final, y_pred,
            best_model_params_list, feature_freq, patient_err_df, scaler_final)