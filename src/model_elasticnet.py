import time
import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import joblib
import contextlib, io

import preprocess


def run_tuned_elasticnet(
    df_combined, feature_list, target_col='pain_reduction', random_state=42,
    target_transformer=None,
):
    """ElasticNet with Optuna nested CV.

      1. Inner CV (4×5=20) + Optuna (50 trials) tunes ElasticNet HPs on X_train.
      2. Train final fold model on X_train → evaluate on X_test.
      3. Final model: median HPs across outer folds, trained on full X.

    Returns: results_df, final_model, X_final, y_pred, best_model_params_list, patient_err_df
    """
    from sklearn.linear_model import ElasticNet
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
    print(f"  ElasticNet + Optuna — {target_col}")
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

        # 1. Encode — fit on X_train only, transform X_test
        if cat_cols:
            oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            X_train[cat_cols] = oe.fit_transform(X_train[cat_cols].astype(str))
            X_test[cat_cols]  = oe.transform(X_test[cat_cols].astype(str))
        X_train = X_train.astype(float)
        X_test  = X_test.astype(float)

        # 2. Impute — fit on X_train only, transform X_test
        X_train_imp, imputer = preprocess.impute_iterative(
            X_train, ex_cols=None, iterations=10,
            random_state=random_state, verbose=False)
        X_train_imp = pd.DataFrame(
            X_train_imp, columns=selected_cols, index=X_train.index)
        X_test_imp = pd.DataFrame(
            imputer.transform(X_test),
            columns=selected_cols, index=X_test.index)

        # 3. Scale — fit on X_train only, transform X_test
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train_imp),
            columns=selected_cols, index=X_train.index)
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test_imp),
            columns=selected_cols, index=X_test.index)

        inner_splits = list(inner_cv.split(X_train_scaled))

        def _fit_inner_en(itr, ival, params):
            m = ElasticNet(**params, max_iter=5000, random_state=random_state)
            m.fit(X_train_scaled.iloc[itr], y_train_fit.iloc[itr])
            return np.sqrt(mean_squared_error(
                y_train_fit.iloc[ival],
                m.predict(X_train_scaled.iloc[ival])))

        def model_objective(trial):
            params = dict(
                alpha    = trial.suggest_float('alpha',    1e-4, 10.0, log=True),
                l1_ratio = trial.suggest_float('l1_ratio', 0.0,  1.0),
            )
            rmses = joblib.Parallel(n_jobs=4, prefer='threads')(
                joblib.delayed(_fit_inner_en)(itr, ival, params)
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

        fold_model = ElasticNet(**best_model_params, max_iter=5000, random_state=random_state)
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

    # Final model — fit all transformers on full X, no test set
    X_final = X.copy()
    if cat_cols:
        oe_final = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_final[cat_cols] = oe_final.fit_transform(X_final[cat_cols].astype(str))
    X_final = X_final.astype(float)

    X_final_imp, _ = preprocess.impute_iterative(
        X_final, ex_cols=None, iterations=10,
        random_state=random_state, verbose=False)
    X_final_imp = pd.DataFrame(X_final_imp, columns=selected_cols, index=X_final.index)

    scaler_final = StandardScaler()
    X_final = pd.DataFrame(
        scaler_final.fit_transform(X_final_imp),
        columns=selected_cols, index=X_final_imp.index)

    if target_transformer is not None:
        pt_final    = clone(target_transformer)
        y_final_fit = pd.Series(
            pt_final.fit_transform(y.values.reshape(-1, 1)).ravel(), index=y.index)
    else:
        pt_final, y_final_fit = None, y

    hp_final = {
        k: statistics.median([p[k] for p in best_model_params_list])
        for k in best_model_params_list[0]}
    print(f"  Final model HPs (median): {hp_final}")

    final_model = ElasticNet(**hp_final, max_iter=5000, random_state=random_state)
    final_model.fit(X_final, y_final_fit)

    y_pred_raw = pd.Series(
        final_model.predict(X_final),
        index=range(len(X_final)), dtype='float64')
    y_pred = (pd.Series(
        pt_final.inverse_transform(y_pred_raw.values.reshape(-1, 1)).ravel(),
        index=y_pred_raw.index, dtype='float64')
              if pt_final is not None else y_pred_raw)

    return (results_df, final_model, X_final, y_pred,
            best_model_params_list, patient_err_df)
