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


""" This file contains all functions used for Elasticnet Modeling.
1. elasticnet_mrmr 
First, ElasticNet is ran with MRMR feature selection in each outer fold
All model and MRMR hyperparameters are tuned with Optuna.
A feature frequency list is returned, showing selected features across all outer folds,

2. elasticnet_threshold_analysis
The returned feature frequency list is then used to run a feature-threshold analysis.
Up to 11 models are run based on different subsets of features: all features, features selected
in >= 10% of outer folds, and up to 9 further thresholds sampled from the unique frequencies
present in the data (data-driven, not fixed at 10% increments).
Performance metrics are stored across all feature subsets, and used for plotting.

3. run_tuned_elasticnet
After selecting a cut-off, the remaining features from the feature frequency list is used as
input data for the final model.
Model hyperparameters are tuned with optuna, and performance metrics are stored.
Final model for shap analysis used the selected features, and the median of 
the best model hyperparameters across all outer folds. This model is later used for SHAP analysis.

"""


def prep_for_mrmr(X_train, cat_cols, random_state=42):
    """OrdinalEncode + iterativeImpute for MRMR."""
    out = X_train.copy()

    # Encode, iterative imputer needs numeric imput only
    cats = [c for c in cat_cols if c in out.columns]
    if cats:
        oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        out[cats] = oe.fit_transform(out[cats].astype(str))
    # impute
    out, _ = preprocess.impute_iterative(
        out.astype(float), ex_cols=None, iterations=10,
        random_state=random_state, verbose=False)
    return out


def elasticnet_mrmr(
    df_combined, target_col='pain_reduction', random_state=42,
    target_transformer=None,
):
    """ElasticNet with MRMR (K + RFCQ params tuned by Optuna) inside each outer CV fold.
      1. Tune K and RFCQ params (n_estimators, max_depth, min_samples_leaf) via Optuna with 20 trials
         on a 75-25 split of X_train. K candidates: all features 40, 30, 20, 10 features.
      2. Re-run MRMR on full X_train with best K + best RFCQ params -> selected feature subset for an outer fold
      3. Inner CV (4×5=20) + Optuna (50 trials) tunes ElasticNet hyperparameters
      Returns results dataframe with metrics, and a feature frequency list

    Returns: results_df, feature_freq,
    """
    from sklearn.linear_model import ElasticNet
    from feature_engine.selection import MRMR
    from sklearn.model_selection import train_test_split
    from collections import Counter
    import optuna, warnings, statistics

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=RuntimeWarning)

    N_TRIALS_MRMR  = 20
    N_TRIALS_MODEL = 50

    y = df_combined[target_col].copy()
    valid = y.notna()

    exclude = {'Patient', 'Timepoint', target_col, 'pain_reduction',
               'pain_reduction_pct', 'pain_under_load_reduction',
               'pain_under_load_reduction_pct'}
    feature_cols = [c for c in df_combined.columns if c not in exclude]
    X = df_combined[feature_cols].copy()

    X, y = X[valid].reset_index(drop=True), y[valid].reset_index(drop=True)
    cat_cols = X.select_dtypes(include=['category', 'object']).columns.tolist()
    
    # Sugesstions for num. selected features : from all features to selecting 10 features
    # only include values smaller than p
    p = len(feature_cols)
    print(f"\n{'='*65}")
    print(f"  ElasticNet + MRMR (K tuned by Optuna) — {target_col}")
    print(f"  n={len(X)}, p={p}")
    print(f"  Outer 4×5=20 | Inner 4×5=20 | Model trials={N_TRIALS_MODEL}")
    print(f"{'='*65}")

    outer_cv = RepeatedKFold(n_splits=4, n_repeats=5, random_state=random_state)
    inner_cv = RepeatedKFold(n_splits=4, n_repeats=5, random_state=random_state)

    fold_results               = []
    best_model_params_list     = []
    best_mrmr_params_per_fold  = []
    selected_features_per_fold = []
    start = time.time()

    for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X), start=1):
        print(f"\n{'─'*65}")
        print(f"  Outer fold {outer_fold}/{outer_cv.get_n_splits()}")
        print(f"{'─'*65}")

        X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Target transform, fit on y_train only
        if target_transformer is not None:
            pt_fold = clone(target_transformer)
            y_train_fit = pd.Series(
                pt_fold.fit_transform(y_train.values.reshape(-1, 1)).ravel(),
                index=y_train.index)
        else:
            pt_fold, y_train_fit = None, y_train

        # ── Step 1: Tune mrmr parameters on 75-25 split of X_train ─────────────────────────
        X_train_mrmr = prep_for_mrmr(X_train, cat_cols, random_state)

        X_tr_mrmr, X_val_mrmr, y_tr, y_val = train_test_split(
            X_train_mrmr, y_train_fit, test_size=0.25, random_state=random_state)

        def mrmr_objective(trial):
            k                = trial.suggest_categorical('K',                [40, 30, 20, 10])
            n_estimators     = trial.suggest_categorical('n_estimators',     [50, 100, 200, 300])
            max_depth        = trial.suggest_categorical('max_depth',        [2, 4, 6, 8])
            min_samples_leaf = trial.suggest_categorical('min_samples_leaf', [3, 5, 8])

            mrmr_t = MRMR(
                method='RFCQ',
                max_features=k,
                scoring='neg_mean_squared_error',
                param_grid={'n_estimators':    [n_estimators],
                            'max_depth':        [max_depth],
                            'min_samples_leaf': [min_samples_leaf]},
                cv=5, regression=True,
                random_state=random_state, n_jobs=-1)
            
            #with contextlib.redirect_stderr(io.StringIO()):
            mrmr_t.fit(X_tr_mrmr, y_tr)
            sel_cols = list(mrmr_t.transform(X_tr_mrmr).columns)

            if len(sel_cols) == 0:
                return 1e6

            probe = ElasticNet(max_iter=5000, random_state=random_state)
            probe.fit(X_tr_mrmr[sel_cols], y_tr)
            return np.sqrt(mean_squared_error(y_val, probe.predict(X_val_mrmr[sel_cols])))

        mrmr_study = optuna.create_study(direction='minimize')
        #with contextlib.redirect_stderr(io.StringIO()):
        mrmr_study.optimize(
                mrmr_objective, n_trials=N_TRIALS_MRMR,
                show_progress_bar=False)

        best_mrmr = mrmr_study.best_params
        best_k    = best_mrmr['K']
        best_mrmr_params_per_fold.append(best_mrmr)
        print(f"  Best MRMR params: {best_mrmr}  RMSE={mrmr_study.best_value:.4f}")

        # ── Step 2: Re-run MRMR on full X_train with best K + best RFCQ params ─
        mrmr_full = MRMR(
            method='RFCQ',
            max_features=best_k,
            scoring='neg_mean_squared_error',
            param_grid={'n_estimators':    [best_mrmr['n_estimators']],
                        'max_depth':        [best_mrmr['max_depth']],
                        'min_samples_leaf': [best_mrmr['min_samples_leaf']]},
            cv=5, regression=True,
            random_state=random_state, n_jobs=-1)
        
        mrmr_full.fit(X_train_mrmr, y_train_fit)
        selected_cols = list(mrmr_full.transform(X_train_mrmr).columns)
        selected_features_per_fold.append(selected_cols)
        print(f"  {len(selected_cols)} selected features: {selected_cols}")
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

        # ── Step 3: Inner CV Optuna for ElasticNet HPs ───────────────────────
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
                print(f"    Trial {trial.number+1:>3}/{N_TRIALS_MODEL}: "
                      f"RMSE={trial.value:.4f}  {trial.params}")

        model_study = optuna.create_study(direction='minimize')
        with contextlib.redirect_stderr(io.StringIO()):
            model_study.optimize(model_objective, n_trials=N_TRIALS_MODEL,
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
        print(f"  Outer Fold {outer_fold} | K={best_k} | Features={len(selected_cols)}: {selected_cols}")
        print(f"    MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}")

    print(f"\n  Training time: {(time.time()-start)/60:.1f} min")

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
        print(f"    {m:<5}: {mv:.3f} ± {sv:.4f})")

    print(f"\n  Per-fold parameters:")
    print(f"  {'Fold':>4}  {'K':>4}  {'n_est':>5}  {'depth':>5}  {'leaf':>4}  "
          f"{'alpha':>9}  {'l1_ratio':>8}")
    print(f"  {'─'*55}")
    for i, (mrmr_p, model_p) in enumerate(
            zip(best_mrmr_params_per_fold, best_model_params_list), start=1):
        print(f"  {i:>4}  {mrmr_p['K']:>4}  {mrmr_p['n_estimators']:>5}  "
              f"{mrmr_p['max_depth']:>5}  {mrmr_p['min_samples_leaf']:>4}  "
              f"{model_p['alpha']:>9.4f}  {model_p['l1_ratio']:>8.4f}")

    # ── Feature frequency ─────────────────────────────────────────────────────
    freq = Counter(f for fold in selected_features_per_fold for f in fold)
    feature_freq = (
        pd.Series(dict(freq), name='selection_count')
        .reindex(feature_cols, fill_value=0)
        .sort_values(ascending=False))
    feature_freq.index.name = 'feature'

    print(f"\n  Complete Feature selection frequency (out of {n_outer} outer folds):")
    for feat, cnt in feature_freq.items():
            print(f"    {cnt:>2}/{n_outer}  ({cnt/n_outer*100:5.1f}%)  {feat}")

    return results_df, feature_freq


#_________________________________________________________________________________________________
# Feature-Treshold Analysis
#_________________________________________________________________________________________________

def elasticnet_threshold_analysis(
    df_combined, feature_freq, target_col='pain_reduction',
    random_state=42, target_transformer=None):
    """ElasticNet + Optuna nested CV across feature-frequency threshold subsets.

    Evaluates 11 subsets (all features → most-frequent features) with outer 4×5=20 CV
    and inner 4×5=20 CV + Optuna (50 trials). Use the returned sweep_df to plot and
    choose a feature threshold, then pass the chosen feature list to run_tuned_elasticnet.

    Returns
    -------
    sweep_df : pd.DataFrame  — columns: threshold, threshold_label, n_features,
               mean_MAE, std_MAE, mean_RMSE, std_RMSE, mean_R2, std_R2
    """
    from sklearn.linear_model import ElasticNet
    import optuna, warnings

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=RuntimeWarning)

    N_TRIALS = 50

    # Convert selection counts to frequencies, build threshold grid
    freq_norm    = feature_freq / 20
    unique_freqs = sorted(freq_norm[freq_norm >= 0.10].unique())
    indices      = np.linspace(0, len(unique_freqs) - 1, 9, dtype=int)
    THRESHOLDS   = sorted(set(
        [0.0, 0.10] + [unique_freqs[i] for i in indices if unique_freqs[i] > 0.10]
    ))

    y = df_combined[target_col].copy()

    exclude = {'Patient', 'Timepoint', target_col, 'pain_reduction',
               'pain_reduction_pct', 'pain_under_load_reduction',
               'pain_under_load_reduction_pct'}
    feature_cols = [c for c in df_combined.columns if c not in exclude]
    X = df_combined[feature_cols].copy()

    valid = y.notna()
    X, y  = X[valid].reset_index(drop=True), y[valid].reset_index(drop=True)

    cat_cols = X.select_dtypes(include=['category', 'object']).columns.tolist()

    outer_cv = RepeatedKFold(n_splits=4, n_repeats=5, random_state=random_state)
    inner_cv = RepeatedKFold(n_splits=4, n_repeats=5, random_state=random_state)

    sweep_results = []
    total_start   = time.time()

    for threshold in THRESHOLDS:
        if threshold == 0.0:
            selected_cols = feature_cols.copy()
            thresh_label  = 'all'
        else:
            selected_cols = freq_norm[freq_norm >= threshold].index.tolist()
            thresh_label  = f'>={threshold*100:.0f}%'

        if len(selected_cols) == 0:
            print(f"\n  Threshold {thresh_label}: no features, skipped.")
            continue

        n_features = len(selected_cols)
        print(f"\n{'='*65}")
        print(f"  Threshold {thresh_label}:  {n_features} features")
        print(f"  {selected_cols[:8]}{'...' if n_features > 8 else ''}")
        print(f"{'='*65}")

        fold_results = []
        thresh_start = time.time()

        for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X), start=1):
            X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            if target_transformer is not None:
                pt_fold     = clone(target_transformer)
                y_train_fit = pd.Series(
                    pt_fold.fit_transform(y_train.values.reshape(-1, 1)).ravel(),
                    index=y_train.index)
            else:
                pt_fold, y_train_fit = None, y_train

            # Encode
            X_train_sel = X_train[selected_cols].copy()
            X_test_sel  = X_test[selected_cols].copy()
            cats_sel = [c for c in cat_cols if c in selected_cols]
            if cats_sel:
                oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                X_train_sel[cats_sel] = oe.fit_transform(X_train_sel[cats_sel].astype(str))
                X_test_sel[cats_sel]  = oe.transform(X_test_sel[cats_sel].astype(str))
            X_train_sel = X_train_sel.astype(float)
            X_test_sel  = X_test_sel.astype(float)

            # Impute
            X_train_imp, imputer = preprocess.impute_iterative(
                X_train_sel, ex_cols=None, iterations=10,
                random_state=random_state, verbose=False)
            X_train_imp = pd.DataFrame(
                X_train_imp, columns=selected_cols, index=X_train_sel.index)
            X_test_imp = pd.DataFrame(
                imputer.transform(X_test_sel),
                columns=selected_cols, index=X_test_sel.index)

            # Scale
            scaler = StandardScaler()
            X_train_sc = pd.DataFrame(
                scaler.fit_transform(X_train_imp),
                columns=selected_cols, index=X_train_sel.index)
            X_test_sc = pd.DataFrame(
                scaler.transform(X_test_imp),
                columns=selected_cols, index=X_test_sel.index)

            inner_splits = list(inner_cv.split(X_train_sc))

            def _fit_inner(itr, ival, params):
                m = ElasticNet(**params, max_iter=5000, random_state=random_state)
                m.fit(X_train_sc.iloc[itr], y_train_fit.iloc[itr])
                return np.sqrt(mean_squared_error(
                    y_train_fit.iloc[ival],
                    m.predict(X_train_sc.iloc[ival])))

            def model_objective(trial):
                params = dict(
                    alpha    = trial.suggest_float('alpha',    1e-4, 10.0, log=True),
                    l1_ratio = trial.suggest_float('l1_ratio', 0.0,  1.0),
                )
                rmses = joblib.Parallel(n_jobs=4, prefer='threads')(
                    joblib.delayed(_fit_inner)(itr, ival, params)
                    for itr, ival in inner_splits)
                return np.mean(rmses)

            model_study = optuna.create_study(direction='minimize')
            with contextlib.redirect_stderr(io.StringIO()):
                model_study.optimize(model_objective, n_trials=N_TRIALS,
                                     show_progress_bar=False)

            best_params = model_study.best_params
            print(f"  Fold {outer_fold:>2}/20: Trial {model_study.best_trial.number+1:>2}/{N_TRIALS}"
                  f"  RMSE={model_study.best_value:.4f}  {best_params}")

            fold_model = ElasticNet(**best_params, max_iter=5000, random_state=random_state)
            fold_model.fit(X_train_sc, y_train_fit)

            preds_raw = fold_model.predict(X_test_sc)
            preds = (pt_fold.inverse_transform(preds_raw.reshape(-1, 1)).ravel()
                     if pt_fold is not None else preds_raw)

            mae  = mean_absolute_error(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            r2   = r2_score(y_test, preds)
            fold_results.append({'Fold': outer_fold, 'MAE': mae, 'RMSE': rmse, 'R2': r2})

        res     = pd.DataFrame(fold_results)
        n_folds = len(fold_results)
        t_crit  = stats.t.ppf(0.975, df=n_folds - 1)

        mean_mae,  std_mae  = res['MAE'].mean(),  res['MAE'].std()
        mean_rmse, std_rmse = res['RMSE'].mean(), res['RMSE'].std()
        mean_r2,   std_r2   = res['R2'].mean(),   res['R2'].std()

        elapsed = (time.time() - thresh_start) / 60
        print(f"\n  {thresh_label}  {n_features} features  {elapsed:.1f} min")
        for label, mv, sv in [('MAE', mean_mae, std_mae),
                               ('RMSE', mean_rmse, std_rmse),
                               ('R2', mean_r2, std_r2)]:
            ci = t_crit * sv / np.sqrt(n_folds)
            print(f"    {label}: {mv:.3f} +/- {sv:.4f}  "
                  f"(95% CI [{mv-ci:.3f}, {mv+ci:.3f}])")

        sweep_results.append({
            'threshold':       threshold,
            'threshold_label': thresh_label,
            'n_features':      n_features,
            'mean_MAE':        mean_mae,  'std_MAE':  std_mae,
            'mean_RMSE':       mean_rmse, 'std_RMSE': std_rmse,
            'mean_R2':         mean_r2,   'std_R2':   std_r2,
        })

    total_elapsed = (time.time() - total_start) / 60
    print(f"\n{'='*65}")
    print(f"  Total time: {total_elapsed:.1f} min")
    print(f"{'='*65}")

    return pd.DataFrame(sweep_results)



#_________________________________________________________________________________________________
# Final Elasticnet Model
#_________________________________________________________________________________________________
def run_tuned_elasticnet(
    df_combined, feature_list, target_col='pain_reduction', random_state=42,
    target_transformer=None
):
    """ElasticNet with Optuna nested CV.
      Runs Elasticnet model on a selected sets of features, after feature selection and feature-threshold analysis.
      SHAP analysis is performed on the final model.
      1. Inner CV (4×5=20) + Optuna (50 trials) tunes ElasticNet hyperparameters on X_train
      2. Train final fold model on X_train → evaluate on X_test
      3. Final model: median HPs across outer folds, trained on full X

    Returns: results_df, final_model, X_final, y_pred, best_model_params_list, patient_err_df, scaler_final
    """
    from sklearn.linear_model import ElasticNet
    import optuna, warnings, statistics
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Optuna trials
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
        print(f"    {m:<5}: {mv:.3f} ± {sv:.4f}")

    # Final model uses the selected features, and the median hyperparameters across outer folds.
    X_final = X.copy()
    # encode
    if cat_cols:
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
    
    # reverse transform predictions
    if target_transformer is not None:
        pt_final    = clone(target_transformer)
        y_final_fit = pd.Series(
            pt_final.fit_transform(y.values.reshape(-1, 1)).ravel(), index=y.index)
    else:
        pt_final, y_final_fit = None, y

    hp_final = {
        k: statistics.median([p[k] for p in best_model_params_list])
        for k in best_model_params_list[0]}
    print(f"  Final model hyperparameters (median across outer folds): {hp_final}")

    final_model = ElasticNet(**hp_final, max_iter=5000, random_state=random_state)
    final_model.fit(X_final, y_final_fit)

    y_pred_raw = pd.Series(
        final_model.predict(X_final),
        index=range(len(X_final)), dtype='float64')
    y_pred = (pd.Series(
        pt_final.inverse_transform(y_pred_raw.values.reshape(-1, 1)).ravel(),
        index=y_pred_raw.index, dtype='float64')
              if pt_final is not None else y_pred_raw)

    return (final_model, X_final, y_pred,
            patient_err_df, scaler_final)
