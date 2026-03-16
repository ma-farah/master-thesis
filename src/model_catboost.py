
# Modeling functions — baseline and advanced regressors
import time
import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
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

def _prep_for_rent(X_imp, cat_cols):
    """OrdinalEncode categorical columns in an already-imputed DataFrame.
    RENT requires a fully numeric NaN-free matrix."""
    out = X_imp.copy()
    cats_present = [c for c in cat_cols if c in out.columns]
    if cats_present:
        oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        # astype(str) is safe here because X_imp is already imputed — no real NaN to corrupt
        out[cats_present] = oe.fit_transform(out[cats_present].astype(str))
    return out.astype(float)


# _____________________________________________________________________________
# Tuned CatBoost Model + Rent Feature Selection + Optuna tuning
# ══════════════════════════════════════════════════════════════════════════════

def run_advanced_catboost_rent(
    df_combined, target_col='pain_reduction_pct', random_state=42,
    tau_3=0.95, target_transformer=None,
):
    """CatBoostRegressor with Optuna-tuned RENT and Model Hyperparameters.

    Per outer fold:
      0. Iterative imputer on outer fold X_train (used for RENT tuning only)
      1. Tune RENT HPs (C, l1_ratio, τ₁, τ₂) via Optuna on 75-25 split of imputed X_train.
      2. Re-run RENT on full imputed X_train with best HPs → selected feature subset.
      3. Inner CV (4×5=20) + Optuna (20 trials) tunes CatBoost HPs on raw X_train (NaN intact).
      4. Train final fold model on raw X_train → evaluate on raw X_test (no imputation needed).
      5. Final model: features selected in ≥75% of outer folds, median HPs across outer folds.

    MICE imputation is only used to prepare NaN-free input for RENT (ElasticNet requires this).
    CatBoost handles missing values natively — X_test is never imputed.

    Returns: results_df, final_model, X_final, y_pred, best_model_params_list, feature_freq
    """
    from RENT import RENT
    from catboost import CatBoostRegressor
    import optuna, warnings, statistics
    from collections import Counter
    
    for cat in [FutureWarning, RuntimeWarning]:
        warnings.filterwarnings('ignore', category=cat, module='RENT')
    for pat in ['.*less than 75% GPU memory.*', '.*joblib.*', '.*loky.*']:
        warnings.filterwarnings('ignore', message=pat)
    optuna.logging.set_verbosity(optuna.logging.WARNING)


    N_TRIALS = 20 # prøv 50 etterhvert   

    y            = df_combined[target_col].copy()

    exclude      = {'Patient', 'Timepoint', target_col, 'pain_reduction',
                    'pain_reduction_pct', 'pain_under_load_reduction',
                    'pain_under_load_reduction_pct'}
    feature_cols = [c for c in df_combined.columns if c not in exclude]
    X            = df_combined[feature_cols].copy()

    valid = y.notna()
    X, y  = X[valid].reset_index(drop=True), y[valid].reset_index(drop=True)

    # Identify categorical columns and  keep as category dtype throughout pipeline!
    cat_cols = X.select_dtypes(include=['category', 'object']).columns.tolist()

    print(f"\n{'='*65}")
    print(f"  CatBoost + Optuna + RENT — {target_col}")
    print(f"  n={len(X)}, p={len(feature_cols)}, τ₃={tau_3}")
    print(f"  Outer 4×5=20 | Inner 4×5=20 | RENT & Optuna trials={N_TRIALS} | K=100")  
    print(f"{'='*65}")

    outer_cv = RepeatedKFold(n_splits=4, n_repeats=5, random_state=random_state)  #
    inner_cv = RepeatedKFold(n_splits=4, n_repeats=5, random_state=random_state)  # TEST with 25 repeats

    # Storing best parameters
    fold_results = []
    best_model_params_list, selected_features_per_fold = [], []
    start = time.time()

    # Splitting into outer folds
    for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X), start=1):
        print(f"\n{'─'*65}")
        print(f"  Outer fold {outer_fold}/{outer_cv.get_n_splits()}")
        print(f"{'─'*65}")

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Power-transform target-value
        if target_transformer is not None:
            pt_fold     = clone(target_transformer)
            y_train_fit = pd.Series(
                pt_fold.fit_transform(y_train.values.reshape(-1, 1)).ravel(),
                index=y_train.index)
        else:
            pt_fold, y_train_fit = None, y_train

        # Prepare CatBoost-ready X_train
        X_train_cb = _prep_for_catboost(X_train, cat_cols)

        # ---- Iterative Imputer on X_train for Rent Tuning --------
        X_train_imp, imputer = preprocess.impute_iterative(
            X_train, ex_cols=None, iterations=10, 
            random_state=42, verbose=False)

        # Ordinal-encode imputed X_train to a fully numeric dataframe; free imputed copy
        X_train_rent = _prep_for_rent(X_train_imp, cat_cols)
        del X_train_imp

        # ---- Step 1: Tune RENT HPs on 75-25 split of imputed X_train ---------
        # Splitting the imputed+encoded matrix for rent-tuning and feature selection
        X_tr_rent, X_val_rent, y_tr, y_val = train_test_split(
            X_train_rent, y_train_fit, test_size=0.25, random_state=random_state)

        # Slice the already-prepared CatBoost matrix — no extra conversion needed
        X_tr_cb  = X_train_cb.loc[X_tr_rent.index]
        X_val_cb = X_train_cb.loc[X_val_rent.index]

        # Precompute reset index once instead of repeating inside every Optuna trial
        X_tr_rent_reset = X_tr_rent.reset_index(drop=True)

        # Suggested Rent Parameters for tuning
        def rent_objective(trial):
            c_val    = trial.suggest_float('C',        1e-3, 10,  log=True)
            l1_ratio = trial.suggest_float('l1_ratio', 0.1,  1.0)
            tau_1    = trial.suggest_float('tau_1',    0.7,  0.95)
            tau_2    = trial.suggest_float('tau_2',    0.7,  0.95)

            # Running RENT on imputed+encoded data to select features
            rent_t = RENT.RENT_Regression(
                data=X_tr_rent_reset,
                target=y_tr.values, feat_names=feature_cols,
                C=[c_val], l1_ratios=[l1_ratio], autoEnetParSel=False,
                poly='OFF', testsize_range=(0.25, 0.25), K=100,  
                random_state=random_state, verbose=0)
            with contextlib.redirect_stderr(io.StringIO()):
                rent_t.train()
            sel_idx = rent_t.select_features(
                tau_1_cutoff=tau_1, tau_2_cutoff=tau_2, tau_3_cutoff=tau_3)
            # penalize over selection
            if len(sel_idx) == 0:
                return 1e6
            if len(sel_idx) > 45:
                return 1e6
            sel_cols    = [feature_cols[i] for i in sel_idx]
            cat_sel     = [c for c in cat_cols if c in sel_cols]

            # Baseline CatBoost on nan-intact data to score the selected features
            probe = CatBoostRegressor(
                iterations=500, depth=6, random_seed=random_state, loss_function='RMSE',  
                cat_features=cat_sel,
                task_type='CPU', thread_count=-1, logging_level='Silent')
            with contextlib.redirect_stderr(io.StringIO()):
                probe.fit(X_tr_cb[sel_cols], y_tr)
            return np.sqrt(mean_squared_error(y_val, probe.predict(X_val_cb[sel_cols])))

        rent_study = optuna.create_study(direction='minimize')
        with contextlib.redirect_stderr(io.StringIO()):
            rent_study.optimize(rent_objective, n_trials=50, n_jobs=1, show_progress_bar=False)

        # Store best parameters and RMSE score (in transformed space!)
        best_rent = rent_study.best_params
        print(f"  Best RENT RMSE: {rent_study.best_value:.4f} Best Parameters: {best_rent}")

        # -- Step 2: Re-run RENT on full imputed X_train with best HPs ─────────
        rent_full = RENT.RENT_Regression(
            data=X_train_rent.reset_index(drop=True),
            target=y_train_fit.values, feat_names=feature_cols,
            C=[best_rent['C']], l1_ratios=[best_rent['l1_ratio']],
            autoEnetParSel=False, poly='OFF', testsize_range=(0.25, 0.25),
            K=100, random_state=random_state, verbose=0)  
        
        with contextlib.redirect_stderr(io.StringIO()):
            rent_full.train()
        sel_idx_outer = rent_full.select_features(
            tau_1_cutoff=best_rent['tau_1'],
            tau_2_cutoff=best_rent['tau_2'],
            tau_3_cutoff=tau_3)

        # Selected Features for this Outer Fold
        selected_cols  = ([feature_cols[i] for i in sel_idx_outer]
                          if len(sel_idx_outer) > 0 else feature_cols)
        cat_cols_inner = [c for c in cat_cols if c in selected_cols]
        selected_features_per_fold.append(selected_cols)

        suffix = '...' if len(selected_cols) > 8 else ''
        print(f"  RENT Selected: {len(selected_cols)}/{len(feature_cols)} features — "
              f"{selected_cols[:8]}{suffix}")


        # ── Step 3: Inner CV CatBoost Hyperparameter tuning with Optuna  ─────────────────────────
        # CatBoost handles NaN, no imputation needed, using nan-dataset
   
        inner_splits = list(inner_cv.split(X_train_cb))
        def _fit_inner(itr, ival, params):
            # Train and evaluate one inner-fold CatBoost model on dataset (with nan)
            m = CatBoostRegressor(
                iterations=500, **params, cat_features=cat_cols_inner, loss_function='RMSE',  # TEST
                random_seed=random_state, task_type='CPU', thread_count=-1,
                logging_level='Silent')
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
                print(f"    Trial {trial.number+1:>3}/{N_TRIALS}: "
                      f"RMSE={trial.value:.4f}  {trial.params}")

        model_study = optuna.create_study(direction='minimize')
        with contextlib.redirect_stderr(io.StringIO()):
            model_study.optimize(model_objective, n_trials=N_TRIALS,
                                 callbacks=[_cb], show_progress_bar=False) #check n jobs
        
        # Get the best parameters for the best trial
        best_model_params = model_study.best_params
        best_model_params_list.append(best_model_params)
        print(f"  Best Trial: {model_study.best_trial.number}   RMSE={model_study.best_value:.4f}  {best_model_params}")


        # ── Step 4: Train on full X_train --> evaluate on X_test ───────────────
        # conserve category types:
        X_test_cb  = _prep_for_catboost(X_test, cat_cols)

        fold_model = CatBoostRegressor(
            iterations=1000, **best_model_params, cat_features=cat_cols_inner, loss_function='RMSE',  # TEST: 100 (production: 1000)
            random_seed=random_state, task_type='CPU', thread_count=-1, logging_level='Silent')
        with contextlib.redirect_stderr(io.StringIO()):
            fold_model.fit(X_train_cb[selected_cols], y_train_fit)

        preds_raw = fold_model.predict(X_test_cb[selected_cols])
        preds     = (pt_fold.inverse_transform(preds_raw.reshape(-1, 1)).ravel()
                     if pt_fold is not None else preds_raw)

        mae  = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2   = r2_score(y_test, preds)
        fold_results.append({'Fold': outer_fold, 'MAE': mae, 'MSE': rmse**2, 'RMSE': rmse, 'R2': r2})
        print(f"  Outer Fold {outer_fold} |  Features={len(selected_cols)}: {selected_cols}")
        print(f"    MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}")

    print(f"\n  Training time: {(time.time()-start)/60:.1f} min")


    # ── RESULTS SUMMARY ───────────────────────────────────────────────────────
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

    # ── Feature selection frequency ───────────────────────────────────────────
    freq         = Counter(f for fold in selected_features_per_fold for f in fold)
    feature_freq = (pd.Series(dict(freq), name='selection_count')
                    .reindex(feature_cols, fill_value=0)
                    .sort_values(ascending=False))
    feature_freq.index.name = 'feature'

    print(f"\n  Top 30 RENT feature-selection frequencies:")
    for feat, cnt in freq.most_common(30):
        print(f"    {cnt:>3}/{n_outer}  {feat}{'   (≥75%)' if cnt/n_outer >= 0.75 else ''}")

    if not [f for f, cnt in freq.items() if cnt / n_outer >= 0.75]:
        print("   Warning: No features met ≥75% threshold - falling back to top 10 selected features for final model.")


    # ── Final model ───────────────────────────────────────────────────────────
    # Trained on full X (all outer folds combined) using features stable across ≥75% of folds
    # and median hyperparameters from all outer folds.
    final_cols = ([f for f, cnt in freq.items() if cnt / n_outer >= 0.75]
                  or [f for f, _ in freq.most_common(10)])
    print(f"\n  Final model: {len(final_cols)} features (≥75%): {final_cols}")

    # Prepare full dataset for CatBoost (category → object, NaN preserved)
    X_final        = _prep_for_catboost(X[final_cols], cat_cols)
    cat_cols_final = [c for c in cat_cols if c in final_cols]

    if target_transformer is not None:
        pt_final    = clone(target_transformer)
        y_final_fit = pd.Series(
            pt_final.fit_transform(y.values.reshape(-1, 1)).ravel(), index=y.index)
    else:
        pt_final, y_final_fit = None, y

    # Aggregate per-fold best HPs into a single set, using the median
    hp_final = {k: (int(round(statistics.median([p[k] for p in best_model_params_list])))
                    if isinstance(best_model_params_list[0][k], int)
                    else statistics.median([p[k] for p in best_model_params_list]))
                for k in best_model_params_list[0]}
    print(f"  Final model HPs (median): {hp_final}")

    final_model = CatBoostRegressor(
        iterations=1000, loss_function='RMSE', custom_metric=['MAE', 'R2'],  # TEST: 100 (production: 1000)
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
            best_model_params_list, feature_freq)

