# Modeling functions — HistGradientBoosting with RENT + Optuna
import time
import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy import stats
from sklearn.model_selection import RepeatedKFold, train_test_split
from sklearn.preprocessing import OrdinalEncoder
import joblib
import contextlib, io
import statistics

import preprocess


def _prep_for_hgb(df, cat_cols, encoder=None):
    """OrdinalEncode categorical columns for HistGradientBoosting.

    HGB requires categorical features to be non-negative integers (NaN allowed).
    Fills NaN with '__missing__' before encoding so every value maps to a valid integer.

    Returns
    -------
    out     : pd.DataFrame  encoded copy
    encoder : fitted OrdinalEncoder (pass back in for transform-only calls)
    """
    out = df.copy()
    cats_present = [c for c in cat_cols if c in out.columns]
    if cats_present:
        for c in cats_present:
            out[c] = out[c].astype(object).fillna('__missing__')
        if encoder is None:
            encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=0)
            out[cats_present] = encoder.fit_transform(out[cats_present]).astype(float)
        else:
            out[cats_present] = encoder.transform(out[cats_present]).astype(float)
    return out, encoder


def _prep_for_rent(X_imp, cat_cols):
    """OrdinalEncode categorical columns in an already-imputed DataFrame for RENT.
    RENT requires a fully numeric NaN-free matrix."""
    out = X_imp.copy()
    cats_present = [c for c in cat_cols if c in out.columns]
    if cats_present:
        oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        out[cats_present] = oe.fit_transform(out[cats_present].astype(str))
    return out.astype(float)


# _____________________________________________________________________________
# Tuned HistGradientBoosting + RENT Feature Selection + Optuna
# ══════════════════════════════════════════════════════════════════════════════

def run_advanced_hgb_rent(
    df_combined, target_col='pain_reduction_pct', random_state=42,
    tau_3=0.95, target_transformer=None,
):
    """HistGradientBoostingRegressor with Optuna-tuned RENT and Model Hyperparameters.

    Per outer fold:
      0. MICE imputation on outer fold X_train (used for RENT only).
      1. Tune RENT HPs (C, l1_ratio, τ₁, τ₂) via Optuna on 75-25 split of imputed X_train.
      2. Re-run RENT on full imputed X_train with best HPs → selected feature subset.
      3. Inner CV (4×5=20) + Optuna (N_TRIALS) tunes HGB HPs on OrdinalEncoded X_train (NaN intact).
      4. Train final fold model on X_train → evaluate on X_test.
      5. Final model: features selected in ≥75% of outer folds, median HPs across outer folds.

    HGB handles missing values natively for numeric features.
    Categorical features are OrdinalEncoded (encoder fit on train only).

    Returns: results_df, final_model, X_final, y_pred, best_model_params_list, feature_freq
    """
    from RENT import RENT
    import optuna, warnings
    from collections import Counter

    for cat in [FutureWarning, RuntimeWarning]:
        warnings.filterwarnings('ignore', category=cat, module='RENT')
    for pat in ['.*joblib.*', '.*loky.*']:
        warnings.filterwarnings('ignore', message=pat)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    N_TRIALS = 20

    y            = df_combined[target_col].copy()
    exclude      = {'Patient', 'Timepoint', target_col, 'pain_reduction',
                    'pain_reduction_pct', 'pain_under_load_reduction',
                    'pain_under_load_reduction_pct'}
    feature_cols = [c for c in df_combined.columns if c not in exclude]
    X            = df_combined[feature_cols].copy()

    valid = y.notna()
    X, y  = X[valid].reset_index(drop=True), y[valid].reset_index(drop=True)

    cat_cols = X.select_dtypes(include=['category', 'object']).columns.tolist()

    print(f"\n{'='*65}")
    print(f"  HGB + Optuna + RENT — {target_col}")
    print(f"  n={len(X)}, p={len(feature_cols)}, τ₃={tau_3}")
    print(f"  Outer 4×5=20 | Inner 4×5=20 | RENT & Optuna trials={N_TRIALS} | K=100")
    print(f"{'='*65}")

    outer_cv = RepeatedKFold(n_splits=4, n_repeats=5, random_state=random_state)
    inner_cv = RepeatedKFold(n_splits=4, n_repeats=5, random_state=random_state)

    fold_results                          = []
    best_model_params_list, selected_features_per_fold = [], []
    start = time.time()

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

        # OrdinalEncode categoricals for HGB — fit encoder on train only
        X_train_hgb, oe_fold = _prep_for_hgb(X_train, cat_cols)
        X_test_hgb,  _       = _prep_for_hgb(X_test,  cat_cols, encoder=oe_fold)

        # MICE imputation on X_train for RENT (ElasticNet requires NaN-free numeric matrix)
        X_train_imp, _ = preprocess.impute_iterative(
            X_train, ex_cols=None, iterations=10, random_state=42, verbose=False)
        X_train_rent   = _prep_for_rent(X_train_imp, cat_cols)
        del X_train_imp

        # ── Step 1: Tune RENT HPs on 75-25 split ──────────────────────────────
        X_tr_rent, X_val_rent, y_tr, y_val = train_test_split(
            X_train_rent, y_train_fit, test_size=0.25, random_state=random_state)

        X_tr_hgb        = X_train_hgb.loc[X_tr_rent.index]
        X_val_hgb       = X_train_hgb.loc[X_val_rent.index]
        X_tr_rent_reset = X_tr_rent.reset_index(drop=True)

        def rent_objective(trial):
            c_val    = trial.suggest_float('C',        1e-3, 10,  log=True)
            l1_ratio = trial.suggest_float('l1_ratio', 0.1,  1.0)
            tau_1    = trial.suggest_float('tau_1',    0.7,  0.95)
            tau_2    = trial.suggest_float('tau_2',    0.7,  0.95)

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
            if len(sel_idx) == 0 or len(sel_idx) > 45:
                return 1e6

            sel_cols = [feature_cols[i] for i in sel_idx]
            cat_mask = [c in cat_cols for c in sel_cols]

            probe = HistGradientBoostingRegressor(
                max_iter=300, max_depth=6, random_state=random_state,
                categorical_features=cat_mask if any(cat_mask) else None)
            probe.fit(X_tr_hgb[sel_cols], y_tr)
            return np.sqrt(mean_squared_error(y_val, probe.predict(X_val_hgb[sel_cols])))

        rent_study = optuna.create_study(direction='minimize')
        with contextlib.redirect_stderr(io.StringIO()):
            rent_study.optimize(rent_objective, n_trials=50, n_jobs=1, show_progress_bar=False)

        best_rent = rent_study.best_params
        print(f"  Best RENT RMSE: {rent_study.best_value:.4f}  Best Parameters: {best_rent}")

        # ── Step 2: Re-run RENT on full X_train with best HPs ─────────────────
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

        selected_cols  = ([feature_cols[i] for i in sel_idx_outer]
                          if len(sel_idx_outer) > 0 else feature_cols)
        cat_mask_inner = [c in cat_cols for c in selected_cols]
        selected_features_per_fold.append(selected_cols)

        suffix = '...' if len(selected_cols) > 8 else ''
        print(f"  RENT Selected: {len(selected_cols)}/{len(feature_cols)} features — "
              f"{selected_cols[:8]}{suffix}")

        # ── Step 3: Inner CV HGB Hyperparameter tuning with Optuna ───────────
        inner_splits = list(inner_cv.split(X_train_hgb))

        def _fit_inner(itr, ival, params):
            m = HistGradientBoostingRegressor(
                **params, random_state=random_state,
                categorical_features=cat_mask_inner if any(cat_mask_inner) else None)
            m.fit(X_train_hgb.iloc[itr][selected_cols], y_train_fit.iloc[itr])
            return np.sqrt(mean_squared_error(
                y_train_fit.iloc[ival],
                m.predict(X_train_hgb.iloc[ival][selected_cols])))

        def model_objective(trial):
            params = dict(
                max_iter          = trial.suggest_int(  'max_iter',          100,  500),
                max_depth         = trial.suggest_int(  'max_depth',         3,    10),
                learning_rate     = trial.suggest_float('learning_rate',     1e-3, 0.3,  log=True),
                l2_regularization = trial.suggest_float('l2_regularization', 0.0,  10.0),
                min_samples_leaf  = trial.suggest_int(  'min_samples_leaf',  5,    50),
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
        model_study.optimize(model_objective, n_trials=N_TRIALS,
                             callbacks=[_cb], show_progress_bar=False)

        best_model_params = model_study.best_params
        best_model_params_list.append(best_model_params)
        print(f"  Best Trial: {model_study.best_trial.number}   "
              f"RMSE={model_study.best_value:.4f}  {best_model_params}")

        # ── Step 4: Train on full X_train → evaluate on X_test ───────────────
        cat_mask_sel = [c in cat_cols for c in selected_cols]
        fold_model   = HistGradientBoostingRegressor(
            **best_model_params, random_state=random_state,
            categorical_features=cat_mask_sel if any(cat_mask_sel) else None)
        fold_model.fit(X_train_hgb[selected_cols], y_train_fit)

        preds_raw = fold_model.predict(X_test_hgb[selected_cols])
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
        print("   Warning: No features met ≥75% threshold — falling back to top 10.")

    # ── Final model ───────────────────────────────────────────────────────────
    final_cols = ([f for f, cnt in freq.items() if cnt / n_outer >= 0.75]
                  or [f for f, _ in freq.most_common(10)])
    print(f"\n  Final model: {len(final_cols)} features (≥75%): {final_cols}")

    X_final, oe_final  = _prep_for_hgb(X[final_cols], cat_cols)
    cat_mask_final     = [c in cat_cols for c in final_cols]

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

    final_model = HistGradientBoostingRegressor(
        **hp_final, random_state=random_state,
        categorical_features=cat_mask_final if any(cat_mask_final) else None)
    final_model.fit(X_final, y_final_fit)

    y_pred_raw = pd.Series(final_model.predict(X_final),
                           index=range(len(X_final)), dtype='float64')
    y_pred     = (pd.Series(pt_final.inverse_transform(y_pred_raw.values.reshape(-1, 1)).ravel(),
                            index=y_pred_raw.index, dtype='float64')
                  if pt_final is not None else y_pred_raw)

    return (results_df, final_model, X_final, y_pred,
            best_model_params_list, feature_freq)
