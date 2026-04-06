# Trying other feature selection methods
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



#_________________________________________________________________________________________________
# ElasticNet + RFE
#_________________________________________________________________________________________________

def elasticnet_rfe(
    df_combined, target_col='pain_reduction_pct', random_state=42,
    target_transformer=None,
):
    """ElasticNet with RFE (n_features_to_select + step tuned by Optuna) inside each outer CV fold.
      1. Tune n_features_to_select and step via Optuna (20 trials) on a 75-25 split of X_train.
         n_features_to_select candidates: [40, 30, 20, 10]. step: float in (0.0, 1.0).
         RFE estimator: Ridge (stable coef_ for feature ranking, decoupled from final ElasticNet).
      2. Re-run RFE on full X_train with best params -> selected feature subset for the outer fold.
      3. Inner CV (4×5=20) + Optuna (50 trials) tunes ElasticNet hyperparameters.

    Returns: results_df, feature_freq, selected_features_per_fold
    """
    from sklearn.linear_model import ElasticNet, Ridge
    from sklearn.feature_selection import RFE
    from sklearn.model_selection import train_test_split
    from collections import Counter
    import optuna, warnings, statistics

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=RuntimeWarning)

    N_TRIALS_RFE   = 20
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

    p = len(feature_cols)
    print(f"\n{'='*80}")
    print(f" Nested CV - ElasticNet + RFE + Optuna — {target_col}")
    print(f"  n={len(X)}, p={p}")
    print(f"  Outer 4×5=20 | Inner 4×5=20 | Optuna Trials Model={N_TRIALS_MODEL} | Optuna Trials RFE={N_TRIALS_RFE}")
    print(f"{'='*80}")

    outer_cv = RepeatedKFold(n_splits=4, n_repeats=5, random_state=random_state)
    inner_cv = RepeatedKFold(n_splits=4, n_repeats=5, random_state=random_state)

    fold_results               = []
    best_model_params_list     = []
    best_rfe_params_per_fold   = []
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

        # ── Step 1: Tune RFE params on 75-25 split of X_train ───────────────
        X_train_rfe = prep_for_mrmr(X_train, cat_cols, random_state)  # encode + impute

        X_tr_rfe, X_val_rfe, y_tr, y_val = train_test_split(
            X_train_rfe, y_train_fit, test_size=0.25, random_state=random_state)

        # Scale for RFE tuning split
        scaler_rfe = StandardScaler()
        X_tr_rfe_s  = pd.DataFrame(scaler_rfe.fit_transform(X_tr_rfe),  columns=feature_cols)
        X_val_rfe_s = pd.DataFrame(scaler_rfe.transform(X_val_rfe),     columns=feature_cols)

        def rfe_objective(trial):
            n_features = trial.suggest_categorical('n_features_to_select', [30, 20, 15, 10])
            step       = trial.suggest_categorical('step', [1, 5, 10])

            # step=0.0 means remove 1 feature per iteration; clamp to at least 1
            rfe = RFE(
                estimator=Ridge(random_state=random_state),
                n_features_to_select=min(n_features, X_tr_rfe_s.shape[1]),
                step=max(step, 1e-6))
            rfe.fit(X_tr_rfe_s, y_tr)
            sel_cols = [c for c, s in zip(feature_cols, rfe.support_) if s]

            if len(sel_cols) == 0:
                return 1e6

            probe = ElasticNet(max_iter=5000, random_state=random_state)
            probe.fit(X_tr_rfe_s[sel_cols], y_tr)
            return np.sqrt(mean_squared_error(y_val, probe.predict(X_val_rfe_s[sel_cols])))

        rfe_study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=random_state))
        rfe_study.optimize(rfe_objective, n_trials=N_TRIALS_RFE, show_progress_bar=False)

        best_rfe   = rfe_study.best_params
        best_rfe_params_per_fold.append(best_rfe)
        print(f"  Best RFE params: {best_rfe}  RMSE={rfe_study.best_value:.4f}")

        # ── Step 2: Re-run RFE on full X_train with best params ──────────────
        X_train_full = prep_for_mrmr(X_train, cat_cols, random_state)
        scaler_full  = StandardScaler()
        X_train_full_s = pd.DataFrame(
            scaler_full.fit_transform(X_train_full), columns=feature_cols)

        rfe_full = RFE(
            estimator=Ridge(random_state=random_state),
            n_features_to_select=min(best_rfe['n_features_to_select'], X_train_full_s.shape[1]),
            step=best_rfe['step'])
        rfe_full.fit(X_train_full_s, y_train_fit)
        selected_cols = [c for c, s in zip(feature_cols, rfe_full.support_) if s]
        selected_features_per_fold.append(selected_cols)
        print(f"  {len(selected_cols)} selected features: {selected_cols}")

        # ── Encode + impute + scale for ElasticNet ───────────────────────────
        X_train_sel = X_train[selected_cols].copy()
        X_test_sel  = X_test[selected_cols].copy()

        cats_sel = [c for c in cat_cols if c in selected_cols]
        if cats_sel:
            oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            X_train_sel[cats_sel] = oe.fit_transform(X_train_sel[cats_sel].astype(str))
            X_test_sel[cats_sel]  = oe.transform(X_test_sel[cats_sel].astype(str))

        X_train_sel = X_train_sel.astype(float)
        X_test_sel  = X_test_sel.astype(float)

        X_train_imp, imputer = preprocess.impute_iterative(
            X_train_sel, ex_cols=None, iterations=10,
            random_state=random_state, verbose=False)
        X_train_imp = pd.DataFrame(X_train_imp, columns=selected_cols, index=X_train_sel.index)
        X_test_imp  = pd.DataFrame(
            imputer.transform(X_test_sel), columns=selected_cols, index=X_test_sel.index)

        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train_imp), columns=selected_cols, index=X_train_sel.index)
        X_test_scaled  = pd.DataFrame(
            scaler.transform(X_test_imp), columns=selected_cols, index=X_test_sel.index)

        print('     Running 20 Inner Folds, 50 Optuna Trials...')

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

        model_study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=random_state))
        with contextlib.redirect_stderr(io.StringIO()):
            model_study.optimize(model_objective, n_trials=N_TRIALS_MODEL,
                                 show_progress_bar=False)

        best_model_params = model_study.best_params
        best_model_params_list.append(best_model_params)
        print(f"     Best Trial:  {model_study.best_trial.number}/{N_TRIALS_MODEL}"
              f"   RMSE={model_study.best_value:.4f}  {best_model_params}")

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
        print(f"  MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}")

    print(f"\n  Training time: {(time.time()-start)/60:.1f} min")

    results_df  = pd.DataFrame(fold_results)
    metric_cols = ['MAE', 'MSE', 'RMSE', 'R2']
    mean_row    = {'Fold': 'Mean', **{m: results_df[m].mean() for m in metric_cols}}
    std_row     = {'Fold': 'Std',  **{m: results_df[m].std()  for m in metric_cols}}
    results_df  = pd.concat(
        [results_df, pd.DataFrame([mean_row, std_row])], ignore_index=True)

    n_outer = len(fold_results)
    print(f"\n{'='*65}\n  SUMMARY — {target_col}\n{'='*65}")
    for m in metric_cols:
        mv, sv = mean_row[m], std_row[m]
        print(f"    {m:<5}: {mv:.3f} ± {sv:.4f})")

    freq = Counter(f for fold in selected_features_per_fold for f in fold)
    feature_freq = (
        pd.Series(dict(freq), name='selection_count')
        .reindex(feature_cols, fill_value=0)
        .sort_values(ascending=False))
    feature_freq.index.name = 'feature'

    print(f"\n  Complete Feature Selection Frequency List:")
    for feat, cnt in feature_freq.items():
        print(f"    {cnt:>2}/{n_outer}  {cnt/n_outer*100:4.1f}%  {feat}")

    return results_df, feature_freq, selected_features_per_fold


#_________________________________________________________________________________________________
# ElasticNet + RENT
#_________________________________________________________________________________________________

def elasticnet_rent(
    df_combined, target_col='pain_reduction_pct', random_state=42,
    tau_3=0.975, target_transformer=None,
):
    """ElasticNet with RENT (C, l1_ratio, τ₁ tuned by Optuna) inside each outer CV fold.
      1. Tune RENT HPs (C, l1_ratio, τ₁) via Optuna (20 trials) on a 75-25 split of X_train.
      2. Re-run RENT on full X_train with best HPs -> selected feature subset for the outer fold.
      3. Inner CV (4×5=20) + Optuna (50 trials) tunes ElasticNet hyperparameters.

    Returns: results_df, feature_freq, selected_features_per_fold
    """
    from sklearn.linear_model import ElasticNet
    from RENT import RENT
    from sklearn.model_selection import train_test_split
    from collections import Counter
    import optuna, warnings, statistics

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    for pat in ['.*joblib.*', '.*loky.*']:
        warnings.filterwarnings('ignore', message=pat)

    N_TRIALS_RENT  = 20
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

    p = len(feature_cols)
    print(f"\n{'='*80}")
    print(f" Nested CV - ElasticNet + RENT + Optuna — {target_col}")
    print(f"  n={len(X)}, p={p}, τ₃={tau_3}")
    print(f"  Outer 4×5=20 | Inner 4×5=20 | Optuna Trials Model={N_TRIALS_MODEL} | Optuna Trials RENT={N_TRIALS_RENT} | K=100")
    print(f"{'='*80}")

    outer_cv = RepeatedKFold(n_splits=4, n_repeats=5, random_state=random_state)
    inner_cv = RepeatedKFold(n_splits=4, n_repeats=5, random_state=random_state)

    fold_results               = []
    best_model_params_list     = []
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

        # ── Impute + encode X_train for RENT (fully numeric, NaN-free) ───────
        X_train_rent = prep_for_mrmr(X_train, cat_cols, random_state)

        # ── Step 1: Tune RENT HPs on 75-25 split of X_train ─────────────────
        X_tr_rent, X_val_rent, y_tr, y_val = train_test_split(
            X_train_rent, y_train_fit, test_size=0.25, random_state=random_state)

        scaler_probe = StandardScaler()
        X_tr_s  = pd.DataFrame(scaler_probe.fit_transform(X_tr_rent),  columns=feature_cols)
        X_val_s = pd.DataFrame(scaler_probe.transform(X_val_rent),     columns=feature_cols)
        X_tr_rent_reset = X_tr_rent.reset_index(drop=True)

        def rent_objective(trial):
            c_val    = trial.suggest_float('C',        1e-3, 10.0, log=True)
            l1_ratio = trial.suggest_float('l1_ratio', 0.1,  1.0)
            tau_1    = trial.suggest_float('tau_1',    0.0,  1.0)

            rent_t = RENT.RENT_Regression(
                data=X_tr_rent_reset,
                target=y_tr.values, feat_names=feature_cols,
                C=[c_val], l1_ratios=[l1_ratio], autoEnetParSel=False,
                poly='OFF', testsize_range=(0.25, 0.25), K=100,
                random_state=random_state, verbose=0)

            with contextlib.redirect_stderr(io.StringIO()):
                rent_t.train()
            sel_idx = rent_t.select_features(
                tau_1_cutoff=tau_1, tau_2_cutoff=tau_1, tau_3_cutoff=tau_3)

            if len(sel_idx) == 0:
                return 1e6

            sel_cols = [feature_cols[i] for i in sel_idx]
            probe = ElasticNet(max_iter=5000, random_state=random_state)
            probe.fit(X_tr_s[sel_cols], y_tr)
            return np.sqrt(mean_squared_error(y_val, probe.predict(X_val_s[sel_cols])))

        rent_study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=random_state))
        with contextlib.redirect_stderr(io.StringIO()):
            rent_study.optimize(rent_objective, n_trials=N_TRIALS_RENT,
                                n_jobs=1, show_progress_bar=False)

        best_rent = rent_study.best_params
        print(f"  Best RENT params: {best_rent}  RMSE={rent_study.best_value:.4f}")

        # ── Step 2: Re-run RENT on full X_train with best HPs ────────────────
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
            tau_2_cutoff=best_rent['tau_1'],
            tau_3_cutoff=tau_3)

        selected_cols = ([feature_cols[i] for i in sel_idx_outer]
                         if len(sel_idx_outer) > 0 else feature_cols)
        selected_features_per_fold.append(selected_cols)
        suffix = '...' if len(selected_cols) > 8 else ''
        print(f"  {len(selected_cols)} selected features: {selected_cols[:8]}{suffix}")

        # ── Encode + impute + scale for ElasticNet ───────────────────────────
        X_train_sel = X_train[selected_cols].copy()
        X_test_sel  = X_test[selected_cols].copy()

        cats_sel = [c for c in cat_cols if c in selected_cols]
        if cats_sel:
            oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            X_train_sel[cats_sel] = oe.fit_transform(X_train_sel[cats_sel].astype(str))
            X_test_sel[cats_sel]  = oe.transform(X_test_sel[cats_sel].astype(str))

        X_train_sel = X_train_sel.astype(float)
        X_test_sel  = X_test_sel.astype(float)

        X_train_imp, imputer = preprocess.impute_iterative(
            X_train_sel, ex_cols=None, iterations=10,
            random_state=random_state, verbose=False)
        X_train_imp = pd.DataFrame(X_train_imp, columns=selected_cols, index=X_train_sel.index)
        X_test_imp  = pd.DataFrame(
            imputer.transform(X_test_sel), columns=selected_cols, index=X_test_sel.index)

        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train_imp), columns=selected_cols, index=X_train_sel.index)
        X_test_scaled  = pd.DataFrame(
            scaler.transform(X_test_imp), columns=selected_cols, index=X_test_sel.index)

        print('     Running 20 Inner Folds, 50 Optuna Trials...')

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

        model_study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=random_state))
        with contextlib.redirect_stderr(io.StringIO()):
            model_study.optimize(model_objective, n_trials=N_TRIALS_MODEL,
                                 show_progress_bar=False)

        best_model_params = model_study.best_params
        best_model_params_list.append(best_model_params)
        print(f"     Best Trial:  {model_study.best_trial.number}/{N_TRIALS_MODEL}"
              f"   RMSE={model_study.best_value:.4f}  {best_model_params}")

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
        print(f"  MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}")

    print(f"\n  Training time: {(time.time()-start)/60:.1f} min")

    results_df  = pd.DataFrame(fold_results)
    metric_cols = ['MAE', 'MSE', 'RMSE', 'R2']
    mean_row    = {'Fold': 'Mean', **{m: results_df[m].mean() for m in metric_cols}}
    std_row     = {'Fold': 'Std',  **{m: results_df[m].std()  for m in metric_cols}}
    results_df  = pd.concat(
        [results_df, pd.DataFrame([mean_row, std_row])], ignore_index=True)

    n_outer = len(fold_results)
    print(f"\n{'='*65}\n  SUMMARY — {target_col}\n{'='*65}")
    for m in metric_cols:
        mv, sv = mean_row[m], std_row[m]
        print(f"    {m:<5}: {mv:.3f} ± {sv:.4f})")

    freq = Counter(f for fold in selected_features_per_fold for f in fold)
    feature_freq = (
        pd.Series(dict(freq), name='selection_count')
        .reindex(feature_cols, fill_value=0)
        .sort_values(ascending=False))
    feature_freq.index.name = 'feature'

    print(f"\n  Complete Feature Selection Frequency List:")
    for feat, cnt in feature_freq.items():
        print(f"    {cnt:>2}/{n_outer}  {cnt/n_outer*100:4.1f}%  {feat}")

    return results_df, feature_freq, selected_features_per_fold
