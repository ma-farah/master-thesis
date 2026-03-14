import time
import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import joblib, os
import contextlib, io

import preprocess


def _encode_categoricals(X):
    """OrdinalEncode all categorical/object columns once before CV.
    Returns fully encoded DataFrame — NaN in numeric cols preserved,
    categorical cols are now numeric. No statistics learned — no leakage."""
    out = X.copy()
    all_non_numeric = out.select_dtypes(
        include=['category', 'object']).columns.tolist()
    if all_non_numeric:
        oe = OrdinalEncoder(
            handle_unknown='use_encoded_value', unknown_value=-1)
        out[all_non_numeric] = oe.fit_transform(
            out[all_non_numeric].astype(str))
    return out.astype(float)


def run_advanced_elasticnet_rent(
    df_combined, target_col='pain_reduction', random_state=42,
    tau_3=0.95, target_transformer=None,
):
    """ElasticNet with Optuna-tuned RENT and Model Hyperparameters.

    Per outer fold:
      0. Encode categoricals once before CV loop — no leakage
      1. Tune RENT HPs via Optuna on 75-25 split of imputed+scaled X_train
      2. Re-run RENT on full imputed+scaled X_train with best HPs
      3. Inner CV (4×5=20) + Optuna (50 trials) tunes ElasticNet HPs
      4. Train final fold model on X_train → evaluate on X_test
      5. Final model: features selected in ≥75% of outer folds, median HPs

    Categorical encoding: done once before CV — no leakage risk.
    Imputation + scaling: fitted on X_train only, applied to X_test.

    Returns: results_df, final_model, X_final, y_pred,
             best_model_params_list, feature_freq, scaler_final
             (scaler_final needed to inverse-transform feature units for SHAP)
    """
    from RENT import RENT
    from sklearn.linear_model import ElasticNet
    import optuna, warnings, statistics
    from collections import Counter

    for cat in [FutureWarning, RuntimeWarning]:
        warnings.filterwarnings('ignore', category=cat, module='RENT')
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    N_TRIALS = 50

    y            = df_combined[target_col].copy()
    exclude      = {'Patient', 'Timepoint', target_col, 'pain_reduction',
                    'pain_reduction_pct', 'pain_under_load_reduction',
                    'pain_under_load_reduction_pct'}
    feature_cols = [c for c in df_combined.columns if c not in exclude]
    X            = df_combined[feature_cols].copy()

    valid = y.notna()
    X, y  = X[valid].reset_index(drop=True), y[valid].reset_index(drop=True)

    # ── Encode categoricals ONCE before CV loop ───────────────────────────────
    # Only converts strings → integers, learns no statistics — no leakage
    X_enc = _encode_categoricals(X)

    print(f"\n{'='*65}")
    print(f"  ElasticNet + Optuna + RENT — {target_col}")
    print(f"  n={len(X)}, p={len(feature_cols)}, τ₃={tau_3}")
    print(f"  Outer 4×5=20 | Inner 4×5=20 | RENT & Optuna trials={N_TRIALS} | K=100")
    print(f"{'='*65}")

    outer_cv = RepeatedKFold(n_splits=4, n_repeats=5, random_state=random_state)
    inner_cv = RepeatedKFold(n_splits=4, n_repeats=5, random_state=random_state)

    fold_results               = []
    best_model_params_list     = []
    selected_features_per_fold = []
    start = time.time()

    for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X_enc), start=1):
        print(f"\n{'─'*65}")
        print(f"  Outer fold {outer_fold}/{outer_cv.get_n_splits()}")
        print(f"{'─'*65}")

        X_train, X_test = X_enc.iloc[train_idx], X_enc.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx],     y.iloc[test_idx]

        # ── Target transform ──────────────────────────────────────────────────
        if target_transformer is not None:
            pt_fold     = clone(target_transformer)
            y_train_fit = pd.Series(
                pt_fold.fit_transform(y_train.values.reshape(-1, 1)).ravel(),
                index=y_train.index)
        else:
            pt_fold, y_train_fit = None, y_train

        # ── Impute numeric NaN — fit on X_train only ──────────────────────────
        X_train_imp, imputer = preprocess.impute_iterative(
            X_train, ex_cols=None, iterations=10,
            random_state=42, verbose=False)
        X_train_imp = pd.DataFrame(
            X_train_imp, columns=feature_cols, index=X_train.index)

        # ── Scale — fit on X_train only ───────────────────────────────────────
        scaler         = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train_imp),
            columns=feature_cols, index=X_train.index)


        # ── X_test — impute + scale using X_train fitted objects ──────────────
        X_test_imp = pd.DataFrame(
            imputer.transform(X_test),              # transform only, no fit
            columns=feature_cols, index=X_test.index)
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test_imp),            # transform only, no fit
            columns=feature_cols, index=X_test.index)

        # ── Step 1: Tune RENT HPs on 75-25 split ─────────────────────────────
        X_tr_rent, X_val_rent, y_tr, y_val = train_test_split(
            X_train_scaled, y_train_fit, test_size=0.25,
            random_state=random_state)
        X_tr_rent_reset = X_tr_rent.reset_index(drop=True)

        def rent_objective(trial):
            c_val    = trial.suggest_float('C',        1e-3, 10,  log=True)
            l1_ratio = trial.suggest_float('l1_ratio', 0.1,  1.0)
            tau_1    = trial.suggest_float('tau_1',    0.6,  0.95)
            tau_2    = trial.suggest_float('tau_2',    0.6,  0.95)

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
            if len(sel_idx) == 0:
                return 1e6

            sel_cols = [feature_cols[i] for i in sel_idx]

            # Probe ElasticNet to score selected features
            probe = ElasticNet(
                alpha=0.1, l1_ratio=l1_ratio,
                max_iter=5000, random_state=random_state)
            probe.fit(X_tr_rent[sel_cols], y_tr)
            preds = probe.predict(X_val_rent[sel_cols])
            return np.sqrt(mean_squared_error(y_val, preds))

        rent_study = optuna.create_study(direction='minimize')
        with contextlib.redirect_stderr(io.StringIO()):
            rent_study.optimize(rent_objective, n_trials=N_TRIALS,
                                n_jobs=1, show_progress_bar=False)

        best_rent = rent_study.best_params
        print(f"  Best RENT RMSE: {rent_study.best_value:.4f}  "
              f"Best Parameters: {best_rent}")

        # ── Step 2: Re-run RENT on full scaled X_train ────────────────────────
        rent_full = RENT.RENT_Regression(
            data=X_train_scaled.reset_index(drop=True),
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

        selected_cols = ([feature_cols[i] for i in sel_idx_outer]
                         if len(sel_idx_outer) > 0 else feature_cols)
        selected_features_per_fold.append(selected_cols)

        suffix = '...' if len(selected_cols) > 8 else ''
        print(f"  RENT Selected: {len(selected_cols)}/{len(feature_cols)} features — "
              f"{selected_cols[:8]}{suffix}")

        # ── Step 3: Inner CV ElasticNet HP tuning ────────────────────────────
        inner_splits = list(inner_cv.split(X_train_scaled))

        def _fit_inner_en(itr, ival, params):
            m = ElasticNet(**params, max_iter=5000, random_state=random_state)
            m.fit(X_train_scaled.iloc[itr][selected_cols],
                  y_train_fit.iloc[itr])
            return np.sqrt(mean_squared_error(
                y_train_fit.iloc[ival],
                m.predict(X_train_scaled.iloc[ival][selected_cols])))

        def model_objective(trial):
            params = dict(
                alpha    = trial.suggest_float('alpha',    1e-4, 10.0, log=True),
                l1_ratio = trial.suggest_float('l1_ratio', 0.0,  1.0),
            )
            rmses = joblib.Parallel(n_jobs=-1, prefer='threads')(
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

        # ── Step 4: Train on full X_train → evaluate on X_test ───────────────
        # X_test already prepared above using X_train fitted imputer + scaler
        fold_model = ElasticNet(
            **best_model_params, max_iter=5000, random_state=random_state)
        fold_model.fit(X_train_scaled[selected_cols], y_train_fit)

        preds_raw = fold_model.predict(X_test_scaled[selected_cols])
        preds     = (pt_fold.inverse_transform(preds_raw.reshape(-1, 1)).ravel()
                     if pt_fold is not None else preds_raw)

        mae  = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2   = r2_score(y_test, preds)
        fold_results.append({
            'Fold': outer_fold, 'MAE': mae, 'MSE': rmse**2, 'RMSE': rmse, 'R2': r2})
        print(f"  Outer Fold {outer_fold} | Features={len(selected_cols)}: {selected_cols}")
        print(f"    MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}")

    print(f"\n  Training time: {(time.time()-start)/60:.1f} min")

    # ── Results summary ───────────────────────────────────────────────────────
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
        print(f"    {m:<5}: {mv:.3f} ± {sv:.4f}   "
              f"(95% CI [{mv-ci:.3f}, {mv+ci:.3f}])")

    # ── Feature selection frequency ───────────────────────────────────────────
    freq         = Counter(f for fold in selected_features_per_fold for f in fold)
    feature_freq = (pd.Series(dict(freq), name='selection_count')
                    .reindex(feature_cols, fill_value=0)
                    .sort_values(ascending=False))
    feature_freq.index.name = 'feature'

    print(f"\n  Top 30 RENT feature-selection frequencies:")
    for feat, cnt in freq.most_common(30):
        print(f"    {cnt:>3}/{n_outer}  {feat}"
              f"{'   (≥75%)' if cnt/n_outer >= 0.75 else ''}")

    if not [f for f, cnt in freq.items() if cnt / n_outer >= 0.75]:
        print("  Warning:  No features met ≥75% threshold — "
              "falling back to top 10 selected features.")

    # ── Final model ───────────────────────────────────────────────────────────
    final_cols = ([f for f, cnt in freq.items() if cnt / n_outer >= 0.75]
                  or [f for f, _ in freq.most_common(10)])
    print(f"\n  Final model: {len(final_cols)} features (≥75%): {final_cols}")

    # Encode → impute → scale full dataset for final model
    X_final_imp, _ = preprocess.impute_iterative(
        X_enc, ex_cols=None, iterations=10,
        random_state=42, verbose=False)
    X_final_imp    = pd.DataFrame(
        X_final_imp, columns=feature_cols, index=X_enc.index)
    scaler_final   = StandardScaler()
    X_final_scaled = pd.DataFrame(
        scaler_final.fit_transform(X_final_imp),
        columns=feature_cols, index=X_enc.index)
    X_final        = X_final_scaled[final_cols]

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

    final_model = ElasticNet(
        **hp_final, max_iter=5000, random_state=random_state)
    final_model.fit(X_final, y_final_fit)

    y_pred_raw = pd.Series(
        final_model.predict(X_final),
        index=range(len(X_final)), dtype='float64')
    y_pred = (pd.Series(
        pt_final.inverse_transform(y_pred_raw.values.reshape(-1, 1)).ravel(),
        index=y_pred_raw.index, dtype='float64')
              if pt_final is not None else y_pred_raw)

    return (results_df, final_model, X_final, y_pred,
            best_model_params_list, feature_freq, scaler_final)