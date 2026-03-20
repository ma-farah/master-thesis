# HistGradientBoosting threshold sweep — MRMR feature frequency list
import time
import warnings
import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy import stats
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import OrdinalEncoder
import joblib

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
            encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)
            out[cats_present] = encoder.fit_transform(out[cats_present]).astype(float)
        else:
            out[cats_present] = encoder.transform(out[cats_present]).astype(float)
    return out, encoder


# _____________________________________________________________________________
# HGB threshold sweep — MRMR feature frequency list + Optuna nested CV
# ══════════════════════════════════════════════════════════════════════════════

def run_hgb_threshold_sweep(
    df_combined, feature_freq, target_col='pain_reduction',
    random_state=42, target_transformer=None,
):
    """Sweep over MRMR frequency thresholds using HGB + Optuna nested CV.

    For each threshold (all features, >=5%, >=10%, ..., >=55%):
      - Select features with selection_freq >= threshold from feature_freq
      - Outer 4x5=20 folds, Inner 4x5=20 folds, 50 Optuna trials per outer fold
      - Report mean MAE, RMSE, R2 +/- std and 95% CI across outer folds

    Parameters
    ----------
    df_combined        : pd.DataFrame
    feature_freq       : pd.Series  — output of get_mrmr_frequency()
    target_col         : str
    random_state       : int
    target_transformer : sklearn transformer or None  (e.g. PowerTransformer)

    Returns
    -------
    sweep_df      : pd.DataFrame  — one row per threshold, summary metrics
    sweep_results : list of dict  — includes per-fold results for plotting
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=RuntimeWarning)

    N_TRIALS   = 20
    # 0.0 (all features), then >=10%, then 9 evenly-spaced thresholds up to max frequency
    unique_freqs = sorted(feature_freq[feature_freq >= 0.10].unique())
    indices      = np.linspace(0, len(unique_freqs) - 1, 9, dtype=int)
    THRESHOLDS   = [0.0, 0.10] + [unique_freqs[i] for i in indices if unique_freqs[i] > 0.10]

    y = df_combined[target_col].copy()
    exclude = {'Patient', 'Timepoint', target_col, 'pain_reduction',
               'pain_reduction_pct', 'pain_under_load_reduction',
               'pain_under_load_reduction_pct'}
    feature_cols = [c for c in df_combined.columns if c not in exclude]
    X = df_combined[feature_cols].copy()

    valid = y.notna()
    X, y = X[valid].reset_index(drop=True), y[valid].reset_index(drop=True)

    cat_cols = X.select_dtypes(include=['category', 'object']).columns.tolist()

    outer_cv = RepeatedKFold(n_splits=4, n_repeats=5, random_state=random_state)
    inner_cv = RepeatedKFold(n_splits=4, n_repeats=5, random_state=random_state)

    sweep_results = []
    total_start   = time.time()

    for threshold in THRESHOLDS:
        if threshold == 0.0:
            selected_cols = feature_cols.copy()
            thresh_label  = 'all features'
            folds_equiv   = 0
        else:
            selected_cols = feature_freq[feature_freq >= threshold].index.tolist()
            thresh_label  = f'>={threshold*100:.0f}%'
            folds_equiv   = int(threshold * 20)

        if len(selected_cols) == 0:
            print(f"\n  Threshold {thresh_label} — no features, skipping.")
            continue

        n_features   = len(selected_cols)
        cat_mask_sel = [c in cat_cols for c in selected_cols]

        print(f"\n{'='*65}")
        if threshold == 0.0:
            print(f"  Threshold: ALL features — {n_features} features")
        else:
            print(f"  Features selected in >={folds_equiv}/20 outer folds: {thresh_label}"
                  f" — Number of features: {n_features}")
        print(f"  {selected_cols[:8]}{'...' if n_features > 8 else ''}")
        print(f"{'='*65}")

        fold_results           = []
        best_model_params_list = []
        thresh_start           = time.time()

        for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X), start=1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            if target_transformer is not None:
                pt_fold     = clone(target_transformer)
                y_train_fit = pd.Series(
                    pt_fold.fit_transform(y_train.values.reshape(-1, 1)).ravel(),
                    index=y_train.index)
            else:
                pt_fold, y_train_fit = None, y_train

            X_train_hgb, oe_fold = _prep_for_hgb(X_train, cat_cols)
            X_test_hgb,  _       = _prep_for_hgb(X_test,  cat_cols, encoder=oe_fold)

            inner_splits_fold = list(inner_cv.split(X_train_hgb))

            def _fit_inner(itr, ival, params, sel_cols, cm):
                m = HistGradientBoostingRegressor(
                    **params, random_state=random_state,
                    categorical_features=cm if any(cm) else None)
                m.fit(X_train_hgb.iloc[itr][sel_cols], y_train_fit.iloc[itr])
                return np.sqrt(mean_squared_error(
                    y_train_fit.iloc[ival],
                    m.predict(X_train_hgb.iloc[ival][sel_cols])))

            def model_objective(trial):
                params = dict(
                    max_depth         = trial.suggest_int(  'max_depth',          2,   8),
                    learning_rate     = trial.suggest_float('learning_rate',   1e-3, 0.3, log=True),
                    min_samples_leaf  = trial.suggest_int(  'min_samples_leaf',   5,  30),
                    l2_regularization = trial.suggest_float('l2_regularization', 0.0, 1.0),
                    max_iter          = 300,
                )
                rmses = joblib.Parallel(n_jobs=4, prefer='threads')(
                    joblib.delayed(_fit_inner)(itr, ival, params, selected_cols, cat_mask_sel)
                    for itr, ival in inner_splits_fold)
                return np.mean(rmses)

            model_study = optuna.create_study(direction='minimize')
            model_study.optimize(model_objective, n_trials=N_TRIALS, show_progress_bar=False)

            best_params = model_study.best_params
            best_model_params_list.append(best_params)
            print(f"  Outer Fold {outer_fold:>2}/20 | "
                  f"Best Trial {model_study.best_trial.number+1:>2}/{N_TRIALS}  "
                  f"RMSE={model_study.best_value:.4f}  {best_params}")

            fold_model = HistGradientBoostingRegressor(
                **best_params, random_state=random_state,
                categorical_features=cat_mask_sel if any(cat_mask_sel) else None)
            fold_model.fit(X_train_hgb[selected_cols], y_train_fit)

            preds_raw = fold_model.predict(X_test_hgb[selected_cols])
            preds = (pt_fold.inverse_transform(preds_raw.reshape(-1, 1)).ravel()
                     if pt_fold is not None else preds_raw)

            mae  = mean_absolute_error(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            r2   = r2_score(y_test, preds)
            fold_results.append({'Fold': outer_fold, 'MAE': mae, 'RMSE': rmse, 'R2': r2})

        # ── Threshold summary ─────────────────────────────────────────────────
        res     = pd.DataFrame(fold_results)
        n_outer = len(fold_results)
        t_crit  = stats.t.ppf(0.975, df=n_outer - 1)

        mean_mae,  std_mae  = res['MAE'].mean(),  res['MAE'].std()
        mean_rmse, std_rmse = res['RMSE'].mean(), res['RMSE'].std()
        mean_r2,   std_r2   = res['R2'].mean(),   res['R2'].std()

        ci_mae  = t_crit * std_mae  / np.sqrt(n_outer)
        ci_rmse = t_crit * std_rmse / np.sqrt(n_outer)
        ci_r2   = t_crit * std_r2   / np.sqrt(n_outer)

        elapsed = (time.time() - thresh_start) / 60
        print(f"\n  ── {thresh_label} | {n_features} features | {elapsed:.1f} min ──")
        print(f"    MAE:  {mean_mae:.3f} +/- {std_mae:.4f}  "
              f"(95% CI [{mean_mae-ci_mae:.3f}, {mean_mae+ci_mae:.3f}])")
        print(f"    RMSE: {mean_rmse:.3f} +/- {std_rmse:.4f}  "
              f"(95% CI [{mean_rmse-ci_rmse:.3f}, {mean_rmse+ci_rmse:.3f}])")
        print(f"    R2:   {mean_r2:.3f} +/- {std_r2:.4f}  "
              f"(95% CI [{mean_r2-ci_r2:.3f}, {mean_r2+ci_r2:.3f}])")

        sweep_results.append({
            'threshold':       threshold,
            'threshold_label': thresh_label,
            'n_features':      n_features,
            'mean_MAE':        mean_mae,  'std_MAE':  std_mae,
            'mean_RMSE':       mean_rmse, 'std_RMSE': std_rmse,
            'mean_R2':         mean_r2,   'std_R2':   std_r2,
            'fold_results':    res.to_dict('records'),
        })

    total_elapsed = (time.time() - total_start) / 60
    print(f"\n{'='*65}")
    print(f"  Total time: {total_elapsed:.1f} min")
    print(f"{'='*65}")

    sweep_df = pd.DataFrame([
        {k: v for k, v in r.items() if k != 'fold_results'}
        for r in sweep_results
    ])
    return sweep_df, sweep_results
