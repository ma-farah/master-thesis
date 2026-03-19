
# MRMR feature frequency list (model-agnostic)
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import RepeatedKFold
from collections import Counter
import time
import warnings
import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from scipy import stats
from sklearn.model_selection import RepeatedKFold
import joblib
import preprocess


def _prep_for_mrmr(X_train, cat_cols, random_state=42):
    """IterativeImpute + OrdinalEncode for MRMR.
    MRMR requires a fully numeric, NaN-free matrix."""
    X_imp, _ = preprocess.impute_iterative(
        X_train, ex_cols=None, iterations=10,
        random_state=random_state, verbose=False)
    out = X_imp.copy()
    if cat_cols:
        oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        out[cat_cols] = oe.fit_transform(out[cat_cols].astype(str))
    return out.astype(float)



def get_mrmr_frequency(
    df_combined, target_col='pain_reduction', random_state=42, K=15,
):
    """Run MRMR feature selection across 20 outer folds and return selection frequencies.

    No model training — purely feature selection to produce a model-agnostic
    frequency list for use across CatBoost, ElasticNet, PLS, and Random Forest.

    Parameters
    ----------
    df_combined : pd.DataFrame
    target_col  : str
    random_state: int
    K           : int   — max features to select per fold (default 15)

    Returns
    -------
    feature_freq : pd.Series
        Index = feature name, values = selection frequency [0.0, 1.0],
        sorted descending. Same format as RENT feature_freq output.
    """
    from feature_engine.selection import MRMR

    y = df_combined[target_col].copy()
    exclude = {'Patient', 'Timepoint', target_col, 'pain_reduction',
               'pain_reduction_pct', 'pain_under_load_reduction',
               'pain_under_load_reduction_pct'}
    feature_cols = [c for c in df_combined.columns if c not in exclude]
    X = df_combined[feature_cols].copy()

    valid = y.notna()
    X, y = X[valid].reset_index(drop=True), y[valid].reset_index(drop=True)

    cat_cols = X.select_dtypes(include=['category', 'object']).columns.tolist()

    print(f"\n{'='*60}")
    print(f"  MRMR Feature Frequency — {target_col}")
    print(f"  n={len(X)}, p={len(feature_cols)}, K={K}")
    print(f"  Outer CV: RepeatedKFold(n_splits=4, n_repeats=5) = 20 folds")
    print(f"{'='*60}")

    outer_cv = RepeatedKFold(n_splits=4, n_repeats=5, random_state=random_state)
    selected_features_per_fold = []

    for fold, (train_idx, _) in enumerate(outer_cv.split(X), start=1):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]

        X_train_mrmr = _prep_for_mrmr(X_train, cat_cols, random_state)

        mrmr_sel = MRMR(
            method='RFCQ',
            max_features=K,
            scoring='neg_mean_squared_error',
            param_grid={'n_estimators': [50, 100, 200, 300, 400, 500], 'max_depth': [2, 3, 4, 5, 6, 7],
                        'min_samples_leaf': [3, 5, 8]},
            cv=5,
            regression=True,
            random_state=random_state,
            n_jobs=-1,
        )
        mrmr_sel.fit(X_train_mrmr, y_train)
        selected_cols = list(mrmr_sel.transform(X_train_mrmr).columns)
        selected_features_per_fold.append(selected_cols)

        print(f" Outer Fold {fold:>2}/20 — {len(selected_cols)} features: {selected_cols[:6]}"
              f"{'...' if len(selected_cols) > 6 else ''}")

    n_folds = len(selected_features_per_fold)
    freq = Counter(f for fold in selected_features_per_fold for f in fold)

    feature_freq = (
        pd.Series({f: cnt / n_folds for f, cnt in freq.items()}, name='selection_freq')
        .reindex(feature_cols, fill_value=0.0)
        .sort_values(ascending=False)
    )
    feature_freq.index.name = 'feature'

    print(f"\n  Feature Frequency List (selected in ≥1 fold):")
    for feat, val in feature_freq.items():
        if val > 0:
            print(f"    {val*100:5.1f}%  {feat}")

    return feature_freq




def _prep_for_enet(X_train, X_test, cat_cols, random_state=42):
    """Impute + OrdinalEncode + StandardScale for ElasticNet.

    ElasticNet requires a fully numeric, NaN-free, scaled matrix.
    Encoder and scaler are fit on X_train only, applied to both.
    """
    X_train_imp, imputer = preprocess.impute_iterative(
        X_train, ex_cols=None, iterations=10, random_state=random_state, verbose=False)
    X_test_imp = pd.DataFrame(
        imputer.transform(X_test), columns=X_test.columns, index=X_test.index) 

    if cat_cols:
        cats = [c for c in cat_cols if c in X_train_imp.columns]
        if cats:
            oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            X_train_imp[cats] = oe.fit_transform(X_train_imp[cats].astype(str))
            X_test_imp[cats]  = oe.transform(X_test_imp[cats].astype(str))

    X_train_out = X_train_imp.astype(float)
    X_test_out  = X_test_imp.astype(float)

    scaler      = StandardScaler()
    X_train_out = pd.DataFrame(
        scaler.fit_transform(X_train_out), columns=X_train_out.columns)
    X_test_out  = pd.DataFrame(
        scaler.transform(X_test_out), columns=X_test_out.columns)

    return X_train_out, X_test_out


# _____________________________________________________________________________
# ElasticNet threshold sweep — MRMR feature frequency list + Optuna nested CV
# ══════════════════════════════════════════════════════════════════════════════

def run_enet_threshold(
    df_combined, feature_freq, target_col='pain_reduction',
    random_state=42, target_transformer=None,
):
    """Sweep over MRMR frequency thresholds using ElasticNet + Optuna nested CV.

    For each threshold (all features, >=10%, evenly spaced to max frequency):
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

    N_TRIALS = 50

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

        n_features = len(selected_cols)

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

            X_train_enc, X_test_enc = _prep_for_enet(
                X_train[selected_cols], X_test[selected_cols], cat_cols, random_state)

            inner_splits_fold = list(inner_cv.split(X_train_enc))

            def _fit_inner(itr, ival, params):
                m = ElasticNet(**params, random_state=random_state, max_iter=5000)
                m.fit(X_train_enc.iloc[itr], y_train_fit.iloc[itr])
                return np.sqrt(mean_squared_error(
                    y_train_fit.iloc[ival],
                    m.predict(X_train_enc.iloc[ival])))

            def model_objective(trial):
                params = dict(
                    alpha    = trial.suggest_float('alpha',    1e-4, 10.0, log=True),
                    l1_ratio = trial.suggest_float('l1_ratio', 0.0,  1.0),
                )
                rmses = joblib.Parallel(n_jobs=-1, prefer='threads')(
                    joblib.delayed(_fit_inner)(itr, ival, params)
                    for itr, ival in inner_splits_fold)
                return np.mean(rmses)

            def _cb(study, trial):
                if trial.state.name == 'COMPLETE':
                    print(f"    Trial {trial.number+1:>2}/{N_TRIALS}: "
                          f"RMSE={trial.value:.4f}  {trial.params}")

            model_study = optuna.create_study(direction='minimize')
            model_study.optimize(model_objective, n_trials=N_TRIALS,
                                 callbacks=[_cb], show_progress_bar=False)

            best_params = model_study.best_params
            best_model_params_list.append(best_params)
            print(f"  Outer Fold {outer_fold:>2}/20 | "
                  f"Best Trial {model_study.best_trial.number+1:>2}/{N_TRIALS}  "
                  f"RMSE={model_study.best_value:.4f}  {best_params}")

            fold_model = ElasticNet(**best_params, random_state=random_state, max_iter=5000)
            fold_model.fit(X_train_enc, y_train_fit)

            preds_raw = fold_model.predict(X_test_enc)
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
