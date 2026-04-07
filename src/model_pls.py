import time
import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import OneHotEncoder, TargetEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cross_decomposition import PLSRegression
import joblib
import contextlib, io
import preprocess



# ── Encoding constants ──────────────────────────────────────────────────────
BINARY_MAPS = {
    'gender':     {'m': 1, 'f': 0},
    'overweight': {'ja': 1, 'nein': 0},
}
OHE_COLS        = ['target_volume_side']
TARGET_ENC_COLS = ['diagnosis', 'target_volume']


def encode_categoricals(X_train, y_train, X_test=None, random_state=42,
                        ohe_categories=None):
    """Encode categoricals — fit on X_train/y_train only, transform both.

    Binary:  gender (m=1, f=0), overweight (ja=1, nein=0)
    OHE:     target_volume_side (L, R, B)
    Target:  diagnosis, target_volume  (smoothed for small groups)

    Returns (X_train_enc, X_test_enc | None, encoders dict).
    """
    X_tr = X_train.copy()
    X_te = X_test.copy() if X_test is not None else None
    encoders = {}

    cat_cols = list(BINARY_MAPS.keys()) + OHE_COLS + TARGET_ENC_COLS
    present_cat_cols = [c for c in cat_cols if c in X_tr.columns]

    if present_cat_cols:
        # Vi bruker 'most_frequent' siden missing-andelen er lav
        imputer = SimpleImputer(strategy='most_frequent')
        
        # Fit på train, transform på begge
        X_tr[present_cat_cols] = imputer.fit_transform(X_tr[present_cat_cols])
        if X_te is not None:
            X_te[present_cat_cols] = imputer.transform(X_te[present_cat_cols])
        encoders['cat_imputer'] = imputer

    # 1. Binary mapping
    for col, mapping in BINARY_MAPS.items():
        if col in X_tr.columns:
            str_map = {str(k): v for k, v in mapping.items()}
            X_tr[col] = X_tr[col].astype(str).str.lower().map(str_map)
            if X_te is not None:
                X_te[col] = X_te[col].astype(str).str.lower().map(str_map)

    # 2. One-hot encoding
    ohe_present = [c for c in OHE_COLS if c in X_tr.columns]
    if ohe_present:
        ohe = OneHotEncoder(
            categories=ohe_categories if ohe_categories else 'auto',
            sparse_output=False,
            handle_unknown='ignore')
        train_ohe = pd.DataFrame(
            ohe.fit_transform(X_tr[ohe_present].astype(str)),
            columns=ohe.get_feature_names_out(ohe_present),
            index=X_tr.index)
        X_tr = X_tr.drop(columns=ohe_present).join(train_ohe)
        if X_te is not None:
            test_ohe = pd.DataFrame(
                ohe.transform(X_te[ohe_present].astype(str)),
                columns=ohe.get_feature_names_out(ohe_present),
                index=X_te.index)
            X_te = X_te.drop(columns=ohe_present).join(test_ohe)
        encoders['ohe'] = ohe

    # 3. Target encoding (with automatic smoothing for small groups)
    te_present = [c for c in TARGET_ENC_COLS if c in X_tr.columns]
    if te_present:
        te = TargetEncoder(smooth='auto', random_state=random_state)
        X_tr[te_present] = te.fit_transform(X_tr[te_present].astype(str), y_train)
        if X_te is not None:
            X_te[te_present] = te.transform(X_te[te_present].astype(str))
        encoders['te'] = te

    return X_tr, X_te, encoders


#_________________________________________________________________________________________________
# PLS + MRMR
#_________________________________________________________________________________________________

def pls_mrmr(
    df_combined, target_col='pain_reduction_pct', random_state=42,
    target_transformer=None,
):
    """PLSRegression with MRMR (K + RFCQ params tuned by Optuna) inside each outer CV fold.
      1. Tune K and RFCQ params (n_estimators, max_depth, min_samples_leaf) via Optuna (20 trials)
         on a 75-25 split of X_train. K candidates: [30, 20, 15, 10].
      2. Re-run MRMR on full X_train with best K + best RFCQ params -> selected feature subset.
      3. Inner CV (4×5=20) + Optuna (50 trials) tunes PLSRegression hyperparameter (n_components).

    Returns: results_df, feature_freq, selected_features_per_fold
    """
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

    # Precompute OHE categories for consistent encoding across folds
    ohe_cats = [sorted(X[c].dropna().astype(str).unique())
                for c in OHE_COLS if c in X.columns]
    # Build full list of encoded feature names (for feature frequency tracking)
    all_enc_cols = []
    for c in feature_cols:
        if c in OHE_COLS:
            for v in sorted(X[c].dropna().astype(str).unique()):
                all_enc_cols.append(f"{c}_{v}")
        else:
            all_enc_cols.append(c)

    p = len(feature_cols)
    print(f"\n{'='*80}")
    print(f" Nested CV - PLS + MRMR + Optuna — {target_col}")
    print(f"  n={len(X)}, p={p}")
    print(f"  Outer 4×5=20 | Inner 4×5=20 | Optuna Trials Model={N_TRIALS_MODEL} | Optuna Trials MRMR={N_TRIALS_MRMR}")
    print(f"{'='*80}")

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

        if target_transformer is not None:
            pt_fold = clone(target_transformer)
            y_train_fit = pd.Series(
                pt_fold.fit_transform(y_train.values.reshape(-1, 1)).ravel(),
                index=y_train.index)
        else:
            pt_fold, y_train_fit = None, y_train

        # ── Encode categoricals (fit on train only) ────────────────────────
        X_train_enc, X_test_enc, _ = encode_categoricals(
            X_train, y_train_fit, X_test, random_state, ohe_categories=ohe_cats)
     
        # ── Impute (full encoded set; save imputer for test transform) ─────────
        X_train_mrmr, full_imputer = preprocess.impute_iterative(
            X_train_enc.astype(float), ex_cols=None, iterations=10,
            random_state=random_state, verbose=False)
        
        # ── Step 1: Tune mrmr parameters on 75-25 split of X_train ───────────────────────── 
        X_tr_mrmr, X_val_mrmr, y_tr, y_val = train_test_split(
            X_train_mrmr, y_train_fit, test_size=0.25, random_state=random_state)


        def mrmr_objective(trial):
            k                = trial.suggest_categorical('K',                [10, 15, 20])
            n_estimators     = trial.suggest_categorical('n_estimators',     [100, 200, 300, 400])
            max_depth        = trial.suggest_int('max_depth',                 3, 8)
            min_samples_leaf = trial.suggest_int('min_samples_leaf',          5, 20)

            mrmr_t = MRMR(
                method='RFCQ', max_features=k,
                scoring='neg_mean_squared_error',
                param_grid={'n_estimators':    [n_estimators],
                            'max_depth':        [max_depth],
                            'min_samples_leaf': [min_samples_leaf]},
                cv=5, regression=True, random_state=random_state, n_jobs=-1)
            mrmr_t.fit(X_tr_mrmr, y_tr)
            sel_cols = list(mrmr_t.transform(X_tr_mrmr).columns)

            if len(sel_cols) == 0:
                return 1e6

            scaler_probe = StandardScaler()
            X_tr_s  = scaler_probe.fit_transform(X_tr_mrmr[sel_cols])
            X_val_s = scaler_probe.transform(X_val_mrmr[sel_cols])
            nc = min(5, len(sel_cols), len(y_tr) - 1)
            probe = PLSRegression(n_components=max(1, nc))
            probe.fit(X_tr_s, y_tr)
            return np.sqrt(mean_squared_error(y_val, probe.predict(X_val_s).ravel()))

        mrmr_study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=random_state))
        mrmr_study.optimize(mrmr_objective, n_trials=N_TRIALS_MRMR, show_progress_bar=False)

        best_mrmr = mrmr_study.best_params
        best_k    = best_mrmr['K']
        best_mrmr_params_per_fold.append(best_mrmr)
        print(f"  Best MRMR params: {best_mrmr}  RMSE={mrmr_study.best_value:.4f}")

        # ── Step 2: Re-run MRMR on full X_train with best params ─────────────
        mrmr_full = MRMR(
            method='RFCQ', max_features=best_k,
            scoring='neg_mean_squared_error',
            param_grid={'n_estimators':    [best_mrmr['n_estimators']],
                        'max_depth':        [best_mrmr['max_depth']],
                        'min_samples_leaf': [best_mrmr['min_samples_leaf']]},
            cv=5, regression=True, random_state=random_state, n_jobs=-1)
        
        mrmr_full.fit(X_train_mrmr, y_train_fit)
        selected_cols = list(mrmr_full.transform(X_train_mrmr).columns)
        selected_features_per_fold.append(selected_cols)
        print(f"  {len(selected_cols)} selected features: {selected_cols}")

        # Select from already-imputed data; transform test with same imputer
        X_train_sel = X_train_mrmr[selected_cols]
        arr = full_imputer.transform(X_test_enc.astype(float))
        X_test_mrmr = pd.DataFrame(arr, columns=X_train_mrmr.columns, index=X_test_enc.index)
        X_test_sel  = X_test_mrmr[selected_cols]

        # Scale
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train_sel),
            columns=selected_cols, index=X_train_sel.index)
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test_sel),
            columns=selected_cols, index=X_test_sel.index)

        print('     Running 20 Inner Folds, 50 Optuna Trials...')

        # ── Step 3: Inner CV Optuna for PLS HPs ──────────────────────────────
        inner_splits = list(inner_cv.split(X_train_scaled))

        def _fit_inner_pls(itr, ival, params):
            nc = min(params['n_components'], len(itr) - 1, len(selected_cols))
            m  = PLSRegression(n_components=max(1, nc))
            m.fit(X_train_scaled.iloc[itr], y_train_fit.iloc[itr])
            return np.sqrt(mean_squared_error(
                y_train_fit.iloc[ival],
                m.predict(X_train_scaled.iloc[ival]).ravel()))

        def model_objective(trial):
            n_c_max = len(selected_cols)
            params  = dict(n_components=trial.suggest_int('n_components', 1, n_c_max))
            rmses = joblib.Parallel(n_jobs=4, prefer='threads')(
                joblib.delayed(_fit_inner_pls)(itr, ival, params)
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

        nc_fold   = min(best_model_params['n_components'], len(X_train_scaled) - 1, len(selected_cols))
        fold_model = PLSRegression(n_components=max(1, nc_fold))
        fold_model.fit(X_train_scaled, y_train_fit)

        preds_raw = fold_model.predict(X_test_scaled).ravel()
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
        .reindex(all_enc_cols, fill_value=0)
        .sort_values(ascending=False))
    feature_freq.index.name = 'feature'

    print(f"\n  Complete Feature Selection Frequency List:")
    for feat, cnt in feature_freq.items():
        print(f"    {cnt:>2}/{n_outer}  {cnt/n_outer*100:4.1f}%  {feat}")

    return results_df, feature_freq, selected_features_per_fold


#_________________________________________________________________________________________________
# PLS Feature-Threshold Analysis
#_________________________________________________________________________________________________

def pls_threshold_analysis(
    df_combined, feature_freq, target_col='pain_reduction_pct',
    random_state=42, target_transformer=None):
    """PLSRegression + Optuna nested CV across feature-frequency threshold subsets.

    Evaluates subsets (all features → most-frequent features) with outer 4×5=20 CV
    and inner 4×5=20 CV + Optuna (50 trials). Use the returned sweep_df to plot and
    choose a feature threshold, then pass the chosen feature list to run_tuned_pls.

    Returns
    -------
    sweep_df : pd.DataFrame  — columns: threshold, threshold_label, n_features,
               mean_MAE, std_MAE, mean_RMSE, std_RMSE, mean_R2, std_R2
    """
    import optuna, warnings

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=RuntimeWarning)

    N_TRIALS = 50

    y = df_combined[target_col].copy()
    exclude = {'Patient', 'Timepoint', target_col, 'pain_reduction',
               'pain_reduction_pct', 'pain_under_load_reduction',
               'pain_under_load_reduction_pct'}
    feature_cols = [c for c in df_combined.columns if c not in exclude]
    X = df_combined[feature_cols].copy()

    valid = y.notna()
    X, y  = X[valid].reset_index(drop=True), y[valid].reset_index(drop=True)

    # Precompute OHE categories for consistent encoding across folds
    ohe_cats = [sorted(X[c].dropna().astype(str).unique())
                for c in OHE_COLS if c in X.columns]

    outer_cv = RepeatedKFold(n_splits=4, n_repeats=5, random_state=random_state)
    inner_cv = RepeatedKFold(n_splits=4, n_repeats=5, random_state=random_state)

    sweep_results = []
    total_start   = time.time()

    last_count = int(feature_freq[feature_freq > 0].max())   
    steps = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    
    for step in steps:
        is_last = False
        if step == 0:
            selected_cols = feature_freq.index.tolist()
            thresh_label  = 'all'
            pct_str = ' '
            current = 0
        else:
            selected_cols = feature_freq[feature_freq >= step].index.tolist()
            current = step
        
            # If empty at step, try step + 1
            if len(selected_cols) == 0:
                selected_cols = feature_freq[feature_freq >= step + 1].index.tolist()
                current = step + 1               
                thresh_label = f'>={current}/20'
                pct_str = f'{current/20*100:.0f}%'
            
            # If still empty, skip
            if len(selected_cols) == 0:
                print(f"\n  No features at threshold {step} or {step + 1}.. Skipping.")
                continue
            
            thresh_label = f'>={step}/20'
            pct_str = f'{step/20*100:.0f}%'
            
            if current >= last_count:
                is_last = True

        n_features = len(selected_cols)
        print(f"\n{'='*65}")
        print(f"  Threshold  {thresh_label} ({pct_str}):  {n_features} features")
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

            # Encode (fit on train only)
            X_train_enc, X_test_enc, _ = encode_categoricals(
                X_train, y_train_fit, X_test, random_state, ohe_categories=ohe_cats)

            # Impute full encoded set (fit on train only), then select
            X_train_mrmr, full_imputer = preprocess.impute_iterative(
                X_train_enc.astype(float), ex_cols=None, iterations=10,
                random_state=random_state, verbose=False)
            arr = full_imputer.transform(X_test_enc.astype(float))
            X_test_mrmr = pd.DataFrame(arr, columns=X_train_mrmr.columns, index=X_test_enc.index)
            X_train_sel = X_train_mrmr[selected_cols]
            X_test_sel  = X_test_mrmr[selected_cols]

            # Scale
            scaler = StandardScaler()
            X_train_sc = pd.DataFrame(
                scaler.fit_transform(X_train_sel),
                columns=selected_cols, index=X_train_sel.index)
            X_test_sc = pd.DataFrame(
                scaler.transform(X_test_sel),
                columns=selected_cols, index=X_test_sel.index)


            inner_splits = list(inner_cv.split(X_train_sc))

            def _fit_inner(itr, ival, params):
                nc = min(params['n_components'], len(itr) - 1, len(selected_cols))
                m  = PLSRegression(n_components=max(1, nc))
                m.fit(X_train_sc.iloc[itr], y_train_fit.iloc[itr])
                return np.sqrt(mean_squared_error(
                    y_train_fit.iloc[ival],
                    m.predict(X_train_sc.iloc[ival]).ravel()))

            def model_objective(trial):
                n_c_max = len(selected_cols)
                params  = dict(n_components=trial.suggest_int('n_components', 1, n_c_max))
                rmses = joblib.Parallel(n_jobs=4, prefer='threads')(
                    joblib.delayed(_fit_inner)(itr, ival, params)
                    for itr, ival in inner_splits)
                return np.mean(rmses)

            model_study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=random_state))
            with contextlib.redirect_stderr(io.StringIO()):
                model_study.optimize(model_objective, n_trials=N_TRIALS,
                                     show_progress_bar=False)

            best_params = model_study.best_params
            print(f"  Outer Fold {outer_fold:>2}/20:  Best Trial {model_study.best_trial.number+1:>2}/{N_TRIALS}"
                  f"  RMSE={model_study.best_value:.4f}  {best_params}")

            nc_fold    = min(best_params['n_components'], len(X_train_sc) - 1, len(selected_cols))
            fold_model = PLSRegression(n_components=max(1, nc_fold))
            fold_model.fit(X_train_sc, y_train_fit)

            preds_raw = fold_model.predict(X_test_sc).ravel()
            preds = (pt_fold.inverse_transform(preds_raw.reshape(-1, 1)).ravel()
                     if pt_fold is not None else preds_raw)

            mae  = mean_absolute_error(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            r2   = r2_score(y_test, preds)
            fold_results.append({'Fold': outer_fold, 'MAE': mae, 'RMSE': rmse, 'R2': r2})

        res     = pd.DataFrame(fold_results)
        mean_mae,  std_mae  = res['MAE'].mean(),  res['MAE'].std()
        mean_rmse, std_rmse = res['RMSE'].mean(), res['RMSE'].std()
        mean_r2,   std_r2   = res['R2'].mean(),   res['R2'].std()

        print(f"\n  {thresh_label}  {n_features} features")
        for label, mv, sv in [('MAE', mean_mae, std_mae),
                               ('RMSE', mean_rmse, std_rmse),
                               ('R2', mean_r2, std_r2)]:
            print(f"    {label}: {mv:.3f} ± {sv:.4f}")

        sweep_results.append({
            'threshold':       step,
            'threshold_label': thresh_label,
            'n_features':      n_features,
            'mean_MAE':        mean_mae,  'std_MAE':  std_mae,
            'mean_RMSE':       mean_rmse, 'std_RMSE': std_rmse,
            'mean_R2':         mean_r2,   'std_R2':   std_r2,
        })
        if is_last:
            break

    total_elapsed = (time.time() - total_start) / 60
    print(f"\n{'='*65}")
    print(f"  Total time: {total_elapsed:.1f} min")
    print(f"{'='*65}")

    return pd.DataFrame(sweep_results)


#_________________________________________________________________________________________________
# Final PLS Model
#_________________________________________________________________________________________________

def run_tuned_pls(
    df_combined, feature_list, target_col='pain_reduction', random_state=42,
    target_transformer=None,
):
    """PLSRegression + Optuna nested CV on a fixed feature list.
      1. Inner CV (4×5=20) + Optuna (50 trials) tunes n_components
      2. Train final fold model on X_train → evaluate on X_test
      3. Final model: median n_components across outer folds, trained on full X

    Returns: results_df, final_model, X_final, y_pred, patient_err_df, scaler_final
    """
    import optuna, warnings, statistics

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    N_TRIALS = 50

    y = df_combined[target_col].copy()
    valid = y.notna()

    # feature_list contains encoded column names (e.g. target_volume_side_L);
    # we need ALL original columns so encode_categoricals can produce them.
    exclude = {'Patient', 'Timepoint', target_col, 'pain_reduction',
               'pain_reduction_pct', 'pain_under_load_reduction',
               'pain_under_load_reduction_pct'}
    all_feature_cols = [c for c in df_combined.columns if c not in exclude]
    X = df_combined[all_feature_cols].copy()

    selected_cols = list(feature_list)

    X, y = X[valid].reset_index(drop=True), y[valid].reset_index(drop=True)
    patient_id_map = df_combined.loc[valid, 'Patient'].reset_index(drop=True)

    # OHE categories for consistent encoding across folds
    ohe_cats = [sorted(X[c].dropna().astype(str).unique())
                for c in OHE_COLS if c in X.columns]

    print(f"\n{'='*65}")
    print(f"  PLSRegression + Optuna — {target_col}")
    print(f"  n={len(X)}, p={len(selected_cols)}")
    print(f"  Outer 4×5=20 | Inner 4×5=20 | Optuna trials={N_TRIALS}")
    print(f"{'='*65}")

    outer_cv = RepeatedKFold(n_splits=4, n_repeats=5, random_state=random_state)
    inner_cv = RepeatedKFold(n_splits=4, n_repeats=5, random_state=random_state)

    fold_results           = []
    best_model_params_list = []
    patient_errors         = []
    start = time.time()

    for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X), start=1):
        print(f"\n{'─'*65}")
        print(f"  Outer fold {outer_fold}/{outer_cv.get_n_splits()}")
        print(f"{'─'*65}")

        X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        if target_transformer is not None:
            pt_fold = clone(target_transformer)
            y_train_fit = pd.Series(
                pt_fold.fit_transform(y_train.values.reshape(-1, 1)).ravel(),
                index=y_train.index)
        else:
            pt_fold, y_train_fit = None, y_train

        # Encode (fit on train only)
        X_train_enc, X_test_enc, _ = encode_categoricals(
            X_train, y_train_fit, X_test, random_state, ohe_categories=ohe_cats)

        # Impute full encoded set (fit on train only), then select
        X_train_mrmr, full_imputer = preprocess.impute_iterative(
            X_train_enc.astype(float), ex_cols=None, iterations=10,
            random_state=random_state, verbose=False)
        arr = full_imputer.transform(X_test_enc.astype(float))
        X_test_mrmr = pd.DataFrame(arr, columns=X_train_mrmr.columns, index=X_test_enc.index)
        X_train_sel = X_train_mrmr[selected_cols]
        X_test_sel  = X_test_mrmr[selected_cols]

        # Scale
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train_sel),
            columns=selected_cols, index=X_train_sel.index)
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test_sel),
            columns=selected_cols, index=X_test_sel.index)

        inner_splits = list(inner_cv.split(X_train_scaled))

        def _fit_inner_pls(itr, ival, params):
            nc = min(params['n_components'], len(itr) - 1, len(selected_cols))
            m  = PLSRegression(n_components=max(1, nc))
            m.fit(X_train_scaled.iloc[itr], y_train_fit.iloc[itr])
            return np.sqrt(mean_squared_error(
                y_train_fit.iloc[ival],
                m.predict(X_train_scaled.iloc[ival]).ravel()))

        def model_objective(trial):
            n_c_max = len(selected_cols)
            params  = dict(n_components=trial.suggest_int('n_components', 1, n_c_max))
            rmses = joblib.Parallel(n_jobs=4, prefer='threads')(
                joblib.delayed(_fit_inner_pls)(itr, ival, params)
                for itr, ival in inner_splits)
            return np.mean(rmses)

        def _cb(study, trial):
            if trial.state.name == 'COMPLETE':
                print(f"    Trial {trial.number+1:>3}/{N_TRIALS}: "
                      f"RMSE={trial.value:.4f}  {trial.params}")

        model_study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=random_state))
        with contextlib.redirect_stderr(io.StringIO()):
            model_study.optimize(model_objective, n_trials=N_TRIALS,
                                 callbacks=[_cb], show_progress_bar=False)

        best_model_params = model_study.best_params
        best_model_params_list.append(best_model_params)
        print(f"  Best Trial: {model_study.best_trial.number}  "
              f"RMSE={model_study.best_value:.4f}  {best_model_params}")

        nc_fold    = min(best_model_params['n_components'], len(X_train_scaled) - 1, len(selected_cols))
        fold_model = PLSRegression(n_components=max(1, nc_fold))
        fold_model.fit(X_train_scaled, y_train_fit)

        preds_raw = fold_model.predict(X_test_scaled).ravel()
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
                      .sort_values('mean_mae', ascending=False)
                      .round({'mean_mae': 2}))

    results_df  = pd.DataFrame(fold_results)
    metric_cols = ['MAE', 'MSE', 'RMSE', 'R2']
    mean_row    = {'Fold': 'Mean', **{m: results_df[m].mean() for m in metric_cols}}
    std_row     = {'Fold': 'Std',  **{m: results_df[m].std()  for m in metric_cols}}
    results_df  = pd.concat(
        [results_df, pd.DataFrame([mean_row, std_row])], ignore_index=True)

    print(f"\n{'='*65}\n  SUMMARY — {target_col}\n{'='*65}")
    for m in metric_cols:
        mv, sv = mean_row[m], std_row[m]
        print(f"    {m:<5}: {mv:.3f} ± {sv:.4f}")

    # Final model: median n_components, trained on full X
    if target_transformer is not None:
        pt_final    = clone(target_transformer)
        y_final_fit = pd.Series(
            pt_final.fit_transform(y.values.reshape(-1, 1)).ravel(), index=y.index)
    else:
        pt_final, y_final_fit = None, y

    # Encode full dataset
    X_final_enc, _, _ = encode_categoricals(
        X, y_final_fit, random_state=random_state, ohe_categories=ohe_cats)

    # Impute full encoded set, then select
    X_final_mrmr, _ = preprocess.impute_iterative(
        X_final_enc.astype(float), ex_cols=None, iterations=10,
        random_state=random_state, verbose=False)
    X_final_sel = X_final_mrmr[selected_cols]

    # Scale
    scaler_final = StandardScaler()
    X_final = pd.DataFrame(
        scaler_final.fit_transform(X_final_sel),
        columns=selected_cols, index=X_final_sel.index)

    nc_median   = int(round(statistics.median([p['n_components'] for p in best_model_params_list])))
    hp_final    = {'n_components': min(nc_median, len(X_final) - 1, len(selected_cols))}
    print(f"  Final model hyperparameters (median across outer folds): {hp_final}")

    final_model = PLSRegression(n_components=max(1, hp_final['n_components']))
    final_model.fit(X_final, y_final_fit)

    y_pred_raw = pd.Series(
        final_model.predict(X_final).ravel(),
        index=range(len(X_final)), dtype='float64')
    
    y_pred = (pd.Series(
        pt_final.inverse_transform(y_pred_raw.values.reshape(-1, 1)).ravel(),
        index=y_pred_raw.index, dtype='float64')
              if pt_final is not None else y_pred_raw)

    return (results_df, final_model, X_final, y_pred,
            patient_err_df, scaler_final)
