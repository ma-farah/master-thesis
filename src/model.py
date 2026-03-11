# Modeling functions — baseline and advanced regressors
import time
import optuna
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import HistGradientBoostingRegressor
from catboost import CatBoostRegressor, Pool
# shap is imported lazily inside plot_shap_regressor / plot_shap_pipeline
# to avoid kernel crashes on Windows with restricted execution policies.
import joblib, os
import contextlib, io


# ── Constants ─────────────────────────────────────────────────────────────────


# Metadata / date columns never used as model features.
CL_MODEL_DROP_COLS = ['Date', 'date', 'response', 'response_category', 'improvement_percent', 'measurement_timepoint', 'pain_scale', 'pain_under_load']

# Pain questionnaire / non-feature clinical columns — excluded from modeling.
CL_QUESTIONNAIRE_COLS = [
    'pain_night', 'pain_daytime', 'pain_at_rest', 'morning_stiffness',
    'pain_points',   # high correlation with target_volume  
]

# path to save models
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')



def prepare_model_input(df, target_col):
    """Strip non-feature columns, keeping Patient, Timepoint, model features, and target_col.

    Call this once on df_immu_alone / df_combined before any model function.

    Removes
    -------
    - Metadata/date columns  (Date, date, measurement_timepoint)
    - Pain questionnaire / non-feature cols (pain_under_load, pain_night,
                               pain_daytime, pain_at_rest, morning_stiffness, pain_points)
    - Leaky outcome columns   (anything matching CL_MODEL_LEAKY_PATTERNS —
                               response*, improvement_percent, *_reduction*)
      → target_col is always preserved even if it matches a leaky pattern.

    Parameters
    ----------
    df         : pd.DataFrame  Modeling dataset (df_immu_alone or df_combined).
    target_col : str           The single regression target to keep.

    Returns
    -------
    pd.DataFrame — Patient, Timepoint, feature columns, and target_col only.
    """
    to_drop = set(CL_MODEL_DROP_COLS + CL_QUESTIONNAIRE_COLS)
    # remove columns that specifically match to_drop:
    drop = {c for c in df.columns if c in to_drop}
    if drop:
        print(f"  prepare_model_input: dropping {len(drop)} cols — {sorted(drop)}")
    return df.drop(columns=list(drop))


# ══════════════════════════════════════════════════════════════════════════════
# DATASET CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

def construct_datasets_targets(df1, column_name, timepoints):
    """Compute per-patient regression targets from a clinical column across two timepoints.

    For column_name and timepoints [t_a, t_b], computes per patient:
      - {col}_t{ta}            : raw baseline value (T_a)
      - {col}_t{tb}            : raw post-treatment value (T_b)  ← leaky, for reference only
      - {col}_reduction        : absolute reduction  = value_ta - value_tb
      - {col}_reduction_pct    : percent reduction   = reduction / value_ta × 100

    Only patients that satisfy ALL of the following are included:
      - have a non-NaN measurement at T_a
      - have a non-NaN measurement at T_b
      - have a non-NaN computed reduction (i.e. no division-by-zero when value_ta == 0)

    Prints a summary of target distributions and the list of eligible patient IDs.

    Parameters
    ----------
    df1         : pd.DataFrame  Cleaned clinical dataset (df_cl_vis).
                                Must contain 'Patient', 'Timepoint', and column_name.
    column_name : str           Column to build targets from, e.g. 'pain_scale',
                                'pain_daytime', 'pain_under_load'.
    timepoints  : list[int]     [t_a, t_b] — reduction is computed as t_a minus t_b.
                                Typically [1, 2] (baseline → first follow-up).

    Returns
    -------
    targets : pd.DataFrame
        One row per eligible patient with columns:
          Patient, {col}_t{ta}, {col}_t{tb}, {prefix}_reduction, {prefix}_reduction_pct
        where prefix = column_name with '_scale' stripped (e.g. 'pain_scale' → 'pain').
        Patients with NaN in any computed target column are excluded.
    """
    t_a, t_b = timepoints[0], timepoints[1]
    col_ta  = f'{column_name}_t{t_a}'
    col_tb  = f'{column_name}_t{t_b}'

    # Strip '_scale' from the prefix so 'pain_scale' becomes 'pain_reduction'
    # and 'pain_reduction_pct' rather than 'pain_scale_reduction[_pct]'.
    prefix  = column_name.replace('_scale', '')
    col_red = f'{prefix}_reduction'
    col_pct = f'{prefix}_reduction_pct'

    # Extract the column at each timepoint (one row per patient, drop duplicates)
    ta_vals = (
        df1[df1['Timepoint'] == t_a][['Patient', column_name]]
        .rename(columns={column_name: col_ta})
        .drop_duplicates('Patient')
        .reset_index(drop=True)
    )
    tb_vals = (
        df1[df1['Timepoint'] == t_b][['Patient', column_name]]
        .rename(columns={column_name: col_tb})
        .drop_duplicates('Patient')
        .reset_index(drop=True)
    )

    # Inner join: keep only patients present at BOTH timepoints with non-NaN values
    targets = ta_vals.merge(tb_vals, on='Patient', how='inner')
    targets = targets.dropna(subset=[col_ta, col_tb]).reset_index(drop=True)

    # Absolute reduction (positive = improvement)
    targets[col_red] = targets[col_ta] - targets[col_tb]

    # Percent reduction: set to NaN when baseline is 0 to avoid division-by-zero
    targets[col_pct] = np.where(
        targets[col_ta] != 0,
        (targets[col_ta] - targets[col_tb]) / targets[col_ta] * 100,
        np.nan,
    )

    # Drop any patient whose computed target columns contain NaN
    # (handles division-by-zero and any other edge cases)
    targets = targets.dropna(subset=[col_red, col_pct]).reset_index(drop=True)

    # Print summary
    eligible = sorted(targets['Patient'].tolist())
    print(f"\n{'='*60}")
    print(f"  Targets: '{column_name}'  (T{t_a} → T{t_b})")
    print(f"{'='*60}")
    print(f"  Patients with T{t_a} values    : {ta_vals[col_ta].notna().sum()}")
    print(f"  Patients with T{t_b} values    : {tb_vals[col_tb].notna().sum()}")
    print(f"  Eligible (non-NaN both, n)    : {len(targets)}")
    print(f"  Eligible patient IDs          : {eligible}")
    print(f"\n  Target distributions:")
    for c in [col_red, col_pct]:
        s = targets[c]
        print(f"    {c:<42s}  mean={s.mean():.3f}  std={s.std():.3f}"
              f"  [{s.min():.3f}, {s.max():.3f}]")

    return targets



def create_model_datasets(df_cl, df_im, targets, timepoints):
    """Create wide-format modeling datasets from clinical and immunological data.

    Patients with NaN in all target columns after merging are excluded 

    Parameters
    ----------
    df_cl      : pd.DataFrame  Cleaned clinical dataset 
                               Must contain 'Patient', 'Timepoint', and clinical features.
    df_im      : pd.DataFrame  Cleaned Immunological dataset 
                               Must contain 'Patient', 'Timepoint', and immu features.
    targets    : pd.DataFrame  Output from construct_datasets_targets().
                               Must contain 'Patient' + target columns.
    timepoints : list[int]     [t_a, t_b] to define the immunological difference direction.
                               Typically [1, 2].

    Returns
    -------
    df_combined : pd.DataFrame
        One row per patient: immunological T_b−T_a difference features
        + clinical T_a baseline features + target columns.
    """
    t_a, t_b = timepoints[0], timepoints[1]
    id_cols  = {'Patient', 'Timepoint', 'Date', 'date', 'measurement_timepoint'}

    # ── IMMUNOLOGICAL: T_a − T_b differences only (one row per patient) ────────

    # Restrict to the two timepoints of interest
    df_im_tp = df_im[df_im['Timepoint'].isin([t_a, t_b])].copy()

    # Identify patients that have measurements at BOTH timepoints
    tp_counts     = df_im_tp.groupby('Patient')['Timepoint'].nunique()
    patients_both = tp_counts[tp_counts == 2].index
    df_im_tp      = df_im_tp[df_im_tp['Patient'].isin(patients_both)]

    # Feature columns = everything except ID columns
    im_feat_cols = [c for c in df_im_tp.columns if c not in id_cols]

    # Extract T_a and T_b separately; rename columns with timepoint suffix (temporary)
    df_im_ta = (
        df_im_tp[df_im_tp['Timepoint'] == t_a][['Patient'] + im_feat_cols]
        .rename(columns={c: f'{c}_t{t_a}' for c in im_feat_cols})
        .reset_index(drop=True)
    )
    df_im_tb = (
        df_im_tp[df_im_tp['Timepoint'] == t_b][['Patient'] + im_feat_cols]
        .rename(columns={c: f'{c}_t{t_b}' for c in im_feat_cols})
        .reset_index(drop=True)
    )

    # Merge to align T_a and T_b rows; compute difference; drop raw T_a and T_b columns
    df_im_merged = df_im_ta.merge(df_im_tb, on='Patient', how='inner')
    diff_cols = {}
    for c in im_feat_cols:
        col_name         = f'{c}_t{t_b}_minus_t{t_a}'
        diff_cols[c]     = col_name
        df_im_merged[col_name] = df_im_merged[f'{c}_t{t_b}'] - df_im_merged[f'{c}_t{t_a}']

    # Keep only Patient + difference columns (discard raw T_a and T_b feature columns)
    df_im_wide = df_im_merged[['Patient'] + list(diff_cols.values())].copy()

    # ── CLINICAL: T_a baseline rows only ─────────────────────────────────────────
    # df_cl contains all modeling columns at this point; prepare_model_input()
    # is called downstream (before model functions) to strip non-feature columns.

    cl_feat_cols = [c for c in df_cl.columns if c not in id_cols]

    df_cl_t1 = (
        df_cl[df_cl['Timepoint'] == t_a][['Patient'] + cl_feat_cols]
        .drop_duplicates('Patient')
        .reset_index(drop=True)
    )

    print(f"\n  Clinical features: {len(cl_feat_cols)} (pain/leaky cols pre-filtered upstream)")

    # ── TARGETS: include baseline (_t{t_a}), exclude leaky post-treatment (_t{t_b}) ──
    leaky_tp_cols = [c for c in targets.columns if c.endswith(f'_t{t_b}')]
    target_merge  = ['Patient'] + [c for c in targets.columns
                                   if c != 'Patient' and c not in leaky_tp_cols]

    # ── MERGE into final dataset ──────────────────────────────────────────────

    # Combined: immu difference features + clinical T_a baseline + target columns
    df_combined = (
        df_im_wide
        .merge(df_cl_t1, on='Patient', how='inner')
        .merge(targets[target_merge], on='Patient', how='inner')
    )

    baseline_cols = [c for c in target_merge if c.endswith(f'_t{t_a}')]

    # Drop leaky / metadata columns
    drop_cols = set(CL_MODEL_DROP_COLS + CL_QUESTIONNAIRE_COLS)
    drop = {c for c in df_combined.columns if c in drop_cols}
    if drop:
        print(f"  prepare_model_input: dropping {len(drop)} cols — {sorted(drop)}")
        df_combined = df_combined.drop(columns=list(drop), errors='ignore')

    # Drop baseline target values (e.g. pain_scale_t1) — regression-to-mean confound
    baseline_present = [c for c in baseline_cols if c in df_combined.columns]
    if baseline_present:
        df_combined = df_combined.drop(columns=baseline_present)
        print(f"  Dropped baseline target cols : {baseline_present}")

    print(f"\nModel dataset ready (T{t_a}–T{t_b} immunological differences + clinical baseline):")
    print(f"  Immunological diff features : {len(diff_cols)}  "
          f"(one T{t_b}−T{t_a} diff per original feature)")
    print(f"  Clinical baseline features  : {len(cl_feat_cols)}")
    print(f"  df_combined : shape={df_combined.shape}, "
          f"patients={df_combined['Patient'].nunique()}")

    return df_combined



# ══════════════════════════════════════════════════════════════════════════════
# BASELINE CATBOOST
# ══════════════════════════════════════════════════════════════════════════════

def regression_metrics(y_true, y_pred):
    """Compute standard regression metrics for a single prediction array.

    Parameters
    ----------
    y_true : array-like   Ground-truth target values.
    y_pred : array-like   Model-predicted values, same length as y_true.

    Returns
    -------
    dict with keys 'MAE', 'MSE', 'RMSE', 'R2' (float values).
    """
    # Compute each metric individually so callers can inspect any subset
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, y_pred)
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}



def run_catboost_regressor(df_model, target_col, name,
                           n_splits=5, n_repeats=5, random_state=42,
                           target_transformer=None):
    """Run a baseline CatBoostRegressor with RepeatedKFold cross-validation.
    Parameters uses CPU as task type
    ----------
    df_model     : pd.DataFrame   Dataset containing features and target column.
    target_col   : str            Name of the regression target column.
    name         : str            Label used in printed output and summaries.
    n_splits     : int            Number of CV folds (default 5).
    n_repeats    : int            Number of CV repetitions (default 5).
    random_state : int            Seed for RepeatedKFold and CatBoost (default 42).

    Returns
    -------
    results_df : pd.DataFrame       Per-fold MAE/MSE/RMSE/R2 plus Mean and Std rows.
    model      : CatBoostRegressor  Model trained on the final CV fold.
    X          : pd.DataFrame       Feature matrix used (rows with non-NaN target).
    y_pred     : pd.Series          Out-of-fold predictions aligned to X's index.
    """
    # Build the exclusion set: ID columns + target_col.

    y = df_model[target_col].copy() # target
    # exlude id features and other targets
    exclude = ['Patient', 'Timepoint', target_col, 'pain_reduction', 'pain_reduction_pct', 'pain_under_load_reduction', 'pain_under_load_reduction_pct']
    feature_cols = [c for c in df_model.columns if c not in exclude]
    X = df_model[feature_cols].copy()

    valid = y.notna()
    X, y = X[valid].reset_index(drop=True), y[valid].reset_index(drop=True)

    # Convert category dtype to str so CatBoost treats them as categorical features
    for col in X.select_dtypes(include=['category', 'object']).columns:
        X[col] = X[col].astype(str)
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()

    # Print run header with dataset dimensions and CV configuration
    print(f"\n{'='*65}")
    print(f"  CatBoost Regressor Baseline — {name}")
    print(f"  Target : {target_col}")
    print(f"  Samples: {len(X)},  Features: {len(feature_cols)}")
    print(f"  CV     : {n_splits}-fold × {n_repeats} repeats = {n_splits * n_repeats} fits")
    print(f"{'='*65}")

    # Initialise CV splitter and containers for per-fold results and predictions
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    fold_results = []
    y_pred = pd.Series(np.nan, index=range(len(X)), dtype='float64')

    # Train one CatBoostRegressor per fold, collect metrics, and store predictions
    for fold, (train_idx, test_idx) in enumerate(rkf.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        if target_transformer is not None:
            pt_fold     = clone(target_transformer)
            y_train_fit = pd.Series(
                pt_fold.fit_transform(y_train.values.reshape(-1, 1)).ravel(),
                index=y_train.index,
            )
        else:
            pt_fold     = None
            y_train_fit = y_train

        model = CatBoostRegressor(
            iterations=300,
            loss_function='RMSE',
            random_seed=random_state,
            task_type='CPU', 
            thread_count=-1, #using all cores in paralell
            verbose=0,
        )
        model.fit(Pool(X_train, y_train_fit, cat_features=cat_cols))

        preds_raw = model.predict(X_test)
        preds = (pt_fold.inverse_transform(preds_raw.reshape(-1, 1)).ravel()
                 if pt_fold is not None else preds_raw)
        y_pred.iloc[test_idx] = preds

        # Metrics computed in original-space (inverse-transformed if transformer provided)
        mae  = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mse  = rmse ** 2
        r2   = r2_score(y_test, preds)
        m = {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}
        fold_results.append({'Fold': fold + 1, **m})
        print(f"  Fold {fold+1:>2}: MAE={m['MAE']:.3f}  MSE={m['MSE']:.3f}  "
              f"RMSE={m['RMSE']:.3f}  R²={m['R2']:.3f}")

    # Append Mean and Std summary rows to the per-fold results DataFrame
    results_df = pd.DataFrame(fold_results)
    metric_cols = ['MAE', 'MSE', 'RMSE', 'R2']
    mean_row = {'Fold': 'Mean', **{m: results_df[m].mean() for m in metric_cols}}
    std_row  = {'Fold': 'Std',  **{m: results_df[m].std()  for m in metric_cols}}
    results_df = pd.concat(
        [results_df, pd.DataFrame([mean_row, std_row])], ignore_index=True)

    # Print the mean ± std ± 95% CI summary for each metric
    n_folds = n_splits * n_repeats
    t_crit  = stats.t.ppf(0.975, df=n_folds - 1)
    print(f"\n  Summary ({n_splits}x{n_repeats} CV, 95% CI):")
    for m in metric_cols:
        mv  = results_df.loc[results_df['Fold'] == 'Mean', m].iloc[0]
        sv  = results_df.loc[results_df['Fold'] == 'Std',  m].iloc[0]
        ci  = t_crit * sv / np.sqrt(n_folds)
        print(f"    {m:<5}: {mv:.3f} ± {sv:.4f}  (95% CI [{mv - ci:.3f}, {mv + ci:.3f}])")

    return results_df, model, X, y_pred



def plot_shap_regressor(model, X, name):
    """SHAP bar + beeswarm plots for a fitted CatBoostRegressor."""
    import shap
    print(f"\n=== SHAP Analysis: {name} ===")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    shap.summary_plot(shap_values, X, plot_type="bar", show=False, max_display=20)
    plt.title(f"SHAP Feature Importance — {name}")
    plt.tight_layout()
    plt.show()

    shap.summary_plot(shap_values, X, show=False, max_display=20)
    plt.title(f"SHAP Beeswarm — {name}")
    plt.tight_layout()
    plt.show()

    return shap_values


def print_regression_summary(results_dict, target_col):
    """Print a mean ± std summary table across all datasets for a given target."""
    metric_cols = ['MAE', 'MSE', 'RMSE', 'R2']
    rows = []
    for ds_name, res_df in results_dict.items():
        fold_rows = res_df[~res_df['Fold'].isin(['Mean', 'Std'])]
        row = {'Dataset': ds_name}
        for m in metric_cols:
            mv = fold_rows[m].mean()
            sv = fold_rows[m].std()
            row[m] = f"{mv:.3f} ± {sv:.4f}"
        rows.append(row)
    summary = pd.DataFrame(rows)
    print(f"\n{'='*75}")
    print(f"  CATBOOST BASELINE SUMMARY — Target: {target_col}  (mean ± std)")
    print(f"{'='*75}")
    print(summary.to_string(index=False))
    return summary


# ══════════════════════════════════════════════════════════════════════════════
# ADVANCED CATBOOST + RENT FEATURE SELECTION  (Nested CV + Optuna)
# ══════════════════════════════════════════════════════════════════════════════
def run_advanced_catboost_rent(
    df_combined, target_col='pain_reduction_pct', random_state=42,
    tau_3=0.95, target_transformer=None,
):
    """CatBoostRegressor with Optuna-tuned RENT + nested CV.
    Per outer fold:
      1. Tune RENT HPs (C, l1_ratio, τ₁, τ₂) via Optuna on a 75-25 split of X_train — 20 trials, K=100.
      2. Re-run RENT on full X_train with best HPs → selected feature subset.
      3. Inner CV (4×5=20) + Optuna (20 trials) tunes CatBoost HPs on selected features.
      4. Best trial params → train on full X_train → evaluate on X_test (inverse-transformed).
      5. Final model: features selected in ≥50% of outer folds, median HPs across outer folds.

    Returns: results_df, final_model, X_final, y_pred,
             selected_features_per_fold, best_rent_params_list,
             best_model_params_list, feature_freq
    """
    import optuna, warnings, statistics
    from collections import Counter
    from sklearn.model_selection import train_test_split
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OrdinalEncoder
    from RENT import RENT

    for cat in [FutureWarning, RuntimeWarning]:
        warnings.filterwarnings('ignore', category=cat, module='RENT')
    for pat in ['.*less than 75% GPU memory.*', '.*joblib.*', '.*loky.*']:
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
    for col in X.select_dtypes(include=['category', 'object']).columns:
        X[col] = X[col].astype(str)
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()

    print(f"\n{'='*65}")
    print(f"  CatBoost + Optuna + RENT — {target_col}")
    print(f"  n={len(X)}, p={len(feature_cols)}, τ₃={tau_3}")
    print(f"  Outer 4×5=20 | Inner 4×5=20 | RENT & model trials={N_TRIALS} | K=100")
    print(f"{'='*65}")

    outer_cv = RepeatedKFold(n_splits=4, n_repeats=5, random_state=random_state)
    inner_cv = RepeatedKFold(n_splits=4, n_repeats=5, random_state=random_state)

    fold_results, best_rent_params_list   = [], []
    best_model_params_list, selected_features_per_fold = [], []
    start = time.time()

    def _encode_for_rent(df_feat, cols):
        """OrdinalEncode + median-impute a feature DataFrame for RENT input."""
        df_enc = df_feat.copy()
        mask   = [c for c in df_feat.columns if df_feat[c].dtype == object]
        if mask:
            oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            df_enc[mask] = oe.fit_transform(df_feat[mask])
        imp = SimpleImputer(strategy='median')
        return pd.DataFrame(imp.fit_transform(df_enc.astype(float)), columns=cols)

    for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X), start=1):
        print(f"\n{'─'*65}")
        print(f"  Outer fold {outer_fold}/{outer_cv.get_n_splits()}")
        print(f"{'─'*65}")

        X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        if target_transformer is not None:
            pt_fold     = clone(target_transformer)
            y_train_fit = pd.Series(
                pt_fold.fit_transform(y_train.values.reshape(-1, 1)).ravel(),
                index=y_train.index)
        else:
            pt_fold, y_train_fit = None, y_train

        # ── Step 1: Tune RENT HPs on 75-25 split of X_train ──────────────────
        print(f"  [Step 1] RENT HP tuning ({N_TRIALS} trials, 75-25 split)")
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train_fit, test_size=0.25, random_state=random_state)
        X_tr_rent = _encode_for_rent(X_tr, feature_cols)

        def rent_objective(trial):
            c_val    = trial.suggest_float('C',        1e-3, 10,  log=True)
            l1_ratio = trial.suggest_float('l1_ratio', 0.1,  1.0)
            tau_1    = trial.suggest_float('tau_1',    0.6,  0.9)
            tau_2    = trial.suggest_float('tau_2',    0.6,  0.9)
            rent_t   = RENT.RENT_Regression(
                data=X_tr_rent, target=y_tr.values, feat_names=feature_cols,
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

            probe    = CatBoostRegressor(
                iterations=300, depth=5, random_seed=random_state,
                cat_features=[c for c in cat_cols if c in sel_cols],
                task_type='GPU', devices='0', gpu_ram_part=0.6, logging_level='Silent')
            with contextlib.redirect_stderr(io.StringIO()):
                probe.fit(X_tr[sel_cols], y_tr)
            return np.sqrt(mean_squared_error(y_val, probe.predict(X_val[sel_cols])))

        rent_study = optuna.create_study(direction='minimize')
        with contextlib.redirect_stderr(io.StringIO()):
            rent_study.optimize(rent_objective, n_trials=N_TRIALS, show_progress_bar=False)
        
        best_rent = rent_study.best_params
        best_rent_params_list.append(best_rent)
        print(f"  Best RENT RMSE={rent_study.best_value:.4f}  {best_rent}")

        # ── Step 2: Re-run RENT on full X_train with best HPs ────────────────
        print(f"  [Step 2] RENT on full X_train with best HPs")
        X_train_rent = _encode_for_rent(X_train, feature_cols)

        rent_full    = RENT.RENT_Regression(
            data=X_train_rent, target=y_train_fit.values, feat_names=feature_cols,
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
        cat_cols_inner = [c for c in cat_cols if c in selected_cols]
        selected_features_per_fold.append(selected_cols)

        suffix = '...' if len(selected_cols) > 8 else ''
        print(f"  RENT Selected : {len(selected_cols)}/{len(feature_cols)} features — "
              f"{selected_cols[:8]}{suffix}")

        # ── Step 3: Inner CV + Optuna — CatBoost HPs ─────────────────────────
        print(f"  [Step 3] CatBoost HP tuning with 20 trials, on 20 inner folds each)") #fix typo

        # Pre-compute splits once — reused across all Optuna trials
        inner_splits = list(inner_cv.split(X_train))

        def _fit_inner(itr, ival, params):
            m = CatBoostRegressor(
                iterations=300, **params, cat_features=cat_cols_inner,
                random_seed=random_state, task_type='CPU', thread_count=1,
                logging_level='Silent')
            with contextlib.redirect_stderr(io.StringIO()):
                m.fit(X_train.iloc[itr][selected_cols], y_train_fit.iloc[itr])
            return np.sqrt(mean_squared_error(
                y_train_fit.iloc[ival],
                m.predict(X_train.iloc[ival][selected_cols])))

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
                                 callbacks=[_cb], show_progress_bar=False)
        best_model_params = model_study.best_params
        best_model_params_list.append(best_model_params)
        print(f"  Best Trial: {model_study.best_trial.number} - RMSE={model_study.best_value:.4f}  {best_model_params}")

        # ── Step 4: Train on full X_train → eval on X_test ───────────────────
        fold_model = CatBoostRegressor(
            iterations=300, **best_model_params, cat_features=cat_cols_inner,
            random_seed=random_state, task_type='GPU', devices='0',
            gpu_ram_part=0.6, logging_level='Silent')
        with contextlib.redirect_stderr(io.StringIO()):
            fold_model.fit(X_train[selected_cols], y_train_fit)

        preds_raw = fold_model.predict(X_test[selected_cols])
        preds     = (pt_fold.inverse_transform(preds_raw.reshape(-1, 1)).ravel()
                     if pt_fold is not None else preds_raw)

        mae  = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2   = r2_score(y_test, preds)
        fold_results.append({'Fold': outer_fold, 'MAE': mae, 'MSE': rmse**2, 'RMSE': rmse, 'R2': r2})
        print(f"  Outer {outer_fold} | features={len(selected_cols)}: {selected_cols}")
        print(f"    MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}")

    print(f"\n  Training time: {(time.time()-start)/60:.1f} min")

    # ── Results summary ───────────────────────────────────────────────────────
    results_df  = pd.DataFrame(fold_results)
    metric_cols = ['MAE', 'MSE', 'RMSE', 'R2']
    mean_row    = {'Fold': 'Mean', **{m: results_df[m].mean() for m in metric_cols}}
    std_row     = {'Fold': 'Std',  **{m: results_df[m].std()  for m in metric_cols}}
    results_df  = pd.concat([results_df, pd.DataFrame([mean_row, std_row])], ignore_index=True)

    n_outer = len(fold_results)
    t_crit  = stats.t.ppf(0.975, df=n_outer - 1)
    print(f"\n{'='*65}\n  SUMMARY — {target_col}  (4×5 outer CV, 95% CI)\n{'='*65}")
    for m in metric_cols:
        mv, sv = mean_row[m], std_row[m]
        ci = t_crit * sv / np.sqrt(n_outer)
        print(f"    {m:<5}: {mv:.3f} ± {sv:.4f}  (95% CI [{mv-ci:.3f}, {mv+ci:.3f}])")

    # ── Feature selection frequency ───────────────────────────────────────────
    freq         = Counter(f for fold in selected_features_per_fold for f in fold)
    feature_freq = (pd.Series(dict(freq), name='selection_count')
                    .reindex(feature_cols, fill_value=0)
                    .sort_values(ascending=False))
    feature_freq.index.name = 'feature'
    print(f"\n  RENT feature selection frequency ({n_outer} outer folds):")
    for feat, cnt in freq.most_common():
        print(f"    {cnt:>3}/{n_outer}  {feat}{'  ◀ (≥50%)' if cnt/n_outer >= 0.5 else ''}") # bytte til valgte i hver i fold

    # ── Final model ───────────────────────────────────────────────────────────
    final_cols = ([f for f, cnt in freq.items() if cnt / n_outer >= 0.5]
                  or [f for f, _ in freq.most_common(10)])
    print(f"\n  Final model: {len(final_cols)} features (≥50%): {final_cols}")

    X_final        = X[final_cols]
    cat_cols_final = [c for c in cat_cols if c in final_cols]

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

    final_model = CatBoostRegressor(
        iterations=300, loss_function='RMSE', custom_metric=['MAE', 'R2'],
        cat_features=cat_cols_final, random_seed=random_state,
        task_type='GPU', devices='0', gpu_ram_part=0.6, logging_level='Silent',
        **hp_final)
    with contextlib.redirect_stderr(io.StringIO()):
        final_model.fit(X_final, y_final_fit)

    y_pred_raw = pd.Series(final_model.predict(X_final), index=range(len(X_final)), dtype='float64')
    y_pred     = (pd.Series(pt_final.inverse_transform(y_pred_raw.values.reshape(-1, 1)).ravel(),
                            index=y_pred_raw.index, dtype='float64')
                  if pt_final is not None else y_pred_raw)

    return (results_df, final_model, X_final, y_pred,
            selected_features_per_fold, best_rent_params_list,
            best_model_params_list, feature_freq)


# ══════════════════════════════════════════════════════════════════════════════
# ADVANCED HGB + RENT  (Nested CV + Optuna)
# ══════════════════════════════════════════════════════════════════════════════

def run_advanced_hgb_rent(
    df_combined, target_col='pain_reduction_pct', random_state=42,
    tau_1=0.7, tau_2=0.75, tau_3=0.95,
):
    """HistGradientBoostingRegressor with RENT feature selection + nested CV + Optuna.

    Outer CV : RepeatedKFold(n_splits=4, n_repeats=5)  = 20 outer folds.
    Inner CV : RepeatedKFold(n_splits=4, n_repeats=25) = 100 fits per trial.
    Optuna   : 20 trials per outer fold, scoring = neg_root_mean_squared_error.
    Tuned    : learning_rate (log 0.01–0.3), max_depth (2–8),
               min_samples_leaf (5–40), l2_regularization (0–1).

    Parameters
    ----------
    df_combined : pd.DataFrame  Combined T1 dataset (immunological + clinical).
    target_col  : str           Regression target (default: 'pain_reduction_pct').
    random_state: int           Random seed (default 42).
    tau_1       : float         RENT τ₁ cutoff — selection frequency (default 0.7).
    tau_2       : float         RENT τ₂ cutoff — sign consistency (default 0.75).
    tau_3       : float         RENT τ₃ cutoff — t-test threshold (default 0.95).

    Returns
    -------
    results_df              : pd.DataFrame  Per-fold metrics + Mean/Std rows.
    best_params_df          : pd.DataFrame  Best hyperparameters per outer fold.
    final_model             : Pipeline      Fitted (HGB) on full data.
    X_final                 : pd.DataFrame  OrdinalEncoded feature matrix (RENT-selected cols).
    y_pred                  : pd.Series     Full-data predictions from final_model.
    selected_features_per_fold : list[list[str]]  Selected feature names per outer fold.
    """
    import optuna
    import warnings
    try:
        from optuna.integration import OptunaSearchCV
    except ImportError:
        from optuna_integration import OptunaSearchCV
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OrdinalEncoder, StandardScaler
    from sklearn.pipeline import Pipeline
    from RENT import RENT

    warnings.filterwarnings('ignore', category=FutureWarning, module='RENT')
    warnings.filterwarnings('ignore', category=RuntimeWarning, module='RENT')
    warnings.filterwarnings('ignore', message='OptunaSearchCV is experimental')

    # Non-feature columns are already removed by prepare_model_input().
    id_cols = ['Patient', 'Timepoint']
    exclude = set(id_cols) | {target_col}

    feature_cols = [c for c in df_combined.columns if c not in exclude]
    X_raw = df_combined[feature_cols].copy()
    y = df_combined[target_col].copy()

    valid = y.notna()
    X_raw, y = X_raw[valid].reset_index(drop=True), y[valid].reset_index(drop=True)

    # ── OrdinalEncode str/category columns once upfront ───────────────────────
    X_oe = X_raw.copy()
    cat_cols_list = [c for c in X_oe.columns
                     if X_oe[c].dtype == object or str(X_oe[c].dtype) == 'category']
    if cat_cols_list:
        oe_global = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_oe[cat_cols_list] = oe_global.fit_transform(X_oe[cat_cols_list].astype(str))
    # X_oe is now fully numeric (may still contain NaN — HGB handles this natively)

    print(f"\n{'='*65}")
    print(f"  HGB + RENT — {target_col}")
    print(f"  Samples: {len(X_oe)},  Features: {len(feature_cols)}")
    print(f"  RENT: K=100, τ₁={tau_1}, τ₂={tau_2}, τ₃={tau_3}")
    print(f"  Preprocessing: OrdinalEncode (upfront) — HGB handles NaN natively")
    print(f"  Outer: 4×5=20 folds  |  Inner: 4×25=100 fits/trial  |  Trials: 20")
    print(f"{'='*65}")

    outer_cv = RepeatedKFold(n_splits=4, n_repeats=5,  random_state=random_state)
    inner_cv = RepeatedKFold(n_splits=4, n_repeats=25, random_state=random_state) # 100 inner folds

    param_distributions = {
        'model__learning_rate':      optuna.distributions.FloatDistribution(0.01, 0.3, log=True),
        'model__max_depth':          optuna.distributions.IntDistribution(2, 8),
        'model__min_samples_leaf':   optuna.distributions.IntDistribution(5, 40),
        'model__l2_regularization':  optuna.distributions.FloatDistribution(0.0, 1.0),
    }

    fold_results           = []
    best_params_list       = []
    selected_features_per_fold = []
    optuna.logging.set_verbosity(optuna.logging.INFO)
    start = time.time()

    for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X_oe), start=1):
        print(f"\n  Outer fold {outer_fold}/{outer_cv.get_n_splits()}")

        X_oe_train = X_oe.iloc[train_idx].copy()
        X_oe_test  = X_oe.iloc[test_idx].copy()
        y_train    = y.iloc[train_idx]
        y_test     = y.iloc[test_idx]

        # ── Preprocess X_train for RENT (fit on train only) ───────────────────
        imputer_rent = SimpleImputer(strategy='median').fit(X_oe_train)
        X_imp_train  = imputer_rent.transform(X_oe_train)
        scaler_rent  = StandardScaler().fit(X_imp_train)
        X_rent_train = pd.DataFrame(
            scaler_rent.transform(X_imp_train), columns=feature_cols)

        # ── RENT feature selection on preprocessed X_train ───────────────────
        rent_model = RENT.RENT_Regression(
            data=X_rent_train,
            target=y_train.values,
            feat_names=feature_cols,
            C=[0.1, 1, 10],
            l1_ratios=[0.1, 0.5, 0.9],
            autoEnetParSel=True,
            poly='OFF',
            testsize_range=(0.25, 0.25),
            K=100,
            random_state=random_state,
            verbose=0,
        )
        rent_model.train()
        selected_idx = rent_model.select_features(
            tau_1_cutoff=tau_1, tau_2_cutoff=tau_2, tau_3_cutoff=tau_3)

        if len(selected_idx) == 0:
            print(f"    RENT: 0 features selected — using all {len(feature_cols)}")
            selected_cols = feature_cols
        else:
            selected_cols = [feature_cols[i] for i in selected_idx]
            preview = selected_cols[:8]
            suffix  = '...' if len(selected_cols) > 8 else ''
            print(f"    RENT: {len(selected_cols)}/{len(feature_cols)} features — {preview}{suffix}")

        selected_features_per_fold.append(selected_cols)

        # ── Inner CV + Optuna: Pipeline(HGB) — NaN handled natively ──────────
        inner_pipe = Pipeline([
            ('model', HistGradientBoostingRegressor(
                max_iter=300,
                random_state=random_state,
                early_stopping=False,
            )),
        ])

        optuna_search = OptunaSearchCV(
            estimator=inner_pipe,
            param_distributions=param_distributions,
            cv=inner_cv,
            scoring='neg_root_mean_squared_error',
            n_trials=20,
            n_jobs=-1,
            verbose=0,
        )

        optuna_search.fit(X_oe_train[selected_cols], y_train)
        best_params_list.append(optuna_search.best_params_)

        preds = optuna_search.predict(X_oe_test[selected_cols])
        rmse  = np.sqrt(mean_squared_error(y_test, preds))
        mae   = mean_absolute_error(y_test, preds)
        r2    = r2_score(y_test, preds)
        mse   = rmse ** 2

        fold_results.append({'Fold': outer_fold, 'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2})
        print(f"    MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}")
        print(f"    Best params: {optuna_search.best_params_}")

    elapsed = time.time() - start
    print(f"\n  Training time: {elapsed:.1f}s  ({elapsed/60:.1f} min)")

    # Results DataFrame + summary
    results_df  = pd.DataFrame(fold_results)
    metric_cols = ['MAE', 'MSE', 'RMSE', 'R2']
    mean_row = {'Fold': 'Mean', **{m: results_df[m].mean() for m in metric_cols}}
    std_row  = {'Fold': 'Std',  **{m: results_df[m].std()  for m in metric_cols}}
    results_df = pd.concat(
        [results_df, pd.DataFrame([mean_row, std_row])], ignore_index=True)

    n_outer = len(fold_results)
    t_crit  = stats.t.ppf(0.975, df=n_outer - 1)
    print(f"\n  Summary (4×5 outer CV + RENT, 20 Optuna trials, 95% CI):")
    for m in metric_cols:
        mv = mean_row[m]; sv = std_row[m]
        ci = t_crit * sv / np.sqrt(n_outer)
        print(f"    {m:<5}: {mv:.3f} ± {sv:.4f}  (95% CI [{mv - ci:.3f}, {mv + ci:.3f}])")

    best_params_df = pd.DataFrame(best_params_list)
    best_params_df.index = [f"Fold {i+1}" for i in range(len(best_params_list))]
    print(f"\n  Best hyperparameters per outer fold:")
    print(best_params_df.to_string())

    # Feature selection frequency across folds
    from collections import Counter
    all_selected = [f for fold_feats in selected_features_per_fold for f in fold_feats]
    freq = Counter(all_selected)
    print(f"\n  RENT feature selection frequency (top 20 across {n_outer} folds):")
    for feat, cnt in freq.most_common(20):
        print(f"    {cnt:>3}/{n_outer}  {feat}")

    # ── Final model on full dataset ───────────────────────────────────────────
    # RENT on full OrdinalEncoded X_oe
    imp_full = SimpleImputer(strategy='median').fit(X_oe)
    sca_full = StandardScaler().fit(imp_full.transform(X_oe))
    X_rent_full = pd.DataFrame(
        sca_full.transform(imp_full.transform(X_oe)), columns=feature_cols)

    rent_final = RENT.RENT_Regression(
        data=X_rent_full,
        target=y.values,
        feat_names=feature_cols,
        C=[0.1, 1, 10],
        l1_ratios=[0.1, 0.5, 0.9],
        autoEnetParSel=True,
        poly='OFF',
        testsize_range=(0.25, 0.25),
        K=100,
        random_state=random_state,
        verbose=0,
    )
    rent_final.train()
    final_idx  = rent_final.select_features(
        tau_1_cutoff=tau_1, tau_2_cutoff=tau_2, tau_3_cutoff=tau_3)
    final_cols = [feature_cols[i] for i in final_idx] if len(final_idx) > 0 else feature_cols
    print(f"\n  Final model RENT selected {len(final_cols)}/{len(feature_cols)} features.")

    X_final = X_oe[final_cols]

    bp = optuna_search.best_params_
    final_model = Pipeline([
        ('model', HistGradientBoostingRegressor(
            max_iter=300,
            random_state=random_state,
            early_stopping=False,
            learning_rate=bp['model__learning_rate'],
            max_depth=bp['model__max_depth'],
            min_samples_leaf=bp['model__min_samples_leaf'],
            l2_regularization=bp['model__l2_regularization'],
        )),
    ])
    final_model.fit(X_final, y)
    y_pred = pd.Series(final_model.predict(X_final), index=range(len(X_final)), dtype='float64')

    return results_df, best_params_df, final_model, X_final, y_pred, selected_features_per_fold



# ══════════════════════════════════════════════════════════════════════════════
# ADVANCED ELASTICNET + RENT  (Nested CV + Optuna)
# ══════════════════════════════════════════════════════════════════════════════

def run_advanced_elasticnet_rent(
    df_combined, target_col='pain_reduction_pct', random_state=42,
    tau_1=0.7, tau_2=0.75, tau_3=0.95,
):
    """ElasticNet with RENT feature selection + nested CV + Optuna.

      1. OrdinalEncoder — applied once before the outer CV loop to convert any
         str/category columns to integers (fixed mapping, no statistical fit).
      2. SimpleImputer (median) — fitted on X_train inside each outer fold;
         also re-fitted on each inner split via the sklearn Pipeline.
      3. StandardScaler — same: fitted on X_train per outer fold for RENT;
         re-fitted per inner split via the Pipeline.

    RENT runs on the fully preprocessed (imputed + scaled) X_train of each
    outer fold.  The Optuna inner CV uses a pipeline so preprocessing is fit
    only on inner-train splits.

    Outer CV : RepeatedKFold(n_splits=4, n_repeats=5)  = 20 outer folds.
    Inner CV : RepeatedKFold(n_splits=4, n_repeats=25) = 100 fits per trial.
    Optuna   : 20 trials per outer fold, scoring = neg_root_mean_squared_error.
    Tuned    : alpha (1e-4 – 100, log), l1_ratio (0.01 – 1.0).

    Parameters
    ----------
    df_combined : pd.DataFrame  Combined T1 dataset (immunological + clinical).
    target_col  : str           Regression target (default: 'pain_reduction_pct').
    random_state: int           Random seed (default 42).
    tau_1       : float         RENT τ₁ cutoff — selection frequency (default 0.7).
    tau_2       : float         RENT τ₂ cutoff — sign consistency (default 0.75).
    tau_3       : float         RENT τ₃ cutoff — t-test threshold (default 0.95).

    Returns
    -------
    results_df              : pd.DataFrame  Per-fold metrics + Mean/Std rows.
    best_params_df          : pd.DataFrame  Best hyperparameters per outer fold.
    final_pipeline          : Pipeline      Fitted (Imputer→Scaler→ElasticNet) on full data.
    X_final                 : pd.DataFrame  OrdinalEncoded feature matrix (RENT-selected cols).
    y_pred                  : pd.Series     Full-data predictions from final_pipeline.
    selected_features_per_fold : list[list[str]]  Selected feature names per outer fold.
    """
    import optuna
    try:
        from optuna.integration import OptunaSearchCV
    except ImportError:
        from optuna_integration import OptunaSearchCV
    import warnings
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OrdinalEncoder, StandardScaler
    from sklearn.linear_model import ElasticNet
    from sklearn.pipeline import Pipeline
    from sklearn.exceptions import ConvergenceWarning
    from RENT import RENT

    # Suppress known harmless warnings:
    # - FutureWarning: RENT uses deprecated .applymap() (old pandas API, no effect on results)
    # - RuntimeWarning: RENT divide-by-zero in τ₃ when feature std=0 (handled as NaN internally)
    # - ExperimentalWarning: OptunaSearchCV is marked experimental but works correctly
    warnings.filterwarnings('ignore', category=FutureWarning, module='RENT')
    warnings.filterwarnings('ignore', category=RuntimeWarning, module='RENT')
    warnings.filterwarnings('ignore', message='OptunaSearchCV is experimental')

    # Non-feature columns are already removed by prepare_model_input().
    id_cols = ['Patient', 'Timepoint']
    exclude = set(id_cols) | {target_col}

    feature_cols = [c for c in df_combined.columns if c not in exclude]
    X_raw = df_combined[feature_cols].copy()
    y = df_combined[target_col].copy()

    valid = y.notna()
    X_raw, y = X_raw[valid].reset_index(drop=True), y[valid].reset_index(drop=True)

    # ── OrdinalEncode str/category columns once upfront ───────────────────────
    # This is a fixed integer mapping (gender, diagnosis, …) — not a statistical
    # fit, so doing it before the outer CV loop introduces negligible leakage.
    X_oe = X_raw.copy()
    cat_cols_list = [c for c in X_oe.columns
                     if X_oe[c].dtype == object or str(X_oe[c].dtype) == 'category']
    if cat_cols_list:
        oe_global = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_oe[cat_cols_list] = oe_global.fit_transform(X_oe[cat_cols_list].astype(str))
    # X_oe is now fully numeric (may still contain NaN)

    print(f"\n{'='*65}")
    print(f"  ElasticNet + RENT — {target_col}")
    print(f"  Samples: {len(X_oe)},  Features: {len(feature_cols)}")
    print(f"  RENT: K=100, τ₁={tau_1}, τ₂={tau_2}, τ₃={tau_3}")
    print(f"  Preprocessing: OrdinalEncode → Impute (median) → StandardScale")
    print(f"  Outer: 4×5=20 folds  |  Inner: 4×25=100 fits/trial  |  Trials: 20")
    print(f"{'='*65}")

    outer_cv = RepeatedKFold(n_splits=4, n_repeats=5,  random_state=random_state)
    inner_cv = RepeatedKFold(n_splits=4, n_repeats=25, random_state=random_state)

    # Optuna search space: alpha (regularisation strength) and l1_ratio (L1/L2 mix).
    # Lower bound 1e-2 avoids near-zero alpha that causes severe overfitting on
    # small inner-train splits (~45 samples) and wastes Optuna trials.
    param_distributions = {
        'model__alpha':     optuna.distributions.FloatDistribution(1e-2, 100.0, log=True),
        'model__l1_ratio':  optuna.distributions.FloatDistribution(0.01, 1.0),
    }

    fold_results           = []
    best_params_list       = []
    selected_features_per_fold = []
    optuna.logging.set_verbosity(optuna.logging.INFO)
    start = time.time()

    for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X_oe), start=1):
        print(f"\n  Outer fold {outer_fold}/{outer_cv.get_n_splits()}")

        X_oe_train = X_oe.iloc[train_idx].copy()
        X_oe_test  = X_oe.iloc[test_idx].copy()
        y_train    = y.iloc[train_idx]
        y_test     = y.iloc[test_idx]

        # ── Preprocess X_train for RENT (fit on train only) ───────────────────
        imputer_rent = SimpleImputer(strategy='median').fit(X_oe_train)
        X_imp_train  = imputer_rent.transform(X_oe_train)
        scaler_rent  = StandardScaler().fit(X_imp_train)
        X_rent_train = pd.DataFrame(
            scaler_rent.transform(X_imp_train), columns=feature_cols)

        # ── RENT feature selection on preprocessed X_train ───────────────────
        rent_model = RENT.RENT_Regression(
            data=X_rent_train,
            target=y_train.values,
            feat_names=feature_cols,
            C=[0.1, 1, 10],
            l1_ratios=[0.1, 0.5, 0.9],
            autoEnetParSel=True,
            poly='OFF',
            testsize_range=(0.25, 0.25),
            K=100,
            random_state=random_state,
            verbose=0,
        )
        rent_model.train()
        selected_idx = rent_model.select_features(
            tau_1_cutoff=tau_1, tau_2_cutoff=tau_2, tau_3_cutoff=tau_3)

        if len(selected_idx) == 0:
            print(f"    RENT: 0 features selected — using all {len(feature_cols)}")
            selected_cols = feature_cols
        else:
            selected_cols = [feature_cols[i] for i in selected_idx]
            preview = selected_cols[:8]
            suffix  = '...' if len(selected_cols) > 8 else ''
            print(f"    RENT: {len(selected_cols)}/{len(feature_cols)} features — {preview}{suffix}")

        selected_features_per_fold.append(selected_cols)

        # ── Inner CV + Optuna with Pipeline (re-fits preprocessing per split) ─
        # Pipeline ensures Imputer and Scaler are fit only on inner-train data.
        inner_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler',  StandardScaler()),
            ('model',   ElasticNet(max_iter=100000, random_state=random_state)),
        ])

        optuna_search = OptunaSearchCV(
            estimator=inner_pipe,
            param_distributions=param_distributions,
            cv=inner_cv,
            scoring='neg_root_mean_squared_error',
            n_trials=50,
            n_jobs=-1,
            verbose=0,
        )

        import warnings
        from sklearn.exceptions import ConvergenceWarning
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ConvergenceWarning)
            optuna_search.fit(X_oe_train[selected_cols], y_train)
        best_params_list.append(optuna_search.best_params_)

        preds = optuna_search.predict(X_oe_test[selected_cols])
        rmse  = np.sqrt(mean_squared_error(y_test, preds))
        mae   = mean_absolute_error(y_test, preds)
        r2    = r2_score(y_test, preds)
        mse   = rmse ** 2

        fold_results.append({'Fold': outer_fold, 'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2})
        print(f"    MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}")
        print(f"    Best params: {optuna_search.best_params_}")

    elapsed = time.time() - start
    print(f"\n  Training time: {elapsed:.1f}s  ({elapsed/60:.1f} min)")

    # Results DataFrame + summary
    results_df  = pd.DataFrame(fold_results)
    metric_cols = ['MAE', 'MSE', 'RMSE', 'R2']
    mean_row = {'Fold': 'Mean', **{m: results_df[m].mean() for m in metric_cols}}
    std_row  = {'Fold': 'Std',  **{m: results_df[m].std()  for m in metric_cols}}
    results_df = pd.concat(
        [results_df, pd.DataFrame([mean_row, std_row])], ignore_index=True)

    n_outer = len(fold_results)
    t_crit  = stats.t.ppf(0.975, df=n_outer - 1)
    print(f"\n  Summary (4×5 outer CV + RENT, 20 Optuna trials, 95% CI):")
    for m in metric_cols:
        mv = mean_row[m]; sv = std_row[m]
        ci = t_crit * sv / np.sqrt(n_outer)
        print(f"    {m:<5}: {mv:.3f} ± {sv:.4f}  (95% CI [{mv - ci:.3f}, {mv + ci:.3f}])")

    best_params_df = pd.DataFrame(best_params_list)
    best_params_df.index = [f"Fold {i+1}" for i in range(len(best_params_list))]
    print(f"\n  Best hyperparameters per outer fold:")
    print(best_params_df.to_string())

    # Feature selection frequency across folds
    from collections import Counter
    all_selected = [f for fold_feats in selected_features_per_fold for f in fold_feats]
    freq = Counter(all_selected)
    print(f"\n  RENT feature selection frequency (top 20 across {n_outer} folds):")
    for feat, cnt in freq.most_common(20):
        print(f"    {cnt:>3}/{n_outer}  {feat}")

    # ── Final model on full dataset ───────────────────────────────────────────
    # RENT on full OrdinalEncoded X_oe
    imp_full = SimpleImputer(strategy='median').fit(X_oe)
    sca_full = StandardScaler().fit(imp_full.transform(X_oe))
    X_rent_full = pd.DataFrame(
        sca_full.transform(imp_full.transform(X_oe)), columns=feature_cols)

    rent_final = RENT.RENT_Regression(
        data=X_rent_full,
        target=y.values,
        feat_names=feature_cols,
        C=[0.1, 1, 10],
        l1_ratios=[0, 0.1, 0.25, 0.5, 0.75, 0.9, 1],
        autoEnetParSel=True,
        poly='OFF',
        testsize_range=(0.25, 0.25),
        K=100,
        random_state=random_state,
        verbose=0,
    )
    rent_final.train()
    final_idx  = rent_final.select_features(
        tau_1_cutoff=tau_1, tau_2_cutoff=tau_2, tau_3_cutoff=tau_3)
    final_cols = [feature_cols[i] for i in final_idx] if len(final_idx) > 0 else feature_cols
    print(f"\n  Final model RENT selected {len(final_cols)}/{len(feature_cols)} features.")

    X_final = X_oe[final_cols]

    final_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler()),
        ('model',   ElasticNet(
            max_iter=100000,
            random_state=random_state,
            alpha=optuna_search.best_params_['model__alpha'],
            l1_ratio=optuna_search.best_params_['model__l1_ratio'],
        )),
    ])
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        final_pipeline.fit(X_final, y)
    y_pred = pd.Series(final_pipeline.predict(X_final), index=range(len(X_final)), dtype='float64')

    return results_df, best_params_df, final_pipeline, X_final, y_pred, selected_features_per_fold


def plot_shap_pipeline(pipeline, X_final, name, top_n=20):
    """SHAP bar + beeswarm plots for a fitted sklearn Pipeline.

    Selects the appropriate SHAP explainer based on the model type:
      - ElasticNet  → shap.LinearExplainer  (exact, fast)
      - HGB         → shap.TreeExplainer    (exact, fast)
    
    For pipelines with preprocessing steps (Imputer → Scaler → model), X is
    transformed through those steps before SHAP is applied to the model directly,
    so SHAP values are in the standardised feature space.

    Parameters
    ----------
    pipeline : sklearn Pipeline  Fitted pipeline with last step named 'model'.
    X_final  : pd.DataFrame      OrdinalEncoded feature matrix (RENT-selected cols).
    name     : str               Label shown in plot titles.
    top_n    : int               Maximum features to display (default 20).
    """
    import shap
    from sklearn.linear_model import ElasticNet

    feature_names  = list(X_final.columns)
    model          = pipeline.named_steps['model']
    step_names     = list(pipeline.named_steps.keys())
    preprocess_steps = step_names[:-1]   # all steps before 'model'

    # Transform X through preprocessing steps (imputer, scaler) if present
    X_t = X_final.copy()
    for sname in preprocess_steps:
        X_t = pipeline.named_steps[sname].transform(X_t)
    X_t = pd.DataFrame(X_t, columns=feature_names)

    print(f"\n=== SHAP Analysis: {name} ===")

    if isinstance(model, ElasticNet):
        explainer   = shap.LinearExplainer(model, X_t)
        shap_values = explainer.shap_values(X_t)
    elif isinstance(model, HistGradientBoostingRegressor):
        # HGB pipeline has no preprocessing steps; pass X_final directly
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_final)
        X_t = X_final.copy()
    else:
        # PLS and any other model: permutation-based (model-agnostic)
        explainer   = shap.Explainer(model.predict, X_t)
        shap_values = explainer(X_t).values

    shap.summary_plot(shap_values, X_t, plot_type="bar", show=False, max_display=top_n)
    plt.title(f"SHAP Feature Importance — {name}")
    plt.tight_layout()
    plt.show()

    shap.summary_plot(shap_values, X_t, show=False, max_display=top_n)
    plt.title(f"SHAP Beeswarm — {name}")
    plt.tight_layout()
    plt.show()

    return shap_values

