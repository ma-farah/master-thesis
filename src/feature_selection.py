
# MRMR feature frequency list (model-agnostic)
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import RepeatedKFold
from collections import Counter

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
            bar = '█' * int(val * 20)
            print(f"    {val*100:5.1f}%  {bar:<20}  {feat}")

    return feature_freq
