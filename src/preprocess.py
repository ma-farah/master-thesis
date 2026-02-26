# functions for preprocessing data
# imports
import pandas as pd
import numpy as np
import re
from skrub import TableReport


# ══════════════════════════════════════════════════════════════════════════════
# COMMON CLEANING FUNCTIONS  (work on both immunological and clinical datasets)
# ══════════════════════════════════════════════════════════════════════════════

def replace_missing_markers(df, skip_cols=None, verbose=False):
    """Replace German missing-value strings with NaN in all object columns.

    Handles all capitalisation and punctuation variants of:
      - 'k.A.' (keine Angabe — no data entered)
      - 'n.D.' (nicht durchgeführt — not performed)

    Parameters
    ----------
    df        : pd.DataFrame  (modified in-place)
    skip_cols : iterable of str, optional — columns to leave untouched
    verbose   : bool — print per-column replacement counts
    """
    pattern   = r'^([kK]\.?[aA]\.?|[nN]\.?[dD]\.?)$'
    skip_cols = set(skip_cols or [])

    for col in df.columns:
        if col in skip_cols or df[col].dtype != object:
            continue
        str_col = df[col].astype(str).str.strip()
        mask = str_col.str.match(pattern, na=False) | (str_col == "")
        if mask.sum() > 0:
            if verbose:
                print(f"  {col}: replaced {mask.sum()} null markers")
            df.loc[mask, col] = np.nan


def drop_columns(df, columns, verbose=True):
    """Drop a pre-specified list of columns, ignoring any that are absent.

    Parameters
    ----------
    df       : pd.DataFrame
    columns  : list of str  — columns to remove
    verbose  : bool

    Returns
    -------
    pd.DataFrame with columns removed (copy)
    """
    present = [c for c in columns if c in df.columns]
    missing = [c for c in columns if c not in df.columns]
    if verbose:
        print(f"  Dropping {len(present)} columns.")
        if missing:
            print(f"  Not found (already absent): {missing}")
    return df.drop(columns=present)


def drop_rows_by_index(df, indices, verbose=True):
    """Drop rows by explicit index labels, ignoring any that are absent.

    Parameters
    ----------
    df      : pd.DataFrame
    indices : list of int/label  — index values to drop
    verbose : bool

    Returns
    -------
    pd.DataFrame with rows removed (copy)
    """
    present = [i for i in indices if i in df.index]
    if verbose:
        print(f"  Dropping {len(present)} rows at index: {present}")
    return df.drop(index=present)


def drop_high_nan_columns(df, threshold=0.25, exclude_cols=None,
                           timepoint_col='Timepoint', check_per_timepoint=True,
                           verbose=True):
    """Drop columns whose overall missing rate exceeds *threshold*.

    Optionally prints a per-timepoint breakdown first so you can verify
    the threshold is consistent across timepoints before dropping.

    Parameters
    ----------
    df                  : pd.DataFrame
    threshold           : float   fraction, default 0.25
    exclude_cols        : list of str — never drop these (e.g. id columns)
    timepoint_col       : str
    check_per_timepoint : bool — print per-timepoint >threshold columns first
    verbose             : bool

    Returns
    -------
    pd.DataFrame with high-NaN columns removed (copy)
    """
    exclude_cols = list(exclude_cols or [])

    if check_per_timepoint and timepoint_col in df.columns:
        print(f"  Per-timepoint columns >{threshold*100:.0f}% NaN:")
        for _tp in sorted(df[timepoint_col].dropna().unique()):
            _df_tp   = df[df[timepoint_col] == _tp]
            _na_tp   = _df_tp.drop(columns=[c for c in exclude_cols if c in _df_tp.columns]).isna().mean()
            _high    = sorted(_na_tp[_na_tp > threshold].index.tolist())
            print(f"    T{_tp} ({len(_high)} columns): {_high}")

    na_frac      = df.isna().mean()
    cols_to_drop = [c for c in na_frac[na_frac > threshold].index if c not in exclude_cols]

    if verbose:
        print(f"\n  Overall columns >{threshold*100:.0f}% NaN ({len(cols_to_drop)}): "
              f"{sorted(cols_to_drop)}")
        print(f"  Dropping {len(cols_to_drop)} columns.")

    return df.drop(columns=cols_to_drop).copy()


# ══════════════════════════════════════════════════════════════════════════════
# IMMUNOLOGICAL DATASET SPECIFIC CLEANING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

# Columns marked yellow in source file — excluded from all analysis
IM_EXCLUDED_COLUMNS = [
    "ID_Subset",
    "CD123lo Bas.1",
    "T cells.1",
    "TH.1",
    "TC.1",
    "T4:T8 ratio.1",
    "DCs.1",
    "MDCs .1",
    "PDCs.1",
    "LIN-/16+/HLA+/123+",
    "undefined",
    "T8hi.1",
    "T8lo.1",
    "DNT.1",
    "DPT.1",
    "CD28-",
    "CD28- ",
    "CD28-.1",
    "mDC",
    "pDC",
    "mDC.1",
    "pDC.1",
    "Eos_CD25+",
    "DC_CD25+",
    "T_CTLA4+",
    "TH_CTLA4+",
    "TC_CTLA4+",
    "TH naive_CTLA4+",
    "TC naive_CTLA4+",
    "DC_PDL1+",
    "mDC_PDL1+",
    "mDC-1_PDL1+",
    "mDC-2_PDL1+",
    "pDC_PDL1+",
    "DC_CD80+",
    "mDC_CD80+",
    "mDC-1_CD80+",
    "mDC-2_CD80+",
    "pDC_CD80+",
    "DC_CD86+",
    "mDC_CD86+",
    "mDC-1_CD86+",
    "mDC-2_CD86+",
    "pDC_CD86+",
]

# Empty/junk rows at the bottom of the immunological Excel sheet
IM_EMPTY_ROW_INDICES = list(range(823, 829)) + [78]


def clean_im(df_im, verbose=True):
    """Full cleaning pipeline for the raw immunological dataset.

    Steps
    -----
    1. Drop pre-determined excluded columns + rename Messdatum → Date
    2. Drop known empty rows (bottom of Excel + row 78)
    3. Replace German NaN markers (k.A. / n.D. variants)
    4. Fix dtypes: Date → datetime, Patient/Timepoint → Int64, features → float64
    5. Make baseline CatBoost copy  (df_im_bcat)
    6. Drop columns with >25% NaN overall
    7. Make EDA/visualization copy  (df_im_vis)

    Parameters
    ----------
    df_im   : pd.DataFrame   raw immunological data as loaded from Excel
    verbose : bool

    Returns
    -------
    df_im      : cleaned full dataset (after dtype fix, before NaN drop)
    df_im_bcat : copy taken before >25% NaN drop (for baseline CatBoost)
    df_im_vis  : copy taken after  >25% NaN drop (for EDA / visualization)
    """
    # 1 — drop excluded columns + rename date column
    if verbose:
        print("  [1] Dropping excluded columns and renaming Messdatum → Date")
    df_im = drop_columns(df_im, IM_EXCLUDED_COLUMNS, verbose=verbose)
    if 'Messdatum' in df_im.columns:
        df_im = df_im.rename(columns={'Messdatum': 'Date'})

    # 2 — drop known empty rows
    if verbose:
        print("  [2] Dropping known empty rows")
    df_im = drop_rows_by_index(df_im, IM_EMPTY_ROW_INDICES, verbose=verbose)
    df_im = df_im.reset_index(drop=True)

    # 3 — replace German NaN markers
    if verbose:
        print("  [3] Replacing German NaN markers")
    replace_missing_markers(df_im, skip_cols=["Patient", "Timepoint"], verbose=verbose)

    # 4 — fix dtypes
    if verbose:
        print("  [4] Fixing dtypes")
    df_im = fix_dtypes_im(df_im)

    # 5 — baseline CatBoost copy (before >25% NaN drop)
    df_im_bcat = df_im.copy()

    # 6 — drop high-NaN columns
    if verbose:
        print("  [6] Dropping columns with >25% NaN")
    _exclude = ["Date", "Patient", "Timepoint"]
    df_im = drop_high_nan_columns(
        df_im, threshold=0.25, exclude_cols=_exclude,
        check_per_timepoint=True, verbose=verbose
    )

    # 7 — EDA/visualization copy (after >25% NaN drop)
    df_im_vis = df_im.copy()

    if verbose:
        print(f"\n  df_im      : {df_im.shape}")
        print(f"  df_im_bcat : {df_im_bcat.shape}")
        print(f"  df_im_vis  : {df_im_vis.shape}")

    return df_im, df_im_bcat, df_im_vis


def fix_dtypes_im(df_im):
    """Convert immunological dataset columns to correct dtypes.

    - Date      → datetime64
    - Patient   → Int64  (nullable integer)
    - Timepoint → Int64
    - All other columns → float64  (coerce non-numeric to NaN)

    Returns
    -------
    pd.DataFrame (copy)
    """
    df = df_im.copy()
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    if 'Patient' in df.columns:
        df['Patient'] = pd.to_numeric(df['Patient'], errors='coerce').astype('Int64')
    if 'Timepoint' in df.columns:
        df['Timepoint'] = pd.to_numeric(df['Timepoint'], errors='coerce').astype('Int64')

    _id_cols   = ['Date', 'Patient', 'Timepoint']
    _feat_cols = [c for c in df.columns if c not in _id_cols]
    for col in _feat_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


# ── Outlier removal (after PyOD + expert review) ─────────────────────────────

# Confirmed outlier observations from PyOD ensemble + expert review.
# Each entry is a (Patient, Timepoint) pair — only that specific observation
# is removed, not all timepoints for that patient.
IM_CONFIRMED_OUTLIERS = [
    (221, 2),
    (163, 1),
    (150, 1),
    (159, 2),
    (109, 5),
    (266, 4),
]


def remove_outlier_observations(df, outliers=None,
                                 patient_col='Patient', timepoint_col='Timepoint',
                                 verbose=True):
    """Remove specific (patient, timepoint) observations confirmed as outliers.

    Works for both immunological and clinical datasets — pass the appropriate
    outlier list via *outliers*. Defaults to IM_CONFIRMED_OUTLIERS.

    Parameters
    ----------
    df             : pd.DataFrame   source dataset (df_im_vis or df_cl_vis)
    outliers       : list of (patient_id, timepoint) tuples, or None
                     → defaults to IM_CONFIRMED_OUTLIERS
    patient_col    : str
    timepoint_col  : str
    verbose        : bool

    Returns
    -------
    pd.DataFrame — copy with outlier rows removed and index reset
    """
    if outliers is None:
        outliers = IM_CONFIRMED_OUTLIERS

    mask = pd.Series(False, index=df.index)
    for patient, timepoint in outliers:
        mask |= (df[patient_col] == patient) & (df[timepoint_col] == timepoint)

    result = df[~mask].reset_index(drop=True)

    if verbose:
        print(f"  Removed {mask.sum()} observations:")
        for patient, timepoint in outliers:
            found = ((df[patient_col] == patient) & (df[timepoint_col] == timepoint)).sum()
            status = "removed" if found else "not found"
            print(f"    Patient {patient}  T{timepoint}  ({status})")
        print(f"  Shape before: {df.shape}  →  after: {result.shape}")

    return result


