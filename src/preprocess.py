# functions for preprocessing data
# imports
import pandas as pd
import numpy as np
import re
import datetime as dt
from collections import defaultdict
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
    # Capture Patient IDs at the rows being dropped before they are removed
    _to_drop_pts = df_im.loc[
        [i for i in IM_EMPTY_ROW_INDICES if i in df_im.index], 'Patient'
    ].dropna().unique().tolist() if 'Patient' in df_im.columns else []
    df_im = drop_rows_by_index(df_im, IM_EMPTY_ROW_INDICES, verbose=verbose)
    df_im = df_im.reset_index(drop=True)
    if verbose and 'Patient' in df_im.columns:
        if _to_drop_pts:
            print(f"  Patient IDs in dropped rows: {sorted(_to_drop_pts)}")
        print(f"  Unique patients remaining: {df_im['Patient'].dropna().nunique()}")

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
        print(f"  Unique patients remaining: {result[patient_col].nunique()}")

    return result


# ══════════════════════════════════════════════════════════════════════════════
# CLINICAL DATASET SPECIFIC CLEANING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

# ── Constants ─────────────────────────────────────────────────────────────────

# Patient-level columns: constant across timepoints, filled only in first row
CL_PATIENT_LEVEL_COLS = [
    'Patient', 'Unnamed: 2', 'Age at start', 'Gender', 'Weight [kg]', 'Height [cm]',
    'Overweight? BMI', 'Besserung nach Nachuntersuchung laut Arztbrief in %',
    'Comments questionnaire', 'Diagnosis', 'Target volume', 'single fraction',
    'kummulative dose (x) - if two targets were applied', 'FHA', 'kV', 'mA',
    'Filter', 'Response', 'further comments',
]

# Patients irradiated at multiple different body parts — excluded from analysis
CL_MULTI_BODY_PATIENTS = [3, 45, 184, 162, 179, 156, 54, 47]

# German → English column rename map
CL_RENAME_MAP = {
    "Patient": "Patient", "Timepoint": "Timepoint",
    "Age at start": "age_at_start", "Gender": "gender",
    "Weight [kg]": "weight_kg", "Height [cm]": "height_cm",
    "Overweight? BMI": "overweight_bmi",
    "Erfassungszeitpunkt": "measurement_timepoint", "Datum": "date",
    "Beschwerden seit": "symptoms_months", "vorherige Therapie": "previous_therapy",
    "unter Belastung": "pain_under_load", "bei Nacht": "pain_night",
    "tagsüber": "pain_daytime", "in Ruhe": "pain_at_rest",
    "bei ersten Schritten/Morgensteifigkeit": "morning_stiffness",
    "Schmerzskala": "pain_scale", "Schmerzpunkte": "pain_points",
    "Besserung nach Nachuntersuchung laut Arztbrief in %": "improvement_percent",
    "Diagnosis": "diagnosis", "Target volume": "target_volume",
    "single fraction": "single_fraction",
    "kummulative dose (x) - if two targets were applied": "cumulative_dose",
    "FHA": "fha", "kV": "kv", "mA": "ma", "Filter": "filter", "Response": "response",
}

CL_CATEGORICAL_COLS = [
    'gender', 'overweight', 'pain_points', 'diagnosis',
    'target_volume', 'filter_material',
    'response', 'response_category',
]

# Column name patterns that identify leaky/metadata columns (must not be model features)
CL_LEAKY_PATTERNS = [
    'response', 'improvement_percent', 'pain_reduction_pct',
    'response_pct', 'response_category',
]


# ── Clinical helper functions ─────────────────────────────────────────────────

def move_column_after(df, col_to_move, after_col):
    """Move col_to_move to the position immediately after after_col."""
    cols = df.columns.tolist()
    cols.insert(cols.index(after_col) + 1, cols.pop(cols.index(col_to_move)))
    return df[cols]


def extract_numeric(series):
    """Extract numeric value from ordinal questionnaire entries (scale 1–4 or 1–5).

    Handles:
      - comma-separated multi-select  "1,2"               → average
      - range                          "2-3"               → midpoint
      - number with parenthetical text "3 (tag), 4 (nacht)"→ average of all numbers
      - leading number with text       "3 left side"       → 3
    """
    s = series.astype(str).str.strip()

    def parse_entry(val):
        if val in ('nan', '', 'None'):
            return np.nan
        if re.match(r'^\d+(\s*,\s*\d+)+$', val):
            return np.mean([float(x) for x in val.split(',')])
        m = re.match(r'^(\d+)\s*[-–]\s*(\d+)', val)
        if m:
            return (float(m.group(1)) + float(m.group(2))) / 2
        all_nums = re.findall(r'\b(\d+)\b', val)
        if len(all_nums) > 1:
            return np.mean([float(x) for x in all_nums])
        if len(all_nums) == 1:
            return float(all_nums[0])
        return np.nan

    return s.apply(parse_entry)


def extract_continuous(series):
    """Extract numeric value from continuous scale entries (e.g. pain_scale 0–10).

    Handles German decimal commas, ranges (→ midpoint), trailing text, and
    "Ruhe" (at rest) entries where both load and rest values are present
    (prefers the resting value).
    """
    s = series.astype(str).str.strip()

    def parse_entry(val):
        if val in ('nan', '', 'None'):
            return np.nan
        m_ruhe = re.search(r'(\d+[.,]?\d*)\s*(?:aus\s+der\s+)?[Rr]uhe', val)
        if m_ruhe:
            return float(m_ruhe.group(1).replace(',', '.'))
        m = re.match(r'^(\d+[.,]?\d*)\s*[-–]\s*(\d+[.,]?\d*)\s*$', val)
        if m:
            return (float(m.group(1).replace(',', '.')) +
                    float(m.group(2).replace(',', '.'))) / 2
        m = re.match(r'^(\d+[.,]?\d*)', val)
        if m:
            return float(m.group(1).replace(',', '.'))
        return np.nan

    return s.apply(parse_entry)


def split_bmi_column(df, col_name='overweight_bmi'):
    """Split combined overweight/BMI column into 'overweight' (ja/nein) and 'bmi' (float).

    Input format: "ja (28.5)", "nein", "n.D" (missing).
    """
    col_idx = df.columns.get_loc(col_name)
    is_missing = df[col_name].str.contains(r'^n\.?D\.?$', case=False, na=True)
    overweight = df[col_name].str.extract(r'(ja|nein)', flags=re.IGNORECASE)[0].str.lower()
    bmi = df[col_name].str.extract(r'\((\d+[,.]?\d*)\)?')[0].str.replace(',', '.').astype(float)
    overweight = overweight.where(~is_missing, pd.NA)
    bmi = bmi.where(~is_missing, pd.NA)
    df = df.drop(columns=[col_name])
    df.insert(col_idx, 'overweight', overweight)
    df.insert(col_idx + 1, 'bmi', bmi)
    return df


def parse_symptoms_duration(series, date_series=None):
    """Convert German symptom duration strings to numeric months.

    Handles: "3 Monate", "2 Jahre", "6-12 Mo.", "1,5 J.", "1/2 J.",
    ranges → midpoint, German decimals, fractions, ~approx, >greater-than.
    Date entries (2023-04-01, ~02/2022, Okt/Nov 2022) → months from measurement date.
    Vague entries (Jahre, mehrere, täglich) → NaN.
    Standalone numbers without unit → assumed months.
    """
    month_map = {
        'jan': 1, 'feb': 2, 'mär': 3, 'mar': 3, 'apr': 4, 'mai': 5,
        'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'okt': 10, 'oct': 10,
        'nov': 11, 'dez': 12, 'dec': 12,
    }

    def parse_entry(val, meas_date):
        if pd.isna(val):
            return pd.NA
        s = str(val).strip()

        if s.lower() in ('einige jahre', 'einige j.', 'einge j.'):
            return 12.0
        if s.lower() in ('jahre', 'jahre ', 'mehrere', 'mehrere jahre',
                         'mehrere monate', 'mehreren mo.', 'täglich'):
            return pd.NA

        date_match = re.match(r'^(\d{4})-(\d{2})-(\d{2})', s)
        if date_match:
            if pd.notna(meas_date):
                symptom_date = pd.Timestamp(
                    f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}")
                return max(0, (pd.Timestamp(meas_date) - symptom_date).days / 30.44)
            return pd.NA

        de_date_match = re.match(r'^~?(\d{1,2})\.(\d{1,2})\.(\d{4})$', s.strip())
        if de_date_match:
            if pd.notna(meas_date):
                symptom_date = pd.Timestamp(
                    f"{de_date_match.group(3)}-{int(de_date_match.group(2)):02d}"
                    f"-{int(de_date_match.group(1)):02d}"
                )
                return max(0, (pd.Timestamp(meas_date) - symptom_date).days / 30.44)
            return pd.NA

        my_match = re.match(r'^~?(\d{2})/(\d{4})$', s)
        if my_match:
            if pd.notna(meas_date):
                symptom_date = pd.Timestamp(f"{my_match.group(2)}-{my_match.group(1)}-01")
                return max(0, (pd.Timestamp(meas_date) - symptom_date).days / 30.44)
            return pd.NA

        mon_match = re.match(r'^(\w{3})\w*(?:/\w+)?\s+(\d{4})', s, re.IGNORECASE)
        if mon_match:
            paren_match = re.search(r'\((\d+)\s*Mo', s)
            if paren_match:
                return float(paren_match.group(1))
            mon_key = mon_match.group(1).lower()
            year = int(mon_match.group(2))
            if mon_key in month_map and pd.notna(meas_date):
                symptom_date = pd.Timestamp(f"{year}-{month_map[mon_key]:02d}-01")
                return max(0, (pd.Timestamp(meas_date) - symptom_date).days / 30.44)
            return pd.NA

        s_clean = re.sub(r'\(\?\)', '', s)
        s_clean = re.sub(r'\([^)]*\)', '', s_clean).strip()
        s_clean = re.sub(r'^[~><]\s*', '', s_clean)
        s_clean = re.sub(r'^akut\s+', '', s_clean, flags=re.IGNORECASE)

        is_years = bool(re.search(r'(Jahr\w*|J\.?\b)', s_clean, re.IGNORECASE))

        frac_match = re.match(r'(\d+)/(\d+)', s_clean)
        if frac_match:
            number = float(frac_match.group(1)) / float(frac_match.group(2))
            return number * 12 if is_years else number

        range_match = re.search(r'(\d+[,.]?\d*)\s*-\s*(\d+[,.]?\d*)', s_clean)
        if range_match:
            start = float(range_match.group(1).replace(',', '.'))
            end = float(range_match.group(2).replace(',', '.'))
            number = (start + end) / 2
            return number * 12 if is_years else number

        num_match = re.search(r'(\d+[,.]?\d*)', s_clean)
        if num_match:
            number = float(num_match.group(1).replace(',', '.'))
            return number * 12 if is_years else number

        return pd.NA

    if date_series is not None:
        return pd.Series(
            [parse_entry(v, d) for v, d in zip(series, date_series)],
            index=series.index,
        )
    return series.apply(lambda v: parse_entry(v, None))


def standardize_target_volume(series):
    """Standardize target_volume: map body part variants to English names,
    extract treatment side into a separate series.

    Returns (body_part_series, target_side_series).
    """
    body_part_map = [
        ('Achilles Tendon', ['achillessehne', 'achilles tendon']),
        ('Heel',            ['heel', 'ferse']),
        ('Foot',            ['foot', 'forefoot']),
        ('Ankle',           ['ankle']),
        ('Knee',            ['knee', 'knie']),
        ('Hip',             ['hip', 'hüfte']),
        ('Elbow',           ['elbow', 'ellbow']),
        ('Shoulder',        ['shoulder', 'schulter']),
        ('Thumb',           ['thumb', 'carpometacarpal', 'daumensattelgelenk']),
        ('Hand',            ['hand']),
        ('Finger',          ['finger']),
        ('Toe',             ['toe', 'zehe']),
        ('Trochanter',      ['trochanter']),
        ('Wrist',           ['wrist']),
    ]

    def extract_side(s):
        s_check = s.strip()
        lower = s_check.lower()
        if 'both sides' in lower:
            return 'B'
        if re.search(r'[LR]\s*[+&]\s*[LR]', s_check):
            return 'B'
        if re.search(r'[LR]\s*,\s*[LR](?!\w)', s_check):
            return 'B'
        if re.search(r'\b[LR][LR]\b', s_check):
            return 'B'
        if 'links' in lower:
            return 'L'
        if 'recht' in lower:
            return 'R'
        if 'left' in lower:
            return 'L'
        if 'right' in lower:
            return 'R'
        if re.search(r'\bL\s*$', s_check):
            return 'L'
        if re.search(r'\bR\s*$', s_check):
            return 'R'
        if re.search(r'\bl\s*$', s_check):
            return 'L'
        if re.search(r'\br\s*$', s_check):
            return 'R'
        return pd.NA

    def match_body_part(s):
        lower = s.lower().strip()
        matched = []
        for name, keywords in body_part_map:
            if any(kw in lower for kw in keywords):
                if name not in matched:
                    matched.append(name)
        if len(matched) == 0:
            return s.strip()
        return ', '.join(matched)

    body_parts = pd.Series(pd.NA, index=series.index)
    sides = pd.Series(pd.NA, index=series.index)
    for idx, val in series.items():
        if pd.isna(val):
            continue
        s = str(val).strip()
        sides[idx] = extract_side(s)
        body_parts[idx] = match_body_part(s)

    return body_parts, sides


def standardize_diagnosis(series):
    """Standardize diagnosis: map German/English variants to English names.

    Combined diagnoses are kept as 'Name1, Name2'.
    """
    diagnosis_map = [
        ('Achillodynia',          ['achillodynie', 'achilliodynie', 'achyllodynie', 'achillodynia', 'tendinitis']),
        ('Calcaneodynia',         ['calcaneodynie', 'calcaneodynia', 'heel calcaneodynia']),
        ('Heel Spur',             ['heel spur', 'fersensporn']),
        ('Elbow Syndrome',        ['ellbow', 'elbow', 'ellenbogen', 'epicondylitis', 'epiconilitis']),
        ('Rhizarthrosis',         ['rhizarthros', 'rizarthros', 'daumensattelgelenk', 'thumb cmc', 'carpometacarpal']),
        ('Gonarthrosis',          ['gonarthros', 'kniegelenk']),
        ('Finger Arthritis',      ['fingergelenk', 'fingerpolyarth', 'finger joint arthritis', 'finger arthritis']),
        ('Shoulder Syndrome',     ['shouldersyndrom', 'shoulder syndrom', 'schulter']),
        ('Ankle Arthrosis',       ['sprunggelenk', 'ankle', 'arthrosis upper ankle']),
        ('Midfoot Arthrosis',     ['mittelfuß', 'midfoot', 'forefoot']),
        ('Plantar Fasciitis',     ['plantarfasz', 'plantar']),
        ('Trochanter Tendopathy', ['trochanter']),
        ('Toe Arthrosis',         ['zehenarthros', 'zehengrundgelenk']),
        ('Rheumatoid Arthritis',  ['rheumatoid', 'rheumatoide']),
        ('Wrist Arthrosis',       ['wrist arthritis', 'wrist arthrosis', 'handgelenk']),
    ]

    def match_diagnosis(s):
        lower = s.lower().strip()
        matched = []
        for name, keywords in diagnosis_map:
            if any(kw in lower for kw in keywords):
                if name not in matched:
                    matched.append(name)
        if len(matched) == 0:
            return s.strip()
        return ', '.join(matched)

    diagnoses = pd.Series(pd.NA, index=series.index)
    for idx, val in series.items():
        if pd.isna(val):
            continue
        diagnoses[idx] = match_diagnosis(str(val).strip())

    return diagnoses


def standardize_pain_points(series):
    """Standardize pain_points: map German body parts to English, extract side (L/R/B).

    Pure number entries become NaN. Returns standardized 'BodyPart Side, BodyPart Side' format.
    """
    body_part_keywords = [
        ('Achilles Tendon', ['achillessehne']),
        ('Ankle',           ['fußgelenk', 'fußknöchel', 'knöchel']),
        ('Heel',            ['ferse', 'fersen', 'ferser', 'fersensporn', 'fersenaußenseite']),
        ('Foot',            ['fuß', 'füße', 'fußsohle', 'fußaußenseite', 'fußknochen', 'mittelfuß', 'ballen']),
        ('Toe',             ['zehen', 'zehe']),
        ('Fibula',          ['wadenbein']),
        ('Calf',            ['wade', 'waden']),
        ('Shin',            ['schienbein']),
        ('Knee',            ['knie']),
        ('Thigh',           ['oberschenkel']),
        ('Leg',             ['bein']),
        ('Hip',             ['hüfte']),
        ('Groin',           ['leistengegend', 'leiste']),
        ('Buttocks',        ['po']),
        ('Back',            ['rücken']),
        ('Neck',            ['nacken']),
        ('Shoulder',        ['schulter']),
        ('Upper Arm',       ['oberarm']),
        ('Forearm',         ['unterarm']),
        ('Elbow',           ['ellenbogen', 'ellbogen', 'ellenbogengelenk']),
        ('Arm',             [r'\barm\b']),
        ('Wrist',           ['handgelenk', 'hangelenk']),
        ('Thumb',           ['daumen', 'daumensattelgelenk']),
        ('Hand',            [r'\bhand\b', 'hände']),
        ('Finger',          ['finger']),
    ]

    def find_side(seg):
        s = seg.lower().strip()
        if re.search(r'beide|bds', s):
            return 'B'
        if re.search(r'li\s*[+&/]\s*re|re\s*[+&/]\s*li', s):
            return 'B'
        if re.search(r'li\s+u\.?\s+re|re\s+u\.?\s+li', s):
            return 'B'
        if re.search(r'li\s+und\s+re|re\s+und\s+li', s):
            return 'B'
        if re.search(r'\bli\b|\blinks\b|\blinke[rns]?\b', s):
            return 'L'
        if re.search(r'\bre\b|\brechts\b|\brechte[rns]?\b|\brecht\b', s):
            return 'R'
        return ''

    def find_body_part(seg):
        s = seg.lower().strip()
        for name, keywords in body_part_keywords:
            for kw in keywords:
                if kw.startswith(r'\b'):
                    if re.search(kw, s):
                        return name
                else:
                    if kw in s:
                        return name
        return None

    def parse_entry(val):
        if pd.isna(val):
            return pd.NA
        s = str(val).strip()
        if re.match(r'^\d+$', s):
            return pd.NA
        s_clean = s.replace('(', '').replace(')', '')
        s_clean = re.sub(r'[?]', '', s_clean)
        s_clean = re.sub(r'(\D)\d+\b', r'\1', s_clean)
        segments = re.split(r'[,;]', s_clean)
        results = []
        last_body_part = None

        for seg in segments:
            seg = seg.strip()
            if not seg:
                continue
            body = find_body_part(seg)
            side = find_side(seg)
            if body is None and side and last_body_part:
                body = last_body_part
            if body:
                entry = f"{body} {side}".strip()
                if entry not in results:
                    results.append(entry)
                last_body_part = body

        if not results:
            return s.strip()

        sides_by_part = defaultdict(set)
        order = []
        for entry in results:
            parts = entry.rsplit(' ', 1)
            if len(parts) == 2 and parts[1] in ('L', 'R', 'B'):
                part, side = parts
            else:
                part, side = entry, ''
            if part not in order:
                order.append(part)
            sides_by_part[part].add(side)

        merged = []
        for part in order:
            sides = sides_by_part[part]
            if 'B' in sides or ('L' in sides and 'R' in sides):
                merged.append(f"{part} B")
            elif 'L' in sides:
                merged.append(f"{part} L")
            elif 'R' in sides:
                merged.append(f"{part} R")
            else:
                merged.append(part)

        return ', '.join(merged)

    return series.apply(parse_entry)


def split_filter_column(df, col_name='filter'):
    """Split filter column into filter_mm (float) and filter_material (Cu/Al).

    Handles German decimal commas, duplicate entries, and various formats.
    """
    col_idx = df.columns.get_loc(col_name)

    def parse_filter(val):
        if pd.isna(val):
            return pd.NA, pd.NA
        s = str(val).strip()
        s = s.split('\n')[0].strip()
        material = pd.NA
        if re.search(r'Cu', s, re.IGNORECASE):
            material = 'Cu'
        elif re.search(r'Al', s, re.IGNORECASE):
            material = 'Al'
        num_match = re.search(r'(\d+[,.]?\d*)', s)
        if num_match:
            num_str = num_match.group(1).replace(',', '.')
            return float(num_str), material
        return pd.NA, material

    parsed = df[col_name].apply(parse_filter)
    df.insert(col_idx, 'filter_mm', parsed.apply(lambda x: x[0]))
    df.insert(col_idx + 1, 'filter_material', parsed.apply(lambda x: x[1]))
    return df.drop(columns=[col_name])


def parse_cumulative_dose(val):
    """Parse total dose from mixed format strings.

    Handles: "L: 3; R: 6" (sum both sides), "3(6)" (prefer parenthesized total),
    "3\\n3" duplicates (take first line), standalone numbers.
    """
    if pd.isna(val):
        return pd.NA
    s = str(val).strip()
    if re.search(r'[LR]\s*:', s):
        numbers = re.findall(r'(\d+\.?\d*)', s)
        return sum(float(n) for n in numbers) if numbers else pd.NA
    paren_match = re.search(r'\((\d+\.?\d*)', s)
    if paren_match:
        return float(paren_match.group(1))
    s = s.split('\n')[0].strip()
    num_match = re.match(r'^(\d+\.?\d*)$', s)
    if num_match:
        return float(num_match.group(1))
    return pd.NA


def encode_therapy_columns(df, col_name='previous_therapy'):
    """Encode comma-separated therapy codes (1–7) into binary indicator columns.

    Input: "1,3,5" or "1,2,3 (medicine)".
    Output: previous_therapy_1 … previous_therapy_7 (int 0/1).
    """
    col_idx = df.columns.get_loc(col_name)
    for i in range(1, 8):
        binary_col = df[col_name].str.contains(rf'\b{i}\b', na=False).astype(int)
        df.insert(col_idx + i - 1, f'previous_therapy_{i}', binary_col)
    return df.drop(columns=[col_name])


def standardize_response(df, response_col='response', verbose=True):
    """Parse raw response column into response_category (CR/PR/NI) and response_percent (float).

    response_category is metadata only — NOT a modeling target.
    Multiple categories in one entry are kept as comma-separated ('CR, NI', 'PR, CR').
    Unrecognized entries are kept as-is for manual review.
    """
    df = df.copy()
    raw = df[response_col].astype(str).str.strip()

    phrase_map = {
        'no improvement':                  'ni',
        'no imrovement':                   'ni',
        'no imrpvovemnet':                 'ni',
        'recovery only on the right side': 'pr',
        'initial improvement':             'pr',
        'subtotal remission':              'pr',
        'improvement':                     'pr',
        'pd':                              'pr',
    }

    categories = pd.Series(pd.NA, index=df.index, dtype=object)
    percents   = pd.Series(np.nan, index=df.index, dtype='float64')
    _null_marker_pat = re.compile(r'^([kK]\.?[aA]\.?|[nN]\.?[dD]\.?)$')

    for idx, val in raw.items():
        if val in ('nan', '', 'None', 'NaN'):
            continue
        if _null_marker_pat.match(val.strip()):
            continue

        s = val.lower().strip()
        for phrase, replacement in phrase_map.items():
            s = s.replace(phrase.lower(), replacement)

        s = re.sub(r'\b([lr])\s*[>~=]\s*(\d+)', r'pr > \2', s)

        range_m  = re.search(r'(\d+)\s*[-–]\s*(\d+)', s)
        single_m = re.search(r'[>~<]?\s*(\d+)\s*%?', s)
        if range_m:
            percents[idx] = (float(range_m.group(1)) + float(range_m.group(2))) / 2
        elif single_m:
            percents[idx] = float(single_m.group(1))

        found = []
        if re.search(r'\bni\b', s):
            found.append('NI')
        if re.search(r'\bcr\b', s):
            found.append('CR')
        if re.search(r'\bpr\b', s):
            found.append('PR')

        if found:
            categories[idx] = ', '.join(found)
        else:
            categories[idx] = val.strip()

    df['response_category'] = categories.astype('category')
    df['response_percent']  = percents

    if verbose:
        print(f"\nResponse categories:\n"
              f"{df['response_category'].value_counts(dropna=False).to_string()}")
        print(f"\nResponse percent — {df['response_percent'].notna().sum()} "
              f"entries with a numeric value:")
        print(df['response_percent'].describe())

    return df


# ── Clinical pipeline functions ───────────────────────────────────────────────

def forward_fill_clinical(df_cl, verbose=True):
    """Create working copy: forward-fill patient-level columns and extract Timepoint.

    Patient-level columns (demographics, treatment parameters) are only filled
    in the first row per patient in the raw Excel format.  This step propagates
    them to all timepoint rows.

    Parameters
    ----------
    df_cl   : pd.DataFrame   raw clinical data as loaded from Excel
    verbose : bool

    Returns
    -------
    pd.DataFrame — working copy with forward-filled columns and Timepoint extracted
    """
    df = df_cl.copy()
    fill_cols = [c for c in CL_PATIENT_LEVEL_COLS + ['Unnamed: 0'] if c in df.columns]

    df['Patient_Group'] = df['Patient'].notna().cumsum()
    df[fill_cols] = df.groupby('Patient_Group')[fill_cols].ffill()
    df = df.drop(columns=['Patient_Group'])

    # Extract timepoint number from Erfassungszeitpunkt (e.g., "01.01.1" → 1)
    if 'Erfassungszeitpunkt' in df.columns:
        df['Timepoint'] = (
            df['Erfassungszeitpunkt']
            .str.extract(r'\d+\.\d+\.(\d+)')[0]
            .astype(float)
        )

    if verbose:
        print(f"  df_cl initialised: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def exclude_patients_cl(df_cl_clean, multi_body_patients=None, verbose=True):
    """Exclude Ausschluss patients, multi-body-part patients, + EORTC columns.

    Steps
    -----
    1. Remove rows flagged as 'Ausschluss' in the Unnamed: 0 column
    2. Remove patients irradiated at multiple body parts (CL_MULTI_BODY_PATIENTS)
    3. Drop the EORTC health/function questionnaire column range

    Parameters
    ----------
    df_cl_clean         : pd.DataFrame
    multi_body_patients : list of int, optional — defaults to CL_MULTI_BODY_PATIENTS
    verbose             : bool

    Returns
    -------
    pd.DataFrame — copy with excluded patients and EORTC columns removed
    """
    if multi_body_patients is None:
        multi_body_patients = CL_MULTI_BODY_PATIENTS

    df = df_cl_clean.copy()

    # 1 — Ausschluss exclusion
    if 'Unnamed: 0' in df.columns:
        exclude_mask = df['Unnamed: 0'].str.contains('Ausschluss', case=False, na=False)
        excluded_patients = df.loc[exclude_mask, 'Patient'].dropna().unique()
        if verbose:
            print(f"  Excluded {len(excluded_patients)} patients by Ausschluss keyword: "
                  f"{excluded_patients}")
        df = df[~exclude_mask]

    # 2 — Multi-body-part exclusion
    if verbose:
        print(f"\n  Verifying multi-body-part patients (to be excluded):")
        for pid in multi_body_patients:
            rows = df[df['Patient'] == pid]
            if len(rows) > 0:
                col = 'Target volume' if 'Target volume' in df.columns else 'target_volume'
                volumes = rows[col].dropna().unique() if col in df.columns else []
                print(f"    Patient {pid}: Target volume(s) = {volumes}")
            else:
                print(f"    Patient {pid}: not found in dataset")
    df = df[~df['Patient'].isin(multi_body_patients)]
    if verbose:
        print(f"  Removed {len(multi_body_patients)} multi-body-part patients")

    # 3 — Drop EORTC questionnaire column range
    q_cols_to_drop = []
    try:
        col_list    = df.columns.tolist()
        start_col   = 'Schwierigkeiten körperlicher Anstrengung'
        end_col_options = [
            'Allgemeinzustand Gesundheit HEUTE',
            'Allgemeinzustand Gesundheut HEUTE',
        ]
        end_col = next((c for c in end_col_options if c in col_list), None)
        if start_col not in col_list:
            if verbose:
                print(f"  Warning: EORTC start column '{start_col}' not found — no columns dropped")
        elif end_col is None:
            if verbose:
                print(f"  Warning: EORTC end column not found — no columns dropped")
        else:
            start_idx      = col_list.index(start_col)
            end_idx        = col_list.index(end_col)
            q_cols_to_drop = col_list[start_idx : end_idx + 1]
            df = df.drop(columns=q_cols_to_drop)
            if verbose:
                print(f"\n  Dropped {len(q_cols_to_drop)} EORTC questionnaire columns "
                      f"('{start_col}' to '{end_col}')")
    except Exception as e:
        if verbose:
            print(f"  Warning: Could not drop EORTC columns: {e}")

    if verbose:
        print(f"\n  After exclusions: {df['Patient'].nunique()} patients, {len(df)} rows")

    return df


def rename_columns_cl(df_cl_clean, rename_map=None, verbose=True):
    """Rename clinical columns from German to English using CL_RENAME_MAP.

    Also moves the Timepoint column to immediately after Patient.

    Parameters
    ----------
    df_cl_clean : pd.DataFrame
    rename_map  : dict, optional — defaults to CL_RENAME_MAP
    verbose     : bool

    Returns
    -------
    pd.DataFrame — copy with renamed columns
    """
    if rename_map is None:
        rename_map = CL_RENAME_MAP

    df = df_cl_clean.rename(columns=rename_map)
    if 'Timepoint' in df.columns and 'Patient' in df.columns:
        df = move_column_after(df, 'Timepoint', 'Patient')

    if verbose:
        actual_renames = {k: v for k, v in rename_map.items()
                          if k in df_cl_clean.columns and k != v}
        print(f"  Renamed {len(actual_renames)} columns")
    return df


def drop_unused_cl(df_cl_clean, verbose=True):
    """Drop metadata columns and rows with no measurement date.

    Steps
    -----
    1. Drop admin columns ('Unnamed: 0', 'Unnamed: 2', 'further comments',
       'Comments questionnaire')
    2. Drop rows with no date (completely empty measurement slots from Excel)

    Note: The all-NaN clinical row check is intentionally NOT done here.
    It is deferred to clean_cl step [7b], which runs AFTER NaN marker
    replacement (step 7), so that rows whose entries consist entirely of
    k.A./n.D. markers are also caught.

    Requires columns already renamed via rename_columns_cl.

    Parameters
    ----------
    df_cl_clean : pd.DataFrame
    verbose     : bool

    Returns
    -------
    pd.DataFrame — copy with unused columns and empty rows removed
    """
    df = df_cl_clean.copy()

    # 1 — Drop metadata/admin columns
    cols_to_drop = ['Unnamed: 0', 'Unnamed: 2', 'further comments', 'Comments questionnaire']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # 2 — Drop rows with no date (blank Excel rows)
    if 'date' in df.columns:
        no_date = df[df['date'].isna()]
        df = df[df['date'].notna()].copy()
        if verbose:
            if len(no_date) > 0:
                dropped_pts = sorted(no_date['Patient'].dropna().unique().tolist())
                print(f"  Dropped {len(no_date)} rows with no date "
                      f"(Patient IDs in dropped rows: {dropped_pts})")
            else:
                print(f"  Dropped 0 rows with no date")
            print(f"  Unique patients remaining: {df['Patient'].nunique()}")

    if verbose:
        print(f"\n  After drop_unused: {df['Patient'].nunique()} patients, {len(df)} rows")
    return df


def manual_corrections_cl(df_cl_clean, verbose=True):
    """Apply known manual data corrections to the clinical dataset.

    Corrections applied
    -------------------
    - Patient 248 T2 : pain_daytime '22' → '2'  (confirmed typo)
    - Patient 219    : removed (used a different questionnaire)
    - Patient 89     : assign correct timepoints by date (T2 = 27.03.2019,
                       T5 = 05.07.2019); drop unmatched row (10.05.2019)
    - Patient 113    : filter value corrected to '0.2mm'
    - Patient 182    : filter value '32mm' (typo, true value unknown) → NaN

    Parameters
    ----------
    df_cl_clean : pd.DataFrame
    verbose     : bool

    Returns
    -------
    pd.DataFrame — copy with corrections applied
    """
    df = df_cl_clean.copy()

    # Patient 248 T2: pain_daytime typo
    mask_248 = (df['Patient'] == 248) & (df['Timepoint'] == 2)
    if mask_248.sum() > 0:
        df.loc[mask_248, 'pain_daytime'] = '2'
        if verbose:
            print("  Patient 248 T2 pain_daytime set to '2' (was '22')")
    elif verbose:
        print("  Warning: Patient 248 T2 not found — correction skipped")

    # Patient 219: different questionnaire — remove all rows
    n_before = len(df)
    df = df[df['Patient'] != 219].copy()
    if verbose:
        print(f"  Removed Patient 219 (different questionnaire): "
              f"{n_before - len(df)} rows dropped")

    # Patient 89: assign correct timepoints by date, drop unmatched row
    p89          = df['Patient'] == 89
    mask_89_t2   = p89 & (df['date'] == dt.datetime(2019, 3, 27))
    mask_89_t5   = p89 & (df['date'] == dt.datetime(2019, 7,  5))
    mask_89_drop = p89 & (df['date'] == dt.datetime(2019, 5, 10))

    if mask_89_t2.sum() > 0:
        df.loc[mask_89_t2, 'Timepoint'] = 2
        if verbose:
            print("  Patient 89 row 27.03.2019 → Timepoint 2")
    elif verbose:
        print("  Warning: Patient 89 row dated 27.03.2019 not found")

    if mask_89_t5.sum() > 0:
        df.loc[mask_89_t5, 'Timepoint'] = 5
        if verbose:
            print("  Patient 89 row 05.07.2019 → Timepoint 5")
    elif verbose:
        print("  Warning: Patient 89 row dated 05.07.2019 not found")

    if mask_89_drop.sum() > 0:
        df = df[~mask_89_drop].copy()
        if verbose:
            print("  Removed Patient 89 row dated 10.05.2019 (unmatched timepoint)")
    elif verbose:
        print("  Warning: Patient 89 row dated 10.05.2019 not found — removal skipped")

    # Patient 113: filter value typo
    mask_113 = df['Patient'] == 113
    if mask_113.sum() > 0:
        df.loc[mask_113, 'filter'] = '0.2mm'
        if verbose:
            print(f"  Patient 113 filter set to '0.2mm' ({mask_113.sum()} rows)")
    elif verbose:
        print("  Warning: Patient 113 not found — filter correction skipped")

    # Patient 182: filter value '32mm' is a typo, true value unknown → NaN
    mask_182 = df['Patient'] == 182
    if mask_182.sum() > 0:
        df.loc[mask_182, 'filter'] = pd.NA
        if verbose:
            print(f"  Patient 182 filter set to NaN (was '32mm', {mask_182.sum()} rows)")
    elif verbose:
        print("  Warning: Patient 182 not found — filter correction skipped")

    return df


def parse_transform_cl(df_cl_clean, verbose=True):
    """Parse and transform all clinical columns to structured values.

    Steps
    -----
    1.  diagnosis       → standardized English names
    2.  target_volume   → body part + side combined string
    3.  pain_points     → standardized English body parts + side
    4.  filter          → filter_mm (float) + filter_material (Cu/Al)
    5.  cumulative_dose → numeric (Gy)
    6.  gender          → 'w' → 'f'
    7.  overweight_bmi  → overweight (ja/nein) + bmi (float)
    8.  symptoms_months → numeric months
    9.  previous_therapy→ binary columns previous_therapy_1 … _7
    10. response        → response_category + response_percent
    11. ordinal columns → extract_numeric
    12. pain_scale      → extract_continuous (handles German decimal comma)

    Requires columns already renamed via rename_columns_cl.

    Parameters
    ----------
    df_cl_clean : pd.DataFrame
    verbose     : bool — print before/after value distributions for each column

    Returns
    -------
    pd.DataFrame — copy with all columns parsed and transformed
    """
    df = df_cl_clean.copy()

    # 1 — diagnosis
    if verbose:
        print("\n=== diagnosis (before) ===")
        print(df['diagnosis'].value_counts(dropna=False).to_string())
    df['diagnosis'] = standardize_diagnosis(df['diagnosis'])
    if verbose:
        print("\n=== diagnosis (after) ===")
        print(df['diagnosis'].value_counts().to_dict())

    # 2 — target_volume: standardize + merge side into one string
    if verbose:
        print("\n=== target_volume (before) ===")
        print(df['target_volume'].value_counts(dropna=False).head(20).to_string())
    df['target_volume'], df['target_side'] = standardize_target_volume(df['target_volume'])
    df = move_column_after(df, 'target_side', 'target_volume')
    df['target_volume'] = df.apply(
        lambda r: f"{r['target_volume']} {r['target_side']}".strip()
                  if pd.notna(r['target_volume']) and pd.notna(r['target_side'])
                     and r['target_side'] != ''
                  else r['target_volume'],
        axis=1,
    )
    df = df.drop(columns=['target_side'])
    if verbose:
        print("\n=== target_volume (after) ===")
        print(df['target_volume'].value_counts().to_dict())

    # 3 — pain_points
    if verbose:
        print("\n=== pain_points (before) ===")
        print(df['pain_points'].value_counts(dropna=False).head(20).to_string())
    df['pain_points'] = standardize_pain_points(df['pain_points'])
    if verbose:
        print("\n=== pain_points (after) ===")
        print(df['pain_points'].value_counts().head(20).to_dict())

    # 4 — filter → filter_mm + filter_material
    if 'filter' in df.columns:
        if verbose:
            print("\n=== filter (before) ===")
            print(df['filter'].value_counts(dropna=False).to_string())
        df = split_filter_column(df)
        if verbose:
            print("\n=== filter (after) ===")
            print(f"  filter_mm      : {sorted(df['filter_mm'].dropna().unique())}")
            print(f"  filter_material: {df['filter_material'].value_counts().to_dict()}")

    # 5 — cumulative_dose
    if 'cumulative_dose' in df.columns:
        if verbose:
            print("\n=== cumulative_dose (before) ===")
            print(df['cumulative_dose'].value_counts(dropna=False).to_string())
        df['cumulative_dose'] = pd.to_numeric(
            df['cumulative_dose'].apply(parse_cumulative_dose), errors='coerce'
        )
        if verbose:
            print("\n=== cumulative_dose (after) ===")
            print(sorted(df['cumulative_dose'].dropna().unique()))

    # 6 — gender: 'w' → 'f'
    if 'gender' in df.columns:
        if verbose:
            print("\n=== gender (before) ===")
            print(df['gender'].value_counts(dropna=False).to_string())
        df['gender'] = df['gender'].replace('w', 'f')
        if verbose:
            print("\n=== gender (after) ===")
            print(df['gender'].value_counts().to_dict())

    # 7 — overweight_bmi → overweight + bmi
    if 'overweight_bmi' in df.columns:
        if verbose:
            print("\n=== overweight_bmi (before) ===")
            print(df['overweight_bmi'].value_counts(dropna=False).head(20).to_string())
        df = split_bmi_column(df)
        if verbose:
            print("\n=== overweight / bmi (after) ===")
            print(f"  overweight: {df['overweight'].value_counts().to_dict()}")
            bmi_valid = df['bmi'].dropna()
            if len(bmi_valid) > 0:
                print(f"  bmi: range {bmi_valid.min():.1f}–{bmi_valid.max():.1f}, "
                      f"{df['bmi'].isna().sum()} missing")

    # 8 — symptoms_months: parse duration strings to numeric months
    if 'symptoms_months' in df.columns:
        if verbose:
            print("\n=== symptoms_months (before) ===")
            print(df['symptoms_months'].value_counts(dropna=False).head(20).to_string())
        date_col = df['date'] if 'date' in df.columns else None
        df['symptoms_months'] = pd.to_numeric(
            parse_symptoms_duration(df['symptoms_months'], date_col), errors='coerce'
        )
        if verbose:
            valid = df['symptoms_months'].dropna()
            if len(valid) > 0:
                print(f"\n=== symptoms_months (after) ===")
                print(f"  range {valid.min():.0f}–{valid.max():.0f} months, "
                      f"{df['symptoms_months'].isna().sum()} missing")

    # 9 — previous_therapy → binary indicator columns
    if 'previous_therapy' in df.columns:
        if verbose:
            print("\n=== previous_therapy (before) ===")
            print(df['previous_therapy'].value_counts(dropna=False).head(20).to_string())
        df = encode_therapy_columns(df)
        if verbose:
            therapy_cols = [f'previous_therapy_{i}' for i in range(1, 8)
                            if f'previous_therapy_{i}' in df.columns]
            print("\n=== previous_therapy (after: binary columns) ===")
            print(df[therapy_cols].sum().to_dict())

    # 10 — response → response_category + response_percent
    if 'response' in df.columns:
        df = standardize_response(df, response_col='response', verbose=verbose)
        if 'response_category' in df.columns:
            df = move_column_after(df, 'response_category', 'response')
        if 'response_percent' in df.columns:
            df = move_column_after(df, 'response_percent', 'response_category')

    # 11 — Ordinal questionnaire columns → extract numeric
    ordinal_cols = ['pain_under_load', 'pain_at_rest', 'pain_daytime',
                    'pain_night', 'morning_stiffness']
    if verbose:
        print("\n=== Ordinal questionnaire columns (before extraction) ===")
        for col in ordinal_cols:
            if col in df.columns:
                uniq = df[col].dropna().unique()
                print(f"\n  {col} ({len(uniq)} unique):")
                for v in sorted(uniq, key=lambda x: str(x)):
                    print(f"    {repr(v)}")
        print("\n=== Extracting ordinal values ===")
    for col in ordinal_cols:
        if col in df.columns:
            df[col] = extract_numeric(df[col])
            if verbose:
                print(f"  {col}: unique after = {sorted(df[col].dropna().unique())}")

    # 12 — pain_scale (continuous): German decimal comma, ranges → midpoint
    if 'pain_scale' in df.columns:
        if verbose:
            print("\n=== pain_scale (before extraction) ===")
            uniq_ps = df['pain_scale'].dropna().unique()
            print(f"  pain_scale ({len(uniq_ps)} unique):")
            for v in sorted(uniq_ps, key=lambda x: str(x)):
                print(f"    {repr(v)}")
        df['pain_scale'] = extract_continuous(df['pain_scale'])
        if verbose:
            uniq_after = sorted(df['pain_scale'].dropna().unique())
            print(f"\n=== pain_scale (after extraction) ===")
            print(f"  pain_scale ({len(uniq_after)} unique): {uniq_after}")

    return df


def fix_dtypes_cl(df_cl_clean, verbose=True):
    """Convert clinical dataset columns to correct dtypes.

    - Patient, Timepoint : coerce to numeric; drop rows where unparseable; cast to int64
    - measurement_timepoint : str
    - date               : datetime64
    - Categorical columns (CL_CATEGORICAL_COLS) : category
    - All other columns  : float64 (coerce non-numeric to NaN)

    Parameters
    ----------
    df_cl_clean : pd.DataFrame
    verbose     : bool

    Returns
    -------
    pd.DataFrame — copy with correct dtypes
    """
    df = df_cl_clean.copy()

    df['Patient']   = pd.to_numeric(df['Patient'],   errors='coerce')
    df['Timepoint'] = pd.to_numeric(df['Timepoint'], errors='coerce')

    n_before  = len(df)
    _bad_rows = df[df[['Patient', 'Timepoint']].isna().any(axis=1)]
    if len(_bad_rows) > 0 and verbose:
        print(f"  Rows with unparseable Patient or Timepoint (dropping):")
        print(_bad_rows[['Patient', 'Timepoint']].to_string())
    df = df.dropna(subset=['Patient', 'Timepoint']).copy()
    if verbose and (n_before - len(df)) > 0:
        print(f"  Dropped {n_before - len(df)} rows with unparseable Patient/Timepoint")

    df['Patient']   = df['Patient'].astype('int64')
    df['Timepoint'] = df['Timepoint'].astype('int64')

    if 'measurement_timepoint' in df.columns:
        df['measurement_timepoint'] = df['measurement_timepoint'].astype(str)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

    for col in CL_CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype('category')

    exclude_for_float = (set(CL_CATEGORICAL_COLS) |
                         {'Patient', 'Timepoint', 'measurement_timepoint', 'date'})
    cols_to_float = [c for c in df.columns if c not in exclude_for_float]
    df[cols_to_float] = (
        df[cols_to_float]
        .apply(lambda s: pd.to_numeric(s, errors='coerce'))
        .astype('float64')
    )

    if verbose:
        print("\n=== Dtype summary (clinical) ===")
        print(df.dtypes.value_counts())
        print(f"Shape: {df.shape}, Patients: {df['Patient'].nunique()}")

    return df


def remove_no_pain_scale_rows(df, verbose=True):
    """Remove rows where pain_scale is NaN.

    Parameters
    ----------
    df      : pd.DataFrame
    verbose : bool

    Returns
    -------
    pd.DataFrame — copy with pain_scale NaN rows removed and index reset
    """
    n_before        = len(df)
    patients_before = df['Patient'].nunique()
    result = df[df['pain_scale'].notna()].reset_index(drop=True)
    if verbose:
        print(f"  Dropped {n_before - len(result)} rows with NaN pain_scale "
              f"({patients_before - result['Patient'].nunique()} patients lost)")
        print(f"  Shape: {result.shape}, Patients: {result['Patient'].nunique()}")
    return result


def create_target_variables(df_cl_vis, df_cl_mod=None, verbose=True):
    """Compute pain reduction targets from T1 and T2 pain_scale values.

    Targets
    -------
    pain_scale_t1        : T1 pain level (float)
    pain_scale_t2        : T2 pain level — secondary regression target
    pain_scale_reduction : T1 − T2 absolute reduction (reference only)
    pain_reduction_pct   : (T1 − T2) / T1 × 100 — primary regression target

    Only patients with BOTH T1 and T2 pain_scale values are included.
    Raises ValueError if any patient has pain_scale_t1 = 0 (undefined percentage).

    Parameters
    ----------
    df_cl_vis : pd.DataFrame  — full clinical dataset (T1–T5, used to extract targets)
    df_cl_mod : pd.DataFrame, optional
        If given, show per-timepoint pain_scale distribution from df_cl_mod.
    verbose   : bool

    Returns
    -------
    pain_targets : pd.DataFrame
        One row per patient with columns:
        Patient, pain_scale_t1, pain_scale_t2, pain_scale_reduction, pain_reduction_pct
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    pain_t1 = (
        df_cl_vis[df_cl_vis['Timepoint'] == 1][['Patient', 'pain_scale']]
        .rename(columns={'pain_scale': 'pain_scale_t1'})
        .dropna(subset=['pain_scale_t1'])
    )
    pain_t2 = (
        df_cl_vis[df_cl_vis['Timepoint'] == 2][['Patient', 'pain_scale']]
        .rename(columns={'pain_scale': 'pain_scale_t2'})
        .dropna(subset=['pain_scale_t2'])
    )

    pain_targets = pain_t1.merge(pain_t2, on='Patient', how='inner')
    pain_targets['pain_scale_reduction'] = (
        pain_targets['pain_scale_t1'] - pain_targets['pain_scale_t2']
    )

    zero_t1 = pain_targets[pain_targets['pain_scale_t1'] == 0]
    if len(zero_t1) > 0:
        raise ValueError(
            f"Cannot compute pain_reduction_pct: {len(zero_t1)} patient(s) have "
            f"pain_scale_t1 = 0: {zero_t1['Patient'].tolist()}"
        )

    pain_targets['pain_reduction_pct'] = (
        (pain_targets['pain_scale_t1'] - pain_targets['pain_scale_t2'])
        / pain_targets['pain_scale_t1'] * 100
    )

    if verbose:
        print(f"\nPatients with T1 + T2 pain_scale (usable for regression): {len(pain_targets)}")
        print(f"pain_scale_t2 range:      "
              f"{pain_targets['pain_scale_t2'].min():.1f} – "
              f"{pain_targets['pain_scale_t2'].max():.1f}")
        print(f"pain_scale_reduction:     "
              f"{pain_targets['pain_scale_reduction'].min():.1f} – "
              f"{pain_targets['pain_scale_reduction'].max():.1f} pts")
        print(f"pain_reduction_pct range: "
              f"{pain_targets['pain_reduction_pct'].min():.1f} – "
              f"{pain_targets['pain_reduction_pct'].max():.1f} %  "
              f"(positive = improvement, negative = worsening)")
        print(f"pain_reduction_pct stats:\n{pain_targets['pain_reduction_pct'].describe()}")

        colors = sns.color_palette('mako', 5)
        fig, axes = plt.subplots(1, 3, figsize=(18, 4))

        axes[0].hist(pain_targets['pain_scale_t2'].dropna(), bins=20, color=colors[1])
        axes[0].set_title('pain_scale_t2 (T2 pain level)')
        axes[0].set_xlabel('Pain Scale (0–10)')
        axes[0].set_ylabel('Number of Patients')

        axes[1].hist(pain_targets['pain_scale_reduction'].dropna(), bins=20, color=colors[2])
        axes[1].set_title('pain_scale_reduction (T1 − T2 pts, reference)')
        axes[1].set_xlabel('Point Reduction (positive = improvement)')
        axes[1].axvline(0, color='white', linestyle='--', linewidth=1, label='No change')
        axes[1].legend()

        axes[2].hist(pain_targets['pain_reduction_pct'].dropna(), bins=20, color=colors[3])
        axes[2].set_title('pain_reduction_pct (% relative to T1, primary target)')
        axes[2].set_xlabel('Pain Reduction (%)')
        axes[2].axvline(0, color='white', linestyle='--', linewidth=1, label='No change')
        axes[2].legend()

        plt.suptitle('Distribution of Potential Regression Targets', fontweight='bold')
        plt.tight_layout()
        plt.show()

        # Per-timepoint pain_scale distribution (only when df_cl_mod is provided)
        if df_cl_mod is not None:
            timepoints = sorted(df_cl_mod['Timepoint'].dropna().unique().astype(int))
            colors_tp  = sns.color_palette('mako', len(timepoints))
            n_cols     = min(3, len(timepoints))
            n_rows     = (len(timepoints) + n_cols - 1) // n_cols
            fig2, axes2 = plt.subplots(n_rows, n_cols,
                                       figsize=(6 * n_cols, 4 * n_rows),
                                       squeeze=False)
            axes2_flat = axes2.flatten()

            for i, (tp, color) in enumerate(zip(timepoints, colors_tp)):
                data = df_cl_mod.loc[df_cl_mod['Timepoint'] == tp, 'pain_scale'].dropna()
                axes2_flat[i].hist(data, bins=15, color=color, edgecolor='white')
                axes2_flat[i].set_title(f'T{tp}  (n={len(data)})')
                axes2_flat[i].set_xlabel('Pain Scale (0–10)')
                axes2_flat[i].set_ylabel('Count')
                if len(data) > 0:
                    axes2_flat[i].axvline(data.median(), color='white', linestyle='--',
                                          linewidth=1.5, label=f'Median {data.median():.1f}')
                    axes2_flat[i].legend(fontsize=9)

            for j in range(len(timepoints), len(axes2_flat)):
                axes2_flat[j].set_visible(False)

            plt.suptitle('Distribution of pain_scale by Timepoint', fontweight='bold')
            plt.tight_layout()
            plt.show()

    return pain_targets


def clean_cl(df_cl, verbose=True):
    """Full cleaning pipeline for the raw clinical dataset.

    Steps
    -----
    1.  Forward-fill patient-level columns + extract Timepoint  (forward_fill_clinical)
    2.  Exclude Ausschluss patients, multi-body patients, EORTC columns  (exclude_patients_cl)
    3.  Rename columns German → English  (rename_columns_cl)
    4.  Drop admin columns + empty/all-NaN-questionnaire rows  (drop_unused_cl)
    5.  Apply manual data corrections  (manual_corrections_cl)
    6.  Parse/transform all columns  (parse_transform_cl)
    7.  Replace German NaN markers in-place  (replace_missing_markers)
    7b. Drop rows where date is NaN OR all columns from symptoms_months
        onwards are NaN (runs AFTER marker replacement so k.A./n.D. rows
        are caught too). Flag questionnaire_missing=True for rows where
        pain_under_load → pain_points are all NaN.
    8.  Fix dtypes  (fix_dtypes_cl)
    9.  Make baseline CatBoost copy  (df_cl_bcat)
    10. Drop columns with >25% NaN  (drop_high_nan_columns)
    11. Make EDA/visualization copy  (df_cl_vis)

    Parameters
    ----------
    df_cl   : pd.DataFrame   raw clinical data as loaded from Excel
    verbose : bool

    Returns
    -------
    df_cl_clean : cleaned dataset (after dtype fix, before >25% NaN drop)
    df_cl_bcat  : copy taken before >25% NaN drop (for baseline CatBoost)
    df_cl_vis   : copy taken after  >25% NaN drop (for EDA / visualization)
    """
    if verbose:
        print("\n  [1] Forward-filling patient-level columns + extracting Timepoint")
    df_cl_clean = forward_fill_clinical(df_cl, verbose=verbose)

    if verbose:
        print("\n  [2] Excluding patients and EORTC columns")
    df_cl_clean = exclude_patients_cl(df_cl_clean, verbose=verbose)

    if verbose:
        print("\n  [3] Renaming columns German → English")
    df_cl_clean = rename_columns_cl(df_cl_clean, verbose=verbose)

    if verbose:
        print("\n  [4] Dropping unused columns and empty rows")
    df_cl_clean = drop_unused_cl(df_cl_clean, verbose=verbose)

    if verbose:
        print("\n  [5] Applying manual corrections")
    df_cl_clean = manual_corrections_cl(df_cl_clean, verbose=verbose)

    if verbose:
        print("\n  [6] Parsing and transforming columns")
    df_cl_clean = parse_transform_cl(df_cl_clean, verbose=verbose)

    if verbose:
        print("\n  [7] Replacing German NaN markers")
    replace_missing_markers(df_cl_clean, skip_cols=["Patient", "Timepoint"], verbose=verbose)

    # [7b] — Drop empty rows + flag missing questionnaire data
    # Runs AFTER NaN marker replacement so k.A./n.D. entries are already NaN.
    #
    # Drop condition (either is sufficient):
    #   (a) date is NaN  — belt-and-suspenders over step [4]; also catches any
    #       edge cases where a date became NaN after parsing.
    #   (b) every column from symptoms_months onwards is NaN — patient had a
    #       valid date but provided no clinical data at all (or all entries
    #       were k.A./n.D. markers now converted to NaN).
    # --- Drop rows with missing date or no clinical data ---
    
    drop_mask = pd.Series(False, index=df_cl_clean.index)

    if 'date' in df_cl_clean.columns:
        drop_mask |= df_cl_clean['date'].isna()

    if 'symptoms_months' in df_cl_clean.columns:
        from_sym = df_cl_clean.loc[:, 'symptoms_months':]
        drop_mask |= from_sym.isna().all(axis=1)

    if verbose and drop_mask.any():
        print(f"\n  [7b] Dropping {drop_mask.sum()} rows "
            f"(date NaN or all columns from symptoms_months onwards NaN):")
        print(df_cl_clean.loc[drop_mask, ['Patient', 'Timepoint']].to_string())

    df_cl_clean = df_cl_clean.loc[~drop_mask].copy()

    # --- Drop rows with missing questionnaire ---
    if {'pain_under_load', 'pain_points'}.issubset(df_cl_clean.columns):

        q_mask = df_cl_clean.loc[:, 'pain_under_load':'pain_points'].isna().all(axis=1)

        if verbose and q_mask.any():
            print(f"\n  [7c] Dropping {q_mask.sum()} rows with missing questionnaire:")
            print(df_cl_clean.loc[q_mask, ['Patient', 'Timepoint']].to_string())

        df_cl_clean = df_cl_clean.loc[~q_mask].copy()

    if verbose:
        print(f"\n  Unique patients remaining: {df_cl_clean['Patient'].nunique()}")
    
    if verbose:
        print("\n  [8] Fixing dtypes")
    df_cl_clean = fix_dtypes_cl(df_cl_clean, verbose=verbose)

    # 9 — baseline CatBoost copy (before >25% NaN drop)
    df_cl_bcat = df_cl_clean.copy()

    # 10 — drop high-NaN columns
    if verbose:
        print("\n  [10] Dropping columns with >25% NaN")
    _exclude = ['Patient', 'Timepoint', 'pain_scale', 'date', 'measurement_timepoint']
    df_cl_clean = drop_high_nan_columns(
        df_cl_clean, threshold=0.25, exclude_cols=_exclude,
        check_per_timepoint=True, verbose=verbose,
    )

    # 11 — EDA/visualization copy (after >25% NaN drop)
    df_cl_vis = df_cl_clean.copy()

    if verbose:
        print(f"\n  df_cl_clean : {df_cl_clean.shape}")
        print(f"  df_cl_bcat  : {df_cl_bcat.shape}")
        print(f"  df_cl_vis   : {df_cl_vis.shape}")

    return df_cl_clean, df_cl_bcat, df_cl_vis



# ══════════════════════════════════════════════════════════════════════════════
# IMPUTATION  (produces df_*_imputed for PyOD; both immunological and clinical)
# ══════════════════════════════════════════════════════════════════════════════

def impute_miceforest(df, id_cols, name, num_datasets=5, iterations=10,
                      mean_match_candidates=5, random_state=42):
    """MICE imputation with miceforest. Handles both numeric-only and mixed-type datasets.

    Numeric columns: averaged across all datasets.
    Categorical columns (dtype='category'): mode across all datasets.

    Parameters
    ----------
    df                    : pd.DataFrame  contains id_cols + feature cols
    id_cols               : list[str]     columns to preserve unchanged
    name                  : str           label for print output
    num_datasets          : int           number of imputed datasets (default 5)
    iterations            : int           MICE iterations per dataset (default 10)
    mean_match_candidates : int
        5 for numeric-only datasets (immunological);
        0 to disable KD-tree mean matching for mixed categorical/numeric data (clinical)
    random_state          : int

    Returns
    -------
    df_imputed : pd.DataFrame  same column order as input, all NaN filled
    """
    import miceforest as mf

    print(f"\nMICE imputation (miceforest) — {name} dataset")

    feat_cols = [c for c in df.columns if c not in id_cols]

    def _clean_col(col):
        col = col.strip()
        col = re.sub(r'[^\w]', '_', col)
        col = re.sub(r'_+', '_', col)
        return col

    rename_map  = {c: _clean_col(c) for c in feat_cols}
    reverse_map = {v: k for k, v in rename_map.items()}

    X = (df[feat_cols]
         .reset_index(drop=True)
         .rename(columns=rename_map))

    # Remove unused categories to prevent LightGBM errors
    for c in X.select_dtypes('category').columns:
        X[c] = X[c].cat.remove_unused_categories()

    cat_renamed = [rename_map[c] for c in feat_cols if df[c].dtype.name == 'category']
    num_renamed = [rename_map[c] for c in feat_cols if df[c].dtype.name != 'category']

    # Record which columns have NaN before imputation (using renamed X)
    nan_before = X.isna().sum()
    cols_with_nan = nan_before[nan_before > 0].rename(index=reverse_map)

    kernel = mf.ImputationKernel(
        X,
        num_datasets=num_datasets,
        mean_match_candidates=mean_match_candidates,
        random_state=random_state,
    )
    kernel.mice(iterations)

    datasets  = [kernel.complete_data(i) for i in range(num_datasets)]
    X_imputed = datasets[0].copy()

    # Numeric: mean across all datasets
    if num_renamed:
        X_imputed[num_renamed] = sum(d[num_renamed] for d in datasets) / num_datasets

    # Categorical: mode across all datasets
    if cat_renamed:
        cat_stack = pd.concat(datasets, axis=0, keys=range(num_datasets))
        for c in cat_renamed:
            X_imputed[c] = (
                cat_stack[c]
                .groupby(level=1)
                .agg(lambda x: x.mode()[0])
            )

    X_imputed  = X_imputed.rename(columns=reverse_map).reindex(columns=feat_cols)
    df_imputed = pd.concat(
        [df[id_cols].reset_index(drop=True),
         X_imputed.reset_index(drop=True)],
        axis=1,
    )[df.columns]

    total_imputed = int(cols_with_nan.sum())
    print(f"  Imputed {total_imputed} values across {len(cols_with_nan)} columns:")
    for col, n in cols_with_nan.items():
        print(f"    {col}: {n}")
    print(f"  df_imputed shape : {df_imputed.shape}")
    print(f"  Remaining NaN    : {df_imputed.isna().sum().sum()}")
    print(f"\nTableReport of miceforest-imputed {name} dataset:")
    TableReport(df_imputed)

    return df_imputed


def impute_median(df, id_cols, name):
    """Column-wise median imputation for numeric features.

    Parameters
    ----------
    df      : pd.DataFrame
    id_cols : list[str]   columns to preserve unchanged
    name    : str

    Returns
    -------
    df_median : pd.DataFrame  same column order as input, numeric NaN filled
    """
    print(f"\nMedian imputation — {name} dataset")

    feat_cols = [c for c in df.columns if c not in id_cols]
    df_median = df.reset_index(drop=True).copy()
    num_feats = [c for c in feat_cols if pd.api.types.is_numeric_dtype(df_median[c])]

    nan_before = df_median[num_feats].isna().sum()
    cols_with_nan = nan_before[nan_before > 0]

    for col in num_feats:
        df_median[col] = df_median[col].fillna(df_median[col].median())

    total_imputed = int(cols_with_nan.sum())
    print(f"  Imputed {total_imputed} values across {len(cols_with_nan)} columns:")
    for col, n in cols_with_nan.items():
        print(f"    {col}: {n}")
    print(f"  Remaining NaN: {df_median[num_feats].isna().sum().sum()}")
    print(f"\nTableReport of median-imputed {name} dataset:")
    TableReport(df_median)

    return df_median

