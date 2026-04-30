# functions for preprocessing data
# imports
import pandas as pd
import numpy as np
import re
import datetime as dt
from collections import defaultdict
from skrub import TableReport

# Helper function: replace missing value markers
def replace_missing_markers(df, skip_cols=None, verbose=False):
    """Replace German missing-value strings with NaN in all object columns.

    Handles all capitalisation and punctuation variants of:
      - 'k.A.' (keine Angabe — no data entered)
      - 'n.D.' (nicht durchgeführt — not performed)

    Parameters
    ----------
    df        : pd.DataFrame  
    skip_cols : iterable of str -  optional columns to skip
    verbose   : bool            — set to true to print per-column replacement counts
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

    return df


def remove_nan_cols(df, id_cols=None, threshold=0.25, verbose=True):
    """Remove feature columns with more than `threshold` fraction of NaN values.

    Parameters
    ----------
    df        : pd.DataFrame
    id_cols   : list of str  columns to exclude from NaN check (default: Patient, Timepoint
    threshold : float        NaN fraction cutoff (default 0.25)
    verbose   : bool

    Returns
    -------
    pd.DataFrame — copy with high-NaN columns removed
    """
    if id_cols is None:
        id_cols = ['Patient', 'Timepoint']

    id_cols_present = [c for c in id_cols if c in df.columns]
    nan_frac        = df.drop(columns=id_cols_present).isna().mean()
    high_nan_cols   = nan_frac[nan_frac > threshold].index.tolist()

    df = df.drop(columns=high_nan_cols)

    if verbose:
        print(f"  Dropped {len(high_nan_cols)} columns with >{threshold:.0%} NaN: {high_nan_cols}")

    return df



def remove_for_modeling(df, verbose=True):
    """ Remove columns not needed for modeling. """
    to_drop = ['filter', 'kv', 'ma', 'fha', 'single_fraction', 'pain_points', 'date',
               'measurement_timepoint', 'Date']

    dropped = [c for c in to_drop if c in df.columns]
    df = df.drop(columns=dropped)

    if verbose:
        print(f"  Dropped {len(dropped)} columns: {dropped}")

    return df


# Immunological Dataset Preprocessing Functions
# ══════════════════════════════════════════════════════════════════════════════

# Predetermined columns to be excluded from all analysis:
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
    "Basophils.1",   # duplicate column
]

# Empty rows at the bottom of dataset
IM_EMPTY_ROW_INDICES = list(range(823, 829)) 


def drop_rename_cols_im(df_im_vis, verbose=True):
    """ Drop pre-determined columns and rename columns in immunological dataset. """

    cols_present = [c for c in IM_EXCLUDED_COLUMNS if c in df_im_vis.columns]
    df_im_vis = df_im_vis.drop(columns=cols_present)
    
    if verbose:
        print(f"  Dropped {len(cols_present)} columns")

    if 'Messdatum' in df_im_vis.columns:
        df_im_vis = df_im_vis.rename(columns={'Messdatum': 'Date'})
    
    return df_im_vis


def drop_empty_rows_im(df_im_vis, verbose=True):
    """ Drops found empty rows in dataset and rows with no immunulogical data. """

    # Verify found rows are actually empty 
    rows_present = [i for i in IM_EMPTY_ROW_INDICES if i in df_im_vis.index]
    feat_cols    = df_im_vis.columns[df_im_vis.columns.get_loc('PMN'):]

    if verbose:
        print("Verifying found empty rows:")

    for i in rows_present:
        filled  = df_im_vis.loc[i, feat_cols].dropna()
        patient = df_im_vis.loc[i, 'Patient']
        if len(filled) == 0:
            if verbose:
                print(f"  Row {i}  Patient={patient}  OK — all NaN")
        else:
            if verbose:
                print(f"  Row {i}  Patient={patient}  WARNING — {len(filled)} non-NaN values:")
                print(filled.to_string())
   
    to_drop_pts = df_im_vis.loc[rows_present, 'Patient'].dropna().unique().tolist()
    df_im_vis   = df_im_vis.drop(index=rows_present).reset_index(drop=True)
    
    if verbose:
        print(f"\nDropped {len(rows_present)}")

    # Rows where all features (PMN - last) are NaN
    if 'PMN' in df_im_vis.columns:
        empty_mask = df_im_vis[df_im_vis.columns[df_im_vis.columns.get_loc('PMN'):]].isna().all(axis=1)
    if empty_mask.any():
        if verbose:
            print(f"\nDropping {empty_mask.sum()} all-NaN feature rows:")
            print(df_im_vis.loc[empty_mask, ['Patient', 'Timepoint']].to_string())
        df_im_vis = df_im_vis[~empty_mask].reset_index(drop=True)

    # Patients with NaN Timepoints:
        nan_tp = df_im_vis['Timepoint'].isna() & df_im_vis['Patient'].notna()
        if nan_tp.any():
            if verbose:
                print(f"\nPatients with NaN Timepoint ({nan_tp.sum()} rows):")
                print(df_im_vis.loc[nan_tp, ['Patient', 'Timepoint']].to_string())
    
    if verbose:
        print(f"\nUnique patients remaining: {df_im_vis['Patient'].dropna().nunique()}")
    
    return df_im_vis



def clean_im(df_im, verbose=True):
    """Full cleaning pipeline for the raw immunological dataset.
    -----
    1. Drop pre-determined excluded columns + rename Messdatum to Date
    2. Replace German NaN markers (k.A. / n.D. variants)
    3. Drop known empty rows (bottom of Excel dataset and rows with no data)
    4. Fix dtypes: Date → datetime, Patient/Timepoint → Int64, features → float64

    Parameters
    ----------
    df_im   : pd.DataFrame    raw immunological data
    verbose : bool            set to False to silence all outprints

    Returns
    -------
    df_im_vis  : copy of cleaned dataset
    """

    df_im_vis = df_im.copy()
    
    # 1 — drop predetermined excluded columns + rename date column
    if verbose:
        print("  [1] Dropping Pre-Determined Columns:")
    df_im_vis = drop_rename_cols_im(df_im_vis, verbose=verbose)

    # 2 — replace German NaN markers
    if verbose:
        print("  [2] Replacing German NaN markers")
    df_im_vis = replace_missing_markers(df_im_vis, skip_cols=["Patient", "Timepoint"], verbose=verbose)

    # 3 — drop known empty rows and rows with no data
    if verbose:
        print("  [3] Dropping empty rows and no-data rows")
    df_im_vis = drop_empty_rows_im(df_im_vis, verbose=verbose)

    # 4 — fix dtypes
    if verbose:
        print("  [4] Fixing dtypes")
    df_im_vis = fix_dtypes_im(df_im_vis, verbose=verbose)

    if verbose:
        print(f"\n  Shape df_im_vis  : {df_im_vis.shape}")

    return df_im_vis


def fix_dtypes_im(df_im_vis, verbose=True):
    """ Convert immunological dataset columns to correct dtypes."""

    if 'Date' in df_im_vis.columns:
        df_im_vis['Date'] = pd.to_datetime(df_im_vis['Date'], errors='coerce')
    if 'Patient' in df_im_vis.columns:
        df_im_vis['Patient'] = pd.to_numeric(df_im_vis['Patient'], errors='coerce').astype('Int64')
    if 'Timepoint' in df_im_vis.columns:
        df_im_vis['Timepoint'] = pd.to_numeric(df_im_vis['Timepoint'], errors='coerce').astype('Int64')

    _id_cols   = ['Date', 'Patient', 'Timepoint']
    _feat_cols = [c for c in df_im_vis.columns if c not in _id_cols]
    df_im_vis[_feat_cols] = (
        df_im_vis[_feat_cols]
        .apply(lambda s: pd.to_numeric(s, errors='coerce'))
        .astype('float64')
    )
    
    if verbose:
        print(f"\nData types after cleaning:")
        print(df_im_vis.dtypes.value_counts())

    return df_im_vis


# ── Outlier removal (after PyOD + review with dataset-owner) ─────────────────────────────

# 6 PyOD flagged outliers removed + 2 additional patients with found abnormal measurement (also flagged)
IM_CONFIRMED_OUTLIERS = [
    (221, 2),
    (163, 1),
    (150, 1),
    (159, 2),
    (109, 5),
    (266, 4),
    (254, 1),   
    (229, 2)] 


def remove_outlier_observations(df, outliers=None, verbose=True):
    """Remove specific (patient, timepoint) observations confirmed as outliers.

    Parameters
    ----------
    df             : pd.DataFrame  
    outliers       : list of (patient_id, timepoint) tuples, or None
                     defaults to IM_CONFIRMED_OUTLIERS
    verbose        : bool

    Returns
    -------
    pd.DataFrame — copy with outlier rows removed and index reset
    """

    if outliers is None:
        outliers = IM_CONFIRMED_OUTLIERS

    mask = pd.Series(False, index=df.index)
    for patient, timepoint in outliers:
        mask |= (df['Patient' ] == patient) & (df['Timepoint'] == timepoint)

    result = df[~mask].reset_index(drop=True)

    if verbose:
        print(f"  Removed {mask.sum()} observations:")
        for patient, timepoint in outliers:
            found = ((df['Patient' ] == patient) & (df['Timepoint'] == timepoint)).sum()
            status = "removed" if found else "not found"
            print(f"    Patient {patient}  T{timepoint}  ({status})")
        print(f"  Shape before: {df.shape}  Shape after: {result.shape}")
        print(f"  Rows removed: {df.shape[0] - result.shape[0]}")

    return result



# Clinical Dataset Preprocessing Functions
# ══════════════════════════════════════════════════════════════════════════════

# Constants

# Patient-level columns: constant across timepoints, filled only in first row
CL_PATIENT_LEVEL_COLS = [
    'Patient', 'Unnamed: 2', 'Age at start', 'Gender', 'Weight [kg]', 'Height [cm]',
    'Overweight? BMI', 'Besserung nach Nachuntersuchung laut Arztbrief in %',
    'Comments questionnaire', 'Diagnosis', 'Target volume', 'single fraction',
    'kummulative dose (x) - if two targets were applied', 'FHA', 'kV', 'mA',
    'Filter', 'Response', 'further comments',
]

# Patients irradiated at multiple different body parts — excluded from analysis 
CL_MULTI_BODY_PATIENTS = [3, 45, 184, 149, 150, 162, 179, 156, 54, 47]

# German to English column rename map
CL_RENAME_MAP = {
    "Patient": "Patient", "Timepoint": "Timepoint",
    "Age at start": "age_at_start", "Gender": "gender",
    "Weight [kg]": "weight_kg", "Height [cm]": "height_cm",
    "Overweight? BMI": "overweight_bmi",
    "Erfassungszeitpunkt": "measurement_timepoint", "Datum": "date",
    "Beschwerden seit": "complaints_since", "vorherige Therapie": "previous_therapy",
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
    'gender', 'overweight', 'complaints_since', 'pain_points', 'diagnosis',
    'target_volume', 'target_volume_side',
    'response']

# Pain questionnaire ordinal columns 
CL_PAIN_QUESTIONNAIRE_COLS = [
    'pain_under_load', 'pain_night', 'pain_daytime', 'pain_at_rest', 'morning_stiffness']


# ── Clinical helper functions ─────────────────────────────────────────────────

def move_column_after(df, col_to_move, after_col):
    """Move col_to_move to the position immediately after after_col."""
    cols = df.columns.tolist()
    cols.insert(cols.index(after_col) + 1, cols.pop(cols.index(col_to_move)))
    return df[cols]


def extract_numeric(series):
    """Extract numeric value from ordinal questionnaire entries (scale 1–4).

    Handles:
      - k.A./kA/ka                                        → NaN
      - decimal-encoded pair     "2.3"                    → average(2, 3) = 2.5
      - comma-separated ratings  "1,2" or "3, 4"         → average
      - parenthetical notes stripped first:
            "3 (abends 4)"  → "3"  → 3      (note inside parens, not a rating)
            "3 (tag), 4 (nacht)" → "3, 4"  → 3.5  (two ratings with context)
      - leading number with trailing text  "4 re"         → 4
    """
    s = series.astype(str).str.strip()

    def parse_entry(val):
        if val in ('nan', '', 'None', 'k.A.', 'kA', 'ka', 'k.a.'):
            return np.nan

        # Strip parenthetical content so descriptive notes don't contribute numbers
        # e.g. "3 (abends 4)" → "3",  "3 (tag), 4 (nacht)" → "3, 4"
        val_clean = re.sub(r'\([^)]*\)', '', val).strip()

        # Comma-separated ratings: "1,2" or "3, 4" → average
        if re.match(r'^\d+(\s*,\s*\d+)+$', val_clean):
            return np.mean([float(x) for x in val_clean.split(',')])

        # All remaining numbers (handles "2.3" → [2,3] → 2.5, "4 re" → [4] → 4)
        all_nums = re.findall(r'\b(\d+)\b', val_clean)
        if len(all_nums) > 1:
            return np.mean([float(x) for x in all_nums])
        if len(all_nums) == 1:
            return float(all_nums[0])
        return np.nan

    return s.apply(parse_entry)


def extract_continuous(series):
    """Extract numeric value from continuous scale entries (pain_scale from 0–10).

    Priority order:
      1. MW=  (Mittelwert, pre-calculated midpoint)     -> use directly
      2. semicolon-separated  ('1,2; 8,6')              -> take first value
      3. bilateral li/re pair ('5 (li), 8 (re)')        -> average both sides
      4. range + parenthesised value ('4-7 (6)')        -> use parenthesised value
      5. pure range ('5,5-7,5', '3-9')                  -> take midpoint
      6. leading number with trailing text              -> use leading number
    """
    s = series.astype(str).str.strip()

    def parse_entry(val):
        if val in ('nan', '', 'None', '?', 'k.A.', 'kA', 'ka', 'k.a.'):
            return np.nan

        # Correcting confirmed typo (dash used instead of a decimal point)
        if val == '3-9':
            return 3.9

        # 1 — MW= (pre-calculated midpoint already in the entry)
        mw = re.search(r'MW\s*=\s*(\d+[.,]\d+|\d+)', val, re.IGNORECASE)
        if mw:
            return float(mw.group(1).replace(',', '.'))

        # 2 — semicolon-separated: take the first value  ('1,2; 8,6' → 1.2)
        if ';' in val:
            first = val.split(';')[0].strip()
            m = re.match(r'^(\d+[.,]?\d*)', first)
            if m:
                return float(m.group(1).replace(',', '.'))

        # 3 — bilateral li/re: average both sides  ('5 (li), 8 (re)' → 6.5)
        #     \b after li/re ensures 'linke' / 'rechte' are not accidentally matched
        lr_nums = re.findall(r'(\d+[.,]?\d*)\s*\(?(?:li|re)\b', val, re.IGNORECASE)
        if lr_nums:
            nums = [float(n.replace(',', '.')) for n in lr_nums]
            return sum(nums) / len(nums)

        # 4 — range with parenthesised representative: '4-7 (6)' → 6
        m = re.match(r'^\d+[.,]?\d*\s*[-–]\s*\d+[.,]?\d*\s*\((\d+[.,]?\d*)\)', val)
        if m:
            return float(m.group(1).replace(',', '.'))

        # 5 — pure range: '5,5-7,5', '3-9' → midpoint
        m = re.match(r'^(\d+[.,]?\d*)\s*[-–]\s*(\d+[.,]?\d*)\s*$', val)
        if m:
            return (float(m.group(1).replace(',', '.')) +
                    float(m.group(2).replace(',', '.'))) / 2

        # 6 — leading number with any trailing text
        #     covers: '6 (long description...)', '8 (m. Schmerzmittel 4)',
        #             '7,3-dauernd bei Belastung, 10 aus der Ruhe', '1,6', etc.
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
        if 'both sides' in lower or re.search(r'[LR]\s*[+&]\s*[LR]|[LR]\s*,\s*[LR](?!\w)|\b[LR]{2}\b', s_check):
            return 'B'
        if 'links' in lower or 'left' in lower:
            return 'L'
        if 'recht' in lower or 'right' in lower:
            return 'R'
        m = re.search(r'\b([LRlr])\s*$', s_check)
        if m:
            return m.group(1).upper()
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
    """Standardize diagnosis column by mapping German/English variants to English names:

    Combined diagnoses are mapped as 'Name1, Name2'.
    """
    diagnosis_map = [
        ('Achillodynia',            ['achillodynie', 'achilliodynie', 'achyllodynie', 'achillodynia', 'tendinitis']),
        ('Heel/Plantar',            ['calcaneodynie', 'calcaneodynia', 'heel calcaneodynia', 'heel spur', 'fersensporn', 'plantarfasz', 'plantar']),
        ('Foot/Ankle',              ['sprunggelenk', 'ankle', 'arthrosis upper ankle', 'mittelfuß', 'midfoot', 'forefoot', 'arthrosis right foot', 'zehenarthros', 'zehengrundgelenk']),
        ('Hand/Wrist',              ['rhizarthros', 'rizarthros', 'daumensattelgelenk', 'thumb cmc', 'carpometacarpal',
                                     'fingergelenk', 'fingerpolyarth', 'finger joint arthritis', 'finger arthritis', "dupuytren's disease",
                                     'wrist arthritis', 'wrist arthrosis', 'handgelenk', 'painful  tendon sheath right (wrist)']),
        ('Elbow Syndrome',          ['ellbow', 'elbow', 'ellenbogen', 'epicondylitis', 'epiconilitis']),
        ('Shoulder Syndrome',       ['shouldersyndrom', 'shoulder syndrom', 'schulter', 'omarthrosis']),
        ('Gonarthrosis',            ['gonarthros', 'kniegelenk']),
        ('Trochanter Tendopathy',   ['trochanter']),
        ('Rheumatoid Arthritis',    ['rheumatoid', 'rheumatoide']),
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
    """Standardize pain_points: map German body parts to English.

    Pure number entries become NaN. Returns 'BodyPart, BodyPart' format (side stripped).
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
        ('Under Arm',       ['unterarm']),
        ('Elbow',           ['ellenbogen', 'ellbogen', 'ellenbogengelenk']),
        ('Arm',             [r'\barm\b']),
        ('Wrist',           ['handgelenk', 'hangelenk']),
        ('Thumb',           ['daumen', 'daumensattelgelenk']),
        ('Hand',            [r'\bhand\b', 'hände']),
        ('Finger',          ['finger']),
    ]

    def find_body_part(seg):
        s = seg.lower().strip()
        for name, keywords in body_part_keywords:
            if any(re.search(kw, s) for kw in keywords):
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
        for seg in segments:
            seg = seg.strip()
            if not seg:
                continue
            body = find_body_part(seg)
            if body and body not in results:
                results.append(body)
        if not results:
            return s.strip()
        return ', '.join(results)

    return series.apply(parse_entry)



def parse_cumulative_dose(val):
    """Parse total cumulative dose from mixed format strings.

    Handles: standalone values, double values "L: 3; R: 6" (sum both sides), "3(6)" (takes the parenthesized total),
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


def forward_fill_clinical(df_cl, verbose=True):
    """Forward-fill clinical patient columns and extract timepoint to make a Timepoint column.
    Makes it so clinical and immunulogical dataset has the same format with one timepoint per row.

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


def exclude_predetermined(df_cl_clean, multi_body_patients=None, verbose=True):
    """Exclude predetermined patients and columns.

    Parameters
    ----------
    df_cl_clean         : pd.DataFrame
    multi_body_patients : list of int, optional — defaults to CL_MULTI_BODY_PATIENTS
    verbose             : bool

    Returns
    -------
    pd.DataFrame — copy with excluded patients and predetermined columns removed
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
        print(f"\n  Verifying multi-body-part patients:")
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

    # 3 — Drop predetermined questionnaire column range
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
                print(f"  Warning: start column '{start_col}' not found — no EORTC columns dropped")
        elif end_col is None:
            if verbose:
                print(f"  Warning: end column not found — no EORTC columns dropped")
        else:
            start_idx      = col_list.index(start_col)
            end_idx        = col_list.index(end_col)
            q_cols_to_drop = col_list[start_idx : end_idx + 1]
            df = df.drop(columns=q_cols_to_drop)
            if verbose:
                print(f"\n  Dropped {len(q_cols_to_drop)} questionnaire columns "
                      f"('{start_col}' to '{end_col}')")
    except Exception as e:
        if verbose:
            print(f"  Warning: Could not drop columns: {e}")

    # 4 — Drop empty columns 
    admin_cols = ['Unnamed: 0', 'Unnamed: 2', 'further comments', 'Comments questionnaire']
    dropped = [c for c in admin_cols if c in df.columns]
    df = df.drop(columns=dropped)
    if verbose:
        print(f"\n  Dropped {len(dropped)} empty columns: {dropped}")

    if verbose:
        print(f"\n  After exclusions: {df['Patient'].nunique()} patients, {len(df)} rows")

    return df



def rename_columns_cl(df_cl_clean, rename_map=None, verbose=True):
    """Rename clinical columns from German to English using the CL_RENAME_MAP.

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


def drop_rows_cl(df_cl_clean, verbose=True):
    """Drop empty and invalid rows from the clinical dataset.

    Steps
    -----
    1. Drop rows with no measurement date (blank rows in raw dataset)
    2. Drop rows where all questionnaire columns (complaints_since to
       improvement_percent) are NaN (empty or fully-marker-filled visits)

    Parameters
    ----------
    df_cl_clean : pd.DataFrame
    verbose     : bool

    Returns
    -------
    pd.DataFrame — copy with invalid rows removed
    """
    df = df_cl_clean.copy()

    drop_mask = pd.Series(False, index=df.index)

    # 1 — Rows with no date
    if 'date' in df.columns:
        drop_mask |= df['date'].isna()

    # 2 — Rows where the entire questionnaire section is NaN
    if {'complaints_since', 'improvement_percent'}.issubset(df.columns):
        drop_mask |= df.loc[:, 'complaints_since':'improvement_percent'].isna().all(axis=1)

    if verbose:
        if drop_mask.any():
            print(f"  Dropping {drop_mask.sum()} empty rows:")
        else:
            print(f"  No empty rows to drop")

    df = df.loc[~drop_mask].copy()

    if verbose:
        print(f"\nAfter dropping empty rows: {df['Patient'].nunique()} patients, {len(df)} rows")

    return df


def manual_corrections_cl(df_cl_clean, verbose=True):
    """Apply known manual data corrections to the clinical dataset.

    Corrections applied
    -------------------
    - Patient 248 T2 : pain_daytime '22' → '2'  (confirmed typo)
    - Patient 219    : removed (used a different questionnaire)
    - Patient 89     : assign correct timepoints by date (T2 = 27.03.2019,
                       T5 = 05.07.2019); drop unmatched row (10.05.2019)
    - Patient 21 - 

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

    # Patient 89: assign correct timepoints by date, drop the unmatched row
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

    return df


def parse_transform_cl(df_cl_clean, verbose=True):
    """Parse and transform all clinical columns to structured values.

    Steps
    -----
    1.  diagnosis       -> standardized English names
    2.  target_volume   -> body part + side combined string
    3.  pain_points     -> standardized English body parts + side
    4.  cumulative_dose -> numeric (Gy)
    5.  gender          -> 'w' -> 'f'
    6.  overweight_bmi  -> overweight (ja/nein) + bmi (float)
    7.  previous_therapy-> binary columns previous_therapy_1 … _7
    8.  ordinal columns -> extract_numeric values
    9.  pain_scale      -> extract_continuous values

    Requires columns already renamed.

    Parameters
    ----------
    df_cl_clean : pd.DataFrame
    verbose     : bool         — print before/after value distributions for each column

    Returns
    -------
    pd.DataFrame — copy with all columns parsed and transformed
    """
    df = df_cl_clean.copy()

    # 1 — diagnosis
    _pt = df.drop_duplicates(subset=['Patient'])
    if verbose:
        print('Parsing and Transforming Clinical columns: \nUnique Value Counts before and after')
        print(f"\n--- diagnosis (BEFORE, {_pt['diagnosis'].nunique()} unique values) ---")
        print(_pt['diagnosis'].value_counts(dropna=False).to_string())
    df['diagnosis'] = standardize_diagnosis(df['diagnosis'])
    if verbose:
        _pt_after = df.drop_duplicates(subset=['Patient'])
        print(f"\n--- diagnosis (AFTER, {_pt_after['diagnosis'].nunique()} unique values) ---")
        print(_pt_after['diagnosis'].value_counts().to_dict())
      

    # 2 — target_volume: standardize body part; keep side as separate column
    if verbose:
        _tv_pt = df.drop_duplicates(subset=['Patient'])
        _tv_n_unique = _tv_pt['target_volume'].nunique()
        print(f"\n--- target_volume (BEFORE, {_tv_n_unique} unique values) ---")
        print(_tv_pt['target_volume'].value_counts(dropna=False).to_string())
    df['target_volume'], df['target_volume_side'] = standardize_target_volume(df['target_volume'])
    df = move_column_after(df, 'target_volume_side', 'target_volume')

    if verbose and 'target_volume_side' in df.columns and 'Patient' in df.columns:
        pt_side = df.drop_duplicates(subset=['Patient']).set_index('Patient')['target_volume_side']
        n_single = pt_side.isin(['L', 'R']).sum()
        n_both   = (pt_side == 'B').sum()
        print(f"\n  Patients treated at one side (L / R): {n_single}")
        print(f"  Patients treated at two sides (L&R):  {n_both}")
        counts = pt_side.value_counts(dropna=False)
        l = counts.get('L', 0); r = counts.get('R', 0); b = counts.get('B', 0)
        print(f"\n--- target_volume_side (AFTER) ---")
        print(f"  L: {l}   R: {r}   B: {b}")

    if verbose:
        _tv_after = df.drop_duplicates(subset=['Patient'])['target_volume']
        print(f"\n--- target_volume (AFTER, {_tv_after.nunique()} unique values) ---")
        print(_tv_after.value_counts().to_dict())

    # 3 — pain_points
    if verbose:
        print("\n--- pain_points (BEFORE) ---") 
        print(df['pain_points'].value_counts(dropna=False).head(20).to_string())
    df['pain_points'] = standardize_pain_points(df['pain_points'])
    if verbose:
        print("\n--- pain_points (AFTER) ---")
        print(df['pain_points'].value_counts().head(20).to_dict())

    # 4 — cumulative_dose
    if 'cumulative_dose' in df.columns:
        if verbose:
            print(f"\n--- cumulative_dose (BEFORE, {df['cumulative_dose'].nunique()} unique values) ---")
            print(df['cumulative_dose'].value_counts(dropna=False).to_string())
        df['cumulative_dose'] = pd.to_numeric(
            df['cumulative_dose'].apply(parse_cumulative_dose), errors='coerce'
        )
        if verbose:
            print(f"\n--- cumulative_dose (AFTER, {df['cumulative_dose'].nunique()} unique values) ---")
            print(sorted(df['cumulative_dose'].dropna().unique()))

    # 5 — gender: 'w' -> 'f'
    if 'gender' in df.columns:
        if verbose:
            print(f"\n--- gender (BEFORE, {df['gender'].nunique()} unique values) ---")
            print(df['gender'].value_counts(dropna=False).to_string())
        df['gender'] = df['gender'].replace('w', 'f')
        if verbose:
            print(f"\n--- gender (AFTER, {df['gender'].nunique()} unique values) ---")
            print(df['gender'].value_counts().to_dict())

    # 6 — overweight_bmi -> overweight + bmi
    if 'overweight_bmi' in df.columns:
        if verbose:
            _owbmi_n_unique = df['overweight_bmi'].nunique()
            print(f"\n--- overweight_bmi (BEFORE, {_owbmi_n_unique} unique values) ---")
            print(df['overweight_bmi'].value_counts(dropna=False).to_string())
        df = split_bmi_column(df)
        if verbose:
            _pt_ow = df.drop_duplicates(subset=['Patient'])
            print(f"\n--- overweight / bmi (AFTER, {_pt_ow['overweight'].nunique()} unique overweight values, "
                  f"{_pt_ow['bmi'].nunique()} unique bmi values) ---")
            print(f"  overweight (unique patients): {_pt_ow['overweight'].value_counts(dropna=False).to_dict()}")
            bmi_valid = _pt_ow['bmi'].dropna()
            if len(bmi_valid) > 0:
                print(f"  bmi (unique patients): range {bmi_valid.min():.1f}–{bmi_valid.max():.1f}, "
                      f"{_pt_ow['bmi'].isna().sum()} missing")

    # 7 — previous_therapy -> binary indicator columns
    if 'previous_therapy' in df.columns:
        if verbose:
            print("\n--- previous_therapy (BEFORE) ---")
            print(df['previous_therapy'].value_counts(dropna=False).to_string())
        df = encode_therapy_columns(df)
        if verbose:
            therapy_cols = [f'previous_therapy_{i}' for i in range(1, 8)
                            if f'previous_therapy_{i}' in df.columns]
            print("\n--- previous_therapy (AFTER: binary columns) ---")
            print(df[therapy_cols].sum().to_dict())

    # 9 — Ordinal questionnaire columns -> extract numeric
    ordinal_cols = ['pain_under_load', 'pain_at_rest', 'pain_daytime',
                    'pain_night', 'morning_stiffness']
    for col in ordinal_cols:
        if col not in df.columns:
            continue
        if verbose:
            uniq_before = df[col].dropna().unique()
            print(f"\n--- {col} (BEFORE) ---")
            for v in sorted(uniq_before, key=lambda x: str(x)):
                print(f"  {repr(v)}")
        df[col] = extract_numeric(df[col])
        if verbose:
            print(f"--- {col} (AFTER) ---")
            print(f"  {sorted(df[col].dropna().unique())}")

    # 10 — pain_scale (continuous): German decimal comma, ranges -> midpoint
    if 'pain_scale' in df.columns:
        if verbose:
            print("\n--- pain_scale (BEFORE) ---")
            uniq_ps = df['pain_scale'].dropna().unique()
            for v in sorted(uniq_ps, key=lambda x: str(x)):
                print(f"    {repr(v)}")
        df['pain_scale'] = extract_continuous(df['pain_scale'])
        if verbose:
            print(f"\n--- pain_scale (AFTER) ---")
            print(f"  range: {df['pain_scale'].min():.1f} – {df['pain_scale'].max():.1f}")

    return df


def fix_dtypes_cl(df_cl_clean, verbose=True):
    """ Convert clinical dataset columns to correct dtypes. """
    df = df_cl_clean.copy()

    df['Patient']   = pd.to_numeric(df['Patient'],   errors='coerce')
    df['Timepoint'] = pd.to_numeric(df['Timepoint'], errors='coerce')

    n_before  = len(df)
    _bad_rows = df[df[['Patient', 'Timepoint']].isna().any(axis=1)]
    if len(_bad_rows) > 0 and verbose:
        print(f"  Dropping rows with unknown Patient or Timepoint:")
        print(_bad_rows[['Patient', 'Timepoint']].to_string())
    df = df.dropna(subset=['Patient', 'Timepoint']).copy()

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
                         {'Patient', 'Timepoint', 'measurement_timepoint', 'date',
                          'complaints_since'})
    cols_to_float = [c for c in df.columns if c not in exclude_for_float]
    df[cols_to_float] = (
        df[cols_to_float]
        .apply(lambda s: pd.to_numeric(s, errors='coerce'))
        .astype('float64')
    )

    if verbose:
        print("\n--- Dtype summary (clinical dataset) ---")
        print(df.dtypes.value_counts())
        print(f"Shape: {df.shape}, Patients: {df['Patient'].nunique()}")

    return df



def clean_cl(df_cl, verbose=True):
    """Full cleaning pipeline for the raw clinical dataset.

    Steps
    -----
    1.  Forward-fill patient-level columns + extract Timepoint  (forward_fill_clinical)
    2.  Exclude predetermined patients + drop predetermined columns  (exclude_predetermined)
    3.  Rename columns German to English  (rename_columns_cl)
    4.  Apply manual data corrections  (manual_corrections_cl)
    5.  Parse/transform columns  (parse_transform_cl)
    6.  Replace German NaN markers  (replace_missing_markers)
    7.  Drop empty rows, no date or empty questionnaire  (drop_rows_cl)
    8.  Fix dtypes  (fix_dtypes_cl)
    9.  Return df_cl_vis with all columns

    Parameters
    ----------
    df_cl   : pd.DataFrame   raw clinical data as loaded from Excel
    verbose : bool.      -   set to False to supress all outprints 

    Returns
    -------
    df_cl_vis : full cleaned dataset
    """

    df_cl_vis = df_cl.copy()   # keep the raw input untouched!1

    if verbose:
        print("\n  [1] Forward-filling patient-level columns + extracting Timepoint column")
    df_cl_vis = forward_fill_clinical(df_cl_vis, verbose=verbose)

    if verbose:
        print("\n  [2] Excluding pre-determined patients and columns")
    df_cl_vis = exclude_predetermined(df_cl_vis, verbose=verbose)

    if verbose:
        print("\n  [3] Renaming columns from German to English")
    df_cl_vis = rename_columns_cl(df_cl_vis, verbose=verbose)

    if verbose:
        print("\n  [4] Applying manual corrections")
    df_cl_vis = manual_corrections_cl(df_cl_vis, verbose=verbose)

    if verbose:
        print("\n  [5] Parsing and transforming columns")
    df_cl_vis = parse_transform_cl(df_cl_vis, verbose=verbose)

    if verbose:
        print("\n  [6] Replacing NaN markers")
    df_cl_vis = replace_missing_markers(df_cl_vis, skip_cols=["Patient", "Timepoint"], verbose=verbose)

    if verbose:
        print("\n  [7] Dropping empty rows")
    df_cl_vis = drop_rows_cl(df_cl_vis, verbose=verbose)

    if verbose:
        print("\n  [8] Correcting dtypes")
    df_cl_vis = fix_dtypes_cl(df_cl_vis, verbose=verbose)

    if verbose:
        print(f"\n Shape df_cl_vis : {df_cl_vis.shape}")

    return df_cl_vis



# ══════════════════════════════════════════════════════════════════════════════
# IMPUTATION 
# ══════════════════════════════════════════════════════════════════════════════


def impute_iterative(df, ex_cols=None, iterations=20, random_state=42, verbose=False):
    """Iterative imputation using sklearn IterativeImputer.
    Numeric columns: IterativeImputer (BayesianRidge).
    Categorical/object columns: SimpleImputer (Majority vote).

    Parameters
    ----------
    df         : pd.DataFrame
    ex_cols    : list of str or None  — columns to exclude (e.g. id columns)
    iterations : int                  — max iterations (default 10)
    random_state : int
    verbose    : bool

    Returns
    -------
    df_imputed : pd.DataFrame  — same shape as df
    imputer    : fitted ColumnTransformer  — use imputer.transform() on test set
    """

    from sklearn.experimental import enable_iterative_imputer  
    from sklearn.impute import IterativeImputer, SimpleImputer
    from sklearn.compose import ColumnTransformer

    ex_cols   = list(ex_cols) if ex_cols is not None else []
    feat_cols = [c for c in df.columns if c not in ex_cols]
    orig_index = df.index

    X = df[feat_cols].reset_index(drop=True)

    num_cols = X.select_dtypes('number').columns.tolist()
    cat_cols = X.select_dtypes(['category', 'object']).columns.tolist()

    # Record NaN before imputation
    nan_before    = X.isna().sum()
    cols_with_nan = nan_before[nan_before > 0]

    transformers = []
    if num_cols:
        transformers.append((
            'num',
            IterativeImputer(max_iter=iterations, random_state=random_state),
            num_cols
        ))
    if cat_cols:
        transformers.append((
            'cat',
            SimpleImputer(strategy='most_frequent'),
            cat_cols
        ))

    imputer = ColumnTransformer(transformers=transformers, remainder='passthrough')
    arr = imputer.fit_transform(X)

    out_cols = num_cols + cat_cols
    X_imputed = pd.DataFrame(arr, columns=out_cols).reindex(columns=feat_cols)

    # Restore dtypes
    for c in num_cols:
        X_imputed[c] = pd.to_numeric(X_imputed[c])
    for c in cat_cols:
        X_imputed[c] = X_imputed[c].astype(df[c].dtype)

    X_imputed.index = orig_index

    if ex_cols:
        df_imputed = pd.concat(
            [df[ex_cols].reset_index(drop=True),
             X_imputed.reset_index(drop=True)],
            axis=1,
        )[df.columns]
    else:
        df_imputed = X_imputed

    if verbose:
        total_imputed = int(cols_with_nan.sum())
        print(f"  Imputed {total_imputed} values across {len(cols_with_nan)} columns:")
        for col, n in cols_with_nan.items():
            print(f"    {col}: {n}")
        print(f"  Shape: {df_imputed.shape}  -  Remaining NaN: {df_imputed.isna().sum().sum()}")

    return df_imputed, imputer




def impute_median(df, ex_cols, verbose=True):
    """Column-wise median imputation for numeric features.
    """
    if verbose:
        print(f"\nMedian imputation:")

    feat_cols = [c for c in df.columns if c not in ex_cols]
    df_median = df.reset_index(drop=True).copy()
    num_feats = [c for c in feat_cols if pd.api.types.is_numeric_dtype(df_median[c])]

    nan_before = df_median[num_feats].isna().sum()
    cols_with_nan = nan_before[nan_before > 0]

    for col in num_feats:
        df_median[col] = df_median[col].fillna(df_median[col].median())

    total_imputed = int(cols_with_nan.sum())
    if verbose:
        print(f"  Imputed {total_imputed} values across {len(cols_with_nan)} columns:")
        for col, n in cols_with_nan.items():
            print(f"    {col}: {n}")
            print(f"  Remaining NaN: {df_median[num_feats].isna().sum().sum()}")

    return df_median


