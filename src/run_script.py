# File for running and testing code

#%% imports
import pandas as pd
import numpy as np
from pathlib import Path
from skrub import TableReport
import scikit_na as na
import re
import lightgbm
import miceforest as mf
import hoggorm as ho
import seaborn as sns
import matplotlib.pyplot as plt
import prince as ps
import pyod as pyod
from skrub import Cleaner
from missing_methods import pca as mm_pca, rv2 as mm_rv2
from missing_methods.sk import StandardScaler as MM_StandardScaler


#%%############# Shared utility functions ###########################

def replace_missing_markers(df, skip_cols=None, verbose=False):
    """Replace German missing-value strings with NaN in all object columns.

    Handles all capitalisation and punctuation variants of:
      - 'k.A.' (keine, keine Angabe — none / no data entered)
      - 'n.D.' (nicht durchgeführt / nicht definiert — not performed)

    Parameters
    ----------
    df : pd.DataFrame  (modified in-place)
    skip_cols : iterable of str, optional — columns to leave untouched (e.g. Patient, Timepoint)
    verbose : bool — if True, print per-column replacement counts
    """
    pattern   = r'^([kK]\.?[aA]\.?|[nN]\.?[dD]\.?)$'
    skip_cols = set(skip_cols or [])

    for col in df.columns:
        if col in skip_cols:
            continue
        if df[col].dtype != object:
            continue
        str_col = df[col].astype(str).str.strip()
        mask = str_col.str.match(pattern, na=False) | (str_col == "")
        if mask.sum() > 0:
            if verbose:
                print(f"  {col}: replaced {mask.sum()} null markers")
            df.loc[mask, col] = np.nan


#%%############# Loading raw datasets ###########################


# reading excel file with raw  (UPDATED)
data_dir = Path(__file__).resolve().parents[1] / "data"
data = data_dir / "LDRT_raw.xlsx"

# immunological data/blood samples, columns starts at row 5
df_im = pd.read_excel(
    data,
    sheet_name="IPT ",
    header=4,
    engine="openpyxl"
)


# Clinical data and questionarries, columns starts at row 2
df_cl = pd.read_excel(
    data,
    sheet_name="Patient data & Pain",
    header=1,
    engine="openpyxl"
)


#%%################  RAW IMMUNOLOGICAL DATASET ###########################


print("TableReport of raw immunological dataset:")
TableReport(df_im, max_plot_columns=138)

# 46 columns to exclude from further analysis, as mention in dataset.
# Around 6-7 outliers and missing values for almost each variable, 
# also 6 missing values for patient IDs, maybe it is the same patients? 
# Not all patients have been measured at all timepoints 1-5. Which ones are that?

# na analysis of immunological dataset
print("Na analysis of immunological dataset:")
na.altair.plot_heatmap(df_im)


# Raw immunological dataset statistics
print("Raw immunological dataset statistics:")
print('shape of dataset:', df_im.shape)
print("unique patients:", df_im["Patient"].nunique())
print("timepoints:", df_im["Timepoint"].nunique())
print("\n")
print("measurements per timepoint:")
print(df_im["Timepoint"].value_counts().sort_index())
print("\n")

# Patients with measurements from t1 through t5
patients_t1 = set(df_im[df_im["Timepoint"] == 1]["Patient"])
patients_t2 = set(df_im[df_im["Timepoint"] == 2]["Patient"])
patients_t3 = set(df_im[df_im["Timepoint"] == 3]["Patient"])
patients_t4 = set(df_im[df_im["Timepoint"] == 4]["Patient"])
patients_t5 = set(df_im[df_im["Timepoint"] == 5]["Patient"])    


print("Patients with measurements at only timepoint 1:", len(patients_t1 - (patients_t2 | patients_t3 | patients_t4 | patients_t5)))
print("Patients with measurements at timepoint 1 and 2:", len(patients_t1 & patients_t2))
print("Patients with measurements at timepoint 1,2 and 3:", len(patients_t1 & patients_t2 & patients_t3))
print("Patients with measurements at timepoint 1,2,3 and 4:", len(patients_t1 & patients_t2 & patients_t3 & patients_t4))
print("Patients with measurements at timepoint 1,2,3,4 and 5:", len(patients_t1 & patients_t2 & patients_t3 & patients_t4 & patients_t5))   


# Bar plot of number of unique patients per timepoint
plt.figure(figsize=(8, 5))
patient_counts = df_im.groupby("Timepoint")["Patient"].nunique()
sns.barplot(
    x=patient_counts.index,
    y=patient_counts.values,
    color="teal"
)
plt.title("Number of Patients per Timepoint (Immunological Dataset)")
plt.xlabel("Timepoint")
plt.ylabel("Number of unique patients")
plt.show()  


#%%############# Cleaning immunulogical dataset ####################################


# Removing columns that can be exlcuded (marked yellow in dataset): 43 columns + Id Subset

dropped_columns = [
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

df_im = df_im.drop(columns=dropped_columns)
df_im = df_im.rename(columns={'Messdatum': 'Date'})

# Removing empty rows in the bottom of excel file (row 829 to 834 in excel file and row 84
df_im = df_im.drop(index=range(823, 829))
df_im = df_im.drop(index=78)

# Replace all German missing-value markers (k.A. / n.D. variants) with NaN.
# Done before df_im_vis is copied so the baseline dataset is also clean.
replace_missing_markers(df_im, skip_cols=["Patient", "Timepoint"])

# change datatypes to correct type
df_im["Date"] = pd.to_datetime(df_im["Date"], errors="coerce")
df_im["Patient"] = pd.to_numeric(df_im["Patient"], errors="coerce").astype("Int64")
df_im["Timepoint"] = pd.to_numeric(df_im["Timepoint"], errors="coerce").astype("Int64")

exclude_cols = ["Date", "Patient", "Timepoint"]

# For numeric columns, convert explicitly
feature_cols = df_im.columns.difference(exclude_cols)
for col in feature_cols:
    df_im[col] = pd.to_numeric(df_im[col], errors="coerce")

TableReport(df_im, max_plot_columns=180)

# Copy for catboost baseline-modeling:
df_im_bcat = df_im.copy()

# Per-timepoint NaN check (T1/T2/T3) — run BEFORE the overall drop.
# Purpose: verify that the >25% threshold is consistent across timepoints.
# If T1/T2/T3 lists match the overall list → the threshold is unambiguous and
# dropping is safe regardless of which timepoint you analyse.
# If a column appears in the overall list but NOT in T1 → it is sparse only at
# later timepoints (T4/T5 dropout), meaning T1 baseline data is actually fine.
# That discrepancy would be worth flagging to the expert before proceeding.
_id_drop_cols = ['Patient', 'Timepoint', 'Date']
for _tp in [1, 2, 3]:
    _df_tp = df_im[df_im['Timepoint'] == _tp]
    _na_tp = _df_tp.drop(columns=[c for c in _id_drop_cols if c in _df_tp.columns]).isna().mean()
    _high_nan_tp = sorted(_na_tp[_na_tp > 0.25].index.tolist())
    print(f"T{_tp} columns >25% NaN ({len(_high_nan_tp)}): {_high_nan_tp}")

# Overall >25% NaN across all timepoints — these are the columns actually dropped
na_frac = df_im.isna().mean()
cols_to_drop = na_frac[na_frac > 0.25].index.tolist()
print(f"\nOverall columns >25% NaN ({len(cols_to_drop)}): {sorted(cols_to_drop)}")

df_im = df_im.drop(columns=cols_to_drop).copy()
print('Dropped columns:', cols_to_drop)

""" 
Dropped columns: ['TC_CD25hi', 'B_CD25hi', 'Eos_HLADR+', 'Mo2_HLADRhi', 'TC_HLADRhi', 
'NK_HLADRhi', 'Eos_CD69+', 'Bas_CD69+', 'Mo_CD69+', 'B_CD69+', 'DC_CD69+', 
'TH naive_PD1+', 'TH eff_PD1+', 'TC naive_PD1+'
"""

# Copy for EDA / visualization (after >25% NaN drop)
df_im_vis = df_im.copy()
# Note: df_im_mod (outlier-removed modeling copy) is created after PyOD outlier detection



#%%########## Pearson correlation — immunological dataset (with missing values)
# pandas .corr() computes pairwise Pearson r, dropping NaN per pair independently.
# Uses df_im_vis (not imputed)

df_pearson_feat = df_im_vis.drop(columns=[c for c in exclude_cols if c in df_im_vis.columns])
pearson_matrix = df_pearson_feat.corr(method='pearson')

# Top correlated pairs by |r| — upper triangle only (no diagonal, no duplicates)
upper_tri = pearson_matrix.where(np.triu(np.ones(pearson_matrix.shape), k=1).astype(bool))
pearson_pairs = (
    upper_tri.stack()
    .reset_index()
    .rename(columns={'level_0': 'Feature_1', 'level_1': 'Feature_2', 0: 'Pearson_r'})
    .assign(Abs_r=lambda x: x['Pearson_r'].abs())
    .sort_values('Abs_r', ascending=False)
    .drop(columns='Abs_r')
    .reset_index(drop=True)
)

print("\nTop 40 Most Correlated Feature Pairs (Pearson r):")
print("=" * 80)
print(pearson_pairs.head(40).to_string(index=False))

print("\nTop 40 Most Negatively Correlated Feature Pairs (Pearson r):")
print("=" * 80)
print(upper_tri.stack()
      .reset_index()
      .rename(columns={'level_0': 'Feature_1', 'level_1': 'Feature_2', 0: 'Pearson_r'})
      .sort_values('Pearson_r', ascending=True)
      .head(40)
      .reset_index(drop=True)
      .to_string(index=False))

# Full heatmap — lower triangle only
mask_full = np.triu(np.ones_like(pearson_matrix, dtype=bool))
fig, ax = plt.subplots(figsize=(18, 16))
sns.heatmap(
    pearson_matrix,
    mask=mask_full,
    cmap='mako',
    center=0,
    vmin=-1,
    vmax=1,
    square=True,
    linewidths=0.2,
    cbar_kws={'label': 'Pearson r', 'shrink': 0.8},
    ax=ax,
)
ax.set_title('Pearson Correlation — Immunological Dataset (with nan)',
             fontsize=14, fontweight='bold')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=7)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=7)
plt.tight_layout()
plt.show()

# Focused heatmap — features appearing in the top 30 pairs by |r|
top_pearson_features = set()
for _, row in pearson_pairs.head(30).iterrows():
    top_pearson_features.add(row['Feature_1'])
    top_pearson_features.add(row['Feature_2'])
top_pearson_features = sorted(list(top_pearson_features))

print(f"\nFeatures in top 30 Pearson pairs: {len(top_pearson_features)}")
focused_pearson = pearson_matrix.loc[top_pearson_features, top_pearson_features]
mask_focused = np.triu(np.ones_like(focused_pearson, dtype=bool))

fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(
    focused_pearson,
    mask=mask_focused,
    annot=True,
    fmt='.2f',
    cmap='mako',
    center=0,
    vmin=-1,
    vmax=1,
    square=True,
    linewidths=0.3,
    cbar_kws={'label': 'Pearson r'},
    ax=ax,
)
ax.set_title(f'Pearson Correlation — Top {len(top_pearson_features)} Features (Immunological)',
             fontsize=14, fontweight='bold')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=9)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
plt.tight_layout()
plt.show()



#%% ########## RV2 matrix — immunological dataset (missing-methods, NaN-native) ##########
# Re-implements the hoggorm RV2 analysis without requiring imputed data.
# mm_rv2() scales inner products by the proportion of observed entries,
# so patients with some missing markers are still included.

print("RV2 matrix — immunological dataset (missing-methods, NaN-native)")

_id_cols    = ["Patient", "Timepoint", "Date"]
_timepoints = [1, 2, 3, 4, 5]

# Per-timepoint slices from df_im_vis (NOT imputed)
_dfs_r = {t: df_im_vis[df_im_vis["Timepoint"] == t] for t in _timepoints}

_n_tp      = len(_timepoints)
_rv2_mm    = np.zeros((_n_tp, _n_tp))
_n_comm_mm = np.zeros((_n_tp, _n_tp), dtype=int)

# Pre-extract patient sets once per timepoint (avoids repeated set() calls)
_pt_sets = {t: set(_dfs_r[t]["Patient"]) for t in _timepoints}

# Diagonal: RV2(A, A) = 1 by definition
for _i, _ti in enumerate(_timepoints):
    _rv2_mm[_i, _i]    = 1.0
    _n_comm_mm[_i, _i] = len(_dfs_r[_ti])

# Upper triangle only — RV2 is symmetric so mirror to lower triangle
# Reduces mm_rv2 calls from 20 → 10 for a 5-timepoint matrix
from itertools import combinations as _combns
for (_i, _ti), (_j, _tj) in _combns(enumerate(_timepoints), 2):
    _common = _pt_sets[_ti] & _pt_sets[_tj]
    _n = len(_common)
    _n_comm_mm[_i, _j] = _n_comm_mm[_j, _i] = _n
    _A_raw = (_dfs_r[_ti][_dfs_r[_ti]["Patient"].isin(_common)]
              .sort_values("Patient").drop(columns=_id_cols).values.astype(float))
    _B_raw = (_dfs_r[_tj][_dfs_r[_tj]["Patient"].isin(_common)]
              .sort_values("Patient").drop(columns=_id_cols).values.astype(float))
    # Standardise A and B SEPARATELY — matches ho.standardise(A, mode=0) in the
    # hoggorm approach. Each matrix gets its own column means/stdevs, removing
    # between-timepoint mean differences so RV2 compares correlation structure only.
    # Joint standardisation (stacking A+B) would leave mean-shifts intact and
    # suppress RV2 toward zero — which is what caused the near-zero values.
    _A = MM_StandardScaler().fit_transform(_A_raw)
    _B = MM_StandardScaler().fit_transform(_B_raw)
    _rv2_mm[_i, _j] = _rv2_mm[_j, _i] = mm_rv2(_A, _B)

_rv2_mm_df = pd.DataFrame(
    _rv2_mm,
    index=[f"T{t}" for t in _timepoints],
    columns=[f"T{t}" for t in _timepoints]
)

# Annotation: RV2 value + number of common patients per cell
_annot_mm = pd.DataFrame(
    [[f"{_rv2_mm[_i,_j]:.2f}\n(n={_n_comm_mm[_i,_j]})" for _j in range(_n_tp)]
     for _i in range(_n_tp)],
    index=_rv2_mm_df.index,
    columns=_rv2_mm_df.columns
)

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(_rv2_mm_df, annot=_annot_mm, fmt="", cmap="crest",
            vmin=0, vmax=1, square=True, ax=ax)
ax.set_title("RV2 Similarity — Immunological Dataset\n(missing-methods, NaN-native)")
plt.tight_layout()
plt.show()

print(_rv2_mm_df.round(3))



#%% ########## PCA per timepoint T1-T5 — immunological dataset (missing-methods) ##########
# Each timepoint is analysed in its own PCA space.
# Data: df_im_vis (NOT imputed) — NaN handled natively by NIPALS.
# Standardised before PCA so all features contribute equally.

print("Per-timepoint PCA — immunological dataset (missing-methods approach)")

from adjustText import adjust_text as _adj

_ncomp_tp   = 10
_cum_col_tp = sns.color_palette("crest", 1)[0]
_mako5_tp   = sns.color_palette("mako", 5)

for _t in _timepoints:
    _df_t          = _dfs_r[_t]
    _n_t           = len(_df_t)
    _patient_ids_t = _df_t["Patient"].values
    print(f"\n  T{_t}: {_n_t} patients")

    _feat_names_t = _df_t.drop(columns=_id_cols).columns.tolist()
    _Xs_t         = MM_StandardScaler().fit_transform(
                        _df_t.drop(columns=_id_cols).values.astype(float))
    _res_t        = mm_pca(_Xs_t, ncomp=_ncomp_tp)
    _scores_t     = _res_t["scores"]
    _loadings_t   = _res_t["loadings"]  # (n_features, ncomp)
    _exp_t        = _res_t["explained"] / _res_t["explained"].sum() * 100

    # Scree plot
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(1, _ncomp_tp + 1), _exp_t,
           color=sns.color_palette("mako", _ncomp_tp), label="Per-PC %")
    ax.plot(range(1, _ncomp_tp + 1), np.cumsum(_exp_t),
            marker="o", color=_cum_col_tp, linewidth=1.5, label="Cumulative %")
    ax.set_xticks(range(1, _ncomp_tp + 1))
    ax.set_xlabel("Principal Components.")
    ax.set_ylabel("Explained Variance (%)")
    ax.set_title(f"Scree Plot — Immunological Dataset T{_t} ")
    ax.legend()
    plt.tight_layout()
    plt.show()

    # Score plot — label top 20 furthest from origin
    _dist_t  = np.sqrt(_scores_t[:, 0]**2 + _scores_t[:, 1]**2)
    _top20_t = np.argsort(_dist_t)[::-1][:20]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(_scores_t[:, 0], _scores_t[:, 1],
               c=[_mako5_tp[_t - 1]], s=40, zorder=3,
               edgecolors="white", linewidth=0.4, alpha=0.85,
               label=f"T{_t} (n={_n_t})")
    _texts_t = [ax.text(_scores_t[_i, 0], _scores_t[_i, 1],
                        str(_patient_ids_t[_i]),
                        fontsize=7, fontweight="bold", color="black", zorder=5)
                for _i in _top20_t]
    _adj(_texts_t, ax=ax, expand=(1.5, 1.5),
         arrowprops=dict(arrowstyle="-", color="grey", lw=0.5))
    ax.axhline(0, color="grey", lw=0.5, linestyle="--")
    ax.axvline(0, color="grey", lw=0.5, linestyle="--")
    ax.set_xlabel(f"PC1 ({_exp_t[0]:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({_exp_t[1]:.1f}% variance)")
    ax.set_title(f"PCA Score Plot — Immunological Dataset T{_t}\n"
                 f"(top 20 patients furthest from pca-origin labelled)")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()

    # Printed table: top 20 furthest from origin
    print(f"  Top 20 patients furthest from pca-origin at T{_t}:")
    print(f"  {'Patient':>10}  {'PC1':>8}  {'PC2':>8}  {'Distance':>10}")
    for _i in _top20_t:
        print(f"  {_patient_ids_t[_i]:>10}  "
              f"{_scores_t[_i,0]:>8.3f}  {_scores_t[_i,1]:>8.3f}  "
              f"{_dist_t[_i]:>10.3f}")

    # Top 10 loadings for PC1 and PC2
    for _pc_i, _pc_name in enumerate(["PC1", "PC2"]):
        _abs_l   = np.abs(_loadings_t[:, _pc_i])
        _top10_l = np.argsort(_abs_l)[::-1][:10]
        print(f"\n  Top 10 loadings — {_pc_name} (T{_t}):")
        print(f"  {'Feature':>40}  {'Loading':>10}")
        for _k in _top10_l:
            print(f"  {_feat_names_t[_k]:>40}  {_loadings_t[_k, _pc_i]:>10.4f}")




#%% ########## Trajectory PCA: T1↔T2, T2↔T3, T1↔T3 (missing-methods) ###

print("Trajectory PCA — immunological dataset")

# We use df_im_vis (cleaned but NOT imputed).
# missing-methods NIPALS handles NaN natively via scaled inner products,
# so we do not need to impute before this analysis.
id_cols_im = ["Patient", "Timepoint", "Date"]

# Colour palette: consistently one colour per timepoint across all plots.
# T1 = mako[0], T2 = mako[2], T3 = mako[4] 
_mako5     = sns.color_palette("mako", 5)
tp_colors  = {1: _mako5[0], 2: _mako5[2], 3: _mako5[4]}
tp_labels  = {1: "T1", 2: "T2", 3: "T3"}

ncomp     = 10           # PCs to extract; PC1+PC2 used for the score plot
cum_color = sns.color_palette("crest", 1)[0]


def _filter_tp(tp, patients):
    """Return rows for *tp* restricted to *patients*, sorted by Patient ID."""
    return (
        df_im_vis[
            (df_im_vis["Timepoint"] == tp)
            & (df_im_vis["Patient"].isin(patients))
        ]
        .sort_values("Patient")
        .reset_index(drop=True)
    )


# Each tuple: (timepoint A, timepoint B, arrow colour)
# The arrow colour is the destination timepoint's colour (shows where you land).
pairs = [
    (1, 2, tp_colors[2], "T1 → T2"),
    (2, 3, tp_colors[3], "T2 → T3"),
    (1, 3, tp_colors[3], "T1 → T3"),
]

for tp_a, tp_b, arrow_color, label in pairs:

    # ── 1. Patients present at BOTH timepoints in this pair ───────────────────
    patients_pair = (
        set(df_im_vis[df_im_vis["Timepoint"] == tp_a]["Patient"])
        & set(df_im_vis[df_im_vis["Timepoint"] == tp_b]["Patient"])
    )
    n_pair = len(patients_pair)
    print(f"  {label}: {n_pair} patients")

    # ── 2. Filter + sort (sorting guarantees row i = same patient in A and B) ─
    df_a = _filter_tp(tp_a, patients_pair)
    df_b = _filter_tp(tp_b, patients_pair)

    # ── 3. Feature matrices ────────────────────────────────────────────────────
    Xa = df_a.drop(columns=id_cols_im).values.astype(float)
    Xb = df_b.drop(columns=id_cols_im).values.astype(float)

    # Stack A on top of B → one matrix for a shared PC space.
    # Each patient appears twice: once as timepoint A, once as timepoint B.
    X_pair = np.vstack([Xa, Xb])   # shape: (2 * n_pair, n_features)

    # ── 4. Standardize before PCA ─────────────────────────────────────────────
    # Without scaling, features with large absolute variance (e.g. raw cell counts)
    # dominate PC1 entirely. StandardScaler here handles NaN natively — it computes
    # mean and std from observed values only, leaving NaN positions as NaN.
    scaler = MM_StandardScaler()
    X_pair = scaler.fit_transform(X_pair)

    # ── 5. NIPALS PCA using missing-methods ────────────────────────────────────
    feat_names_pair = df_a.drop(columns=id_cols_im).columns.tolist()
    res         = mm_pca(X_pair, ncomp=ncomp)
    scores      = res["scores"]    # (2*n_pair, ncomp)
    loadings    = res["loadings"]  # (n_features, ncomp)
    explained   = res["explained"] # (ncomp,)  — raw sum-of-squares, NOT %

    # ── 6. Split scores + recover patient IDs ────────────────────────────────
    # Patient IDs preserved from the sorted filter — row i = same patient in A and B.
    patient_ids = df_a["Patient"].values
    sc_a = scores[:n_pair, :]
    sc_b = scores[n_pair:, :]

    # ── 7. Explained variance % ───────────────────────────────────────────────
    exp_pct = explained / explained.sum() * 100

    # ── 8. Scree plot ─────────────────────────────────────────────────────────
    bar_colors = sns.color_palette("mako", ncomp)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(1, ncomp + 1), exp_pct, color=bar_colors, label="Per-PC %")
    ax.plot(
        range(1, ncomp + 1), np.cumsum(exp_pct),
        marker="o", color=cum_color, linewidth=1.5, label="Cumulative %"
    )
    ax.set_xticks(range(1, ncomp + 1))   # show every integer 1–10
    ax.set_xlabel("Principal Components.")
    ax.set_ylabel("Explained Variance (%)")
    ax.set_title(f"Scree Plot — Immunological Dataset\n{label}")
    ax.legend()
    plt.tight_layout()
    plt.show()

    # ── 9. Longest trajectories — printed table ───────────────────────────────
    # For trajectory PCA the meaningful metric is arrow length: how much did
    # each patient's immune profile shift between the two timepoints?
    # (Distance from origin belongs in single-timepoint outlier analysis.)
    N_PRINT    = 20

    traj_len = np.sqrt(
        (sc_b[:, 0] - sc_a[:, 0])**2 +
        (sc_b[:, 1] - sc_a[:, 1])**2
    )
    top_traj_idx = np.argsort(traj_len)[::-1][:N_PRINT]

    print(f"\n  Top {N_PRINT} Largest Trajecotry Lengths {label}:")
    print(f"  {'Patient':>10}  {'PC1 T'+str(tp_a):>9}  {'PC2 T'+str(tp_a):>9}"
          f"  {'PC1 T'+str(tp_b):>9}  {'PC2 T'+str(tp_b):>9}  {'Traj. length':>13}")
    for i in top_traj_idx:
        print(f"  {patient_ids[i]:>10}"
              f"  {sc_a[i,0]:>9.3f}  {sc_a[i,1]:>9.3f}"
              f"  {sc_b[i,0]:>9.3f}  {sc_b[i,1]:>9.3f}"
              f"  {traj_len[i]:>13.3f}")

    # Top 10 loadings for PC1 and PC2
    for _pc_i, _pc_name in enumerate(["PC1", "PC2"]):
        _abs_l   = np.abs(loadings[:, _pc_i])
        _top10_l = np.argsort(_abs_l)[::-1][:10]
        print(f"\n  Top 10 loadings — {_pc_name} ({label}):")
        print(f"  {'Feature':>40}  {'Loading':>10}")
        for _k in _top10_l:
            print(f"  {feat_names_pair[_k]:>40}  {loadings[_k, _pc_i]:>10.4f}")

    # ── 10. Trajectory score plot ─────────────────────────────────────────────
    from adjustText import adjust_text

    # Top 20 by trajectory length get labelled.
    label_idx = np.argsort(traj_len)[::-1][:20]

    fig, ax = plt.subplots(figsize=(11, 9))

    ax.scatter(
        sc_a[:, 0], sc_a[:, 1],
        c=[tp_colors[tp_a]], label=tp_labels[tp_a],
        s=40, zorder=3, edgecolors="white", linewidth=0.4, alpha=0.8
    )
    ax.scatter(
        sc_b[:, 0], sc_b[:, 1],
        c=[tp_colors[tp_b]], label=tp_labels[tp_b],
        s=40, zorder=3, edgecolors="white", linewidth=0.4, alpha=0.8
    )

    # One arrow per patient from A → B.
    # annotation_clip=False prevents arrows from being cut off at the axes edge.
    for i in range(n_pair):
        ax.annotate(
            "",
            xy    =(sc_b[i, 0], sc_b[i, 1]),
            xytext=(sc_a[i, 0], sc_a[i, 1]),
            annotation_clip=False,
            arrowprops=dict(
                arrowstyle="-|>",
                color=arrow_color,
                lw=0.8,
                alpha=0.3,
                mutation_scale=7,
            ),
        )

    # Label top 20 by trajectory length — placed at arrow midpoint, non-overlapping.
    texts = []
    for i in label_idx:
        mx = (sc_a[i, 0] + sc_b[i, 0]) / 2
        my = (sc_a[i, 1] + sc_b[i, 1]) / 2
        texts.append(ax.text(mx, my, str(patient_ids[i]),
                             fontsize=8, fontweight="bold",
                             color="black", zorder=5))
    adjust_text(
        texts, ax=ax,
        expand=(1.5, 1.5),
        arrowprops=dict(arrowstyle="-", color="grey", lw=0.6)
    )

    ax.axhline(0, color="grey", lw=0.5, linestyle="--")
    ax.axvline(0, color="grey", lw=0.5, linestyle="--")
    ax.set_xlabel(f"PC1 ({exp_pct[0]:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({exp_pct[1]:.1f}% variance)")
    ax.set_title(
        f"Trajectory PCA — Immunological Dataset\n"
        f"{label}  (top 20 longest trajectories labelled)"
    )
    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()




#%% ########## MFA T1-T3 — immunological dataset (missing-methods, NaN-native) ##########
# Multiple Factor Analysis: each timepoint is a separate block.
# Approach: standardise each block, normalise by its first eigenvalue (NIPALS),
# then run NIPALS PCA on the horizontally stacked matrix — equivalent to the
# FactoMineR/prince MFA definition but NaN-native throughout.

print("MFA T1-T3 — immunological dataset (missing-methods, NaN-native)")

_patients_mfa = (
    set(_dfs_r[1]["Patient"])
    & set(_dfs_r[2]["Patient"])
    & set(_dfs_r[3]["Patient"])
)
_n_mfa = len(_patients_mfa)
print(f"  Patients with T1+T2+T3: {_n_mfa}")

def _get_mfa_block(tp, patients):
    return (_dfs_r[tp][_dfs_r[tp]["Patient"].isin(patients)]
            .sort_values("Patient").reset_index(drop=True))

_df_mfa1         = _get_mfa_block(1, _patients_mfa)
_df_mfa2         = _get_mfa_block(2, _patients_mfa)
_df_mfa3         = _get_mfa_block(3, _patients_mfa)
_patient_ids_mfa = _df_mfa1["Patient"].values

# Feature names for the stacked MFA matrix: prefix each column with its block (T1/T2/T3)
_feat_cols_mfa  = _df_mfa1.drop(columns=_id_cols).columns.tolist()
_feat_names_mfa = ([f"T1_{c}" for c in _feat_cols_mfa] +
                   [f"T2_{c}" for c in _feat_cols_mfa] +
                   [f"T3_{c}" for c in _feat_cols_mfa])

def _mfa_normalise(X):
    """Standardise then divide by sqrt(first eigenvalue) — NaN-native NIPALS."""
    Xs   = MM_StandardScaler().fit_transform(X)
    lam1 = mm_pca(Xs, ncomp=1)["explained"][0]
    return Xs / np.sqrt(lam1)

# Stack three normalised blocks horizontally → shape: (n_patients, 3 * n_features)
_X_mfa_all = np.hstack([
    _mfa_normalise(_df_mfa1.drop(columns=_id_cols).values.astype(float)),
    _mfa_normalise(_df_mfa2.drop(columns=_id_cols).values.astype(float)),
    _mfa_normalise(_df_mfa3.drop(columns=_id_cols).values.astype(float)),
])

_ncomp_mfa    = 5
_res_mfa      = mm_pca(_X_mfa_all, ncomp=_ncomp_mfa)
_scores_mfa   = _res_mfa["scores"]
_loadings_mfa = _res_mfa["loadings"]  # (3 * n_features, ncomp)
_exp_mfa      = _res_mfa["explained"] / _res_mfa["explained"].sum() * 100

# Scree plot
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(range(1, _ncomp_mfa + 1), _exp_mfa,
       color=sns.color_palette("mako", _ncomp_mfa), label="Per-PC %")
ax.plot(range(1, _ncomp_mfa + 1), np.cumsum(_exp_mfa),
        marker="o", color=sns.color_palette("crest", 1)[0],
        linewidth=1.5, label="Cumulative %")
ax.set_xticks(range(1, _ncomp_mfa + 1))
ax.set_xlabel("Principal Components.")
ax.set_ylabel("Explained Variance (%)")
ax.set_title("Scree Plot — MFA Immunological Dataset T1+T2+T3\n(missing-methods)")
ax.legend()
plt.tight_layout()
plt.show()

# Global score plot — label top 20 furthest from origin
_dist_mfa  = np.sqrt(_scores_mfa[:, 0]**2 + _scores_mfa[:, 1]**2)
_top20_mfa = np.argsort(_dist_mfa)[::-1][:20]

fig, ax = plt.subplots(figsize=(9, 7))
ax.scatter(_scores_mfa[:, 0], _scores_mfa[:, 1],
           c=[sns.color_palette("mako", 1)[0]],
           s=40, zorder=3, edgecolors="white", linewidth=0.4, alpha=0.85,
           label=f"Patients (n={_n_mfa})")
_texts_mfa = [ax.text(_scores_mfa[_i, 0], _scores_mfa[_i, 1],
                      str(_patient_ids_mfa[_i]),
                      fontsize=7, fontweight="bold", color="black", zorder=5)
              for _i in _top20_mfa]
_adj(_texts_mfa, ax=ax, expand=(1.5, 1.5),
     arrowprops=dict(arrowstyle="-", color="grey", lw=0.5))
ax.axhline(0, color="grey", lw=0.5, linestyle="--")
ax.axvline(0, color="grey", lw=0.5, linestyle="--")
ax.set_xlabel(f"PC1 ({_exp_mfa[0]:.1f}% variance)")
ax.set_ylabel(f"PC2 ({_exp_mfa[1]:.1f}% variance)")
ax.set_title("MFA Score Plot — Immunological Dataset T1+T2+T3\n"
             "(missing-methods, top 20 furthest labelled)")
ax.legend(loc="best")
plt.tight_layout()
plt.show()

# Top 10 loadings for Dim1 and Dim2 (MFA)
# Each feature name is prefixed with its block (T1/T2/T3) so you can see
# which timepoint drives each dimension most.
for _pc_i, _pc_name in enumerate(["Dim1", "Dim2"]):
    _abs_l   = np.abs(_loadings_mfa[:, _pc_i])
    _top10_l = np.argsort(_abs_l)[::-1][:10]
    print(f"\n  Top 10 loadings — {_pc_name} (MFA T1+T2+T3):")
    print(f"  {'Feature':>45}  {'Loading':>10}")
    for _k in _top10_l:
        print(f"  {_feat_names_mfa[_k]:>45}  {_loadings_mfa[_k, _pc_i]:>10.4f}")





#%%########### Imputing missing values using miceforest and median
# in order to be able to run pyod outlier detection


# handling name issues - mice forest does not take symbols
feature_cols = df_im.columns.difference(exclude_cols)

def clean_colname(col):
    col = col.strip()
    col = re.sub(r"[^\w]", "_", col)
    col = re.sub(r"_+", "_", col)
    return col

# map for renaming columns
rename_map = {c: clean_colname(c) for c in feature_cols}

# rename columns
df_im2 = df_im.reset_index(drop=True).rename(columns=rename_map)

# imputing with miceforest with renamed columns
X_im = df_im2[list(rename_map.values())].copy()

import miceforest as mf

# MICE imputation with mean matching
kernel = mf.ImputationKernel(
    X_im,
    num_datasets=5,
    mean_match_candidates=5,  # impute from 5 nearest observed values
    random_state=42
)

kernel.mice(10)    # 10 iterations

# Average imputed values across all 5 datasets
imputed_datasets = [kernel.complete_data(dataset=i) for i in range(5)]
X_imputed_renamed = sum(imputed_datasets) / len(imputed_datasets)

# changing back to original column names
reverse_rename_map = {v: k for k, v in rename_map.items()}
X_imputed = X_imputed_renamed.rename(columns=reverse_rename_map)

# reindex to preserve original column order
X_imputed = X_imputed.reindex(columns=feature_cols)

# final imputation
df_im_imputed = pd.concat(
    [
        df_im[exclude_cols].reset_index(drop=True),
        X_imputed.reset_index(drop=True)
    ],
    axis=1
)

# ensure final dataframe has columns in same order as original
df_im_imputed = df_im_imputed[df_im.columns]

# New tablereport of imputed data
TableReport(df_im_imputed, max_plot_columns=138)


# Median imputation:
df_im_median = df_im.copy()
for col in feature_cols:
    median_value = df_im_median[col].median()
    df_im_median[col] = df_im_median[col].fillna(median_value)  


# Caluculate correlation between datasets:







#%%######### PyOD Ensemble Outlier Detection (Zyran approach) - Immunological Dataset ########

# This section uses the pre-built outlier detection framework from:
# https://gitlab.com/zryan.rz/master_outlier_detection_h23
#
# Pipeline:
#   1. Miceforest-imputed / median-imputed immunological data -> scale with StandardScaler
#   2. GEC (Gaussian Ensemble Comparison): fits all candidate algorithms and
#      selects the 6 most *dissimilar* ones to form a diverse ensemble
#   3. visualiser_OD: fits the 6 selected algorithms, aggregates scores via
#      median probability across algorithms, and produces three plots:
#        - PCA biplot (hoggorm NIPALS PCA)
#        - Scatter: median probability vs. average confidence (marker size =
#          std of confidence, colour = std of probability)
#        - Pairplots of PC1-5 coloured by median probability / confidence
#   Contamination is fixed at 0.1 / 0.05 (trying both)

import sys
import random
from pathlib import Path

# Make pyod_zyran folder importable into this file from src/ 
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pyod_zyran.GEC import calculate_GEC
from pyod_zyran.Visualisering import visualiser_OD
from sklearn.preprocessing import StandardScaler
from pyod.models.qmcd import QMCD
from pyod.models.inne import INNE
from pyod.models.knn import KNN as KNN_od
from pyod.models.lof import LOF as LOF_od
from pyod.models.iforest import IForest as IForest_od
from pyod.models.pca import PCA as PCA_od
from pyod.models.loda import LODA
from pyod.models.hbos import HBOS
from pyod.models.ocsvm import OCSVM
from pyod.models.ecod import ECOD as ECOD_od
from pyod.models.copod import COPOD as COPOD_od
from pyod.models.lscp import LSCP

if not hasattr(np, 'bool'):
    np.bool = bool

# --- Data: imputed immunological dataset (with miceforest) ---
# Drop ID columns, scale data
X_ens = df_im_imputed[feature_cols].copy()
patient_labels = (
    df_im_imputed["Patient"].astype(str) + "-T" + df_im_imputed["Timepoint"].astype(str)
).tolist()

scaler_ens = StandardScaler()
X_sc = pd.DataFrame(scaler_ens.fit_transform(X_ens), columns=X_ens.columns)

# --- Build candidate algorithm list  ---
contamination = 0.05 # set to 0.1 for standard contamination value?
random.seed(42)
detector_list_lscp = [IForest_od(n_estimators=n) for n in random.sample(range(5, 200), 10)]

list_OD_classes   = [QMCD, INNE, KNN_od, LOF_od, IForest_od, PCA_od, LODA, HBOS, OCSVM, ECOD_od, COPOD_od]
list_OD_strings   = [cls.__name__ for cls in list_OD_classes]
list_OD_init      = [LSCP(detector_list=detector_list_lscp, contamination=contamination)
                     if cls == LSCP
                     else cls(contamination=contamination)
                     for cls in list_OD_classes]

# --- GEC function — select 6 most dissimilar algorithms ---
print("Running GEC to select 6 most dissimilar algorithms...")
final_selected_algos, tau_dissimilarity_df = calculate_GEC(
    X_sc.values,
    list_OD_init,
    list_OD_strings,
    percentages=[0.90, 0.98, 1.00]
)
print(f"GEC selected algorithms: {final_selected_algos}")


# --- Re-initialise only the selected 6 algorithms ---
algo_class_map     = {cls.__name__: cls for cls in list_OD_classes}
initialized_modules = [
    algo_class_map[name](contamination=contamination)
    for name in final_selected_algos
    if name in algo_class_map
]
print(f"Ensemble: {len(initialized_modules)} algorithms with contamination={contamination}")

# --- Plot using visualiser_OD ---
print("Running visualiser_OD...")
no_od_df, y_prob_mean, y_conf_mean, y_prob_arr, y_conf_arr, train_scores_ens = visualiser_OD(
    X_sc_ens,
    initialized_modules,
    patient_labels,
    visualize=True
)

# --- Summary ---
print(f"\n=== Outlier Detection Summary (contamination={contamination}) ===")
for n in [1, 3, len(initialized_modules)]:
    label = f"Flagged by >= {n} algorithm{'s' if n > 1 else ''}"
    print(f"{label}: {(no_od_df['No. OD Detected'] >= n).sum()}")

# Observations in the upper-right quadrant: median probability > 0.9 AND avg confidence > 0.9
high_prob_conf_mask = (y_prob_mean > 0.9) & (y_conf_mean > 0.9)
outlier_candidates = no_od_df[high_prob_conf_mask].copy()
outlier_candidates['Median_Probability'] = y_prob_mean[high_prob_conf_mask]
outlier_candidates['Avg_Confidence'] = y_conf_mean[high_prob_conf_mask]
outlier_candidates = outlier_candidates.sort_values('Median_Probability', ascending=False)

print(f"\n=== Upper-right Quadrant Observations (median prob. > 0.9  &  avg confidence > 0.9) ===")
print(f"Total: {len(outlier_candidates)}")
print(outlier_candidates.to_string())



# Run using median imputed dataset:











# %%################ RAW CLINICAL DATASET #############################

# Table report of clinical dataset
print("TableReport of raw clinical dataset:")
TableReport(df_cl, max_plot_columns=138)

# na analysis of clinical dataset
print("Na analysis of clinical dataset:")
na.altair.plot_heatmap(df_cl)


# Raw clinical dataset statistics
print(f"\n=== Raw clinical dataset overview ===")
print(f"  Shape         : {df_cl.shape[0]} rows × {df_cl.shape[1]} columns")
print(f"  Patients      : {df_cl['Patient'].dropna().nunique()}")
print(f"  Missing values: {df_cl.isna().sum().sum()} total "
      f"({df_cl.isna().mean().mean()*100:.1f}% of all cells)")

# Missing values per column (top 15 most incomplete)
missing = (df_cl.isna().mean() * 100).sort_values(ascending=False)
print(f"\nTop 15 columns by missingness (%):\n{missing.head(15).round(1).to_string()}")

# Patient count per timepoint (using Erfassungszeitpunkt)
if 'Erfassungszeitpunkt' in df_cl.columns:
    tp_counts = (
        df_cl[df_cl['Datum'].notna()]
        ['Erfassungszeitpunkt']
        .str.extract(r'\d+\.\d+\.(\d+)')[0]
        .value_counts()
        .sort_index()
    )
    print(f"\nRows per timepoint:\n{tp_counts.to_string()}")

# --- Plots ---
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
colors = sns.color_palette('mako', 5)

# 1. Missing value rate per column
missing_plot = missing[missing > 0]
axes[0].barh(range(min(20, len(missing_plot))),
             missing_plot.values[:20][::-1],
             color=colors[2])
axes[0].set_yticks(range(min(20, len(missing_plot))))
axes[0].set_yticklabels(missing_plot.index[:20][::-1], fontsize=7)
axes[0].set_xlabel('Missing (%)')
axes[0].set_title('Top 20 columns by missingness')

# 2. Rows per timepoint
if 'Erfassungszeitpunkt' in df_cl.columns:
    axes[1].bar(tp_counts.index.astype(str), tp_counts.values, color=colors[1])
    axes[1].set_xlabel('Timepoint')
    axes[1].set_ylabel('Row count')
    axes[1].set_title('Rows per timepoint')

# 3. Age distribution (if column present)
age_col = next((c for c in df_cl.columns if 'age' in c.lower() or 'alter' in c.lower()), None)
if age_col:
    age_vals = pd.to_numeric(df_cl[age_col], errors='coerce').dropna()
    axes[2].hist(age_vals, bins=20, color=colors[3], edgecolor='white')
    axes[2].set_xlabel('Age')
    axes[2].set_ylabel('Count')
    axes[2].set_title(f'Age distribution (n={len(age_vals)})')
    print(f"\nAge — mean: {age_vals.mean():.1f}, "
          f"median: {age_vals.median():.1f}, "
          f"range: {age_vals.min():.0f}–{age_vals.max():.0f}")
else:
    axes[2].set_visible(False)

plt.suptitle('Raw Clinical Dataset — Overview', fontweight='bold')
plt.tight_layout()
plt.show()




#%%############ Cleaning clinical dataset #############################
# Pipeline: Forward-fill -> Exclude patients -> Rename -> Define nulls ->
# Deduplicate categories -> Extract numerics -> Transform columns -> Change dtype 
# -> drop na>25% columns -> visualize -> prepare target before modeling


# Clinical preprocessing: helper functions

def move_column_after(df, col_to_move, after_col):
    """Move a column to position right after another column."""
    cols = df.columns.tolist()
    cols.insert(cols.index(after_col) + 1, cols.pop(cols.index(col_to_move)))
    return df[cols]


def extract_numeric(series):
    """Extract numeric value from ordinal questionnaire entries (scale 1-4 or 1-5).
    Handles: comma-separated multi-select "1,2" -> avg, multiple numbers with text
    "3 (tagsüber), 4 (nachts)" -> avg, leading number "3 left side" -> 3.
    """
    s = series.astype(str).str.strip()

    def parse_entry(val):
        if val in ('nan', '', 'None'):
            return np.nan
        # Comma-separated multi-select: "3,4" -> average
        if re.match(r'^\d+(\s*,\s*\d+)+$', val):
            return np.mean([float(x) for x in val.split(',')])
        # Range: "2-3", "1 - 4"
        m = re.match(r'^(\d+)\s*[-–]\s*(\d+)', val)
        if m:
            return (float(m.group(1)) + float(m.group(2))) / 2
        # Multiple numbers with text: "3 (tagsüber), 4 (nachts)" -> average all
        all_nums = re.findall(r'\b(\d+)\b', val)
        if len(all_nums) > 1:
            return np.mean([float(x) for x in all_nums])
        if len(all_nums) == 1:
            return float(all_nums[0])
        return np.nan

    return s.apply(parse_entry)


def extract_continuous(series):
    """Extract numeric value from continuous scale entries (e.g., pain_scale 1-10).
    Comma is German decimal ("9,7" = 9.7).
    Handles:
      - German decimals: "9,7" -> 9.7
      - Ranges: "20-30" -> midpoint 25.0
      - Trailing text: "40 (left side)" -> 40
      - Ruhe (at rest) entries: "7,3-dauernd bei Belastung, 10 aus der Ruhe" -> 10
        (prefer the resting pain value when both load and rest values are given)
    """
    s = series.astype(str).str.strip()

    def parse_entry(val):
        if val in ('nan', '', 'None'):
            return np.nan
        # Ruhe (at rest): extract the number directly before "Ruhe" or "aus der Ruhe"
        # e.g. "7,3-dauernd bei Belastung, 10 aus der Ruhe" -> 10
        m_ruhe = re.search(r'(\d+[.,]?\d*)\s*(?:aus\s+der\s+)?[Rr]uhe', val)
        if m_ruhe:
            return float(m_ruhe.group(1).replace(',', '.'))
        # Pure range: "20-30", "10 - 20" -> midpoint
        m = re.match(r'^(\d+[.,]?\d*)\s*[-–]\s*(\d+[.,]?\d*)\s*$', val)
        if m:
            return (float(m.group(1).replace(',', '.')) +
                    float(m.group(2).replace(',', '.'))) / 2
        # Leading number (with optional text): "9,7" -> 9.7, "40 (left side)" -> 40
        m = re.match(r'^(\d+[.,]?\d*)', val)
        if m:
            return float(m.group(1).replace(',', '.'))
        return np.nan

    return s.apply(parse_entry)


def split_bmi_column(df, col_name='overweight_bmi'):
    """Split combined overweight/BMI column into two columns.
    Input format: "ja (28.5)", "nein", "n.D" (missing).
    Output: 'overweight' (ja/nein) + 'bmi' (float).
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
    # German month name to number
    month_map = {'jan': 1, 'feb': 2, 'mär': 3, 'mar': 3, 'apr': 4, 'mai': 5,
                 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'okt': 10, 'oct': 10,
                 'nov': 11, 'dez': 12, 'dec': 12}

    def parse_entry(val, meas_date):
        if pd.isna(val):
            return pd.NA
        s = str(val).strip()

        # "einige Jahre" / "einge j." = approximately 1 year → 12 months
        if s.lower() in ('einige jahre', 'einige j.', 'einge j.'):
            return 12.0

        # Vague / unparseable entries → NaN
        if s.lower() in ('jahre', 'jahre ', 'mehrere', 'mehrere jahre',
                         'mehrere monate', 'mehreren mo.', 'täglich'):
            return pd.NA

        # Full date string: "2023-04-01 00:00:00" → calc months from measurement date
        date_match = re.match(r'^(\d{4})-(\d{2})-(\d{2})', s)
        if date_match:
            if pd.notna(meas_date):
                symptom_date = pd.Timestamp(f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}")
                return max(0, (pd.Timestamp(meas_date) - symptom_date).days / 30.44)
            return pd.NA

        # German date "01.04.2023" (DD.MM.YYYY) → calc months from measurement date
        de_date_match = re.match(r'^~?(\d{1,2})\.(\d{1,2})\.(\d{4})$', s.strip())
        if de_date_match:
            if pd.notna(meas_date):
                symptom_date = pd.Timestamp(
                    f"{de_date_match.group(3)}-{int(de_date_match.group(2)):02d}-{int(de_date_match.group(1)):02d}"
                )
                return max(0, (pd.Timestamp(meas_date) - symptom_date).days / 30.44)
            return pd.NA

        # "~02/2022" → month/year
        my_match = re.match(r'^~?(\d{2})/(\d{4})$', s)
        if my_match:
            if pd.notna(meas_date):
                symptom_date = pd.Timestamp(f"{my_match.group(2)}-{my_match.group(1)}-01")
                return max(0, (pd.Timestamp(meas_date) - symptom_date).days / 30.44)
            return pd.NA

        # "Okt/Nov 2022" or "Feb 2022 (7Mo.)"
        mon_match = re.match(r'^(\w{3})\w*(?:/\w+)?\s+(\d{4})', s, re.IGNORECASE)
        if mon_match:
            # Check for explicit months in parens first: "Feb 2022 (7Mo.)"
            paren_match = re.search(r'\((\d+)\s*Mo', s)
            if paren_match:
                return float(paren_match.group(1))
            mon_key = mon_match.group(1).lower()
            year = int(mon_match.group(2))
            if mon_key in month_map and pd.notna(meas_date):
                symptom_date = pd.Timestamp(f"{year}-{month_map[mon_key]:02d}-01")
                return max(0, (pd.Timestamp(meas_date) - symptom_date).days / 30.44)
            return pd.NA

        # Remove parenthetical comments: "120 Mo. (?)" → "120 Mo.", "2 (?) Mo." → "2 Mo."
        s_clean = re.sub(r'\(\?\)', '', s)
        # "20 J. (akut 2 Mo.)" → take the main number (20 J.)
        s_clean = re.sub(r'\([^)]*\)', '', s_clean).strip()
        # Remove leading ~, >, <, "akut"
        s_clean = re.sub(r'^[~><]\s*', '', s_clean)
        s_clean = re.sub(r'^akut\s+', '', s_clean, flags=re.IGNORECASE)

        # Detect unit: Jahre/J. = years, Monate/Mo. = months
        is_years = bool(re.search(r'(Jahr\w*|J\.?\b)', s_clean, re.IGNORECASE))

        # Handle fractions: "1/2 J." → 0.5
        frac_match = re.match(r'(\d+)/(\d+)', s_clean)
        if frac_match:
            number = float(frac_match.group(1)) / float(frac_match.group(2))
            return number * 12 if is_years else number

        # Handle ranges: "6-12 Mo." → midpoint 9, "4-5 Jahre" → 4.5 years
        range_match = re.search(r'(\d+[,.]?\d*)\s*-\s*(\d+[,.]?\d*)', s_clean)
        if range_match:
            start = float(range_match.group(1).replace(',', '.'))
            end = float(range_match.group(2).replace(',', '.'))
            number = (start + end) / 2
            return number * 12 if is_years else number

        # Handle German decimal: "1,5 J." → 1.5
        num_match = re.search(r'(\d+[,.]?\d*)', s_clean)
        if num_match:
            number = float(num_match.group(1).replace(',', '.'))
            return number * 12 if is_years else number

        return pd.NA

    if date_series is not None:
        return pd.Series(
            [parse_entry(v, d) for v, d in zip(series, date_series)],
            index=series.index
        )
    return series.apply(lambda v: parse_entry(v, None))


def standardize_target_volume(series):
    """Standardize target_volume column: map body part variants to English names,
    extract treatment side into separate column.
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
    """Standardize diagnosis column: map German/English variants to standardized
    English diagnosis names. Combined diagnoses kept as 'Name1, Name2'.
    Side is NOT extracted here — use target_side from target_volume instead.
    Returns standardized diagnosis series.
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
    """Standardize pain_points column: map German body parts to English,
    extract side (L/R/B) per body part. Pure number entries (2, 3, 4) become NaN.
    Returns standardized series with format 'BodyPart Side, BodyPart Side'.
    """
    # Order matters: more specific compound words before shorter substrings
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
        ('Arm',             [r'\barm\b']),  # word boundary to avoid matching oberarm/unterarm
        ('Wrist',           ['handgelenk', 'hangelenk']),
        ('Thumb',           ['daumen', 'daumensattelgelenk']),
        ('Hand',            [r'\bhand\b', 'hände']),
        ('Finger',          ['finger']),
    ]

    def find_side(seg):
        """Extract side from a text segment."""
        s = seg.lower().strip()
        # Bilateral patterns first
        if re.search(r'beide|bds', s):
            return 'B'
        if re.search(r'li\s*[+&/]\s*re|re\s*[+&/]\s*li', s):
            return 'B'
        if re.search(r'li\s+u\.?\s+re|re\s+u\.?\s+li', s):
            return 'B'
        if re.search(r'li\s+und\s+re|re\s+und\s+li', s):
            return 'B'
        # Left
        if re.search(r'\bli\b|\blinks\b|\blinke[rns]?\b', s):
            return 'L'
        # Right
        if re.search(r'\bre\b|\brechts\b|\brechte[rns]?\b|\brecht\b', s):
            return 'R'
        return ''

    def find_body_part(seg):
        """Find body part in a text segment."""
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
        # Pure numbers → NaN
        if re.match(r'^\d+$', s):
            return pd.NA
        # Remove parentheses but keep content inside (they may contain body parts)
        s_clean = s.replace('(', '').replace(')', '')
        # Remove question marks and trailing digits stuck to words ("Daumen re3" → "Daumen re")
        s_clean = re.sub(r'[?]', '', s_clean)
        s_clean = re.sub(r'(\D)\d+\b', r'\1', s_clean)
        # Split by comma and semicolon
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
                # Side-only segment: applies to previous body part (e.g., "Ferse li, re")
                body = last_body_part
            if body:
                entry = f"{body} {side}".strip()
                if entry not in results:
                    results.append(entry)
                last_body_part = body

        if not results:
            return s.strip()  # keep original if nothing matched

        # Collapse same body part with both L and R sides into B
        # e.g. ["Heel R", "Heel L"] -> ["Heel B"]
        from collections import defaultdict
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
        # Handle duplicate entries like "0,2\n0,2" — take the first
        s = s.split('\n')[0].strip()
        # Extract material (Cu or Al)
        material = pd.NA
        if re.search(r'Cu', s, re.IGNORECASE):
            material = 'Cu'
        elif re.search(r'Al', s, re.IGNORECASE):
            material = 'Al'
        # Extract numeric value: replace German comma with dot
        num_match = re.search(r'(\d+[,.]?\d*)', s)
        if num_match:
            num_str = num_match.group(1).replace(',', '.')
            return float(num_str), material
        return pd.NA, material

    parsed = df[col_name].apply(parse_filter)
    df.insert(col_idx, 'filter_mm', parsed.apply(lambda x: x[0]))
    df.insert(col_idx + 1, 'filter_material', parsed.apply(lambda x: x[1]))
    return df.drop(columns=[col_name])



# Cumulative dose: parse total dose from mixed formats
def parse_cumulative_dose(val):
    if pd.isna(val):
        return pd.NA
    s = str(val).strip()
    # "L: 3;   R: 6" → sum both sides
    if re.search(r'[LR]\s*:', s):
        numbers = re.findall(r'(\d+\.?\d*)', s)
        return sum(float(n) for n in numbers) if numbers else pd.NA
    # "3(6)" or "3 (6Gy Right)" → take number inside parentheses (= total)
    paren_match = re.search(r'\((\d+\.?\d*)', s)
    if paren_match:
        return float(paren_match.group(1))
    # "3\n3" (duplicate) → take first line
    s = s.split('\n')[0].strip()
    # Standalone number
    num_match = re.match(r'^(\d+\.?\d*)$', s)
    if num_match:
        return float(num_match.group(1))
    return pd.NA



def encode_therapy_columns(df, col_name='previous_therapy'):
    """Encode comma-separated therapy codes (1-7) into binary columns.
    Input: "1,3,5" or "1,2,3 (medicine)". Output: previous_therapy_1 ... previous_therapy_7.
    """
    col_idx = df.columns.get_loc(col_name)
    for i in range(1, 8):
        binary_col = df[col_name].str.contains(rf'\b{i}\b', na=False).astype(int)
        df.insert(col_idx + i - 1, f'previous_therapy_{i}', binary_col)
    return df.drop(columns=[col_name])



def standardize_response(df, response_col='response'):
    """Parse the raw response column into two structured columns:

    response_category : standardized category string (CR / PR / NI).
        - Known phrases ('subtotal remission', 'recovery only on the right side', etc.) → PR.
        - Typo 'no imrovement' → NI.
        - Multiple categories in one entry kept as comma-separated, e.g. 'CR, NI' or 'PR, CR'.
        - Entries that cannot be mapped are kept as-is (for manual review).

    response_percent  : numeric percentage extracted from the entry (float).
        - 'PR > 80'   → 80.0
        - 'CR (100%)' → 100.0
        - 'PR~75'     → 75.0
        - '80-90'     → 85.0  (midpoint of range)
        - NaN when no number is present.

    Note: response_category is metadata only — NOT used as a modeling target.
    Regression targets (pain_scale_t2, pain_scale_reduction) are created in Step 9.
    """
    df = df.copy()
    raw = df[response_col].astype(str).str.strip()

    # Phrase → canonical token mapping (applied before category detection).
    # IMPORTANT: longer/more-specific phrases must come before shorter ones so
    # that e.g. 'no improvement' is replaced with 'ni' before 'improvement'
    # is replaced with 'pr' (otherwise 'no improvement' → 'no pr' → wrong).
    phrase_map = {
        'no improvement':                  'ni',   # must precede 'improvement'
        'no imrovement':                   'ni',   # typo variant
        'no imrpvovemnet':                 'ni',   # typo variant
        'recovery only on the right side': 'pr',
        'initial improvement':             'pr',   # must precede 'improvement'
        'subtotal remission':              'pr',
        'improvement':                     'pr',
        'pd':                              'pr',   # PD abbreviation in bilateral entries → PR
    }

    categories = pd.Series(pd.NA, index=df.index, dtype=object)
    percents   = pd.Series(np.nan, index=df.index, dtype='float64')

    _null_marker_pat = re.compile(r'^([kK]\.?[aA]\.?|[nN]\.?[dD]\.?)$')

    for idx, val in raw.items():
        if val in ('nan', '', 'None', 'NaN'):
            continue
        # Skip German null markers (n.D / n.D. / k.A. variants) — leave as pd.NA
        if _null_marker_pat.match(val.strip()):
            continue

        s = val.lower().strip()

        # Apply phrase replacements
        for phrase, replacement in phrase_map.items():
            s = s.replace(phrase.lower(), replacement)

        # Bilateral entries: standalone side marker followed by a number (e.g. 'R > 50',
        # 'L=30') means that side had a partial response — convert to a 'pr > N' token
        # so both the category and percentage are captured correctly.
        # '\b([lr])\b' matches only standalone L/R, not the 'r' inside 'cr' or 'pr'.
        s = re.sub(r'\b([lr])\s*[>~=]\s*(\d+)', r'pr > \2', s)

        # Extract numeric percentage from entry
        # Range: "80-90" → 85 (midpoint); single: ">80", "~75", "100%" → number
        range_m = re.search(r'(\d+)\s*[-–]\s*(\d+)', s)
        single_m = re.search(r'[>~<]?\s*(\d+)\s*%?', s)
        if range_m:
            percents[idx] = (float(range_m.group(1)) + float(range_m.group(2))) / 2
        elif single_m:
            percents[idx] = float(single_m.group(1))

        # Detect which response categories are present in the entry.
        # NI must be checked before PR/CR so 'ni' token (mapped above) is caught.
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
            # Keep original for unrecognized entries so they can be reviewed
            categories[idx] = val.strip()

    df['response_category'] = categories.astype('category')
    df['response_percent']  = percents

    print(f"\nResponse categories:\n{df['response_category'].value_counts(dropna=False).to_string()}")
    print(f"\nResponse percent — {df['response_percent'].notna().sum()} entries with a numeric value:")
    print(df['response_percent'].describe())
    return df



#%% 1 — Forward fill + timepoint + clean copy
########################################################
# TableReport(df_cl) + NA heatmap + raw statistics: already run above (lines ~800-870)

# Patient-level columns: constant across timepoints, only filled in first row per patient
patient_level_cols = [
    'Patient', 'Unnamed: 2', 'Age at start', 'Gender', 'Weight [kg]', 'Height [cm]',
    'Overweight? BMI', 'Besserung nach Nachuntersuchung laut Arztbrief in %',
    'Comments questionnaire', 'Diagnosis', 'Target volume', 'single fraction',
    'kummulative dose (x) - if two targets were applied', 'FHA', 'kV', 'mA',
    'Filter', 'Response', 'further comments'
]

df_cl['Patient_Group'] = df_cl['Patient'].notna().cumsum()
df_cl[patient_level_cols + ['Unnamed: 0']] = (
    df_cl.groupby('Patient_Group')[patient_level_cols + ['Unnamed: 0']].ffill()
)
df_cl = df_cl.drop(columns=['Patient_Group'])

# Extract timepoint number from Erfassungszeitpunkt (e.g., "01.01.1" → 1)
df_cl['Timepoint'] = (
    df_cl['Erfassungszeitpunkt']
    .str.extract(r'\d+\.\d+\.(\d+)')[0]
    .astype(float)
)

# Working copy — date filter deferred to step 4 so Ausschluss check (Unnamed:0) still works
df_cl_clean = df_cl.copy()
print(f"\ndf_cl_clean initialised: {df_cl_clean.shape[0]} rows × {df_cl_clean.shape[1]} columns")


#%% 2 — Exclusions + EORTC column drop
########################################################

# Exclude patients marked with "Ausschluss" keyword (uses Unnamed: 0, before it's dropped)
exclude_mask = df_cl_clean['Unnamed: 0'].str.contains('Ausschluss', case=False, na=False)
excluded_patients = df_cl_clean.loc[exclude_mask, 'Patient'].dropna().unique()
print(f"Excluded {len(excluded_patients)} patients by Ausschluss keyword: {excluded_patients}")
df_cl_clean = df_cl_clean[~exclude_mask]

# Exclude patients irradiated at MULTIPLE DIFFERENT body parts in the same course
multi_body_patients = [3, 45, 184, 162, 179, 156, 54, 47, 219]
print(f"\nVerifying multi-body-part patients (to be excluded):")
for pid in multi_body_patients:
    rows = df_cl_clean[df_cl_clean['Patient'] == pid]
    if len(rows) > 0:
        volumes = rows['Target volume'].dropna().unique()
        print(f"  Patient {pid}: Target volume(s) = {volumes}")
    else:
        print(f"  Patient {pid}: not found in dataset")
df_cl_clean = df_cl_clean[~df_cl_clean['Patient'].isin(multi_body_patients)]
print(f"Removed {len(multi_body_patients)} multi-body-part patients")

# Drop EORTC health/function questionnaire columns
try:
    col_list = df_cl_clean.columns.tolist()
    start_col = 'Schwierigkeiten körperlicher Anstrengung'
    end_col_options = [
        'Allgemeinzustand Gesundheit HEUTE',
        'Allgemeinzustand Gesundheut HEUTE',
    ]
    end_col = next((c for c in end_col_options if c in col_list), None)
    if start_col not in col_list:
        print(f"Warning: EORTC start column '{start_col}' not found — no columns dropped")
    elif end_col is None:
        print(f"Warning: EORTC end column not found — no columns dropped")
    else:
        start_idx      = col_list.index(start_col)
        end_idx        = col_list.index(end_col)
        q_cols_to_drop = col_list[start_idx : end_idx + 1]
        df_cl_clean    = df_cl_clean.drop(columns=q_cols_to_drop)
        print(f"\nDropped {len(q_cols_to_drop)} EORTC questionnaire columns:")
        print(f"  From: '{start_col}'")
        print(f"  To  : '{end_col}'")
        print(f"  Cols: {q_cols_to_drop}")
except Exception as e:
    print(f"Warning: Could not drop EORTC columns: {e}")

print(f"\nAfter exclusions: {df_cl_clean['Patient'].nunique()} patients, {len(df_cl_clean)} rows")


#%% 3 — Rename columns
########################################################

clinical_names = {
    # Patient demographics
    "Patient": "Patient", "Timepoint": "Timepoint",
    "Age at start": "age_at_start", "Gender": "gender",
    "Weight [kg]": "weight_kg", "Height [cm]": "height_cm",
    "Overweight? BMI": "overweight_bmi",

    # Dates and timings
    "Erfassungszeitpunkt": "measurement_timepoint", "Datum": "date",
    "Beschwerden seit": "symptoms_months", "vorherige Therapie": "previous_therapy",

    # Pain characteristics
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
df_cl_clean = df_cl_clean.rename(columns=clinical_names)
df_cl_clean = move_column_after(df_cl_clean, 'Timepoint', 'Patient')
print(f"Columns renamed: {len(clinical_names)}")


#%% 4 — Drop unused columns + empty rows
########################################################

# Drop metadata/admin columns not needed for analysis
cols_to_drop = ['Unnamed: 0', 'Unnamed: 2', 'further comments', 'Comments questionnaire']
df_cl_clean = df_cl_clean.drop(columns=[c for c in cols_to_drop if c in df_cl_clean.columns])

# Drop rows with no measurement date (empty slots)
n_before = len(df_cl_clean)
df_cl_clean = df_cl_clean[df_cl_clean['date'].notna()].copy()
print(f"Dropped {n_before - len(df_cl_clean)} rows with no date (empty measurement slots)")

# Drop rows where ALL questionnaire columns are NaN (completely empty questionnaire rows)
questionnaire_range = df_cl_clean.loc[:, 'symptoms_months':'improvement_percent'].columns
n_before = len(df_cl_clean)
all_q_nan = df_cl_clean[questionnaire_range].isna().all(axis=1)
dropped_q = df_cl_clean[all_q_nan][['Patient', 'Timepoint']]
if len(dropped_q) > 0:
    print(f"\nDropping {all_q_nan.sum()} rows with all questionnaire columns NaN:")
    print(dropped_q.to_string())
df_cl_clean = df_cl_clean[~all_q_nan].copy()
print(f"\nAfter step 4: {df_cl_clean['Patient'].nunique()} patients, {len(df_cl_clean)} rows")


#%% 5 — Baseline copy + quick inspect
########################################################

# df_cl_bcat: English column names, raw (unparsed) values.
# CatBoost handles raw strings natively — this is the baseline modeling input.
# Regression targets will be joined later from pain_targets (step 12).
df_cl_bcat = df_cl_clean.copy()

TableReport(df_cl_bcat, max_plot_columns=100)


#%% 6a — Manual corrections
########################################################

# Patient 248 T2: pain_daytime was entered as "22" — confirmed typo, should be 2
mask_248 = (df_cl_clean['Patient'] == 248) & (df_cl_clean['Timepoint'] == 2)
if mask_248.sum() > 0:
    df_cl_clean.loc[mask_248, 'pain_daytime'] = '2'
    print("Manual correction: Patient 248 T2 pain_daytime set to '2' (was '22')")
else:
    print("Warning: Patient 248 T2 not found — correction skipped")
# Add further manual corrections here if needed


#%% 6b — Parse/transform columns
########################################################

# 1 — diagnosis
print("\n=== diagnosis (before) ===")
print(df_cl_clean['diagnosis'].value_counts(dropna=False).to_string())
df_cl_clean['diagnosis'] = standardize_diagnosis(df_cl_clean['diagnosis'])
print("\n=== diagnosis (after) ===")
print(df_cl_clean['diagnosis'].value_counts().to_dict())

# 2 — target_volume: standardize + combine into "BodyPart Side"
print("\n=== target_volume (before) ===")
print(df_cl_clean['target_volume'].value_counts(dropna=False).head(20).to_string())
df_cl_clean['target_volume'], df_cl_clean['target_side'] = standardize_target_volume(
    df_cl_clean['target_volume'])
df_cl_clean = move_column_after(df_cl_clean, 'target_side', 'target_volume')
df_cl_clean['target_volume'] = df_cl_clean.apply(
    lambda r: f"{r['target_volume']} {r['target_side']}".strip()
              if pd.notna(r['target_volume']) and pd.notna(r['target_side']) and r['target_side'] != ''
              else r['target_volume'],
    axis=1
)
df_cl_clean = df_cl_clean.drop(columns=['target_side'])
print("\n=== target_volume (after) ===")
print(df_cl_clean['target_volume'].value_counts().to_dict())

# 3 — pain_points
print("\n=== pain_points (before) ===")
print(df_cl_clean['pain_points'].value_counts(dropna=False).head(20).to_string())
df_cl_clean['pain_points'] = standardize_pain_points(df_cl_clean['pain_points'])
print("\n=== pain_points (after) ===")
print(df_cl_clean['pain_points'].value_counts().head(20).to_dict())

# 4 — filter → filter_mm + filter_material
print("\n=== filter (before) ===")
print(df_cl_clean['filter'].value_counts(dropna=False).to_string())
df_cl_clean = split_filter_column(df_cl_clean)
print("\n=== filter (after) ===")
print(f"filter_mm    : {sorted(df_cl_clean['filter_mm'].dropna().unique())}")
print(f"filter_material: {df_cl_clean['filter_material'].value_counts().to_dict()}")

# 5 — cumulative_dose
print("\n=== cumulative_dose (before) ===")
print(df_cl_clean['cumulative_dose'].value_counts(dropna=False).to_string())
df_cl_clean['cumulative_dose'] = pd.to_numeric(
    df_cl_clean['cumulative_dose'].apply(parse_cumulative_dose),
    errors='coerce'
)
print("\n=== cumulative_dose (after) ===")
print(sorted(df_cl_clean['cumulative_dose'].dropna().unique()))

# 6 — gender: 'w' → 'f'
print("\n=== gender (before) ===")
print(df_cl_clean['gender'].value_counts(dropna=False).to_string())
df_cl_clean['gender'] = df_cl_clean['gender'].replace('w', 'f')
print("\n=== gender (after) ===")
print(df_cl_clean['gender'].value_counts().to_dict())

# 7 — overweight_bmi → overweight + bmi
print("\n=== overweight_bmi (before) ===")
print(df_cl_clean['overweight_bmi'].value_counts(dropna=False).head(20).to_string())
df_cl_clean = split_bmi_column(df_cl_clean)
print("\n=== overweight / bmi (after) ===")
print(f"overweight: {df_cl_clean['overweight'].value_counts().to_dict()}")
print(f"bmi: range {df_cl_clean['bmi'].min():.1f}–{df_cl_clean['bmi'].max():.1f}, "
      f"{df_cl_clean['bmi'].isna().sum()} missing")

# 8 — symptoms_months
print("\n=== symptoms_months (before) ===")
print(df_cl_clean['symptoms_months'].value_counts(dropna=False).head(20).to_string())
df_cl_clean['symptoms_months'] = pd.to_numeric(
    parse_symptoms_duration(df_cl_clean['symptoms_months'], df_cl_clean['date']),
    errors='coerce'
)
print("\n=== symptoms_months (after) ===")
print(f"range {df_cl_clean['symptoms_months'].min():.0f}–{df_cl_clean['symptoms_months'].max():.0f} months, "
      f"{df_cl_clean['symptoms_months'].isna().sum()} missing")

# 9 — previous_therapy → binary columns
print("\n=== previous_therapy (before) ===")
print(df_cl_clean['previous_therapy'].value_counts(dropna=False).head(20).to_string())
df_cl_clean = encode_therapy_columns(df_cl_clean)
therapy_cols = [f'previous_therapy_{i}' for i in range(1, 8)]
print("\n=== previous_therapy (after: binary columns) ===")
print(df_cl_clean[therapy_cols].sum().to_dict())

# 10 — response → response_category + response_percent
df_cl_clean = standardize_response(df_cl_clean, response_col='response')
df_cl_clean = move_column_after(df_cl_clean, 'response_category', 'response')
df_cl_clean = move_column_after(df_cl_clean, 'response_percent', 'response_category')

# 11 — Ordinal questionnaire columns
ordinal_cols = ['pain_under_load', 'pain_at_rest', 'pain_daytime', 'pain_night', 'morning_stiffness']
print("\n=== Ordinal questionnaire columns (before extraction) ===")
for col in ordinal_cols:
    if col in df_cl_clean.columns:
        uniq = df_cl_clean[col].dropna().unique()
        print(f"\n  {col} ({len(uniq)} unique):")
        for v in sorted(uniq, key=lambda x: str(x)):
            print(f"    {repr(v)}")

print("\n=== Extracting ordinal values ===")
for col in ordinal_cols:
    if col in df_cl_clean.columns:
        df_cl_clean[col] = extract_numeric(df_cl_clean[col])
        print(f"  {col}: unique after = {sorted(df_cl_clean[col].dropna().unique())}")

# 12 — pain_scale (continuous)
print("\n=== pain_scale (before extraction) ===")
uniq_ps = df_cl_clean['pain_scale'].dropna().unique()
print(f"  pain_scale ({len(uniq_ps)} unique):")
for v in sorted(uniq_ps, key=lambda x: str(x)):
    print(f"    {repr(v)}")

df_cl_clean['pain_scale'] = extract_continuous(df_cl_clean['pain_scale'])
print("\n=== pain_scale (after extraction) ===")
uniq_after = sorted(df_cl_clean['pain_scale'].dropna().unique())
print(f"  pain_scale ({len(uniq_after)} unique): {uniq_after}")


#%% 7 — Replace missing markers
########################################################

print("\n=== Replacing null markers ('kA' and 'nD' variants) ===")
replace_missing_markers(df_cl_clean, skip_cols=["Patient", "Timepoint"], verbose=True)

# Safety check: Patient and Timepoint must not have NaN
for id_col in ['Patient', 'Timepoint']:
    nan_count = df_cl_clean[id_col].isna().sum()
    if nan_count > 0:
        print(f"Warning: {nan_count} NaN in {id_col} — will be dropped in step 8")
    else:
        print(f"OK: {id_col} has no NaN values")


#%% 8 — Dtype conversion + visualization copy
########################################################

categorical_cols = [
    'gender', 'overweight', 'pain_points', 'diagnosis',
    'target_volume', 'filter_material',
    'response', 'response_category'       # response_percent stays float
]

# Coerce ids first; drop any rows where they cannot be parsed
df_cl_clean['Patient']   = pd.to_numeric(df_cl_clean['Patient'],   errors='coerce')
df_cl_clean['Timepoint'] = pd.to_numeric(df_cl_clean['Timepoint'], errors='coerce')

n_before    = len(df_cl_clean)
_bad_rows   = df_cl_clean[df_cl_clean[['Patient', 'Timepoint']].isna().any(axis=1)]
if len(_bad_rows) > 0:
    print(f"Rows with unparseable Patient or Timepoint (about to be dropped):")
    print(_bad_rows[['Patient', 'Timepoint']].to_string())
df_cl_clean = df_cl_clean.dropna(subset=['Patient', 'Timepoint'])
n_dropped   = n_before - len(df_cl_clean)
if n_dropped > 0:
    print(f"Dropped {n_dropped} rows with unparseable Patient or Timepoint values")

df_cl_clean['Patient']   = df_cl_clean['Patient'].astype('int64')
df_cl_clean['Timepoint'] = df_cl_clean['Timepoint'].astype('int64')

if 'measurement_timepoint' in df_cl_clean.columns:
    df_cl_clean['measurement_timepoint'] = df_cl_clean['measurement_timepoint'].astype(str)
if 'date' in df_cl_clean.columns:
    df_cl_clean['date'] = pd.to_datetime(df_cl_clean['date'], errors='coerce')

for col in categorical_cols:
    if col in df_cl_clean.columns:
        df_cl_clean[col] = df_cl_clean[col].astype('category')

exclude_for_float = set(categorical_cols) | {'Patient', 'Timepoint', 'measurement_timepoint', 'date'}
cols_to_float = [c for c in df_cl_clean.columns if c not in exclude_for_float]
df_cl_clean[cols_to_float] = (
    df_cl_clean[cols_to_float]
    .apply(lambda s: pd.to_numeric(s, errors='coerce'))
    .astype('float64')
)

print("\n=== Dtype summary (clinical) ===")
print(df_cl_clean.dtypes.value_counts())
print(f"Shape: {df_cl_clean.shape}, Patients: {df_cl_clean['Patient'].nunique()}")

TableReport(df_cl_clean, max_plot_columns=100)

# Drop columns with >25% missing values — calculated across all timepoints (T1–T5)
# (same strategy as immunological dataset)
na_frac_cl = df_cl_clean.isna().mean()
cl_cols_to_drop = na_frac_cl[na_frac_cl > 0.25].index.tolist()
# Never drop patient/timepoint identifiers or the primary target
cl_cols_to_drop = [c for c in cl_cols_to_drop
                   if c not in ['Patient', 'Timepoint', 'pain_scale']]
print(f"\nDropping {len(cl_cols_to_drop)} clinical columns with >25% missing: {cl_cols_to_drop}")
df_cl_clean = df_cl_clean.drop(columns=cl_cols_to_drop)
print(f"Shape after NaN drop: {df_cl_clean.shape}")

# Visualization copy: all timepoints, all patients, after dtype conversion + >25% NaN drop
df_cl_vis = df_cl_clean.copy()


#%%##### VISUALIZATION (placeholder) ###########################################
# TODO: Use df_cl_vis for EDA (all timepoints, all patients, not imputed)
# Use df_cl_imputed (miceforest) for FAMD / MFA (requires complete data)
#
# Planned visualisations:
#   - Distribution plots per feature / per timepoint
#   - PCA (prince library — handles NaN via imputation internally, or
#           use sklearn PCA after imputing a viz copy)
#   - Correlation matrix (phik.phik_matrix — handles mixed types and NaN)
#   - NA heatmap
################################################################################


#%% ########## MFA Score Plots coloured by clinical variables ##########
# Re-uses immunological MFA scores (_scores_mfa, _patient_ids_mfa, _exp_mfa).
# Clinical metadata is aggregated across T1+T2+T3 per patient and matched by
# Patient ID. Patients with no clinical record are excluded from that plot.

# ── Aggregate clinical metadata across T1+T2+T3 per patient ─────────────────
# Stable categorical variables (gender, diagnosis, response_category):
#   sort by Timepoint so groupby.first() picks T1 when available, else T2/T3.
# pain_at_rest (ordinal, changes over time): mean across available timepoints,
#   then rounded to nearest integer so legend labels show as "1", "2", etc.
_cl_t123 = df_cl_vis[df_cl_vis["Timepoint"].isin([1, 2, 3])].copy()

_cat_meta = (
    _cl_t123.sort_values("Timepoint")
    .groupby("Patient")[["gender", "diagnosis", "response_category"]]
    .first()
)
_ord_meta = (
    _cl_t123.sort_values("Timepoint")
    .groupby("Patient")[["pain_at_rest", "pain_scale"]]
    .first()  # T1 baseline value; falls back to T2/T3 only if T1 is missing
)
_cl_meta = _cat_meta.join(_ord_meta, how="outer")

# Specs: (column, title, kind, palette, vmin, vmax)
#   kind="cat"  → discrete colors + legend
#   kind="cont" → continuous colormap + colorbar
_mfa_color_specs = [
    ("gender",            "Gender",            "cat",  "mako",  None, None),
    ("diagnosis",         "Diagnosis",         "cat",  "tab20", None, None),
    ("response_category", "Response Category", "cat",  "mako", None, None),
    ("pain_at_rest",      "Pain at Rest",      "cont", "mako_r",  1,    5   ),
    ("pain_scale",        "Pain Scale (T1–T3 mean)", "cont", "mako_r", 0, 10),
]

print("MFA score plots coloured by clinical variables:")
for _col, _title, _kind, _pal, _vmin, _vmax in _mfa_color_specs:

    # Map each MFA patient to its aggregated clinical value (np.nan if not found)
    _vals = np.array(
        [_cl_meta.loc[p, _col] if p in _cl_meta.index else np.nan
         for p in _patient_ids_mfa],
        dtype=object
    )
    _mask_nan = pd.isna(_vals)

    # Exclude patients with no value for this variable
    _keep     = ~_mask_nan
    _sc_plt   = _scores_mfa[_keep, :]
    _v_plt    = _vals[_keep]
    _pids_plt = np.array(_patient_ids_mfa)[_keep]
    print(f"  {_title}: {_keep.sum()} patients plotted ({_mask_nan.sum()} excluded — no clinical data)")

    fig, ax = plt.subplots(figsize=(9, 7))

    if _kind == "cat":
        # ── Discrete: one scatter series per category ─────────────────────
        _cats       = sorted({str(v) for v in _v_plt})
        _cat_colors = dict(zip(_cats, sns.color_palette(_pal, len(_cats))))
        for _cat in _cats:
            _mask_c = np.array([str(v) == _cat for v in _v_plt])
            ax.scatter(_sc_plt[_mask_c, 0], _sc_plt[_mask_c, 1],
                       c=[_cat_colors[_cat]], s=40, zorder=3,
                       edgecolors="white", linewidth=0.4, alpha=0.85,
                       label=f"{_cat} (n={_mask_c.sum()})")
        ax.legend(title=_title, loc="best", fontsize=8)
        # Label top 20 patients furthest from origin
        _dist_cl  = np.sqrt(_sc_plt[:, 0]**2 + _sc_plt[:, 1]**2)
        _top20_cl = np.argsort(_dist_cl)[::-1][:20]
        _texts_cl = [ax.text(_sc_plt[_i, 0], _sc_plt[_i, 1],
                             str(_pids_plt[_i]),
                             fontsize=7, fontweight="bold", color="black", zorder=5)
                     for _i in _top20_cl]
        _adj(_texts_cl, ax=ax, expand=(1.5, 1.5),
             arrowprops=dict(arrowstyle="-", color="grey", lw=0.5))

    else:
        # ── Continuous: colormap + colorbar ───────────────────────────────
        _num_v = pd.to_numeric(pd.Series(_v_plt), errors="coerce").values
        _valid = ~np.isnan(_num_v)
        sc = ax.scatter(_sc_plt[_valid, 0], _sc_plt[_valid, 1],
                        c=_num_v[_valid], cmap=_pal,
                        vmin=_vmin, vmax=_vmax,
                        s=40, zorder=3,
                        edgecolors="white", linewidth=0.4, alpha=0.85)
        plt.colorbar(sc, ax=ax, label=_title)
        # Label top 20 patients furthest from origin
        _pids_cont = _pids_plt[_valid]
        _sc_cont   = _sc_plt[_valid]
        _dist_cl   = np.sqrt(_sc_cont[:, 0]**2 + _sc_cont[:, 1]**2)
        _top20_cl  = np.argsort(_dist_cl)[::-1][:20]
        _texts_cl  = [ax.text(_sc_cont[_i, 0], _sc_cont[_i, 1],
                              str(_pids_cont[_i]),
                              fontsize=7, fontweight="bold", color="black", zorder=5)
                      for _i in _top20_cl]
        _adj(_texts_cl, ax=ax, expand=(1.5, 1.5),
             arrowprops=dict(arrowstyle="-", color="grey", lw=0.5))

    ax.axhline(0, color="grey", lw=0.5, linestyle="--")
    ax.axvline(0, color="grey", lw=0.5, linestyle="--")
    ax.set_xlabel(f"Dim1 ({_exp_mfa[0]:.1f}% variance)")
    ax.set_ylabel(f"Dim2 ({_exp_mfa[1]:.1f}% variance)")
    ax.set_title(
        f"MFA Score Plot — Immunological T1+T2+T3\n"
        f"Coloured by {_title}"
    )
    plt.tight_layout()
    plt.show()














#%% 10 — Modeling copy placeholder (df_cl_mod)
########################################################
# df_cl_mod will be created after clinical PyOD outlier detection:
#   df_cl_mod = df_cl_vis with manually confirmed outlier patients removed
# Target variables (pain_reduction_pct, pain_scale_t2) are then merged into df_cl_mod.


#%% 11 — Target variables + distributions
########################################################
# Note: pain_targets is derived from df_cl_vis here as a temporary stand-in.
# Once clinical PyOD is complete and df_cl_mod (outlier-removed) is created,
# this block should be moved after df_cl_mod creation and use df_cl_mod instead.

# Extract T1 and T2 pain_scale per patient
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

# Inner join: only patients with BOTH T1 and T2 pain_scale values
pain_targets = pain_t1.merge(pain_t2, on='Patient', how='inner')

# Raw point reduction (reference only — not modeling target)
pain_targets['pain_scale_reduction'] = pain_targets['pain_scale_t1'] - pain_targets['pain_scale_t2']

# Percent reduction relative to T1 — primary modeling target
# T1 = 0 is undefined: raise immediately rather than silently produce NaN/inf
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

print(f"\nPatients with T1 + T2 pain_scale (usable for regression): {len(pain_targets)}")
print(f"pain_scale_t2 range:      {pain_targets['pain_scale_t2'].min():.1f} – {pain_targets['pain_scale_t2'].max():.1f}")
print(f"pain_scale_reduction:     {pain_targets['pain_scale_reduction'].min():.1f} – {pain_targets['pain_scale_reduction'].max():.1f} pts")
print(f"pain_reduction_pct range: {pain_targets['pain_reduction_pct'].min():.1f} – {pain_targets['pain_reduction_pct'].max():.1f} %")
print(f"  (positive = improvement, negative = worsening)")
print(f"pain_reduction_pct stats:\n{pain_targets['pain_reduction_pct'].describe()}")

# Note: merging targets into df_cl_mod will happen after clinical PyOD + outlier removal.

# Distribution of regression targets (one row per patient)
fig, axes = plt.subplots(1, 3, figsize=(18, 4))
targets_per_patient = pain_targets
colors = sns.color_palette('mako', 5)

axes[0].hist(targets_per_patient['pain_scale_t2'].dropna(), bins=20, color=colors[1])
axes[0].set_title('pain_scale_t2 (T2 pain level)')
axes[0].set_xlabel('Pain Scale (0–10)')
axes[0].set_ylabel('Number of Patients')

axes[1].hist(targets_per_patient['pain_scale_reduction'].dropna(), bins=20, color=colors[2])
axes[1].set_title('pain_scale_reduction (T1 − T2 pts, reference)')
axes[1].set_xlabel('Point Reduction (positive = improvement)')
axes[1].axvline(0, color='white', linestyle='--', linewidth=1, label='No change')
axes[1].legend()

axes[2].hist(targets_per_patient['pain_reduction_pct'].dropna(), bins=20, color=colors[3])
axes[2].set_title('pain_reduction_pct (% relative to T1, primary target)')
axes[2].set_xlabel('Pain Reduction (%)')
axes[2].axvline(0, color='white', linestyle='--', linewidth=1, label='No change')
axes[2].legend()

plt.suptitle('Distribution of Regression Targets', fontweight='bold')
plt.tight_layout()
plt.show()

# Distribution of pain_scale by timepoint (T1–T5, 2-row layout)
timepoints = [1, 2, 3, 4, 5]
colors_tp = sns.color_palette('mako', 5)
fig = plt.figure(figsize=(15, 8))

top_axes = [fig.add_subplot(2, 3, i + 1) for i in range(3)]
bot_axes = [fig.add_subplot(2, 3, 5), fig.add_subplot(2, 3, 6)]
plot_axes = top_axes + bot_axes

for ax, tp, color in zip(plot_axes, timepoints, colors_tp):
    data = df_cl_mod.loc[df_cl_mod['Timepoint'] == tp, 'pain_scale'].dropna()
    ax.hist(data, bins=15, color=color, edgecolor='white')
    ax.set_title(f'T{tp}  (n={len(data)})')
    ax.set_xlabel('Pain Scale (0–10)')
    ax.set_ylabel('Count')
    if len(data) > 0:
        ax.axvline(data.median(), color='white', linestyle='--', linewidth=1.5,
                   label=f'Median {data.median():.1f}')
        ax.legend(fontsize=9)

plt.suptitle('Distribution of pain_scale by Timepoint', fontweight='bold')
plt.tight_layout()
plt.show()

print(df_cl_mod.groupby('Timepoint')['pain_scale'].describe())
TableReport(df_cl_mod, max_plot_columns=100)


#%%##### BASELINE CATBOOST #####################################################

#%% 12 — Prepare baseline datasets + regression helpers
########################################################

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold
from catboost import CatBoostRegressor, Pool
import shap


def regression_metrics(y_true, y_pred):
    """Return dict of MAE, MSE, RMSE, R² for a regression prediction."""
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, y_pred)
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}


def run_catboost_regressor(df_model, target_col, name,
                           n_splits=5, n_repeats=5, random_state=42):
    """5-fold × 5-repeat RepeatedKFold CatBoostRegressor (25 fits). No hyperparameter tuning.
    Returns (results_df, last_trained_model, X_features, y_pred_series).

    Automatically excluded from features:
      - ID columns  : Patient, Timepoint, Date, date, measurement_timepoint
      - Leaky cols  : any column whose name contains 'response', 'improvement_percent',
                      'pain_scale', or 'pain_reduction_pct'
    """
    always_exclude = ['Patient', 'Timepoint', 'Date', 'date', 'measurement_timepoint']
    leaky_patterns = ['response', 'improvement_percent', 'pain_scale', 'pain_reduction_pct']
    exclude = set(always_exclude + [target_col])
    for col in df_model.columns:
        if any(pat in col.lower() for pat in leaky_patterns):
            exclude.add(col)

    feature_cols = [c for c in df_model.columns if c not in exclude]
    X = df_model[feature_cols].copy()
    y = df_model[target_col].copy()

    valid = y.notna()
    X, y = X[valid].reset_index(drop=True), y[valid].reset_index(drop=True)

    for col in X.select_dtypes(include=['category', 'object']).columns:
        X[col] = X[col].astype(str)
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()

    print(f"\n{'='*65}")
    print(f"  CatBoost Regressor Baseline — {name}")
    print(f"  Target : {target_col}")
    print(f"  Samples: {len(X)},  Features: {len(feature_cols)}")
    print(f"  CV     : {n_splits}-fold × {n_repeats} repeats = {n_splits * n_repeats} fits")
    print(f"{'='*65}")

    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    fold_results = []
    y_pred = pd.Series(np.nan, index=range(len(X)), dtype='float64')

    for fold, (train_idx, test_idx) in enumerate(rkf.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = CatBoostRegressor(iterations=300, random_seed=random_state, verbose=0)
        model.fit(
            Pool(X_train, y_train, cat_features=cat_cols),
            eval_set=Pool(X_test, y_test, cat_features=cat_cols),
            use_best_model=False
        )

        preds = model.predict(X_test)
        y_pred.iloc[test_idx] = preds

        m = regression_metrics(y_test, preds)
        fold_results.append({'Fold': fold + 1, **m})
        print(f"  Fold {fold+1:>2}: MAE={m['MAE']:.3f}  MSE={m['MSE']:.3f}  "
              f"RMSE={m['RMSE']:.3f}  R²={m['R2']:.3f}")

    results_df = pd.DataFrame(fold_results)
    metric_cols = ['MAE', 'MSE', 'RMSE', 'R2']
    mean_row = {'Fold': 'Mean', **{m: results_df[m].mean() for m in metric_cols}}
    std_row  = {'Fold': 'Std',  **{m: results_df[m].std()  for m in metric_cols}}
    results_df = pd.concat(
        [results_df, pd.DataFrame([mean_row, std_row])], ignore_index=True)

    print(f"\n  Summary ({n_splits}x{n_repeats} CV):")
    for m in metric_cols:
        mv = results_df.loc[results_df['Fold'] == 'Mean', m].iloc[0]
        sv = results_df.loc[results_df['Fold'] == 'Std',  m].iloc[0]
        print(f"    {m:<5}: {mv:.3f} ± {sv:.4f}")

    return results_df, model, X, y_pred


def plot_shap_regressor(model, X, name):
    """SHAP bar + beeswarm plots for a fitted CatBoostRegressor."""
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
            row[m] = f"{fold_rows[m].mean():.3f} ± {fold_rows[m].std():.4f}"
        rows.append(row)
    summary = pd.DataFrame(rows)
    print(f"\n{'='*70}")
    print(f"  CATBOOST BASELINE SUMMARY — Target: {target_col}")
    print(f"{'='*70}")
    print(summary.to_string(index=False))
    return summary


# Patients eligible for baseline modeling (have both T1 and T2 pain_scale)
model_patients = set(pain_targets['Patient'].values)

# Immunological T1 baseline: from df_im_vis (pre-imputation), filtered to model_patients
df_im_raw_t1 = (
    df_im_vis[
        (df_im_vis['Timepoint'] == 1) &
        (df_im_vis['Patient'].isin(model_patients))
    ]
    .copy()
    .reset_index(drop=True)
)
df_im_raw_t1 = df_im_raw_t1.merge(
    pain_targets[['Patient', 'pain_scale_t2', 'pain_reduction_pct']],
    on='Patient', how='left'
)

# Clinical T1 baseline: from df_cl_bcat (English names, raw unparsed values)
df_cl_bcat_t1 = (
    df_cl_bcat[
        (df_cl_bcat['Timepoint'] == 1) &
        (df_cl_bcat['Patient'].isin(model_patients))
    ]
    .copy()
    .reset_index(drop=True)
)
df_cl_bcat_t1 = df_cl_bcat_t1.merge(
    pain_targets[['Patient', 'pain_scale_t2', 'pain_reduction_pct']],
    on='Patient', how='left'
)

# Combined T1 baseline: inner join on Patient
df_bcat_combined_t1 = df_im_raw_t1.merge(
    df_cl_bcat_t1.drop(columns=['Timepoint'], errors='ignore'),
    on='Patient', how='inner',
    suffixes=('_im', '_cl')
)

print(f"\nBaseline T1 datasets:")
print(f"  Immunological : {df_im_raw_t1.shape},  patients: {df_im_raw_t1['Patient'].nunique()}")
print(f"  Clinical      : {df_cl_bcat_t1.shape},  patients: {df_cl_bcat_t1['Patient'].nunique()}")
print(f"  Combined      : {df_bcat_combined_t1.shape}, patients: {df_bcat_combined_t1['Patient'].nunique()}")


#%% 12b — Run baseline CatBoost
########################################################

print("\n" + "="*70)
print("  CATBOOST BASELINE REGRESSOR — Target: pain_reduction_pct")
print("="*70)

target = 'pain_reduction_pct'

res_im_red, model_im_red, X_im_red, ypred_im_red = run_catboost_regressor(
    df_im_raw_t1, target, "Immunological (raw T1)")

res_cl_red, model_cl_red, X_cl_red, ypred_cl_red = run_catboost_regressor(
    df_cl_bcat_t1, target, "Clinical (reduced T1)")

res_comb_red, model_comb_red, X_comb_red, ypred_comb_red = run_catboost_regressor(
    df_bcat_combined_t1, target, "Combined (reduced T1)")

summary_cb_red = print_regression_summary(
    {"Immunological": res_im_red, "Clinical": res_cl_red, "Combined": res_comb_red},
    target
)

#%% SHAP — pain_reduction_pct

shap_im_red   = plot_shap_regressor(model_im_red,   X_im_red,   "Immunological — pain_reduction_pct")
shap_cl_red   = plot_shap_regressor(model_cl_red,   X_cl_red,   "Clinical — pain_reduction_pct")
shap_comb_red = plot_shap_regressor(model_comb_red, X_comb_red, "Combined — pain_reduction_pct")


print("\n" + "="*70)
print("  CATBOOST BASELINE REGRESSOR — Target: pain_scale_t2")
print("="*70)

target = 'pain_scale_t2'

res_im_t2, model_im_t2, X_im_t2, ypred_im_t2 = run_catboost_regressor(
    df_im_raw_t1, target, "Immunological (raw T1)")

res_cl_t2, model_cl_t2, X_cl_t2, ypred_cl_t2 = run_catboost_regressor(
    df_cl_bcat_t1, target, "Clinical (reduced T1)")

res_comb_t2, model_comb_t2, X_comb_t2, ypred_comb_t2 = run_catboost_regressor(
    df_bcat_combined_t1, target, "Combined (reduced T1)")

summary_cb_t2 = print_regression_summary(
    {"Immunological": res_im_t2, "Clinical": res_cl_t2, "Combined": res_comb_t2},
    target
)

#%% SHAP — pain_scale_t2

shap_im_t2   = plot_shap_regressor(model_im_t2,   X_im_t2,   "Immunological — pain_scale_t2")
shap_cl_t2   = plot_shap_regressor(model_cl_t2,   X_cl_t2,   "Clinical — pain_scale_t2")
shap_comb_t2 = plot_shap_regressor(model_comb_t2, X_comb_t2, "Combined — pain_scale_t2")




#%%##### ADVANCED CATBOOST (placeholder) #######################################

#%% 13 — Prepare combined clean dataset
# df_im_t1: T1 rows from df_im_imputed (or df_im_vis?), filtered to model_patients
# df_cl_mod_t1: T1 rows from df_cl_mod, filtered to model_patients
# df_combined_mod_t1: inner join on Patient → combined clean+transformed, NOT imputed
# TableReport + print patient count
#
# Nested CV structure:
#   Outer: 5-fold KFold (evaluation)
#   Inner: 4-fold KFold (hyperparameter tuning via Optuna)
#   Imputation: fit on outer train fold only (no leakage)
#   Objective: minimize RMSE
#   Report: avg MAE, MSE, RMSE, R² across outer folds
#   SHAP: fit final model on full training data, explain test predictions





#%%##### ADVANCED HGB (placeholder) ############################################

#%% 14 — HistGradientBoosting advanced model
# Same nested CV structure as section 13 above
# Handles numeric NaN natively → simpler imputation strategy (categoricals only)
# OrdinalEncoder for categorical features inside pipeline
# Objective: minimize RMSE
# Report: avg MAE, MSE, RMSE, R² across outer folds
# Feature importance: HGB built-in + SHAP

# %%
