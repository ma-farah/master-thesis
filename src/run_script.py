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

# copy of reduced raw dataset for baseline modeling
df_im_reduced = df_im.copy()

TableReport(df_im_reduced, max_plot_columns=100)


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


# Removing columns with more than 25% missing values:
na_frac = df_im.isna().mean()
cols_to_drop = na_frac[na_frac > 0.25].index.tolist()
df_im = df_im.drop(columns=cols_to_drop).copy()
print('Dropped columns:', cols_to_drop)

""" 
Dropped columns: ['TC_CD25hi', 'B_CD25hi', 'Eos_HLADR+', 'Mo2_HLADRhi', 'TC_HLADRhi', 
'NK_HLADRhi', 'Eos_CD69+', 'Bas_CD69+', 'Mo_CD69+', 'B_CD69+', 'DC_CD69+', 
'TH naive_PD1+', 'TH eff_PD1+', 'TC naive_PD1+'
"""

# New Tablereport
TableReport(df_im, max_plot_columns=138)


#%%########### Imputing missing values using miceforest and median

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



#%%############# RV / RV2 analysis across timepoints ##########################

#NB! Patient ID 83 has two timepoint 4 measurements (take average)
# Needs to be the same shape in order to to RV2 analysis.
# patient ID 137 have two t4 and two t3 measurements.

# new dataset given, lets check duplicates again:

# print duplicated timepoint-measurements:
df_im_imputed[
    df_im_imputed.duplicated(subset=["Patient", "Timepoint"], keep=False)
].sort_values(["Patient", "Timepoint"])
# no duplicates


# dataframes for each time-point t1, t2, t3, t4, t5
df_t1 = df_im_imputed[df_im_imputed["Timepoint"] == 1]
df_t2 = df_im_imputed[df_im_imputed["Timepoint"] == 2]
df_t3 = df_im_imputed[df_im_imputed["Timepoint"] == 3]
df_t4 = df_im_imputed[df_im_imputed["Timepoint"] == 4]
df_t5 = df_im_imputed[df_im_imputed["Timepoint"] == 5]

# checking size of each dataframe:
print("T1 shape:", df_t1.shape)
print("T2 shape:", df_t2.shape)
print("T3 shape:", df_t3.shape)
print("T4 shape:", df_t4.shape)
print("T5 shape:", df_t5.shape)

# storing dataframes in a dictionary for easy access
dfs = {
    1: df_t1,
    2: df_t2,
    3: df_t3,
    4: df_t4,
    5: df_t5
}

id_cols = ["Patient", "Timepoint", "Date"]
timepoints = [1, 2, 3, 4, 5]


# find the common patients between two timepoints:
def common_patients(df_a, df_b, id_col="Patient"):
    common = np.intersect1d(df_a[id_col], df_b[id_col])
    # first dataframe
    A = (
        df_a[df_a[id_col].isin(common)]
        .sort_values(id_col)
        .reset_index(drop=True)
    )
    # second dataframe
    B = (
        df_b[df_b[id_col].isin(common)]
        .sort_values(id_col)
        .reset_index(drop=True)
    )

    return A, B


# calculating rv2 for all combinations of timepoints t1 to t5 (10 combinations)
n = len(timepoints)
rv2_matrix = np.zeros((n, n))
n_common = np.zeros((n, n), dtype=int)


for i, ti in enumerate(timepoints):
    for j, tj in enumerate(timepoints):

        if i == j:     # if we are comparing the same dataframe with itself
            rv2_matrix[i, j] = 1.0       # correlation = 1 
            n_common[i, j] = dfs[ti].shape[0]  
           
        else:
            A, B = common_patients(dfs[ti], dfs[tj])   # getting common patients
            n_common[i, j] = A.shape[0]              # number of common patients

            X = ho.standardise(                     # standardising data before calculating RV2
                A.drop(columns=id_cols).values,     # dropping id columns
                mode=0                              # column-wise standardisation
            )

            Y = ho.standardise(
                B.drop(columns=id_cols).values,
                mode=0
            )

            rv2 = ho.RV2coeff([X, Y])[0, 1]     # calculating RV2 coefficient
            rv2_matrix[i, j] = rv2              # storing rv2 value



# dataframe of number of common patients inbetween comparisons:
n_common_df = pd.DataFrame(
    n_common,
    index=[f"T{t}" for t in timepoints],        # row labels
    columns=[f"T{t}" for t in timepoints]      # column labels
)
n_common_df.style.set_caption(
    "Number of common patients between timepoint comparisons")


# convert results to dataframe before plotting       
rv2_df = pd.DataFrame(
    rv2_matrix,
    index=[f"T{t}" for t in timepoints],
    columns=[f"T{t}" for t in timepoints]
)


# plotting heatmap of rv2 values
plt.figure(figsize=(8, 6))
sns.heatmap(
    rv2_df,
    annot=True,
    fmt=".2f",
    cmap="viridis",
    vmin=-1,
    vmax=1,
    square=True
)

plt.title("RV2 similarity across timepoints T1 to T5")
plt.tight_layout()
plt.show()


"""
201 patients have measurements in both T1 and T2
130 patients have both T1 and T3
123 patients have both T1 and T4
76 patients have both T1 and T5
"""

# check rv2 for seperate datatsets: what we would like to see change vs not.
# lekocytes stable - mDC downregulate, m1,m3 and m3?


#%%########## Correlation analysis between features for immunological dataset

# prøv med pearsons coeffiseint!

import phik

# phik correlation matrix for immunological dataset
df_features = df_im_imputed.drop(columns=id_cols)
phik_matrix = df_features.phik_matrix(interval_cols=None)

# Extract top correlated pairs
# Get upper triangle of correlation matrix (avoid duplicates and diagonal)
upper_triangle = np.triu(phik_matrix, k=1)
upper_triangle_df = pd.DataFrame(
    upper_triangle,
    index=phik_matrix.index,
    columns=phik_matrix.columns
)

# Convert to long format and sort by correlation
correlations = []
for i in range(len(upper_triangle_df)):
    for j in range(i+1, len(upper_triangle_df.columns)):
        correlations.append({
            'Feature_1': upper_triangle_df.index[i],
            'Feature_2': upper_triangle_df.columns[j],
            'Phik_Correlation': upper_triangle_df.iloc[i, j]
        })

correlations_df = pd.DataFrame(correlations)
correlations_df = correlations_df.sort_values('Phik_Correlation', ascending=False)

# Display top 20 correlated pairs
print("\nTop 40 Most Correlated Feature Pairs:")
print("="*80)
print(correlations_df.head(40).to_string(index=False))
print("\n")

# Parent categories such as T-cells, Monocytes, Eosinophils, Basophils, B-cells, NK-cells.. have high correlation with their subcategories, which is expected. 
# Does T-cell represent the total amount, and their subtypes are percentages of these? if so - might be redundant?

# Focused heatmap: only features that appear in top 20 correlations
top_features = set()
for _, row in correlations_df.head(30).iterrows():
    top_features.add(row['Feature_1'])
    top_features.add(row['Feature_2'])
top_features = sorted(list(top_features))

print(f"\nNumber of features involved in top 30 correlations: {len(top_features)}")
print("Features:", top_features)

# Create focused heatmap
focused_phik = phik_matrix.loc[top_features, top_features]

plt.figure(figsize=(14, 12))
sns.heatmap(
    focused_phik,
    annot=True,
    fmt='.2f',
    cmap="viridis",
    vmin=0,
    vmax=1,
    square=True,
    cbar_kws={"label": "Phik Correlation Coefficient"},
    xticklabels=top_features,
    yticklabels=top_features
)
plt.title(f'Phik Correlation Matrix - Top {len(top_features)} Most Correlated Features',
          fontsize=14, fontweight='bold')
plt.xticks(rotation=90, fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.show()


#%% ############## PCA analysis immu dataset ########################
# using prince package for pca analysis:

# has automatic scaling

# Pca for timepoint 1 - 5 individually
for t in timepoints:
    df_t = dfs[t]
    # Set Patient ID as index so it shows in the plot
    X_t = df_t.set_index('Patient').drop(columns=['Timepoint', 'Date'])

    pca = ps.PCA(
        n_components=3,
        n_iter=3,
        copy=True,
        engine='sklearn',
        check_input=True,
        random_state=42
    )

    pca = pca.fit(X_t)

    # plotting results of PCA, scatter plot of patients at timepoint t
    chart = pca.plot(
        X_t,
        x_component=0,
        y_component=1,
        show_row_markers=True,
        show_column_markers=False,
        show_row_labels=False,  # Show patient IDs
        show_column_labels=False
    ).properties(
        title=f'PCA of Immunological Data at Timepoint {t}'
    )
    chart.display()

    # Scores (coordinates) for each patient
    # sort after patient id that has longest distance away from the center in the PCA plot, to see if they are outliers in the raw data as well.
    row_coords = pca.transform(X_t)
    row_coords['Distance'] = np.sqrt(row_coords[0]**2 + row_coords[1]**2)
    row_coords = row_coords.sort_values('Distance', ascending=False)
    print(f"Top 10 Patients with highest distance from center for Timepoint {t}:")
    print(row_coords.head(10))
    print("\n")

    # Top contributing variables to PC1:
    loading_scores = pca.column_correlations 
    print(f"Top 10 contributing variables to PC1 for Timepoint {t}:")
    print(loading_scores[0].abs().sort_values(ascending=False).head(10))
    print("\n")

    # Top contributing variables to PC2:
    print(f"Top 10 contributing variables to PC2 for Timepoint {t}:")
    print(loading_scores[1].abs().sort_values(ascending=False).head(10))
    print("\n")
    
       



#%%  Combined PCA for timepoint t1-t2, t2-t3 and t1-t3 

# finding common patients with measuements at t1 and t2:
patients_t12 = set(df_t1["Patient"]) & set(df_t2["Patient"])
# common patients with measuements at t1 and t3:
patients_t13 = set(df_t1["Patient"]) & set(df_t3["Patient"])
# common patients with measuements at t2 and t3:
patients_t23 = set(df_t2["Patient"]) & set(df_t3["Patient"])

# function to sort dataframe by patient ids
def sortdfs(df, patients):
    return (
        df[df["Patient"].isin(patients)]
        .sort_values("Patient")
        .set_index("Patient")
    )

t12 = sortdfs(df_t1, patients_t12)
t22 = sortdfs(df_t2, patients_t12)
t13 = sortdfs(df_t1, patients_t13)
t23 = sortdfs(df_t3, patients_t13)
t32 = sortdfs(df_t2, patients_t23)
t33 = sortdfs(df_t3, patients_t23)

print('Combined PCA:')

# PCA for timepoints combined
for (df_a, df_b, label) in [(t12, t22, "T1 and T2"), (t13, t23, "T1 and T3"), (t32, t33, "T2 and T3")]:
    # Patient is already index, drop only Timepoint and Date
    X_a = df_a.drop(columns=['Timepoint', 'Date'])
    X_b = df_b.drop(columns=['Timepoint', 'Date'])

    X_combined = pd.concat([X_a, X_b], axis=0)

    pca = ps.PCA(
        n_components=3,
        n_iter=3,
        copy=True,
        engine='sklearn',
        check_input=True,
        random_state=42
    )

    pca = pca.fit(X_combined)

    # plotting results of PCA (Patient ID as index but not shown)
    chart = pca.plot(
        X_combined,
        x_component=0,
        y_component=1,
        show_row_markers=True,
        show_column_markers=False,
        show_row_labels=False,  # Don't show patient IDs
        show_column_labels=False
    ).properties(
        title=f'PCA of Immunological Data at {label}'
    )
    chart.display()

    # Scores (coordinates) for each patient
    # sort after patient id that has longest distance away from the center in the PCA plot, to see if they are outliers in the raw data as well.
    row_coords = pca.transform(X_combined)
    row_coords['Distance'] = np.sqrt(row_coords[0]**2 + row_coords[1]**2)
    row_coords = row_coords.sort_values('Distance', ascending=False)
    print(f"Top 10 Patients with highest distance from center for {label}:")
    print(row_coords.head(10))
    print("\n")

    # Top contributing variables to PC1 and PC2
    loading_scores = pca.column_correlations  # property, not method
    print(f"Top contributing variables to PC1 for {label}:")
    print(loading_scores[0].abs().sort_values(ascending=False).head(10))
    print("\n")
    print(f"Top contributing variables to PC2 for {label}:")
    print(loading_scores[1].abs().sort_values(ascending=False).head(10))
    print("\n")


# Trajectory pca plot....


#%%############ MFA for timepoints 1, 2 and 3 combined 

print("MFA for timepoints 1, 2 and 3 combined:")

# finding common patients with measuements at t1, t2 and t3 all together
patients_t123 = (
    set(df_t1["Patient"])
    & set(df_t2["Patient"])
    & set(df_t3["Patient"])
)

# number patients with measuements at time 1, 2 and 3 = 121 patients out of 250
# sorting dataframes by patient ids
df1 = sortdfs(df_t1, patients_t123)
df2 = sortdfs(df_t2, patients_t123)
df3 = sortdfs(df_t3, patients_t123)

# Dropping timepoint and date columns from analysis (Patient is already index)
X1 = df1.drop(columns=['Timepoint', 'Date'])
X2 = df2.drop(columns=['Timepoint', 'Date'])
X3 = df3.drop(columns=['Timepoint', 'Date'])

# need to define group name to get multi-index formated dataset
def group_name(df, group_name):
    df = df.copy()
    df.columns = pd.MultiIndex.from_product(
        [[group_name], df.columns]
    )
    return df

# creating multi-index columns for mfa
X1_m = group_name(X1, "T1")
X2_m = group_name(X2, "T2")
X3_m = group_name(X3, "T3")

dataset = pd.concat([X1_m, X2_m, X3_m], axis=1)
groups = dataset.columns.levels[0].tolist()

# MFA analysis
mfa = ps.MFA(
    n_components=3,
    n_iter=3,
    copy=True,
    engine='sklearn',
    check_input=True,
    random_state=42
)

mfa = mfa.fit(dataset, groups=groups, supplementary_groups=None)

# plotting results of MFA 
mfa.plot(
    dataset,
    show_partial_rows=True
)

# Scores for each patient at different timepoints-groups
# sort ater patient id that has longest distance away from the center in the MFA plot, to see if they are outliers in the raw data as well.
mfa.partial_row_coordinates(dataset)


#%%######### PyOD Ensemble Outlier Detection (Zyran approach) - Immunological Dataset ########

# This section uses the pre-built outlier detection framework from:
# https://gitlab.com/zryan.rz/master_outlier_detection_h23
#
# Pipeline:
#   1. Miceforest-imputed and median-imputed immunological data -> StandardScaler
#   2. GEC (Gaussian Ensemble Comparison): fits all candidate algorithms and
#      selects the 6 most *dissimilar* ones to form a diverse ensemble
#   3. visualiser_OD: fits the 6 selected algorithms, aggregates scores via
#      median probability across algorithms, and produces three plots:
#        - PCA biplot (hoggorm NIPALS PCA)
#        - Scatter: median probability vs. average confidence (marker size =
#          std of confidence, colour = std of probability)
#        - Pairplots of PC1-5 coloured by median probability / confidence
#   Contamination is fixed at 0.1

import sys
import random
from pathlib import Path

# Make pyod_zyran importable from src/
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

# --- Data: median-imputed immunological dataset (already computed above) ---
# Drop ID columns, scale to zero mean / unit variance
X_ens = df_im_imputed[feature_cols].copy()
patient_labels = (
    df_im_imputed["Patient"].astype(str) + "-T" + df_im_imputed["Timepoint"].astype(str)
).tolist()

scaler_ens = StandardScaler()
X_sc = pd.DataFrame(scaler_ens.fit_transform(X_ens), columns=X_ens.columns)

# --- Build candidate algorithm list (matches notebook) ---
contamination = 0.05
random.seed(42)
detector_list_lscp = [IForest_od(n_estimators=n) for n in random.sample(range(5, 200), 10)]

list_OD_classes   = [QMCD, INNE, KNN_od, LOF_od, IForest_od, PCA_od, LODA, HBOS, OCSVM, ECOD_od, COPOD_od]
list_OD_strings   = [cls.__name__ for cls in list_OD_classes]
list_OD_init      = [LSCP(detector_list=detector_list_lscp, contamination=contamination)
                     if cls == LSCP
                     else cls(contamination=contamination)
                     for cls in list_OD_classes]

# --- Step 1: GEC — select 6 most dissimilar algorithms ---
print("Running GEC to select 6 most dissimilar algorithms...")
final_selected_algos, tau_dissimilarity_df = calculate_GEC(
    X_sc.values,
    list_OD_init,
    list_OD_strings,
    percentages=[0.90, 0.98, 1.00]
)
print(f"GEC selected algorithms: {final_selected_algos}")

"""
GEC selected algorithms: ['LODA', 'ECOD', 'COPOD', 'HBOS', 'QMCD', 'IForest']
"""

# --- Step 2: Re-initialise only the selected 6 algorithms ---
algo_class_map     = {cls.__name__: cls for cls in list_OD_classes}
initialized_modules = [
    algo_class_map[name](contamination=contamination)
    for name in final_selected_algos
    if name in algo_class_map
]
print(f"Ensemble: {len(initialized_modules)} algorithms with contamination={contamination}")

# --- Step 3: visualiser_OD ---
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














# %%################ RAW CLINICAL DATASET #############################

# Table report of clinical dataset
print("TableReport of raw clinical dataset:")
TableReport(df_cl, max_plot_columns=138)

# na analysis of clinical dataset
print("Na analysis of clinical dataset:")
na.altair.plot_heatmap(df_cl)


# Raw clinical dataset statistics







#%%############ Cleaning clinical dataset #############################
# Pipeline: Forward-fill -> Exclude patients -> Rename -> Define nulls ->
# Deduplicate categories -> Extract numerics -> Transform columns -> Change dtype 
# -> drop na>25% columns -> visualize -> prepare target before modeling


#%% Clinical preprocessing: helper functions

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

        # Vague / unparseable entries → NaN
        if s.lower() in ('jahre', 'jahre ', 'mehrere', 'mehrere jahre',
                         'mehreren mo.', 'einige jahre', 'einge j.', 'täglich'):
            return pd.NA

        # Full date string: "2023-04-01 00:00:00" → calc months from measurement date
        date_match = re.match(r'^(\d{4})-(\d{2})-(\d{2})', s)
        if date_match:
            if pd.notna(meas_date):
                symptom_date = pd.Timestamp(f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}")
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

    # Phrase → canonical token mapping (applied before category detection)
    phrase_map = {
        'no imrovement':                   'no improvement',    # typo fix
        'recovery only on the right side': 'pr',
        'improvement':                     'pr',
        'initial improvement':             'pr',
        'subtotal remission':              'pr',
    }

    categories = pd.Series(pd.NA, index=df.index, dtype=object)
    percents   = pd.Series(pd.NA, index=df.index, dtype='float64')

    for idx, val in raw.items():
        if val in ('nan', '', 'None', 'NaN'):
            continue

        s = val.lower().strip()

        # Apply phrase replacements
        for phrase, replacement in phrase_map.items():
            s = s.replace(phrase.lower(), replacement)

        # Extract numeric percentage from entry
        # Range: "80-90" → 85 (midpoint); single: ">80", "~75", "100%" → number
        range_m = re.search(r'(\d+)\s*[-–]\s*(\d+)', s)
        single_m = re.search(r'[>~<]?\s*(\d+)\s*%?', s)
        if range_m:
            percents[idx] = (float(range_m.group(1)) + float(range_m.group(2))) / 2
        elif single_m:
            percents[idx] = float(single_m.group(1))

        # Detect which response categories are present in the entry
        found = []
        if re.search(r'\bcr\b', s):
            found.append('CR')
        if re.search(r'\bpr\b', s):
            found.append('PR')
        if re.search(r'no\s*imp', s):
            found.append('NI')

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



#%% Step 1: Forward-fill patient-level data within each patient group
# Patient-level columns are constant across timepoints but only filled in the first row per patient

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


#%% Step 2: Extract timepoint, filter rows, exclude patients

# Extract timepoint number from Erfassungszeitpunkt (e.g., "01.01.1" -> 1)
df_cl['Timepoint'] = (
    df_cl['Erfassungszeitpunkt']
    .str.extract(r'\d+\.\d+\.(\d+)')[0]
    .astype(float)
)
df_cl = move_column_after(df_cl, 'Timepoint', 'Patient')

# Keep only rows with actual measurement data
df_cl_clean = df_cl[df_cl['Datum'].notna()].copy()
print(f"\nRows with measurement data: {len(df_cl_clean)}")

# Exclude patients marked with "Ausschluss" keyword
exclude_mask = df_cl_clean['Unnamed: 0'].str.contains('Ausschluss', case=False, na=False)
excluded_patients = df_cl_clean.loc[exclude_mask, 'Patient'].unique()
print(f"Excluded {len(excluded_patients)} patients by keyword: {excluded_patients}")
df_cl_clean = df_cl_clean[~exclude_mask]

# NOTE: Patients with missing/NaN Response are KEPT — response is NOT used as target variable.
# We use pain_scale instead (see Step 9). Response column is retained as a metadata column only.

# Drop the general health/function assessment questionnaire columns
# ONLY the block from "Schwierigkeiten körperlicher Anstrengung" (inclusive) to
# "Allgemeinzustand Gesundheit HEUTE" (inclusive) is removed.
# Everything before this block (symptoms_months → pain_points) is KEPT.
# Everything after this block (improvement_percent onwards) is also KEPT.
try:
    col_list = df_cl_clean.columns.tolist()

    start_col = 'Schwierigkeiten körperlicher Anstrengung'

    # Accept both correct spelling and known Excel typo ('Gesundheut' vs 'Gesundheit')
    end_col_options = [
        'Allgemeinzustand Gesundheit HEUTE',
        'Allgemeinzustand Gesundheut HEUTE',
    ]
    end_col = next((c for c in end_col_options if c in col_list), None)

    if start_col not in col_list:
        print(f"Warning: start column '{start_col}' not found — no questionnaire columns dropped")
    elif end_col is None:
        print(f"Warning: end column 'Allgemeinzustand...' not found — no questionnaire columns dropped")
    else:
        start_idx      = col_list.index(start_col)
        end_idx        = col_list.index(end_col)
        q_cols_to_drop = col_list[start_idx : end_idx + 1]   # inclusive of both boundaries
        df_cl_clean    = df_cl_clean.drop(columns=q_cols_to_drop)
        print(f"Dropped {len(q_cols_to_drop)} health/function questionnaire columns")
        print(f"  From : '{start_col}'")
        print(f"  To   : '{end_col}'")
        print(f"  Cols : {q_cols_to_drop}")
except Exception as e:
    print(f"Warning: Could not drop questionnaire columns: {e}")




# Remove patients irradiated at MULTIPLE DIFFERENT body parts in the same treatment course.
# E.g. shoulder + heel simultaneously — these are distinct from bilateral (same part, both sides).
# Verify target volumes before removing.
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

# Drop no longer needed columns
cols_to_drop = ['Unnamed: 0', 'Unnamed: 2', 'Comments questionnaire', 'further comments']
df_cl_clean = df_cl_clean.drop(columns=[c for c in cols_to_drop if c in df_cl_clean.columns])

print(f"\nAfter exclusions: {df_cl_clean['Patient'].nunique()} patients, {len(df_cl_clean)} rows")

# Quick inspection after initial cleaning step
TableReport(df_cl_clean, max_plot_columns=100)


#%% Step 3: Rename columns (German to  English) + create baseline copy

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
print(f"Columns renamed: {len(clinical_names)}")

# Baseline copy: raw data with English column names, before any numeric transforms or imputation.
# CatBoost can handle raw strings and missing values natively — this copy preserves that raw state.
# Target variable (pain_scale_t2 / pain_scale_reduction) will be joined later from the clean dataset.
df_cl_raw_baseline = df_cl_clean.copy()
TableReport(df_cl_raw_baseline, max_plot_columns=100)


#%% Step 4: Clean null markers

# Replace German missing markers across ALL columns: k.A./ka/kA and n.D./n.D
# Skip ID columns — Patient and Timepoint must never be nulled out here
null_pattern = r'^([kK]\.?[aA]\.?|[nN]\.?[dD]\.?)$'
null_skip = {'Patient', 'Timepoint'}
print("\n=== Replacing null markers ('kA' and 'nD' varations) ===")
for col in df_cl_clean.columns:
    if col in null_skip:
        continue
    str_col = df_cl_clean[col].astype(str).str.strip()
    mask = str_col.str.match(null_pattern, na=False)
    if mask.sum() > 0:
        print(f"  {col}: replaced {mask.sum()} null markers")
        df_cl_clean.loc[mask, col] = pd.NA


# Manual data corrections (known entry errors identified by inspection)
# These are applied before numeric extraction so the corrected value goes through
# the normal parsing pipeline.

# Patient 248, T2: pain_daytime was entered as "22" — confirmed typo, should be 2
mask_248 = (df_cl_clean['Patient'] == 248) & (df_cl_clean['Timepoint'] == 2)
if mask_248.sum() > 0:
    df_cl_clean.loc[mask_248, 'pain_daytime'] == 2.0
    print("Manual correction: Patient 248 T2 pain_daytime set to 2 (was '22')")
else:
    print("Warning: Patient 248 T2 not found — correction skipped")


#%% Step 5: Extract/convert numeric values from mixed text/number columns

# Ordinal questionnaire columns (scale 1-4 or 1-5): multi-select "1,2" -> avg
# EORTC questionnaire columns already dropped at Step 2, so only pain columns remain.
# Skip: pain_points (categorical string), pain_scale (continuous, comma = German decimal)
all_cols = df_cl_clean.loc[:, 'pain_under_load':'pain_points'].columns
ordinal_cols = [c for c in all_cols if c not in ('pain_points', 'pain_scale')]

print("\n=== Unique raw values — ordinal questionnaire columns (before extraction) ===")
for col in ordinal_cols:
    if col in df_cl_clean.columns:
        uniq = df_cl_clean[col].dropna().unique()
        print(f"\n  {col} ({len(uniq)} unique):")
        for v in sorted(uniq, key=lambda x: str(x)):
            print(f"    {repr(v)}")

print("\n=== Extracting numeric values (ordinal questionnaire columns) ===")
n_converted = 0
for col in ordinal_cols:
    if col in df_cl_clean.columns:
        original_numeric = pd.to_numeric(df_cl_clean[col], errors='coerce')
        extracted = extract_numeric(df_cl_clean[col])
        was_text = original_numeric.isna() & extracted.notna()
        if was_text.sum() > 0:
            print(f"  {col}: extracted {was_text.sum()} values from text entries")
            n_converted += was_text.sum()
        df_cl_clean[col] = extracted
print(f"Total ordinal text entries converted: {n_converted}")


# Continuous column: comma = German decimal, ranges -> midpoint
# pain_scale (1-10): "9,7" -> 9.7, "7-8" -> 7.5
print("\n=== Unique raw values — pain_scale (before extraction) ===")
if 'pain_scale' in df_cl_clean.columns:
    uniq_ps = df_cl_clean['pain_scale'].dropna().unique()
    print(f"  pain_scale ({len(uniq_ps)} unique):")
    for v in sorted(uniq_ps, key=lambda x: str(x)):
        print(f"    {repr(v)}")

print("\n=== Extracting numeric values (continuous columns) ===")
for col in ['pain_scale']:
    if col in df_cl_clean.columns:
        original_numeric = pd.to_numeric(df_cl_clean[col], errors='coerce')
        extracted = extract_continuous(df_cl_clean[col])
        was_text = original_numeric.isna() & extracted.notna()
        if was_text.sum() > 0:
            print(f"  {col}: extracted {was_text.sum()} values from text entries")
        df_cl_clean[col] = extracted
        uniq_after = sorted(df_cl_clean[col].dropna().unique())
        print(f"\n  pain_scale unique values after extraction ({len(uniq_after)}):")
        for v in uniq_after:
            print(f"    {v}")

TableReport(df_cl_clean)

#%% Step 6: Column-specific transformations

# Diagnosis: standardize names (side extracted from target_volume instead)
df_cl_clean['diagnosis'] = standardize_diagnosis(df_cl_clean['diagnosis'])
print(f"\nDiagnosis: {df_cl_clean['diagnosis'].nunique()} unique categories")
print(f"  Categories: {df_cl_clean['diagnosis'].value_counts().to_dict()}")

# Target volume: standardize body part names + extract treatment side
df_cl_clean['target_volume'], df_cl_clean['target_side'] = standardize_target_volume(df_cl_clean['target_volume'])
df_cl_clean = move_column_after(df_cl_clean, 'target_side', 'target_volume')
print(f"\nTarget volume: {df_cl_clean['target_volume'].nunique()} unique categories")
print(f"  Categories: {df_cl_clean['target_volume'].value_counts().to_dict()}")
print(f"  Target side: {df_cl_clean['target_side'].value_counts().to_dict()}")

# Pain points: standardize body part names + side per body part
df_cl_clean['pain_points'] = standardize_pain_points(df_cl_clean['pain_points'])
print(f"\nPain points: {df_cl_clean['pain_points'].nunique()} unique categories (was 149)")
print(f"  Top 20: {df_cl_clean['pain_points'].value_counts().head(40).to_dict()}")


# Filter: split into filter_mm (thickness) and filter_material (Cu/Al)
df_cl_clean = split_filter_column(df_cl_clean)
print(f"\nFilter mm: {sorted(df_cl_clean['filter_mm'].dropna().unique())}")
print(f"Filter material: {df_cl_clean['filter_material'].value_counts().to_dict()}")


df_cl_clean['cumulative_dose'] = pd.to_numeric(
    df_cl_clean['cumulative_dose'].apply(parse_cumulative_dose),
    errors='coerce'
)
print(f"\nCumulative dose: {sorted(df_cl_clean['cumulative_dose'].dropna().unique())}")


# Gender: standardize 'w' (German: weiblich) to 'f' (female)
df_cl_clean['gender'] = df_cl_clean['gender'].replace('w', 'f')
print(f"\nGender: {df_cl_clean['gender'].value_counts().to_dict()}")


# BMI: split overweight_bmi -> overweight (ja/nein) + bmi (float)
df_cl_clean = split_bmi_column(df_cl_clean)
missing_bmi = (df_cl_clean['bmi'].isna() & df_cl_clean['overweight'].notna()).sum()
print(f"BMI split: {missing_bmi} patients with overweight status but missing BMI value")


# Symptoms duration: German strings to numeric months
df_cl_clean['symptoms_months'] = pd.to_numeric(
    parse_symptoms_duration(df_cl_clean['symptoms_months'], df_cl_clean['date']),
    errors='coerce'
)
print(f"Symptoms: range {df_cl_clean['symptoms_months'].min():.0f}-{df_cl_clean['symptoms_months'].max():.0f} months, "
      f"{df_cl_clean['symptoms_months'].isna().sum()} missing")


# Previous therapy: comma-separated codes (1-7) to binary columns
df_cl_clean = encode_therapy_columns(df_cl_clean)
therapy_cols = [f'previous_therapy_{i}' for i in range(1, 8)]
print(f"Therapy encoding: {df_cl_clean[therapy_cols].sum().to_dict()}")



# Response: parse into response_category (CR/PR/NI or combinations) and response_percent (numeric).
# response_category is metadata for reference; regression targets use pain_scale (see Step 9).
df_cl_clean = standardize_response(df_cl_clean, response_col='response')
df_cl_clean = move_column_after(df_cl_clean, 'response_category', 'response')
df_cl_clean = move_column_after(df_cl_clean, 'response_percent', 'response_category')


#%% Step 7: Remove rows with no questionnaire data at all

# A row is considered "empty" if ALL columns between 'date' and 'response' are NaN.
# This covers: symptoms_months, previous_therapy, pain columns, pain_scale, pain_points.
# (EORTC questionnaire columns already dropped at Step 2, so this range is narrower.)
n_before = len(df_cl_clean)
df_cl_clean = df_cl_clean[df_cl_clean['pain_scale'].notna()].copy()
print(f"\nRemoved {n_before - len(df_cl_clean)} rows with missing pain_scale "
      f"({df_cl_clean['Patient'].nunique()} patients remaining)")

TableReport(df_cl_clean)


# distribution of pain_scale at T1-T5: T1/T2/T3 top row, T4/T5 bottom row
timepoints = [1, 2, 3, 4, 5]
colors = sns.color_palette('mako', 5)
fig = plt.figure(figsize=(15, 8))

# Top row: T1, T2, T3 (3 columns)
top_axes = [fig.add_subplot(2, 3, i + 1) for i in range(3)]
# Bottom row: T4, T5 centred using columns 1 and 2 of a 3-col grid
bot_axes = [fig.add_subplot(2, 3, 5), fig.add_subplot(2, 3, 6)]

plot_axes = top_axes + bot_axes

for ax, tp, color in zip(plot_axes, timepoints, colors):
    data = df_cl_clean.loc[df_cl_clean['Timepoint'] == tp, 'pain_scale'].dropna()
    ax.hist(data, bins=15, color=color, edgecolor='white')
    ax.set_title(f'T{tp}  (n={len(data)})')
    ax.set_xlabel('Pain Scale (0-10)')
    ax.set_ylabel('Count')
    if len(data) > 0:
        ax.axvline(data.median(), color='white', linestyle='--', linewidth=1.5,
                   label=f'Median {data.median():.1f}')
        ax.legend(fontsize=9)

plt.suptitle('Distribution of pain_scale by Timepoint', fontweight='bold')
plt.tight_layout()
plt.show()

print(df_cl_clean.groupby('Timepoint')['pain_scale'].describe())


#%% Step 8: final dtype conversion
# measurement_timepoint is a panel group ID, not a date — keep as string
# Columns with more than 25% missing will be dropped later in this step.

categorical_cols = [
    'gender', 'overweight', 'pain_points', 'diagnosis',
    'target_volume', 'target_side', 'filter_material',
    'response', 'response_category'          # response_percent stays float
]

# Ensure id/date columns keep appropriate types
if 'Patient' in df_cl_clean.columns:
    df_cl_clean['Patient'] = pd.to_numeric(df_cl_clean['Patient'], errors='coerce')
if 'Timepoint' in df_cl_clean.columns:
    df_cl_clean['Timepoint'] = pd.to_numeric(df_cl_clean['Timepoint'], errors='coerce')

# Drop rows where Patient or Timepoint couldn't be parsed — they are unusable
n_before = len(df_cl_clean)
df_cl_clean = df_cl_clean.dropna(subset=['Patient', 'Timepoint'])
n_dropped = n_before - len(df_cl_clean)
if n_dropped > 0:
    print(f"Dropped {n_dropped} rows with unparseable Patient or Timepoint values")

df_cl_clean['Patient']   = df_cl_clean['Patient'].astype('int64')
df_cl_clean['Timepoint'] = df_cl_clean['Timepoint'].astype('int64')
if 'measurement_timepoint' in df_cl_clean.columns:
    df_cl_clean['measurement_timepoint'] = df_cl_clean['measurement_timepoint'].astype(str)
if 'date' in df_cl_clean.columns:
    df_cl_clean['date'] = pd.to_datetime(df_cl_clean['date'], errors='coerce')

# Cast categorical string columns to 'category' (if present)
for col in categorical_cols:
    if col in df_cl_clean.columns:
        df_cl_clean[col] = df_cl_clean[col].astype('category')

# All remaining columns (except ids, date, measurement_timepoint, and categoricals) -> float
exclude_for_float = set(categorical_cols) | {'Patient', 'Timepoint', 'measurement_timepoint', 'date'}
cols_to_float = [c for c in df_cl_clean.columns if c not in exclude_for_float]
df_cl_clean[cols_to_float] = df_cl_clean[cols_to_float].apply(lambda s: pd.to_numeric(s, errors='coerce')).astype('float64')

# Verification
print("\n=== Manual dtype assignment (clinical) ===")
print(df_cl_clean.dtypes.value_counts())
TableReport(df_cl_clean, max_plot_columns=100)


#%% Drop columns with >25% missing values

# Protect pain_scale and improvement_percent from auto-dropping:
# - pain_scale is the primary target variable (must be retained)
# - improvement_percent is kept as supplementary metadata
protected_cols = ['pain_scale', 'improvement_percent']

missing_pct = df_cl_clean.isna().mean()
cols_to_drop_na = [
    c for c in missing_pct[missing_pct > 0.25].index
    if c not in protected_cols
]
print(f"Dropping {len(cols_to_drop_na)} columns with >25% missing:")
print(missing_pct[cols_to_drop_na].sort_values(ascending=False))

df_cl_clean = df_cl_clean.drop(columns=cols_to_drop_na)
print(f"\nColumns remaining: {len(df_cl_clean.columns)}")

TableReport(df_cl_clean, max_plot_columns=100)


#%% Step 9: Prepare target variables and modeling datasets

# Save visualization copy BEFORE pain_scale filter — preserves ALL patients
# (not limited to those with pain targets) for use in PCA / MFA / exploration.
df_cl_for_viz = df_cl_clean.copy()


# --- Compute regression targets ---
# pain_scale_t2        : pain scale value AT timepoint T2 (post-treatment outcome)
# pain_scale_reduction : T1 pain_scale MINUS T2 pain_scale
#                        positive value = improvement, negative = worsening

# Extract T1 and T2 pain_scale per patient
pain_t1 = (
    df_cl_clean[df_cl_clean['Timepoint'] == 1][['Patient', 'pain_scale']]
    .rename(columns={'pain_scale': 'pain_scale_t1'})
    .dropna(subset=['pain_scale_t1'])
)
pain_t2 = (
    df_cl_clean[df_cl_clean['Timepoint'] == 2][['Patient', 'pain_scale']]
    .rename(columns={'pain_scale': 'pain_scale_t2'})
    .dropna(subset=['pain_scale_t2'])
)

# Keep only patients with BOTH T1 and T2 pain_scale values (needed for regression target)
pain_targets = pain_t1.merge(pain_t2, on='Patient', how='inner')
pain_targets['pain_scale_reduction'] = pain_targets['pain_scale_t1'] - pain_targets['pain_scale_t2']

print(f"\nPatients with T1 + T2 pain_scale (usable for regression): {len(pain_targets)}")
print(f"pain_scale_t2 range:        {pain_targets['pain_scale_t2'].min():.1f} — {pain_targets['pain_scale_t2'].max():.1f}")
print(f"pain_scale_reduction range: {pain_targets['pain_scale_reduction'].min():.1f} — {pain_targets['pain_scale_reduction'].max():.1f}")
print(f"  (positive = improvement, negative = worsening)")
print(f"pain_scale_reduction stats:\n{pain_targets['pain_scale_reduction'].describe()}")

# Filter clinical dataset to patients that have valid regression targets
df_cl_clean = df_cl_clean[df_cl_clean['Patient'].isin(pain_targets['Patient'])].copy()

# Add regression targets as patient-level columns (same value repeated across all timepoints per patient)
df_cl_clean = df_cl_clean.merge(
    pain_targets[['Patient', 'pain_scale_t2', 'pain_scale_reduction']],
    on='Patient',
    how='left'
)
print(f"\nFinal clinical dataset: {df_cl_clean['Patient'].nunique()} patients, {len(df_cl_clean)} rows")


# --- Distribution of regression targets ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

targets_per_patient = df_cl_clean.drop_duplicates('Patient')

# pain_scale_t2
axes[0].hist(
    targets_per_patient['pain_scale_t2'].dropna(),
    bins=20,
    color=sns.color_palette('mako', 3)[1]
)
axes[0].set_title('Distribution: pain_scale_t2 (T2 pain level)')
axes[0].set_xlabel('Pain Scale (0-10)')
axes[0].set_ylabel('Number of Patients')

# pain_scale_reduction
axes[1].hist(
    targets_per_patient['pain_scale_reduction'].dropna(),
    bins=20,
    color=sns.color_palette('mako', 3)[2]
)
axes[1].set_title('Distribution: pain_scale_reduction (T1 - T2)')
axes[1].set_xlabel('Pain Reduction (positive = improvement)')
axes[1].set_ylabel('Number of Patients')
axes[1].axvline(0, color='red', linestyle='--', linewidth=1, label='No change')
axes[1].legend()

plt.tight_layout()
plt.show()


# --- Copy for advanced modeling (before imputation) ---
# Cleaned and transformed but NOT imputed. Used for the final nested-CV modeling pipeline (Phase 4).
# Missing values will be handled inside the cross-validation loop to avoid leakage.
df_cl_for_modeling = df_cl_clean.copy()
print(f"\nAdvanced modeling dataset (before imputation): {df_cl_for_modeling['Patient'].nunique()} patients, {len(df_cl_for_modeling)} rows")
TableReport(df_cl_for_modeling, max_plot_columns=100)


# --- Merge immunological + clinical datasets (T1 baseline, for modeling) ---
# Goal: predict post-treatment pain from PRE-treatment (T1) immunological + clinical values.
# Both datasets are in their pre-imputation state so that imputation can be done inside CV.

print("\n=== Merging immunological + clinical datasets (T1 baseline, before imputation) ===")

# T1 immunological — uses df_im_reduced (columns dropped, not yet imputed)
df_im_t1 = df_im_reduced[df_im_reduced['Timepoint'] == 1].copy()

# T1 clinical — not imputed
df_cl_t1 = df_cl_for_modeling[df_cl_for_modeling['Timepoint'] == 1].copy()

# Inner join on Patient (only patients present in both datasets)
# Drop Timepoint from clinical side to avoid duplication; suffix any other overlapping cols
df_combined_t1 = df_im_t1.merge(
    df_cl_t1.drop(columns=['Timepoint'], errors='ignore'),
    on='Patient',
    how='inner',
    suffixes=('_im', '_cl')
)

print(f"Immunological T1:  {df_im_t1['Patient'].nunique()} patients")
print(f"Clinical T1:       {df_cl_t1['Patient'].nunique()} patients")
print(f"Combined T1:       {df_combined_t1['Patient'].nunique()} patients")
print(f"\nTarget: pain_scale_t2\n{df_combined_t1['pain_scale_t2'].describe()}")
print(f"\nTarget: pain_scale_reduction\n{df_combined_t1['pain_scale_reduction'].describe()}")

TableReport(df_combined_t1, max_plot_columns=200)


#%%############### VISUALIZATION DATASETS #####################################
# All timepoints, fully cleaned + imputed.
# Used for PCA, MFA, correlation analysis, and other exploratory visualisations.
#
#   df_im_imputed      : immunological — already prepared above (MICE imputation)
#   df_cl_for_viz      : clinical before pain_scale filter — all patients
#   df_cl_imputed      : clinical, median (numeric) + mode (categorical) imputed
#   df_combined_imputed: merged im + cl on Patient + Timepoint (inner join)


# Identify column groups in clinical (excluding IDs / target columns)
_viz_id  = ['Patient', 'Timepoint', 'date', 'measurement_timepoint']
_viz_cat = df_cl_for_viz.select_dtypes(
    include=['category', 'object']).columns.difference(_viz_id).tolist()
_viz_num = df_cl_for_viz.select_dtypes(
    include=['float64', 'Int64', 'int64']).columns.difference(_viz_id).tolist()

df_cl_imputed = df_cl_for_viz.copy()

# Median imputation for numeric columns
for _col in _viz_num:
    if df_cl_imputed[_col].isna().any():
        df_cl_imputed[_col] = df_cl_imputed[_col].fillna(df_cl_imputed[_col].median())

# Mode imputation for categorical columns
for _col in _viz_cat:
    if df_cl_imputed[_col].isna().any():
        _mode = df_cl_imputed[_col].dropna().mode()
        if len(_mode) > 0:
            df_cl_imputed[_col] = df_cl_imputed[_col].fillna(_mode.iloc[0])

print(f"Clinical (viz, imputed): {df_cl_imputed['Patient'].nunique()} patients, {len(df_cl_imputed)} rows")
TableReport(df_cl_imputed, max_plot_columns=100)

# Combined visualization dataset — inner join on Patient + Timepoint
# Only rows where a patient has measurements in BOTH datasets at the SAME timepoint
df_combined_imputed = df_im_imputed.merge(
    df_cl_imputed,
    on=['Patient', 'Timepoint'],
    how='inner',
    suffixes=('_im', '_cl')
)
print(f"\nCombined imputed (viz): {df_combined_imputed['Patient'].nunique()} patients, "
      f"{len(df_combined_imputed)} rows")
TableReport(df_combined_imputed, max_plot_columns=200)


#%%############### BASELINE MODELING DATASETS #################################
# Almost raw data: only exclusion criteria and empty rows removed.
# NO transformation, parsing, or imputation applied.
# Filtered to T1 only; only patients present in pain_targets (have both T1 + T2 pain_scale).

# Patients eligible for modeling (have both T1 and T2 pain_scale)
model_patients = set(pain_targets['Patient'].values)

# --- Immunological raw T1 ---
df_im_raw_t1 = (
    df_im_reduced[
        (df_im_reduced['Timepoint'] == 1) &
        (df_im_reduced['Patient'].isin(model_patients))
    ]
    .copy()
    .reset_index(drop=True)
)
# Attach regression targets so run_catboost_regressor can find the target column
df_im_raw_t1 = df_im_raw_t1.merge(
    pain_targets[['Patient', 'pain_scale_t2', 'pain_scale_reduction']],
    on='Patient', how='left'
)

# --- Clinical raw T1: add pain targets (computed from cleaned data) ---
# df_cl_raw_baseline = right after rename, before any numeric parsing or transforms
df_cl_raw_t1 = (
    df_cl_raw_baseline[
        (df_cl_raw_baseline['Timepoint'] == 1) &
        (df_cl_raw_baseline['Patient'].isin(model_patients))
    ]
    .copy()
    .reset_index(drop=True)
)
# Attach regression targets (pain_scale is still a raw string here — treated as cat feature)
df_cl_raw_t1 = df_cl_raw_t1.merge(
    pain_targets[['Patient', 'pain_scale_t2', 'pain_scale_reduction']],
    on='Patient', how='left'
)

# --- Combined raw T1 ---
df_combined_raw_t1 = df_im_raw_t1.merge(
    df_cl_raw_t1.drop(columns=['Timepoint'], errors='ignore'),
    on='Patient', how='inner',
    suffixes=('_im', '_cl')
)

print(f"\nBaseline raw T1 datasets:")
print(f"  Immunological: {df_im_raw_t1.shape},  patients: {df_im_raw_t1['Patient'].nunique()}")
print(f"  Clinical:      {df_cl_raw_t1.shape},  patients: {df_cl_raw_t1['Patient'].nunique()}")
print(f"  Combined:      {df_combined_raw_t1.shape}, patients: {df_combined_raw_t1['Patient'].nunique()}")


#%%############### REGRESSION HELPERS #########################################

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
    """5-fold × 5-repeat RepeatedKFold CatBoostRegressor. No hyperparameter tuning.
    Returns (results_df, last_trained_model, X_features, y_pred_series).

    Automatically excluded from features:
      - ID columns  : Patient, Timepoint, Date, date, measurement_timepoint
      - Leaky cols  : any column whose name contains 'response', 'improvement_percent',
                      or 'pain_scale' (catches the raw score AND the other target)
    """
    always_exclude  = ['Patient', 'Timepoint', 'Date', 'date', 'measurement_timepoint']
    leaky_patterns  = ['response', 'improvement_percent', 'pain_scale']
    exclude = set(always_exclude + [target_col])
    for col in df_model.columns:
        if any(pat in col.lower() for pat in leaky_patterns):
            exclude.add(col)

    feature_cols = [c for c in df_model.columns if c not in exclude]
    X = df_model[feature_cols].copy()
    y = df_model[target_col].copy()

    # Drop rows where target is NaN
    valid = y.notna()
    X, y = X[valid].reset_index(drop=True), y[valid].reset_index(drop=True)

    # CatBoost requires string for categorical features
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

        model = CatBoostRegressor(
            iterations=300,
            random_seed=random_state,
            verbose=0
        )
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

    # Bar plot: mean |SHAP| per feature
    shap.summary_plot(shap_values, X, plot_type="bar", show=False, max_display=20)
    plt.title(f"SHAP Feature Importance — {name}")
    plt.tight_layout()
    plt.show()

    # Beeswarm: direction + magnitude
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


#%%############### BASELINE CATBOOST — pain_scale_reduction ###################

print("\n" + "="*70)
print("  CATBOOST BASELINE REGRESSOR — Target: pain_scale_reduction")
print("="*70)

target = 'pain_scale_reduction'

res_im_red, model_im_red, X_im_red, ypred_im_red = run_catboost_regressor(
    df_im_raw_t1, target, "Immunological (raw T1)")

res_cl_red, model_cl_red, X_cl_red, ypred_cl_red = run_catboost_regressor(
    df_cl_raw_t1, target, "Clinical (raw T1)")

res_comb_red, model_comb_red, X_comb_red, ypred_comb_red = run_catboost_regressor(
    df_combined_raw_t1, target, "Combined (raw T1)")

summary_cb_red = print_regression_summary(
    {"Immunological": res_im_red, "Clinical": res_cl_red, "Combined": res_comb_red},
    target
)

#%% SHAP — pain_scale_reduction

shap_im_red   = plot_shap_regressor(model_im_red,   X_im_red,   "Immunological — pain_scale_reduction")
shap_cl_red   = plot_shap_regressor(model_cl_red,   X_cl_red,   "Clinical — pain_scale_reduction")
shap_comb_red = plot_shap_regressor(model_comb_red, X_comb_red, "Combined — pain_scale_reduction")


#%%############### BASELINE CATBOOST — pain_scale_t2 ##########################

print("\n" + "="*70)
print("  CATBOOST BASELINE REGRESSOR — Target: pain_scale_t2")
print("="*70)

target = 'pain_scale_t2'

res_im_t2, model_im_t2, X_im_t2, ypred_im_t2 = run_catboost_regressor(
    df_im_raw_t1, target, "Immunological (raw T1)")

res_cl_t2, model_cl_t2, X_cl_t2, ypred_cl_t2 = run_catboost_regressor(
    df_cl_raw_t1, target, "Clinical (raw T1)")

res_comb_t2, model_comb_t2, X_comb_t2, ypred_comb_t2 = run_catboost_regressor(
    df_combined_raw_t1, target, "Combined (raw T1)")

summary_cb_t2 = print_regression_summary(
    {"Immunological": res_im_t2, "Clinical": res_cl_t2, "Combined": res_comb_t2},
    target
)

#%% SHAP — pain_scale_t2

shap_im_t2   = plot_shap_regressor(model_im_t2,   X_im_t2,   "Immunological — pain_scale_t2")
shap_cl_t2   = plot_shap_regressor(model_cl_t2,   X_cl_t2,   "Clinical — pain_scale_t2")
shap_comb_t2 = plot_shap_regressor(model_comb_t2, X_comb_t2, "Combined — pain_scale_t2")


#%%############### ADVANCED MODELING DATASET ##################################
# Cleaned + transformed + parsed, NOT imputed, T1 only.
# df_combined_t1 (from Step 9) is already this dataset.
# HistGradientBoostingRegressor handles numeric NaN natively.

print(f"\nAdvanced modeling dataset (df_combined_t1): {df_combined_t1.shape}, "
      f"patients: {df_combined_t1['Patient'].nunique()}")
print(df_combined_t1[['pain_scale_reduction', 'pain_scale_t2']].describe())


#%%############### ADVANCED MODELING: HistGradientBoosting ####################

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer


def run_hgb_regressor(df_model, target_col, name,
                      n_splits=5, n_repeats=5, random_state=42):
    """5-fold × 5-repeat RepeatedKFold HistGradientBoostingRegressor. No tuning.
    Numeric NaN handled natively. Categorical columns encoded via OrdinalEncoder.
    Returns (results_df, fitted_pipeline, X_features, y_pred_series).
    """
    always_exclude = ['Patient', 'Timepoint', 'Date', 'date', 'measurement_timepoint']
    leaky_patterns = ['response', 'improvement_percent', 'pain_scale']
    exclude = set(always_exclude + [target_col])
    for col in df_model.columns:
        if any(pat in col.lower() for pat in leaky_patterns):
            exclude.add(col)

    feature_cols = [c for c in df_model.columns if c not in exclude]
    X = df_model[feature_cols].copy().reset_index(drop=True)
    y = df_model[target_col].copy().reset_index(drop=True)

    # Drop rows with NaN target
    valid = y.notna()
    X, y = X[valid].reset_index(drop=True), y[valid].reset_index(drop=True)

    cat_cols = X.select_dtypes(include=['category', 'object']).columns.tolist()

    # Pipeline: OrdinalEncoder for categoricals (NaN → -2, unknown → -1),
    # numeric columns passed through (HGB handles NaN natively)
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OrdinalEncoder(
                handle_unknown='use_encoded_value',
                unknown_value=-1,
                encoded_missing_value=-2
            ), cat_cols),
        ],
        remainder='passthrough',
        verbose_feature_names_out=False
    )

    pipeline = Pipeline([
        ('preprocess', preprocessor),
        ('model', HistGradientBoostingRegressor(
            max_iter=500,
            random_state=random_state
        ))
    ])

    print(f"\n{'='*65}")
    print(f"  HistGradientBoosting Regressor — {name}")
    print(f"  Target : {target_col}")
    print(f"  Samples: {len(X)},  Features: {len(feature_cols)}")
    print(f"  CV     : {n_splits}-fold × {n_repeats} repeats = {n_splits * n_repeats} fits")
    print(f"{'='*65}")

    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    fold_results = []
    y_pred_arr = np.full(len(y), np.nan)

    for fold, (train_idx, test_idx) in enumerate(rkf.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        y_pred_arr[test_idx] = preds

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

    return results_df, pipeline, X, pd.Series(y_pred_arr, name='y_pred')


#%%############### RUN HGB — pain_scale_reduction #############################

print("\n" + "="*70)
print("  HGB ADVANCED MODELING — Target: pain_scale_reduction")
print("="*70)

res_hgb_red, pipeline_hgb_red, X_hgb_red, ypred_hgb_red = run_hgb_regressor(
    df_combined_t1, 'pain_scale_reduction', "Combined clean T1")

# Built-in feature importance from the HGB model
hgb_model     = pipeline_hgb_red.named_steps['model']
try:
    feat_names = pipeline_hgb_red.named_steps['preprocess'].get_feature_names_out()
except Exception:
    feat_names = X_hgb_red.columns.tolist()

importance_df = pd.DataFrame({
    'Feature':    feat_names,
    'Importance': hgb_model.feature_importances_
}).sort_values('Importance', ascending=False).reset_index(drop=True)

print(f"\nTop 20 features by HGB importance (pain_scale_reduction):")
print(importance_df.head(20).to_string(index=False))

# Bar plot of feature importance
plt.figure(figsize=(10, 8))
sns.barplot(
    data=importance_df.head(20),
    x='Importance', y='Feature',
    palette='mako', orient='h'
)
plt.title("HGB Feature Importance — pain_scale_reduction", fontweight='bold')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()

# %%
