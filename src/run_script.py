# File for running and testing code

# imports
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



# %%################ RAW CLINICAL DATASET #############################


# Table report of clinical dataset
print("TableReport of raw clinical dataset:")
TableReport(df_cl, max_plot_columns=138)

# A lot of null values dues to empty rows (1658 rows),
# as well as other comments/notes in the excel sheet.
# Need to structure data based on timepoints 1,2,3,4...
# Patients with missing treatment/ response information - remove
# Patients with "Ausschluss/ Exclude" - remove
# Patients with only 1 measured timepoint - remove
# Combine Survey questions/columns ?

# na analysis of clinical dataset
print("Na analysis of clinical dataset:")
na.altair.plot_heatmap(df_cl)


# Raw clinical dataset statistics



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
print("measurements per timepoint:")
print(df_im["Timepoint"].value_counts().sort_index())
print("number of patients per timepoint:")
print(df_im.groupby("Timepoint")["Patient"].nunique().sort_index())
print("\n")

# Patients with measurements from t1 through t5
patients_t1 = set(df_im[df_im["Timepoint"] == 1]["Patient"])
patients_t2 = set(df_im[df_im["Timepoint"] == 2]["Patient"])
patients_t3 = set(df_im[df_im["Timepoint"] == 3]["Patient"])
patients_t4 = set(df_im[df_im["Timepoint"] == 4]["Patient"])
patients_t5 = set(df_im[df_im["Timepoint"] == 5]["Patient"])    

print("Patients with measurements at timepoint 1 and 2:", len(patients_t1 & patients_t2))
print("Patients with measurements at timepoint 1,2 and 3:", len(patients_t1 & patients_t2 & patients_t3))
print("Patients with measurements at timepoint 1,2,3 and 4:", len(patients_t1 & patients_t2 & patients_t3 & patients_t4))
print("Patients with measurements at timepoint 1,2,3,4 and 5:", len(patients_t1 & patients_t2 & patients_t3 & patients_t4 & patients_t5))   

# Bar plot of measurements per timepoint
plt.figure(figsize=(8, 5))
sns.countplot(x="Timepoint", data=df_im, order=sorted(df_im["Timepoint"].unique()))
plt.title("Number of measurements per Timepoint")
plt.xlabel("Timepoint")
plt.ylabel("Number of measurements")
plt.show()

# Bar plot of number of unique patients per timepoint
plt.figure(figsize=(8, 5))
sns.barplot(
    x=df_im.groupby("Timepoint")["Patient"].nunique().index,
    y=df_im.groupby("Timepoint")["Patient"].nunique().values
)
plt.title("Number of unique patients per Timepoint")
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

df_im_raw = df_im.copy()  # copy of raw dataset
df_im = df_im.drop(columns=dropped_columns)

# Correcting datatypes

# husk å skrive opp alle kolonner som har blitt endret!

# Changing columntyoe to date/time type
df_im["Messdatum"] = pd.to_datetime(
    df_im["Messdatum"], errors="coerce")

# All other columns should be Float type (except Messdatum, Patient and Timepoint)
exclude_cols = ["Messdatum", "Patient", "Timepoint"]
float_cols = df_im.columns.difference(exclude_cols)
df_im[float_cols] = df_im[float_cols].apply(
    pd.to_numeric, errors="coerce"
)

# Removing empty rows from row 829 tto 834
df_im = df_im.drop(index=range(823, 829)) #is this correct?

# remove empty measurement row at index 84?
df_im = df_im.drop(index=77)

# Removing columns with more than 25% missing values:
na_frac = df_im.isna().mean()
cols_to_drop = na_frac[na_frac > 0.25].index.tolist()
df_im = df_im.drop(columns=cols_to_drop).copy()

"""
Dropped Columns: 
['TC_CD25hi', 'B_CD25hi', 'Eos_HLADR+', 'Mo2_HLADRhi', 'TC_HLADRhi', 'NK_HLADRhi', 
'Eos_CD69+', 'Bas_CD69+', 'Mo_CD69+', 'B_CD69+', 'DC_CD69+', 'TH naive_PD1+', 
'TH eff_PD1+', 'TC naive_PD1+'

"""

# New Tablereport
TableReport(df_im, max_plot_columns=138)



#%% Imputing missing values using miceforest and median

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
df_im2 = df_im.rename(columns=rename_map)

# imputing with miceforest with renamed columns
X_im = df_im2[list(rename_map.values())]

kernel = mf.ImputationKernel(
    data=X_im,
    datasets=3,   # 3 imputed datasets to create
    random_state=42
)

kernel.mice(5) # 5 iterations

X_imputed_renamed = kernel.complete_data(dataset=1) # get the 1st completed dataset

# changing back to original column names
reverse_rename_map = {v: k for k, v in rename_map.items()}
X_imputed = X_imputed_renamed.rename(columns=reverse_rename_map)

# final imputation
df_im_imputed = pd.concat(
    [
        df_im[exclude_cols].reset_index(drop=True),
        X_imputed.reset_index(drop=True)
    ],
    axis=1
)

# New tablereport of imputed data
TableReport(df_im_imputed, max_plot_columns=138)


# compare with median imputation.....?
df_im_median = df_im.copy()
for col in feature_cols:
    median_value = df_im_median[col].median()
    df_im_median[col] = df_im_median[col].fillna(median_value)  

TableReport(df_im_median, max_plot_columns=138)


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

id_cols = ["Patient", "Timepoint", "Messdatum"]
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
    columns=[f"T{t}" for t in timepoints].      # column labels
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
    cmap="coolwarm",
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

#%% ############## PCA analysis immu dataset ########################
# using prince package for pca analysis:

# scale data before?

# Pca for timepoint 1 - 5 individually
for t in timepoints:
    df_t = dfs[t]
    X_t = df_t.drop(columns=id_cols)

    pca = ps.PCA(
        n_components=3,
        n_iter=3,
        copy=True,
        engine='sklearn',
        check_input=True,
        random_state=42
    )

    pca = pca.fit(X_t)

    # plotting results of PCA, scatter plot of patients at timepoint t, colored by....
    pca.plot_rows(
        X_t,
        ax=None,
        figsize=(8, 6),
        show_points=True,
        labels=None,
        color_labels=None,
        ellipse_outline=False,
        ellipse_fill=False,
        confidence_level=0.95,
        title=f'PCA of Immunological Data at Timepoint {t}',
        show=True
    )

    # Explained variance ratio for pc 1, 2 and 3
    print(f"Explained variance ratio for Timepoint {t}:")
    print(pca.explained_inertia_)

    # Scores (coordinates) for each patient
    row_coords = pca.row_coordinates(X_t)
    print(f"Row coordinates for Timepoint {t}:")
    print(row_coords.head())

    # Top contributing variables to PC1 and PC2
    loading_scores = pca.column_correlations(X_t)
    print(f"Top contributing variables to PC1 and PC2 for Timepoint {t}:")
    print(loading_scores.head())
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

# PCA for timepoints 1 and 2 combined
for (df_a, df_b, label) in [(t12, t22, "T1 and T2"), (t13, t23, "T1 and T3"), (t32, t33, "T2 and T3")]:
    X_a = df_a.drop(columns=id_cols)
    X_b = df_b.drop(columns=id_cols)

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

    # plotting results of PCA
    pca.plot_rows(
        X_combined,
        ax=None,
        figsize=(8, 6),
        show_points=True,
        labels=None,
        color_labels=None,
        ellipse_outline=False,
        ellipse_fill=False,
        confidence_level=0.95,
        title=f'PCA of Immunological Data at {label}',
        show=True
    )

    # Explained variance ratio for pc 1, 2 and 3
    print(f"Explained variance ratio for {label}:")
    print(pca.explained_inertia_)

    # Scores (coordinates) for each patient
    row_coords = pca.row_coordinates(X_combined)
    print(f"Row coordinates for {label}:")
    print(row_coords.head())

    # Top contributing variables to PC1 and PC2
    loading_scores = pca.column_correlations(X_combined)
    print(f"Top contributing variables to PC1 and PC2 for {label}:")
    print(loading_scores.head())
    print("\n")




#%%############ MFA for timepoints 1, 2 and 3 combined 

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

mfa_cols = id_cols = ["Patient", "Timepoint", "Messdatum"]   
 

# Dropping patient id and timepoint columns from analysis
X1 = df1.drop(columns=mfa_cols)
X2 = df2.drop(columns=mfa_cols)
X3 = df3.drop(columns=mfa_cols)

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

# Eigenvalues for Dim 0, 1 and 2 (explained variance pc 1, 2 and 3)
mfa.eigenvalues_summary

# Scores for each patient at different timepoints-groups
mfa.partial_row_coordinates(dataset)

# Pasient 221 at timepoint 2 is an extreme outlier in MFA plot and
# has extreme values in raw data file?
dataset.loc[221, "T2"]

#%%######### PyOD for outlier detection immunological dataset ########

# Using an ensemble of outlier detection methods from pyod package:
# Isolation Forest (IForest): tree-based method, very different from pca, great for high-dimensional continous data
# Local Outlier Factor (LOF): proximity-based method measuring density of points, widely used, good for datasets with clustered observations
# Empirical Cumulative Distribution (ECOD): probabilistic-based method, non-parametric, good for datasets with unknown distributions
# Copula-Based Outlier Detection (COPOD): probabilistic method using correlations, parameter-free, effective for high-dimensional data with complex dependencies

from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.ecod import ECOD
from pyod.models.copod import COPOD
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Using imputed immunological dataset without ID columns
X_pyod = df_im_imputed.drop(columns=id_cols).values
patient_ids = df_im_imputed["Patient"].values
timepoints = df_im_imputed["Timepoint"].values

contamination = 0.05  # assuming 5% outliers in dataset
models = {
    'IsolationForest': IForest(
        contamination=contamination,
        n_estimators=100,      # Number of trees (default=100, stable)
        random_state=42
    ),
    'LOF': LOF(
        contamination=contamination,
        n_neighbors=20,        # Number of neighbors (default=20)
        # Higher = smoother decision boundary, more global
        # Lower = more sensitive to local density variations
        metric='minkowski'
    ),
    'ECOD': ECOD(
        contamination=contamination
        # Parameter-free! Uses empirical cumulative distribution
    ),
    'COPOD': COPOD(
        contamination=contamination
        # Parameter-free! Uses copula-based approach
    )
}

results = {}
outlier_counts = {}

for name, model in models.items():
    print(f"Fitting model: {name}...")
    
    # Fit model
    model.fit(X_pyod)
    
    # Get predictions: 0 = inlier, 1 = outlier
    predictions = model.predict(X_pyod)
    
    # Get outlier scores (higher = more outlier-like)
    scores = model.decision_function(X_pyod)
    
    # Store results
    results[name] = {
        'predictions': predictions,
        'scores': scores,
        'scores_normalized': (scores - np.min(scores)) / (np.max(scores) - np.min(scores)),
        'model': model
    }
    
    # Count outliers
    n_outliers = np.sum(predictions == 1)
    outlier_counts[name] = n_outliers
    
    print(f" Found {n_outliers} outliers ({n_outliers/len(X_pyod)*100:.2f}%)\n")

# Plot for each model the outlier scores
# dimensionality reduction with PCA to 2D for visualization

# scale data before
scaler = StandardScaler()
X_pyod_scaled = scaler.fit_transform(X_pyod)

pca_pyod = PCA(n_components=2, random_state=42)
X_2d = pca_pyod.fit_transform(X_pyod_scaled)
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
axes = axes.flatten()

for idx, (name, data) in enumerate(results.items()):
    ax = axes[idx]

    # Create dataframe for seaborn
    plot_df = pd.DataFrame({
        'PC1': X_2d[:, 0],
        'PC2': X_2d[:, 1],
        'Outlier': ['Outlier' if pred == 1 else 'Inlier' for pred in data['predictions']],
        'Score': data['scores_normalized'],
        'Patient': patient_ids,
        'Timepoint': timepoints
    })
    
    # Seaborn scatter plot (sized based on outlier score)
    sns.scatterplot(
        data=plot_df,
        x='PC1',
        y='PC2',
        hue='Outlier',
        size='Score',
        sizes=(20, 200),
        palette={'Inlier': 'blue', 'Outlier': 'red'},
        alpha=0.6,
        ax=ax,
        legend=True
    )
    
    # Annotate outliers with Patient-Timepoint
    outliers_df = plot_df[plot_df['Outlier'] == 'Outlier']
    for _, row in outliers_df.iterrows():
        ax.annotate(
            f"P{int(row['Patient'])}-T{int(row['Timepoint'])}",
            (row['PC1'], row['PC2']),
            fontsize=8,
            xytext=(5, 5),
            textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7)
        )
    
    ax.set_xlabel(f'PC1 ({pca_pyod.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11)
    ax.set_ylabel(f'PC2 ({pca_pyod.explained_variance_ratio_[1]*100:.1f}%)', fontsize=11)
    ax.set_title(f'{name}\n{outlier_counts[name]} outliers detected', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(title='', loc='best', fontsize=10)

plt.tight_layout()
plt.suptitle('PyOD Outlier Detection - Immunological Dataset', fontsize=16, fontweight='bold', y=1.02)
plt.show()


# Consensesous analysis

# Create consensus matrix
consensus_matrix = np.column_stack([
    results['IsolationForest']['predictions'],
    results['LOF']['predictions'],
    results['ECOD']['predictions'],
    results['COPOD']['predictions']
])

# Count how many algorithms flagged each sample
consensus_count = consensus_matrix.sum(axis=1)

# Show which algorithms agree on which samples for top 30 most flagged samples
top_n = 30
top_indices = np.argsort(consensus_count)[-top_n:]

plt.figure(figsize=(14, 8))
sns.heatmap(
    consensus_matrix[top_indices].T,
    cmap='RdYlGn_r',
    cbar_kws={'label': 'Outlier (1) vs Inlier (0)'},
    yticklabels=['IForest', 'LOF', 'ECOD', 'COPOD'],
    xticklabels=[f"P{int(patient_ids[i])}-T{int(timepoints[i])}" for i in top_indices],
    linewidths=0.5,
    vmin=0,
    vmax=1
)
plt.xlabel('Patient-Timepoint', fontsize=12)
plt.ylabel('Algorithm', fontsize=12)
plt.title(f'Consensus Heatmap: Top {top_n} Most Flagged Samples', fontsize=14, fontweight='bold')
plt.xticks(rotation=90, fontsize=8)
plt.tight_layout()
plt.show()


# Find all samples flagged by 3 or more algorithms
high_consensus_mask = consensus_count >= 3
high_consensus_indices = np.where(high_consensus_mask)[0]

print("\n")
print(f"CONSENSUS ANALYSIS:")
print(f"Total samples flagged by 3 or more algorithms out of 4: {len(high_consensus_indices)}\n")

# Create consensus dataframe
consensus_df = pd.DataFrame({
    'Dataset index': high_consensus_indices,                            # Original index in dataset
    'Patient': patient_ids[high_consensus_indices],                     # Patient IDs
    'Timepoint': timepoints[high_consensus_indices].astype(int),          # Timepoints
    "Avg Score": np.mean([
        results['IsolationForest']['scores_normalized'][high_consensus_indices],
        results['LOF']['scores_normalized'][high_consensus_indices],
        results['ECOD']['scores_normalized'][high_consensus_indices],
        results['COPOD']['scores_normalized'][high_consensus_indices]
    ], axis=0),
    'IsolationForest': consensus_matrix[high_consensus_indices, 0],       
    'LOF': consensus_matrix[high_consensus_indices, 1],
    'ECOD': consensus_matrix[high_consensus_indices, 2],
    'COPOD': consensus_matrix[high_consensus_indices, 3],
    'Consensus_Count': consensus_count[high_consensus_indices]
}).sort_values('Consensus_Count', ascending=False)

print(consensus_df.to_string(index=False))


# Displaying the top 10 consensus outliers (first 10 columns from dataset)
top_10_consensus = df_im_imputed.iloc[high_consensus_indices].head(10)
print("\nMeasurements for Top 10 Consensus Outliers (first ten columns):")
print(top_10_consensus.iloc[:, :10].to_string(index=False))


# Bar plot of number of consensus outliers per timepoint
timepoint_outliers = consensus_df.groupby('Timepoint').size().reset_index(name='Count')
plt.figure(figsize=(10, 6))
sns.barplot(data=timepoint_outliers, x='Timepoint', y='Count', palette='viridis')
plt.xlabel('Timepoint', fontsize=12, fontweight='bold')
plt.ylabel('Number of Consensus Outliers (≥3 flags)', fontsize=12, fontweight='bold')
plt.title('Consensus Outliers by Timepoint', fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()


#%%############ Cleaning clinical dataset #############################

# Translating and renaming clinical dataset column-naames from german to english

clinical_names = {
    # Patientt demographics
    "Unnamed: 0": "treatment_location",
    "Patient": "Patient",
    "Age at start": "age_at_start",
    "Gender": "sex",
    "Weight [kg]": "weight_kg",
    "Height [cm]": "height_cm",
    "Overweight? BMI": "overweight_bmi",

    # Dates and timings
    "Erfassungszeitpunkt": "assessment_timepoint",
    "Datum": "assessment_date",
    "Beschwerden seit": "symptoms_since",
    "vorherige Therapie": "previous_therapy",

    # Pain characteristics
    "unter Belastung": "pain_under_load",
    "bei Nacht": "pain_night",
    "tagsüber": "pain_daytime",
    "in Ruhe": "pain_at_rest",
    "bei ersten Schritten/Morgensteifigkeit": "pain_morning_stiffness",
    "Schmerzskala": "pain_scale",
    "Schmerzpunkte": "pain_points",

    # Functional limitations
    "Schwierigkeiten körperlicher Anstrengung": "difficulty_physical_exertion",
    "Schwierigkeiten bei längerem Spaziergang": "difficulty_long_walk",
    "Schwierigkeiten bei kurzer Strecke": "difficulty_short_distance",
    "tagsüber liegen oder sitzen": "lying_or_sitting_daytime",
    "Hilfe Essen, Anziehen, Waschen, Toilette": "help_with_adl",
    "Einschränkung bei Alltag": "limitation_daily_life",
    "Einschränkung bei Hobbys/Freizeit": "limitation_leisure",

    # General health symptoms
    "kurzatmig": "shortness_of_breath",
    "Schmerzen": "general_pain",
    "Ausruhen": "need_to_rest",
    "Schlafstörungen": "sleep_disturbance",
    "Schwach": "weakness",
    "Appetitmangel": "loss_of_appetite",
    "Übelkeit": "nausea",
    "Erbrochen": "vomiting",
    "Verstopfung": "constipation",
    "Durchfall": "diarrhea",
    "Müde": "fatigue",

    # Cognitive / phycological impairments
    "Beeinträchtigung im Alltag": "impairment_daily_life",
    "Konzentrationsstörung": "concentration_difficulty",
    "angespannt": "feeling_tense",
    "Sorgen gemacht": "worry",
    "reizbar": "irritability",
    "niedergeschlagen": "depressed_mood",
    "Erinnerungsprobleme": "memory_problems",

    # Social impairments
    "Beeinträchtigung Familienleben": "family_life_impairment",
    "Beeinträchtigung Zusammensein Menschen": "social_interaction_impairment",
    "finanzielle Schwierigkeiten": "financial_difficulties",

    # Quality of life
    "Gesundheitszustand insgesamt": "overall_health_status",
    "Lebensqualität insgesamt": "overall_quality_of_life",

    # Daily tasks
    "Strecken": "stretching",
    "Heben und Tragen von 10kg": "lift_carry_10kg",
    "Waschen und Abtrocknen": "wash_and_dry",
    "Bücken": "bending",
    "Haare Waschen im Waschbecken": "wash_hair",
    "Sitzen 1h": "sit_1h",
    "Stehen 30min": "stand_30min",
    "Aufsetzen": "sit_up",
    "Strümpfe an-/ausziehen": "put_on_socks",
    "Gegenstand aus Sitzposition aufheben": "pick_object_seated",
    "Gegenstand auf Tisch stellen": "place_object_table",
    "Schnell laufen 100m": "fast_walk_100m",

    # EQ-5D like questionarre
    "Beweglichkeit/Mobilität": "mobility",
    "für sich selbst sorgen": "self_care",
    "alltägliche Tätigkeiten": "usual_activities",
    "Schmerzen/körperliche Beschwerden": "pain_discomfort",
    "Angst/Niedergeschlagenheit": "anxiety_depression",

    # Clinical outcomes and treatment variables
    "Allgemeinzustand Gesundheut HEUTE": "health_status_today",
    "Besserung nach Nachuntersuchung laut Arztbrief in %": "improvement_percent",
    "Comments questionnaire": "comments_questionnaire",
    "Diagnosis": "diagnosis",
    "Target volume": "target_volume",
    "single fraction": "single_fraction",
    "kummulative dose (x) - if two targets were applied": "cumulative_dose",
    "FHA": "fha",
    "kV": "kv",
    "mA": "ma",
    "Filter": "filter",
    "Response": "response",
    "further comments:": "comments_additional",
}

# map new columnnames to dataset...



# cleaning up column names in cliical dataset
immu_names = {
    "Patient": "patient_id",
    "Timepoint": "timepoint",
    "Messdatum": "measurement_date",
    "ID_Subset": "subset_id",
}

# map new columnnames to dataset...

