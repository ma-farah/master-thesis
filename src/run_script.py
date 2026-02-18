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



#%%############# Comparing miceforest vs median imputation ##########################

# Create mask of originally missing values in df_im
missing_mask = df_im[feature_cols].isna()

# Extract only the imputed values (originally missing) from both methods
mice_imputed_values = df_im_imputed[feature_cols][missing_mask]
median_imputed_values = df_im_median[feature_cols][missing_mask]

# Flatten to 1D arrays for overall comparison
mice_flat = mice_imputed_values.values.flatten()
median_flat = median_imputed_values.values.flatten()

# Create mask for positions where BOTH methods have valid (non-NaN) values
valid_mask = ~np.isnan(mice_flat) & ~np.isnan(median_flat)
mice_flat = mice_flat[valid_mask]
median_flat = median_flat[valid_mask]

# Calculate differences
differences = mice_flat - median_flat

print("\n" + "="*80)
print("COMPARISON OF MICEFOREST VS MEDIAN IMPUTATION")
print("="*80)
print(f"\nTotal number of imputed values: {len(differences)}")
print(f"Mean absolute difference: {np.abs(differences).mean():.4f}")
print(f"Median absolute difference: {np.median(np.abs(differences)):.4f}")
print(f"Max absolute difference: {np.abs(differences).max():.4f}")
print(f"Std of differences: {np.std(differences):.4f}")


# Feature-wise comparison: which features have largest imputation differences?
feature_differences = {}

for col in feature_cols:
    # Get originally missing values for this feature
    col_mask = missing_mask[col].values  # Convert to numpy array for index-independent masking

    if col_mask.sum() > 0:  # If there were missing values
        mice_vals = df_im_imputed[col].values[col_mask]
        median_vals = df_im_median[col].values[col_mask]

        # Only compare positions where BOTH have valid values
        valid = ~np.isnan(mice_vals) & ~np.isnan(median_vals)
        if valid.sum() > 0:
            mice_valid = mice_vals[valid]
            median_valid = median_vals[valid]

            # Calculate metrics
            mae = np.mean(np.abs(mice_valid - median_valid))
            rmse = np.sqrt(np.mean((mice_valid - median_valid)**2))
            max_diff = np.max(np.abs(mice_valid - median_valid))
            n_missing = valid.sum()

            feature_differences[col] = {
                'MAE': mae,
                'RMSE': rmse,
                'Max_Diff': max_diff,
                'N_Missing': n_missing
            }

# Create dataframe and sort by MAE
feature_diff_df = pd.DataFrame(feature_differences).T
feature_diff_df = feature_diff_df.sort_values('MAE', ascending=False)

print("\n" + "="*80)
print("TOP 20 FEATURES WITH LARGEST IMPUTATION DIFFERENCES (sorted by MAE)")
print("="*80)
print(feature_diff_df.head(20).to_string())
print("\n")

# Lets look at the distrubution of the the high-difference features, top 10:
# Are these features normally distributed or heavily skewed?
high_mae_features = ['B_HLADR+', 'TC_PD1+', 'TH_PD1+', 'Mo_HLADRhi', 'Mo1_HLADRhi', 'Mo3_HLADRhi', 'Mo3_HLADR+', 'TC eff_PD1+', 'TH CM_PD1+', 'Mo2_HLADR+', 'TC_HLADR+']

fig, axes = plt.subplots(1, len(high_mae_features), figsize=(20, 4))
for ax, col in zip(axes, high_mae_features):
    df_im[col].dropna().hist(ax=ax, bins=30)
    ax.axvline(df_im[col].median(), color='r', linestyle='--', label='Median')
    ax.set_title(col)
plt.tight_layout()
plt.show()

# Most seem very skewed. 
# Lets see at the immputed values, side by side:

 #for top 10 features, show what each method imputed:

for col in high_mae_features:
    col_mask = missing_mask[col].values
    if col_mask.sum() > 0:
        comparison = pd.DataFrame({
            'MICE': df_im_imputed[col].values[col_mask],
            'Median': df_im_median[col].values[col_mask],
            'Diff': df_im_imputed[col].values[col_mask] - df_im_median[col].values[col_mask]
        })
        print(f"\n{col} (median of non-missing: {df_im[col].median():.2f}):")
        print(comparison.to_string())


# Seems like MICE is imputing pretty high values, compared to the median. for example for TC_PD1+, 
# mice imputes extremely high values. Mice seems to overfit to correlations in that specific row?
# adding mean-matching in the kernel to only impute values that are observed in the original data, choosing 5 neighbours. might help with this issue.
# and adding more datasets to the kernel, to get more stable imputations. 
# lets try 5 datasets, with 10 iterations, + updating miceforest  from 5.2.6 to 6.0.5 and specifing num neigbours =5 instead of 1 dataset.


# Now we have less extreme differences, but still some features have qute high differences compared to median.
# max differene before was 49. Now it is 20.

# Correlation between imputed values only
corr_imputed = np.corrcoef(median_flat, mice_flat)[0, 1]

plt.figure(figsize=(8, 6))
plt.scatter(median_flat, mice_flat, alpha=0.5, s=20, color='steelblue', edgecolors='none')
plt.plot([median_flat.min(), median_flat.max()],
         [median_flat.min(), median_flat.max()],
         'r--', linewidth=2, label='Perfect agreement (y=x)')
plt.text(0.05, 0.95, f'Correlation: {corr_imputed:.4f}\n(n={len(mice_flat)} imputed values)',
         transform=plt.gca().transAxes, fontsize=11,
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
plt.xlabel('Median Imputed Values', fontsize=12, fontweight='bold')
plt.ylabel('MICE Imputed Values', fontsize=12, fontweight='bold')
plt.title('Imputed Values Comparison: Median vs MICE', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\nCorrelation (imputed values only): {corr_imputed:.4f}")



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

print("Running PyOD outlier detection on immunological dataset")
# Using imputed immunological dataset without ID columns
patient_ids = df_im_imputed["Patient"].values
timepoints = df_im_imputed["Timepoint"].values
X_pyod = X_pyod = df_im_imputed.drop(columns=exclude_cols).values

contamination = 0.05    # standard contamination fraction from pyod library (assuming 5% of samples are outliers, can be adjusted based on domain knowledge or expected outlier proportion)
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

# Summary table
summary_df = pd.DataFrame({
    'Algorithm': list(outlier_counts.keys()),
    'Outliers Found': list(outlier_counts.values()),
    'Percentage of Dataset': [f"{(count/len(X_pyod)*100):.2f}%" for count in outlier_counts.values()]
})
print("\nOutlier Detection Summary:")
print(summary_df.to_string(index=False))
print("\n")


# Plot for each model 
# scale data before!
scaler = StandardScaler()
X_pyod_scaled = scaler.fit_transform(X_pyod)

pca_pyod = PCA(n_components=2, random_state=42)
X_2d = pca_pyod.fit_transform(X_pyod_scaled)
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
axes = axes.flatten()

for idx, (name, data) in enumerate(results.items()):
    ax = axes[idx]

    # Create dataframe
    plot_df = pd.DataFrame({
        'PC1': X_2d[:, 0],
        'PC2': X_2d[:, 1],
        'is_outlier': data['predictions'] == 1,
        'Score': data['scores_normalized'],
        'Patient': patient_ids,
        'Timepoint': timepoints
    })

    # Split inliers and outliers
    inliers_df = plot_df[~plot_df['is_outlier']]
    outliers_df = plot_df[plot_df['is_outlier']]

    # Plot inliers in blue
    ax.scatter(
        inliers_df['PC1'], inliers_df['PC2'],
        c='steelblue', s=30, alpha=0.5, label='Inliers', edgecolors='none'
    )

    # Plot outliers colored by score (high score = high certainty)
    scatter = ax.scatter(
        outliers_df['PC1'], outliers_df['PC2'],
        c=outliers_df['Score'], cmap='Reds', s=80, alpha=0.8,
        edgecolors='black', linewidths=0.5, label='Outliers'
    )

    # Add colorbar for outlier scores
    if len(outliers_df) > 0:
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label('Outlier Score', fontsize=10)

    # Annotate only outliers with Patient-Timepoint labels
    for _, row in outliers_df.iterrows():
        ax.annotate(
            f"P{int(row['Patient'])}-T{int(row['Timepoint'])}",
            (row['PC1'], row['PC2']),
            fontsize=8,
            xytext=(5, 5),
            textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=0.8)
        )

    ax.set_xlabel(f'PC1 ({pca_pyod.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11)
    ax.set_ylabel(f'PC2 ({pca_pyod.explained_variance_ratio_[1]*100:.1f}%)', fontsize=11)
    ax.set_title(f'{name}\n{outlier_counts[name]} outliers detected', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)

plt.tight_layout()
plt.suptitle('PyOD Outlier Detection - Immunological Dataset', fontsize=16, fontweight='bold', y=1.02)
plt.show()


#%%############## Consensesous analysis ############

# Create consensus matrix
consensus_matrix = np.column_stack([
    results['IsolationForest']['predictions'],
    results['LOF']['predictions'],
    results['ECOD']['predictions'],
    results['COPOD']['predictions']
])

# Count how many algorithms flagged each sample
consensus_count = consensus_matrix.sum(axis=1)

# Count samples with highest scores (most outlier-like) across all algorithms
print("\nConsensus Analysis:")
print(f"Number of samples flagged by at least 3 algorithms: {(consensus_count >= 3).sum()} out of {len(consensus_count)} ({(consensus_count >= 3).mean()*100:.2f}%)")
print(f"Number of samples flagged by all 4 algorithms: {(consensus_count == 4).sum()} out of {len(consensus_count)} ({(consensus_count == 4).mean()*100:.2f}%)") 
print(" ")       

# Samples flagged by all 4 algorithms
average_scores = np.mean([r['scores_normalized'] for r in results.values()], axis=0)
all_flagged = pd.DataFrame({
    'Patient': patient_ids, 'Timepoint': timepoints, 'Avg_Score': average_scores})
print("Samples flagged by all 4 algorithms (sorted by avg score):")
print(all_flagged.to_string(index=False))


# Bar plot of consensus outliers by timepoint
plt.figure(figsize=(10, 6))
outlier_counts = all_flagged['Timepoint'].value_counts().sort_index().reset_index()
outlier_counts.columns = ['Timepoint', 'Count']
sns.barplot(data=outlier_counts, x='Timepoint', y='Count', color='teal')
plt.xlabel('Timepoint', fontsize=12, fontweight='bold')
plt.ylabel('Number of Consensus Outliers', fontsize=12, fontweight='bold')
plt.title('Consensus Outliers by Timepoint (Across all 4 Algorithms)', fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()                                                        

# Plot showing algorithm and number flagged samples
plt.figure(figsize=(8, 5))
algorithms = list(outlier_counts.keys())
counts = list(outlier_counts.values())
sns.barplot(x=algorithms, y=counts, hue=algorithms, palette='crest', legend=False)
plt.xlabel('Outlier Detection Algorithm', fontsize=12, fontweight='bold')
plt.ylabel('Number of Outliers Detected', fontsize=12, fontweight='bold')
plt.title('Outliers Detected by Each Algorithm', fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()




#%%######### PyOD Ensemble Outlier Detection (Zyran approach) - Immunological Dataset ########

# This section uses the pre-built outlier detection framework from:
# https://gitlab.com/zryan.rz/master_outlier_detection_h23
#
# Pipeline:
#   1. Median-imputed immunological data -> StandardScaler
#   2. GEC (Gaussian Ensemble Comparison): fits all candidate algorithms and
#      selects the 6 most *dissimilar* ones to form a diverse ensemble
#   3. visualiser_OD: fits the 6 selected algorithms, aggregates scores via
#      median probability across algorithms, and produces three plots:
#        - PCA biplot (hoggorm NIPALS PCA)
#        - Scatter: median probability vs. average confidence (marker size =
#          std of confidence, colour = std of probability)
#        - Pairplots of PC1-5 coloured by median probability / confidence
#   Contamination is fixed at 0.1 (standard PyOD default).

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
X_sc_ens = pd.DataFrame(scaler_ens.fit_transform(X_ens), columns=X_ens.columns)

# --- Build candidate algorithm list (matches notebook) ---
contamination = 0.1
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
    X_sc_ens.values,
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



# --- Step 3: Estimate contamination with gammaGMM ---
# Fits a Bayesian Dirichlet-Process GMM on anomaly scores from 3 base detectors
# to derive a posterior distribution for the contamination fraction γ.
# The median of that posterior is used as the point estimate.
from pyod_zyran.gammaGMM import run_gammaGMM

print("\nEstimating contamination with gammaGMM...")
gamma_samples = run_gammaGMM(
    X_sc_ens.values,
    ad_list=[KNN_od(), IForest_od(), LOF_od()],
    cpu=1,
    verbose=False
)
contamination_est = float(np.median(gamma_samples))
print(f"Estimated contamination: {contamination_est:.4f} (default was 0.1)")

# Plot posterior distribution of contamination
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(gamma_samples, bins=60, color=sns.color_palette("mako", 1)[0], edgecolor="none", alpha=0.85)
ax.axvline(contamination_est, color="black", linestyle="--", lw=1.5,
           label=f"Median = {contamination_est:.4f}")
ax.axvline(0.1, color="grey", linestyle=":", lw=1.2, label="Default = 0.10")
ax.set_xlabel("Contamination fraction (γ)")
ax.set_ylabel("Posterior samples")
ax.set_title("gammaGMM: Posterior distribution of contamination", fontweight="bold")
ax.legend()
plt.tight_layout()
plt.show()

# --- Step 5: Re-initialise selected algorithms with estimated contamination ---
initialized_modules_cont = [
    algo_class_map[name](contamination=contamination_est)
    for name in final_selected_algos
    if name in algo_class_map
]

# --- Step 6: visualiser_OD with estimated contamination (primary result) ---
print(f"Running visualiser_OD with gammaGMM contamination={contamination_est:.4f}...")
no_od_df_cont, y_prob_mean_cont, y_conf_mean_cont, y_prob_arr_cont, y_conf_arr_cont, train_scores_ens_cont = visualiser_OD(
    X_sc_ens,
    initialized_modules_cont,
    patient_labels,
    visualize=True,
    figure_append_name='Contamination'
)

# --- Summary ---
print(f"\n=== Outlier Detection Summary (gammaGMM contamination={contamination_est:.4f}) ===")
for n in [1, 3, len(initialized_modules_cont)]:
    label = f"Flagged by >= {n} algorithm{'s' if n > 1 else ''}"
    print(f"{label}: {(no_od_df_cont['No. OD Detected'] >= n).sum()}")

print("\nTop 20 most-flagged samples:")
print(no_od_df_cont.sort_values("No. OD Detected", ascending=False).head(20))


#%% Data exploration raw clinical dataset





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
    """Extract numeric value from continuous scale entries (e.g., pain_scale 1-10,
    and health_status_today 0-100). Comma is German decimal ("9,7" = 9.7).
    Handles: German decimals, ranges "20-30" -> midpoint, trailing text "40 (left side)" -> 40.
    """
    s = series.astype(str).str.strip()

    def parse_entry(val):
        if val in ('nan', '', 'None'):
            return np.nan
        # Range: "20-30", "10 - 20"
        m = re.match(r'^(\d+[.,]?\d*)\s*[-–]\s*(\d+[.,]?\d*)', val)
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
        ('Foot',            ['foot']),
        ('Forefoot',        ['forefoot']),
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
        return ', '.join(results)

    return series.apply(parse_entry)


def split_filter_column(df, col_name='filter'):
    """Split filter column into filter_mm (float) and filter_material (Cu/Al).
    Handles German decimal commas, duplicate entries, and various formats.
    """
    col_idx = df.columns.get_loc(col_name)

    def parse_filter(val):
        if pd.isna(val):
            return pd.NA, pd.N
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

# Exclude patients with missing/invalid Response values
invalid_response_mask = df_cl_clean['Response'].isna() | df_cl_clean['Response'].isin(['n.D', 'n.D.'])
patients_invalid_response = df_cl_clean.loc[invalid_response_mask, 'Patient'].unique()
df_cl_clean = df_cl_clean[~df_cl_clean['Patient'].isin(patients_invalid_response)]
print(f"Excluded {len(patients_invalid_response)} patients with invalid Response: {patients_invalid_response}")

# Drop no longer needed columns 'Unnamed: 2', ?
cols_to_drop = ['Unnamed: 0', 'Comments questionnaire', 'further comments']
df_cl_clean = df_cl_clean.drop(columns=[c for c in cols_to_drop if c in df_cl_clean.columns])

print(f"\nAfter exclusions: {df_cl_clean['Patient'].nunique()} patients, {len(df_cl_clean)} rows")


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

    # Functional limitations
    "Schwierigkeiten körperlicher Anstrengung": "difficulty_physical_exertion",
    "Schwierigkeiten bei längerem Spaziergang": "difficulty_long_walk",
    "Schwierigkeiten bei kurzer Strecke": "difficulty_short_distance",
    "tagsüber liegen oder sitzen": "difficulty_laying_sitting_daytime",
    "Hilfe Essen, Anziehen, Waschen, Toilette": "help_with_adl",
    "Einschränkung bei Alltag": "limitation_daily_life",
    "Einschränkung bei Hobbys/Freizeit": "limitation_leisure",

    # General health symptoms
    "kurzatmig": "shortness_of_breath", "Schmerzen": "pain_general",
    "Ausruhen": "need_to_rest", "Schlafstörungen": "sleep_disturbance",
    "Schwach": "weakness", "Appetitmangel": "loss_of_appetite",
    "Übelkeit": "nausea", "Erbrochen": "vomiting",
    "Verstopfung": "constipation", "Durchfall": "diarrhea", "Müde": "fatigue",

    # Cognitive/psychological impairments
    "Beeinträchtigung im Alltag": "impairment_daily_life",
    "Konzentrationsstörung": "concentration_difficulty",
    "angespannt": "feeling_tense", "Sorgen gemacht": "worry",
    "reizbar": "irritability", "niedergeschlagen": "depressed",
    "Erinnerungsprobleme": "memory_problems",

    # Social impairments
    "Beeinträchtigung Familienleben": "family_life_impairment",
    "Beeinträchtigung Zusammensein Menschen": "social_impairment",
    "finanzielle Schwierigkeiten": "financial_difficulties",

    # Quality of life
    "Gesundheitszustand insgesamt": "health_status_overall",
    "Lebensqualität insgesamt": "life_quality_overall",

    # Daily tasks (physical function)
    "Strecken": "stretching", "Heben und Tragen von 10kg": "lift_10kg",
    "Waschen und Abtrocknen": "wash_and_dry", "Bücken": "bending",
    "Haare Waschen im Waschbecken": "wash_hair", "Sitzen 1h": "sit_1h",
    "Stehen 30min": "stand_30min", "Aufsetzen": "sit_up",
    "Strümpfe an-/ausziehen": "put_on_socks",
    "Gegenstand aus Sitzposition aufheben": "pick_object_seated",
    "Gegenstand auf Tisch stellen": "place_object_table",
    "Schnell laufen 100m": "run_fast_100m",

    # EQ-5D-like questionnaire
    "Beweglichkeit/Mobilität": "mobility", "für sich selbst sorgen": "self_care",
    "alltägliche Tätigkeiten": "daily_activities",
    "Schmerzen/körperliche Beschwerden": "pain_discomfort",
    "Angst/Niedergeschlagenheit": "anxiety_depression",

    # Clinical outcomes and treatment variables
    "Allgemeinzustand Gesundheut HEUTE": "health_status_today",
    "Besserung nach Nachuntersuchung laut Arztbrief in %": "improvement_percent",
    "Diagnosis": "diagnosis", "Target volume": "target_volume",
    "single fraction": "single_fraction",
    "kummulative dose (x) - if two targets were applied": "cumulative_dose",
    "FHA": "fha", "kV": "kv", "mA": "ma", "Filter": "filter", "Response": "response",
}
df_cl_clean = df_cl_clean.rename(columns=clinical_names)
print(f"Columns renamed: {len(clinical_names)}")

# Baseline copy: raw data with timepoint column and English names (before any transforms)
df_cl_reduced = df_cl_clean.copy()


#%% Step 4: Clean null markers

# Fix duplicate entries separated by newlines (e.g., "3\n3" → "3")
print("=== Removing newline-duplicated entries ===")
for col in df_cl_clean.columns:
    str_col = df_cl_clean[col].astype(str)
    has_newline = str_col.str.contains('\n', na=False)
    if has_newline.sum() > 0:
        def dedup_newline(val):
            if pd.isna(val):
                return val
            s = str(val)
            if '\n' not in s:
                return val
            parts = [p.strip() for p in s.split('\n') if p.strip()]
            if len(parts) > 1 and len(set(parts)) == 1:
                return parts[0]  # all parts identical → keep one
            return parts[0]  # different parts → keep first
        df_cl_clean[col] = df_cl_clean[col].apply(dedup_newline)
        print(f"  {col}: fixed {has_newline.sum()} entries with newlines")


# Replace German missing markers across ALL columns: k.A./ka/kA and n.D./n.D
null_pattern = r'^([kK]\.?[aA]\.?|[nN]\.?[dD]\.?)$'
print("\n=== Replacing null markers ('kA' and 'nD' varations) ===")
for col in df_cl_clean.columns:
    str_col = df_cl_clean[col].astype(str).str.strip()
    mask = str_col.str.match(null_pattern, na=False)
    if mask.sum() > 0:
        print(f"  {col}: replaced {mask.sum()} null markers")
        df_cl_clean.loc[mask, col] = pd.NA



#%% Step 5: Extract/convert numeric values from mixed text/number columns

# Ordinal questionnaire columns (scale 1-4 or 1-5): multi-select "1,2" -> avg
# Skip: pain_points (categorical), pain_scale (continuous, comma = decimal), health_status_today (0-100 scale)
all_cols = df_cl_clean.loc[:, 'pain_under_load':'health_status_today'].columns
ordinal_cols = [c for c in all_cols if c not in ('pain_points', 'pain_scale', 'health_status_today')]

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


# Continuous columns: comma = German decimal, ranges -> midpoint
# pain_scale (1-10): "9,7" -> 9.7
# health_status_today (0-100): "20-30" -> 25, "40 (left side)" -> 40
print("\n=== Extracting numeric values (continuous columns) ===")
for col in ['pain_scale', 'health_status_today']:
    if col in df_cl_clean.columns:
        original_numeric = pd.to_numeric(df_cl_clean[col], errors='coerce')
        extracted = extract_continuous(df_cl_clean[col])
        was_text = original_numeric.isna() & extracted.notna()
        if was_text.sum() > 0:
            print(f"  {col}: extracted {was_text.sum()} values from text entries")
        df_cl_clean[col] = extracted



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
print(f"  Top 10: {df_cl_clean['pain_points'].value_counts().head(10).to_dict()}")

# Filter: split into filter_mm (thickness) and filter_material (Cu/Al)
df_cl_clean = split_filter_column(df_cl_clean)
print(f"\nFilter mm: {sorted(df_cl_clean['filter_mm'].dropna().unique())}")
print(f"Filter material: {df_cl_clean['filter_material'].value_counts().to_dict()}")

df_cl_clean['cumulative_dose'] = df_cl_clean['cumulative_dose'].apply(parse_cumulative_dose)
print(f"\nCumulative dose: {sorted(df_cl_clean['cumulative_dose'].dropna().unique())}")

# Gender: standardize 'w' (German: weiblich) to 'f' (female)
df_cl_clean['gender'] = df_cl_clean['gender'].replace('w', 'f')
print(f"\nGender: {df_cl_clean['gender'].value_counts().to_dict()}")

# BMI: split overweight_bmi -> overweight (ja/nein) + bmi (float)
df_cl_clean = split_bmi_column(df_cl_clean)
missing_bmi = (df_cl_clean['bmi'].isna() & df_cl_clean['overweight'].notna()).sum()
print(f"BMI split: {missing_bmi} patients with overweight status but missing BMI value")

# Symptoms duration: German strings to numeric months
df_cl_clean['symptoms_months'] = parse_symptoms_duration(df_cl_clean['symptoms_months'], df_cl_clean['date'])
print(f"Symptoms: range {df_cl_clean['symptoms_months'].min():.0f}-{df_cl_clean['symptoms_months'].max():.0f} months, "
      f"{df_cl_clean['symptoms_months'].isna().sum()} missing")

# Previous therapy: comma-separated codes (1-7) to binary columns
df_cl_clean = encode_therapy_columns(df_cl_clean)
therapy_cols = [f'previous_therapy_{i}' for i in range(1, 8)]
print(f"Therapy encoding: {df_cl_clean[therapy_cols].sum().to_dict()}")


# Improvement percent: fill 0 for patients with "no improvement" response
no_improvement_mask = (
    df_cl_clean['response'].str.lower().str.startswith('no', na=False) &
    df_cl_clean['improvement_percent'].isna()
)
df_cl_clean.loc[no_improvement_mask, 'improvement_percent'] = 0
print(f"Improvement: filled {no_improvement_mask.sum()} NaN values with 0 for 'no improvement' responses")


#%% Step 7: Remove empty questionnaire rows

questionnaire_cols = df_cl_clean.loc[:, 'symptoms_months':'health_status_today'].columns
empty_questionnaire_mask = (
    df_cl_clean['date'].notna() &
    df_cl_clean[questionnaire_cols].isna().all(axis=1)
)
print(f"\nRemoving {empty_questionnaire_mask.sum()} empty questionnaire rows")
if empty_questionnaire_mask.sum() > 0:
    print(df_cl_clean.loc[empty_questionnaire_mask, ['Patient', 'Timepoint', 'date', 'response', 'diagnosis']])
df_cl_clean = df_cl_clean[~empty_questionnaire_mask]


#%% Step 8: final dtype conversion
# Cleaner auto-detects: dates -> datetime, numeric strings -> numeric, null strings -> NaN
# measurement_timepoint is a panel group ID, not a date — keep as string
# remove columns with more than 35% missing

categorical_cols = [
    'gender', 'overweight', 'pain_points', 'diagnosis',
    'target_volume', 'target_side', 'filter_material', 'response'
]

# Ensure id/date columns keep appropriate types
if 'Patient' in df_cl_clean.columns:
    df_cl_clean['Patient'] = pd.to_numeric(df_cl_clean['Patient'], errors='coerce').astype('Int64')
if 'Timepoint' in df_cl_clean.columns:
    df_cl_clean['Timepoint'] = pd.to_numeric(df_cl_clean['Timepoint'], errors='coerce').astype('Int64')
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



TableReport(df_cl_clean, max_plot_columns=100)

#%%
# Clean response variable to classes containing CR, PR or NI for no improvement.

# Columns with "no improvement" AND CR at the same measurement will be put in PR category
# Columns with CR AND PR at same measurement will be put in PR category.
# This will most likely lead to a class imbalace where partial response is biggest?

# Columns with notes indiciating partial response, change into PR:
# recovery only on the right side", "improvement", "initial improvement", "subtotal remission" needs to be assigned to PR
# one column with typo "no imrovement" needs to be assigned to NI

def create_response_category(df, response_col='response'):
    """Create variable response_category (CR, PR, NI) from the raw "response" column.
    With groups CR, PR, NI. Mixed responses (CR+NI, CR+PR) -> PR.
    """
    df = df.copy()
    clean = df[response_col].astype(str).str.strip().str.lower()
    
    # Fix specific entries and typo
    pr_terms = ['recovery only on the right side', 'improvement', 'initial improvement', 'subtotal remission']
    clean = clean.replace(pr_terms, 'pr')
    clean = clean.str.replace('no imrovement', 'no improvement')
    
    # Detect patterns
    has_cr = clean.str.contains(r'\bcr\b', na=False)
    has_pr = clean.str.contains(r'\bpr\b', na=False)
    has_ni = clean.str.contains(r'no\s*imp', na=False)
    
    # Mixed responses -> PR
    clean = clean.where(~(has_cr & (has_ni | has_pr)), 'pr')
    
    # Recalculate after fixes
    has_cr = clean.str.contains(r'\bcr\b', na=False)
    has_ni = clean.str.contains(r'no\s*imp', na=False)
    
    # Assign categories (CR > PR > NI)
    df['response_category'] = pd.NA
    df.loc[has_ni, 'response_category'] = 'NI'
    df.loc[clean.str.contains(r'\bpr\b', na=False), 'response_category'] = 'PR'
    df.loc[has_cr, 'response_category'] = 'CR'
    
    print(f"Response categories:\n{df['response_category'].value_counts(dropna=False)}")
    return df

# creating cleaned response category
df_cl_reduced1 = create_response_category(df_cl_reduced, response_col='response')


#%% Plot response category distribution (unique patients)
response_per_patient = df_cl_reduced1.drop_duplicates(subset='Patient')[['Patient', 'response_category']]
counts = response_per_patient['response_category'].value_counts()

fig, ax = plt.subplots(figsize=(6, 4))
counts.plot(kind='bar', ax=ax, color=sns.color_palette('mako', n_colors=len(counts)))
ax.set_title('Response Category Distribution (unique patients) in Clin. Dataset')
ax.set_xlabel('Response Category')
ax.set_ylabel('Number of Patients')
for i, (cat, count) in enumerate(counts.items()):
    ax.text(i, count + 1, str(count), ha='center', fontweight='bold')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
plt.tight_layout()
plt.show()

print(f"Total unique patients: {len(response_per_patient)}")
print(counts.to_string())




#%%  improvement_percent variable as a target variable

def clean_improvement_percent(df, col='improvement_percent'):
    """Clean improvement_percent column: handle ranges, L/R splits, special entries.
    Returns df with numeric improvement_percent (0-100).
    """
    df = df.copy()
    raw = df[col].astype(str).str.strip()
    
    # Step 1: Handle specific known entries
    # "nicht schmerzfrei" (not pain-free) -> 0
    raw = raw.replace({'nicht schmerzfrei': '0', 'nan': pd.NA, '': pd.NA})
    
    # "02/2017: 90; 08/2017: 50" -> take latest value (50)
    date_pattern = raw.str.contains(r'\d{2}/\d{4}', na=False)
    if date_pattern.any():
        # Extract last number after the last colon
        raw[date_pattern] = raw[date_pattern].str.extract(r'.*:\s*(\d+)')[0]
    
    # Step 2: Handle L/R entries (e.g., "L: 100 R:0", "links: 100%; rechts: 80-90")
    lr_pattern = raw.str.contains(r'[LlRr]|links|rechts', na=False)
    
    def parse_lr(entry):
        """Extract all numbers/ranges from L/R entries and average them."""
        import re
        # Find all numbers or ranges
        parts = re.findall(r'(\d+)\s*-\s*(\d+)|(\d+)', entry)
        values = []
        for start, end, single in parts:
            if start and end:  # range like 80-90
                values.append((float(start) + float(end)) / 2)
            elif single:  # single number
                values.append(float(single))
        return np.mean(values) if values else np.nan
    
    if lr_pattern.any():
        raw[lr_pattern] = raw[lr_pattern].apply(parse_lr)
    
    # Step 3: Handle simple ranges (e.g., "80-90")
    range_pattern = raw.astype(str).str.match(r'^\s*[<>~]?\s*\d+\s*-\s*\d+\s*$', na=False)
    if range_pattern.any():
        start = raw[range_pattern].str.extract(r'(\d+)\s*-')[0].astype(float)
        end = raw[range_pattern].str.extract(r'-\s*(\d+)')[0].astype(float)
        raw[range_pattern] = (start + end) / 2
    
    # Step 4: Convert to numeric
    df[col] = pd.to_numeric(raw, errors='coerce')
    
    # Step 5: Remove patients with missing improvement_percent
    before = df['Patient'].nunique()
    missing_patients = df[df[col].isna()]['Patient'].unique()
    df = df[~df['Patient'].isin(missing_patients)]
    after = df['Patient'].nunique()
    
    print(f"\n=== Improvement Percent Cleaning ===")
    print(f"Patients removed (missing): {len(missing_patients)}")
    print(f"Patients remaining: {after} (was {before})")
    print(f"\n=== Improvement Percent Stats ===")
    print(df[col].describe())
    print(f"\nValue counts:\n{df[col].value_counts().sort_index()}")
    
    return df

# Usage:
df_cl_reduced2 = clean_improvement_percent(df_cl_reduced)


#%% Reduce clinical dataset?

# remove all columns with more than 25% missing variables. 


missing_pct = df_cl_clean.isna().mean()
cols_to_keep = missing_pct[missing_pct <= 0.35].index
df_cl_red= df_cl_clean[cols_to_keep].copy()

# Verification
print(f"Columns before: {len(df_cl_clean.columns)}")
print(f"Columns after: {len(df_cl_red.columns)}")
print(f"Dropped {len(df_cl_clean.columns) - len(df_cl_red.columns)} columns with >35% missing:")
print(missing_pct[missing_pct > 0.35].sort_values(ascending=False))

TableReport(df_cl_red, max_plot_columns=100)


#%% Basline model for immunological dataset - CatBoost

TableReport(df_im_reduced, max_plot_columns=130)
TableReport(df_cl_reduced1, max_plot_columns=130) # using category as target
TableReport(df_cl_reduced2, max_plot_columns=130) # using improvement_percent as target

#%% Step 9: Prepare baseline datasets for modeling

# --- First, check what we have ---
print("=== Dataset overview: Target variable: Response Category ===")
print(f"Immunological (df_im_reduced): {df_im_reduced['Patient'].nunique()} patients, {len(df_im_reduced)} rows")
print(f"Clinical (df_cl_reduced): {df_cl_reduced1['Patient'].nunique()} patients, {len(df_cl_reduced1)} rows")

# --- Check common patients ---
im_patients = set(df_im_reduced['Patient'].unique())
cl_patients = set(df_cl_reduced1['Patient'].unique())
common_patients = im_patients & cl_patients

print(f"\n=== Patient overlap ===")
print(f"Immunological only: {len(im_patients - cl_patients)}")
print(f"Clinical only: {len(cl_patients - im_patients)}")
print(f"Common patients: {len(common_patients)}")


print('')
print("=== Dataset overview: Target variable: Improvement in Percent ===")
print(f"Immunological (df_im_reduced): {df_im_reduced['Patient'].nunique()} patients, {len(df_im_reduced)} rows")
print(f"Clinical (df_cl_reduced2): {df_cl_reduced2['Patient'].nunique()} patients, {len(df_cl_reduced2)} rows")

# --- Check common patients ---
im_patients = set(df_im_reduced['Patient'].unique())
cl_patients = set(df_cl_reduced2['Patient'].unique())
common_patients = im_patients & cl_patients

print(f"\n=== Patient overlap ===")
print(f"Immunological only: {len(im_patients - cl_patients)}")
print(f"Clinical only: {len(cl_patients - im_patients)}")
print(f"Common patients: {len(common_patients)}")


#%% Check timepoint availability for common patients

# Filter to common patients
df_im_common = df_im_reduced[df_im_reduced['Patient'].isin(common_patients)]
df_cl_common = df_cl_reduced2[df_cl_reduced2['Patient'].isin(common_patients)]

# Check which timepoints each patient has in BOTH datasets
im_pt = df_im_common.groupby('Patient')['Timepoint'].apply(set)
cl_pt = df_cl_common.groupby('Patient')['Timepoint'].apply(set)

# Combine to find common timepoints per patient
common_timepoints = pd.DataFrame({'im': im_pt, 'cl': cl_pt}).dropna()
common_timepoints['both'] = common_timepoints.apply(lambda x: x['im'] & x['cl'], axis=1)

print("\nTimepoint availability (using improvement_percent as target")
# Count patients with T1+T2+T3
has_t123 = common_timepoints['both'].apply(lambda x: {1, 2, 3}.issubset(x)).sum()
print(f"Patients with T1 + T2 + T3: {has_t123}")

# Count patients with T1+T2
has_t12 = common_timepoints['both'].apply(lambda x: {1, 2}.issubset(x)).sum()
print(f"Patients with T1 + T2: {has_t12}")

# Count patients with just T1
has_t1 = common_timepoints['both'].apply(lambda x: 1 in x).sum()
print(f"Patients with T1: {has_t1}")

"""
Immunological: 264 patients, 822 rows
Clinical (Response category as target): 206 patients, 687 rows

Using Response Cateory as Target
=== Patient overlap ===
Immunological only: 58
Clinical only: 0
Common patients: 206

Patients with T1 + T2 + T3: 84
Patients with T1 + T2: 140
Patients with T1: 182
--------------------------------------------
Using Improvement in percent as Target
Immunological: 264 patients, 829 rows
Clinical (df_cl_reduced2): 127 patients, 492 rows

=== Patient overlap ===
Immunological only: 138
Clinical only: 0
Common patients: 127

Patients with T1 + T2 + T3: 62
Patients with T1 + T2: 99
Patients with T1: 111
"""


#%% BASELINE DATASETS FOR CATBOOST MODEL USING RESPONSE CATEGORY AS TARGET
# T1 only (pre-treatment) — one row per patient, raw data, no imputation
# check if pre-treatment measurements alone predict treatment response?

from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
import shap
import numpy as np
import pandas as pd

target_col = 'response_category'
id_cols = ['Patient', 'Timepoint']
leaky_cols = ['response', 'improvement_percent']

# --- Step 1: Add response_category to immunological dataset (patient-level label) ---
response_map = df_cl_reduced1[['Patient', 'response_category']].drop_duplicates(subset='Patient')
df_im_with_response = df_im_reduced.merge(response_map, on='Patient', how='left')
df_im_with_response = df_im_with_response.dropna(subset=[target_col])

# --- Step 2: Create combined dataset (all timepoints), merge on Patient + Timepoint ---
# response_category already in df_im_with_response from Step 1, drop from clinical to avoid duplication
df_cl_for_merge = df_cl_reduced1.drop(columns=[target_col], errors='ignore')
df_combined = df_im_with_response.merge(
    df_cl_for_merge, on=['Patient', 'Timepoint'], how='inner',
    suffixes=('_im', '_cl'))

# --- Step 3: Filter to T1 for baseline ---
df_im_baseline = df_im_with_response[df_im_with_response['Timepoint'] == 1].copy()
#df_im_baseline = df_im_baseline.dropna(subset='response_category')
df_cl_baseline = df_cl_reduced1[df_cl_reduced1['Timepoint'] == 1].copy()
df_combined_baseline = df_combined[df_combined['Timepoint'] == 1].copy()

TableReport(df_im_baseline, max_plot_columns=180)
TableReport(df_cl_baseline, max_plot_columns=180)
TableReport(df_combined_baseline, max_plot_columns=180)

#%% CatBoost baseline: 5-fold StratifiedKFold, built-in metrics
# T1 only = one row per patient, so StratifiedKFold preserves class distribution

def run_catboost_baseline(df_model, target_col, name):
    """Run 5-fold StratifiedKFold CatBoost classifier on T1 data. No tuning.
    Returns results_df (per-fold + mean), last trained model, X, y_pred.
    """
    # Base exclusion list
    exclude = ['Patient', 'Timepoint', target_col]
    
    # Add ALL leaky columns: response and improvement_percent (any variant)
    # Catches: 'response', 'response_T1', 'response_T2', 'response_im', 'response_cl'
    # Catches: 'improvement_percent', 'improvement_percent_T1', 'improvement_percent_T2', etc.
    leaky_patterns = ['response', 'improvement_percent']
    for col in df_model.columns:
        for pattern in leaky_patterns:
            if pattern in col.lower():
                exclude.append(col)
                break
    
    # Remove duplicates and non-existent columns
    exclude = list(set([col for col in exclude if col in df_model.columns]))
    
    
    feature_cols = [c for c in df_model.columns if c not in exclude]
    X = df_model[feature_cols].copy()
    y = df_model[target_col].copy()

    # Convert categoricals to string (CatBoost requirement for cat_features)
    for col in X.select_dtypes(include=['category', 'object']).columns:
        X[col] = X[col].astype(str)
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()

    print(f"\n{'='*60}")
    print(f"  CatBoost Baseline: {name}")
    print(f"{'='*60}")
    print(f"  Samples: {len(X)}, Features: {X.shape[1]}")

    skf = StratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
    fold_results = []
    y_pred = pd.Series(index=X.index, dtype='object')

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        train_pool = Pool(X_train, y_train, cat_features=cat_cols)
        test_pool = Pool(X_test, y_test, cat_features=cat_cols)

        model = CatBoostClassifier(
            random_seed=42,
            verbose=0,
            iterations=0 # my laptop crashes due to large dataset
        )
        model.fit(train_pool, eval_set=test_pool, use_best_model=False)

        # Predict class labels
        preds = model.predict(test_pool, prediction_type='Class').flatten()
        y_pred.iloc[test_idx] = preds

        # Compute metrics using CatBoost eval_metrics (post-training)
        metrics = model.eval_metrics(
            test_pool,
            ['Accuracy', 'TotalF1:average=Weighted', 'AUC:type=Mu', 'MCC']
        )

        fold_result = {
            'Fold': fold + 1,
            'Accuracy': metrics['Accuracy'][-1],
            'F1_total': metrics['TotalF1:average=Weighted'][-1],
            'AUC': metrics['AUC:type=Mu'][-1],
            'MCC': metrics['MCC'][-1],
            'Train_size': len(train_idx),
            'Test_size': len(test_idx)
        }
        fold_results.append(fold_result)

        print(f"  Fold {fold+1}: Acc={fold_result['Accuracy']:.4f}  "
              f"F1={fold_result['F1_total']:.4f}  "
              f"AUC={fold_result['AUC']:.4f}  "
              f"MCC={fold_result['MCC']:.4f}")

    results_df = pd.DataFrame(fold_results)

    # Add mean row
    metric_cols = ['Accuracy', 'F1_total', 'AUC', 'MCC']
    mean_row = {m: results_df[m].mean() for m in metric_cols}
    mean_row['Fold'] = 'Mean'
    results_df = pd.concat([results_df, pd.DataFrame([mean_row])], ignore_index=True)

    print(f"\n  Mean Across 5 Folds")
    for m in metric_cols:
        print(f"  {m}: {mean_row[m]:.4f}")

    return results_df, model, X, y_pred


#%%
# Run for all three datasets
print("\n" + "="*70)
print("  RUNNING CATBOOST BASELINES")
print("="*70)

target_col = 'response_category'
res_im, model_im, X_im, y_pred_im = run_catboost_baseline(
    df_im_baseline, target_col, "Immunological (T1 only)")

res_cl, model_cl, X_cl, y_pred_cl = run_catboost_baseline(
    df_cl_baseline, target_col, "Clinical (T1 only")

res_comb, model_comb, X_comb, y_pred_comb = run_catboost_baseline(
    df_combined_baseline, target_col, "Combined (T1 only")


#%% Summary results table

summary_rows = []
metric_cols = ['Accuracy', 'F1_total', 'AUC', 'MCC']
for name, res in [("Immunological", res_im),
                   ("Clinical", res_cl),
                   ("Combined", res_comb)]:
    fold_rows = res[res['Fold'] != 'Mean']
    row = {'Dataset': name}
    for m in metric_cols:
        mean_val = fold_rows[m].mean()
        std_val = fold_rows[m].std()
        row[m] = f"{mean_val:.4f} ± {std_val:.4f}"
    summary_rows.append(row)

df_summary = pd.DataFrame(summary_rows)
print("\n")
print("  BASELINE CATBOOST MODEL (T1) RESULTS SUMMARY")
print("="*70)
print(df_summary.to_string(index=False))


#%% SHAP analysis for each model
# Compute SHAP values and plot summary for each dataset

def compute_and_plot_shap(model, X, name):
    """Compute SHAP values and create summary plot."""
    print(f"\n=== SHAP Analysis: {name} ===")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Convert list of arrays to numpy array for consistent shape handling
    if isinstance(shap_values, list):
        shap_values = np.array(shap_values)
    print(f"  SHAP values shape: {shap_values.shape}, X shape: {X.shape}")

    # Summary plot (bar) - mean absolute SHAP across all classes
    shap.summary_plot(shap_values, X, plot_type="bar",
                      class_names=list(model.classes_),
                      show=False, max_display=25)
    plt.title(f"SHAP Feature Importance - {name}")
    plt.tight_layout()
    plt.show()
    return shap_values

shap_im = compute_and_plot_shap(model_im, X_im, "Immunological")
shap_cl = compute_and_plot_shap(model_cl, X_cl, "Clinical")
shap_comb = compute_and_plot_shap(model_comb, X_comb, "Combined")


#%% ===================================================================
# BASELINE MODEL 2: T1 + T2 (patients with both timepoints)
# Pivot to wide format: one row per patient, features suffixed with _T1 / _T2
# =====================================================================

# --- Step 1: Find patients that have BOTH T1 and T2 ---
patients_t1_im = set(df_im_with_response[df_im_with_response['Timepoint'] == 1]['Patient'])
patients_t2_im = set(df_im_with_response[df_im_with_response['Timepoint'] == 2]['Patient'])
patients_t1t2_im = patients_t1_im & patients_t2_im

patients_t1_cl = set(df_cl_reduced1[df_cl_reduced1['Timepoint'] == 1]['Patient'])
patients_t2_cl = set(df_cl_reduced1[df_cl_reduced1['Timepoint'] == 2]['Patient'])
patients_t1t2_cl = patients_t1_cl & patients_t2_cl

def pivot_t1t2(df, patients_with_both, id_col='Patient', tp_col='Timepoint'):
    """Pivot T1+T2 long format to wide: one row per patient.
    Feature columns get _T1 / _T2 suffixes. Patient-level columns (same across
    timepoints) are kept as single columns without suffix.
    """
    # Filter to patients with both T1 and T2, timepoints 1 and 2 only
    df_filt = df[
        (df[id_col].isin(patients_with_both)) &
        (df[tp_col].isin([1, 2]))
    ].copy()

    # Separate T1 and T2
    df_t1 = df_filt[df_filt[tp_col] == 1].set_index(id_col).drop(columns=[tp_col])
    df_t2 = df_filt[df_filt[tp_col] == 2].set_index(id_col).drop(columns=[tp_col])

    # Find columns that are identical across T1 and T2 (patient-level / static)
    common_patients = df_t1.index.intersection(df_t2.index)
    static_cols = []
    for col in df_t1.columns:
        if col in df_t2.columns:
            t1_vals = df_t1.loc[common_patients, col].reset_index(drop=True)
            t2_vals = df_t2.loc[common_patients, col].reset_index(drop=True)
            # Check if values are identical (treating NaN == NaN)
            if t1_vals.equals(t2_vals):
                static_cols.append(col)

    # Separate static and time-varying columns
    varying_cols = [c for c in df_t1.columns if c not in static_cols]

    # Build wide dataframe
    df_static = df_t1.loc[common_patients, static_cols]
    df_t1_varying = df_t1.loc[common_patients, varying_cols].add_suffix('_T1')
    df_t2_varying = df_t2.loc[common_patients, varying_cols].add_suffix('_T2')

    df_wide = pd.concat([df_static, df_t1_varying, df_t2_varying], axis=1)
    df_wide = df_wide.reset_index()

    print(f"  Pivot: {len(common_patients)} patients, "
          f"{len(static_cols)} static cols, "
          f"{len(varying_cols)} varying cols × 2 = {len(varying_cols)*2} cols, "
          f"total features: {df_wide.shape[1] - 1}")

    return df_wide

# --- Step 2: Pivot each dataset to wide format ---
print("\n=== Pivoting T1+T2 to wide format ===")

print("\nImmunological:")
df_im_baseline_t1t2 = pivot_t1t2(df_im_with_response, patients_t1t2_im)

print("\nClinical:")
df_cl_baseline_t1t2 = pivot_t1t2(df_cl_reduced1, patients_t1t2_cl)

# --- Step 3: Combined dataset — merge wide immunological + wide clinical ---
patients_t1t2_both = patients_t1t2_im & patients_t1t2_cl
# Re-pivot with shared patient set for consistent merge
print("\nCombined (re-pivot with shared patients):")
df_im_wide_shared = pivot_t1t2(df_im_with_response, patients_t1t2_both)
df_cl_wide_shared = pivot_t1t2(df_cl_reduced1, patients_t1t2_both)
# Drop target from clinical side to avoid duplication
df_cl_wide_for_merge = df_cl_wide_shared.drop(columns=[target_col], errors='ignore')
df_combined_baseline_t1t2 = df_im_wide_shared.merge(
    df_cl_wide_for_merge, on='Patient', how='inner', suffixes=('_im', '_cl'))

print(f"\n=== T1+T2 Wide Baseline Datasets ===")
print(f"Immunological: {df_im_baseline_t1t2.shape}, Patients: {df_im_baseline_t1t2['Patient'].nunique()}")
print(f"Clinical: {df_cl_baseline_t1t2.shape}, Patients: {df_cl_baseline_t1t2['Patient'].nunique()}")
print(f"Combined: {df_combined_baseline_t1t2.shape}, Patients: {df_combined_baseline_t1t2['Patient'].nunique()}")
print(f"Target distribution:\n{df_im_baseline_t1t2[target_col].value_counts().to_string()}")

TableReport(df_combined_baseline_t1t2, max_plot_columns=320)

#%% Run T1+T2 baselines (wide format = one row per patient, reuse run_catboost_baseline)
print("\n" + "="*70)
print("  RUNNING CATBOOST BASELINE MODELS (T1+T2 wide)")
print("="*70)

target_col= 'response_category'

res_im_t1t2, model_im_t1t2, X_im_t1t2, y_pred_im_t1t2 = run_catboost_baseline(
    df_im_baseline_t1t2, target_col, "Immunological (T1+T2)")

res_cl_t1t2, model_cl_t1t2, X_cl_t1t2, y_pred_cl_t1t2 = run_catboost_baseline(
    df_cl_baseline_t1t2, target_col, "Clinical (T1+T2)")

res_comb_t1t2, model_comb_t1t2, X_comb_t1t2, y_pred_comb_t1t2 = run_catboost_baseline(
    df_combined_baseline_t1t2, target_col, "Combined (T1+T2)")


#%% T1+T2 Summary results table
summary_rows_t1t2 = []
metric_cols = ['Accuracy', 'F1_total', 'AUC', 'MCC']
for name, res in [("Immunological", res_im_t1t2),
                   ("Clinical", res_cl_t1t2),
                   ("Combined", res_comb_t1t2)]:
    fold_rows = res[res['Fold'] != 'Mean']
    row = {'Dataset': name}
    for m in metric_cols:
        mean_val = fold_rows[m].mean()
        std_val = fold_rows[m].std()
        row[m] = f"{mean_val:.4f} ± {std_val:.4f}"
    summary_rows_t1t2.append(row)

df_summary_t1t2 = pd.DataFrame(summary_rows_t1t2)
print("\n")
print("  BASELINE CATBOOST MODEL (T1+T2 wide) RESULTS SUMMARY")
print("="*70)
print(df_summary_t1t2.to_string(index=False))


#%% Classification plots for T1+T2 models
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

target_col = 'response_category'
class_order = ['CR', 'PR', 'NI']

# --- Confusion matrices (one per dataset) ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (name, df_src, y_pred) in zip(axes, [
    ("Immunological (T1+T2)", df_im_baseline_t1t2, y_pred_im_t1t2),
    ("Clinical (T1+T2)", df_cl_baseline_t1t2, y_pred_cl_t1t2),
    ("Combined (T1+T2)", df_combined_baseline_t1t2, y_pred_comb_t1t2),
]):
    y_true = df_src[target_col].loc[y_pred.index]
    cm = confusion_matrix(y_true, y_pred, labels=class_order)
    disp = ConfusionMatrixDisplay(cm, display_labels=class_order)
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title(name, fontsize=12)
fig.suptitle("Confusion Matrices — CatBoost Baseline (T1+T2)", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

# --- Predicted vs True class distribution ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (name, df_src, y_pred) in zip(axes, [
    ("Immunological (T1+T2)", df_im_baseline_t1t2, y_pred_im_t1t2),
    ("Clinical (T1+T2)", df_cl_baseline_t1t2, y_pred_cl_t1t2),
    ("Combined (T1+T2)", df_combined_baseline_t1t2, y_pred_comb_t1t2),
]):
    y_true = df_src[target_col].loc[y_pred.index]
    true_counts = y_true.value_counts().reindex(class_order, fill_value=0)
    pred_counts = y_pred.value_counts().reindex(class_order, fill_value=0)
    x = np.arange(len(class_order))
    width = 0.35
    ax.bar(x - width/2, true_counts.values, width, label='True', color='steelblue')
    ax.bar(x + width/2, pred_counts.values, width, label='Predicted', color='salmon')
    ax.set_xticks(x)
    ax.set_xticklabels(class_order)
    ax.set_ylabel('Count')
    ax.set_title(name, fontsize=12)
    ax.legend()
fig.suptitle("True vs Predicted Class Distribution — CatBoost (T1+T2)", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()


#%% SHAP analysis for T1+T2 models
shap_im_t1t2 = compute_and_plot_shap(model_im_t1t2, X_im_t1t2, "Immunological (T1+T2)")
shap_cl_t1t2 = compute_and_plot_shap(model_cl_t1t2, X_cl_t1t2, "Clinical (T1+T2)")
shap_comb_t1t2 = compute_and_plot_shap(model_comb_t1t2, X_comb_t1t2, "Combined (T1+T2)")



# %%
