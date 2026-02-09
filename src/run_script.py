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

# Removing empty rows in the bottom of excel file (row 829 to 834 in excel file)
# check if rows 822-830 are empty:
empty_rows = df_im.loc[822:830].isna().all(axis=1)
print("Empty rows in range 822-830:")
print(empty_rows)

# Row 78 (row 84 in excel file) has no measurements except for patient, timepoint and date
# check first:
print("Row 78 has measurements:")
print(df_im.loc[78].notna().sum(), "non-null values out of", len(df_im.loc[78]))
print(df_im.loc[78])

# removing empty rows
df_im = df_im.drop(index=range(823, 829)) 
df_im = df_im.drop(index=78)

# copy of reduced raw dataset for baseline modeling
df_im_reduced = df_im.copy()

# Correcting datatypes
# husk å skrive opp alle kolonner som har blitt endret!

# Changing columntyoe to date/time type
df_im["Messdatum"] = pd.to_datetime(
    df_im["Messdatum"], errors="coerce")

# Change Patient and Timepoint to integer type
df_im["Patient"] = pd.to_numeric(df_im["Patient"], errors="coerce").astype("Int64")
df_im["Timepoint"] = pd.to_numeric(df_im["Timepoint"], errors="coerce").astype("Int64")

# All other columns should be Float type (except Messdatum, Patient and Timepoint)
exclude_cols = ["Messdatum", "Patient", "Timepoint"]
float_cols = df_im.columns.difference(exclude_cols)
df_im[float_cols] = df_im[float_cols].apply(
    pd.to_numeric, errors="coerce"
)



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
    X_t = df_t.set_index('Patient').drop(columns=['Timepoint', 'Messdatum'])

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
    # Patient is already index, drop only Timepoint and Messdatum
    X_a = df_a.drop(columns=['Timepoint', 'Messdatum'])
    X_b = df_b.drop(columns=['Timepoint', 'Messdatum'])

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
X1 = df1.drop(columns=['Timepoint', 'Messdatum'])
X2 = df2.drop(columns=['Timepoint', 'Messdatum'])
X3 = df3.drop(columns=['Timepoint', 'Messdatum'])

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

contamination = 0.05    # standard contamination fraction from pyod library (assuming 10% of samples are outliers, can be adjusted based on domain knowledge or expected outlier proportion)
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




#%% Data exploration raw clinical dataset





#%%############ Cleaning clinical dataset #############################
# Pipeline: Load -> Forward-fill patient info -> Filter exclusions -> Rename -> Transform columns

# --- Helper function to move column after another column ---
def move_column_after(df, col_to_move, after_col):
    """Move a column to position right after another column."""
    cols = df.columns.tolist()
    cols.insert(cols.index(after_col) + 1, cols.pop(cols.index(col_to_move)))
    return df[cols]


#%% Step 1: Forward-fill patient-level data within each patient group
# So that clinical dataset is the same format as immunoligcal:

# Patient-level columns that should be constant across all timepoints for a patient
patient_level_cols = [
    'Patient', 'Unnamed: 2', 'Age at start', 'Gender', 'Weight [kg]', 'Height [cm]',
    'Overweight? BMI', 'Besserung nach Nachuntersuchung laut Arztbrief in %',
    'Comments questionnaire', 'Diagnosis', 'Target volume', 'single fraction',
    'kummulative dose (x) - if two targets were applied', 'FHA', 'kV', 'mA',
    'Filter', 'Response', 'further comments'
]

# Create patient group identifier and forward-fill within groups
df_cl['Patient_Group'] = df_cl['Patient'].notna().cumsum()
df_cl[patient_level_cols + ['Unnamed: 0']] = (
    df_cl.groupby('Patient_Group')[patient_level_cols + ['Unnamed: 0']].ffill()
)
df_cl = df_cl.drop(columns=['Patient_Group'])


#%% Step 2: Extract Timepoint measurement and create column

# Extract Timepoint number from Erfassungszeitpunkt string (e.g., "01.01.1" -> 1)
df_cl['Timepoint'] = (
    df_cl['Erfassungszeitpunkt']
    .str.extract(r'\d+\.\d+\.(\d+)')[0]
    .astype(float)
)
df_cl = move_column_after(df_cl, 'Timepoint', 'Patient')

# Keep rows with actual measurement data
df_cl_clean = df_cl[df_cl['Datum'].notna()].copy()
print(f"\nRows with measurement data: {len(df_cl_clean)}")

# Define exclusion keywords for raw dataset (German terms for: excluded, file locked, letter unavailable, questionnaires missing)
# 'Akte gesperrt', 'arztbrief kann nicht geöffnet werden', 'Fragebögen fehlen'?
exclude_keywords = ['Ausschluss']
exclude_mask = df_cl_clean['Unnamed: 0'].str.contains('|'.join(exclude_keywords), case=False, na=False)

# Print and apply exclusions
excluded_patients = df_cl_clean.loc[exclude_mask, 'Patient'].unique()
print(f"\n=== Exclusion Step 1: Exclusion keywords ===")
print(f"Excluded {len(excluded_patients)} patients with IDs: {excluded_patients}")
"""
removed 18 patient ids: 2.  39.  40.  44.  67.  71.  79.  96.  98. 100. 131. 161. 162. 168.
216. 243. 260. 264.]
"""
df_cl_clean = df_cl_clean[~exclude_mask]

#%% Step 3: Remove patients with invalid Response and drop helper columns

# Identify patients with missing/invalid Response values
invalid_response_mask = df_cl_clean['Response'].isna() | df_cl_clean['Response'].isin(['n.D', 'n.D.'])
patients_invalid_response = df_cl_clean.loc[invalid_response_mask, 'Patient'].unique()
df_cl_clean = df_cl_clean[~df_cl_clean['Patient'].isin(patients_invalid_response)]

print(f"\n=== Exclusion Step 2: Invalid Response ===")
print(f"Removed {len(patients_invalid_response)} patients with missing/invalid Response values")
print(f"Removed patient IDs: {patients_invalid_response}")
"""
removed 36 ids: [  8.   9.  10.  22.  23.  36.  38.  90.  92. 101. 103. 104. 108. 114.
 124. 127. 129. 151. 152. 186. 233. 252. 253. 261. 262. 263. 266. 267.
 268. 269. 270. 271. 272. 273. 274. 275.]
"""

# Drop helper columns no longer needed (check existence to avoid errors)
cols_to_drop = ['Unnamed: 0', 'Comments questionnaire', 'further comments']
df_cl_clean = df_cl_clean.drop(columns=[c for c in cols_to_drop if c in df_cl_clean.columns])


print(f"\n=== After exclusions ===")
print(f"Remaining: {df_cl_clean['Patient'].nunique()} patients, {len(df_cl_clean)} rows")
TableReport(df_cl_clean, max_plot_columns=180)
#%% Step 4: Rename columns (German -> English)
# Rationale: Standardize column names for easier coding and library compatibility

clinical_names = {
    # Patient demographics
    "Patient": "Patient", "Timepoint": "Timepoint",
    "Age at start": "age_at_start", "Gender": "gender",
    "Weight [kg]": "weight_kg", "Height [cm]": "height_cm",

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
print(f"\n Columns renamed: {len(clinical_names)}  ")

#%% Step 5: Remove empty data rows and fill improvement_percent for non-responders

# Identify rows with date but no actual questionnaire data (symptoms_months to health_status_today)
questionnaire_cols = df_cl_clean.loc[:, 'symptoms_months':'health_status_today'].columns
empty_questionnaire_mask = (
    df_cl_clean['date'].notna() &
    df_cl_clean[questionnaire_cols].isna().all(axis=1)
)
print(f"\n=== Removing empty questionnaire rows ===")
print(f"Rows to remove: {empty_questionnaire_mask.sum()}")
if empty_questionnaire_mask.sum() > 0:
    print(df_cl_clean.loc[empty_questionnaire_mask, ['Patient', 'Timepoint', 'date']])
df_cl_clean = df_cl_clean[~empty_questionnaire_mask]

# copy for baseline model
df_cl_reduced = df_cl_clean.copy()


#%% Feature engineering

# Fill improvement_percent with 0 for patients with "no improvement" response
# If response explicitly says "no improvement", the improvement percentage is 0
no_improvement_mask = (
    df_cl_clean['response'].str.lower().str.startswith('no', na=False) &
    df_cl_clean['improvement_percent'].isna()
)
df_cl_clean.loc[no_improvement_mask, 'improvement_percent'] = 0
print(f"Filled {no_improvement_mask.sum()} improvement_percent values with 0 for 'no improvement' responses")

# er det ok spørre anna?


#%% Step 6: Transform column values to usable formats
# This section handles: gender coding, BMI extraction, symptom duration parsing, therapy encoding

# --- 6a: Gender - standardize 'w' (German: weiblich) to 'f' (female) ---
df_cl_clean['gender'] = df_cl_clean['gender'].replace('w', 'f')
print(f"\n=== Gender value counts ===\n{df_cl_clean['gender'].value_counts()}")

# --- 6b: Split "Overweight? BMI" into two separate columns ---
# Original format: "ja (28.5)" or "nein" or "n.D" (missing)
def split_bmi_column(df, col_name='Overweight? BMI'):
    """Extract overweight status and BMI value from combined column."""
    col_idx = df.columns.get_loc(col_name)

    # Identify missing data markers (n.D, n.D.)
    is_missing = df[col_name].str.contains(r'^n\.?D\.?$', case=False, na=True)

    # Extract overweight status (ja/nein) and BMI value
    overweight = df[col_name].str.extract(r'(ja|nein)', flags=re.IGNORECASE)[0].str.lower()
    bmi = df[col_name].str.extract(r'\((\d+[,.]?\d*)\)?')[0].str.replace(',', '.').astype(float)

    # Set both to NaN where original was missing
    overweight = overweight.where(~is_missing, pd.NA)
    bmi = bmi.where(~is_missing, pd.NA)

    # Replace original column with two new columns at same position
    df = df.drop(columns=[col_name])
    df.insert(col_idx, 'overweight', overweight)
    df.insert(col_idx + 1, 'bmi', bmi)
    return df

df_cl_clean = split_bmi_column(df_cl_clean)

# Verify: Check for patients with overweight status but missing BMI
missing_bmi_mask = df_cl_clean['bmi'].isna() & df_cl_clean['overweight'].notna()
print(f"\n=== BMI/Overweight split verification ===")
print(f"Patients with overweight status but missing BMI: {missing_bmi_mask.sum()}")
if missing_bmi_mask.sum() > 0:
    print(df_cl_clean.loc[missing_bmi_mask, ['Patient', 'overweight']])



#%% --- 6c: Convert symptom duration to numeric months ---
# Original format: "3 Monate", "2 Jahre", "6-12 Mo.", "< 1 J" etc.
def parse_symptoms_duration(series):
    """Convert symptom duration strings to numeric months.
    Handles single values, ranges (takes midpoint), years/months.
    """
    # Extract numbers: single value or range (start-end)
    single = series.str.extract(r'[<>~]?\s*(\d+)(?!\s*-\s*\d)')[0].astype(float)
    range_start = series.str.extract(r'[<>~]?\s*(\d+)\s*-\s*\d+')[0].astype(float)
    range_end = series.str.extract(r'[<>~]?\s*\d+\s*-\s*(\d+)')[0].astype(float)

    # Use midpoint for ranges, single value otherwise
    number = range_start.add(range_end).div(2).fillna(single)

    # Detect if unit is years (J, Jahr, Jahre) vs months (Mo, Monat, Monate)
    unit = series.str.extract(r'(Monat\w*|Mo\.?|Jahr\w*|J\.?)', flags=re.IGNORECASE)[0]
    is_years = unit.str.lower().str.startswith('j').fillna(False)

    # Convert years to months
    return number.where(~is_years, number * 12)

df_cl_clean['symptoms_months'] = parse_symptoms_duration(df_cl_clean['symptoms_months'])

# Verify symptom duration conversion
print(f"\n=== Symptom duration conversion verification ===")
print(f"Dtype: {df_cl_clean['symptoms_months'].dtype}")
print(f"Range: {df_cl_clean['symptoms_months'].min()} - {df_cl_clean['symptoms_months'].max()} months")
print(f"Missing: {df_cl_clean['symptoms_months'].isna().sum()}")
print(f"Sample (Patient 16):\n{df_cl_clean.loc[df_cl_clean['Patient'] == 16, ['Timepoint', 'symptoms_months']]}")
# NB: How to handle words, comments, and dates in this column?

#%% --- 6d: Encode previous therapy as binary columns ---
# SKIP Original format: comma-separated numbers like "1,3,5" indicating therapy types 1-7
def encode_therapy_columns(df, col_name='previous_therapy'):
    """Create binary columns for each therapy type (1-7) from comma-separated string."""
    col_idx = df.columns.get_loc(col_name)

    # Create binary columns directly at the correct position
    for i in range(1, 8):
        binary_col = df[col_name].str.contains(rf'\b{i}\b', na=False).astype(int)
        df.insert(col_idx + i - 1, f'previous_therapy_{i}', binary_col)

    # Drop original column
    return df.drop(columns=[col_name])

df_cl_clean = encode_therapy_columns(df_cl_clean)

# Verify therapy encoding
print(f"\n=== Previous therapy encoding verification ===")
therapy_cols = [f'previous_therapy_{i}' for i in range(1, 8)]
print(df_cl_clean[therapy_cols].sum())


#%% Step 7: Set proper datatypes
# Integer columns: Patient ID, Timepoint, Age
# Date columns: measurement date
# Categorical: gender, diagnosis, response, etc.
# Float: questionnaire scores (PROBLEM: some contain comments - see note below)

# --- Define column type mappings ---
int_cols = ['Patient', 'Timepoint', 'age_at_start']
date_cols = ['date']

# --- Apply conversions ---
# Integer columns (use Int64 for nullable integers)
for col in int_cols:
    df_cl_clean[col] = pd.to_numeric(df_cl_clean[col], errors='coerce').astype('Int64')
# Date column
for col in date_cols:
    df_cl_clean[col] = pd.to_datetime(df_cl_clean[col], errors='coerce')


# PROBLEM: Cannot convert questionnaire columns to float because some entries contain comments
# TODO: Need to extract numeric values and handle/preserve comments separately

# --- Verification ---
print(f"\n=== Final datatype verification ===")
print(f"Shape: {df_cl_clean.shape}")
print(f"\nInteger columns:")
for col in int_cols:
    print(f"  {col}: {df_cl_clean[col].dtype}, missing: {df_cl_clean[col].isna().sum()}")
print(f"\nCategorical columns:")
for col in cat_cols:
    if col in df_cl_clean.columns:
        print(f"  {col}: {df_cl_clean[col].nunique()} unique values")
print(f"\nDate range: {df_cl_clean['date'].min()} to {df_cl_clean['date'].max()}")

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
    """Create response_category (CR, PR, NI) from raw response column.
    Priority: CR > PR > NI. Mixed responses (CR+NI, CR+PR) -> PR.
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
    Returns results_df (per-fold + mean), last trained model, X.
    """
    exclude = ['Patient', 'Timepoint', 'improvement_percent', 'response', target_col] 
    feature_cols = [c for c in df_model.columns if c not in exclude]
    X = df_model[feature_cols].copy()
    y = df_model[target_col].copy()

    # Convert categoricals to string (CatBoost requirement for cat_features)
    for col in X.select_dtypes(include=['category', 'object']).columns:
        X[col] = X[col].astype(str)
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()

    print(f"\n{'='*60}")
    print(f"  CatBoost Baseline (T1 only): {name}")
    print(f"{'='*60}")
    print(f"  Samples: {len(X)}, Features: {X.shape[1]}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        train_pool = Pool(X_train, y_train, cat_features=cat_cols)
        test_pool = Pool(X_test, y_test, cat_features=cat_cols)

        model = CatBoostClassifier(
            random_seed=42,
            verbose=0,
            iterations=500, # kernel crashing with 1000.
            custom_metric=[
                'Accuracy',
                'TotalF1:average=Weighted',
                'AUC:type=Mu',
                'MCC', 
            ]
        )
        model.fit(train_pool, eval_set=test_pool, use_best_model=False)

        # Get metrics from CatBoost's built-in evaluation
        evals = model.get_evals_result()

        fold_result = {
            'Fold': fold + 1,
            'Accuracy': evals['validation']['Accuracy'][-1],
            'F1_weighted': evals['validation']['TotalF1:average=Weighted'][-1],
            'AUC': evals['validation']['AUC:type=Mu'][-1],
            'MCC': evals['validation']['MCC'][-1],
            'Train_size': len(train_idx),
            'Test_size': len(test_idx)
        }
        fold_results.append(fold_result)

        print(f"  Fold {fold+1}: Acc={fold_result['Accuracy']:.4f}  "
              f"F1={fold_result['F1_weighted']:.4f}  "
              f"AUC={fold_result['AUC']:.4f}  "
              f"MCC={fold_result['MCC']:.4f}")

    results_df = pd.DataFrame(fold_results)

    # Add mean row
    metric_cols = ['Accuracy', 'F1_weighted', 'AUC', 'MCC']
    mean_row = {m: results_df[m].mean() for m in metric_cols}
    mean_row['Fold'] = 'Mean'
    results_df = pd.concat([results_df, pd.DataFrame([mean_row])], ignore_index=True)

    print(f"\n  Mean Across 5 Folds")
    for m in metric_cols:
        print(f"  {m}: {mean_row[m]:.4f}")

    return results_df, model, X


#%%
# Run for all three datasets
print("\n" + "="*70)
print("  RUNNING CATBOOST BASELINES")
print("="*70)

target_col = 'response_category'

res_im, model_im, X_im = run_catboost_baseline(
    df_im_baseline, target_col, "Immunological")

res_cl, model_cl, X_cl = run_catboost_baseline(
    df_cl_baseline, target_col, "Clinical")

res_comb, model_comb, X_comb = run_catboost_baseline(
    df_combined_baseline, target_col, "Combined")


#%% Summary results table

summary_rows = []
metric_cols = ['Accuracy', 'F1_weighted', 'AUC', 'MCC']
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
print("\n" + "="*70)
print("  BASELINE RESULTS SUMMARY")
print("="*70)
print(df_summary.to_string(index=False))


#%% SHAP analysis for each model
# Compute SHAP values and plot summary for each dataset

def compute_and_plot_shap(model, X, name):
    """Compute SHAP values and create summary plot."""
    print(f"\n=== SHAP Analysis: {name} ===")
    print(f"  Classes: {list(model.classes_)}")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Summary plot (bar) - mean absolute SHAP across all classes
    shap.summary_plot(shap_values, X, plot_type="bar",
                      class_names=list(model.classes_),
                      show=False, max_display=20)
    plt.title(f"SHAP Feature Importance - {name}")
    plt.tight_layout()
    plt.show()

    # Per-class summary plots
    for i, cls in enumerate(model.classes_):
        shap.summary_plot(shap_values[i], X, show=False, max_display=15)
        plt.title(f"SHAP Values - {name} - Class: {cls}")
        plt.tight_layout()
        plt.show()

    return shap_values

shap_im = compute_and_plot_shap(model_im, X_im, "Immunological")
shap_cl = compute_and_plot_shap(model_cl, X_cl, "Clinical")
shap_comb = compute_and_plot_shap(model_comb, X_comb, "Combined")


