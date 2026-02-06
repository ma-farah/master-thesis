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

df_im_raw = df_im.copy()  # copy of raw dataset
df_im = df_im.drop(columns=dropped_columns)

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






#%% Basline model for immunological dataset - CatBoost

