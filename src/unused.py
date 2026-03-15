
#%% old trajecotry pca

# Combined PCA for timepoint t1-t2, t2-t3 and t1-t3 

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




#%% Old pyOD outlier detection code:

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



#%% comparing median vs miceforest imputation (immu dataset)
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





#%%############# RV / RV2 analysis across timepoints ##########################

#NB! Patient ID 83 has two timepoint 4 measurements 
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



#%% Old modeling dataset prep functions (replaced by construct_datasets_targets + create_model_datasets)


def prepare_baseline_datasets(df_im_vis, df_cl_bcat, pain_targets):
    """Build the three T1 modeling datasets for baseline CatBoost (old version).

    Replaced by construct_datasets_targets() + create_model_datasets().
    Kept here for reference.
    """
    model_patients = set(pain_targets['Patient'].values)

    df_im_raw_t1 = (
        df_im_vis[
            (df_im_vis['Timepoint'] == 1) &
            (df_im_vis['Patient'].isin(model_patients))
        ]
        .copy()
        .reset_index(drop=True)
    )
    df_im_raw_t1 = df_im_raw_t1.merge(
        pain_targets[['Patient', 'pain_scale_reduction', 'pain_reduction_pct']],
        on='Patient', how='left'
    )

    df_cl_bcat_t1 = (
        df_cl_bcat[
            (df_cl_bcat['Timepoint'] == 1) &
            (df_cl_bcat['Patient'].isin(model_patients))
        ]
        .copy()
        .reset_index(drop=True)
    )
    df_cl_bcat_t1 = df_cl_bcat_t1.merge(
        pain_targets[['Patient', 'pain_scale_reduction', 'pain_reduction_pct']],
        on='Patient', how='left'
    )

    df_bcat_combined_t1 = df_im_raw_t1.merge(
        df_cl_bcat_t1,
        on=['Patient', 'Timepoint'], how='inner',
        suffixes=('_im', '_cl')
    )

    return df_im_raw_t1, df_cl_bcat_t1, df_bcat_combined_t1


def run_baseline_catboost(df_im_raw_t1, df_cl_bcat_t1, df_bcat_combined_t1):
    """Run baseline CatBoost on three datasets (old version).

    Replaced by direct calls to run_catboost_regressor() in results.py.
    Kept here for reference.
    """
    import model as _model
    results     = {}
    shap_values = {}

    for target in ['pain_scale_reduction', 'pain_reduction_pct']:
        res_im,   model_im,   X_im,   ypred_im   = _model.run_catboost_regressor(
            df_im_raw_t1,       target, "Immunological (raw T1)")
        res_cl,   model_cl,   X_cl,   ypred_cl   = _model.run_catboost_regressor(
            df_cl_bcat_t1,      target, "Clinical (raw T1)")
        res_comb, model_comb, X_comb, ypred_comb = _model.run_catboost_regressor(
            df_bcat_combined_t1, target, "Combined (raw T1)")

        results[target] = {
            'Immunological': (res_im,   model_im,   X_im,   ypred_im),
            'Clinical':      (res_cl,   model_cl,   X_cl,   ypred_cl),
            'Combined':      (res_comb, model_comb, X_comb, ypred_comb),
        }
    return results, shap_values


def prepare_advanced_dataset(df_im_vis, df_cl_mod, pain_targets):
    """Build combined T1 dataset for advanced CatBoost (old version).

    Replaced by construct_datasets_targets() + create_model_datasets().
    Kept here for reference.
    """
    model_patients = set(pain_targets['Patient'].values)

    df_im_t1 = (
        df_im_vis[
            (df_im_vis['Timepoint'] == 1) &
            (df_im_vis['Patient'].isin(model_patients))
        ]
        .copy()
        .reset_index(drop=True)
    )
    df_im_t1 = df_im_t1.merge(
        pain_targets[['Patient', 'pain_scale_reduction', 'pain_reduction_pct']],
        on='Patient', how='left'
    )

    df_cl_t1 = (
        df_cl_mod[
            (df_cl_mod['Timepoint'] == 1) &
            (df_cl_mod['Patient'].isin(model_patients))
        ]
        .copy()
        .reset_index(drop=True)
    )
    df_cl_t1 = df_cl_t1.merge(
        pain_targets[['Patient', 'pain_scale_reduction', 'pain_reduction_pct']],
        on='Patient', how='left'
    )

    df_combined = df_im_t1.merge(
        df_cl_t1, on=['Patient', 'Timepoint'], how='inner',
        suffixes=('_im', '_cl')
    )
    df_combined = (
        df_combined
        .drop(columns=['pain_scale_reduction_im', 'pain_reduction_pct_im'], errors='ignore')
        .rename(columns={
            'pain_scale_reduction_cl': 'pain_scale_reduction',
            'pain_reduction_pct_cl':   'pain_reduction_pct',
        })
    )
    return df_combined


#%%
# creates a respon se category column based on response column

def standardize_response(df, response_col='response', verbose=True):
    """Parse raw response column into response_category (CR/PR/NI). 

    Multiple categories in one entry are kept as comma-separated ('CR, NI', 'PR, CR').
    Unrecognized entries are kept as-is
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

        found = []
        if re.search(r'\bni\b', s):
            found.append('NI')
        if re.search(r'\bcr\b', s):
            found.append('CR')
        if re.search(r'\bpr\b', s):
            found.append('PR')

        categories[idx] = ', '.join(found) if found else val.strip()

    df['response_category'] = categories.astype('category')

    if verbose:
        print(f"\nResponse categories:\n"
              f"{df['response_category'].value_counts(dropna=False).to_string()}")

    return df

#%% old catboost approach rent and optuna on innder folds:

def run_advanced_catboost_rent(
    df_combined, target_col='pain_reduction_pct', random_state=42,
    tau_3=0.90, target_transformer=None,
):
    """CatBoostRegressor with Optuna-tuned RENT + nested CV (per-inner-fold tuning).

    For each outer fold → for each inner fold:
      Study 1 — RENT HPs : tune C, l1_ratio, τ₁, τ₂ using a fixed probe CatBoost
                           evaluated on the inner val split.  τ₃ is fixed at tau_3.
      Study 2 — Model HPs: tune depth, learning_rate, l2_leaf_reg, etc. on the
                           features selected by Study 1, evaluated on the inner val split.
      Pick the best inner fold (lowest Study 2 val RMSE) → use its selected features
      and model params to train on full X_train and evaluate on X_test.

    Outer CV : RepeatedKFold(n_splits=4, n_repeats=5) = 20 outer folds.
    Inner CV : RepeatedKFold(n_splits=4, n_repeats=2) =  8 inner folds per outer fold. # change to 4 repeats later?
    Study 1  : 50 Optuna trials for RENT HPs (K=100 RENT splits per trial).
    Study 2  : 50 Optuna trials for CatBoost HPs.

    Parameters
    ----------
    df_combined        : pd.DataFrame  Combined dataset (immunological + clinical).
    target_col         : str           Regression target (default: 'pain_reduction_pct').
    random_state       : int           Random seed (default 42).
    tau_3              : float         Fixed RENT τ₃ t-test threshold (default 0.90).
    target_transformer : transformer   Optional sklearn-compatible power transformer.

    Returns
    -------
    results_df                 : pd.DataFrame       Per-fold metrics + Mean/Std rows.
    final_model                : CatBoostRegressor  Final model trained on full dataset.
    X_final                    : pd.DataFrame       Features used by final model.
    y_pred                     : pd.Series          Full-data predictions (original scale).
    selected_features_per_fold : list[list[str]]    Features selected per outer fold.
    best_rent_params_list      : list[dict]         Best RENT HPs per outer fold.
    """
    import optuna
    import warnings
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OrdinalEncoder
    from RENT import RENT

    warnings.filterwarnings('ignore', message='.*less than 75% GPU memory.*')
    warnings.filterwarnings('ignore', category=FutureWarning, module='RENT')
    warnings.filterwarnings('ignore', category=RuntimeWarning, module='RENT')
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    N_RENT_TRIALS  = 20
    N_MODEL_TRIALS = 20

    y = df_combined[target_col].copy()
    exclude = ['Patient', 'Timepoint', target_col,
               'pain_reduction', 'pain_reduction_pct',
               'pain_under_load_reduction', 'pain_under_load_reduction_pct']
    feature_cols = [c for c in df_combined.columns if c not in exclude]
    X = df_combined[feature_cols].copy()

    valid = y.notna()
    X, y = X[valid].reset_index(drop=True), y[valid].reset_index(drop=True)

    for col in X.select_dtypes(include=['category', 'object']).columns:
        X[col] = X[col].astype(str)
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()

    print(f"\n{'='*65}")
    print(f"  CatBoost + RENT (Optuna-tuned) — {target_col}")
    print(f"  Samples: {len(X)},  Features: {len(feature_cols)}")
    print(f"  τ₃={tau_3} (fixed)  |  τ₁, τ₂, C, l1_ratio tuned via Optuna")
    print(f"  Outer: 4×5=20 folds  |  Inner: 4×2=8 folds") # try with 1 first
    print(f"  OptunaStudy 1 (RENT HPs):  {N_RENT_TRIALS} trials × K=100 RENT splits  (per inner fold)")
    print(f"  Optuna Study 2 (model HPs): {N_MODEL_TRIALS}                   (per inner fold)")
    print(f" Total model fits 20x8x50x100 + 20x8x50 + 20 = approx. 808 020")
    print(f"{'='*65}")

    outer_cv = RepeatedKFold(n_splits=4, n_repeats=5, random_state=random_state)
    inner_cv = RepeatedKFold(n_splits=4, n_repeats=2, random_state=random_state) # try 2 repeat first

    fold_results               = []
    best_rent_params_list      = []
    best_model_params_list     = []
    selected_features_per_fold = []
    start = time.time()

    for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X), start=1):
        print(f"\n  ── Outer fold {outer_fold}/{outer_cv.get_n_splits()} ──")

        X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        if target_transformer is not None:
            pt_fold     = clone(target_transformer)
            y_train_fit = pd.Series(
                pt_fold.fit_transform(y_train.values.reshape(-1, 1)).ravel(),
                index=y_train.index,
            )
        else:
            pt_fold     = None
            y_train_fit = y_train

        # ── Inner CV: per-fold RENT + CatBoost tuning ────────────────────────
        # For each inner fold: Study 1 (RENT HPs) → Study 2 (model HPs).
        # Pick the best inner fold (lowest Study 2 val RMSE).
        inner_fold_log = []  # (val_rmse, selected_cols, model_params, cat_cols_inner, rent_params)

        for inner_fold_idx, (inner_train_idx, inner_val_idx) in enumerate(
            inner_cv.split(X_train), start=1
        ):
            X_it = X_train.iloc[inner_train_idx].copy()
            y_it = y_train_fit.iloc[inner_train_idx]
            X_iv = X_train.iloc[inner_val_idx].copy()
            y_iv = y_train_fit.iloc[inner_val_idx]
          
            # Prepare RENT input for this inner fold's training data
            X_it_enc = X_it.copy()
            cat_mask_cols = [c for c in X_it.columns if X_it[c].dtype == object]
            if cat_mask_cols:
                oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                X_it_enc[cat_mask_cols] = oe.fit_transform(X_it[cat_mask_cols])
            imputer   = SimpleImputer(strategy='median')
            X_it_rent = pd.DataFrame(
                imputer.fit_transform(X_it_enc.astype(float)), columns=feature_cols)

            # ── Study 1: Tune RENT HPs — probe evaluated on inner val split ──
            def rent_objective(trial):
                c_val    = trial.suggest_float('C',        1e-3, 10, log=True)
                l1_ratio = trial.suggest_float('l1_ratio', 0.1,  1.0)
                tau_1    = trial.suggest_float('tau_1',    0.6,  0.9)
                tau_2    = trial.suggest_float('tau_2',    0.6,  0.9)

                rent_t = RENT.RENT_Regression(
                    data=X_it_rent, target=y_it.values,
                    feat_names=feature_cols,
                    C=[c_val], l1_ratios=[l1_ratio],
                    autoEnetParSel=False, poly='OFF',
                    testsize_range=(0.25, 0.25), K=100,
                    random_state=random_state, verbose=0,
                )
                rent_t.train()
                sel_idx = rent_t.select_features(
                    tau_1_cutoff=tau_1, tau_2_cutoff=tau_2, tau_3_cutoff=tau_3)
                if len(sel_idx) == 0:
                    return 1e6
                sel_cols     = [feature_cols[i] for i in sel_idx]
                cat_cols_sel = [c for c in cat_cols if c in sel_cols]
                probe = CatBoostRegressor(
                    iterations=300, depth=5,
                    cat_features=cat_cols_sel,
                    random_seed=random_state,
                    task_type='GPU', devices='0',
                    gpu_ram_part=0.6, logging_level='Silent',
                )
                with contextlib.redirect_stderr(io.StringIO()):
                    probe.fit(X_it[sel_cols], y_it)
                preds = probe.predict(X_iv[sel_cols])
                return np.sqrt(mean_squared_error(y_iv, preds))
         
            rent_study = optuna.create_study(direction='minimize')
            rent_study.optimize(rent_objective, n_trials=N_RENT_TRIALS, show_progress_bar=False)
            best_rent_inner = rent_study.best_params

            # Re-run RENT with best HPs → selected features for this inner fold
            rent_final = RENT.RENT_Regression(
                data=X_it_rent, target=y_it.values,
                feat_names=feature_cols,
                C=[best_rent_inner['C']], l1_ratios=[best_rent_inner['l1_ratio']],
                autoEnetParSel=False, poly='OFF',
                testsize_range=(0.25, 0.25), K=100,
                random_state=random_state, verbose=0,
            )
            rent_final.train()
            sel_idx_inner = rent_final.select_features(
                tau_1_cutoff=best_rent_inner['tau_1'],
                tau_2_cutoff=best_rent_inner['tau_2'],
                tau_3_cutoff=tau_3,
            )
            selected_cols  = ([feature_cols[i] for i in sel_idx_inner]
                              if len(sel_idx_inner) > 0 else feature_cols)
            cat_cols_inner = [c for c in cat_cols if c in selected_cols]

            # ── Study 2: Tune CatBoost HPs on selected features ──────────────
            def model_objective(trial):
                params = {
                    'depth':               trial.suggest_int('depth', 3, 10),
                    'learning_rate':       trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
                    'l2_leaf_reg':         trial.suggest_float('l2_leaf_reg', 1, 10.0, log=True),
                    'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
                }
                m = CatBoostRegressor(
                    iterations=300, **params,
                    cat_features=cat_cols_inner,
                    random_seed=random_state,
                    task_type='GPU', devices='0', gpu_ram_part=0.6,
                    logging_level='Silent',
                )
                with contextlib.redirect_stderr(io.StringIO()):
                    m.fit(X_it[selected_cols], y_it)
                preds = m.predict(X_iv[selected_cols])
                return np.sqrt(mean_squared_error(y_iv, preds))

            model_study = optuna.create_study(direction='minimize')
            model_study.optimize(model_objective, n_trials=N_MODEL_TRIALS, show_progress_bar=False)
            best_model_params_inner = model_study.best_params
            val_rmse_inner          = model_study.best_value
            inner_fold_log.append((
                val_rmse_inner, selected_cols,
                best_model_params_inner, cat_cols_inner, best_rent_inner,
            ))
            print(f" Inner fold {inner_fold_idx}/{inner_cv.get_n_splits()} - val RMSE={val_rmse_inner:.4f} - selected features={len(selected_cols)}")

        # ── Pick best inner fold ──────────────────────────────────────────────
        best_inner = min(inner_fold_log, key=lambda x: x[0])
        val_rmse_best, selected_cols, best_model_params, cat_cols_inner, best_rent = best_inner

        selected_features_per_fold.append(selected_cols)
        best_model_params_list.append(best_model_params)
        best_rent_params_list.append(best_rent)

        suffix = '...' if len(selected_cols) > 8 else ''
        print(f"    Best inner fold val RMSE={val_rmse_best:.4f}")
        print(f"    RENT: {len(selected_cols)}/{len(feature_cols)} features — "
              f"{selected_cols[:8]}{suffix}")
        print(f"    Model hyperparameters: {best_model_params}")

        # ── Train on full X_train with best inner fold's params ───────────────
        fold_model = CatBoostRegressor(
            iterations=300, **best_model_params,
            cat_features=cat_cols_inner,
            random_seed=random_state,
            task_type='GPU', devices='0', gpu_ram_part=0.6,
            logging_level='Silent',
        )
        with contextlib.redirect_stderr(io.StringIO()):
            fold_model.fit(X_train[selected_cols], y_train_fit)

        preds_raw = fold_model.predict(X_test[selected_cols])
        preds = (pt_fold.inverse_transform(preds_raw.reshape(-1, 1)).ravel()
                 if pt_fold is not None else preds_raw)

        mae  = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mse  = rmse ** 2
        r2   = r2_score(y_test, preds)
        fold_results.append({'Fold': outer_fold, 'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2})
        print(f"      MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}")

    elapsed = time.time() - start
    print(f"\n  Training time: {elapsed:.1f}s  ({elapsed/60:.1f} min)")


    # ── Results summary ───────────────────────────────────────────────────────
    results_df  = pd.DataFrame(fold_results)
    metric_cols = ['MAE', 'MSE', 'RMSE', 'R2']
    mean_row = {'Fold': 'Mean', **{m: results_df[m].mean() for m in metric_cols}}
    std_row  = {'Fold': 'Std',  **{m: results_df[m].std()  for m in metric_cols}}
    results_df = pd.concat(
        [results_df, pd.DataFrame([mean_row, std_row])], ignore_index=True)

    n_outer = len(fold_results)
    t_crit  = stats.t.ppf(0.975, df=n_outer - 1)
    print(f"\n  Summary (4×5 outer CV, 95% CI):")
    for m in metric_cols:
        mv = mean_row[m]; sv = std_row[m]
        ci = t_crit * sv / np.sqrt(n_outer)
        print(f"    {m:<5}: {mv:.3f} ± {sv:.4f}  (95% CI [{mv - ci:.3f}, {mv + ci:.3f}])")

    # ── Feature selection frequency ───────────────────────────────────────────
    from collections import Counter
    all_selected = [f for fold_feats in selected_features_per_fold for f in fold_feats]
    freq = Counter(all_selected)
    print(f"\n  RENT feature selection frequency in ({n_outer} outer folds):")
    for feat, cnt in freq.most_common():
        marker = ' ◀' if cnt / n_outer >= 0.5 else ''
        print(f"    {cnt:>3}/{n_outer}  {feat}{marker}")


    # ── Final model: features selected in ≥50% of outer folds ────────────────
    final_cols = [f for f, cnt in freq.items() if cnt / n_outer >= 0.5]
    if not final_cols:
        print(f"\n  Warning: no feature met ≥50% threshold — falling back to top 10 by frequency")
        final_cols = [f for f, _ in freq.most_common(10)]
    print(f"\n  Final model: {len(final_cols)} features selected (≥50% frequency): {final_cols}")

    X_final        = X[final_cols]
    cat_cols_final = [c for c in cat_cols if c in final_cols]

    if target_transformer is not None:
        pt_final    = clone(target_transformer)
        y_final_fit = pd.Series(
            pt_final.fit_transform(y.values.reshape(-1, 1)).ravel(), index=y.index)
    else:
        pt_final    = None
        y_final_fit = y

    # Aggregate model Hyperprameters across outer folds: use median for continuous params,
    # mode for integer params.  avoid cherry picking model(?)
    import statistics
    _all_keys = best_model_params_list[0].keys()
    best_model_params_final = {}
    for k in _all_keys:
        vals = [p[k] for p in best_model_params_list]
        if isinstance(vals[0], int):
            best_model_params_final[k] = int(round(statistics.median(vals)))
        else:
            best_model_params_final[k] = statistics.median(vals)
    final_model = CatBoostRegressor(
        iterations=300,
        loss_function='RMSE',
        custom_metric=['MAE', 'R2'],
        cat_features=cat_cols_final,
        random_seed=random_state,
        task_type='GPU', devices='0', gpu_ram_part=0.6,
        logging_level='Silent',
        **best_model_params_final,
    )

    with contextlib.redirect_stderr(io.StringIO()):
        final_model.fit(X_final, y_final_fit)

    y_pred_raw = pd.Series(final_model.predict(X_final), index=range(len(X_final)), dtype='float64')
    y_pred = (pd.Series(pt_final.inverse_transform(y_pred_raw.values.reshape(-1, 1)).ravel(),
                        index=y_pred_raw.index, dtype='float64')
              if pt_final is not None else y_pred_raw)

    return results_df, final_model, X_final, y_pred, selected_features_per_fold, best_rent_params_list

#%%
# pain under load target 
print('\n1.3: CatBoost (Nested CV + RENT + Optuna) — pain_under_load_reduction (T1-T2)')
_pt = PowerTransformer(method='yeo-johnson', standardize=True)

cb_ul_results, cb_ul_model, cb_ul_X, cb_ul_ypred, cb_ul_model_params, cb_ul_freq = \
    model.run_advanced_catboost_rent(
        model_datasets['pain_under_load_reduction'],
        target_col='pain_under_load_reduction',
        target_transformer=_pt,
    )

# Save model and feature matrix so SHAP can be run without retraining
cb_ul_model.save_model(os.path.join(MODEL_DIR, 'cb_ul_model.cbm'))
joblib.dump(cb_ul_X, os.path.join(MODEL_DIR, 'cb_ul_X.pkl'))
print(' Saved cb_ul_model.cbm and cb_ul_X.pkl to', os.path.abspath(MODEL_DIR))

#%%


def standardize_response(df, response_col='response', verbose=True):
    """Parse raw response column into response_category (CR/PR/NI). 

    Multiple categories in one entry are kept as comma-separated ('CR, NI', 'PR, CR').
    Unrecognized entries are kept as-is
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

        found = []
        if re.search(r'\bni\b', s):
            found.append('NI')
        if re.search(r'\bcr\b', s):
            found.append('CR')
        if re.search(r'\bpr\b', s):
            found.append('PR')

        categories[idx] = ', '.join(found) if found else val.strip()

    df['response_category'] = categories.astype('category')

    if verbose:
        print(f"\nResponse categories:\n"
              f"{df['response_category'].value_counts(dropna=False).to_string()}")

    return df

#%%

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
        if re.search(r'beide|bds|li\s*[+&/]\s*re|re\s*[+&/]\s*li|li\s+u\.?\s+re|re\s+u\.?\s+li|li\s+und\s+re|re\s+und\s+li', s):
            return 'B'
        if re.search(r'\bli\b|\blinks\b|\blinke[rns]?\b', s):
            return 'L'
        if re.search(r'\bre\b|\brechts\b|\brechte[rns]?\b|\brecht\b', s):
            return 'R'
        return ''

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


#%% 

def parse_symptoms_duration(series, date_series=None):
    """Convert months since complaints/symtpoms to numeric value in months.

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
