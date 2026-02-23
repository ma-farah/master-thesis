
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
