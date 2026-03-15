# imports
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import phik
from missing_methods import pca as mm_pca, rv2 as mm_rv2
from missing_methods.sk import StandardScaler as MM_StandardScaler
from adjustText import adjust_text as _adj
from itertools import combinations as _combns
from scipy import stats
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(data_path=None):
    """Load immunological and clinical sheets from LDRT_raw.xlsx.

    Parameters
    ----------
    data_path : str or Path, optional

    Returns
    -------
    df_im : pd.DataFrame   raw immunological dataset 
    df_cl : pd.DataFrame   raw clinical dataset 
    """
    if data_path is None:
        data_path = Path(__file__).resolve().parents[1] / "data" / "LDRT_raw.xlsx"
    data_path = Path(data_path)

    df_im = pd.read_excel(data_path, sheet_name="IPT ", header=4, engine="openpyxl")
    df_cl = pd.read_excel(data_path, sheet_name="Patient data & Pain", header=1, engine="openpyxl")
    return df_im, df_cl


# ── Raw dataset overview ──────────────────────────────────────────────────────

def dataset_overview(df, name):
    """Print basic statistics for a dataset.

    Parameters
    ----------
    df            : pd.DataFrame
    name          : str   label used in print headers (e.g. 'Immunological')
    """
    print(f"\n{'='*60}")
    print(f"Raw {name} Dataset Overview")
    print(f"{'='*60}")
    print(f"  Shape         : {df.shape[0]} rows × {df.shape[1]} columns")

    patient_col='Patient'
    timepoint_col='Timepoint'

    if patient_col in df.columns:
        print(f"  Patients      : {df[patient_col].dropna().nunique()}")

    if timepoint_col in df.columns:
        print(f"  Timepoints    : {df[timepoint_col].dropna().nunique()}")
        print(f"\nRows per timepoint:")
        print(df[timepoint_col].value_counts().sort_index().to_string())

        if patient_col in df.columns:
            no_tp_mask = df[timepoint_col].isna() & df[patient_col].notna()
            if no_tp_mask.any():
                print(f"\nPatients with unknown Timepoints ({no_tp_mask.sum()} rows):")
                print(df.loc[no_tp_mask, [patient_col, timepoint_col]].to_string())

    total_nan  = df.isna().sum().sum()
    total_vals = df.shape[0] * df.shape[1]
    print(f"\nMissing values: {total_nan} ({total_nan / total_vals * 100:.1f}% of all cells)")

    # Column dtype breakdown
    dtype_counts = df.dtypes.value_counts()
    print(f"\nColumn dtypes:")
    for dtype, count in dtype_counts.items():
        print(f"{dtype}: {count} columns")
    print("\n")



# ── Patient timepoint coverage ────────────────────────────────────────────────

def patient_timepoint_summary(df, name):
    """Print cumulative patient coverage across timepoints and plot a bar plot.

    Parameters
    ----------
    df            : pd.DataFrame
    name          : str   label used in titles
    """

    timepoint_col='Timepoint'
    patient_col='Patient'

    timepoints = sorted(df[timepoint_col].dropna().unique())

    pt_sets = {
        t: set(df.loc[df[timepoint_col] == t, patient_col].dropna())
        for t in timepoints
    }

    print(f"Patient timepoint coverage — {name} Dataset:")

    # Cumulative intersection: patients present at T1, T1+T2, T1+T2+T3, …
    cumulative = pt_sets[timepoints[0]]
    tp_labels  = [f"T{int(timepoints[0])}"]
    for t in timepoints[1:]:
        tp_labels.append(f"T{int(t)}")
        print(f"Patients with measurements at {' & '.join(tp_labels)}:  "
              f"{len(cumulative & pt_sets[t])}")
        cumulative = cumulative & pt_sets[t]

    # Patients with ONLY T1
    others = set().union(*(pt_sets[t] for t in timepoints[1:]))
    print(f"  Patients at only T{int(timepoints[0])} : "
          f"{len(pt_sets[timepoints[0]] - others)}")

    # Bar plot: unique patients per timepoint
    patient_counts = df.groupby(timepoint_col)[patient_col].nunique().sort_index()
    _bar_color = sns.color_palette("mako", len(patient_counts))

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=patient_counts.index, y=patient_counts.values,
                hue=patient_counts.index, palette=_bar_color, legend=False, ax=ax)
    for bar, n in zip(ax.patches, patient_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"n={n}", ha='center', va='bottom', fontsize=9)
    ax.set_xticklabels([str(t) for t in patient_counts.index])
    ax.set_title(f"Unique Patients per Timepoint - {name} Dataset")
    ax.set_xlabel("Timepoint")
    ax.set_ylabel("Num unique patients")
    plt.tight_layout()
    plt.show()


# ── Clinical distribution plots ───────────────────────────────────────────────

def plot_clinical_distributions(df_cl_vis):
    """Plot baseline demographic and pain distributions for the clinical dataset

    Parameters
    ----------
    df_cl_vis : pd.DataFrame   cleaned clinical dataset
    """
    print('Clinical dataset - Distribution Plots')

    cl_t1  = df_cl_vis[df_cl_vis['Timepoint'] == 1].drop_duplicates('Patient').copy()
    mako3  = sns.color_palette('mako', 3)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # Age
    sns.histplot(cl_t1['age_at_start'].dropna(), kde=True, ax=axes[0],
                 color=mako3[1], bins=15)
    axes[0].set_title('Age distribution')
    axes[0].set_xlabel('Age')
    axes[0].set_ylabel('Count')

    # Gender
    gender_counts = cl_t1['gender'].value_counts()
    axes[1].bar(gender_counts.index.astype(str), gender_counts.values,
                color=mako3[:len(gender_counts)])
    axes[1].set_title('Gender distribution')
    axes[1].set_ylabel('Count')

    # Diagnosis
    diag_counts = cl_t1['diagnosis'].value_counts()
    axes[2].barh(diag_counts.index.astype(str), diag_counts.values,
                 color=mako3[1])
    axes[2].set_title('Diagnosis distribution')
    axes[2].set_xlabel('Count')
    axes[2].invert_yaxis()

    plt.tight_layout()
    plt.show()

    # Pain scale distribution per timepoint histogram with median line
    timepoints  = sorted(df_cl_vis['Timepoint'].dropna().unique().astype(int))
    mako_tp     = sns.color_palette('mako', len(timepoints))

    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(10, 7))
    fig.suptitle('Distribution of pain_scale by Timepoint', fontsize=13, fontweight='bold')
    gs  = GridSpec(2, 6, figure=fig)
    all_axes = [
        fig.add_subplot(gs[0, 0:2]),
        fig.add_subplot(gs[0, 2:4]),
        fig.add_subplot(gs[0, 4:6]),
        fig.add_subplot(gs[1, 1:3]),
        fig.add_subplot(gs[1, 3:5]),
    ]

    for i, t in enumerate(timepoints):
        ax      = all_axes[i]
        vals    = (df_cl_vis.loc[df_cl_vis['Timepoint'] == t]
                             .drop_duplicates(subset='Patient')['pain_scale']
                             .dropna())
        median  = vals.median()

        ax.hist(vals, bins=range(0, 12), color=mako_tp[i], edgecolor='white', linewidth=0.4, alpha=0.9)
        ax.axvline(median, color='white', linestyle='--', linewidth=1.5, label=f'Median {median:.1f}')
        ax.legend(fontsize=8, loc='upper right', framealpha=0.6)
        ax.set_title(f'T{t}  (n={len(vals)})', fontsize=10)
        ax.set_xlabel('Pain Scale (0-10)', fontsize=9)
        ax.set_ylabel('Count', fontsize=9)
        ax.set_xlim(0, 11)
        ax.set_ylim(0, 50)

    plt.tight_layout()
    plt.show()



def t_test_pain_reduction(df_cl_vis):
    """
    One-sample t-test for wether the mean pain_scale reduction is significantly
    greater than zero for each baseline-vs-followup timepoint pair 
    H1: reduction > 0, using significance level at 0.05

    """
    followups = sorted(df_cl_vis['Timepoint'].dropna().unique().astype(int))
    followups = [t for t in followups if t > 1]
    pairs     = [(1, t) for t in followups]

    records = []
    for t_a, t_b in pairs:
        t1 = (df_cl_vis[df_cl_vis['Timepoint'] == t_a][['Patient', 'pain_scale']]
              .drop_duplicates('Patient').dropna()
              .rename(columns={'pain_scale': 'pain_t1'}))
        
        t2 = (df_cl_vis[df_cl_vis['Timepoint'] == t_b][['Patient', 'pain_scale']]
              .drop_duplicates('Patient').dropna()
              .rename(columns={'pain_scale': f'pain_t{t_b}'}))

        df_r        = t1.merge(t2, on='Patient')
        df_r['red'] = df_r['pain_t1'] - df_r[f'pain_t{t_b}']
        vals        = df_r['red'].dropna()

        t_stat, p_two = stats.ttest_1samp(vals, popmean=0)
        p_one         = p_two / 2

        records.append({
            'Comparison': f'T1 -> T{t_b}',
            'n'         : len(vals),
            'Mean'      : round(vals.mean(), 3),
            'std'       : round(vals.std(), 3),
            't'         : round(t_stat, 3),
            'p-value'   : f'{p_one:.2e}',
        })

    result_df = pd.DataFrame(records)

    cols   = result_df.columns.tolist()
    widths = {'Comparison': 13, 'n': 6, 'Mean': 8, 'std': 7, 't': 8, 'p-value': 10}
    header = '  '.join(f'{c:>{widths[c]}}' for c in cols)

    print("\nOne-sample t-test: pain_scale reduction vs 0  (H1: reduction > 0)\n")
    print(header)
    print('_' * 66)
    for _, row in result_df.iterrows():
        print('  '.join(f'{str(row[c]):>{widths[c]}}' for c in cols))

    return result_df


def plot_diagnosis_reduction(df_cl_vis, col, timepoints, min_n=10, figsize=(14, 6)):
    """
    Boxplot of pain reduction by diagnosis, computed from df_cl_vis. 

    Matches patients present at both timepoints, computes reduction as
    col(t_a) - col(t_b), filters to diagnosis groups with enough patients,
    then runs one-way ANOVA and eventually Tukey HSD post-hoc for significant tests

    Parameters
    ----------
    df_cl_vis  : pd.DataFrame  cleaned clinical dataset
    col        : str           column to compute reduction from, e.g. 'pain_scale'
    timepoints : list[int]     [t_a, t_b], reduction = col_ta - col_tb
    min_n      : int           minimum patients per diagnosis group (default 10)
    figsize    : tuple
    """
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    from statannotations.Annotator import Annotator
    from itertools import combinations

    t_a, t_b     = timepoints[0], timepoints[1]
    target_col   = f'{col}_reduction'
    tp_label     = f'T{t_a}-T{t_b}'

    # 1 — Extract col + diagnosis at t_a; col at t_b; match on Patient
    ta = (df_cl_vis[df_cl_vis['Timepoint'] == t_a][['Patient', 'diagnosis', col]]
          .drop_duplicates('Patient').dropna(subset=[col]))
    tb = (df_cl_vis[df_cl_vis['Timepoint'] == t_b][['Patient', col]]
          .drop_duplicates('Patient').dropna(subset=[col]))

    merged = ta.merge(tb, on='Patient', suffixes=(f'_t{t_a}', f'_t{t_b}'))
    merged[target_col] = merged[f'{col}_t{t_a}'] - merged[f'{col}_t{t_b}']
    merged = merged.dropna(subset=[target_col, 'diagnosis'])
    merged['diagnosis'] = merged['diagnosis'].astype(str)

    print(f"\nPatients with {col} at both T{t_a} and T{t_b}: {len(merged)}")

    # 2 — Filter to diagnosis groups with enough patients
    counts   = merged['diagnosis'].value_counts()
    valid_dx = counts[counts >= min_n].index.tolist()
    plot_df  = merged[merged['diagnosis'].isin(valid_dx)].copy()

    print(f"Diagnosis-groups with n ≥ {min_n}: {valid_dx}")

    # Sort by median descending
    order = (plot_df.groupby('diagnosis')[target_col]
                    .median()
                    .sort_values(ascending=False)
                    .index.tolist())

    # 3 — One-way ANOVA (exclude groups with zero variance this causes F=nan)
    groups          = [plot_df.loc[plot_df['diagnosis'] == dx, target_col].values
                       for dx in order]
    groups          = [g for g in groups if g.std() > 0]
    f_stat, p_anova = stats.f_oneway(*groups)

    # Print summary table
    widths = {'Diagnosis': 28, 'n': 5, 'Mean': 8, 'std': 7, 'F': 7, 'p-value': 10}
    header = '  '.join(f'{c:>{widths[c]}}' for c in widths)

    print(f"\nOne-way ANOVA — {col} reduction by diagnosis ({tp_label})\n")
    print(header)
    print('_' * 84)
    for dx in order:
        vals = plot_df.loc[plot_df['diagnosis'] == dx, target_col].dropna()
        print(f"  {dx:<26}  {len(vals):>{widths['n']}}  {vals.mean():>{widths['Mean']}.3f}  "
              f"{vals.std():>{widths['std']}.3f}  {f_stat:>{widths['F']}.3f}  "
              f"{p_anova:>{widths['p-value']}.2e}")

    # 4 — Tukey HSD post-hoc
    tukey    = pairwise_tukeyhsd(plot_df[target_col], plot_df['diagnosis'])
    tukey_df = pd.DataFrame(
        data    = tukey._results_table.data[1:],
        columns = tukey._results_table.data[0],
    )
    pval_map = {}
    for _, row in tukey_df.iterrows():
        pval_map[(row['group1'], row['group2'])] = row['p-adj']
        pval_map[(row['group2'], row['group1'])] = row['p-adj']

    all_pairs = list(combinations(order, 2))
    sig_pairs = [(a, b) for a, b in all_pairs if pval_map.get((a, b), 1.0) < 0.05]
    sig_pvals = [pval_map[(a, b)] for a, b in sig_pairs]

    # 5 — Plot
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(
        data=plot_df, x='diagnosis', y=target_col,
        order=order, palette='mako', width=0.5,
        flierprops=dict(marker='o', markersize=4, alpha=0.5), ax=ax,
    )
    sns.stripplot(
        data=plot_df, x='diagnosis', y=target_col,
        order=order, color='black', alpha=0.3, size=3, jitter=True, ax=ax,
    )

    # 6 — Annotate only significant Tukey pairs 
    if sig_pairs:
        annotator = Annotator(ax, sig_pairs, data=plot_df,
                              x='diagnosis', y=target_col, order=order)
        annotator.configure(text_format='star', loc='inside', verbose=False)
        annotator.set_pvalues_and_annotate(sig_pvals)

    # 7 — Labels
    anova_str = f"ANOVA F={f_stat:.2f}, p={p_anova:.3f}"
    ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Diagnosis', fontsize=12)
    ax.set_ylabel(f'{col} reduction ({tp_label})', fontsize=12)
    ax.set_title(f'{col} Reduction by Diagnosis — {tp_label}\n'
                 f'(n ≥ {min_n} per group; {anova_str}; Tukey HSD brackets = p<0.05)',
                 fontsize=12)
    ax.tick_params(axis='x', rotation=35)

    for i, dx in enumerate(order):
        n = counts[dx]
        ax.text(i, ax.get_ylim()[0] - 0.15, f'n={n}',
                ha='center', va='top', fontsize=8, color='gray')

    plt.tight_layout()
    plt.show()

    return fig



# ── Pearson correlation ───────────────────────────────────────────────────────

def pearson_correlation(df, ex_cols, name, n_top=40):
    """Compute pairwise Pearson r on only numeric columns, print top pairs, and plot heatmaps.

    Parameters
    ----------
    df      : pd.DataFrame  dataframe (not imputed) 
    ex_cols : list[str]     columns to exclude
    name    : str           label used in titles and print headers
    n_top   : int           number of top pairs to print (default is set to 40)

    Returns
    -------
    pearson_matrix : pd.DataFrame  symmetric correlation matrix
    pearson_pairs  : pd.DataFrame  upper-triangle pairs sorted by |r| descending
    """
    print(f"\nPearson Correlation ({name} dataset)")

    feat_cols      = [c for c in df.select_dtypes(include='number').columns
                      if c not in ex_cols]
    pearson_matrix = df[feat_cols].corr(method='pearson')

    upper_tri = pearson_matrix.where(
        np.triu(np.ones(pearson_matrix.shape), k=1).astype(bool))
    pearson_pairs = (
        upper_tri.stack()
        .reset_index()
        .rename(columns={'level_0': 'Feature_1', 'level_1': 'Feature_2', 0: 'Pearson_r'})
        .assign(Abs_r=lambda x: x['Pearson_r'].abs())
        .sort_values('Abs_r', ascending=False)
        .drop(columns='Abs_r')
        .reset_index(drop=True)
    )

    print(f"\nTop {n_top} Most Correlated Feature Pairs (Pearson r):")
    print("=" * 80)
    print(pearson_pairs.head(n_top).to_string(index=False))

    print(f"\nTop {n_top} Most Negatively Correlated Feature Pairs (Pearson r):")
    print("=" * 80)
    print(upper_tri.stack()
          .reset_index()
          .rename(columns={'level_0': 'Feature_1', 'level_1': 'Feature_2', 0: 'Pearson_r'})
          .sort_values('Pearson_r', ascending=True)
          .head(n_top)
          .reset_index(drop=True)
          .to_string(index=False))

    # Full heatmap plot showing lower triangle only:
    mask_full = np.triu(np.ones_like(pearson_matrix, dtype=bool))
    fig, ax   = plt.subplots(figsize=(18, 16))
    sns.heatmap(
        pearson_matrix, mask=mask_full,
        cmap='mako', center=0, vmin=-1, vmax=1,
        square=True, linewidths=0.2,
        cbar_kws={'label': 'Pearson r', 'shrink': 0.8}, ax=ax,
    )
    ax.set_title(f'Pearson Correlation ({name} Dataset)',
                 fontsize=14, fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=7)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=7)
    plt.tight_layout()
    plt.show()

    return pearson_matrix, pearson_pairs


def pearson_correlation_pain(df_cl_vis, df_im_vis, ex_cols_im, n_top=20):
    """Pearson correlation between each immunological feature and pain_scale.

    Merges df_im_vis with pain_scale from df_cl_vis on (Patient, Timepoint),
    keeping only rows present in both datasets.

    Parameters
    ----------
    df_cl_vis  : pd.DataFrame   clinical dataset (must contain Patient, Timepoint, pain_scale)
    df_im_vis  : pd.DataFrame   immunological dataset
    ex_cols_im : list[str]      non-feature columns in df_im_vis (e.g. ['Patient','Timepoint','Date'])
    n_top      : int            number of top/bottom correlations to print and plot 

    Returns
    -------
    results : pd.DataFrame  columns: Feature, 
    """
    print("\nPearson Correlations - Immunological Features × pain_scale")

    pain   = df_cl_vis[['Patient', 'Timepoint', 'pain_scale']].dropna(subset=['pain_scale'])
    merged = df_im_vis.merge(pain, on=['Patient', 'Timepoint'], how='inner')
    print(f"({merged['Patient'].nunique()} unique patients)")

    feat_cols = [c for c in df_im_vis.columns if c not in ex_cols_im]

    records = []
    for col in feat_cols:
        sub = merged[['pain_scale', col]].dropna()
        if len(sub) < 5:
            continue
        r = stats.pearsonr(sub[col], sub['pain_scale']).statistic
        records.append({'Feature': col, 'r': r})

    results = (pd.DataFrame(records)
               .sort_values('r', ascending=False)
               .reset_index(drop=True))

    top_pos = results.head(n_top)
    top_neg = results.tail(n_top).sort_values('r')

    def _print_table(df, label):
        print(f"\n{label}:")
        print(f"  {'Feature':<35}  {'r':>7}")
        print("  " + "-" * 45)
        for _, row in df.iterrows():
            print(f"  {row['Feature']:<35}  {row['r']:>7.3f}")

    _print_table(top_pos, f"Top {n_top} Most Positive Correlations with pain_scale")
    _print_table(top_neg, f"Top {n_top} Most Negative Correlations with pain_scale")

    def _bar_plot(df, title, color):
        fig, ax = plt.subplots(figsize=(9, 0.45 * len(df) + 1.5))
        ax.barh(df['Feature'][::-1], df['r'][::-1], color=color)
        ax.set_xlabel('Pearson r')
        ax.set_title(title)
        plt.tight_layout()
        plt.show()

    _bar_plot(top_pos,
              f'Top {n_top} Positive Correlations Immunological Features × pain_scale',
              '#d62728')
    _bar_plot(top_neg,
              f'Top {n_top} Negative Correlations — Immunological Features × pain_scale',
              '#1f77b4')

    return results


# ── Phik correlation ──────────────────────────────────────────────────────────

def phik_correlation(df, ex_cols, num_cols, name, n_top=40):
    """Compute phik correlation matrix (can use on data with mixed feature types)

    Parameters
    ----------
    df       : pd.DataFrame  dataframe
    ex_cols  : list[str]     columns to exclude 
    num_cols : list[str]     numeric columns (needed for phik interval_cols)
    name     : str           label used in titles
    n_top    : int           number of top pairs to print, standard is top 40.

    Returns
    -------
    phik_matrix : pd.DataFrame
    phik_pairs  : pd.DataFrame  sorted by phik descending
    """
    print(f"\nPhik Correlation ({name} Dataset) ")

    feat_cols = [c for c in df.columns if c not in ex_cols]
    df_phik   = df[feat_cols].copy()

    # phik requires categorical columns to be string-typed (not category dtype);
    # also ensure no literal 'nan' strings slip through as valid categories:
    for c in df_phik.select_dtypes(['category', 'object']).columns:
        df_phik[c] = df_phik[c].astype(str).replace('nan', np.nan)

    interval_cols = [c for c in num_cols if c in feat_cols]
    phik_matrix   = df_phik.phik_matrix(interval_cols=interval_cols)

    upper = phik_matrix.where(
        np.triu(np.ones(phik_matrix.shape), k=1).astype(bool))
    phik_pairs = (
        upper.stack()
        .reset_index()
        .rename(columns={'level_0': 'Feature_1', 'level_1': 'Feature_2', 0: 'phik'})
        .sort_values('phik', ascending=False)
        .reset_index(drop=True)
    )
    nonzero_pairs = phik_pairs[phik_pairs['phik'] > 0].reset_index(drop=True)
    zero_pairs    = phik_pairs[phik_pairs['phik'] == 0].reset_index(drop=True)

    print(f"\nTop {n_top} Most Strongly Associated Feature Pairs (phik):")
    print("=" * 80)
    print(nonzero_pairs.head(n_top).to_string(index=False))

    print(f"\nTop {n_top} Weakest Computable Associations (phik > 0, sorted ascending):")
    print("=" * 80)
    print(nonzero_pairs.tail(n_top).sort_values('phik').reset_index(drop=True).to_string(index=False))


    # Full heatmap
    mask_full = np.triu(np.ones_like(phik_matrix, dtype=bool))
    fig, ax   = plt.subplots(figsize=(16, 14))
    sns.heatmap(
        phik_matrix, mask=mask_full,
        cmap='mako', vmin=0, vmax=1,
        square=True, linewidths=0.2,
        cbar_kws={'label': 'phik', 'shrink': 0.8}, ax=ax,
    )
    ax.set_title(f'Phik Correlation ({name} Dataset)',
                 fontsize=14, fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    plt.tight_layout()
    plt.show()

    return phik_matrix, phik_pairs


def phik_correlation_pain(df, target, ex_cols, name, num_cols):
    """PhiK correlation between every feature and a single target column.

    Computes the full phik_matrix and extracts the target column, so all
    pairwise statistics (including significance) are consistent with the
    matrix computation. PhiK is [0,1] with no direction.

    Parameters
    ----------
    df       : pd.DataFrame  dataframe (must contain target column)
    target   : str           name of the target column (e.g. 'pain_scale')
    ex_cols  : list[str]     columns to exclude (target must NOT be in ex_cols)
    name     : str           label used in titles
    num_cols : list[str]     numeric/interval columns for phik

    Returns
    -------
    result : pd.DataFrame  columns: Feature, phik  (sorted descending, target excluded)
    """
    print(f"\nPhiK — Features × {target} ({name} Dataset)")

    feat_cols = [c for c in df.columns if c not in ex_cols]
    if target not in feat_cols:
        raise ValueError(f"'{target}' not in feat_cols — remove it from ex_cols")

    df_phik = df[feat_cols].copy()
    for c in df_phik.select_dtypes(['category', 'object']).columns:
        df_phik[c] = df_phik[c].astype(str).replace('nan', np.nan)

    interval_cols = [c for c in num_cols if c in feat_cols]
    phik_matrix   = df_phik.phik_matrix(interval_cols=interval_cols)

    result = (phik_matrix[[target]]
              .drop(index=target)
              .rename(columns={target: 'phik'})
              .reset_index()
              .rename(columns={'index': 'Feature'})
              .sort_values('phik', ascending=False)
              .reset_index(drop=True))

    nonzero = result[result['phik'] > 0].reset_index(drop=True)
    zero_n  = (result['phik'] == 0).sum()

    print(f"  Features: {len(result)}  |  computable (phik > 0): {len(nonzero)}  "
          f"|  skipped (phik = 0): {zero_n}")

    print(f"\nAll features × '{target}' (phik, sorted descending):")
    print(f"  {'Feature':<35}  {'phik':>6}")
    print("  " + "-" * 45)
    for _, row in nonzero.iterrows():
        print(f"  {row['Feature']:<35}  {row['phik']:>6.3f}")

    fig, ax = plt.subplots(figsize=(9, 0.45 * len(nonzero) + 1.5))
    ax.barh(nonzero['Feature'][::-1], nonzero['phik'][::-1], color='#2ca02c')
    ax.set_xlabel('PhiK')
    ax.set_xlim(0, 1)
    ax.set_title(f'All Features × {target} ({name} Dataset, PhiK)')
    plt.tight_layout()
    plt.show()

    return result


# ── RV2 matrix ────────────────────────────────────────────────────────────────

def rv2_matrix(df, timepoints, ex_cols, name):
    """Compute pairwise RV2 similarity matrix across timepoints.

    Uses missing-methods and therefore handles NaN natively.

    Parameters
    ----------
    df         : pd.DataFrame   dataframe
    timepoints : list[int]      list of timepoints to include
    ex_cols    : list[str]      list of columns to exclude (for dataset with mixed types, exclude categorical columns)
    name       : str            label for titles and prints
   
    Returns
    -------
    rv2_df : pd.DataFrame  symmetric RV2 matrix with T-labels

    """
    dfs_tp  = {t: df[df['Timepoint'] == t] for t in timepoints}
    pt_sets = {t: set(dfs_tp[t]['Patient']) for t in timepoints}
    n_tp    = len(timepoints)
    rv2_mat = np.zeros((n_tp, n_tp))
    n_comm  = np.zeros((n_tp, n_tp), dtype=int)

    def _get_feat_vals(df_t, patients):
        df_f = df_t[df_t['Patient'].isin(patients)].sort_values('Patient')
        cols = [c for c in df_f.columns if c not in ex_cols]
        return df_f[cols].values.astype(float)

    for i, ti in enumerate(timepoints):
        rv2_mat[i, i] = 1.0
        n_comm[i, i]  = len(dfs_tp[ti])

    for (i, ti), (j, tj) in _combns(enumerate(timepoints), 2):
        common         = pt_sets[ti] & pt_sets[tj]
        n              = len(common)
        n_comm[i, j]   = n_comm[j, i] = n
        A = MM_StandardScaler().fit_transform(_get_feat_vals(dfs_tp[ti], common))
        B = MM_StandardScaler().fit_transform(_get_feat_vals(dfs_tp[tj], common))
        rv2_mat[i, j]  = rv2_mat[j, i] = mm_rv2(A, B)

    rv2_df = pd.DataFrame(
        rv2_mat,
        index=[f"T{t}" for t in timepoints],
        columns=[f"T{t}" for t in timepoints],
    )
    annot = pd.DataFrame(
        [[f"{rv2_mat[i,j]:.2f}\n(n={n_comm[i,j]})" for j in range(n_tp)]
         for i in range(n_tp)],
        index=rv2_df.index, columns=rv2_df.columns,
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(rv2_df, annot=annot, fmt="", cmap="mako_r",
                vmin=0, vmax=1, square=True, ax=ax)
    ax.set_title(f"RV2 Similarity Matrix for {name} Dataset")
    plt.tight_layout()
    plt.show()

    return rv2_df


# ── PCA per timepoint ─────────────────────────────────────────────────────────

def pca_per_timepoint(df, timepoints, ex_cols, name, ncomp=10):
    """Run PCA per timepoint, plot scree + score plots, print loadings.

    Parameters
    ----------
    df         : pd.DataFrame   dataframe (not imputed)
    timepoints : list[int]
    ex_cols    : list[str]      columns excluded from feature matrix
    name       : str            label for titles
    ncomp      : int            number of PCs to extract

    Returns
    -------
    pca_store : dict  {t: {scores, loadings, exp, df, patient_ids, feat_names}}
    """
    print(f"\nPCA per Timepoint for {name} Dataset")

    mako_tp = sns.color_palette("mako", len(timepoints))
    cum_col = sns.color_palette("crest", 1)[0]
    pca_store = {}

    for idx, t in enumerate(timepoints):
        df_t        = df[df['Timepoint'] == t].reset_index(drop=True)
        n_t         = len(df_t)
        patient_ids = df_t['Patient'].values
        cols        = [c for c in df_t.columns if c not in ex_cols]
        feat_names  = [c for c in cols if c in df_t.columns]

        Xs       = MM_StandardScaler().fit_transform(df_t[feat_names].values.astype(float))
        res      = mm_pca(Xs, ncomp=ncomp)
        scores   = res['scores']
        loadings = res['loadings']
        exp      = res['explained'] / res['explained'].sum() * 100

        pca_store[t] = {
            'scores':      scores,
            'loadings':    loadings,
            'exp':         exp,
            'df':          df_t,
            'patient_ids': patient_ids,
            'feat_names':  feat_names,
        }

        # Scree plot
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(range(1, ncomp + 1), exp,
               color=sns.color_palette("mako", ncomp), label="Per-PC %")
        ax.plot(range(1, ncomp + 1), np.cumsum(exp),
                marker="o", color=cum_col, linewidth=1.5, label="Cumulative %")
        ax.set_xticks(range(1, ncomp + 1))
        ax.set_xlabel("Principal Components")
        ax.set_ylabel("Explained Variance (%)")
        ax.set_title(f"Scree Plot for {name} Dataset T{t}")
        ax.legend()
        plt.tight_layout()
        plt.show()

        # Score plot labeling top 20 furthest from origin
        dist  = np.sqrt(scores[:, 0]**2 + scores[:, 1]**2)
        top20 = np.argsort(dist)[::-1][:20]

        fig, ax = plt.subplots(figsize=(9, 7))
        ax.scatter(scores[:, 0], scores[:, 1],
                   c=[mako_tp[idx]], s=40, zorder=3,
                   edgecolors='white', linewidth=0.4, alpha=0.85,
                   label=f"T{t} (n={n_t})")
        texts = [ax.text(scores[i, 0], scores[i, 1], str(patient_ids[i]),
                         fontsize=7, fontweight='bold', color='black', zorder=5)
                 for i in top20]
        _adj(texts, ax=ax, expand=(1.5, 1.5),
             arrowprops=dict(arrowstyle="-", color="grey", lw=0.5))
        ax.axhline(0, color='grey', lw=0.5, linestyle='--')
        ax.axvline(0, color='grey', lw=0.5, linestyle='--')
        ax.set_xlabel(f"PC1 ({exp[0]:.1f}% variance)")
        ax.set_ylabel(f"PC2 ({exp[1]:.1f}% variance)")
        ax.set_title(f"PCA Score Plot for {name} Dataset T{t}\n"
                     f"(top 20 patients furthest from pca-origin are labelled)")
        ax.legend(loc='best')
        plt.tight_layout()
        plt.show()

        # Print top 20 patients furthest from pca-origin
        print(f"  Top 20 patients furthest from pca-origin at T{t}:")
        print(f"  {'Patient':>10}  {'PC1':>8}  {'PC2':>8}  {'Distance':>10}")
        for i in top20:
            print(f"  {patient_ids[i]:>10}  "
                  f"{scores[i,0]:>8.3f}  {scores[i,1]:>8.3f}  "
                  f"{dist[i]:>10.3f}")

        # Top 10 loadings for PC1 and PC2
        for pc_i, pc_name in enumerate(['PC1', 'PC2']):
            abs_l  = np.abs(loadings[:, pc_i])
            top10l = np.argsort(abs_l)[::-1][:10]
            print(f"\n  Top 10 loadings for {pc_name} (T{t}):")
            print(f"  {'Feature':>40}  {'Loading':>10}")
            for k in top10l:
                print(f"  {feat_names[k]:>40}  {loadings[k, pc_i]:>10.4f}")

    return pca_store


# ── PCA colored by clinical adata ───────────────────────────────────────────────────

def pca_colored(pca_store, timepoints, color_configs, name,
                color_source_df=None):
    
    """Plotting PCA score plots T1–T5, colored by clinical variables.

    Parameters
    ----------
    pca_store       : dict,  saved output of pca_per_timepoint          
    timepoints      : list[int]
    color_configs   : list of tuples (col_name, col_type, palette)
                      col_type ∈ {'categorical', 'continuous'}
    name            : str   label for figure title
    color_source_df : pd.DataFrame  or None - reference for clinical dataset
                      If None: reads color values from pca_store (self-coloring).
                      If provided: looks up patient IDs at the matching timepoint (cross-dataset coloring).
    """

    for col, col_type, palette in color_configs:
        fig, axes = plt.subplots(1, len(timepoints), figsize=(22, 5), sharey=False)
        fig.suptitle(f'{name} PCA Score Plots T1–T5  |  coloured by {col}',
                     fontsize=13, fontweight='bold')

        if col_type == 'categorical':
            if color_source_df is None:
                all_vals = pd.concat([
                    pca_store[t]['df'][col].astype(str)
                    for t in timepoints
                    if col in pca_store[t]['df'].columns
                ]).replace({'nan': np.nan, '<NA>': np.nan}).dropna().unique()
            else:
                all_vals = (
                    color_source_df[col].astype(str)
                    .replace({'nan': np.nan, '<NA>': np.nan})
                    .dropna().unique()
                )
            categories    = sorted(all_vals)
            cat_palette   = sns.color_palette(palette, len(categories))
            cat_color_map = dict(zip(categories, cat_palette))

        for i, t in enumerate(timepoints):
            ax      = axes[i]
            d       = pca_store[t]
            scores  = d['scores']
            exp_t   = d['exp']
            pt_ids  = d['patient_ids']
            n_t     = len(pt_ids)

            ax.axhline(0, color='grey', lw=0.5, linestyle='--')
            ax.axvline(0, color='grey', lw=0.5, linestyle='--')

            # Resolve color values: self-coloring or cross-dataset
            if color_source_df is None:
                df_t      = d['df']
                color_ser = (df_t[col] if col in df_t.columns
                             else pd.Series([np.nan] * n_t))
            else:
                cl_lookup = (
                    color_source_df[color_source_df['Timepoint'] == t]
                    .set_index('Patient')[col]
                )
                color_ser = pd.Series(
                    [cl_lookup.loc[p] if p in cl_lookup.index else np.nan
                     for p in pt_ids],
                    dtype=object,
                )

            if col_type == 'continuous':
                vals  = pd.to_numeric(color_ser, errors='coerce').values
                valid = ~np.isnan(vals)
                sc = ax.scatter(
                    scores[valid, 0], scores[valid, 1],
                    c=vals[valid], cmap='mako', vmin=0, vmax=10,
                    s=30, alpha=0.85, edgecolors='white', linewidth=0.3, zorder=3)
                if (~valid).sum() > 0:
                    ax.scatter(scores[~valid, 0], scores[~valid, 1],
                               c='lightgrey', s=20, alpha=0.5, zorder=1)
                if i == len(timepoints) - 1:
                    fig.colorbar(sc, ax=ax, label=col, shrink=0.85)
            else:  # categorical
                vals_str = color_ser.astype(str).replace({'nan': np.nan, '<NA>': np.nan})
                for cat in categories:
                    mask = (vals_str == cat).values
                    if mask.sum() > 0:
                        ax.scatter(
                            scores[mask, 0], scores[mask, 1],
                            color=cat_color_map[cat], s=30, alpha=0.85,
                            edgecolors='white', linewidth=0.3, zorder=3,
                            label=cat if i == 0 else '_nolegend_')
                nan_mask = vals_str.isna().values
                if nan_mask.sum() > 0:
                    ax.scatter(scores[nan_mask, 0], scores[nan_mask, 1],
                               c='lightgrey', s=20, alpha=0.5, zorder=1,
                               label='missing' if i == 0 else '_nolegend_')

            ax.set_xlabel(f"PC1 ({exp_t[0]:.1f}%)", fontsize=9)
            ax.set_ylabel(f"PC2 ({exp_t[1]:.1f}%)", fontsize=9)
            ax.set_title(f"T{t}  (n={n_t})", fontsize=10)

        if col_type == 'categorical':
            axes[0].legend(fontsize=7, loc='best', framealpha=0.7)
        plt.tight_layout()
        plt.show()



# ── PyOD Zryan outlier detection ──────────────────────────────────────────────

def run_pyod_zryan(df_imputed, feature_cols, contamination=0.05, name='', random_state=42):
    """Ensemble outlier detection using Zryan approach. Code is adapted from Zryan´s original github repo:

    Pipeline:
      1. Scale feature columns with StandardScaler.
      2. GEC selects 6 most dissimilar algorithms from a candidate pool of 11.
      3. visualiser_OD fits ensemble, aggregates median probability + confidence, plots.
      4. Print summary; return outlier candidate DataFrame.

    Requires df_imputed (no NaN in feature_cols) — run after impute_miceforest or
    impute_median.

    Parameters
    ----------
    df_imputed   : pd.DataFrame  imputed dataset, with numeric features only
    feature_cols : list[str]     feature columns
    contamination: float
    name         : str           label for print headers
    random_state : int

    Returns
    -------
    no_od_df          : pd.DataFrame  flagged-by-N-algorithms summary
    outlier_candidates: pd.DataFrame  high probability + confidence observations
    """
    import sys
    import random
    from pathlib import Path
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
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from pyod_zyran.GEC import calculate_GEC
    from pyod_zyran.Visualisering import visualiser_OD

    if not hasattr(np, 'bool'):
        np.bool = bool

    print(f"\nPyOD Outlier Detection for {name} dataset")

    X_ens          = df_imputed[feature_cols].copy()
    patient_labels = (
        df_imputed['Patient'].astype(str) + "-T" +
        df_imputed['Timepoint'].astype(str)
    ).tolist()

    scaler = StandardScaler()
    X_sc   = pd.DataFrame(scaler.fit_transform(X_ens), columns=X_ens.columns)

    random.seed(random_state)
    detector_list_lscp = [IForest_od(n_estimators=n)
                          for n in random.sample(range(5, 200), 10)]

    list_OD_classes = [QMCD, INNE, KNN_od, LOF_od, IForest_od, PCA_od,
                       LODA, HBOS, OCSVM, ECOD_od, COPOD_od]
    list_OD_strings = [cls.__name__ for cls in list_OD_classes]
    list_OD_init    = [
        LSCP(detector_list=detector_list_lscp, contamination=contamination)
        if cls == LSCP
        else cls(contamination=contamination)
        for cls in list_OD_classes
    ]

    print("Running GEC to select 6 most dissimilar algorithms...")
    final_algos, _ = calculate_GEC(
        X_sc.values, list_OD_init, list_OD_strings,
        percentages=[0.90, 0.98, 1.00],
    )
    print(f"GEC selected: {final_algos}")

    algo_class_map = {cls.__name__: cls for cls in list_OD_classes}
    modules = [
        algo_class_map[n](contamination=contamination)
        for n in final_algos if n in algo_class_map
    ]
    print(f"Ensemble: {len(modules)} algorithms, contamination={contamination}")

    print("Running visualiser_OD...")
    no_od_df, y_prob_mean, y_conf_mean, y_prob_arr, y_conf_arr, _ = visualiser_OD(
        X_sc, modules, patient_labels, visualize=True,
    )

    print(f"\n=== Outlier Detection Summary — {name} (contamination={contamination}) ===")
    for n in [1, 3, len(modules)]:
        lbl = f"Flagged by >= {n} algorithm{'s' if n > 1 else ''}"
        print(f"{lbl}: {(no_od_df['No. OD Detected'] >= n).sum()}")

    mask = (y_prob_mean > 0.9) & (y_conf_mean > 0.9)
    outlier_candidates = no_od_df[mask].copy()
    outlier_candidates['Median_Probability'] = y_prob_mean[mask]
    outlier_candidates['Avg_Confidence']     = y_conf_mean[mask]
    outlier_candidates = outlier_candidates.sort_values('Median_Probability', ascending=False)

    print(f"\n=== Top Upper-right Quadrant (median prob. > 0.9 & avg conf. > 0.9) — {name} ===")
    print(f"Total: {len(outlier_candidates)}")
    print(outlier_candidates.to_string())

    return no_od_df, outlier_candidates


# ── Trajectory PCA — immunological  ───────────────────────────────────

def trajectory_pca_im(df, pairs, ex_cols, ncomp=10):
    """Trajectory PCA, stacking two timepoints together and drawing arrows for patient trajectories..

    Parameters
    ----------
    df      : pd.DataFrame   dataframe (not imputed)
    pairs   : list of tuples  (tp_a, tp_b, arrow_color, label)
              e.g. [(1, 2, color, 'T1 → T2'), ...]
    ex_cols : list[str]
    ncomp   : int
    """
    print("\nTrajectory PCA — immunological dataset")

    cum_col   = sns.color_palette("crest", 1)[0]
    _mako5    = sns.color_palette("mako", 5)
    tp_colors = {1: _mako5[0], 2: _mako5[2], 3: _mako5[4]}
    tp_labels = {1: "T1", 2: "T2", 3: "T3"}

    def _filter_tp(tp, patients):
        return (df[(df['Timepoint'] == tp) & (df['Patient'].isin(patients))]
                .sort_values('Patient').reset_index(drop=True))

    for tp_a, tp_b, arrow_color, label in pairs:
        patients_pair = (
            set(df[df['Timepoint'] == tp_a]['Patient'])
            & set(df[df['Timepoint'] == tp_b]['Patient'])
        )
        n_pair = len(patients_pair)
        print(f"  {label}: {n_pair} patients")

        df_a       = _filter_tp(tp_a, patients_pair)
        df_b       = _filter_tp(tp_b, patients_pair)
        feat_names = [c for c in df_a.columns if c not in ex_cols]

        X_pair  = np.vstack([df_a[feat_names].values.astype(float),
                              df_b[feat_names].values.astype(float)])
        X_pair  = MM_StandardScaler().fit_transform(X_pair)

        res         = mm_pca(X_pair, ncomp=ncomp)
        scores      = res['scores']
        loadings    = res['loadings']
        exp_pct     = res['explained'] / res['explained'].sum() * 100
        patient_ids = df_a['Patient'].values
        sc_a        = scores[:n_pair, :]
        sc_b        = scores[n_pair:, :]

        # Scree plot
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(range(1, ncomp + 1), exp_pct,
               color=sns.color_palette("mako", ncomp), label="Per-PC %")
        ax.plot(range(1, ncomp + 1), np.cumsum(exp_pct),
                marker="o", color=cum_col, linewidth=1.5, label="Cumulative %")
        ax.set_xticks(range(1, ncomp + 1))
        ax.set_xlabel("Principal Components.")
        ax.set_ylabel("Explained Variance (%)")
        ax.set_title(f"Scree Plot — Immunological Dataset\n{label}")
        ax.legend()
        plt.tight_layout()
        plt.show()

        # Trajectory lengths table
        N_PRINT  = 20
        traj_len = np.sqrt((sc_b[:, 0] - sc_a[:, 0])**2 + (sc_b[:, 1] - sc_a[:, 1])**2)
        top_idx  = np.argsort(traj_len)[::-1][:N_PRINT]

        print(f"\n  Top {N_PRINT} Largest Trajectory Lengths {label}:")
        print(f"  {'Patient':>10}  {'PC1 T'+str(tp_a):>9}  {'PC2 T'+str(tp_a):>9}"
              f"  {'PC1 T'+str(tp_b):>9}  {'PC2 T'+str(tp_b):>9}  {'Traj. length':>13}")
        for i in top_idx:
            print(f"  {patient_ids[i]:>10}"
                  f"  {sc_a[i,0]:>9.3f}  {sc_a[i,1]:>9.3f}"
                  f"  {sc_b[i,0]:>9.3f}  {sc_b[i,1]:>9.3f}")
             

        # Top 10 loadings
        for pc_i, pc_name in enumerate(['PC1', 'PC2']):
            abs_l  = np.abs(loadings[:, pc_i])
            top10l = np.argsort(abs_l)[::-1][:10]
            print(f"\n  Top 10 loadings for {pc_name} ({label}):")
            print(f"  {'Feature':>40}  {'Loading':>10}")
            for k in top10l:
                print(f"  {feat_names[k]:>40}  {loadings[k, pc_i]:>10.4f}")

        # Trajectory score plot
        label_idx = np.argsort(traj_len)[::-1][:20]
        fig, ax   = plt.subplots(figsize=(11, 9))

        ax.scatter(sc_a[:, 0], sc_a[:, 1],
                   c=[tp_colors.get(tp_a, _mako5[0])],
                   label=tp_labels.get(tp_a, f"T{tp_a}"),
                   s=40, zorder=3, edgecolors='white', linewidth=0.4, alpha=0.8)
        ax.scatter(sc_b[:, 0], sc_b[:, 1],
                   c=[tp_colors.get(tp_b, _mako5[-1])],
                   label=tp_labels.get(tp_b, f"T{tp_b}"),
                   s=40, zorder=3, edgecolors='white', linewidth=0.4, alpha=0.8)

        for i in range(n_pair):
            ax.annotate(
                "", xy=(sc_b[i, 0], sc_b[i, 1]), xytext=(sc_a[i, 0], sc_a[i, 1]),
                annotation_clip=False,
                arrowprops=dict(arrowstyle="-|>", color=arrow_color,
                                lw=0.8, alpha=0.3, mutation_scale=7),
            )

        texts = []
        for i in label_idx:
            mx = (sc_a[i, 0] + sc_b[i, 0]) / 2
            my = (sc_a[i, 1] + sc_b[i, 1]) / 2
            texts.append(ax.text(mx, my, str(patient_ids[i]),
                                 fontsize=8, fontweight='bold', color='black', zorder=5))
        _adj(texts, ax=ax, expand=(1.5, 1.5),
             arrowprops=dict(arrowstyle="-", color="grey", lw=0.6))

        ax.axhline(0, color='grey', lw=0.5, linestyle='--')
        ax.axvline(0, color='grey', lw=0.5, linestyle='--')
        ax.set_xlabel(f"PC1 ({exp_pct[0]:.1f}% variance)")
        ax.set_ylabel(f"PC2 ({exp_pct[1]:.1f}% variance)")
        ax.set_title(f"Trajectory PCA for Immunological Dataset\n"
                     f"{label}  (top 20 longest trajectories labelled)")
        ax.legend(loc='best')
        plt.tight_layout()
        plt.show()


# ── MFA — immunological  ─────────────────────────────────────────────

def mfa_im(df, timepoints, ex_cols, ncomp=5):
    """Plotting Multiple Factor Analysis, 

    Each timepoint is a block, normalised by sqrt(first eigenvalue), then stacked
    horizontally for a joint PCA. Uses missing-methods so no imputation is required.

    Parameters
    ----------
    df         : pd.DataFrame  df_im_vis (NOT imputed)
    timepoints : list[int]     timepoints to include (e.g. [1, 2, 3])
    ex_cols    : list[str]
    ncomp      : int
    """
    tp_label = '+'.join(f"T{t}" for t in timepoints)
    print(f"\nMFA {tp_label} — immunological dataset")

    pt_sets      = {t: set(df[df['Timepoint'] == t]['Patient']) for t in timepoints}
    patients_mfa = set.intersection(*pt_sets.values())
    n_mfa        = len(patients_mfa)
    print(f"  Patients with all timepoints ({tp_label}): {n_mfa}")

    def _get_block(tp):
        return (df[(df['Timepoint'] == tp) & (df['Patient'].isin(patients_mfa))]
                .sort_values('Patient').reset_index(drop=True))

    blocks      = {t: _get_block(t) for t in timepoints}
    patient_ids = blocks[timepoints[0]]['Patient'].values
    feat_cols   = [c for c in blocks[timepoints[0]].columns if c not in ex_cols]
    feat_names  = [f"T{t}_{c}" for t in timepoints for c in feat_cols]

    def _mfa_normalise(X):
        Xs   = MM_StandardScaler().fit_transform(X)
        lam1 = mm_pca(Xs, ncomp=1)['explained'][0]
        return Xs / np.sqrt(lam1)

    X_mfa = np.hstack([
        _mfa_normalise(blocks[t][feat_cols].values.astype(float))
        for t in timepoints
    ])

    res      = mm_pca(X_mfa, ncomp=ncomp)
    scores   = res['scores']
    loadings = res['loadings']
    exp      = res['explained'] / res['explained'].sum() * 100

    # Scree plot
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(1, ncomp + 1), exp,
           color=sns.color_palette("mako", ncomp), label="Per-PC %")
    ax.plot(range(1, ncomp + 1), np.cumsum(exp),
            marker="o", color=sns.color_palette("crest", 1)[0],
            linewidth=1.5, label="Cumulative %")
    ax.set_xticks(range(1, ncomp + 1))
    ax.set_xlabel("Principal Components.")
    ax.set_ylabel("Explained Variance (%)")
    ax.set_title(f"Scree Plot for MFA of Immunological Dataset {tp_label})")
    ax.legend()
    plt.tight_layout()
    plt.show()

    # Score plot of top 20 furthest from origin
    dist  = np.sqrt(scores[:, 0]**2 + scores[:, 1]**2)
    top20 = np.argsort(dist)[::-1][:20]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(scores[:, 0], scores[:, 1],
               c=[sns.color_palette("mako", 1)[0]],
               s=40, zorder=3, edgecolors='white', linewidth=0.4, alpha=0.85,
               label=f"Patients (n={n_mfa})")
    texts = [ax.text(scores[i, 0], scores[i, 1], str(patient_ids[i]),
                     fontsize=7, fontweight='bold', color='black', zorder=5)
             for i in top20]
    _adj(texts, ax=ax, expand=(1.5, 1.5),
         arrowprops=dict(arrowstyle="-", color="grey", lw=0.5))
    ax.axhline(0, color='grey', lw=0.5, linestyle='--')
    ax.axvline(0, color='grey', lw=0.5, linestyle='--')
    ax.set_xlabel(f"PC1 ({exp[0]:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({exp[1]:.1f}% variance)")
    ax.set_title(f"MFA Score Plot For Immunological Dataset {tp_label}\n"
                 f"top 20 furthest from origin labelled)")
    ax.legend(loc='best')
    plt.tight_layout()
    plt.show()

    # Top 10 loadings for PC1 and PC2
    for pc_i, pc_name in enumerate(['PC1', 'PC2']):
        abs_l  = np.abs(loadings[:, pc_i])
        top10l = np.argsort(abs_l)[::-1][:10]
        print(f"\n  Top 10 loadings — {pc_name} (MFA {tp_label}):")
        print(f"  {'Feature':>45}  {'Loading':>10}")
        for k in top10l:
            print(f"  {feat_names[k]:>45}  {loadings[k, pc_i]:>10.4f}")



# ── Immunological features × pain_scale (Pearson) ────────────────────────────
