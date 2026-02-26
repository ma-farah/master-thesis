# imports
import pandas as pd
import numpy as np
import re
from pathlib import Path
from skrub import TableReport
import scikit_na as na
import hoggorm as ho
import prince as ps
import matplotlib.pyplot as plt
import seaborn as sns
import phik
from missing_methods import pca as mm_pca, rv2 as mm_rv2
from missing_methods.sk import StandardScaler as MM_StandardScaler
from adjustText import adjust_text as _adj


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

def dataset_overview(df, name, patient_col='Patient', timepoint_col='Timepoint',
                     max_plot_columns=138):
    """Print TableReport, NA heatmap and basic statistics for a raw dataset.

    Parameters
    ----------
    df               : pd.DataFrame
    name             : str   label used in print headers (e.g. 'Immunological')
    patient_col      : str   column containing patient IDs
    timepoint_col    : str   column containing timepoint labels
    max_plot_columns : int   passed to TableReport
    """
    print(f"\n{'='*60}")
    print(f"  {name} Dataset Overview")
    print(f"{'='*60}")

    print(f"\nTableReport of raw {name} dataset:")
    TableReport(df, max_plot_columns=max_plot_columns)

    print(f"\nNA heatmap — {name} dataset:")
    na.altair.plot_heatmap(df)

    print(f"\n=== Raw {name} dataset statistics ===")
    print(f"  Shape         : {df.shape[0]} rows × {df.shape[1]} columns")
    if patient_col in df.columns:
        print(f"  Patients      : {df[patient_col].dropna().nunique()}")
    if timepoint_col in df.columns:
        print(f"  Timepoints    : {df[timepoint_col].dropna().nunique()}")
        print(f"\n  Measurements per timepoint:")
        print(df[timepoint_col].value_counts().sort_index().to_string())
    print(f"\n  Missing values: {df.isna().sum().sum()} total "
          f"({df.isna().mean().mean()*100:.1f}% of all cells)")


# ── Patient timepoint coverage ────────────────────────────────────────────────

def patient_timepoint_summary(df, name, patient_col='Patient', timepoint_col='Timepoint',
                               timepoints=None):
    """Print cumulative patient coverage across timepoints and show a bar plot.

    Parameters
    ----------
    df            : pd.DataFrame
    name          : str   label used in titles
    patient_col   : str
    timepoint_col : str
    timepoints    : list of int, optional
        Ordered list of timepoints to check. Defaults to sorted unique values.
    """
    if timepoints is None:
        timepoints = sorted(df[timepoint_col].dropna().unique())

    pt_sets = {
        t: set(df.loc[df[timepoint_col] == t, patient_col].dropna())
        for t in timepoints
    }

    print(f"\n=== Patient timepoint coverage — {name} ===")

    # Cumulative intersection: patients present at T1, T1+T2, T1+T2+T3, …
    cumulative = pt_sets[timepoints[0]]
    tp_labels  = [f"T{timepoints[0]}"]
    for t in timepoints[1:]:
        tp_labels.append(f"T{t}")
        print(f"  Patients with measurements at {' & '.join(tp_labels)}: "
              f"{len(cumulative & pt_sets[t])}")
        cumulative = cumulative & pt_sets[t]

    # Patients with ONLY T1 (no follow-up)
    others = set().union(*(pt_sets[t] for t in timepoints[1:]))
    print(f"  Patients at T{timepoints[0]} only (no follow-up): "
          f"{len(pt_sets[timepoints[0]] - others)}")

    # Bar plot: unique patients per timepoint
    patient_counts = df.groupby(timepoint_col)[patient_col].nunique().sort_index()
    _bar_color = sns.color_palette("mako", len(patient_counts))

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=patient_counts.index, y=patient_counts.values,
                palette=_bar_color, ax=ax)
    ax.set_title(f"Unique Patients per Timepoint — {name} Dataset")
    ax.set_xlabel("Timepoint")
    ax.set_ylabel("Number of unique patients")
    plt.tight_layout()
    plt.show()


