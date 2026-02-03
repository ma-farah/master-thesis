# Loading and exploring raw datasets

# Lag Klasse med funksjoner - lese, rense, imputere 

#%%
import pandas as pd
from pathlib import Path

# reading excel file with raw data
data_dir = Path(__file__).resolve().parents[1] / "data"
data = data_dir / "LDRT_raw.xlsx"

#%%

# immunological data/blood samples, columns starts at row 5
df_im = pd.read_excel(
    data,
    sheet_name="IPT ",
    header=4,
    engine="openpyxl"
)

# patient IDs from 1 - 269

# Clinical data and questionarries, columns starts at row 2
df_cl = pd.read_excel(
    data,
    sheet_name="Patient data & Pain",
    header=1,
    engine="openpyxl"
)

# Patient IDs from 1 to 276. 


# %%################ RAW CLINICAL DATASET #############################
from skrub import TableReport
import scikit_na as na

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


# checking rows with missing values, and patient ids: 
rows_with_na = df_im[df_im.isna().any(axis=1)]
print("patient id-s with missing values:")
rows_with_na["Patient"].value_counts()

# Patient Ids : 30, 223, 224, 226, 227, 228, 229 and 230 have missing values in one row/timepoint each.



#%%################# Patients and number of timepoint measurements ###################

group_sizes = df_im.groupby("Patient").size()

for n_timepoints, patients in group_sizes.groupby(group_sizes):
    rows = df_im[df_im["Patient"].isin(patients.index)]

    print(f"Amount of patients with {n_timepoints} measured timepoints: {len(patients)}")
    print(rows[["Patient", "Timepoint"]])

#%% Num. Patients with 1-5 timepoints:
    
tp_per_patient = (
df_im.groupby("Patient")["Timepoint"].apply(lambda x: sorted(set(x)))
)

def max_tp(tps):
    count = 0
    for tp in tps:
        if tp == count + 1:
            count += 1
        else:
            break
    return count

max_tp = tp_per_patient.apply(max_tp)

summary = (
    max_tp.value_counts()
          .sort_index()
          .rename_axis("tp    n")
          .rename("n_patients")
)

print(summary)


#%%#############  CLEANING DATASET ####################################


# Removing columns that can be exlcuded (marked yellow in dataset): 43 columns + Id Subset
# + columns with more than 25 % missing values:

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

# Changing columns with incorrect datatypes
# husk å skrive opp alle kolonner som har blitt endretn
# changing into date/time type
df_im["Messdatum"] = pd.to_datetime(
    df_im["Messdatum"], errors="coerce")

# All columns should be Float type except for "Messdatum" 
exclude_cols = ["Messdatum"]
float_cols = df_im.columns.difference(exclude_cols)
df_im[float_cols] = df_im[float_cols].apply(
    pd.to_numeric, errors="coerce"
)


# Removing empty rows from row 829 tto 834
df_im = df_im.drop(index=range(823, 829)) #is this correct
# remove empty row at index 84
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



#%%################ Imputing missing values using miceforest ###########

# handling name issues - mice forest does not take symbols
import re
import lightgbm
import miceforest as mf
print("lightgbm:", lightgbm.__version__)
print("miceforest:", mf.__version__)

id_cols = ["Patient", "Timepoint", "Messdatum"]
feature_cols = df_im.columns.difference(id_cols)

def clean_colname(col):
    col = col.strip()
    col = re.sub(r"[^\w]", "_", col)
    col = re.sub(r"_+", "_", col)
    return col

# map
rename_map = {c: clean_colname(c) for c in feature_cols}

# rename
df_im2 = df_im.rename(columns=rename_map)

# miceforsest with renamed columns
X_im = df_im2[list(rename_map.values())]

import miceforest as mf

kernel = mf.ImputationKernel(
    data=X_im,
    datasets=3,
    random_state=42
)

kernel.mice(5)

X_imputed_renamed = kernel.complete_data(dataset=1)

# changing back to original names
reverse_rename_map = {v: k for k, v in rename_map.items()}

X_imputed = X_imputed_renamed.rename(columns=reverse_rename_map)

# final imputation
df_im_imputed = pd.concat(
    [
        df_im[id_cols].reset_index(drop=True),
        X_imputed.reset_index(drop=True)
    ],
    axis=1
)

# New tablereport of imputed data
TableReport(df_im_imputed, max_plot_columns=138)


#%%  Array matrix RV / RV2

import hoggorm as ho
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


#NB! Patient ID 83 has two timepoint 4 measurements (take average)
# Needs to be the same shape in order to to RV2 analysis.
# patient ID 137 have two t4 and two t3 measurements.
# Only use the average of both entries for now:

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
            A, B = common_patients(dfs[ti], dfs[tj])   # exctract common patients
            n_common[i, j] = A.shape[0]

            X = ho.standardise(
                A.drop(columns=id_cols).values,
                mode=0
            )

            Y = ho.standardise(
                B.drop(columns=id_cols).values,
                mode=0
            )

            rv2 = ho.RV2coeff([X, Y])[0, 1]     # calculating RV2
            rv2_matrix[i, j] = rv2



# number of common patients inbetween comparisons:
n_common_df = pd.DataFrame(
    n_common,
    index=[f"T{t}" for t in timepoints],
    columns=[f"T{t}" for t in timepoints]
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
# lekocytes stable - mDC downregulate, m1,m3 and m3


#%%############ MFA for timepoints 1, 2 and 3
import prince as ps

# finding common patients with measuements at t1, t2 and t3 all together
patients_t123 = (
    set(df_t1["Patient"])
    & set(df_t2["Patient"])
    & set(df_t3["Patient"])
)

#patients_t123 = sorted(patients_t123)
# number patients with measuements at time 1, 2 and 3 = 121 patients out of 250

def sortdfs(df, patients):
    return (
        df[df["Patient"].isin(patients)]
        .sort_values("Patient")
        .set_index("Patient")
    )

df1 = sortdfs(df_t1, patients_t123)
df2 = sortdfs(df_t2, patients_t123)
df3 = sortdfs(df_t3, patients_t123)

mfa_cols = id_cols = ["Timepoint", "Messdatum"]   
 

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

X1_m = group_name(X1, "T1")
X2_m = group_name(X2, "T2")
X3_m = group_name(X3, "T3")

dataset = pd.concat([X1_m, X2_m, X3_m], axis=1)
groups = dataset.columns.levels[0].tolist()

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

# Scores (coordinates) for each patient at different timepoints-groups
mfa.partial_row_coordinates(dataset)

# Pasient 221 at timepoint 2 is an extreme outlier in MFA plot and
# has extreme values in raw data file?
dataset.loc[221, "T2"]


#%% PCA (prince)

"""
Sette T1 and T2 oppå hverandre - kjør PCA (prince)
sjekk scores og loadings
sette t2+t3, t1 + t3

"""



#%% ############## Exploratory PCA analysis ########################
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# only including numeric features, excluding patient ids, timepoints, dates.
pca_exclude = ["Patient", "Timepoint"]
df_meta = df_im[pca_exclude].copy()

X_pca =(
    df_im_imputed_avg
    .select_dtypes(include="number")
    .drop(columns=pca_exclude)
)

X_pca_clean = X_pca.dropna(axis=0)

# keep matching meta data before scaling
df_meta = df_meta.loc[X_pca_clean.index]

# scaling 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_pca_clean)

# pca with 2 components:
pca = PCA(n_components=2, random_state=42)
pca_raw = pca.fit_transform(X_scaled)

df_pca = pd.DataFrame(
    pca_raw,
    columns=["PC1", "PC2"],
    index=X_pca_clean.index
)

df_pca = pd.concat([df_pca, df_meta], axis=1)


timepoints = sorted(df_pca["Timepoint"].dropna().unique())

for tp in timepoints:
    subset = df_pca[df_pca["Timepoint"] == tp]
    plt.scatter(
        subset["PC1"],
        subset["PC2"],
        label=f"Timepoint {int(tp)}",
        alpha=0.7
    )

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA of immunological data, colored by timepoint 1-5")
plt.legend(title="Timepoint")
plt.axhline(0)
plt.axvline(0)

plt.show()

#%%     Identifying outliers patient ids:

# Distance from pc center in the PCA space
df_pca["pca_distance"] = np.sqrt(df_pca["PC1"]**2 + df_pca["PC2"]**2)

# keepin top 2 % observtions that are the furtherest away from pc center (outliers)
threshold = df_pca["pca_distance"].quantile(0.98)
outliers = df_pca[df_pca["pca_distance"] > threshold]

print("Number of outlier points in the top-2 percent: ", outliers.shape[0])
print("Outliers with patient ids, respective timepoint...")
print(outliers[["Patient", "Timepoint", "PC1", "PC2", "pca_distance"]])

plt.figure(figsize=(7, 6))

# Plot all obervations
plt.scatter(
    df_pca["PC1"],
    df_pca["PC2"],
    alpha=0.2,
    label="All observations"
)

# Plot outliers with red color
plt.scatter(
    outliers["PC1"],
    outliers["PC2"],
    color="red",
    alpha=0.9,
    label="Outliers"
)

for _, row in outliers.iterrows():
    plt.text(
        row["PC1"],
        row["PC2"],
        f'P{int(row["Patient"])}-T{int(row["Timepoint"])}',
        fontsize=8
    )

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA of immunological data with labeled outliers")
plt.legend()
plt.axhline(0)
plt.axvline(0)
plt.show()

# Patient 221 is an extreme outlier at time point 2.
# Pateint 163 is an noticble outlier at timepoint 1.
# Patient 8 is an outlier at timepoint 1 and 2. 


# Checking top 10 PC1 loadings 
loadings_pc1 = pd.Series(
    pca.components_[0],
    index=X_pca_clean.columns
).sort_values(ascending=False)
loadings_pc1.head(10)

# ---> PC1 Loadnings are evenly driven by multiple variables related to immune cell abundances
# Very similar values for the top 10, none of the features dominante PC1.
# PC1 shows overall immune-cell profile. 


# Checking top 10 PC2 Loadings
loadings_pc2 = pd.Series(
    pca.components_[1],
    index=X_pca_clean.columns
).sort_values(ascending=False)
loadings_pc2.head(10)

# ---> PC2 Loadings are dominated by T-Cell activity and inflammation responses
# PC2 seperates samples by degree of T-cell activiation/ immune-regulation balances
# Outliers with higher negative PC2 values might have bigger differences in T-cell activity?




# %% Renaming columns

# Translating and renaming clinical dataset column-naames from german to english

clinical_names = {
    # Patientt demographics
    "Unnamed: 0": "treatment_location",
    "Patient": "patient_id",
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

