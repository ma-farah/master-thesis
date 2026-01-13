# Loading and exploring raw datasets

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

# Table report of clinical dataset
print("TableReport of raw clinical dataset:")
TableReport(df_cl, max_plot_columns=138)

# A lot of null values dues to empty rows (1658 rows),
# as well as other comments/notes in the excel sheet.
# Need to structure data based on timepoints 1,2,3,4...
# Patients with missing treatment/ response information - remove
# Patients with "Ausschluss/ Exclude" - remove
# Patients with less than x? measure-timepoints - remove
# Combine Survey questions/columns ?



#%%################  RAW IMMUNOLOGICAL DATASET ###########################


print("TableReport of raw immunological dataset:")
TableReport(df_im, max_plot_columns=138)

# 46 columns to exclude from further analysis, as mention in dataset.
# Around 6-7 outliers and missing values for almost each variable, 
# also 6 missing values for patient IDs, maybe it is the same patients? 
# Not all patients have been measured at all timepoints 1-5. Which ones are that?


rows_with_na = df_im[df_im.isna().any(axis=1)]
print("patient id-s with missing values:")
rows_with_na["Patient"].value_counts()

# Patient Ids : 30, 223, 224, 226, 227, 228, 229 and 230 have missing values in one row/timepoint each.

# Patients that has only one timepoints measured:
patient_counts = df_im["Patient"].value_counts()
single_timepoint_patients = patient_counts[patient_counts == 1].index
single_timepoint_rows = df_im[df_im["Patient"].isin(single_timepoint_patients)]
print("Amount of patients with only one measured timepoint:", len(single_timepoint_patients))
print(single_timepoint_rows[["Patient", "Timepoint"]])



#%%#############  CLEANING DATASET ####################################

# Removing columns that can be exlcuded (marked yellow in dataset): 43 columns
dropped_columns = [
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

# transform feature types to correct types

df_im = df_im.drop(columns=dropped_columns)
print("TableReport of im. dataset, after removing columns:")
TableReport(df_im, max_plot_columns=138)


#%% ############## Exploratory PCA analysis ########################
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# only including numeric features, excluding patient ids, timepoints, dates.
pca_exclude = ["Patient", "Timepoint"]
df_meta = df_im[pca_exclude].copy()

X_pca =(
    df_im
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
plt.title("PCA of raw immunological data, colored by timepoint 1-5")
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

