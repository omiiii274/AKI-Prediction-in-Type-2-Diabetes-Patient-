# I import the core libraries I need.
# import os: lets me work with file paths and directories (e.g. os.path.join, os.makedirs).
# import pandas as pd: gives me pandas for handling tabular clinical data (DataFrames).
# import numpy as np: gives me NumPy for numerical operations such as NaN handling and ratios.
import os
import pandas as pd
import numpy as np


# PATH CONFIGURATION 
# RAW_DATA_DIRECTORY is a simple string that points to my local MIMIC-IV dataset folder.
RAW_DATA_DIRECTORY = "/Users/apple/Desktop/Dissertation/MIMIC-IV Dataset"

# COHORT_PATH points to the previously built diabetes ICU cohort (Parquet file).
COHORT_PATH = "data/diabetes_cohort_list.parquet"

# OUTPUT_DIRECTORY defines where I will store the processed feature matrix.
OUTPUT_DIRECTORY = "data/processed"

# os.path.join() safely combines the directory and filename into one path string.
OUTPUT_FILE = os.path.join(OUTPUT_DIRECTORY, "kdigo_labeled_features.parquet")


# os.makedirs(path, exist_ok=True) creates the output directory if it doesn’t already exist.
# The exist_ok=True parameter ensures that no error is raised if the folder is already present.
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)


# SCRIPT HEADER

# I use print() here to display a clear header when the script runs,
# making the console output easier to follow.
print("=" * 60)
print("STARTING ADVANCED RENAL & GLYCEMIC FEATURE ENGINEERING")
print("=" * 60)


# LOAD DIABETES ICU COHORT 

# pd.read_parquet() loads the cohort file into a pandas DataFrame called 'cohort'.
cohort = pd.read_parquet(COHORT_PATH)

# cohort['hadm_id'].unique() gets all unique hospital admission IDs in the cohort.
# set(...) converts that array to a Python set, which is efficient for membership checks later.
target_hadm_ids = set(cohort['hadm_id'].unique())
print(f"[INFO] Target Cohort Admissions: {len(target_hadm_ids):,}")


# LOAD ICU STAYS (ANCHOR TIMES) 

# os.path.join() again builds the full path to icustays.csv.
icustays_path = os.path.join(RAW_DATA_DIRECTORY, "icustays.csv")

print("[INFO] Loading icustays.csv for ICU-anchored time-window filtering...")

# pd.read_csv() reads the ICU stays file into a DataFrame.
# usecols=['hadm_id', 'intime'] means I only load the admission ID and ICU entry time,
# which keeps memory usage lower.
icustays_df = pd.read_csv(icustays_path, usecols=['hadm_id', 'intime'])

# pd.to_datetime() converts the 'intime' column from text into proper datetime objects,
# allowing me to do time arithmetic later.
icustays_df['intime'] = pd.to_datetime(icustays_df['intime'])


# If an admission has multiple ICU stays, I want to use the earliest ICU entry as the anchor.
# .sort_values('intime') sorts the DataFrame by ICU entry time.
# .drop_duplicates(subset='hadm_id', keep='first') keeps only the first ICU stay per admission.
icustays_df = icustays_df.sort_values('intime').drop_duplicates(subset='hadm_id', keep='first')

# I then filter to just admissions that belong to my T2DM cohort using .isin().
icustays_df = icustays_df[icustays_df['hadm_id'].isin(target_hadm_ids)]

# .rename(columns={'intime': 'icu_intime'}) renames the column to reflect its role as ICU admission time.
icustays_df = icustays_df.rename(columns={'intime': 'icu_intime'})


# LOAD LABEVENTS (CREATININE & GLUCOSE) 

# I build the path to labevents.csv in the same way.
labevents_path = os.path.join(RAW_DATA_DIRECTORY, "labevents.csv")

# 'chunks' will store portions of labevents so I can process the file in manageable pieces.
chunks = []

print("[INFO] Processing stream from labevents...")

# pd.read_csv(..., chunksize=250000, low_memory=False) reads the labevents file in chunks of 250,000 rows.
# This streaming approach avoids loading the entire file into memory at once.
for chunk in pd.read_csv(labevents_path, chunksize=250000, low_memory=False):

    # First, I filter the chunk to only admissions in my target cohort.
    # .isin(target_hadm_ids) creates a boolean mask, and I use it to subset the DataFrame.
    filtered_chunk = chunk[chunk['hadm_id'].isin(target_hadm_ids)]

    # I then filter to only the lab tests I care about:
    # itemid 50912 = creatinine, 50931 and 50809 = glucose-related measurements.
    filtered_chunk = filtered_chunk[filtered_chunk['itemid'].isin([50912, 50931, 50809])]

    # I append this filtered_chunk to the chunks list for later concatenation.
    chunks.append(filtered_chunk)


# pd.concat(chunks, axis=0, ignore_index=True) combines all the individual chunks
# into one large DataFrame called raw_labs.
raw_labs = pd.concat(chunks, axis=0, ignore_index=True)

# pd.to_datetime() converts charttime into datetime objects.
raw_labs['charttime'] = pd.to_datetime(raw_labs['charttime'])

print(f"[INFO] Extracted {len(raw_labs):,} clean clinical entries.")


# .sort_values(by=['hadm_id', 'charttime']) sorts all lab events by admission then time,
# which ensures temporal calculations are consistent.
raw_labs = raw_labs.sort_values(by=['hadm_id', 'charttime'])


# ANCHOR LABS TO ICU INTIME

# DataFrame.merge(other, on='hadm_id', how='inner') joins raw_labs and icustays_df
# using hadm_id as the key, keeping only records present in both (inner join).
raw_labs = raw_labs.merge(icustays_df, on='hadm_id', how='inner')

# I compute hours since ICU admission using datetime arithmetic:
# (charttime - icu_intime) gives a Timedelta, .dt.total_seconds() converts it to seconds,
# and dividing by 3600.0 turns it into hours.
raw_labs['hours_since_icu_admit'] = (raw_labs['charttime'] - raw_labs['icu_intime']).dt.total_seconds() / 3600.0


# I remove any lab measurements that occurred before ICU admission.
# This ensures I only work with ICU-anchored data.
raw_labs = raw_labs[raw_labs['hours_since_icu_admit'] >= 0]


print("[INFO] Engineering predictor matrices (0-24h post-ICU-admission window)...")

# I create a boolean mask selecting all lab values within the first 24 hours of ICU stay.
obs_mask = raw_labs['hours_since_icu_admit'] <= 24.0

# obs_data is the subset of raw_labs restricted to this 0–24 hour observation window.
obs_data = raw_labs[obs_mask]


# GLYCEMIC FEATURES (GLUCOSE) 

# glucose_mask selects rows where itemid corresponds to glucose measurements (50931 or 50809).
glucose_mask = obs_data['itemid'].isin([50931, 50809])

# creat_mask selects rows where itemid is 50912 (creatinine).
creat_mask = obs_data['itemid'] == 50912


# obs_data[glucose_mask] filters to glucose entries in the observation window.
# .groupby('hadm_id')['valuenum'] groups glucose values by admission.
glc_groups = obs_data[glucose_mask].groupby('hadm_id')['valuenum']

# .mean(), .count(), and .std() are pandas GroupBy methods:
# - mean(): average glucose,
# - count(): number of readings,
# - std(): standard deviation of glucose values.
glc_mean = glc_groups.mean().rename('glucose_mean')
glc_count = glc_groups.count().rename('glucose_reading_count')
glc_std = glc_groups.std().rename('glucose_std')

# I compute coefficient of variation (CV) as (std / mean) * 100, and rename the Series.
glc_cv = ((glc_std / glc_mean) * 100).rename('glucose_cv')


# groupby('hadm_id').apply(...) lets me apply a custom function per admission.
# I use a lambda to calculate "time in range" (TIR): percentage of readings between 70 and 180.
# .between(70, 180) returns a boolean Series, .sum() counts in-range readings,
# len(x) is the total number of glucose readings.
glc_tir = obs_data[glucose_mask].groupby('hadm_id').apply(
    lambda x: (x['valuenum'].between(70, 180).sum() / len(x) * 100) if len(x) > 0 else np.nan,
    include_groups=False
).rename('glucose_time_in_range')



# CREATININE FEATURES 

# I define a helper function to calculate a simple slope for creatinine over 24h.
# It takes a group (DataFrame for one hadm_id), and returns:
# last value - first value. If there are fewer than 2 readings, slope is undefined (np.nan).
def calculate_slope(group):
    if len(group) < 2:
        return np.nan
    return group['valuenum'].iloc[-1] - group['valuenum'].iloc[0]


# For creatinine in the observation window, I again use groupby('hadm_id').apply().
# apply(calculate_slope) passes each group to my custom slope function.
creat_slope = obs_data[creat_mask].groupby('hadm_id').apply(
    calculate_slope, include_groups=False
).rename('creatinine_slope_24h')

# .count() here returns how many creatinine readings each admission has.
creat_count = obs_data[creat_mask].groupby('hadm_id')['valuenum'].count().rename('creatinine_reading_count')

# .min() is used to approximate a baseline creatinine in the first 24 hours.
baseline_creat = obs_data[creat_mask].groupby('hadm_id')['valuenum'].min().rename('baseline_creatinine')


# BUILD FEATURE MATRIX

# pd.concat([...], axis=1) horizontally concatenates the different feature Series
# (glucose and creatinine features) into a single DataFrame.
# .reset_index() turns hadm_id back into a regular column.
features_df = pd.concat(
    [glc_mean, glc_std, glc_cv, glc_count, glc_tir, creat_slope, creat_count, baseline_creat],
    axis=1
).reset_index()


print("[INFO] Computing target parameters (24-72h post-ICU-admission prediction horizon)...")

# I now define the prediction window as 24–72 hours after ICU admission.
# The pred_mask boolean Series selects creatinine labs in that time range.
pred_mask = (raw_labs['hours_since_icu_admit'] > 24.0) & (raw_labs['hours_since_icu_admit'] <= 72.0)

# I subset raw_labs using pred_mask and creatinine itemid (50912) to get relevant target data.
pred_data = raw_labs[pred_mask & (raw_labs['itemid'] == 50912)]


# I merge baseline creatinine from features_df into pred_data using DataFrame.merge(),
# so I can compute KDIGO criteria relative to each patient’s baseline.
pred_data = pred_data.merge(features_df[['hadm_id', 'baseline_creatinine']], on='hadm_id', how='inner')

# creat_ratio: current creatinine / baseline creatinine.
pred_data['creat_ratio'] = pred_data['valuenum'] / pred_data['baseline_creatinine']

# creat_diff: absolute change from baseline.
pred_data['creat_diff'] = pred_data['valuenum'] - pred_data['baseline_creatinine']



# KDIGO LABEL ASSIGNMENT

# I define a function that assigns KDIGO AKI stage based on all creatinine measurements
# for a given admission in the 24–72h window.
def assign_kdigo(group):
    # .max() returns the maximum observed ratio/difference/value within the group.
    max_ratio = group['creat_ratio'].max()
    max_diff = group['creat_diff'].max()
    max_val = group['valuenum'].max()

    # KDIGO criteria:
    # Stage 3 if ratio >= 3.0 or absolute creatinine >= 4.0 mg/dL.
    if max_ratio >= 3.0 or max_val >= 4.0:
        return 3
    # Stage 2 if ratio between 2.0 and 3.0.
    elif 2.0 <= max_ratio < 3.0:
        return 2
    # Stage 1 if ratio >= 1.5 or absolute increase >= 0.3 mg/dL.
    elif max_ratio >= 1.5 or max_diff >= 0.3:
        return 1
    # Otherwise, no AKI (Stage 0).
    return 0


# pred_data.groupby('hadm_id').apply(assign_kdigo) applies my KDIGO function per admission.
# It returns a Series of KDIGO stages indexed by hadm_id.
labels_series = pred_data.groupby('hadm_id').apply(assign_kdigo, include_groups=False).rename('kdigo_stage')

# .reset_index() converts the index (hadm_id) back into a normal column,
# giving me a two-column DataFrame: hadm_id and kdigo_stage.
labels_df = labels_series.reset_index()


print("[INFO] Merging clean datasets")

# BUILD FINAL MATRIX 

# cohort.merge(features_df, on='hadm_id', how='inner') joins cohort and engineered features
# by admission ID, keeping only admissions present in both.
final_matrix = cohort.merge(features_df, on='hadm_id', how='inner')

# I then merge KDIGO labels to get the final target variable.
final_matrix = final_matrix.merge(labels_df, on='hadm_id', how='inner')


# I keep icu_intime in the final output so that other scripts (e.g., vitals feature engineering)
# can reuse the exact same time anchor.
icu_intime_lookup = icustays_df[['hadm_id', 'icu_intime']]
final_matrix = final_matrix.merge(icu_intime_lookup, on='hadm_id', how='left')


# final_matrix.to_parquet(path, index=False) writes the consolidated feature + label matrix
# to disk in Parquet format for efficient storage and later modelling.
final_matrix.to_parquet(OUTPUT_FILE, index=False)


print("=" * 60)
print(f"SUCCESS! Consolidated matrix written to: {OUTPUT_FILE}")

# final_matrix.shape gives (rows, columns).
print(f"Dimensions: {final_matrix.shape[0]:,} records x {final_matrix.shape[1]} metrics")

# final_matrix['kdigo_stage'].value_counts() counts the number of admissions in each KDIGO stage.
# .sort_index() sorts stages by their numeric order (0, 1, 2, 3).
print(f"Target distribution (KDIGO Stages):\n{final_matrix['kdigo_stage'].value_counts().sort_index()}")
print("=" * 60)