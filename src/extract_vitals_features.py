# I import the core libraries required for this script.
# import os: lets me handle file paths and directories (e.g. os.path.join, os.makedirs).
# import pandas as pd: gives me pandas for working with large tabular clinical datasets.
# import numpy as np: gives me NumPy for numerical operations such as regression and slope calculation.
import os
import pandas as pd
import numpy as np


# A simple print() call to show that the vitals extraction process is starting.
print("Starting Vitals Extraction from CSV Files")


# 1. PATH CONFIGURATIONS 

# os.path.expanduser() expands '~' to my home directory so the path is robust on my machine.
RAW_DATA_DIR = os.path.expanduser("~/Desktop/Dissertation/MIMIC-IV Dataset")

# os.path.join() safely builds the full path to chartevents.csv.
CHARTEVENTS_FILE = os.path.join(RAW_DATA_DIR, "chartevents.csv")

# Paths to the input (V1) and output (V2) feature matrices in Parquet format.
V1_DATA_PATH = "data/processed/kdigo_labeled_features.parquet"
V2_DATA_PATH = "data/processed/kdigo_labeled_features_v2.parquet"


# 2. LOAD COHORT WITH ICU ANCHOR 

# os.path.exists() checks if the V1 parquet file actually exists before proceeding.
# If not, I raise a FileNotFoundError with a helpful message.
if not os.path.exists(V1_DATA_PATH):
    raise FileNotFoundError(f"Could not find V1 parquet file at {V1_DATA_PATH}.")


# pd.read_parquet() loads the existing KDIGO feature matrix into a DataFrame v1_df.
v1_df = pd.read_parquet(V1_DATA_PATH)

# v1_df['hadm_id'].unique() returns all unique admission IDs; set(...) converts this to a Python set
# for fast membership checks when filtering chartevents.
cohort_hadm_ids = set(v1_df['hadm_id'].unique())
print(f"Loaded V1 dataset. Target cohort size: {len(cohort_hadm_ids)} admissions.")


# I ensure that icu_intime is present; this column was added in the KDIGO feature script.
# If it is missing, I raise a ValueError asking the user to re-run the earlier script.
if 'icu_intime' not in v1_df.columns:
    raise ValueError("icu_intime column not found in V1 dataset. Re-run extract_kdigo_features.py first.")


# I build a lookup table for ICU admission times:
# v1_df[['hadm_id', 'icu_intime']] selects just the columns I need.
# .dropna(subset=['icu_intime']) removes rows where icu_intime is missing.
# .drop_duplicates(subset='hadm_id') ensures only one icu_intime per admission.
icu_intime_lookup = v1_df[['hadm_id', 'icu_intime']].dropna(subset=['icu_intime']).drop_duplicates(subset='hadm_id')

# pd.to_datetime() converts icu_intime from strings to datetime objects to allow time arithmetic.
icu_intime_lookup['icu_intime'] = pd.to_datetime(icu_intime_lookup['icu_intime'])
print(f"[INFO] Using icu_intime anchor for {len(icu_intime_lookup):,} admissions.")


# 3. STREAM & FILTER CHARTEVENTS 
# I define a set of item IDs that correspond to the vital signs I care about.
# Using a set makes lookups via .isin() efficient.
vitals_itemids = {
    220045,          # Heart Rate
    220179, 220050,  # Systolic Blood Pressure
    220180, 220051,  # Diastolic Blood Pressure
    220052, 220181,  # Mean Arterial Pressure
    220210,          # Respiratory Rate
    223761, 223762,  # Temperature (F, C)
    220277           # SpO2
}

# A mapping dictionary that translates raw itemids into human-readable vital names.
# I will use .map() later to create a vital_name column.
id_mapping = {
    220045: 'hr',
    220179: 'sbp',
    220050: 'sbp',
    220180: 'dbp',
    220051: 'dbp',
    220052: 'map',
    220181: 'map',
    220210: 'rr',
    223761: 'temp',
    223762: 'temp',
    220277: 'spo2'
}

# I define plausible physiological ranges for each vital sign as (low, high) tuples.
# These will be used to remove outlier or implausible values.
PLAUSIBLE_RANGES = {
    'hr':   (20, 300),
    'sbp':  (40, 300),
    'dbp':  (20, 200),
    'map':  (20, 250),
    'rr':   (4, 60),
    'temp': (30, 44),
    'spo2': (50, 100)
}


print("Streaming chartevents.csv and applying strict 24h ICU window filter...")
vitals_chunks = []  # This list will store filtered chunks of chartevents.


# pd.read_csv(..., chunksize=100000) reads chartevents.csv in chunks of 100,000 rows.
# usecols specifies only the columns I need, reducing memory usage.
for chunk in pd.read_csv(
    CHARTEVENTS_FILE,
    usecols=['hadm_id', 'charttime', 'itemid', 'valuenum'],
    chunksize=100000
):
    # First filter: keep only rows where hadm_id is in my cohort set using .isin().
    chunk_filtered = chunk[chunk['hadm_id'].isin(cohort_hadm_ids)]

    # Second filter: keep only itemids that correspond to my chosen vitals.
    chunk_filtered = chunk_filtered[chunk_filtered['itemid'].isin(vitals_itemids)]

    # If the filtered chunk is not empty, I append it to vitals_chunks.
    if not chunk_filtered.empty:
        vitals_chunks.append(chunk_filtered)


# If no chunks had matching vitals, I raise an error to warn the user.
if not vitals_chunks:
    raise ValueError("No matching vital signs found in chartevents.csv for your cohort.")


# pd.concat(vitals_chunks, ignore_index=True) combines all filtered chunks into a single DataFrame.
raw_vitals = pd.concat(vitals_chunks, ignore_index=True)
print(f"Extracted {len(raw_vitals)} raw vital recordings.")


# ANCHOR VITALS TO ICU INTIME 

# DataFrame.merge() joins raw_vitals with icu_intime_lookup on hadm_id.
# how='inner' keeps only admissions present in both DataFrames.
raw_vitals = raw_vitals.merge(icu_intime_lookup, on='hadm_id', how='inner')

# pd.to_datetime() converts charttime to datetime, enabling time comparisons.
raw_vitals['charttime'] = pd.to_datetime(raw_vitals['charttime'])


# I restrict vitals to the first 24 hours after ICU admission.
# The boolean mask checks whether charttime lies between icu_intime and icu_intime + 24 hours
# using pd.Timedelta(hours=24).
raw_vitals = raw_vitals[
    (raw_vitals['charttime'] >= raw_vitals['icu_intime']) &
    (raw_vitals['charttime'] <= raw_vitals['icu_intime'] + pd.Timedelta(hours=24))
]


# .dropna(subset=['valuenum']) removes any rows where the numeric vital value is missing.
raw_vitals = raw_vitals.dropna(subset=['valuenum'])
print(f"Post-Hygiene Filter (ICU time window): {len(raw_vitals)} records remain.")


# 4. STANDARDIZATION & FEATURE ENGINEERING

# .map(id_mapping) converts itemid values into human-readable vital_name labels (hr, sbp, etc.).
raw_vitals['vital_name'] = raw_vitals['itemid'].map(id_mapping)


# For temperature recorded in Fahrenheit (itemid 223761), I convert the values to Celsius:
# (F - 32) * 5/9 using vectorised operations with .loc[].
raw_vitals.loc[raw_vitals['itemid'] == 223761, 'valuenum'] = (
    (raw_vitals.loc[raw_vitals['itemid'] == 223761, 'valuenum'] - 32) * 5 / 9
)


# I now apply physiological range filtering.
before_count = len(raw_vitals)

# I start with a mask of all True values for each row using pd.Series(True, index=raw_vitals.index).
mask = pd.Series(True, index=raw_vitals.index)

# I loop through each vital type and its plausible range.
for vital, (low, high) in PLAUSIBLE_RANGES.items():
    # vital_rows identifies rows corresponding to the current vital_name.
    vital_rows = raw_vitals['vital_name'] == vital
    # .between(low, high) checks if values are within the plausible range.
    # ~ operator negates the result to find out-of-range values.
    out_of_range = vital_rows & ~raw_vitals['valuenum'].between(low, high)
    # I update the mask to exclude out-of-range readings (logical AND with the negation).
    mask &= ~out_of_range

# I apply the combined mask to keep only plausible vital readings.
raw_vitals = raw_vitals[mask]
print(f"Post-Hygiene Filter (physiological range): {len(raw_vitals)} records remain "
      f"({before_count - len(raw_vitals)} outlier readings removed).")



# I define a helper function to compute the slope (rate of change) for a vital over time.
# It will be used with groupby().apply().
def compute_slope(group):
    # If fewer than two measurements exist, I return 0.0 as slope (no trend can be estimated).
    if len(group) < 2:
        return 0.0
    # I compute time in hours since the first measurement:
    # (charttime - first charttime), then .dt.total_seconds()/3600.
    x = (group['charttime'] - group['charttime'].iloc[0]).dt.total_seconds() / 3600.0
    # y is the array of vital values.
    y = group['valuenum'].values
    # If all measurements occur at the same time (no time spread), slope is 0.0.
    if x.max() == 0:
        return 0.0
    # np.polyfit(x, y, 1) fits a line (degree 1) to the data and returns coefficients.
    # [0] is the slope of that line.
    slope = np.polyfit(x, y, 1)[0]
    return slope



print("Computing Summary Statistics & Slope (Rate of Change)...")


# I sort the data by hadm_id, vital_name, and charttime so that statistics and slopes are computed
# in a consistent temporal order.
raw_vitals = raw_vitals.sort_values(['hadm_id', 'vital_name', 'charttime'])


# 4. GROUPED SUMMARY STATISTICS 
# raw_vitals.groupby(['hadm_id', 'vital_name'])['valuenum'].agg(...) groups by admission and vital type,
# then uses .agg() to compute multiple summary statistics at once:
# mean, std, min, max, count.
agg_stats = raw_vitals.groupby(['hadm_id', 'vital_name'])['valuenum'].agg(
    mean='mean', std='std', min='min', max='max', count='count'
).reset_index()


# I compute slopes per (hadm_id, vital_name) using groupby().apply(compute_slope).
# include_groups=False ensures the function receives only the group data, not the group labels.
slope_series = raw_vitals.groupby(['hadm_id', 'vital_name']).apply(
    compute_slope, include_groups=False
).rename('slope').reset_index()


# DataFrame.merge() joins the summary statistics with the slopes
# based on hadm_id and vital_name.
agg_stats = agg_stats.merge(slope_series, on=['hadm_id', 'vital_name'])


# PIVOT TO WIDE FORMAT 
# .pivot(index='hadm_id', columns='vital_name') transforms the long table (one row per vital)
# into a wide table with one row per admission and separate columns per vital statistic.
vitals_wide = agg_stats.pivot(index='hadm_id', columns='vital_name')

# The pivoted DataFrame has a MultiIndex on columns (stat, vital_name).
# I flatten it into single strings like 'hr_mean', 'sbp_max' using a list comprehension.
vitals_wide.columns = [f'{vital}_{stat}' for stat, vital in vitals_wide.columns]

# .reset_index() brings hadm_id back as a regular column.
vitals_wide = vitals_wide.reset_index()


# I merge the vitals summary features into the existing V1 feature matrix using pd.merge().
# how='left' keeps all admissions from v1_df and adds vitals where available.
final_v2_df = pd.merge(v1_df, vitals_wide, on='hadm_id', how='left')


# 5. DATA QUALITY SUMMARY

print("\n" + "=" * 60)
print("DATA QUALITY SUMMARY")
print("=" * 60)

# I loop through columns in final_v2_df and focus on those ending with '_mean',
# which represent key vitals summary features.
for col in final_v2_df.columns:
    if col.endswith('_mean'):
        # .isna().sum() counts missing values.
        missing = final_v2_df[col].isna().sum()
        total = len(final_v2_df)
        # I print the number and percentage of non-null values for each vital mean.
        print(f"{col}: {total - missing:,}/{total:,} non-null ({(total-missing)/total*100:.1f}%)")
print("=" * 60)


# 5. SAVE VERSIONED PARQUET 
# os.path.dirname(V2_DATA_PATH) extracts the directory part of the path.
# os.makedirs(..., exist_ok=True) ensures this directory exists.
os.makedirs(os.path.dirname(V2_DATA_PATH), exist_ok=True)

# final_v2_df.to_parquet(path, index=False) saves the updated feature matrix (including vitals)
# as a Parquet file for future modelling.
final_v2_df.to_parquet(V2_DATA_PATH, index=False)


# Final confirmation message.
print(f"Step complete! Versioned matrix saved successfully as {V2_DATA_PATH}")