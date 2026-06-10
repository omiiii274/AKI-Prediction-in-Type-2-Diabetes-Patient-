import os  # Provides utilities for working with file paths and directories
import pandas as pd  # Used for reading, cleaning, and analysing tabular data
import numpy as np  # Supports numerical calculations and missing-value handling

RAW_DATA_DIRECTORY = "/Users/apple/Desktop/Dissertation/MIMIC-IV Dataset"  # Directory containing the raw MIMIC-IV CSV files
COHORT_PATH = "data/diabetes_cohort_list.parquet"  # cohort file containing the target admissions
OUTPUT_DIRECTORY = "data/processed"  # Folder where the final engineered dataset will be stored
OUTPUT_FILE = os.path.join(OUTPUT_DIRECTORY, "kdigo_labeled_features.parquet")  # Full path to the final parquet output

os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)  # Create the output folder if it does not already exist

print("=" * 60)  # Print a separator for cleaner console output
print("STARTING ADVANCED RENAL & GLYCEMIC FEATURE ENGINEERING")  # Indicate that the pipeline has started
print("=" * 60)  # Print another separator

cohort = pd.read_parquet(COHORT_PATH)  # Load the cohort into a DataFrame
target_hadm_ids = set(cohort['hadm_id'].unique())  # Store unique hospital admission IDs for fast filtering
print(f"[INFO] Target Cohort Admissions: {len(target_hadm_ids):,}")  # Display the number of admissions in the cohort

labevents_path = os.path.join(RAW_DATA_DIRECTORY, "labevents.csv")  # Path to the large lab events file
chunks = []  # List used to collect filtered chunks from the CSV file

print("[INFO] Processing stream from labevents...")  # Show progress while reading the large file
for chunk in pd.read_csv(labevents_path, chunksize=250000, low_memory=False):  # Read the file in manageable blocks
    filtered_chunk = chunk[chunk['hadm_id'].isin(target_hadm_ids)]  # Keep only rows for admissions in the cohort
    filtered_chunk = filtered_chunk[filtered_chunk['itemid'].isin([50912, 50927, 50809, 51520])]  # Keep creatinine and glucose labs only
    chunks.append(filtered_chunk)  # Save the filtered rows for later combination

raw_labs = pd.concat(chunks, axis=0, ignore_index=True)  # Combine all filtered chunks into one table
raw_labs['charttime'] = pd.to_datetime(raw_labs['charttime'])  # Convert lab timestamps into datetime format
print(f"[INFO] Extracted {len(raw_labs):,} clean clinical entries.")  # Report the number of retained observations

raw_labs = raw_labs.sort_values(by=['hadm_id', 'charttime'])  # Sort observations by admission and time
raw_labs['admission_start'] = raw_labs.groupby('hadm_id')['charttime'].transform('min')  # Find each admission's first observation time
raw_labs['hours_since_admit'] = (raw_labs['charttime'] - raw_labs['admission_start']).dt.total_seconds() / 3600.0  # Calculate time since admission in hours

print("[INFO] Engineering predictor matrices (0-24h observation window)...")  # Indicate the start of feature construction
obs_mask = raw_labs['hours_since_admit'] <= 24.0  # Keep only data from the first 24 hours
obs_data = raw_labs[obs_mask]  # Create the 24-hour feature window

glucose_mask = obs_data['itemid'].isin([50927, 50809, 51520])  # Identify glucose measurements
creat_mask = obs_data['itemid'] == 50912  # Identify creatinine measurements

glc_groups = obs_data[glucose_mask].groupby('hadm_id')['valuenum']  # Group glucose values by admission
glc_mean = glc_groups.mean().rename('glucose_mean')  # Compute average glucose per admission
glc_std = glc_groups.std().fillna(0)  # Compute glucose variability and replace missing values with zero
glc_cv = ((glc_std / glc_mean) * 100).rename('glucose_cv')  # Convert variability into coefficient of variation

glc_tir = obs_data[glucose_mask].groupby('hadm_id').apply(  # Calculate the proportion of glucose values in range
    lambda x: (x['valuenum'].between(70, 180).sum() / len(x) * 100) if len(x) > 0 else np.nan,  # Safe-range percentage
    include_groups=False  # Prevent group labels from being included in the output
).rename('glucose_time_in_range')  # Name the output series

def calculate_slope(group):  # Define a helper function for creatinine trend calculation
    if len(group) < 2:  # Require at least two values to compute a change
        return np.nan  # Return missing if there are not enough observations
    return group['valuenum'].iloc[-1] - group['valuenum'].iloc[0]  # Calculate change from first to last value

creat_slope = obs_data[creat_mask].groupby('hadm_id').apply(calculate_slope, include_groups=False).rename('creatinine_slope_24h')  # Compute 24-hour creatinine change
baseline_creat = obs_data[creat_mask].groupby('hadm_id')['valuenum'].min().rename('baseline_creatinine')  # Use the minimum creatinine in the first 24h as baseline

features_df = pd.concat([glc_mean, glc_cv, glc_tir, creat_slope, baseline_creat], axis=1).reset_index()  # Combine all engineered features into one table

print("[INFO] Computing target parameters (24-72h prediction horizon)...")  # Indicate target creation stage
pred_mask = (raw_labs['hours_since_admit'] > 24.0) & (raw_labs['hours_since_admit'] <= 72.0)  # Keep future observations from 24 to 72 hours
pred_data = raw_labs[pred_mask & (raw_labs['itemid'] == 50912)]  # Keep only creatinine values in the prediction window

pred_data = pred_data.merge(features_df[['hadm_id', 'baseline_creatinine']], on='hadm_id', how='inner')  # Attach baseline creatinine to future creatinine records
pred_data['creat_ratio'] = pred_data['valuenum'] / pred_data['baseline_creatinine']  # Compute creatinine ratio relative to baseline
pred_data['creat_diff'] = pred_data['valuenum'] - pred_data['baseline_creatinine']  # Compute absolute creatinine increase

def assign_kdigo(group):  # Define the KDIGO staging function
    max_ratio = group['creat_ratio'].max()  # Find the largest creatinine ratio in the follow-up window
    max_diff = group['creat_diff'].max()  # Find the largest absolute increase
    max_val = group['valuenum'].max()  # Find the highest absolute creatinine value
    
    if max_ratio >= 3.0 or max_val >= 4.0:  # KDIGO stage 3: severe AKI
        return 3
    elif 2.0 <= max_ratio < 3.0:  # KDIGO stage 2: moderate AKI
        return 2
    elif max_ratio >= 1.5 or max_diff >= 0.3:  # KDIGO stage 1: mild AKI
        return 1
    return 0  # No AKI

labels_series = pred_data.groupby('hadm_id').apply(assign_kdigo, include_groups=False).rename('kdigo_stage')  # Assign an AKI stage to each admission
labels_df = labels_series.reset_index()  # Convert the result into a DataFrame for merging

print("[INFO] Merging clean datasets...")  # Indicate that final dataset assembly is starting
final_matrix = cohort.merge(features_df, on='hadm_id', how='inner')  # Merge cohort data with engineered predictors
final_matrix = final_matrix.merge(labels_df, on='hadm_id', how='inner')  # Merge cohort data with KDIGO labels

fill_cols = ['glucose_mean', 'glucose_cv', 'glucose_time_in_range', 'creatinine_slope_24h']  # Columns where missing values will be imputed
for col in fill_cols:  # Loop through each selected feature column
    final_matrix[col] = final_matrix[col].fillna(final_matrix[col].median())  # Replace missing values with the median of that column

final_matrix.to_parquet(OUTPUT_FILE, index=False)  # Save the final dataset in an efficient parquet format

print("=" * 60)  # Print a final separator
print(f"SUCCESS! Consolidated matrix written to: {OUTPUT_FILE}")  # Confirm the output path
print(f"Dimensions: {final_matrix.shape[0]:,} records x {final_matrix.shape[1]} metrics")  # Show dataset dimensions
print(f"Target distribution (KDIGO Stages):\n{final_matrix['kdigo_stage'].value_counts().sort_index()}")  # Show class balance
print("=" * 60)  # Print a closing separator