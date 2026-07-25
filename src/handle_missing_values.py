import pandas as pd   # I use pandas for working with tabular clinical data (DataFrames).
import numpy as np    # I use NumPy for numerical operations, including detecting infinities.
import os             # I use os for handling file paths and directories.


# PATH CONFIGURATION
# os.path.expanduser() expands '~' to my home directory so the raw MIMIC-IV path is robust on my machine.
RAW_DATA_DIR = os.path.expanduser("~/Desktop/Dissertation/MIMIC-IV Dataset")

# INPUT_PATH points to the latest engineered feature matrix (KDIGO + vitals).
INPUT_PATH = "data/processed/kdigo_labeled_features_v2.parquet"

# DEMOGRAPHICS_PATH points to the separate cohort file containing patient-level demographics.
DEMOGRAPHICS_PATH = "data/extract_patient_cohort.parquet"

# os.path.join() safely constructs the full path to admissions.csv inside the raw data directory.
ADMISSIONS_FILE = os.path.join(RAW_DATA_DIR, "admissions.csv")

# OUTPUT_PATH is where I will write the final, fully imputed feature matrix.
OUTPUT_PATH = "data/processed/final_feature_matrix.parquet"

# MISSING_REPORT_PATH stores a detailed report of missing data for documentation.
MISSING_REPORT_PATH = "data/processed/missing_data_report.csv"


print("=" * 60)
print("MISSING VALUE ANALYSIS & IMPUTATION")
print("=" * 60)


# 1. LOAD DATA

# pd.read_parquet() loads the engineered feature matrix into a pandas DataFrame.
df = pd.read_parquet(INPUT_PATH)
print(f"[INFO] Loaded dataset: {df.shape[0]:,} rows x {df.shape[1]} columns")


# 1a. MERGE DEMOGRAPHICS
# This fixes a previous gap: demographics are needed for fairness analysis later.

print("\n[INFO] Merging patient demographics...")

# pd.read_parquet() again loads the patient cohort with demographic variables (e.g. anchor_age, gender).
demo_df = pd.read_parquet(DEMOGRAPHICS_PATH)


# I now need the admission year to correctly compute age at the time of THIS admission.
# pd.read_csv() loads admissions.csv but only the columns I need via usecols.
admissions_year = pd.read_csv(ADMISSIONS_FILE, usecols=['hadm_id', 'admittime'])

# pd.to_datetime() converts admittime from string to datetime objects.
admissions_year['admittime'] = pd.to_datetime(admissions_year['admittime'])

# .dt.year extracts the year component from the datetime.
admissions_year['admit_year'] = admissions_year['admittime'].dt.year

# I keep only hadm_id and admit_year to avoid carrying unnecessary columns.
admissions_year = admissions_year[['hadm_id', 'admit_year']]


# df.merge(..., on='hadm_id', how='left') joins admission year onto my feature matrix
# based on the hospital admission ID, keeping all rows in df.
df = df.merge(admissions_year, on='hadm_id', how='left')

# I then merge demographics using subject_id to bring in anchor_age and anchor_year.
df = df.merge(demo_df, on='subject_id', how='left')


# anchor_age is the patient's age at a fixed anchor_year (as defined in MIMIC).
# I adjust it so age_at_admission reflects the actual age at the time of this specific admission:
# anchor_age + (admit_year - anchor_year).
df['age_at_admission'] = df['anchor_age'] + (df['admit_year'] - df['anchor_year'])

print(f"[INFO] After demographics merge: {df.shape[0]:,} rows x {df.shape[1]} columns")


#  1b. HANDLE INFINITE VALUES 
# np.isinf() checks for infinite values in numeric columns.
# df.select_dtypes(include=[np.number]) selects only numeric columns.
# .sum().sum() aggregates across all columns and rows to count total infinities.
inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()

if inf_count > 0:
    print(f"[WARNING] Found {inf_count} infinite values — converting to NaN before imputation.")

# df.replace([np.inf, -np.inf], np.nan) replaces positive and negative infinity with NaN
# so they can be handled by the later imputation step instead of corrupting medians.
df = df.replace([np.inf, -np.inf], np.nan)


# 2. IDENTIFY COLUMNS WITH MISSING VALUES

# I define columns to exclude from imputation:
# IDs, target, and raw demographic fields I won’t model directly.
exclude_cols = [
    'subject_id', 'hadm_id', 'icd_code', 'icd_version', 'kdigo_stage',
    'anchor_age', 'anchor_year', 'admit_year', 'dod'
]

# df.select_dtypes(include=[np.number]).columns.tolist() collects all numeric columns.
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# feature_cols keeps only numeric columns that are not in the exclusion list.
feature_cols = [c for c in numeric_cols if c not in exclude_cols]


# df[feature_cols].isna().sum() computes the number of NaNs per feature column.
missing_summary = df[feature_cols].isna().sum()

# I filter to columns with at least one missing value and sort them descending.
missing_summary = missing_summary[missing_summary > 0].sort_values(ascending=False)


print(f"\n[INFO] {len(missing_summary)} columns have missing values out of {len(feature_cols)} feature columns.")
print("\nTop columns by missing count:")
print(missing_summary.head(20))


# I create a detailed missing-value report for documentation.
# pd.DataFrame({...}) constructs a new DataFrame with column name, count, and percentage missing.
missing_report = pd.DataFrame({
    'column': missing_summary.index,
    'missing_count': missing_summary.values,
    'missing_pct': (missing_summary.values / len(df) * 100).round(2)
})

# os.makedirs(os.path.dirname(MISSING_REPORT_PATH), exist_ok=True) ensures the output folder exists.
os.makedirs(os.path.dirname(MISSING_REPORT_PATH), exist_ok=True)

# missing_report.to_csv() writes the report to disk for later inclusion in an appendix.
missing_report.to_csv(MISSING_REPORT_PATH, index=False)
print(f"\n[INFO] Full missing-value report saved to: {MISSING_REPORT_PATH}")


# 3. EXTRACT ROWS WITH ANY MISSING 

# df[feature_cols].isna().any(axis=1) creates a boolean mask for rows with at least one missing feature.
rows_with_any_null = df[df[feature_cols].isna().any(axis=1)]

print(f"\n[INFO] {len(rows_with_any_null):,} admissions ({len(rows_with_any_null)/len(df)*100:.1f}%) "
      f"have at least one missing feature value.")

# I save these rows separately for further inspection using to_parquet().
null_rows_path = "data/processed/rows_with_missing_values.parquet"
rows_with_any_null.to_parquet(null_rows_path, index=False)
print(f"[INFO] Saved these rows separately for inspection to: {null_rows_path}")


# 4. CREATE MISSING-INDICATOR COLUMNS

print("\n[INFO] Creating missing-indicator columns...")

indicator_cols = []

# I loop over each column that has missing values.
for col in missing_summary.index:
    # I construct a new column name indicating missingness, e.g. 'glucose_mean_was_missing'.
    indicator_name = f"{col}_was_missing"
    # df[col].isna() creates a boolean Series; .astype(int) turns True/False into 1/0.
    df[indicator_name] = df[col].isna().astype(int)
    indicator_cols.append(indicator_name)

print(f"[INFO] Created {len(indicator_cols)} indicator columns.")


# 5. IMPUTATION STRATEGY (MEDIAN)

print("\n[INFO] Applying median imputation to remaining missing values...")

imputation_log = []

# I iterate through each column that has missing values, as captured in missing_summary.
for col in missing_summary.index:
    # df[col].median() calculates the median value for that column.
    median_val = df[col].median()
    # df[col].isna().sum() counts how many rows are missing in that column.
    n_missing = df[col].isna().sum()
    # .fillna(median_val) replaces NaNs with the median.
    df[col] = df[col].fillna(median_val)
    # I log the operation for transparency.
    imputation_log.append({'column': col, 'median_used': median_val, 'rows_imputed': n_missing})


# I convert the log list into a DataFrame for saving.
imputation_log_df = pd.DataFrame(imputation_log)

# I define a path for the imputation log and save it with to_csv().
imputation_log_path = "data/processed/imputation_log.csv"
imputation_log_df.to_csv(imputation_log_path, index=False)
print(f"[INFO] Imputation log saved to: {imputation_log_path}")


# 6. VERIFY NO MISSING VALUES REMAIN

# df[feature_cols].isna().sum().sum() sums NaNs across all feature columns
# to confirm that imputation has handled them.
remaining_nulls = df[feature_cols].isna().sum().sum()
print(f"\n[INFO] Remaining nulls in feature columns after imputation: {remaining_nulls}")


# I specifically check age_at_admission, which is not part of feature_cols but is important for analysis.
age_nulls = df['age_at_admission'].isna().sum()
if age_nulls > 0:
    print(f"[WARNING] {age_nulls} rows have missing age_at_admission — check demographics merge.")


# 7. SAVE FINAL FEATURE MATRIX

# df.to_parquet(path, index=False) writes the fully processed and imputed feature matrix to disk.
df.to_parquet(OUTPUT_PATH, index=False)

print(f"\n[SUCCESS] Final feature matrix saved to: {OUTPUT_PATH}")
print(f"Final shape: {df.shape[0]:,} rows x {df.shape[1]} columns "
      f"({len(indicator_cols)} indicator columns added)")
print("=" * 60)