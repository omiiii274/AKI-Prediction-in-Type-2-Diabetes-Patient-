import pandas as pd
import os

# We define clear paths so your supervisor can see exactly where data is read and written.
RAW_DATA_DIR = os.path.expanduser("~/Desktop/Dissertation/MIMIC-IV Dataset")
DIAGNOSES_FILE = os.path.join(RAW_DATA_DIR, "diagnoses_icd.csv")

OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "diabetes_cohort_list.parquet")

def extract_diabetic_cohort():
    print("STARTING DIAGNOSES EXTRACTION SYSTEM FOR T2DM COHORT IDENTIFICATION")
    
    # Check if the source file is present
    if not os.path.exists(DIAGNOSES_FILE):
        print(f"Error: Cannot find the diagnosis file at: {DIAGNOSES_FILE}")
        return

    # Using Python sets for high-speed, unique ID tracking

    t2dm_patients = set()
    esrd_patients = set()
    
    # Memory management: process the massive file in chunks of 1,000,000 rows
    chunk_size = 1000000
    print(f"Reading {DIAGNOSES_FILE} sequentially in chunks of {chunk_size:,} rows...")

    for chunk_num, chunk in enumerate(pd.read_csv(DIAGNOSES_FILE, usecols=["subject_id", "icd_code"], chunksize=chunk_size), start=1):
        # Clean the string data to avoid leading/trailing whitespace mismatches
        chunk["icd_code"] = chunk["icd_code"].astype(str).str.strip()
        
        # 1. Identify Type 2 Diabetes Mellitus (T2DM)
        # In ICD-10, E11.x represents Type 2 Diabetes
        t2dm_mask = chunk["icd_code"].str.startswith("E11")
        t2dm_ids = chunk[t2dm_mask]["subject_id"].unique()
        t2dm_patients.update(t2dm_ids)
        
        # 2. Identify End-Stage Renal Disease (ESRD) Exclusion Criteria
        # In ICD-10, N18.6 represents ESRD (patients already on dialysis)
        esrd_mask = chunk["icd_code"].str.startswith("N186")
        esrd_ids = chunk[esrd_mask]["subject_id"].unique()
        esrd_patients.update(esrd_ids)
        
        print(f"   Processed chunk {chunk_num}: Checked {chunk_num * chunk_size:,} rows...")

    print("\nEXTRACTION SUMMARY")
    print(f"Total unique T2DM patients found: {len(t2dm_patients):,}")
    print(f"Total unique ESRD patients found: {len(esrd_patients):,}")

    # 3. Apply Inclusion/Exclusion Logic
    # Convert sets to dataframes to perform the exclusion drop
    cohort_df = pd.DataFrame(list(t2dm_patients), columns=["subject_id"])
    
    # Keep only the patients whose subject_id is NOT in the ESRD exclusion set
    cohort_df = cohort_df[~cohort_df["subject_id"].isin(esrd_patients)]
    
    print(f"Final filtered cohort size (T2DM without ESRD): {len(cohort_df):,}")

    # Save the cohort list cleanly
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cohort_df.to_parquet(OUTPUT_FILE, index=False)
    print(f"Success! Saved cohort list to: {OUTPUT_FILE}")

if __name__ == "__main__":
    extract_diabetic_cohort()