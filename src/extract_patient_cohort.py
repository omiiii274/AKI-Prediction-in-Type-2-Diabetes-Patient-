import pandas as pd
import os

# FILE PATH CONFIGURATION
RAW_DATA_DIRECTORY = os.path.expanduser("~/Desktop/Dissertation/MIMIC-IV Dataset")
INPUT_PATIENTS_FILE = os.path.join(RAW_DATA_DIRECTORY, "patients.csv")

OUTPUT_DIRECTORY = "data"
FINAL_OUTPUT_FILE = os.path.join(OUTPUT_DIRECTORY, "extract_patient_cohort.parquet")

def run_cohort_extraction():
    print("="*60)
    print("STARTING SEPARATE PATIENT DEMOGRAPHIC EXTRACTION")
    print("="*60)
    
    # Step 1: Verify that the source dataset file exists
    if not os.path.exists(INPUT_PATIENTS_FILE):
        print(f"Error: The source file was not found at: {INPUT_PATIENTS_FILE}")
        return

    print(f"Successfully located source file: {INPUT_PATIENTS_FILE}")
    print("Reading full patients file into memory...")

    # Step 2: Read the complete file (No nrows limit so you don't lose your study population)
    raw_dataframe = pd.read_csv(INPUT_PATIENTS_FILE)
    
    # Step 3: Isolate the demographic dictionary columns
    # CRITICAL: We must include anchor_year and dod for time-series alignment later
    target_columns = ["subject_id", "gender", "anchor_age", "anchor_year", "dod"]
    filtered_cohort_dataframe = raw_dataframe[target_columns]
    
    print(f"\n[INFO] Total patient records imported: {len(filtered_cohort_dataframe):,}")
    print("\nSAMPLE EXTRACTED DATA:")
    print(filtered_cohort_dataframe.head(5))

    # Step 4: Save the demographic file as a standalone parquet
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    
    print(f"\nSaving standalone demographics to: {FINAL_OUTPUT_FILE}")
    filtered_cohort_dataframe.to_parquet(FINAL_OUTPUT_FILE, index=False)
    
    print("STANDALONE EXTRACTION SUCCESSFUL!")
    print("="*60)

if __name__ == "__main__":
    run_cohort_extraction()