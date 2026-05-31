import pandas as pd
import os

# Updated to match your exact case-sensitive path and subfolder
DESKTOP_DATA_DIR = os.path.expanduser("~/Desktop/Dissertation/MIMIC-IV Dataset")
OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "raw_cohort.parquet")

def extract_cohort():
    # Matching your exact file name with the .html extension for now
    patients_file = os.path.join(DESKTOP_DATA_DIR, "patients.csv")
    
    if not os.path.exists(patients_file):
        print(f"Error: Could not find {patients_file}")
        print("Please verify the file name inside your Desktop/Dissertation/MIMIC-IV Dataset/ patients.csv .")
        return

    print(f"Reading file from: {patients_file}...")
    
    try:
        # Trying to read it, but warning you below why this might fail
        df = pd.read_csv(patients_file, nrows=1000)
        
        available_cols = [col for col in ['subject_id', 'gender', 'anchor_age'] if col in df.columns]
        df_cohort = df[available_cols]
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        df_cohort.to_parquet(OUTPUT_FILE, index=False)
        print("Extraction complete! Check data/raw_cohort.parquet")
        
    except Exception as e:
        print(f"Failed to read the file. Error: {e}")
        print("\nCRITICAL WARNING: Your file ends in '.html'. If this is an HTML webpage rather than raw data, Pandas cannot parse it. You need to download the actual raw 'patients.csv.gz' file from PhysioNet, not its download link page.")

if __name__ == "__main__":
    extract_cohort()