import pandas as pd
import os

# Define clear paths
RAW_DATA_DIR = os.path.expanduser("~/Desktop/Dissertation/MIMIC-IV Dataset")
DIAGNOSES_FILE = os.path.join(RAW_DATA_DIR, "diagnoses_icd.csv")

OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "diabetes_cohort_list.parquet")

def extract_diabetic_cohort():
    print("="*65)
    print("STARTING LONGITUDINAL COHORT EXTRACTION FOR T2DM RESEARCH")
    print("="*65)
    
    if not os.path.exists(DIAGNOSES_FILE):
        print(f"Error: Cannot find the diagnosis file at: {DIAGNOSES_FILE}")
        return

    # Temporary storage for matching chunks
    t2dm_chunks = []
    esrd_hadm_ids = set()
    
    chunk_size = 1000000
    print(f"Reading {DIAGNOSES_FILE} sequentially in chunks of {chunk_size:,} rows...")

    # Step 1: Sequential processing chunk by chunk
    for chunk_num, chunk in enumerate(pd.read_csv(DIAGNOSES_FILE, chunksize=chunk_size), start=1):
        # Clean data type formatting
        chunk["icd_code"] = chunk["icd_code"].astype(str).str.strip()
        
        # Track baseline ESRD exclusion records across the entire dataset via hadm_id
        # ICD-9: 5856 | ICD-10: N186
        esrd_mask = chunk["icd_code"].str.startswith(("N186", "5856"))
        esrd_hadm_ids.update(chunk[esrd_mask]["hadm_id"].unique())
        
        # Filter for Type 2 Diabetes Mellitus (T2DM)
        # ICD-9: 25000, 25002, 25010, 25020... | ICD-10: E11
        t2dm_mask = chunk["icd_code"].str.startswith(("E11", "25000", "25002", "25010", "25020"))
        t2dm_records = chunk[t2dm_mask][["subject_id", "hadm_id", "icd_code", "icd_version"]]
        
        t2dm_chunks.append(t2dm_records)
        print(f"   Processed chunk {chunk_num}: Logged matching cohort records...")

    # Step 2: Consolidate data extractions
    print("\n[INFO] Compiling master dataframes...")
    raw_cohort_df = pd.concat(t2dm_chunks, ignore_index=True)
    
    print(f"[INFO] Initial T2DM records extracted: {len(raw_cohort_df):,}")
    print(f"[INFO] Admissions flagged with prior ESRD: {len(esrd_hadm_ids):,}")

    # Step 3: Apply Exclusion Logic via hospital admission mapping
    # Drop rows if that specific hospital stay already has an ESRD diagnosis
    filtered_cohort_df = raw_cohort_df[~raw_cohort_df["hadm_id"].isin(esrd_hadm_ids)]
    
    # Deduplicate so we keep one row per patient admission stay
    filtered_cohort_df = filtered_cohort_df.drop_duplicates(subset=["subject_id", "hadm_id"])

    print("\nEXTRACTION SUMMARY")
    print(f"Final Filtered Cohort Size (Unique Admissions): {len(filtered_cohort_df):,}")
    print(f"Unique Patient Count: {filtered_cohort_df['subject_id'].nunique():,}")

    # Step 4: Save the relational schema layout
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filtered_cohort_df.to_parquet(OUTPUT_FILE, index=False)
    print(f"\nSuccess! Saved multi-column cohort matrix to: {OUTPUT_FILE}")
    print("="*65)

if __name__ == "__main__":
    extract_diabetic_cohort()