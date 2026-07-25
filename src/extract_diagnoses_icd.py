import pandas as pd   # I use the pandas library for handling large table-like medical datasets (DataFrames).
import os             # I use the os module for working with file paths and checking if files/folders exist.


# Define clear paths
# os.path.expanduser() expands '~' to my home directory, so the path works reliably on my machine.
RAW_DATA_DIR = os.path.expanduser("~/Desktop/Dissertation/MIMIC-IV Dataset")

# os.path.join() safely combines folder and filename into a full path (better than manual string concatenation).
DIAGNOSES_FILE = os.path.join(RAW_DATA_DIR, "diagnoses_icd.csv")
ICUSTAYS_FILE = os.path.join(RAW_DATA_DIR, "icustays.csv")

# I define where I want to store the final cohort file.
OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "diabetes_cohort_list.parquet")


def extract_diabetic_cohort():
    """
    Main function: extracts a longitudinal Type 2 diabetes (T2DM) ICU cohort from MIMIC-IV,
    applying inclusion (T2DM + ICU) and exclusion (ESRD) criteria.
    """

    # print() is used here just to show a clear header when the script runs.
    print("=" * 65)
    print("STARTING LONGITUDINAL COHORT EXTRACTION FOR T2DM RESEARCH")
    print("=" * 65)


    # os.path.exists() checks if the diagnoses file actually exists before loading it.
    if not os.path.exists(DIAGNOSES_FILE):
        print(f"Error: Cannot find the diagnosis file at: {DIAGNOSES_FILE}")
        return  # return exits the function early if the file is missing.


    # Similarly, I check for the ICU stays file to ensure I can restrict to ICU admissions.
    if not os.path.exists(ICUSTAYS_FILE):
        print(f"Error: Cannot find icustays.csv at: {ICUSTAYS_FILE}")
        return


    # pd.read_csv() loads the ICU stays file into a pandas DataFrame.
    # The usecols argument tells pandas to load only the 'hadm_id' column,
    # which reduces memory usage because I don't need all columns here.
    icustays_df = pd.read_csv(ICUSTAYS_FILE, usecols=['hadm_id'])

    # .unique() returns all unique admission IDs, and set() converts them to a Python set
    # for fast membership testing later (when filtering ICU admissions).
    icu_hadm_ids = set(icustays_df['hadm_id'].unique())
    print(f"[INFO] Total unique admissions with an ICU stay in MIMIC-IV: {len(icu_hadm_ids):,}")


    # I prepare containers to accumulate results across chunks.
    t2dm_chunks = []         # This list will store T2DM records DataFrames from each chunk.
    esrd_hadm_ids = set()    # This set collects hadm_ids with ESRD diagnoses.
    esrd_subject_ids = set() # This set collects subject_ids (patients) with ESRD diagnosed.
    chunk_size = 1000000     # I choose a large chunk size to balance speed and memory.


    # Here I build ICD‑9 T2DM codes programmatically using a tuple comprehension and range().
    # range(10) loops over 0–9, and I combine that with suffixes '0' and '2', following clinical coding rules.
    icd9_t2dm_codes = tuple(f"250{comp}{suffix}" for comp in range(10) for suffix in ('0', '2'))

    # ICD‑10 T2DM codes start with the prefix "E11".
    icd10_t2dm_prefix = ("E11",)

    # I combine ICD‑9 and ICD‑10 patterns into one tuple for easier prefix matching later.
    all_t2dm_prefixes = icd10_t2dm_prefix + icd9_t2dm_codes


    print(f"Reading {DIAGNOSES_FILE} sequentially in chunks of {chunk_size:,} rows...")


    # pd.read_csv(..., chunksize=chunk_size) uses pandas' chunking functionality:
    # instead of loading the whole file at once, it yields smaller DataFrames one by one.
    # enumerate() gives me both the chunk number and the chunk itself, starting from 1.
    for chunk_num, chunk in enumerate(pd.read_csv(DIAGNOSES_FILE, chunksize=chunk_size), start=1):

        # .astype(str) converts the icd_code column to string type,
        # and .str.strip() removes any leading/trailing spaces, ensuring clean codes for prefix matching.
        chunk["icd_code"] = chunk["icd_code"].astype(str).str.strip()


        # .str.startswith() is a pandas string method that checks if each icd_code
        # begins with one of the ESRD patterns ("N186" for ICD‑10, "5856" for ICD‑9).
        esrd_mask = chunk["icd_code"].str.startswith(("N186", "5856"))

        # .unique() gets unique IDs; .update() on a set adds all these IDs efficiently.
        esrd_hadm_ids.update(chunk[esrd_mask]["hadm_id"].unique())
        esrd_subject_ids.update(chunk[esrd_mask]["subject_id"].unique())


        # Similarly, I use .str.startswith(all_t2dm_prefixes) to flag T2DM diagnoses.
        t2dm_mask = chunk["icd_code"].str.startswith(all_t2dm_prefixes)

        # Here I subset the DataFrame to only the relevant columns:
        # subject_id (patient), hadm_id (admission), icd_code, and icd_version.
        t2dm_records = chunk[t2dm_mask][["subject_id", "hadm_id", "icd_code", "icd_version"]]

        # I append the resulting T2DM records DataFrame to t2dm_chunks
        # so I can later concatenate all chunks into one big cohort.
        t2dm_chunks.append(t2dm_records)


        # Simple progress logging with print().
        print(f"   Processed chunk {chunk_num}: Logged matching cohort records...")


    print("\n[INFO] Compiling master dataframes...")

    # pd.concat() merges all the individual chunk DataFrames in t2dm_chunks
    # into a single DataFrame representing all T2DM diagnosis records found.
    raw_cohort_df = pd.concat(t2dm_chunks, ignore_index=True)

    # I use len() to show how many rows we have, and len(set) to show ESRD coverage.
    print(f"[INFO] Initial T2DM records extracted: {len(raw_cohort_df):,}")
    print(f"[INFO] Admissions flagged with ESRD (this stay): {len(esrd_hadm_ids):,}")
    print(f"[INFO] Patients flagged with ESRD (any prior stay): {len(esrd_subject_ids):,}")


    # Step 3: Apply exclusion logic (ESRD) AND inclusion logic (ICU stay required)
    # .isin() checks whether each hadm_id/subject_id appears in our ESRD sets and ICU sets.
    # The ~ operator negates the boolean mask (i.e., "not in").
    filtered_cohort_df = raw_cohort_df[
        ~raw_cohort_df["hadm_id"].isin(esrd_hadm_ids) &
        ~raw_cohort_df["subject_id"].isin(esrd_subject_ids) &
        raw_cohort_df["hadm_id"].isin(icu_hadm_ids)   # NEW: restrict to ICU admissions only
    ]


    # .drop_duplicates() removes duplicate rows where subject_id and hadm_id are the same,
    # so each patient‑admission pair appears only once in our final cohort.
    filtered_cohort_df = filtered_cohort_df.drop_duplicates(subset=["subject_id", "hadm_id"])


    print("\nEXTRACTION SUMMARY")
    # len() counts total ICU admissions in the filtered cohort.
    print(f"Final Filtered Cohort Size (Unique ICU Admissions): {len(filtered_cohort_df):,}")
    # .nunique() counts distinct patients (unique subject_id values).
    print(f"Unique Patient Count: {filtered_cohort_df['subject_id'].nunique():,}")


    # os.makedirs() creates the output directory if it does not exist.
    # exist_ok=True means it will not raise an error if the folder is already there.
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # DataFrame.to_parquet() saves the filtered cohort to disk in Parquet format,
    # which is efficient for large datasets and preserves column types.
    filtered_cohort_df.to_parquet(OUTPUT_FILE, index=False)

    print(f"\nSuccess! Saved multi-column cohort matrix to: {OUTPUT_FILE}")
    print("=" * 65)


# This if block uses Python’s standard “entry point” pattern.
# __name__ == "__main__" is True when this file is run directly (e.g. python script.py),
# so extract_diabetic_cohort() will execute.
# If I import this file as a module elsewhere, the function will NOT run automatically.
if __name__ == "__main__":
    extract_diabetic_cohort()