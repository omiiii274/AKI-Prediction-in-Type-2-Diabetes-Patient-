import pandas as pd
import numpy as np
import os

RAW_DATA_DIR = os.path.expanduser("~/Desktop/Dissertation/MIMIC-IV Dataset")
TRIAGE_FILE = os.path.join(RAW_DATA_DIR, "triage.csv")
EDSTAYS_FILE = os.path.join(RAW_DATA_DIR, "edstays.csv")
COHORT_PATH = "data/processed/final_feature_matrix.parquet"
OUTPUT_PATH = "data/processed/ed_symptom_features.parquet"

print("=" * 60)
print("EXTRACTING MIMIC-IV-ED TRIAGE DATA FOR SYMPTOM MODEL")
print("=" * 60)

# 1. LOAD EXISTING AKI COHORT
cohort = pd.read_parquet(COHORT_PATH)

# FIX: aki_binary wasn't saved in the parquet file — recreate it from kdigo_stage
cohort['aki_binary'] = (cohort['kdigo_stage'] > 0).astype(int)

cohort_subjects = set(cohort['subject_id'].unique())
print(f"[INFO] Existing cohort: {len(cohort_subjects):,} unique patients")

# 2. LOAD ED STAYS
if not os.path.exists(EDSTAYS_FILE):
    raise FileNotFoundError(f"Could not find {EDSTAYS_FILE}")

print(f"[INFO] Loading ED stays from: {EDSTAYS_FILE}")
edstays = pd.read_csv(EDSTAYS_FILE, usecols=['subject_id', 'stay_id', 'hadm_id', 'intime'])
edstays = edstays[edstays['subject_id'].isin(cohort_subjects)]
print(f"[INFO] ED stays matching cohort patients: {len(edstays):,}")

# 3. LOAD TRIAGE DATA
print(f"[INFO] Loading triage records from: {TRIAGE_FILE}")
triage = pd.read_csv(TRIAGE_FILE)
print(f"[INFO] Total triage records loaded: {len(triage):,}")

# FIX: drop subject_id from triage before merging, since edstays already
# has subject_id — avoids pandas creating subject_id_x/subject_id_y
if 'subject_id' in triage.columns:
    triage = triage.drop(columns=['subject_id'])

# 4. MERGE — link triage to our cohort's ED stays
ed_merged = edstays.merge(triage, on='stay_id', how='inner')
print(f"[INFO] Matched triage records for cohort patients: {len(ed_merged):,}")

# 5. LINK TO AKI LABEL AT PATIENT LEVEL
patient_aki_status = (
    cohort.groupby('subject_id')['aki_binary']
    .max()
    .reset_index()
    .rename(columns={'aki_binary': 'ever_aki'})
)

print(f"[INFO] patient_aki_status shape: {patient_aki_status.shape}")
print(f"[INFO] patient_aki_status columns: {list(patient_aki_status.columns)}")

# Verify subject_id exists before merging (sanity check)
assert 'subject_id' in ed_merged.columns, "subject_id missing from ed_merged!"
assert 'subject_id' in patient_aki_status.columns, "subject_id missing from patient_aki_status!"

ed_merged = ed_merged.merge(patient_aki_status, on='subject_id', how='inner')
print(f"[INFO] Final ED symptom dataset: {len(ed_merged):,} records")
print(f"[INFO] AKI prevalence in this subset: {ed_merged['ever_aki'].mean()*100:.1f}%")

# 6. SELECT RELEVANT COLUMNS
keep_cols = [
    'subject_id', 'stay_id', 'chiefcomplaint',
    'temperature', 'heartrate', 'resprate', 'o2sat',
    'sbp', 'dbp', 'pain', 'acuity', 'ever_aki'
]
ed_final = ed_merged[[c for c in keep_cols if c in ed_merged.columns]].copy()

ed_final = ed_final.dropna(subset=['chiefcomplaint'])
ed_final['chiefcomplaint'] = ed_final['chiefcomplaint'].str.lower().str.strip()

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
ed_final.to_parquet(OUTPUT_PATH, index=False)

print(f"\n[SUCCESS] Saved ED symptom features to: {OUTPUT_PATH}")
print(f"Final shape: {ed_final.shape}")
print("=" * 60)
