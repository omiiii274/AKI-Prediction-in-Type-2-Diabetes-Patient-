import pandas as pd  # I use pandas to load and manipulate the clinical dataset for fairness analysis.
import numpy as np   # I use NumPy for numerical operations such as NaN handling.
from sklearn.model_selection import train_test_split  # I use this to recreate the same test split as model training.
from sklearn.metrics import recall_score  # I use recall_score to measure sensitivity within each subgroup.
import joblib  # I use joblib to load the trained XGBoost model.
import os  # I use os to construct file paths for model and output locations.


# This extends the original src/fairness_audit.py (age only) to also
# cover sex, so the audit is two-dimensional rather than single-dimension.
# Same split logic and conventions as the rest of src/.


INPUT_PATH = "data/processed/final_feature_matrix.parquet"
MODEL_DIR = "models"
OUTPUT_DIR = "models"
RANDOM_STATE = 42


print("=" * 60)
print("MULTIDIMENSIONAL FAIRNESS AUDIT (AGE + SEX)")
print("=" * 60)


# 1. LOAD DATA AND RECREATE THE SAME TEST SET
# I load the same processed feature matrix used for model training.
df = pd.read_parquet(INPUT_PATH)
# I create the binary AKI target using the same KDIGO definition as the main cohort.
df['aki_binary'] = (df['kdigo_stage'] > 0).astype(int)


# I split by subject_id rather than admission to avoid patient leakage between train and test.
patient_aki_status = df.groupby('subject_id')['aki_binary'].max().reset_index()
patient_aki_status.columns = ['subject_id', 'patient_ever_aki']


train_subjects, test_subjects = train_test_split(
    patient_aki_status['subject_id'], test_size=0.2, random_state=RANDOM_STATE,
    # I stratify on patient-level AKI status to preserve class balance across splits.
    stratify=patient_aki_status['patient_ever_aki']
)
test_df = df[df['subject_id'].isin(test_subjects)].copy()


print(f"[INFO] Test set: {len(test_df):,} admissions from {test_df['subject_id'].nunique():,} patients")


# Sanity check: confirm the 'gender' column is present and see its values
# before running the audit, since MIMIC-IV typically encodes this as 'M'/'F'.
if 'gender' not in test_df.columns:
    raise ValueError(
        "'gender' column not found in the feature matrix. "
        "Check that extract_patient_cohort.py's gender column survived "
        "the merge into final_feature_matrix.parquet."
    )
print(f"[INFO] Gender value counts in test set:\n{test_df['gender'].value_counts()}")


# 2. LOAD XGBOOST MODEL (the deployed / primary model)
xgb_model = joblib.load(os.path.join(MODEL_DIR, "xgboost.joblib"))
# I use the feature names stored in the model to ensure exact alignment with training.
feature_cols = list(xgb_model.feature_names_in_)


X_test = test_df[feature_cols]
test_df['xgb_pred'] = xgb_model.predict(X_test)
test_df['xgb_proba'] = xgb_model.predict_proba(X_test)[:, 1]


# 3. HELPER: COMPUTE GROUP METRICS
def group_metrics(group_df, group_name):
    n = len(group_df)
    ppr = group_df['xgb_pred'].mean()  # positive prediction rate
    recall = recall_score(group_df['aki_binary'], group_df['xgb_pred'])

    # False positive rate: predicted positive among true negatives
    negatives = group_df[group_df['aki_binary'] == 0]
    fpr = negatives['xgb_pred'].mean() if len(negatives) > 0 else np.nan

    return {
        'group': group_name,
        'n': n,
        'positive_prediction_rate': round(ppr, 4),
        'recall_sensitivity': round(recall, 4),
        'false_positive_rate': round(fpr, 4)
    }


# 4. AGE DIMENSION (same threshold as the original single-dimension check)
# I use 65 years as the age threshold, consistent with common clinical definitions of older adults.
young = test_df[test_df['age_at_admission'] < 65]
old = test_df[test_df['age_at_admission'] >= 65]


age_rows = [
    group_metrics(young, 'Age <65'),
    group_metrics(old, 'Age >=65')
]


age_df = pd.DataFrame(age_rows)
age_gap_ppr = abs(age_df.loc[0, 'positive_prediction_rate'] - age_df.loc[1, 'positive_prediction_rate'])
age_gap_recall = abs(age_df.loc[0, 'recall_sensitivity'] - age_df.loc[1, 'recall_sensitivity'])


print("\n" + "-" * 60)
print("AGE SUBGROUP RESULTS")
print("-" * 60)
print(age_df.to_string(index=False))
print(f"\nPPR gap:    {age_gap_ppr*100:.1f} percentage points")
print(f"Recall gap: {age_gap_recall*100:.1f} percentage points")


# 5. SEX DIMENSION (new)
# MIMIC-IV encodes gender as 'M' / 'F'. Adjust the values below if your
# extracted column uses a different encoding (check the value_counts()
# printed above before trusting these labels).
female = test_df[test_df['gender'] == 'F']
male = test_df[test_df['gender'] == 'M']


sex_rows = [
    group_metrics(female, 'Female'),
    group_metrics(male, 'Male')
]


sex_df = pd.DataFrame(sex_rows)
sex_gap_ppr = abs(sex_df.loc[0, 'positive_prediction_rate'] - sex_df.loc[1, 'positive_prediction_rate'])
sex_gap_recall = abs(sex_df.loc[0, 'recall_sensitivity'] - sex_df.loc[1, 'recall_sensitivity'])


print("\n" + "-" * 60)
print("SEX SUBGROUP RESULTS")
print("-" * 60)
print(sex_df.to_string(index=False))
print(f"\nPPR gap:    {sex_gap_ppr*100:.1f} percentage points")
print(f"Recall gap: {sex_gap_recall*100:.1f} percentage points")


# 6. SAVE COMBINED OUTPUT
# I tag each row with its dimension so the Streamlit app can separate age and sex tabs.
age_df['dimension'] = 'age'
sex_df['dimension'] = 'sex'
combined = pd.concat([age_df, sex_df], ignore_index=True)
combined = combined[['dimension', 'group', 'n', 'positive_prediction_rate',
                      'recall_sensitivity', 'false_positive_rate']]


combined.to_csv(os.path.join(OUTPUT_DIR, "fairness_audit_extended.csv"), index=False)


print("\n" + "=" * 60)
print(f"[SUCCESS] Saved to: {OUTPUT_DIR}/fairness_audit_extended.csv")
print("=" * 60)
print(
    "\n[NOTE] This remains a preliminary, single-model check. It reports "
    "raw group differences (demographic-parity-style) without controlling "
    "for illness severity or computing full equalised-odds metrics, so "
    "gaps here should be read as candidates for further investigation, "
    "not as confirmed evidence of unfair bias (see dissertation Section 5.4)."
)