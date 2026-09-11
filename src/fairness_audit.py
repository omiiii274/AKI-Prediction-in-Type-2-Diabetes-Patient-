import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
import joblib
import os

# Extends my original age-only fairness check to also cover sex, so this is
# a two-dimension audit rather than a single one. Same split conventions as
# the rest of src/.

INPUT_PATH = "data/processed/final_feature_matrix.parquet"
MODEL_DIR = "models"
OUTPUT_DIR = "models"
RANDOM_STATE = 42

print("=" * 60)
print("MULTIDIMENSIONAL FAIRNESS AUDIT (AGE + SEX)")
print("=" * 60)

df = pd.read_parquet(INPUT_PATH)
df['aki_binary'] = (df['kdigo_stage'] > 0).astype(int)

# Splitting by subject_id, not hadm_id - same reasoning as everywhere else
# in this project, since some patients have multiple ICU admissions.
patient_aki_status = df.groupby('subject_id')['aki_binary'].max().reset_index()
patient_aki_status.columns = ['subject_id', 'patient_ever_aki']

train_subjects, test_subjects = train_test_split(
    patient_aki_status['subject_id'], test_size=0.2, random_state=RANDOM_STATE,
    stratify=patient_aki_status['patient_ever_aki']
)
test_df = df[df['subject_id'].isin(test_subjects)].copy()

print(f"[INFO] Test set: {len(test_df):,} admissions from {test_df['subject_id'].nunique():,} patients")

# Checking this before running the audit rather than assuming - if this
# column got dropped somewhere in the merge, I'd rather get a clear error
# here than a fairness result that's silently based on an empty subgroup.
if 'gender' not in test_df.columns:
    raise ValueError(
        "'gender' column not found in the feature matrix. "
        "Check that extract_patient_cohort.py's gender column survived "
        "the merge into final_feature_matrix.parquet."
    )
print(f"[INFO] Gender value counts in test set:\n{test_df['gender'].value_counts()}")

# Using the deployed XGBoost model, since that's the one the app actually
# ships predictions from.
xgb_model = joblib.load(os.path.join(MODEL_DIR, "xgboost.joblib"))
feature_cols = list(xgb_model.feature_names_in_)

X_test = test_df[feature_cols]
test_df['xgb_pred'] = xgb_model.predict(X_test)
test_df['xgb_proba'] = xgb_model.predict_proba(X_test)[:, 1]


def group_metrics(group_df, group_name):
    """Positive prediction rate, recall, and false positive rate for one subgroup."""
    n = len(group_df)
    ppr = group_df['xgb_pred'].mean()
    recall = recall_score(group_df['aki_binary'], group_df['xgb_pred'])

    negatives = group_df[group_df['aki_binary'] == 0]
    fpr = negatives['xgb_pred'].mean() if len(negatives) > 0 else np.nan

    return {
        'group': group_name,
        'n': n,
        'positive_prediction_rate': round(ppr, 4),
        'recall_sensitivity': round(recall, 4),
        'false_positive_rate': round(fpr, 4)
    }


# 65 is the cutoff NHS guidance itself uses for higher AKI risk, so it felt
# like a clinically grounded threshold rather than an arbitrary one.
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

# Checked the value_counts() above first to confirm gender really is 'M'/'F'
# in this dataset before filtering on it.
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

# Tagging each row with its dimension so the app can split age and sex into
# separate tabs later without needing two different CSVs.
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
    "\n[NOTE] This is still a preliminary, single-model check. It reports "
    "raw group differences without controlling for illness severity or "
    "computing full equalised-odds metrics, so these gaps are candidates "
    "for further investigation, not confirmed evidence of bias "
    "(see dissertation Section 5.4)."
)
