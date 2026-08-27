import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

df = pd.read_parquet("data/processed/final_feature_matrix.parquet")
df['aki_binary'] = (df['kdigo_stage'] > 0).astype(int)

patient_aki_status = df.groupby('subject_id')['aki_binary'].max().reset_index()
patient_aki_status.columns = ['subject_id', 'patient_ever_aki']
train_subjects, test_subjects = train_test_split(
    patient_aki_status['subject_id'], test_size=0.2, random_state=42,
    stratify=patient_aki_status['patient_ever_aki']
)
test_df = df[df['subject_id'].isin(test_subjects)].copy()

xgb_model = joblib.load("models/xgboost.joblib")
feature_cols = list(xgb_model.feature_names_in_)

young = test_df[test_df['age_at_admission'] < 65]
old = test_df[test_df['age_at_admission'] >= 65]

young_preds = xgb_model.predict(young[feature_cols])
old_preds = xgb_model.predict(old[feature_cols])

young_rate = young_preds.mean()
old_rate = old_preds.mean()

print(f"Age <65  — positive prediction rate: {young_rate:.3f} (n={len(young)})")
print(f"Age 65+  — positive prediction rate: {old_rate:.3f} (n={len(old)})")
print(f"Disparity: {abs(young_rate - old_rate)*100:.1f} percentage points")

# Also check actual recall (sensitivity) per group — more clinically meaningful
from sklearn.metrics import recall_score
young_recall = recall_score(young['aki_binary'], young_preds)
old_recall = recall_score(old['aki_binary'], old_preds)
print(f"\nAge <65  — Recall (sensitivity): {young_recall:.3f}")
print(f"Age 65+  — Recall (sensitivity): {old_recall:.3f}")