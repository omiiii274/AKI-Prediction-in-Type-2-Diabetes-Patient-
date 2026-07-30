import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score, roc_auc_score, classification_report,
    confusion_matrix, precision_score, recall_score
)
import joblib
import os


# PATH CONFIGURATION
# I set the input and output paths here so the script stays organised and easy to change later.
INPUT_PATH = "data/processed/final_feature_matrix.parquet"
MODEL_OUTPUT_DIR = "models"
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)


print("=" * 60)
print("MODEL: RANDOM FOREST")
print("=" * 60)


# 1. LOAD FINAL FEATURE MATRIX
# I load the cleaned and prepared dataset because this is the final version I want to use for modelling.
df = pd.read_parquet(INPUT_PATH)
print(f"[INFO] Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")


# 2. CREATE BINARY TARGET (AKI vs No-AKI) — same as baseline
# I create a simple binary target so the model can learn a clear yes/no prediction for AKI.
df['aki_binary'] = (df['kdigo_stage'] > 0).astype(int)
print(f"[INFO] AKI prevalence: {df['aki_binary'].mean()*100:.1f}%")


# 3. PATIENT-LEVEL TRAIN/TEST SPLIT
# I split by patient instead of by row so that the same patient does not appear in both train and test sets.
# This is important because it prevents data leakage and makes the evaluation fairer.
print("\n[INFO] Performing patient-level train/test split (same as baseline)...")

patient_aki_status = df.groupby('subject_id')['aki_binary'].max().reset_index()
patient_aki_status.columns = ['subject_id', 'patient_ever_aki']

train_subjects, test_subjects = train_test_split(
    patient_aki_status['subject_id'],
    test_size=0.2,
    random_state=42,
    stratify=patient_aki_status['patient_ever_aki']
)

train_df = df[df['subject_id'].isin(train_subjects)].copy()
test_df = df[df['subject_id'].isin(test_subjects)].copy()

print(f"[INFO] Train set: {len(train_df):,} admissions from {train_df['subject_id'].nunique():,} patients")
print(f"[INFO] Test set:  {len(test_df):,} admissions from {test_df['subject_id'].nunique():,} patients")

overlap = set(train_df['subject_id']) & set(test_df['subject_id'])
print(f"[VERIFY] Patient overlap between train/test: {len(overlap)} (should be 0)")


# 4. SELECT FEATURE COLUMNS
# I exclude IDs, outcome columns, and non-numeric columns because the model only needs usable input features.
exclude_cols = [
    'subject_id', 'hadm_id', 'icd_code', 'icd_version', 'kdigo_stage',
    'aki_binary', 'icu_intime', 'dod', 'gender'
]
feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ['int64', 'float64', 'int32']]

print(f"\n[INFO] Using {len(feature_cols)} features for modeling.")

X_train = train_df[feature_cols]
y_train = train_df['aki_binary']
X_test = test_df[feature_cols]
y_test = test_df['aki_binary']


# I do not scale the data here because Random Forest works well with raw numeric values.
# Unlike Logistic Regression, it does not need standardisation to perform properly.


# 5. TRAIN RANDOM FOREST
# I train a Random Forest because it can capture non-linear patterns and interactions between features.
# I also use class weighting to help the model pay more attention to the minority AKI class.
print("\n[INFO] Training Random Forest...")

model = RandomForestClassifier(
    n_estimators=300,        # I use many trees so the model is more stable and reliable.
    max_depth=10,            # I limit tree depth to reduce overfitting.
    min_samples_leaf=5,      # I avoid tiny leaves so the trees do not become too specific.
    class_weight='balanced', # I balance the classes because AKI cases may be less common.
    random_state=42,         # I fix the seed so the results are reproducible.
    n_jobs=-1                # I use all CPU cores to make training faster.
)
model.fit(X_train, y_train)


# 6. EVALUATE
# I evaluate on the test set so I can measure how well the model generalises to unseen data.
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

auprc = average_precision_score(y_test, y_pred_proba)
auc_roc = roc_auc_score(y_test, y_pred_proba)
precision_aki = precision_score(y_test, y_pred)
recall_aki = recall_score(y_test, y_pred)

print("\n" + "=" * 60)
print("RANDOM FOREST RESULTS")
print("=" * 60)
print(f"AUPRC: {auprc:.4f}")
print(f"AUC-ROC: {auc_roc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No AKI', 'AKI']))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# 7. FEATURE IMPORTANCE
# I extract feature importance so I can see which variables the Random Forest relied on most.
# This helps with interpretability, even though the meaning is different from regression coefficients.
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 15 features by importance:")
print(importance_df.head(15).to_string(index=False))


# 8. SAVE MODEL AND RESULTS
# I save the trained model so I can reuse it later without retraining.
joblib.dump(model, os.path.join(MODEL_OUTPUT_DIR, "random_forest.joblib"))

# I save the feature importance table so I can review and report the most useful variables.
importance_df.to_csv(os.path.join(MODEL_OUTPUT_DIR, "random_forest_feature_importance.csv"), index=False)

# I save the performance metrics so I can compare this model with Logistic Regression and XGBoost later.
metrics_summary = pd.DataFrame({
    'model': ['Random Forest'],
    'auprc': [auprc],
    'auc_roc': [auc_roc],
    'precision_aki': [precision_aki],
    'recall_aki': [recall_aki]
})
metrics_path = os.path.join(MODEL_OUTPUT_DIR, "model_comparison.csv")

# I append the new results to the comparison file so I keep one clean summary table for all models.
if os.path.exists(metrics_path):
    existing = pd.read_csv(metrics_path)
    existing = existing[existing['model'] != 'Random Forest']  # I remove any old Random Forest row before saving again.
    metrics_summary = pd.concat([existing, metrics_summary], ignore_index=True)

metrics_summary.to_csv(metrics_path, index=False)

print(f"\n[SUCCESS] Model saved to: {MODEL_OUTPUT_DIR}/random_forest.joblib")
print(f"[SUCCESS] Metrics appended to: {metrics_path}")
print("=" * 60)