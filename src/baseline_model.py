import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, roc_auc_score, classification_report,
    confusion_matrix, precision_recall_curve, precision_score, recall_score
)
from sklearn.preprocessing import StandardScaler
import joblib
import os


# PATH CONFIGURATION
# I define the input file and output folder here so the script stays tidy and easy to update.
INPUT_PATH = "data/processed/final_feature_matrix.parquet"
MODEL_OUTPUT_DIR = "models"
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)


print("=" * 60)
print("BASELINE MODEL: LOGISTIC REGRESSION")
print("=" * 60)


# 1. LOAD DATA
# I load the prepared dataset that already contains the features I need for modelling.
df = pd.read_parquet(INPUT_PATH)
print(f"[INFO] Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")


# 2. CREATE TARGET VARIABLE
# I turn kdigo_stage into a binary target so the model predicts AKI vs no AKI.
df['aki_binary'] = (df['kdigo_stage'] > 0).astype(int)
print(f"[INFO] AKI prevalence: {df['aki_binary'].mean()*100:.1f}%")


# 3. PATIENT-LEVEL SPLITTING
# I split by patient instead of by row so the same patient does not appear in both train and test sets.
# This avoids data leakage and gives a fairer evaluation.
print("\n[INFO] Performing patient-level train/test split...")

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
print(f"[INFO] Train AKI prevalence: {train_df['aki_binary'].mean()*100:.1f}%")
print(f"[INFO] Test AKI prevalence:  {test_df['aki_binary'].mean()*100:.1f}%")

overlap = set(train_df['subject_id']) & set(test_df['subject_id'])
print(f"[VERIFY] Patient overlap between train/test: {len(overlap)} (should be 0)")


# 4. FEATURE SELECTION
# I remove IDs, target columns, and non-numeric fields so the model only uses real predictors.
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


# 5. FEATURE SCALING
# I standardise the features because Logistic Regression performs better when variables are on a similar scale.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 6. MODEL TRAINING
# I train a balanced Logistic Regression model as a simple and interpretable baseline.
print("\n[INFO] Training baseline Logistic Regression...")

model = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    random_state=42
)
model.fit(X_train_scaled, y_train)


# 7. MODEL EVALUATION
# I test the model on unseen data to see how well it performs in practice.
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

auprc = average_precision_score(y_test, y_pred_proba)
auc_roc = roc_auc_score(y_test, y_pred_proba)
precision_aki = precision_score(y_test, y_pred)
recall_aki = recall_score(y_test, y_pred)

print("\n" + "=" * 60)
print("BASELINE MODEL RESULTS")
print("=" * 60)
print(f"AUPRC: {auprc:.4f}")
print(f"AUC-ROC: {auc_roc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No AKI', 'AKI']))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# 8. FEATURE IMPORTANCE
# I look at the coefficients to see which features push the prediction toward AKI or away from it.
coef_df = pd.DataFrame({
    'feature': feature_cols,
    'coefficient': model.coef_[0]
})
coef_df = coef_df.sort_values('coefficient', key=abs, ascending=False)

print("\nTop 15 features by |coefficient| (standardized):")
print(coef_df.head(15).to_string(index=False))


# 9. SAVE MODEL
# I save the trained model and scaler so I can reuse them later without retraining.
joblib.dump(model, os.path.join(MODEL_OUTPUT_DIR, "baseline_logistic_regression.joblib"))
joblib.dump(scaler, os.path.join(MODEL_OUTPUT_DIR, "scaler.joblib"))

# I also save the coefficients so I can use them in analysis or reporting.
coef_df.to_csv(os.path.join(MODEL_OUTPUT_DIR, "baseline_lr_coefficients.csv"), index=False)


# 10. SAVE METRICS FOR COMPARISON
# I store the performance results in a shared file so I can compare this model with RF and XGBoost.
metrics_summary = pd.DataFrame({
    'model': ['Logistic Regression'],
    'auprc': [auprc],
    'auc_roc': [auc_roc],
    'precision_aki': [precision_aki],
    'recall_aki': [recall_aki]
})
metrics_path = os.path.join(MODEL_OUTPUT_DIR, "model_comparison.csv")

if os.path.exists(metrics_path):
    existing = pd.read_csv(metrics_path)
    existing = existing[existing['model'] != 'Logistic Regression']
    metrics_summary = pd.concat([existing, metrics_summary], ignore_index=True)

metrics_summary.to_csv(metrics_path, index=False)

print(f"\n[SUCCESS] Model saved to: {MODEL_OUTPUT_DIR}/baseline_logistic_regression.joblib")
print(f"[SUCCESS] Metrics appended to: {metrics_path}")
print("=" * 60)