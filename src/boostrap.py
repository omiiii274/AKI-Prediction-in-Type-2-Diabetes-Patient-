import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score
import joblib
import os


# PATH CONFIGURATION
# Same paths and conventions as the rest of the src/ scripts.
INPUT_PATH = "data/processed/final_feature_matrix.parquet"
MODEL_DIR = "models"
OUTPUT_DIR = "models"
N_BOOTSTRAPS = 1000
RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)

print("=" * 60)
print("BOOTSTRAP 95% CONFIDENCE INTERVALS (AUPRC / AUC-ROC)")
print("=" * 60)


# 1. LOAD DATA AND RECREATE THE SAME TEST SET
# I recreate the exact patient-level split used in train_xgBoost.py /
# train_random_forest.py / baseline_model.py so the bootstrap runs on
# the identical held-out test set that produced the reported point estimates.
df = pd.read_parquet(INPUT_PATH)
df['aki_binary'] = (df['kdigo_stage'] > 0).astype(int)

patient_aki_status = df.groupby('subject_id')['aki_binary'].max().reset_index()
patient_aki_status.columns = ['subject_id', 'patient_ever_aki']

train_subjects, test_subjects = train_test_split(
    patient_aki_status['subject_id'],
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=patient_aki_status['patient_ever_aki']
)

test_df = df[df['subject_id'].isin(test_subjects)].copy()
print(f"[INFO] Test set: {len(test_df):,} admissions from {test_df['subject_id'].nunique():,} patients")


# 2. LOAD THE THREE TRAINED MODELS
# I load the models exactly as saved, rather than retraining, so the
# bootstrap reflects uncertainty in the *evaluation*, not in training.
lr_model = joblib.load(os.path.join(MODEL_DIR, "baseline_logistic_regression.joblib"))
rf_model = joblib.load(os.path.join(MODEL_DIR, "random_forest.joblib"))
xgb_model = joblib.load(os.path.join(MODEL_DIR, "xgboost.joblib"))

scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))

print("[INFO] Loaded Logistic Regression, Random Forest, and XGBoost models.")


# 3. BUILD FEATURE MATRICES
# Same exclude_cols convention as train_xgBoost.py / treeshap_analysis.py.
exclude_cols = [
    'subject_id', 'hadm_id', 'icd_code', 'icd_version', 'kdigo_stage',
    'aki_binary', 'icu_intime', 'dod', 'gender'
]
feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ['int64', 'float64', 'int32']]

X_test = test_df[feature_cols]
y_test = test_df['aki_binary'].values

X_test_scaled = scaler.transform(X_test)  # for Logistic Regression only


# 4. COMPUTE PREDICTED PROBABILITIES ONCE
# I compute predictions a single time on the full test set, then resample
# the *indices* for each bootstrap iteration. This is much faster than
# re-running .predict_proba() 1,000 times per model and gives identical results.
lr_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
rf_proba = rf_model.predict_proba(X_test)[:, 1]
xgb_proba = xgb_model.predict_proba(X_test)[:, 1]

model_probas = {
    'Logistic Regression': lr_proba,
    'Random Forest': rf_proba,
    'XGBoost': xgb_proba
}


# 5. BOOTSTRAP RESAMPLING
# For each of 1,000 iterations, I resample the test set WITH replacement
# and recompute AUPRC / AUC-ROC. The spread of these 1,000 values gives
# an empirical estimate of how much the metric would move under a
# different random sample of patients.
n = len(y_test)
results = {name: {'auprc': [], 'auc_roc': []} for name in model_probas}

print(f"\n[INFO] Running {N_BOOTSTRAPS} bootstrap resamples (n={n:,} per resample)...")

rng = np.random.RandomState(RANDOM_STATE)

for i in range(N_BOOTSTRAPS):
    idx = rng.randint(0, n, size=n)
    y_boot = y_test[idx]

    # Skip the (extremely rare) resample with only one class present,
    # since AUPRC/AUC-ROC are undefined in that case.
    if len(np.unique(y_boot)) < 2:
        continue

    for name, proba in model_probas.items():
        proba_boot = proba[idx]
        results[name]['auprc'].append(average_precision_score(y_boot, proba_boot))
        results[name]['auc_roc'].append(roc_auc_score(y_boot, proba_boot))

    if (i + 1) % 200 == 0:
        print(f"  ...completed {i + 1}/{N_BOOTSTRAPS} resamples")


# 6. SUMMARISE: POINT ESTIMATE + 95% PERCENTILE INTERVAL
# The point estimate is computed on the ORIGINAL (non-resampled) test set,
# matching the number already reported in Table 2. The CI is the
# [2.5th, 97.5th] percentile of the bootstrap distribution around it.
rows = []
for name, proba in model_probas.items():
    point_auprc = average_precision_score(y_test, proba)
    point_auc = roc_auc_score(y_test, proba)

    auprc_lo, auprc_hi = np.percentile(results[name]['auprc'], [2.5, 97.5])
    auc_lo, auc_hi = np.percentile(results[name]['auc_roc'], [2.5, 97.5])

    rows.append({
        'model': name,
        'auprc': round(point_auprc, 4),
        'auprc_ci_low': round(auprc_lo, 4),
        'auprc_ci_high': round(auprc_hi, 4),
        'auc_roc': round(point_auc, 4),
        'auc_roc_ci_low': round(auc_lo, 4),
        'auc_roc_ci_high': round(auc_hi, 4),
        'n_bootstrap_iterations_used': len(results[name]['auprc'])
    })

ci_df = pd.DataFrame(rows)

print("\n" + "=" * 60)
print("BOOTSTRAP RESULTS (1,000 resamples, 95% percentile interval)")
print("=" * 60)
print(ci_df.to_string(index=False))

ci_df.to_csv(os.path.join(OUTPUT_DIR, "bootstrap_confidence_intervals.csv"), index=False)
print(f"\n[SUCCESS] Saved to: {OUTPUT_DIR}/bootstrap_confidence_intervals.csv")
print("=" * 60)