import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
import optuna
from sklearn.metrics import (
    average_precision_score, roc_auc_score, classification_report,
    confusion_matrix, precision_score, recall_score
)
import joblib
import os


# PATH CONFIGURATION
# I set the file paths here so the script is neat and easy to update later if needed.
INPUT_PATH = "data/processed/final_feature_matrix.parquet"
MODEL_OUTPUT_DIR = "models"
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)


print("=" * 60)
print("MODEL: XGBOOST (with Optuna Hyperparameter Tuning)")
print("=" * 60)


# 1. LOAD FINAL FEATURE MATRIX
# I load the prepared dataset because this is the cleaned version I want to use for training.
df = pd.read_parquet(INPUT_PATH)
print(f"[INFO] Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")


# I create a binary target so the model can predict AKI as a simple yes/no outcome.
df['aki_binary'] = (df['kdigo_stage'] > 0).astype(int)
print(f"[INFO] AKI prevalence: {df['aki_binary'].mean()*100:.1f}%")


# 2. PATIENT-LEVEL TRAIN/TEST SPLIT
# I split by patient instead of by row so the same patient does not appear in both training and testing.
# This avoids data leakage and makes the evaluation more trustworthy.
print("\n[INFO] Performing patient-level train/test split (same as LR and RF)...")

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


# 3. SELECT FEATURE COLUMNS
# I remove IDs, outcome variables, and non-useful columns so the model only learns from real predictors.
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


# 4. CALCULATE scale_pos_weight
# I calculate the class imbalance from the training data so the model gives more attention to AKI cases.
# This is important because AKI is the smaller class.
n_no_aki = (y_train == 0).sum()
n_aki = (y_train == 1).sum()
scale_pos_weight = n_no_aki / n_aki
print(f"\n[INFO] Computed scale_pos_weight: {scale_pos_weight:.2f} "
      f"(No AKI: {n_no_aki:,}, AKI: {n_aki:,})")


# 5. FURTHER SPLIT TRAINING DATA FOR OPTUNA VALIDATION
# I split the training set again so Optuna can tune the model using a validation set
# without touching the final test set. This keeps the test results fair.
train_patient_status = train_df.groupby('subject_id')['aki_binary'].max().reset_index()
train_patient_status.columns = ['subject_id', 'patient_ever_aki']

opt_train_subjects, opt_val_subjects = train_test_split(
    train_patient_status['subject_id'],
    test_size=0.2,
    random_state=42,
    stratify=train_patient_status['patient_ever_aki']
)

opt_train_df = train_df[train_df['subject_id'].isin(opt_train_subjects)]
opt_val_df = train_df[train_df['subject_id'].isin(opt_val_subjects)]

X_opt_train = opt_train_df[feature_cols]
y_opt_train = opt_train_df['aki_binary']
X_opt_val = opt_val_df[feature_cols]
y_opt_val = opt_val_df['aki_binary']

print(f"[INFO] Optuna training subset: {len(X_opt_train):,} rows")
print(f"[INFO] Optuna validation subset: {len(X_opt_val):,} rows")


# 6. OPTUNA HYPERPARAMETER TUNING
# I use Optuna to search for the best XGBoost settings instead of choosing them manually.
# The goal is to improve AUPRC and make the model better at finding AKI cases.
print("\n[INFO] Starting Optuna hyperparameter search (this may take a while)...")

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42,
        'eval_metric': 'aucpr',
        'use_label_encoder': False
    }

    # I train one trial model at a time and check how well it performs on the validation set.
    model = xgb.XGBClassifier(**params)
    model.fit(X_opt_train, y_opt_train)

    val_proba = model.predict_proba(X_opt_val)[:, 1]
    val_auprc = average_precision_score(y_opt_val, val_proba)

    # I return AUPRC because it is more suitable for imbalanced data like AKI prediction.
    return val_auprc


# I set the number of tuning trials so Optuna can test many combinations and find a strong model.
N_TRIALS = 100

study = optuna.create_study(direction='maximize', study_name='xgboost_aki_tuning')
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

print(f"\n[INFO] Best trial AUPRC (validation): {study.best_value:.4f}")
print(f"[INFO] Best hyperparameters found:")
for key, value in study.best_params.items():
    print(f"    {key}: {value}")


# I save the Optuna trial history so I can review the tuning process later if needed.
optuna_log = study.trials_dataframe()
optuna_log.to_csv(os.path.join(MODEL_OUTPUT_DIR, "optuna_xgboost_trials.csv"), index=False)
print(f"\n[INFO] Optuna trial log saved to: {MODEL_OUTPUT_DIR}/optuna_xgboost_trials.csv")


# 7. TRAIN FINAL XGBOOST MODEL ON FULL TRAINING SET WITH BEST PARAMETERS
# After tuning, I retrain the model on the full training set so it can learn from all available training data.
print("\n[INFO] Training final XGBoost model on full training set with best hyperparameters...")

best_params = study.best_params.copy()
best_params['scale_pos_weight'] = scale_pos_weight
best_params['random_state'] = 42
best_params['eval_metric'] = 'aucpr'
best_params['use_label_encoder'] = False

final_model = xgb.XGBClassifier(**best_params)
final_model.fit(X_train, y_train)


# 8. EVALUATE ON THE HELD-OUT TEST SET
# I evaluate only once on the test set so I get an honest estimate of how the model performs on unseen data.
y_pred = final_model.predict(X_test)
y_pred_proba = final_model.predict_proba(X_test)[:, 1]

auprc = average_precision_score(y_test, y_pred_proba)
auc_roc = roc_auc_score(y_test, y_pred_proba)
precision_aki = precision_score(y_test, y_pred)
recall_aki = recall_score(y_test, y_pred)

print("\n" + "=" * 60)
print("XGBOOST RESULTS (Optuna-tuned)")
print("=" * 60)
print(f"AUPRC: {auprc:.4f}")
print(f"AUC-ROC: {auc_roc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No AKI', 'AKI']))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# 9. FEATURE IMPORTANCE
# I extract feature importance so I can see which variables the model relied on most.
# This helps me interpret the model in a simpler way.
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 15 features by importance:")
print(importance_df.head(15).to_string(index=False))


# 10. SAVE MODEL AND RESULTS
# I save the trained model so I can reuse it later without having to retrain from scratch.
joblib.dump(final_model, os.path.join(MODEL_OUTPUT_DIR, "xgboost.joblib"))

# I save the feature importance table so I can use it in analysis or reporting.
importance_df.to_csv(os.path.join(MODEL_OUTPUT_DIR, "xgboost_feature_importance.csv"), index=False)

# I save the best hyperparameters so I have a record of the tuning results for my dissertation.
best_params_df = pd.DataFrame([study.best_params])
best_params_df.to_csv(os.path.join(MODEL_OUTPUT_DIR, "xgboost_best_params.csv"), index=False)

# I store the final performance metrics in one comparison file so I can compare all models easily.
metrics_summary = pd.DataFrame({
    'model': ['XGBoost'],
    'auprc': [auprc],
    'auc_roc': [auc_roc],
    'precision_aki': [precision_aki],
    'recall_aki': [recall_aki]
})
metrics_path = os.path.join(MODEL_OUTPUT_DIR, "model_comparison.csv")

if os.path.exists(metrics_path):
    existing = pd.read_csv(metrics_path)
    existing = existing[existing['model'] != 'XGBoost']  # I remove any old XGBoost row before saving again.
    metrics_summary = pd.concat([existing, metrics_summary], ignore_index=True)

metrics_summary.to_csv(metrics_path, index=False)

print(f"\n[SUCCESS] Model saved to: {MODEL_OUTPUT_DIR}/xgboost.joblib")
print(f"[SUCCESS] Best params saved to: {MODEL_OUTPUT_DIR}/xgboost_best_params.csv")
print(f"[SUCCESS] Metrics appended to: {metrics_path}")
print("=" * 60)