import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_score, recall_score
)
import joblib
import os

# This is an empirical test of the causal claim made in Section 5.2:
# that dbp_count (diastolic blood pressure measurement frequency) is a
# confounded proxy for monitoring intensity rather than a true driver
# of AKI risk. If that's correct, removing it should barely move AUPRC.
#
# I reuse the exact same XGBoost hyperparameters already found by Optuna
# (models/xgboost_best_params.csv) rather than re-running the full
# 100-trial search, so this is a clean like-for-like comparison: same
# model, same data, same split — the only thing that changes is whether
# dbp_count is included as a feature.

INPUT_PATH = "data/processed/final_feature_matrix.parquet"
MODEL_DIR = "models"
OUTPUT_DIR = "models"
RANDOM_STATE = 42

print("=" * 60)
print("CAUSAL SENSITIVITY TEST: XGBOOST WITH / WITHOUT dbp_count")
print("=" * 60)


# 1. LOAD DATA AND RECREATE THE SAME SPLIT
df = pd.read_parquet(INPUT_PATH)
df['aki_binary'] = (df['kdigo_stage'] > 0).astype(int)

patient_aki_status = df.groupby('subject_id')['aki_binary'].max().reset_index()
patient_aki_status.columns = ['subject_id', 'patient_ever_aki']

train_subjects, test_subjects = train_test_split(
    patient_aki_status['subject_id'], test_size=0.2, random_state=RANDOM_STATE,
    stratify=patient_aki_status['patient_ever_aki']
)

train_df = df[df['subject_id'].isin(train_subjects)].copy()
test_df = df[df['subject_id'].isin(test_subjects)].copy()


# 2. FULL FEATURE SET (same exclude_cols as train_xgBoost.py)
exclude_cols = [
    'subject_id', 'hadm_id', 'icd_code', 'icd_version', 'kdigo_stage',
    'aki_binary', 'icu_intime', 'dod', 'gender'
]
full_feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ['int64', 'float64', 'int32']]

if 'dbp_count' not in full_feature_cols:
    raise ValueError(
        "'dbp_count' not found in the feature matrix. Check the exact "
        "column name in final_feature_matrix.parquet (it may be named "
        "slightly differently, e.g. 'dbp_measurement_count')."
    )

ablated_feature_cols = [c for c in full_feature_cols if c != 'dbp_count']

print(f"[INFO] Full feature set:    {len(full_feature_cols)} features")
print(f"[INFO] Ablated feature set: {len(ablated_feature_cols)} features (dbp_count removed)")


# 3. LOAD THE ALREADY-TUNED HYPERPARAMETERS
# I reuse Optuna's best hyperparameters rather than re-tuning, so this is
# a controlled comparison: hyperparameters, split, and random_state are
# identical between the two runs — only the feature set changes.
best_params = pd.read_csv(os.path.join(MODEL_DIR, "xgboost_best_params.csv")).iloc[0].to_dict()

# Optuna's int-typed params get read back as floats from CSV; cast them.
int_params = ['n_estimators', 'max_depth', 'min_child_weight']
for p in int_params:
    if p in best_params:
        best_params[p] = int(best_params[p])

print(f"\n[INFO] Reusing tuned hyperparameters: {best_params}")


def train_and_evaluate(feature_cols, label):
    X_train = train_df[feature_cols]
    y_train = train_df['aki_binary']
    X_test = test_df[feature_cols]
    y_test = test_df['aki_binary']

    n_no_aki = (y_train == 0).sum()
    n_aki = (y_train == 1).sum()
    scale_pos_weight = n_no_aki / n_aki

    params = dict(best_params)
    params['scale_pos_weight'] = scale_pos_weight
    params['random_state'] = RANDOM_STATE
    params['eval_metric'] = 'aucpr'
    params['use_label_encoder'] = False

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        'variant': label,
        'n_features': len(feature_cols),
        'auprc': round(average_precision_score(y_test, y_proba), 4),
        'auc_roc': round(roc_auc_score(y_test, y_proba), 4),
        'precision_aki': round(precision_score(y_test, y_pred), 4),
        'recall_aki': round(recall_score(y_test, y_pred), 4)
    }

    return model, metrics


# 4. TRAIN FULL MODEL (as a same-conditions baseline for this comparison)
print("\n[INFO] Training full-feature XGBoost (for like-for-like comparison)...")
_, full_metrics = train_and_evaluate(full_feature_cols, 'Full model (with dbp_count)')

# 5. TRAIN ABLATED MODEL
print("[INFO] Training XGBoost with dbp_count removed...")
_, ablated_metrics = train_and_evaluate(ablated_feature_cols, 'Ablated model (without dbp_count)')


# 6. COMPARE
results_df = pd.DataFrame([full_metrics, ablated_metrics])

delta = {
    'variant': 'Absolute change (ablated - full)',
    'n_features': ablated_metrics['n_features'] - full_metrics['n_features'],
    'auprc': round(ablated_metrics['auprc'] - full_metrics['auprc'], 4),
    'auc_roc': round(ablated_metrics['auc_roc'] - full_metrics['auc_roc'], 4),
    'precision_aki': round(ablated_metrics['precision_aki'] - full_metrics['precision_aki'], 4),
    'recall_aki': round(ablated_metrics['recall_aki'] - full_metrics['recall_aki'], 4)
}
results_df = pd.concat([results_df, pd.DataFrame([delta])], ignore_index=True)

print("\n" + "=" * 60)
print("RESULTS: dbp_count ABLATION")
print("=" * 60)
print(results_df.to_string(index=False))

auprc_change = delta['auprc']
print(f"\n[INTERPRETATION] AUPRC changed by {auprc_change:+.4f} when dbp_count was removed.")
if abs(auprc_change) < 0.01:
    print(
        "This is a negligible change, consistent with the hypothesis that "
        "dbp_count functions as a confounded proxy (correlated with illness "
        "severity via monitoring intensity) rather than an independent "
        "physiological driver the model actually depends on."
    )
else:
    print(
        "This change is larger than a rounding effect. If replicated, this "
        "would weaken the 'confounded proxy' interpretation in Section 5.2 "
        "and should be reported honestly rather than downplayed."
    )

results_df.to_csv(os.path.join(OUTPUT_DIR, "dbp_count_ablation_results.csv"), index=False)
print(f"\n[SUCCESS] Saved to: {OUTPUT_DIR}/dbp_count_ablation_results.csv")
print("=" * 60)