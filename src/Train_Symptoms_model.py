import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score, classification_report
import joblib
import os

print("=" * 60)
print("TRAINING SYMPTOM-BASED AKI RISK MODEL")
print("=" * 60)

X = pd.read_parquet("data/processed/symptom_model_features.parquet")
y = pd.read_parquet("data/processed/symptom_model_labels.parquet")['ever_aki']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

n_no_aki = (y_train == 0).sum()
n_aki = (y_train == 1).sum()
scale_pos_weight = n_no_aki / max(n_aki, 1)

print(f"[INFO] Train: {len(X_train):,} | Test: {len(X_test):,}")
print(f"[INFO] scale_pos_weight: {scale_pos_weight:.2f}")

symptom_model = xgb.XGBClassifier(
    n_estimators=150,
    max_depth=4,
    learning_rate=0.05,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric='aucpr'
)
symptom_model.fit(X_train, y_train)

y_pred_proba = symptom_model.predict_proba(X_test)[:, 1]
y_pred = symptom_model.predict(X_test)

auprc = average_precision_score(y_test, y_pred_proba)
auc_roc = roc_auc_score(y_test, y_pred_proba)

print(f"\n[RESULTS] AUPRC: {auprc:.4f} | AUC-ROC: {auc_roc:.4f}")
print(classification_report(y_test, y_pred, target_names=['No AKI', 'AKI']))

joblib.dump(symptom_model, "models/symptom_xgboost.joblib")
print("\n[SUCCESS] Symptom model saved to models/symptom_xgboost.joblib")
print("=" * 60)