import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

X = pd.read_parquet("data/processed/symptom_model_features.parquet")
y = pd.read_parquet("data/processed/symptom_model_labels.parquet")['ever_aki']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Logistic Regression
lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
lr_proba = lr.predict_proba(X_test)[:, 1]
print(f"LR — AUPRC: {average_precision_score(y_test, lr_proba):.4f}, AUC-ROC: {roc_auc_score(y_test, lr_proba):.4f}")

# Random Forest
rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_proba = rf.predict_proba(X_test)[:, 1]
print(f"RF — AUPRC: {average_precision_score(y_test, rf_proba):.4f}, AUC-ROC: {roc_auc_score(y_test, rf_proba):.4f}")