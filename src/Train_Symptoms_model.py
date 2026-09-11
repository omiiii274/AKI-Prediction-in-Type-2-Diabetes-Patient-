import pandas as pd  # I use pandas to load and manipulate the symptom feature and label datasets.
import numpy as np   # I use NumPy for numerical operations if needed during preprocessing or analysis.
from sklearn.model_selection import train_test_split  # I use this to create reproducible train/test splits.
import xgboost as xgb  # I use XGBoost as the classifier for symptom-based AKI risk prediction.
from sklearn.metrics import average_precision_score, roc_auc_score, classification_report  # I use these to evaluate ranking and classification performance.
import joblib  # I use joblib to save the trained model efficiently.
import os  # I use os for potential filesystem handling, although it is not used in this script version.


print("=" * 60)
print("TRAINING SYMPTOM-BASED AKI RISK MODEL")
print("=" * 60)


# I load the preprocessed feature matrix containing the symptom-model input variables.
X = pd.read_parquet("data/processed/symptom_model_features.parquet")
# I load the label file and select ever_aki as the binary target: 1 indicates a patient has experienced AKI.
y = pd.read_parquet("data/processed/symptom_model_labels.parquet")['ever_aki']


# I reserve 20% of the observations for unbiased final evaluation and train using the remaining 80%.
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    # I fix the random seed so this split can be reproduced exactly.
    random_state=42,
    # I preserve the AKI/non-AKI class proportion in both training and test data.
    stratify=y
)


# I count non-AKI training records to quantify the majority class.
n_no_aki = (y_train == 0).sum()
# I count AKI training records to quantify the minority class.
n_aki = (y_train == 1).sum()
# I calculate the class-weight ratio so XGBoost gives greater importance to under-represented AKI cases.
scale_pos_weight = n_no_aki / max(n_aki, 1)


print(f"[INFO] Train: {len(X_train):,} | Test: {len(X_test):,}")
print(f"[INFO] scale_pos_weight: {scale_pos_weight:.2f}")


# I configure an XGBoost classifier for the symptom-based AKI prediction task.
symptom_model = xgb.XGBClassifier(
    # I fit 150 boosted trees, balancing model capacity with training efficiency.
    n_estimators=150,
    # I limit each tree to depth four to control complexity and reduce overfitting on sparse symptom features.
    max_depth=4,
    # I use a low learning rate so each tree makes a measured contribution to the final model.
    learning_rate=0.05,
    # I apply the calculated positive-class weight to compensate for AKI class imbalance.
    scale_pos_weight=scale_pos_weight,
    # I set a fixed seed to make training reproducible.
    random_state=42,
    # I monitor area under the precision-recall curve because it is informative for imbalanced AKI outcomes.
    eval_metric='aucpr'
)
# I train the model using only the training features and labels.
symptom_model.fit(X_train, y_train)


# I generate AKI probabilities for the held-out test set to calculate threshold-independent metrics.
y_pred_proba = symptom_model.predict_proba(X_test)[:, 1]
# I generate default threshold-based class predictions for the classification report.
y_pred = symptom_model.predict(X_test)


# I calculate AUPRC, the primary ranking metric for this imbalanced binary classification problem.
auprc = average_precision_score(y_test, y_pred_proba)
# I calculate AUC-ROC to report overall class-discrimination performance.
auc_roc = roc_auc_score(y_test, y_pred_proba)


print(f"\n[RESULTS] AUPRC: {auprc:.4f} | AUC-ROC: {auc_roc:.4f}")
# I print precision, recall, F1-score, and support for the No AKI and AKI classes.
print(classification_report(y_test, y_pred, target_names=['No AKI', 'AKI']))


# I save the fitted model so the deployed Streamlit symptom checker can load the same trained estimator.
joblib.dump(symptom_model, "models/symptom_xgboost.joblib")
print("\n[SUCCESS] Symptom model saved to models/symptom_xgboost.joblib")
print("=" * 60)