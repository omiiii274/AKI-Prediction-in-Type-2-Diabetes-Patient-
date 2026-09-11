import pandas as pd  # I use pandas to load and manipulate the processed clinical dataset.
import numpy as np   # I use NumPy for numerical operations if needed during preprocessing.
from sklearn.model_selection import train_test_split  # I use this to create reproducible train/test splits.
import xgboost as xgb  # I use XGBoost as the classifier for the clinician-input model.
from sklearn.metrics import average_precision_score, roc_auc_score  # I use these to evaluate ranking performance.
import joblib  # I use joblib to save the trained model efficiently.
import os  # I use os to construct file paths for saving outputs.


INPUT_PATH = "data/processed/final_feature_matrix.parquet"
MODEL_OUTPUT_DIR = "models"


# I load the full feature matrix that was created by the main preprocessing pipeline.
df = pd.read_parquet(INPUT_PATH)
# I create the binary AKI target using the same KDIGO definition as the main cohort.
df['aki_binary'] = (df['kdigo_stage'] > 0).astype(int)


# Only the features the doctor will actually enter
# I restrict the model to variables a clinician can realistically provide at the bedside.
clinician_features = [
    'age_at_admission',
    'baseline_creatinine',
    'creatinine_slope_24h',
    'glucose_mean',
    'glucose_cv',
    'map_mean',
    'hr_mean'
]


# Same patient-level split as before, for consistency
# I split by subject_id rather than admission to avoid patient leakage between train and test.
patient_aki_status = df.groupby('subject_id')['aki_binary'].max().reset_index()
patient_aki_status.columns = ['subject_id', 'patient_ever_aki']


train_subjects, test_subjects = train_test_split(
    patient_aki_status['subject_id'],
    test_size=0.2, random_state=42,
    # I stratify on patient-level AKI status to preserve class balance across splits.
    stratify=patient_aki_status['patient_ever_aki']
)


train_df = df[df['subject_id'].isin(train_subjects)].copy()
test_df = df[df['subject_id'].isin(test_subjects)].copy()


X_train = train_df[clinician_features]
y_train = train_df['aki_binary']
X_test = test_df[clinician_features]
y_test = test_df['aki_binary']


# I calculate the class imbalance ratio to inform XGBoost's internal weighting.
n_no_aki = (y_train == 0).sum()
n_aki = (y_train == 1).sum()
scale_pos_weight = n_no_aki / n_aki


# I configure XGBoost with slightly more capacity than the symptom model because these are structured numeric features.
clinician_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    # I monitor AUPRC during training because AKI is the minority class.
    eval_metric='aucpr'
)
clinician_model.fit(X_train, y_train)


y_pred_proba = clinician_model.predict_proba(X_test)[:, 1]
print(f"Clinician-only model AUPRC: {average_precision_score(y_test, y_pred_proba):.4f}")
print(f"Clinician-only model AUC-ROC: {roc_auc_score(y_test, y_pred_proba):.4f}")


joblib.dump(clinician_model, os.path.join(MODEL_OUTPUT_DIR, "clinician_xgboost.joblib"))
print("Saved clinician-only model.")