# I start by importing all the necessary libraries.
# import pandas as pd: gives me the pandas library, which I use to handle tabular data (DataFrames).
# import numpy as np: gives me NumPy, which supports efficient numerical operations.
# Together, these form the core of my data manipulation stack.
import pandas as pd
import numpy as np


# from sklearn.model_selection import train_test_split:
# I import the train_test_split() function, which is used to split data into
# training and testing sets in a controlled way (including stratification).
from sklearn.model_selection import train_test_split


# from sklearn.linear_model import LogisticRegression:
# I import the LogisticRegression class, which implements a baseline linear
# classification model suitable for binary outcomes such as AKI vs no AKI.
from sklearn.linear_model import LogisticRegression


# from sklearn.metrics import (...):
# I import several evaluation functions:
# - average_precision_score(): computes area under the precision–recall curve (AUPRC).
# - roc_auc_score(): computes area under the ROC curve (AUC-ROC).
# - classification_report(): summarizes precision, recall, F1 and support per class.
# - confusion_matrix(): shows counts of true vs predicted classes.
# - precision_recall_curve(): gives points to plot the full precision–recall curve.
from sklearn.metrics import (
    average_precision_score, roc_auc_score, classification_report,
    confusion_matrix, precision_recall_curve
)


# from sklearn.preprocessing import StandardScaler:
# I import the StandardScaler class, which standardises features by removing the mean
# and scaling to unit variance. I use its fit_transform() and transform() methods later.
from sklearn.preprocessing import StandardScaler


# import joblib: provides dump() and load() functions
# that allow me to save and reload trained models and preprocessors efficiently.
import joblib


# import os: gives me access to file system-related functions,
# such as os.makedirs() and os.path.join() for directory and path management.
import os



# PATH CONFIGURATION 
# I define where my processed dataset is stored as a simple string path.
INPUT_PATH = "data/processed/final_feature_matrix.parquet"

# I define the directory where I want to save my trained model and related files.
MODEL_OUTPUT_DIR = "models"

# os.makedirs(path, exist_ok=True) creates the directory if it doesn't exist.
# The exist_ok=True flag avoids raising an error if the folder is already present.
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)



# HEADER PRINTING
# print() is a basic Python function I use to display a visual separator.
print("=" * 60)
# Another print() call to show a clear title for this script’s output block.
print("BASELINE MODEL: LOGISTIC REGRESSION")
print("=" * 60)



# 1. LOAD DATA 

# pd.read_parquet() reads a Parquet file into a pandas DataFrame.
# I use this to load my final feature matrix, which has already been preprocessed.
df = pd.read_parquet(INPUT_PATH)

# Here I use f-string formatting and df.shape to report:
# - df.shape[0]: number of rows (admissions),
# - df.shape[1]: number of columns (features + metadata).
print(f"[INFO] Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")



# 2. CREATE TARGET VARIABLE 

# I create a new column 'aki_binary' using a vectorised comparison:
# (df['kdigo_stage'] > 0) returns a boolean Series.
# .astype(int) converts True/False into 1/0.
# This turns AKI severity into a simple binary classification label.
df['aki_binary'] = (df['kdigo_stage'] > 0).astype(int)

# df['aki_binary'].mean() computes the average of 0/1 values,
# which corresponds to the proportion of AKI cases in the dataset.
print(f"[INFO] AKI prevalence: {df['aki_binary'].mean()*100:.1f}%")



# 3. PATIENT-LEVEL SPLITTING 

# print() informs the user (and my future self) that I am now performing
# a patient-level train/test split instead of admission-level.
print("\n[INFO] Performing patient-level train/test split...")


# df.groupby('subject_id') groups all rows by patient ID.
# ['aki_binary'].max() returns, for each patient, the maximum AKI status across their admissions,
# effectively marking whether they EVER had AKI.
# reset_index() converts the grouped result back into a regular DataFrame.
patient_aki_status = df.groupby('subject_id')['aki_binary'].max().reset_index()

# I then rename the columns of this summary DataFrame using .columns assignment
# to make their meaning clear and explicit.
patient_aki_status.columns = ['subject_id', 'patient_ever_aki']


# train_test_split() here operates on the list of patient IDs.
# Parameters:
# - test_size=0.2 reserves 20% of patients for testing.
# - random_state=42 ensures reproducible splitting.
# - stratify=patient_aki_status['patient_ever_aki'] maintains similar AKI prevalence
#   in both train and test sets.
train_subjects, test_subjects = train_test_split(
    patient_aki_status['subject_id'],
    test_size=0.2,
    random_state=42,
    stratify=patient_aki_status['patient_ever_aki']
)


# df[ df['subject_id'].isin(train_subjects) ] uses .isin() to filter rows
# whose subject_id appears in the training patient list.
# .copy() creates an independent copy of the filtered DataFrame to avoid chained assignment issues.
train_df = df[df['subject_id'].isin(train_subjects)].copy()
test_df = df[df['subject_id'].isin(test_subjects)].copy()


# len(train_df) returns the number of admissions in the training set.
# train_df['subject_id'].nunique() counts unique patients.
print(f"[INFO] Train set: {len(train_df):,} admissions from {train_df['subject_id'].nunique():,} patients")
print(f"[INFO] Test set:  {len(test_df):,} admissions from {test_df['subject_id'].nunique():,} patients")

# I again use .mean() on the binary AKI column to calculate AKI prevalence
# separately for the training and test sets.
print(f"[INFO] Train AKI prevalence: {train_df['aki_binary'].mean()*100:.1f}%")
print(f"[INFO] Test AKI prevalence:  {test_df['aki_binary'].mean()*100:.1f}%")


# set(train_df['subject_id']) and set(test_df['subject_id']) create Python sets
# of patient IDs in train and test. The & operator computes the intersection.
# If the intersection size is zero, it confirms no patient appears in both sets,
# reducing risk of data leakage.
overlap = set(train_df['subject_id']) & set(test_df['subject_id'])
print(f"[VERIFY] Patient overlap between train/test: {len(overlap)} (should be 0)")



# 4. FEATURE SELECTION 
# I list columns that should be excluded from modelling:
# IDs, target-related columns, date/time fields, and categorical variables
# that I have not yet encoded.
exclude_cols = [
    'subject_id', 'hadm_id', 'icd_code', 'icd_version', 'kdigo_stage',
    'aki_binary', 'icu_intime', 'dod', 'gender'
]


# I build feature_cols using a list comprehension:
# - I iterate over df.columns.
# - I keep only columns not in exclude_cols.
# - I also ensure they are numeric types (int64, float64, int32) via df[c].dtype.
feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ['int64', 'float64', 'int32']]

print(f"\n[INFO] Using {len(feature_cols)} features for modeling.")


# I now separate my features and target into training and testing sets.
# train_df[feature_cols] uses DataFrame column selection to build X_train.
X_train = train_df[feature_cols]
# train_df['aki_binary'] selects the target column as y_train.
y_train = train_df['aki_binary']
# Similarly for X_test and y_test from the test_df.
X_test = test_df[feature_cols]
y_test = test_df['aki_binary']



# 5. FEATURE SCALING 

# I create an instance of StandardScaler().
# This object will learn scaling parameters from the training data.
scaler = StandardScaler()

# scaler.fit_transform(X_train) first calls fit() to learn the mean and standard deviation
# of each feature on the training data, then calls transform() to apply scaling.
X_train_scaled = scaler.fit_transform(X_train)

# scaler.transform(X_test) applies the previously learned scaling parameters to the test set
# without refitting, which is critical to avoid leakage.
X_test_scaled = scaler.transform(X_test)



# 6. MODEL TRAINING 

print("\n[INFO] Training baseline Logistic Regression...")


# I instantiate the LogisticRegression model with:
# - class_weight='balanced' to automatically adjust weights for minority/majority classes.
# - max_iter=1000 to allow enough iterations for convergence.
# - random_state=42 for reproducibility of the underlying solver.
model = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    random_state=42
)


# model.fit(X_train_scaled, y_train) is the core training call.
# It adjusts model coefficients so that the predicted probabilities align as closely as possible
# with the observed AKI labels in the training data.
model.fit(X_train_scaled, y_train)



# 7. MODEL EVALUATION

# model.predict(X_test_scaled) uses the learned model to assign class labels (0 or 1)
# to each admission in the test set.
y_pred = model.predict(X_test_scaled)

# model.predict_proba(X_test_scaled) returns probabilities for both classes.
# [:, 1] selects the probability of the positive class (AKI = 1),
# which I use for threshold-independent evaluation metrics.
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]


# average_precision_score(y_test, y_pred_proba) calculates AUPRC,
# which is especially informative for imbalanced datasets like AKI.
auprc = average_precision_score(y_test, y_pred_proba)

# roc_auc_score(y_test, y_pred_proba) calculates AUC-ROC,
# measuring how well the model ranks AKI vs non-AKI cases across thresholds.
auc_roc = roc_auc_score(y_test, y_pred_proba)


# Print a visual separator for the result section.
print("\n" + "=" * 60)
print("BASELINE MODEL RESULTS")
print("=" * 60)


# I format the AUPRC and AUC-ROC values to 4 decimal places for readability.
print(f"AUPRC: {auprc:.4f}")  # Focus metric for imbalanced data
print(f"AUC-ROC: {auc_roc:.4f}")  # Overall discrimination ability


# classification_report(y_test, y_pred, target_names=[...]) prints precision, recall,
# F1-score and support for each class ('No AKI' and 'AKI') in a neat text table.
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No AKI', 'AKI']))


# confusion_matrix(y_test, y_pred) returns a 2x2 matrix showing:
# - True negatives, false positives, false negatives, true positives.
# This helps to see specific error types.
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))



# 8. FEATURE IMPORTANCE 

# I build a new DataFrame using pd.DataFrame(), with:
# - 'feature' column listing feature names.
# - 'coefficient' column storing the corresponding logistic regression coefficients
#   from model.coef_[0].
coef_df = pd.DataFrame({
    'feature': feature_cols,
    'coefficient': model.coef_[0]
})


# coef_df.sort_values() sorts the DataFrame by the 'coefficient' column.
# key=abs tells pandas to use the absolute value of the coefficient for sorting,
# so the strongest positive and negative effects appear at the top.
coef_df = coef_df.sort_values('coefficient', key=abs, ascending=False)


print("\nTop 15 features by |coefficient| (standardized):")
# .head(15) selects the top 15 rows; .to_string(index=False) prints them in a readable table
# without the index column.
print(coef_df.head(15).to_string(index=False))



# 9. SAVE MODEL

# joblib.dump(model, path) serializes and writes the trained model object to disk
# so that I can reload it later without retraining.
joblib.dump(model, os.path.join(MODEL_OUTPUT_DIR, "baseline_logistic_regression.joblib"))

# I also save the scaler using joblib.dump(), because any new data must be scaled
# with the same parameters learned from the training set.
joblib.dump(scaler, os.path.join(MODEL_OUTPUT_DIR, "scaler.joblib"))

# coef_df.to_csv(path, index=False) saves the coefficients DataFrame as a CSV file
# for further analysis, reporting, or plotting.
coef_df.to_csv(os.path.join(MODEL_OUTPUT_DIR, "baseline_lr_coefficients.csv"), index=False)


print(f"\n[SUCCESS] Model saved to: {MODEL_OUTPUT_DIR}/baseline_logistic_regression.joblib")
print("=" * 60)