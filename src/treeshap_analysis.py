import os
import joblib
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
import matplotlib.pyplot as plt

from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split


# ============================================================
# PATH CONFIGURATION
# ============================================================

# I define the input dataset, model folder, and output folder here
# so that all file locations are easy to manage in one place.
INPUT_PATH = "data/processed/final_feature_matrix.parquet"
MODEL_DIR = "models"
OUTPUT_DIR = "app_assets/shap"

os.makedirs(OUTPUT_DIR, exist_ok=True)


print("=" * 60)
print("TREESHAP ANALYSIS: RANDOM FOREST vs XGBOOST")
print("=" * 60)


# ============================================================
# 1. LOAD DATA AND RECREATE THE TEST SET
# ============================================================

# I load the same processed dataset used when training the models.
df = pd.read_parquet(INPUT_PATH)
print(f"[INFO] Loaded dataset: {df.shape[0]:,} rows x {df.shape[1]} columns")


# I recreate the binary AKI outcome so that it matches the target
# used in the Logistic Regression, Random Forest, and XGBoost scripts.
df["aki_binary"] = (df["kdigo_stage"] > 0).astype(int)


# I calculate whether each patient ever experienced AKI.
# This allows me to split the data at patient level rather than row level.
patient_aki_status = (
    df.groupby("subject_id")["aki_binary"]
    .max()
    .reset_index()
)

patient_aki_status.columns = [
    "subject_id",
    "patient_ever_aki"
]


# I recreate the same 80/20 patient-level split used during model training.
# Using the same random_state ensures that the same patients are selected
# for the test set.
train_subjects, test_subjects = train_test_split(
    patient_aki_status["subject_id"],
    test_size=0.2,
    random_state=42,
    stratify=patient_aki_status["patient_ever_aki"]
)


# I select only the patients assigned to the test set.
test_df = df[df["subject_id"].isin(test_subjects)].copy()


# I exclude identifiers, target-related columns, date fields,
# and variables that were not used as model features.
exclude_cols = [
    "subject_id",
    "hadm_id",
    "icd_code",
    "icd_version",
    "kdigo_stage",
    "aki_binary",
    "icu_intime",
    "dod",
    "gender"
]


# I select only numeric columns that were used as predictors.
feature_cols = [
    column
    for column in df.columns
    if column not in exclude_cols
    and df[column].dtype in ["int64", "float64", "int32"]
]


# I create the test feature matrix and target variable.
X_test = test_df[feature_cols]
y_test = test_df["aki_binary"]


print(
    f"[INFO] Test set: {len(X_test):,} rows, "
    f"{len(feature_cols)} features"
)


# I check that the test set contains no missing values before running SHAP.
# Missing values can sometimes cause problems during explanation or plotting.
missing_values = X_test.isnull().sum().sum()

if missing_values > 0:
    print(
        f"[WARNING] X_test contains {missing_values:,} missing values."
    )
else:
    print("[VERIFY] No missing values found in X_test.")


# ============================================================
# 2. LOAD TRAINED MODELS
# ============================================================

# I load the trained Random Forest and XGBoost models.
# This allows me to explain the already-trained models without retraining them.
rf_model = joblib.load(
    os.path.join(MODEL_DIR, "random_forest.joblib")
)

xgb_model = joblib.load(
    os.path.join(MODEL_DIR, "xgboost.joblib")
)

print("[INFO] Loaded Random Forest and XGBoost models.")


# I check that the number and names of the selected features
# match the features expected by the saved models.
if hasattr(rf_model, "feature_names_in_"):
    rf_features = list(rf_model.feature_names_in_)

    if rf_features != feature_cols:
        print(
            "[WARNING] Random Forest feature names do not exactly match "
            "the current feature list."
        )

if hasattr(xgb_model, "feature_names_in_"):
    xgb_features = list(xgb_model.feature_names_in_)

    if xgb_features != feature_cols:
        print(
            "[WARNING] XGBoost feature names do not exactly match "
            "the current feature list."
        )


# ============================================================
# 3. CREATE A SAMPLE FOR SHAP ANALYSIS
# ============================================================

# I use a representative sample instead of the full test set.
# This makes the SHAP calculations and plots faster while still
# providing useful global explanations.
SAMPLE_SIZE = min(2000, len(X_test))

X_shap_sample = X_test.sample(
    n=SAMPLE_SIZE,
    random_state=42
)

print(
    f"[INFO] Using a sample of {SAMPLE_SIZE:,} rows "
    "for SHAP computation."
)


# ============================================================
# 4. COMPUTE SHAP VALUES FOR RANDOM FOREST
# ============================================================

print("\n[INFO] Computing TreeSHAP values for Random Forest...")


# I create a TreeExplainer for the Random Forest because SHAP
# supports tree-based models directly.
rf_explainer = shap.TreeExplainer(rf_model)
rf_shap_raw = rf_explainer.shap_values(X_shap_sample)


# Depending on the installed SHAP version, Random Forest SHAP values
# may be returned as:
# - a list containing one array per class, or
# - a three-dimensional array containing both classes.
# I select the AKI class, which is class 1.
if isinstance(rf_shap_raw, list):
    rf_shap_values = rf_shap_raw[1]

elif isinstance(rf_shap_raw, np.ndarray) and rf_shap_raw.ndim == 3:
    rf_shap_values = rf_shap_raw[:, :, 1]

else:
    rf_shap_values = rf_shap_raw


print(
    f"[INFO] Random Forest SHAP shape: "
    f"{rf_shap_values.shape}"
)

print("[INFO] Random Forest TreeSHAP values computed.")


# ============================================================
# 5. COMPUTE SHAP VALUES FOR XGBOOST
# ============================================================

print("\n[INFO] Computing native TreeSHAP values for XGBoost...")


# I access the underlying XGBoost booster.
# I use XGBoost's native contribution method because the installed
# SHAP version cannot directly read the base_score format in this model.
xgb_booster = xgb_model.get_booster()


# I convert the SHAP sample into an XGBoost DMatrix.
# Feature names are supplied to maintain the correct feature order.
xgb_dmatrix = xgb.DMatrix(
    X_shap_sample,
    feature_names=list(X_shap_sample.columns)
)


# pred_contribs=True returns one contribution for each feature.
# The final column contains the base value for each observation.
xgb_contributions = xgb_booster.predict(
    xgb_dmatrix,
    pred_contribs=True
)


# I separate the feature contributions from the base values.
xgb_shap_values = xgb_contributions[:, :-1]
xgb_base_values = xgb_contributions[:, -1]


print(
    f"[INFO] XGBoost SHAP shape: "
    f"{xgb_shap_values.shape}"
)

print("[INFO] XGBoost native SHAP values computed.")


# I verify that both models produced one SHAP value per feature.
if rf_shap_values.shape[1] != len(feature_cols):
    raise ValueError(
        "Random Forest SHAP values do not match the number of features."
    )

if xgb_shap_values.shape[1] != len(feature_cols):
    raise ValueError(
        "XGBoost SHAP values do not match the number of features."
    )


# ============================================================
# 6. GLOBAL SHAP FEATURE IMPORTANCE
# ============================================================

# I calculate the average absolute SHAP value for each feature.
# A larger value means that the feature had a stronger influence
# on the model's predictions overall.
rf_global_importance = np.abs(rf_shap_values).mean(axis=0)
xgb_global_importance = np.abs(xgb_shap_values).mean(axis=0)


# I combine the importance values from both models into one table.
shap_importance_df = pd.DataFrame({
    "feature": feature_cols,
    "rf_shap_importance": rf_global_importance,
    "xgb_shap_importance": xgb_global_importance
})


# I sort the table according to Random Forest importance.
shap_importance_df = shap_importance_df.sort_values(
    "rf_shap_importance",
    ascending=False
)


# I save the global SHAP importance values for later analysis
# and use in the dissertation.
shap_importance_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "shap_global_importance.csv"
    ),
    index=False
)


print(
    f"\n[INFO] Global SHAP importance saved to: "
    f"{OUTPUT_DIR}/shap_global_importance.csv"
)


# ============================================================
# 7. CROSS-MODEL FEATURE STABILITY
# ============================================================

# I compare the feature rankings from Random Forest and XGBoost.
# Spearman correlation measures whether both models rank important
# features in a similar order.
print(
    "\n[INFO] Computing Spearman rank correlation between "
    "Random Forest and XGBoost SHAP importances..."
)


rho, p_value = spearmanr(
    shap_importance_df["rf_shap_importance"],
    shap_importance_df["xgb_shap_importance"]
)


print("\n" + "=" * 60)
print("CROSS-ARCHITECTURE STABILITY RESULT")
print("=" * 60)
print(f"Spearman's rho (RF vs XGBoost): {rho:.4f}")
print(f"P-value: {p_value:.6f}")


if rho > 0.70:
    print(
        "Target achieved: rho > 0.70. "
        "The explanations are relatively stable across models."
    )
else:
    print(
        "Target not met: rho <= 0.70. "
        "The models show some differences in their feature rankings."
    )

print("=" * 60)


# I save the stability result so that it can be reported later.
stability_result = pd.DataFrame({
    "comparison": ["Random Forest vs XGBoost"],
    "spearman_rho": [rho],
    "p_value": [p_value],
    "target_met": [rho > 0.70]
})


stability_result.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "cross_architecture_stability.csv"
    ),
    index=False
)


# ============================================================
# 8. SHAP SUMMARY PLOTS
# ============================================================

print("\n[INFO] Generating SHAP summary plots...")


# I create a beeswarm plot showing how the most important features
# influence Random Forest predictions.
plt.figure()

shap.summary_plot(
    rf_shap_values,
    X_shap_sample,
    show=False,
    max_display=15
)

plt.title("SHAP Summary — Random Forest")
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "rf_shap_summary.png"
    ),
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# I create the equivalent SHAP summary plot for XGBoost.
plt.figure()

shap.summary_plot(
    xgb_shap_values,
    X_shap_sample,
    show=False,
    max_display=15
)

plt.title("SHAP Summary — XGBoost")
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "xgb_shap_summary.png"
    ),
    dpi=150,
    bbox_inches="tight"
)

plt.close()


print(
    f"[INFO] Summary plots saved to "
    f"{OUTPUT_DIR}/rf_shap_summary.png and "
    f"{OUTPUT_DIR}/xgb_shap_summary.png"
)


# ============================================================
# 9. SELECT A HIGH-RISK SAMPLE
# ============================================================

# I calculate the XGBoost AKI probability for every observation
# in the SHAP sample.
xgb_probs_sample = xgb_model.predict_proba(
    X_shap_sample
)[:, 1]


# I select the observation with the highest predicted AKI probability.
high_risk_idx = np.argmax(xgb_probs_sample)

high_risk_probability = xgb_probs_sample[high_risk_idx]


print(
    "\n[INFO] Generating waterfall plots for a high-risk sample "
    f"(predicted AKI probability: {high_risk_probability:.3f})..."
)


# ============================================================
# 10. RANDOM FOREST WATERFALL PLOT
# ============================================================

# I obtain the Random Forest base value for the AKI class.
rf_base_value = rf_explainer.expected_value

if isinstance(rf_base_value, (list, np.ndarray)):
    rf_base_value = rf_base_value[1]


# I create a SHAP Explanation object for the selected high-risk observation.
rf_explanation = shap.Explanation(
    values=rf_shap_values[high_risk_idx],
    base_values=rf_base_value,
    data=X_shap_sample.iloc[high_risk_idx].values,
    feature_names=feature_cols
)


plt.figure()

shap.waterfall_plot(
    rf_explanation,
    max_display=15,
    show=False
)

plt.title(
    "SHAP Waterfall — Random Forest "
    "(Sample High-Risk Patient)"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "rf_shap_waterfall.png"
    ),
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 11. XGBOOST WATERFALL PLOT
# ============================================================

# I use the native XGBoost contributions and the base value
# corresponding to the selected high-risk observation.
#
# These contributions are on the raw model-margin scale rather
# than directly on the probability scale.
xgb_explanation = shap.Explanation(
    values=xgb_shap_values[high_risk_idx],
    base_values=xgb_base_values[high_risk_idx],
    data=X_shap_sample.iloc[high_risk_idx].values,
    feature_names=feature_cols
)


plt.figure()

shap.waterfall_plot(
    xgb_explanation,
    max_display=15,
    show=False
)

plt.title(
    "SHAP Waterfall — XGBoost "
    "(Sample High-Risk Patient)"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "xgb_shap_waterfall.png"
    ),
    dpi=150,
    bbox_inches="tight"
)

plt.close()


print(
    f"[INFO] Waterfall plots saved to "
    f"{OUTPUT_DIR}/rf_shap_waterfall.png and "
    f"{OUTPUT_DIR}/xgb_shap_waterfall.png"
)


# ============================================================
# 12. FEATURE RANKING COMPARISON
# ============================================================

# I create separate rankings for both models so I can compare
# which features are considered important by each architecture.
comparison_table = shap_importance_df.copy()

comparison_table["rf_rank"] = (
    comparison_table["rf_shap_importance"]
    .rank(ascending=False, method="min")
    .astype(int)
)

comparison_table["xgb_rank"] = (
    comparison_table["xgb_shap_importance"]
    .rank(ascending=False, method="min")
    .astype(int)
)


# I sort the table according to the Random Forest ranking.
comparison_table = comparison_table.sort_values(
    "rf_rank",
    ascending=True
)


print(
    "\nTop 15 features — Random Forest SHAP ranking "
    "(with XGBoost rank for comparison):"
)

print(
    comparison_table.head(15).to_string(index=False)
)


# I save the ranking comparison for reporting and further analysis.
comparison_table.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "shap_rank_comparison.csv"
    ),
    index=False
)


# ============================================================
# 13. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("[SUCCESS] TreeSHAP analysis complete.")
print(f"All outputs saved to: {OUTPUT_DIR}/")
print("=" * 60)