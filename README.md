# AKI Risk Prediction in Type 2 Diabetes Patients

MSc dissertation project — an explainable machine learning system for predicting Acute Kidney
Injury (AKI) risk in Type 2 diabetic ICU patients, built on the MIMIC-IV database, with a
3-tab Streamlit app for researchers, clinicians, and patients.

## What this project does

AKI is common in ICU patients and even more common in people with Type 2 diabetes, but most
existing prediction models are either accurate-but-unexplainable, or don't focus on diabetic
patients specifically. This project:

- Builds a diabetes-specific feature set (glycaemic variability, renal trend, vitals) from
  MIMIC-IV
- Trains and compares three models — Logistic Regression, Random Forest, and Optuna-tuned
  XGBoost
- Explains the models with TreeSHAP, and checks whether the explanations agree across the two
  tree-based architectures (Spearman ρ = 0.85)
- Quantifies uncertainty with bootstrap confidence intervals rather than reporting a single
  point estimate
- Runs a preliminary fairness check across age and sex
- Ships a working Streamlit app with three views: a research dashboard, a clinician-facing
  risk tool, and an NHS-guideline-aligned symptom checker for patients

## Project structure

```
├── src/                    all the pipeline scripts (extraction, cleaning, training, evaluation)
├── notebooks/               exploratory notebook, cell by cell, with output explanations
├── models/                  trained model files and saved evaluation results (CSVs, charts)
├── app_assets/shap/         saved SHAP plots used by the app and in the dissertation
├── streamlit_app.py         the actual app (in src/)
└── data/                    NOT included - MIMIC-IV is credentialed data, kept local only
```

## Pipeline order (if running from scratch)

The scripts in `src/` depend on each other's outputs, so they need to run roughly in this order:

1. `extract_diagnoses_icd.py` → cohort extraction (T2DM + ICU-only + ESRD excluded)
2. `extract_kdigo_features.py` → KDIGO labelling + renal/glycaemic features
3. `extract_vitals_features.py` → vitals features (ICU-intime anchored)
4. `extract_patient_cohort.py` → demographics
5. `handle_missing_values.py` → imputation + missing-indicator flags
6. `baseline_model.py`, `train_random_forest.py`, `train_xgBoost.py` → the three models
7. `treeshap_analysis.py` → SHAP explanations + cross-architecture stability check
8. `boostrap.py` → bootstrap confidence intervals
9. `fairness_audit.py` → age/sex fairness check
10. `MIMIC_ED_Extraction.py`, `Feature_Engineering_Symptoms.py`, `Train_Symptoms_model.py`,
    `lr_rf_symptom.py` → the symptom-only model for the Patient tab
11. `clinician.py` → the 7-feature model used by the Clinician tab

## Running the app

```bash
streamlit run src/streamlit_app.py
```

## Data

This project uses MIMIC-IV and MIMIC-ED, which require PhysioNet credentialing and a signed
data use agreement. Raw data is not included in this repo and was never uploaded anywhere -
only the processing code is here. If you have your own credentialed access, the raw CSVs need
to sit in a local `MIMIC-IV Dataset` folder referenced at the top of the extraction scripts.
