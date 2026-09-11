import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap
import time
import plotly.graph_objects as go
import os


# ============================================================
# PAGE CONFIG
# ============================================================
# wide layout gives the charts/tables more room than Streamlit's default narrow column
st.set_page_config(page_title="AKI Risk Predictor", layout="wide")


# ============================================================
# LOAD MODELS AND DATA
# ============================================================
# Streamlit reruns the whole script on every click, so caching stops the models
# and dataset getting reloaded from disk every single time.
@st.cache_resource
def load_models():
    rf_model = joblib.load("models/random_forest.joblib")
    xgb_model = joblib.load("models/xgboost.joblib")
    lr_model = joblib.load("models/baseline_logistic_regression.joblib")
    scaler = joblib.load("models/scaler.joblib")  # only LR needs this
    return rf_model, xgb_model, lr_model, scaler


@st.cache_data
def load_data():
    df = pd.read_parquet("data/processed/final_feature_matrix.parquet")
    df['aki_binary'] = (df['kdigo_stage'] > 0).astype(int)
    return df


# Separate, smaller model trained only on the 7 fields a doctor can actually
# type in - Clinician tab shouldn't be silently filling in cohort averages.
@st.cache_resource
def load_clinician_model():
    return joblib.load("models/clinician_xgboost.joblib")


# Need the exact fitted TF-IDF vectoriser from training, not a fresh one,
# otherwise the word-to-column mapping won't match.
@st.cache_resource
def load_symptom_model():
    model = joblib.load("models/symptom_xgboost.joblib")
    tfidf = joblib.load("models/symptom_tfidf_vectorizer.joblib")
    vitals_cols = joblib.load("models/symptom_vitals_cols.joblib")
    return model, tfidf, vitals_cols


rf_model, xgb_model, lr_model, scaler = load_models()
df = load_data()
clinician_model = load_clinician_model()
symptom_model, symptom_tfidf, symptom_vitals_cols = load_symptom_model()


# ============================================================
# FEATURE COLUMN ALIGNMENT
# ============================================================
# feature_names_in_ stores the exact columns the model was trained on, in order -
# using this instead of retyping the list avoids a silent mismatch later.
if hasattr(xgb_model, 'feature_names_in_'):
    feature_cols = list(xgb_model.feature_names_in_)
else:
    # fallback for older library versions that don't expose feature_names_in_
    exclude_cols = ['subject_id', 'hadm_id', 'icd_code', 'icd_version', 'kdigo_stage',
                    'aki_binary', 'icu_intime', 'dod', 'gender']
    feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ['int64', 'float64', 'int32']]


# ============================================================
# PLOTLY RISK GAUGE METER (shared helper used by all three tabs)
# ============================================================
def render_gauge_meter(risk_proba):
    risk_pct = risk_proba * 100

    # same 0.3/0.6 cut-points reused for the colour, label, and shaded bands below
    if risk_proba < 0.3:
        bar_color = "#2ecc71"       # green
        risk_label = "LOW RISK"
    elif risk_proba < 0.6:
        bar_color = "#f39c12"       # amber
        risk_label = "MODERATE RISK"
    else:
        bar_color = "#e74c3c"       # red
        risk_label = "HIGH RISK"

    # Plotly's built-in gauge/dial type - draws the dial and the number together
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_pct,
        number={'suffix': "%", 'font': {'size': 32, 'weight': 'bold', 'color': bar_color}},
        title={'text': f"<b>{risk_label}</b>", 'font': {'size': 20, 'color': bar_color}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "gray"},
            'bar': {'color': bar_color, 'thickness': 0.35},
            'bgcolor': "white",
            'borderwidth': 1,
            'bordercolor': "#e0e0e0",
            # faint shaded zones on the dial face, same 30/60 thresholds as above
            'steps': [
                {'range': [0, 30], 'color': 'rgba(46, 204, 113, 0.20)'},
                {'range': [30, 60], 'color': 'rgba(243, 156, 18, 0.20)'},
                {'range': [60, 100], 'color': 'rgba(231, 76, 60, 0.20)'}
            ],
        }
    ))

    # transparent background so it blends into the page instead of showing a white box
    fig.update_layout(
        height=260,
        margin=dict(l=30, r=30, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'family': "Arial"}
    )
    return fig


# ============================================================
# ROLE SELECTOR - substitute for a proper login system
# ============================================================
# whatever is picked here controls which if/elif/else branch runs below,
# so this basically behaves like three separate pages in one script
st.sidebar.title("AKI Risk Predictor")
role = st.sidebar.radio("I am a:", ["Researcher", "Clinician", "Daily User"])


# ============================================================
# TAB 1: RESEARCH VIEW
# ============================================================
if role == "Researcher":
    st.title("Clinical Research View")
    st.caption("Cohort-level model performance and patient-level SHAP explanations.")

    # rebuild the same split used in training so this is evaluated on the real held-out set
    @st.cache_data
    def get_test_set():
        from sklearn.model_selection import train_test_split
        # split by patient (subject_id), not admission - some patients have multiple
        # ICU stays, and splitting by admission would leak the same patient into both sets
        patient_aki_status = df.groupby('subject_id')['aki_binary'].max().reset_index()
        patient_aki_status.columns = ['subject_id', 'patient_ever_aki']
        train_subjects, test_subjects = train_test_split(
            patient_aki_status['subject_id'],
            test_size=0.2, random_state=42,
            # Stratifying on whether the patient ever had AKI keeps the AKI prevalence roughly
            # equal between the training and test partitions.
            stratify=patient_aki_status['patient_ever_aki']
        )
        test_df = df[df['subject_id'].isin(test_subjects)].copy()
        X_test = test_df[feature_cols]
        y_test = test_df['aki_binary']
        return X_test, y_test

    X_test, y_test = get_test_set()

    # ---------------- MODEL COMPARISON TABLE ----------------
    st.subheader("Model Comparison")

    # model_comparison.csv holds each model's single point-estimate metrics, calculated once
    # during training and saved to disk, rather than recalculated live on every page load.
    comparison_df = pd.read_csv("models/model_comparison.csv")
    # bootstrap_confidence_intervals.csv holds the 95% confidence interval bounds calculated
    # separately via 1,000-iteration bootstrap resampling (see the notebook for that process).
    ci_df = pd.read_csv("models/bootstrap_confidence_intervals.csv")

    # Merge the two files together on the model name so each row has both the point estimate and
    # its associated confidence interval in one place.
    display_df = comparison_df.merge(
        ci_df[['model', 'auprc_ci_low', 'auprc_ci_high', 'auc_roc_ci_low', 'auc_roc_ci_high']],
        on='model'
    )
    # Build combined "point estimate [lower, upper]" strings so the table communicates both the
    # single best-guess number and how much uncertainty surrounds it, in one readable cell.
    display_df['AUPRC (95% CI)'] = display_df.apply(
        lambda r: f"{r['auprc']:.4f}  [{r['auprc_ci_low']:.3f}, {r['auprc_ci_high']:.3f}]", axis=1
    )
    display_df['AUC-ROC (95% CI)'] = display_df.apply(
        lambda r: f"{r['auc_roc']:.4f}  [{r['auc_roc_ci_low']:.3f}, {r['auc_roc_ci_high']:.3f}]", axis=1
    )

    # Select and rename only the columns needed for display, then centre-align the numeric
    # columns using pandas' Styler so the table is easier to visually scan.
    styled_df = display_df[['model', 'AUPRC (95% CI)', 'AUC-ROC (95% CI)', 'precision_aki', 'recall_aki']].rename(
        columns={'model': 'Model', 'precision_aki': 'Precision (AKI)', 'recall_aki': 'Recall (AKI)'}
    ).style.set_properties(**{'text-align': 'center'}).set_table_styles([
        {'selector': 'th', 'props': [('text-align', 'center')]}
    ])
    st.dataframe(styled_df, use_container_width=True)

    # flagging the overlap explicitly - XGBoost isn't proven statistically better here,
    # just has the higher point estimate
    st.caption(
        "Confidence intervals obtained via 1,000-iteration bootstrap resampling of the held-out "
        "test set. Note that XGBoost's and Random Forest's AUPRC intervals overlap, meaning the "
        "two cannot be confidently distinguished on this metric alone at the 95% level."
    )

    # ---------------- BOOTSTRAP CI ERROR-BAR CHART ----------------
    # kept collapsed so it doesn't push the page down for anyone who's happy with the table
    with st.expander("View Bootstrap Confidence Interval Chart", expanded=False):
        fig_ci, axes_ci = plt.subplots(1, 2, figsize=(11, 4.5))
        colors_ci = ['#4C72B0', '#55A868', '#8172B2']

        # yerr wants the distance from the bar top to each bound, not the raw CI values
        axes_ci[0].bar(
            ci_df['model'], ci_df['auprc'],
            yerr=[ci_df['auprc'] - ci_df['auprc_ci_low'], ci_df['auprc_ci_high'] - ci_df['auprc']],
            capsize=8, color=colors_ci, alpha=0.85
        )
        axes_ci[0].set_title('AUPRC (95% CI)')
        axes_ci[0].set_ylabel('AUPRC')
        axes_ci[0].set_ylim(0, 1)

        axes_ci[1].bar(
            ci_df['model'], ci_df['auc_roc'],
            yerr=[ci_df['auc_roc'] - ci_df['auc_roc_ci_low'], ci_df['auc_roc_ci_high'] - ci_df['auc_roc']],
            capsize=8, color=colors_ci, alpha=0.85
        )
        axes_ci[1].set_title('AUC-ROC (95% CI)')
        axes_ci[1].set_ylabel('AUC-ROC')
        axes_ci[1].set_ylim(0, 1)

        plt.tight_layout()
        st.pyplot(fig_ci)
        plt.close(fig_ci)  # closing the figure frees memory; without this, repeated reruns would
                            # slowly accumulate unused figure objects.

        # ---------------- PLAIN MODEL COMPARISON TABLE (point estimates only) ----------------
        styled_df = comparison_df.style.set_properties(
            subset=['auprc', 'auc_roc', 'precision_aki', 'recall_aki'],
            **{'text-align': 'center'}
        ).set_table_styles([
            {'selector': 'th', 'props': [('text-align', 'center')]}
        ]).format({
            'auprc': '{:.4f}', 'auc_roc': '{:.4f}',
            'precision_aki': '{:.4f}', 'recall_aki': '{:.4f}'
        })
        st.dataframe(styled_df, use_container_width=True)

        # ---------------- BENCHMARK CONTEXT ----------------
        # compares my results against Tomašev et al. (2019) so the numbers have context
        st.caption("""
        Benchmark context: Tomašev et al. (2019) achieved a high discrimination performance (AUC-ROC = 0.93) using a deep recurrent architecture trained on a multi-center longitudinal dataset for general AKI detection. In contrast, our XGBoost baseline yielded an AUC-ROC of 0.85 within a single-center cohort restricted exclusively to Type 2 Diabetes Mellitus (T2DM) patients.
        Direct performance comparisons are limited due to key methodological differences: Tomašev et al. leveraged continuous sequential EHR streams across a broad inpatient population, whereas this study evaluates cross-sectional tabular features in a high-risk subgroup with pre-existing metabolic and microvascular vulnerability. Rather than competing with general-population deep learning models, our findings establish a specialized, interpretable benchmark tailored specifically to T2DM-associated AKI risk, laying the groundwork for targeted clinical decision support.
        """)

        # ---------------- OVERALL MODEL COMPARISON BAR CHART ----------------
        with st.expander("View Model Comparison Chart", expanded=False):
            metrics = ['auprc', 'auc_roc', 'precision_aki', 'recall_aki']
            metric_labels = ['AUPRC', 'AUC-ROC', 'Precision (AKI)', 'Recall (AKI)']
            colors_map = {'Logistic Regression': '#4C72B0', 'Random Forest': '#55A868', 'XGBoost': '#8172B2'}

            fig_comp, ax_comp = plt.subplots(figsize=(9, 5))
            x = np.arange(len(metrics))
            width = 0.25
            # Plot one set of grouped bars per model, offsetting each model's bars by "width" so
            # three bars sit side-by-side under each metric label instead of overlapping.
            for i, model_name in enumerate(comparison_df['model']):
                values = comparison_df[comparison_df['model'] == model_name][metrics].values[0]
                ax_comp.bar(x + i*width, values, width, label=model_name,
                            color=colors_map.get(model_name, '#888888'))
            ax_comp.set_xticks(x + width)  # centre the tick labels under the middle bar of each group
            ax_comp.set_xticklabels(metric_labels)
            ax_comp.set_ylim(0, 1)
            ax_comp.set_ylabel('Score')
            ax_comp.set_title('Model Comparison: LR vs Random Forest vs XGBoost')
            ax_comp.legend()
            plt.tight_layout()
            st.pyplot(fig_comp)
            plt.close(fig_comp)

            # ---------------- ABBREVIATION GLOSSARY ----------------
            # added after supervisor feedback - not everyone will know sbp/dbp/map etc.
            with st.expander("What do these abbreviations mean?"):
                st.markdown("""
                | Abbreviation | Meaning |
                |---|---|
                | sbp | Systolic Blood Pressure |
                | dbp | Diastolic Blood Pressure |
                | map | Mean Arterial Pressure |
                | hr | Heart Rate |
                | rr | Respiratory Rate |
                | spo2 | Peripheral Oxygen Saturation |
                | temp | Temperature |
                | AUPRC | Area Under the Precision-Recall Curve |
                | AUC-ROC | Area Under the Receiver Operating Characteristic Curve |
                | _mean | Average value over the first 24 hours |
                | _std | Standard deviation (variability) |
                | _min / _max | Minimum / Maximum value in 24h |
                | _count | Number of times the value was measured |
                | _slope | Rate of change over time |
                | _was_missing | Flag indicating this value was originally missing and later imputed |
                """)

    # ---------------- CLICKABLE MODEL SELECTOR BUTTONS ----------------
    # stores the chosen model in session_state so it survives the rerun on every click
    st.write("**Click a model to see its detailed results:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Logistic Regression", key="btn_lr", use_container_width=True):
            st.session_state['selected_model'] = 'Logistic Regression'
    with col2:
        if st.button("Random Forest", key="btn_rf", use_container_width=True):
            st.session_state['selected_model'] = 'Random Forest'
    with col3:
        if st.button("XGBoost", key="btn_xgb", use_container_width=True):
            st.session_state['selected_model'] = 'XGBoost'

    # A "Hide Results" button only appears once a model has actually been selected. It deletes
    # the session_state key and calls st.rerun() to immediately refresh the page without it.
    if 'selected_model' in st.session_state:
        if st.button("Hide Results", key="btn_hide"):
            del st.session_state['selected_model']
            st.rerun()

    # This whole block only runs if a model is currently selected in session_state.
    if 'selected_model' in st.session_state:
        selected = st.session_state['selected_model']
        st.markdown(f"### {selected} — Detailed Results")

        # only LR needs scaling - it was trained on standardised features,
        # the tree models split on raw values directly
        if selected == 'Logistic Regression':
            X_test_scaled = scaler.transform(X_test)
            y_pred = lr_model.predict(X_test_scaled)
            y_pred_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
        elif selected == 'Random Forest':
            y_pred = rf_model.predict(X_test)
            y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
        else:
            y_pred = xgb_model.predict(X_test)
            y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]

        from sklearn.metrics import (
            average_precision_score, roc_auc_score, roc_curve,
            precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay
        )

        # AUPRC is emphasised throughout this dissertation because AKI is the minority class
        # (~24% prevalence); AUPRC is more informative than accuracy in this kind of imbalance.
        auprc = average_precision_score(y_test, y_pred_proba)
        auc_roc = roc_auc_score(y_test, y_pred_proba)

        m_col1, m_col2 = st.columns(2)
        m_col1.metric("AUPRC", f"{auprc:.4f}")
        m_col2.metric("AUC-ROC", f"{auc_roc:.4f}")

        st.write("**Performance Graphs**")
        g_col1, g_col2, g_col3 = st.columns(3)

        # Confusion matrix: shows the raw counts of correct/incorrect predictions per class.
        with g_col1:
            fig1, ax1 = plt.subplots(figsize=(4, 4))
            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No AKI', 'AKI'])
            disp.plot(ax=ax1, cmap='Blues', values_format='d', colorbar=False)
            ax1.set_title("Confusion Matrix")
            st.pyplot(fig1)
            plt.close(fig1)

        # ROC curve: plots true-positive rate against false-positive rate across every possible
        # decision threshold, summarised by the AUC-ROC number shown in the legend.
        with g_col2:
            fig2, ax2 = plt.subplots(figsize=(4, 4))
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {auc_roc:.3f}')
            # The diagonal dashed line represents a model with no discriminative ability at all
            # (equivalent to random guessing), included as a visual reference point.
            ax2.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
            ax2.set_xlabel('False Positive Rate')
            ax2.set_ylabel('True Positive Rate')
            ax2.set_title("ROC Curve")
            ax2.legend(loc='lower right', fontsize=8)
            st.pyplot(fig2)
            plt.close(fig2)

        # Precision-recall curve: more informative than the ROC curve specifically because AKI is
        # a rare/minority outcome in this dataset.
        with g_col3:
            fig3, ax3 = plt.subplots(figsize=(4, 4))
            precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)
            ax3.plot(recall_vals, precision_vals, color='darkgreen', lw=2, label=f'AUPRC = {auprc:.3f}')
            # dashed line = the no-skill baseline, equal to AKI prevalence in the test set
            ax3.axhline(y=y_test.mean(), color='gray', lw=1, linestyle='--')
            ax3.set_xlabel('Recall')
            ax3.set_ylabel('Precision')
            ax3.set_title("Precision-Recall Curve")
            ax3.legend(loc='upper right', fontsize=8)
            st.pyplot(fig3)
            plt.close(fig3)

        # SHAP images are only available for the two tree-based models, since these were the two
        # architectures actually compared in the cross-architecture SHAP stability analysis.
        if selected in ['Random Forest', 'XGBoost']:
            st.write("**SHAP Analysis**")

            shap_img_map = {
                'Random Forest': 'app_assets/shap/rf_shap_summary.png',
                'XGBoost': 'app_assets/shap/xgb_shap_summary.png'
            }
            shap_img_path = shap_img_map[selected]

            # Defensive check: if the SHAP analysis notebook/script hasn't been run yet, show a
            # helpful message instead of letting the app crash on a missing file.
            if os.path.exists(shap_img_path):
                st.image(shap_img_path, caption=f"SHAP Summary — {selected}")
            else:
                st.info("SHAP summary image not found. Run treeshap_analysis.py first.")

            # Display the pre-computed Spearman correlation between the two models' SHAP
            # importance rankings, if that file has been generated.
            stability_path = "app_assets/shap/cross_architecture_stability.csv"
            if os.path.exists(stability_path):
                stability_df = pd.read_csv(stability_path)
                rho = stability_df['spearman_rho'].values[0]
                st.caption(f"Cross-architecture SHAP stability (RF vs XGBoost): Spearman ρ = {rho:.4f}")
        else:
            st.info("SHAP analysis is available for Random Forest and XGBoost only.")

    st.divider()  # a plain horizontal rule to visually separate the model-comparison section
    st.divider()  # from the fairness-audit section below it

    # ---------------- PRELIMINARY FAIRNESS AUDIT ----------------
    st.subheader("Preliminary Fairness Audit")
    st.caption(
        "A full demographic parity and equalised odds audit across age, sex, and ethnicity was "
        "out of scope given project time constraints. A preliminary two-dimension check (age and "
        "sex) is presented below as a partial indicator."
    )

    fairness_df = pd.read_csv("models/fairness_audit_extended.csv")

    # The CSV holds both age-based and sex-based rows together, distinguished by a "dimension"
    # column, so each is filtered out separately here.
    age_results = fairness_df[fairness_df['dimension'] == 'age'].reset_index(drop=True)
    sex_results = fairness_df[fairness_df['dimension'] == 'sex'].reset_index(drop=True)

    # st.tabs() creates a small tabbed sub-section so age and sex results don't have to be shown
    # stacked on top of each other, keeping the page shorter.
    f_tab1, f_tab2 = st.tabs(["Age", "Sex"])

    with f_tab1:
        st.dataframe(
            age_results[['group', 'n', 'positive_prediction_rate', 'recall_sensitivity', 'false_positive_rate']].rename(
                columns={'group': 'Group', 'n': 'N', 'positive_prediction_rate': 'Positive Prediction Rate',
                         'recall_sensitivity': 'Recall (Sensitivity)', 'false_positive_rate': 'False Positive Rate'}
            ),
            use_container_width=True
        )
        # Gaps are reported as simple percentage-point differences (not relative percentages),
        # matching the demographic-parity convention used throughout the dissertation report.
        age_ppr_gap = abs(age_results.loc[0, 'positive_prediction_rate'] - age_results.loc[1, 'positive_prediction_rate']) * 100
        age_recall_gap = abs(age_results.loc[0, 'recall_sensitivity'] - age_results.loc[1, 'recall_sensitivity']) * 100
        st.warning(f"PPR gap: {age_ppr_gap:.1f} percentage points | Recall gap: {age_recall_gap:.1f} percentage points (favouring 65+ group)")

    with f_tab2:
        st.dataframe(
            sex_results[['group', 'n', 'positive_prediction_rate', 'recall_sensitivity', 'false_positive_rate']].rename(
                columns={'group': 'Group', 'n': 'N', 'positive_prediction_rate': 'Positive Prediction Rate',
                         'recall_sensitivity': 'Recall (Sensitivity)', 'false_positive_rate': 'False Positive Rate'}
            ),
            use_container_width=True
        )
        sex_ppr_gap = abs(sex_results.loc[0, 'positive_prediction_rate'] - sex_results.loc[1, 'positive_prediction_rate']) * 100
        sex_recall_gap = abs(sex_results.loc[0, 'recall_sensitivity'] - sex_results.loc[1, 'recall_sensitivity']) * 100
        st.info(f"PPR gap: {sex_ppr_gap:.1f} percentage points | Recall gap: {sex_recall_gap:.1f} percentage points")

    # A collapsed chart comparing all four groups (age <65, age 65+, female, male) side by side
    # on both metrics, so the size of the age gap versus the sex gap can be visually compared.
    with st.expander("View Fairness Comparison Chart (Age vs Sex)", expanded=False):
        fig_f, axes_f = plt.subplots(1, 2, figsize=(11, 4.5))

        groups_all = ['Age <65', 'Age 65+', 'Female', 'Male']
        colors_f = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']

        ppr_values = list(age_results['positive_prediction_rate']) + list(sex_results['positive_prediction_rate'])
        recall_values = list(age_results['recall_sensitivity']) + list(sex_results['recall_sensitivity'])

        axes_f[0].bar(groups_all, ppr_values, color=colors_f)
        axes_f[0].set_title('Positive Prediction Rate by Group')
        axes_f[0].set_ylabel('Rate')
        axes_f[0].set_ylim(0, 0.5)
        # A vertical line visually separates the two age bars from the two sex bars.
        axes_f[0].axvline(x=1.5, color='gray', linestyle='--', linewidth=1)

        axes_f[1].bar(groups_all, recall_values, color=colors_f)
        axes_f[1].set_title('Recall (Sensitivity) by Group')
        axes_f[1].set_ylabel('Recall')
        axes_f[1].set_ylim(0, 1)
        axes_f[1].axvline(x=1.5, color='gray', linestyle='--', linewidth=1)

        plt.tight_layout()
        st.pyplot(fig_f)
        plt.close(fig_f)

    # deliberately not calling this "bias" - could just as easily be genuine clinical
    # risk difference, full discussion is in the report
    st.caption(
        "The age-based gap is substantially larger than the sex-based gap, suggesting the "
        "disparity observed is concentrated in the age dimension rather than reflecting a "
        "general fairness issue across all demographic axes. This gap could reflect genuine "
        "higher AKI incidence in older patients rather than model bias — see the dissertation "
        "report for full discussion."
    )

    # ---------------- PATIENT-LEVEL PREDICTION ----------------
    st.subheader("Select a Patient")
    # subject_id identifies a patient; a single patient can have multiple ICU admissions, so the
    # second dropdown (hadm_id) is needed to pin down one specific admission for that patient.
    subject_ids = df['subject_id'].unique()
    selected_subject = st.selectbox("Subject ID", subject_ids)

    patient_rows = df[df['subject_id'] == selected_subject]
    selected_hadm = st.selectbox("Admission ID (hadm_id)", patient_rows['hadm_id'].unique())

    patient_row = df[df['hadm_id'] == selected_hadm].iloc[0]

    # Wrapping the single patient's row in a DataFrame (rather than passing a Series) preserves
    # the full column schema the model expects, in the same shape as during training.
    df_patient = pd.DataFrame([patient_row[feature_cols]])

    if st.button("Run Risk Assessment", key="research_run"):
        start_time = time.time()  # marks the beginning of the timed prediction+explanation block

        risk_proba = xgb_model.predict_proba(df_patient)[0, 1]

        st.plotly_chart(render_gauge_meter(risk_proba), use_container_width=True)

        st.subheader("Contributing Factors")
        # TreeExplainer is SHAP's fast, exact explainer specifically for tree-based models like
        # XGBoost, as opposed to the slower, approximate explainer needed for other model types.
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer(df_patient)

        fig, ax = plt.subplots(figsize=(8, 5))
        # The waterfall plot automatically sorts this one patient's features by SHAP impact and
        # shows the top contributors, collapsing the rest into a single "other features" bar.
        shap.waterfall_plot(shap_values[0], max_display=10, show=False)
        st.pyplot(fig)
        plt.close()

        # A short plain-English legend explaining how to read the colours and numbers on the
        # chart above, since SHAP output isn't self-explanatory to a non-technical reader.
        st.markdown("""
        **How to read this chart:**
        - 🔴 **Red bars** — this factor **increased** the patient's AKI risk
        - 🔵 **Blue bars** — this factor **decreased** the patient's AKI risk
        - The number on each bar shows how much that factor pushed the risk score up or down
        - Bars are ordered by impact — the biggest contributors are at the top
        - See the abbreviation glossary above if any feature names are unclear
        """)

        elapsed = (time.time() - start_time) * 1000  # convert seconds to milliseconds
        st.caption(f"Prediction latency: {elapsed:.0f} ms")

    # ---------------- COHORT SNAPSHOT ----------------
    st.subheader("Cohort Snapshot")
    st.write(f"Total admissions: {len(df):,}")
    st.write(f"AKI prevalence: {df['aki_binary'].mean()*100:.1f}%")
    st.dataframe(df[['subject_id', 'hadm_id', 'kdigo_stage']].head(10), use_container_width=True)


# ============================================================
# TAB 2: CLINICIAN INPUT VIEW
# ============================================================
elif role == "Clinician":
    st.title("Clinician Input View")
    st.caption("Enter patient values manually to get a risk prediction and recommendations.")
    # earlier version filled gaps with cohort averages - dropped that, this only uses
    # what the clinician actually types in
    st.info("This prediction is based entirely on the values you enter below — no other patient data is assumed or imputed.")

    # Two columns split the eight input fields into a compact, side-by-side layout rather than
    # one long vertical list.
    col1, col2 = st.columns(2)
    with col1:
        patient_name = st.text_input("Patient Name (optional)", "")
        age = st.number_input("Age", min_value=18, max_value=120, value=65)
        baseline_creatinine = st.number_input("Baseline Creatinine (mg/dL)", min_value=0.0, max_value=15.0, value=1.0, step=0.1)
        creatinine_slope = st.number_input("Creatinine Slope 24h", min_value=-5.0, max_value=10.0, value=0.0, step=0.1)

    with col2:
        glucose_mean = st.number_input("Glucose Mean (mg/dL)", min_value=0.0, max_value=600.0, value=150.0)
        glucose_cv = st.number_input("Glucose CV (%)", min_value=0.0, max_value=100.0, value=20.0)
        map_mean = st.number_input("Mean Arterial Pressure", min_value=0.0, max_value=200.0, value=75.0)
        hr_mean = st.number_input("Heart Rate (mean)", min_value=0.0, max_value=250.0, value=85.0)

    if st.button("Predict Risk", key="clinician_predict"):
        # keys have to match the column names clinician_xgboost.joblib was trained on
        input_row = pd.DataFrame([{
            'age_at_admission': age,
            'baseline_creatinine': baseline_creatinine,
            'creatinine_slope_24h': creatinine_slope,
            'glucose_mean': glucose_mean,
            'glucose_cv': glucose_cv,
            'map_mean': map_mean,
            'hr_mean': hr_mean
        }])

        # Note: this uses the separate, smaller clinician_model (7 features), not the main
        # 104-feature xgb_model used in the Research tab.
        risk_proba = clinician_model.predict_proba(input_row)[0, 1]

        st.plotly_chart(render_gauge_meter(risk_proba), use_container_width=True)

        # Recommendation text is manually written for each risk band, informed by general
        # KDIGO-based clinical management principles, rather than generated by the model itself.
        if risk_proba < 0.3:
            recommendations = [
                "Continue routine monitoring.",
                "No immediate action required beyond standard care."
            ]
        elif risk_proba < 0.6:
            recommendations = [
                "Reassess fluid balance and avoid nephrotoxic agents where possible.",
                "Recheck serum creatinine within 12 hours.",
                "Monitor glucose closely given elevated variability."
            ]
        else:
            recommendations = [
                "Urgent renal function reassessment recommended.",
                "Review and hold nephrotoxic medications immediately.",
                "Consider nephrology consult."
            ]

        st.write("**Clinical Recommendations:**")
        for rec in recommendations:
            st.write(f"- {rec}")

        st.subheader("Top Contributing Factors")
        # Build a fresh TreeExplainer around the smaller, 7-feature clinician model, since it is
        # a genuinely different trained model object from the main research XGBoost model.
        explainer = shap.TreeExplainer(clinician_model)
        shap_values = explainer.shap_values(input_row)

        # Sorting by the absolute SHAP value (rather than the raw signed value) ensures the
        # single strongest factor appears first, regardless of whether it pushed risk up or down.
        shap_df = pd.DataFrame({
            'feature': input_row.columns,
            'shap_value': shap_values[0]
        }).sort_values('shap_value', key=abs, ascending=False)

        fig, ax = plt.subplots(figsize=(7, 4))
        # Colour each bar red if it pushed risk up (positive SHAP value) or green if it pushed
        # risk down (negative SHAP value), using a compact list comprehension.
        colors = ['#e74c3c' if v > 0 else '#2ecc71' for v in shap_df['shap_value']]
        ax.barh(shap_df['feature'], shap_df['shap_value'], color=colors)
        ax.set_xlabel("SHAP value (impact on risk)")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("""
        **How to read this chart:**
        - 🔴 **Red bars** — this factor **increased** the patient's AKI risk
        - 🟢 **Green bars** — this factor **decreased** the patient's AKI risk
        """)


# ============================================================
# TAB 3: DAILY USER — SYMPTOM CHECKER (ML-based, NHS-guideline-aligned)
# ============================================================
elif role == "Daily User":
    st.title("Acute Kidney Injury (AKI) Symptom Checker")
    st.caption("Based on NHS guidelines for Acute Kidney Injury awareness.")

    # A short, plain-language explanation of AKI itself, since this tab's audience is assumed to
    # have no clinical background at all, unlike the Research or Clinician tabs.
    st.info(
        "**What is AKI?** Acute kidney injury (AKI) is when something suddenly causes your "
        "kidneys to stop working as well as they should. It can be very serious for some people."
    )

    st.subheader("Are you experiencing any of these symptoms?")
    # Free-text input lets the model use its trained TF-IDF vocabulary directly on the user's own
    # words, in addition to (or instead of) the structured checkboxes below.
    free_text = st.text_area(
        "Describe how you're feeling (optional):",
        placeholder="e.g. I've been peeing a lot less than usual and feel very thirsty..."
    )

    st.write("Or select what applies:")
    col1, col2 = st.columns(2)
    with col1:
        symptom_pee = st.checkbox("Peeing a lot less than usual")
        symptom_thirst = st.checkbox("Feeling very thirsty")
        symptom_nausea = st.checkbox("Feeling sick (nausea) or being sick (vomiting)")
    with col2:
        symptom_swelling = st.checkbox("Swollen feet or legs")
        symptom_fatigue = st.checkbox("Feeling tired, dizzy, confused or sleepy")
        symptom_breathless = st.checkbox("Feeling breathless, especially when lying down")

    # These risk-factor checkboxes are shown for context/awareness (matching NHS guidance on who
    # is more likely to get AKI), but are not currently fed into the ML model itself.
    st.subheader("Do any of these apply to you?")
    st.caption("NHS guidance notes AKI is more common in people with these risk factors.")
    r_col1, r_col2 = st.columns(2)
    with r_col1:
        risk_age = st.checkbox("Aged 65 or older")
        risk_prior_aki = st.checkbox("Have had AKI before")
    with r_col2:
        risk_surgery = st.checkbox("Recently had surgery")
        risk_meds = st.checkbox("Taking ACE inhibitors, NSAIDs, or diuretics")

    if st.button("Check My Risk", key="patient_check"):
        # turning checkboxes into phrases that overlap with the MIMIC-IV-ED vocabulary the
        # TF-IDF vectoriser was trained on, so a checkbox-only submission still means something
        checkbox_symptoms = []
        if symptom_pee: checkbox_symptoms.append("peeing less urine output reduced")
        if symptom_thirst: checkbox_symptoms.append("thirsty")
        if symptom_nausea: checkbox_symptoms.append("nausea vomiting sick")
        if symptom_swelling: checkbox_symptoms.append("swollen legs feet edema")
        if symptom_fatigue: checkbox_symptoms.append("tired dizzy confused sleepy")
        if symptom_breathless: checkbox_symptoms.append("breathless shortness of breath")

        # Combine any free-text description with the checkbox-derived phrases into a single
        # lowercase string, since the vectoriser was trained on lowercase text.
        combined_text = (free_text.strip() + " " + " ".join(checkbox_symptoms)).strip().lower()

        if not combined_text:
            st.warning("Please describe your symptoms or select from the checkboxes above.")
        else:
            # transform(), not fit_transform() - has to map onto the same feature space
            # the model was trained on, not fit a new vocabulary from one sentence
            text_features = symptom_tfidf.transform([combined_text])
            X_input = pd.DataFrame(
                text_features.toarray(),
                columns=[f"symptom_{w}" for w in symptom_tfidf.get_feature_names_out()]
            )
            # reindex() restores the exact column set and order the model expects, filling any
            # engineered keyword-flag columns not already present with 0.
            X_input = X_input.reindex(columns=symptom_model.feature_names_in_, fill_value=0)

            risk_proba = symptom_model.predict_proba(X_input)[0, 1]

            st.plotly_chart(render_gauge_meter(risk_proba), use_container_width=True)

            # the symptom model barely beats random chance (see evaluation), so these two
            # urgent symptoms override its probability regardless of what it predicts
            urgent_flag = symptom_pee or symptom_breathless

            if risk_proba >= 0.6 or urgent_flag:
                st.error("Urgent action recommended")
                st.markdown("""
                Ask for an **urgent GP appointment** or get help from **NHS 111** if you think you may have
                symptoms of acute kidney injury (AKI). You can get help from 111 online or call 111.

                **What you can do:**
                - Contact NHS 111 or an urgent GP appointment.
                - Speak to a doctor before taking any new medicines or supplements, especially anti-inflammatory
                  medicines (NSAIDs) such as ibuprofen.
                - If you feel very breathless or are unable to pee at all, seek emergency care.
                """)
            elif risk_proba >= 0.3:
                st.warning("Speak to a doctor")
                st.markdown("""
                Based on what you've described, it's worth getting checked. Ask for a GP appointment
                or contact NHS 111 for advice.

                **What you can do:**
                - Stay hydrated — drink enough fluids so your pee is pale and regular throughout the day,
                  especially in hot weather, or if you're being sick or have diarrhoea.
                - Speak to a doctor before taking any new medicines or supplements.
                - Make sure you know the symptoms of AKI to watch out for.
                """)
            else:
                st.success("Low risk reported")
                st.markdown("""
                No strong symptoms of acute kidney injury were detected. To lower your chances of AKI,
                especially if you're at higher risk:

                **What you can do:**
                - Stay hydrated — drink enough fluids so your pee is pale and regular throughout the day.
                - Speak to a doctor before taking any new medicines or supplements.
                - Know the symptoms of AKI: peeing a lot less than usual, feeling very thirsty, nausea or
                  vomiting, swollen feet or legs, feeling tired/dizzy/confused, or feeling breathless.
                """)

            # A closing safety/attribution note shown regardless of which risk band was reached.
            st.caption(
                "Information adapted from NHS guidance on acute kidney injury (AKI). "
                "This tool is for awareness only and is not a medical diagnosis. "
                "Always consult a healthcare professional for medical concerns."
            )