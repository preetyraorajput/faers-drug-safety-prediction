"""
AE Seriousness Console — real deployment app for the FAERS XGBoost model.

Run with:  streamlit run app.py
Needs, in the same folder: best_xgb_tuned_model.pkl, deployment_bundle.pkl
(both produced by the final export cell in your notebook — the version trained
WITHOUT reporting_delay_days, since that field isn't knowable at real triage time)
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="AE Seriousness Console", page_icon=":stethoscope:", layout="centered")

# ---------------------------------------------------------------------------
# Styling — clean clinical look, safe to screenshot/share on GitHub
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');

html, body, [class*="css"]  {
    font-family: 'IBM Plex Sans', -apple-system, sans-serif;
}

.stApp {
    background-color: #eef2f5;
}

h1 {
    color: #16232e;
    font-weight: 700;
    letter-spacing: -0.02em;
}

[data-testid="stCaptionContainer"] {
    color: #5b6b78;
}

/* Card look for the form */
[data-testid="stForm"] {
    background-color: #ffffff;
    border: 1px solid #dde3e8;
    border-radius: 14px;
    padding: 28px 28px 12px 28px;
    box-shadow: 0 1px 3px rgba(16, 24, 32, 0.06);
}

label {
    font-weight: 600 !important;
    color: #2c3a47 !important;
}

.stButton>button, [data-testid="stFormSubmitButton"]>button {
    background-color: #1d6fa5;
    color: #ffffff;
    border: none;
    border-radius: 8px;
    padding: 0.55em 1.6em;
    font-weight: 600;
    transition: background-color 0.15s ease;
}
.stButton>button:hover, [data-testid="stFormSubmitButton"]>button:hover {
    background-color: #155a87;
    color: #ffffff;
}

/* Result cards */
div[data-testid="stAlertContentError"], div[data-testid="stAlertContentSuccess"] {
    font-size: 1.05rem;
    font-weight: 600;
}

hr {
    border-color: #dde3e8;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_and_bundle():
    model = joblib.load("best_xgb_tuned_model.pkl")
    bundle = joblib.load("deployment_bundle.pkl")
    return model, bundle


# Decision threshold — the probability above which a case is flagged "Likely Serious".
# Set to 0.4 rather than the default 0.5: this gave the best F1 balance on your test
# set (0.820) while still catching ~89.5% of truly serious cases, with meaningfully
# fewer false alarms than a more aggressive 0.3 threshold.
DECISION_THRESHOLD = 0.4

# Reporter occupation codes, spelled out for anyone unfamiliar with FAERS abbreviations
OCCUPATION_LABELS = {
    "MD": "MD — Physician",
    "PH": "PH — Pharmacist",
    "OT": "OT — Other health professional",
    "CN": "CN — Consumer / non-health-professional",
    "LW": "LW — Lawyer",
    "HP": "HP — Health professional (unspecified)",
    "RN": "RN — Registered nurse",
    "Unknown": "Unknown",
}


def derive_age_group_encoded(age_years):
    """Same bucket logic as the training notebook's Step 7d / 11d."""
    if age_years < (28 / 365):
        code = "N"
    elif age_years < 2:
        code = "I"
    elif age_years < 12:
        code = "C"
    elif age_years < 18:
        code = "T"
    elif age_years < 65:
        code = "A"
    else:
        code = "E"
    order = ["N", "I", "C", "T", "A", "E"]
    return order.index(code)


def build_feature_vector(bundle, age_years, sex, drug, reporter, n_reactions, n_concomitant):
    """
    Note: no reporting-delay input here on purpose. This model is trained to be used
    at triage time, before a case has been submitted to FDA — at that point, how long
    it will eventually take to reach FDA isn't knowable yet, so that feature was
    removed from training entirely rather than approximated or guessed here.
    """
    feature_columns = bundle["feature_columns"]
    medians = bundle["medians"]

    # Start from training medians for every column — this is what silently
    # fills the fields the user was never asked about.
    row = pd.Series(medians, index=feature_columns, dtype="float64")

    # Overwrite with what the user actually told us
    row["age_years"] = age_years
    row["age_years_was_missing"] = 0
    row["age_group_encoded"] = derive_age_group_encoded(age_years)

    if "n_reactions_log" in row.index:
        row["n_reactions_log"] = np.log1p(max(n_reactions, 0))
    if "n_concomitant_drugs_log" in row.index:
        row["n_concomitant_drugs_log"] = np.log1p(max(n_concomitant, 0))

    # One-hot fields: zero out the whole group, then set the selected category
    for col in feature_columns:
        if col.startswith("sex_"):
            row[col] = 0.0
        if col.startswith("occp_cod_"):
            row[col] = 0.0
        if col.startswith("primary_suspect_drugs_bucketed_"):
            row[col] = 0.0

    sex_col = f"sex_{sex}"
    if sex_col in row.index:
        row[sex_col] = 1.0

    occp_col = f"occp_cod_{reporter}"
    if occp_col in row.index:
        row[occp_col] = 1.0

    drug_col = f"primary_suspect_drugs_bucketed_{drug}"
    if drug_col not in row.index:
        drug_col = "primary_suspect_drugs_bucketed_Other"
    if drug_col in row.index:
        row[drug_col] = 1.0

    return row.reindex(feature_columns).to_frame().T


def main():
    model, bundle = load_model_and_bundle()

    st.title("AE Seriousness Console")
    st.caption("FAERS case triage — predicts Serious vs. Non-Serious from report fields, at the "
               "moment a case is received, before it's submitted to FDA. Model: XGBoost (tuned). "
               "Illustrative tool, not a substitute for clinical review.")

    with st.form("case_form"):
        col1, col2 = st.columns(2)
        with col1:
            age_years = st.number_input("Patient age (years)", min_value=0, max_value=110, value=58)
        with col2:
            sex = st.selectbox("Sex", options=bundle["sex_categories"])

        drug = st.selectbox("Primary suspect drug", options=sorted(bundle["drug_categories"]))

        reporter = st.selectbox(
            "Reporter occupation",
            options=bundle["occp_categories"],
            format_func=lambda code: OCCUPATION_LABELS.get(code, code),
            help="Who is submitting this report — a physician, pharmacist, nurse, the "
                 "patient/consumer themselves, or another party."
        )

        n_reactions = st.slider("Number of reactions reported", 1, 10, 3,
                                 help="How many distinct adverse reactions were listed on this report.")
        n_concomitant = st.slider("Concomitant drugs", 0, 12, 2,
                                   help="How many other drugs the patient was taking at the same time, "
                                        "besides the primary suspect drug.")

        submitted = st.form_submit_button("Run prediction")

    if submitted:
        X_row = build_feature_vector(bundle, age_years, sex, drug, reporter, n_reactions, n_concomitant)
        proba = model.predict_proba(X_row)[0, 1]
        is_serious = proba >= DECISION_THRESHOLD

        st.divider()
        if is_serious:
            st.error(f"**Likely Serious** — P(serious) = {proba:.2f}")
        else:
            st.success(f"**Likely Non-Serious** — P(serious) = {proba:.2f}")

        st.progress(float(proba))
        st.caption(f"Decision threshold: {DECISION_THRESHOLD:.2f}")


if __name__ == "__main__":
    main()
