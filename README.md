AE Seriousness Console — Predicting Adverse Event Seriousness from FDA FAERS Data

A machine learning model and deployed app that predicts whether a reported drug adverse event is likely to be classified as Serious, using data from the FDA's public FAERS (FDA Adverse Event Reporting System) database.

The goal: help triage a new adverse event report at the moment it's received — before it has even been submitted to the FDA — using only information that's actually knowable at that point.

Try it live: faers-drug-safety-prediction-qzf2uzzskvkryl5qfvwd7q.streamlit.app — no install needed, just open the link.

Live app

Built with Streamlit. A user enters six fields (patient age, sex, primary suspect drug, reporter occupation, number of reactions reported, number of concomitant drugs) and gets back a Serious / Non-Serious prediction with the model's confidence.

To run it yourself locally instead:

pip install -r requirements.txt
streamlit run app.py

Needs best_xgb_tuned_model.pkl and deployment_bundle.pkl in the same folder (both included in this repo).

Model selection

Three models were tuned and compared — not just on a same-period test set, but also on genuinely out-of-time data (real 2026 Q1 reports the models never saw during training). This is the metric that actually matters for a model meant to be deployed on future, unseen data.

Model	Test-set ROC-AUC	Out-of-time (2026) ROC-AUC	Drop, test → future
Random Forest	0.8745	0.8499	0.0246
LightGBM	0.8797 (best on test)	0.8629	0.0168
XGBoost	0.8744	0.8646 (best on future data)	0.0098 (smallest drop)

XGBoost was chosen as the final model — not because it had the top same-period test score (LightGBM did, narrowly), but because its performance held up best on genuinely unseen future data. A model that degrades the least when the world moves on is the one worth deploying.

A deliberate accuracy trade-off, for deployment honesty

While preparing this model for real use, a feature called reporting_delay_days (how long a report took to reach the FDA) was found to be a deployment-validity problem: at the moment this tool would actually be used — during triage, before a report is submitted — that delay simply isn't knowable yet. Keeping it would mean the deployed app depended on information that doesn't exist at prediction time.

The feature was dropped entirely, and the model was retrained without it — accepting a small, honest cost in exchange for a model that only uses information genuinely available at real-world prediction time:

Metric	With delay feature	Without delay feature	Difference
Test Accuracy	0.7902	0.7877	−0.25 pts
Test ROC-AUC	0.8744	0.8722	−0.22 pts
2026 (out-of-time) Accuracy	0.7685	0.7569	−1.16 pts
2026 (out-of-time) ROC-AUC	0.8646	0.8596	−0.50 pts

This is the model shipped in this repo: XGBoost, tuned, 82 features, trained without reporting_delay_days.

Decision threshold

The app flags a case "Likely Serious" at a probability threshold of 0.4, not the default 0.5 — chosen after comparing precision/recall/F1 at 0.3, 0.4, and 0.5 on the test set. 0.4 gives the best F1 score (0.820) while still catching about 89.5% of truly serious cases, with meaningfully fewer false alarms than a more aggressive 0.3 threshold.

Interpretability

Feature importance and SHAP values were cross-referenced against official FDA drug labeling (via DailyMed) for the 28 drugs the models flagged as important predictors — checking whether the model's Serious/Non-Serious signal for each drug lines up with that drug's real FDA boxed-warning status. See the full report for the detailed findings, including cases where the model's signal and a drug's boxed-warning status don't perfectly align (expected — a boxed warning reflects a specific rare risk, not the general severity of most reports for that drug).

Repository contents
File	What it is
FDA 2025.ipynb	Full notebook: data cleaning, EDA, model training/tuning, out-of-time evaluation, SHAP interpretability, and the final deployment export
app.py	The Streamlit deployment app
requirements.txt	Python dependencies
run_app.bat	Windows shortcut to launch the app
best_xgb_tuned_model.pkl	Final trained XGBoost model (no reporting_delay_days)
deployment_bundle.pkl	Feature column list, category options, and training medians used to reconstruct the full feature vector from the app's 6 input fields
Predicting Adverse Event Seriousness A Machine Learning Approach to FDA FAERS Data.docx	Full written project report

Note: the raw FAERS data files (multiple GB) are not included in this repo — they're publicly available from the FDA FAERS Quarterly Data Files.

Disclaimer

This is an educational/illustrative project, not a validated clinical tool. Predictions should never substitute for clinical judgment or official pharmacovigilance review.
