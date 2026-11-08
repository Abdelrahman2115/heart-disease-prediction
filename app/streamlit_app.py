"""Streamlit interface for the trained heart disease model.

Run from the project root:
    streamlit run app/streamlit_app.py

EDUCATIONAL ONLY. This app estimates how a model trained on a small,
historical research dataset would label a record. It is not a diagnostic
tool and must not be used for medical decisions.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heart import config  # noqa: E402
from heart.predict import load_model, predict_one  # noqa: E402

st.set_page_config(page_title="Heart Disease Model (Educational)", page_icon="❤️", layout="centered")

st.title("Heart disease prediction: educational demo")
st.warning(
    "**Not medical advice.** This model reproduces labels from a 1980s research dataset of ~300 "
    "patients referred for cardiac testing. Its output is not a diagnosis and says nothing reliable "
    "about any real person. If you have symptoms, contact a healthcare professional."
)

if not config.MODEL_PATH.exists():
    st.error("No trained model found. Run `python -m heart.train` first.")
    st.stop()

_, meta = load_model()
CODE_NOTE = ("The Kaggle source documents these codes inconsistently, so they are shown as raw codes. "
             "See docs/walkthrough.md for what each code is associated with in this data.")

with st.form("patient"):
    st.subheader("Clinical measurements")
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age (years)", 18, 100, 55)
        sex = st.radio("Sex", options=[1, 0], format_func=lambda v: "Male" if v else "Female", horizontal=True)
        trestbps = st.number_input("Resting blood pressure (mm Hg)", 80, 220, 130)
        chol = st.number_input("Serum cholesterol (mg/dl)", 100, 600, 240)
        fbs = st.radio("Fasting blood sugar > 120 mg/dl", [0, 1], format_func=lambda v: "Yes" if v else "No",
                       horizontal=True)
        thalach = st.number_input("Max heart rate in exercise test", 60, 220, 150)
    with c2:
        exang = st.radio("Exercise-induced angina", [0, 1], format_func=lambda v: "Yes" if v else "No",
                         horizontal=True)
        oldpeak = st.number_input("ST depression (oldpeak)", 0.0, 7.0, 1.0, step=0.1)
        cp = st.selectbox("Chest pain type code (cp)", [0, 1, 2, 3], help=CODE_NOTE)
        restecg = st.selectbox("Resting ECG code (restecg)", [0, 1, 2], help=CODE_NOTE)
        slope = st.selectbox("ST slope code (slope)", [0, 1, 2], help=CODE_NOTE)
        ca = st.selectbox("Major vessels coloured by fluoroscopy (ca)", [0, 1, 2, 3])
        thal = st.selectbox("Thallium test code (thal)", [1, 2, 3], help=CODE_NOTE)
    submitted = st.form_submit_button("Estimate")

if submitted:
    record = dict(age=age, sex=sex, cp=cp, trestbps=trestbps, chol=chol, fbs=fbs, restecg=restecg,
                  thalach=thalach, exang=exang, oldpeak=oldpeak, slope=slope, ca=ca, thal=thal)
    pred = predict_one(record)
    st.metric("Model probability of the dataset's 'disease' label", f"{pred.probability:.0%}")
    if pred.flagged:
        st.error(f"Above the screening threshold ({pred.threshold:.2f}): the model would flag this record.")
    else:
        st.success(f"Below the screening threshold ({pred.threshold:.2f}): the model would not flag this record.")
    st.caption(
        f"The threshold is set low on purpose: it was chosen so the model catches about "
        f"{config.TARGET_SENSITIVITY:.0%} of disease cases in training data, which means accepting more "
        "false alarms. A probability here means 'records like this were usually labelled X in this "
        "dataset', not 'this person's chance of having heart disease'."
    )

with st.expander("About the model"):
    t = meta["test_metrics_chosen_threshold"]
    st.markdown(
        f"- **Model:** {meta['model_name']} (L1-regularised), selected by cross-validated ROC-AUC\n"
        f"- **Trained on:** {meta['n_train']} unique patients; tested once on {meta['n_test']} held-out patients\n"
        f"- **Test sensitivity:** {t['recall_sensitivity']:.2f} · **specificity:** {t['specificity']:.2f} · "
        f"**ROC-AUC:** {t['roc_auc']:.2f}\n"
        "- Limitations: small sample, mostly male, single historical source, referral population."
    )
