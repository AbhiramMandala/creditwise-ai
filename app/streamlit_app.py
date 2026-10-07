"""Streamlit dashboard: Overview, Risk, Segmentation, Anomaly, Explainability,
Fairness, Performance, Loan, Finance, Reports. All numbers computed live."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402
from src.anomaly import fit_anomaly  # noqa: E402
from src.clustering import elbow_table, fit_clusters  # noqa: E402
from src.credit_risk import (  # noqa: E402
    global_importance,
    load_model,
    predict_one,
)
from src.data_loader import basic_profile, load_data, numeric_summary, validate_data  # noqa: E402
from src.fairness import fairness_report  # noqa: E402
from src.finance_advisor import finance_health  # noqa: E402
from src.loan_recommender import recommend_loan  # noqa: E402

st.set_page_config(page_title=config.APP_NAME, layout="wide")
st.title(config.APP_NAME)
st.caption("Decision-support only — predictions are not loan approvals or financial advice.")

CAT_OPTS = {
    "checking_status": ["<0", "0<=X<200", ">=200", "no checking"],
    "credit_history": ["critical/other existing credit", "existing paid", "delayed previously",
                       "no credits/all paid", "all paid"],
    "purpose": ["radio/tv", "education", "furniture/equipment", "new car", "used car",
                "business", "domestic appliance", "repairs", "other", "retraining"],
    "savings_status": ["<100", "100<=X<500", "500<=X<1000", ">=1000", "no known savings"],
    "employment": ["unemployed", "<1", "1<=X<4", "4<=X<7", ">=7"],
    "personal_status": ["male div/sep", "female div/dep/mar", "male single", "male mar/wid", "female single"],
    "other_parties": ["none", "co applicant", "guarantor"],
    "property_magnitude": ["real estate", "life insurance", "car", "no known property"],
    "other_payment_plans": ["bank", "stores", "none"],
    "housing": ["rent", "own", "for free"],
    "job": ["unemp/unskilled non res", "unskilled resident", "skilled", "high qualif/self emp/mgmt"],
    "own_telephone": ["none", "yes"],
    "foreign_worker": ["yes", "no"],
}
DEFAULTS = {"duration": 12.0, "credit_amount": 3000.0, "installment_commitment": 4.0,
            "residence_since": 3.0, "age": 35.0, "existing_credits": 1.0, "num_dependents": 1.0,
            **{k: v[0] for k, v in CAT_OPTS.items()}}


def customer_form(prefix: str) -> dict:
    c1, c2, c3 = st.columns(3)
    rec: dict = {}
    with c1:
        rec["age"] = st.number_input("Age", 18, 100, int(DEFAULTS["age"]), key=prefix + "age")
        rec["duration"] = st.number_input("Duration (months)", 0, 120, int(DEFAULTS["duration"]), key=prefix + "dur")
        rec["credit_amount"] = st.number_input("Credit amount", 0, 200000, int(DEFAULTS["credit_amount"]), key=prefix + "amt")
        rec["installment_commitment"] = st.number_input("Instalment %", 0, 100, int(DEFAULTS["installment_commitment"]), key=prefix + "inst")
    with c2:
        rec["residence_since"] = st.number_input("Years at residence", 0, 50, int(DEFAULTS["residence_since"]), key=prefix + "res")
        rec["existing_credits"] = st.number_input("Existing credits", 0, 20, int(DEFAULTS["existing_credits"]), key=prefix + "ex")
        rec["num_dependents"] = st.number_input("Dependents", 0, 20, int(DEFAULTS["num_dependents"]), key=prefix + "dep")
        for k in ["checking_status", "credit_history", "purpose", "savings_status"]:
            rec[k] = st.selectbox(k.replace("_", " ").title(), CAT_OPTS[k], key=prefix + k)
    with c3:
        for k in ["employment", "personal_status", "other_parties", "property_magnitude",
                  "other_payment_plans", "housing", "job", "own_telephone", "foreign_worker"]:
            rec[k] = st.selectbox(k.replace("_", " ").title(), CAT_OPTS[k], key=prefix + k)
    return rec


@st.cache_data
def _load_df():
    df = load_data()
    return df


@st.cache_resource
def _risk_pipe():
    try:
        return load_model()
    except FileNotFoundError:
        return None


df = _load_df()
problems = validate_data(df)
if problems:
    st.warning("Data-quality notes: " + "; ".join(problems))

tabs = st.tabs(["Overview", "Risk", "Segmentation", "Anomaly", "Explainability",
                "Fairness", "Performance", "Loan", "Finance", "Reports"])

with tabs[0]:
    st.header("Overview")
    prof = basic_profile(df)
    a, b, c, d = st.columns(4)
    a.metric("Records", prof["n_rows"])
    b.metric("Good (low risk)", prof["class_counts"].get("good", 0))
    c.metric("Bad (high risk)", prof["class_counts"].get("bad", 0))
    d.metric("Missing / dup", f"{prof['missing_values']} / {prof['duplicate_rows']}")
    st.bar_chart(df["class"].value_counts())
    fig, ax = plt.subplots()
    sns.histplot(df["credit_amount"], bins=25, kde=True, ax=ax)
    ax.set_title("Credit amount distribution")
    st.pyplot(fig)
    st.dataframe(numeric_summary(df))

with tabs[1]:
    st.header("Credit Risk Prediction")
    rec = customer_form("risk_")
    if st.button("Predict risk", key="pred"):
        pipe = _risk_pipe()
        if pipe is None:
            st.error("Model not trained. Run `python scripts/train.py` first.")
        else:
            out = predict_one(pipe, rec)
            st.success(f"{out['risk_level']} — P(good)={out['p_good']:.2%}, P(bad)={out['p_bad']:.2%}")
            st.write("Top reasons (from model features):")
            st.dataframe(pd.DataFrame(out["explanations"]))

with tabs[2]:
    st.header("Customer Segmentation (KMeans, scaled)")
    k = st.slider("k", 2, 6, config.N_CLUSTERS)
    pipe_c, labeled, info = fit_clusters(df, k)
    st.write(f"Silhouette: {info['metrics'].get('silhouette', float('nan')):.3f}")
    st.bar_chart(labeled["cluster"].value_counts().sort_index())
    st.dataframe(pd.DataFrame(info["profiles"]).T)
    st.write("Elbow / silhouette table:")
    st.dataframe(pd.DataFrame(elbow_table(df)))

with tabs[3]:
    st.header("Anomaly Detection (IsolationForest)")
    st.info("Anomalies need human review — they are NOT confirmed fraud.")
    _, scored = fit_anomaly(df)
    st.metric("Flagged", int(scored["anomaly"].sum()))
    st.dataframe(scored[scored["anomaly"] == 1].head(20))

with tabs[4]:
    st.header("Model Explainability")
    pipe = _risk_pipe()
    if pipe is None:
        st.error("Train the model first.")
    else:
        imp = global_importance(pipe)
        st.bar_chart(pd.DataFrame(imp).set_index("feature")["importance"])
        rec2 = customer_form("exp_")
        if st.button("Explain this customer", key="exp"):
            st.dataframe(pd.DataFrame(predict_one(pipe, rec2)["explanations"]))

with tabs[5]:
    st.header("Fairness Analysis")
    pipe = _risk_pipe()
    if pipe is None:
        st.error("Train the model first.")
    else:
        rep = fairness_report(df, pipe)
        for attr in ["personal_status", "employment", "age_band"]:
            st.subheader(f"By {attr} (approval = predicted good)")
            st.dataframe(pd.DataFrame(rep[attr]["groups"]))
            st.write(f"Max approval-rate gap: {rep[attr]['max_approval_gap']:.1%}")
        st.caption(rep["note"])

with tabs[6]:
    st.header("Model Performance")
    if config.RISK_METRICS_PATH.exists():
        m = json.loads(config.RISK_METRICS_PATH.read_text())
        st.write(f"Selected model: **{m['selected']['best_model']}** (v{m['model_version']})")
        st.dataframe({k: {kk: round(vv, 3) if isinstance(vv, float) else vv
                          for kk, vv in v.items() if kk != "confusion_matrix"}
                      for k, v in m["models"].items()}.copy())
        for name, mm in m["models"].items():
            fig, ax = plt.subplots()
            sns.heatmap(mm["confusion_matrix"], annot=True, fmt="d", ax=ax, cmap="Blues",
                        xticklabels=["bad", "good"], yticklabels=["bad", "good"])
            ax.set_title(name)
            st.pyplot(fig)
    else:
        st.error("No metrics yet — run `python scripts/train.py`.")

with tabs[7]:
    st.header("Loan Recommendation (decision support)")
    p = st.slider("Predicted P(good) from Risk tab", 0.0, 1.0, 0.65)
    inc = st.number_input("Monthly income", 0, 1000000, 5000, key="li")
    exp = st.number_input("Monthly expenses", 0, 1000000, 3000, key="le")
    amt = st.number_input("Requested amount", 0, 1000000, 8000, key="la")
    dur = st.number_input("Duration (months)", 1, 120, 24, key="ld")
    if st.button("Recommend", key="rec"):
        r = recommend_loan(p, float(inc), float(exp), float(amt), int(dur))
        st.write(f"Decision: **{r['decision']}** | max affordable ≈ ${r['max_affordable_amount']:,.0f}")
        st.write("Products:", ", ".join(r["products"]))
        for reason in r["reasons"]:
            st.write("- " + reason)
        st.caption(r["disclaimer"])

with tabs[8]:
    st.header("Personal Finance Insights")
    fi = st.number_input("Income", 0, 1000000, 5000, key="fi")
    fe = st.number_input("Expenses", 0, 1000000, 3000, key="fe2")
    fs = st.number_input("Savings", 0, 10000000, 10000, key="fs")
    fd = st.number_input("Debt", 0, 10000000, 5000, key="fd")
    if st.button("Assess", key="fin"):
        h = finance_health(float(fi), float(fe), float(fs), float(fd))
        st.metric("Health score", h["health_score"])
        st.write(h["notes"])
        st.caption(h["disclaimer"])

with tabs[9]:
    st.header("Reports")
    st.download_button("Download dataset profile (JSON)",
                       json.dumps(basic_profile(df), indent=2), "profile.json")
    if config.RISK_METRICS_PATH.exists():
        st.download_button("Download model metrics (JSON)",
                           config.RISK_METRICS_PATH.read_text(), "metrics.json")
