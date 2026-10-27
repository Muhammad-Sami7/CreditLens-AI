"""
CreditLens-AI
About Page

Plain, honest description of what the project is, how it works and
what it has not been validated for. Static text: nothing here calls
the API or calculates anything.
"""

from __future__ import annotations

import streamlit as st

from components.markup import render_html

# Reported model quality, copied from the project's evaluation notebooks
# (weighted ensemble; calibration notebook for Brier and the 0.18 row).
# Keep EVAL_SET equal to the data those numbers were computed on.
EVAL_SET = "development validation set"

METRICS = [
    ("ROC-AUC", "0.79", f"Weighted ensemble, {EVAL_SET}"),
    ("BRIER SCORE", "0.065", "After calibration, fitted on the same validation set (raw: 0.165)"),
    ("MODELS", "3", "LightGBM, XGBoost and CatBoost ensemble"),
    ("FEATURES", "297", "From the application and six history tables"),
]

HOW_IT_WORKS = [
    ("1", "Data",
     "The public Home Credit Default Risk dataset: applications joined "
     "with credit-bureau, previous-loan, instalment, point-of-sale and "
     "credit-card histories, giving 297 model features."),
    ("2", "Models",
     "An ensemble of LightGBM, XGBoost and CatBoost (weights 0.35 / "
     "0.35 / 0.30). Raw ensemble scores run far above real default "
     "rates, so isotonic calibration maps them to observed frequencies: "
     "an applicant shown at 10% belongs to a group where about 10% "
     "defaulted."),
    ("3", "Decision",
     "A calibrated probability at or above 0.18 is flagged for review. "
     "0.18 maximises F1 on calibrated probabilities, so it is a "
     "screening aid, not a verdict. The backend applies it, never "
     "this app."),
    ("4", "Evidence",
     "SHAP shows which fields pushed the estimate up or down, and an "
     "AI-written summary turns that evidence into plain language for "
     "the analyst."),
    ("5", "Audit trail",
     "Every assessment is saved to PostgreSQL and appears on the "
     "dashboard and in each customer's history."),
]

LIMITS = [
    "Built on a public dataset. It has never seen your applicants, and "
    "it must not be used for real lending decisions.",
    "The strongest signals are anonymised external credit scores; a real "
    "lender would need its own equivalent.",
    "It assesses customers from the dataset; it does not take new, "
    "unseen applicants.",
    "Gender is an input of the trained model. A real deployment would "
    "need a fairness review and, most likely, its removal.",
]


def render_about() -> None:

    render_html(
        """
        <div class="page-eyebrow">ABOUT</div>
        <div class="page-title">About CreditLens-AI</div>
        <div class="page-description">
            A credit-risk decision-support prototype: calibrated
            probabilities, transparent evidence and honest uncertainty.
        </div>
        """
    )

    columns = st.columns(len(METRICS), gap="medium")

    for column, (label, value, caption) in zip(columns, METRICS):
        with column:
            render_html(
                f"""
                <div class="kpi-card">
                    <div class="metric-label">{label}</div>
                    <div class="kpi-value">{value}</div>
                    <div class="kpi-caption">{caption}</div>
                </div>
                """
            )

    render_html("<div class='section-gap'></div>")

    render_html(
        """
        <div class="panel">
            <div class="section-label">PURPOSE</div>
            <div class="section-title">What this project shows</div>
            <p>
                An end-to-end machine-learning system: feature
                engineering across six history tables, an ensemble
                with calibration, explainability, an API with a
                database, and an analyst-facing app. It is a
                demonstration built on public data, not a lending
                product.
            </p>
        </div>
        """
    )

    render_html("<div class='section-gap'></div>")

    steps = "".join(
        f"""
        <div style="display:grid;grid-template-columns:150px 1fr;gap:16px;
                    padding:14px 0;border-bottom:1px solid #E2E8F0;
                    font-size:14px;line-height:1.55;color:#334155;">
            <strong style="color:#0F172A;">{number}. {title}</strong>
            <span>{text}</span>
        </div>
        """
        for number, title, text in HOW_IT_WORKS
    )

    render_html(
        f"""
        <div class="panel">
            <div class="section-label">HOW IT WORKS</div>
            <div class="section-title">From data to a decision</div>
            {steps}
        </div>
        """
    )

    render_html("<div class='section-gap'></div>")

    items = "".join(f"<li>{text}</li>" for text in LIMITS)

    render_html(
        f"""
        <div class="panel">
            <div class="section-label">LIMITATIONS</div>
            <div class="section-title">What it has not been validated for</div>
            <ul>{items}</ul>
        </div>
        """
    )
