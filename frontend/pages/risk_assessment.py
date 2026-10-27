"""
CreditLens-AI
Risk Assessment Page

Primary workflow: the analyst enters (or picks) a customer ID and the
backend retrieves the customer's data and runs the production pipeline
(``POST /assess/{customer_id}``).

If the connected backend does not provide that endpoint, the page says
so and offers the custom-data form (``POST /predict``) instead. Nothing
is calculated in the frontend.
"""

from __future__ import annotations

import json
from typing import Any, Callable

import streamlit as st

from api_client import (
    EP_ASSESS,
    EP_PREDICT,
    EP_SAMPLE_CUSTOMERS,
    APIError,
    CreditLensAPI,
)
from components.customer_profile import render_customer_profile
from components.feedback import show_api_error
from components.markup import render_html
from components.model_breakdown import render_model_breakdown
from components.prediction_card import render_prediction_card
from components.probability_chart import render_probability_chart
from components.recommendation_panel import (
    render_recommendation_panel,
)
from components.risk_factors import render_risk_factors

STATE_KEY = "latest_assessment"
SOURCE_KEY = "latest_assessment_source"

SAMPLE_PLACEHOLDER = "— none —"


def _run(
    fetch: Callable[[], dict[str, Any]],
    source: str,
) -> None:
    """Run an assessment and store only the API response."""

    st.session_state.pop(STATE_KEY, None)

    with st.spinner(
        "Running risk models and preparing the explanation. "
        "This can take up to a minute..."
    ):
        try:
            result = fetch()
        except APIError as exc:
            show_api_error(exc)
            return
        except Exception:
            st.error(
                "The risk service returned an unexpected response."
            )
            return

    st.session_state[STATE_KEY] = result
    st.session_state[SOURCE_KEY] = source


def _render_custom_data_form(
    api: CreditLensAPI,
    expanded: bool,
    title: str,
) -> None:

    with st.expander(title, expanded=expanded):
        st.caption(
            "For developers and testing. Provide production feature "
            "data as JSON (pasted or uploaded): an object whose keys "
            "are the production feature names. Fields left out are "
            "treated as missing data, which can reduce the "
            "reliability of the result; use null for an empty value."
        )

        advanced_id = st.text_input(
            "Customer ID for custom data",
            key="advanced_customer_id",
        )

        uploaded = st.file_uploader(
            "Upload a .json file",
            type=["json"],
        )

        pasted = st.text_area(
            "…or paste JSON",
            height=160,
            placeholder='{"feature_name": 0.5, "other_feature": null}',
        )

        if not st.button("Run with custom data"):
            return

        raw = (
            uploaded.getvalue().decode("utf-8", errors="replace")
            if uploaded is not None
            else pasted
        )

        if not advanced_id.strip():
            st.error("Please enter a customer ID.")
            return

        if not raw.strip():
            st.error("Please upload or paste customer data.")
            return

        try:
            features = json.loads(raw)
        except json.JSONDecodeError:
            st.error(
                "The data is not valid JSON. Please review it "
                "and try again."
            )
            return

        if not isinstance(features, dict):
            st.error(
                "Customer data must be a JSON object of "
                "field names and values."
            )
            return

        _run(
            lambda: api.predict(advanced_id.strip(), features),
            "Custom data",
        )


def _render_inputs(api: CreditLensAPI) -> None:

    capabilities = api.get_capabilities()

    can_lookup = api.supports(capabilities, EP_ASSESS)
    can_list = api.supports(capabilities, EP_SAMPLE_CUSTOMERS)
    can_predict = api.supports(capabilities, EP_PREDICT)

    if not can_lookup:
        st.warning(
            "Customer lookup by ID is not available on the connected "
            "API (it does not provide POST /assess/{customer_id}). "
            "Use the custom-data form below, or update the backend."
        )

        if can_predict:
            _render_custom_data_form(
                api,
                expanded=True,
                title="Run an assessment with custom customer data",
            )

        return

    render_html(
        """
        <div class="panel">
            <div class="section-label">CUSTOMER</div>
            <div class="section-title">Who would you like to assess?</div>
        </div>
        """
    )

    samples: list[str] = []

    if can_list:
        try:
            samples = api.get_sample_customers(limit=25)
        except APIError:
            samples = []

    left, right = st.columns(2, gap="large")

    with left:
        typed_id = st.text_input(
            "Customer ID",
            placeholder="Enter the customer ID",
        )

    with right:
        picked_id = (
            st.selectbox(
                "Or choose a customer from the list",
                [SAMPLE_PLACEHOLDER, *samples],
            )
            if samples
            else SAMPLE_PLACEHOLDER
        )

    customer_id = typed_id.strip() or (
        "" if picked_id == SAMPLE_PLACEHOLDER else picked_id
    )

    if st.button("Run Risk Assessment", type="primary"):

        if not customer_id:
            st.error("Please enter or choose a customer ID.")
        else:
            _run(
                lambda: api.assess_customer(customer_id),
                "Customer lookup",
            )

    if can_predict:
        _render_custom_data_form(
            api,
            expanded=False,
            title="Advanced: use custom customer data (technical users)",
        )


def _render_report_download(
    result: dict[str, Any],
    source: str | None = None,
) -> None:
    """Offer the assessment as a PDF (built from the API response only)."""

    # Imported here so a missing optional package never stops the app.
    try:
        from components.report_pdf import build_assessment_pdf
    except ImportError:
        st.caption(
            "PDF reports need the 'reportlab' package. "
            "Install it with: pip install reportlab"
        )
        return

    try:
        pdf = build_assessment_pdf(
            result,
            source=source or st.session_state.get(
                SOURCE_KEY, "Customer lookup"
            ),
        )
    except Exception:
        st.caption("The PDF report could not be prepared for this result.")
        return

    st.download_button(
        "Download PDF report",
        data=pdf,
        file_name=f"creditlens_assessment_{result['customer_id']}.pdf",
        mime="application/pdf",
    )


def _render_result(
    result: dict[str, Any],
    source: str | None = None,
) -> None:

    source = source or st.session_state.get(
        SOURCE_KEY, "Customer lookup"
    )

    explanation = result.get("llm_explanation") or {}
    completeness = result.get("feature_completeness")

    render_html("<div class='section-gap'></div>")

    _render_report_download(result, source)

    if completeness and completeness.get("missing_count", 0) > 0:
        st.warning(
            f"{completeness['missing_count']:,} of "
            f"{completeness['expected_count']:,} data fields were "
            "not provided for this customer. Missing fields are "
            "treated as 'no data', which can make this result "
            "less reliable."
        )

    render_prediction_card(
        risk_level=result["risk_level"],
        probability=result["probability"],
        threshold=result["threshold"],
        prediction=result["prediction"],
        decision=result["decision"],
    )

    render_html("<div class='section-gap'></div>")

    render_html(
        """
        <div class="panel">
            <div class="section-label">RISK vs. REVIEW THRESHOLD</div>
            <div class="section-title">
                Where this customer sits relative to the review threshold
            </div>
        </div>
        """
    )

    render_probability_chart(result)

    render_html("<div class='section-gap'></div>")

    left, right = st.columns([1, 1], gap="large")

    with left:
        render_risk_factors(
            key_risk_factors=explanation.get("key_risk_factors"),
            protective_factors=explanation.get("protective_factors"),
            shap=result.get("shap"),
        )

    with right:
        render_recommendation_panel(
            risk_level=result["risk_level"],
            decision=result["decision"],
            llm_explanation=explanation,
        )

    render_html("<div class='section-gap'></div>")

    render_customer_profile(
        customer_id=result["customer_id"],
        risk_level=result["risk_level"],
        decision=result["decision"],
        completeness=completeness,
        source=source,
    )

    with st.expander(
        "Technical details: how each model scored this customer"
    ):
        render_model_breakdown(result)


def render_risk_assessment() -> None:

    render_html(
        """
        <div class="page-eyebrow">CREDIT DECISION SUPPORT</div>
        <div class="page-title">Risk Assessment</div>
        <div class="page-description">
            Get a model-assisted risk assessment for analyst review.
            Results support, and do not replace, the analyst's decision.
        </div>
        """
    )

    _render_inputs(CreditLensAPI())

    result = st.session_state.get(STATE_KEY)

    if result:
        _render_result(result)


def render_assessment_result(
    result: dict[str, Any],
    source: str,
) -> None:
    """Public entry point so other pages show results identically."""

    _render_result(result, source)
