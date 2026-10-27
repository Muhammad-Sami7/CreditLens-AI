"""
CreditLens-AI
Customer History Page

Shows persisted assessments from the backend history endpoint.
No data is created or estimated in the frontend.
"""

from __future__ import annotations

import html
from datetime import datetime
from typing import Any

import streamlit as st

from api_client import EP_HISTORY, APIError, CreditLensAPI
from components.feedback import (
    render_feature_unavailable,
    show_api_error,
)
from components.markup import render_html


def _format_timestamp(value: str) -> str:

    try:
        return datetime.fromisoformat(value).strftime(
            "%d %b %Y, %H:%M %Z"
        ).strip()
    except (ValueError, TypeError):
        return str(value)


def _render_item(record: dict[str, Any]) -> None:

    is_high = record["prediction"] == 1

    css = "history-item-high" if is_high else "history-item-low"

    render_html(
        f"""
        <div class="history-item {css}">
            <div>
                <div class="history-level">
                    {html.escape(record["risk_level"])}
                </div>
                <div class="history-meta">
                    Assessment #{record["prediction_id"]}
                </div>
            </div>
            <div class="history-meta">
                {html.escape(record["decision"])}<br>
                {html.escape(_format_timestamp(record["created_at"]))}<br>
                Review threshold at the time:
                {record["threshold"] * 100:.2f}%
            </div>
            <div class="history-probability">
                {record["probability"] * 100:.2f}%
            </div>
        </div>
        """
    )


def render_customer_history() -> None:

    render_html(
        """
        <div class="page-eyebrow">CUSTOMER INTELLIGENCE</div>
        <div class="page-title">Customer History</div>
        <div class="page-description">
            Review a customer's previous risk assessments,
            newest first.
        </div>
        """
    )

    api = CreditLensAPI()

    if not api.supports(api.get_capabilities(), EP_HISTORY):
        render_feature_unavailable(
            "Customer history is not available on the connected API.",
            "GET /customers/{customer_id}/predictions",
        )
        return

    customer_id = st.text_input(
        "Customer ID",
        placeholder="Enter customer ID",
    ).strip()

    render_html("<div class='section-gap'></div>")

    if not customer_id:
        render_html(
            """
            <div class="empty-panel">
                <div class="empty-panel-title">
                    Enter a customer ID to see their history
                </div>
                <div class="empty-panel-text">
                    Previous assessments for that customer will
                    appear here.
                </div>
            </div>
            """
        )
        return

    try:
        with st.spinner("Loading assessment history..."):
            records = api.get_customer_history(customer_id)
    except APIError as exc:
        show_api_error(exc)
        return

    if not records:
        render_html(
            """
            <div class="empty-panel">
                <div class="empty-panel-title">
                    No previous assessments found for this customer.
                </div>
                <div class="empty-panel-text">
                    Run a Risk Assessment to create the first record.
                </div>
            </div>
            """
        )
        return

    latest = records[0]

    render_html(
        f"""
        <div class="status-strip">
            <div>
                {len(records)} assessment{"s" if len(records) != 1 else ""}
                &middot; latest result:
                <strong>{html.escape(latest["risk_level"])}</strong>
            </div>
            <div class="status-meta">
                Customer {html.escape(customer_id)}
            </div>
        </div>
        """
    )

    for record in records:
        _render_item(record)
