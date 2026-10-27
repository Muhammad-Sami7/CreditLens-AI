"""
CreditLens-AI
Dashboard Page

All figures come from ``GET /api/v1/dashboard/summary`` (computed
from persisted predictions). Nothing is calculated or estimated
in the frontend.
"""

from __future__ import annotations

import html
from datetime import datetime
from typing import Any

import streamlit as st

from api_client import EP_DASHBOARD, APIError, CreditLensAPI
from components.feedback import render_feature_unavailable
from components.markup import render_html


def _kpi(label: str, value: str, caption: str) -> str:

    return f"""
    <div class="kpi-card">
        <div class="metric-label">{html.escape(label)}</div>
        <div class="kpi-value">{html.escape(value)}</div>
        <div class="kpi-caption">{html.escape(caption)}</div>
    </div>
    """


def _render_health(api: CreditLensAPI) -> bool:

    try:
        healthy = str(api.health().get("status", "")).lower() == "healthy"
    except Exception:
        healthy = False

    text = (
        "Assessment service is running"
        if healthy
        else "Assessment service is unreachable"
    )

    color = "#16A34A" if healthy else "#B91C1C"

    render_html(
        f"""
        <div class="status-strip">
            <div>
                <span class="status-dot" style="background:{color};"></span>
                {text}
            </div>
        </div>
        """
    )

    return healthy


def _render_recent(recent: list[dict[str, Any]]) -> None:

    rows = []

    for record in recent:

        try:
            when = datetime.fromisoformat(record["created_at"]).strftime(
                "%d %b %Y, %H:%M"
            )
        except (ValueError, TypeError):
            when = str(record["created_at"])

        rows.append(
            f"""
            <div class="weight-row">
                <span>{html.escape(str(record["customer_id"]))}
                    &middot; {html.escape(record["risk_level"])}</span>
                <strong>{record["probability"] * 100:.2f}%
                    &middot; {html.escape(when)}</strong>
            </div>
            """
        )

    render_html(
        f"""
        <div class="panel">
            <div class="section-label">RECENT ASSESSMENTS</div>
            <div class="section-title">Latest results</div>
            {"".join(rows)}
        </div>
        """
    )


def render_dashboard() -> None:

    render_html(
        """
        <div class="page-eyebrow">CREDIT RISK INTELLIGENCE</div>
        <div class="page-title">Portfolio Overview</div>
        <div class="page-description">
            Overview of assessments run through CreditLens-AI.
        </div>
        """
    )

    api = CreditLensAPI()

    _render_health(api)

    render_html("<div class='section-gap'></div>")

    if not api.supports(api.get_capabilities(), EP_DASHBOARD):
        render_feature_unavailable(
            "Assessment analytics are not available on the connected API.",
            "GET /dashboard/summary",
        )
        return

    try:
        summary = api.get_dashboard_summary()
    except APIError:
        render_html(
            """
            <div class="empty-panel">
                <div class="empty-panel-title">
                    Assessment analytics are currently unavailable.
                </div>
                <div class="empty-panel-text">
                    Totals and recent assessments could not be
                    loaded. Risk assessments can still be run.
                </div>
            </div>
            """
        )
        return

    total = summary["total_assessments"]
    high = summary["high_risk_assessments"]
    low = summary["low_risk_assessments"]
    average = summary["average_probability"]

    columns = st.columns(4, gap="medium")

    cards = [
        ("Total Assessments", f"{total:,}", "Saved assessments"),
        ("High Risk", f"{high:,}", "Flagged for review"),
        ("Low Risk", f"{low:,}", "Below the review threshold"),
        (
            "Average Estimated Risk",
            f"{average * 100:.2f}%" if average is not None else "—",
            "Mean estimated default risk",
        ),
    ]

    for column, (label, value, caption) in zip(columns, cards):
        with column:
            render_html(_kpi(label, value, caption))

    render_html("<div class='section-gap'></div>")

    if total == 0:
        render_html(
            """
            <div class="empty-panel">
                <div class="empty-panel-title">No assessments yet</div>
                <div class="empty-panel-text">
                    Run a Risk Assessment and the results will
                    appear here.
                </div>
            </div>
            """
        )
        return

    left, right = st.columns([1.2, 1], gap="large")

    with left:
        _render_recent(summary["recent_assessments"])

    with right:
        high_pct = high / total * 100
        low_pct = low / total * 100

        render_html(
            f"""
            <div class="panel">
                <div class="section-label">RISK DISTRIBUTION</div>
                <div class="section-title">High vs. low risk</div>
                <div class="dist-bar">
                    <div class="dist-high" style="width:{high_pct:.2f}%;"></div>
                    <div class="dist-low" style="width:{low_pct:.2f}%;"></div>
                </div>
                <div class="dist-legend">
                    <span>High risk {high_pct:.1f}% ({high:,})</span>
                    <span>Low risk {low_pct:.1f}% ({low:,})</span>
                </div>
            </div>
            """
        )
