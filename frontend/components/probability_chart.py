"""
CreditLens-AI
Probability vs. Production Threshold Gauge

Visualizes the calibrated probability against the production
decision threshold on a 0-100% scale. Both values, and the
final decision text, come directly from the API response —
this component does not recompute or reinterpret the decision.
"""

from __future__ import annotations

import html
from typing import Any

import streamlit as st
from components.markup import render_html


def render_probability_chart(
    result: dict[str, Any],
) -> None:

    probability = float(result.get("probability", 0.0))
    threshold = float(result.get("threshold", 0.0))
    decision = str(result.get("decision", ""))
    risk_level = str(result.get("risk_level", ""))

    probability_pct = max(0.0, min(probability * 100, 100.0))
    threshold_pct = max(0.0, min(threshold * 100, 100.0))

    is_high = probability >= threshold

    def edge_class(percent: float) -> str:
        # Keep label boxes inside the track near 0% / 100%.
        if percent < 8:
            return "gauge-edge-start"
        if percent > 92:
            return "gauge-edge-end"
        return ""

    fill_class = "gauge-fill-high" if is_high else "gauge-fill-low"
    interpretation_class = (
        "gauge-interpretation-high"
        if is_high
        else "gauge-interpretation-low"
    )

    render_html(
        f"""
        <div class="gauge-wrap">

            <div
                class="gauge-marker gauge-marker-probability {edge_class(probability_pct)}"
                style="left:{probability_pct:.4f}%;"
            >
                <div class="gauge-marker-label">
                    {probability * 100:.2f}%
                </div>
                <div class="gauge-marker-line"></div>
            </div>

            <div class="gauge-track">
                <div
                    class="gauge-fill {fill_class}"
                    style="width:{probability_pct:.4f}%;"
                ></div>
            </div>

            <div
                class="gauge-marker gauge-marker-threshold {edge_class(threshold_pct)}"
                style="left:{threshold_pct:.4f}%;"
            >
                <div class="gauge-marker-line"></div>
                <div class="gauge-marker-label">
                    Threshold {threshold * 100:.2f}%
                </div>
            </div>

        </div>

        <div class="gauge-scale">
            <span>0%</span>
            <span>25%</span>
            <span>50%</span>
            <span>75%</span>
            <span>100%</span>
        </div>

        <div class="gauge-interpretation {interpretation_class}">
            Estimated default risk of
            <strong>{probability * 100:.2f}%</strong>
            {"meets or exceeds" if is_high else "is below"}
            the review threshold of
            <strong>{threshold * 100:.2f}%</strong>.
            Result: <strong>{html.escape(risk_level)}</strong>
            &mdash; {html.escape(decision)}.
        </div>
        """)
