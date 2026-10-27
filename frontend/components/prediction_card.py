"""
CreditLens-AI
Prediction Result Card
"""

from __future__ import annotations

import streamlit as st
from components.markup import render_html


def render_prediction_card(
    risk_level: str,
    probability: float,
    threshold: float,
    prediction: int,
    decision: str,
) -> None:

    probability_pct = probability * 100
    threshold_pct = threshold * 100

    is_high = prediction == 1

    status_class = "risk-high" if is_high else "risk-low"

    # ``risk_level`` and ``decision`` are rendered exactly as
    # returned by the API. The frontend does not infer or
    # recreate this text from the probability/threshold itself.
    status_text = risk_level.upper()
    status_description = decision

    render_html(
        f"""
        <div class="prediction-card {status_class}">

            <div class="prediction-card-header">
                <div>
                    <div class="section-label">
                        RISK DECISION
                    </div>

                    <div class="prediction-status">
                        {status_text}
                    </div>
                </div>

                <div class="prediction-score">
                    {probability_pct:.2f}%
                </div>
            </div>

            <div class="prediction-divider"></div>

            <div class="prediction-grid">

                <div>
                    <div class="metric-label">
                        Estimated Default Risk
                    </div>

                    <div class="metric-value">
                        {probability_pct:.2f}%
                    </div>
                </div>

                <div>
                    <div class="metric-label">
                        Review Threshold
                    </div>

                    <div class="metric-value">
                        {threshold_pct:.2f}%
                    </div>
                </div>

                <div>
                    <div class="metric-label">
                        Model Output
                    </div>

                    <div class="metric-value">
                        {prediction}
                    </div>
                </div>

            </div>

            <div class="prediction-description">
                {status_description}
            </div>

        </div>
        """)