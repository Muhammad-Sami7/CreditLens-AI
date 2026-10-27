"""
CreditLens-AI
Model Breakdown Component
"""

from __future__ import annotations

from typing import Any

import streamlit as st
from components.markup import render_html


def render_model_breakdown(
    result: dict[str, Any],
) -> None:

    probabilities = result.get(
        "model_probabilities",
        {},
    )

    lightgbm = float(
        probabilities.get("lightgbm", 0)
    )

    xgboost = float(
        probabilities.get("xgboost", 0)
    )

    catboost = float(
        probabilities.get("catboost", 0)
    )

    ensemble = float(
        result.get(
            "ensemble_probability",
            0,
        )
    )

    calibrated = float(
        result.get(
            "probability",
            0,
        )
    )

    # Weights come from the API (the backend's ensemble configuration).
    # If the backend does not send them, show a dash instead of guessing.
    weights = result.get("model_weights") or {}

    def weight_label(model: str) -> str:
        value = weights.get(model)
        return f"{float(value) * 100:.0f}%" if value is not None else "—"

    rows = [
        ("LightGBM", lightgbm, weight_label("lightgbm")),
        ("XGBoost", xgboost, weight_label("xgboost")),
        ("CatBoost", catboost, weight_label("catboost")),
        ("Weighted Ensemble", ensemble, "—"),
        ("Isotonic Calibration", calibrated, "—"),
    ]

    render_html(
        """
        <div class="section-header">
            <div>
                <div class="section-label">
                    MODEL INTELLIGENCE
                </div>

                <div class="section-title">
                    Prediction Breakdown
                </div>
            </div>
        </div>
        """)

    for name, probability, weight in rows:

        render_html(
            f"""
            <div class="model-row">

                <div class="model-name">
                    {name}
                </div>

                <div class="model-weight">
                    {weight}
                </div>

                <div class="model-probability">
                    {probability * 100:.2f}%
                </div>

            </div>
            """)