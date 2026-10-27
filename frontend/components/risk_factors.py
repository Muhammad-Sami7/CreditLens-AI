"""
CreditLens-AI
Risk / Protective Factors Component

Shows (1) the structured SHAP contributions returned by the API
as proportional bars, when available, and (2) the AI-written
plain-language factors. Contributions describe how the model
scored this customer; they do not prove causation.
"""

from __future__ import annotations

import html
from typing import Any

from components.feature_names import friendly_name, replace_feature_names
from components.formatting import display_value
from components.markup import render_html

TOP_SHAP_ROWS = 5


def _agreement_text(factor: dict[str, Any]) -> str:
    """'· 3 of 3 models' when the API reports model agreement."""

    agreeing = factor.get("models_agreeing")

    if agreeing is None:
        return ""

    return f" &middot; {int(agreeing)} of 3 models agree"


def _shap_rows(
    factors: list[dict[str, Any]],
    css_kind: str,
    scale: float,
) -> str:

    rows = []

    for factor in factors[:TOP_SHAP_ROWS]:

        contribution = float(factor["shap_value"])

        width = (
            min(abs(contribution) / scale * 100, 100)
            if scale > 0
            else 0
        )

        rows.append(
            f"""
            <div class="shap-row">
                <div class="shap-name"
                     title="{html.escape(str(factor["feature"]))}">
                    {html.escape(friendly_name(factor["feature"]))}
                </div>
                <div class="shap-value">
                    value: {html.escape(display_value(str(factor["feature"]), factor.get("feature_value")))}
                </div>
                <div>
                    <div class="shap-bar-track">
                        <div class="shap-bar shap-bar-{css_kind}"
                             style="width:{width:.1f}%;"></div>
                    </div>
                    <div class="shap-contribution">
                        contribution {contribution:+.4f}{_agreement_text(factor)}
                    </div>
                </div>
            </div>
            """
        )

    return "".join(rows)


def _text_items(
    items: list[str],
    css_kind: str,
    label: str,
) -> str:

    return "".join(
        f"""
        <div class="factor-item factor-item-{css_kind}">
            <span class="factor-marker factor-marker-{css_kind}">
                {label}
            </span>
            <span>{html.escape(replace_feature_names(str(item)))}</span>
        </div>
        """
        for item in items
    )


def render_risk_factors(
    key_risk_factors: list[str] | None = None,
    protective_factors: list[str] | None = None,
    shap: dict[str, Any] | None = None,
) -> None:

    key_risk_factors = key_risk_factors or []
    protective_factors = protective_factors or []

    positive = (shap or {}).get("positive_factors") or []
    negative = (shap or {}).get("negative_factors") or []

    render_html(
        """
        <div class="section-header">
            <div>
                <div class="section-label">EXPLAINABILITY</div>
                <div class="section-title">Why the model scored this customer this way</div>
            </div>
        </div>
        """
    )

    if not (
        key_risk_factors or protective_factors or positive or negative
    ):
        render_html(
            """
            <div class="empty-panel">
                <div class="empty-panel-title">No factor evidence returned</div>
                <div class="empty-panel-text">
                    The explanation service did not return any risk
                    or protective factors for this assessment.
                </div>
            </div>
            """
        )
        return

    scale = max(
        [abs(float(f["shap_value"])) for f in positive + negative]
        or [0.0]
    )

    # ---- Raising risk ------------------------------------------

    render_html(
        '<div class="factor-section-title">'
        "Factors Increasing Modeled Risk</div>"
    )

    if positive:
        render_html(_shap_rows(positive, "risk", scale))

    if key_risk_factors:
        render_html(
            '<div class="factor-list">'
            + _text_items(key_risk_factors, "risk", "RISK")
            + "</div>"
        )
    elif not positive:
        render_html(
            '<div class="empty-panel"><div class="empty-panel-text">'
            "No risk-increasing factors were returned."
            "</div></div>"
        )

    # ---- Lowering risk -----------------------------------------

    render_html(
        '<div class="factor-section-title">'
        "Factors Supporting Lower Modeled Risk</div>"
    )

    if negative:
        render_html(_shap_rows(negative, "protective", scale))

    if protective_factors:
        render_html(
            '<div class="factor-list">'
            + _text_items(protective_factors, "protective", "PROTECTIVE")
            + "</div>"
        )
    elif not negative:
        render_html(
            '<div class="empty-panel"><div class="empty-panel-text">'
            "No protective factors were returned."
            "</div></div>"
        )

    note = (
        "Bars show each field's contribution to the model's score "
        "(longer = stronger), combined across the LightGBM, XGBoost and CatBoost models. Fields are named in plain words; hover over a name for the technical field. "
        if (positive or negative)
        else ""
    )

    render_html(
        f"""
        <div class="explanation-disclaimer">
            {note}Contributions reflect how the model scored this
            customer. They do not prove that a field caused the outcome.
        </div>
        """
    )
