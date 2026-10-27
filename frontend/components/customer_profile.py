"""
CreditLens-AI
Customer Profile Component

Shows only what the API actually returned for this assessment.
"""

from __future__ import annotations

import html
from typing import Any

from components.markup import render_html


def render_customer_profile(
    customer_id: str,
    risk_level: str,
    decision: str,
    completeness: dict[str, Any] | None = None,
    source: str = "Customer lookup",
) -> None:

    render_html(
        """
        <div class="section-header">
            <div>
                <div class="section-label">
                    CUSTOMER
                </div>
                <div class="section-title">
                    Assessment Summary
                </div>
            </div>
        </div>
        """
    )

    if completeness:
        data_value = (
            f"{completeness['provided_count']:,} of "
            f"{completeness['expected_count']:,} fields"
        )

        if completeness.get("null_count"):
            data_value += f" ({completeness['null_count']:,} empty)"
    else:
        data_value = "Not reported"

    items = [
        ("Customer ID", customer_id),
        ("Risk Level", risk_level),
        ("Decision", decision),
        ("Data Available", data_value),
        ("Input Source", source),
    ]

    cells = "".join(
        f"""
        <div class="profile-item">
            <div class="metric-label">{html.escape(label)}</div>
            <div class="profile-value">{html.escape(str(value))}</div>
        </div>
        """
        for label, value in items
    )

    render_html(f'<div class="profile-card">{cells}</div>')
