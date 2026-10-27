"""
CreditLens-AI
Display formatting shared by the page and the PDF report.

Numbers are shown the way a loan officer reads them: thousands
separators for large amounts, no scientific notation.
"""

from __future__ import annotations

from typing import Any


def format_number(value: Any) -> str:
    """49018.5 -> '49,019'; 1575000.0 -> '1,575,000'; 0.1062 -> '0.1062'."""

    if value is None:
        return "no data"

    if isinstance(value, bool):
        return "yes" if value else "no"

    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)

    if number != number:  # NaN
        return "no data"

    magnitude = abs(number)

    if magnitude >= 1000:
        return f"{number:,.0f}"

    if number == int(number):
        return str(int(number))

    if 0 < magnitude < 0.0001:
        return f"{number:.2e}"

    return f"{number:.4g}"


def display_value(feature: str, value: Any) -> str:
    """Value text for a factor row."""

    return format_number(value)

