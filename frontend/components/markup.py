"""
CreditLens-AI
HTML rendering helper.

Streamlit renders ``st.markdown`` through a CommonMark parser.
In CommonMark an HTML block ends at the first blank line, and
any following line indented by 4+ spaces becomes a *code block*
(raw HTML shown as text). Hand-formatted, indented HTML triggers
this easily, so every HTML snippet goes through ``render_html``,
which removes indentation and blank lines before rendering.
"""

from __future__ import annotations

import streamlit as st


def compact_html(markup: str) -> str:
    """Strip per-line indentation and drop blank lines."""

    lines = (line.strip() for line in markup.splitlines())

    return "\n".join(line for line in lines if line)


def render_html(markup: str) -> None:
    st.markdown(
        compact_html(markup),
        unsafe_allow_html=True,
    )
