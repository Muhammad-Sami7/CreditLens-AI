"""
CreditLens-AI
User feedback helpers (errors / unavailable features).
"""

from __future__ import annotations

import streamlit as st

from api_client import APIError
from components.markup import render_html


def show_api_error(error: APIError) -> None:
    """Show a friendly error, with the raw detail tucked away."""

    st.error(str(error))

    if error.detail:
        with st.expander("Technical details"):
            st.code(error.detail, language=None)


def render_feature_unavailable(title: str, endpoint: str) -> None:
    """
    Honest empty state for an optional feature whose endpoint the
    connected backend does not provide.
    """

    render_html(
        f"""
        <div class="empty-panel">
            <div class="empty-panel-title">{title}</div>
            <div class="empty-panel-text">
                The connected API does not provide <code>{endpoint}</code>.
                Update the backend to the version that includes it
                (see the project instructions) and refresh this page.
            </div>
        </div>
        """
    )
