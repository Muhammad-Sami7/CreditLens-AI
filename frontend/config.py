"""
CreditLens-AI
Frontend Configuration
"""

from __future__ import annotations

import os
from pathlib import Path


# ============================================================
# Project Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# API Configuration
# ============================================================

API_BASE_URL = os.getenv(
    "CREDITLENS_API_URL",
    "http://127.0.0.1:8000/api/v1",
)

API_TIMEOUT = int(
    os.getenv(
        "CREDITLENS_API_TIMEOUT",
        "180",
    )
)


# ============================================================
# Application Configuration
# ============================================================

APP_NAME = "CreditLens-AI"
APP_VERSION = "1.0.0"

APP_TAGLINE = (
    "Financial Risk Intelligence"
)

APP_DESCRIPTION = (
    "Production credit-risk assessment "
    "powered by calibrated machine learning."
)


# ============================================================
# Production Configuration
# ============================================================

PRODUCTION_THRESHOLD = 0.18

ENSEMBLE_WEIGHTS = {
    "LightGBM": 0.35,
    "XGBoost": 0.35,
    "CatBoost": 0.30,
}


# ============================================================
# Visual Configuration
# ============================================================

COLORS = {
    "background": "#F5F7FA",
    "surface": "#FFFFFF",
    "surface_alt": "#F8FAFC",

    "navy": "#0F172A",
    "navy_light": "#1E293B",

    "text": "#172033",
    "text_muted": "#64748B",

    "border": "#E2E8F0",

    "blue": "#2563EB",
    "blue_light": "#EFF6FF",

    "green": "#15803D",
    "green_light": "#F0FDF4",

    "amber": "#B45309",
    "amber_light": "#FFFBEB",

    "red": "#B91C1C",
    "red_light": "#FEF2F2",
}


# ============================================================
# Navigation
# ============================================================

NAVIGATION_ITEMS = [
    "Dashboard",
    "Risk Assessment",
    "Customer History",
    "About",
]