"""
============================================================
CreditLens-AI
Production Feature Schema
============================================================

Defines the production feature contract used by the trained
ensemble models.

The authoritative feature list is:

models/feature_columns.json
"""

from __future__ import annotations

import json
from pathlib import Path

from src.utils.logger import get_logger


logger = get_logger(__name__)


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_COLUMNS_PATH = (
    PROJECT_ROOT
    / "models"
    / "feature_columns.json"
)


# ============================================================
# Feature Schema
# ============================================================

def load_feature_columns() -> list[str]:
    """
    Load the exact production feature columns.

    Returns
    -------
    list[str]
        Ordered list of production feature names.
    """

    if not FEATURE_COLUMNS_PATH.exists():

        raise FileNotFoundError(
            "Production feature schema not found: "
            f"{FEATURE_COLUMNS_PATH}"
        )

    with FEATURE_COLUMNS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        columns = json.load(file)

    if not isinstance(columns, list):

        raise ValueError(
            "feature_columns.json must contain a list."
        )

    if not columns:

        raise ValueError(
            "Production feature schema is empty."
        )

    if len(columns) != len(set(columns)):

        raise ValueError(
            "Duplicate feature names detected "
            "in feature_columns.json."
        )

    logger.info(
        "Loaded production feature schema: %d features.",
        len(columns),
    )

    return columns