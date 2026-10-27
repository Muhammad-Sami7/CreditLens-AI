"""
============================================================
CreditLens-AI
Production Feature Service
============================================================

Responsible for converting an assembled customer feature
record into the exact feature matrix expected by the
production models.

The trained models expect exactly 297 production features.

This service does NOT retrain models and does NOT recreate
the notebook feature-engineering pipeline.

Its responsibility is production feature alignment,
validation, and consistency.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.features.feature_schema import (
    load_feature_columns,
)
from src.utils.logger import get_logger


logger = get_logger(__name__)


class ProductionFeatureService:
    """
    Production feature preparation service.

    Responsibilities
    ----------------
    - Load production feature contract.
    - Validate incoming features.
    - Add missing features.
    - Remove unexpected features.
    - Preserve feature ordering.
    - Return model-ready DataFrame.
    """

    def __init__(
        self,
        feature_columns: list[str] | None = None,
    ) -> None:

        self.feature_columns = (
            feature_columns
            if feature_columns is not None
            else load_feature_columns()
        )

        if not self.feature_columns:

            raise ValueError(
                "Production feature schema cannot be empty."
            )

        logger.info(
            "ProductionFeatureService initialized "
            "with %d features.",
            len(self.feature_columns),
        )

    # ========================================================
    # Properties
    # ========================================================

    @property
    def feature_count(self) -> int:
        """Return the production feature count."""

        return len(self.feature_columns)

    # ========================================================
    # Validation
    # ========================================================

    def validate_schema(
        self,
        df: pd.DataFrame,
    ) -> dict[str, Any]:
        """
        Validate the incoming DataFrame against the
        production feature contract.

        Returns
        -------
        dict
            Schema validation information.
        """

        if not isinstance(df, pd.DataFrame):

            raise TypeError(
                "Expected pandas.DataFrame."
            )

        incoming_columns = set(df.columns)
        expected_columns = set(self.feature_columns)

        missing = sorted(
            expected_columns - incoming_columns
        )

        unexpected = sorted(
            incoming_columns - expected_columns
        )

        return {
            "valid": len(missing) == 0
            and len(unexpected) == 0,

            "expected_count": self.feature_count,

            "incoming_count": len(df.columns),

            "missing_features": missing,

            "unexpected_features": unexpected,
        }

    # ========================================================
    # Feature Preparation
    # ========================================================

    def prepare(
        self,
        df: pd.DataFrame,
        *,
        allow_missing: bool = True,
    ) -> pd.DataFrame:
        """
        Convert an assembled feature DataFrame into the exact
        production feature matrix expected by the models.

        Parameters
        ----------
        df:
            Incoming customer feature DataFrame.

        allow_missing:
            If True, missing production features are created
            with NaN.

            If False, missing features raise an error.

        Returns
        -------
        pandas.DataFrame
            Model-ready feature DataFrame with exactly the
            production feature columns in the correct order.
        """

        if not isinstance(df, pd.DataFrame):

            raise TypeError(
                "Production features must be provided "
                "as a pandas DataFrame."
            )

        working_df = df.copy()

        # ----------------------------------------------------
        # Remove duplicate columns
        # ----------------------------------------------------

        if working_df.columns.duplicated().any():

            duplicated = (
                working_df.columns[
                    working_df.columns.duplicated()
                ]
                .tolist()
            )

            raise ValueError(
                "Duplicate feature columns detected: "
                f"{duplicated}"
            )

        # ----------------------------------------------------
        # Identify missing features
        # ----------------------------------------------------

        missing_features = [
            column
            for column in self.feature_columns
            if column not in working_df.columns
        ]

        # ----------------------------------------------------
        # Handle missing features
        # ----------------------------------------------------

        if missing_features:

            if not allow_missing:

                raise ValueError(
                    "Missing production features: "
                    f"{missing_features}"
                )

            logger.warning(
                "Creating %d missing production features "
                "with NaN.",
                len(missing_features),
            )

            for column in missing_features:

                working_df[column] = np.nan

        # ----------------------------------------------------
        # Keep ONLY production features
        # ----------------------------------------------------

        working_df = working_df[
            self.feature_columns
        ]

        # ----------------------------------------------------
        # Validate final feature count
        # ----------------------------------------------------

        if len(working_df.columns) != self.feature_count:

            raise ValueError(
                "Production feature count mismatch. "
                f"Expected {self.feature_count}, "
                f"received {len(working_df.columns)}."
            )

        # ----------------------------------------------------
        # Validate feature ordering
        # ----------------------------------------------------

        if list(working_df.columns) != self.feature_columns:

            raise ValueError(
                "Production feature ordering mismatch."
            )

        logger.info(
            "Production feature matrix prepared: "
            "%d rows × %d features.",
            len(working_df),
            len(working_df.columns),
        )

        return working_df

    # ========================================================
    # Single Customer
    # ========================================================

    def prepare_customer(
        self,
        features: dict[str, Any],
        *,
        allow_missing: bool = True,
    ) -> pd.DataFrame:
        """
        Prepare one customer's feature dictionary.

        Returns
        -------
        pandas.DataFrame
            One-row production feature matrix.
        """

        if not isinstance(features, dict):

            raise TypeError(
                "Customer features must be provided "
                "as a dictionary."
            )

        customer_df = pd.DataFrame(
            [features]
        )

        return self.prepare(
            customer_df,
            allow_missing=allow_missing,
        )

    # ========================================================
    # Schema Report
    # ========================================================

    def schema_report(
        self,
        df: pd.DataFrame,
    ) -> dict[str, Any]:
        """
        Generate a human-readable schema report.
        """

        validation = self.validate_schema(df)

        return {
            "production_feature_count": self.feature_count,

            "incoming_feature_count": len(
                df.columns
            ),

            "missing_feature_count": len(
                validation["missing_features"]
            ),

            "unexpected_feature_count": len(
                validation["unexpected_features"]
            ),

            "missing_features": validation[
                "missing_features"
            ],

            "unexpected_features": validation[
                "unexpected_features"
            ],

            "schema_exact_match": validation[
                "valid"
            ],
        }