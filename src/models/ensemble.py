"""
============================================================
CreditLens-AI
Ensemble Prediction
============================================================

Production ensemble logic for combining model probabilities.

The ensemble configuration is loaded from:
    models/ensemble_config.json

Production configuration currently defines:
    LightGBM  : 0.35
    XGBoost   : 0.35
    CatBoost  : 0.30
    Threshold : 0.18 (calibrated production threshold)

Author: Muhammad Sami
"""

from __future__ import annotations

from typing import Any

import numpy as np


class EnsemblePredictor:
    """
    Weighted soft-voting ensemble.

    The ensemble combines probability outputs from:
        - LightGBM
        - XGBoost
        - CatBoost

    Weights and threshold can be supplied from the production
    ensemble configuration.
    """

    DEFAULT_WEIGHTS = {
        "lightgbm": 0.35,
        "xgboost": 0.35,
        "catboost": 0.30,
    }

    DEFAULT_THRESHOLD = 0.18

    def __init__(
        self,
        ensemble_config: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the ensemble.

        Parameters
        ----------
        ensemble_config:
            Production configuration loaded from
            models/ensemble_config.json.
        """

        self.weights = self.DEFAULT_WEIGHTS.copy()
        self.threshold = self.DEFAULT_THRESHOLD

        if ensemble_config is not None:
            self._load_config(ensemble_config)

        self._validate_configuration()

    # ========================================================
    # Configuration
    # ========================================================

    def _load_config(
        self,
        ensemble_config: dict[str, Any],
    ) -> None:
        """
        Load ensemble weights and threshold from production
        configuration.
        """

        ensemble_section = ensemble_config.get(
            "ensemble",
            {},
        )

        configured_weights = ensemble_section.get(
            "weights"
        )

        if configured_weights is not None:
            self.weights = {
                "lightgbm": float(
                    configured_weights["lightgbm"]
                ),
                "xgboost": float(
                    configured_weights["xgboost"]
                ),
                "catboost": float(
                    configured_weights["catboost"]
                ),
            }

        configured_threshold = ensemble_config.get(
            "threshold"
        )

        if configured_threshold is not None:
            self.threshold = float(
                configured_threshold
            )

    # ========================================================
    # Validation
    # ========================================================

    def _validate_configuration(self) -> None:
        """
        Validate ensemble weights and threshold.
        """

        required_models = {
            "lightgbm",
            "xgboost",
            "catboost",
        }

        if set(self.weights.keys()) != required_models:
            raise ValueError(
                "Ensemble weights must contain exactly "
                "lightgbm, xgboost, and catboost."
            )

        if any(
            weight < 0
            for weight in self.weights.values()
        ):
            raise ValueError(
                "Ensemble weights cannot be negative."
            )

        weight_sum = sum(
            self.weights.values()
        )

        if not np.isclose(
            weight_sum,
            1.0,
            atol=1e-6,
        ):
            raise ValueError(
                "Ensemble weights must sum to 1.0. "
                f"Current sum: {weight_sum}"
            )

        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(
                "Ensemble threshold must be between "
                f"0 and 1. Current value: {self.threshold}"
            )

    # ========================================================
    # Probability ensemble
    # ========================================================

    def weighted_probability(
        self,
        lgb: np.ndarray,
        xgb: np.ndarray,
        cat: np.ndarray,
    ) -> np.ndarray:
        """
        Combine model probabilities using weighted soft voting.

        Parameters
        ----------
        lgb:
            LightGBM positive-class probabilities.

        xgb:
            XGBoost positive-class probabilities.

        cat:
            CatBoost positive-class probabilities.

        Returns
        -------
        np.ndarray
            Weighted ensemble probabilities.
        """

        lgb = np.asarray(lgb, dtype=float)
        xgb = np.asarray(xgb, dtype=float)
        cat = np.asarray(cat, dtype=float)

        if not (
            lgb.shape
            == xgb.shape
            == cat.shape
        ):
            raise ValueError(
                "LightGBM, XGBoost, and CatBoost "
                "probability arrays must have identical shapes."
            )

        if np.any(~np.isfinite(lgb)):
            raise ValueError(
                "LightGBM probabilities contain invalid values."
            )

        if np.any(~np.isfinite(xgb)):
            raise ValueError(
                "XGBoost probabilities contain invalid values."
            )

        if np.any(~np.isfinite(cat)):
            raise ValueError(
                "CatBoost probabilities contain invalid values."
            )

        if np.any((lgb < 0) | (lgb > 1)):
            raise ValueError(
                "LightGBM probabilities must be between 0 and 1."
            )

        if np.any((xgb < 0) | (xgb > 1)):
            raise ValueError(
                "XGBoost probabilities must be between 0 and 1."
            )

        if np.any((cat < 0) | (cat > 1)):
            raise ValueError(
                "CatBoost probabilities must be between 0 and 1."
            )

        ensemble_probability = (
            lgb * self.weights["lightgbm"]
            + xgb * self.weights["xgboost"]
            + cat * self.weights["catboost"]
        )

        return np.clip(
            ensemble_probability,
            0.0,
            1.0,
        )

    # ========================================================
    # Probability → prediction
    # ========================================================

    def probability_to_prediction(
        self,
        probability: float,
    ) -> int:
        """
        Convert a probability into a binary risk prediction.

        Production threshold:
            0.18

        Returns
        -------
        int
            1 = higher predicted credit risk
            0 = lower predicted credit risk
        """

        probability = float(probability)

        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                "Probability must be between 0 and 1."
            )

        return int(
            probability >= self.threshold
        )