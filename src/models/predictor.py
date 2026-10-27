"""
CreditLens-AI
Production Predictor

Author: Muhammad Sami
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.models.model_loader import ModelLoader
from src.models.ensemble import EnsemblePredictor
from src.utils.logger import get_logger


logger = get_logger(__name__)


class CreditLensPredictor:
    """
    Production prediction service for CreditLens-AI.

    Loads the production models, ensemble configuration,
    isotonic calibrator, and feature configuration once.
    """

    def __init__(
        self,
        model_loader: ModelLoader | None = None,
    ) -> None:

        logger.info("Loading production models...")

        # --------------------------------------------------
        # Load production assets
        # --------------------------------------------------

        self.loader = (
            model_loader
            if model_loader is not None
            else ModelLoader()
        )

        self.loader.load_models()

        self.assets = self.loader.get_models()

        # --------------------------------------------------
        # Models
        # --------------------------------------------------

        self.lightgbm = self.assets["lightgbm"]
        self.xgboost = self.assets["xgboost"]
        self.catboost = self.assets["catboost"]

        # --------------------------------------------------
        # Probability calibrator
        # --------------------------------------------------

        self.calibrator = self.assets["calibrator"]

        # --------------------------------------------------
        # Feature configuration
        # --------------------------------------------------

        self.feature_columns = self.assets["feature_columns"]

        # --------------------------------------------------
        # Ensemble configuration
        # --------------------------------------------------

        self.ensemble_config = self.assets["ensemble_config"]

        self.ensemble = EnsemblePredictor(
            self.ensemble_config
        )

        logger.info(
            "Production models loaded successfully."
        )

        logger.info(
            "Production feature count: %d",
            len(self.feature_columns),
        )

        logger.info(
            "Ensemble weights: %s",
            self.ensemble.weights,
        )

        logger.info(
            "Production threshold: %.4f",
            self.ensemble.threshold,
        )

    # ======================================================
    # Input Validation
    # ======================================================

    def _validate_input(
        self,
        df: pd.DataFrame,
    ) -> None:
        """
        Validate prediction input before inference.
        """

        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                "Prediction input must be a pandas DataFrame."
            )

        if df.empty:
            raise ValueError(
                "Prediction input DataFrame is empty."
            )

        missing_columns = [
            column
            for column in self.feature_columns
            if column not in df.columns
        ]

        if missing_columns:
            preview = missing_columns[:10]

            raise ValueError(
                "Prediction input is missing "
                f"{len(missing_columns)} required feature(s). "
                f"Examples: {preview}"
            )

    # ======================================================
    # Prediction
    # ======================================================

    def predict(
        self,
        df: pd.DataFrame,
    ) -> dict[str, Any]:
        """
        Generate production credit-risk predictions.

        Returns individual model probabilities,
        raw ensemble probability, calibrated probability,
        final prediction, and production threshold.
        """

        # --------------------------------------------------
        # Validate input
        # --------------------------------------------------

        self._validate_input(df)

        # --------------------------------------------------
        # Select production features
        # --------------------------------------------------

        X = df.loc[
            :,
            self.feature_columns,
        ].copy()

        # --------------------------------------------------
        # Individual model predictions
        # --------------------------------------------------

        lgb_prob = self.lightgbm.predict_proba(
            X
        )[:, 1]

        xgb_prob = self.xgboost.predict_proba(
            X
        )[:, 1]

        cat_prob = self.catboost.predict_proba(
            X
        )[:, 1]

        # --------------------------------------------------
        # Weighted ensemble
        # --------------------------------------------------

        ensemble_prob = (
            self.ensemble.weighted_probability(
                lgb_prob,
                xgb_prob,
                cat_prob,
            )
        )

        # --------------------------------------------------
        # Isotonic calibration
        # --------------------------------------------------

        calibrated_prob = np.asarray(
            self.calibrator.predict(
                ensemble_prob
            ),
            dtype=float,
        )

        calibrated_prob = np.clip(
            calibrated_prob,
            0.0,
            1.0,
        )

        # --------------------------------------------------
        # Final prediction
        # --------------------------------------------------

        prediction = np.asarray(
            [
                self.ensemble.probability_to_prediction(
                    probability
                )
                for probability in calibrated_prob
            ],
            dtype=int,
        )

        # --------------------------------------------------
        # Return prediction results
        # --------------------------------------------------

        return {
            "prediction": prediction,
            "probability": calibrated_prob,

            "lightgbm_probability": lgb_prob,
            "xgboost_probability": xgb_prob,
            "catboost_probability": cat_prob,

            "ensemble_probability": ensemble_prob,

            "threshold": self.ensemble.threshold,
        }