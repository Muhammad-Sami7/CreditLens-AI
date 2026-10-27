"""
============================================================
CreditLens-AI
Production Model Loader
============================================================

Centralized loader for all production model artifacts.

This module is the only application component that directly
interacts with serialized model files.

Responsibilities:
- Load LightGBM, XGBoost, and CatBoost models
- Load the isotonic calibration model
- Load feature schema
- Load ensemble configuration
- Load model metadata
- Validate production artifacts
- Expose a simple module-level load_models() interface

Author: Muhammad Sami
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from src.config import (
    CATBOOST_MODEL_PATH,
    ENSEMBLE_CONFIG_PATH,
    FEATURE_COLUMNS_PATH,
    ISOTONIC_MODEL_PATH,
    LIGHTGBM_MODEL_PATH,
    MODEL_METADATA_PATH,
    XGBOOST_MODEL_PATH,
)


class ModelLoader:
    """
    Singleton-style production model loader.

    Models are loaded only once and kept in memory for the
    lifetime of the application.
    """

    def __init__(self) -> None:
        self.lightgbm_model: LGBMClassifier | None = None
        self.xgboost_model: XGBClassifier | None = None
        self.catboost_model: CatBoostClassifier | Any | None = None

        self.calibrator: Any | None = None

        self.feature_columns: list[str] | None = None
        self.ensemble_config: dict[str, Any] | None = None
        self.metadata: dict[str, Any] | None = None

        self.loaded = False

    # ========================================================
    # Internal helpers
    # ========================================================

    @staticmethod
    def _check_file(path: Path, name: str) -> None:
        """
        Verify that a required production artifact exists.
        """

        if not path.exists():
            raise FileNotFoundError(
                f"{name} not found: {path}"
            )

        if not path.is_file():
            raise FileNotFoundError(
                f"{name} is not a file: {path}"
            )

    @staticmethod
    def _load_json(path: Path, name: str) -> dict[str, Any] | list[Any]:
        """
        Load a JSON artifact safely.
        """

        ModelLoader._check_file(path, name)

        try:
            with open(path, "r", encoding="utf-8") as file:
                return json.load(file)

        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{name} contains invalid JSON: {path}"
            ) from exc

    # ========================================================
    # Main loading logic
    # ========================================================

    def load_models(self) -> None:
        """
        Load all production artifacts once.
        """

        if self.loaded:
            return

        print("=" * 60)
        print("Loading Production Models")
        print("=" * 60)

        # ----------------------------------------------------
        # Validate serialized model files
        # ----------------------------------------------------

        self._check_file(
            LIGHTGBM_MODEL_PATH,
            "LightGBM model",
        )

        self._check_file(
            XGBOOST_MODEL_PATH,
            "XGBoost model",
        )

        self._check_file(
            CATBOOST_MODEL_PATH,
            "CatBoost model",
        )

        self._check_file(
            ISOTONIC_MODEL_PATH,
            "Isotonic calibrator",
        )

        # ----------------------------------------------------
        # LightGBM
        # ----------------------------------------------------

        self.lightgbm_model = joblib.load(
            LIGHTGBM_MODEL_PATH
        )

        print("✓ Regularized LightGBM Loaded")

        # ----------------------------------------------------
        # XGBoost
        # ----------------------------------------------------

        self.xgboost_model = joblib.load(
            XGBOOST_MODEL_PATH
        )

        print("✓ Optimized XGBoost Loaded")

        # ----------------------------------------------------
        # CatBoost
        # ----------------------------------------------------

        self.catboost_model = joblib.load(
            CATBOOST_MODEL_PATH
        )

        print("✓ Optimized CatBoost Loaded")

        # ----------------------------------------------------
        # Isotonic calibration model
        # ----------------------------------------------------

        self.calibrator = joblib.load(
            ISOTONIC_MODEL_PATH
        )

        print("✓ Isotonic Calibrator Loaded")

        # ----------------------------------------------------
        # Feature columns
        # ----------------------------------------------------

        feature_columns = self._load_json(
            FEATURE_COLUMNS_PATH,
            "Feature columns",
        )

        if not isinstance(feature_columns, list):
            raise TypeError(
                "feature_columns.json must contain a JSON list."
            )

        if not feature_columns:
            raise ValueError(
                "feature_columns.json is empty."
            )

        if not all(
            isinstance(column, str)
            for column in feature_columns
        ):
            raise TypeError(
                "All feature column names must be strings."
            )

        self.feature_columns = feature_columns

        print(
            f"✓ Feature Columns Loaded "
            f"({len(self.feature_columns)} features)"
        )

        # ----------------------------------------------------
        # Ensemble configuration
        # ----------------------------------------------------

        ensemble_config = self._load_json(
            ENSEMBLE_CONFIG_PATH,
            "Ensemble configuration",
        )

        if not isinstance(ensemble_config, dict):
            raise TypeError(
                "ensemble_config.json must contain a JSON object."
            )

        self.ensemble_config = ensemble_config

        print("✓ Ensemble Configuration Loaded")

        # ----------------------------------------------------
        # Model metadata
        # ----------------------------------------------------

        metadata = self._load_json(
            MODEL_METADATA_PATH,
            "Model metadata",
        )

        if not isinstance(metadata, dict):
            raise TypeError(
                "model_metadata.json must contain a JSON object."
            )

        self.metadata = metadata

        print("✓ Model Metadata Loaded")

        # ----------------------------------------------------
        # Production schema validation
        # ----------------------------------------------------

        expected_features = (
            self.metadata
            .get("dataset", {})
            .get("total_features")
        )

        if expected_features is not None:
            if expected_features != len(self.feature_columns):
                raise ValueError(
                    "Feature count mismatch: "
                    f"metadata expects {expected_features}, "
                    f"but feature_columns.json contains "
                    f"{len(self.feature_columns)} features."
                )

        # ----------------------------------------------------
        # Production ensemble validation
        # ----------------------------------------------------

        ensemble_section = self.ensemble_config.get(
            "ensemble",
            {},
        )

        weights = ensemble_section.get(
            "weights",
            {},
        )

        threshold = self.ensemble_config.get(
            "threshold"
        )

        if not weights:
            raise ValueError(
                "No ensemble weights found in ensemble_config.json."
            )

        required_weights = {
            "lightgbm",
            "xgboost",
            "catboost",
        }

        if not required_weights.issubset(weights.keys()):
            raise ValueError(
                "Ensemble configuration must contain weights for "
                "lightgbm, xgboost, and catboost."
            )

        weight_sum = sum(
            float(weights[name])
            for name in required_weights
        )

        if not abs(weight_sum - 1.0) < 1e-6:
            raise ValueError(
                f"Ensemble weights must sum to 1.0. "
                f"Current sum: {weight_sum}"
            )

        if threshold is None:
            raise ValueError(
                "Production ensemble threshold is missing."
            )

        threshold = float(threshold)

        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                f"Invalid ensemble threshold: {threshold}"
            )

        # ----------------------------------------------------
        # Mark loader as ready
        # ----------------------------------------------------

        self.loaded = True

        print("=" * 60)
        print("Production Environment Ready")
        print(f"Features : {len(self.feature_columns)}")
        print(
            "Weights  : "
            f"LGB={weights['lightgbm']}, "
            f"XGB={weights['xgboost']}, "
            f"CAT={weights['catboost']}"
        )
        print(f"Threshold: {threshold}")
        print("=" * 60)

    # ========================================================
    # Public access
    # ========================================================

    def get_models(self) -> dict[str, Any]:
        """
        Return all production assets.
        """

        if not self.loaded:
            self.load_models()

        return {
            "lightgbm": self.lightgbm_model,
            "xgboost": self.xgboost_model,
            "catboost": self.catboost_model,
            "calibrator": self.calibrator,
            "feature_columns": self.feature_columns,
            "ensemble_config": self.ensemble_config,
            "metadata": self.metadata,
        }


# ============================================================
# Application-level singleton
# ============================================================

_model_loader = ModelLoader()


def load_models() -> dict[str, Any]:
    """
    Application-level interface for loading production models.

    This is the function used by predictor.py and other
    application services.
    """

    return _model_loader.get_models()