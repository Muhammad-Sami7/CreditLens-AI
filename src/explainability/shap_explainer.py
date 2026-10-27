"""
============================================================
CreditLens-AI
Production SHAP Explainer
============================================================

Provides production SHAP explanations for the CreditLens-AI
credit-risk prediction system.

Architecture:

Customer Features
        |
Production Feature Service
        |
297 Production Features
        |
Production Model Loader
        |
LightGBM + XGBoost + CatBoost
        |
SHAP TreeExplainer (one per model)
        |
Weighted combination (production ensemble weights)
        |
Feature Contributions
        |
Positive / Negative Risk Factors

Author: Muhammad Sami
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap

from src.models.model_loader import ModelLoader
from src.utils.logger import get_logger


logger = get_logger(__name__)


class SHAPExplainer:
    """
    Production SHAP explainability service.

    The production prediction is a weighted ensemble of LightGBM,
    XGBoost and CatBoost, so the explanation covers all three.

    Each model is explained with SHAP TreeExplainer. A feature's
    ensemble contribution is the weighted sum of its three
    per-model contributions, using the same weights as the
    production ensemble (models/ensemble_config.json).

    Units: every model's SHAP values are in that model's raw
    log-odds space, and the weighted sum is in the same space. The
    result shows which features pushed the ensemble toward or away
    from risk and how strongly. It is not an additive breakdown of
    the final calibrated probability (calibration is a separate,
    non-linear step). The per-model values and the number of models
    that agree on each feature's direction are returned too.
    """

    def __init__(
        self,
        model_loader: ModelLoader | None = None,
    ) -> None:

        self.model_loader = (
            model_loader
            if model_loader is not None
            else ModelLoader()
        )

        # Load production models if they have not
        # already been loaded.
        try:
            models = self.model_loader.get_models()
        except Exception:
            self.model_loader.load_models()
            models = self.model_loader.get_models()

        # ----------------------------------------------------
        # Production LightGBM model
        # ----------------------------------------------------

        if "lightgbm" not in models:

            raise ValueError(
                "Production LightGBM model was not found "
                "in ModelLoader."
            )

        self.model = models["lightgbm"]

        # XGBoost and CatBoost are explained as well, because the
        # production decision is made by the ensemble of all three.
        for required in ("xgboost", "catboost"):

            if required not in models:

                raise ValueError(
                    f"Production {required} model was not found "
                    "in ModelLoader."
                )

        self.models = {
            "lightgbm": models["lightgbm"],
            "xgboost": models["xgboost"],
            "catboost": models["catboost"],
        }

        # Ensemble weights come from the SAME production config the
        # predictor uses, so explanation and prediction always agree
        # on how much each model counts.
        configured = (
            (models.get("ensemble_config") or {})
            .get("ensemble", {})
            .get("weights")
        )

        if not configured:

            raise ValueError(
                "Ensemble weights were not found in "
                "ensemble_config.json."
            )

        weights = {
            name: float(configured[name])
            for name in self.models
        }

        total_weight = sum(weights.values())

        if total_weight <= 0:

            raise ValueError(
                "Ensemble weights must sum to a positive value."
            )

        # Normalised so the combined contribution stays on the same
        # scale as a single model's contribution.
        self.weights = {
            name: weight / total_weight
            for name, weight in weights.items()
        }

        # ----------------------------------------------------
        # Production feature schema
        # ----------------------------------------------------

        if "feature_columns" not in models:

            raise ValueError(
                "Production feature columns were not found "
                "in ModelLoader."
            )

        self.feature_columns = list(
            models["feature_columns"]
        )

        if len(self.feature_columns) != 297:

            raise ValueError(
                "Production feature schema must contain "
                f"297 features. Found "
                f"{len(self.feature_columns)}."
            )

        # ----------------------------------------------------
        # SHAP TreeExplainer
        # ----------------------------------------------------

        logger.info(
            "Initializing SHAP TreeExplainer..."
        )

        # One TreeExplainer per tree model. The ensemble itself and
        # the calibrator are not tree models, so they are explained
        # through their members. Contributions are in each model's raw
        # log-odds space.
        self.explainers = {
            name: shap.TreeExplainer(model)
            for name, model in self.models.items()
        }

        # Kept for backward compatibility (LightGBM explainer).
        self.explainer = self.explainers["lightgbm"]

        logger.info(
            "SHAP explainer initialized with "
            "%d production features.",
            len(self.feature_columns),
        )

    # ========================================================
    # Feature Preparation
    # ========================================================

    def _prepare_features(
        self,
        features: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Validate and order the production feature matrix.
        """

        if not isinstance(features, pd.DataFrame):

            raise TypeError(
                "features must be a pandas DataFrame."
            )

        missing_features = [
            feature
            for feature in self.feature_columns
            if feature not in features.columns
        ]

        if missing_features:

            raise ValueError(
                "Missing production features: "
                + ", ".join(
                    missing_features[:20]
                )
                + (
                    "..."
                    if len(missing_features) > 20
                    else ""
                )
            )

        # Column order must match training exactly: tree models read
        # features by position, so a different order would give
        # wrong scores without raising an error.
        # Keep exactly the production feature schema
        # and preserve production ordering.
        prepared = features[
            self.feature_columns
        ].copy()

        # SHAP/model input should be numeric.
        #
        # ProductionFeatureService is expected to have
        # already performed the necessary preprocessing.
        for column in prepared.columns:

            if not pd.api.types.is_numeric_dtype(
                prepared[column]
            ):

                try:

                    prepared[column] = pd.to_numeric(
                        prepared[column],
                        errors="coerce",
                    )

                except Exception as exc:

                    raise ValueError(
                        f"Feature '{column}' could not "
                        "be converted to numeric."
                    ) from exc

        return prepared

    # ========================================================
    # SHAP Value Calculation
    # ========================================================

    def _calculate_shap_values(
        self,
        features: pd.DataFrame,
        model_name: str = "lightgbm",
    ) -> np.ndarray:
        """
        Calculate SHAP values for one production model.
        """

        # Depending on the SHAP version, binary classifiers return a
        # list of two arrays, one 3-D array, or one 2-D array. The code
        # below normalises all of them to (rows, features) for the
        # positive (default-risk) class.
        shap_values = self.explainers[model_name].shap_values(
            features
        )

        # ----------------------------------------------------
        # Handle binary classification output formats.
        # ----------------------------------------------------

        if isinstance(shap_values, list):

            if len(shap_values) == 2:

                # Positive-class explanation.
                shap_values = shap_values[1]

            elif len(shap_values) == 1:

                shap_values = shap_values[0]

            else:

                raise ValueError(
                    "Unexpected SHAP output format."
                )

        shap_values = np.asarray(
            shap_values
        )

        # Some SHAP/model combinations may return
        # (rows, features, classes).
        if shap_values.ndim == 3:

            if shap_values.shape[-1] == 2:

                shap_values = shap_values[:, :, 1]

            else:

                raise ValueError(
                    "Unexpected 3-dimensional SHAP "
                    "output shape: "
                    f"{shap_values.shape}"
                )

        if shap_values.ndim != 2:

            raise ValueError(
                "Unexpected SHAP output dimensions: "
                f"{shap_values.shape}"
            )

        if shap_values.shape[1] != len(
            self.feature_columns
        ):

            raise ValueError(
                "SHAP feature count mismatch. "
                f"Expected {len(self.feature_columns)}, "
                f"received {shap_values.shape[1]}."
            )

        return shap_values

    # ========================================================
    # Explanation
    # ========================================================

    def explain(
        self,
        features: pd.DataFrame,
        top_n: int = 10,
    ) -> dict[str, Any]:
        """
        Generate a production SHAP explanation.

        Parameters
        ----------
        features:
            Production feature matrix containing the
            297 expected features.

        top_n:
            Number of positive and negative factors
            to return.

        Returns
        -------
        dict
            Production SHAP explanation.
        """

        if top_n <= 0:

            raise ValueError(
                "top_n must be greater than zero."
            )

        prepared = self._prepare_features(
            features
        )

        # SHAP for each of the three models, one customer (row 0).
        model_rows = {
            name: self._calculate_shap_values(
                prepared,
                name,
            )[0]
            for name in self.models
        }

        # Ensemble contribution = weighted sum of the per-model
        # contributions, using the production ensemble weights.
        row_shap = sum(
            self.weights[name] * model_rows[name]
            for name in self.models
        )

        row_features = prepared.iloc[0]

        # ----------------------------------------------------
        # Build feature contribution records.
        # ----------------------------------------------------

        contributions: list[dict[str, Any]] = []

        for index, feature in enumerate(
            self.feature_columns
        ):

            shap_value = float(
                row_shap[index]
            )

            feature_value = row_features[
                feature
            ]

            # Missing values are reported as None so the API and UI
            # can show "no data" instead of NaN.
            if pd.isna(feature_value):

                feature_value_output = None

            elif isinstance(
                feature_value,
                (np.integer, np.floating),
            ):

                feature_value_output = float(
                    feature_value
                )

            else:

                feature_value_output = feature_value

            per_model = {
                name: float(model_rows[name][index])
                for name in self.models
            }

            direction = (
                "positive"
                if shap_value > 0
                else "negative"
                if shap_value < 0
                else "neutral"
            )

            # How many of the three models push this feature the
            # same way as the ensemble does. 3 = full agreement.
            if direction == "neutral":

                agreeing = 0

            else:

                sign = 1 if direction == "positive" else -1

                agreeing = sum(
                    1
                    for value in per_model.values()
                    if value * sign > 0
                )

            contributions.append(
                {
                    "feature": feature,
                    "feature_value": (
                        feature_value_output
                    ),
                    "shap_value": shap_value,
                    "abs_shap_value": abs(
                        shap_value
                    ),
                    "direction": direction,
                    "model_shap_values": per_model,
                    "models_agreeing": agreeing,
                }
            )

        # ----------------------------------------------------
        # Sort by absolute SHAP magnitude.
        # ----------------------------------------------------

        # `top_factors` = largest absolute influence in either
        # direction; the positive/negative lists below are ranked
        # separately by signed size.
        contributions.sort(
            key=lambda item: item[
                "abs_shap_value"
            ],
            reverse=True,
        )

        # ----------------------------------------------------
        # Positive risk factors.
        #
        # Positive SHAP contribution means the feature
        # pushes the model toward the positive/risk class.
        # ----------------------------------------------------

        positive_factors = sorted(
            [
                item
                for item in contributions
                if item["shap_value"] > 0
            ],
            key=lambda item: item[
                "shap_value"
            ],
            reverse=True,
        )[:top_n]

        # ----------------------------------------------------
        # Negative risk factors.
        #
        # Negative SHAP contribution means the feature
        # pushes the prediction away from the positive
        # risk class.
        # ----------------------------------------------------

        negative_factors = sorted(
            [
                item
                for item in contributions
                if item["shap_value"] < 0
            ],
            key=lambda item: item[
                "shap_value"
            ],
        )[:top_n]

        # ----------------------------------------------------
        # Base value.
        # ----------------------------------------------------

        # Weighted base value (the model's average output), on the
        # same log-odds scale as the contributions.
        base_value = sum(
            self.weights[name]
            * self._base_value(self.explainers[name])
            for name in self.models
        )

        # ----------------------------------------------------
        # Return production explanation.
        # ----------------------------------------------------

        return {
            "base_value": base_value,

            "shap_values": row_shap.tolist(),

            "feature_contributions": contributions,

            "positive_factors": positive_factors,

            "negative_factors": negative_factors,

            "top_factors": contributions[
                :top_n
            ],

            "feature_count": len(
                self.feature_columns
            ),

            "models_explained": list(self.models),

            "model_weights": dict(self.weights),
        }

    @staticmethod
    def _base_value(explainer: Any) -> float:
        """
        Return the positive-class base value of one explainer.

        SHAP may return a scalar or an array (one value per class).
        """

        base_value = explainer.expected_value

        if isinstance(base_value, (list, np.ndarray)):

            base_array = np.asarray(base_value).reshape(-1)

            return float(
                base_array[1] if len(base_array) >= 2
                else base_array[0]
            )

        return float(base_value)

    # ========================================================
    # Convenience Method
    # ========================================================

    def explain_customer(
        self,
        customer_features: pd.DataFrame,
        top_n: int = 10,
    ) -> dict[str, Any]:
        """
        Alias for explain() intended for application-level
        customer explanation calls.
        """

        return self.explain(
            customer_features,
            top_n=top_n,
        )

