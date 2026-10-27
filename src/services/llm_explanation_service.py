from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field

load_dotenv()


class LLMExplanation(BaseModel):
    """Structured explanation returned by the LLM."""

    summary: str = Field(
        description="A concise explanation of the final credit-risk result."
    )

    risk_explanation: str = Field(
        description=(
            "Why the customer received the given risk level, "
            "based only on supplied evidence."
        )
    )

    key_risk_factors: list[str] = Field(
        description="The most important factors contributing to risk."
    )

    protective_factors: list[str] = Field(
        description="Factors that support a lower-risk interpretation."
    )

    analyst_actions: list[str] = Field(
        description="Practical actions a loan analyst may consider next."
    )


class LLMExplanationService:
    """
    Generates a natural-language explanation of an existing
    CreditLens-AI risk prediction.

    The LLM:
    - does NOT make the risk decision
    - does NOT change the prediction
    - does NOT override the calibrated threshold
    - only explains the existing ML + SHAP result
    """

    DEFAULT_MODEL = "gemini-3.6-flash"

    def __init__(self, model: str | None = None) -> None:
        self.api_key = os.getenv("GEMINI_API_KEY")

        if not self.api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not configured in the environment."
            )

        self.model = model or os.getenv(
            "GEMINI_MODEL",
            self.DEFAULT_MODEL,
        )

        self.client = genai.Client(api_key=self.api_key)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_prediction_consistency(
        *,
        prediction: int,
        probability: float,
        threshold: float,
        risk_level: str,
    ) -> None:
        """Validate that the supplied result is internally consistent."""

        expected_prediction = int(probability >= threshold)

        if prediction != expected_prediction:
            raise ValueError(
                "Prediction is inconsistent with probability and threshold: "
                f"prediction={prediction}, "
                f"probability={probability:.6f}, "
                f"threshold={threshold:.6f}"
            )

        expected_risk = (
            "High Risk"
            if probability >= threshold
            else "Low Risk"
        )

        if risk_level != expected_risk:
            raise ValueError(
                "Risk level is inconsistent with probability and threshold: "
                f"risk_level={risk_level}, "
                f"expected={expected_risk}"
            )

    # ------------------------------------------------------------------
    # SHAP normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_top_factors(
        shap_result: dict[str, Any] | None,
        prediction_result: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Extract SHAP factors from the dedicated SHAP result.

        The service also supports prediction results that already contain
        top_factors so the prediction pipeline can be used directly.
        """

        if shap_result:
            top_factors = shap_result.get("top_factors")

            if isinstance(top_factors, list):
                return top_factors

            feature_contributions = shap_result.get(
                "feature_contributions"
            )

            if isinstance(feature_contributions, list):
                return feature_contributions

        top_factors = prediction_result.get("top_factors")

        if isinstance(top_factors, list):
            return top_factors

        return []

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    @classmethod
    def _build_prompt(
        cls,
        *,
        prediction: int,
        risk_level: str,
        probability: float,
        ensemble_probability: float,
        threshold: float,
        model_probabilities: dict[str, float],
        shap_result: dict[str, Any] | None,
        prediction_result: dict[str, Any],
        customer_context: dict[str, Any] | None = None,
    ) -> str:
        """Build the evidence-grounded LLM prompt."""

        top_factors = cls._extract_top_factors(
            shap_result,
            prediction_result,
        )

        factors_text: list[dict[str, Any]] = []

        for factor in top_factors:
            feature_name = factor.get(
                "feature_name",
                factor.get("feature", "Unknown feature"),
            )

            feature_value = factor.get(
                "feature_value",
                factor.get("value"),
            )

            shap_value = factor.get(
                "shap_value",
                0.0,
            )

            direction = factor.get(
                "direction",
                "neutral",
            )

            factors_text.append(
                {
                    "feature": feature_name,
                    "value": feature_value,
                    "shap_value": float(shap_value),
                    "direction": direction,
                }
            )

        evidence = {
            "final_prediction": prediction,
            "risk_level": risk_level,
            "calibrated_probability": probability,
            "ensemble_probability_before_calibration": (
                ensemble_probability
            ),
            "production_threshold": threshold,
            "individual_model_probabilities": (
                model_probabilities
            ),
            "top_shap_factors": factors_text,
            "customer_context": customer_context or {},
        }

        return f"""
You are the explanation layer of CreditLens-AI,
an AI-assisted credit-risk platform.

Your role is ONLY to explain an already-computed
machine-learning result.

You MUST NOT:
- change the prediction
- change the risk level
- recommend a different classification
- invent customer information
- invent financial facts
- claim that a feature is causal
- override the production threshold
- make a new credit decision

The production ML system has already made the decision.

Your job is to explain the existing result clearly
to a loan analyst.

SHAP interpretation:
- A positive SHAP contribution means the feature pushed
  the model toward higher predicted risk.
- A negative SHAP contribution means the feature pushed
  the model toward lower predicted risk.
- SHAP values represent model contribution, not causality.
- Do not describe a feature as inherently good or bad.
- Interpret each factor using its supplied value and direction.

The explanation must be:
- concise
- professional
- evidence-based
- understandable to a loan analyst
- consistent with the supplied prediction
- free from unsupported assumptions

Return ONLY the requested structured JSON object.

CreditLens-AI evidence:

{json.dumps(evidence, indent=2, default=str)}
"""

    # ------------------------------------------------------------------
    # Gemini generation
    # ------------------------------------------------------------------

    def _generate_explanation(
        self,
        prompt: str,
    ) -> LLMExplanation:
        """Generate the structured explanation using Gemini."""

        interaction = self.client.interactions.create(
            model=self.model,
            input=prompt,
            response_format={
                "type": "text",
                "mime_type": "application/json",
                "schema": LLMExplanation.model_json_schema(),
            },
        )

        output_text = getattr(
            interaction,
            "output_text",
            None,
        )

        if not output_text:
            raise RuntimeError(
                "Gemini returned an empty response."
            )

        try:
            return LLMExplanation.model_validate_json(
                output_text
            )
        except Exception as exc:
            raise RuntimeError(
                "Gemini returned invalid structured output."
            ) from exc

    # ------------------------------------------------------------------
    # Main explanation method
    # ------------------------------------------------------------------

    async def explain(
        self,
        *,
        prediction: int,
        risk_level: str,
        probability: float,
        ensemble_probability: float,
        threshold: float,
        model_probabilities: dict[str, float],
        shap_result: dict[str, Any] | None = None,
        prediction_result: dict[str, Any] | None = None,
        customer_context: dict[str, Any] | None = None,
    ) -> LLMExplanation:
        """
        Generate a structured explanation.

        SHAP can be supplied separately through shap_result.
        """

        prediction_result = prediction_result or {}

        self._validate_prediction_consistency(
            prediction=prediction,
            probability=probability,
            threshold=threshold,
            risk_level=risk_level,
        )

        prompt = self._build_prompt(
            prediction=prediction,
            risk_level=risk_level,
            probability=probability,
            ensemble_probability=ensemble_probability,
            threshold=threshold,
            model_probabilities=model_probabilities,
            shap_result=shap_result,
            prediction_result=prediction_result,
            customer_context=customer_context,
        )

        return await asyncio.to_thread(
            self._generate_explanation,
            prompt,
        )

    # ------------------------------------------------------------------
    # Prediction-result wrapper
    # ------------------------------------------------------------------

    async def explain_prediction(
        self,
        prediction_result: dict[str, Any],
        *,
        shap_result: dict[str, Any] | None = None,
        customer_context: dict[str, Any] | None = None,
    ) -> LLMExplanation:
        """
        Generate an explanation from a CreditLens-AI prediction result.

        SHAP can be passed independently using shap_result.
        """

        model_probabilities = prediction_result.get(
            "model_probabilities",
            {},
        )

        return await self.explain(
            prediction=int(
                prediction_result["prediction"]
            ),
            risk_level=str(
                prediction_result["risk_level"]
            ),
            probability=float(
                prediction_result["probability"]
            ),
            ensemble_probability=float(
                prediction_result["ensemble_probability"]
            ),
            threshold=float(
                prediction_result["threshold"]
            ),
            model_probabilities={
                "lightgbm": float(
                    model_probabilities["lightgbm"]
                ),
                "xgboost": float(
                    model_probabilities["xgboost"]
                ),
                "catboost": float(
                    model_probabilities["catboost"]
                ),
            },
            shap_result=shap_result,
            prediction_result=prediction_result,
            customer_context=customer_context,
        )