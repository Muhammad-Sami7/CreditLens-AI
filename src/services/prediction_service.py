"""
============================================================
CreditLens-AI
Prediction Service
============================================================

Application-level service sitting between the API,
production prediction engine, SHAP explainability,
business risk engine, and PostgreSQL persistence.

Author: Muhammad Sami
"""
from __future__ import annotations

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from src.business.risk_engine import RiskEngine
from src.database.repository import (
    create_model_prediction,
    create_prediction,
    create_risk_factor,
    get_or_create_customer,
)
from src.explainability.shap_explainer import SHAPExplainer
from src.features.feature_service import ProductionFeatureService
from src.models.predictor import CreditLensPredictor
from src.services.llm_explanation_service import (
    LLMExplanation,
    LLMExplanationService,
)
from src.utils.logger import get_logger


logger = get_logger(__name__)


class PredictionService:
    """
    Application service for complete credit-risk prediction.

    Responsibilities
    ----------------
    - Prepare production features.
    - Run the production ML predictor.
    - Apply business risk assessment.
    - Generate SHAP explanations.
    - Generate LLM-based natural-language explanations.
    - Persist prediction results.
    - Persist individual model probabilities.
    - Persist top SHAP risk factors.

    Decision authority
    ------------------
    The ML prediction and calibrated production threshold
    determine the final credit-risk decision.

    SHAP and the LLM are explanation layers only.
    The LLM cannot change the prediction or risk decision.
    """

    def __init__(
        self,
        predictor: CreditLensPredictor | None = None,
        feature_service: ProductionFeatureService | None = None,
        shap_explainer: SHAPExplainer | None = None,
        llm_explanation_service: LLMExplanationService | None = None,
    ) -> None:

        self.predictor = (
            predictor
            if predictor is not None
            else CreditLensPredictor()
        )

        self.feature_service = (
            feature_service
            if feature_service is not None
            else ProductionFeatureService()
        )

        self.shap_explainer = (
            shap_explainer
            if shap_explainer is not None
            else SHAPExplainer()
        )

        self.llm_explanation_service = (
            llm_explanation_service
            if llm_explanation_service is not None
            else LLMExplanationService()
        )

        self.risk_engine = RiskEngine(
            threshold=self.predictor.ensemble.threshold
        )

        logger.info(
            "Prediction service initialized."
        )

    # ========================================================
    # Prediction
    # ========================================================

    async def predict(
        self,
        session: AsyncSession,
        *,
        customer_id: str | int | float,
        features: dict[str, Any],
        top_n: int = 10,
    ) -> dict[str, Any]:
        """
        Generate, explain, and persist a complete
        credit-risk prediction.

        Parameters
        ----------
        session:
            Active asynchronous SQLAlchemy session.

        customer_id:
            Customer identifier.

        features:
            Assembled customer feature dictionary.

        top_n:
            Number of top SHAP factors to generate and persist.

        Returns
        -------
        dict
            Complete application-level prediction result,
            including ML prediction, SHAP explanation,
            and LLM explanation.
        """

        if top_n <= 0:
            raise ValueError(
                "top_n must be greater than zero."
            )

        customer_id_str = str(customer_id)

        # ----------------------------------------------------
        # 1. Prepare production features
        # ----------------------------------------------------

        # Validate and order the 297 production features. With
        # allow_missing=True, absent fields become NaN ("no data"),
        # which the tree models handle natively; the API reports how
        # many were missing so the user can judge reliability.
        customer_df = (
            self.feature_service.prepare_customer(
                features,
                allow_missing=True,
            )
        )

        logger.info(
            "Production features prepared for customer %s.",
            customer_id_str,
        )

        # ----------------------------------------------------
        # 2. Run production ML prediction
        # ----------------------------------------------------

        # `probability` is the CALIBRATED probability (isotonic
        # calibration applied to the weighted ensemble). It is the value
        # the business decision uses. `ensemble_probability` is the
        # weighted average BEFORE calibration, and the per-model values
        # are the raw scores of each model. Only the calibrated value
        # is compared with the threshold.
        result = self.predictor.predict(
            customer_df
        )

        prediction = int(
            result["prediction"][0]
        )

        probability = float(
            result["probability"][0]
        )

        ensemble_probability = float(
            result["ensemble_probability"][0]
        )

        threshold = float(
            result["threshold"]
        )

        model_probabilities = {
            "lightgbm": float(
                result["lightgbm_probability"][0]
            ),
            "xgboost": float(
                result["xgboost_probability"][0]
            ),
            "catboost": float(
                result["catboost_probability"][0]
            ),
        }

        # ----------------------------------------------------
        # 3. Business risk assessment
        # ----------------------------------------------------

        # The decision comes from the RiskEngine alone. Nothing after
        # this line (SHAP, Gemini, persistence) may change it.
        assessment = self.risk_engine.assess(
            probability
        )

        # ----------------------------------------------------
        # 4. Generate SHAP explanation
        # ----------------------------------------------------

        # SHAP covers all three ensemble models (LightGBM, XGBoost,
        # CatBoost), combined with the production ensemble weights, in
        # raw log-odds units. It is evidence for HOW the ensemble scored
        # this customer; it is not an exact decomposition of the
        # calibrated probability. The UI labels these as "contribution".
        shap_result = (
            self.shap_explainer.explain_customer(
                customer_df,
                top_n=top_n,
            )
        )

        top_factors = shap_result[
            "top_factors"
        ]

        logger.info(
            "SHAP explanation generated for customer %s.",
            customer_id_str,
        )

        # ----------------------------------------------------
        # 5. Generate LLM explanation
        # ----------------------------------------------------
        #
        # IMPORTANT:
        # The LLM receives the already-computed ML decision
        # and the actual SHAP evidence.
        #
        # The LLM does NOT make or modify the decision.
        # ----------------------------------------------------

        prediction_result = {
            "prediction": prediction,
            "risk_level": assessment.risk_level,
            "probability": probability,
            "ensemble_probability": ensemble_probability,
            "threshold": assessment.threshold,
            "model_probabilities": model_probabilities,
        }

        # The explanation is a convenience. If the explanation service
        # fails (network, quota, a 503 from the provider), the decision,
        # SHAP evidence and audit record are still valid and are still
        # returned; only the written summary is replaced by an honest
        # notice. The decision never depends on this step.
        try:

            llm_explanation = (
                await self.llm_explanation_service.explain_prediction(
                    prediction_result=prediction_result,
                    shap_result=shap_result,
                )
            )

            logger.info(
                "LLM explanation generated for customer %s.",
                customer_id_str,
            )

        except Exception:

            logger.exception(
                "LLM explanation unavailable for customer %s; "
                "returning the prediction without it.",
                customer_id_str,
            )

            llm_explanation = LLMExplanation(
                summary=(
                    "The AI-written explanation is temporarily "
                    "unavailable. The risk estimate, decision and "
                    "model evidence shown are unaffected."
                ),
                risk_explanation=(
                    "Please use the model evidence (risk factors and "
                    "per-model scores) for this assessment, or run it "
                    "again later to get the written explanation."
                ),
                key_risk_factors=[],
                protective_factors=[],
                analyst_actions=[],
            )

        # ----------------------------------------------------
        # 6. Persist customer
        # ----------------------------------------------------

        customer = await get_or_create_customer(
            session,
            customer_id_str,
        )

        # ----------------------------------------------------
        # 7. Persist final prediction
        # ----------------------------------------------------

        prediction_record = await create_prediction(
            session,
            customer_id=customer.customer_id,
            prediction=prediction,
            risk_level=assessment.risk_level,
            decision=assessment.decision,
            probability=probability,
            ensemble_probability=ensemble_probability,
            threshold=assessment.threshold,
        )

        # ----------------------------------------------------
        # 8. Persist individual model predictions
        # ----------------------------------------------------

        for model_name, model_probability in (
            model_probabilities.items()
        ):

            # Per-model flag stored for audit only. It compares each
            # model's RAW score with the production threshold, so it
            # can differ from the final decision (which uses the
            # calibrated ensemble probability).
            model_prediction = int(
                model_probability
                >= assessment.threshold
            )

            await create_model_prediction(
                session,
                prediction_id=prediction_record.id,
                model_name=model_name,
                probability=model_probability,
                prediction=model_prediction,
            )

        # ----------------------------------------------------
        # 9. Persist top SHAP risk factors
        # ----------------------------------------------------

        for rank, factor in enumerate(
            top_factors,
            start=1,
        ):

            feature_value = factor.get(
                "feature_value"
            )

            if feature_value is not None:
                feature_value_str = str(
                    feature_value
                )
            else:
                feature_value_str = None

            await create_risk_factor(
                session,
                prediction_id=prediction_record.id,
                feature_name=str(
                    factor["feature"]
                ),
                feature_value=feature_value_str,
                shap_value=float(
                    factor["shap_value"]
                ),
                direction=str(
                    factor["direction"]
                ),
                rank=rank,
            )

        # ----------------------------------------------------
        # 10. Commit complete transaction
        # ----------------------------------------------------

        # Everything above only staged rows in the session. A single
        # commit makes the assessment, model scores and risk factors
        # appear together; on failure the rollback leaves no partial
        # record behind.
        try:

            await session.commit()

        except Exception:

            await session.rollback()

            logger.exception(
                "Database commit failed for customer %s.",
                customer_id_str,
            )

            raise

        logger.info(
            "Prediction persisted successfully "
            "for customer %s. prediction_id=%d.",
            customer_id_str,
            prediction_record.id,
        )

        # ----------------------------------------------------
        # 11. Return complete application result
        # ----------------------------------------------------

        return {
            "customer_id": customer_id_str,

            "prediction": prediction,

            "risk_level": assessment.risk_level,

            "decision": assessment.decision,

            "probability": probability,

            "ensemble_probability": ensemble_probability,

            "threshold": assessment.threshold,

            "model_probabilities": model_probabilities,

            "shap": {
                "base_value": float(
                    shap_result["base_value"]
                ),

                "top_factors": top_factors,

                "positive_factors": (
                    shap_result["positive_factors"]
                ),

                "negative_factors": (
                    shap_result["negative_factors"]
                ),

                "feature_count": int(
                    shap_result["feature_count"]
                ),

                "models_explained": shap_result.get(
                    "models_explained", []
                ),
            },

            "llm_explanation": (
                llm_explanation.model_dump()
            ),

            "prediction_id": prediction_record.id,
        }