"""
============================================================
CreditLens-AI
Prediction API Routes
============================================================

Author: Muhammad Sami
"""

from __future__ import annotations

import math
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.schemas import (
    CustomerPredictionHistoryResponse,
    DashboardRecentAssessment,
    DashboardSummaryResponse,
    FeatureCompleteness,
    LLMExplanationResponse,
    ModelProbabilities,
    PredictionHistoryItem,
    PredictionRequest,
    PredictionResponse,
    SampleCustomersResponse,
    ShapExplanation,
    ShapFactor,
)
from src.database.connection import get_db
from src.database.repository import (
    get_customer_predictions,
    get_dashboard_summary,
)
from src.services.customer_feature_store import (
    CustomerFeatureStore,
    CustomerNotFoundError,
    FeatureStoreUnavailableError,
)
from src.services.prediction_service import PredictionService
from src.utils.logger import get_logger


logger = get_logger(__name__)

router = APIRouter(
    prefix="/api/v1",
    tags=["Credit Risk"],
)


# ------------------------------------------------------------
# Services
# ------------------------------------------------------------

prediction_service = PredictionService()


customer_feature_store = CustomerFeatureStore()


# ------------------------------------------------------------
# Response builders
# ------------------------------------------------------------

def _build_shap(
    shap: dict[str, Any] | None,
) -> ShapExplanation | None:

    if not shap:
        return None

    def factors(items: list[dict[str, Any]]) -> list[ShapFactor]:
        return [
            ShapFactor(
                feature=item["feature"],
                feature_value=item.get("feature_value"),
                shap_value=item["shap_value"],
                direction=item["direction"],
                rank=position,
                models_agreeing=item.get("models_agreeing"),
            )
            for position, item in enumerate(items, start=1)
        ]

    return ShapExplanation(
        base_value=shap.get("base_value"),
        feature_count=shap.get("feature_count"),
        models_explained=shap.get("models_explained", []),
        top_factors=factors(shap.get("top_factors", [])),
        positive_factors=factors(
            shap.get("positive_factors", [])
        ),
        negative_factors=factors(
            shap.get("negative_factors", [])
        ),
    )


def _is_empty(value: Any) -> bool:
    """True for JSON null / NaN (a value that is present but empty)."""

    return value is None or (
        isinstance(value, float) and math.isnan(value)
    )


def _normalise_missing_values(
    features: dict[str, Any],
) -> dict[str, Any]:
    """
    Convert JSON ``null`` (None) into ``float('nan')``.

    The API contract allows ``null`` feature values, but
    ``pd.DataFrame([features])`` turns a None-only column into an
    ``object`` column, which LightGBM/XGBoost reject
    ("pandas dtypes must be int, float or bool"). NaN is how the
    models represent a missing value, so this preserves meaning.
    No feature is added, removed or rescaled.
    """

    return {
        name: (float("nan") if value is None else value)
        for name, value in features.items()
    }


def _build_feature_completeness(
    features: dict[str, Any],
) -> FeatureCompleteness | None:

    try:
        expected = list(
            prediction_service.feature_service.feature_columns
        )
    except AttributeError:
        return None

    provided = sum(1 for column in expected if column in features)

    empty = sum(
        1
        for column in expected
        if column in features and _is_empty(features[column])
    )

    return FeatureCompleteness(
        expected_count=len(expected),
        provided_count=provided,
        missing_count=len(expected) - provided,
        null_count=empty,
    )


def _build_response(
    result: dict[str, Any],
    features: dict[str, Any],
    completeness: FeatureCompleteness | None = None,
) -> PredictionResponse:

    explanation = result["llm_explanation"]

    return PredictionResponse(
        customer_id=result["customer_id"],
        prediction=result["prediction"],
        risk_level=result["risk_level"],
        decision=result["decision"],
        probability=result["probability"],
        ensemble_probability=result["ensemble_probability"],
        threshold=result["threshold"],
        model_probabilities=ModelProbabilities(
            lightgbm=result["model_probabilities"]["lightgbm"],
            xgboost=result["model_probabilities"]["xgboost"],
            catboost=result["model_probabilities"]["catboost"],
        ),
        llm_explanation=LLMExplanationResponse(
            summary=explanation["summary"],
            risk_explanation=explanation["risk_explanation"],
            key_risk_factors=explanation["key_risk_factors"],
            protective_factors=explanation["protective_factors"],
            analyst_actions=explanation["analyst_actions"],
        ),
        prediction_id=result.get("prediction_id"),
        shap=_build_shap(result.get("shap")),
        feature_completeness=(
            completeness
            if completeness is not None
            else _build_feature_completeness(features)
        ),
        # Same weights the predictor used for this assessment.
        model_weights=dict(
            prediction_service.predictor.ensemble.weights
        ),
    )


async def _run_prediction(
    session: AsyncSession,
    customer_id: str,
    features: dict[str, Any],
    completeness: FeatureCompleteness | None = None,
) -> PredictionResponse:

    try:

        result = await prediction_service.predict(
            session,
            customer_id=customer_id,
            features=_normalise_missing_values(features),
            top_n=10,
        )

        return _build_response(result, features, completeness)

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        logger.exception(
            "Prediction failed for customer %s.",
            customer_id,
        )

        raise HTTPException(
            status_code=500,
            detail="Credit-risk prediction failed.",
        ) from exc


# ------------------------------------------------------------
# Health
# ------------------------------------------------------------

@router.get("/health")
def health_check() -> dict[str, str]:
    """
    Check whether the CreditLens prediction service is alive.
    """

    return {
        "status": "healthy",
        "service": "CreditLens-AI",
        "version": "1.0.0",
    }


# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------

@router.post(
    "/predict",
    response_model=PredictionResponse,
)
async def predict_credit_risk(
    request: PredictionRequest,
    session: AsyncSession = Depends(get_db),
) -> PredictionResponse:
    """
    Generate, explain, and persist a credit-risk
    prediction for one customer.
    """

    return await _run_prediction(
        session,
        request.customer_id,
        request.features,
    )


# ------------------------------------------------------------
# Server-side customer assessment
# ------------------------------------------------------------

@router.post(
    "/assess/{customer_id}",
    response_model=PredictionResponse,
)
async def assess_customer(
    customer_id: str,
    session: AsyncSession = Depends(get_db),
) -> PredictionResponse:
    """
    Assess a customer by ID. The production features are
    retrieved server-side and run through the same prediction
    pipeline as ``/predict``.
    """

    try:

        features = customer_feature_store.get_features(
            customer_id
        )

    except CustomerNotFoundError as exc:

        raise HTTPException(
            status_code=404,
            detail="Customer was not found in the feature store.",
        ) from exc

    except FeatureStoreUnavailableError as exc:

        logger.error("Feature store unavailable: %s", exc)

        raise HTTPException(
            status_code=503,
            detail="Customer lookup is currently unavailable.",
        ) from exc

    return await _run_prediction(
        session,
        customer_id.strip(),
        features,
    )


@router.get(
    "/customers/sample",
    response_model=SampleCustomersResponse,
)
def list_sample_customers(
    limit: int = Query(20, ge=1, le=100),
) -> SampleCustomersResponse:
    """
    List customer IDs that can be assessed by ID.
    """

    try:

        return SampleCustomersResponse(
            customer_ids=customer_feature_store.sample_customer_ids(
                limit
            )
        )

    except FeatureStoreUnavailableError as exc:

        logger.error("Feature store unavailable: %s", exc)

        raise HTTPException(
            status_code=503,
            detail="Customer lookup is currently unavailable.",
        ) from exc


# ------------------------------------------------------------
# Customer prediction history
# ------------------------------------------------------------

@router.get(
    "/customers/{customer_id}/predictions",
    response_model=CustomerPredictionHistoryResponse,
)
async def get_customer_prediction_history(
    customer_id: str,
    limit: int = Query(50, ge=1, le=200),
    session: AsyncSession = Depends(get_db),
) -> CustomerPredictionHistoryResponse:
    """
    Return persisted predictions for one customer, newest
    first. A customer with no history returns an empty list.
    """

    try:

        records = await get_customer_predictions(
            session,
            customer_id,
            limit=limit,
        )

    except Exception as exc:

        logger.exception(
            "History lookup failed for customer %s.",
            customer_id,
        )

        raise HTTPException(
            status_code=503,
            detail="Prediction history is currently unavailable.",
        ) from exc

    return CustomerPredictionHistoryResponse(
        customer_id=customer_id,
        predictions=[
            PredictionHistoryItem(
                prediction_id=record.id,
                prediction=record.prediction,
                risk_level=record.risk_level,
                decision=record.decision,
                probability=record.probability,
                ensemble_probability=record.ensemble_probability,
                threshold=record.threshold,
                created_at=record.created_at,
            )
            for record in records
        ],
    )


# ------------------------------------------------------------
# Dashboard summary
# ------------------------------------------------------------

@router.get(
    "/dashboard/summary",
    response_model=DashboardSummaryResponse,
)
async def get_dashboard_summary_endpoint(
    session: AsyncSession = Depends(get_db),
) -> DashboardSummaryResponse:
    """
    Aggregate metrics over persisted predictions.
    """

    try:

        summary = await get_dashboard_summary(session)

    except Exception as exc:

        logger.exception("Dashboard summary failed.")

        raise HTTPException(
            status_code=503,
            detail="Assessment analytics are currently unavailable.",
        ) from exc

    return DashboardSummaryResponse(
        total_assessments=summary["total_assessments"],
        high_risk_assessments=summary["high_risk_assessments"],
        low_risk_assessments=summary["low_risk_assessments"],
        average_probability=summary["average_probability"],
        recent_assessments=[
            DashboardRecentAssessment(
                prediction_id=record.id,
                customer_id=record.customer_id,
                prediction=record.prediction,
                risk_level=record.risk_level,
                decision=record.decision,
                probability=record.probability,
                created_at=record.created_at,
            )
            for record in summary["recent_assessments"]
        ],
    )
