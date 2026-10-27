"""
============================================================
CreditLens-AI
Database Repository
============================================================

Application-level database operations.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import (
    Customer,
    ModelPrediction,
    ModelVersion,
    Prediction,
    RiskFactor,
)


# ============================================================
# Customer Operations
# ============================================================

async def get_or_create_customer(
    session: AsyncSession,
    customer_id: str,
) -> Customer:

    result = await session.execute(
        select(Customer).where(
            Customer.customer_id == customer_id
        )
    )

    customer = result.scalar_one_or_none()

    if customer is not None:
        return customer

    customer = Customer(
        customer_id=customer_id,
    )

    session.add(customer)

    await session.flush()

    return customer


# ============================================================
# Prediction Operations
# ============================================================

async def create_prediction(
    session: AsyncSession,
    *,
    customer_id: str,
    prediction: int,
    risk_level: str,
    decision: str,
    probability: float,
    ensemble_probability: float,
    threshold: float,
) -> Prediction:

    prediction_record = Prediction(
        customer_id=customer_id,
        prediction=prediction,
        risk_level=risk_level,
        decision=decision,
        probability=probability,
        ensemble_probability=ensemble_probability,
        threshold=threshold,
    )

    session.add(prediction_record)

    await session.flush()

    return prediction_record


# ============================================================
# Model Prediction
# ============================================================

async def create_model_prediction(
    session: AsyncSession,
    *,
    prediction_id: int,
    model_name: str,
    probability: float,
    prediction: int | None = None,
) -> ModelPrediction:

    record = ModelPrediction(
        prediction_id=prediction_id,
        model_name=model_name,
        probability=probability,
        prediction=prediction,
    )

    session.add(record)

    await session.flush()

    return record


# ============================================================
# Risk Factor
# ============================================================

async def create_risk_factor(
    session: AsyncSession,
    *,
    prediction_id: int,
    feature_name: str,
    feature_value: str | None,
    shap_value: float,
    direction: str,
    rank: int,
) -> RiskFactor:

    record = RiskFactor(
        prediction_id=prediction_id,
        feature_name=feature_name,
        feature_value=feature_value,
        shap_value=shap_value,
        direction=direction,
        rank=rank,
    )

    session.add(record)

    await session.flush()

    return record


# ============================================================
# Customer History
# ============================================================

async def get_customer_predictions(
    session: AsyncSession,
    customer_id: str,
    limit: int | None = None,
) -> list[Prediction]:
    """
    Return a customer's predictions, newest first.

    ``limit`` is optional; ``None`` keeps the original
    behaviour of returning every record.
    """

    statement = (
        select(Prediction)
        .where(
            Prediction.customer_id == customer_id
        )
        .order_by(
            Prediction.created_at.desc(),
            Prediction.id.desc(),
        )
    )

    if limit is not None:
        statement = statement.limit(limit)

    result = await session.execute(statement)

    return list(result.scalars().all())


# ============================================================
# Dashboard Summary
# ============================================================

async def get_dashboard_summary(
    session: AsyncSession,
    recent_limit: int = 10,
) -> dict[str, Any]:
    """
    Aggregate persisted predictions for the dashboard.

    High risk is ``prediction == 1`` (the binary flag produced
    by the production risk engine); low risk is ``prediction == 0``.
    The average uses the calibrated ``probability`` column only.
    """

    totals = (
        await session.execute(
            select(
                func.count(Prediction.id),
                func.count(Prediction.id).filter(
                    Prediction.prediction == 1
                ),
                func.count(Prediction.id).filter(
                    Prediction.prediction == 0
                ),
                func.avg(Prediction.probability),
            )
        )
    ).one()

    recent = (
        await session.execute(
            select(Prediction)
            .order_by(
                Prediction.created_at.desc(),
                Prediction.id.desc(),
            )
            .limit(recent_limit)
        )
    ).scalars().all()

    total, high, low, average = totals

    return {
        "total_assessments": int(total or 0),
        "high_risk_assessments": int(high or 0),
        "low_risk_assessments": int(low or 0),
        "average_probability": (
            float(average) if average is not None else None
        ),
        "recent_assessments": list(recent),
    }


# ============================================================
# Model Version
# ============================================================

async def create_model_version(
    session: AsyncSession,
    *,
    model_name: str,
    version: str,
    metadata: dict[str, Any] | None = None,
) -> ModelVersion:

    record = ModelVersion(
        model_name=model_name,
        version=version,
        model_metadata=metadata,
    )

    session.add(record)

    await session.flush()

    return record