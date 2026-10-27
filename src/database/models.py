"""
============================================================
CreditLens-AI
Database Models
============================================================

SQLAlchemy ORM models for customer records,
prediction history, model probabilities,
SHAP risk factors, and model versions.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.connection import Base


# ============================================================
# Customer
# ============================================================

class Customer(Base):
    """
    Stores application-level customer information.
    """

    __tablename__ = "customers"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    customer_id: Mapped[str] = mapped_column(
        String(100),
        unique=True,
        nullable=False,
        index=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    predictions: Mapped[list["Prediction"]] = relationship(
        back_populates="customer",
        cascade="all, delete-orphan",
    )


# ============================================================
# Prediction
# ============================================================

class Prediction(Base):
    """
    Stores the final production credit-risk assessment.
    """

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    customer_id: Mapped[str] = mapped_column(
        String(100),
        ForeignKey(
            "customers.customer_id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    prediction: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    risk_level: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    decision: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    probability: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    ensemble_probability: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    threshold: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )

    customer: Mapped["Customer"] = relationship(
        back_populates="predictions",
    )

    model_predictions: Mapped[list["ModelPrediction"]] = relationship(
        back_populates="prediction_record",
        cascade="all, delete-orphan",
    )

    risk_factors: Mapped[list["RiskFactor"]] = relationship(
        back_populates="prediction",
        cascade="all, delete-orphan",
    )


# ============================================================
# Individual Model Predictions
# ============================================================

class ModelPrediction(Base):
    """
    Stores individual probabilities produced by
    LightGBM, XGBoost, and CatBoost.
    """

    __tablename__ = "model_predictions"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    prediction_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey(
            "predictions.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    model_name: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    probability: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )
    prediction: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )
    prediction_record: Mapped["Prediction"] = relationship(
        back_populates="model_predictions",
    )


# ============================================================
# SHAP Risk Factors
# ============================================================

class RiskFactor(Base):
    """
    Stores important SHAP factors for a prediction.
    """

    __tablename__ = "risk_factors"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    prediction_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey(
            "predictions.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    feature_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    feature_value: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    shap_value: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    direction: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
    )

    rank: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    prediction: Mapped["Prediction"] = relationship(
        back_populates="risk_factors",
    )


# ============================================================
# Model Versions
# ============================================================

class ModelVersion(Base):
    """
    Stores production model metadata and version information.
    """

    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    model_name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
    )

    version: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    model_metadata: Mapped[dict[str, Any] | None] = mapped_column(
        "metadata",
        JSONB,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )