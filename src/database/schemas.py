"""
============================================================
CreditLens-AI
Database Schemas
============================================================

Pydantic schemas used by the database/application layer.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel


# ============================================================
# Customer
# ============================================================

class CustomerCreate(BaseModel):

    customer_id: str


class CustomerResponse(BaseModel):

    customer_id: str
    created_at: datetime

    model_config = {
        "from_attributes": True,
    }


# ============================================================
# Prediction
# ============================================================

class PredictionCreate(BaseModel):

    customer_id: str

    prediction: int

    risk_level: str

    decision: str

    probability: float

    ensemble_probability: float

    threshold: float


class PredictionResponse(PredictionCreate):

    id: int
    created_at: datetime

    model_config = {
        "from_attributes": True,
    }


# ============================================================
# Model Prediction
# ============================================================

class ModelPredictionCreate(BaseModel):

    prediction_id: int

    model_name: str

    probability: float

    prediction: int | None = None


# ============================================================
# Risk Factor
# ============================================================

class RiskFactorCreate(BaseModel):

    prediction_id: int

    feature_name: str

    feature_value: str | None = None

    shap_value: float

    direction: str

    rank: int


# ============================================================
# Model Version
# ============================================================

class ModelVersionCreate(BaseModel):

    model_name: str

    version: str

    metadata: dict[str, Any] | None = None