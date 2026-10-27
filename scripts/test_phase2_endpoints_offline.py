"""
===========================================================
CreditLens-AI
Phase 2 Endpoint Test (offline)
===========================================================

Exercises the Phase 2 API additions WITHOUT the real models,
Gemini or PostgreSQL:

- structured SHAP + prediction_id + feature_completeness
- GET /customers/{id}/predictions
- GET /dashboard/summary
- POST /assess/{id}  and  GET /customers/sample

The ML/SHAP/LLM PredictionService is replaced by a stub that
persists real rows through the real repository into a
temporary SQLite database. This verifies routing, schemas,
SQL and empty/error behaviour - not model quality.

Requires: aiosqlite  (pip install aiosqlite)

Run from the project root:
    python scripts/test_phase2_endpoints_offline.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

TMP = Path(tempfile.mkdtemp())
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TMP / 'test.db'}"

import asyncio  # noqa: E402

import pandas as pd  # noqa: E402
from sqlalchemy.dialects.postgresql import JSONB  # noqa: E402
from sqlalchemy.ext.compiler import compiles  # noqa: E402


@compiles(JSONB, "sqlite")
def _jsonb_as_json(element, compiler, **kw):  # noqa: ANN001
    return "JSON"


# ---------------------------------------------------------
# Stub PredictionService (replaces models / SHAP / Gemini)
# ---------------------------------------------------------

from src.database.repository import (  # noqa: E402
    create_prediction,
    get_or_create_customer,
)

FEATURE_COLUMNS = ["f1", "f2", "f3"]


class _Features:
    feature_columns = FEATURE_COLUMNS


class StubPredictionService:

    feature_service = _Features()

    # The routes read the ensemble weights from the predictor.
    predictor = types.SimpleNamespace(
        ensemble=types.SimpleNamespace(
            weights={"lightgbm": 0.35, "xgboost": 0.35, "catboost": 0.30}
        )
    )

    async def predict(self, session, *, customer_id, features, top_n=10):
        high = customer_id.startswith("H")
        probability = 0.40 if high else 0.04
        await get_or_create_customer(session, customer_id)
        record = await create_prediction(
            session,
            customer_id=customer_id,
            prediction=1 if high else 0,
            risk_level="High Risk" if high else "Low Risk",
            decision="Review Required" if high else "Below Risk Threshold",
            probability=probability,
            ensemble_probability=0.30,
            threshold=0.18,
        )
        await session.commit()

        def factor(name, value, shap):
            return {
                "feature": name,
                "feature_value": value,
                "shap_value": shap,
                "abs_shap_value": abs(shap),
                "direction": "positive" if shap > 0 else "negative",
            }

        pos = [factor("EXT_SOURCE_3", 0.2, 0.15), factor("X", None, 0.05)]
        neg = [factor("EXT_SOURCE_MEAN", 0.7, -0.36)]
        return {
            "customer_id": customer_id,
            "prediction": record.prediction,
            "risk_level": record.risk_level,
            "decision": record.decision,
            "probability": probability,
            "ensemble_probability": 0.30,
            "threshold": 0.18,
            "model_probabilities": {
                "lightgbm": 0.31, "xgboost": 0.29, "catboost": 0.30,
            },
            "shap": {
                "base_value": -2.1,
                "top_factors": pos + neg,
                "positive_factors": pos,
                "negative_factors": neg,
                "feature_count": 3,
            },
            "llm_explanation": {
                "summary": "s", "risk_explanation": "r",
                "key_risk_factors": ["a"], "protective_factors": ["b"],
                "analyst_actions": ["c"],
            },
            "prediction_id": record.id,
        }


stub_module = types.ModuleType("src.services.prediction_service")
stub_module.PredictionService = StubPredictionService
sys.modules["src.services.prediction_service"] = stub_module

# ---------------------------------------------------------
# App under test
# ---------------------------------------------------------

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from src.api import routes  # noqa: E402
from src.database.connection import Base, engine  # noqa: E402
from src.services.customer_feature_store import (  # noqa: E402
    CustomerFeatureStore,
)

import src.database.models  # noqa: E402,F401  (register tables)

parquet = TMP / "features.parquet"
pd.DataFrame(
    {
        "SK_ID_CURR": [100001, 100002, 100003],
        "f1": [0.1, None, 0.3],
        "f2": [1, 2, 3],
        "f3": [0.5, 0.6, None],
    }
).to_parquet(parquet)
routes.customer_feature_store = CustomerFeatureStore(data_path=parquet)

app = FastAPI()
app.include_router(routes.router)

passed = failed = 0


def check(name: str, condition: bool) -> None:
    global passed, failed
    if condition:
        passed += 1
        print(f"PASS  {name}")
    else:
        failed += 1
        print(f"FAIL  {name}")


async def create_tables() -> None:
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)


asyncio.run(create_tables())

with TestClient(app) as client:

    # --- empty database ---------------------------------------
    r = client.get("/api/v1/dashboard/summary")
    body = r.json()
    check("summary 200 on empty DB", r.status_code == 200)
    check("empty: totals are 0", body["total_assessments"] == 0)
    check("empty: average_probability is null",
          body["average_probability"] is None)
    check("empty: recent is []", body["recent_assessments"] == [])

    r = client.get("/api/v1/customers/NOPE/predictions")
    check("history 200 (not 404) for unknown customer",
          r.status_code == 200)
    check("history empty list for unknown customer",
          r.json() == {"customer_id": "NOPE", "predictions": []})

    # --- /predict with partial features -----------------------
    r = client.post(
        "/api/v1/predict",
        json={"customer_id": "L1", "features": {"f1": 1.0, "f2": 2.0}},
    )
    body = r.json()
    check("predict 200", r.status_code == 200)
    check("predict: prediction_id returned",
          isinstance(body.get("prediction_id"), int))
    check("predict: structured SHAP returned",
          body["shap"]["top_factors"][0]["feature"] == "EXT_SOURCE_3")
    check("predict: SHAP rank is 1-based",
          [f["rank"] for f in body["shap"]["top_factors"]] == [1, 2, 3])
    check("predict: null feature_value preserved",
          body["shap"]["top_factors"][1]["feature_value"] is None)
    check("predict: completeness counts absent columns",
          body["feature_completeness"] ==
          {"expected_count": 3, "provided_count": 2, "missing_count": 1,
           "null_count": 0})
    check("predict: original fields intact",
          body["decision"] == "Below Risk Threshold"
          and body["threshold"] == 0.18)

    # --- /assess ----------------------------------------------
    r = client.post("/api/v1/assess/100001")
    body = r.json()
    check("assess 200 for known customer", r.status_code == 200)
    check("assess: completeness full (null != missing)",
          body["feature_completeness"]["missing_count"] == 0)
    r = client.post("/api/v1/assess/999999")
    check("assess 404 for unknown customer", r.status_code == 404)
    r = client.post("/api/v1/assess/H-risky")
    check("assess unknown 'H-risky' is 404 (not in store)",
          r.status_code == 404)

    # high-risk row via /predict
    client.post("/api/v1/predict",
                json={"customer_id": "H1", "features": {"f1": 1.0}})
    client.post("/api/v1/predict",
                json={"customer_id": "L1", "features": {"f1": 1.0}})

    r = client.get("/api/v1/customers/sample?limit=2")
    check("sample customers limited",
          r.json()["customer_ids"] == ["100001", "100002"])

    # --- history ----------------------------------------------
    r = client.get("/api/v1/customers/L1/predictions")
    items = r.json()["predictions"]
    check("history returns 2 records for L1", len(items) == 2)
    check("history newest first",
          items[0]["prediction_id"] > items[1]["prediction_id"])
    check("history item has timestamp + threshold",
          "created_at" in items[0] and items[0]["threshold"] == 0.18)
    r = client.get("/api/v1/customers/L1/predictions?limit=1")
    check("history limit honoured", len(r.json()["predictions"]) == 1)
    r = client.get("/api/v1/customers/L1/predictions?limit=0")
    check("history limit=0 rejected (422)", r.status_code == 422)

    # --- dashboard --------------------------------------------
    r = client.get("/api/v1/dashboard/summary")
    s = r.json()
    check("summary total = 4 (404s save nothing)", s["total_assessments"] == 4)
    check("summary high=1 low=4",
          s["high_risk_assessments"] == 1 and s["low_risk_assessments"] == 3)
    check("summary avg uses calibrated probability",
          abs(s["average_probability"] - ((0.04 * 3 + 0.40) / 4)) < 1e-9)
    check("summary recent newest first, <=10",
          0 < len(s["recent_assessments"]) <= 10
          and s["recent_assessments"][0]["prediction_id"] == 4)

    # --- existing contract ------------------------------------
    r = client.get("/api/v1/health")
    check("health unchanged", r.json()["status"] == "healthy")

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
