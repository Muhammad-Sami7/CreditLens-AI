"""
============================================================
CreditLens-AI
Database CRUD Test
============================================================

Verifies that the repository layer can create and read
customers, predictions, model predictions, risk factors,
and model versions.
"""

from __future__ import annotations

import asyncio

from sqlalchemy import select

from src.database.connection import AsyncSessionLocal
from src.database.models import ModelVersion
from src.database.repository import (
    create_model_prediction,
    create_model_version,
    create_prediction,
    create_risk_factor,
    get_customer_predictions,
    get_or_create_customer,
)


async def test_database_crud() -> None:
    print("=" * 70)
    print("CreditLens-AI - Database CRUD Test")
    print("=" * 70)

    test_customer_id = "CRUD_TEST_CUSTOMER"

    async with AsyncSessionLocal() as session:
        try:
            # --------------------------------------------------------
            # 1. Customer
            # --------------------------------------------------------

            customer = await get_or_create_customer(
                session,
                test_customer_id,
            )

            print(f"✓ Customer created/found: {customer.customer_id}")

            # --------------------------------------------------------
            # 2. Prediction
            # --------------------------------------------------------

            prediction = await create_prediction(
                session,
                customer_id=test_customer_id,
                prediction=0,
                risk_level="Low Risk",
                decision="Below Risk Threshold",
                probability=0.042242,
                ensemble_probability=0.300549,
                threshold=0.18,
            )

            print(
                f"✓ Prediction created: "
                f"id={prediction.id}, "
                f"probability={prediction.probability:.6f}"
            )

            # --------------------------------------------------------
            # 3. Individual model predictions
            # --------------------------------------------------------

            await create_model_prediction(
                session,
                prediction_id=prediction.id,
                model_name="LightGBM",
                probability=0.307807,
                prediction=0,
            )

            await create_model_prediction(
                session,
                prediction_id=prediction.id,
                model_name="XGBoost",
                probability=0.286364,
                prediction=0,
            )

            await create_model_prediction(
                session,
                prediction_id=prediction.id,
                model_name="CatBoost",
                probability=0.308630,
                prediction=0,
            )

            print("✓ Individual model predictions created")

            # --------------------------------------------------------
            # 4. SHAP risk factors
            # --------------------------------------------------------

            await create_risk_factor(
                session,
                prediction_id=prediction.id,
                feature_name="EXT_SOURCE_3",
                feature_value="0.42",
                shap_value=0.147858,
                direction="positive",
                rank=1,
            )

            await create_risk_factor(
                session,
                prediction_id=prediction.id,
                feature_name="EXT_SOURCE_MEAN",
                feature_value="0.68",
                shap_value=-0.360785,
                direction="negative",
                rank=1,
            )

            print("✓ SHAP risk factors created")

            # --------------------------------------------------------
            # 5. Model version
            # --------------------------------------------------------

            model_version = await create_model_version(
                session,
                model_name="CreditLens Ensemble",
                version="1.0.0",
                metadata={
                    "lightgbm_weight": 0.35,
                    "xgboost_weight": 0.35,
                    "catboost_weight": 0.30,
                    "calibration": "isotonic",
                    "threshold": 0.18,
                },
            )

            print(
                f"✓ Model version created: "
                f"id={model_version.id}, "
                f"version={model_version.version}"
            )

            # --------------------------------------------------------
            # 6. Customer history
            # --------------------------------------------------------

            history = await get_customer_predictions(
                session,
                test_customer_id,
            )

            print(
                f"✓ Customer history retrieved: "
                f"{len(history)} prediction(s)"
            )

            # --------------------------------------------------------
            # 7. Verify model version metadata
            # --------------------------------------------------------

            result = await session.execute(
                select(ModelVersion).where(
                    ModelVersion.id == model_version.id
                )
            )

            stored_model_version = result.scalar_one()

            assert stored_model_version.model_metadata is not None
            assert (
                stored_model_version.model_metadata["calibration"]
                == "isotonic"
            )
            assert (
                stored_model_version.model_metadata["threshold"]
                == 0.18
            )

            print("✓ Model version metadata verified")

            # --------------------------------------------------------
            # 8. Roll back test data
            # --------------------------------------------------------

            await session.rollback()

            print("✓ Test transaction rolled back")
            print("✓ No CRUD test data was permanently stored")

            print("=" * 70)
            print("DATABASE CRUD TEST PASSED")
            print("=" * 70)

        except Exception:
            await session.rollback()
            raise


if __name__ == "__main__":
    asyncio.run(test_database_crud())