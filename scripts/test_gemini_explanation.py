from __future__ import annotations

import asyncio

from src.services.llm_explanation_service import LLMExplanationService


async def main() -> None:
    print("=" * 70)
    print("CreditLens-AI - Gemini Explanation Service Test")
    print("=" * 70)

    service = LLMExplanationService()

    prediction_result = {
        "prediction": 0,
        "risk_level": "Low Risk",
        "probability": 0.042242,
        "ensemble_probability": 0.300549,
        "threshold": 0.18,
        "model_probabilities": {
            "lightgbm": 0.307807,
            "xgboost": 0.286364,
            "catboost": 0.308630,
        },
    }

    shap_result = {
        "top_factors": [
            {
                "feature": "EXT_SOURCE_MEAN",
                "feature_value": 0.72,
                "shap_value": -0.360785,
                "direction": "negative",
            },
            {
                "feature": "EXT_SOURCE_3",
                "feature_value": 0.65,
                "shap_value": 0.147858,
                "direction": "positive",
            },
            {
                "feature": "BUREAU_RECENT_LOAN_DAYS",
                "feature_value": 120,
                "shap_value": 0.134925,
                "direction": "positive",
            },
            {
                "feature": "EXT_SOURCE_2",
                "feature_value": 0.70,
                "shap_value": -0.156610,
                "direction": "negative",
            },
        ],
        "positive_factors": [
            {
                "feature": "EXT_SOURCE_3",
                "feature_value": 0.65,
                "shap_value": 0.147858,
                "direction": "positive",
            },
            {
                "feature": "BUREAU_RECENT_LOAN_DAYS",
                "feature_value": 120,
                "shap_value": 0.134925,
                "direction": "positive",
            },
        ],
        "negative_factors": [
            {
                "feature": "EXT_SOURCE_MEAN",
                "feature_value": 0.72,
                "shap_value": -0.360785,
                "direction": "negative",
            },
            {
                "feature": "EXT_SOURCE_2",
                "feature_value": 0.70,
                "shap_value": -0.156610,
                "direction": "negative",
            },
        ],
    }

    customer_context = {
        "customer_type": "individual",
        "data_source": "anonymized_demo_data",
    }

    print("\n[1/3] Initializing Gemini service...")
    print(f"✓ Model: {service.model}")

    print("\n[2/3] Sending explanation request...")

    explanation = await service.explain_prediction(
        prediction_result=prediction_result,
        shap_result=shap_result,
        customer_context=customer_context,
    )

    print("✓ Gemini response received")

    print("\n[3/3] Validating structured response...")

    if not explanation.summary:
        raise RuntimeError("Summary is empty.")

    if not explanation.risk_explanation:
        raise RuntimeError("Risk explanation is empty.")

    if not isinstance(explanation.key_risk_factors, list):
        raise RuntimeError("key_risk_factors is not a list.")

    if not isinstance(explanation.protective_factors, list):
        raise RuntimeError("protective_factors is not a list.")

    if not isinstance(explanation.analyst_actions, list):
        raise RuntimeError("analyst_actions is not a list.")

    print("✓ Pydantic structured output validated")

    print("\n" + "=" * 70)
    print("GEMINI EXPLANATION RESULT")
    print("=" * 70)

    print("\nSummary:")
    print(explanation.summary)

    print("\nRisk Explanation:")
    print(explanation.risk_explanation)

    print("\nKey Risk Factors:")
    for factor in explanation.key_risk_factors:
        print(f"  - {factor}")

    print("\nProtective Factors:")
    for factor in explanation.protective_factors:
        print(f"  - {factor}")

    print("\nAnalyst Actions:")
    for action in explanation.analyst_actions:
        print(f"  - {action}")

    print("\n" + "=" * 70)
    print("✓ GEMINI EXPLANATION SERVICE TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())