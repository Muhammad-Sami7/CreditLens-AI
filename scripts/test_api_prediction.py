"""
===========================================================
CreditLens-AI
FastAPI Prediction Endpoint Test
===========================================================

Tests the complete production prediction pipeline through
the actual FastAPI /api/v1/predict endpoint.

Pipeline:

test_processed.parquet
        |
297 production features
        |
FastAPI
        |
CreditLensPredictor
        |
LightGBM
XGBoost
CatBoost
        |
Weighted Ensemble
        |
Isotonic Calibration
        |
Calibrated Production Threshold
        |
Final Risk Prediction
        |
SHAP Explainability
        |
Gemini Explanation
        |
Structured API Response

Author: Muhammad Sami
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests


# =========================================================
# Configuration
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test_processed.parquet"
)

API_URL = "http://127.0.0.1:8000/api/v1/predict"

HEALTH_URL = "http://127.0.0.1:8000/api/v1/health"

ID_COLUMN = "SK_ID_CURR"

EXPECTED_FEATURE_COUNT = 297

EXPECTED_THRESHOLD = 0.18


# =========================================================
# Main Test
# =========================================================

def main() -> None:

    print("=" * 70)
    print("CreditLens-AI - FastAPI Prediction Endpoint Test")
    print("=" * 70)

    # -----------------------------------------------------
    # 1. Load test dataset
    # -----------------------------------------------------

    print("\n[1/5] Loading test dataset...")

    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Test dataset not found:\n{TEST_DATA_PATH}"
        )

    test_df = pd.read_parquet(
        TEST_DATA_PATH
    )

    print("✓ Test dataset loaded")
    print(f"  Rows    : {len(test_df):,}")
    print(f"  Columns : {len(test_df.columns)}")

    # -----------------------------------------------------
    # 2. Prepare real customer
    # -----------------------------------------------------

    print("\n[2/5] Preparing real test customer...")

    if ID_COLUMN not in test_df.columns:
        raise ValueError(
            f"Required identifier '{ID_COLUMN}' "
            "was not found in the test dataset."
        )

    customer_id = test_df.iloc[0][ID_COLUMN]

    customer_df = (
        test_df
        .iloc[[0]]
        .drop(columns=[ID_COLUMN])
        .copy()
    )

    feature_count = len(
        customer_df.columns
    )

    if feature_count != EXPECTED_FEATURE_COUNT:
        raise AssertionError(
            "Unexpected production feature count: "
            f"{feature_count}. "
            f"Expected {EXPECTED_FEATURE_COUNT}."
        )

    print("✓ Customer prepared")
    print(f"  Customer ID   : {customer_id}")
    print(f"  Feature count : {feature_count}")

    # -----------------------------------------------------
    # 3. Check API
    # -----------------------------------------------------

    print("\n[3/5] Checking FastAPI endpoint...")

    try:

        health_response = requests.get(
            HEALTH_URL,
            timeout=10,
        )

    except requests.RequestException as exc:

        raise RuntimeError(
            "Could not connect to FastAPI.\n"
            "Make sure the API is running with:\n"
            "python -m uvicorn src.main:app --reload"
        ) from exc

    if health_response.status_code != 200:

        raise RuntimeError(
            "FastAPI health check failed.\n"
            f"Status code: {health_response.status_code}\n"
            f"Response: {health_response.text}"
        )

    print("✓ FastAPI is running")

    # -----------------------------------------------------
    # 4. Send prediction request
    # -----------------------------------------------------

    print("\n[4/5] Sending prediction request...")

    payload = {

        "customer_id": str(customer_id),

        "features": {
            column: (
                None
                if pd.isna(value)
                else value.item()
                if hasattr(value, "item")
                else value
            )

            for column, value
            in customer_df.iloc[0].items()
        },
    }

    try:

        response = requests.post(
            API_URL,
            json=payload,
            timeout=300,
        )

    except requests.RequestException as exc:

        raise RuntimeError(
            "Prediction request failed."
        ) from exc

    print(
        f"✓ API responded "
        f"(HTTP {response.status_code})"
    )

    if response.status_code != 200:

        print("\nAPI Response:")
        print(response.text)

        raise RuntimeError(
            "Prediction API returned "
            f"HTTP {response.status_code}"
        )

    result = response.json()

    # -----------------------------------------------------
    # 5. Validate production response
    # -----------------------------------------------------

    print("\n[5/5] Validating production response...")

    # -----------------------------------------------------
    # Basic response fields
    # -----------------------------------------------------

    required_fields = [
        "customer_id",
        "prediction",
        "risk_level",
        "decision",
        "probability",
        "ensemble_probability",
        "threshold",
        "model_probabilities",
        "llm_explanation",
    ]

    for field in required_fields:

        if field not in result:

            raise AssertionError(
                f"Required response field is missing: "
                f"{field}"
            )

    print("✓ Required production response fields present")

    # -----------------------------------------------------
    # Extract prediction values
    # -----------------------------------------------------

    prediction = int(
        result["prediction"]
    )

    calibrated_probability = float(
        result["probability"]
    )

    ensemble_probability = float(
        result["ensemble_probability"]
    )

    threshold = float(
        result["threshold"]
    )

    risk_level = result["risk_level"]

    decision = result["decision"]

    model_probabilities = (
        result["model_probabilities"]
    )

    lightgbm_probability = float(
        model_probabilities["lightgbm"]
    )

    xgboost_probability = float(
        model_probabilities["xgboost"]
    )

    catboost_probability = float(
        model_probabilities["catboost"]
    )

    # -----------------------------------------------------
    # Probability validation
    # -----------------------------------------------------

    probabilities = {

        "LightGBM": lightgbm_probability,

        "XGBoost": xgboost_probability,

        "CatBoost": catboost_probability,

        "Raw Ensemble": ensemble_probability,

        "Calibrated": calibrated_probability,
    }

    for name, probability in probabilities.items():

        if not 0.0 <= probability <= 1.0:

            raise AssertionError(
                f"{name} probability is outside "
                f"[0, 1]: {probability}"
            )

    print("✓ All probabilities are within [0, 1]")

    # -----------------------------------------------------
    # Threshold validation
    # -----------------------------------------------------

    if abs(threshold - EXPECTED_THRESHOLD) > 1e-9:

        raise AssertionError(
            "Unexpected production threshold: "
            f"{threshold}"
        )

    print(
        f"✓ Production threshold confirmed: "
        f"{threshold:.4f}"
    )

    # -----------------------------------------------------
    # Prediction validation
    # -----------------------------------------------------

    expected_prediction = int(
        calibrated_probability >= threshold
    )

    if prediction != expected_prediction:

        raise AssertionError(
            "Final prediction does not match "
            "calibrated probability and threshold."
        )

    print(
        "✓ Final prediction uses calibrated "
        "probability"
    )

    # -----------------------------------------------------
    # Risk level validation
    # -----------------------------------------------------

    expected_risk_level = (
        "High Risk"
        if expected_prediction == 1
        else "Low Risk"
    )

    if risk_level != expected_risk_level:

        raise AssertionError(
            "Risk level does not match "
            "final prediction."
        )

    print(
        "✓ Risk level matches final prediction"
    )

    # -----------------------------------------------------
    # LLM explanation validation
    # -----------------------------------------------------

    llm_explanation = result[
        "llm_explanation"
    ]

    if not isinstance(
        llm_explanation,
        dict,
    ):

        raise AssertionError(
            "llm_explanation must be "
            "a JSON object."
        )

    required_llm_fields = [
        "summary",
        "risk_explanation",
        "key_risk_factors",
        "protective_factors",
        "analyst_actions",
    ]

    for field in required_llm_fields:

        if field not in llm_explanation:

            raise AssertionError(
                "Required LLM explanation field "
                f"is missing: {field}"
            )

    print(
        "✓ Gemini explanation object "
        "present in API response"
    )

    # -----------------------------------------------------
    # LLM text validation
    # -----------------------------------------------------

    text_fields = [
        "summary",
        "risk_explanation",
    ]

    for field in text_fields:

        value = llm_explanation[field]

        if not isinstance(
            value,
            str,
        ):

            raise AssertionError(
                f"LLM field '{field}' "
                "must be a string."
            )

        if not value.strip():

            raise AssertionError(
                f"LLM field '{field}' "
                "must not be empty."
            )

    print(
        "✓ Gemini textual explanation "
        "fields are valid"
    )

    # -----------------------------------------------------
    # LLM list validation
    # -----------------------------------------------------

    list_fields = [
        "key_risk_factors",
        "protective_factors",
        "analyst_actions",
    ]

    for field in list_fields:

        value = llm_explanation[field]

        if not isinstance(
            value,
            list,
        ):

            raise AssertionError(
                f"LLM field '{field}' "
                "must be a list."
            )

        for item in value:

            if not isinstance(
                item,
                str,
            ):

                raise AssertionError(
                    f"Every item in "
                    f"'{field}' must be a string."
                )

    print(
        "✓ Gemini structured lists "
        "are valid"
    )

    # -----------------------------------------------------
    # LLM decision consistency validation
    # -----------------------------------------------------

    explanation_text = (
        llm_explanation["summary"]
        + " "
        + llm_explanation["risk_explanation"]
    ).lower()

    if prediction == 0:

        if "low risk" not in explanation_text:

            raise AssertionError(
                "Gemini explanation does not "
                "reflect the production Low Risk "
                "result."
            )

    elif prediction == 1:

        if "high risk" not in explanation_text:

            raise AssertionError(
                "Gemini explanation does not "
                "reflect the production High Risk "
                "result."
            )

    print(
        "✓ Gemini explanation is consistent "
        "with production risk result"
    )

    # -----------------------------------------------------
    # Display final result
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("FASTAPI PRODUCTION PREDICTION RESULT")
    print("=" * 70)

    print(
        f"Customer ID             : "
        f"{result['customer_id']}"
    )

    print(
        f"LightGBM probability    : "
        f"{lightgbm_probability:.6f}"
    )

    print(
        f"XGBoost probability     : "
        f"{xgboost_probability:.6f}"
    )

    print(
        f"CatBoost probability    : "
        f"{catboost_probability:.6f}"
    )

    print(
        f"Ensemble probability    : "
        f"{ensemble_probability:.6f}"
    )

    print(
        f"Calibrated probability  : "
        f"{calibrated_probability:.6f}"
    )

    print(
        f"Production threshold    : "
        f"{threshold:.6f}"
    )

    print(
        f"Final prediction        : "
        f"{prediction}"
    )

    print(
        f"Risk decision           : "
        f"{risk_level}"
    )

    print(
        f"Decision                : "
        f"{decision}"
    )

    print()
    print("GEMINI EXPLANATION")
    print("-" * 70)

    print(
        f"Summary:\n"
        f"{llm_explanation['summary']}"
    )

    print()

    print(
        f"Risk Explanation:\n"
        f"{llm_explanation['risk_explanation']}"
    )

    print()

    print("Key Risk Factors:")

    for factor in llm_explanation[
        "key_risk_factors"
    ]:

        print(f"  - {factor}")

    print()

    print("Protective Factors:")

    for factor in llm_explanation[
        "protective_factors"
    ]:

        print(f"  - {factor}")

    print()

    print("Analyst Actions:")

    for action in llm_explanation[
        "analyst_actions"
    ]:

        print(f"  - {action}")

    print("=" * 70)
    print("✓ FASTAPI PRODUCTION + GEMINI TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
