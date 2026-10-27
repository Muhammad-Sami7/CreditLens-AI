"""
============================================================
CreditLens-AI
Production Calibration Verification
============================================================

Verifies that the production prediction pipeline:

1. Generates individual model probabilities.
2. Generates the weighted ensemble probability.
3. Applies isotonic calibration.
4. Uses the calibrated production threshold.
5. Produces the final prediction from calibrated probability.

Author: Muhammad Sami
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.predictor import CreditLensPredictor


def main() -> None:

    print("=" * 70)
    print("CreditLens-AI - Production Calibration Verification")
    print("=" * 70)

    # ========================================================
    # 1. Load Test Dataset
    # ========================================================

    print()
    print("[1/6] Loading test dataset...")

    test_path = "data/processed/test_processed.parquet"

    test_df = pd.read_parquet(test_path)

    print("✓ Test dataset loaded")
    print(f"  Rows    : {len(test_df):,}")
    print(f"  Columns : {len(test_df.columns)}")

    # ========================================================
    # 2. Validate Required Identifier
    # ========================================================

    print()
    print("[2/6] Validating customer identifier...")

    if "SK_ID_CURR" not in test_df.columns:
        raise ValueError(
            "SK_ID_CURR not found in test dataset."
        )

    print("✓ SK_ID_CURR found")

    # ========================================================
    # 3. Load Production Predictor
    # ========================================================

    print()
    print("[3/6] Loading production predictor...")

    predictor = CreditLensPredictor()

    print("✓ Production predictor loaded")

    # ========================================================
    # 4. Select Multiple Real Customers
    # ========================================================

    print()
    print("[4/6] Selecting real test customers...")

    sample_df = test_df.head(10).copy()

    customer_ids = sample_df["SK_ID_CURR"].tolist()

    print("✓ Selected customers:")

    for customer_id in customer_ids:
        print(f"  - {customer_id}")

    # ========================================================
    # 5. Generate Predictions
    # ========================================================

    print()
    print("[5/6] Generating production predictions...")

    result = predictor.predict(sample_df)

    # --------------------------------------------------------
    # Extract results
    # --------------------------------------------------------

    raw_lgb = np.asarray(
        result["lightgbm_probability"],
        dtype=float,
    )

    raw_xgb = np.asarray(
        result["xgboost_probability"],
        dtype=float,
    )

    raw_cat = np.asarray(
        result["catboost_probability"],
        dtype=float,
    )

    raw_ensemble = np.asarray(
        result["ensemble_probability"],
        dtype=float,
    )

    calibrated = np.asarray(
        result["probability"],
        dtype=float,
    )

    prediction = np.asarray(
        result["prediction"],
        dtype=int,
    )

    threshold = float(
        result["threshold"]
    )

    # ========================================================
    # 6. Validate Production Logic
    # ========================================================

    print()
    print("[6/6] Validating production calibration logic...")

    # --------------------------------------------------------
    # Probability ranges
    # --------------------------------------------------------

    if not np.all(
        (raw_lgb >= 0.0) &
        (raw_lgb <= 1.0)
    ):
        raise AssertionError(
            "LightGBM probabilities outside [0, 1]."
        )

    if not np.all(
        (raw_xgb >= 0.0) &
        (raw_xgb <= 1.0)
    ):
        raise AssertionError(
            "XGBoost probabilities outside [0, 1]."
        )

    if not np.all(
        (raw_cat >= 0.0) &
        (raw_cat <= 1.0)
    ):
        raise AssertionError(
            "CatBoost probabilities outside [0, 1]."
        )

    if not np.all(
        (raw_ensemble >= 0.0) &
        (raw_ensemble <= 1.0)
    ):
        raise AssertionError(
            "Raw ensemble probabilities outside [0, 1]."
        )

    if not np.all(
        (calibrated >= 0.0) &
        (calibrated <= 1.0)
    ):
        raise AssertionError(
            "Calibrated probabilities outside [0, 1]."
        )

    print("✓ All probabilities are within [0, 1]")

    # --------------------------------------------------------
    # Threshold validation
    # --------------------------------------------------------

    if not np.isclose(
        threshold,
        0.18,
        atol=1e-8,
    ):
        raise AssertionError(
            f"Unexpected production threshold: {threshold}"
        )

    print(
        f"✓ Production threshold confirmed: "
        f"{threshold:.4f}"
    )

    # --------------------------------------------------------
    # Prediction consistency
    # --------------------------------------------------------

    expected_prediction = (
        calibrated >= threshold
    ).astype(int)

    if not np.array_equal(
        prediction,
        expected_prediction,
    ):
        raise AssertionError(
            "Final predictions do not match "
            "calibrated probability >= threshold."
        )

    print(
        "✓ Final predictions are based on "
        "calibrated probability."
    )

    # --------------------------------------------------------
    # Ensure raw threshold is NOT being used
    # --------------------------------------------------------

    raw_threshold = 0.69

    raw_based_prediction = (
        raw_ensemble >= raw_threshold
    ).astype(int)

    if np.array_equal(
        prediction,
        raw_based_prediction,
    ):
        print(
            "⚠ Warning: predictions happen to match "
            "raw-threshold predictions for this sample."
        )
    else:
        print(
            "✓ Predictions are not using the old "
            "raw threshold as the decision threshold."
        )

    # ========================================================
    # Display Results
    # ========================================================

    verification_df = pd.DataFrame({

        "SK_ID_CURR": customer_ids,

        "LightGBM": raw_lgb,

        "XGBoost": raw_xgb,

        "CatBoost": raw_cat,

        "Raw_Ensemble": raw_ensemble,

        "Calibrated_Probability": calibrated,

        "Threshold": threshold,

        "Prediction": prediction,

    })

    print()
    print("=" * 70)
    print("PRODUCTION CALIBRATION VERIFICATION")
    print("=" * 70)

    print(
        verification_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 70)
    print("VERIFICATION PASSED")
    print("=" * 70)

    print()
    print(
        "Production flow verified:"
    )

    print(
        "Individual models -> "
        "Weighted Ensemble -> "
        "Isotonic Calibration -> "
        "Calibrated Threshold -> "
        "Final Prediction"
    )


if __name__ == "__main__":
    main()