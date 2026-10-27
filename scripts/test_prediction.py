"""
===========================================================
CreditLens-AI
Production Prediction Test
===========================================================

Tests the complete production prediction pipeline using
a real customer row from test_processed.parquet.

Pipeline:

test_processed.parquet
        |
297 production features
        |

LightGBM
XGBoost
CatBoost
        |
Weighted Ensemble
        |
Isotonic Calibration
        |
Production Threshold
        |
Final Risk Prediction

Author: Muhammad Sami
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.models.predictor import CreditLensPredictor


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

ID_COLUMN = "SK_ID_CURR"


# =========================================================
# Main Test
# =========================================================

def main() -> None:

    print("=" * 70)
    print("CreditLens-AI - Production Prediction Test")
    print("=" * 70)

    # -----------------------------------------------------
    # 1. Check test data
    # -----------------------------------------------------

    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Test dataset not found:\n{TEST_DATA_PATH}"
        )

    print("\n[1/5] Loading test dataset...")

    test_df = pd.read_parquet(TEST_DATA_PATH)

    print(f"✓ Test dataset loaded")
    print(f"  Rows    : {len(test_df):,}")
    print(f"  Columns : {len(test_df.columns)}")

    # -----------------------------------------------------
    # 2. Validate identifier
    # -----------------------------------------------------

    print("\n[2/5] Validating customer identifier...")

    if ID_COLUMN not in test_df.columns:
        raise ValueError(
            f"Required identifier column '{ID_COLUMN}' "
            "was not found in the test dataset."
        )

    print(f"✓ {ID_COLUMN} found")

    # -----------------------------------------------------
    # 3. Load production predictor
    # -----------------------------------------------------

    print("\n[3/5] Loading production prediction pipeline...")

    predictor = CreditLensPredictor()

    print("✓ Production predictor loaded")

    # -----------------------------------------------------
    # 4. Prepare one real customer
    # -----------------------------------------------------

    print("\n[4/5] Preparing one real test customer...")

    customer_id = test_df.iloc[0][ID_COLUMN]

    customer_df = test_df.iloc[[0]].drop(
        columns=[ID_COLUMN]
    ).copy()

    print(f"✓ Customer ID: {customer_id}")
    print(f"✓ Feature count: {len(customer_df.columns)}")

    # -----------------------------------------------------
    # 5. Generate prediction
    # -----------------------------------------------------

    print("\n[5/5] Generating production prediction...")

    result = predictor.predict(customer_df)
    print("\nDEBUG: Prediction pipeline returned successfully.")
    print("DEBUG: Result keys:", result.keys())
    print("DEBUG: Raw result:", result)
    prediction = int(result["prediction"][0])

    calibrated_probability = float(
        result["probability"][0]
    )

    ensemble_probability = float(
        result["ensemble_probability"][0]
    )

    lightgbm_probability = float(
        result["lightgbm_probability"][0]
    )

    xgboost_probability = float(
        result["xgboost_probability"][0]
    )

    catboost_probability = float(
        result["catboost_probability"][0]
    )

    threshold = float(predictor.ensemble.threshold)

    print()
    print("=" * 70)
    print("PRODUCTION PREDICTION RESULT")
    print("=" * 70)

    print(f"Customer ID             : {customer_id}")
    print(f"LightGBM probability    : {lightgbm_probability:.6f}")
    print(f"XGBoost probability     : {xgboost_probability:.6f}")
    print(f"CatBoost probability     : {catboost_probability:.6f}")
    print(f"Ensemble probability    : {ensemble_probability:.6f}")
    print(f"Calibrated probability  : {calibrated_probability:.6f}")
    print(f"Production threshold    : {threshold:.6f}")
    print(f"Final prediction        : {prediction}")

    if prediction == 1:
        print("Risk decision            : HIGH RISK")
    else:
        print("Risk decision            : LOW RISK")

    print("=" * 70)
    print("✓ Production prediction test completed")
    print("=" * 70)


if __name__ == "__main__":
    main()