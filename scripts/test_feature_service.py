"""
============================================================
CreditLens-AI
Production Feature Service Test
============================================================
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.features.feature_service import (
    ProductionFeatureService,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test_processed.parquet"
)

ID_COLUMN = "SK_ID_CURR"


def main() -> None:

    print("=" * 70)
    print("CreditLens-AI - Production Feature Service Test")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load test data
    # --------------------------------------------------------

    print("\n[1/5] Loading processed test dataset...")

    if not TEST_DATA_PATH.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n{TEST_DATA_PATH}"
        )

    test_df = pd.read_parquet(
        TEST_DATA_PATH
    )

    print("✓ Test dataset loaded")
    print(f"  Rows    : {len(test_df):,}")
    print(f"  Columns : {len(test_df.columns)}")

    # --------------------------------------------------------
    # 2. Remove customer identifier
    # --------------------------------------------------------

    print("\n[2/5] Preparing feature data...")

    if ID_COLUMN not in test_df.columns:

        raise ValueError(
            f"{ID_COLUMN} not found."
        )

    customer_id = test_df.iloc[0][ID_COLUMN]

    customer_df = (
        test_df
        .iloc[[0]]
        .drop(columns=[ID_COLUMN])
        .copy()
    )

    print(
        f"✓ Customer selected: {customer_id}"
    )

    # --------------------------------------------------------
    # 3. Initialize service
    # --------------------------------------------------------

    print("\n[3/5] Loading production feature schema...")

    service = ProductionFeatureService()

    print(
        f"✓ Production feature count: "
        f"{service.feature_count}"
    )

    # --------------------------------------------------------
    # 4. Prepare customer
    # --------------------------------------------------------

    print("\n[4/5] Preparing production features...")

    prepared_df = service.prepare(
        customer_df,
        allow_missing=True,
    )

    print("✓ Production feature matrix prepared")
    print(
        f"  Rows    : {len(prepared_df)}"
    )
    print(
        f"  Columns : {len(prepared_df.columns)}"
    )

    # --------------------------------------------------------
    # 5. Validate
    # --------------------------------------------------------

    print("\n[5/5] Validating feature contract...")

    assert len(prepared_df.columns) == 297

    assert list(
        prepared_df.columns
    ) == service.feature_columns

    assert (
        prepared_df.columns.duplicated().sum()
        == 0
    )

    print("✓ Feature count = 297")
    print("✓ Feature ordering is correct")
    print("✓ No duplicate features")
    print("✓ Production feature contract verified")

    print()
    print("=" * 70)
    print("FEATURE SERVICE VERIFICATION PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()