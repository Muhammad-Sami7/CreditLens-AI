"""
============================================================
CreditLens-AI
Production SHAP Explainability Test
============================================================

Tests the production SHAP explainability pipeline using
a real customer row from test_processed.parquet.

Pipeline:

Test Customer
     |
Production Feature Service
     |
297 Production Features
     |
Production Model Loader
     |
LightGBM
     |
SHAP Explainer
     |
Feature Contributions
     |
Top Risk Factors

Author: Muhammad Sami
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.explainability.shap_explainer import SHAPExplainer
from src.features.feature_service import ProductionFeatureService
from src.models.model_loader import ModelLoader


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test_processed.parquet"
)

ID_COLUMN = "SK_ID_CURR"


# ============================================================
# Main Test
# ============================================================

def main() -> None:

    print("=" * 70)
    print("CreditLens-AI - Production SHAP Explainability Test")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load test dataset
    # --------------------------------------------------------

    print("\n[1/6] Loading processed test dataset...")

    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Test dataset not found:\n{TEST_DATA_PATH}"
        )

    test_df = pd.read_parquet(TEST_DATA_PATH)

    print("✓ Test dataset loaded")
    print(f"  Rows    : {len(test_df):,}")
    print(f"  Columns : {len(test_df.columns)}")

    # --------------------------------------------------------
    # 2. Validate customer identifier
    # --------------------------------------------------------

    print("\n[2/6] Validating customer identifier...")

    if ID_COLUMN not in test_df.columns:
        raise ValueError(
            f"Required identifier column '{ID_COLUMN}' "
            "was not found."
        )

    print(f"✓ {ID_COLUMN} found")

    # --------------------------------------------------------
    # 3. Prepare real customer
    # --------------------------------------------------------

    print("\n[3/6] Preparing real test customer...")

    customer_id = test_df.iloc[0][ID_COLUMN]

    customer_df = (
        test_df.iloc[[0]]
        .drop(columns=[ID_COLUMN])
        .copy()
    )

    print(f"✓ Customer ID: {customer_id}")

    # --------------------------------------------------------
    # 4. Load production feature service
    # --------------------------------------------------------

    print("\n[4/6] Loading production feature service...")

    feature_service = ProductionFeatureService()

    production_features = (
        feature_service.prepare(
            customer_df
        )
    )

    print("✓ Production features prepared")
    print(
        f"  Rows    : {len(production_features)}"
    )
    print(
        f"  Columns : {len(production_features.columns)}"
    )

    # --------------------------------------------------------
    # 5. Load production models
    # --------------------------------------------------------

    print("\n[5/6] Loading production models...")

    model_loader = ModelLoader()
    model_loader.load_models()

    models = model_loader.get_models()

    print("✓ Production models loaded")
    print(
        f"  Models : "
        f"{', '.join(models.keys())}"
    )

    # --------------------------------------------------------
    # 6. Generate SHAP explanation
    # --------------------------------------------------------

    print("\n[6/6] Generating SHAP explanation...")

    explainer = SHAPExplainer(
        model_loader=model_loader
    )

    explanation = explainer.explain(
        production_features
    )

    print("✓ SHAP explanation generated")

    # --------------------------------------------------------
    # Production explanation result
    # --------------------------------------------------------

    print()

    print("=" * 70)

    print(
        "PRODUCTION SHAP EXPLANATION"
    )

    print("=" * 70)

    print(
        f"Customer ID : {customer_id}"
    )

    print(
        f"Feature count : "
        f"{len(explainer.feature_columns)}"
    )

    print(
        f"SHAP values : "
        f"{len(explanation['shap_values'])}"
    )

    # --------------------------------------------------------
    # Display top positive risk factors
    # --------------------------------------------------------

    positive_factors = (
        explanation.get(
            "positive_factors",
            []
        )
    )

    negative_factors = (
        explanation.get(
            "negative_factors",
            []
        )
    )

    print()
    print("-" * 70)
    print("TOP POSITIVE RISK FACTORS")
    print("-" * 70)

    if positive_factors:

        for index, factor in enumerate(
            positive_factors[:10],
            start=1,
        ):

            print(
                f"{index:>2}. "
                f"{factor['feature']} "
                f"| SHAP: "
                f"{factor['shap_value']:.6f}"
            )

    else:

        print("No positive risk factors found.")

    print()
    print("-" * 70)
    print("TOP NEGATIVE RISK FACTORS")
    print("-" * 70)

    if negative_factors:

        for index, factor in enumerate(
            negative_factors[:10],
            start=1,
        ):

            print(
                f"{index:>2}. "
                f"{factor['feature']} "
                f"| SHAP: "
                f"{factor['shap_value']:.6f}"
            )

    else:

        print("No negative risk factors found.")

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("VALIDATION")
    print("-" * 70)

    if len(production_features.columns) != 297:

        raise ValueError(
            "Production feature count is not 297."
        )

    if len(explanation["shap_values"]) != 297:

        raise ValueError(
            "SHAP explanation does not contain "
            "297 feature contributions."
        )

    print("✓ Feature count = 297")
    print("✓ SHAP contribution count = 297")
    print("✓ SHAP explanation generated successfully")

    print()
    print("=" * 70)
    print(
        "SHAP EXPLAINABILITY VERIFICATION PASSED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()

