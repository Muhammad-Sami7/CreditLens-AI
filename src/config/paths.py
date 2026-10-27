"""
===========================================================
CreditLens-AI
Project Paths
===========================================================
"""

from pathlib import Path

# Project Root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
DOCS_DIR = PROJECT_ROOT / "docs"

# Model Files
LIGHTGBM_MODEL_PATH = MODELS_DIR / "regularized_lightgbm.pkl"
XGBOOST_MODEL_PATH = MODELS_DIR / "optimized_xgboost.pkl"
CATBOOST_MODEL_PATH = MODELS_DIR / "optimized_catboost.pkl"

ISOTONIC_MODEL_PATH = MODELS_DIR / "isotonic_calibrator.pkl"

FEATURE_COLUMNS_PATH = MODELS_DIR / "feature_columns.json"
ENSEMBLE_CONFIG_PATH = MODELS_DIR / "ensemble_config.json"

MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"

MODEL_COMPARISON_PATH = MODELS_DIR / "model_comparison.csv"
THRESHOLD_RESULTS_PATH = MODELS_DIR / "threshold_results.csv"

CALIBRATION_RESULTS_PATH = MODELS_DIR / "calibration_results.json"
BRIER_SCORES_PATH = MODELS_DIR / "brier_scores.csv"
CALIBRATION_CURVE_PATH = MODELS_DIR / "calibration_curve.csv"

ENSEMBLE_IMPORTANCE_PATH = MODELS_DIR / "ensemble_feature_importance.csv"