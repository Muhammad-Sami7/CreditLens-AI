from pathlib import Path

# Root directory
ROOT_DIR = Path(__file__).resolve().parents[2]

# Models
MODELS_DIR = ROOT_DIR / "models"

LIGHTGBM_MODEL = MODELS_DIR / "regularized_lightgbm.pkl"
XGBOOST_MODEL = MODELS_DIR / "optimized_xgboost.pkl"
CATBOOST_MODEL = MODELS_DIR / "optimized_catboost.pkl"

CALIBRATOR = MODELS_DIR / "isotonic_calibrator.pkl"

FEATURE_COLUMNS = MODELS_DIR / "feature_columns.json"
ENSEMBLE_CONFIG = MODELS_DIR / "ensemble_config.json"
MODEL_METADATA = MODELS_DIR / "model_metadata.json"

# Logging

LOG_LEVEL = "INFO"