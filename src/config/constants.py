"""
===========================================================
CreditLens-AI
Application Constants
===========================================================
"""

# Application
PROJECT_NAME = "CreditLens-AI"
VERSION = "1.0.0"

# Random Seed
RANDOM_STATE = 42

# Ensemble Weights
LIGHTGBM_WEIGHT = 0.35
XGBOOST_WEIGHT = 0.35
CATBOOST_WEIGHT = 0.30

# Default Threshold
# ============================================================
# Default Production Decision Threshold
# ============================================================

# Calibrated production threshold selected in Notebook 18.
# This is only a fallback value.
# The primary production value comes from:
# models/ensemble_config.json
DEFAULT_THRESHOLD = 0.18

# SHAP
SHAP_BACKGROUND_SAMPLES = 100

# LLM
LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS = 512

# Database
DATABASE_NAME = "creditlens_ai"

# Risk Labels
LOW_RISK = "Low Risk"
MEDIUM_RISK = "Medium Risk"
HIGH_RISK = "High Risk"

# Business Decisions
APPROVE = "Approve"
MANUAL_REVIEW = "Manual Review"
REJECT = "Reject"