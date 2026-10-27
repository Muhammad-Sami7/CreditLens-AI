# Architecture

## Request flow (assess by customer ID)

1. The analyst enters an ID in Streamlit (`frontend/pages/risk_assessment.py`).
2. `frontend/api_client.py` calls `POST /api/v1/assess/{customer_id}`. All HTTP lives in this file.
3. The route loads the customer's 297 production features from the feature store
   (`src/services/customer_feature_store.py`, backed by `data/processed/test_processed.parquet`).
4. Missing values (`None`) become NaN, which the tree models handle natively.
5. `PredictionService.predict()`:
   - validates and orders the features (`src/features`)
   - scores with LightGBM, XGBoost and CatBoost, combines them with the configured weights
   - applies the isotonic calibrator
   - `RiskEngine` compares the calibrated probability with the production threshold
   - SHAP explains the prediction
   - the Gemini service writes the narrative from that evidence
   - the result is saved to PostgreSQL
6. The response returns to Streamlit, which renders it.

## Components

| Path | Responsibility |
|------|----------------|
| `src/api/routes.py`, `schemas.py` | HTTP endpoints and request/response models |
| `src/services/prediction_service.py` | Orchestrates the whole pipeline |
| `src/models/` | Model loading, predictor, ensemble |
| `src/explainability/shap_explainer.py` | SHAP for LightGBM, XGBoost and CatBoost, combined with the ensemble weights |
| `src/business/risk_engine.py` | Threshold, risk level, decision |
| `src/services/llm_explanation_service.py` | Gemini explanation, checked against the SHAP evidence |
| `src/database/` | SQLAlchemy models, repository, connection |
| `frontend/` | Streamlit UI (display only) |

## Design decisions

- **Frontend never decides.** Probabilities, thresholds and decisions come only from the API.
- **Calibrated probability.** The threshold is applied after isotonic calibration.
- **Startup.** The service and models load when the API starts, so the API is slow to become ready; the frontend shows a clear "unavailable" message until it is.
- **Graceful degradation.** The frontend reads `/openapi.json` and only offers features the connected backend provides.
- **Explanation is optional.** If the Gemini service fails, the prediction, decision and SHAP evidence are still returned and saved; only the written summary is replaced by a notice.
- **Secrets** are read from environment variables (`.env`), never stored in code.
