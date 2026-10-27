# API reference

Base URL: `http://127.0.0.1:8000/api/v1` (interactive docs at `/docs`).

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Service status |
| POST | `/predict` | Assess with caller-supplied features (`customer_id`, `features` object); keys are production feature names, `null` = missing |
| POST | `/assess/{customer_id}` | Assess a customer; features are looked up server-side. 404 if unknown, 503 if the feature store is unavailable |
| GET | `/customers/sample?limit=20` | Customer IDs available for lookup (1-100) |
| GET | `/customers/{customer_id}/predictions?limit=50` | Saved assessments, newest first (1-200) |
| GET | `/dashboard/summary` | Totals, high/low counts, average estimated risk, recent assessments |

## Assessment response (`/predict`, `/assess`)

`prediction`, `risk_level`, `decision`, `probability` (calibrated), `ensemble_probability`
(uncalibrated), `threshold`, `model_probabilities`, `llm_explanation`
(summary, risk explanation, key risk factors, protective factors, analyst actions),
`prediction_id`, `shap` (top factors with direction and contribution),
`feature_completeness` (expected, provided, missing and null counts).

## Errors

| Status | Meaning |
|--------|---------|
| 400 | The models could not accept the data |
| 404 | Customer not found |
| 422 | Invalid request body |
| 503 | A dependency (such as the feature store) is unavailable |

If the Gemini explanation service fails, the assessment still succeeds:
`llm_explanation.summary` then says the written explanation is temporarily
unavailable, its lists are empty, and the decision, SHAP evidence and saved
record are unaffected.
