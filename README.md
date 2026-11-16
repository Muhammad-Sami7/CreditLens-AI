# CreditLens-AI

**Explainable credit-risk decision support.** CreditLens-AI estimates the
probability that a loan applicant will have payment difficulties, explains the
estimate with SHAP and a plain-language AI summary, and keeps an audit log of
every assessment in PostgreSQL. It supports a loan officer's decision; it does not make it.

Built on the public [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk)
dataset. **This is a portfolio prototype, not a lending product.**

| ROC-AUC | Gini | KS | PR-AUC | Brier | Models | Features |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0.793** | **0.585** | **0.442** | **0.284** | **0.065** | LightGBM + XGBoost + CatBoost | 297 |

<sub>Calibrated ensemble on a 61,503-customer validation split (8.07% defaulters).
PR-AUC of a random model: 0.081. Brier of "always predict the average": 0.074.
See the [evaluation note](#model-performance) before treating these as final-test numbers.</sub>

<p align="center">
  <img src="docs/images/demo.gif" alt="Risk assessment walkthrough: result, SHAP evidence, AI explanation, PDF report" width="900">
</p>

<p align="center">
  <img src="docs/images/screenshot_dashboard.png" alt="Dashboard" width="800">
  <br>
  <sub>Dashboard: assessments, risk split and latest results</sub>
</p>

<p align="center">
  <img src="docs/images/screenshot_report_pdf.jpeg" alt="PDF report" width="560">
  <br>
  <sub>Downloadable PDF report for each assessment</sub>
</p>

## What it does

1. An analyst looks up a customer and runs an assessment.
2. A weighted ensemble of three gradient-boosted models scores the customer;
   isotonic calibration turns the score into a probability that matches observed
   default frequencies.
3. A business rule flags the customer for review when the calibrated probability
   is at or above **0.18**.
4. SHAP shows which fields pushed the estimate up or down, and a Gemini-written
   summary explains that evidence in plain language.
5. The assessment is saved to PostgreSQL as an audit record (decision, probability,
   threshold, model version, ensemble weights, input completeness, SHAP factors and
   the explanation shown) and appears on the dashboard and in the customer's
   history. A PDF report can be downloaded.

If the AI-explanation service is down, the prediction, decision and SHAP evidence
are still returned; only the written summary is replaced by a notice.

## Architecture

```mermaid
flowchart LR
    UI["Streamlit app<br/>(frontend/)"] -- HTTP --> API["FastAPI<br/>(src/api)"]
    API --> PS["PredictionService"]
    PS --> FS["Feature service<br/>297 features"]
    PS --> ENS["Ensemble + isotonic calibration<br/>(models/)"]
    ENS --> RE["RiskEngine<br/>threshold 0.18"]
    PS --> SHAP["SHAP explainer"]
    PS --> LLM["Gemini explanation"]
    PS --> DB[("PostgreSQL")]
```

- **The frontend never decides.** Probabilities, thresholds and decisions come only
  from the API; the app displays them.
- **The threshold applies to the calibrated probability**, never to the raw
  ensemble score.
- **The explanation is optional.** The decision never depends on Gemini.
- **The database is an audit log, not part of the model.** Predictions do not
  depend on PostgreSQL; it records each assessment so a past decision can be
  reviewed and traced to the model version, weights and threshold that produced it.
  If the record cannot be saved, the API returns an error instead of an
  unrecorded decision.

More detail: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) ·
[docs/API.md](docs/API.md) · [docs/MODEL_CARD.md](docs/MODEL_CARD.md)

## Model performance

Evaluated on a 61,503-customer validation split (8.07% defaulters); the other
246,008 customers of `application_train` were used for training. Every number
below was recomputed from the saved model files without retraining
(`scripts/compare_models_colab.py`).

### Seven models, one ensemble

| Model | ROC-AUC | Gini | KS | PR-AUC | Top-decile lift |
|---|:---:|:---:|:---:|:---:|:---:|
| **Ensemble, calibrated (production)** | **0.7926** | **0.585** | **0.442** | 0.284 | 3.83x |
| Ensemble, raw | 0.7912 | 0.582 | 0.442 | 0.291 | 3.82x |
| XGBoost (optimized) | 0.7901 | 0.580 | 0.441 | 0.288 | 3.79x |
| XGBoost (regularized) | 0.7897 | 0.579 | 0.439 | 0.288 | 3.79x |
| LightGBM (regularized) | 0.7895 | 0.579 | 0.441 | 0.288 | 3.78x |
| CatBoost (optimized) | 0.7891 | 0.578 | 0.434 | 0.288 | 3.79x |
| LightGBM (optimized) | 0.7888 | 0.578 | 0.439 | 0.287 | 3.75x |
| LightGBM (baseline) | 0.7875 | 0.575 | 0.437 | 0.285 | 3.77x |
| CatBoost (regularized) | 0.7840 | 0.568 | 0.432 | 0.281 | 3.70x |

<p align="center">
  <img src="docs/images/roc_all_models.png" alt="ROC curves of all models" width="480">
  <img src="docs/images/auc_confidence_intervals.png" alt="ROC-AUC of each model with 95% bootstrap confidence intervals" width="480">
</p>

**The models are statistically tied.** With about 5,000 defaulters in the
validation split, ROC-AUC has a 95% bootstrap interval of roughly +/-0.007, and the
intervals of all models overlap almost completely. Their rankings are also highly
redundant (pairwise rank correlation 0.95-0.99). The limit is the information in
the 297 features, not the algorithm, so more tuning would not help much.

The ensemble is used for stability, a calibrated probability, a
"models agreeing" signal and SHAP evidence across all three models, not because
it is much more accurate: it is +0.005 AUC over the baseline LightGBM and
+0.0025 over the best single model. The ensemble overfits somewhat
(train AUC 0.873 vs 0.791 on validation).

### Calibration

Raw model scores run far above real default rates (they were trained with class
weighting): an applicant scored 0.35 defaulted roughly 5% of the time. Isotonic
calibration fixes that.

| | Brier score |
|---|:---:|
| Always predict the average default rate (baseline) | 0.074 |
| Raw ensemble | 0.165 |
| Platt scaling | 0.0657 |
| **Isotonic calibration (production)** | **0.0653** |

For a fair comparison, every model was calibrated the same way (isotonic,
5-fold cross-fitted so each probability is out-of-sample). Afterwards all models
are almost identical:

| Model | Brier (raw) | Brier (calibrated) | ECE (calibrated) |
|---|:---:|:---:|:---:|
| Ensemble | 0.1650 | **0.0655** | 0.0013 |
| XGBoost (regularized) | 0.1762 | 0.0656 | 0.0012 |
| XGBoost (optimized) | 0.1701 | 0.0657 | 0.0012 |
| CatBoost (optimized) | 0.1639 | 0.0657 | 0.0014 |
| LightGBM (regularized) | 0.1627 | 0.0657 | 0.0017 |
| LightGBM (optimized) | 0.1416 | 0.0657 | 0.0011 |
| LightGBM (baseline) | 0.1615 | 0.0658 | 0.0021 |
| CatBoost (regularized) | 0.1829 | 0.0660 | 0.0013 |

<p align="center">
  <img src="docs/images/calibration_fair_all_models.png" alt="Calibration of all models after equal isotonic calibration" width="480">
</p>

### What it means for a lender

<p align="center">
  <img src="docs/images/lift_deciles.png" alt="Lift and cumulative capture by risk decile" width="640">
</p>

The riskiest 10% of applicants default at 3.8 times the average rate, and the
riskiest 30% contain 70.7% of all defaulters. The production threshold (0.18)
maximises F1 on calibrated probabilities. It is a screening aid: most flagged
applicants do not default.

| Precision | Recall | F1 | Flagged |
|:---:|:---:|:---:|:---:|
| 30.4% | 39.5% | 0.344 | 10.5% of customers |

<p align="center">
  <img src="docs/images/threshold_tradeoff.png" alt="Precision, recall and F1 across thresholds" width="480">
</p>

<details>
<summary>Confusion matrix at the production threshold</summary>

52,053 true negatives, 4,485 false positives, 3,005 false negatives and 1,960
true positives. Accuracy (87.8%) is not informative here because about 92% of
customers do not default. More plots (KS curve, precision-recall, confusion
matrix) are in [docs/MODEL_CARD.md](docs/MODEL_CARD.md).

</details>

> **Honest evaluation note.** The validation split was used during development
> (model tuning and selection), and the isotonic calibrator and the 0.18
> threshold were fitted on that same split. The calibrated metrics are therefore
> not an independent final test. Cross-fitting the calibration changes the
> ensemble's Brier score only from 0.0653 to 0.0655.

## Tech stack

| Layer | Technology |
|---|---|
| Models | LightGBM, XGBoost, CatBoost (weights 0.35 / 0.35 / 0.30) |
| Calibration | Isotonic regression (scikit-learn) |
| Explainability | SHAP (TreeExplainer) across all three models |
| LLM | Gemini API (explanation only; never changes the decision) |
| Backend | FastAPI, SQLAlchemy (async), Pydantic |
| Database | PostgreSQL (assessment audit log) |
| Frontend | Streamlit |

## Project structure

```
src/
  api/              FastAPI routes and schemas
  business/         RiskEngine (threshold, risk level, decision)
  config/           settings and paths
  database/         SQLAlchemy models and repository (audit log)
  explainability/   SHAP for the three-model ensemble
  features/         feature contract and preparation (297 features)
  models/           model loading, predictor, ensemble
  services/         prediction pipeline, customer lookup, Gemini explanation
frontend/           Streamlit app (pages, components, API client)
models/             trained models, calibrator and configuration
scripts/            environment checks, tests and audits
docs/               architecture, API, model card, images
```

Training notebooks and the dataset are not part of this repository. The model
comparison can be reproduced from the saved models with
`scripts/compare_models_colab.py`.

## Run it locally (Windows PowerShell)

Requirements: Python, PostgreSQL, a Gemini API key, the model files in `models/`,
and the processed Home Credit data at `data/processed/test_processed.parquet`
(the customer lookup reads it; download the dataset from Kaggle).

```powershell
git clone https://github.com/Muhammad-Sami7/CreditLens-AI.git
cd CreditLens-AI
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create `.env` in the project root (it is git-ignored; never commit it):

```
DATABASE_URL=postgresql+asyncpg://USER:PASSWORD@localhost:5432/creditlens
GEMINI_API_KEY=your-key
```

```powershell
python scripts\check_environment.py        # prints names only, never secrets
python scripts\create_database_tables.py   # new database
# existing database from an earlier version: python -m scripts.migrate_audit_columns

# terminal 1 - backend (loads the models; takes a while to become ready)
uvicorn src.main:app --host 127.0.0.1 --port 8000

# terminal 2 - frontend
cd frontend
streamlit run app.py
```

Interactive API docs: `http://127.0.0.1:8000/docs`

## Tests and audits

```powershell
python scripts\test_phase2_endpoints_offline.py   # API endpoints against a stub model
python scripts\test_pdf_report.py                 # PDF report
python scripts\test_feature_names.py              # plain-language feature names
python scripts\test_shap_ensemble.py              # SHAP faithfulness, real models
python scripts\test_api_phase2_live.py            # needs the backend running
python scripts\check_threshold.py                 # read-only audit of the 0.18 threshold
python scripts\evaluate_gender_impact.py          # effect of the gender input, by group
```

`scripts/compare_models_colab.py` reproduces the model-comparison tables and plots
from the saved models (run it in Colab with the project on Google Drive).

## Limitations

- **Public data from another lender.** It has not been validated on any real
  portfolio, with a time-based test, or for drift. Do not use it for real lending.
- **The model choice is not statistically supported by accuracy alone.** The
  seven models are within noise of each other; see the comparison above.
- **Development validation only.** There is no untouched final test set (see the
  evaluation note above).
- **Gender is a model input** because the dataset contains it and the model was
  trained with it. Neutralising it changes ROC-AUC only from
  0.857 to 0.853 (in-sample, 40,000 customers), but it changes who is flagged
  (men 15.5% with it, 11.6% without). Many jurisdictions restrict
  its use in credit decisions; the next retrain should drop it. See
  [docs/MODEL_CARD.md](docs/MODEL_CARD.md).
- **The strongest signals are anonymised external credit scores** that a real
  lender would have to replace with its own.
- **Tree models do not extrapolate**, so the app assesses customers from the
  dataset rather than hand-typed applicants.
- SHAP shows how the model scored a customer, not what caused an outcome.
- Gemini text is an AI-written narrative of the model's evidence, not an
  independent assessment.

## Roadmap

- Retrain without the gender input and re-evaluate on a held-out test set
- Better features from the bureau and instalment tables (the model, not tuning, is the limit)
- Docker / docker-compose
- Batch upload of applicants
- Automated pytest suite

## License

See [LICENSE](LICENSE).

## Author

Muhammad Sami
