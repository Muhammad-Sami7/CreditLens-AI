# CreditLens-AI

**Explainable credit-risk decision support.** CreditLens-AI estimates the
probability that a loan applicant will have payment difficulties, explains the
estimate with SHAP and a plain-language AI summary, and saves every assessment
to PostgreSQL. It supports a loan officer's decision; it does not make it.

Built on the public [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk)
dataset. **This is a portfolio prototype, not a lending product.**

| ROC-AUC | Brier score | Models | Features |
|:---:|:---:|:---:|:---:|
| **0.79** (validation) | **0.065** (raw ensemble: 0.165) | LightGBM + XGBoost + CatBoost | 297 |

<!-- Add your screenshots to docs/images/ with these exact names. -->
<p align="center">
  <img src="docs/images/demo.gif" alt="Risk assessment demo: customer lookup, decision, SHAP evidence and AI explanation" width="900">
</p>

<p align="center">
  <img src="docs/images/screenshot_dashboard.png" alt="Dashboard" width="440">
  <img src="docs/images/screenshot_report_pdf.jpeg" alt="PDF report" width="440">
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
5. The assessment is saved to PostgreSQL and appears on the dashboard and in the
   customer's history. A PDF report can be downloaded.

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

More detail: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) ·
[docs/API.md](docs/API.md) · [docs/MODEL_CARD.md](docs/MODEL_CARD.md)

## Model performance

Evaluated on a 61,503-customer validation split (8.07% defaulters); the other
246,008 customers of `application_train` were used for training.

| Model | Train ROC-AUC | Validation ROC-AUC | Gap |
|---|:---:|:---:|:---:|
| Weighted ensemble | 0.873 | **0.791** | 0.082 |
| Optimized XGBoost | 0.853 | 0.790 | 0.063 |
| Regularized LightGBM | 0.881 | 0.789 | 0.092 |
| Optimized CatBoost | 0.879 | 0.789 | 0.090 |

<p align="center">
  <img src="docs/images/roc_curve_comparison.png" alt="ROC curves of the three models" width="520">
</p>

The ensemble ranks applicants slightly better than any single model. All four
models overfit somewhat (train AUC about 0.85-0.88 against 0.79 on validation).

### Calibration

Raw ensemble scores run far above real default rates: an applicant scored 0.35
defaulted roughly 5% of the time. Isotonic calibration fixes that.

| | Brier score |
|---|:---:|
| Always predict the average default rate (baseline) | 0.074 |
| Raw ensemble | 0.165 |
| Platt scaling | 0.0657 |
| **Isotonic calibration (production)** | **0.0653** |

<p align="center">
  <img src="docs/images/calibration_curve.png" alt="Calibration curve: raw ensemble, Platt scaling, isotonic regression" width="480">
</p>

After calibration the ensemble's ROC-AUC is 0.7926 and its PR-AUC is 0.284
(a random model scores 0.081).

<details>
<summary>Operating point at the production threshold (0.18)</summary>

The threshold maximises F1 on calibrated probabilities. It is a screening aid:
most flagged applicants do not default.

| Precision | Recall | F1 | Flagged |
|:---:|:---:|:---:|:---:|
| 30.4% | 39.5% | 0.344 | 10.5% of customers |

Confusion matrix on the validation split: 52,053 true negatives, 4,485 false
positives, 3,005 false negatives, 1,960 true positives. Accuracy (87.8%) is not
informative here because about 92% of customers do not default.

</details>

> **Honest evaluation note.** The validation split was used during development
> (model tuning and selection), and the isotonic calibrator and the 0.18
> threshold were fitted on that same split. The calibrated metrics are therefore
> not an independent final test.

## Tech stack

| Layer | Technology |
|---|---|
| Models | LightGBM, XGBoost, CatBoost (weights 0.35 / 0.35 / 0.30) |
| Calibration | Isotonic regression (scikit-learn) |
| Explainability | SHAP (TreeExplainer) across all three models |
| LLM | Gemini API (explanation only; never changes the decision) |
| Backend | FastAPI, SQLAlchemy (async), Pydantic |
| Database | PostgreSQL |
| Frontend | Streamlit |

## Project structure

```
src/
  api/              FastAPI routes and schemas
  business/         RiskEngine (threshold, risk level, decision)
  config/           settings and paths
  database/         SQLAlchemy models and repository
  explainability/   SHAP for the three-model ensemble
  features/         feature contract and preparation (297 features)
  models/           model loading, predictor, ensemble
  services/         prediction pipeline, customer lookup, Gemini explanation
frontend/           Streamlit app (pages, components, API client)
models/             trained models, calibrator and configuration
scripts/            environment checks, tests and audits
docs/               architecture, API, model card, images
```

Training notebooks and the dataset are not part of this repository.

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
python scripts\create_database_tables.py

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

## Limitations

- **Public data from another lender.** It has not been validated on any real
  portfolio, with a time-based test, or for drift. Do not use it for real lending.
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
- Docker / docker-compose
- Batch upload of applicants
- Automated pytest suite

## License

See [LICENSE](LICENSE).

## Author

Muhammad Sami
