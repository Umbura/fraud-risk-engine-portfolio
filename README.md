# FraudRisk Engine

FraudRisk Engine is a portfolio backend for fraud, risk, and anomaly scoring in financial transactions.

The MVP trains lightweight machine learning models on a reproducible synthetic dataset, selects an operational review threshold, exposes a FastAPI scoring endpoint, and returns human-readable reason codes for risky transactions.

The project is intentionally designed to be reproducible without paid APIs, private data, Kaggle credentials, heavy local models, or copied code from external repositories.

## Why This Project Exists

This project targets entry-level and junior roles involving:

- fraud prevention;
- risk and credit modeling;
- anomaly detection;
- Python backend APIs;
- data science and machine learning;
- explainable operational decision support.

It is aligned with companies and roles previously researched for the portfolio: fraud/risk teams, fintechs, data science junior roles, automation roles, and backend Python roles.

## Implemented Scope

- Synthetic transaction dataset generator.
- Train/validation/test split with stratification.
- Two baseline models:
  - logistic regression;
  - random forest.
- Model selection by validation PR-AUC.
- Threshold selection by operational cost:
  - manual review cost;
  - missed fraud loss based on transaction amount.
- Test metrics:
  - ROC-AUC;
  - PR-AUC;
  - precision;
  - recall;
  - F1;
  - confusion matrix.
- Feature importance report.
- Reason-code explanations for individual transactions.
- FastAPI scoring endpoint:
  - `GET /health`;
  - `POST /score`.
- Local test suite.

## Safety and Anti-Overfitting Notes

The current dataset is synthetic. This is useful for a public portfolio because it avoids private financial data, but it can make model performance look cleaner than a real fraud environment.

Controls included in the MVP:

- train/validation/test split;
- threshold selected on validation data, not test data;
- final metrics reported only on the holdout test set;
- target column is explicitly excluded from features;
- duplicate transaction IDs are checked;
- synthetic-data limitation is documented in the metrics report.

The next step for a stronger version is to add a public real-world benchmark dataset and compare performance against this synthetic baseline.

## Quickstart

Install dependencies:

```bash
uv sync --extra dev --extra api
```

Create the dataset, train the model, and write reports:

```bash
uv run python scripts/create_dataset.py
uv run python scripts/train_model.py
uv run python scripts/run_eval.py
```

Start the API:

```bash
uv run uvicorn fraudrisk_engine.api:app --reload
```

Open the docs:

```text
http://127.0.0.1:8000/docs
```

## Example Score Request

```bash
curl -X POST http://127.0.0.1:8000/score \
  -H "Content-Type: application/json" \
  -d "{
    \"amount\": 850.0,
    \"account_age_days\": 12,
    \"customer_tx_count_24h\": 7,
    \"merchant_risk_score\": 0.91,
    \"device_trust_score\": 0.18,
    \"velocity_1h\": 6,
    \"distance_from_home_km\": 980.0,
    \"hour\": 2,
    \"payment_attempts\": 4,
    \"prior_chargebacks\": 1,
    \"is_foreign_card\": 1,
    \"is_high_risk_mcc\": 1,
    \"is_weekend\": 1,
    \"is_new_device\": 1,
    \"customer_segment\": \"new\",
    \"channel\": \"ecommerce\"
  }"
```

## Local Validation

```bash
uv run pytest
uv run ruff check .
uv run python scripts/train_model.py
uv run python scripts/run_eval.py
```

Reports are generated in `reports/` and the model artifact is generated in `models/`. These files are intentionally ignored by Git because they are reproducible local outputs.

Latest local MVP result, using the synthetic holdout test set:

| Metric | Value |
| --- | ---: |
| Selected model | random_forest |
| Fraud base rate | 9.08% |
| ROC-AUC | 0.7531 |
| PR-AUC | 0.2418 |
| Precision | 0.1631 |
| Recall | 0.8440 |
| F1 | 0.2734 |

Detailed results are documented in `docs/results_2026-07-02.md`.

## Repository Layout

```text
src/fraudrisk_engine/   package source
scripts/                dataset, training, and evaluation commands
tests/                  automated tests
data/                   generated local dataset
models/                 generated local model artifact
reports/                generated local metrics and sample predictions
docs/                   technical notes
```

## External References

The implementation is original. The project direction was informed by public references such as:

- Fraud Detection Handbook: https://github.com/Fraud-Detection-Handbook/fraud-detection-handbook
- PyOD: https://github.com/yzhao062/pyod
- Amazon Fraud Dataset Benchmark: https://github.com/amazon-science/fraud-dataset-benchmark
- scikit-learn documentation: https://scikit-learn.org/

No external repository code is copied into this project.

## Roadmap

- Add a real public benchmark dataset.
- Add XGBoost as an optional model.
- Add SHAP explanations as an optional extra.
- Add anomaly detection baselines with PyOD.
- Add MLflow experiment tracking.
- Add Dockerfile and GitHub Actions.
