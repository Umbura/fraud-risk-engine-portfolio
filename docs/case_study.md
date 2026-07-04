# Case Study: FraudRisk Engine

## Problem

Fraud operations teams need a way to prioritize suspicious transactions without reviewing every payment manually. The objective is not to maximize accuracy, but to create a review queue that captures a high share of fraud cases under a constrained review budget.

## Solution

FraudRisk Engine is a small backend system for transaction risk scoring and review triage.

The system:

- receives transaction features through a FastAPI endpoint;
- calculates a fraud probability;
- maps the score to `approve`, `review`, or `block`;
- stores scored transactions in SQLite;
- exposes a pending-review queue;
- accepts manual review decisions;
- exposes transaction audit records;
- summarizes operational metrics;
- supports batch CSV scoring;
- documents performance on a real public fraud dataset.

## Architecture

```text
CSV/API transaction
        |
        v
Feature validation with Pydantic
        |
        v
Model scoring pipeline
        |
        +--> Stateless response: /score
        |
        +--> Persisted response: /transactions/score
                 |
                 v
              SQLite
                 |
                 +--> /reviews/pending
                 |
                 +--> /reviews/{transaction_id}/decision
                 |
                 +--> /transactions/{transaction_id}
                 |
                 +--> /metrics/summary
```

## Modeling

The local API model uses a synthetic transaction dataset with readable operational features:

- amount;
- account age;
- recent transaction velocity;
- merchant risk score;
- device trust score;
- distance from home;
- chargeback history;
- channel and customer segment.

The benchmark model uses the OpenML/Kaggle ULB credit-card fraud dataset. This dataset is real and heavily imbalanced, but its main features are anonymized PCA components.

## Real Benchmark Result

Dataset:

- 284,807 transactions;
- 492 fraud cases;
- 0.173% fraud base rate.

Evaluation:

- temporal split by `Time`;
- model and threshold selected on validation;
- final metrics reported on test.

Latest holdout result:

| Metric | Value |
| --- | ---: |
| Selected model | XGBoost |
| ROC-AUC | 0.9762 |
| PR-AUC | 0.7609 |
| Review rate | 0.817% |
| Frauds caught | 44 / 52 |
| Recall | 0.8462 |
| Precision | 0.1261 |

## Why Precision Is Low

Fraud is extremely rare in this benchmark. The test set fraud base rate is 0.122%, so a random review queue would be expected to find roughly 1 fraud per 820 reviewed transactions. The model found 44 frauds in 349 reviewed transactions.

In this context, precision is interpreted together with recall and review rate.

## Engineering Decisions

- FastAPI was used for a clear scoring contract.
- SQLite was used to keep the portfolio project simple and reproducible.
- Pydantic validates transaction payloads before scoring.
- Transaction lookup and summary endpoints make the scoring workflow auditable.
- Generated data, model artifacts, reports, and SQLite files are excluded from Git.
- CI runs linting, tests, and a lightweight training smoke run.
- The real benchmark is separated from the API model because the real dataset uses anonymized PCA features.

## Limitations

- No authentication.
- No production monitoring.
- No drift detection.
- No live feedback loop for retraining.
- SQLite is suitable for a portfolio demo, not high-volume production traffic.
- Business-readable explanations are available for the synthetic API model, not for the anonymized OpenML benchmark features.

## Next Production Steps

- Add authentication and role-based access control.
- Add PostgreSQL-backed audit tables.
- Add drift monitoring and score-distribution checks.
- Add batch review export for analysts.
- Add reviewer feedback into a retraining dataset.
- Add per-transaction SHAP explanations for supported models.
