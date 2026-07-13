# Completed Portfolio Scope

Version: 1.1.0

## Goal

Deliver a reproducible fraud-risk backend that demonstrates modeling, API design, operational review, auditability, monitoring, and evaluation on a real imbalanced dataset.

## Included

- Synthetic data generation with readable business features.
- Logistic regression, random forest, and optional XGBoost training.
- Validation-based review and high-risk thresholds.
- Real OpenML/Kaggle ULB credit-card fraud benchmark with temporal holdout.
- FastAPI stateless and persisted scoring.
- SQLite audit records, review queue, and manual review decisions.
- Batch CSV scoring and operational summaries.
- Reason codes and optional SHAP global explanations.
- Optional API-key authentication.
- PSI drift monitoring for input features and model scores.
- Docker execution, automated tests, linting, and GitHub Actions CI.
- End-to-end local demonstration script.

## Deliberately Outside Scope

- Real-time payment-network integration.
- Automatic movement, blocking, or reversal of funds.
- Centralized identity, RBAC, and secret rotation.
- Managed PostgreSQL, distributed queues, and high-availability deployment.
- Production telemetry, paging, incident response, and retraining orchestration.
- Compliance, privacy, and fairness certification.

These items require organizational infrastructure and governance. Their absence does not prevent the repository from meeting its intended portfolio scope.

## Reuse Policy

The implementation is original. External repositories and public datasets are cited as references; external source code was not copied into the project.
