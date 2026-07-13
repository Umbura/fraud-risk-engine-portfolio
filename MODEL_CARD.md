# Model Card: FraudRisk Engine

## Model Purpose

FraudRisk Engine is a portfolio fraud-detection backend for transaction risk scoring and review prioritization.

The model output is intended to support operational triage. It is not intended to approve, deny, block, or reverse financial transactions without human review and additional production controls.

## Model Variants

The repository supports three model families:

- logistic regression;
- random forest;
- XGBoost, when the optional `boosting` dependency is installed.

The latest real benchmark selected XGBoost based on validation recall at a fixed review budget.

## Data

### Synthetic Dataset

The synthetic dataset is generated locally by `scripts/create_dataset.py`. It contains readable transaction features such as amount, device trust score, merchant risk score, velocity, and chargeback history.

Purpose:

- local API demonstrations;
- deterministic tests;
- readable reason-code examples.

Limitation:

- synthetic data does not represent the full complexity of real fraud behavior.

### Real Benchmark Dataset

The real benchmark uses the OpenML/Kaggle ULB credit-card fraud dataset.

- OpenML: https://www.openml.org/d/42175
- Kaggle reference: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Dataset summary:

| Field | Value |
| --- | ---: |
| Rows | 284,807 |
| Fraud cases | 492 |
| Fraud base rate | 0.173% |
| Main features | `V1` to `V28`, `Time`, `Amount` |

Limitation:

- `V1` to `V28` are anonymized PCA components. This supports detection benchmarking, but weakens business-readable explanations.

## Evaluation

The real benchmark uses a temporal split by `Time`:

| Partition | Rows | Frauds | Fraud rate |
| --- | ---: | ---: | ---: |
| Train | 199,364 | 384 | 0.193% |
| Validation | 42,721 | 56 | 0.131% |
| Test | 42,722 | 52 | 0.122% |

Model and threshold selection are performed on validation data. Final metrics are reported on the temporal holdout test set.

Latest test result:

| Metric | Value |
| --- | ---: |
| Selected model | XGBoost |
| ROC-AUC | 0.9762 |
| PR-AUC | 0.7609 |
| Reviewed transactions | 349 |
| Review rate | 0.817% |
| Frauds caught | 44 / 52 |
| Fraud transaction amount captured | 77.99% |
| Lift over holdout base rate | 103.58x |
| Precision | 0.1261 |
| Recall | 0.8462 |
| F1 | 0.2195 |
| False positives | 305 |
| False negatives | 8 |

## Intended Use

Appropriate uses:

- portfolio demonstration;
- fraud-modeling baseline;
- review-queue prioritization prototype;
- FastAPI scoring example;
- discussion artifact for junior data, backend, fraud, or risk roles.

Inappropriate uses:

- automatic financial blocking without review;
- production deployment without monitoring;
- customer-impacting decisions without governance;
- fairness or compliance claims without additional analysis.

## Risks And Limitations

- Fraud labels are rare, so accuracy is not a useful success metric.
- Precision can look low even when the review queue is materially better than random sampling.
- Fraud-value recall measures the `Amount` associated with detected fraud labels; it is not a claim about prevented financial loss.
- Real benchmark features are anonymized and do not support intuitive business explanations.
- The synthetic API model and the real benchmark model serve different purposes.
- Drift monitoring uses PSI on recent records persisted by the portfolio API; it is not a substitute for production telemetry or alerting.
- API-key authentication is a minimal portfolio control and does not provide user identity, roles, or key rotation.

## Operational Notes

Implemented operational controls:

- SQLite persistence for scored transactions;
- pending-review queue;
- manual review decision endpoint;
- batch scoring with CSV output and JSON summary;
- optional API-key protection for operational endpoints;
- feature and score drift reports with a configurable minimum sample size.

Production hardening would still require:

- centralized authentication and role-based authorization;
- external service and model monitoring;
- threshold governance;
- governed drift alerts and response procedures;
- reviewer feedback loop;
- incident handling process;
- periodic retraining;
- data privacy review.

## Reproducibility

Install dependencies:

```bash
uv sync --extra dev --extra api --extra boosting
```

Run local validation:

```bash
uv run pytest
uv run ruff check .
```

Run the real benchmark:

```bash
uv run python scripts/fetch_openml_creditcard.py
uv run python scripts/benchmark_openml_creditcard.py --include-xgboost
```
