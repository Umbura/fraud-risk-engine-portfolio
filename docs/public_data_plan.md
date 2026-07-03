# Public Data Plan

The project now has two data modes:

1. Reproducible synthetic portfolio dataset used by the backend API.
2. Public credit-card fraud sample used only as a smoke test for public-data integration.

## Why the Public Sample Is Not the Main Metric

The current public sample has:

- 100 rows;
- 60% fraud rate;
- anonymized PCA-style columns;
- no reliable benchmark split.

It is useful to prove that the project can ingest a public fraud schema, but it is too small and distorted for serious claims.

## Next Real Benchmark Options

- OpenML creditcard dataset: https://www.openml.org/d/1597
- Amazon Fraud Dataset Benchmark: https://github.com/amazon-science/fraud-dataset-benchmark
- Feedzai Bank Account Fraud dataset suite: https://github.com/feedzai/bank-account-fraud

The OpenML/Kaggle-style credit-card dataset is larger and more useful, but it should be downloaded intentionally because it is much heavier than the current MVP sample.
