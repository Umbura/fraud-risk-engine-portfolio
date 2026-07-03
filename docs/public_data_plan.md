# Public Data Plan

The project now has three data modes:

1. Reproducible synthetic portfolio dataset used by the backend API.
2. Public credit-card fraud sample used only as a smoke test for public-data integration.
3. OpenML/Kaggle ULB credit-card fraud dataset used as the first real benchmark.

## Why the Public Sample Is Not the Main Metric

The current public sample has:

- 100 rows;
- 60% fraud rate;
- anonymized PCA-style columns;
- no reliable benchmark split.

It is useful to prove that the project can ingest a public fraud schema, but it is too small and distorted for serious claims.

## Next Real Benchmark Options

- OpenML credit-card fraud dataset: https://www.openml.org/d/42175
- Amazon Fraud Dataset Benchmark: https://github.com/amazon-science/fraud-dataset-benchmark
- Feedzai Bank Account Fraud dataset suite: https://github.com/feedzai/bank-account-fraud

The OpenML/Kaggle ULB credit-card dataset is the best immediate benchmark for this project because it is real fraud data, compact enough for local training, and heavily imbalanced. Its main limitation is explainability: most features are anonymized PCA components, so it is stronger for detection metrics than for business-readable reason codes.
