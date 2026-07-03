# OpenML Credit-Card Fraud Benchmark Results

Date: 2026-07-02

## Dataset

Source: https://www.openml.org/d/42175

Reference dataset page: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

This benchmark uses the public ULB credit-card fraud dataset through OpenML. It contains real transactions, but the main predictors are PCA-anonymized features (`V1` to `V28`). That makes it suitable for detection metrics, but less suitable for business-readable reason codes.

## Split

The benchmark used a temporal split by `Time`:

| Partition | Rows | Frauds | Fraud rate |
| --- | ---: | ---: | ---: |
| Train | 199,364 | 384 | 0.193% |
| Validation | 42,721 | 56 | 0.131% |
| Test | 42,722 | 52 | 0.122% |
| Full dataset | 284,807 | 492 | 0.173% |

Model selection and threshold selection were performed on validation only. The final numbers below are from the temporal holdout test set.

## Model Selection

The review budget was set to approximately 1% of validation transactions. The model selection rule was highest validation recall at that fixed review budget.

| Model | Validation ROC-AUC | Validation PR-AUC | Precision at review budget | Recall at review budget |
| --- | ---: | ---: | ---: | ---: |
| XGBoost | 0.9838 | 0.8473 | 0.1215 | 0.9286 |
| Random forest | 0.9836 | 0.8685 | 0.1192 | 0.9107 |
| Logistic regression | 0.9828 | 0.8394 | 0.1192 | 0.9107 |

Selected model: `xgboost`

Selected threshold: `0.2266`

## Holdout Test Result

| Metric | Value |
| --- | ---: |
| ROC-AUC | 0.9762 |
| PR-AUC | 0.7609 |
| Reviewed transactions | 349 |
| Actual review rate | 0.817% |
| Frauds caught | 44 / 52 |
| Precision | 0.1261 |
| Recall | 0.8462 |
| F1 | 0.2195 |
| False positives | 305 |
| False negatives | 8 |

## Interpretation

At a sub-1% review rate on the temporal test set, the model caught 84.6% of fraud cases. Precision is low in absolute terms because the base fraud rate is only 0.122% in the test period, but the review queue is much richer than random selection.

This is a stronger portfolio result than the synthetic-only MVP because it demonstrates performance on a heavily imbalanced real fraud dataset. The current limitation is explainability: feature importance points to anonymized variables such as `V14`, `V10`, and `V12`, not business concepts like device risk or merchant behavior.
