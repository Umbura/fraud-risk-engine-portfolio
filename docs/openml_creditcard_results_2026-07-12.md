# OpenML Credit-Card Fraud Benchmark Results

Date: 2026-07-12

## Dataset

Source: https://www.openml.org/d/42175

Reference dataset page: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Normalized CSV SHA-256: `ee3241bfc42ce15ec94587056623ca8720f24b7c7746c39dc7d17f1e75657168`

The benchmark uses the public ULB credit-card fraud dataset through OpenML. It contains real labeled transactions, while `V1` to `V28` are PCA-anonymized. The dataset supports detection and ranking evaluation but does not support business-readable explanations for those variables.

| Partition | Rows | Frauds | Fraud rate |
| --- | ---: | ---: | ---: |
| Train | 199,364 | 384 | 0.193% |
| Validation | 42,721 | 56 | 0.131% |
| Test | 42,722 | 52 | 0.122% |
| Full dataset | 284,807 | 492 | 0.173% |

## Evaluation Design

- Temporal split ordered by `Time`: 70% train, 15% validation, and 15% test.
- Models fitted on train only.
- Model and primary operating threshold selected on validation only.
- Final metrics measured once on the temporal test holdout.
- Sensitivity thresholds selected separately on validation for each review budget and then measured on test.

The review-budget curve therefore does not choose thresholds from the test labels or test probabilities.

## Model Selection

| Model | Validation ROC-AUC | Validation PR-AUC | Precision at budget | Recall at budget |
| --- | ---: | ---: | ---: | ---: |
| XGBoost | 0.9838 | 0.8473 | 0.1215 | 0.9286 |
| Random forest | 0.9836 | 0.8685 | 0.1192 | 0.9107 |
| Logistic regression | 0.9828 | 0.8394 | 0.1192 | 0.9107 |

Selected model: `xgboost`

Selected threshold: `0.2266`

## Primary Holdout Result

| Metric | Value |
| --- | ---: |
| ROC-AUC | 0.9762 |
| PR-AUC | 0.7609 |
| Reviewed transactions | 349 / 42,722 |
| Actual review rate | 0.817% |
| Frauds caught | 44 / 52 |
| Fraud-case recall | 84.62% |
| Fraud transaction amount captured | 4,811.24 / 6,168.88 |
| Fraud-value recall | 77.99% |
| Precision | 12.61% |
| Lift over test base rate | 103.58x |
| False positives | 305 |
| False negatives | 8 |

`Fraud-value recall` is the share of the dataset's `Amount` associated with detected fraud labels. It is not an estimate of prevented loss, recovered funds, or business savings.

## Review-Budget Sensitivity

Each threshold below was selected from validation probabilities for the stated target budget. Actual review rates differ on test because score distributions changed over time.

| Validation target | Actual test review | Reviewed | Frauds caught | Case recall | Value recall | Precision | Lift |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.10% | 0.096% | 41 | 36 / 52 | 69.23% | 57.43% | 87.80% | 721.38x |
| 0.25% | 0.232% | 99 | 39 / 52 | 75.00% | 61.54% | 39.39% | 323.65x |
| 0.50% | 0.426% | 182 | 40 / 52 | 76.92% | 61.74% | 21.98% | 180.57x |
| 1.00% | 0.817% | 349 | 44 / 52 | 84.62% | 77.99% | 12.61% | 103.58x |
| 2.00% | 1.727% | 738 | 44 / 52 | 84.62% | 77.99% | 5.96% | 48.98x |
| 5.00% | 4.309% | 1,841 | 46 / 52 | 88.46% | 78.12% | 2.50% | 20.53x |

## Interpretation

- The primary operating point reviewed fewer than 1% of test transactions while capturing 84.6% of fraud cases.
- The same point captured 78.0% of the transaction amount attached to fraud labels.
- The review queue was approximately 104 times richer in fraud than random review at the holdout base rate.
- Increasing the actual review rate from 0.817% to 1.727% doubled workload without finding an additional fraud case in this test period.
- A very narrow 0.096% queue retained 69.2% case recall and 87.8% precision, illustrating strong ranking quality but lower fraud-value coverage.

The 1% validation budget is the strongest practical balance observed in this experiment. This is a holdout result, not a guarantee that the same threshold will remain optimal under future drift.

## Reproduce

```bash
uv sync --extra dev --extra boosting --extra analysis
uv run python scripts/fetch_openml_creditcard.py
uv run python scripts/benchmark_openml_creditcard.py --include-xgboost
uv run python scripts/render_openml_analysis.py
```
