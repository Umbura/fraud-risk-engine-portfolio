from __future__ import annotations

import argparse
import json
from pathlib import Path

from fraudrisk_engine.data import write_dataset
from fraudrisk_engine.training import ThresholdConfig, train_and_evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate fraud-risk model.")
    parser.add_argument("--dataset", default="data/transactions.csv")
    parser.add_argument("--model", default="models/fraud_model.joblib")
    parser.add_argument("--reports", default="reports")
    parser.add_argument("--rows", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--review-cost", type=float, default=5.0)
    parser.add_argument("--missed-fraud-multiplier", type=float, default=1.0)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        write_dataset(dataset_path, n_rows=args.rows, seed=args.seed)

    report = train_and_evaluate(
        dataset_path=dataset_path,
        model_path=args.model,
        reports_dir=args.reports,
        seed=args.seed,
        threshold_config=ThresholdConfig(
            review_cost=args.review_cost,
            missed_fraud_multiplier=args.missed_fraud_multiplier,
        ),
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
