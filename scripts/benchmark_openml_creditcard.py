from __future__ import annotations

import argparse
import json

from fraudrisk_engine.openml_creditcard import run_openml_creditcard_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a real-world benchmark on the OpenML credit-card fraud dataset."
    )
    parser.add_argument("--dataset", default="data/openml_creditcard.csv")
    parser.add_argument("--output", default="reports/openml_creditcard_benchmark.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=120)
    parser.add_argument("--review-rate", type=float, default=0.01)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--include-xgboost", action="store_true")
    args = parser.parse_args()

    report = run_openml_creditcard_benchmark(
        args.dataset,
        args.output,
        seed=args.seed,
        n_estimators=args.n_estimators,
        review_rate=args.review_rate,
        include_xgboost=args.include_xgboost,
        max_rows=args.max_rows,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
