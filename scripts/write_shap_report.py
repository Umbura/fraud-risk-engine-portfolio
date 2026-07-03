from __future__ import annotations

import argparse
import json

from fraudrisk_engine.shap_report import write_shap_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Write optional SHAP feature report.")
    parser.add_argument("--dataset", default="data/transactions.csv")
    parser.add_argument("--model", default="models/fraud_model.joblib")
    parser.add_argument("--output", default="reports/shap_summary.json")
    parser.add_argument("--max-rows", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    report = write_shap_report(
        dataset_path=args.dataset,
        model_path=args.model,
        output_path=args.output,
        max_rows=args.max_rows,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
