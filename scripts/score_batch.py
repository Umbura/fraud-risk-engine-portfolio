from __future__ import annotations

import argparse
import json

from fraudrisk_engine.batch import score_batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a CSV batch of transactions.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", default="models/fraud_model.joblib")
    parser.add_argument("--output", default="reports/batch_scores.csv")
    parser.add_argument("--summary", default="reports/batch_summary.json")
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()

    summary = score_batch(
        input_path=args.input,
        model_path=args.model,
        output_path=args.output,
        summary_path=args.summary,
        top_n=args.top_n,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
