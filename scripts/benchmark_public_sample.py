from __future__ import annotations

import argparse
import json

from fraudrisk_engine.public_benchmark import run_public_creditcard_smoke_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a smoke benchmark on a public fraud sample.")
    parser.add_argument("--dataset", default="data/public_creditcard_sample.csv")
    parser.add_argument("--output", default="reports/public_sample_benchmark.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    report = run_public_creditcard_smoke_benchmark(args.dataset, args.output, seed=args.seed)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
