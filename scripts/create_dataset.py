from __future__ import annotations

import argparse

from fraudrisk_engine.data import write_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Create synthetic fraud-risk dataset.")
    parser.add_argument("--output", default="data/transactions.csv")
    parser.add_argument("--rows", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = write_dataset(args.output, n_rows=args.rows, seed=args.seed)
    print(f"Wrote {len(df)} rows to {args.output}")
    print(f"Fraud rate: {df['is_fraud'].mean():.4f}")


if __name__ == "__main__":
    main()
