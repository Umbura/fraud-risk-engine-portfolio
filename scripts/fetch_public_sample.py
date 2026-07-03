from __future__ import annotations

import argparse

from fraudrisk_engine.public_benchmark import fetch_public_creditcard_sample


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch a small public credit-card fraud sample.")
    parser.add_argument("--output", default="data/public_creditcard_sample.csv")
    args = parser.parse_args()

    path = fetch_public_creditcard_sample(args.output)
    print(f"Wrote public sample to {path}")


if __name__ == "__main__":
    main()
