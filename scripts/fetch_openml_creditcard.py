from __future__ import annotations

import argparse

from fraudrisk_engine.openml_creditcard import fetch_openml_creditcard


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch the OpenML credit-card fraud dataset and save it as CSV."
    )
    parser.add_argument("--output", default="data/openml_creditcard.csv")
    parser.add_argument("--data-id", type=int, default=42175)
    args = parser.parse_args()

    output_path = fetch_openml_creditcard(args.output, data_id=args.data_id)
    print(f"Saved OpenML credit-card fraud dataset to {output_path}")


if __name__ == "__main__":
    main()
