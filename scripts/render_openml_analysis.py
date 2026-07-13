from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render review-budget results from the real OpenML benchmark."
    )
    parser.add_argument("--report", default="reports/openml_creditcard_benchmark.json")
    parser.add_argument("--output", default="docs/assets/openml-budget-analysis.png")
    args = parser.parse_args()

    report_path = Path(args.report)
    output_path = Path(args.output)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    curve = report["review_budget_analysis"]
    selected = report["test_metrics"]

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Rendering requires: uv sync --extra analysis") from exc

    review_rates = [row["review_rate"] * 100 for row in curve]
    case_recall = [row["recall"] * 100 for row in curve]
    amount_recall = [row["fraud_amount_recall"] * 100 for row in curve]
    lift = [row["lift_over_base_rate"] for row in curve]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#cfd5dc",
            "axes.labelcolor": "#48515b",
            "xtick.color": "#59636e",
            "ytick.color": "#59636e",
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(12.8, 7.2), dpi=100)
    figure.patch.set_facecolor("#f4f6f8")
    figure.subplots_adjust(left=0.07, right=0.96, top=0.70, bottom=0.13, wspace=0.28)

    figure.text(0.07, 0.94, "FraudRisk Engine", fontsize=20, fontweight="bold", color="#20242a")
    figure.text(
        0.07,
        0.85,
        "Review-budget sensitivity on the real temporal holdout",
        fontsize=16,
        fontweight="bold",
        color="#20242a",
    )
    figure.text(
        0.07,
        0.81,
        "Thresholds selected on validation; every point measured on 42,722 unseen transactions.",
        fontsize=10,
        color="#68727c",
    )

    callouts = [
        ("84.6%", "fraud-case recall"),
        (f"{selected['fraud_amount_recall'] * 100:.1f}%", "fraud-value recall"),
        (f"{selected['lift_over_base_rate']:.0f}x", "lift over base rate"),
        (f"{selected['review_rate'] * 100:.3f}%", "actual review rate"),
    ]
    for index, (value, label) in enumerate(callouts):
        x = 0.53 + index * 0.115
        figure.text(x, 0.94, value, fontsize=16, fontweight="bold", color="#087a53")
        figure.text(x, 0.905, label, fontsize=8, color="#68727c")

    for axis in axes:
        axis.set_facecolor("#ffffff")
        axis.grid(axis="y", color="#e5e9ed", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)

    axes[0].plot(review_rates, case_recall, marker="o", linewidth=2.3, color="#0b8f61", label="Fraud cases")
    axes[0].plot(review_rates, amount_recall, marker="s", linewidth=2.3, color="#397dcc", label="Fraud value")
    axes[0].set_title("Capture improves as review capacity grows", loc="left", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Actual holdout review rate (%)")
    axes[0].set_ylabel("Captured fraud (%)")
    axes[0].set_ylim(0, 105)
    axes[0].legend(frameon=False, loc="lower right")

    axes[1].plot(review_rates, lift, marker="o", linewidth=2.3, color="#e09a00")
    axes[1].set_title("Queue enrichment over random review", loc="left", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Actual holdout review rate (%)")
    axes[1].set_ylabel("Lift over fraud base rate (x)")
    axes[1].set_ylim(bottom=0)
    for x, y in zip(review_rates, lift, strict=True):
        axes[1].annotate(f"{y:.0f}x", (x, y), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)

    figure.text(
        0.07,
        0.055,
        "Dataset: OpenML/Kaggle ULB | Selected model: XGBoost | Temporal split: 70/15/15",
        fontsize=8,
        color="#747c86",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, facecolor=figure.get_facecolor())
    plt.close(figure)
    print(f"Wrote OpenML review-budget chart to {output_path}")


if __name__ == "__main__":
    main()
