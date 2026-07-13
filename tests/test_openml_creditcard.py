from pathlib import Path

import numpy as np
import pandas as pd

from fraudrisk_engine.openml_creditcard import (
    normalize_creditcard_frame,
    review_budget_analysis,
    run_openml_creditcard_benchmark,
    threshold_for_review_rate,
)


def test_normalize_creditcard_frame_accepts_case_variation() -> None:
    frame = pd.DataFrame(
        [
            {
                "time": 1,
                **{f"v{idx}": float(idx) for idx in range(1, 29)},
                "amount": 42.0,
                "class": "1",
            }
        ]
    )

    normalized = normalize_creditcard_frame(frame)

    assert "Class" in normalized.columns
    assert normalized["Class"].iloc[0] == 1


def test_openml_creditcard_benchmark_runs_on_local_sample(tmp_path: Path) -> None:
    dataset = tmp_path / "openml_creditcard_sample.csv"
    output = tmp_path / "openml_report.json"
    rows = []
    for idx in range(180):
        is_fraud = int(idx % 12 == 0)
        rows.append(
            {
                "Time": idx,
                **{
                    f"V{col}": float((idx % (col + 2)) + (8 if is_fraud and col in {4, 14} else 0))
                    for col in range(1, 29)
                },
                "Amount": float(350 + idx if is_fraud else 15 + idx % 70),
                "Class": is_fraud,
            }
        )
    pd.DataFrame(rows).to_csv(dataset, index=False)

    report = run_openml_creditcard_benchmark(
        dataset,
        output,
        n_estimators=20,
        review_rate=0.10,
    )

    assert output.exists()
    assert report["dataset"]["rows"] == 180
    assert report["selected_model"] in {"logistic_regression", "random_forest"}
    assert report["test_metrics"]["reviewed_transactions"] > 0
    assert report["test_metrics"]["fraud_amount_recall"] >= 0
    assert len(report["review_budget_analysis"]) == 6
    assert report["amount_analysis"]["full_dataset"]["fraud"]["transactions"] == 15
    assert len(report["source"]["dataset_sha256"]) == 64


def test_review_budget_analysis_uses_validation_thresholds() -> None:
    validation_target = pd.Series([0, 0, 0, 1])
    validation_probabilities = np.array([0.05, 0.10, 0.20, 0.80])
    test_target = pd.Series([0, 1, 0, 1])
    test_probabilities = np.array([0.02, 0.85, 0.30, 0.70])
    test_amounts = pd.Series([20.0, 300.0, 40.0, 100.0])

    analysis = review_budget_analysis(
        validation_target,
        validation_probabilities,
        test_target,
        test_probabilities,
        test_amounts,
        review_rates=(0.25,),
    )

    expected_threshold = threshold_for_review_rate(validation_probabilities, 0.25)
    assert analysis[0]["threshold"] == expected_threshold
    assert analysis[0]["frauds_caught"] == 1
    assert analysis[0]["fraud_amount_recall"] == 0.75
