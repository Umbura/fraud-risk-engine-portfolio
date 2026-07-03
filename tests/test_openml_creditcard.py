from pathlib import Path

import pandas as pd

from fraudrisk_engine.openml_creditcard import (
    normalize_creditcard_frame,
    run_openml_creditcard_benchmark,
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
