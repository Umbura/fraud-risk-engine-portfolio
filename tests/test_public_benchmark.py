from pathlib import Path

import pandas as pd

from fraudrisk_engine.public_benchmark import run_public_creditcard_smoke_benchmark


def test_public_smoke_benchmark_runs_on_local_sample(tmp_path: Path) -> None:
    dataset = tmp_path / "public_sample.csv"
    output = tmp_path / "public_report.json"
    rows = []
    for idx in range(40):
        rows.append(
            {
                "Time": idx,
                **{f"V{col}": float(idx % (col + 1)) for col in range(1, 29)},
                "Amount": float(idx + 1),
                "Class": int(idx % 4 == 0),
            }
        )
    pd.DataFrame(rows).to_csv(dataset, index=False)

    report = run_public_creditcard_smoke_benchmark(dataset, output)

    assert output.exists()
    assert report["rows"] == 40
    assert report["results"]
