from pathlib import Path

import pytest

from fraudrisk_engine.data import write_dataset
from fraudrisk_engine.shap_report import write_shap_report
from fraudrisk_engine.training import train_and_evaluate


def test_write_shap_report_when_dependency_is_available(tmp_path: Path) -> None:
    pytest.importorskip("shap")

    dataset = tmp_path / "transactions.csv"
    model = tmp_path / "fraud_model.joblib"
    reports = tmp_path / "reports"
    shap_output = reports / "shap_summary.json"
    write_dataset(dataset, n_rows=1000, seed=23)
    train_and_evaluate(dataset, model, reports, seed=23, n_estimators=30)

    report = write_shap_report(dataset, model, shap_output, max_rows=25, seed=23)

    assert shap_output.exists()
    assert report["top_features"]
