from pathlib import Path

from fraudrisk_engine.data import write_dataset
from fraudrisk_engine.training import train_and_evaluate


def test_train_and_evaluate_creates_model_and_metrics(tmp_path: Path) -> None:
    dataset = tmp_path / "transactions.csv"
    model = tmp_path / "fraud_model.joblib"
    reports = tmp_path / "reports"
    write_dataset(dataset, n_rows=1200, seed=11)

    report = train_and_evaluate(
        dataset_path=dataset,
        model_path=model,
        reports_dir=reports,
        seed=11,
        n_estimators=40,
    )

    assert model.exists()
    assert (reports / "metrics.json").exists()
    assert report["best_model"] in {"logistic_regression", "random_forest"}
    assert 0.05 <= report["threshold"] <= 0.95
    assert report["test_metrics"]["roc_auc"] >= 0.70
    assert report["sanity_checks"]["uses_target_as_feature"] is False
