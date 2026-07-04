from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from fraudrisk_engine.api import create_app
from fraudrisk_engine.batch import score_batch
from fraudrisk_engine.data import FEATURE_COLUMNS


class AlwaysRiskyModel:
    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        probabilities = np.full(len(frame), 0.9)
        return np.column_stack([1 - probabilities, probabilities])


def _reference_stats() -> dict[str, dict[str, float]]:
    return {
        "amount": {"p10": 10.0, "median": 40.0, "p75": 80.0, "p90": 120.0, "p95": 180.0},
        "merchant_risk_score": {
            "p10": 0.05,
            "median": 0.20,
            "p75": 0.40,
            "p90": 0.70,
            "p95": 0.90,
        },
        "device_trust_score": {
            "p10": 0.30,
            "median": 0.70,
            "p75": 0.85,
            "p90": 0.95,
            "p95": 0.98,
        },
        "velocity_1h": {"p10": 0.0, "median": 1.0, "p75": 2.0, "p90": 4.0, "p95": 5.0},
        "distance_from_home_km": {
            "p10": 5.0,
            "median": 50.0,
            "p75": 120.0,
            "p90": 250.0,
            "p95": 400.0,
        },
        "payment_attempts": {
            "p10": 1.0,
            "median": 1.0,
            "p75": 2.0,
            "p90": 3.0,
            "p95": 4.0,
        },
        "prior_chargebacks": {
            "p10": 0.0,
            "median": 0.0,
            "p75": 0.0,
            "p90": 1.0,
            "p95": 1.0,
        },
        "account_age_days": {
            "p10": 30.0,
            "median": 300.0,
            "p75": 500.0,
            "p90": 800.0,
            "p95": 1000.0,
        },
    }


def _write_fake_model(path: Path) -> None:
    artifact = {
        "model": AlwaysRiskyModel(),
        "feature_columns": FEATURE_COLUMNS,
        "threshold": 0.30,
        "high_risk_threshold": 0.70,
        "reference_stats": _reference_stats(),
        "metadata": {"best_model": "always_risky", "seed": 1},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)


def _payload(transaction_id: str = "txn_test_001") -> dict[str, object]:
    return {
        "transaction_id": transaction_id,
        "amount": 850.0,
        "account_age_days": 12,
        "customer_tx_count_24h": 7,
        "merchant_risk_score": 0.91,
        "device_trust_score": 0.18,
        "velocity_1h": 6,
        "distance_from_home_km": 980.0,
        "hour": 2,
        "payment_attempts": 4,
        "prior_chargebacks": 1,
        "is_foreign_card": 1,
        "is_high_risk_mcc": 1,
        "is_weekend": 1,
        "is_new_device": 1,
        "customer_segment": "new",
        "channel": "ecommerce",
    }


def test_operational_review_workflow(tmp_path: Path) -> None:
    model_path = tmp_path / "model.joblib"
    database_path = tmp_path / "fraudrisk.sqlite"
    _write_fake_model(model_path)

    client = TestClient(create_app(model_path=model_path, database_path=database_path))

    score_response = client.post("/transactions/score", json=_payload())
    assert score_response.status_code == 200
    score_body = score_response.json()
    assert score_body["transaction_id"] == "txn_test_001"
    assert score_body["decision"] == "block"
    assert score_body["stored"] is True

    duplicate_response = client.post("/transactions/score", json=_payload())
    assert duplicate_response.status_code == 409

    pending_response = client.get("/reviews/pending")
    assert pending_response.status_code == 200
    pending_body = pending_response.json()
    assert pending_body["count"] == 1
    assert pending_body["items"][0]["transaction_id"] == "txn_test_001"

    review_response = client.post(
        "/reviews/txn_test_001/decision",
        json={
            "review_decision": "fraud",
            "reviewer": "analyst",
            "notes": "Confirmed in manual review.",
        },
    )
    assert review_response.status_code == 200
    reviewed_body = review_response.json()
    assert reviewed_body["review_decision"] == "fraud"
    assert reviewed_body["reviewer"] == "analyst"

    assert client.get("/reviews/pending").json()["count"] == 0


def test_score_batch_writes_scores_and_summary(tmp_path: Path) -> None:
    model_path = tmp_path / "model.joblib"
    input_path = tmp_path / "batch_input.csv"
    output_path = tmp_path / "batch_scores.csv"
    summary_path = tmp_path / "batch_summary.json"
    _write_fake_model(model_path)

    frame = pd.DataFrame([_payload("txn_batch_001"), _payload("txn_batch_002")])
    frame.to_csv(input_path, index=False)

    summary = score_batch(
        input_path=input_path,
        model_path=model_path,
        output_path=output_path,
        summary_path=summary_path,
        top_n=1,
    )

    scored = pd.read_csv(output_path)
    assert summary_path.exists()
    assert summary["rows"] == 2
    assert summary["decision_counts"]["block"] == 2
    assert len(summary["top_risks"]) == 1
    assert set(["transaction_id", "fraud_probability", "decision"]).issubset(scored.columns)
