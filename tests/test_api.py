from pathlib import Path

import joblib

from fraudrisk_engine.api import score_transaction
from fraudrisk_engine.data import write_dataset
from fraudrisk_engine.training import train_and_evaluate


def test_score_transaction_returns_decision(tmp_path: Path) -> None:
    dataset = tmp_path / "transactions.csv"
    model = tmp_path / "fraud_model.joblib"
    reports = tmp_path / "reports"
    write_dataset(dataset, n_rows=1200, seed=19)
    train_and_evaluate(dataset, model, reports, seed=19, n_estimators=40)

    artifact = joblib.load(model)
    payload = score_transaction(
        artifact,
        {
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
        },
    )

    assert 0 <= payload.fraud_probability <= 1
    assert payload.decision in {"approve", "review", "block"}
    assert payload.reason_codes
