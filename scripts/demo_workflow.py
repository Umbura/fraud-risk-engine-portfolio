from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd
from fastapi.testclient import TestClient

from fraudrisk_engine.api import create_app
from fraudrisk_engine.data import FEATURE_COLUMNS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the complete local fraud-review workflow.")
    parser.add_argument("--dataset", default="data/transactions.csv")
    parser.add_argument("--model", default="models/fraud_model.joblib")
    parser.add_argument("--database", default="data/fraudrisk.sqlite")
    parser.add_argument("--output", default="reports/demo_workflow.json")
    parser.add_argument("--rows", type=int, default=250)
    parser.add_argument("--api-key", default=None)
    return parser.parse_args()


def suspicious_payload() -> dict[str, object]:
    return {
        "amount": 1250.0,
        "account_age_days": 5,
        "customer_tx_count_24h": 12,
        "merchant_risk_score": 0.96,
        "device_trust_score": 0.12,
        "velocity_1h": 8,
        "distance_from_home_km": 1450.0,
        "hour": 3,
        "payment_attempts": 5,
        "prior_chargebacks": 2,
        "is_foreign_card": 1,
        "is_high_risk_mcc": 1,
        "is_weekend": 1,
        "is_new_device": 1,
        "customer_segment": "new",
        "channel": "ecommerce",
    }


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    model_path = Path(args.model)
    database_path = Path(args.database)
    output_path = Path(args.output)
    if not dataset_path.exists() or not model_path.exists():
        raise SystemExit(
            "Dataset or model missing. Run scripts/create_dataset.py and scripts/train_model.py first."
        )

    frame = pd.read_csv(dataset_path).head(max(args.rows, 1))
    payloads = json.loads(frame[FEATURE_COLUMNS].to_json(orient="records"))
    payloads.append(suspicious_payload())

    runtime_api_key = args.api_key
    if runtime_api_key is None:
        runtime_api_key = os.getenv("FRAUDRISK_API_KEY", "")
    headers = {"X-API-Key": runtime_api_key} if runtime_api_key else {}
    client = TestClient(
        create_app(
            model_path=model_path,
            database_path=database_path,
            api_key=runtime_api_key,
        )
    )

    run_id = f"{datetime.now(UTC):%Y%m%dT%H%M%S}_{uuid4().hex[:8]}"
    scored = []
    for index, payload in enumerate(payloads):
        response = client.post(
            "/transactions/score",
            json={"transaction_id": f"demo_{run_id}_{index:04d}", **payload},
            headers=headers,
        )
        response.raise_for_status()
        scored.append(response.json())

    pending_response = client.get("/reviews/pending?limit=5", headers=headers)
    pending_response.raise_for_status()
    review_candidates = [item for item in scored if item["decision"] in {"review", "block"}]
    reviewed_transaction = None
    if review_candidates:
        transaction_id = max(
            review_candidates,
            key=lambda item: item["fraud_probability"],
        )["transaction_id"]
        review_response = client.post(
            f"/reviews/{transaction_id}/decision",
            json={
                "review_decision": "needs_more_info",
                "reviewer": "portfolio_demo",
                "notes": "Manual-review workflow demonstration; no real fraud label assigned.",
            },
            headers=headers,
        )
        review_response.raise_for_status()
        reviewed_transaction = review_response.json()["transaction_id"]

    metrics_response = client.get("/metrics/summary", headers=headers)
    metrics_response.raise_for_status()
    drift_response = client.get(
        "/monitoring/drift?min_samples=200",
        headers=headers,
    )
    drift_response.raise_for_status()
    drift = drift_response.json()
    highest_risk = max(scored, key=lambda item: item["fraud_probability"])
    result = {
        "run_id": run_id,
        "scored_transactions": len(scored),
        "decision_counts": dict(Counter(item["decision"] for item in scored)),
        "highest_risk": {
            "transaction_id": highest_risk["transaction_id"],
            "fraud_probability": highest_risk["fraud_probability"],
            "decision": highest_risk["decision"],
        },
        "reviewed_transaction": reviewed_transaction,
        "pending_queue_preview_count": pending_response.json()["count"],
        "operational_summary": metrics_response.json(),
        "drift": {
            "status": drift["status"],
            "sample_size": drift["sample_size"],
            "top_features": [
                {
                    "feature": item["feature"],
                    "psi": item["psi"],
                    "status": item["status"],
                }
                for item in drift["features"][:5]
            ],
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
