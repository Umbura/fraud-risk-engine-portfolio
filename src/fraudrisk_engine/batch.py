"""Batch scoring utilities for transaction CSV files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import joblib
import pandas as pd

from fraudrisk_engine.data import FEATURE_COLUMNS
from fraudrisk_engine.explain import reason_codes
from fraudrisk_engine.training import decision_from_probability


def score_batch(
    input_path: str | Path,
    model_path: str | Path,
    output_path: str | Path,
    summary_path: str | Path,
    top_n: int = 10,
) -> dict[str, Any]:
    input_path = Path(input_path)
    model_path = Path(model_path)
    output_path = Path(output_path)
    summary_path = Path(summary_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(input_path)
    missing = sorted(set(FEATURE_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")
    if "transaction_id" not in frame.columns:
        frame["transaction_id"] = [f"batch_{uuid4().hex}" for _ in range(len(frame))]

    artifact = joblib.load(model_path)
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]
    review_threshold = float(artifact["threshold"])
    high_risk_threshold = float(artifact["high_risk_threshold"])
    model_name = artifact["metadata"]["best_model"]
    reference_stats = artifact["reference_stats"]
    probabilities = model.predict_proba(frame[feature_columns])[:, 1]

    rows: list[dict[str, Any]] = []
    for (_, row), probability in zip(frame.iterrows(), probabilities, strict=True):
        record = row[FEATURE_COLUMNS].to_dict()
        decision = decision_from_probability(
            float(probability),
            review_threshold,
            high_risk_threshold,
        )
        reasons = reason_codes(record, reference_stats)
        rows.append(
            {
                "transaction_id": str(row["transaction_id"]),
                "fraud_probability": float(probability),
                "decision": decision,
                "model_name": model_name,
                "reason_codes": "; ".join(reason["message"] for reason in reasons[:3]),
            }
        )

    scored = pd.DataFrame(rows).sort_values(
        ["fraud_probability", "transaction_id"],
        ascending=[False, True],
    )
    scored.to_csv(output_path, index=False)

    decision_counts = scored["decision"].value_counts().to_dict()
    summary = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "rows": int(len(scored)),
        "average_fraud_probability": float(scored["fraud_probability"].mean()),
        "max_fraud_probability": float(scored["fraud_probability"].max()),
        "decision_counts": {
            "approve": int(decision_counts.get("approve", 0)),
            "review": int(decision_counts.get("review", 0)),
            "block": int(decision_counts.get("block", 0)),
        },
        "top_risks": scored.head(top_n).to_dict(orient="records"),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
