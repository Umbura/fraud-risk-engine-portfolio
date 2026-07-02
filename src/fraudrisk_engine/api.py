"""FastAPI scoring service for the fraud-risk MVP."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from fraudrisk_engine.explain import reason_codes
from fraudrisk_engine.schemas import (
    HealthResponse,
    TransactionScoreRequest,
    TransactionScoreResponse,
)
from fraudrisk_engine.training import decision_from_probability


class ModelStore:
    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        self._artifact: dict[str, Any] | None = None

    def load(self) -> dict[str, Any]:
        if self._artifact is None:
            if not self.model_path.exists():
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "Model artifact not found. Run: "
                        "uv run python scripts/create_dataset.py && "
                        "uv run python scripts/train_model.py"
                    ),
                )
            self._artifact = joblib.load(self.model_path)
        return self._artifact


def score_transaction(artifact: dict[str, Any], record: dict[str, Any]) -> TransactionScoreResponse:
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]
    frame = pd.DataFrame([record], columns=feature_columns)
    probability = float(model.predict_proba(frame)[:, 1][0])
    review_threshold = float(artifact["threshold"])
    high_risk_threshold = float(artifact["high_risk_threshold"])
    decision = decision_from_probability(
        probability,
        review_threshold,
        high_risk_threshold,
    )
    return TransactionScoreResponse(
        fraud_probability=probability,
        decision=decision,
        review_threshold=review_threshold,
        high_risk_threshold=high_risk_threshold,
        reason_codes=reason_codes(record, artifact["reference_stats"]),
        model_name=artifact["metadata"]["best_model"],
    )


def create_app(model_path: str | Path | None = None) -> FastAPI:
    runtime_model_path = model_path or os.getenv(
        "FRAUDRISK_MODEL_PATH",
        "models/fraud_model.joblib",
    )
    store = ModelStore(runtime_model_path)
    app = FastAPI(
        title="FraudRisk Engine",
        version="0.1.0",
        description="Fraud, risk, and anomaly scoring backend with operational reason codes.",
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        path = Path(runtime_model_path)
        return HealthResponse(
            status="ok",
            model_path=str(path),
            model_loaded=path.exists(),
        )

    @app.post("/score", response_model=TransactionScoreResponse)
    def score(payload: TransactionScoreRequest) -> TransactionScoreResponse:
        artifact = store.load()
        return score_transaction(artifact, payload.model_dump())

    return app


app = create_app()
