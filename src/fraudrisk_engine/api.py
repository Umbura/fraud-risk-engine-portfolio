"""FastAPI scoring service for the fraud-risk MVP."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, status

from fraudrisk_engine.explain import reason_codes
from fraudrisk_engine.schemas import (
    HealthResponse,
    PendingReviewResponse,
    ReviewDecisionRequest,
    ReviewedTransactionResponse,
    StoredTransactionScoreRequest,
    StoredTransactionScoreResponse,
    TransactionScoreRequest,
    TransactionScoreResponse,
)
from fraudrisk_engine.storage import TransactionStore
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


def model_version(artifact: dict[str, Any]) -> str:
    metadata = artifact.get("metadata", {})
    best_model = metadata.get("best_model", "unknown_model")
    seed = metadata.get("seed", "unknown_seed")
    return f"{best_model}:seed={seed}"


def reviewed_transaction_response(record: dict[str, Any]) -> ReviewedTransactionResponse:
    return ReviewedTransactionResponse(
        transaction_id=record["transaction_id"],
        fraud_probability=record["fraud_probability"],
        decision=record["decision"],
        model_name=record["model_name"],
        created_at=record["created_at"],
        reviewed_at=record["reviewed_at"],
        review_decision=record["review_decision"],
        reviewer=record["reviewer"],
        review_notes=record["review_notes"],
    )


def create_app(
    model_path: str | Path | None = None,
    database_path: str | Path | None = None,
) -> FastAPI:
    runtime_model_path = model_path or os.getenv(
        "FRAUDRISK_MODEL_PATH",
        "models/fraud_model.joblib",
    )
    runtime_database_path = database_path or os.getenv(
        "FRAUDRISK_DB_PATH",
        "data/fraudrisk.sqlite",
    )
    store = ModelStore(runtime_model_path)
    transaction_store = TransactionStore(runtime_database_path)
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

    @app.post("/transactions/score", response_model=StoredTransactionScoreResponse)
    def score_and_store(payload: StoredTransactionScoreRequest) -> StoredTransactionScoreResponse:
        artifact = store.load()
        transaction_id = payload.transaction_id or f"txn_{uuid4().hex}"
        record = payload.model_dump(exclude={"transaction_id"})
        score_response = score_transaction(artifact, record)
        score_payload = score_response.model_dump()
        try:
            transaction_store.insert_score(
                transaction_id=transaction_id,
                payload=record,
                score=score_payload,
                model_version=model_version(artifact),
            )
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
        return StoredTransactionScoreResponse(
            transaction_id=transaction_id,
            **score_payload,
        )

    @app.get("/reviews/pending", response_model=PendingReviewResponse)
    def pending_reviews(
        limit: int = Query(default=50, ge=1, le=500),
    ) -> PendingReviewResponse:
        records = transaction_store.list_pending_reviews(limit=limit)
        items = [reviewed_transaction_response(record) for record in records]
        return PendingReviewResponse(items=items, count=len(items))

    @app.post("/reviews/{transaction_id}/decision", response_model=ReviewedTransactionResponse)
    def review_decision(
        transaction_id: str,
        payload: ReviewDecisionRequest,
    ) -> ReviewedTransactionResponse:
        record = transaction_store.record_review_decision(
            transaction_id=transaction_id,
            review_decision=payload.review_decision,
            reviewer=payload.reviewer,
            notes=payload.notes,
        )
        if record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Transaction not found: {transaction_id}",
            )
        return reviewed_transaction_response(record)

    return app


app = create_app()
