"""Pydantic schemas for the scoring API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TransactionScoreRequest(BaseModel):
    amount: float = Field(gt=0)
    account_age_days: int = Field(ge=0)
    customer_tx_count_24h: int = Field(ge=0)
    merchant_risk_score: float = Field(ge=0, le=1)
    device_trust_score: float = Field(ge=0, le=1)
    velocity_1h: int = Field(ge=0)
    distance_from_home_km: float = Field(ge=0)
    hour: int = Field(ge=0, le=23)
    payment_attempts: int = Field(ge=1)
    prior_chargebacks: int = Field(ge=0)
    is_foreign_card: int = Field(ge=0, le=1)
    is_high_risk_mcc: int = Field(ge=0, le=1)
    is_weekend: int = Field(ge=0, le=1)
    is_new_device: int = Field(ge=0, le=1)
    customer_segment: str
    channel: str


class StoredTransactionScoreRequest(TransactionScoreRequest):
    transaction_id: str | None = Field(default=None, min_length=1, max_length=120)


class ReasonCode(BaseModel):
    feature: str
    value: float | int | str
    benchmark: str
    severity: str
    message: str


class TransactionScoreResponse(BaseModel):
    fraud_probability: float
    decision: str
    review_threshold: float
    high_risk_threshold: float
    reason_codes: list[ReasonCode]
    model_name: str


class StoredTransactionScoreResponse(TransactionScoreResponse):
    transaction_id: str
    stored: bool = True


class ReviewDecisionRequest(BaseModel):
    review_decision: Literal["fraud", "legitimate", "needs_more_info"]
    reviewer: str = Field(default="manual_review", min_length=1, max_length=120)
    notes: str | None = Field(default=None, max_length=1000)


class ReviewedTransactionResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    decision: str
    review_threshold: float | None = None
    high_risk_threshold: float | None = None
    model_name: str
    model_version: str | None = None
    created_at: str
    reviewed_at: str | None
    review_decision: str | None
    reviewer: str | None
    review_notes: str | None
    payload: dict[str, float | int | str] | None = None
    reason_codes: list[ReasonCode] | None = None


class PendingReviewResponse(BaseModel):
    items: list[ReviewedTransactionResponse]
    count: int


class OperationalSummaryResponse(BaseModel):
    total_transactions: int
    average_fraud_probability: float
    max_fraud_probability: float
    decision_counts: dict[str, int]
    pending_reviews: int
    completed_reviews: int
    review_decision_counts: dict[str, int]


class DriftFeatureResponse(BaseModel):
    feature: str
    kind: Literal["numeric", "categorical"]
    psi: float
    status: Literal["stable", "warning", "drift"]
    missing_rate: float
    buckets: list[str]
    expected_proportions: list[float]
    observed_proportions: list[float]


class DriftReportResponse(BaseModel):
    status: Literal["stable", "warning", "drift", "insufficient_data"]
    sample_size: int
    minimum_sample_size: int
    reference_rows: int
    generated_at: str
    model_name: str
    model_version: str
    features: list[DriftFeatureResponse]


class HealthResponse(BaseModel):
    status: str
    model_path: str
    model_loaded: bool
    authentication_enabled: bool
