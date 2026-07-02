"""Pydantic schemas for the scoring API."""

from __future__ import annotations

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


class HealthResponse(BaseModel):
    status: str
    model_path: str
    model_loaded: bool
