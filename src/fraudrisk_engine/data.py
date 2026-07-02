"""Synthetic transaction data generation for the portfolio MVP.

The dataset is intentionally synthetic to keep the project reproducible without
Kaggle credentials, paid APIs, or private financial data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

NUMERIC_FEATURES = [
    "amount",
    "account_age_days",
    "customer_tx_count_24h",
    "merchant_risk_score",
    "device_trust_score",
    "velocity_1h",
    "distance_from_home_km",
    "hour",
    "payment_attempts",
    "prior_chargebacks",
]

BINARY_FEATURES = [
    "is_foreign_card",
    "is_high_risk_mcc",
    "is_weekend",
    "is_new_device",
]

CATEGORICAL_FEATURES = [
    "customer_segment",
    "channel",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "is_fraud"


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-values))


def generate_transactions(n_rows: int = 6000, seed: int = 42) -> pd.DataFrame:
    """Generate a reproducible transaction dataset with realistic fraud signals."""

    rng = np.random.default_rng(seed)

    amount = np.round(rng.lognormal(mean=4.15, sigma=0.9, size=n_rows), 2)
    account_age_days = np.round(rng.gamma(shape=2.5, scale=140, size=n_rows)).astype(int)
    customer_tx_count_24h = rng.poisson(lam=2.0, size=n_rows)
    merchant_risk_score = rng.beta(a=1.3, b=4.2, size=n_rows)
    device_trust_score = rng.beta(a=5.0, b=1.6, size=n_rows)
    velocity_1h = rng.poisson(lam=1.2, size=n_rows)
    distance_from_home_km = np.round(rng.gamma(shape=1.4, scale=85, size=n_rows), 2)
    hour = rng.integers(low=0, high=24, size=n_rows)
    payment_attempts = np.clip(rng.poisson(lam=1.1, size=n_rows), 1, 8)
    prior_chargebacks = rng.binomial(n=3, p=0.035, size=n_rows)

    is_foreign_card = rng.binomial(n=1, p=0.12, size=n_rows)
    is_high_risk_mcc = rng.binomial(n=1, p=0.09, size=n_rows)
    is_weekend = rng.binomial(n=1, p=0.28, size=n_rows)
    is_new_device = rng.binomial(n=1, p=0.18, size=n_rows)

    customer_segment = rng.choice(
        ["new", "regular", "vip", "dormant"],
        size=n_rows,
        p=[0.22, 0.56, 0.12, 0.10],
    )
    channel = rng.choice(
        ["card_present", "ecommerce", "wallet", "bank_transfer"],
        size=n_rows,
        p=[0.35, 0.42, 0.15, 0.08],
    )

    night_hour = ((hour <= 5) | (hour >= 23)).astype(int)
    high_amount = (amount > np.quantile(amount, 0.88)).astype(int)
    low_device_trust = (device_trust_score < 0.45).astype(int)
    far_from_home = (distance_from_home_km > np.quantile(distance_from_home_km, 0.90)).astype(int)
    high_velocity = (velocity_1h >= 4).astype(int)

    logit = (
        -5.7
        + 0.55 * np.log1p(amount)
        + 1.10 * is_foreign_card
        + 1.35 * is_high_risk_mcc
        + 1.30 * is_new_device
        + 1.65 * merchant_risk_score
        - 1.70 * device_trust_score
        + 0.45 * high_amount
        + 0.95 * low_device_trust
        + 0.90 * far_from_home
        + 0.85 * high_velocity
        + 0.55 * night_hour
        + 0.70 * (payment_attempts >= 3).astype(int)
        + 1.20 * (prior_chargebacks > 0).astype(int)
        + 0.45 * (customer_segment == "new").astype(int)
        + 0.35 * (customer_segment == "dormant").astype(int)
        + 0.35 * (channel == "ecommerce").astype(int)
        + 0.30 * (channel == "wallet").astype(int)
    )

    fraud_probability = _sigmoid(logit)
    is_fraud = rng.binomial(n=1, p=fraud_probability)

    event_start = np.datetime64("2026-01-01T00:00:00")
    event_offsets = rng.integers(0, 60 * 60 * 24 * 120, size=n_rows)
    event_timestamp = event_start + event_offsets.astype("timedelta64[s]")

    df = pd.DataFrame(
        {
            "transaction_id": [f"txn_{idx:06d}" for idx in range(n_rows)],
            "event_timestamp": event_timestamp.astype(str),
            "amount": amount,
            "account_age_days": account_age_days,
            "customer_tx_count_24h": customer_tx_count_24h,
            "merchant_risk_score": np.round(merchant_risk_score, 4),
            "device_trust_score": np.round(device_trust_score, 4),
            "velocity_1h": velocity_1h,
            "distance_from_home_km": distance_from_home_km,
            "hour": hour,
            "payment_attempts": payment_attempts,
            "prior_chargebacks": prior_chargebacks,
            "is_foreign_card": is_foreign_card,
            "is_high_risk_mcc": is_high_risk_mcc,
            "is_weekend": is_weekend,
            "is_new_device": is_new_device,
            "customer_segment": customer_segment,
            "channel": channel,
            "is_fraud": is_fraud,
            "simulated_fraud_probability": np.round(fraud_probability, 5),
        }
    )
    return df


def write_dataset(path: str | Path, n_rows: int = 6000, seed: int = 42) -> pd.DataFrame:
    """Generate and write the dataset to CSV."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_transactions(n_rows=n_rows, seed=seed)
    df.to_csv(output_path, index=False)
    return df
