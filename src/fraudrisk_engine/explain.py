"""Lightweight reason-code explanations for operational fraud review."""

from __future__ import annotations

from typing import Any

import pandas as pd


def build_reference_stats(train_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Build simple distribution references used by reason-code explanations."""

    numeric_columns = [
        "amount",
        "merchant_risk_score",
        "device_trust_score",
        "velocity_1h",
        "distance_from_home_km",
        "payment_attempts",
        "prior_chargebacks",
        "account_age_days",
    ]
    stats: dict[str, dict[str, float]] = {}
    for column in numeric_columns:
        series = train_df[column].astype(float)
        stats[column] = {
            "p10": float(series.quantile(0.10)),
            "median": float(series.quantile(0.50)),
            "p75": float(series.quantile(0.75)),
            "p90": float(series.quantile(0.90)),
            "p95": float(series.quantile(0.95)),
        }
    return stats


def reason_codes(record: dict[str, Any], reference_stats: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    """Return human-readable risk drivers for a single transaction.

    This is not a replacement for SHAP. It is a low-cost MVP explanation layer
    based on operational thresholds and training distribution references.
    """

    reasons: list[dict[str, Any]] = []

    def add(feature: str, value: Any, benchmark: str, message: str, severity: str = "medium") -> None:
        reasons.append(
            {
                "feature": feature,
                "value": value,
                "benchmark": benchmark,
                "severity": severity,
                "message": message,
            }
        )

    amount_p90 = reference_stats["amount"]["p90"]
    if float(record["amount"]) >= amount_p90:
        add(
            "amount",
            float(record["amount"]),
            f"p90={amount_p90:.2f}",
            "Transaction amount is unusually high for the training population.",
            "high",
        )

    trust_p10 = reference_stats["device_trust_score"]["p10"]
    if float(record["device_trust_score"]) <= trust_p10:
        add(
            "device_trust_score",
            float(record["device_trust_score"]),
            f"p10={trust_p10:.3f}",
            "Device trust score is in the lowest training decile.",
            "high",
        )

    velocity_p90 = reference_stats["velocity_1h"]["p90"]
    if float(record["velocity_1h"]) >= velocity_p90:
        add(
            "velocity_1h",
            int(record["velocity_1h"]),
            f"p90={velocity_p90:.1f}",
            "Recent transaction velocity is elevated.",
        )

    distance_p90 = reference_stats["distance_from_home_km"]["p90"]
    if float(record["distance_from_home_km"]) >= distance_p90:
        add(
            "distance_from_home_km",
            float(record["distance_from_home_km"]),
            f"p90={distance_p90:.2f}",
            "Transaction happened far from the customer's usual geography.",
        )

    if int(record["is_foreign_card"]) == 1:
        add("is_foreign_card", 1, "binary flag", "Foreign-card transaction.")

    if int(record["is_high_risk_mcc"]) == 1:
        add("is_high_risk_mcc", 1, "binary flag", "Merchant category is marked as high risk.")

    if int(record["is_new_device"]) == 1:
        add("is_new_device", 1, "binary flag", "Transaction originated from a new device.")

    if int(record["prior_chargebacks"]) > 0:
        add(
            "prior_chargebacks",
            int(record["prior_chargebacks"]),
            "greater than zero",
            "Customer has prior chargeback history.",
            "high",
        )

    if int(record["payment_attempts"]) >= 3:
        add(
            "payment_attempts",
            int(record["payment_attempts"]),
            ">=3",
            "Multiple payment attempts increase operational risk.",
        )

    return reasons[:5]
