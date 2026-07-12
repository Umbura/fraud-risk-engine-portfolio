"""Lightweight data-drift monitoring for scored transactions."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from fraudrisk_engine.data import BINARY_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES

EPSILON = 1e-6
PSI_WARNING_THRESHOLD = 0.10
PSI_DRIFT_THRESHOLD = 0.25


def _normalized(values: np.ndarray) -> np.ndarray:
    smoothed = np.asarray(values, dtype=float) + EPSILON
    return smoothed / smoothed.sum()


def _numeric_profile(series: pd.Series, bins: int = 5) -> dict[str, Any]:
    clean = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if clean.size == 0:
        raise ValueError("Cannot build a numeric drift profile from an empty series.")

    quantiles = np.quantile(clean, np.linspace(0, 1, bins + 1)[1:-1])
    boundaries = np.unique(quantiles).astype(float)
    histogram_edges = np.concatenate(([-np.inf], boundaries, [np.inf]))
    counts, _ = np.histogram(clean, bins=histogram_edges)
    proportions = _normalized(counts)
    return {
        "kind": "numeric",
        "boundaries": boundaries.tolist(),
        "expected_proportions": proportions.tolist(),
    }


def _categorical_profile(series: pd.Series) -> dict[str, Any]:
    clean = series.fillna("__missing__").astype(str)
    categories = sorted(clean.unique().tolist())
    counts = np.array([(clean == category).sum() for category in categories] + [0])
    return {
        "kind": "categorical",
        "categories": categories,
        "expected_proportions": _normalized(counts).tolist(),
    }


def build_monitoring_reference(
    frame: pd.DataFrame,
    fraud_probabilities: np.ndarray | pd.Series | None = None,
) -> dict[str, Any]:
    """Build the distribution profile stored alongside a trained model."""

    features: dict[str, dict[str, Any]] = {}
    for feature in NUMERIC_FEATURES:
        features[feature] = _numeric_profile(frame[feature])
    for feature in BINARY_FEATURES + CATEGORICAL_FEATURES:
        features[feature] = _categorical_profile(frame[feature])

    if fraud_probabilities is not None:
        score_series = pd.Series(np.asarray(fraud_probabilities, dtype=float))
        features["fraud_probability"] = _numeric_profile(score_series)

    return {
        "profile_version": 1,
        "reference_rows": int(len(frame)),
        "features": features,
    }


def population_stability_index(
    expected_proportions: list[float],
    observed_proportions: list[float],
) -> float:
    """Calculate PSI between two aligned discrete distributions."""

    expected = _normalized(np.asarray(expected_proportions, dtype=float))
    observed = _normalized(np.asarray(observed_proportions, dtype=float))
    if expected.shape != observed.shape:
        raise ValueError("Expected and observed distributions must have the same shape.")
    return float(np.sum((observed - expected) * np.log(observed / expected)))


def _drift_status(psi: float) -> str:
    if psi >= PSI_DRIFT_THRESHOLD:
        return "drift"
    if psi >= PSI_WARNING_THRESHOLD:
        return "warning"
    return "stable"


def _observed_numeric(series: pd.Series, profile: dict[str, Any]) -> tuple[list[str], list[float]]:
    clean = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    boundaries = np.asarray(profile["boundaries"], dtype=float)
    edges = np.concatenate(([-np.inf], boundaries, [np.inf]))
    counts, _ = np.histogram(clean, bins=edges)
    labels = []
    for index in range(len(edges) - 1):
        lower = "-inf" if np.isneginf(edges[index]) else f"{edges[index]:.4g}"
        upper = "inf" if np.isposinf(edges[index + 1]) else f"{edges[index + 1]:.4g}"
        labels.append(f"[{lower}, {upper})")
    return labels, _normalized(counts).tolist()


def _observed_categorical(
    series: pd.Series,
    profile: dict[str, Any],
) -> tuple[list[str], list[float]]:
    clean = series.fillna("__missing__").astype(str)
    categories = profile["categories"]
    known = set(categories)
    counts = [(clean == category).sum() for category in categories]
    counts.append((~clean.isin(known)).sum())
    return [*categories, "__other__"], _normalized(np.asarray(counts)).tolist()


def evaluate_drift(
    reference: dict[str, Any],
    records: list[dict[str, Any]],
    min_samples: int = 200,
) -> dict[str, Any]:
    """Compare recent persisted transactions with a model reference profile."""

    rows = []
    for record in records:
        row = dict(record.get("payload", {}))
        row["fraud_probability"] = record.get("fraud_probability")
        rows.append(row)
    observed = pd.DataFrame(rows)

    feature_results: list[dict[str, Any]] = []
    for feature, profile in reference["features"].items():
        series = (
            observed[feature]
            if feature in observed
            else pd.Series(np.nan, index=observed.index, dtype=float)
        )
        missing_rate = float(series.isna().mean()) if len(observed) else 0.0
        if profile["kind"] == "numeric":
            labels, observed_proportions = _observed_numeric(series, profile)
        else:
            labels, observed_proportions = _observed_categorical(series, profile)
        expected_proportions = profile["expected_proportions"]
        psi = population_stability_index(expected_proportions, observed_proportions)
        feature_results.append(
            {
                "feature": feature,
                "kind": profile["kind"],
                "psi": psi,
                "status": _drift_status(psi),
                "missing_rate": missing_rate,
                "buckets": labels,
                "expected_proportions": expected_proportions,
                "observed_proportions": observed_proportions,
            }
        )

    feature_results.sort(key=lambda item: item["psi"], reverse=True)
    if len(records) < min_samples:
        overall_status = "insufficient_data"
    elif any(item["status"] == "drift" for item in feature_results):
        overall_status = "drift"
    elif any(item["status"] == "warning" for item in feature_results):
        overall_status = "warning"
    else:
        overall_status = "stable"

    return {
        "status": overall_status,
        "sample_size": len(records),
        "minimum_sample_size": min_samples,
        "reference_rows": int(reference["reference_rows"]),
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "features": feature_results,
    }
