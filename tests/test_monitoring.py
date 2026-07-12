import numpy as np

from fraudrisk_engine.data import FEATURE_COLUMNS, generate_transactions
from fraudrisk_engine.monitoring import build_monitoring_reference, evaluate_drift


def _records(frame, probabilities: np.ndarray) -> list[dict[str, object]]:
    payloads = frame[FEATURE_COLUMNS].to_dict(orient="records")
    return [
        {"payload": payload, "fraud_probability": float(probability)}
        for payload, probability in zip(payloads, probabilities, strict=True)
    ]


def test_drift_report_distinguishes_stable_and_shifted_data() -> None:
    reference_frame = generate_transactions(n_rows=500, seed=31)
    reference_probabilities = np.linspace(0.01, 0.99, len(reference_frame))
    reference = build_monitoring_reference(reference_frame, reference_probabilities)

    stable_report = evaluate_drift(
        reference,
        _records(reference_frame, reference_probabilities),
        min_samples=50,
    )
    assert stable_report["status"] == "stable"
    assert max(item["psi"] for item in stable_report["features"]) < 0.001

    shifted = reference_frame.copy()
    shifted["amount"] = shifted["amount"] * 20
    shifted["device_trust_score"] = 0.01
    shifted["customer_segment"] = "unknown_segment"
    shifted_probabilities = np.full(len(shifted), 0.99)
    drift_report = evaluate_drift(
        reference,
        _records(shifted, shifted_probabilities),
        min_samples=50,
    )
    assert drift_report["status"] == "drift"
    drifted_features = {
        item["feature"] for item in drift_report["features"] if item["status"] == "drift"
    }
    assert {"amount", "device_trust_score", "customer_segment"}.issubset(drifted_features)


def test_drift_report_requires_a_minimum_sample() -> None:
    reference_frame = generate_transactions(n_rows=100, seed=12)
    probabilities = np.linspace(0.05, 0.95, len(reference_frame))
    reference = build_monitoring_reference(reference_frame, probabilities)

    report = evaluate_drift(
        reference,
        _records(reference_frame.head(5), probabilities[:5]),
        min_samples=50,
    )
    assert report["status"] == "insufficient_data"
    assert report["sample_size"] == 5
