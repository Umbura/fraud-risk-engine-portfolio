"""Optional SHAP reporting.

SHAP is intentionally optional because it adds extra dependencies and can be
expensive on larger datasets.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from fraudrisk_engine.data import FEATURE_COLUMNS


def write_shap_report(
    dataset_path: str | Path,
    model_path: str | Path,
    output_path: str | Path,
    max_rows: int = 150,
    seed: int = 42,
) -> dict[str, Any]:
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError("SHAP support requires: uv sync --extra explain") from exc

    dataset_path = Path(dataset_path)
    model_path = Path(model_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    artifact = joblib.load(model_path)
    pipeline = artifact["model"]
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["model"]

    df = pd.read_csv(dataset_path)
    sample = df.sample(n=min(max_rows, len(df)), random_state=seed)
    transformed = preprocessor.transform(sample[FEATURE_COLUMNS])
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(classifier, "estimators_") or classifier.__class__.__name__.startswith("XGB"):
        explainer = shap.TreeExplainer(classifier)
    elif hasattr(classifier, "coef_"):
        explainer = shap.LinearExplainer(classifier, transformed)
    else:
        explainer = shap.Explainer(classifier.predict_proba, transformed)
    shap_values = explainer.shap_values(transformed)
    if isinstance(shap_values, list):
        values = np.asarray(shap_values[-1])
    else:
        values = np.asarray(shap_values)
    if values.ndim == 3:
        values = values[:, :, -1]

    mean_abs = np.abs(values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:20]
    report = {
        "model": artifact["metadata"]["best_model"],
        "sample_rows": int(len(sample)),
        "top_features": [
            {
                "feature": str(feature_names[idx]),
                "mean_abs_shap": float(mean_abs[idx]),
            }
            for idx in order
        ],
        "note": (
            "SHAP report was generated on a sample of the local dataset. "
            "Use it as explainability evidence for the MVP, not as a final fairness audit."
        ),
    }
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report
