"""Public credit-card fraud sample utilities."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PUBLIC_SAMPLE_URL = (
    "https://raw.githubusercontent.com/psundaravadivel/"
    "Credit-Card-Fraud-Detection/main/CreditcardDataset.csv"
)


def fetch_public_creditcard_sample(
    output_path: str | Path,
    url: str = PUBLIC_SAMPLE_URL,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, output_path)  # noqa: S310 - explicit public sample URL.
    df = pd.read_csv(output_path)
    required = {"Time", "Amount", "Class"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Downloaded sample is missing required columns: {missing}")
    return output_path


def run_public_creditcard_smoke_benchmark(
    dataset_path: str | Path,
    output_path: str | Path,
    seed: int = 42,
) -> dict[str, Any]:
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(dataset_path)

    feature_columns = [column for column in df.columns if column != "Class"]
    target = df["Class"].astype(int)
    if target.nunique() < 2:
        raise ValueError("Public sample must contain both classes.")

    train, test = train_test_split(
        df,
        test_size=0.30,
        stratify=target,
        random_state=seed,
    )
    models = {
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=80,
            max_depth=5,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        ),
    }

    results: list[dict[str, Any]] = []
    for name, model in models.items():
        model.fit(train[feature_columns], train["Class"])
        probabilities = model.predict_proba(test[feature_columns])[:, 1]
        results.append(
            {
                "model": name,
                "roc_auc": float(roc_auc_score(test["Class"], probabilities)),
                "pr_auc": float(average_precision_score(test["Class"], probabilities)),
            }
        )

    report = {
        "dataset_path": str(dataset_path),
        "rows": int(len(df)),
        "fraud_rate": float(target.mean()),
        "source_url": PUBLIC_SAMPLE_URL,
        "results": results,
        "warning": (
            "This is a tiny public sample used only to validate the public-data adapter. "
            "It is not a reliable benchmark for model claims."
        ),
    }
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report
