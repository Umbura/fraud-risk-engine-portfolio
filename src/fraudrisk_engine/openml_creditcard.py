"""OpenML credit-card fraud benchmark utilities."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

OPENML_CREDITCARD_DATA_ID = 42175
OPENML_CREDITCARD_URL = "https://www.openml.org/d/42175"
KAGGLE_CREDITCARD_URL = "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"

TARGET_COLUMN = "Class"
FEATURE_COLUMNS = ["Time", *[f"V{idx}" for idx in range(1, 29)], "Amount"]
REQUIRED_COLUMNS = [*FEATURE_COLUMNS, TARGET_COLUMN]


def fetch_openml_creditcard(
    output_path: str | Path,
    data_id: int = OPENML_CREDITCARD_DATA_ID,
) -> Path:
    """Download the OpenML credit-card fraud dataset and save a normalized CSV."""
    from sklearn.datasets import fetch_openml

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bunch = fetch_openml(data_id=data_id, as_frame=True)
    if bunch.frame is not None:
        frame = bunch.frame.copy()
    else:
        frame = pd.DataFrame(bunch.data).copy()
        if TARGET_COLUMN not in frame.columns and bunch.target is not None:
            frame[TARGET_COLUMN] = bunch.target

    normalized = normalize_creditcard_frame(frame)
    normalized.to_csv(output_path, index=False)
    return output_path


def normalize_creditcard_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize OpenML/Kaggle-like credit-card fraud columns and target."""
    rename_map: dict[str, str] = {}
    lower_to_original = {str(column).lower(): column for column in frame.columns}
    for expected in REQUIRED_COLUMNS:
        if expected in frame.columns:
            continue
        original = lower_to_original.get(expected.lower())
        if original is not None:
            rename_map[original] = expected

    normalized = frame.rename(columns=rename_map).copy()
    missing = sorted(set(REQUIRED_COLUMNS) - set(normalized.columns))
    if missing:
        raise ValueError(f"Credit-card fraud dataset is missing columns: {missing}")

    normalized = normalized[REQUIRED_COLUMNS].copy()
    for column in FEATURE_COLUMNS:
        normalized[column] = pd.to_numeric(normalized[column], errors="raise")
    normalized[TARGET_COLUMN] = pd.to_numeric(
        normalized[TARGET_COLUMN],
        errors="raise",
    ).astype(int)

    invalid_targets = sorted(set(normalized[TARGET_COLUMN].unique()) - {0, 1})
    if invalid_targets:
        raise ValueError(f"Target column must contain only 0/1 values. Found: {invalid_targets}")

    return normalized.dropna().reset_index(drop=True)


def run_openml_creditcard_benchmark(
    dataset_path: str | Path,
    output_path: str | Path,
    seed: int = 42,
    n_estimators: int = 120,
    review_rate: float = 0.01,
    include_xgboost: bool = False,
    max_rows: int | None = None,
) -> dict[str, Any]:
    """Train baseline models and report holdout results on the real fraud dataset."""
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame = normalize_creditcard_frame(pd.read_csv(dataset_path))
    if max_rows is not None and max_rows < len(frame):
        frame = _stratified_sample(frame, max_rows=max_rows, seed=seed)

    train, validation, test, split_strategy = make_splits(frame, seed=seed)
    models = candidate_models(
        train[TARGET_COLUMN],
        seed=seed,
        n_estimators=n_estimators,
        include_xgboost=include_xgboost,
    )

    leaderboard: list[dict[str, Any]] = []
    fitted_models: dict[str, Any] = {}
    for name, model in models.items():
        model.fit(train[FEATURE_COLUMNS], train[TARGET_COLUMN])
        fitted_models[name] = model

        probabilities = model.predict_proba(validation[FEATURE_COLUMNS])[:, 1]
        threshold = threshold_for_review_rate(probabilities, review_rate=review_rate)
        metrics = evaluate_threshold(validation[TARGET_COLUMN], probabilities, threshold)
        leaderboard.append(
            {
                "model": name,
                "validation_roc_auc": _safe_roc_auc(validation[TARGET_COLUMN], probabilities),
                "validation_pr_auc": float(
                    average_precision_score(validation[TARGET_COLUMN], probabilities)
                ),
                "validation_precision_at_budget": metrics["precision"],
                "validation_recall_at_budget": metrics["recall"],
                "validation_review_rate": metrics["review_rate"],
                "threshold": float(threshold),
            }
        )

    leaderboard = sorted(
        leaderboard,
        key=lambda row: (
            -row["validation_recall_at_budget"],
            -row["validation_precision_at_budget"],
            -row["validation_pr_auc"],
        ),
    )
    selected = leaderboard[0]
    selected_model = fitted_models[selected["model"]]
    test_probabilities = selected_model.predict_proba(test[FEATURE_COLUMNS])[:, 1]
    test_metrics = evaluate_threshold(test[TARGET_COLUMN], test_probabilities, selected["threshold"])

    report = {
        "source": {
            "openml_data_id": OPENML_CREDITCARD_DATA_ID,
            "openml_url": OPENML_CREDITCARD_URL,
            "kaggle_reference_url": KAGGLE_CREDITCARD_URL,
            "warning": (
                "This is a real public fraud dataset, but V1-V28 are anonymized PCA features. "
                "It supports detection benchmarking better than business-readable reason codes."
            ),
        },
        "dataset": dataset_summary(frame),
        "split": {
            "strategy": split_strategy,
            "train": dataset_summary(train),
            "validation": dataset_summary(validation),
            "test": dataset_summary(test),
        },
        "review_budget": {
            "target_review_rate": float(review_rate),
            "selection_rule": "highest validation recall at fixed review budget",
        },
        "selected_model": selected["model"],
        "leaderboard": leaderboard,
        "test_metrics": {
            "roc_auc": _safe_roc_auc(test[TARGET_COLUMN], test_probabilities),
            "pr_auc": float(average_precision_score(test[TARGET_COLUMN], test_probabilities)),
            **test_metrics,
        },
        "feature_importance": feature_importance(selected_model),
        "sanity_checks": {
            "uses_target_as_feature": TARGET_COLUMN in FEATURE_COLUMNS,
            "real_public_dataset": True,
            "rows_after_optional_sampling": int(len(frame)),
            "optional_max_rows": max_rows,
        },
    }

    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def make_splits(
    frame: pd.DataFrame,
    seed: int = 42,
    validation_size: float = 0.15,
    test_size: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """Use a temporal split when every partition still contains fraud cases."""
    ordered = frame.sort_values("Time").reset_index(drop=True)
    train_end = int(len(ordered) * (1.0 - validation_size - test_size))
    validation_end = int(len(ordered) * (1.0 - test_size))
    train = ordered.iloc[:train_end].copy()
    validation = ordered.iloc[train_end:validation_end].copy()
    test = ordered.iloc[validation_end:].copy()
    if _all_partitions_have_two_classes(train, validation, test):
        return train, validation, test, "temporal by Time: 70% train, 15% validation, 15% test"

    train_validation, test = train_test_split(
        frame,
        test_size=test_size,
        stratify=frame[TARGET_COLUMN],
        random_state=seed,
    )
    relative_validation_size = validation_size / (1.0 - test_size)
    train, validation = train_test_split(
        train_validation,
        test_size=relative_validation_size,
        stratify=train_validation[TARGET_COLUMN],
        random_state=seed,
    )
    return (
        train.copy(),
        validation.copy(),
        test.copy(),
        "stratified fallback: temporal split did not preserve both classes",
    )


def candidate_models(
    target: pd.Series,
    seed: int = 42,
    n_estimators: int = 120,
    include_xgboost: bool = False,
) -> dict[str, Any]:
    models: dict[str, Any] = {
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1200,
                        class_weight="balanced",
                        solver="lbfgs",
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=9,
            min_samples_leaf=10,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        ),
    }

    if include_xgboost:
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise RuntimeError("XGBoost support requires: uv sync --extra boosting") from exc

        positive = int((target == 1).sum())
        negative = int((target == 0).sum())
        models["xgboost"] = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=4,
            learning_rate=0.06,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_weight=3,
            scale_pos_weight=float(negative / max(positive, 1)),
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=-1,
        )
    return models


def threshold_for_review_rate(probabilities: np.ndarray, review_rate: float) -> float:
    if not 0 < review_rate <= 1:
        raise ValueError("review_rate must be in the interval (0, 1].")
    k = max(1, math.ceil(len(probabilities) * review_rate))
    ordered = np.sort(probabilities)[::-1]
    return float(ordered[k - 1])


def evaluate_threshold(
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    reviewed = int(predictions.sum())
    total_frauds = int(y_true.sum())
    return {
        "threshold": float(threshold),
        "reviewed_transactions": reviewed,
        "review_rate": float(reviewed / len(y_true)),
        "frauds_caught": int(tp),
        "total_frauds": total_frauds,
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }


def dataset_summary(frame: pd.DataFrame) -> dict[str, Any]:
    frauds = int(frame[TARGET_COLUMN].sum())
    return {
        "rows": int(len(frame)),
        "frauds": frauds,
        "legitimate": int(len(frame) - frauds),
        "fraud_rate": float(frame[TARGET_COLUMN].mean()),
    }


def feature_importance(model: Any, limit: int = 12) -> list[dict[str, Any]]:
    estimator = model
    transformed_names = FEATURE_COLUMNS
    if isinstance(model, Pipeline):
        estimator = model.named_steps["model"]

    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = np.abs(estimator.coef_[0])
    else:
        return []

    order = np.argsort(values)[::-1][:limit]
    return [
        {
            "feature": str(transformed_names[index]),
            "importance": float(values[index]),
        }
        for index in order
    ]


def _stratified_sample(frame: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if max_rows <= 0:
        raise ValueError("max_rows must be positive.")
    fraction = max_rows / len(frame)
    sampled = (
        frame.groupby(TARGET_COLUMN, group_keys=False)
        .sample(frac=fraction, random_state=seed)
        .reset_index(drop=True)
    )
    return sampled


def _all_partitions_have_two_classes(*partitions: pd.DataFrame) -> bool:
    return all(partition[TARGET_COLUMN].nunique() == 2 for partition in partitions)


def _safe_roc_auc(y_true: pd.Series, probabilities: np.ndarray) -> float | None:
    if y_true.nunique() < 2:
        return None
    return float(roc_auc_score(y_true, probabilities))
