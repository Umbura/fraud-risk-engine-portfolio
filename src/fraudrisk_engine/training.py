"""Model training, threshold selection, and evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
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
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fraudrisk_engine.data import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)
from fraudrisk_engine.explain import build_reference_stats, reason_codes
from fraudrisk_engine.monitoring import build_monitoring_reference


@dataclass(frozen=True)
class ThresholdConfig:
    review_cost: float = 5.0
    missed_fraud_multiplier: float = 1.0
    minimum_recall: float = 0.65


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("binary", "passthrough", BINARY_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def candidate_models(
    seed: int = 42,
    n_estimators: int = 220,
    include_xgboost: bool = False,
) -> dict[str, Pipeline]:
    preprocessor = build_preprocessor()
    models = {
        "logistic_regression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1500,
                        class_weight="balanced",
                        solver="lbfgs",
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=9,
                        min_samples_leaf=12,
                        class_weight="balanced_subsample",
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }
    if include_xgboost:
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise RuntimeError(
                "XGBoost support requires: uv sync --extra boosting"
            ) from exc

        models["xgboost"] = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=n_estimators,
                        max_depth=4,
                        learning_rate=0.06,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        min_child_weight=3,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    return models


def _probabilities(model: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(frame[FEATURE_COLUMNS])[:, 1]


def classification_metrics(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "threshold": float(threshold),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }


def operational_cost(
    frame: pd.DataFrame,
    y_true: pd.Series,
    predictions: np.ndarray,
    config: ThresholdConfig,
) -> float:
    review_cost = float(predictions.sum()) * config.review_cost
    missed_fraud_mask = (y_true.to_numpy() == 1) & (predictions == 0)
    missed_fraud_loss = (
        frame.loc[missed_fraud_mask, "amount"].sum() * config.missed_fraud_multiplier
    )
    return float(review_cost + missed_fraud_loss)


def select_threshold(
    validation_frame: pd.DataFrame,
    y_true: pd.Series,
    probabilities: np.ndarray,
    config: ThresholdConfig,
) -> dict[str, float]:
    best: dict[str, float] | None = None

    for threshold in np.linspace(0.05, 0.95, 181):
        predictions = (probabilities >= threshold).astype(int)
        recall = recall_score(y_true, predictions, zero_division=0)
        cost = operational_cost(validation_frame, y_true, predictions, config)
        precision = precision_score(y_true, predictions, zero_division=0)
        candidate = {
            "threshold": float(threshold),
            "cost": float(cost),
            "recall": float(recall),
            "precision": float(precision),
        }
        if recall < config.minimum_recall:
            continue
        if best is None or candidate["cost"] < best["cost"]:
            best = candidate

    if best is not None:
        return best

    for threshold in np.linspace(0.05, 0.95, 181):
        predictions = (probabilities >= threshold).astype(int)
        cost = operational_cost(validation_frame, y_true, predictions, config)
        candidate = {
            "threshold": float(threshold),
            "cost": float(cost),
            "recall": float(recall_score(y_true, predictions, zero_division=0)),
            "precision": float(precision_score(y_true, predictions, zero_division=0)),
        }
        if best is None or candidate["cost"] < best["cost"]:
            best = candidate

    if best is None:
        raise RuntimeError("Unable to select threshold.")
    return best


def _feature_importance(model: Pipeline) -> list[dict[str, Any]]:
    classifier = model.named_steps["model"]
    preprocessor = model.named_steps["preprocessor"]
    try:
        names = preprocessor.get_feature_names_out()
    except Exception:
        names = np.array(FEATURE_COLUMNS)

    if hasattr(classifier, "feature_importances_"):
        values = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        values = np.abs(classifier.coef_[0])
    else:
        return []

    order = np.argsort(values)[::-1][:15]
    return [
        {"feature": str(names[idx]), "importance": float(values[idx])}
        for idx in order
    ]


def train_and_evaluate(
    dataset_path: str | Path,
    model_path: str | Path,
    reports_dir: str | Path,
    seed: int = 42,
    n_estimators: int = 220,
    include_xgboost: bool = False,
    threshold_config: ThresholdConfig | None = None,
) -> dict[str, Any]:
    threshold_config = threshold_config or ThresholdConfig()
    dataset_path = Path(dataset_path)
    model_path = Path(model_path)
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(dataset_path)
    missing = sorted(set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    train_val, test = train_test_split(
        df,
        test_size=0.20,
        stratify=df[TARGET_COLUMN],
        random_state=seed,
    )
    train, validation = train_test_split(
        train_val,
        test_size=0.25,
        stratify=train_val[TARGET_COLUMN],
        random_state=seed,
    )

    leaderboard: list[dict[str, Any]] = []
    trained_models = candidate_models(
        seed=seed,
        n_estimators=n_estimators,
        include_xgboost=include_xgboost,
    )

    for name, model in trained_models.items():
        model.fit(train[FEATURE_COLUMNS], train[TARGET_COLUMN])
        probabilities = _probabilities(model, validation)
        threshold_info = select_threshold(
            validation,
            validation[TARGET_COLUMN],
            probabilities,
            threshold_config,
        )
        metrics = classification_metrics(
            validation[TARGET_COLUMN],
            probabilities,
            threshold_info["threshold"],
        )
        leaderboard.append(
            {
                "model": name,
                "validation_pr_auc": metrics["pr_auc"],
                "validation_roc_auc": metrics["roc_auc"],
                "validation_recall": metrics["recall"],
                "validation_precision": metrics["precision"],
                "validation_cost": threshold_info["cost"],
                "threshold": threshold_info["threshold"],
            }
        )

    leaderboard = sorted(
        leaderboard,
        key=lambda item: (item["validation_cost"], -item["validation_pr_auc"]),
    )
    best_name = leaderboard[0]["model"]
    best_model = trained_models[best_name]
    best_threshold = float(leaderboard[0]["threshold"])
    high_risk_threshold = float(min(0.95, max(best_threshold + 0.25, best_threshold * 1.75)))

    validation_probabilities = _probabilities(best_model, validation)
    test_probabilities = _probabilities(best_model, test)
    test_metrics = classification_metrics(test[TARGET_COLUMN], test_probabilities, best_threshold)
    test_predictions = (test_probabilities >= best_threshold).astype(int)
    test_cost = operational_cost(test, test[TARGET_COLUMN], test_predictions, threshold_config)

    artifact = {
        "model": best_model,
        "feature_columns": FEATURE_COLUMNS,
        "threshold": best_threshold,
        "high_risk_threshold": high_risk_threshold,
        "reference_stats": build_reference_stats(train),
        "monitoring_reference": build_monitoring_reference(
            validation,
            validation_probabilities,
        ),
        "metadata": {
            "artifact_version": "1.0",
            "best_model": best_name,
            "seed": seed,
            "train_rows": int(len(train)),
            "validation_rows": int(len(validation)),
            "test_rows": int(len(test)),
            "fraud_rate": float(df[TARGET_COLUMN].mean()),
            "threshold_config": threshold_config.__dict__,
            "include_xgboost": include_xgboost,
        },
    }
    joblib.dump(artifact, model_path)

    sample_records = test.head(8).copy()
    sample_probabilities = test_probabilities[: len(sample_records)]
    sample_payload = []
    for (_, row), probability in zip(sample_records.iterrows(), sample_probabilities, strict=True):
        record = row[FEATURE_COLUMNS].to_dict()
        sample_payload.append(
            {
                "transaction_id": row.get("transaction_id"),
                "actual_is_fraud": int(row[TARGET_COLUMN]),
                "fraud_probability": float(probability),
                "decision": decision_from_probability(
                    float(probability),
                    best_threshold,
                    high_risk_threshold,
                ),
                "reason_codes": reason_codes(record, artifact["reference_stats"]),
            }
        )

    report = {
        "best_model": best_name,
        "leaderboard": leaderboard,
        "threshold": best_threshold,
        "high_risk_threshold": high_risk_threshold,
        "test_operational_cost": float(test_cost),
        "test_metrics": test_metrics,
        "feature_importance": _feature_importance(best_model),
        "sanity_checks": {
            "uses_target_as_feature": TARGET_COLUMN in FEATURE_COLUMNS,
            "duplicate_transaction_ids": int(df["transaction_id"].duplicated().sum()),
            "synthetic_dataset": True,
            "holdout_split": "60% train, 20% validation, 20% test stratified by target",
        },
    }

    (reports_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (reports_dir / "sample_predictions.json").write_text(
        json.dumps(sample_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report


def decision_from_probability(
    probability: float,
    review_threshold: float,
    high_risk_threshold: float,
) -> str:
    if probability >= high_risk_threshold:
        return "block"
    if probability >= review_threshold:
        return "review"
    return "approve"
