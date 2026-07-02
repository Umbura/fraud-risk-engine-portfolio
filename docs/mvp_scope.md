# MVP Scope

Date: 2026-07-02

## Goal

Build a partial but publishable backend for fraud, risk, and anomaly scoring.

## What Is Included

- Synthetic dataset generation.
- Lightweight supervised fraud model.
- Operational threshold selection.
- Reason-code explanations.
- FastAPI scoring endpoint.
- Tests and lint.
- Reproducible local commands with `uv`.

## What Is Not Included Yet

- Real bank or card data.
- Paid APIs.
- Heavy model downloads.
- XGBoost.
- SHAP.
- PyOD.
- MLflow.

## Reuse Policy

This project does not copy code from reference repositories. It reuses common project structure patterns and public ML concepts only:

- `pyproject.toml` and `uv` project management.
- `src/` package layout.
- scripts for dataset/training/evaluation.
- README-first portfolio documentation.
- public metric conventions for fraud detection.

External references are cited in the README and should be used for study or future extensions, not as pasted source code.
