from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    metrics_path = Path("reports/metrics.json")
    if not metrics_path.exists():
        raise SystemExit("Missing reports/metrics.json. Run: uv run python scripts/train_model.py")
    report = json.loads(metrics_path.read_text(encoding="utf-8"))
    summary = {
        "best_model": report["best_model"],
        "threshold": report["threshold"],
        "high_risk_threshold": report["high_risk_threshold"],
        "test_operational_cost": report["test_operational_cost"],
        "test_metrics": report["test_metrics"],
        "sanity_checks": report["sanity_checks"],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
