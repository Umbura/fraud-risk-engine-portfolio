"""SQLite storage for operational fraud-review workflows."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


class TransactionStore:
    """Small SQLite repository for scored transactions and review decisions."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._migrate()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _migrate(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS scored_transactions (
                    transaction_id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    reason_codes_json TEXT NOT NULL,
                    fraud_probability REAL NOT NULL,
                    decision TEXT NOT NULL,
                    review_threshold REAL NOT NULL,
                    high_risk_threshold REAL NOT NULL,
                    model_name TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    reviewed_at TEXT,
                    review_decision TEXT,
                    reviewer TEXT,
                    review_notes TEXT
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_scored_transactions_review_queue
                ON scored_transactions(decision, reviewed_at, fraud_probability DESC)
                """
            )

    def insert_score(
        self,
        transaction_id: str,
        payload: dict[str, Any],
        score: dict[str, Any],
        model_version: str,
    ) -> dict[str, Any]:
        now = utc_now_iso()
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO scored_transactions (
                        transaction_id,
                        payload_json,
                        reason_codes_json,
                        fraud_probability,
                        decision,
                        review_threshold,
                        high_risk_threshold,
                        model_name,
                        model_version,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        transaction_id,
                        json.dumps(payload, ensure_ascii=False, sort_keys=True),
                        json.dumps(score["reason_codes"], ensure_ascii=False, sort_keys=True),
                        score["fraud_probability"],
                        score["decision"],
                        score["review_threshold"],
                        score["high_risk_threshold"],
                        score["model_name"],
                        model_version,
                        now,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"Transaction already exists: {transaction_id}") from exc
        stored = self.get_transaction(transaction_id)
        if stored is None:
            raise RuntimeError("Inserted transaction could not be loaded.")
        return stored

    def list_pending_reviews(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM scored_transactions
                WHERE reviewed_at IS NULL
                  AND decision IN ('review', 'block')
                ORDER BY fraud_probability DESC, created_at ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def record_review_decision(
        self,
        transaction_id: str,
        review_decision: str,
        reviewer: str,
        notes: str | None = None,
    ) -> dict[str, Any] | None:
        with self._connect() as connection:
            result = connection.execute(
                """
                UPDATE scored_transactions
                SET reviewed_at = ?,
                    review_decision = ?,
                    reviewer = ?,
                    review_notes = ?
                WHERE transaction_id = ?
                """,
                (utc_now_iso(), review_decision, reviewer, notes, transaction_id),
            )
            if result.rowcount == 0:
                return None
        return self.get_transaction(transaction_id)

    def get_transaction(self, transaction_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM scored_transactions WHERE transaction_id = ?",
                (transaction_id,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def operational_summary(self) -> dict[str, Any]:
        with self._connect() as connection:
            totals = connection.execute(
                """
                SELECT
                    COUNT(*) AS total_transactions,
                    AVG(fraud_probability) AS average_fraud_probability,
                    MAX(fraud_probability) AS max_fraud_probability,
                    SUM(CASE WHEN decision = 'approve' THEN 1 ELSE 0 END) AS approved,
                    SUM(CASE WHEN decision = 'review' THEN 1 ELSE 0 END) AS review,
                    SUM(CASE WHEN decision = 'block' THEN 1 ELSE 0 END) AS blocked,
                    SUM(CASE WHEN reviewed_at IS NULL AND decision IN ('review', 'block') THEN 1 ELSE 0 END) AS pending_reviews,
                    SUM(CASE WHEN reviewed_at IS NOT NULL THEN 1 ELSE 0 END) AS completed_reviews,
                    SUM(CASE WHEN review_decision = 'fraud' THEN 1 ELSE 0 END) AS confirmed_fraud,
                    SUM(CASE WHEN review_decision = 'legitimate' THEN 1 ELSE 0 END) AS confirmed_legitimate,
                    SUM(CASE WHEN review_decision = 'needs_more_info' THEN 1 ELSE 0 END) AS needs_more_info
                FROM scored_transactions
                """
            ).fetchone()

        return {
            "total_transactions": int(totals["total_transactions"] or 0),
            "average_fraud_probability": float(totals["average_fraud_probability"] or 0.0),
            "max_fraud_probability": float(totals["max_fraud_probability"] or 0.0),
            "decision_counts": {
                "approve": int(totals["approved"] or 0),
                "review": int(totals["review"] or 0),
                "block": int(totals["blocked"] or 0),
            },
            "pending_reviews": int(totals["pending_reviews"] or 0),
            "completed_reviews": int(totals["completed_reviews"] or 0),
            "review_decision_counts": {
                "fraud": int(totals["confirmed_fraud"] or 0),
                "legitimate": int(totals["confirmed_legitimate"] or 0),
                "needs_more_info": int(totals["needs_more_info"] or 0),
            },
        }

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "transaction_id": row["transaction_id"],
            "payload": json.loads(row["payload_json"]),
            "reason_codes": json.loads(row["reason_codes_json"]),
            "fraud_probability": float(row["fraud_probability"]),
            "decision": row["decision"],
            "review_threshold": float(row["review_threshold"]),
            "high_risk_threshold": float(row["high_risk_threshold"]),
            "model_name": row["model_name"],
            "model_version": row["model_version"],
            "created_at": row["created_at"],
            "reviewed_at": row["reviewed_at"],
            "review_decision": row["review_decision"],
            "reviewer": row["reviewer"],
            "review_notes": row["review_notes"],
        }
