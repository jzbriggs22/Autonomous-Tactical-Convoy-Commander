"""SQLite-backed storage for governance data.

Uses WAL mode and a per-connection transaction model. Thread-safe via RLock.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS decisions (
    event_id            TEXT PRIMARY KEY,
    agent_id            TEXT NOT NULL,
    timestamp           TEXT NOT NULL,
    case_id             TEXT NOT NULL,
    case_category       TEXT NOT NULL,
    is_high_risk        INTEGER NOT NULL DEFAULT 0,
    high_risk_score     REAL    NOT NULL DEFAULT 0.0,
    decision            TEXT NOT NULL,
    resolution_time_ms  INTEGER NOT NULL,
    ground_truth        TEXT,
    metadata_json       TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_dec_agent_ts  ON decisions (agent_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_dec_cat       ON decisions (agent_id, case_category, timestamp);
CREATE INDEX IF NOT EXISTS idx_dec_hr        ON decisions (agent_id, is_high_risk, timestamp);

CREATE TABLE IF NOT EXISTS baselines (
    agent_id        TEXT NOT NULL,
    category        TEXT NOT NULL,
    metric          TEXT NOT NULL,
    computed_at     TEXT NOT NULL,
    value           REAL NOT NULL,
    sample_count    INTEGER NOT NULL,
    PRIMARY KEY (agent_id, category, metric)
);

CREATE TABLE IF NOT EXISTS alerts (
    alert_id        TEXT PRIMARY KEY,
    agent_id        TEXT NOT NULL,
    timestamp       TEXT NOT NULL,
    rule_name       TEXT NOT NULL,
    severity        TEXT NOT NULL,
    message         TEXT NOT NULL,
    metrics_json    TEXT NOT NULL DEFAULT '{}',
    acknowledged    INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_alert_agent_ts ON alerts (agent_id, timestamp);

CREATE TABLE IF NOT EXISTS rollbacks (
    rollback_id     TEXT PRIMARY KEY,
    agent_id        TEXT NOT NULL,
    timestamp       TEXT NOT NULL,
    trigger_rule    TEXT NOT NULL,
    reason          TEXT NOT NULL,
    metrics_json    TEXT NOT NULL DEFAULT '{}',
    resolved        INTEGER NOT NULL DEFAULT 0,
    resolved_at     TEXT,
    resolved_by     TEXT
);
CREATE INDEX IF NOT EXISTS idx_rb_agent_ts ON rollbacks (agent_id, timestamp);
"""


@dataclass
class DecisionRecord:
    event_id: str
    agent_id: str
    timestamp: datetime
    case_id: str
    case_category: str
    is_high_risk: bool
    high_risk_score: float
    decision: str
    resolution_time_ms: int
    ground_truth: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class AlertRecord:
    alert_id: str
    agent_id: str
    timestamp: datetime
    rule_name: str
    severity: str
    message: str
    metrics: dict = field(default_factory=dict)
    acknowledged: bool = False


@dataclass
class RollbackRecord:
    rollback_id: str
    agent_id: str
    timestamp: datetime
    trigger_rule: str
    reason: str
    metrics: dict = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None


class GovernanceDB:
    """Thread-safe SQLite store for all governance data."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self._db_path = str(db_path)
        self._lock = threading.RLock()
        self._conn: Optional[sqlite3.Connection] = None
        self._init()

    def _init(self) -> None:
        with self._lock:
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            try:
                yield self._conn
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    # ── decisions ────────────────────────────────────────────────────────────

    def insert_decision(self, rec: DecisionRecord) -> None:
        with self._tx() as c:
            c.execute(
                """
                INSERT OR REPLACE INTO decisions
                  (event_id, agent_id, timestamp, case_id, case_category,
                   is_high_risk, high_risk_score, decision, resolution_time_ms,
                   ground_truth, metadata_json)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    rec.event_id, rec.agent_id, rec.timestamp.isoformat(),
                    rec.case_id, rec.case_category, int(rec.is_high_risk),
                    rec.high_risk_score, rec.decision, rec.resolution_time_ms,
                    rec.ground_truth, json.dumps(rec.metadata),
                ),
            )

    def set_ground_truth(self, event_id: str, agent_id: str, ground_truth: str) -> bool:
        with self._tx() as c:
            cur = c.execute(
                "UPDATE decisions SET ground_truth=? WHERE event_id=? AND agent_id=?",
                (ground_truth, event_id, agent_id),
            )
            return cur.rowcount > 0

    def get_recent_decisions(
        self,
        agent_id: str,
        *,
        category: Optional[str] = None,
        limit: int = 500,
        high_risk_only: bool = False,
        oldest_first: bool = False,
    ) -> list[DecisionRecord]:
        q = "SELECT * FROM decisions WHERE agent_id=?"
        params: list = [agent_id]
        if category and category != "*":
            q += " AND case_category=?"
            params.append(category)
        if high_risk_only:
            q += " AND is_high_risk=1"
        order = "ASC" if oldest_first else "DESC"
        q += f" ORDER BY timestamp {order} LIMIT ?"
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(q, params).fetchall()
        return [self._to_decision(r) for r in rows]

    def count_decisions(self, agent_id: str, *, category: Optional[str] = None) -> int:
        q = "SELECT COUNT(*) FROM decisions WHERE agent_id=?"
        params: list = [agent_id]
        if category and category != "*":
            q += " AND case_category=?"
            params.append(category)
        with self._lock:
            return self._conn.execute(q, params).fetchone()[0]

    def _to_decision(self, row: sqlite3.Row) -> DecisionRecord:
        return DecisionRecord(
            event_id=row["event_id"],
            agent_id=row["agent_id"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            case_id=row["case_id"],
            case_category=row["case_category"],
            is_high_risk=bool(row["is_high_risk"]),
            high_risk_score=row["high_risk_score"],
            decision=row["decision"],
            resolution_time_ms=row["resolution_time_ms"],
            ground_truth=row["ground_truth"],
            metadata=json.loads(row["metadata_json"]),
        )

    # ── baselines ────────────────────────────────────────────────────────────

    def upsert_baseline(
        self, agent_id: str, category: str, metric: str,
        value: float, sample_count: int,
    ) -> None:
        with self._tx() as c:
            c.execute(
                """
                INSERT INTO baselines (agent_id, category, metric, computed_at, value, sample_count)
                VALUES (?,?,?,?,?,?)
                ON CONFLICT (agent_id, category, metric) DO UPDATE SET
                  computed_at=excluded.computed_at,
                  value=excluded.value,
                  sample_count=excluded.sample_count
                """,
                (
                    agent_id, category, metric,
                    datetime.now(timezone.utc).isoformat(),
                    value, sample_count,
                ),
            )

    def get_baseline(
        self, agent_id: str, category: str, metric: str
    ) -> Optional[tuple[float, int]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT value, sample_count FROM baselines "
                "WHERE agent_id=? AND category=? AND metric=?",
                (agent_id, category, metric),
            ).fetchone()
        return (row["value"], row["sample_count"]) if row else None

    # ── alerts ───────────────────────────────────────────────────────────────

    def insert_alert(self, rec: AlertRecord) -> None:
        with self._tx() as c:
            c.execute(
                """
                INSERT OR REPLACE INTO alerts
                  (alert_id, agent_id, timestamp, rule_name, severity, message, metrics_json)
                VALUES (?,?,?,?,?,?,?)
                """,
                (
                    rec.alert_id, rec.agent_id, rec.timestamp.isoformat(),
                    rec.rule_name, rec.severity, rec.message,
                    json.dumps(rec.metrics),
                ),
            )

    def get_recent_alerts(self, agent_id: str, limit: int = 50) -> list[AlertRecord]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM alerts WHERE agent_id=? ORDER BY timestamp DESC LIMIT ?",
                (agent_id, limit),
            ).fetchall()
        return [
            AlertRecord(
                alert_id=r["alert_id"],
                agent_id=r["agent_id"],
                timestamp=datetime.fromisoformat(r["timestamp"]),
                rule_name=r["rule_name"],
                severity=r["severity"],
                message=r["message"],
                metrics=json.loads(r["metrics_json"]),
                acknowledged=bool(r["acknowledged"]),
            )
            for r in rows
        ]

    def get_last_alert_time(self, agent_id: str, rule_name: str) -> Optional[datetime]:
        with self._lock:
            row = self._conn.execute(
                "SELECT timestamp FROM alerts "
                "WHERE agent_id=? AND rule_name=? ORDER BY timestamp DESC LIMIT 1",
                (agent_id, rule_name),
            ).fetchone()
        return datetime.fromisoformat(row["timestamp"]) if row else None

    # ── rollbacks ────────────────────────────────────────────────────────────

    def insert_rollback(self, rec: RollbackRecord) -> None:
        with self._tx() as c:
            c.execute(
                """
                INSERT OR IGNORE INTO rollbacks
                  (rollback_id, agent_id, timestamp, trigger_rule, reason, metrics_json)
                VALUES (?,?,?,?,?,?)
                """,
                (
                    rec.rollback_id, rec.agent_id, rec.timestamp.isoformat(),
                    rec.trigger_rule, rec.reason, json.dumps(rec.metrics),
                ),
            )

    def get_rollbacks(self, agent_id: str, limit: int = 10) -> list[RollbackRecord]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM rollbacks WHERE agent_id=? ORDER BY timestamp DESC LIMIT ?",
                (agent_id, limit),
            ).fetchall()
        return [
            RollbackRecord(
                rollback_id=r["rollback_id"],
                agent_id=r["agent_id"],
                timestamp=datetime.fromisoformat(r["timestamp"]),
                trigger_rule=r["trigger_rule"],
                reason=r["reason"],
                metrics=json.loads(r["metrics_json"]),
                resolved=bool(r["resolved"]),
                resolved_at=(
                    datetime.fromisoformat(r["resolved_at"]) if r["resolved_at"] else None
                ),
                resolved_by=r["resolved_by"],
            )
            for r in rows
        ]

    def has_active_rollback(self, agent_id: str) -> bool:
        with self._lock:
            count = self._conn.execute(
                "SELECT COUNT(*) FROM rollbacks WHERE agent_id=? AND resolved=0",
                (agent_id,),
            ).fetchone()[0]
        return count > 0

    def resolve_rollback(
        self, agent_id: str, rollback_id: str, resolved_by: str
    ) -> bool:
        with self._tx() as c:
            cur = c.execute(
                """UPDATE rollbacks SET resolved=1, resolved_at=?, resolved_by=?
                   WHERE rollback_id=? AND agent_id=? AND resolved=0""",
                (
                    datetime.now(timezone.utc).isoformat(),
                    resolved_by,
                    rollback_id,
                    agent_id,
                ),
            )
            return cur.rowcount > 0

    def resolve_all_rollbacks(self, agent_id: str, resolved_by: str) -> int:
        with self._tx() as c:
            cur = c.execute(
                """UPDATE rollbacks SET resolved=1, resolved_at=?, resolved_by=?
                   WHERE agent_id=? AND resolved=0""",
                (datetime.now(timezone.utc).isoformat(), resolved_by, agent_id),
            )
            return cur.rowcount

    def acknowledge_alert(self, agent_id: str, alert_id: str) -> bool:
        with self._tx() as c:
            cur = c.execute(
                "UPDATE alerts SET acknowledged=1 WHERE alert_id=? AND agent_id=?",
                (alert_id, agent_id),
            )
            return cur.rowcount > 0

    def close(self) -> None:
        with self._lock:
            if self._conn:
                self._conn.close()
                self._conn = None
