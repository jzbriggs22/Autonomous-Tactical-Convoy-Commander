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

from .migrations import apply_migrations


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
    config_version: str = ""


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
            apply_migrations(self._conn)

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
                   ground_truth, metadata_json, config_version)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    rec.event_id, rec.agent_id, rec.timestamp.isoformat(),
                    rec.case_id, rec.case_category, int(rec.is_high_risk),
                    rec.high_risk_score, rec.decision, rec.resolution_time_ms,
                    rec.ground_truth, json.dumps(rec.metadata), rec.config_version,
                ),
            )

    def get_decision_by_id(self, event_id: str, agent_id: str) -> Optional[DecisionRecord]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM decisions WHERE event_id=? AND agent_id=?",
                (event_id, agent_id),
            ).fetchone()
        return self._to_decision(row) if row else None

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

    def count_high_risk_decisions(self, agent_id: str) -> int:
        with self._lock:
            return self._conn.execute(
                "SELECT COUNT(*) FROM decisions WHERE agent_id=? AND is_high_risk=1",
                (agent_id,),
            ).fetchone()[0]

    @staticmethod
    def _parse_ts(value: str) -> datetime:
        """Parse a stored ISO timestamp, treating legacy naive values as UTC.

        Consumers subtract these from aware datetimes; a naive result would
        raise TypeError deep inside dashboard/drift windowing.
        """
        ts = datetime.fromisoformat(value)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts

    def _to_decision(self, row: sqlite3.Row) -> DecisionRecord:
        return DecisionRecord(
            event_id=row["event_id"],
            agent_id=row["agent_id"],
            timestamp=self._parse_ts(row["timestamp"]),
            case_id=row["case_id"],
            case_category=row["case_category"],
            is_high_risk=bool(row["is_high_risk"]),
            high_risk_score=row["high_risk_score"],
            decision=row["decision"],
            resolution_time_ms=row["resolution_time_ms"],
            ground_truth=row["ground_truth"],
            metadata=json.loads(row["metadata_json"]),
            config_version=row["config_version"] if "config_version" in row.keys() else "",
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
                timestamp=self._parse_ts(r["timestamp"]),
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
        return self._parse_ts(row["timestamp"]) if row else None

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
                timestamp=self._parse_ts(r["timestamp"]),
                trigger_rule=r["trigger_rule"],
                reason=r["reason"],
                metrics=json.loads(r["metrics_json"]),
                resolved=bool(r["resolved"]),
                resolved_at=(
                    self._parse_ts(r["resolved_at"]) if r["resolved_at"] else None
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

    def get_unacknowledged_alerts_older_than(
        self, agent_id: str, cutoff: datetime
    ) -> list[AlertRecord]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM alerts WHERE agent_id=? AND acknowledged=0 AND timestamp < ? "
                "ORDER BY timestamp ASC",
                (agent_id, cutoff.isoformat()),
            ).fetchall()
        return [
            AlertRecord(
                alert_id=r["alert_id"],
                agent_id=r["agent_id"],
                timestamp=self._parse_ts(r["timestamp"]),
                rule_name=r["rule_name"],
                severity=r["severity"],
                message=r["message"],
                metrics=json.loads(r["metrics_json"]),
                acknowledged=False,
            )
            for r in rows
        ]

    # ── metric snapshots (history tracking) ────────────────────────────────

    def insert_metric_snapshot(
        self, agent_id: str, category: str, metric: str,
        value: float, sample_count: int,
    ) -> None:
        with self._tx() as c:
            c.execute(
                """INSERT INTO metric_snapshots
                   (agent_id, timestamp, category, metric, value, sample_count)
                   VALUES (?,?,?,?,?,?)""",
                (
                    agent_id, datetime.now(timezone.utc).isoformat(),
                    category, metric, value, sample_count,
                ),
            )

    def get_metric_history(
        self, agent_id: str, category: str, metric: str, limit: int = 100
    ) -> list[tuple[datetime, float, int]]:
        """Returns (timestamp, value, sample_count) tuples, newest first."""
        with self._lock:
            rows = self._conn.execute(
                """SELECT timestamp, value, sample_count FROM metric_snapshots
                   WHERE agent_id=? AND category=? AND metric=?
                   ORDER BY timestamp DESC LIMIT ?""",
                (agent_id, category, metric, limit),
            ).fetchall()
        return [
            (self._parse_ts(r["timestamp"]), r["value"], r["sample_count"])
            for r in rows
        ]

    def close(self) -> None:
        with self._lock:
            if self._conn:
                self._conn.close()
                self._conn = None

    # ── data retention ──────────────────────────────────────────────────────

    def purge_old_decisions(self, agent_id: str, keep_days: int = 90) -> int:
        """Delete decisions older than keep_days. Returns count deleted."""
        cutoff = datetime.now(timezone.utc) - __import__("datetime").timedelta(days=keep_days)
        with self._tx() as c:
            cur = c.execute(
                "DELETE FROM decisions WHERE agent_id=? AND timestamp < ?",
                (agent_id, cutoff.isoformat()),
            )
            return cur.rowcount

    def purge_old_snapshots(self, agent_id: str, keep_days: int = 90) -> int:
        """Delete metric snapshots older than keep_days."""
        cutoff = datetime.now(timezone.utc) - __import__("datetime").timedelta(days=keep_days)
        with self._tx() as c:
            cur = c.execute(
                "DELETE FROM metric_snapshots WHERE agent_id=? AND timestamp < ?",
                (agent_id, cutoff.isoformat()),
            )
            return cur.rowcount

    def purge_old_alerts(self, agent_id: str, keep_days: int = 180) -> int:
        """Delete acknowledged alerts older than keep_days."""
        cutoff = datetime.now(timezone.utc) - __import__("datetime").timedelta(days=keep_days)
        with self._tx() as c:
            cur = c.execute(
                "DELETE FROM alerts WHERE agent_id=? AND acknowledged=1 AND timestamp < ?",
                (agent_id, cutoff.isoformat()),
            )
            return cur.rowcount

    def get_table_counts(self, agent_id: str) -> dict[str, int]:
        """Return row counts per table for this agent (for health monitoring)."""
        tables = {
            "decisions": "SELECT COUNT(*) FROM decisions WHERE agent_id=?",
            "baselines": "SELECT COUNT(*) FROM baselines WHERE agent_id=?",
            "alerts": "SELECT COUNT(*) FROM alerts WHERE agent_id=?",
            "rollbacks": "SELECT COUNT(*) FROM rollbacks WHERE agent_id=?",
            "metric_snapshots": "SELECT COUNT(*) FROM metric_snapshots WHERE agent_id=?",
        }
        counts = {}
        with self._lock:
            for name, sql in tables.items():
                counts[name] = self._conn.execute(sql, (agent_id,)).fetchone()[0]
        return counts

    def list_agents(self) -> list[str]:
        """Return all distinct agent IDs present in the database."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT agent_id FROM decisions ORDER BY agent_id"
            ).fetchall()
        return [r[0] for r in rows]

    def export_decisions(
        self,
        agent_id: str,
        *,
        category: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        limit: int = 10000,
    ) -> list[DecisionRecord]:
        """Export decisions matching filters (for compliance/audit)."""
        q = "SELECT * FROM decisions WHERE agent_id=?"
        params: list = [agent_id]
        if category and category != "*":
            q += " AND case_category=?"
            params.append(category)
        if since:
            q += " AND timestamp >= ?"
            params.append(since.isoformat())
        if until:
            q += " AND timestamp <= ?"
            params.append(until.isoformat())
        q += " ORDER BY timestamp ASC LIMIT ?"
        params.append(min(limit, 50000))
        with self._lock:
            rows = self._conn.execute(q, params).fetchall()
        return [self._to_decision(r) for r in rows]
