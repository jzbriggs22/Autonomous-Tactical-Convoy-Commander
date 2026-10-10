"""Database schema migration system.

Migrations are forward-only, idempotent, and tracked in `schema_migrations`.
Each migration has a unique version int and a human-readable description.
Run `apply_migrations(conn)` once at DB init; it's a no-op if up-to-date.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Callable

_BOOTSTRAP = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version     INTEGER PRIMARY KEY,
    description TEXT    NOT NULL,
    applied_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);
"""


@dataclass(frozen=True)
class Migration:
    version: int
    description: str
    up: Callable[[sqlite3.Connection], None]


def _v1_initial_schema(c: sqlite3.Connection) -> None:
    c.executescript("""
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
        metadata_json       TEXT NOT NULL DEFAULT '{}',
        config_version      TEXT NOT NULL DEFAULT ''
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

    CREATE TABLE IF NOT EXISTS metric_snapshots (
        snapshot_id     INTEGER PRIMARY KEY AUTOINCREMENT,
        agent_id        TEXT NOT NULL,
        timestamp       TEXT NOT NULL,
        category        TEXT NOT NULL,
        metric          TEXT NOT NULL,
        value           REAL NOT NULL,
        sample_count    INTEGER NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_snap_agent_cat
        ON metric_snapshots (agent_id, category, metric, timestamp);

    CREATE TABLE IF NOT EXISTS audit_log (
        seq             INTEGER PRIMARY KEY AUTOINCREMENT,
        agent_id        TEXT NOT NULL,
        timestamp       TEXT NOT NULL,
        action          TEXT NOT NULL,
        actor           TEXT NOT NULL,
        resource_type   TEXT NOT NULL,
        resource_id     TEXT,
        detail_json     TEXT NOT NULL DEFAULT '{}',
        prev_hash       TEXT NOT NULL,
        entry_hash      TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_audit_agent_ts ON audit_log (agent_id, timestamp);
    CREATE INDEX IF NOT EXISTS idx_audit_action   ON audit_log (agent_id, action);
    """)


def _v2_decisions_ground_truth_index(c: sqlite3.Connection) -> None:
    c.execute(
        "CREATE INDEX IF NOT EXISTS idx_dec_gt "
        "ON decisions (agent_id, ground_truth) "
        "WHERE ground_truth IS NOT NULL"
    )


def _v3_alerts_acknowledged_index(c: sqlite3.Connection) -> None:
    c.execute(
        "CREATE INDEX IF NOT EXISTS idx_alert_ack "
        "ON alerts (agent_id, acknowledged, timestamp)"
    )


def _v4_decisions_config_version_index(c: sqlite3.Connection) -> None:
    c.execute(
        "CREATE INDEX IF NOT EXISTS idx_dec_cfg "
        "ON decisions (agent_id, config_version)"
    )


def _v5_rollbacks_resolved_index(c: sqlite3.Connection) -> None:
    c.execute(
        "CREATE INDEX IF NOT EXISTS idx_rb_resolved "
        "ON rollbacks (agent_id, resolved)"
    )


MIGRATIONS: list[Migration] = [
    Migration(1, "initial schema — all tables and base indexes", _v1_initial_schema),
    Migration(2, "decisions: partial index on ground_truth", _v2_decisions_ground_truth_index),
    Migration(3, "alerts: composite index on acknowledged + timestamp", _v3_alerts_acknowledged_index),
    Migration(4, "decisions: index on config_version for policy audit queries", _v4_decisions_config_version_index),
    Migration(5, "rollbacks: index on resolved status", _v5_rollbacks_resolved_index),
]


def apply_migrations(conn: sqlite3.Connection) -> list[int]:
    """Apply all pending migrations. Returns list of newly applied version numbers."""
    conn.executescript(_BOOTSTRAP)
    conn.commit()

    applied = {
        row[0]
        for row in conn.execute("SELECT version FROM schema_migrations").fetchall()
    }
    newly_applied: list[int] = []
    for m in MIGRATIONS:
        if m.version in applied:
            continue
        m.up(conn)
        conn.execute(
            "INSERT INTO schema_migrations (version, description) VALUES (?, ?)",
            (m.version, m.description),
        )
        conn.commit()
        newly_applied.append(m.version)

    return newly_applied


def current_version(conn: sqlite3.Connection) -> int:
    """Return the highest applied migration version (0 if none)."""
    conn.executescript(_BOOTSTRAP)
    conn.commit()
    row = conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()
    return row[0] or 0


def migration_status(conn: sqlite3.Connection) -> list[dict]:
    """Return all migrations with applied/pending status."""
    conn.executescript(_BOOTSTRAP)
    conn.commit()
    applied = {
        row[0]: row[1]
        for row in conn.execute(
            "SELECT version, applied_at FROM schema_migrations"
        ).fetchall()
    }
    return [
        {
            "version": m.version,
            "description": m.description,
            "status": "applied" if m.version in applied else "pending",
            "applied_at": applied.get(m.version),
        }
        for m in MIGRATIONS
    ]
