"""Tests for the database migration system."""

from __future__ import annotations

import sqlite3

import pytest
from fastapi.testclient import TestClient

from ai_governance.api import app, init_services
from ai_governance.auth import reset as reset_auth
from ai_governance.config import GovernanceConfig
from ai_governance.migrations import (
    MIGRATIONS,
    Migration,
    apply_migrations,
    current_version,
    migration_status,
)
from ai_governance.storage import GovernanceDB


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    yield c
    c.close()


@pytest.fixture(autouse=True)
def _clean_auth():
    reset_auth()
    yield
    reset_auth()


class TestApplyMigrations:
    def test_bootstrap_creates_tracking_table(self, conn):
        apply_migrations(conn)
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "schema_migrations" in tables

    def test_all_migrations_applied_on_fresh_db(self, conn):
        newly = apply_migrations(conn)
        assert len(newly) == len(MIGRATIONS)
        assert newly == [m.version for m in MIGRATIONS]

    def test_idempotent_second_run(self, conn):
        apply_migrations(conn)
        newly = apply_migrations(conn)
        assert newly == []

    def test_current_version_correct(self, conn):
        assert current_version(conn) == 0
        apply_migrations(conn)
        assert current_version(conn) == max(m.version for m in MIGRATIONS)

    def test_all_tables_created(self, conn):
        apply_migrations(conn)
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        for expected in [
            "decisions", "baselines", "alerts", "rollbacks",
            "metric_snapshots", "audit_log",
        ]:
            assert expected in tables, f"Missing table: {expected}"

    def test_indexes_created(self, conn):
        apply_migrations(conn)
        indexes = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        }
        assert "idx_dec_agent_ts" in indexes
        assert "idx_alert_agent_ts" in indexes
        assert "idx_audit_agent_ts" in indexes

    def test_incremental_migration(self, conn):
        # Apply only v1 manually
        m1 = next(m for m in MIGRATIONS if m.version == 1)
        m1.up(conn)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations "
            "(version INTEGER PRIMARY KEY, description TEXT NOT NULL, "
            "applied_at TEXT NOT NULL DEFAULT (datetime('now')))"
        )
        conn.execute(
            "INSERT INTO schema_migrations (version, description) VALUES (1, 'manual v1')"
        )
        conn.commit()

        # Now apply_migrations should apply v2+ only
        newly = apply_migrations(conn)
        assert 1 not in newly
        assert all(v >= 2 for v in newly)

    def test_migration_status_shows_all(self, conn):
        apply_migrations(conn)
        status = migration_status(conn)
        assert len(status) == len(MIGRATIONS)
        for s in status:
            assert s["status"] == "applied"
            assert s["applied_at"] is not None

    def test_migration_status_shows_pending_before_apply(self, conn):
        # bootstrap only
        from ai_governance.migrations import _BOOTSTRAP
        conn.executescript(_BOOTSTRAP)
        conn.commit()
        status = migration_status(conn)
        assert all(s["status"] == "pending" for s in status)


class TestMigrationDataIntegrity:
    def test_governance_db_uses_migrations(self):
        db = GovernanceDB(":memory:")
        with db._lock:
            version = current_version(db._conn)
        assert version == max(m.version for m in MIGRATIONS)

    def test_data_survives_migration(self, tmp_path):
        db_path = tmp_path / "test.db"

        # Simulate a DB that only has v1 applied
        conn = sqlite3.connect(str(db_path))
        m1 = next(m for m in MIGRATIONS if m.version == 1)
        m1.up(conn)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations "
            "(version INTEGER PRIMARY KEY, description TEXT NOT NULL, "
            "applied_at TEXT NOT NULL DEFAULT (datetime('now')))"
        )
        conn.execute(
            "INSERT INTO schema_migrations (version, description) VALUES (1, 'v1')"
        )
        conn.execute(
            "INSERT INTO decisions (event_id, agent_id, timestamp, case_id, "
            "case_category, is_high_risk, high_risk_score, decision, resolution_time_ms) "
            "VALUES ('ev1','ag1','2024-01-01T00:00:00','c1','billing',0,0.0,'resolve',100)"
        )
        conn.commit()
        conn.close()

        # Open through GovernanceDB — should apply remaining migrations
        db = GovernanceDB(str(db_path))
        decisions = db.get_recent_decisions("ag1", limit=10)
        assert len(decisions) == 1
        assert decisions[0].event_id == "ev1"


class TestMigrationEndpoint:
    def test_migrations_endpoint_returns_status(self):
        cfg = GovernanceConfig.default_customer_service()
        db = GovernanceDB(":memory:")
        init_services(cfg, db)
        client = TestClient(app)
        resp = client.get("/admin/migrations")
        assert resp.status_code == 200
        data = resp.json()
        assert "schema_version" in data
        assert "migrations" in data
        assert data["schema_version"] == max(m.version for m in MIGRATIONS)
        assert all(m["status"] == "applied" for m in data["migrations"])
