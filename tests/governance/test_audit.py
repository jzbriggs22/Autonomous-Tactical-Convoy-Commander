"""Tests for the tamper-evident audit log with hash-chain integrity."""

from __future__ import annotations

import threading
from datetime import datetime, timezone

import pytest

from ai_governance.audit import AuditEntry, AuditLog
from ai_governance.storage import GovernanceDB


@pytest.fixture
def audit():
    db = GovernanceDB(":memory:")
    return AuditLog(db)


@pytest.fixture
def audit_with_db():
    db = GovernanceDB(":memory:")
    return AuditLog(db), db


class TestAppend:
    def test_append_returns_entry(self, audit):
        entry = audit.append(
            agent_id="agent-1",
            action="decision.ingested",
            actor="system",
            resource_type="event",
            resource_id="evt-001",
            detail={"case_category": "billing_dispute"},
        )
        assert isinstance(entry, AuditEntry)
        assert entry.seq == 1
        assert entry.agent_id == "agent-1"
        assert entry.action == "decision.ingested"
        assert entry.actor == "system"
        assert entry.resource_type == "event"
        assert entry.resource_id == "evt-001"
        assert entry.detail == {"case_category": "billing_dispute"}
        assert entry.prev_hash == "genesis"
        assert len(entry.entry_hash) == 64  # SHA-256 hex

    def test_sequential_entries_chain(self, audit):
        e1 = audit.append("a", "decision.ingested", "sys", "event")
        e2 = audit.append("a", "baseline.computed", "sys", "baseline")
        e3 = audit.append("a", "drift.detected", "sys", "drift")
        assert e1.prev_hash == "genesis"
        assert e2.prev_hash == e1.entry_hash
        assert e3.prev_hash == e2.entry_hash

    def test_entries_have_unique_hashes(self, audit):
        entries = [
            audit.append("a", f"action.{i}", "sys", "test")
            for i in range(10)
        ]
        hashes = {e.entry_hash for e in entries}
        assert len(hashes) == 10

    def test_detail_defaults_to_empty_dict(self, audit):
        entry = audit.append("a", "config.loaded", "sys", "config")
        assert entry.detail == {}

    def test_resource_id_optional(self, audit):
        entry = audit.append("a", "config.loaded", "sys", "config")
        assert entry.resource_id is None

    def test_timestamp_is_utc(self, audit):
        entry = audit.append("a", "test", "sys", "test")
        assert entry.timestamp.tzinfo is not None


class TestGetEntries:
    def test_get_entries_by_agent(self, audit):
        audit.append("agent-A", "test", "sys", "test")
        audit.append("agent-B", "test", "sys", "test")
        audit.append("agent-A", "test2", "sys", "test")

        entries = audit.get_entries("agent-A")
        assert len(entries) == 2
        assert all(e.agent_id == "agent-A" for e in entries)

    def test_get_entries_by_action(self, audit):
        audit.append("a", "decision.ingested", "sys", "event")
        audit.append("a", "alert.fired", "sys", "alert")
        audit.append("a", "decision.ingested", "sys", "event")

        entries = audit.get_entries("a", action="decision.ingested")
        assert len(entries) == 2
        assert all(e.action == "decision.ingested" for e in entries)

    def test_get_entries_respects_limit(self, audit):
        for i in range(20):
            audit.append("a", "test", "sys", "test")

        entries = audit.get_entries("a", limit=5)
        assert len(entries) == 5

    def test_get_entries_returns_newest_first(self, audit):
        audit.append("a", "first", "sys", "test")
        audit.append("a", "second", "sys", "test")
        audit.append("a", "third", "sys", "test")

        entries = audit.get_entries("a")
        assert entries[0].action == "third"
        assert entries[-1].action == "first"

    def test_get_entries_empty(self, audit):
        entries = audit.get_entries("nonexistent")
        assert entries == []


class TestVerifyChain:
    def test_valid_chain(self, audit):
        for i in range(10):
            audit.append("a", f"action.{i}", "sys", "test", detail={"n": i})

        valid, broken_seq = audit.verify_chain("a")
        assert valid is True
        assert broken_seq is None

    def test_empty_chain_is_valid(self, audit):
        valid, broken_seq = audit.verify_chain("a")
        assert valid is True
        assert broken_seq is None

    def test_single_entry_chain(self, audit):
        audit.append("a", "test", "sys", "test")
        valid, broken_seq = audit.verify_chain("a")
        assert valid is True

    def test_tampered_hash_detected(self, audit_with_db):
        log, db = audit_with_db
        log.append("a", "first", "sys", "test")
        log.append("a", "second", "sys", "test")
        log.append("a", "third", "sys", "test")

        with db._lock:
            db._conn.execute(
                "UPDATE audit_log SET entry_hash='tampered' WHERE seq=2"
            )
            db._conn.commit()

        valid, broken_seq = log.verify_chain("a")
        assert valid is False
        assert broken_seq == 2

    def test_tampered_prev_hash_detected(self, audit_with_db):
        log, db = audit_with_db
        log.append("a", "first", "sys", "test")
        log.append("a", "second", "sys", "test")

        with db._lock:
            db._conn.execute(
                "UPDATE audit_log SET prev_hash='tampered' WHERE seq=2"
            )
            db._conn.commit()

        valid, broken_seq = log.verify_chain("a")
        assert valid is False
        assert broken_seq == 2

    def test_per_agent_verification(self, audit):
        audit.append("a", "test", "sys", "test")
        audit.append("b", "test", "sys", "test")

        valid_a, _ = audit.verify_chain("a")
        valid_b, _ = audit.verify_chain("b")
        assert valid_a is True
        assert valid_b is True


class TestConcurrency:
    def test_concurrent_appends_maintain_chain(self, audit):
        errors = []

        def worker(agent_id, n):
            try:
                for i in range(n):
                    audit.append(agent_id, f"action.{i}", "worker", "test")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=worker, args=("shared", 20))
            for _ in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        entries = audit.get_entries("shared", limit=200)
        assert len(entries) == 80
        valid, broken_seq = audit.verify_chain("shared")
        assert valid is True


class TestPersistence:
    def test_chain_survives_reload(self):
        db = GovernanceDB(":memory:")
        log1 = AuditLog(db)
        log1.append("a", "first", "sys", "test")
        log1.append("a", "second", "sys", "test")

        log2 = AuditLog(db)
        log2.append("a", "third", "sys", "test")

        entries = log2.get_entries("a")
        assert len(entries) == 3
        valid, _ = log2.verify_chain("a")
        assert valid is True
