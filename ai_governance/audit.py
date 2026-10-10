"""Tamper-evident audit log for governance actions.

Records every governance-significant event with a hash chain:
each entry includes the SHA-256 hash of the previous entry,
making retroactive modification detectable.
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from .storage import GovernanceDB

_AUDIT_SCHEMA = """
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
"""


@dataclass
class AuditEntry:
    seq: int
    agent_id: str
    timestamp: datetime
    action: str
    actor: str
    resource_type: str
    resource_id: Optional[str]
    detail: dict
    prev_hash: str
    entry_hash: str


class AuditLog:
    """Append-only audit log with hash-chain integrity.

    Actions recorded:
      decision.ingested     — agent made a decision
      baseline.computed     — baseline was frozen
      drift.detected        — drift detection ran (with results)
      alert.fired           — threshold or condition triggered an alert
      alert.acknowledged    — PM acknowledged an alert
      rollback.triggered    — rollback was activated
      rollback.resolved     — PM resolved a rollback (agent resumed)
      config.loaded         — governance config was loaded/changed
    """

    def __init__(self, db: GovernanceDB) -> None:
        self._db = db
        self._lock = threading.Lock()
        self._ensure_schema()
        self._prev_hashes: dict[str, str] = {}

    def _ensure_schema(self) -> None:
        with self._db._lock:
            self._db._conn.executescript(_AUDIT_SCHEMA)
            self._db._conn.commit()

    def _get_prev_hash(self, agent_id: str) -> str:
        if agent_id not in self._prev_hashes:
            with self._db._lock:
                row = self._db._conn.execute(
                    "SELECT entry_hash FROM audit_log "
                    "WHERE agent_id=? ORDER BY seq DESC LIMIT 1",
                    (agent_id,),
                ).fetchone()
            self._prev_hashes[agent_id] = row["entry_hash"] if row else "genesis"
        return self._prev_hashes[agent_id]

    def append(
        self,
        agent_id: str,
        action: str,
        actor: str,
        resource_type: str,
        resource_id: Optional[str] = None,
        detail: Optional[dict] = None,
    ) -> AuditEntry:
        """Append an entry to the audit log. Returns the new entry."""
        with self._lock:
            now = datetime.now(timezone.utc)
            detail_json = json.dumps(detail or {}, sort_keys=True)
            prev_hash = self._get_prev_hash(agent_id)

            payload = f"{prev_hash}|{now.isoformat()}|{action}|{actor}|{detail_json}"
            entry_hash = hashlib.sha256(payload.encode()).hexdigest()

            with self._db._tx() as c:
                c.execute(
                    """INSERT INTO audit_log
                       (agent_id, timestamp, action, actor, resource_type,
                        resource_id, detail_json, prev_hash, entry_hash)
                       VALUES (?,?,?,?,?,?,?,?,?)""",
                    (
                        agent_id, now.isoformat(), action, actor,
                        resource_type, resource_id, detail_json,
                        prev_hash, entry_hash,
                    ),
                )
                seq = c.execute("SELECT last_insert_rowid()").fetchone()[0]

            entry = AuditEntry(
                seq=seq,
                agent_id=agent_id,
                timestamp=now,
                action=action,
                actor=actor,
                resource_type=resource_type,
                resource_id=resource_id,
                detail=detail or {},
                prev_hash=prev_hash,
                entry_hash=entry_hash,
            )
            self._prev_hashes[agent_id] = entry_hash
            return entry

    def get_entries(
        self,
        agent_id: str,
        *,
        action: Optional[str] = None,
        limit: int = 100,
    ) -> list[AuditEntry]:
        q = "SELECT * FROM audit_log WHERE agent_id=?"
        params: list = [agent_id]
        if action:
            q += " AND action=?"
            params.append(action)
        q += " ORDER BY seq DESC LIMIT ?"
        params.append(limit)
        with self._db._lock:
            rows = self._db._conn.execute(q, params).fetchall()
        return [
            AuditEntry(
                seq=r["seq"],
                agent_id=r["agent_id"],
                timestamp=datetime.fromisoformat(r["timestamp"]),
                action=r["action"],
                actor=r["actor"],
                resource_type=r["resource_type"],
                resource_id=r["resource_id"],
                detail=json.loads(r["detail_json"]),
                prev_hash=r["prev_hash"],
                entry_hash=r["entry_hash"],
            )
            for r in rows
        ]

    def verify_chain(self, agent_id: str) -> tuple[bool, Optional[int]]:
        """Verify the hash chain is intact. Returns (valid, first_broken_seq)."""
        with self._db._lock:
            rows = self._db._conn.execute(
                "SELECT * FROM audit_log WHERE agent_id=? ORDER BY seq ASC",
                (agent_id,),
            ).fetchall()

        prev_hash = "genesis"
        for row in rows:
            detail_json = row["detail_json"]
            payload = (
                f"{prev_hash}|{row['timestamp']}|{row['action']}"
                f"|{row['actor']}|{detail_json}"
            )
            expected = hashlib.sha256(payload.encode()).hexdigest()
            if expected != row["entry_hash"]:
                return False, row["seq"]
            if row["prev_hash"] != prev_hash:
                return False, row["seq"]
            prev_hash = row["entry_hash"]

        return True, None
