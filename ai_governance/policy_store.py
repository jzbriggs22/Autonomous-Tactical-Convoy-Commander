"""Governance policy version store.

Tracks the history of GovernanceConfig changes with timestamps and
change descriptions. Supports rollback to any prior policy version.
Stored in the same SQLite database as governance events.
"""

from __future__ import annotations

import json
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from .config import GovernanceConfig


@dataclass
class PolicyVersion:
    policy_id: str
    agent_id: str
    created_at: str
    config_version: str
    config_fingerprint: str
    changed_by: str
    change_description: str
    config_json: str
    is_active: bool

    def to_config(self) -> GovernanceConfig:
        return GovernanceConfig.model_validate(json.loads(self.config_json))


class PolicyStore:
    """Persistent history of governance config versions.

    Each config change is saved as a new PolicyVersion. The active
    version is the one most recently committed. Previous versions are
    retained for audit and rollback.
    """

    def __init__(self, db) -> None:
        self._db = db
        self._lock = threading.RLock()
        self._ensure_table()

    def _ensure_table(self) -> None:
        with self._tx() as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS policy_versions (
                    policy_id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    config_version TEXT NOT NULL,
                    config_fingerprint TEXT NOT NULL,
                    changed_by TEXT NOT NULL,
                    change_description TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    is_active INTEGER NOT NULL DEFAULT 0
                )
            """)
            c.execute("""
                CREATE INDEX IF NOT EXISTS idx_policy_versions_agent
                ON policy_versions (agent_id, created_at DESC)
            """)

    @contextmanager
    def _tx(self):
        with self._lock:
            with self._db._conn:
                yield self._db._conn.cursor()

    def commit(
        self,
        config: GovernanceConfig,
        changed_by: str,
        description: str,
    ) -> PolicyVersion:
        now = datetime.now(timezone.utc)
        policy_id = str(uuid.uuid4())
        with self._tx() as c:
            c.execute(
                "UPDATE policy_versions SET is_active=0 WHERE agent_id=?",
                (config.agent_id,),
            )
            c.execute(
                """INSERT INTO policy_versions
                   (policy_id, agent_id, created_at, config_version, config_fingerprint,
                    changed_by, change_description, config_json, is_active)
                   VALUES (?,?,?,?,?,?,?,?,1)""",
                (
                    policy_id, config.agent_id, now.isoformat(),
                    config.version, config.fingerprint,
                    changed_by, description, config.to_json(),
                ),
            )
        return PolicyVersion(
            policy_id=policy_id,
            agent_id=config.agent_id,
            created_at=now.isoformat(),
            config_version=config.version,
            config_fingerprint=config.fingerprint,
            changed_by=changed_by,
            change_description=description,
            config_json=config.to_json(),
            is_active=True,
        )

    def list_versions(
        self, agent_id: str, limit: int = 20
    ) -> list[PolicyVersion]:
        with self._lock:
            rows = self._db._conn.execute(
                """SELECT * FROM policy_versions WHERE agent_id=?
                   ORDER BY created_at DESC LIMIT ?""",
                (agent_id, limit),
            ).fetchall()
        return [self._to_version(r) for r in rows]

    def get_active(self, agent_id: str) -> Optional[PolicyVersion]:
        with self._lock:
            row = self._db._conn.execute(
                "SELECT * FROM policy_versions WHERE agent_id=? AND is_active=1",
                (agent_id,),
            ).fetchone()
        return self._to_version(row) if row else None

    def get_version(self, policy_id: str) -> Optional[PolicyVersion]:
        with self._lock:
            row = self._db._conn.execute(
                "SELECT * FROM policy_versions WHERE policy_id=?",
                (policy_id,),
            ).fetchone()
        return self._to_version(row) if row else None

    def rollback_to(self, policy_id: str) -> Optional[PolicyVersion]:
        """Make a prior policy version the active one. Returns updated version or None."""
        with self._tx() as c:
            row = c.execute(
                "SELECT * FROM policy_versions WHERE policy_id=?", (policy_id,)
            ).fetchone()
            if row is None:
                return None
            agent_id = row["agent_id"]
            c.execute(
                "UPDATE policy_versions SET is_active=0 WHERE agent_id=?",
                (agent_id,),
            )
            c.execute(
                "UPDATE policy_versions SET is_active=1 WHERE policy_id=?",
                (policy_id,),
            )
        return self.get_version(policy_id)

    def _to_version(self, row) -> PolicyVersion:
        return PolicyVersion(
            policy_id=row["policy_id"],
            agent_id=row["agent_id"],
            created_at=row["created_at"],
            config_version=row["config_version"],
            config_fingerprint=row["config_fingerprint"],
            changed_by=row["changed_by"],
            change_description=row["change_description"],
            config_json=row["config_json"],
            is_active=bool(row["is_active"]),
        )
