"""Ingestion layer: validates, classifies risk, and stores agent decision events."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from .config import GovernanceConfig, HighRiskPattern
from .storage import DecisionRecord, GovernanceDB
from .structured import GovernanceDecision

VALID_DECISIONS = frozenset({"resolve", "escalate", "deny", "defer", "partial_resolve"})


@dataclass
class IngestRequest:
    """One agent decision event submitted by the caller."""

    case_id: str
    case_category: str
    decision: str
    resolution_time_ms: int
    metadata: dict = field(default_factory=dict)
    ground_truth: Optional[str] = None
    timestamp: Optional[datetime] = None
    event_id: Optional[str] = None


@dataclass
class IngestResult:
    event_id: str
    is_high_risk: bool
    high_risk_score: float  # 0.0–1.0
    matched_patterns: list[str]


class ValidationError(ValueError):
    pass


class IngestionLayer:
    """Entry point for all agent decision events.

    Responsibilities:
    - Input validation (raises ValidationError on bad input)
    - High-risk classification via pattern matching
    - Persistence via GovernanceDB
    """

    def __init__(self, config: GovernanceConfig, db: GovernanceDB) -> None:
        self._config = config
        self._db = db
        self._compiled: list[tuple[HighRiskPattern, re.Pattern]] = [
            (p, re.compile(p.pattern, re.IGNORECASE))
            for p in config.high_risk_patterns
        ]

    def ingest(self, req: IngestRequest) -> IngestResult:
        self._validate(req)
        is_high_risk, score, matched = self._classify_risk(req)
        rec = DecisionRecord(
            event_id=req.event_id or str(uuid.uuid4()),
            agent_id=self._config.agent_id,
            timestamp=req.timestamp or datetime.now(timezone.utc),
            case_id=req.case_id,
            case_category=req.case_category,
            is_high_risk=is_high_risk,
            high_risk_score=score,
            decision=req.decision,
            resolution_time_ms=req.resolution_time_ms,
            ground_truth=req.ground_truth,
            metadata=req.metadata,
            config_version=f"{self._config.version}:{self._config.fingerprint}",
        )
        self._db.insert_decision(rec)
        return IngestResult(
            event_id=rec.event_id,
            is_high_risk=is_high_risk,
            high_risk_score=score,
            matched_patterns=matched,
        )

    def ingest_structured(
        self,
        decision: GovernanceDecision,
        case_id: Optional[str] = None,
        resolution_time_ms: int = 0,
        ground_truth: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> IngestResult:
        """Ingest a structured GovernanceDecision (from outlines/instructor/Pydantic).

        Maps GovernanceDecision fields to IngestRequest, storing risk_level,
        confidence, and flags in metadata for downstream inspection.
        """
        req = IngestRequest(
            case_id=case_id or str(uuid.uuid4()),
            case_category=decision.case_category,
            decision=decision.decision,
            resolution_time_ms=resolution_time_ms,
            metadata={
                "risk_level": decision.risk_level,
                "confidence": decision.confidence,
                "flags": decision.flags,
            },
            ground_truth=ground_truth,
            timestamp=timestamp,
        )
        return self.ingest(req)

    def ingest_batch(self, requests: list[IngestRequest]) -> list[IngestResult]:
        return [self.ingest(r) for r in requests]

    def add_ground_truth(self, event_id: str, ground_truth: str) -> bool:
        """Attach ground truth label to an existing event. Returns False if event not found."""
        if ground_truth not in VALID_DECISIONS:
            raise ValidationError(
                f"Invalid ground truth {ground_truth!r}. "
                f"Must be one of: {sorted(VALID_DECISIONS)}"
            )
        return self._db.set_ground_truth(event_id, self._config.agent_id, ground_truth)

    # ── private ──────────────────────────────────────────────────────────────

    def _validate(self, req: IngestRequest) -> None:
        if not req.case_id or not req.case_id.strip():
            raise ValidationError("case_id is required")
        if not req.case_category or not req.case_category.strip():
            raise ValidationError("case_category is required")
        if req.decision not in VALID_DECISIONS:
            raise ValidationError(
                f"Invalid decision {req.decision!r}. "
                f"Must be one of: {sorted(VALID_DECISIONS)}"
            )
        if req.resolution_time_ms < 0:
            raise ValidationError("resolution_time_ms must be non-negative")
        if req.event_id is not None and not req.event_id.strip():
            raise ValidationError("event_id must be non-empty if provided")

    def _classify_risk(
        self, req: IngestRequest
    ) -> tuple[bool, float, list[str]]:
        # Build a flat dict of all checkable fields
        checkable = {
            "case_category": req.case_category,
            "decision": req.decision,
            **{k: str(v) for k, v in req.metadata.items()},
        }

        total_weight = 0.0
        matched: list[str] = []
        for pattern, compiled in self._compiled:
            target = checkable.get(pattern.field, "")
            if compiled.search(target):
                total_weight += pattern.weight
                matched.append(pattern.name)

        # Normalize: max possible weight is sum of all pattern weights
        max_weight = sum(p.weight for p in self._config.high_risk_patterns) or 1.0
        score = min(total_weight / max_weight, 1.0)
        return total_weight > 0.0, score, matched
