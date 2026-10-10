"""Event replay engine for governance config testing.

Replays historical decision events against a candidate config to predict
how policy changes would affect drift scores, alert firing, and risk
classification — without affecting the live system.

Usage:
    replayer = EventReplayer(live_config, candidate_config, db)
    result = replayer.replay(limit=1000)
    print(result.summary())
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from .alerts import AlertEngine
from .config import GovernanceConfig
from .drift import DriftDetector
from .ingestion import IngestionLayer, IngestRequest
from .storage import DecisionRecord, GovernanceDB


@dataclass
class ReplayDiff:
    """Comparison between live and candidate config on a single event."""
    event_id: str
    case_category: str
    decision: str
    live_is_high_risk: bool
    candidate_is_high_risk: bool
    risk_changed: bool
    live_score: float
    candidate_score: float
    live_patterns: list[str]
    candidate_patterns: list[str]


@dataclass
class ReplayResult:
    events_replayed: int
    live_config_version: str
    candidate_config_version: str

    live_high_risk_count: int
    candidate_high_risk_count: int
    risk_reclassified_count: int
    new_high_risk_count: int
    no_longer_high_risk_count: int

    live_drift_score: float
    candidate_drift_score: float
    live_violations: int
    candidate_violations: int
    live_alerts_would_fire: int
    candidate_alerts_would_fire: int

    diffs: list[ReplayDiff] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Replay: {self.events_replayed} events",
            f"  Live config:      v{self.live_config_version}",
            f"  Candidate config: v{self.candidate_config_version}",
            "",
            f"  High-risk events:     {self.live_high_risk_count} -> {self.candidate_high_risk_count}",
            f"    Newly classified:   +{self.new_high_risk_count}",
            f"    No longer flagged:  -{self.no_longer_high_risk_count}",
            f"    Total reclassified: {self.risk_reclassified_count}",
            "",
            f"  Drift score:  {self.live_drift_score:.4f} -> {self.candidate_drift_score:.4f}",
            f"  Violations:   {self.live_violations} -> {self.candidate_violations}",
            f"  Alerts:       {self.live_alerts_would_fire} -> {self.candidate_alerts_would_fire}",
        ]
        if self.candidate_violations > self.live_violations:
            lines.append("")
            lines.append("  [!] Candidate config would increase violations")
        if self.candidate_alerts_would_fire > self.live_alerts_would_fire:
            lines.append("  [!] Candidate config would fire more alerts")
        return "\n".join(lines)


class EventReplayer:
    """Replay historical events against a candidate governance config.

    Creates an isolated in-memory database for each config and replays
    all events to compare behavior. The live database is never modified.
    """

    def __init__(
        self,
        live_config: GovernanceConfig,
        candidate_config: GovernanceConfig,
        db: GovernanceDB,
    ) -> None:
        self._live_config = live_config
        self._candidate_config = candidate_config
        self._db = db

    def replay(
        self,
        limit: int = 10000,
        category: Optional[str] = None,
    ) -> ReplayResult:
        records = self._db.get_recent_decisions(
            self._live_config.agent_id,
            category=category,
            limit=limit,
            oldest_first=True,
        )

        live_sandbox = GovernanceDB(":memory:")
        cand_sandbox = GovernanceDB(":memory:")

        live_layer = IngestionLayer(self._live_config, live_sandbox)
        cand_layer = IngestionLayer(self._candidate_config, cand_sandbox)

        diffs: list[ReplayDiff] = []
        live_hr = 0
        cand_hr = 0
        new_hr = 0
        no_longer_hr = 0
        risk_changed = 0

        for rec in records:
            req = IngestRequest(
                case_id=rec.case_id,
                case_category=rec.case_category,
                decision=rec.decision,
                resolution_time_ms=rec.resolution_time_ms,
                metadata=rec.metadata,
                ground_truth=rec.ground_truth,
                timestamp=rec.timestamp,
                event_id=str(uuid.uuid4()),
            )

            live_result = live_layer.ingest(req)
            req.event_id = str(uuid.uuid4())
            cand_result = cand_layer.ingest(req)

            if live_result.is_high_risk:
                live_hr += 1
            if cand_result.is_high_risk:
                cand_hr += 1

            changed = live_result.is_high_risk != cand_result.is_high_risk
            if changed:
                risk_changed += 1
                if cand_result.is_high_risk and not live_result.is_high_risk:
                    new_hr += 1
                elif not cand_result.is_high_risk and live_result.is_high_risk:
                    no_longer_hr += 1

            if changed:
                diffs.append(ReplayDiff(
                    event_id=rec.event_id,
                    case_category=rec.case_category,
                    decision=rec.decision,
                    live_is_high_risk=live_result.is_high_risk,
                    candidate_is_high_risk=cand_result.is_high_risk,
                    risk_changed=True,
                    live_score=live_result.high_risk_score,
                    candidate_score=cand_result.high_risk_score,
                    live_patterns=live_result.matched_patterns,
                    candidate_patterns=cand_result.matched_patterns,
                ))

        live_det = DriftDetector(self._live_config, live_sandbox)
        cand_det = DriftDetector(self._candidate_config, cand_sandbox)

        live_det.compute_baseline()
        cand_det.compute_baseline()

        live_report = live_det.detect()
        cand_report = cand_det.detect()

        live_eng = AlertEngine(self._live_config, live_sandbox)
        cand_eng = AlertEngine(self._candidate_config, cand_sandbox)

        live_alerts = live_eng.evaluate(live_report)
        cand_alerts = cand_eng.evaluate(cand_report)

        return ReplayResult(
            events_replayed=len(records),
            live_config_version=f"{self._live_config.version}:{self._live_config.fingerprint}",
            candidate_config_version=f"{self._candidate_config.version}:{self._candidate_config.fingerprint}",
            live_high_risk_count=live_hr,
            candidate_high_risk_count=cand_hr,
            risk_reclassified_count=risk_changed,
            new_high_risk_count=new_hr,
            no_longer_high_risk_count=no_longer_hr,
            live_drift_score=live_report.overall_drift_score,
            candidate_drift_score=cand_report.overall_drift_score,
            live_violations=len(live_report.violations),
            candidate_violations=len(cand_report.violations),
            live_alerts_would_fire=len(live_alerts),
            candidate_alerts_would_fire=len(cand_alerts),
            diffs=diffs,
        )
