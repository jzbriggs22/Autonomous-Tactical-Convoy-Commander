"""FastAPI HTTP server exposing governance endpoints.

Start with:
    uvicorn ai_governance.api:app --port 8080
    # or: GOVERNANCE_DB=governance.db python -m ai_governance.api
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from .alerts import AlertEngine
from .audit import AuditLog
from .config import GovernanceConfig
from .dashboard import DashboardBuilder
from .drift import DriftDetector
from .ingestion import IngestRequest, IngestionLayer, ValidationError
from .storage import GovernanceDB
from .structured import DecodeError, DecisionDecoder, GovernanceDecision
from .webhooks import WebhookDispatcher

# ── app + lazy singleton wiring ──────────────────────────────────────────────

app = FastAPI(
    title="AI Governance API",
    version="1.0.0",
    description="Observability and drift detection for AI agents",
)


class _Services:
    config: GovernanceConfig
    db: GovernanceDB
    ingestion: IngestionLayer
    detector: DriftDetector
    engine: AlertEngine
    dashboard: DashboardBuilder
    webhooks: WebhookDispatcher
    audit: AuditLog


_svc: Optional[_Services] = None


def _get_svc() -> _Services:
    global _svc
    if _svc is None:
        _svc = _Services()
        _svc.config = GovernanceConfig.default_customer_service()
        _svc.db = GovernanceDB(os.environ.get("GOVERNANCE_DB", ":memory:"))
        _svc.ingestion = IngestionLayer(_svc.config, _svc.db)
        _svc.detector = DriftDetector(_svc.config, _svc.db)
        _svc.engine = AlertEngine(_svc.config, _svc.db)
        _svc.dashboard = DashboardBuilder(
            _svc.config, _svc.db, _svc.detector, _svc.engine
        )
        _svc.webhooks = WebhookDispatcher()
        _svc.audit = AuditLog(_svc.db)
    return _svc


def init_services(
    config: GovernanceConfig, db: GovernanceDB, webhooks: WebhookDispatcher = None
) -> _Services:
    """Explicit initialization — used by tests to inject in-memory DB."""
    global _svc
    _svc = _Services()
    _svc.config = config
    _svc.db = db
    _svc.ingestion = IngestionLayer(config, db)
    _svc.detector = DriftDetector(config, db)
    _svc.engine = AlertEngine(config, db)
    _svc.dashboard = DashboardBuilder(config, db, _svc.detector, _svc.engine)
    _svc.webhooks = webhooks or WebhookDispatcher()
    _svc.audit = AuditLog(db)
    return _svc


# ── request / response models ────────────────────────────────────────────────

class DecisionRequest(BaseModel):
    case_id: str
    case_category: str
    decision: str
    resolution_time_ms: int = Field(ge=0)
    metadata: dict = Field(default_factory=dict)
    ground_truth: Optional[str] = None
    timestamp: Optional[datetime] = None
    event_id: Optional[str] = None


class GroundTruthRequest(BaseModel):
    ground_truth: str


class BaselineRequest(BaseModel):
    categories: Optional[list[str]] = None


class ResolveRollbackRequest(BaseModel):
    resolved_by: str


class AcknowledgeAlertRequest(BaseModel):
    pass  # presence of the POST is the ack


class StructuredDecisionRequest(BaseModel):
    """Wrapper for a GovernanceDecision submitted as structured JSON.

    Accepts either a pre-formed GovernanceDecision payload (nested under
    ``decision``) or raw JSON text in ``raw`` — the decoder validates both.
    """
    decision: Optional[GovernanceDecision] = None
    raw: Optional[str] = None
    case_id: Optional[str] = None
    resolution_time_ms: int = Field(default=0, ge=0)
    ground_truth: Optional[str] = None


# ── endpoints ────────────────────────────────────────────────────────────────

@app.post("/events", status_code=status.HTTP_201_CREATED)
def ingest_event(body: DecisionRequest):
    svc = _get_svc()
    req = IngestRequest(
        case_id=body.case_id,
        case_category=body.case_category,
        decision=body.decision,
        resolution_time_ms=body.resolution_time_ms,
        metadata=body.metadata,
        ground_truth=body.ground_truth,
        timestamp=body.timestamp,
        event_id=body.event_id,
    )
    try:
        result = svc.ingestion.ingest(req)
    except ValidationError as exc:
        raise HTTPException(422, str(exc))
    svc.audit.append(
        svc.config.agent_id, "decision.ingested", "system", "event",
        resource_id=result.event_id,
        detail={"category": body.case_category, "decision": body.decision,
                "is_high_risk": result.is_high_risk},
    )
    return {
        "event_id": result.event_id,
        "is_high_risk": result.is_high_risk,
        "high_risk_score": result.high_risk_score,
        "matched_patterns": result.matched_patterns,
    }


@app.post("/events/structured", status_code=status.HTTP_201_CREATED)
def ingest_structured_event(body: StructuredDecisionRequest):
    """Ingest a GovernanceDecision — the constrained-output path.

    Accepts either ``decision`` (pre-validated Pydantic object) or
    ``raw`` (JSON string validated via outlines_core regex before Pydantic).
    """
    svc = _get_svc()
    _dec = DecisionDecoder()
    if body.decision is not None:
        gov_decision = body.decision
    elif body.raw is not None:
        try:
            gov_decision = _dec.decode(body.raw)
        except DecodeError as exc:
            raise HTTPException(422, str(exc))
    else:
        raise HTTPException(422, "Provide either 'decision' or 'raw'")

    try:
        result = svc.ingestion.ingest_structured(
            gov_decision,
            case_id=body.case_id,
            resolution_time_ms=body.resolution_time_ms,
            ground_truth=body.ground_truth,
        )
    except ValidationError as exc:
        raise HTTPException(422, str(exc))

    svc.audit.append(
        svc.config.agent_id, "decision.ingested", "system", "event",
        resource_id=result.event_id,
        detail={"category": gov_decision.case_category,
                "decision": gov_decision.decision,
                "risk_level": gov_decision.risk_level,
                "is_high_risk": result.is_high_risk,
                "structured": True},
    )
    return {
        "event_id": result.event_id,
        "is_high_risk": result.is_high_risk,
        "high_risk_score": result.high_risk_score,
        "matched_patterns": result.matched_patterns,
        "risk_level": gov_decision.risk_level,
        "confidence": gov_decision.confidence,
    }


@app.get("/events/schema")
def get_decision_schema():
    """Return the GovernanceDecision JSON schema for use in LLM prompts."""
    return GovernanceDecision.model_json_schema()


@app.post("/events/batch", status_code=status.HTTP_201_CREATED)
def ingest_batch(body: list[DecisionRequest]):
    svc = _get_svc()
    if len(body) > 1000:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "Max 1000 events per batch"
        )
    results = []
    for b in body:
        req = IngestRequest(
            case_id=b.case_id,
            case_category=b.case_category,
            decision=b.decision,
            resolution_time_ms=b.resolution_time_ms,
            metadata=b.metadata,
            ground_truth=b.ground_truth,
            timestamp=b.timestamp,
            event_id=b.event_id,
        )
        try:
            r = svc.ingestion.ingest(req)
        except ValidationError as exc:
            raise HTTPException(422, str(exc))
        results.append({
            "event_id": r.event_id,
            "is_high_risk": r.is_high_risk,
            "high_risk_score": r.high_risk_score,
        })
    return results


@app.post("/events/{event_id}/ground-truth")
def add_ground_truth(event_id: str, body: GroundTruthRequest):
    svc = _get_svc()
    try:
        found = svc.ingestion.add_ground_truth(event_id, body.ground_truth)
    except ValidationError as exc:
        raise HTTPException(422, str(exc))
    if not found:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Event {event_id!r} not found")
    return {"status": "updated", "event_id": event_id}


@app.get("/drift")
def get_drift():
    svc = _get_svc()
    report = svc.detector.detect()
    fired = svc.engine.evaluate(report)
    for f in fired:
        svc.webhooks.dispatch_async({
            "alert_id": f.alert_id,
            "rule_name": f.rule_name,
            "severity": f.severity,
            "message": f.message,
            "triggered_rollback": f.triggered_rollback,
            "agent_id": svc.config.agent_id,
        })
        action = "rollback.triggered" if f.triggered_rollback else "alert.fired"
        svc.audit.append(
            svc.config.agent_id, action, "system", "alert",
            resource_id=f.alert_id,
            detail={"rule": f.rule_name, "severity": f.severity,
                    "triggered_rollback": f.triggered_rollback},
        )
    svc.audit.append(
        svc.config.agent_id, "drift.detected", "system", "drift",
        detail={"overall_score": round(report.overall_drift_score, 4),
                "violations": len(report.violations), "alerts_fired": len(fired)},
    )
    return {
        "agent_id": report.agent_id,
        "overall_drift_score": round(report.overall_drift_score, 4),
        "category_drift_scores": {
            k: round(v, 4) for k, v in report.category_drift_scores.items()
        },
        "violations": [
            {
                "category": v.category,
                "metric": v.metric,
                "rule": v.rule_name,
                "baseline": round(v.baseline_value, 4),
                "recent": round(v.recent_value, 4),
                "delta": round(v.delta, 4),
                "severity": v.severity,
            }
            for v in report.violations
        ],
        "alerts_fired": len(fired),
    }


@app.get("/dashboard")
def get_dashboard():
    svc = _get_svc()
    snap = svc.dashboard.build()
    return snap.to_dict()


@app.get("/alerts")
def get_alerts(limit: int = 20):
    svc = _get_svc()
    alerts = svc.db.get_recent_alerts(svc.config.agent_id, limit=min(limit, 100))
    return [
        {
            "alert_id": a.alert_id,
            "timestamp": a.timestamp.isoformat(),
            "rule": a.rule_name,
            "severity": a.severity,
            "message": a.message,
            "acknowledged": a.acknowledged,
        }
        for a in alerts
    ]


@app.post("/alerts/{alert_id}/acknowledge")
def acknowledge_alert(alert_id: str):
    svc = _get_svc()
    found = svc.db.acknowledge_alert(svc.config.agent_id, alert_id)
    if not found:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Alert {alert_id!r} not found")
    svc.audit.append(
        svc.config.agent_id, "alert.acknowledged", "pm", "alert",
        resource_id=alert_id,
    )
    return {"status": "acknowledged", "alert_id": alert_id}


@app.get("/status")
def get_status():
    svc = _get_svc()
    is_safe, reason = svc.engine.is_agent_safe()
    return {"is_safe": is_safe, "reason": reason, "agent_id": svc.config.agent_id}


@app.get("/rollbacks")
def get_rollbacks():
    svc = _get_svc()
    rollbacks = svc.db.get_rollbacks(svc.config.agent_id, limit=20)
    return [
        {
            "rollback_id": r.rollback_id,
            "timestamp": r.timestamp.isoformat(),
            "trigger_rule": r.trigger_rule,
            "reason": r.reason,
            "resolved": r.resolved,
            "resolved_at": r.resolved_at.isoformat() if r.resolved_at else None,
            "resolved_by": r.resolved_by,
        }
        for r in rollbacks
    ]


@app.post("/rollbacks/{rollback_id}/resolve")
def resolve_rollback(rollback_id: str, body: ResolveRollbackRequest):
    svc = _get_svc()
    found = svc.db.resolve_rollback(svc.config.agent_id, rollback_id, body.resolved_by)
    if not found:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"Rollback {rollback_id!r} not found or already resolved",
        )
    svc.audit.append(
        svc.config.agent_id, "rollback.resolved", body.resolved_by, "rollback",
        resource_id=rollback_id,
    )
    return {"status": "resolved", "rollback_id": rollback_id, "resolved_by": body.resolved_by}


@app.post("/rollbacks/resolve-all")
def resolve_all_rollbacks(body: ResolveRollbackRequest):
    svc = _get_svc()
    count = svc.db.resolve_all_rollbacks(svc.config.agent_id, body.resolved_by)
    if count > 0:
        svc.audit.append(
            svc.config.agent_id, "rollback.resolved", body.resolved_by, "rollback",
            detail={"resolve_all": True, "count": count},
        )
    return {"status": "resolved", "count": count, "resolved_by": body.resolved_by}


@app.post("/baseline/compute")
def compute_baseline(body: BaselineRequest = None):
    svc = _get_svc()
    categories = body.categories if body else None
    results = svc.detector.compute_baseline(categories)
    svc.audit.append(
        svc.config.agent_id, "baseline.computed", "system", "baseline",
        detail={"categories": list(results.keys()),
                "event_counts": {cat: m.total_events for cat, m in results.items()}},
    )
    return {
        "categories_computed": list(results.keys()),
        "event_counts": {cat: m.total_events for cat, m in results.items()},
    }


@app.get("/metrics/history/{category}/{metric}")
def get_metric_history(category: str, metric: str, limit: int = 50):
    svc = _get_svc()
    history = svc.db.get_metric_history(
        svc.config.agent_id, category, metric, limit=min(limit, 200)
    )
    baseline = svc.db.get_baseline(svc.config.agent_id, category, metric)
    return {
        "category": category,
        "metric": metric,
        "baseline_value": baseline[0] if baseline else None,
        "baseline_samples": baseline[1] if baseline else None,
        "history": [
            {
                "timestamp": ts.isoformat(),
                "value": round(val, 4),
                "sample_count": sc,
            }
            for ts, val, sc in history
        ],
    }


@app.get("/audit")
def get_audit(action: Optional[str] = None, limit: int = 50):
    svc = _get_svc()
    entries = svc.audit.get_entries(
        svc.config.agent_id, action=action, limit=min(limit, 200)
    )
    return [
        {
            "seq": e.seq,
            "timestamp": e.timestamp.isoformat(),
            "action": e.action,
            "actor": e.actor,
            "resource_type": e.resource_type,
            "resource_id": e.resource_id,
            "detail": e.detail,
            "entry_hash": e.entry_hash,
        }
        for e in entries
    ]


@app.get("/audit/verify")
def verify_audit():
    svc = _get_svc()
    valid, broken_seq = svc.audit.verify_chain(svc.config.agent_id)
    return {
        "agent_id": svc.config.agent_id,
        "chain_valid": valid,
        "first_broken_seq": broken_seq,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
