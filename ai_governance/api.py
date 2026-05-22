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

import time as _time

from .alerts import AlertEngine
from .audit import AuditLog
from .auth import AuthMiddleware, configure_from_env
from .config import GovernanceConfig
from .dashboard import DashboardBuilder
from .drift import DriftDetector
from .ingestion import IngestRequest, IngestionLayer, ValidationError
from .storage import GovernanceDB
from .structured import DecodeError, DecisionDecoder, GovernanceDecision
from .webhooks import WebhookDispatcher
from .ui import DASHBOARD_HTML
from . import metrics as _m

# ── app + lazy singleton wiring ──────────────────────────────────────────────

app = FastAPI(
    title="AI Governance API",
    version="1.0.0",
    description="Observability and drift detection for AI agents",
)
app.add_middleware(AuthMiddleware)
configure_from_env()


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
    t0 = _time.monotonic()
    try:
        result = svc.ingestion.ingest(req)
    except ValidationError as exc:
        raise HTTPException(422, str(exc))
    _m.ingestion_duration.labels(agent_id=svc.config.agent_id).observe(_time.monotonic() - t0)
    _m.events_ingested.labels(
        agent_id=svc.config.agent_id,
        case_category=body.case_category,
        decision=body.decision,
    ).inc()
    if result.is_high_risk:
        _m.high_risk_events.labels(
            agent_id=svc.config.agent_id, case_category=body.case_category,
        ).inc()
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
    _m.drift_checks.labels(agent_id=svc.config.agent_id).inc()
    _m.drift_score.labels(agent_id=svc.config.agent_id).set(report.overall_drift_score)
    is_safe, _ = svc.engine.is_agent_safe()
    _m.agent_safe.labels(agent_id=svc.config.agent_id).set(1 if is_safe else 0)
    fired = svc.engine.evaluate(report)
    for f in fired:
        _m.alerts_fired.labels(
            agent_id=svc.config.agent_id,
            severity=f.severity,
            rule_name=f.rule_name,
        ).inc()
        if f.triggered_rollback:
            _m.rollbacks_triggered.labels(
                agent_id=svc.config.agent_id, trigger_rule=f.rule_name,
            ).inc()
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


@app.get("/health")
def health_check():
    """Health check endpoint — verifies DB is responsive and returns system stats."""
    svc = _get_svc()
    try:
        counts = svc.db.get_table_counts(svc.config.agent_id)
        is_safe, reason = svc.engine.is_agent_safe()
        valid, broken_seq = svc.audit.verify_chain(svc.config.agent_id)
        return {
            "status": "healthy",
            "agent_id": svc.config.agent_id,
            "config_version": svc.config.version,
            "is_safe": is_safe,
            "audit_chain_valid": valid,
            "table_counts": counts,
        }
    except Exception as exc:
        return {"status": "unhealthy", "error": str(exc)}


@app.get("/config")
def get_config():
    """Return the active governance configuration (read-only)."""
    svc = _get_svc()
    return {
        "agent_id": svc.config.agent_id,
        "version": svc.config.version,
        "min_baseline_events": svc.config.min_baseline_events,
        "recent_window_size": svc.config.recent_window_size,
        "high_risk_patterns": [
            {"name": p.name, "field": p.field, "pattern": p.pattern, "weight": p.weight}
            for p in svc.config.high_risk_patterns
        ],
        "drift_thresholds": [
            {
                "name": t.name, "category": t.category, "metric": t.metric,
                "max_delta": t.max_delta, "direction": t.direction.value,
                "severity": t.severity.value,
            }
            for t in svc.config.drift_thresholds
        ],
        "rollback_conditions": [
            {"name": c.name, "description": c.description, "expression": c.expression}
            for c in svc.config.rollback_conditions
        ],
    }


@app.get("/export/decisions")
def export_decisions(
    category: Optional[str] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    limit: int = 10000,
    format: str = "json",
):
    """Export decisions for compliance/audit. Supports JSON and CSV formats."""
    svc = _get_svc()
    since_dt = datetime.fromisoformat(since.replace(" ", "+")) if since else None
    until_dt = datetime.fromisoformat(until.replace(" ", "+")) if until else None
    records = svc.db.export_decisions(
        svc.config.agent_id,
        category=category,
        since=since_dt,
        until=until_dt,
        limit=min(limit, 50000),
    )
    if format == "csv":
        import csv
        import io
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=[
            "event_id", "timestamp", "case_id", "case_category", "is_high_risk",
            "high_risk_score", "decision", "resolution_time_ms", "ground_truth",
        ])
        writer.writeheader()
        for r in records:
            writer.writerow({
                "event_id": r.event_id,
                "timestamp": r.timestamp.isoformat(),
                "case_id": r.case_id,
                "case_category": r.case_category,
                "is_high_risk": r.is_high_risk,
                "high_risk_score": round(r.high_risk_score, 4),
                "decision": r.decision,
                "resolution_time_ms": r.resolution_time_ms,
                "ground_truth": r.ground_truth or "",
            })
        from fastapi.responses import Response
        return Response(
            content=buf.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=decisions_export.csv"},
        )
    return {
        "count": len(records),
        "records": [
            {
                "event_id": r.event_id,
                "timestamp": r.timestamp.isoformat(),
                "case_id": r.case_id,
                "case_category": r.case_category,
                "is_high_risk": r.is_high_risk,
                "high_risk_score": round(r.high_risk_score, 4),
                "decision": r.decision,
                "resolution_time_ms": r.resolution_time_ms,
                "ground_truth": r.ground_truth,
                "metadata": r.metadata,
            }
            for r in records
        ],
    }


@app.post("/admin/purge")
def purge_old_data(
    decisions_days: int = 90,
    snapshots_days: int = 90,
    alerts_days: int = 180,
):
    """Purge old data to prevent unbounded DB growth. Only deletes acknowledged alerts."""
    svc = _get_svc()
    deleted_decisions = svc.db.purge_old_decisions(svc.config.agent_id, keep_days=decisions_days)
    deleted_snapshots = svc.db.purge_old_snapshots(svc.config.agent_id, keep_days=snapshots_days)
    deleted_alerts = svc.db.purge_old_alerts(svc.config.agent_id, keep_days=alerts_days)
    total = deleted_decisions + deleted_snapshots + deleted_alerts
    if total > 0:
        svc.audit.append(
            svc.config.agent_id, "data.purged", "admin", "system",
            detail={
                "decisions_deleted": deleted_decisions,
                "snapshots_deleted": deleted_snapshots,
                "alerts_deleted": deleted_alerts,
                "retention_days": {
                    "decisions": decisions_days,
                    "snapshots": snapshots_days,
                    "alerts": alerts_days,
                },
            },
        )
    return {
        "decisions_deleted": deleted_decisions,
        "snapshots_deleted": deleted_snapshots,
        "alerts_deleted": deleted_alerts,
    }


@app.post("/admin/reload-config")
def reload_config(body: dict):
    """Hot-reload governance config without restarting the server.

    Accepts a full GovernanceConfig JSON body. Validates before swapping.
    Preserves DB, audit log, and webhook dispatcher across reloads.
    """
    svc = _get_svc()
    old_version = svc.config.version
    old_fingerprint = svc.config.fingerprint
    try:
        new_config = GovernanceConfig.model_validate(body)
    except Exception as exc:
        raise HTTPException(422, f"Invalid config: {exc}")
    svc.config = new_config
    svc.ingestion = IngestionLayer(new_config, svc.db)
    svc.detector = DriftDetector(new_config, svc.db)
    svc.engine = AlertEngine(new_config, svc.db)
    svc.dashboard = DashboardBuilder(new_config, svc.db, svc.detector, svc.engine)
    svc.audit.append(
        new_config.agent_id, "config.reloaded", "admin", "config",
        detail={
            "old_version": old_version,
            "old_fingerprint": old_fingerprint,
            "new_version": new_config.version,
            "new_fingerprint": new_config.fingerprint,
        },
    )
    return {
        "status": "reloaded",
        "old_version": f"{old_version}:{old_fingerprint}",
        "new_version": f"{new_config.version}:{new_config.fingerprint}",
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


@app.get("/ui", include_in_schema=False)
def web_dashboard():
    """PM-facing web dashboard — auto-refreshing browser UI."""
    from starlette.responses import HTMLResponse
    return HTMLResponse(content=DASHBOARD_HTML)


@app.get("/metrics")
def prometheus_metrics():
    """Prometheus metrics endpoint for scraping."""
    from starlette.responses import Response as StarletteResponse
    body, content_type = _m.metrics_response()
    return StarletteResponse(content=body, media_type=content_type)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
