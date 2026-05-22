"""CLI for AI governance operations.

Usage:
    python -m ai_governance.cli status          # safety status
    python -m ai_governance.cli dashboard       # terminal dashboard
    python -m ai_governance.cli drift           # run drift detection
    python -m ai_governance.cli alerts          # list recent alerts
    python -m ai_governance.cli export [--csv]  # export decisions
    python -m ai_governance.cli config          # show active config
    python -m ai_governance.cli test            # run behavioral tests
    python -m ai_governance.cli migrate         # show migration status
    python -m ai_governance.cli serve           # start HTTP server
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .config import GovernanceConfig
from .storage import GovernanceDB


def _get_db_path() -> str:
    return os.environ.get("GOVERNANCE_DB", "governance.db")


def _get_config() -> GovernanceConfig:
    cfg_path = os.environ.get("GOVERNANCE_CONFIG")
    if cfg_path:
        p = Path(cfg_path)
        if p.suffix in (".yml", ".yaml"):
            return GovernanceConfig.from_yaml(p)
        return GovernanceConfig.from_json(p)
    return GovernanceConfig.default_customer_service()


def _get_services():
    from .alerts import AlertEngine
    from .audit import AuditLog
    from .dashboard import DashboardBuilder, TerminalDashboard
    from .drift import DriftDetector

    cfg = _get_config()
    db = GovernanceDB(_get_db_path())
    det = DriftDetector(cfg, db)
    eng = AlertEngine(cfg, db)
    audit = AuditLog(db)
    dash_builder = DashboardBuilder(cfg, db, det, eng)
    return cfg, db, det, eng, audit, dash_builder


def cmd_status(_args) -> int:
    cfg, db, det, eng, audit, _ = _get_services()
    is_safe, reason = eng.is_agent_safe()
    total = db.count_decisions(cfg.agent_id)
    counts = db.get_table_counts(cfg.agent_id)
    valid, broken = audit.verify_chain(cfg.agent_id)

    symbol = "\033[92m OK\033[0m" if is_safe else "\033[91m UNSAFE\033[0m"
    print(f"Agent: {cfg.agent_id}")
    print(f"Status:{symbol}")
    print(f"Reason: {reason}")
    print(f"Total decisions: {total}")
    print(f"Audit chain: {'valid' if valid else f'BROKEN at seq {broken}'}")
    for table, count in counts.items():
        print(f"  {table}: {count}")
    return 0 if is_safe else 1


def cmd_dashboard(_args) -> int:
    from .dashboard import TerminalDashboard
    _, _, _, _, _, dash_builder = _get_services()
    snap = dash_builder.build()
    renderer = TerminalDashboard()
    renderer.render(snap)
    return 0


def cmd_drift(_args) -> int:
    cfg, db, det, eng, _, _ = _get_services()
    report = det.detect()
    fired = eng.evaluate(report)

    print(f"Agent: {cfg.agent_id}")
    print(f"Overall drift score: {report.overall_drift_score:.4f}")
    print(f"Categories analyzed: {', '.join(report.categories_analyzed) or '(none)'}")
    print(f"Violations: {len(report.violations)}")
    for v in report.violations:
        print(f"  [{v.severity.upper()}] {v.rule_name}: {v.metric} "
              f"delta={v.delta:+.4f} (max={v.max_allowed:.4f}) in '{v.category}'")
    if fired:
        print(f"\nAlerts fired: {len(fired)}")
        for f in fired:
            print(f"  [{f.severity}] {f.rule_name}"
                  f"{' (ROLLBACK)' if f.triggered_rollback else ''}")
    return 1 if report.violations else 0


def cmd_alerts(_args) -> int:
    cfg, db, _, _, _, _ = _get_services()
    alerts = db.get_recent_alerts(cfg.agent_id, limit=20)
    if not alerts:
        print("No alerts.")
        return 0
    for a in alerts:
        ack = " [ACK]" if a.acknowledged else ""
        print(f"  [{a.severity.upper()}{ack}] {a.timestamp:%Y-%m-%d %H:%M} "
              f"{a.rule_name}: {a.message[:100]}")
    return 0


def cmd_export(args) -> int:
    cfg, db, _, _, _, _ = _get_services()
    recs = db.export_decisions(cfg.agent_id, limit=args.limit)
    if args.csv:
        import csv
        import io
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=[
            "event_id", "timestamp", "case_id", "case_category", "is_high_risk",
            "high_risk_score", "decision", "resolution_time_ms", "ground_truth",
        ])
        writer.writeheader()
        for r in recs:
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
        print(buf.getvalue())
    else:
        for r in recs:
            print(json.dumps({
                "event_id": r.event_id,
                "timestamp": r.timestamp.isoformat(),
                "case_category": r.case_category,
                "decision": r.decision,
                "is_high_risk": r.is_high_risk,
            }))
    print(f"\n# {len(recs)} records exported", file=sys.stderr)
    return 0


def cmd_config(_args) -> int:
    cfg = _get_config()
    print(cfg.to_json())
    return 0


def cmd_test(_args) -> int:
    import subprocess
    test_path = Path(__file__).resolve().parent.parent / "tests" / "governance" / "test_behavioral.py"
    if not test_path.exists():
        print(f"Behavioral test file not found: {test_path}", file=sys.stderr)
        return 1
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_path), "-v", "--timeout=60", "--tb=short"],
        capture_output=False,
    )
    return result.returncode


def cmd_migrate(_args) -> int:
    from .migrations import migration_status, current_version
    db = GovernanceDB(_get_db_path())
    with db._lock:
        version = current_version(db._conn)
        status = migration_status(db._conn)
    print(f"Schema version: {version}")
    for m in status:
        icon = "+" if m["status"] == "applied" else "-"
        applied = f" (applied {m['applied_at']})" if m["applied_at"] else ""
        print(f"  [{icon}] v{m['version']}: {m['description']}{applied}")
    return 0


def cmd_report(args) -> int:
    from .alerts import AlertEngine
    from .audit import AuditLog
    from .drift import DriftDetector
    from .reports import ReportGenerator

    cfg, db, det, eng, audit, _ = _get_services()
    gen = ReportGenerator(cfg, db, det, eng, audit)
    report = gen.generate()
    if args.json:
        print(report.to_json())
    else:
        print(report.to_text())
    return 0


def cmd_serve(_args) -> int:
    import uvicorn
    from .api import app
    uvicorn.run(
        app,
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8080")),
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ai-governance",
        description="AI agent observability and governance CLI",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("status", help="Check agent safety status")
    sub.add_parser("dashboard", help="Render terminal dashboard")
    sub.add_parser("drift", help="Run drift detection")
    sub.add_parser("alerts", help="List recent alerts")

    export_p = sub.add_parser("export", help="Export decisions")
    export_p.add_argument("--csv", action="store_true", help="CSV format")
    export_p.add_argument("--limit", type=int, default=10000)

    sub.add_parser("config", help="Show active governance config")
    sub.add_parser("test", help="Run behavioral regression tests")
    sub.add_parser("migrate", help="Show schema migration status")

    report_p = sub.add_parser("report", help="Generate compliance report")
    report_p.add_argument("--json", action="store_true", help="JSON format")

    sub.add_parser("serve", help="Start HTTP API server")

    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0

    commands = {
        "status": cmd_status,
        "dashboard": cmd_dashboard,
        "drift": cmd_drift,
        "alerts": cmd_alerts,
        "export": cmd_export,
        "config": cmd_config,
        "test": cmd_test,
        "migrate": cmd_migrate,
        "report": cmd_report,
        "serve": cmd_serve,
    }
    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
