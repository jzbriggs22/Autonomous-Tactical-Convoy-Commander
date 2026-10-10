# PR Description — Initial implementation of Autonomous Tactical Convoy Commander simulator

## Summary

This PR contains two independent deliverables sharing one repository:

1. **`convoy_commander`** — a deterministic multi-vehicle convoy simulator for
   fleet coordination in GPS-denied environments: world model, vehicle
   dynamics, position estimation with innovation gating, lossy comms, leader
   election, task auction, formation control, global/local planning, a
   centralised supervisor, and a batch evaluation harness with plots and
   markdown reports across 11 scenarios.

2. **`ai_governance`** — an observability and governance service for AI
   agents built on the premise that an agent can look healthy on aggregate
   metrics while silently drifting on high-risk cases. It provides drift
   detection, high-risk classification with explanations, alerting with
   automatic rollback, predictive breach forecasting, a ground-truth accuracy
   guard, constrained decoding for structured agent output, and a PM-facing
   dashboard answering "is this agent safe to keep running?"

The packages import nothing from each other; the governance stack installs
via the `governance` extra.

## Motivation

- Convoy coordination research needs a reproducible, seeded simulation with
  explicit safety enforcement and failure-mode scenarios.
- AI agent deployments need governance signals that catch category-level
  drift *before* aggregate metrics degrade, with an auditable trail and a
  hard rollback gate suitable for CI/CD and PM decision-making.

## Changes

- ~36.5k lines across 181 files; full breakdown in the commit history
  (each component landed as a focused commit).
- Convoy simulator packages: `core/`, `vehicles/`, `comms/`, `planning/`,
  `coordination/`, `supervisor/`, `sim/`, `metrics/`, `viz/`.
- Governance packages: ingestion, drift, alerts, escalation, forecast,
  feedback, accuracy guard, structured decoding, audit, policy store,
  replay, readiness, reports, retention, scheduler, storage, API, CLI, UI.
- CI: GitHub Actions matrix on Python 3.11/3.12 with coverage; governance
  unit and behavioral suites gated separately; advisory mypy.
- Review fixes from automated review (all verified with regression tests):
  weather fuel-tank debit, inline ground-truth validation, naive-timestamp
  normalization, Docker test-image scope, missing dependency declarations,
  CI timeout budgets for heavy sim tests.

## Test plan

- `python -m pytest tests/ -q --timeout=120 --ignore=tests/governance`
  — convoy suite (~460 tests), verified on 3.11 and 3.12 with coverage.
- `python -m pytest tests/governance/ -q --timeout=60`
  — governance suite (~640 tests incl. 26 behavioral regression tests on a
  fixed 50-decision dataset), verified on 3.11 and 3.12.
- Both suites verified green locally under the exact CI invocations
  (including coverage instrumentation) before push.

## Rollout / migration notes

- No external services required: SQLite storage, self-hosted FastAPI.
- Governance DB schema is migration-managed (`python -m ai_governance.cli migrate`).
- The Docker image intentionally excludes the governance stack (torch-class
  transitive deps); use `pip install -e ".[dev,governance]"` for governance work.

## Known limitations / follow-ups

- Docstring coverage is ~55% (CodeRabbit advisory threshold is 80%) — a
  mechanical docstring sweep is available on request.
- Weather scenarios have known near-miss tuning work outstanding
  (tracked in README "Limitations and Next Steps").
- `sensor_drift_spike` would benefit from the same landmark-density tuning
  applied to `gps_denied`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01AdDejfGeUnNNgaQY7DLpZT
