# Changelog Entry

## [0.1.0] — Unreleased

### Added — Convoy Simulator (`convoy_commander`)
- Deterministic multi-vehicle convoy simulation: world generation (roads,
  obstacles, spoof regions), kinematics + fuel model, position estimation with
  innovation gating, range/loss/latency comms network, Bully leader election,
  CBBA-lite task auction, formation control, multi-objective A* global planning,
  DWA-lite local planning, centralised supervisor, metrics/plots/markdown reports.
- 11 scenarios including GPS-denied, GPS spoofing, degraded comms, leader
  failure, sensor drift spike, weather (heavy rain / winter storm), and
  multi-convoy merge/split.
- Batch evaluation harness (`convoy-commander evaluate`) with reproducibility
  stamping.

### Added — AI Governance Stack (`ai_governance`)
- Drift detection over 10 behavioral metrics per case category with
  baseline/recent-window comparison and configurable thresholds.
- High-risk case classification via weighted regex patterns, with per-event
  explanations (`/events/{id}/explain`).
- Alert engine with escalation chains and automatic rollback records;
  safe AST-evaluated rollback conditions; `/status` safety gate.
- Predictive drift forecasting (`/forecast`) with scheduler-integrated webhook
  alerts and re-alert cooldown.
- Ground-truth feedback pipeline: bulk labels, accuracy reports, and an
  accuracy guard that triggers alerts/rollbacks on accuracy floor breaches.
- Constrained decoding path for structured `GovernanceDecision` ingestion.
- Operations: PM web dashboard (`/ui`), Prometheus metrics, OpenTelemetry
  tracing, HMAC-signed webhooks, hash-chain audit log, policy version store,
  retention, schema migrations, API-key auth + rate limiting, compliance
  reports, event replay for config impact analysis, deployment readiness gate.
- CLI: `status`, `dashboard`, `drift`, `alerts`, `export`, `config`, `test`,
  `migrate`, `report`, `forecast`, `accuracy`, `serve`.

### Fixed
- Weather fuel multipliers now debit the actual fuel tank, not just the
  reported consumption (cold/wind previously never drained fuel faster).
- Inline `ground_truth` labels on ingestion are validated against the decision
  vocabulary (previously arbitrary strings could corrupt accuracy metrics).
- Naive ISO timestamps are normalized to UTC at ingestion and on storage
  reads, preventing `TypeError` in dashboard/drift windowing.
- CI Python 3.12 job: heavy sim integration tests carry explicit 600s timeout
  budgets; convoy suite default per-test timeout raised to 120s to absorb
  coverage instrumentation overhead.
- `governance` extra now declares all runtime dependencies
  (opentelemetry, prometheus-client, rich; httpx in `dev`).
- Docker test image default command skips governance tests (the governance
  stack is deliberately excluded from the slim image).

### Infrastructure
- GitHub Actions CI on Python 3.11 + 3.12 with coverage; separately gated
  governance unit and behavioral suites; advisory mypy.
- ICM staged-development workspace (`CLAUDE.md` + `icm/`).
