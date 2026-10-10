# Architecture Principles

## High-level style
- Modular monolith: two independent packages in one repo —
  `convoy_commander` (simulation) and `ai_governance` (observability/governance service).
  They share nothing but the repository; the governance stack is an optional extra.
- Key patterns we prefer:
  - Config-driven behavior: one validated pydantic config object per subsystem
    (`SimConfig`, `GovernanceConfig`) with cross-field validators and safe defaults.
  - Deterministic simulation: seeded RNG everywhere; same config + seed = same run.
  - Layered sim loop: world events → sensing/estimation → comms → coordination →
    planning → actuation → safety enforcement → metrics.
  - Storage as the integration point: governance subsystems (drift, alerts, feedback,
    forecast) communicate through `GovernanceDB`, not through each other.
  - Advisory vs. enforcing subsystems: predictions and webhooks may fail without
    breaking the enforcing path (drift detection, rollback gates) — isolate with
    catch-log-continue and say so in a comment.
- Boundaries we protect:
  - `core/` (physics, world, config) has no dependency on `sim/`, `coordination/`, or `viz/`.
  - `ai_governance` never imports from `convoy_commander` (and vice versa).
  - The API layer (`api.py`) wires services; domain modules never import FastAPI.

## Decision guidelines
- When to add a new dependency: only when it removes substantial code or risk, is
  well-maintained, and fits an existing extra (`dev` / `geo` / `weather` / `governance`).
  Anything pulling large transitive deps (torch-class) stays out of the default install
  and out of the Docker test image.
- When to create a new module: when a concern has its own data model + lifecycle
  (e.g. `forecast.py`, `feedback.py`, `accuracy_guard.py` are separate modules, not
  additions to `drift.py`). Register public names in `__init__.py` `__all__`.
- Data ownership rules: `GovernanceDB` owns all persistence and timestamp parsing;
  domain modules never touch sqlite directly (tests may, for seeding). Records are
  dataclasses owned by `storage.py`.

## Non-negotiables
- Timezone-aware UTC datetimes at every boundary; naive inputs are normalized, never propagated.
- All external input validated at ingestion (`ValidationError` with context).
- No `eval` on user expressions — `safe_eval` (AST allowlist) only.
- Rollback/safety gates must fail closed; advisory features must fail open.
- Both packages keep CI green on 3.11 and 3.12; behavior changes ship with tests
  including failure states.
