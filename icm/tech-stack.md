# Tech Stack

## Core
- Language / runtime: Python >= 3.11 (CI matrix: 3.11 and 3.12)
- Core libraries (`convoy_commander`): numpy, scipy, networkx, matplotlib, pydantic v2
- Governance stack (`ai_governance`, optional extra `governance`): FastAPI, uvicorn, pyyaml,
  outlines / outlines_core, instructor, opentelemetry-api/sdk, prometheus-client, rich
- Database / storage: SQLite (WAL mode, thread-safe access via locks); no external DB
- CLI entry points: `convoy-commander` (convoy sim), `python -m ai_governance.cli` (governance)

## Optional extras (pyproject)
- `dev`: pytest, pytest-timeout, pytest-cov, mypy, httpx
- `geo`: rasterio, osmnx, pyproj (needs system GDAL)
- `weather`: requests
- `governance`: the governance stack above

## Infrastructure
- Hosting / deployment: library + CLI + self-hosted FastAPI service; no cloud dependency
- CI / CD: GitHub Actions (`.github/workflows/ci.yml`) — pytest on 3.11 + 3.12 with coverage,
  governance suites gated separately, mypy advisory (`continue-on-error`)
- Docker: slim test image for the convoy suite only (governance stack deliberately excluded —
  the `governance` extra pulls torch via outlines)
- Monitoring: Prometheus metrics endpoint + OpenTelemetry tracing in the governance API

## Constraints
- Versions that must be respected: `requires-python >= 3.11`; pydantic v2 API only (no v1 idioms)
- Compatibility requirements: both 3.11 and 3.12 must stay green in CI
- Per-test timeout budgets: CI runs `--timeout=120` for the convoy suite; the heaviest sim
  integration tests carry `@pytest.mark.timeout(600)` — new long-running tests must budget explicitly
- Determinism: sim runs are seeded (`np.random.default_rng(seed)`); tests must not depend on
  wall-clock or unseeded randomness
