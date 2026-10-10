# Coding Standards

These standards apply to all code produced or modified in this workspace.
Claude must follow these rules unless the human explicitly overrides them for a specific task.

## 1. General Principles
- Prefer clarity over cleverness.
- Code is read far more often than it is written.
- Make the happy path obvious; make error paths explicit.
- Small, focused units of work > large multi-purpose functions/classes.
- Leave the codebase better than you found it (within the scope of the current change).

## 2. Naming Conventions
- **Files & folders**: snake_case (`multi_runner.py`, `accuracy_guard.py`).
- **Functions & methods**: verb + noun, descriptive (`get_recent_decisions`, `compute_accuracy`, `has_reached_destination`).
- **Variables**: descriptive nouns. Avoid single letters except in very short loops (`i`, `j`) and math-heavy kinematics code where symbols mirror the equations.
- **Constants**: SCREAMING_SNAKE_CASE (`VALID_DECISIONS`, `NEAR_GOAL_DIST`).
- **Booleans**: prefix with `is`, `has`, `can`, `should` (`is_high_risk`, `has_active_rollback`).
- **Private helpers**: leading underscore (`_parse_ts`, `_classify_risk`).
- **Avoid**: abbreviations that are not universally known, and names that require a comment to understand.

## 3. Function & Module Design
- One function = one responsibility.
- Prefer pure functions when possible; sim state mutation stays inside the owning class.
- Keep functions short (soft limit ~40–50 lines). Sim step loops are the accepted exception, but extract phases into helpers when they grow.
- Prefer early returns over deep nesting.
- Limit parameters. More than 3–4 positional parameters is a signal to use keyword-only args (`*,`) or a config object.
- Avoid side effects that are not obvious from the name.

## 4. Error Handling
- Fail fast and fail loud.
- Never swallow errors silently. Advisory subsystems (forecasting, webhooks) may catch-and-log, but must log with `logger.exception` and say why continuing is safe.
- Use custom exception types at boundaries (`ValidationError`, `DecodeError`, `SafeExprError`).
- Validate inputs at system boundaries (API handlers, CLI entry points, ingestion).
- Catch the narrowest exception that covers the failure (`except (SafeExprError, ArithmeticError)`, never bare `except Exception` unless isolating an advisory step).
- Always include useful context in error messages (what failed + relevant identifiers).

## 5. Logging
- Use the module-level `logger = logging.getLogger(__name__)` pattern.
- Log at appropriate levels: debug / info / warn / error.
- Never log secrets, tokens, or API keys.
- Include agent/correlation IDs when available (the API layer provides correlation middleware).
- Prefer clear, searchable messages over clever ones.

## 6. Comments & Documentation
- Prefer self-documenting code over comments.
- Comments explain *why*, not *what* — especially constraints the code cannot show (timeout budgets, ordering requirements, invariants).
- Module docstrings state purpose and invariants; dataclasses document their invariants (`0 <= fuel <= capacity`).
- Keep comments up to date — outdated comments are worse than no comments.
- TODO comments must include a name or ticket reference.

## 7. Testing Expectations
- New behavior ships with tests, including failure states.
- Cover the happy path + important edge cases + error cases.
- Unit tests for pure logic; API tests through `TestClient`; behavioral regression tests on fixed datasets.
- Avoid testing implementation details — test observable behavior.
- Heavy sim integration tests carry explicit `@pytest.mark.timeout(...)` budgets with a comment explaining why.
- Never trust a piped pytest exit code — check `PIPESTATUS[0]`, not the tail of the pipe.

## 8. Security Basics
- Never commit secrets. Use environment variables.
- Validate and sanitize all external input at ingestion.
- Use parameterized queries — never string-concatenate SQL.
- No `eval`/`exec` on untrusted input — use the AST-based `safe_eval` module for user expressions.
- Follow the principle of least privilege.

## 9. Code Organization
- Group related code together; one subsystem per package (`core/`, `sim/`, `coordination/`, `ai_governance/`).
- Keep modules focused. If a file is doing many unrelated things, split it.
- Public API surface is defined by explicit `__init__.py` re-exports with `__all__`.
- Delete dead code. Do not leave large commented-out blocks.

## 10. Git & PR Hygiene
- Small, focused commits — one component per commit.
- Commit messages explain *why*, not just *what*.
- PRs include a clear description, test plan, and any migration notes.

## 11. Explicitly Avoid
- Magic numbers (extract to named constants or config fields).
- Deeply nested conditionals (prefer early returns or guard clauses).
- God classes / god modules.
- Copy-paste with slight variations (extract the common part).
- Premature optimization.
- Catching generic `Exception` and ignoring it.
- Hard-coded credentials or environment-specific values in source.

## Language-Specific Notes
**Primary language for this workspace**: Python 3.11+ (CI gates 3.11 and 3.12)

- Type hints on public APIs; `from __future__ import annotations` at module top.
- `dataclasses` for plain records; pydantic v2 (`BaseModel`, `Field`, validators) for validated config and API contracts.
- Keyword-only arguments (`*,`) for optional constructor knobs.
- Timezone-aware `datetime.now(timezone.utc)` everywhere; normalize naive timestamps to UTC at boundaries.
- Thread safety via `threading.RLock`/`Lock` around shared state; SQLite in WAL mode.
- pytest with `pytest-timeout` and `pytest-cov`; mypy runs advisory in CI.
