# Developer ICM Workspace — Project Instructions

This repository operates as an Interpretable Context Methodology (ICM) workspace.
The filesystem is the orchestration layer. Claude becomes the specialist required
by the current stage only.

## Core Rules
- One stage, one job.
- Only load the context listed in the active stage's Inputs table.
- Never dump the entire repository into context.
- Every output is an editable artifact a human can review before the next stage.
- Layer 3 (reference / factory) is stable. Layer 4 (working artifacts) changes every run.

## Active Stage Tracking
At the start of every response write exactly one line:
**Active stage: [01-requirements | 02-design | 03-implement | 04-review | 05-docs | None]**

If no stage is declared and the task is ambiguous, ask which stage to use.

## Available Stages (Layer 2 — `icm/`)
- `icm/01-requirements.md` → clarify problem, acceptance criteria, constraints
- `icm/02-design.md` → architecture, interfaces, data model, decisions
- `icm/03-implement.md` → write or modify code
- `icm/04-review.md` → tests, critique, quality, security
- `icm/05-docs.md` → documentation, changelog, PR/handoff notes

## How to Switch or Start
When the user says "start 02-design", "move to implement", "review this", etc.:
1. Set the active stage.
2. Read only that stage's .md file.
3. Follow its Inputs table strictly.
4. Produce the defined Outputs.

## Cross-Stage Handoffs
When moving work forward (e.g. "take the approved design and start implementation"):
- Locate the latest output from the previous stage by naming convention in `icm/artifacts/`.
- Switch to the new stage context.
- Load only what the new Inputs table requires.

## Naming Conventions (Layer 4 artifacts — `icm/artifacts/`)
- Requirements: `[feature]-requirements.md` or `[ticket]-req-v1.md`
- Design: `[feature]-design.md` or `[feature]-adr.md`
- Implementation notes: `[feature]-impl-notes.md`
- Review: `[feature]-review.md` or `[feature]-test-plan.md`
- Docs: `[feature]-docs.md`, `CHANGELOG-entry.md`, `PR-description.md`
- Versions: append `-v2`, `-v3` when iterating inside a stage.

## Layer 3 Reference Material (the factory — `icm/`)
These files are stable and loaded only when a stage's Inputs table lists them:
- `icm/coding-standards.md`
- `icm/tech-stack.md`
- `icm/architecture-principles.md`

## Token Discipline
- Prefer the smallest useful context.
- If a file is not listed in the current Inputs table, do not read it.
- When the user pastes a large document, ask which stage it belongs to before processing.

## My Defaults
- Primary language(s): Python 3.11+ (3.11 and 3.12 are both CI-gated)
- Preferred frameworks / stack: see `icm/tech-stack.md` (pydantic v2, FastAPI, numpy/scipy, pytest)
- Code style preferences: PEP 8, snake_case, type hints on public APIs, dataclasses for records; see `icm/coding-standards.md`
- Testing philosophy: every behavior change ships with tests, including failure states; heavy sim integration tests carry explicit `@pytest.mark.timeout` budgets
- Documentation style: module-level docstrings stating purpose and invariants; Markdown for reports and ADRs
