# 02 – Design

## Purpose
Turn approved requirements into a concrete technical design: architecture, interfaces, data model, key decisions, and edge cases.

## Inputs (load only these)
- This file
- The latest approved `[feature]-requirements.md` (Layer 4, in `icm/artifacts/`)
- architecture-principles.md
- tech-stack.md
- coding-standards.md (only if it affects design decisions)

Do **not** load implementation or review files yet.

## Process
1. Summarize the approved requirements in 2–3 sentences.
2. Propose high-level architecture / component breakdown.
3. Define key interfaces, data models, and contracts.
4. Call out important technical decisions and trade-offs (lightweight ADRs).
5. List edge cases, failure modes, and how they will be handled.
6. Note any new dependencies or infrastructure needs.
7. Identify risks and open design questions for the human.

## Outputs
- `icm/artifacts/[feature]-design.md`
- Optional: `icm/artifacts/[feature]-adr.md` for major decisions

## Done When
A human can review the design and approve it before any significant code is written.
