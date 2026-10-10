# 01 – Requirements

## Purpose
Clarify the problem, stakeholders, constraints, acceptance criteria, and non-goals. Produce a clean, reviewable requirements artifact.

## Inputs (load only these)
- This file
- Any ticket / user request the human provides
- tech-stack.md (only if constraints depend on existing stack)
- architecture-principles.md (only if high-level constraints apply)

Do **not** load design, implementation, or review files.

## Process
1. Restate the problem in one clear sentence.
2. List stakeholders and primary users.
3. Capture functional requirements as user stories or bullet criteria.
4. Capture non-functional requirements (performance, security, compatibility, etc.).
5. Explicitly list non-goals / out-of-scope items.
6. Note open questions or assumptions that need human confirmation.
7. Define "done" acceptance criteria that can be tested later.

## Outputs
- `icm/artifacts/[feature]-requirements.md` (or `[ticket]-req-v1.md`)
- Short list of open questions for the human

## Done When
A human can read the requirements file and decide whether to approve moving to design. No code or detailed design yet.
