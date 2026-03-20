# CLAUDE.md

## Development Guidelines

### Ticket Workflow
- Work on one ticket at a time. Do not start a new ticket until the current one is fully complete and a PR has been created.
- Implement exactly what the ticket specifies. Do not add scope, refactor unrelated code, or make improvements beyond what is asked. The only acceptable reason to deviate from the spec is if it is technically impossible or actively causes an error — document any deviation in the PR description with a clear explanation.
- Do not modify ticket requirements. If a spec is ambiguous or appears incorrect, stop and ask before proceeding.
- If a blocker surfaces mid-implementation, stop and ask for help. Do not make assumptions or try workarounds silently.

### Branching
- Name branches after the ticket number (e.g. `TICKET-123`). Do not invent branch names.

### Pull Request Process
When a ticket is complete:
1. Create a PR targeting the appropriate base branch
2. Review the PR diff line by line against the ticket's acceptance criteria
3. Confirm every requirement is addressed before calling the work done
4. The PR description must include what changed, why, and any deviations from the spec with justification

### Commits
- Make small, focused commits — one logical change per commit.
- Use conventional commit format: `feat:`, `fix:`, `test:`, `refactor:`, `chore:`, `docs:`

### Test-Driven Development (TDD)
- TDD is required for all code. Write a failing test first, then write the minimum code to make it pass, then refactor. The only exception is code with no testable logic (e.g. pure configuration or entrypoint bootstrapping).
- Do not write implementation code before the corresponding test exists.
- Do not mock what you can test for real. Only mock external services that cannot run locally (e.g. remote APIs, cloud services). Mocking internal application code in integration tests is not acceptable.

### Dependencies
- Do not add new dependencies without approval. Flag any required new packages to the user before adding them to requirements or pyproject.toml.

### Code Documentation
- Every function must have a docstring explaining what it does, its parameters, and its return value.
- Only add inline comments to explain *why* something non-obvious is done. Do not comment code that is self-evident — commenting obvious code is noise.

### Code Hygiene
- Do not leave commented-out code in the codebase. If code is removed or replaced, delete it. Git history is the safety net.

### README
No ticket is complete without updating the README. Keep it in sync with the current state of the codebase at all times — update setup steps, commands, and feature documentation as part of every task.
