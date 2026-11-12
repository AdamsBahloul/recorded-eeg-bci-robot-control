# Context file map

Read in this order:
1. `AGENTS.md` — binding operating rules for Codex.
2. `README.md` — quick start and hardware reality.
3. `docs/PROJECT_BRIEF.md` — what the project is and is not.
4. `docs/ARCHITECTURE.md` — component boundaries and command schema.
5. `docs/ROADMAP.md` — milestone order and gates.
6. `docs/DATA_AND_EVALUATION.md` — dataset, preprocessing and scientific validity.
7. `docs/SAFETY.md` — fail-safe behavior and limitations.
8. `docs/TECH_DECISIONS.md` — default technology choices and what not to add.
9. `CODEX_START_PROMPT.md` — first task prompt.

Token-efficiency rule: Codex should refer to these files by path instead of restating them. For a task, read only the relevant files after the initial project orientation.
