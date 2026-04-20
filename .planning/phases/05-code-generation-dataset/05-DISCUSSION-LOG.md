# Phase 5: Code Generation Dataset - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md -- this log preserves the alternatives considered.

**Date:** 2026-04-20
**Phase:** 05-code-generation-dataset
**Areas discussed:** Language distribution, Category distribution, Response style, Reuse Phase 4 infrastructure

---

## Language Distribution

| Option | Description | Selected |
|--------|-------------|----------|
| Python-heavy | ~40% Python, ~25% JS/TS, ~20% Go, ~15% Rust. Aligns with ML/dev audience. | Yes |
| Even split across all | ~20% each for 5 langs. Maximum breadth. | |
| Top 3 only (Python, JS, TS) | Drop Go and Rust. Simpler but limits generality. | |

**User's choice:** Python-heavy
**Notes:** None

---

## Category Distribution

| Option | Description | Selected |
|--------|-------------|----------|
| Utility-heavy | ~50% utility, ~25% file ops, ~25% debugging. Quick helpers is where 1.7B shines. | Yes |
| Even thirds (~33% each) | Balanced but may over-index on debugging. | |
| Debugging-heavy | ~50% debugging. Hardest skill but risky for small model. | |

**User's choice:** Utility-heavy
**Notes:** None

---

## Response Style

| Option | Description | Selected |
|--------|-------------|----------|
| Terse code-first | Code with 1-2 line comment. No preamble. Debugging gets brief "Bug: X, Fix: Y". | Yes |
| Explanation + code | Brief explanation paragraph before code block. | |
| Mixed by category | Different style per category. | |

**User's choice:** Terse code-first
**Notes:** None

---

## Reuse Phase 4 Infrastructure

| Option | Description | Selected |
|--------|-------------|----------|
| New script, same pattern | Create generate_code_data.py with same structure as generate_tool_data.py. | Yes |
| Extend existing script | Add code modes to generate_tool_data.py. | |
| You decide | Claude's discretion on organization. | |

**User's choice:** New script, same pattern
**Notes:** None

---

## Claude's Discretion

- Specific utility function topics beyond template list
- Bug types and debugging scenarios beyond template list
- Whether to include code comments in generated samples
- Test fixtures for generation script
- Batch file naming convention

## Deferred Ideas

None -- discussion stayed within phase scope
