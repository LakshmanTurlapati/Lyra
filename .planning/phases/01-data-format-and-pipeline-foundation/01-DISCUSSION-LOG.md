# Phase 1: Data Format and Pipeline Foundation - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md -- this log preserves the alternatives considered.

**Date:** 2026-04-20
**Phase:** 1-Data Format and Pipeline Foundation
**Areas discussed:** Tool call JSON schema, Token budget per sample, Generation method, Project structure

---

## Tool Call JSON Schema

| Option | Description | Selected |
|--------|-------------|----------|
| OpenAI-compatible (Recommended) | Standard function_call format with name/arguments JSON. Widest ecosystem support. | Yes |
| Hermes/ChatML style | Tool calls as special tokens within ChatML. Used by NousResearch models. | |
| Custom Lyra format | Lyra-specific schema optimized for 1.7B models. | |
| You decide | Claude picks based on SmolLM2 compatibility research. | |

**User's choice:** OpenAI-compatible (Recommended)
**Notes:** None

| Option | Description | Selected |
|--------|-------------|----------|
| Simple first (Recommended) | Start with single function calls. Add complexity in Phase 4. | |
| Full complexity now | Format spec covers single, multi-turn, parallel, and nested from day one. | Yes |
| Progressive | Define simple now, extend schema as needed. | |

**User's choice:** Full complexity now
**Notes:** None

---

## Token Budget Per Sample

| Option | Description | Selected |
|--------|-------------|----------|
| 2048 tokens (Recommended) | Matches SmolLM2's training sequence length. Research strongly recommends. | Yes |
| 4096 tokens | Double native training length. Risks quality degradation. | |
| Mixed lengths | Most under 2048, allow up to 4096 for complex scenarios. | |

**User's choice:** 2048 tokens (Recommended)
**Notes:** None

| Option | Description | Selected |
|--------|-------------|----------|
| Naturally varied (Recommended) | Let conversations be as long as they need within cap. | Yes |
| Target medium | Aim for 800-1500 tokens per sample. | |
| Skew short | Most samples 200-800 tokens. | |

**User's choice:** Naturally varied (Recommended)
**Notes:** None

---

## Generation Method

| Option | Description | Selected |
|--------|-------------|----------|
| Batch API (Recommended) | Anthropic Message Batches API -- 50% cost savings. | |
| Single calls first | Individual API calls for iteration speed. | |
| Both from start | Single calls for dev, batch for bulk. | |

**User's choice:** Other -- "you do it... we are in claude code now... we use what we have at our disposal"
**Notes:** User clarified that data generation happens directly in Claude Code sessions. No external API pipeline needed. Claude Opus writes samples as files.

| Option | Description | Selected |
|--------|-------------|----------|
| One sample per call (Recommended) | Each API call generates one conversation. | |
| Multiple per call | Each call generates 3-5 conversations. | |
| You decide | Claude picks best approach. | |

**User's choice:** Other -- "not an api call"
**Notes:** Reinforced that there are no API calls. Generation is direct file writing in Claude Code.

**Follow-up confirmation:**

| Option | Description | Selected |
|--------|-------------|----------|
| Yes, exactly | Claude Code writes raw data, Python scripts validate/filter. | Yes |
| All in Claude Code | No separate Python pipeline needed. | |
| Let me explain | Different workflow in mind. | |

**User's choice:** Yes, exactly
**Notes:** Confirmed workflow: Claude Code generates JSONL -> Python scripts validate, dedup, filter

---

## Project Structure

| Option | Description | Selected |
|--------|-------------|----------|
| Python (Recommended) | Native HuggingFace/TRL/Unsloth ecosystem. | Yes |
| Python + shell scripts | Python core + shell wrappers. | |
| You decide | Claude picks. | |

**User's choice:** Python (Recommended)
**Notes:** None

| Option | Description | Selected |
|--------|-------------|----------|
| Flat scripts (Recommended) | Top-level scripts/ dir. Simple, no package overhead. | Yes |
| Python package | Installable lyra/ package with pyproject.toml. | |
| Monorepo with subdirs | Separate dirs per concern. | |

**User's choice:** Flat scripts (Recommended)
**Notes:** None

| Option | Description | Selected |
|--------|-------------|----------|
| data/ directory | data/raw/, data/curated/, data/final/. Gitignored. | |
| datasets/ directory | HuggingFace convention. Domain separation. | Yes |
| You decide | Claude picks. | |

**User's choice:** datasets/ directory
**Notes:** None

---

## Claude's Discretion

- Exact ShareGPT JSON field structure (following OpenAI conventions)
- Python script naming and organization
- Validation error message format
- Prompt template file format

## Deferred Ideas

None
