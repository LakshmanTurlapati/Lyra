# Phase 1: Data Format and Pipeline Foundation - Context

**Gathered:** 2026-04-20
**Status:** Ready for planning

<domain>
## Phase Boundary

Establish the ShareGPT data format specification (including OpenAI-compatible tool call schema at full complexity), build Python validation scripts against SmolLM2-1.7B's tokenizer, and create a prompt template library. This phase produces the data contract that ALL downstream generation (Phases 4-6) depends on. No actual training data is generated here -- only the format, validation, and templates.

</domain>

<decisions>
## Implementation Decisions

### Tool Call JSON Schema
- **D-01:** Use OpenAI-compatible function calling format within ShareGPT conversations. Standard `function_call` role with `name` and `arguments` JSON, `observation` role for tool results.
- **D-02:** Spec covers full complexity from day one -- single function calls, multi-turn with results, parallel execution, MCP-style patterns, and CLI/shell commands. All patterns defined in the format spec before any data generation begins.

### Token Budget
- **D-03:** Maximum 2048 tokens per training conversation. This matches SmolLM2-1.7B's native training sequence length and optimizes for the short, practical interactions the model will excel at.
- **D-04:** Natural length distribution within the 2048 cap. Samples range organically from ~200 to ~1800 tokens based on task complexity. No artificial length targeting.

### Generation Method
- **D-05:** Data generation happens directly in Claude Code sessions. Claude Opus writes training samples as JSONL files. No Anthropic API SDK or external API pipeline in the project.
- **D-06:** Python scripts handle all post-generation processing: format validation, tokenizer alignment checks, deduplication, and quality scoring. The pipeline is: Claude Code generates -> Python validates/filters.

### Project Structure
- **D-07:** Python is the primary language. All scripts, validation, training, and evaluation code in Python.
- **D-08:** Flat scripts/ directory with standalone Python files. No package installation overhead -- simple, iterative research structure.
- **D-09:** Data stored in datasets/ directory with domain separation: datasets/tool-calling/, datasets/code/, datasets/knowledge/. Raw generated data and curated filtered data both live here. Gitignored for large files.

### Claude's Discretion
- Exact ShareGPT JSON field names and nesting structure (following OpenAI-compatible conventions)
- Python script naming and internal organization within scripts/
- Validation error message format and logging approach
- Prompt template file format (YAML, JSON, or Markdown)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Data Format
- `.planning/research/STACK.md` -- SmolLM2-1.7B architecture constraints, 8192 context/2048 training seq length
- `.planning/research/FEATURES.md` -- ShareGPT format requirements, tool call conversation ordering
- `.planning/research/ARCHITECTURE.md` -- Pipeline component boundaries and data flow

### Pitfalls
- `.planning/research/PITFALLS.md` -- Tokenizer/chat-template alignment is silent total-loss failure; format must be verified on tokenized samples before any training

### Project
- `.planning/PROJECT.md` -- Core value, constraints, key decisions
- `.planning/REQUIREMENTS.md` -- DATA-01, DATA-02, DATA-06 are this phase's requirements

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- None -- greenfield project, no existing code

### Established Patterns
- None -- first phase establishes all patterns

### Integration Points
- Format spec produced here is consumed by Phase 2 (quality pipeline) and Phases 4-6 (data generation)
- Validation scripts produced here are reused in Phase 2 and Phase 7 (dataset assembly)
- Prompt templates produced here guide Claude Code data generation in Phases 4-6

</code_context>

<specifics>
## Specific Ideas

- OpenAI-compatible tool calling is chosen for maximum ecosystem compatibility (TRL, Unsloth, vLLM all expect this format)
- The 2048 token limit is a hard constraint from SmolLM2's training configuration, not arbitrary
- Claude Code as the generation tool means no API client code, no batch processing infrastructure, no SDK dependencies for data generation
- Prompt templates serve as documentation for HOW to generate data in Claude Code sessions, not as programmatic API inputs

</specifics>

<deferred>
## Deferred Ideas

None -- discussion stayed within phase scope

</deferred>

---

*Phase: 01-data-format-and-pipeline-foundation*
*Context gathered: 2026-04-20*
