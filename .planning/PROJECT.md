# Lyra

## What This Is

An open-source dataset curation and model fine-tuning project that distills Claude Opus reasoning quality into training data for SmolLM2-1.7B. Lyra aims to produce a small language model that punches far above its weight class -- capable of tool calling, code generation, and general knowledge tasks that typically require much larger models. Named in the musical tradition of Haiku, Sonnet, and Opus -- a small constellation that produces clear, harmonious output.

## Core Value

Curate Opus-quality training data that makes a 1.7B parameter model practically useful for day-to-day development tasks -- tool calls, quick code, and general reasoning.

## Requirements

### Validated

- [x] ShareGPT format specification with Pydantic validation (Validated in Phase 1: Data Format and Pipeline Foundation)
- [x] SmolLM2-1.7B tokenizer alignment validation pipeline (Validated in Phase 1)
- [x] Prompt template library for data generation guidance (Validated in Phase 1)
- [x] Data quality curation pipeline with filtering, dedup, scoring (Validated in Phase 2: Data Quality and Curation Pipeline)
- [x] Configurable pipeline via YAML with per-domain overrides (Validated in Phase 2)
- [x] Adaptive output style validation -- terse for code, detailed for reasoning (Validated in Phase 2)
- [x] Custom eval benchmarks for tool-call format, code correctness, argument accuracy (Validated in Phase 3: Evaluation Framework)
- [x] Per-category quality metrics reported separately for tool calls, code, knowledge (Validated in Phase 3)

### Active

- [ ] ShareGPT-format dataset with tool call training samples (function calling, MCP-style, CLI/shell)
- [ ] ShareGPT-format dataset with code generation samples (quick, efficient, trivial code)
- [ ] ShareGPT-format dataset with general knowledge samples (reasoning, Q&A, explanations)
- [ ] Adaptive output style in training data (terse for code, detailed for reasoning)
- [ ] Dataset generation pipeline powered by Claude Opus
- [ ] Fine-tuned SmolLM2-1.7B model weights published on HuggingFace
- [ ] Training and reproduction scripts for end-to-end fine-tuning
- [ ] Custom evaluation benchmarks for tool-call accuracy, code quality, and knowledge
- [ ] All artifacts released under MIT license on HuggingFace

### Out of Scope

- Multi-model support (other base models beyond SmolLM2-1.7B) -- focus on one model first, expand later
- RLHF/DPO alignment -- pure supervised fine-tuning for v1
- Long-context tasks -- 1.7B models have limited context; optimize for short, practical interactions
- Image/multimodal data -- text-only for v1
- Human annotation pipeline -- Opus generates all data; human curation deferred

## Context

SmolLM2-1.7B is HuggingFace's largest SmolLM variant. Its base instruction-following ability is limited, but its size makes it ideal for local deployment. The hypothesis is that carefully curated, Opus-quality training data can close a significant portion of the capability gap between this tiny model and much larger ones -- especially in structured tasks like tool calling where format consistency matters more than raw parameter count.

The dataset is built iteratively: start with ~5K high-quality samples (evenly split across three focus areas), validate improvement via benchmarks, then scale up based on what works. Each phase produces a versioned dataset release.

Three focus areas, evenly weighted:
- **Tool calls (~33%)**: Function calling (JSON-structured), MCP tool use patterns, CLI/shell command generation
- **Code (~33%)**: Quick utility functions, file operations, trivial but correct code, debugging
- **General knowledge (~33%)**: Reasoning chains, factual Q&A, explanations, everyday queries

## Constraints

- **Base model**: SmolLM2-1.7B -- all fine-tuning targets this architecture
- **Data format**: ShareGPT conversation format -- widely supported, HuggingFace native
- **License**: MIT -- maximum permissiveness for open-source adoption
- **Data source**: Claude Opus generates all training data -- no human annotation in v1
- **Scale strategy**: Start small (~5K samples), measure impact, grow iteratively
- **Token limits**: Must account for SmolLM2-1.7B's context window and generation limits in training data design

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| SmolLM2-1.7B as base model | Largest SmolLM -- best balance of capability and local deployability | -- Pending |
| ShareGPT format | Widely supported, native HuggingFace integration, multi-turn capable | -- Pending |
| Even 33/33/33 split across focus areas | Balanced general-purpose model rather than specialist | -- Pending |
| Opus generates all data | Highest quality source, consistent output, no annotation costs | -- Pending |
| Start with ~5K samples | Validate approach before investing in scale | -- Pending |
| Name: Lyra | Greek constellation, musical lineage (Haiku/Sonnet/Opus), "small but clear" | -- Pending |
| MIT license | Maximum adoption, no commercial restrictions | -- Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? -- Move to Out of Scope with reason
2. Requirements validated? -- Move to Validated with phase reference
3. New requirements emerged? -- Add to Active
4. Decisions to log? -- Add to Key Decisions
5. "What This Is" still accurate? -- Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check -- still the right priority?
3. Audit Out of Scope -- reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-04-20 after Phase 1 completion*
