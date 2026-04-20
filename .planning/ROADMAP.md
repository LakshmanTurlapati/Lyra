# Roadmap: Lyra

## Overview

Lyra delivers a fine-tuned SmolLM2-1.7B model trained on Opus-quality synthetic data across three domains: tool calling, code generation, and general knowledge. The roadmap follows a format-first, evaluate-before-train discipline. The data pipeline and format validation are established first, then the evaluation framework is built so every training run produces actionable signal from day one. Data generation proceeds domain-by-domain (tool calls, code, knowledge), followed by dataset assembly, training, benchmarking, and a two-stage release -- core artifacts first, then community enhancements (GGUF, demo Space, versioning).

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Data Format and Pipeline Foundation** - ShareGPT format specification, validation, prompt template system, and Anthropic Batch API generation client
- [ ] **Phase 2: Data Quality and Curation Pipeline** - Deduplication, format validation, quality scoring, configurable pipeline with adaptive output style support
- [ ] **Phase 3: Evaluation Framework** - Custom eval benchmarks, standard benchmark harness integration, and base model baselines before any training
- [ ] **Phase 4: Tool Calling Dataset** - Generate and curate all tool-call domain training data (JSON function calls, multi-turn, parallel, MCP, CLI)
- [ ] **Phase 5: Code Generation Dataset** - Generate and curate all code domain training data (utilities, file ops, debugging)
- [ ] **Phase 6: General Knowledge Dataset** - Generate and curate all knowledge domain training data (reasoning chains, Q&A, explanations)
- [ ] **Phase 7: Dataset Assembly** - Merge domains into final dataset with stratified train/validation/test splits and 33/33/33 domain balance
- [ ] **Phase 8: Fine-Tuning** - QLoRA training on SmolLM2-1.7B with documented scripts, hyperparameters, and consumer GPU targeting
- [ ] **Phase 9: Benchmarking and Core Release** - Run evaluations, produce comparison report, publish model/dataset cards and weights on HuggingFace under MIT
- [ ] **Phase 10: Community Release Enhancements** - GGUF quantized variants, interactive Gradio demo Space, and versioned dataset releases

## Phase Details

### Phase 1: Data Format and Pipeline Foundation
**Goal**: Users can generate correctly-formatted ShareGPT conversations that are validated against SmolLM2-1.7B's tokenizer and chat template
**Depends on**: Nothing (first phase)
**Requirements**: DATA-01, DATA-02, DATA-06
**Success Criteria** (what must be TRUE):
  1. User can run a generation command that produces ShareGPT-format conversations with correct role ordering (human/gpt/function_call/observation)
  2. User can validate any generated conversation against SmolLM2-1.7B's tokenizer and chat template and get pass/fail with error details
  3. User can browse a prompt template library organized by category (tool call, code, general knowledge) with documented system prompts
  4. User can generate local test samples and validate them end-to-end through the format and tokenizer pipeline (Anthropic API generation deferred to Phase 4-6 per D-05)
**Plans**: 3 plans
Plans:
- [x] 01-01-PLAN.md -- Format specification, Pydantic validation schema, and test infrastructure
- [x] 01-02-PLAN.md -- Tokenizer alignment validation, sample generator, and dataset directory structure
- [x] 01-03-PLAN.md -- Prompt template library for all three domains with system prompts

### Phase 2: Data Quality and Curation Pipeline
**Goal**: Users can filter, deduplicate, score, and configure their data generation pipeline so only high-quality samples proceed to training
**Depends on**: Phase 1
**Requirements**: DATA-03, DATA-04, DATA-05
**Success Criteria** (what must be TRUE):
  1. User can run the curation pipeline on raw JSONL and get a filtered output with deduplication, format validation, and quality scores applied
  2. User can configure prompt templates, topic distributions, and quality thresholds via config files and reuse them across runs
  3. User can observe adaptive output styles in generated data -- terse responses for code tasks, detailed chain-of-thought for reasoning tasks
  4. User can see per-sample quality scores and filter by threshold
**Plans**: 2 plans
Plans:
- [x] 02-01-PLAN.md -- Quality scorer with 4 heuristic signals and n-gram Jaccard deduplication
- [x] 02-02-PLAN.md -- Style validator, pipeline config, and orchestrator wiring all stages

### Phase 3: Evaluation Framework
**Goal**: Users can evaluate any SmolLM2-1.7B checkpoint (base or fine-tuned) against standard and custom benchmarks before any training begins
**Depends on**: Phase 1
**Requirements**: EVAL-03, EVAL-04
**Success Criteria** (what must be TRUE):
  1. User can run custom eval benchmarks that measure tool-call format compliance, argument extraction accuracy, and code correctness
  2. User can see per-category quality metrics reported separately for tool calls, code, and general knowledge
  3. User can run the eval suite against base SmolLM2-1.7B to establish baseline scores that later compare against the fine-tuned model
**Plans**: 2 plans
Plans:
- [x] 03-01-PLAN.md -- Pydantic result schemas, eval config, and compare command
- [x] 03-02-PLAN.md -- Unified eval runner CLI and test suite

### Phase 4: Tool Calling Dataset
**Goal**: Users have a complete, curated tool-calling dataset covering all five tool-call patterns (JSON function calls, multi-turn, parallel, MCP, CLI)
**Depends on**: Phase 2
**Requirements**: TOOL-01, TOOL-02, TOOL-03, TOOL-04, TOOL-05
**Success Criteria** (what must be TRUE):
  1. Dataset contains structured JSON function calling samples in OpenAI-compatible format with correct argument types and return values
  2. Dataset contains multi-turn tool calling conversations with the full function_call -> observation -> response cycle
  3. Dataset contains parallel function execution patterns with multiple tools invoked in a single turn
  4. Dataset contains MCP-style tool use patterns (server discovery, tool listing, invocation, result handling)
  5. Dataset contains CLI/shell command generation patterns for bash, git, and file operations
**Plans**: 4 plans
Plans:
- [x] 04-01-PLAN.md -- Tool schema pool (50-100 schemas) and batch generation script
- [x] 04-02-PLAN.md -- Generate single-call (~1,155) and CLI (~825) category batches
- [x] 04-03-PLAN.md -- Generate multi-turn (~495), parallel (~495), and MCP (~330) category batches
- [x] 04-04-PLAN.md -- Full curation pipeline run and quality verification checkpoint

### Phase 5: Code Generation Dataset
**Goal**: Users have a complete, curated code generation dataset covering utility functions, file operations, and debugging
**Depends on**: Phase 2
**Requirements**: CODE-01, CODE-02, CODE-03
**Success Criteria** (what must be TRUE):
  1. Dataset contains quick utility function generation samples across common programming languages (Python, JavaScript, TypeScript, Go, Rust)
  2. Dataset contains file operation and system manipulation code samples with correct error handling
  3. Dataset contains debugging and code fix samples that identify bugs, explain them, and provide corrected code
**Plans**: 3 plans
Plans:
- [ ] 05-01-PLAN.md -- Code generation script (TDD) with 3 category generators and test suite
- [ ] 05-02-PLAN.md -- Generate all 68 batches: 34 utility, 17 file-ops, 17 debugging (~3,400 raw samples)
- [ ] 05-03-PLAN.md -- Curation pipeline run and quality verification checkpoint

### Phase 6: General Knowledge Dataset
**Goal**: Users have a complete, curated general knowledge dataset covering reasoning, Q&A, and explanations
**Depends on**: Phase 2
**Requirements**: KNOW-01, KNOW-02, KNOW-03
**Success Criteria** (what must be TRUE):
  1. Dataset contains reasoning chain samples with explicit step-by-step chain-of-thought that reaches a conclusion
  2. Dataset contains factual Q&A samples spanning diverse domains (science, history, math, technology, everyday life)
  3. Dataset contains explanation and teaching samples with adaptive detail level matching the complexity of the topic
**Plans**: TBD

### Phase 7: Dataset Assembly
**Goal**: Users can merge all three domain datasets into a single final dataset with correct splits and balanced domain representation
**Depends on**: Phase 4, Phase 5, Phase 6
**Requirements**: DATA-07
**Success Criteria** (what must be TRUE):
  1. Final dataset contains stratified train/validation/test splits (e.g., 90/5/5) with all three focus areas represented proportionally in each split
  2. User can verify the 33/33/33 domain balance across splits with a stats command
  3. Final assembled dataset passes the full validation pipeline from Phase 2 without errors
**Plans**: TBD

### Phase 8: Fine-Tuning
**Goal**: Users can fine-tune SmolLM2-1.7B on the assembled dataset using documented scripts on a consumer GPU
**Depends on**: Phase 7, Phase 3
**Requirements**: TRNG-01, TRNG-02, TRNG-03
**Success Criteria** (what must be TRUE):
  1. User can run end-to-end fine-tuning using TRL SFTTrainer with LoRA/PEFT via a single documented script
  2. Training completes on a consumer GPU (8GB+ VRAM) with documented hyperparameters and expected training time
  3. Fine-tuned SmolLM2-1.7B model weights are produced via QLoRA and saved in a format ready for evaluation and release
**Plans**: TBD

### Phase 9: Benchmarking and Core Release
**Goal**: Users can see how Lyra compares to base SmolLM2-1.7B on standard benchmarks and download the model/dataset from HuggingFace under MIT license
**Depends on**: Phase 8
**Requirements**: EVAL-01, EVAL-02, REL-01, REL-02, REL-03, REL-04
**Success Criteria** (what must be TRUE):
  1. Model is evaluated on standard benchmarks (MMLU, MBPP/HumanEval, BFCL) with pass@1 scores published
  2. A comparison report exists showing base SmolLM2-1.7B vs Lyra fine-tuned model across all metrics
  3. Dataset card on HuggingFace includes description, creation methodology, statistics, and limitations
  4. Model card on HuggingFace includes metadata YAML, usage examples, training parameters, and benchmark results
  5. Model weights are published in safetensors format and MIT license is applied to datasets, model weights, scripts, and eval code
**Plans**: TBD

### Phase 10: Community Release Enhancements
**Goal**: Users can run Lyra locally via GGUF quantization, try it in a browser demo, and track dataset evolution across versions
**Depends on**: Phase 9
**Requirements**: REL-05, REL-06, REL-07
**Success Criteria** (what must be TRUE):
  1. GGUF quantized variants (Q4_K_M, Q8_0) are published and loadable in LM Studio and llama.cpp
  2. An interactive Gradio demo Space on HuggingFace showcases all three capability areas (tool calling, code, knowledge)
  3. Dataset releases are versioned with documented changes and metrics per version
**Plans**: TBD
**UI hint**: yes

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8 -> 9 -> 10
Note: Phases 4, 5, 6 can execute in parallel (all depend on Phase 2, none depend on each other).
Phase 3 depends on Phase 1 (not Phase 2) and can execute in parallel with Phase 2.

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Data Format and Pipeline Foundation | 0/3 | Planning complete | - |
| 2. Data Quality and Curation Pipeline | 0/2 | Planning complete | - |
| 3. Evaluation Framework | 0/2 | Planning complete | - |
| 4. Tool Calling Dataset | 0/4 | Planning complete | - |
| 5. Code Generation Dataset | 0/3 | Planning complete | - |
| 6. General Knowledge Dataset | 0/TBD | Not started | - |
| 7. Dataset Assembly | 0/TBD | Not started | - |
| 8. Fine-Tuning | 0/TBD | Not started | - |
| 9. Benchmarking and Core Release | 0/TBD | Not started | - |
| 10. Community Release Enhancements | 0/TBD | Not started | - |
