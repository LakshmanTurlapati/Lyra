# Requirements: Lyra

**Defined:** 2026-04-20
**Core Value:** Curate Opus-quality training data that makes a 1.7B parameter model practically useful for day-to-day development tasks -- tool calls, quick code, and general reasoning.

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### Data Pipeline

- [ ] **DATA-01**: User can generate ShareGPT-format conversation datasets with strict role ordering (human/gpt/function_call/observation)
- [ ] **DATA-02**: User can validate conversation format alignment with SmolLM2-1.7B tokenizer and chat template
- [ ] **DATA-03**: User can filter generated data through deduplication, format validation, and quality scoring pipeline
- [ ] **DATA-04**: User can configure and reuse the generation pipeline with custom prompt templates, topic distributions, and quality thresholds
- [ ] **DATA-05**: Training data includes adaptive output styles -- terse responses for code tasks, detailed chain-of-thought for reasoning tasks
- [ ] **DATA-06**: Prompt template library organized by category (tool call, code, general knowledge) with documented system prompts
- [ ] **DATA-07**: Dataset includes stratified train/validation/test splits across all three focus areas

### Tool Calling

- [ ] **TOOL-01**: Dataset includes structured JSON function calling samples in OpenAI-compatible format
- [ ] **TOOL-02**: Dataset includes multi-turn tool calling with results handling (function_call -> observation -> response)
- [ ] **TOOL-03**: Dataset includes parallel function execution patterns (multiple tools in one turn)
- [ ] **TOOL-04**: Dataset includes MCP-style tool use patterns (server discovery, tool listing, invocation, result handling)
- [ ] **TOOL-05**: Dataset includes CLI/shell command generation patterns (bash, git, file operations)

### Code Generation

- [ ] **CODE-01**: Dataset includes quick utility function generation samples across common languages
- [ ] **CODE-02**: Dataset includes file operation and system manipulation code samples
- [ ] **CODE-03**: Dataset includes debugging and code fix samples (identify bug, explain, fix)

### General Knowledge

- [ ] **KNOW-01**: Dataset includes reasoning chain samples with explicit chain-of-thought
- [ ] **KNOW-02**: Dataset includes factual Q&A samples across diverse domains
- [ ] **KNOW-03**: Dataset includes explanation and teaching samples with adaptive detail level

### Training

- [ ] **TRNG-01**: User can run end-to-end fine-tuning using TRL SFTTrainer + LoRA/PEFT with documented scripts
- [ ] **TRNG-02**: Training targets consumer GPU (8GB+ VRAM) with documented hyperparameters and expected training time
- [ ] **TRNG-03**: Fine-tuned SmolLM2-1.7B model weights produced via QLoRA

### Evaluation

- [ ] **EVAL-01**: Model evaluated on standard benchmarks (MMLU, MBPP/HumanEval, BFCL) with pass@1 scores
- [ ] **EVAL-02**: Published comparison report of base SmolLM2-1.7B vs Lyra fine-tuned model
- [ ] **EVAL-03**: Custom eval benchmarks measuring tool-call format compliance, argument extraction accuracy, and code correctness
- [ ] **EVAL-04**: Per-category quality metrics reported separately for tool calls, code, and general knowledge

### Release

- [ ] **REL-01**: Dataset card on HuggingFace with description, creation methodology, statistics, limitations
- [ ] **REL-02**: Model card on HuggingFace with metadata YAML, usage examples, training params, benchmark results
- [ ] **REL-03**: Model weights published in safetensors format
- [ ] **REL-04**: MIT license applied consistently to datasets, model weights, scripts, and eval code
- [ ] **REL-05**: GGUF quantized variants (Q4_K_M, Q8_0) published for LM Studio / llama.cpp
- [ ] **REL-06**: Interactive Gradio demo Space on HuggingFace showcasing all three capability areas
- [ ] **REL-07**: Versioned dataset releases with documented changes and metrics per version

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Alignment

- **ALIGN-01**: DPO/RLHF alignment training using preference pairs from SFT model outputs
- **ALIGN-02**: Safety filtering and refusal behavior for harmful queries

### Multi-Model

- **MULTI-01**: Support for additional base models beyond SmolLM2-1.7B
- **MULTI-02**: Architecture-agnostic dataset format with model-specific conversion scripts

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Multi-model support in v1 | Splits focus; each model needs different tokenizer, chat template, and context limits |
| RLHF/DPO alignment | Massive complexity (reward model, preference data); SFT sufficient for v1 capability validation |
| Human annotation pipeline | Expensive and slow; Opus generates high-quality data; automated quality filtering replaces manual review |
| Long-context training data | SmolLM2-1.7B trained at 2048 sequence length; optimize for short practical interactions |
| Image/multimodal data | SmolLM2-1.7B is text-only; multimodal requires different architecture |
| Web UI for dataset curation | CLI-first pipeline; users who need UI can use Argilla or HuggingFace data viewer |
| Hosted API/serving infrastructure | Release weights; users deploy on their own infrastructure (Ollama, LM Studio, vLLM) |
| Automated continuous training pipeline | Over-engineering for iterative research project; manual training runs with documented configs |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| DATA-01 | TBD | Pending |
| DATA-02 | TBD | Pending |
| DATA-03 | TBD | Pending |
| DATA-04 | TBD | Pending |
| DATA-05 | TBD | Pending |
| DATA-06 | TBD | Pending |
| DATA-07 | TBD | Pending |
| TOOL-01 | TBD | Pending |
| TOOL-02 | TBD | Pending |
| TOOL-03 | TBD | Pending |
| TOOL-04 | TBD | Pending |
| TOOL-05 | TBD | Pending |
| CODE-01 | TBD | Pending |
| CODE-02 | TBD | Pending |
| CODE-03 | TBD | Pending |
| KNOW-01 | TBD | Pending |
| KNOW-02 | TBD | Pending |
| KNOW-03 | TBD | Pending |
| TRNG-01 | TBD | Pending |
| TRNG-02 | TBD | Pending |
| TRNG-03 | TBD | Pending |
| EVAL-01 | TBD | Pending |
| EVAL-02 | TBD | Pending |
| EVAL-03 | TBD | Pending |
| EVAL-04 | TBD | Pending |
| REL-01 | TBD | Pending |
| REL-02 | TBD | Pending |
| REL-03 | TBD | Pending |
| REL-04 | TBD | Pending |
| REL-05 | TBD | Pending |
| REL-06 | TBD | Pending |
| REL-07 | TBD | Pending |

**Coverage:**
- v1 requirements: 32 total
- Mapped to phases: 0
- Unmapped: 32

---
*Requirements defined: 2026-04-20*
*Last updated: 2026-04-20 after initial definition*
