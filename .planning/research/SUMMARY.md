# Project Research Summary

**Project:** Lyra -- Opus-quality dataset curation and SmolLM2-1.7B fine-tuning
**Domain:** Synthetic data generation, dataset curation, small LLM fine-tuning and evaluation
**Researched:** 2026-04-20
**Confidence:** HIGH

## Executive Summary

Lyra is a knowledge distillation project: use Claude Opus (teacher) to generate high-quality synthetic training data, curate it into a ShareGPT-format dataset, and fine-tune SmolLM2-1.7B (student) across three capability domains -- tool calling, code generation, and general knowledge. The established approach for this class of project is a stage-isolated pipeline with disk boundaries between generation, curation, assembly, training, evaluation, and release. The stack is mature and well-documented: Anthropic's Batch API for cost-efficient data generation, Unsloth + TRL/PEFT for QLoRA fine-tuning on consumer GPUs, and lm-evaluation-harness for benchmarking. Every major component has production-grade library support with verified current versions.

The recommended approach is to start small and validate iteratively. Generate 100-200 samples per domain first, run a training cycle, evaluate, and only then scale to the full 5K target. This "small-batch-first" pattern avoids the most expensive mistake: generating thousands of samples with prompts that produce poor training data. QLoRA via Unsloth is the default training method because it works on any hardware (including free Colab T4) while delivering 90-95% of full fine-tuning quality. The entire pipeline -- from data generation through evaluation -- should be operational end-to-end before scaling up data volume.

The primary risks are concentrated in three areas. First, tool-call format brittleness: at 1.7B parameters, the model has limited capacity to simultaneously maintain JSON syntax correctness and semantic reasoning, requiring strict format standardization and constrained decoding at inference. Second, overfitting on limited data: 5K samples split across three domains gives only ~1,667 per task, which is the floor of what research considers viable, demanding aggressive regularization and out-of-distribution evaluation. Third, circular evaluation: using Opus to both generate and judge data quality creates blind spots that must be broken with programmatic validators and cross-model evaluation signals. All three risks are manageable with upfront planning but catastrophic if ignored.

## Key Findings

### Recommended Stack

The stack follows a six-layer architecture from data generation through experiment tracking. All versions have been verified on PyPI as of April 2026. The critical tooling decisions are: Anthropic's Message Batches API for generation (50% cost reduction, stacks with prompt caching for up to 95% savings), Unsloth 2026.4.6 wrapping TRL's SFTTrainer for 2x faster training with 70% less VRAM, and QLoRA (4-bit via bitsandbytes) as the default training method. SmolLM2-1.7B's 8192-token context window constrains training samples to approximately 4,000-6,000 tokens total.

**Core technologies:**
- **Anthropic SDK + Message Batches API**: Synthetic data generation via Claude Opus -- 50% batch pricing + prompt caching yields ~$5-15 for 5K samples
- **Unsloth + TRL SFTTrainer + PEFT**: Fine-tuning framework -- QLoRA on single consumer GPU (~4-6GB VRAM for 1.7B model), 2x speed over vanilla TRL
- **HuggingFace Datasets**: Dataset loading, formatting, and Hub publishing -- Arrow-backed, native ShareGPT support
- **lm-evaluation-harness**: Standard benchmarking (MMLU, ARC, HellaSwag) -- powers HuggingFace Open LLM Leaderboard
- **Pydantic 2.x**: Schema validation for ShareGPT conversations -- catches malformed samples before training
- **wandb**: Experiment tracking -- native TRL integration, free tier sufficient

**What to avoid:** AutoTrain (black-box), LLaMA-Factory (stability issues), RLHF/DPO (premature for v1), DeepSpeed/FSDP (overkill at 1.7B), and any multi-GPU distributed training.

### Expected Features

**Must have (table stakes):**
- ShareGPT-format dataset with correct role ordering and tool-call extensions
- Complete dataset card and model card on HuggingFace
- Model weights in safetensors format
- End-to-end reproduction scripts
- Data quality filtering pipeline (dedup, format validation, quality scoring)
- Evaluation results with standard benchmarks (MMLU, MBPP/HumanEval, tool-call accuracy)
- Train/validation/test splits stratified across three focus areas
- MIT license on all artifacts

**Should have (differentiators):**
- Tool calling with structured JSON function calls (very few small models do this reliably)
- GGUF quantized variants for consumer deployment (LM Studio, llama.cpp)
- Interactive demo Space on HuggingFace
- Custom eval benchmarks released alongside model
- Generation pipeline released as reusable tool
- Prompt template library for data generation
- Versioned dataset releases with iteration history

**Defer to v2+:**
- MCP-style tool use patterns, adaptive output style encoding, per-category quality metrics, DPO alignment, multi-model support, multimodal, web UI for curation

### Architecture Approach

The architecture is a six-layer stage-isolated pipeline: Data Generation, Data Curation, Dataset Assembly, Training, Evaluation, and Release. Each layer operates independently, reading from and writing to disk (JSONL files at each stage). This design enables rerunning any stage in isolation, inspecting intermediate data, and resuming after failures. The pipeline is config-driven: prompt templates, hyperparameters, and evaluation tasks are all defined in YAML files separate from source code.

**Major components:**
1. **Data Generation Layer** -- Prompt templates (YAML/Jinja2), Anthropic Batch API client, async response collector writing to raw JSONL
2. **Data Curation Layer** -- Format validator (JSON Schema), quality filter (heuristic + optional LLM-scored), deduplication engine (exact + MinHash LSH)
3. **Dataset Assembly Layer** -- Domain mixer (33/33/33 ratio enforcement), ShareGPT formatter, stratified split generator (90/5/5)
4. **Training Layer** -- SmolLM2-1.7B model loader with chat template setup, SFTTrainer with LoRA/QLoRA config, checkpoint manager
5. **Evaluation Layer** -- Standard benchmark runner (lm-eval-harness), custom eval harness (tool-call/code/knowledge), base-vs-tuned comparison reporter
6. **Release Layer** -- HuggingFace uploader (datasets + weights), model card generator, reproduction scripts

### Critical Pitfalls

1. **Tool-call format brittleness at 1.7B parameters** -- Standardize on ONE canonical JSON format throughout the entire dataset. Keep schemas simple (max 3-5 params, flat). Include 15-20% "no tool needed" examples. Use constrained decoding at inference as a safety net.

2. **Catastrophic forgetting across three tasks** -- Monitor per-task validation metrics separately, not just aggregate loss. Mix in 5-10% replay data from base SmolLM2-Instruct distribution. Consider difficulty-weighted sampling rather than naive 33/33/33 random shuffle.

3. **Overfitting on 5K samples (1,667 per domain)** -- Limit to 1-3 epochs. Build a genuinely out-of-distribution test set not generated by Opus. Use LoRA dropout 0.1-0.15. If ID-OOD gap exceeds 15 percentage points, overfitting is confirmed.

4. **Circular evaluation (Opus generates and Opus judges)** -- Use programmatic validators for structured outputs (JSON schema check, code execution). Cross-reference with a different model for open-ended quality scoring. Reserve 5-10% for human spot-check.

5. **SmolLM2 chat template and tokenizer misalignment** -- Verify the exact chat template before generating any data. Test ShareGPT-to-SmolLM2 conversion on 10+ samples with manual inspection. Validate loss masking targets only assistant tokens.

## Implications for Roadmap

Based on the combined research, the project naturally decomposes into five phases with clear dependency ordering.

### Phase 1: Format Specification and Data Generation Pipeline
**Rationale:** Everything downstream depends on getting the data format right. The architecture research identifies format validation as the first component to build. The pitfalls research flags tokenizer misalignment and tool-call format brittleness as must-solve-first problems. Generating data before the format is locked wastes API budget.
**Delivers:** Tool-call JSON schema specification, ShareGPT format validator, prompt template system, Anthropic Batch API client, initial 100-200 samples per domain for pipeline validation.
**Addresses features:** ShareGPT-format dataset, conversation format correctness, prompt template library.
**Avoids pitfalls:** Tokenizer misalignment (by validating format early), single-source homogeneity (by building diversity into prompts from day one).

### Phase 2: Data Curation and Evaluation Framework
**Rationale:** The pitfalls research is emphatic: build evaluation BEFORE training, not after. The architecture research places quality filtering as the gate between raw data and training. Both the circular evaluation pitfall and the "trusting loss over task metrics" pitfall demand that evaluation infrastructure exist before training begins. Curation and evaluation are grouped because they share the same concern: data and model quality measurement.
**Delivers:** Quality filter pipeline (heuristic rules + dedup), deterministic validators for tool calls and code, custom eval task definitions, OOD test set, base model baseline measurements on all metrics.
**Addresses features:** Data quality filtering pipeline, evaluation results framework, train/val/test splits.
**Avoids pitfalls:** Circular evaluation (by building programmatic validators), conflating loss with capability (by defining task metrics upfront), overfitting (by establishing OOD test set).

### Phase 3: Full Data Generation and Training
**Rationale:** With format validated, quality filters built, and evaluation ready, it is now safe to scale data generation to 5K samples and run training. The architecture research recommends this order explicitly: validate on small data first, then scale. Training is the fastest-iterating stage (minutes with LoRA) once the upstream pipeline is solid.
**Delivers:** Complete 5K-sample curated dataset, trained LoRA adapters on SmolLM2-1.7B, training hyperparameter configuration, experiment tracking via wandb.
**Addresses features:** Fine-tuning scripts with documented config, model weights in safetensors format.
**Avoids pitfalls:** Overfitting (by using regularization and early stopping against OOD set), catastrophic forgetting (by monitoring per-task metrics), 5K sample ceiling (by designing pipeline for easy scale-up).
**Uses stack:** Unsloth + TRL SFTTrainer, PEFT LoRA, bitsandbytes QLoRA, wandb.

### Phase 4: Evaluation, Model Card, and HuggingFace Release
**Rationale:** The features research identifies model card, dataset card, and benchmark results as table stakes that cannot be skipped. The architecture research places release as the terminal pipeline stage, dependent on all upstream stages. Evaluation must run against the trained model before any release claims can be made.
**Delivers:** Standard benchmark results (MMLU, MBPP, tool-call accuracy), comparison report (base vs. fine-tuned), complete model card and dataset card, safetensors weights pushed to HuggingFace Hub, MIT-licensed reproduction scripts.
**Addresses features:** Evaluation results with standard benchmarks, model card, dataset card, reproduction scripts, MIT license.
**Avoids pitfalls:** Trusting loss over task metrics (by running full benchmark suite), incomplete release (by following HuggingFace model release checklist).

### Phase 5: Community Release Enhancements
**Rationale:** GGUF variants, demo Space, and custom eval benchmarks are differentiators that increase adoption but are not needed to validate the core approach. The features research explicitly defers these to v0.2.
**Delivers:** GGUF quantized model variants (Q4_K_M, Q8_0), interactive Gradio demo Space on HuggingFace, custom eval benchmarks as a separate release, generation pipeline packaged as a reusable tool.
**Addresses features:** GGUF quantized variants, interactive demo Space, custom eval benchmarks, generation pipeline as reusable tool, versioned dataset releases.

### Phase Ordering Rationale

- **Format first, training last:** The architecture research identifies the format validator as the foundational contract. The pitfalls research warns that tokenizer misalignment wastes all downstream work. Establishing format correctness before spending API budget or GPU time is non-negotiable.
- **Evaluation before training:** Counter-intuitive but strongly supported by research. Pitfall 7 (trusting loss over task metrics) is the most common failure mode in fine-tuning projects. Building the eval harness first means every training run produces actionable signal from step one.
- **Small data validation, then scale:** The architecture research's anti-pattern 2 warns against generating all data before any training. Phase 1 produces 100-200 samples to validate the pipeline; Phase 3 scales to 5K only after the pipeline is proven.
- **Release artifacts grouped together:** Model card, dataset card, and benchmark results are interdependent (the card references the benchmarks). Grouping them ensures consistency.
- **Enhancements are clearly separable:** GGUF, demo Space, and reusable pipeline tools have no upstream dependencies beyond Phase 4's model weights. They can be parallelized or deferred without blocking the core release.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 1:** Tool-call JSON schema design requires research into SmolLM2's exact chat template handling, TRL's tool-call support in SFTTrainer, and the specific ShareGPT extensions for function calling. Multiple format variants exist and the wrong choice causes silent training failures.
- **Phase 3:** Training hyperparameter selection (LoRA rank, learning rate, batch size, epoch count) for multi-task fine-tuning on a 1.7B model with only 5K samples. The pitfalls research flags catastrophic forgetting and overfitting as high-risk areas that need careful tuning.

Phases with well-documented patterns (skip deep research):
- **Phase 2:** Quality filtering (dedup, format validation, heuristic scoring) and evaluation harness setup (lm-eval-harness custom tasks) follow well-established patterns with extensive documentation.
- **Phase 4:** HuggingFace release process has a detailed official checklist. Model card and dataset card formats are standardized.
- **Phase 5:** GGUF conversion and Gradio demo Spaces are thoroughly documented by the community.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | All package versions verified on PyPI as of April 2026. Unsloth has explicit SmolLM2-1.7B support. TRL v1.0+ is a major maturity milestone. Anthropic Batch API pricing confirmed from official docs. |
| Features | HIGH | Feature landscape well-mapped from HuggingFace ecosystem norms, Microsoft research on small model tool calling, and community expectations for open-source model releases. MVP prioritization is clear. |
| Architecture | HIGH | Stage-isolated pipeline pattern is the established standard for data generation and fine-tuning projects. Component boundaries are well-defined. Project structure follows HuggingFace conventions. |
| Pitfalls | HIGH | All seven critical pitfalls sourced from peer-reviewed research or official documentation. Recovery strategies are documented. The single-source homogeneity and tool-call brittleness risks are particularly well-characterized in the literature. |

**Overall confidence:** HIGH

### Gaps to Address

- **SmolLM2 chat template specifics for tool calls:** The exact token-level format for tool-call conversations in SmolLM2's template needs hands-on verification. TRL's tool-call support in SFTTrainer is documented but SmolLM2-specific behavior may differ. Resolve by building 10 sample conversions and inspecting tokenized output in Phase 1.

- **Optimal LoRA rank and hyperparameters for 3-way multi-task at 1.7B:** Research provides ranges (r=16-32, lr=1e-4 to 2e-4, dropout 0.1-0.15) but the optimal configuration for this specific model-size + task-count + dataset-size combination is unknown. Resolve by running a small hyperparameter sweep in Phase 3.

- **Catastrophic forgetting severity with single adapter:** Whether a single LoRA adapter can sustain three tasks at 1.7B without unacceptable regression is empirically uncertain. The pitfalls research flags this but notes single-adapter is acceptable for v1 if per-task metrics remain above thresholds. Resolve by monitoring per-task metrics; fall back to task-specific adapters if needed.

- **Constrained decoding availability across inference backends:** Tool-call reliability at inference depends on constrained decoding (grammar-based sampling), but support varies across Ollama, vLLM, and llama.cpp. Resolve by testing the top 3 backends before Phase 4 release.

## Sources

### Primary (HIGH confidence)
- Anthropic Message Batches API docs -- batch pricing, limits, async polling
- Anthropic pricing page -- 50% batch discount, prompt caching stacking
- HuggingFace TRL SFTTrainer documentation -- dataset format, assistant-only loss, tool calling
- SmolLM2-1.7B and SmolLM2-1.7B-Instruct model cards -- architecture, context window, tokenizer
- Unsloth SmolLM2-1.7B-Instruct page -- confirmed support, pre-quantized models
- PyPI verified versions: anthropic 0.96.0, unsloth 2026.4.6, trl 1.2.0, transformers 5.5.4, peft 0.19.1, datasets 4.8.4, lm-eval 0.4.11, wandb 0.26.0, bitsandbytes 0.49.2, accelerate 1.13.0
- EleutherAI lm-evaluation-harness -- standard benchmark framework
- HuggingFace Model Release Checklist -- release best practices
- SmolLM2 paper (arXiv 2502.02737) -- architecture and training details

### Secondary (MEDIUM confidence)
- Microsoft: Fine-Tuning Small Language Models for Function Calling -- SLM tool-calling comprehensive guide
- Small Language Models for Efficient Agentic Tool Calling (arXiv 2512.15943) -- 77.55% ToolBench results
- Synthetic Eggs in Many Baskets (arXiv 2511.01490) -- diversity and distribution collapse
- Demystifying Synthetic Data in LLM Pre-training (arXiv 2510.01631) -- synthetic data limits
- Mitigating Catastrophic Forgetting in LLMs (EMNLP 2025) -- forgetting strategies
- When Scale is Fixed (arXiv 2504.12491) -- perplexity unreliability for downstream performance
- SmolLM2 fine-tuning blog (HuggingFace) -- end-to-end fine-tuning patterns
- Fine-tuning framework comparison (Spheron) -- Axolotl vs Unsloth vs Torchtune

### Tertiary (needs validation)
- Optimal 33/33/33 domain split vs. difficulty-weighted distribution -- no direct research for this model size and task combination; requires empirical validation
- Single LoRA adapter viability for 3 concurrent tasks at 1.7B -- community anecdotes suggest it works but peer-reviewed evidence is sparse

---
*Research completed: 2026-04-20*
*Ready for roadmap: yes*
