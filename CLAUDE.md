<!-- GSD:project-start source:PROJECT.md -->
## Project

**Lyra**

An open-source dataset curation and model fine-tuning project that distills Claude Opus reasoning quality into training data for SmolLM2-1.7B. Lyra aims to produce a small language model that punches far above its weight class -- capable of tool calling, code generation, and general knowledge tasks that typically require much larger models. Named in the musical tradition of Haiku, Sonnet, and Opus -- a small constellation that produces clear, harmonious output.

**Core Value:** Curate Opus-quality training data that makes a 1.7B parameter model practically useful for day-to-day development tasks -- tool calls, quick code, and general reasoning.

### Constraints

- **Base model**: SmolLM2-1.7B -- all fine-tuning targets this architecture
- **Data format**: ShareGPT conversation format -- widely supported, HuggingFace native
- **License**: MIT -- maximum permissiveness for open-source adoption
- **Data source**: Claude Opus generates all training data -- no human annotation in v1
- **Scale strategy**: Start small (~5K samples), measure impact, grow iteratively
- **Token limits**: Must account for SmolLM2-1.7B's context window and generation limits in training data design
<!-- GSD:project-end -->

<!-- GSD:stack-start source:research/STACK.md -->
## Technology Stack

## Recommended Stack
### Layer 1: Dataset Generation (Claude Opus via Anthropic API)
| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| anthropic (Python SDK) | 0.96.0 | Programmatic access to Claude Opus for synthetic data generation | Official SDK; direct support for Message Batches API which cuts cost 50%; prompt caching stacks for up to 95% savings. Required for generating all training data. |
| Claude Opus 4.6 | claude-opus-4-6 | Teacher model for generating ShareGPT training samples | Best quality-to-cost ratio for data generation. $5/$25 per MTok standard, $2.50/$12.50 batch. Opus 4.7 is newer but 3x more expensive for marginal gains -- not justified for data generation at scale. |
| Message Batches API | v1 | Async bulk generation of training samples | 50% cost reduction, 100K messages per batch, 24hr processing. Perfect for generating 5K+ samples where latency does not matter. Combine with prompt caching for 95% cost reduction on repeated system prompts. |
### Layer 2: Dataset Processing and Format
| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| datasets (HuggingFace) | 4.8.4 | Dataset loading, processing, formatting, and HuggingFace Hub publishing | De facto standard for HuggingFace ecosystem. Native ShareGPT support, Arrow-backed for memory efficiency, push_to_hub for publishing. |
| Python (stdlib json/jsonl) | 3.10+ | ShareGPT format serialization | ShareGPT is just JSON. No special library needed for format construction. Keep it simple -- build conversations as dicts, validate with Pydantic, serialize to JSONL. |
| pydantic | 2.x | Schema validation for ShareGPT conversations | Type-safe validation of conversation format before training. Catches malformed samples early. Already a dependency of anthropic SDK. |
### Layer 3: Fine-Tuning Framework
| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| unsloth | 2026.4.6 | Primary fine-tuning framework -- LoRA/QLoRA with speed optimizations | 2x faster, 70% less VRAM than vanilla TRL. Explicit SmolLM2-1.7B support (pre-quantized models on Hub). Free tier works on single GPU/Colab. Best choice for a 1.7B model where multi-GPU is unnecessary. |
| trl | 1.2.0 | SFTTrainer backend (used by Unsloth under the hood) | HuggingFace's official post-training library. v1.0+ is a major maturity milestone. Unsloth integrates directly with TRL's SFTTrainer, so you get TRL's features with Unsloth's optimizations. |
| transformers | 5.5.4 | Model loading, tokenization, inference | Core HuggingFace library. Required by everything else. v5.x is current stable. |
| peft | 0.19.1 | LoRA adapter configuration and management | Standard for parameter-efficient fine-tuning. LoRA config passed to SFTTrainer. At 1.7B params, QLoRA lets you fine-tune on a single consumer GPU (16GB VRAM). |
| accelerate | 1.13.0 | Training orchestration, mixed precision | Handles device placement, gradient accumulation, mixed precision. Required by TRL/Unsloth. |
| bitsandbytes | 0.49.2 | 4-bit quantization for QLoRA | Enables QLoRA by quantizing base model to 4-bit NF4. At 1.7B params, full fine-tuning is also feasible on 24GB GPUs, but QLoRA is safer and nearly as effective. |
### Layer 4: Evaluation
| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| lm-evaluation-harness (lm-eval) | 0.4.11 | General benchmarks (MMLU, ARC, HellaSwag, etc.) | Industry standard. Powers HuggingFace Open LLM Leaderboard. Supports HuggingFace models natively. Hundreds of built-in tasks. |
| bigcode-evaluation-harness | latest | Code generation benchmarks (HumanEval, MBPP, HumanEval+, MBPP+) | Standard code eval framework. pass@k metrics on Python code generation. Directly measures code quality improvement. |
| BFCL (Berkeley Function Calling Leaderboard) | v4 | Tool calling accuracy benchmark | Purpose-built for function calling evaluation. AST-based evaluation scales to thousands of functions. Only standard benchmark for tool use. |
| Custom eval scripts | -- | Domain-specific evaluation for Lyra's three focus areas | Standard benchmarks will not cover Lyra's specific tool-calling format or output style. Build lightweight custom evals for: (1) JSON tool call format accuracy, (2) code execution pass rate, (3) factual correctness on knowledge questions. |
### Layer 5: Experiment Tracking and Infrastructure
| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| wandb | 0.26.0 | Experiment tracking, loss curves, hyperparameter logging | Native TRL/Unsloth integration (one-line setup). Free tier sufficient for this project. Tracks loss, learning rate, GPU metrics automatically. |
| huggingface_hub | latest | Model and dataset publishing to HuggingFace Hub | Push trained models and datasets to Hub for open-source release. Integrated with all HF libraries. |
| torch | 2.6+ | PyTorch backend for training | Required by all training libraries. Let Unsloth's installer resolve the correct CUDA-matched version. Do not install manually. |
### Layer 6: Data Quality (Optional, Phase 2+)
| Technology | Version | Purpose | When to Use |
|------------|---------|---------|-------------|
| argilla | 2.8.0 | Human-in-the-loop data review and curation | When you want to manually review/filter generated samples before training. Not needed for v1 (Opus quality is sufficient), but valuable for iterative data improvement. |
| distilabel | 1.5.3 | Structured synthetic data pipelines with built-in quality checks | If dataset generation becomes complex enough to need a pipeline framework. Overkill for v1's straightforward generate-then-validate flow. Consider if scaling beyond 10K samples. |
## Installation
# Python 3.10+ required (3.10 is the sweet spot for compatibility)
# Core: Dataset generation
# Core: Fine-tuning (let unsloth handle torch/CUDA resolution)
# Core: Experiment tracking
# Core: Evaluation
# Optional: Code evaluation (install separately, has heavy deps)
# git clone https://github.com/bigcode-project/bigcode-evaluation-harness
# cd bigcode-evaluation-harness && pip install -e .
# Optional: BFCL tool calling eval
# git clone https://github.com/ShishirPatil/gorilla
# cd gorilla/berkeley-function-call-leaderboard && pip install -e .
# Publishing
## Alternatives Considered
| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| Unsloth (fine-tuning) | Axolotl | If you need multi-GPU training or more complex training configs (FSDP, DeepSpeed). Axolotl is YAML-config-driven and more flexible but slower. For a 1.7B model on single GPU, Unsloth wins. |
| Unsloth (fine-tuning) | torchtune (Meta) | If you want a minimal, PyTorch-native approach without HuggingFace dependencies. Less mature ecosystem, fewer model integrations. |
| Unsloth (fine-tuning) | Raw TRL SFTTrainer | If you want maximum control and do not care about speed/memory optimization. Unsloth wraps TRL anyway, so you get TRL's API with better performance. |
| anthropic SDK (data gen) | distilabel | If you need complex multi-step data generation pipelines with built-in caching, retry logic, and quality scoring. For v1's straightforward batch generation, the SDK + custom scripts is simpler and more transparent. |
| anthropic SDK (data gen) | OpenAI SDK + GPT-4o | If cost is the only concern (GPT-4o is cheaper). But Lyra's core value proposition is Opus-quality data, and the project explicitly targets Claude Opus. |
| wandb (tracking) | MLflow | If you need self-hosted experiment tracking with no SaaS dependency. wandb's free tier and TRL integration make it the path of least resistance. |
| lm-eval-harness (eval) | OpenCompass | If you need broader Chinese-language benchmarks or a GUI. lm-eval-harness has wider community adoption and is the HuggingFace standard. |
| QLoRA (training method) | Full fine-tuning | At 1.7B params, full fine-tuning needs ~14GB VRAM (feasible on 24GB GPUs). Produces slightly better results but QLoRA at 4-bit is 90-95% as good. Use full fine-tuning only if you have a dedicated 24GB+ GPU and want to squeeze out the last 5%. |
| QLoRA (training method) | LoRA (16-bit) | Middle ground: more VRAM than QLoRA (~8-10GB for 1.7B) but no quantization artifacts. Reasonable choice if you have the VRAM headroom. |
## What NOT to Use
| Avoid | Why | Use Instead |
|-------|-----|-------------|
| AutoTrain (HuggingFace) | Black-box fine-tuning with limited control over hyperparameters, data formatting, and training loop. You cannot customize the ShareGPT conversation template or tool-call format handling. | Unsloth + TRL SFTTrainer for full control |
| LLaMA-Factory | Feature-rich but poorly maintained, frequent breaking changes, heavy Chinese-language documentation bias. Community reports stability issues with non-LLaMA architectures. | Unsloth or Axolotl |
| OpenAI fine-tuning API | Closed-source, expensive, limited to OpenAI models. Does not fit the open-source mission. | Local fine-tuning with Unsloth |
| RLHF/DPO (for v1) | Premature optimization. DPO requires preference pairs which doubles data generation cost. SFT is sufficient for v1 -- add alignment in v2 if needed. SmolLM2's official training used DPO after SFT. | Pure SFT first, DPO as future enhancement |
| DeepSpeed/FSDP | Multi-GPU distributed training frameworks. Complete overkill for a 1.7B model that fits on a single consumer GPU. Adds complexity for zero benefit at this scale. | Single-GPU training via Unsloth |
| vLLM (for training) | vLLM is an inference server, not a training tool. Useful for serving the fine-tuned model later but irrelevant during fine-tuning. | Unsloth for training; consider vLLM for deployment later |
| Older SmolLM (v1) | SmolLM v1 has been superseded by SmolLM2 with significant improvements in instruction following and reasoning. SmolLM3 exists but may not have 1.7B variant yet. | SmolLM2-1.7B specifically |
## Stack Patterns by Variant
- Use QLoRA (4-bit) via Unsloth -- fits comfortably in 16GB
- Use Unsloth's Colab notebooks as starting templates
- Batch size 2-4, gradient accumulation 4-8
- Training 5K samples takes ~30-60 minutes
- QLoRA still recommended for speed, but full fine-tuning is feasible
- Can increase batch size to 8-16
- Training 5K samples takes ~15-30 minutes
- Full fine-tuning is practical and preferred for best quality
- LoRA 16-bit is the sweet spot (faster than full, no quantization)
- Consider training on larger dataset (10K-50K samples)
- Switch from raw anthropic SDK to distilabel for pipeline management
- Add Argilla for data review workflow
- Consider Axolotl for multi-GPU training if dataset grows significantly
## Version Compatibility
| Package | Compatible With | Notes |
|---------|-----------------|-------|
| unsloth 2026.4.6 | torch 2.1-2.10, CUDA 11.8-13.0 | Let unsloth resolve torch version via `--torch-backend=auto` |
| trl 1.2.0 | transformers >=4.47.0, peft >=0.13.0 | v1.0+ requires Python >=3.10 |
| peft 0.19.1 | transformers >=4.x | Production stable |
| transformers 5.5.4 | Python >=3.10 | Major version bump from 4.x; API is backward-compatible for standard use |
| datasets 4.8.4 | Python >=3.10, Arrow backend | Major version bump from 3.x; same API surface |
| lm-eval 0.4.11 | transformers, vLLM, SGLang backends | Install model backends as extras |
| anthropic 0.96.0 | Python >=3.9 | Batch API support built-in |
| SmolLM2-1.7B | 8192 token context window | Design training samples to fit within this limit. Account for system prompt + conversation turns. Target ~2K-4K tokens per sample max. |
## Key Architecture Decisions
### Why Not distilabel for Data Generation (v1)
- Data generation is straightforward: send prompts to Opus, collect responses, format as ShareGPT
- The anthropic SDK's Batch API handles the heavy lifting (async, retry, cost optimization)
- Custom Python scripts give full transparency into what prompts produce what data
- Distilabel adds abstraction that obscures the data generation process when you most need to understand it
### Why QLoRA over Full Fine-Tuning (Default)
- Works everywhere (Colab, consumer GPU, cloud)
- 90-95% of full fine-tuning quality (verified across multiple studies)
- 2x faster with Unsloth optimizations
- Produces small adapter files (~50-100MB) instead of full model copies (~3.4GB)
- Adapters can be merged into base model for deployment via `merge_and_unload()`
### SmolLM2-1.7B Context Window Implications
- System prompt: ~200-500 tokens
- Tool definitions (for tool-calling samples): ~500-1500 tokens
- Conversation turns: ~1000-4000 tokens
- Reserve: ~1000 tokens for generation buffer
- **Practical max per training sample: ~4000-6000 tokens total**
## Sources
- [HuggingFace TRL v1.0 announcement](https://www.marktechpost.com/2026/04/01/hugging-face-releases-trl-v1-0-a-unified-post-training-stack-for-sft-reward-modeling-dpo-and-grpo-workflows/) -- TRL v1.0 release details (HIGH confidence)
- [PyPI: trl 1.2.0](https://pypi.org/project/trl/) -- Version verified (HIGH confidence)
- [PyPI: transformers 5.5.4](https://pypi.org/project/transformers/) -- Version verified (HIGH confidence)
- [PyPI: peft 0.19.1](https://pypi.org/project/peft/) -- Version verified (HIGH confidence)
- [PyPI: anthropic 0.96.0](https://pypi.org/project/anthropic/) -- Version verified (HIGH confidence)
- [PyPI: unsloth 2026.4.6](https://pypi.org/project/unsloth/) -- Version verified (HIGH confidence)
- [PyPI: datasets 4.8.4](https://pypi.org/project/datasets/) -- Version verified (HIGH confidence)
- [PyPI: accelerate 1.13.0](https://pypi.org/project/accelerate/) -- Version verified (HIGH confidence)
- [PyPI: bitsandbytes 0.49.2](https://pypi.org/project/bitsandbytes/) -- Version verified (HIGH confidence)
- [PyPI: lm-eval 0.4.11](https://pypi.org/project/lm-eval/) -- Version verified (HIGH confidence)
- [PyPI: wandb 0.26.0](https://pypi.org/project/wandb/) -- Version verified (HIGH confidence)
- [PyPI: argilla 2.8.0](https://pypi.org/project/argilla/) -- Version verified (HIGH confidence)
- [PyPI: distilabel 1.5.3](https://pypi.org/project/distilabel/) -- Version verified (HIGH confidence)
- [Unsloth SmolLM2-1.7B-Instruct](https://huggingface.co/unsloth/SmolLM2-1.7B-Instruct) -- SmolLM2 support confirmed (HIGH confidence)
- [SmolLM2-1.7B model card](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B) -- Model specs, 8192 context (HIGH confidence)
- [Anthropic Batch API pricing](https://platform.claude.com/docs/en/about-claude/pricing) -- Batch pricing 50% off, cache stacks (HIGH confidence)
- [Anthropic Message Batches API docs](https://platform.claude.com/docs/en/api/creating-message-batches) -- Batch API parameters and limits (HIGH confidence)
- [BFCL v4 leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html) -- Tool calling benchmark details (HIGH confidence)
- [BigCode Evaluation Harness](https://github.com/bigcode-project/bigcode-evaluation-harness) -- Code eval framework (HIGH confidence)
- [EleutherAI lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) -- General eval framework (HIGH confidence)
- [SmolLM2 fine-tuning blog](https://huggingface.co/blog/prithivMLmods/smollm2-ft) -- Training approach reference (MEDIUM confidence)
- [Fine-tuning framework comparison](https://blog.spheron.network/comparing-llm-fine-tuning-frameworks-axolotl-unsloth-and-torchtune-in-2025) -- Axolotl vs Unsloth vs Torchtune (MEDIUM confidence)
- [Small models for tool calling](https://arxiv.org/html/2512.15943v1) -- Research on fine-tuned SLMs achieving 77.55% on ToolBench (MEDIUM confidence)
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->
## Conventions

Conventions not yet established. Will populate as patterns emerge during development.
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture

Architecture not yet mapped. Follow existing patterns found in the codebase.
<!-- GSD:architecture-end -->

<!-- GSD:skills-start source:skills/ -->
## Project Skills

No project skills found. Add skills to any of: `.claude/skills/`, `.agents/skills/`, `.cursor/skills/`, or `.github/skills/` with a `SKILL.md` index file.
<!-- GSD:skills-end -->

<!-- GSD:workflow-start source:GSD defaults -->
## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:
- `/gsd-quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd-debug` for investigation and bug fixing
- `/gsd-execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->



<!-- GSD:profile-start -->
## Developer Profile

> Profile not yet configured. Run `/gsd-profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
