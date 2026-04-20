# Architecture Research

**Domain:** Dataset curation and small LLM fine-tuning pipeline
**Researched:** 2026-04-20
**Confidence:** HIGH

## Standard Architecture

### System Overview

```
+-----------------------------------------------------------------------+
|                     DATA GENERATION LAYER                             |
|  +----------------+  +----------------+  +-------------------+        |
|  | Prompt         |  | Opus API       |  | Response          |        |
|  | Templates      |  | Client         |  | Collector         |        |
|  | (per domain)   |  | (Batch API)    |  | (JSONL writer)    |        |
|  +-------+--------+  +-------+--------+  +--------+----------+        |
|          |                    |                    |                   |
+----------+--------------------+--------------------+------------------+
           |                    |                    |
+----------v--------------------v--------------------v------------------+
|                     DATA CURATION LAYER                               |
|  +----------------+  +----------------+  +-------------------+        |
|  | Format         |  | Quality        |  | Deduplication     |        |
|  | Validator      |  | Filter         |  | Engine            |        |
|  | (schema check) |  | (heuristic +   |  | (exact + fuzzy)   |        |
|  |                |  |  LLM-scored)   |  |                   |        |
|  +-------+--------+  +-------+--------+  +--------+----------+        |
|          |                    |                    |                   |
+----------+--------------------+--------------------+------------------+
           |                    |                    |
+----------v--------------------v--------------------v------------------+
|                     DATASET ASSEMBLY LAYER                            |
|  +----------------+  +----------------+  +-------------------+        |
|  | Domain         |  | ShareGPT       |  | Split             |        |
|  | Mixer          |  | Formatter      |  | Generator         |        |
|  | (33/33/33)     |  | (conversations)|  | (train/val/test)  |        |
|  +-------+--------+  +-------+--------+  +--------+----------+        |
|          |                    |                    |                   |
+----------+--------------------+--------------------+------------------+
           |                    |                    |
+----------v--------------------v--------------------v------------------+
|                     TRAINING LAYER                                    |
|  +----------------+  +----------------+  +-------------------+        |
|  | Model          |  | SFTTrainer     |  | Checkpoint        |        |
|  | Loader         |  | (TRL + PEFT)   |  | Manager           |        |
|  | (SmolLM2-1.7B) |  | (LoRA config)  |  | (save + resume)   |        |
|  +-------+--------+  +-------+--------+  +--------+----------+        |
|          |                    |                    |                   |
+----------+--------------------+--------------------+------------------+
           |                    |                    |
+----------v--------------------v--------------------v------------------+
|                     EVALUATION LAYER                                  |
|  +----------------+  +----------------+  +-------------------+        |
|  | Benchmark      |  | Custom Eval    |  | Comparison        |        |
|  | Runner         |  | Harness        |  | Reporter          |        |
|  | (lm-eval)      |  | (tool/code/    |  | (base vs tuned)   |        |
|  |                |  |  knowledge)    |  |                   |        |
|  +-------+--------+  +-------+--------+  +--------+----------+        |
|          |                    |                    |                   |
+----------+--------------------+--------------------+------------------+
           |                    |                    |
+----------v--------------------v--------------------v------------------+
|                     RELEASE LAYER                                     |
|  +----------------+  +----------------+  +-------------------+        |
|  | HuggingFace    |  | Model Card     |  | Reproduction      |        |
|  | Uploader       |  | Generator      |  | Scripts           |        |
|  | (datasets +    |  | (metrics,      |  | (end-to-end       |        |
|  |  weights)      |  |  license, etc) |  |  retraining)      |        |
|  +----------------+  +----------------+  +-------------------+        |
+-----------------------------------------------------------------------+
```

### Component Responsibilities

| Component | Responsibility | Typical Implementation |
|-----------|----------------|------------------------|
| Prompt Templates | Domain-specific prompt design for tool-call, code, and knowledge samples | YAML/JSON config files with Jinja2 templates, one set per focus area |
| Opus API Client | Manages Claude Opus calls for data generation, handles batching and rate limits | Python wrapper around Anthropic Message Batches API (50% cost savings) |
| Response Collector | Receives API responses and writes raw JSONL output | Async processor that maps custom_id back to prompt metadata |
| Format Validator | Validates each sample against ShareGPT schema, rejects malformed entries | JSON Schema validation + structural checks (role ordering, required fields) |
| Quality Filter | Scores samples on coherence, correctness, style diversity | Heuristic rules (length, repetition, formatting) + optional LLM-scored quality pass |
| Deduplication Engine | Removes exact and near-duplicate samples | Exact hash + MinHash LSH for fuzzy dedup (threshold ~0.7) |
| Domain Mixer | Enforces the 33/33/33 split across tool-call, code, and knowledge domains | Stratified sampling with per-domain quotas |
| ShareGPT Formatter | Converts validated samples into final ShareGPT conversation format | Standardizes to `{"conversations": [{"from": "human/gpt", "value": "..."}]}` |
| Split Generator | Creates train/validation/test splits with no data leakage | Stratified split preserving domain ratios (e.g., 90/5/5) |
| Model Loader | Loads SmolLM2-1.7B with appropriate quantization and tokenizer setup | `AutoModelForCausalLM.from_pretrained` with chat template configuration |
| SFTTrainer | Runs supervised fine-tuning with LoRA adapters | TRL `SFTTrainer` + PEFT `LoraConfig`, assistant-only loss masking |
| Checkpoint Manager | Saves model checkpoints, enables training resumption | HuggingFace Trainer checkpoint callbacks with configurable intervals |
| Benchmark Runner | Runs standard academic benchmarks (ARC, HellaSwag, MMLU, etc.) | EleutherAI `lm-evaluation-harness` with model loaded via transformers |
| Custom Eval Harness | Domain-specific evaluation for tool-call accuracy, code correctness, knowledge | Custom YAML tasks for lm-eval-harness + dedicated Python scoring scripts |
| Comparison Reporter | Generates side-by-side analysis of base vs. fine-tuned model | Markdown/HTML report with metric tables, sample comparisons, delta analysis |
| HuggingFace Uploader | Pushes datasets and model weights to HuggingFace Hub | `push_to_hub()` with proper dataset card and model card metadata |
| Model Card Generator | Creates standardized model cards with training details, metrics, license | Jinja2-templated markdown following HuggingFace model card spec |
| Reproduction Scripts | End-to-end scripts that reproduce training from scratch | Shell + Python scripts covering data download, training, evaluation |

## Recommended Project Structure

```
lyra/
|-- configs/                    # All configuration files
|   |-- generation/             # Prompt templates and generation configs
|   |   |-- tool_calls.yaml    # Tool-calling prompt templates + schemas
|   |   |-- code.yaml          # Code generation prompt templates
|   |   |-- knowledge.yaml     # General knowledge prompt templates
|   |   +-- batch_config.yaml  # Anthropic Batch API settings
|   |-- training/              # Training hyperparameters
|   |   |-- sft_config.yaml    # SFTTrainer arguments
|   |   +-- lora_config.yaml   # LoRA adapter configuration
|   +-- evaluation/            # Eval task definitions
|       |-- tool_call_eval.yaml
|       |-- code_eval.yaml
|       +-- knowledge_eval.yaml
|-- src/
|   |-- generation/            # Data generation pipeline
|   |   |-- __init__.py
|   |   |-- client.py          # Anthropic API client (batch + single)
|   |   |-- prompts.py         # Prompt builder from templates
|   |   |-- collector.py       # Response collection and raw JSONL writing
|   |   +-- schemas.py         # Tool-calling JSON schemas for training data
|   |-- curation/              # Data quality and validation
|   |   |-- __init__.py
|   |   |-- validator.py       # ShareGPT schema validation
|   |   |-- quality.py         # Quality scoring and filtering
|   |   |-- dedup.py           # Deduplication (exact + fuzzy)
|   |   +-- stats.py           # Dataset statistics and analysis
|   |-- assembly/              # Dataset assembly and formatting
|   |   |-- __init__.py
|   |   |-- formatter.py       # ShareGPT conversation formatter
|   |   |-- mixer.py           # Domain ratio enforcement (33/33/33)
|   |   +-- splitter.py        # Train/val/test split generation
|   |-- training/              # Fine-tuning pipeline
|   |   |-- __init__.py
|   |   |-- model.py           # Model loading and chat template setup
|   |   |-- trainer.py         # SFTTrainer wrapper with LoRA
|   |   +-- callbacks.py       # Custom training callbacks (logging, checkpoints)
|   |-- evaluation/            # Benchmarking and evaluation
|   |   |-- __init__.py
|   |   |-- benchmarks.py      # Standard benchmark runner (lm-eval-harness)
|   |   |-- custom_eval.py     # Custom domain-specific evaluation
|   |   +-- reporter.py        # Comparison report generation
|   +-- release/               # HuggingFace release tooling
|       |-- __init__.py
|       |-- uploader.py        # push_to_hub for datasets + models
|       +-- cards.py           # Model card and dataset card generation
|-- scripts/                   # Entry-point scripts
|   |-- generate.py            # Run data generation pipeline
|   |-- curate.py              # Run curation (validate + filter + dedup)
|   |-- assemble.py            # Assemble final dataset
|   |-- train.py               # Run fine-tuning
|   |-- evaluate.py            # Run evaluation suite
|   +-- release.py             # Push artifacts to HuggingFace
|-- data/                      # Local data directory (gitignored)
|   |-- raw/                   # Raw API responses
|   |-- curated/               # Post-validation, post-filtering
|   |-- datasets/              # Final assembled datasets
|   +-- checkpoints/           # Training checkpoints
|-- tests/                     # Test suite
|   |-- test_validator.py
|   |-- test_formatter.py
|   |-- test_quality.py
|   +-- test_dedup.py
|-- pyproject.toml
+-- README.md
```

### Structure Rationale

- **configs/:** Separated from code so prompt templates and hyperparameters can be iterated without touching source. Three subdirectories mirror the three pipeline stages that have tunable parameters.
- **src/generation/:** Isolated because it depends on the Anthropic API (external dependency) and runs independently of all downstream stages. Batch API logic is complex enough to warrant its own module.
- **src/curation/:** The quality gate between raw data and training data. Every sample passes through validator then quality filter then dedup in sequence. Keeping these as separate modules allows running them independently for debugging.
- **src/assembly/:** Transforms curated samples into the final training format. Separated from curation because the mixing ratio (33/33/33) and split strategy are policy decisions, not quality decisions.
- **src/training/:** Thin wrapper around TRL/PEFT. Kept minimal because the heavy lifting is in the HuggingFace libraries. The wrapper handles SmolLM2-specific quirks (chat template, token limits).
- **src/evaluation/:** Separate from training because evaluation runs against saved checkpoints, not during training. Custom evals for tool-calling accuracy need dedicated scoring logic.
- **src/release/:** Release tooling is a one-time pipeline stage that depends on all upstream stages being complete. Isolated so it can be run independently after manual review.
- **scripts/:** Each script is an entry point for one pipeline stage. This allows running stages independently (e.g., regenerate data without retraining) and composing them into a full pipeline.
- **data/:** Gitignored. Raw, curated, and assembled datasets live here during development. Clear subdirectories prevent accidental use of wrong-stage data.

## Architectural Patterns

### Pattern 1: Stage-Isolated Pipeline

**What:** Each pipeline stage (generate, curate, assemble, train, evaluate, release) operates independently, reading from disk and writing to disk. No in-memory coupling between stages.
**When to use:** Always for this project. The pipeline is iterative -- you will regenerate data many times before training, and retrain many times before releasing.
**Trade-offs:** Slower than an in-memory pipeline (disk I/O overhead), but enables: rerunning any stage in isolation, inspecting intermediate data, sharing data between collaborators, and resuming after failures.

**Example:**
```python
# scripts/generate.py -- reads config, writes raw JSONL
from src.generation.client import BatchClient
from src.generation.prompts import build_prompts

config = load_config("configs/generation/tool_calls.yaml")
prompts = build_prompts(config)
client = BatchClient(config)
client.submit_and_collect(prompts, output_path="data/raw/tool_calls_v1.jsonl")

# scripts/curate.py -- reads raw JSONL, writes curated JSONL
from src.curation.validator import validate_dataset
from src.curation.quality import filter_by_quality
from src.curation.dedup import deduplicate

raw = load_jsonl("data/raw/tool_calls_v1.jsonl")
valid = validate_dataset(raw)
quality = filter_by_quality(valid)
deduped = deduplicate(quality)
save_jsonl(deduped, "data/curated/tool_calls_v1.jsonl")
```

### Pattern 2: Config-Driven Prompt Templates

**What:** Prompt templates are YAML files, not hardcoded strings. Each domain (tool-call, code, knowledge) has its own template set with variation parameters for diversity.
**When to use:** Always. Prompt engineering is the highest-leverage activity in this project. Templates must be easy to iterate without code changes.
**Trade-offs:** Adds indirection (template rendering logic), but the ability to A/B test prompt variations by swapping config files is essential for data quality iteration.

**Example:**
```yaml
# configs/generation/tool_calls.yaml
domain: tool_calls
model: claude-opus-4-20250205
batch_size: 1000

templates:
  - name: function_calling_json
    system: |
      You are generating training data for a small language model.
      Generate a realistic multi-turn conversation where the assistant
      uses function calling to help the user.
    user_template: |
      Create a conversation where the user asks about {{ topic }}
      and the assistant calls the {{ function_name }} tool with
      appropriate arguments. The tool returns {{ result_type }}.
    variables:
      topics: [weather, file_operations, database_queries, ...]
      functions: [get_weather, read_file, query_db, ...]

  - name: mcp_tool_use
    system: |
      Generate training data showing MCP-style tool use patterns...
```

### Pattern 3: Anthropic Batch API with Async Collection

**What:** Data generation uses the Anthropic Message Batches API for 50% cost savings. Requests are submitted as JSONL batches (up to 100,000 requests per batch), results are polled and collected asynchronously.
**When to use:** Always for bulk data generation. Single-request API calls are only for prompt testing/debugging.
**Trade-offs:** 24-hour maximum processing window, no real-time feedback. But 50% cost reduction on Opus ($2.50/MTok input, $12.50/MTok output) is significant at 5K+ sample scale. Batch pricing stacks with prompt caching for even deeper savings.

**Example:**
```python
# src/generation/client.py
import anthropic

class BatchClient:
    def __init__(self, config):
        self.client = anthropic.Anthropic()
        self.model = config["model"]

    def submit_batch(self, requests: list[dict]) -> str:
        batch = self.client.messages.batches.create(
            requests=[
                {
                    "custom_id": req["id"],
                    "params": {
                        "model": self.model,
                        "max_tokens": req.get("max_tokens", 2048),
                        "system": req["system"],
                        "messages": [{"role": "user", "content": req["prompt"]}]
                    }
                }
                for req in requests
            ]
        )
        return batch.id

    def poll_until_complete(self, batch_id: str) -> list[dict]:
        # Poll batch status, collect results when done
        ...
```

### Pattern 4: ShareGPT Conversation Format with Tool Extensions

**What:** All training data uses ShareGPT multi-turn conversation format. Tool-calling samples extend the format with tool_calls and tool role messages following HuggingFace TRL conventions.
**When to use:** For all three domains. Even code and knowledge samples use multi-turn format for consistency.
**Trade-offs:** ShareGPT format is less flexible than raw text, but TRL's `SFTTrainer` has native support for it, including `assistant_only_loss=True` which ensures loss is computed only on assistant responses.

**Example:**
```json
{
  "conversations": [
    {"from": "system", "value": "You are a helpful assistant with access to tools."},
    {"from": "human", "value": "What is the weather in San Francisco?"},
    {"from": "gpt", "value": "", "tool_calls": [
      {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "San Francisco"}}}
    ]},
    {"from": "tool", "name": "get_weather", "value": "{\"temp\": 62, \"condition\": \"foggy\"}"},
    {"from": "gpt", "value": "The weather in San Francisco is 62F and foggy."}
  ],
  "tools": [
    {"type": "function", "function": {"name": "get_weather", "description": "Get current weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
  ]
}
```

### Pattern 5: LoRA Adapter Training (Not Full Fine-Tuning)

**What:** Use LoRA (Low-Rank Adaptation) via PEFT instead of full fine-tuning. Only a small set of adapter parameters are trained while the base model weights are frozen.
**When to use:** Always for this project. SmolLM2-1.7B is small enough that full fine-tuning is possible on consumer hardware, but LoRA provides faster iteration cycles, lower memory footprint, and easier A/B testing of different adapter configurations.
**Trade-offs:** Slight potential quality gap vs. full fine-tuning (typically <2% on benchmarks), but dramatically faster training (minutes vs. hours on a single GPU) and the ability to maintain multiple adapters for different versions.

## Data Flow

### End-to-End Pipeline Flow

```
[Prompt Templates (YAML)]
    |
    v
[Prompt Builder] --> [Batch of prompts with metadata]
    |
    v
[Anthropic Batch API] --> [Raw JSONL responses (data/raw/)]
    |
    v
[Format Validator] --> rejects malformed --> [error log]
    |
    v (valid samples only)
[Quality Filter] --> rejects low-quality --> [rejected log with reasons]
    |
    v (quality samples only)
[Deduplication Engine] --> removes duplicates --> [dedup stats]
    |
    v (unique, quality samples)
[Domain Mixer] --> enforces 33/33/33 ratio
    |
    v
[ShareGPT Formatter] --> standardizes conversation format
    |
    v
[Split Generator] --> [train.jsonl, val.jsonl, test.jsonl in data/datasets/]
    |
    v
[Model Loader] --> SmolLM2-1.7B + tokenizer + chat template
    |
    v
[SFTTrainer + LoRA] --> [checkpoints in data/checkpoints/]
    |
    v
[Benchmark Runner] --> standard benchmarks (ARC, HellaSwag, MMLU)
    |
    v
[Custom Eval Harness] --> domain-specific evals (tool accuracy, code pass@k, knowledge F1)
    |
    v
[Comparison Reporter] --> [eval_report.md with base vs. tuned metrics]
    |
    v
[HuggingFace Uploader] --> datasets repo + model repo + scripts repo
```

### Key Data Flows

1. **Generation flow:** Prompt templates are expanded with variable substitution, batched into Anthropic API requests, and raw responses are collected as JSONL. Each response retains its `custom_id` linking back to the prompt template and domain. This is the most expensive flow (API costs) and should be designed for maximum reuse.

2. **Curation flow:** Raw JSONL is read sequentially through three filters: schema validation (structural correctness), quality scoring (content quality), and deduplication (uniqueness). Each stage produces a log of rejected samples with reasons, enabling prompt template improvement. The curation pipeline is idempotent -- rerunning it on the same raw data produces identical output.

3. **Assembly flow:** Curated samples from all three domains are mixed to the target ratio (33/33/33), formatted into final ShareGPT structure, and split into train/val/test. The mixer pulls from each domain's curated pool and stops when the smallest pool is exhausted (to maintain ratios). Versioned dataset directories (v1, v2, ...) support iterative improvement.

4. **Training flow:** The assembled dataset is loaded, tokenized with the SmolLM2 chat template, and fed to SFTTrainer with LoRA adapters. Training computes loss only on assistant tokens (`assistant_only_loss=True`). Checkpoints are saved at configurable intervals. The training flow is the fastest to iterate (minutes on a single GPU with LoRA).

5. **Evaluation flow:** Each checkpoint is evaluated against both standard benchmarks (via lm-evaluation-harness) and custom domain evals. Results are compared against the base SmolLM2-1.7B-Instruct model. The evaluation flow produces a structured report that informs whether to scale up data generation, adjust prompt templates, or modify training hyperparameters.

6. **Release flow:** Final model weights (LoRA adapters merged back into base model), datasets, model cards, dataset cards, and reproduction scripts are pushed to HuggingFace Hub. This is a one-directional, terminal flow.

### Data Format at Each Stage

| Stage | Format | Location |
|-------|--------|----------|
| Raw responses | JSONL (one API response per line, includes metadata) | `data/raw/{domain}_v{N}.jsonl` |
| Curated samples | JSONL (validated, quality-scored, deduplicated) | `data/curated/{domain}_v{N}.jsonl` |
| Assembled dataset | JSONL in ShareGPT format (train/val/test splits) | `data/datasets/v{N}/train.jsonl` |
| Training input | HuggingFace Dataset (tokenized, with input_ids and labels) | In-memory, loaded from assembled JSONL |
| Checkpoints | Safetensors (LoRA adapter weights) | `data/checkpoints/v{N}/checkpoint-{step}/` |
| Eval results | JSON (metric scores per benchmark) | `data/eval/v{N}/results.json` |
| Released artifacts | HuggingFace Hub repos (parquet datasets, safetensors weights) | `huggingface.co/{org}/lyra-*` |

## Scaling Considerations

| Scale | Architecture Adjustments |
|-------|--------------------------|
| ~5K samples (Phase 1) | Single-machine pipeline. Batch API handles generation. Training on 1 GPU (even consumer-grade with LoRA). lm-eval-harness runs locally. Total pipeline: hours not days. |
| ~25K samples (Phase 2) | Same architecture. Batch API scales naturally (up to 100K requests/batch). Training time increases linearly but stays under a few hours with LoRA. May need to shard raw data files for easier management. |
| ~100K+ samples (Phase 3) | Consider parallelizing curation across multiple processes (dedup is the bottleneck). May want to switch from local JSONL to HuggingFace Datasets format earlier in the pipeline for streaming. Training may benefit from multi-GPU or full fine-tuning at this scale. |

### Scaling Priorities

1. **First bottleneck: Data generation cost.** At 5K samples with Opus Batch API, expect roughly $5-15 depending on prompt/response length. At 100K samples, this becomes $100-300+. Prompt caching + batch pricing (stacked) is the primary mitigation. The architecture supports this natively through the BatchClient design.

2. **Second bottleneck: Curation quality loop.** As dataset grows, the time to inspect, validate, and iterate on data quality increases. The stage-isolated pipeline helps here -- you can re-curate without regenerating. The rejected sample logs are the key feedback mechanism.

3. **Third bottleneck: Evaluation breadth.** Custom evaluation tasks (especially code execution via pass@k) require sandboxed execution environments. At scale, this becomes the slowest evaluation step. Start with simple metrics (exact match, F1) and add sandboxed code execution only when the model shows meaningful code generation ability.

## Anti-Patterns

### Anti-Pattern 1: Monolithic Generate-and-Train Script

**What people do:** Single script that generates data, formats it, and immediately trains on it.
**Why it is wrong:** Cannot inspect intermediate data quality. Cannot rerun training without regenerating (expensive). Cannot debug whether poor model performance is caused by bad data or bad training config. Couples the most expensive stage (API calls) to the fastest stage (training).
**Do this instead:** Stage-isolated pipeline with disk boundaries between stages. Each stage reads input from disk and writes output to disk. This is the core architectural principle.

### Anti-Pattern 2: Generating All Data Before Any Training

**What people do:** Generate the full 5K dataset, curate it, then start training.
**Why it is wrong:** You do not know if your prompt templates produce useful training data until you train and evaluate. Generating all data upfront risks wasting API budget on prompts that produce poor-quality samples.
**Do this instead:** Start with 100-200 samples per domain, run a quick training cycle, evaluate, then scale up the prompts that work. The architecture supports this by versioning data directories (v1, v2, ...).

### Anti-Pattern 3: Training on All Tokens (Not Assistant-Only)

**What people do:** Train the model to predict every token in the conversation, including user prompts and system messages.
**Why it is wrong:** The model wastes capacity learning to generate user messages it will never need to produce. This dilutes the training signal on the responses that actually matter.
**Do this instead:** Use `assistant_only_loss=True` in SFTConfig. This masks loss computation to only assistant/gpt turns. TRL supports this natively for conversational datasets.

### Anti-Pattern 4: Skipping Deduplication

**What people do:** Trust that LLM-generated data is diverse enough to not need dedup.
**Why it is wrong:** LLMs have strong mode-collapse tendencies, especially with similar prompts. Research shows synthetic datasets can contain 10-30% near-duplicates. Duplicate samples cause the model to overfit on repeated patterns.
**Do this instead:** Run both exact-match dedup (fast, catches identical outputs) and fuzzy dedup via MinHash LSH (catches paraphrased duplicates). Threshold of ~0.7 Jaccard similarity is standard.

### Anti-Pattern 5: Evaluating Only on Standard Benchmarks

**What people do:** Run ARC/HellaSwag/MMLU and declare success.
**Why it is wrong:** Standard benchmarks do not measure the specific capabilities Lyra targets: structured tool calling (JSON format compliance, argument correctness), practical code generation (compilable, correct output), and adaptive response style (terse for code, detailed for reasoning). A model can score well on MMLU but fail to produce valid JSON tool calls.
**Do this instead:** Build custom evaluation tasks for each domain. Tool-call eval checks JSON validity, schema compliance, and argument correctness. Code eval checks syntax validity and pass@k on test cases. Knowledge eval checks factual accuracy and reasoning chain quality.

## Integration Points

### External Services

| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| Anthropic API (Opus) | Message Batches API for bulk generation, Messages API for single-prompt testing | Use batch for production generation (50% cost saving). Use single-message for prompt development and debugging. Batch limit: 100K requests or 256 MB per batch. Results available within 1-24 hours. |
| HuggingFace Hub | `push_to_hub()` via `datasets` and `transformers` libraries | Three repos needed: dataset repo, model weights repo, scripts/reproduction repo. Use safetensors format for weights. |
| Weights and Biases | `report_to="wandb"` in SFTConfig training arguments | Track training loss, learning rate, gradient norms, token accuracy. Essential for comparing training runs across dataset versions. |
| lm-evaluation-harness | CLI invocation or Python API against saved model checkpoints | Custom tasks defined as YAML configs in `configs/evaluation/`. Standard tasks (arc, hellaswag, mmlu) via built-in task registry. |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| Generation -> Curation | JSONL files on disk (`data/raw/`) | Each line is a self-contained sample with metadata. No shared state. Generation can run multiple times, curation re-processes everything. |
| Curation -> Assembly | JSONL files on disk (`data/curated/`) | Curated files are per-domain. Assembly reads all domains and mixes them. Domain ratios are enforced at assembly time, not curation time. |
| Assembly -> Training | JSONL files on disk (`data/datasets/`) | Training reads the assembled dataset via HuggingFace `load_dataset("json", ...)`. The dataset is tokenized in memory during training. |
| Training -> Evaluation | Checkpoint directories on disk (`data/checkpoints/`) | Evaluation loads saved checkpoints. No coupling to the training process. Can evaluate any checkpoint independently. |
| Evaluation -> Generation (feedback loop) | Human review of eval reports | The eval report informs prompt template changes. This is a manual step -- the architecture does not automate prompt improvement. The rejected sample logs from curation are the primary feedback signal. |

## Suggested Build Order

Based on component dependencies, the recommended implementation order is:

1. **ShareGPT Format Validator + Formatter (curation + assembly)** -- Build this first because it defines the contract that all other components must satisfy. Every sample in the pipeline ultimately passes through validation and formatting. Getting the schema right early prevents rework.

2. **Single-Request Generation Client + Prompt Templates (generation)** -- Start with single API calls (not batch) to iterate on prompt templates quickly. Build the template system and a few initial prompts per domain. Generate 50-100 samples to validate the format pipeline.

3. **Quality Filter + Dedup (curation)** -- Once you have raw samples flowing, add quality filtering and deduplication. Start with simple heuristic rules (min/max length, repetition detection) and add more filters as you discover failure modes.

4. **Training Pipeline (training)** -- Wire up SFTTrainer with LoRA on the small curated dataset. This validates end-to-end: data format -> tokenization -> training -> checkpoint. Use a minimal config (few steps) just to prove the pipeline works.

5. **Evaluation Harness (evaluation)** -- Build custom eval tasks and wire up lm-eval-harness. Evaluate the first training run against the base model. This closes the feedback loop and tells you if your data is working.

6. **Batch API Client (generation)** -- Once prompts are validated and the pipeline works end-to-end, switch to Batch API for cost-efficient bulk generation. Scale from 100 to 5K samples.

7. **Domain Mixer + Split Generator (assembly)** -- Formalize the 33/33/33 mixing and train/val/test splitting once you have enough data across all three domains.

8. **Release Pipeline (release)** -- Build last, after at least one successful training-evaluation cycle. Model cards, dataset cards, and reproduction scripts are written once the pipeline is stable.

## Sources

- [HuggingFace TRL SFTTrainer documentation](https://huggingface.co/docs/trl/en/sft_trainer) -- Dataset format, assistant-only loss, LoRA/PEFT integration, tool calling support (HIGH confidence)
- [SmolLM2 paper (arXiv 2502.02737)](https://arxiv.org/html/2502.02737v1) -- SmolLM2 architecture (Llama-based, 2048 seq len), multi-stage training, data curation approach (HIGH confidence)
- [Fine-tuning SmolLM2 on custom synthetic data](https://huggingface.co/blog/prithivMLmods/smollm2-ft) -- End-to-end fine-tuning pipeline with SFTTrainer, training configuration patterns (MEDIUM confidence)
- [Fine-tuning with tool calling (Stephen Diehl)](https://www.stephendiehl.com/posts/fine_tuning_tools/) -- Tool call dataset format, TRL 0.19.0+ tool calling support, JSON schema generation (MEDIUM confidence)
- [Anthropic Batch Processing documentation](https://platform.claude.com/docs/en/build-with-claude/batch-processing) -- Batch API limits (100K requests, 256 MB), pricing (50% discount), polling model (HIGH confidence)
- [EleutherAI lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) -- Custom task YAML format, standard benchmark support, model loading via transformers (HIGH confidence)
- [NVIDIA NeMo Curator](https://developer.nvidia.com/blog/scale-and-curate-high-quality-datasets-for-llm-training-with-nemo-curator/) -- Data curation pipeline components: filtering, deduplication, quality scoring (MEDIUM confidence)
- [Synthetic Data Generation survey (arXiv 2503.14023)](https://arxiv.org/html/2503.14023v2) -- Synthetic data generation patterns, evaluation frameworks, LLM-as-generator approaches (MEDIUM confidence)
- [HuggingFace Model Release Checklist](https://huggingface.co/docs/hub/en/model-release-checklist) -- Release best practices, safetensors format, model card requirements (HIGH confidence)
- [Distil Labs SLM Benchmarking](https://www.distillabs.ai/blog/we-benchmarked-12-small-language-models-across-8-tasks-to-find-the-best-base-model-for-fine-tuning/) -- Small model fine-tuning benchmarks, 10K sample generation approach (MEDIUM confidence)

---
*Architecture research for: Dataset curation and small LLM fine-tuning (Lyra project)*
*Researched: 2026-04-20*
