# Phase 3: Evaluation Framework - Research

**Researched:** 2026-04-20
**Domain:** LLM evaluation tooling -- lm-eval-harness, BFCL, evalplus/bigcode, unified CLI orchestration
**Confidence:** HIGH

## Summary

Phase 3 builds a thin orchestration layer (eval_runner) that invokes three standard benchmark suites -- lm-eval-harness for general knowledge (MMLU, ARC, HellaSwag), BFCL for tool-calling accuracy, and evalplus/bigcode for code generation (HumanEval, MBPP) -- then aggregates results into per-category JSON. The user decisions lock us into standard benchmark suites only (no custom tests), a single-script CLI with argparse, MPS-primary/CPU-fallback device strategy, and a separate compare command for base-vs-fine-tuned delta reporting.

The primary challenge is BFCL integration on Apple Silicon. BFCL requires vLLM or sglang backends for model inference, neither of which supports MPS. The recommended approach is a two-phase BFCL flow: generate model responses via plain transformers inference (our code, MPS-compatible), then feed those responses to BFCL's evaluation-only command which uses AST-based checking and needs no GPU.

lm-eval-harness has a clean Python API (`simple_evaluate()`) with explicit MPS device support. evalplus supports an `hf` (transformers) backend that should work on MPS via standard PyTorch device auto-detection. Both are the straightforward path.

**Primary recommendation:** Use lm-eval-harness Python API directly for general knowledge, evalplus CLI with `--backend hf` for code, and a custom BFCL integration that separates generation (transformers on MPS) from evaluation (BFCL AST checker). Wrap all three behind a unified `eval_runner.py` CLI following the curate_pipeline.py argparse pattern.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- D-01: Single script with CLI flags -- `python3 -m scripts.eval_runner --model path/to/model --benchmarks tool-calling,code,knowledge --output results.json`. Consistent with curate_pipeline.py argparse pattern.
- D-02: Device support: MPS (Apple Silicon) primary, CPU fallback. No CUDA support. Auto-detect MPS availability, fall back to CPU if unavailable.
- D-03: Model loading approach is Claude's discretion -- determine cleanest separation of inference logic from eval logic.
- D-04: Use ONLY widely-approved, standard benchmark suites. No custom test cases or proprietary fixtures.
- D-05: Benchmark mapping by category: Tool calling = BFCL, Code = HumanEval/MBPP via bigcode-evaluation-harness, General knowledge = MMLU/ARC/HellaSwag via lm-eval-harness.
- D-06: Our value-add is the unified runner that invokes standard suites and aggregates results per-category. We are a thin orchestration layer, not a benchmark creator.
- D-07: JSON output file: `results/{model_name}_{timestamp}.json` with structured scores per benchmark per category. Machine-readable, version-controllable.
- D-08: CLI prints a summary table to stdout after eval completes -- quick visual feedback showing category, benchmark, and scores.
- D-09: Run eval twice (once on base SmolLM2-1.7B, once on fine-tuned), producing separate JSON result files.
- D-10: Separate compare command reads two JSON files and prints delta table. No built-in history tracking -- simple file-based comparison.
- D-11: Compare output format is Claude's discretion.

### Claude's Discretion
- Model loading architecture (direct transformers vs inference wrapper module)
- Compare command output format (terminal table, JSON diff, or hybrid)
- Which specific lm-eval-harness tasks to include for MMLU/ARC/HellaSwag (task names, few-shot settings)
- Summary table formatting (tabulate, rich, or plain print)
- Results directory structure

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| EVAL-03 | Custom eval benchmarks measuring tool-call format compliance, argument extraction accuracy, and code correctness | BFCL provides AST-based tool-call evaluation (format compliance + argument accuracy); evalplus provides HumanEval+/MBPP+ for code correctness with 80x more tests than vanilla HumanEval. Both are "standard benchmark suites" per D-04 -- BFCL IS the standard tool-call format compliance benchmark; evalplus IS the standard code correctness benchmark. |
| EVAL-04 | Per-category quality metrics reported separately for tool calls, code, and general knowledge | Eval runner aggregates results from three separate benchmark suites into a single JSON with per-category sections. lm-eval-harness returns results keyed by task name; BFCL returns per-category scores; evalplus returns pass@k per dataset. |
</phase_requirements>

## Project Constraints (from CLAUDE.md)

- **Base model:** SmolLM2-1.7B -- all evaluation targets this architecture
- **No emojis** in terminal output, logs, or files
- **Flat scripts/ directory** with standalone Python files -- eval_runner goes here
- **argparse CLI** entry points -- follow curate_pipeline.py pattern
- **Pydantic** for structured data validation -- use for eval config and result schemas
- **YAML** for configuration files -- follow configs/pipeline.yaml pattern
- **pytest** in tests/ directory -- follow existing test patterns
- **pathlib.Path** for all file operations
- **yaml.safe_load** only (never yaml.load)

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| lm-eval | 0.4.11 | General knowledge benchmarks (MMLU, ARC, HellaSwag) | Industry standard; powers HuggingFace Open LLM Leaderboard; Python API with MPS device support [VERIFIED: PyPI, release 2026-02-13] |
| evalplus | 0.3.1 | Code generation benchmarks (HumanEval+, MBPP+) | 80x more tests than vanilla HumanEval; `hf` backend works without vLLM; NeurIPS 2023 + COLM 2024 [VERIFIED: PyPI, release 2024-10-20] |
| bfcl-eval | 2026.3.23 | Tool calling evaluation (AST-based format compliance + argument accuracy) | Berkeley BFCL v4; industry standard for function calling; AST evaluation needs no function execution [VERIFIED: PyPI, release 2026-03-23] |
| transformers | 5.5.4 | Model loading and inference | Already installed in project; required for model loading [VERIFIED: pip3 list on local machine] |
| torch | 2.10+ | PyTorch backend with MPS support | Required for inference; 2.10+ has Python 3.14 support and stable MPS [VERIFIED: PyTorch blog, 2026-01-21] |
| pydantic | 2.12.5 | Result schema validation | Already in requirements.txt; use for EvalResult and CompareResult models [VERIFIED: requirements.txt] |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| accelerate | 1.13.0 | Device management and model loading helpers | Required by evalplus and lm-eval for device placement [CITED: CLAUDE.md tech stack] |
| pyyaml | 6.0+ | Eval config file parsing | Already in requirements.txt; for configs/eval.yaml [VERIFIED: requirements.txt] |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| evalplus (code eval) | bigcode-evaluation-harness directly | bigcode-evaluation-harness is CLI-only (no Python API), uses accelerate launch, and has no explicit MPS documentation. evalplus wraps HumanEval+/MBPP+ with cleaner interface and hf backend. |
| Plain print for summary tables | tabulate or rich library | tabulate adds a dependency for nicer formatting but plain f-string print follows existing curate_pipeline.py pattern. Recommend plain print for consistency with zero new deps for display. |

**Installation:**
```bash
pip install "lm-eval[hf]==0.4.11" "evalplus>=0.3.1" "bfcl-eval==2026.3.23" torch>=2.10 accelerate>=1.13.0
```

**Version verification:**
- lm-eval 0.4.11: Published 2026-02-13 [VERIFIED: PyPI page]
- evalplus 0.3.1: Published 2024-10-20 [VERIFIED: PyPI page]
- bfcl-eval 2026.3.23: Published 2026-03-23 [VERIFIED: PyPI page]
- torch: Let version resolve based on Python 3.14 compatibility -- PyTorch 2.10+ supports 3.14 [VERIFIED: PyTorch 2.10 release blog]

## Architecture Patterns

### Recommended Project Structure
```
scripts/
    eval_runner.py       # Main CLI: --model --benchmarks --output
    eval_compare.py      # Compare CLI: --baseline --candidate --output
    eval_config.py       # Pydantic models for eval config + result schemas
configs/
    eval.yaml            # Default eval configuration (task names, few-shot, batch_size)
results/                 # Output directory (gitignored)
    {model_name}_{timestamp}.json
tests/
    test_eval_runner.py  # Unit tests for eval orchestration logic
    test_eval_compare.py # Unit tests for comparison logic
    test_eval_config.py  # Unit tests for config/schema validation
```

### Pattern 1: Model Loading Wrapper (Claude's Discretion recommendation)

**What:** Separate model loading into a thin utility function in eval_runner.py rather than a separate module.
**When to use:** Always -- keeps things simple in the flat scripts/ pattern while cleanly separating model loading from benchmark invocation.
**Why:** Each benchmark suite has its own model loading mechanism: lm-eval uses HFLM, evalplus uses --model flag, BFCL uses custom handlers. The eval_runner does not need a shared model object -- it passes a model path to each suite. The model path string is the shared interface.

```python
# Source: Pattern derived from lm-eval-harness Python API docs
# [CITED: github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/python-api.md]

def detect_device() -> str:
    """Auto-detect best available device: MPS > CPU (no CUDA per D-02)."""
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
```

### Pattern 2: Three-Suite Orchestration

**What:** Eval runner invokes each benchmark suite independently and collects results.
**When to use:** Core pattern for eval_runner.py.

```python
# Source: lm-eval Python API docs + evalplus CLI docs + BFCL architecture
# [CITED: github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/python-api.md]

import lm_eval

def run_knowledge_benchmarks(model_path: str, device: str, config: dict) -> dict:
    """Run MMLU, ARC, HellaSwag via lm-eval-harness Python API."""
    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=f"pretrained={model_path},dtype=float32",
        tasks=config.get("knowledge_tasks", ["mmlu", "arc_challenge", "hellaswag"]),
        device=device,
        batch_size=config.get("batch_size", "auto"),
        num_fewshot=config.get("num_fewshot", 5),
    )
    return results["results"]
```

```python
# Source: evalplus docs
# [CITED: github.com/evalplus/evalplus]

import subprocess

def run_code_benchmarks(model_path: str, config: dict) -> dict:
    """Run HumanEval+/MBPP+ via evalplus CLI with hf backend."""
    results = {}
    for dataset in config.get("code_datasets", ["humaneval", "mbpp"]):
        cmd = [
            "evalplus.evaluate",
            "--model", model_path,
            "--dataset", dataset,
            "--backend", "hf",
            "--greedy",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        # Parse evalplus output for pass@k metrics
        results[dataset] = parse_evalplus_output(result.stdout)
    return results
```

```python
# Source: BFCL architecture -- generate + evaluate separately
# [CITED: github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard]

def run_tool_calling_benchmarks(model_path: str, device: str, config: dict) -> dict:
    """Run BFCL evaluation: generate responses via transformers, evaluate via BFCL AST."""
    # Step 1: Generate model responses using plain transformers inference
    responses = generate_bfcl_responses(model_path, device, config)
    
    # Step 2: Write responses to BFCL-expected directory structure
    write_bfcl_responses(responses, model_path, config)
    
    # Step 3: Run BFCL evaluate command on pre-generated responses
    cmd = ["bfcl", "evaluate", "--model", model_name, "--test-category", "all"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return parse_bfcl_output(result.stdout)
```

### Pattern 3: Result Schema (Pydantic)

**What:** Typed result model for JSON output per D-07.
**When to use:** All eval output.

```python
# Source: Pattern derived from existing Pydantic usage in validate_format.py
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class BenchmarkResult(BaseModel):
    """Score for a single benchmark."""
    benchmark: str
    metric: str
    score: float
    num_fewshot: Optional[int] = None

class CategoryResult(BaseModel):
    """Scores for all benchmarks in one category."""
    category: str
    benchmarks: list[BenchmarkResult]

class EvalResult(BaseModel):
    """Complete evaluation result for one model."""
    model_path: str
    model_name: str
    timestamp: str
    device: str
    categories: list[CategoryResult]
```

### Pattern 4: Compare Command

**What:** Read two result JSON files, compute deltas, print table.
**When to use:** D-10 compare command.

```python
# Recommendation for Claude's Discretion D-11: hybrid approach
# Print a plain-text table to stdout + optionally write JSON diff to file

def compare_results(baseline_path: Path, candidate_path: Path) -> None:
    baseline = EvalResult.model_validate_json(baseline_path.read_text())
    candidate = EvalResult.model_validate_json(candidate_path.read_text())
    # Match benchmarks by name, compute score deltas
    # Print formatted table with columns: Category | Benchmark | Baseline | Candidate | Delta
```

### Anti-Patterns to Avoid
- **Loading the model once and sharing across suites:** Each suite has its own model loading. Do not pre-load a transformers model and try to inject it into lm-eval or evalplus -- they manage their own model lifecycle.
- **Running BFCL generate with vLLM/sglang on MPS:** These backends require CUDA. Use plain transformers for generation, BFCL evaluate for scoring.
- **Parsing stdout with fragile regex:** Prefer structured output (JSON files) from each suite where available. lm-eval returns a Python dict directly; evalplus writes JSONL results to a known directory; BFCL writes score JSON files.
- **Using FP16 on MPS:** Apple Silicon MPS does not support FP16. Always use float32 (or bfloat16 if supported) on MPS device. [CITED: huggingface.co/docs/transformers/en/perf_train_special]

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Tool-call format compliance scoring | Custom JSON schema validators for function calls | BFCL's AST-based evaluation | BFCL handles edge cases (parameter ordering, type coercion, nested objects) that hand-rolled JSON matching misses. AST evaluation is the industry standard approach. |
| Code correctness testing | Custom test harness that runs generated code against test cases | evalplus HumanEval+/MBPP+ | evalplus has 80x more tests than vanilla HumanEval, handles sandboxed code execution safely, and produces standard pass@k metrics. |
| General knowledge scoring | Custom MMLU/ARC question runners | lm-eval-harness simple_evaluate() | lm-eval handles prompt formatting, few-shot construction, answer extraction, and metric computation for 200+ tasks. |
| Device detection logic | Complex GPU/MPS/CPU detection with fallback chains | PyTorch's torch.backends.mps.is_available() | Two-line check: MPS available? Use it. Otherwise CPU. Per D-02, no CUDA support needed. |

**Key insight:** The entire value of this phase is orchestration -- calling standard tools and aggregating their output. Every benchmark calculation should come from an established library.

## Common Pitfalls

### Pitfall 1: BFCL Requires GPU Inference Backends
**What goes wrong:** BFCL's `bfcl generate` command requires vLLM or sglang, which require CUDA GPUs. Running `bfcl generate` on Apple Silicon fails.
**Why it happens:** BFCL is designed for leaderboard submissions on GPU clusters, not local Mac evaluation.
**How to avoid:** Split BFCL into two steps: (1) Generate model responses using plain transformers inference with MPS/CPU (our code), (2) Feed pre-generated response files to `bfcl evaluate` which only does AST-based comparison and needs no GPU. Set `BFCL_PROJECT_ROOT` environment variable to control where BFCL looks for result files. [VERIFIED: BFCL README confirms evaluate can run on pre-generated responses]
**Warning signs:** Import errors from vLLM/sglang when BFCL tries to start an inference server.

### Pitfall 2: MPS Float16 Unsupported
**What goes wrong:** Model loading with `dtype=float16` crashes or produces garbage results on MPS.
**Why it happens:** Apple Silicon MPS backend does not support FP16 operations. [CITED: huggingface.co/docs/transformers/en/perf_train_special]
**How to avoid:** Always use `dtype=float32` when device is MPS. Set `PYTORCH_ENABLE_MPS_FALLBACK=1` environment variable to fall back to CPU for any operations not yet implemented in MPS.
**Warning signs:** NaN values in model outputs, RuntimeError about unsupported operations on MPS.

### Pitfall 3: lm-eval-harness MPS Correctness Issues
**What goes wrong:** Some lm-eval tasks produce slightly different scores on MPS vs CPU/CUDA due to the MPS backend being in earlier maturity stage.
**Why it happens:** PyTorch MPS backend still has edge cases with certain operations. The lm-eval docs explicitly warn about this. [CITED: lm-eval-harness GitHub -- "MPS back-end is still in early stages of development"]
**How to avoid:** For the first run, verify that a forward pass on `--device cpu` and `--device mps` produce matching results. If they diverge, fall back to CPU for that task. Document any MPS-specific discrepancies in eval output.
**Warning signs:** Evaluation scores that differ significantly from published SmolLM2-1.7B benchmark results.

### Pitfall 4: evalplus Code Execution Security
**What goes wrong:** evalplus executes model-generated Python code to check correctness. Malicious or buggy code could affect the local system.
**Why it happens:** Code evaluation inherently requires execution.
**How to avoid:** evalplus uses sandboxing by default. Do not disable it. Run eval in a disposable environment if concerned. The `--greedy` flag ensures deterministic single-sample generation (no sampling randomness).
**Warning signs:** Long-running evaluation tasks (infinite loops in generated code).

### Pitfall 5: Python 3.14 Compatibility
**What goes wrong:** Some packages may not fully support Python 3.14 yet. The local environment runs Python 3.14.2.
**Why it happens:** Python 3.14 is very new. PyTorch 2.10+ supports it, but downstream packages may lag.
**How to avoid:** Install packages and verify import before writing integration code. If a package fails on 3.14, use a Python 3.12 virtual environment as fallback. evalplus 0.3.1 was released Oct 2024 (before 3.14 existed) -- test compatibility early.
**Warning signs:** Import errors, setuptools build failures during pip install.

### Pitfall 6: SmolLM2-1.7B Memory on MPS
**What goes wrong:** Loading SmolLM2-1.7B in float32 requires ~6.8GB RAM (1.7B params * 4 bytes). On machines with limited unified memory, MPS allocation may fail.
**Why it happens:** float32 is the only reliable dtype on MPS.
**How to avoid:** Use batch_size=1 for inference when memory is tight. lm-eval's `batch_size="auto"` should handle this. Monitor with `torch.mps.current_allocated_memory()`.
**Warning signs:** MPS out-of-memory errors, system slowdown during eval.

## Code Examples

### lm-eval-harness Python API -- Verified Pattern
```python
# Source: lm-eval-harness Python API docs
# [CITED: github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/python-api.md]

import lm_eval

results = lm_eval.simple_evaluate(
    model="hf",
    model_args="pretrained=HuggingFaceTB/SmolLM2-1.7B-Instruct,dtype=float32",
    tasks=["mmlu", "arc_challenge", "hellaswag"],
    device="mps",  # or "cpu"
    batch_size="auto",
    num_fewshot=5,
)

# Results structure:
# results["results"]["mmlu"]["acc,none"] -> float
# results["results"]["arc_challenge"]["acc_norm,none"] -> float
# results["results"]["hellaswag"]["acc_norm,none"] -> float
```

### evalplus CLI -- Verified Pattern
```bash
# Source: evalplus GitHub README
# [CITED: github.com/evalplus/evalplus]

# Generate + evaluate in one step
evalplus.evaluate \
    --model HuggingFaceTB/SmolLM2-1.7B-Instruct \
    --dataset humaneval \
    --backend hf \
    --greedy

# Results written to: evalplus_results/humaneval/SmolLM2-1.7B-Instruct_hf_temp_0.0.jsonl
```

### BFCL Two-Phase Pattern -- Architecture Design
```python
# Source: BFCL README + custom integration design
# [CITED: github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/README.md]

import os
import subprocess
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

def generate_bfcl_responses(model_path: str, device: str) -> None:
    """Generate responses to BFCL test prompts using plain transformers."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype="auto"
    ).to(device)
    
    # Load BFCL test data and generate responses
    # Write to: $BFCL_PROJECT_ROOT/result/{model_name}/BFCL_v3_{category}_result.json
    ...

def evaluate_bfcl_responses(model_name: str) -> dict:
    """Run BFCL AST evaluation on pre-generated responses."""
    os.environ["BFCL_PROJECT_ROOT"] = str(Path("results/bfcl"))
    cmd = ["bfcl", "evaluate", "--model", model_name, "--test-category", "all"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    # Parse score files from $BFCL_PROJECT_ROOT/score/{model_name}/
    return parse_bfcl_scores(model_name)
```

### Device Detection -- Verified Pattern
```python
# Source: PyTorch MPS documentation + HuggingFace Apple Silicon docs
# [CITED: developer.apple.com/metal/pytorch/]

import os
import torch

def detect_device() -> str:
    """Auto-detect best available device per D-02 (MPS > CPU, no CUDA)."""
    if torch.backends.mps.is_available():
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        return "mps"
    return "cpu"
```

## lm-eval-harness Task Names (Claude's Discretion Recommendation)

Recommended tasks and few-shot settings for general knowledge evaluation:

| Task ID | Benchmark | Metric | Few-shot | Rationale |
|---------|-----------|--------|----------|-----------|
| `mmlu` | MMLU (57 subjects) | acc | 5-shot | Standard few-shot setting for MMLU; matches Open LLM Leaderboard |
| `arc_challenge` | ARC-Challenge | acc_norm | 25-shot | Standard setting; normalized accuracy accounts for answer length bias |
| `hellaswag` | HellaSwag | acc_norm | 10-shot | Standard setting; normalized accuracy is the standard metric |

Note: `arc_easy` is also available but ARC-Challenge is the standard reporting benchmark. Include both if runtime permits.

[ASSUMED] These few-shot settings match the Open LLM Leaderboard defaults. The exact current leaderboard settings should be verified against the current HuggingFace leaderboard configuration if precise replication is needed.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| bigcode-evaluation-harness for HumanEval/MBPP | evalplus for HumanEval+/MBPP+ | NeurIPS 2023 | 80x more tests; catches false positives from vanilla HumanEval |
| BFCL v1 (simple function calls) | BFCL v4 (multi-turn + agentic) | 2026-03-06 | Broader evaluation including conversation state management |
| lm-eval-harness CLI-only | lm-eval-harness Python API (simple_evaluate) | v0.4.0+ | Programmatic access eliminates subprocess parsing |
| FP16 default for inference | float32 required on MPS | PyTorch MPS maturity | Must explicitly set dtype=float32 for Apple Silicon |

**Deprecated/outdated:**
- vanilla HumanEval (164 problems, minimal tests): replaced by HumanEval+ (164 problems, 80x tests)
- vanilla MBPP (427 tasks): replaced by MBPP+ (378 curated tasks, 35x tests)
- lm-eval-harness < 0.4.0: old API without simple_evaluate()

## Assumptions Log

> List all claims tagged [ASSUMED] in this research.

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | MMLU 5-shot, ARC 25-shot, HellaSwag 10-shot match current Open LLM Leaderboard defaults | Task Names section | Scores would not be directly comparable to published leaderboards. LOW risk -- these are long-standing conventions. Verify by checking leaderboard configuration. |
| A2 | evalplus hf backend works on MPS via standard PyTorch device auto-detection | Standard Stack / evalplus | If evalplus hf backend does not detect MPS, would need to fall back to CPU or use environment variables. MEDIUM risk -- needs testing at install time. |
| A3 | BFCL test data can be loaded and prompts extracted for custom generation | BFCL Two-Phase Pattern | If BFCL's test data format is undocumented or locked behind the generate command, custom generation would need reverse-engineering. MEDIUM risk -- inspect bfcl-eval package after install. |
| A4 | Python 3.14.2 is compatible with all three evaluation packages | Pitfall 5 | If incompatible, need Python 3.12 venv. MEDIUM risk for evalplus (released before 3.14). LOW risk for lm-eval and bfcl-eval (recent releases). |

## Open Questions

1. **BFCL test data format**
   - What we know: BFCL stores test data and expects responses in specific JSON format at `$BFCL_PROJECT_ROOT/result/{model_name}/`
   - What's unclear: Exact JSON schema for individual test prompts and expected response format. Need to inspect bfcl-eval package after install.
   - Recommendation: Install bfcl-eval early and inspect `bfcl_eval/data/` directory structure. Write a small spike script to understand the format before implementing the full runner.

2. **evalplus MPS compatibility**
   - What we know: evalplus supports hf backend which uses transformers. Transformers supports MPS.
   - What's unclear: Whether evalplus's code execution sandbox has any MPS-specific issues or whether the hf backend properly passes device configuration.
   - Recommendation: Test `evalplus.codegen --model SmolLM2-1.7B --dataset humaneval --backend hf --greedy` early. If MPS fails, CPU fallback for code eval is acceptable (code generation is not latency-sensitive for eval).

3. **evalplus output parsing**
   - What we know: Results go to `evalplus_results/{dataset}/{model}_hf_temp_0.0.jsonl`
   - What's unclear: Exact JSONL schema and whether summary pass@k is embedded in the file or only printed to stdout.
   - Recommendation: Run a test evaluation and inspect output files. May need to parse both JSONL results and stdout.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python | All | Yes | 3.14.2 | Python 3.12 venv if compatibility issues |
| PyTorch | Model inference, lm-eval | No | -- | Must install: pip install torch>=2.10 |
| transformers | Model loading | Yes | 5.5.4 | -- |
| pydantic | Result schemas | Yes | 2.12.5 | -- |
| pyyaml | Config parsing | Yes | 6.0+ | -- |
| pytest | Testing | Yes | (available at /opt/homebrew/bin/pytest) | -- |
| lm-eval | Knowledge benchmarks | No | -- | Must install: pip install lm-eval[hf]==0.4.11 |
| evalplus | Code benchmarks | No | -- | Must install: pip install evalplus>=0.3.1 |
| bfcl-eval | Tool calling benchmarks | No | -- | Must install: pip install bfcl-eval==2026.3.23 |
| accelerate | Device management | No | -- | Must install: pip install accelerate>=1.13.0 |
| CUDA/GPU | -- | No | -- | Not needed per D-02 |

**Missing dependencies with no fallback:**
- torch, lm-eval, evalplus, bfcl-eval, accelerate must be installed. All are pip-installable.

**Missing dependencies with fallback:**
- None -- all missing deps have straightforward pip install paths.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (already configured) |
| Config file | pytest.ini (exists at project root) |
| Quick run command | `pytest tests/test_eval_config.py tests/test_eval_compare.py -x` |
| Full suite command | `pytest tests/ -x` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| EVAL-03 | Eval runner invokes standard benchmarks and returns per-benchmark scores | integration | `pytest tests/test_eval_runner.py -x -k "test_benchmark_invocation"` | Wave 0 |
| EVAL-03 | BFCL response generation + AST evaluation pipeline | integration | `pytest tests/test_eval_runner.py -x -k "test_bfcl"` | Wave 0 |
| EVAL-04 | Results JSON contains per-category sections with separate scores | unit | `pytest tests/test_eval_config.py -x -k "test_result_schema"` | Wave 0 |
| EVAL-04 | Compare command reads two JSON files and computes deltas | unit | `pytest tests/test_eval_compare.py -x -k "test_compare"` | Wave 0 |
| D-07 | JSON output matches EvalResult schema | unit | `pytest tests/test_eval_config.py -x -k "test_json_output"` | Wave 0 |
| D-08 | Summary table prints to stdout | unit | `pytest tests/test_eval_runner.py -x -k "test_summary_table"` | Wave 0 |

Note: Integration tests that actually invoke lm-eval/evalplus/BFCL require model downloads and significant runtime. Mark these with `@pytest.mark.slow` and use mocked responses for fast unit tests.

### Sampling Rate
- **Per task commit:** `pytest tests/test_eval_config.py tests/test_eval_compare.py -x`
- **Per wave merge:** `pytest tests/ -x`
- **Phase gate:** Full suite green before verification

### Wave 0 Gaps
- [ ] `tests/test_eval_runner.py` -- covers eval orchestration logic (mocked benchmark calls)
- [ ] `tests/test_eval_compare.py` -- covers comparison logic
- [ ] `tests/test_eval_config.py` -- covers Pydantic schemas and config loading

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | No | -- |
| V3 Session Management | No | -- |
| V4 Access Control | No | -- |
| V5 Input Validation | Yes | Pydantic validation for JSON result files and config; pathlib.Path for file paths |
| V6 Cryptography | No | -- |

### Known Threat Patterns for Evaluation Stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Malicious code in model-generated output executed by evalplus | Elevation of Privilege | evalplus sandboxed execution (do not disable) |
| Path traversal in model_path argument | Tampering | Validate model_path exists and is a directory or HuggingFace model ID using Path.exists() |
| YAML deserialization attacks in eval config | Tampering | yaml.safe_load only (per existing project convention T-02-04) |
| Subprocess injection via model name in BFCL commands | Tampering | Use list-form subprocess.run (not shell=True); validate model name is alphanumeric/hyphen/underscore only |

## Sources

### Primary (HIGH confidence)
- [lm-eval-harness Python API docs](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/python-api.md) -- simple_evaluate signature, device parameter, results format
- [lm-eval PyPI page](https://pypi.org/project/lm-eval/) -- version 0.4.11, release date 2026-02-13, Python >=3.10
- [evalplus GitHub](https://github.com/evalplus/evalplus) -- hf backend, CLI interface, output format
- [evalplus PyPI page](https://pypi.org/project/evalplus/) -- version 0.3.1, Python >=3.9
- [BFCL GitHub README](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard) -- evaluate separate from generate, BFCL_PROJECT_ROOT, AST evaluation
- [bfcl-eval PyPI page](https://pypi.org/project/bfcl-eval/) -- version 2026.3.23, Python >=3.10
- [PyTorch 2.10 release blog](https://pytorch.org/blog/pytorch-2-10-release-blog/) -- Python 3.14 support confirmed
- [HuggingFace Apple Silicon training docs](https://huggingface.co/docs/transformers/en/perf_train_special) -- MPS limitations (no FP16), PYTORCH_ENABLE_MPS_FALLBACK
- [Apple Metal PyTorch docs](https://developer.apple.com/metal/pytorch/) -- MPS device detection

### Secondary (MEDIUM confidence)
- [lm-eval-harness MPS support discussion](https://github.com/EleutherAI/lm-evaluation-harness) -- MPS "early stages of development" warning
- [BFCL v4 leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html) -- updated 2026-03-06
- [HuggingFace lm-eval SmolLM2 evaluation blog](https://huggingface.co/blog/Neo111x/integrating-benchmarks-into-lm-evaluation-harness) -- SmolLM2 checkpoint evaluation example

### Tertiary (LOW confidence)
- evalplus hf backend MPS compatibility is inferred from transformers MPS support, not directly tested [ASSUMED]
- BFCL custom generation approach (separate generate/evaluate) has not been tested with SmolLM2-1.7B specifically [ASSUMED]

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all packages verified on PyPI with current versions and release dates
- Architecture: HIGH -- lm-eval Python API is well-documented; BFCL two-phase pattern is architecturally sound based on README
- Pitfalls: HIGH -- MPS limitations well-documented by Apple and HuggingFace; BFCL GPU requirement confirmed in docs
- BFCL integration details: MEDIUM -- exact test data format and response schema need hands-on inspection after install
- evalplus MPS compatibility: MEDIUM -- inferred but not directly verified

**Research date:** 2026-04-20
**Valid until:** 2026-05-20 (30 days -- stable ecosystem, major packages recently released)
