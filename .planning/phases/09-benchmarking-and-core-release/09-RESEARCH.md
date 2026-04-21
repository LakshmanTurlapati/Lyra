# Phase 9: Benchmarking and Core Release - Research

**Researched:** 2026-04-20
**Domain:** LLM evaluation (lm-eval-harness, custom inference), Markdown/Mermaid report generation, Git LFS, MIT license scaffolding
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Run lm-eval-harness locally on MPS for knowledge benchmarks (MMLU, ARC-Challenge, HellaSwag). These work reliably on Apple Silicon.
- **D-02:** For tool-calling and code evals, use inference + pattern matching on our test split instead of BFCL/evalplus (which need CUDA). Run model inference on curated test prompts, check outputs for correct JSON tool call format, valid code syntax, and factual accuracy.
- **D-03:** No cloud/CUDA eval runs required. Everything runs locally on MPS with CPU fallback.
- **D-04:** Auto-generated `BENCHMARK.md` in repo root with tables showing base vs Lyra scores, deltas, and summary.
- **D-05:** Include Mermaid charts for visual comparison (bar charts for category scores, radar chart for overall profile).
- **D-06:** Report generated from JSON results files via the existing `eval_compare.py` tool, extended to output Markdown + Mermaid.
- **D-07:** No HuggingFace publishing for now. All artifacts stay on GitHub.
- **D-08:** Model card and dataset card still created as Markdown files in the repo (README.md serves as model card, datasets/README.md as dataset card).
- **D-09:** Ship everything: merged safetensors model, LoRA adapter weights, training/eval scripts, and assembled dataset files.
- **D-10:** Use Git LFS for large files (.safetensors, .bin, large .jsonl files). Keeps repo cloneable without bloating git history.
- **D-11:** MIT license file in repo root. License headers reference MIT in all scripts.

### Claude's Discretion
- Exact Mermaid chart types and styling
- Which lm-eval-harness few-shot settings to use (eval.yaml already has defaults)
- How to structure the custom inference eval (test harness script design)
- Report section ordering and narrative structure
- Git LFS track patterns and .gitattributes configuration

### Deferred Ideas (OUT OF SCOPE)
- HuggingFace Hub publishing -- deferred, may revisit in Phase 10 or later
- Cloud CUDA eval runs for BFCL/evalplus standard scores -- not needed for v1
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| EVAL-01 | Model evaluated on standard benchmarks (MMLU, MBPP/HumanEval, BFCL) with pass@1 scores | lm-eval-harness covers MMLU/ARC/HellaSwag; custom eval_inference.py covers code (pass@1 via ast.parse) and tool-calling (JSON schema check) per D-02 |
| EVAL-02 | Published comparison report of base SmolLM2-1.7B vs Lyra fine-tuned model | eval_compare.py extended with --markdown flag writes BENCHMARK.md with tables + Mermaid charts |
| REL-01 | Dataset card on HuggingFace with description, creation methodology, statistics, limitations | Per D-07/D-08: datasets/README.md serves as dataset card (GitHub only, no HF publishing) |
| REL-02 | Model card on HuggingFace with metadata YAML, usage examples, training params, benchmark results | Per D-07/D-08: README.md serves as model card; auto-populated benchmark section from BENCHMARK.md |
| REL-03 | Model weights published in safetensors format | models/lyra-merged/model.safetensors (3.2GB) and models/lyra-adapter/adapter_model.safetensors (80MB) exist and need Git LFS tracking |
| REL-04 | MIT license applied consistently to datasets, model weights, scripts, and eval code | LICENSE file (MIT) in repo root + license headers in all scripts |
</phase_requirements>

---

## Summary

Phase 9 is a three-part phase: run evaluations, generate a comparison report, and prepare the release. All artifacts from prior phases already exist. The merged model (`models/lyra-merged/model.safetensors`, 3.2GB) and adapter (`models/lyra-adapter/adapter_model.safetensors`, 80MB) are in place. The test split has 181 samples across three domains (tool-calling: 123, code: 30, knowledge: 28). The eval infrastructure (`eval_runner.py`, `eval_compare.py`, `eval_config.py`, `configs/eval.yaml`) is fully operational with 30 passing tests.

The key build work is: (1) extend `eval_runner.py` to support a new `--benchmarks custom` path that loads the test split and runs inference + pattern-matching for tool-calling and code, (2) extend `eval_compare.py` with a `--markdown` flag that emits BENCHMARK.md with Mermaid charts, and (3) set up Git LFS + LICENSE + model/dataset cards. The lm-eval-harness knowledge eval path already exists in `eval_runner.py` and just needs lm-eval installed.

The primary risk is lm-eval installation: it is not currently in the venv (`lm-eval: NOT installed`). Git LFS is also not installed (`git-lfs: NOT FOUND`). Both are `brew`/`pip` installs. Neither is a code problem, but both must happen before evaluation can run. The plan must include setup tasks for these before the actual eval run tasks.

**Primary recommendation:** Four sequential waves -- (1) install dependencies and set up Git LFS + LICENSE, (2) run lm-eval knowledge benchmarks for both models, (3) run custom inference eval for tool-calling and code, (4) generate BENCHMARK.md and write model/dataset cards.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Knowledge eval (MMLU/ARC/HellaSwag) | scripts/ (eval_runner.py) | configs/eval.yaml | lm-eval-harness wraps HuggingFace model backend; invoked via Python API |
| Custom inference eval (tool-calling/code) | scripts/ (new eval_inference.py) | datasets/assembled/ | Loads test split, runs transformers inference on MPS, checks output with regex/ast.parse |
| Comparison report generation | scripts/ (eval_compare.py extended) | results/*.json | Reads two EvalResult JSON files, emits BENCHMARK.md with Mermaid |
| Model card | README.md | BENCHMARK.md | README.md extended with HF-style YAML frontmatter + benchmark table section |
| Dataset card | datasets/README.md | datasets/assembled/ | New file; references assembled/ stats and documents methodology |
| Git LFS tracking | .gitattributes | models/ + datasets/ | Track *.safetensors, *.bin, large *.jsonl via `git lfs track` |
| License scaffolding | LICENSE (repo root) | scripts/ headers | MIT SPDX header in LICENSE file; scripts already have module docstrings to extend |

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| lm-eval | 0.4.11 | Knowledge benchmarks (MMLU, ARC-Challenge, HellaSwag) via simple_evaluate() | Already referenced in eval_runner.py; only version in PyPI [VERIFIED: pip index versions lm-eval] |
| transformers | 5.5.4 | Model loading for custom inference eval | Already installed in venv [VERIFIED: venv pip list] |
| torch | 2.11.0 | MPS inference backend | Already installed in venv [VERIFIED: venv pip list] |
| datasets (HuggingFace) | 4.8.4 | Load test split for custom inference eval | Already installed in venv [VERIFIED: venv pip list] |
| pydantic | 2.12.5 | EvalResult/CompareResult schemas | Already in eval_config.py [VERIFIED: codebase] |
| git-lfs | 3.7.1 | Track large binary files (.safetensors, .bin) without bloating git history | Industry standard for large model files; available via `brew install git-lfs` [VERIFIED: brew info git-lfs] |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| ast (stdlib) | stdlib | Validate Python code syntax in custom code eval | parse-then-compile check is zero-dependency and deterministic |
| json (stdlib) | stdlib | Validate tool-call JSON in custom tool-calling eval | Standard JSON parse is sufficient for format compliance check |
| re (stdlib) | stdlib | Extract JSON blocks from model output (tool-calling detection) | Already used in eval_runner.py for score parsing |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Custom inference eval | evalplus / BFCL | evalplus and BFCL require CUDA; locked out by D-02/D-03 |
| lm-eval Python API | lm-eval CLI subprocess | Python API gives structured dict result; CLI requires parsing stdout, fragile |
| Mermaid in Markdown | matplotlib/PNG charts | Mermaid renders natively on GitHub with no build step or binary blobs |
| git-lfs | .gitignore for model files | Git LFS keeps files accessible to cloners; .gitignore hides them entirely |

**Installation:**
```bash
# Eval harness
pip install 'lm-eval[hf]==0.4.11'

# Git LFS (one-time system install)
brew install git-lfs
git lfs install
```

---

## Architecture Patterns

### System Architecture Diagram

```
datasets/assembled/test/   models/lyra-merged/    HuggingFaceTB/SmolLM2-1.7B-Instruct
        |                         |                           |
        v                         v                           v
 [eval_inference.py]         [eval_runner.py]           [eval_runner.py]
  custom eval                 --benchmarks               --benchmarks
  (tool-calling,              knowledge                  knowledge
   code domains)              (lm-eval-harness API)      (base model)
        |                         |                           |
        v                         v                           v
results/lyra_custom.json   results/lyra_knowledge.json  results/base_knowledge.json
        |___________________________|___________________________|
                                    |
                           [merge step: combine JSONs
                            into single EvalResult per model]
                                    |
                        results/lyra_full.json    results/base_full.json
                                    |
                           [eval_compare.py --markdown]
                                    |
                           BENCHMARK.md (tables + Mermaid charts)
                                    |
                           README.md model card section updated
                           datasets/README.md written
```

### Recommended Project Structure

No new directories needed. New files land in existing locations:

```
.
├── BENCHMARK.md               # auto-generated from eval_compare.py --markdown
├── LICENSE                    # new: MIT license text
├── .gitattributes             # new: git lfs track patterns
├── README.md                  # extended: model card frontmatter + benchmark section
├── datasets/
│   └── README.md              # new: dataset card
├── results/
│   ├── base_knowledge.json    # lm-eval output for SmolLM2-1.7B base
│   ├── lyra_knowledge.json    # lm-eval output for Lyra merged
│   ├── base_custom.json       # custom eval output for SmolLM2-1.7B base
│   ├── lyra_custom.json       # custom eval output for Lyra merged
│   ├── base_full.json         # merged EvalResult (knowledge + custom)
│   ├── lyra_full.json         # merged EvalResult (knowledge + custom)
│   └── compare.json           # CompareResult list
└── scripts/
    └── eval_inference.py      # new: custom inference eval for tool-calling + code
```

### Pattern 1: Custom Inference Eval (D-02)

**What:** Load assembled test split, run transformers inference on each sample, check output against expected pattern.
**When to use:** Tool-calling (JSON format check) and code domains (ast.parse syntax check).

```python
# Source: inferred from existing eval_runner.py patterns + datasets API
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
import json, ast, re

def run_custom_eval(model_path: str, device: str, dataset_dir: str) -> CategoryResult:
    """Run custom inference eval on assembled test split."""
    ds = load_from_disk(dataset_dir)
    test_split = ds["test"]

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    tool_results, code_results = [], []
    for sample in test_split:
        output = run_inference(model, tokenizer, sample, device)
        if sample["domain"] == "tool-calling":
            tool_results.append(check_tool_call_format(output))
        elif sample["domain"] == "code":
            code_results.append(check_code_syntax(output))

    return CategoryResult(category="custom", benchmarks=[
        BenchmarkResult(benchmark="tool-call-format", metric="pass@1",
                        score=sum(tool_results)/len(tool_results) if tool_results else 0.0),
        BenchmarkResult(benchmark="code-syntax", metric="pass@1",
                        score=sum(code_results)/len(code_results) if code_results else 0.0),
    ])


def check_tool_call_format(output: str) -> bool:
    """Return True if output contains valid JSON tool call."""
    # SmolLM2 format: <tool_call>{"name": ..., "arguments": {...}}</tool_call>
    match = re.search(r"<tool_call>(.*?)</tool_call>", output, re.DOTALL)
    if not match:
        return False
    try:
        obj = json.loads(match.group(1))
        return "name" in obj and "arguments" in obj
    except (json.JSONDecodeError, KeyError):
        return False


def check_code_syntax(output: str) -> bool:
    """Return True if output contains syntactically valid Python code block."""
    match = re.search(r"```(?:python|py)?\n(.*?)```", output, re.DOTALL)
    if not match:
        return False
    try:
        ast.parse(match.group(1))
        return True
    except SyntaxError:
        return False
```

**Key detail on test split composition:** 181 samples total -- tool-calling: 123, code: 30, knowledge: 28. The custom eval covers tool-calling and code (153 samples). Knowledge samples in the test split are covered by lm-eval-harness benchmarks instead (they use standard MMLU/ARC/HellaSwag tasks, not the 28 knowledge samples in our test set).

### Pattern 2: Markdown + Mermaid Report Generation (D-04, D-05, D-06)

**What:** Extend `eval_compare.py` with `--markdown OUTPUT_PATH` flag that writes BENCHMARK.md.
**When to use:** After both model JSONs are available.

```python
# Source: Mermaid bar chart syntax -- [ASSUMED] from Mermaid docs
def format_mermaid_bar_chart(results: list[CompareResult]) -> str:
    """Emit Mermaid xychart-beta bar chart comparing base vs Lyra per benchmark."""
    benchmarks = [r.benchmark for r in results]
    base_vals = [f"{r.baseline_score:.4f}" for r in results]
    lyra_vals = [f"{r.candidate_score:.4f}" for r in results]
    bench_labels = str(benchmarks).replace("'", '"')
    return f"""```mermaid
xychart-beta
    title "Base SmolLM2-1.7B vs Lyra"
    x-axis {bench_labels}
    y-axis "Score" 0 --> 1
    bar {base_vals}
    bar {lyra_vals}
```"""


def write_benchmark_md(results: list[CompareResult], output_path: Path,
                       base_name: str, candidate_name: str) -> None:
    """Write BENCHMARK.md with summary table, per-category tables, and Mermaid charts."""
    lines = [
        "# Benchmark Results",
        "",
        f"Base model: `{base_name}` | Fine-tuned: `{candidate_name}`",
        "",
        "## Summary",
        "",
        "| Benchmark | Base | Lyra | Delta |",
        "|-----------|------|------|-------|",
    ]
    for r in results:
        delta_str = f"+{r.delta:.4f}" if r.delta > 0 else f"{r.delta:.4f}"
        lines.append(f"| {r.benchmark} | {r.baseline_score:.4f} | "
                     f"{r.candidate_score:.4f} | {delta_str} |")
    lines += ["", format_mermaid_bar_chart(results)]
    output_path.write_text("\n".join(lines))
```

**Mermaid chart types verified for GitHub rendering:**
- `xychart-beta` (bar charts) -- renders in GitHub Markdown [ASSUMED: standard GitHub Mermaid support]
- `radar` charts -- may require Mermaid v10.9+; use `xychart-beta` as fallback if radar fails

### Pattern 3: EvalResult JSON Merging

**What:** lm-eval knowledge eval and custom inference eval produce separate JSON files. They need to be merged into a single EvalResult before `eval_compare.py` can process them.

```python
# Source: existing EvalConfig/EvalResult schemas in eval_config.py [VERIFIED: codebase]
def merge_eval_results(knowledge_json: Path, custom_json: Path, output: Path) -> None:
    """Merge two EvalResult JSON files into one (combine categories lists)."""
    k = EvalResult.model_validate_json(knowledge_json.read_text())
    c = EvalResult.model_validate_json(custom_json.read_text())
    merged = EvalResult(
        model_path=k.model_path,
        model_name=k.model_name,
        timestamp=k.timestamp,
        device=k.device,
        categories=k.categories + c.categories,  # combine category lists
    )
    output.write_text(merged.model_dump_json(indent=2))
```

This can be a standalone `scripts/eval_merge.py` or a flag on `eval_runner.py`.

### Pattern 4: Git LFS Setup (D-10)

```bash
# Source: git-lfs documentation [ASSUMED: standard git lfs workflow]
git lfs install                          # one-time per machine
git lfs track "*.safetensors"
git lfs track "*.bin"
git lfs track "datasets/**/*.jsonl"      # large JSONL files (if included)
# This writes .gitattributes automatically
git add .gitattributes
```

**Critical:** `.gitattributes` must be committed BEFORE adding the large files. Adding large files to git without LFS tracking first will put them in git history, which cannot be undone without a history rewrite.

**gitignore note:** `models/` is NOT currently in `.gitignore` [VERIFIED: .gitignore]. The assembled dataset Arrow files ARE in `.gitignore` (`datasets/assembled/`). For D-09 (ship everything), the plan must ensure:
1. `models/` remains unignored
2. `datasets/assembled/` gitignore override needed if including Arrow files, OR include raw JSONL instead

**What to track with LFS:**
- `models/lyra-merged/model.safetensors` -- 3.2GB [VERIFIED: du -sh]
- `models/lyra-adapter/adapter_model.safetensors` -- 80MB [VERIFIED: du -sh]
- `models/lyra-adapter/training_args.bin`

### Pattern 5: MIT License File (D-11)

Standard MIT license text for `LICENSE` file in repo root:

```
MIT License

Copyright (c) 2026 [Author]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

Script license headers: add a one-liner `# SPDX-License-Identifier: MIT` at the top of each script file after the shebang, or reference MIT in the module docstring.

### Pattern 6: Model Card (README.md extension, D-08, REL-02)

The existing `README.md` (137 lines) needs:

1. **HuggingFace-compatible YAML frontmatter** at the very top (even though we're not publishing to HF, it's the standard for model cards per REL-02):
```yaml
---
license: mit
base_model: HuggingFaceTB/SmolLM2-1.7B-Instruct
language: en
tags:
  - fine-tuned
  - tool-calling
  - code-generation
  - smollm2
  - lora
  - sft
---
```

2. **Usage examples** section showing how to load and run inference
3. **Training parameters** section (LoRA r=16, alpha=32, dropout=0.05, lr=2e-4, epochs=3, batch=4, grad_accum=4) [VERIFIED: adapter_config.json + train.py defaults]
4. **Benchmark results** section linking to BENCHMARK.md (populated after eval runs)

### Pattern 7: Dataset Card (datasets/README.md, D-08, REL-01)

New file. Contents:
- Dataset description and purpose
- Creation methodology (Claude Opus generation, quality pipeline)
- Statistics (3,624 total: train 3,262 / val 181 / test 181; domain breakdown: tool-calling 68%, code 16.6%, knowledge 15.5%) [VERIFIED: assemble_dataset stats]
- Limitations (synthetic data, SmolLM2-1.7B format-specific, English only)
- HF dataset card YAML frontmatter (for future HF publishing)

### Anti-Patterns to Avoid

- **Running evals before confirming lm-eval is installed:** `lm_eval` is not in the current venv. The plan must install it before running knowledge benchmarks. Install command: `pip install 'lm-eval[hf]==0.4.11'`.
- **Adding .safetensors to git without LFS first:** Will embed 3.2GB into git history. Cannot be fixed without history rewrite. Always commit `.gitattributes` before `git add models/`.
- **Hardcoding model scores in BENCHMARK.md:** BENCHMARK.md must be generated from JSON results files, not manually written. If scores change (re-run), the script regenerates it.
- **Assuming radar chart works on GitHub:** GitHub supports Mermaid, but radar chart support depends on the Mermaid version bundled. Use `xychart-beta` (bar chart) as the primary; radar chart is Claude's discretion but may silently fail to render.
- **Ignoring datasets/assembled/ gitignore:** `datasets/assembled/` is in `.gitignore`. If the plan includes shipping the assembled dataset, it must either: (a) `git add -f datasets/assembled/` and track with LFS, or (b) ship the raw JSONL files from `datasets/tool-calling/curated/` etc. instead.
- **Testing lm-eval on the fine-tuned model first:** Run base model first to establish baseline before spending time on fine-tuned model. Scores without a baseline are not reportable.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Knowledge benchmark scoring | Custom MMLU evaluation loop | lm-eval simple_evaluate() | lm-eval handles few-shot formatting, answer normalization, exact metric definitions; hand-rolling produces non-comparable numbers |
| JSON syntax validation | Custom JSON parser for tool calls | stdlib json.loads() | Standard library parse is more robust than regex-only approaches |
| Python syntax validation | Execute code to check correctness | stdlib ast.parse() | ast.parse() is safe (no execution), deterministic, catches all syntax errors |
| Model file versioning | Custom binary storage | Git LFS | LFS is the standard; alternatives (GitHub Releases, S3 upload) are valid but add complexity the project doesn't need |
| Markdown table generation | Custom string builder | f-string templating (project pattern) | Project convention: plain f-string formatting, no external deps (see eval_compare.py, Phase 3 decision) |

**Key insight:** The custom eval is intentionally not "standard" -- it deliberately measures Lyra-specific behaviors (SmolLM2 `<tool_call>` XML format, concise code blocks) that standard benchmarks don't cover. The value is in the delta vs. base, not in absolute comparability to other models.

---

## Common Pitfalls

### Pitfall 1: lm-eval API result key format

**What goes wrong:** Extracting wrong metric from lm-eval results dict. MMLU returns `"acc,none"`, ARC-Challenge and HellaSwag return `"acc_norm,none"`.

**Why it happens:** lm-eval uses comma-separated metric+filter keys; the "none" filter suffix is not obvious from the task name.

**How to avoid:** The existing `eval_runner.py` already has `metric_map` with the correct keys [VERIFIED: eval_runner.py line 136-140]. No change needed. Just ensure lm-eval version matches.

**Warning signs:** Score of `0.0` for a task that should have a real score; check `results["results"][task_name].keys()` in a debug run.

### Pitfall 2: MPS inference memory pressure for 1.7B model

**What goes wrong:** Loading two large models (base + fine-tuned) in sequence without clearing GPU memory between runs causes OOM on MPS.

**Why it happens:** MPS memory is not automatically freed when Python object goes out of scope in all cases.

**How to avoid:** Explicitly `del model; import gc; gc.collect()` between model loads. Or run base model eval in a separate subprocess call, as `eval_runner.py` already does for BFCL [VERIFIED: eval_runner.py subprocess pattern]. The simplest approach: run base eval, write JSON, exit script, then run Lyra eval in separate invocation.

**Warning signs:** `RuntimeError: MPS backend out of memory` during second model load.

### Pitfall 3: SmolLM2 tool call format in custom eval

**What goes wrong:** Custom eval counts a response as "wrong" because it looks for `{"name": ..., "arguments": ...}` but SmolLM2's format is `<tool_call>JSON</tool_call>`.

**Why it happens:** SmolLM2 uses XML-wrapped JSON tool calls, not bare JSON. Phase 1 decision: "Pre-process TRL-native tool_calls to SmolLM2 `<tool_call>` XML format before tokenization" [VERIFIED: STATE.md decisions].

**How to avoid:** The `check_tool_call_format` function in eval_inference.py must search for `<tool_call>...</tool_call>` wrapper, then parse the interior as JSON.

**Warning signs:** tool-call-format pass@1 near 0.0 for both base and fine-tuned (means the format check itself is wrong, not the model).

### Pitfall 4: Inference input format for custom eval

**What goes wrong:** Custom eval builds incorrect chat-templated inputs, causing model to produce garbage outputs unrelated to the task.

**Why it happens:** SmolLM2's chat template must be applied via `tokenizer.apply_chat_template()` with `add_generation_prompt=True`. Using raw message strings bypasses the template.

**How to avoid:** Use `tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)` to get the formatted prompt string, then tokenize it. Existing `eval_runner.py` uses this pattern for BFCL response generation [VERIFIED: eval_runner.py].

**Warning signs:** Model outputs unrelated text or repeats the prompt.

### Pitfall 5: BENCHMARK.md committed before eval results exist

**What goes wrong:** Placeholder BENCHMARK.md is committed with empty tables; users clone and see broken report.

**Why it happens:** Release tasks run before eval tasks complete.

**How to avoid:** Plan must sequence: eval runs -> merge JSONs -> generate BENCHMARK.md -> commit all three together. BENCHMARK.md should not be committed until it has real numbers.

### Pitfall 6: .gitattributes not committed before model files

**What goes wrong:** `git add models/` before `.gitattributes` is committed means safetensors files go into regular git, not LFS.

**Why it happens:** Developers add files first, configure LFS second.

**How to avoid:** Plan must sequence: `brew install git-lfs` -> `git lfs install` -> `git lfs track "*.safetensors"` -> `git add .gitattributes` -> `git commit .gitattributes` -> then `git add models/`.

**Warning signs:** `git lfs ls-files` shows no tracked files; `git log --stat` shows very large (+/- line counts) for .safetensors files.

---

## Code Examples

### Loading test split for custom eval

```python
# Source: Phase 7 assembly -- tested in production [VERIFIED: dataset exists at datasets/assembled/]
import os
os.chdir("/path/to/lyra")  # must be repo root for relative path
from datasets import load_from_disk
ds = load_from_disk("datasets/assembled")
test = ds["test"]  # 181 samples: tool-calling: 123, code: 30, knowledge: 28
```

### lm-eval simple_evaluate (existing, works)

```python
# Source: eval_runner.py lines 124-131 [VERIFIED: codebase]
import lm_eval
results = lm_eval.simple_evaluate(
    model="hf",
    model_args=f"pretrained={model_path},dtype=float32",
    tasks=["mmlu", "arc_challenge", "hellaswag"],
    device="mps",
    batch_size="auto",
    num_fewshot=5,  # first task's fewshot used as global
)
mmlu_score = results["results"]["mmlu"]["acc,none"]
arc_score = results["results"]["arc_challenge"]["acc_norm,none"]
hellaswag_score = results["results"]["hellaswag"]["acc_norm,none"]
```

### Running inference on a test sample

```python
# Source: inferred from transformers pipeline pattern + existing eval_runner.py [ASSUMED]
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def run_inference_on_sample(model, tokenizer, sample: dict, device: str,
                            max_new_tokens: int = 256) -> str:
    """Run model inference on one test split sample."""
    # Build messages from sample (exclude last assistant turn -- that's the label)
    messages = sample["messages"]
    # Find last assistant turn index for label
    last_asst_idx = max(i for i, m in enumerate(messages) if m["role"] == "assistant")
    prompt_messages = messages[:last_asst_idx]  # everything before last assistant response

    prompt = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy for determinism
        )
    # Decode only the new tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)
```

### EvalResult from eval_config.py (existing schema)

```python
# Source: scripts/eval_config.py [VERIFIED: codebase]
from scripts.eval_config import BenchmarkResult, CategoryResult, EvalResult
from datetime import datetime

result = EvalResult(
    model_path="models/lyra-merged",
    model_name="lyra-merged",
    timestamp=datetime.now().isoformat(),
    device="mps",
    categories=[
        CategoryResult(
            category="custom",
            benchmarks=[
                BenchmarkResult(benchmark="tool-call-format", metric="pass@1", score=0.85),
                BenchmarkResult(benchmark="code-syntax", metric="pass@1", score=0.90),
            ]
        )
    ]
)
result_json = result.model_dump_json(indent=2)
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| evalplus/BFCL for all evals | Custom inference + lm-eval for MPS | Phase 9 decision | evalplus and BFCL require CUDA; custom eval is MPS-compatible and measures Lyra-specific format compliance |
| HuggingFace Hub for model/dataset publishing | GitHub-only release | D-07 | Simpler for v1; HF publishing deferred to Phase 10 |
| lm-eval CLI subprocess | lm-eval Python API (simple_evaluate) | Phase 3 design | Python API gives structured dict; no stdout parsing needed |

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | GitHub renders `xychart-beta` Mermaid bar charts in Markdown | Architecture Patterns, Pattern 2 | BENCHMARK.md charts show as code blocks instead of images; fallback: include ASCII table which already exists |
| A2 | `radar` chart type requires Mermaid v10.9+ on GitHub | Architecture Patterns, Pattern 2 | Radar chart may or may not render; use xychart-beta as primary (Claude's discretion on chart types) |
| A3 | `tokenizer.apply_chat_template(add_generation_prompt=True)` is the correct API for SmolLM2 inference prompts | Code Examples | Inference inputs malformed; model produces garbage; caught immediately by near-zero pass@1 on both models |
| A4 | lm-eval 0.4.11 `simple_evaluate` API signature unchanged since Phase 3 design | Standard Stack | Import error or API mismatch; eval_runner.py tests mock this so it won't surface until actual install |

---

## Open Questions (RESOLVED)

1. **Should datasets/assembled/ Arrow files be tracked with LFS or excluded?**
   - **RESOLVED:** Use `git add -f datasets/assembled/` with LFS tracking for the Arrow files (per D-09 and D-10). The .gitignore exception is handled by `git add -f` (force-add). Add `datasets/assembled/**` to LFS tracking in .gitattributes via `git lfs track "datasets/assembled/**"`. This is implemented in Plan 05, Task 3.
   - What we know: `datasets/assembled/` is currently in `.gitignore`. Arrow files are ~17MB each.
   - Rationale: D-09 explicitly requires shipping assembled dataset files. `git add -f` overrides .gitignore for this specific directory. LFS (D-10) ensures the Arrow files don't bloat regular git history.

2. **What count of knowledge samples from our test set (28 samples) is used in EVAL-01?**
   - **RESOLVED:** EVAL-01 report uses lm-eval benchmark scores (MMLU/ARC/HellaSwag only) on the full benchmark datasets. The 28 knowledge test split samples are not used in EVAL-01 — they exist only for reference. lm-eval runs on standard benchmark tasks (thousands of questions), not our curated test set. The custom eval covers tool-calling (123 samples) and code (30 samples) from the test split only.

3. **Author name for LICENSE file**
   - **RESOLVED:** Copyright holder is "Lakshman Turlapati" (confirmed by revision_context user decision). LICENSE file already written in Plan 01, Task 1 with this name.

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| lm-eval | EVAL-01 knowledge benchmarks | No | -- | None -- must install before knowledge eval |
| git-lfs | REL-03 D-10 large file tracking | No | -- | None -- must install; model files are 3.2GB |
| torch (MPS) | Custom inference eval, lm-eval | Yes | 2.11.0 | CPU fallback (PYTORCH_ENABLE_MPS_FALLBACK=1 already set in eval_runner.py) |
| transformers | Custom inference eval | Yes | 5.5.4 | -- |
| datasets | Load test split | Yes | 4.8.4 | -- |
| pydantic | EvalResult schemas | Yes | 2.12.5 | -- |
| accelerate | lm-eval hf backend | Yes | 1.13.0 | -- |

**Missing dependencies with no fallback:**
- `lm-eval 0.4.11` -- required for EVAL-01 knowledge benchmarks; install: `pip install 'lm-eval[hf]==0.4.11'`
- `git-lfs` -- required for D-10 large file tracking; install: `brew install git-lfs && git lfs install`

**Missing dependencies with fallback:**
- None of the missing items have fallbacks that satisfy the requirements.

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 9.0.3 |
| Config file | pytest.ini |
| Quick run command | `pytest tests/test_eval_compare.py tests/test_eval_config.py tests/test_eval_runner.py -x` |
| Full suite command | `pytest tests/ -x` |

### Phase Requirements -> Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| EVAL-01 | lm-eval knowledge benchmarks run and produce EvalResult JSON | integration (slow) | `pytest tests/test_eval_runner.py -m slow -x` | Stubs exist; Wave 0 adds real test |
| EVAL-01 | custom inference eval produces EvalResult JSON with tool-call-format and code-syntax benchmarks | unit (mocked model) | `pytest tests/test_eval_inference.py -x` | Wave 0 gap |
| EVAL-02 | eval_compare --markdown produces valid BENCHMARK.md with Mermaid block | unit | `pytest tests/test_eval_compare.py::test_write_benchmark_md -x` | Wave 0 gap |
| EVAL-02 | BENCHMARK.md contains Mermaid xychart-beta block | unit | `pytest tests/test_eval_compare.py::test_mermaid_chart_present -x` | Wave 0 gap |
| REL-01 | datasets/README.md exists and contains required sections | smoke | `pytest tests/test_release_artifacts.py::test_dataset_card -x` | Wave 0 gap |
| REL-02 | README.md contains YAML frontmatter with license: mit | smoke | `pytest tests/test_release_artifacts.py::test_model_card_frontmatter -x` | Wave 0 gap |
| REL-03 | .gitattributes tracks *.safetensors with git-lfs | smoke | `pytest tests/test_release_artifacts.py::test_gitattributes_lfs -x` | Wave 0 gap |
| REL-04 | LICENSE file exists at repo root with MIT text | smoke | `pytest tests/test_release_artifacts.py::test_license_file -x` | Wave 0 gap |

### Sampling Rate

- **Per task commit:** `pytest tests/test_eval_compare.py tests/test_eval_config.py -x`
- **Per wave merge:** `pytest tests/ -x --ignore=tests/test_eval_runner.py` (skip slow integration tests in CI)
- **Phase gate:** Full suite green before `/gsd-verify-work`

### Wave 0 Gaps

- [ ] `tests/test_eval_inference.py` -- covers EVAL-01 custom eval with mocked transformers model
- [ ] `tests/test_release_artifacts.py` -- covers REL-01 through REL-04 artifact presence checks
- [ ] Add `test_write_benchmark_md` and `test_mermaid_chart_present` to `tests/test_eval_compare.py`

---

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | -- |
| V3 Session Management | no | -- |
| V4 Access Control | no | -- |
| V5 Input Validation | yes | `EvalResult.model_validate_json` already enforces schema at trust boundary (eval JSON files) |
| V6 Cryptography | no | -- |

### Known Threat Patterns for this stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Malformed JSON eval result file | Tampering | `EvalResult.model_validate_json` rejects invalid input (already implemented in eval_compare.py) |
| Model path injection | Tampering | `MODEL_PATH_PATTERN` regex already validates model paths in eval_runner.py |
| Unsafe YAML load | Tampering | `yaml.safe_load` enforced throughout; documented in eval_config.py |
| Code execution via ast | Elevation | `ast.parse()` used (not `exec()`); parse-only, no execution |

---

## Sources

### Primary (HIGH confidence)
- `scripts/eval_runner.py` -- existing eval runner, lm-eval API usage, metric key format [VERIFIED: codebase read]
- `scripts/eval_compare.py` -- existing comparison tool, table formatting patterns [VERIFIED: codebase read]
- `scripts/eval_config.py` -- Pydantic schemas for EvalResult, BenchmarkResult [VERIFIED: codebase read]
- `configs/eval.yaml` -- benchmark tasks and few-shot settings [VERIFIED: codebase read]
- `tests/test_eval_runner.py`, `tests/test_eval_compare.py` -- 30 passing tests [VERIFIED: pytest run]
- `models/lyra-merged/` -- model.safetensors 3.2GB, all required files present [VERIFIED: ls + du -sh]
- `models/lyra-adapter/` -- adapter_model.safetensors 80MB, adapter_config.json [VERIFIED: ls + du -sh]
- `datasets/assembled/` -- DatasetDict with 181-sample test split [VERIFIED: assemble_dataset stats]
- pip index: lm-eval 0.4.11 is current stable [VERIFIED: pip index versions lm-eval]
- brew: git-lfs 3.7.1 stable, not installed [VERIFIED: brew info git-lfs]
- venv packages: torch 2.11.0, transformers 5.5.4, pydantic 2.12.5, datasets 4.8.4 installed [VERIFIED: pip list]

### Secondary (MEDIUM confidence)
- Phase 1 STATE.md decision: SmolLM2 uses `<tool_call>JSON</tool_call>` XML wrapper format [VERIFIED: STATE.md]
- adapter_config.json: LoRA r=16, alpha=32, dropout=0.05, target modules confirmed [VERIFIED: codebase]
- train.py defaults: lr=2e-4, epochs=3, batch=4, grad_accum=4, max_length=4096 [VERIFIED: codebase]

### Tertiary (LOW confidence / ASSUMED)
- GitHub Mermaid support for xychart-beta [A1, ASSUMED]
- GitHub radar chart Mermaid support [A2, ASSUMED]

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries verified against venv, pip registry, and existing codebase usage
- Architecture: HIGH -- all input/output files verified to exist; patterns derived from working code
- Pitfalls: HIGH -- verified against actual codebase state (lm-eval missing, LFS missing, format decisions in STATE.md)
- Test coverage: MEDIUM -- existing 30 tests pass; new tests for EVAL-01/REL-01 through REL-04 are Wave 0 gaps

**Research date:** 2026-04-20
**Valid until:** 2026-05-20 (stable stack -- lm-eval 0.4.11 has been at this version for months)
