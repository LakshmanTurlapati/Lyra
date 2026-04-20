# Phase 7: Dataset Assembly - Research

**Researched:** 2026-04-20
**Domain:** HuggingFace datasets library, stratified splitting, JSONL merging
**Confidence:** HIGH

## Summary

Phase 7 merges three curated domain JSONL files (tool-calling: 2,470 samples, code: 600 samples, knowledge: 560 samples = 3,630 total) into a single HuggingFace datasets-format dataset with stratified train/validation/test splits at 90/5/5. The user decided to use ALL samples with natural imbalance (~68%/16.5%/15.4%) rather than downsample to 33/33/33.

The core technical challenge is straightforward: load JSONL, add a `domain` metadata column, perform stratified split preserving domain proportions in each split, strip internal `_quality` metadata, handle the `tools` column correctly (present only in tool-calling samples), and save as HuggingFace Arrow-backed DatasetDict with train/validation/test splits.

**Primary recommendation:** Use HuggingFace `datasets` library's `Dataset.from_list()` with `on_mixed_types="use_json"` to handle the heterogeneous `tools` column, then `train_test_split(stratify_by_column="domain")` applied twice to produce 90/5/5 splits. Save with `save_to_disk()` for local use and design for `push_to_hub()` compatibility.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- D-01: Use ALL curated samples from all 3 domains without downsampling. Accept natural imbalance (tool-calling ~68%, code ~16.5%, knowledge ~15.4%).
- D-02: Total dataset: ~3,630 samples (2,470 tool-calling + 600 code + 560 knowledge).
- D-03: Original 33/33/33 target revised -- imbalance accepted.
- D-04: 90/5/5 train/validation/test split.
- D-05: Stratified splits -- each split contains proportional representation of all 3 domains.
- D-06: Approximate split sizes: ~3,267 train / ~182 validation / ~182 test.
- D-07: HuggingFace `datasets` library format (Arrow-backed). Train/validation/test as named splits.
- D-08: Include a `domain` metadata column (values: "tool-calling", "code", "knowledge").
- D-09: Dataset ready for push_to_hub and direct consumption by Unsloth/TRL SFTTrainer.
- D-10: Assembly script reads curated JSONL from all 3 domain directories, adds domain metadata, performs stratified split, validates entire dataset, and saves as HF dataset format.

### Claude's Discretion
- Script naming and internal organization
- Whether to also output JSONL alongside HF format
- Exact stratification algorithm (sklearn train_test_split or manual)
- Stats command implementation details
- Directory structure for final assembled dataset

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| DATA-07 | Dataset includes stratified train/validation/test splits across all three focus areas | HF datasets `train_test_split(stratify_by_column=)` provides native stratification; 90/5/5 split ratios confirmed achievable with 3,630 samples |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| datasets (HuggingFace) | 4.8.4 | Dataset creation, splitting, Arrow storage, Hub publishing | Already in project tech stack (CLAUDE.md). Native stratified split support, Arrow-backed for efficiency, push_to_hub built-in. [VERIFIED: CLAUDE.md tech stack] |
| Python stdlib (json, pathlib) | 3.10+ | JSONL loading, path handling | No external deps for I/O. Established pattern in project scripts. [VERIFIED: codebase inspection] |
| pydantic | 2.12.5 | Validation reuse (Conversation model) | Already a project dependency. Reuse existing validate_format.py Conversation model. [VERIFIED: requirements.txt] |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| scikit-learn | 1.8.x | Alternative stratification if HF's built-in is insufficient | Only if `train_test_split(stratify_by_column=)` edge cases arise with small classes. NOT needed -- HF's native stratification handles this case. [ASSUMED] |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| HF `train_test_split` | sklearn `train_test_split` | sklearn requires explicit label array, adds a dependency. HF's native method is simpler and returns DatasetDict directly. |
| `Dataset.from_list()` | `load_dataset("json", data_files=)` | load_dataset requires pre-split files. from_list is more direct when building from in-memory data. |
| `save_to_disk()` | Writing Parquet/JSONL manually | Loses Arrow metadata, dataset card, Hub compatibility. save_to_disk preserves everything. |

**Installation:**
```bash
pip install datasets>=4.8.0
```

Note: `datasets` is already specified in CLAUDE.md tech stack but NOT in requirements.txt. It must be added to requirements.txt.

## Architecture Patterns

### Recommended Project Structure
```
scripts/
  assemble_dataset.py     # Main assembly script (CLI entry point)
datasets/
  tool-calling/curated/tool-calling-curated.jsonl   # Input (2,470 samples)
  code/curated/code-curated.jsonl                   # Input (600 samples)
  knowledge/curated/knowledge-curated.jsonl         # Input (560 samples)
  assembled/                                         # Output directory
    dataset_dict.json                               # HF DatasetDict metadata
    train/                                          # Arrow files for train split
    validation/                                     # Arrow files for validation split
    test/                                           # Arrow files for test split
    stats.json                                      # Assembly statistics
```

### Pattern 1: Two-Pass Stratified Splitting for 3-Way Split
**What:** HF's `train_test_split` only creates 2 splits at a time. To get train/validation/test, split twice.
**When to use:** Always for 90/5/5 three-way splits.
**Example:**
```python
# Source: HuggingFace datasets docs - train_test_split
from datasets import Dataset, DatasetDict

# First split: 90% train_temp / 10% remaining
split1 = dataset.train_test_split(
    test_size=0.1,  # 10% goes to temp
    stratify_by_column="domain",
    seed=42,
)

# Second split: 50/50 of the 10% remaining -> 5% val, 5% test
split2 = split1["test"].train_test_split(
    test_size=0.5,
    stratify_by_column="domain",
    seed=42,
)

# Assemble final DatasetDict
final = DatasetDict({
    "train": split1["train"],       # ~3,267 samples
    "validation": split2["train"],  # ~182 samples
    "test": split2["test"],         # ~182 samples
})
```

### Pattern 2: Loading and Merging JSONL with Domain Metadata
**What:** Load all three JSONL files, add domain column, strip internal metadata.
**When to use:** During assembly step before splitting.
**Example:**
```python
# Source: Project codebase patterns + HF datasets docs
import json
from pathlib import Path
from datasets import Dataset

def load_domain_jsonl(path: Path, domain: str) -> list[dict]:
    """Load JSONL and add domain metadata, strip _quality."""
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            # Strip internal pipeline metadata
            sample.pop("_quality", None)
            # Add domain metadata column
            sample["domain"] = domain
            samples.append(sample)
    return samples

# Merge all domains
all_samples = []
all_samples.extend(load_domain_jsonl(
    Path("datasets/tool-calling/curated/tool-calling-curated.jsonl"), "tool-calling"))
all_samples.extend(load_domain_jsonl(
    Path("datasets/code/curated/code-curated.jsonl"), "code"))
all_samples.extend(load_domain_jsonl(
    Path("datasets/knowledge/curated/knowledge-curated.jsonl"), "knowledge"))

# Create HF Dataset -- on_mixed_types handles tools column
dataset = Dataset.from_list(all_samples, on_mixed_types="use_json")
```

### Pattern 3: TRL SFTTrainer Compatibility
**What:** TRL v1.2.0 SFTTrainer expects `messages` column in conversational format, optional `tools` column.
**When to use:** Final dataset must be consumable by SFTTrainer directly.
**Example:**
```python
# Source: TRL v1.2.0 SFTTrainer docs - dataset_formats
# SFTTrainer consumes this directly:
from datasets import load_from_disk
from trl import SFTTrainer

dataset = load_from_disk("datasets/assembled")
trainer = SFTTrainer(
    model="unsloth/SmolLM2-1.7B-Instruct",
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
)
```

### Pattern 4: Statistics Report
**What:** Print domain balance per split for verification.
**When to use:** After assembly completes, and as a standalone stats command.
**Example:**
```python
import json

def compute_stats(dataset_dict) -> dict:
    """Compute per-split domain distribution stats."""
    stats = {}
    for split_name, split_data in dataset_dict.items():
        domain_counts = {}
        for domain in split_data["domain"]:
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        total = len(split_data)
        stats[split_name] = {
            "total": total,
            "domains": {
                d: {"count": c, "percent": round(c / total * 100, 1)}
                for d, c in sorted(domain_counts.items())
            },
        }
    return stats
```

### Anti-Patterns to Avoid
- **Loading all JSONL into one giant string then splitting:** Loses structure, wastes memory. Load as list of dicts.
- **Using sklearn for stratification when HF native works:** Adds unnecessary dependency for a simple single-label stratification case.
- **Keeping `_quality` metadata in final dataset:** Internal pipeline metadata should not be in training data. Strip before assembly.
- **Hardcoding sample counts instead of reading actual files:** Counts may drift. Always count from actual file contents.
- **Creating JSONL intermediate before HF format:** Unnecessary step. Go directly from in-memory to HF Dataset.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Stratified splitting | Manual index shuffling with proportion math | `Dataset.train_test_split(stratify_by_column=)` | Edge cases with rounding, small classes, reproducibility |
| Arrow serialization | Custom binary format or Parquet writer | `DatasetDict.save_to_disk()` | Handles sharding, metadata, schema, hub compatibility |
| Hub publishing | Manual git LFS + README generation | `DatasetDict.push_to_hub()` | Handles auth, dataset card, format conversion, versioning |
| Format validation | New validator for assembled data | Existing `scripts/validate_format.py` Conversation model | Already handles all structural rules, tested |
| JSON type handling for tools column | Manual schema definition with Features() | `on_mixed_types="use_json"` | Auto-detects and handles arbitrary JSON in columns |

**Key insight:** The HuggingFace `datasets` library handles all the complex parts (Arrow format, stratification, Hub publishing). The assembly script just needs to load, merge, tag, and orchestrate.

## Common Pitfalls

### Pitfall 1: Tools Column Type Mismatch
**What goes wrong:** Tool-calling samples have `tools` key (list of dicts), code/knowledge samples do not. When merged, Arrow type inference fails or creates inconsistent schema.
**Why it happens:** Arrow requires uniform column types. Mixed None/list causes type errors without explicit handling.
**How to avoid:** Use `on_mixed_types="use_json"` when calling `Dataset.from_list()`. Alternatively, set `tools` to `None` explicitly for samples without tools before creating the Dataset.
**Warning signs:** `ArrowInvalid` errors during Dataset creation, or tools column silently dropped.

### Pitfall 2: Stratification with Small Classes
**What goes wrong:** With only ~28 knowledge samples per split (5% of 560), stratification might fail or produce uneven splits.
**Why it happens:** `train_test_split` requires at least 1 sample per class per split. With test_size=0.1 first pass, knowledge gets ~56 samples in the 10% remainder, then split 50/50 = ~28 each.
**How to avoid:** Verify minimum class size before splitting. With 560 knowledge samples, 5% = 28 which is safe for stratification (sklearn's StratifiedShuffleSplit handles this). Set seed for reproducibility.
**Warning signs:** ValueError about class size or empty domains in a split.

### Pitfall 3: Forgetting to Strip _quality Metadata
**What goes wrong:** Internal `_quality` scoring metadata ends up in the training dataset, adding noise or confusing downstream tools.
**Why it happens:** Curated JSONL files include `_quality` as a side-effect of the curation pipeline.
**How to avoid:** Explicitly `pop("_quality", None)` from every sample during loading.
**Warning signs:** Extra columns in the final dataset beyond `messages`, `tools`, `domain`.

### Pitfall 4: Non-Reproducible Splits
**What goes wrong:** Running the assembly script twice produces different splits, making evaluation comparisons invalid.
**Why it happens:** Default shuffle uses random seed.
**How to avoid:** Always pass explicit `seed=42` (or configurable seed) to `train_test_split`.
**Warning signs:** Different sample counts or validation loss across identical training runs.

### Pitfall 5: Validation Not Running on Final Assembled Data
**What goes wrong:** Assembly introduces format errors (e.g., stripping tools column incorrectly, mangling message structure), discovered only during training.
**Why it happens:** Validation was run on per-domain curated data but not on the final merged dataset.
**How to avoid:** Re-run `validate_format.py` on every sample in the final assembled dataset as a post-assembly step.
**Warning signs:** Training crashes with template errors or assertion failures in SFTTrainer.

### Pitfall 6: Dataset Too Large for Git
**What goes wrong:** Arrow files are binary and large (~10-20MB for 3,630 samples). Git LFS needed or gitignore required.
**Why it happens:** Arrow format is not diff-friendly.
**How to avoid:** Add `datasets/assembled/` to `.gitignore`. Only the assembly script and source JSONL need version control. The assembled dataset is a reproducible artifact.
**Warning signs:** Large git repo, slow clones, GitHub file size warnings.

## Code Examples

### Complete Assembly Flow
```python
# Source: Derived from HF datasets docs + project patterns
#!/usr/bin/env python3
"""assemble_dataset.py -- Merge curated domain datasets into final HF dataset."""
import argparse
import json
import logging
from pathlib import Path

from datasets import Dataset, DatasetDict

logger = logging.getLogger(__name__)

DOMAIN_SOURCES = {
    "tool-calling": "datasets/tool-calling/curated/tool-calling-curated.jsonl",
    "code": "datasets/code/curated/code-curated.jsonl",
    "knowledge": "datasets/knowledge/curated/knowledge-curated.jsonl",
}

def load_domain(path: Path, domain: str) -> list[dict]:
    """Load JSONL, strip _quality, add domain column."""
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            sample.pop("_quality", None)
            sample["domain"] = domain
            # Ensure tools column exists (None for non-tool samples)
            if "tools" not in sample:
                sample["tools"] = None
            samples.append(sample)
    return samples

def assemble(output_dir: Path, seed: int = 42) -> dict:
    """Main assembly: load, merge, split, validate, save."""
    # Load all domains
    all_samples = []
    for domain, rel_path in DOMAIN_SOURCES.items():
        path = Path(rel_path)
        samples = load_domain(path, domain)
        logger.info("Loaded %d samples from %s", len(samples), domain)
        all_samples.extend(samples)

    logger.info("Total samples: %d", len(all_samples))

    # Create HF Dataset
    dataset = Dataset.from_list(all_samples, on_mixed_types="use_json")

    # Two-pass stratified split: 90/5/5
    split1 = dataset.train_test_split(
        test_size=0.1, stratify_by_column="domain", seed=seed
    )
    split2 = split1["test"].train_test_split(
        test_size=0.5, stratify_by_column="domain", seed=seed
    )

    final = DatasetDict({
        "train": split1["train"],
        "validation": split2["train"],
        "test": split2["test"],
    })

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    final.save_to_disk(str(output_dir))

    return final
```

### Validation Integration
```python
# Source: Project codebase -- scripts/validate_format.py reuse
from scripts.validate_format import Conversation

def validate_assembled(dataset_dict: DatasetDict) -> dict:
    """Run format validation on all samples across all splits."""
    results = {"total": 0, "valid": 0, "invalid": 0, "errors": []}
    for split_name, split_data in dataset_dict.items():
        for i, sample in enumerate(split_data):
            results["total"] += 1
            try:
                # Build validation dict (messages + optional tools)
                val_dict = {"messages": sample["messages"]}
                if sample.get("tools"):
                    val_dict["tools"] = sample["tools"]
                Conversation.model_validate(val_dict)
                results["valid"] += 1
            except Exception as e:
                results["invalid"] += 1
                results["errors"].append({
                    "split": split_name, "index": i, "error": str(e)
                })
    return results
```

### Stats Command
```python
# Source: Derived from project patterns
def print_stats(dataset_dict: DatasetDict) -> None:
    """Print domain distribution per split as a table."""
    print(f"{'Split':<12} {'Total':>6} {'tool-calling':>14} {'code':>8} {'knowledge':>11}")
    print("-" * 55)
    for split_name in ["train", "validation", "test"]:
        split = dataset_dict[split_name]
        total = len(split)
        domains = {}
        for d in split["domain"]:
            domains[d] = domains.get(d, 0) + 1
        tc = domains.get("tool-calling", 0)
        co = domains.get("code", 0)
        kn = domains.get("knowledge", 0)
        print(
            f"{split_name:<12} {total:>6} "
            f"{tc:>5} ({tc/total*100:.1f}%) "
            f"{co:>4} ({co/total*100:.1f}%) "
            f"{kn:>4} ({kn/total*100:.1f}%)"
        )
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Classic ShareGPT (from/value) | TRL-native (messages/role/content) | TRL v1.0 (2026-04) | No format conversion needed at training time |
| Manual Arrow/Parquet construction | `Dataset.from_list(on_mixed_types="use_json")` | datasets 4.x | Handles heterogeneous JSON columns automatically |
| dataset_text_field formatting | Direct messages column consumption | TRL v1.0+ | SFTTrainer auto-applies chat template to messages |

**Deprecated/outdated:**
- `dataset_text_field` parameter: Still supported but `messages` column is the modern approach for conversational data
- Manual `apply_chat_template` before training: TRL now does this automatically when it sees a `messages` column

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `on_mixed_types="use_json"` available in datasets 4.8.4 | Architecture Patterns | Would need manual Features() definition with Json() type -- more code but still works |
| A2 | 28 samples per class in test/validation is sufficient for stratified split to succeed | Common Pitfalls | Would need to fallback to manual proportional sampling |
| A3 | scikit-learn not needed for this single-label stratification | Standard Stack | HF datasets uses sklearn internally anyway; if it fails, install sklearn explicitly |

## Open Questions

1. **Whether to output JSONL alongside HF format**
   - What we know: HF format is the primary output per D-07. JSONL is the intermediate format used throughout the project.
   - What's unclear: Whether downstream consumers (evaluation scripts, debugging) benefit from JSONL alongside Arrow.
   - Recommendation: Output both -- a combined JSONL for human inspection and the HF Arrow format for training. Minimal extra code.

2. **Gitignore strategy for assembled output**
   - What we know: Arrow files are binary, large, not diff-friendly. Source JSONL already in git.
   - What's unclear: Whether assembled dataset should be git-tracked or treated as build artifact.
   - Recommendation: Add `datasets/assembled/` to .gitignore. Assembly is reproducible from source JSONL + script.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.10+ | All scripts | Yes | 3.14.2 | -- |
| datasets (HuggingFace) | Assembly, saving, splitting | No (not installed) | -- | Must install: `pip install datasets>=4.8.0` |
| pydantic | Validation reuse | Yes | 2.12.5 | -- |
| pytest | Tests | Yes | Available | -- |

**Missing dependencies with no fallback:**
- `datasets` library must be installed. It is specified in CLAUDE.md tech stack but not in requirements.txt and not currently installed.

**Missing dependencies with fallback:**
- None. All other dependencies are available.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 7.0+ |
| Config file | None explicit (uses pytest defaults) |
| Quick run command | `python -m pytest tests/test_assemble_dataset.py -x -q` |
| Full suite command | `python -m pytest tests/ -x -q` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DATA-07-a | All 3 domains present in each split | unit | `python -m pytest tests/test_assemble_dataset.py::test_all_domains_in_each_split -x` | Wave 0 |
| DATA-07-b | Stratified proportions preserved (within 5% tolerance) | unit | `python -m pytest tests/test_assemble_dataset.py::test_stratified_proportions -x` | Wave 0 |
| DATA-07-c | Total sample count matches expected 3,630 | unit | `python -m pytest tests/test_assemble_dataset.py::test_total_count -x` | Wave 0 |
| DATA-07-d | Split sizes approximate 90/5/5 | unit | `python -m pytest tests/test_assemble_dataset.py::test_split_ratios -x` | Wave 0 |
| DATA-07-e | domain metadata column present with correct values | unit | `python -m pytest tests/test_assemble_dataset.py::test_domain_column -x` | Wave 0 |
| DATA-07-f | Full validation pipeline passes on assembled data | integration | `python -m pytest tests/test_assemble_dataset.py::test_validation_passes -x` | Wave 0 |
| DATA-07-g | _quality metadata stripped from output | unit | `python -m pytest tests/test_assemble_dataset.py::test_no_quality_metadata -x` | Wave 0 |
| DATA-07-h | tools column present (non-null for tool-calling, null for others) | unit | `python -m pytest tests/test_assemble_dataset.py::test_tools_column -x` | Wave 0 |
| DATA-07-i | Reproducible splits with same seed | unit | `python -m pytest tests/test_assemble_dataset.py::test_reproducibility -x` | Wave 0 |
| DATA-07-j | Stats command outputs correct domain distribution | unit | `python -m pytest tests/test_assemble_dataset.py::test_stats_output -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `python -m pytest tests/test_assemble_dataset.py -x -q`
- **Per wave merge:** `python -m pytest tests/ -x -q`
- **Phase gate:** Full suite green before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `tests/test_assemble_dataset.py` -- covers DATA-07 (all sub-requirements a through j)
- [ ] Install `datasets` library: `pip install datasets>=4.8.0`
- [ ] Add `datasets>=4.8.0` to `requirements.txt`

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | No | N/A (local scripts, no auth) |
| V3 Session Management | No | N/A |
| V4 Access Control | No | N/A |
| V5 Input Validation | Yes | Pydantic Conversation model for format validation; json.loads for parsing |
| V6 Cryptography | No | N/A |

### Known Threat Patterns for This Stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Malformed JSONL injection | Tampering | Line-by-line json.loads with try/except (established project pattern) |
| Path traversal in file paths | Tampering | pathlib.Path with hardcoded relative paths; no user-supplied paths in assembly |
| Arbitrary code in JSON | Elevation of Privilege | json.loads (safe parser), no eval/exec. Pydantic validation catches unexpected fields |

## Sources

### Primary (HIGH confidence)
- [HuggingFace datasets main classes docs](https://huggingface.co/docs/datasets/en/package_reference/main_classes) -- Dataset.train_test_split signature, stratify_by_column, save_to_disk, from_list
- [TRL v1.2.0 SFTTrainer docs](https://huggingface.co/docs/trl/en/sft_trainer) -- Dataset format expectations, messages column, tools column, eval_dataset
- [TRL v1.2.0 Dataset formats docs](https://huggingface.co/docs/trl/v1.2.0/en/dataset_formats) -- Tool calling format with tools column, on_mixed_types="use_json"
- Project codebase inspection -- validate_format.py, curate_pipeline.py, curated JSONL structure

### Secondary (MEDIUM confidence)
- [HuggingFace forums - DatasetDict splits](https://discuss.huggingface.co/t/how-to-split-main-dataset-into-train-dev-test-as-datasetdict/1090) -- Two-pass splitting pattern
- [scikit-learn train_test_split docs](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) -- Stratification mechanics (HF uses this internally)

### Tertiary (LOW confidence)
- None -- all critical claims verified against official docs.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - datasets library confirmed in project tech stack and verified against official docs
- Architecture: HIGH - Two-pass splitting pattern verified against HF docs; tool calling format verified against TRL docs
- Pitfalls: HIGH - Based on observed data structure (tools column heterogeneity) and documented HF behaviors

**Research date:** 2026-04-20
**Valid until:** 2026-05-20 (stable -- datasets library API unlikely to change)
