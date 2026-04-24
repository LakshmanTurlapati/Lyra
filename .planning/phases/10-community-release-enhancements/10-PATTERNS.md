# Phase 10: Community Release Enhancements - Pattern Map

**Mapped:** 2026-04-24
**Files analyzed:** 13 (5 new, 8 modified)
**Analogs found:** 10 / 13 (3 files have no close analog — see §No Analog Found)

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `scripts/convert_gguf.sh` (NEW) | script (shell wrapper) | subprocess-orchestration / file-I/O | `scripts/eval_runner.py` (subprocess T-03-05 pattern) + `scripts/eval_inference.py::main` (arg parsing + output path handling) | partial (no existing `.sh` in repo) |
| `scripts/verify_gguf.py` (NEW) | script (CLI) | file-I/O / validation | `scripts/eval_merge.py` (thin CLI reading one file, validating, exit-code) | exact (thin one-file validator pattern) |
| `tests/test_gguf_conversion.py` (NEW) | test | unit / file-existence + mock | `tests/test_release_artifacts.py` (artifact-presence) + `tests/test_eval_runner.py` (`MagicMock`/`patch.dict` for lazy imports) | exact |
| `tests/test_dataset_versioning.py` (NEW) | test | unit / regex + SemVer + file-existence | `tests/test_release_artifacts.py` (REPO_ROOT pattern) | exact |
| `CHANGELOG.md` (NEW) | docs (repo-root) | static markdown | none in repo — use Keep-a-Changelog 1.1.0 (RESEARCH.md §Code Examples) | no analog |
| `.gitattributes` (MODIFY) | config | declarative | current file is placeholder comments only — see RESEARCH.md §Pitfall 4 | extend-in-place |
| `scripts/assemble_dataset.py` (MODIFY) | script (augment) | CRUD / stats emission | self (existing `compute_stats` + `print_stats` at lines 154-224) | exact (extend in-place) |
| `tests/test_release_artifacts.py` (MODIFY) | test | unit / file-existence | self (lines 9-46) | exact (extend in-place) |
| `tests/test_assemble_dataset.py` (MODIFY) | test | unit / fixtures | self (lines 20-159 fixtures; 208-399 tests) | exact (extend in-place) |
| `requirements.txt` (MODIFY) | config | declarative | self (add `gguf==0.18.0`) | exact (one-line addition) |
| `README.md` (MODIFY) | docs (repo-root) | static markdown | self (existing model-card structure, lines 1-12 frontmatter, 24-32 core features, 69-79 setup) | exact (append section) |
| `datasets/README.md` (MODIFY) | docs | static markdown | self (lines 29-40 stats table) | exact (extend stats section) |
| `.planning/REQUIREMENTS.md` (MODIFY) | planning doc | static markdown | self (lines 63-65 v1 Active; 85-94 Out of Scope table; 100-133 Traceability) | exact (move REL-06 row) |
| `.planning/ROADMAP.md` (MODIFY) | planning doc | static markdown | self (lines 190-199 Phase 10 block) | exact (edit goal + success criteria) |

---

## Pattern Assignments

### `scripts/convert_gguf.sh` (script, subprocess-orchestration)

**Analog:** `scripts/eval_runner.py` (subprocess discipline) + `scripts/eval_inference.py::main` (arg handling). No existing `.sh` file in the repo — this is the first shell script under `scripts/`. Structure follows the RESEARCH.md §Pattern 1 shell skeleton but must adopt the Lyra security/validation discipline visible in `scripts/eval_runner.py`.

**Shebang + license header pattern** (every script in `scripts/` uses SPDX, `scripts/eval_inference.py` lines 1-2, `scripts/eval_runner.py` lines 1-2, `scripts/assemble_dataset.py` lines 1-2):
```bash
#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# scripts/convert_gguf.sh -- reproducible GGUF conversion pipeline (Phase 10 D-04).
#
# Usage:
#   scripts/convert_gguf.sh <model_dir> <output_prefix>
#
# Requires: llama.cpp CLI tools on $PATH (brew install llama.cpp) + Python 3.10+
#
# Threat mitigations (mirrors scripts/eval_runner.py discipline):
#   T-03-05: external tools invoked with list-form args (no shell=True equivalent)
#   T-03-07: positional args validated before substitution
```

**Strict-mode and failure pattern** (adopted from RESEARCH.md §Pattern 1; no existing analog in the repo because no existing `.sh` exists):
```bash
set -euo pipefail
```

**Positional-arg validation pattern** (mirrors `scripts/eval_runner.py::_validate_model_path` lines 51-70 — every script that takes a model path validates it first):
```bash
MODEL_DIR="${1:-models/lyra-merged}"
OUTPUT_PREFIX="${2:-lyra-v1.0}"

# Mirror T-03-07 from scripts/eval_runner.py: refuse paths with shell metacharacters
# or traversal components. Accept only [a-zA-Z0-9._/~\-]+
if ! [[ "$MODEL_DIR" =~ ^[a-zA-Z0-9._/~\-]+$ ]]; then
    echo "Error: invalid model dir: $MODEL_DIR" >&2
    exit 1
fi
if [ ! -d "$MODEL_DIR" ]; then
    echo "Error: model dir not found: $MODEL_DIR" >&2
    exit 1
fi

# Precondition check per RESEARCH.md Pitfall 1 — fail loudly if chat_template missing
python -c "import json,sys; d=json.load(open('$MODEL_DIR/tokenizer_config.json')); \
           sys.exit(0 if d.get('chat_template') else 1)" \
    || { echo "Error: $MODEL_DIR/tokenizer_config.json has no chat_template field" >&2; exit 1; }
```

**Pipeline orchestration pattern** (RESEARCH.md §Pattern 1, lines 247-282):
```bash
OUTPUT_DIR="build/gguf"
mkdir -p "$OUTPUT_DIR"

# Step 1: HF -> f16 GGUF (chat_template auto-embedded)
python "${LLAMA_CPP_DIR:-/opt/homebrew/share/llama.cpp}/convert_hf_to_gguf.py" \
    "$MODEL_DIR" \
    --outfile "$OUTPUT_DIR/${OUTPUT_PREFIX}-f16.gguf" \
    --outtype f16

# Step 2: f16 -> Q4_K_M
llama-quantize "$OUTPUT_DIR/${OUTPUT_PREFIX}-f16.gguf" \
    "$OUTPUT_DIR/${OUTPUT_PREFIX}-q4_k_m.gguf" Q4_K_M

# Step 3: f16 -> Q8_0
llama-quantize "$OUTPUT_DIR/${OUTPUT_PREFIX}-f16.gguf" \
    "$OUTPUT_DIR/${OUTPUT_PREFIX}-q8_0.gguf" Q8_0

# Step 4: verify postcondition (Pitfall 1 mitigation)
python scripts/verify_gguf.py "$OUTPUT_DIR/${OUTPUT_PREFIX}-q4_k_m.gguf"
python scripts/verify_gguf.py "$OUTPUT_DIR/${OUTPUT_PREFIX}-q8_0.gguf"
```

**Permissions note:** The script must be `chmod +x` so it is invokable as `scripts/convert_gguf.sh`. No existing repo precedent — document in README "Setup" section.

---

### `scripts/verify_gguf.py` (script, file-I/O / validation)

**Analog:** `scripts/eval_merge.py` (closest match — thin CLI that reads one file, validates it against a schema, returns exit code). Also adopts the lazy-import discipline from `scripts/eval_inference.py::_do_load_model_and_tokenizer` (lines 111-144).

**Header + docstring pattern** (lines 1-18 of `scripts/eval_merge.py`):
```python
#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""verify_gguf.py -- Verify GGUF metadata meets Lyra release requirements (Phase 10 D-07).

Asserts that:
  1. `tokenizer.chat_template` key is present in GGUF metadata
  2. Template content contains the SmolLM2 `<|im_start|>` marker
  3. `general.architecture` and `general.name` are populated

Closes the same class of runtime template-drift bug that caused the Phase 09.1
D-03/D-04 regressions (see .planning/phases/09.1/).

Usage:
  python3 -m scripts.verify_gguf build/gguf/lyra-v1.0-q4_k_m.gguf

Threat mitigations:
  T-03-02: gguf.GGUFReader validates file structure at the trust boundary
  T-03-07: input path validated before open
"""
```

**Imports + logger pattern** (matches `scripts/eval_merge.py` lines 19-23 and `scripts/eval_inference.py` lines 25-37):
```python
import argparse
import logging
import re
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Input path validation (mirrors scripts/eval_runner.py MODEL_PATH_PATTERN T-03-07)
_GGUF_PATH_PATTERN = re.compile(r"^[a-zA-Z0-9._/~\-]+$")
```

**Lazy-import pattern** (adopt from `scripts/eval_inference.py` lines 111-144, and CLAUDE-confirmed convention from RESEARCH.md §Project Constraints "Lazy imports of heavy ML deps"):
```python
def _load_gguf_reader(gguf_path: str):
    """Lazy-import gguf so tests can monkeypatch without the package installed."""
    from gguf import GGUFReader
    return GGUFReader(gguf_path, mode="r")
```

**Core validation pattern** (mirrors RESEARCH.md §Pattern 2; exit-code contract mirrors `scripts/eval_merge.py::main` lines 65-110 return-0-or-1):
```python
REQUIRED_KEYS = ["tokenizer.chat_template", "general.architecture", "general.name"]
SMOLLM2_MARKER = "<|im_start|>"

def verify(gguf_path: Path) -> int:
    reader = _load_gguf_reader(str(gguf_path))
    missing = [k for k in REQUIRED_KEYS if k not in reader.fields]
    if missing:
        print(f"FAIL: {gguf_path} missing metadata keys: {missing}", file=sys.stderr)
        return 1
    chat_template = reader.fields["tokenizer.chat_template"].contents()
    if SMOLLM2_MARKER not in chat_template:
        print(f"FAIL: {gguf_path} chat_template missing {SMOLLM2_MARKER!r} marker", file=sys.stderr)
        return 1
    print(f"OK: {gguf_path} -- chat_template embedded ({len(chat_template)} chars)")
    return 0
```

**CLI entry-point pattern** (exactly mirrors `scripts/eval_merge.py::main` + `__main__` block, lines 65-110):
```python
def main() -> int:
    parser = argparse.ArgumentParser(description="Verify GGUF metadata for Lyra release.")
    parser.add_argument("gguf_path", type=Path, help="Path to GGUF file to verify.")
    args = parser.parse_args()

    if not args.gguf_path.exists():
        print(f"Error: file not found: {args.gguf_path}", file=sys.stderr)
        return 1
    if not _GGUF_PATH_PATTERN.match(str(args.gguf_path)):
        print(f"Error: invalid path characters: {args.gguf_path}", file=sys.stderr)
        return 1

    try:
        return verify(args.gguf_path)
    except Exception as exc:
        print(f"Error verifying GGUF: {exc}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    sys.exit(main())
```

---

### `tests/test_gguf_conversion.py` (test, unit / mocking)

**Analog:** `tests/test_release_artifacts.py` (REPO_ROOT + file-existence discipline, lines 1-46) + `tests/test_eval_runner.py` (`MagicMock` + `patch.dict("sys.modules", {...})` for lazy-imported packages that are not installed, lines 29-44).

**File header + imports pattern** (matches `tests/test_release_artifacts.py` lines 1-6 and `tests/test_eval_runner.py` lines 1-23):
```python
#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Unit tests for scripts/convert_gguf.sh + scripts/verify_gguf.py (Phase 10 REL-05).

Tests exercise:
  - verify_gguf.py exit codes (OK / missing keys / bad template marker)
  - convert_gguf.sh positional-arg validation via subprocess invocation
  - GGUF metadata shape using mocked gguf.GGUFReader (no real conversion)

RED-state stubs land in Wave 0 per Phase 09.2 convention
(see .planning/phases/09.2-tool-call-regression-diagnosis/09.2-01-PLAN.md
for Wave 0 + RED-first discipline).
"""
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).parent.parent
```

**Lazy-import-friendly mock pattern** (exactly mirrors `tests/test_eval_runner.py::test_detect_device_mps` lines 29-36 — the key trick is `patch.dict("sys.modules", {"gguf": mock_gguf})` so the script's `from gguf import GGUFReader` binds to the mock):
```python
def test_verify_gguf_accepts_good_metadata():
    """verify_gguf.py exits 0 when all required keys present + SmolLM2 marker found."""
    mock_field = MagicMock()
    mock_field.contents.return_value = "{% for m in messages %}<|im_start|>{{m.role}}..."
    mock_reader = MagicMock()
    mock_reader.fields = {
        "tokenizer.chat_template": mock_field,
        "general.architecture": MagicMock(),
        "general.name": MagicMock(),
    }
    mock_gguf = MagicMock()
    mock_gguf.GGUFReader.return_value = mock_reader

    with patch.dict("sys.modules", {"gguf": mock_gguf}):
        from scripts.verify_gguf import verify
        assert verify(Path("fake.gguf")) == 0
```

**Subprocess invocation pattern for shell-script testing** (no existing precedent in repo; this is the convention established in RESEARCH.md §Validation Architecture and Wave 0 Gaps):
```python
def test_convert_gguf_rejects_bad_path():
    """convert_gguf.sh exits non-zero on invalid model dir arg."""
    result = subprocess.run(
        [str(REPO_ROOT / "scripts" / "convert_gguf.sh"), "nonexistent_dir_!!"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode != 0
    assert "invalid" in result.stderr.lower() or "not found" in result.stderr.lower()
```

**RED-state stub pattern** (every Phase 09.2 test starts RED in Wave 0 then turns GREEN by implementation — see `tests/test_phase_09_2/test_template_parity.py` for the reference style; mark RED tests with `@pytest.mark.xfail` or leave as `assert False` with a TODO docstring):
```python
@pytest.mark.xfail(reason="Wave 0 RED stub -- GREEN in Plan 1 task T-10-01-04")
def test_verify_gguf_rejects_missing_template():
    """verify_gguf.py exits 1 when tokenizer.chat_template key is absent."""
    # Implementation lands with the scripts/verify_gguf.py source in Plan 1.
    raise NotImplementedError
```

---

### `tests/test_dataset_versioning.py` (test, unit / regex + file-existence)

**Analog:** `tests/test_release_artifacts.py` (same REPO_ROOT + file-existence assertion style, lines 1-46).

**Full-file pattern** (mirrors `tests/test_release_artifacts.py` structure almost verbatim):
```python
#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Smoke tests for dataset versioning scheme (Phase 10 REL-07 / D-08).

Validates:
  - Dataset version tag regex (`dataset-vMAJOR.MINOR.PATCH`)
  - CHANGELOG.md exists at repo root with Keep-a-Changelog 1.1.0 scaffolding
  - datasets/README.md has a "Dataset Versions" section with link to Releases
"""
import re
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent

DATASET_TAG_PATTERN = re.compile(r"^dataset-v\d+\.\d+\.\d+$")


def test_dataset_tag_format_accepts_valid():
    """Valid SemVer dataset tags match the dataset-v prefix regex."""
    for tag in ["dataset-v1.0.0", "dataset-v1.2.10", "dataset-v10.0.0"]:
        assert DATASET_TAG_PATTERN.match(tag), f"{tag} should be valid"


def test_dataset_tag_format_rejects_invalid():
    """Non-SemVer / wrong-prefix tags are rejected."""
    for tag in ["v1.0.0", "dataset-1.0.0", "dataset-v1.0", "data-v1.0.0"]:
        assert not DATASET_TAG_PATTERN.match(tag), f"{tag} should be rejected"


def test_changelog_exists():
    """CHANGELOG.md exists at repo root with Keep-a-Changelog header."""
    path = REPO_ROOT / "CHANGELOG.md"
    assert path.exists(), "CHANGELOG.md not found at repo root"
    text = path.read_text()
    assert "Keep a Changelog" in text, "CHANGELOG.md does not reference Keep a Changelog"
    assert "Semantic Versioning" in text, "CHANGELOG.md does not reference Semantic Versioning"
```

---

### `scripts/assemble_dataset.py` (MODIFY — stats augmentation)

**Analog:** self. The existing `compute_stats` (lines 154-180) and `print_stats` (lines 183-224) and the `stats` subcommand wiring (lines 298-348) provide the exact extension points.

**Existing `compute_stats` signature to extend** (`scripts/assemble_dataset.py` lines 154-180):
```python
def compute_stats(dataset_dict: DatasetDict) -> dict:
    """Compute per-split domain distribution statistics."""
    stats = {}
    for split_name in dataset_dict:
        split_data = dataset_dict[split_name]
        total = len(split_data)
        domain_counts = {}
        for domain_val in split_data["domain"]:
            domain_counts[domain_val] = domain_counts.get(domain_val, 0) + 1
        domains_stats = {}
        for domain, count in sorted(domain_counts.items()):
            pct = (count / total * 100) if total > 0 else 0.0
            domains_stats[domain] = {"count": count, "percent": round(pct, 1)}
        stats[split_name] = {"total": total, "domains": domains_stats}
    return stats
```

**Extension shape per RESEARCH.md §Code Examples / Open Question #4** — add top-level version metadata and keep backward-compatible per-split structure. Add `--json` and `--output PATH` flags to the **existing `stats` subcommand**; do not create a new script. Pattern for subcommand extension lives at lines 298-306:
```python
# stats subcommand (scripts/assemble_dataset.py lines 298-306) — extend:
stats_parser = subparsers.add_parser(
    "stats", help="Print domain distribution statistics"
)
stats_parser.add_argument(
    "--dataset-dir",
    type=str,
    default=DEFAULT_OUTPUT_DIR,
    help=f"Path to saved DatasetDict (default: {DEFAULT_OUTPUT_DIR})",
)
# NEW for Phase 10:
stats_parser.add_argument(
    "--json", action="store_true",
    help="Emit machine-readable JSON instead of the human-readable table.",
)
stats_parser.add_argument(
    "--output", type=Path, default=None,
    help="If set, write JSON to this file (only valid with --json).",
)
```

**Output path mkdir-parents pattern** (exactly as used in `scripts/eval_inference.py::main` line 433 and `scripts/eval_merge.py` line 60):
```python
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(stats, indent=2))
```

**Pydantic schema pattern for stats validation** — reuse the already-installed pydantic 2.12.5 via a small model, matching the `EvalConfig` / `EvalResult` pattern in `scripts/eval_config.py` (imported by every eval script at `scripts/eval_inference.py` line 35, `scripts/eval_runner.py` lines 33-39). Add a new `DatasetStats` Pydantic model next to `compute_stats`:
```python
from pydantic import BaseModel

class DatasetSplitStats(BaseModel):
    total: int
    domains: dict[str, dict[str, float]]  # {"tool-calling": {"count": ..., "percent": ...}}

class DatasetStats(BaseModel):
    schema_version: str  # "1.0.0" per D-08
    dataset_version: str  # matches dataset-vX.Y.Z tag
    generated_at: str  # ISO-8601 UTC
    splits: dict[str, DatasetSplitStats]
```

---

### `tests/test_release_artifacts.py` (MODIFY — add `*.gguf` + CHANGELOG + Dataset Versions checks)

**Analog:** self. Every existing test in this file follows the same three-line shape: path check, read, substring assert.

**Existing pattern to replicate** (lines 18-24):
```python
def test_gitattributes_lfs():
    """gitattributes tracks *.safetensors with git-lfs filter (REL-03 / D-10)."""
    ga_path = REPO_ROOT / ".gitattributes"
    assert ga_path.exists(), ".gitattributes not found at repo root"
    text = ga_path.read_text()
    assert "safetensors" in text, ".gitattributes does not mention safetensors"
    assert "filter=lfs" in text, ".gitattributes does not set filter=lfs"
```

**Extension pattern for Phase 10** — one new test per new assertion, all following the exact same shape:
```python
def test_gitattributes_gguf_lfs():
    """gitattributes tracks *.gguf under git-lfs (REL-05 / Phase 10 D-05)."""
    text = (REPO_ROOT / ".gitattributes").read_text()
    assert "*.gguf" in text, ".gitattributes does not track *.gguf"
    # Line containing *.gguf must also set filter=lfs
    gguf_lines = [l for l in text.splitlines() if "*.gguf" in l]
    assert any("filter=lfs" in l for l in gguf_lines), \
        "*.gguf line does not set filter=lfs"


def test_changelog_exists_at_root():
    """CHANGELOG.md exists at repo root with v1.0.0 section (REL-07 / D-10)."""
    path = REPO_ROOT / "CHANGELOG.md"
    assert path.exists(), "CHANGELOG.md not found"
    text = path.read_text()
    assert "## [" in text, "CHANGELOG.md has no versioned sections"
```

**Extend `test_dataset_card` for "Dataset Versions" section** (existing test at lines 36-46):
```python
# Append to existing test_dataset_card after line 46:
    assert "## Dataset Versions" in text or "### Dataset Versions" in text, \
        "datasets/README.md missing Dataset Versions section (REL-07)"
```

---

### `tests/test_assemble_dataset.py` (MODIFY — stats JSON output test)

**Analog:** self. The fixture pattern (lines 20-159) and `TestAssemblyOutput` class (lines 208-399) provide all scaffolding.

**Extension pattern** — add a new test class after `TestAssemblyOutput`, reusing `domain_fixture_dir`:
```python
class TestStatsJsonOutput:
    """Phase 10 REL-07: stats subcommand can emit machine-readable JSON."""

    def test_compute_stats_returns_pydantic_compatible_dict(self, domain_fixture_dir):
        """compute_stats output validates against DatasetStats schema."""
        from scripts.assemble_dataset import assemble, compute_stats, DatasetStats

        dd = assemble(
            output_dir=str(domain_fixture_dir / "out"),
            seed=42,
            base_dir=str(domain_fixture_dir),
        )
        stats_dict = compute_stats(dd)
        # Wrap into DatasetStats (adds schema_version/dataset_version/generated_at)
        full = {
            "schema_version": "1.0.0",
            "dataset_version": "1.0.0",
            "generated_at": "2026-04-24T00:00:00+00:00",
            "splits": stats_dict,
        }
        model = DatasetStats.model_validate(full)
        assert model.splits["train"].total > 0
```

---

### `.gitattributes` (MODIFY — add LFS rules)

**Analog:** self. Current file (3 comment lines only) must be extended to actually track LFS patterns.

**Current state** (full file, lines 1-3):
```
# Model weights, tokenizers, and assembled dataset binaries are NOT tracked in git.
# They exceed GitHub's 2GB-per-file LFS limit. Publish to HuggingFace Hub instead.
# See .gitignore for the ignored patterns.
```

**Target state** (per RESEARCH.md §Pattern 4 and D-05 / D-10 + Pitfall 4):
```
# Git LFS tracking for large binaries (Phase 9 D-10 + Phase 10 D-05/D-10)
*.safetensors filter=lfs diff=lfs merge=lfs -text
*.gguf        filter=lfs diff=lfs merge=lfs -text
*.arrow       filter=lfs diff=lfs merge=lfs -text

# Dataset JSONL can be large; track under LFS for the assembled dataset bundle
datasets/assembled/**/*.jsonl filter=lfs diff=lfs merge=lfs -text
```

**Load-bearing ordering** (RESEARCH.md §Pitfall 4): add these rules **before** the first GGUF/Arrow file is committed, otherwise `git lfs migrate import` is required. Plan 1 (prep) owns this edit; Plan 2 (execution) produces the binaries.

---

### `requirements.txt` (MODIFY — add gguf)

**Analog:** self. Existing pattern uses exact pins for pydantic (line 1: `pydantic==2.12.5`) and lm-eval (line 14: `lm-eval[hf]==0.4.11`), and `>=X.Y` lower bounds for transformers, torch, etc.

**Extension pattern** — add one line, using the `==` exact-pin style (matches pydantic/lm-eval precedent; RESEARCH.md confirms `gguf 0.18.0` is current):
```
# GGUF metadata inspection (Phase 10 REL-05)
gguf==0.18.0
```

---

### `CHANGELOG.md` (NEW — repo root)

**Analog:** **None in the Lyra repo.** Use Keep-a-Changelog 1.1.0 exactly as shown in RESEARCH.md §Code Examples §Keep-a-Changelog lines 569-597. No extension of a local pattern is possible.

**Structure to author** (verbatim from RESEARCH.md §Code Examples, adapted to Lyra's dual dataset/model release model per Open Question #3 recommendation):
```markdown
# Changelog

All notable changes to the Lyra dataset and model releases are documented in this file.

The format is based on [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [Dataset v1.0.0] - 2026-04-XX

### Added
- Initial release of the 25K curated Lyra training dataset (tool-calling: 23K, code: ~650, knowledge: ~650).
- Stratified 90/5/5 train/validation/test splits.
- Dataset stats JSON with per-split domain breakdown.
- Pinned model-evaluation metrics: see `BENCHMARK-v1.0.0.md` for the model trained on this dataset version.

### Changed
- Tool-calling dedup threshold increased from 0.7 to 0.9 to retain more template diversity at 25K scale.

## [Model v1.0.0] - 2026-04-XX

### Added
- GGUF quantized variants: Q4_K_M and Q8_0 attached to the GitHub Release.
- Chat template embedded in GGUF metadata via `convert_hf_to_gguf.py` (D-07).
```

**Note for planner:** CHANGELOG.md **location** is repo root — this matches where every existing top-level docs file lives (`README.md`, `LICENSE`, `BENCHMARK.md`, all at `/Users/lakshman/Documents/Lyra/`).

---

### `README.md` (MODIFY — add GGUF variants + Setup update)

**Analog:** self. Existing structure has frontmatter (lines 1-12), sections demarcated with `## Heading` (lines 14, 18, 34, 42, 69, etc.).

**Section insertion pattern** — add a new `## GGUF Variants` section after the existing `## Setup` block (ends around line 79). Section shape mirrors existing `## Technology Stack` table style (lines 42-67):
```markdown
## GGUF Variants

Quantized GGUF variants for local inference via LM Studio / llama.cpp. Attached to each model GitHub Release (see [Releases](https://github.com/LakshmanTurlapati/Lyra/releases)).

| Variant | Size | Bits/weight | Use case |
|---------|------|-------------|----------|
| Q4_K_M  | ~1.0 GB | ~4.9 | Balanced quality/size -- recommended for consumer hardware |
| Q8_0    | ~1.8 GB | ~8.5 | Near-original quality |

### Conversion (reproducing from source)

Requires `llama.cpp` CLI tools on `$PATH`:

\```bash
brew install llama.cpp           # or: git clone https://github.com/ggml-org/llama.cpp
pip install gguf==0.18.0
scripts/convert_gguf.sh models/lyra-merged lyra-v1.0
\```

### Verifying the chat template

Every shipped GGUF embeds the SmolLM2 chat template at convert time (Phase 09.1 D-07). Verify on the user side:

\```bash
gguf-dump build/gguf/lyra-v1.0-q4_k_m.gguf --no-tensors | grep -i chat_template
\```
```

**Setup section extension** (current lines 69-79 mention only `pip install -r requirements.txt`) — add a one-line note that GGUF conversion requires llama.cpp install, pointing at the new GGUF Variants section below.

---

### `datasets/README.md` (MODIFY — add Dataset Versions section)

**Analog:** self. Existing stats table (lines 29-40) and section layout (Description → Creation → Statistics → Limitations → License) provide the model.

**Section insertion pattern** — add `## Dataset Versions` between the existing `## Dataset Statistics` (line 29) and `## Limitations` (line 42):
```markdown
## Dataset Versions

Each dataset version is a git tag (`dataset-vMAJOR.MINOR.PATCH`) with a GitHub Release bundle. See [Releases](https://github.com/LakshmanTurlapati/Lyra/releases) for downloads.

| Version | Date | Samples | Notes |
|---------|------|---------|-------|
| v1.0.0 | 2026-04-XX | 25K | Initial release; see CHANGELOG.md |

Version bundle contents (per release):
- `train/`, `validation/`, `test/` JSONL exported from `datasets/assembled/`
- `lyra-dataset-vX.Y.Z-stats.json` -- per-split domain breakdown
- `BENCHMARK-vX.Y.Z.md` -- pinned metrics for the model trained on this dataset version
- CHANGELOG entry -- scope of the change
```

---

### `.planning/REQUIREMENTS.md` (MODIFY — move REL-06 to Out of Scope per D-02)

**Analog:** self. Three spots to edit:

**Edit 1** — remove REL-06 line from v1 Release section (current line 64):
```markdown
- [ ] **REL-06**: Interactive Gradio demo Space on HuggingFace showcasing all three capability areas
```
Delete.

**Edit 2** — append a row to the `Out of Scope` table (between lines 85-94). Table header is at line 85; follow existing row shape:
```markdown
| Interactive Gradio demo Space on HuggingFace | Community release is GitHub-native; interactive demo not pursued in v1 |
```

**Edit 3** — update the Traceability row for REL-06 (current line 132):
```markdown
| REL-06 | Out of Scope | Deferred |
```
Change status from `Pending` to `Out of Scope`.

**Edit 4** — update the `**Coverage:**` totals block (lines 135-138) from 32 → 31 v1 requirements.

---

### `.planning/ROADMAP.md` (MODIFY — drop Gradio Space from Phase 10)

**Analog:** self. Phase 10 block is at lines 190-199.

**Current Phase 10 block** (lines 190-199):
```markdown
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
```

**Target shape** (per D-02):
```markdown
### Phase 10: Community Release Enhancements
**Goal**: Users can run Lyra locally via GGUF quantization and track dataset evolution across versions
**Depends on**: Phase 9 (gated on Phase 09.1/09.2 tool-call-format success criterion per D-06)
**Requirements**: REL-05, REL-07
**Success Criteria** (what must be TRUE):
  1. GGUF quantized variants (Q4_K_M, Q8_0) are published and loadable in LM Studio and llama.cpp
  2. Dataset releases are versioned with documented changes and metrics per version
**Plans**: TBD
**UI hint**: no
```

Changes: drop `REL-06` from Requirements list; drop success criterion #2; renumber remaining criteria; flip `UI hint` to `no`; rewrite goal sentence.

---

## Shared Patterns

### SPDX + shebang header (every script)
**Source:** `scripts/eval_inference.py` lines 1-2, `scripts/eval_runner.py` lines 1-2, `scripts/eval_merge.py` lines 1-2, `scripts/assemble_dataset.py` lines 1-2.
**Apply to:** `scripts/verify_gguf.py`, `scripts/convert_gguf.sh`, `tests/test_gguf_conversion.py`, `tests/test_dataset_versioning.py`.

Every Python script starts with:
```python
#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""<name>.py -- <one-line summary>.

<paragraph explaining what the script does and when it's run.>

Usage:
  python3 -m scripts.<name> [args]
"""
```

Every shell script starts with:
```bash
#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# <name>.sh -- <one-line summary>.
#
# Usage:
#   scripts/<name>.sh [args]
```

### Path validation for T-03-07 (all CLI entry points accepting paths)
**Source:** `scripts/eval_runner.py` lines 48-70 and `scripts/eval_inference.py` lines 326-344.
**Apply to:** `scripts/verify_gguf.py`, `scripts/convert_gguf.sh`.

Every script that takes a user-provided path validates with the same regex:
```python
_PATH_PATTERN = re.compile(r"^[a-zA-Z0-9._/~\-]+$")

def _validate_path(p: str) -> bool:
    if Path(p).exists():
        return True
    if _PATH_PATTERN.match(p):
        return True
    return False
```

### Subprocess discipline (T-03-05)
**Source:** `scripts/eval_runner.py::run_code_benchmarks` lines 167-193 — list-form args, `shell=False` (default), no string concatenation into the command.
**Apply to:** `scripts/convert_gguf.sh` (bash equivalent: never `eval`; always `$VAR` inside double quotes; no `$(...)` with unvalidated input).

### Lazy imports of heavy ML deps
**Source:** `scripts/eval_inference.py::_do_load_model_and_tokenizer` lines 111-144 (torch/transformers imported inside function). `scripts/assemble_dataset.py::validate_assembled` line 236 (imports `scripts.validate_format` lazily).
**Apply to:** `scripts/verify_gguf.py` (import `gguf` lazily inside `_load_gguf_reader` so the module can be imported by tests without `gguf` installed).

### CLI entry-point shape (`main() -> int`)
**Source:** `scripts/eval_merge.py::main` lines 65-110, `scripts/eval_compare.py::main` lines 219-301, `scripts/eval_inference.py::main` lines 347-441.
**Apply to:** `scripts/verify_gguf.py`.

Canonical shape:
```python
def main() -> int:
    parser = argparse.ArgumentParser(description="...")
    parser.add_argument(...)
    args = parser.parse_args()

    # Validate inputs
    if not args.input.exists():
        print(f"Error: file not found: {args.input}", file=sys.stderr)
        return 1

    try:
        result = do_work(args)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    sys.exit(main())
```

### Pytest file-existence / smoke-test style
**Source:** `tests/test_release_artifacts.py` lines 9-46 — every test is 3-6 lines: get path from `REPO_ROOT`, assert `exists()`, read text, assert substring.
**Apply to:** `tests/test_dataset_versioning.py`, new assertions added to `tests/test_release_artifacts.py`.

```python
REPO_ROOT = Path(__file__).parent.parent

def test_<thing>():
    """<what this proves> (<REL-ID>)."""
    path = REPO_ROOT / "<file>"
    assert path.exists(), "<file> not found"
    text = path.read_text()
    assert "<expected substring>" in text, "<human-readable failure>"
```

### MagicMock for lazy-imported packages not in CI
**Source:** `tests/test_eval_runner.py` lines 29-44 — `patch.dict("sys.modules", {"torch": MagicMock()})`.
**Apply to:** `tests/test_gguf_conversion.py` (mock `gguf` module so tests pass even if `gguf` PyPI package is not installed in the venv when the test is first added in RED state).

### Pydantic models for schema-validated JSON at trust boundaries
**Source:** `scripts/eval_config.py` (imported by every eval script). Every JSON IO goes through `EvalResult.model_validate_json(...)` (`scripts/eval_merge.py` lines 49-50, `scripts/eval_compare.py` line 264).
**Apply to:** `scripts/assemble_dataset.py` new `DatasetStats` model for the stats JSON output.

### RED-state test stubs in Wave 0 (Phase 09.2 discipline)
**Source:** Phase 09.2 Plan 01 added RED tests before implementation (see `.planning/phases/09.2-tool-call-regression-diagnosis/09.2-01-PLAN.md`). Tests in `tests/test_phase_09_2/test_template_parity.py` locked in the Phase 09.1 fix.
**Apply to:** Plan 1 (prep) lands all test files as `@pytest.mark.xfail(reason="Wave 0 RED stub")` or `raise NotImplementedError`, then Plans 2-4 remove the xfail marker and implement.

### Planning-doc edit pattern
**Source:** Plans in `.planning/phases/09-benchmarking-and-core-release/09-03-PLAN.md` reference the planning doc via `@/Users/lakshman/Documents/Lyra/.planning/ROADMAP.md` in their `<context>` block. Actual edits are tracked as file modifications in the plan's action list; they use the **Edit** tool on specific line ranges of `.planning/REQUIREMENTS.md` / `.planning/ROADMAP.md`.
**Apply to:** Plan 1 (prep) owns the REQUIREMENTS.md + ROADMAP.md edits per D-02. Each edit is a discrete task (4 edits for REQUIREMENTS.md, 1 multi-line edit for ROADMAP.md) so the planner can check them off individually.

---

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| `scripts/convert_gguf.sh` | shell script | subprocess-orchestration | **No existing `.sh` file in `scripts/`** (verified: `ls scripts/*.sh` returns no matches). First shell script in the repo. The Python analog patterns (subprocess discipline, path validation, license header) transfer but shell syntax must be authored fresh from RESEARCH.md §Pattern 1. |
| `CHANGELOG.md` | docs | static markdown | **No CHANGELOG file exists in the repo.** Use Keep-a-Changelog 1.1.0 exactly as in RESEARCH.md §Code Examples §Keep-a-Changelog (lines 569-597 of RESEARCH.md). |
| `tests/test_gguf_conversion.py` subprocess-to-bash invocation | test | subprocess | No prior test in the repo exercises a `.sh` script via `subprocess.run`. Pattern must be composed from `tests/test_release_artifacts.py` (REPO_ROOT discipline) + Python `subprocess.run([...], capture_output=True, cwd=REPO_ROOT)` convention — shown above under `tests/test_gguf_conversion.py`. |

---

## Metadata

**Analog search scope:**
- `scripts/` (all 33 Python files; no `.sh` present)
- `tests/` (all test files, including `tests/test_phase_09_2/` for RED-state convention)
- `.planning/phases/09-*/` + `.planning/phases/09.1-*/` + `.planning/phases/09.2-*/` for planning-doc edit precedent
- Repo-root docs (`README.md`, `datasets/README.md`, `.gitattributes`, `requirements.txt`, `BENCHMARK.md`)

**Files scanned:** ~45 (scripts/, tests/, planning docs, root docs)
**Pattern extraction date:** 2026-04-24
