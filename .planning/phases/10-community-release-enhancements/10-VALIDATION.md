---
phase: 10
slug: community-release-enhancements
status: populated
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-24
last_updated_by_planner: 2026-04-24
---

# Phase 10 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution. Populated by the planner
> from RESEARCH.md §Validation Architecture and the 4 finalized PLAN.md files. Plan 04 Task 2
> [CHECKPOINT] is responsible for flipping `nyquist_compliant` and `wave_0_complete` to `true`
> after manual UAT results are captured.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x (existing — matches Phase 1–9 convention) |
| **Config file** | `pytest.ini` (existing project config) |
| **Quick run command** | `pytest tests/test_release_artifacts.py tests/test_dataset_versioning.py tests/test_gguf_conversion.py -x -q` |
| **Full suite command** | `pytest -x -q` |
| **Estimated runtime** | ~10 seconds (quick) / ~60 seconds (full — excludes slow-marker integration tests) |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/test_release_artifacts.py tests/test_dataset_versioning.py tests/test_gguf_conversion.py -x -q`
- **After every plan wave:** Run `pytest -x -q`
- **Before `/gsd-verify-work`:** Full suite must be green; manual GGUF-in-LM-Studio UAT completed (10-UAT.md captured)
- **Max feedback latency:** 10 seconds (quick) / 60 seconds (full)

---

## Per-Task Verification Map

*One row per task across all four plans. Status column updated by executor at task-complete time.*

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| T-10-01-01 | 01 | 1 | REL-05 / REL-06 / REL-07 | T-10-03 | CHANGELOG scaffold instructs shasum -a 256 verification; LFS rules land BEFORE any binary commit (Pitfall 4) | unit | `pytest tests/test_release_artifacts.py tests/test_dataset_versioning.py tests/test_gguf_conversion.py -q` | W0 (creates) | ⬜ pending |
| T-10-01-02 | 01 | 1 | REL-05 | T-10-01, T-10-02 | Positional-arg regex validation in convert_gguf.sh; lazy-import gguf.GGUFReader in verify_gguf.py; substring check for SmolLM2 marker (no eval) | unit | `pytest tests/test_gguf_conversion.py -q && bash -n scripts/convert_gguf.sh` | ✓ (produced by task) | ⬜ pending |
| T-10-01-03 | 01 | 1 | REL-07 | T-10-04 | `args.output` rejected if contains `..` components; `args.output.parent.mkdir(parents=True, exist_ok=True)` used for safe dir creation | unit | `pytest tests/test_assemble_dataset.py tests/test_release_artifacts.py -q && python3 -c "from scripts.assemble_dataset import DatasetStats, build_dataset_stats_payload; print('ok')"` | ✓ (produced by task) | ⬜ pending |
| T-10-02-01 | 02 | 2 | REL-05 | T-10-01, T-10-02 | Invoke convert_gguf.sh with literal safe args; trust verify_gguf.py postcondition (halt if exit 1) | integration | `pytest tests/test_gguf_conversion.py -q && python3 -m scripts.verify_gguf build/gguf/lyra-v1.0-q4_k_m.gguf && python3 -m scripts.verify_gguf build/gguf/lyra-v1.0-q8_0.gguf && git lfs ls-files \| grep -qE "lyra-v1\.0-(q4_k_m\|q8_0)\.gguf"` | ✓ (produced by task) | ⬜ pending |
| T-10-02-02 | 02 | 2 | REL-05 | T-10-03, T-10-05 | SHA256 in CHANGELOG + release-notes; pre-release `gh release view model-v1.0.0` existence check | manual + automated | `gh release view model-v1.0.0 --json assets --jq '[.assets[].name] \| sort' \| grep -qE 'lyra-v1\.0-q4_k_m\.gguf' && grep -qE '^## \[Model v1\.0\.0\] - [0-9]' CHANGELOG.md` | ✓ (produced by task) | ⬜ pending |
| T-10-02-03 | 02 | 2 | REL-05 | T-10-03 | Human verifies release page renders + SHA256 of one downloaded asset matches CHANGELOG | manual | (human checkpoint — captured in 10-UAT.md) | ✓ (checkpoint) | ⬜ pending |
| T-10-03-01 | 03 | 2 | REL-07 | T-10-04 | Literal relative paths under `release/`; Plan-01 `..`-component guard applies | integration | `python3 -c "import json; from scripts.assemble_dataset import DatasetStats; DatasetStats.model_validate(json.loads(open('release/dataset-v1.0.0/lyra-dataset-v1.0.0-stats.json').read()))" && wc -l < release/dataset-v1.0.0/SHA256SUMS.txt \| grep -q '^5$'` | ✓ (produced by task) | ⬜ pending |
| T-10-03-02 | 03 | 2 | REL-07 | T-10-03, T-10-05 | SHA256SUMS.txt included as release asset; checksums inlined in CHANGELOG + release-notes; pre-release `gh release view dataset-v1.0.0` existence check | integration | `gh release view dataset-v1.0.0 --json assets --jq '[.assets[].name] \| sort' \| grep -qE 'train\.jsonl' && pytest tests/test_dataset_versioning.py -q && grep -qE '^## \[Dataset v1\.0\.0\] - [0-9]' CHANGELOG.md` | ✓ (produced by task) | ⬜ pending |
| T-10-03-03 | 03 | 2 | REL-07 | T-10-03 | Human confirms gh release download + shasum -a 256 -c passes + head train.jsonl is TRL-native | manual | (human checkpoint — captured in 10-UAT.md) | ✓ (checkpoint) | ⬜ pending |
| T-10-04-01 | 04 | 3 | REL-05 / REL-07 | T-10-03 | SHA256 values copied verbatim from Plan 02/03 SHA256SUMS.txt; reconcile BENCHMARK snapshot against final BENCHMARK.md | docs | `! grep -A5 "^## GGUF Variants" README.md \| grep -q "TBD" && ! grep -A5 "^## Dataset Versions" datasets/README.md \| grep -q "TBD"` | ✓ (updated by task) | ⬜ pending |
| T-10-04-02 | 04 | 3 | REL-05 / REL-07 | T-10-03, T-10-06 | UAT verifies SHA256 integrity of downloaded asset; LM Studio Prompt Template cross-checked against chat_template.jinja (Pitfall 6); `<tool_call>` wrapper output confirms embedded template honored | manual | (human checkpoint — captured in 10-UAT.md; this row + all UAT rows flip ✅ once Plan 04 Task 2 completes) | ✓ (produces 10-UAT.md) | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [x] `tests/test_release_artifacts.py` — Plan 01 Task 1 extends with `test_gitattributes_gguf_lfs`, `test_gitattributes_arrow_lfs`, `test_changelog_exists_at_root`, and extends `test_dataset_card` with "Dataset Versions" assertion.
- [x] `tests/test_dataset_versioning.py` — Plan 01 Task 1 creates with 5 tests (4 GREEN-from-day-1 + 1 xfail for Plan 03 release).
- [x] `tests/test_gguf_conversion.py` — Plan 01 Task 1 creates with 5 tests (3 xfail → GREEN in Plan 01 Task 2 + 1 xfail → GREEN in Plan 01 Task 2 + 1 xfail → GREEN in Plan 02).
- [x] `tests/test_assemble_dataset.py` — Plan 01 Task 1 extends with TestStatsJsonOutput class (2 xfail → GREEN in Plan 01 Task 3).
- [N/A] `tests/conftest.py` — No new fixtures needed; existing `domain_fixture_dir` fixture in test_assemble_dataset.py is reused for Phase 10 stats tests.

*Wave 0 convention: Plan 01 lands every test file (mix of GREEN + xfail); later plans turn xfails GREEN as implementations land. When Plan 04 Task 2 completes, set `wave_0_complete: true` in this file's frontmatter.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Q4_K_M GGUF loads in LM Studio, responds to a tool-call prompt with embedded template honored | REL-05 | Requires a GUI app install; no headless CI equivalent | Plan 04 Task 2 UAT Verification 1 — load `lyra-v1.0-q4_k_m.gguf`, send tool-call probe, confirm `<tool_call>` XML in output, compare visible Prompt Template against `models/lyra-merged/chat_template.jinja` |
| Q8_0 GGUF loads via `llama-cli` command-line smoke test | REL-05 | External binary + interactive prompt | Plan 04 Task 2 UAT Verification 2 — `llama-cli -m lyra-v1.0-q8_0.gguf -p "<tool-call prompt>" -n 128` |
| Post-quantization perplexity sanity check (f16 vs Q4_K_M delta < 10%) | REL-05 | Long-running; optional per CONTEXT.md Claude's discretion | Plan 04 Task 2 UAT Verification 3 — `llama-perplexity -m <f16.gguf> -f val_samples.txt`; repeat with Q4_K_M; compare. Skip allowed. |
| Tagged dataset release downloads and reconstructs bundle | REL-07 | Requires a real `git tag` + `gh release` round-trip | Plan 04 Task 2 UAT Verification 4 — `gh release download dataset-v1.0.0`, `shasum -a 256 -c SHA256SUMS.txt`, `head -1 train.jsonl \| python3 -m json.tool` |
| GitHub Release page renders + at least one asset SHA256 matches CHANGELOG (Model v1.0.0) | REL-05 | Browser / gh UI | Plan 02 Task 3 — visit release URL, download Q4_K_M, verify SHA256 |
| GitHub Release page renders + dataset bundle round-trip (Dataset v1.0.0) | REL-07 | Browser / gh UI | Plan 03 Task 3 — visit release URL, confirm all 6 assets listed + a spot-download of stats.json + shasum match |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies (Per-Task Verification Map populated)
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify (manual checkpoints flanked by automated task commits)
- [ ] Wave 0 covers all MISSING references (LFS rule, GGUF scripts, dataset-stats JSON schema, CHANGELOG presence, version-tag regex)
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s (quick suite ~10s; full suite ~60s per pytest history in STATE.md)
- [ ] `nyquist_compliant: true` set in frontmatter after Plan 04 Task 2 captures UAT results
- [ ] `wave_0_complete: true` set in frontmatter after Plan 01 executes Task 1

**Approval:** pending until Plan 04 Task 2 completes the manual UAT and ticks all boxes above.
