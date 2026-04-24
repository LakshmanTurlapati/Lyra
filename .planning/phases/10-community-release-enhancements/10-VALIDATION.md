---
phase: 10
slug: community-release-enhancements
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-24
---

# Phase 10 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution. Populated by the planner from RESEARCH.md §Validation Architecture.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x (existing — matches Phase 1–9 convention) |
| **Config file** | `pyproject.toml` / `pytest.ini` (existing project config) |
| **Quick run command** | `pytest tests/test_release_artifacts.py -x -q` |
| **Full suite command** | `pytest -x -q` |
| **Estimated runtime** | ~10 seconds (quick) / ~60 seconds (full — excludes slow-marker integration tests) |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/test_release_artifacts.py -x -q`
- **After every plan wave:** Run `pytest -x -q`
- **Before `/gsd-verify-work`:** Full suite must be green; manual GGUF-in-LM-Studio UAT completed
- **Max feedback latency:** 10 seconds (quick) / 60 seconds (full)

---

## Per-Task Verification Map

*Populated by planner during plan creation. Each task in each PLAN.md maps to a row here with threat ref (T-10-XX) and automated command.*

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| TBD | TBD | TBD | REL-05 / REL-07 | TBD | TBD | unit / integration / manual | TBD | W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_release_artifacts.py` — already exists; extend with `test_gitattributes_gguf_lfs`, `test_convert_gguf_script_exists`, `test_verify_gguf_script_exists`
- [ ] `tests/test_dataset_versioning.py` — NEW: stats JSON schema test, version-tag format validator test, CHANGELOG presence test
- [ ] `tests/test_gguf_conversion.py` — NEW: unit tests for argument parsing in `scripts/convert_gguf.sh` wrapper and `scripts/verify_gguf.py` GGUFReader assertions (mocked)
- [ ] `tests/conftest.py` — shared fixtures (tmp_path model-proxy for GGUF conversion tests)

*Wave 0 installs must complete before any plan-level implementation task executes.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Q4_K_M GGUF loads in LM Studio, responds to a tool-call prompt with embedded template honored | REL-05 | Requires a GUI app install; no headless CI equivalent | 1) Install LM Studio; 2) Load `lyra-v1.0-q4_k_m.gguf`; 3) Send the tool-call test prompt from `tests/fixtures/tool_call_smoke.txt`; 4) Confirm response uses `<tool_call>` XML (matches training format) |
| Q8_0 GGUF loads via `llama-cli` command-line smoke test | REL-05 | External binary + interactive prompt | Run `llama-cli -m lyra-v1.0-q8_0.gguf -p "$(cat tests/fixtures/tool_call_smoke.txt)" -n 128` and inspect output |
| Post-quantization perplexity sanity check (f16 vs Q4_K_M delta < 10%) | REL-05 | Long-running (~minutes on 100 samples); gated on 09.1/09.2 weights being available | Run `llama-perplexity -m lyra-v1.0-f16.gguf -f tests/fixtures/perplexity_sample.txt`; repeat with Q4_K_M; compare |
| Tagged dataset release downloads and reconstructs bundle | REL-07 | Requires a real `git tag` + `gh release create` round-trip | 1) `gh release download dataset-v1.0.0`; 2) Unpack; 3) Verify train/validation/test JSONL match SHA256 of local `datasets/assembled/*.jsonl`; 4) Verify BENCHMARK.md snapshot present |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references (currently: LFS rule, GGUF scripts, dataset-stats JSON schema, CHANGELOG presence)
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s
- [ ] `nyquist_compliant: true` set in frontmatter after planner populates per-task map

**Approval:** pending
