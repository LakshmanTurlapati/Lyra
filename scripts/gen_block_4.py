#!/usr/bin/env python3
"""Block 4: Coverage & CI (items 61-85)."""
import json
from pathlib import Path

samples = []

def add(system, user, assistant, followup=None):
    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]
    if followup:
        msgs.append({"role": "user", "content": followup[0]})
        msgs.append({"role": "assistant", "content": followup[1]})
    samples.append({"messages": msgs, "tools": [], "domain": "code"})


add(
    "You are a Python coverage expert.",
    "How do I measure test coverage with coverage.py and pytest?",
    "Install `pytest-cov` and run pytest with `--cov`.\n\n```bash\npip install pytest pytest-cov\npytest --cov=myapp --cov-report=term-missing --cov-report=html\n```\n\nReport types:\n- `term` — short summary\n- `term-missing` — lists uncovered line numbers inline\n- `html` — interactive browser under `htmlcov/`\n- `xml` — for Codecov/Coveralls upload\n- `json` — programmatic consumption\n\nConfigure in `pyproject.toml`:\n\n```toml\n[tool.coverage.run]\nsource = [\"myapp\"]\nbranch = true\nparallel = true\nomit = [\n    \"*/tests/*\",\n    \"*/migrations/*\",\n    \"*/__init__.py\",\n]\n\n[tool.coverage.report]\nfail_under = 85\nexclude_lines = [\n    \"pragma: no cover\",\n    \"raise NotImplementedError\",\n    \"if __name__ == .__main__.:\",\n    \"if TYPE_CHECKING:\",\n]\nshow_missing = true\nskip_covered = false\n```\n\n`branch = true` enables branch coverage so an `if x:` counts both taken and untaken branches. `fail_under = 85` makes the pytest run exit nonzero if coverage drops below 85%, useful as a CI gate.",
)

add(
    "You are a coverage expert.",
    "What is branch coverage vs line coverage?",
    "Line coverage counts executed lines; branch coverage counts executed conditional branches.\n\n```python\ndef classify(score):\n    if score >= 90:\n        return \"A\"\n    if score >= 80:\n        return \"B\"\n    return \"C\"\n\n# test with score=95 only\n# line coverage: 100% (every line ran... sort of)\n# branch coverage: about 50% — we never took the False branch of either `if`\n```\n\nBranch coverage reveals untested paths. Enable with `branch = true` in `pyproject.toml` or `coverage run --branch`.\n\n```bash\ncoverage run --branch -m pytest\ncoverage report --show-missing\n```\n\nOutput annotates partially-covered lines with `->` showing which branch was missed:\n\n```\nName       Stmts   Miss Branch BrPart  Cover   Missing\n-----------------------------------------------------------\ngrader.py      5      0      4      2   78%   3->4, 5->6\n```\n\nAim for branch coverage in any project doing nontrivial conditional logic. Line coverage alone overstates safety because it treats `if x:` as one unit regardless of which side executed.",
)

add(
    "You are a coverage expert.",
    "How do I exclude specific lines or blocks from coverage?",
    "Use `# pragma: no cover` comments and `exclude_lines` patterns.\n\n```python\nimport sys\n\ndef read_input():\n    if sys.stdin.isatty():  # pragma: no cover\n        # interactive path hard to test\n        return input(\"> \")\n    return sys.stdin.read()\n\ndef debug_dump(obj):  # pragma: no cover\n    import pprint\n    pprint.pprint(obj)\n\ndef safe_div(a, b):\n    if b == 0:\n        raise ValueError(\"divide by zero\")  # pragma: no cover\n    return a / b\n```\n\nGlobal patterns in `pyproject.toml`:\n\n```toml\n[tool.coverage.report]\nexclude_lines = [\n    \"pragma: no cover\",\n    \"raise NotImplementedError\",\n    \"raise AssertionError\",\n    \"if __name__ == .__main__.:\",\n    \"if TYPE_CHECKING:\",\n    \"@(abc\\\\.)?abstractmethod\",\n    \"class .*\\\\bProtocol\\\\):\",\n]\n```\n\nUse sparingly. The pragma is a lie-detector: every `no cover` is an admission that code is untested. Save it for truly untestable branches (platform-specific paths, defensive asserts, `if TYPE_CHECKING:` blocks) and justify each one.",
)

add(
    "You are a coverage expert.",
    "Combine coverage data from multiple test runs (parallel, subprocess).",
    "Use `parallel = true` and `coverage combine`.\n\n```toml\n[tool.coverage.run]\nparallel = true\nsource = [\"myapp\"]\n```\n\n```bash\n# each test process writes .coverage.<hostname>.<pid>\ncoverage run -p -m pytest tests/unit\ncoverage run -p -m pytest tests/integration\ncoverage run -p -m pytest tests/e2e\n\n# merge them\ncoverage combine\n\n# report on combined data\ncoverage report --show-missing\ncoverage html\n```\n\nFor code launched as a subprocess, enable coverage inside the child via `.coveragerc` or `COVERAGE_PROCESS_START`:\n\n```bash\nexport COVERAGE_PROCESS_START=$PWD/.coveragerc\npython -c \"import coverage; coverage.process_startup()\"\n# any subprocess that imports this gets coverage\n```\n\nAdd `sitecustomize.py` with `import coverage; coverage.process_startup()` on your PYTHONPATH for transparent subprocess tracking. Common use: testing CLI tools that spawn workers.",
)

add(
    "You are a coverage expert.",
    "How do I gate a pull request on not dropping coverage?",
    "Use coverage's `--fail-under` or diff-cover for per-PR deltas.\n\n```bash\n# hard floor\ncoverage report --fail-under=85\n\n# per-PR diff coverage\npip install diff-cover\ncoverage xml\ndiff-cover coverage.xml --compare-branch=origin/main --fail-under=90\n```\n\nGitHub Actions example:\n\n```yaml\nname: coverage\non: pull_request\njobs:\n  cov:\n    runs-on: ubuntu-latest\n    steps:\n      - uses: actions/checkout@v5\n        with:\n          fetch-depth: 0  # needed for diff-cover\n      - uses: actions/setup-python@v6\n        with:\n          python-version: \"3.12\"\n      - run: pip install -e \".[test]\" coverage diff-cover\n      - run: coverage run -m pytest\n      - run: coverage xml\n      - run: coverage report --fail-under=85\n      - name: Diff coverage\n        run: diff-cover coverage.xml --compare-branch=origin/${{ github.base_ref }} --fail-under=90\n```\n\nThe combination is effective: the absolute floor stops backsliding, while diff-cover enforces that *new code* is well tested even if the overall number moves slowly.",
)

add(
    "You are a CI expert.",
    "Write a GitHub Actions workflow that tests on multiple Python versions.",
    "Matrix build across versions and OSes.\n\n```yaml\nname: tests\n\non:\n  push:\n    branches: [main]\n  pull_request:\n\njobs:\n  test:\n    runs-on: ${{ matrix.os }}\n    strategy:\n      fail-fast: false\n      matrix:\n        os: [ubuntu-latest, macos-latest, windows-latest]\n        python: [\"3.10\", \"3.11\", \"3.12\", \"3.13\"]\n        exclude:\n          - os: windows-latest\n            python: \"3.10\"\n    steps:\n      - uses: actions/checkout@v5\n      - uses: actions/setup-python@v6\n        with:\n          python-version: ${{ matrix.python }}\n          cache: pip\n          cache-dependency-path: pyproject.toml\n      - run: pip install -e \".[test]\"\n      - run: pytest -n auto --cov=myapp --cov-report=xml\n      - uses: codecov/codecov-action@v5\n        with:\n          token: ${{ secrets.CODECOV_TOKEN }}\n          files: ./coverage.xml\n          fail_ci_if_error: false\n```\n\n`fail-fast: false` lets all matrix jobs finish so you see every failure, not just the first. `cache: pip` with `cache-dependency-path` keys off `pyproject.toml` so changes to deps invalidate the cache. `-n auto` uses `pytest-xdist` for parallel execution on multi-core runners.",
)

add(
    "You are a CI expert.",
    "Set up GitHub Actions to cache pip and pre-commit.",
    "Use `actions/cache` or setup-python's built-in caching.\n\n```yaml\nname: ci\non: [push, pull_request]\n\njobs:\n  lint:\n    runs-on: ubuntu-latest\n    steps:\n      - uses: actions/checkout@v5\n      - uses: actions/setup-python@v6\n        with:\n          python-version: \"3.12\"\n          cache: pip\n          cache-dependency-path: |\n            pyproject.toml\n            requirements-dev.txt\n      - name: Cache pre-commit\n        uses: actions/cache@v4\n        with:\n          path: ~/.cache/pre-commit\n          key: precommit-${{ runner.os }}-${{ hashFiles('.pre-commit-config.yaml') }}\n      - run: pip install pre-commit\n      - run: pre-commit run --all-files --show-diff-on-failure\n\n  test:\n    runs-on: ubuntu-latest\n    steps:\n      - uses: actions/checkout@v5\n      - uses: actions/setup-python@v6\n        with:\n          python-version: \"3.12\"\n          cache: pip\n      - run: pip install -e \".[test]\"\n      - run: pytest\n```\n\nSetup-python's `cache: pip` hashes `pyproject.toml`/requirements files automatically; the separate `actions/cache` step is used for non-pip caches like pre-commit. Cache hits typically cut a cold job from 2 minutes to 20 seconds.",
)

add(
    "You are a CI expert.",
    "Write a GitHub Actions workflow that publishes to PyPI on tag.",
    "Trust-based publishing with no long-lived tokens.\n\n```yaml\nname: release\non:\n  push:\n    tags: [\"v*\"]\n\njobs:\n  build:\n    runs-on: ubuntu-latest\n    steps:\n      - uses: actions/checkout@v5\n      - uses: actions/setup-python@v6\n        with:\n          python-version: \"3.12\"\n      - run: pip install build\n      - run: python -m build\n      - uses: actions/upload-artifact@v4\n        with:\n          name: dist\n          path: dist/\n\n  publish:\n    needs: build\n    runs-on: ubuntu-latest\n    environment: pypi\n    permissions:\n      id-token: write  # required for trusted publishing\n    steps:\n      - uses: actions/download-artifact@v4\n        with:\n          name: dist\n          path: dist/\n      - uses: pypa/gh-action-pypi-publish@release/v1\n```\n\nConfigure trusted publishing on PyPI: project settings -> Publishing -> add a GitHub publisher with owner, repo, workflow filename `release.yml`, environment `pypi`. No API tokens needed; GitHub's OIDC token is exchanged for a short-lived PyPI token. Tag `v1.2.3` triggers the release.",
)

add(
    "You are a CI expert.",
    "Show me a pre-commit config for a typical Python project.",
    "`.pre-commit-config.yaml` with ruff, mypy, and sanity checks.\n\n```yaml\nrepos:\n  - repo: https://github.com/pre-commit/pre-commit-hooks\n    rev: v5.0.0\n    hooks:\n      - id: trailing-whitespace\n      - id: end-of-file-fixer\n      - id: check-yaml\n      - id: check-toml\n      - id: check-added-large-files\n        args: [\"--maxkb=500\"]\n      - id: check-merge-conflict\n      - id: debug-statements\n\n  - repo: https://github.com/astral-sh/ruff-pre-commit\n    rev: v0.9.0\n    hooks:\n      - id: ruff\n        args: [--fix]\n      - id: ruff-format\n\n  - repo: https://github.com/pre-commit/mirrors-mypy\n    rev: v1.15.0\n    hooks:\n      - id: mypy\n        additional_dependencies: [types-requests, pydantic]\n\n  - repo: https://github.com/astral-sh/uv-pre-commit\n    rev: 0.5.20\n    hooks:\n      - id: uv-lock  # keep uv.lock in sync\n```\n\nInstall and enable:\n\n```bash\npip install pre-commit\npre-commit install             # run on every commit\npre-commit install --hook-type commit-msg  # also on commit messages\npre-commit run --all-files     # initial full run\npre-commit autoupdate          # bump revs\n```\n\nCommit the `.pre-commit-config.yaml` and pin revs; `autoupdate` gives reproducible upgrades.",
)

add(
    "You are a CI expert.",
    "Write a minimal tox config that runs pytest across Python versions.",
    "`tox.ini` or `[tool.tox]` in `pyproject.toml`.\n\n```ini\n# tox.ini\n[tox]\nenvlist = py310, py311, py312, py313, lint, type\nisolated_build = true\n\n[testenv]\ndeps =\n    pytest\n    pytest-cov\n    hypothesis\ncommands = pytest {posargs:--cov=myapp --cov-report=term-missing}\n\n[testenv:lint]\nskip_install = true\ndeps = ruff\ncommands =\n    ruff check .\n    ruff format --check .\n\n[testenv:type]\ndeps = mypy\ncommands = mypy myapp\n\n[testenv:docs]\ndeps = sphinx\ncommands = sphinx-build -W -b html docs docs/_build\n```\n\nRun:\n\n```bash\ntox                      # all envs\ntox -e py312             # single env\ntox -e py312 -- -x -k login   # pass args to pytest via posargs\ntox -p auto              # parallel\n```\n\nNew alternative: `nox` uses Python config instead of INI, which is easier to express conditional logic. Modern `tox 4.x` is still the workhorse for matrix Python-version testing in libraries.",
)

add(
    "You are a CI expert.",
    "Demonstrate running pytest in parallel with pytest-xdist.",
    "Install `pytest-xdist` and use `-n`.\n\n```bash\npip install pytest-xdist\npytest -n auto          # as many workers as CPUs\npytest -n 4             # exactly 4 workers\npytest -n auto --dist=loadfile   # tests in same file go to same worker\n```\n\nDistribution modes:\n- `load` (default): round-robin; good when tests are uniform\n- `loadfile`: group by file; good when tests share expensive fixtures per file\n- `loadgroup`: explicit grouping via `@pytest.mark.xdist_group`\n- `worksteal`: dynamic rebalancing, best for uneven test durations\n\nMark tests that cannot be parallelized:\n\n```python\nimport pytest\n\n@pytest.mark.xdist_group(\"serial\")\ndef test_uses_global_port_5432():\n    ...\n```\n\nRun with `--dist=loadgroup` and all `serial` tests go to one worker.\n\nGotchas: session fixtures that touch a shared resource (same Docker container port, same tmp file path) will collide. Scope resources per worker via `tmp_path_factory` or include `os.environ.get('PYTEST_XDIST_WORKER', 'master')` in paths and ports.",
)

add(
    "You are a CI expert.",
    "Configure a GitHub Actions workflow that runs integration tests only on main.",
    "Use path filters and `if:` conditions.\n\n```yaml\nname: tests\non:\n  push:\n    branches: [main]\n  pull_request:\n\njobs:\n  unit:\n    runs-on: ubuntu-latest\n    steps:\n      - uses: actions/checkout@v5\n      - uses: actions/setup-python@v6\n        with:\n          python-version: \"3.12\"\n          cache: pip\n      - run: pip install -e \".[test]\"\n      - run: pytest tests/unit -n auto\n\n  integration:\n    if: github.ref == 'refs/heads/main' || contains(github.event.pull_request.labels.*.name, 'run-integration')\n    runs-on: ubuntu-latest\n    services:\n      postgres:\n        image: postgres:16\n        env:\n          POSTGRES_PASSWORD: test\n        ports: [\"5432:5432\"]\n        options: --health-cmd pg_isready --health-interval 5s --health-retries 5\n      redis:\n        image: redis:7\n        ports: [\"6379:6379\"]\n    steps:\n      - uses: actions/checkout@v5\n      - uses: actions/setup-python@v6\n        with: { python-version: \"3.12\", cache: pip }\n      - run: pip install -e \".[test,integration]\"\n      - run: pytest tests/integration -v\n        env:\n          DATABASE_URL: postgres://postgres:test@localhost:5432/postgres\n          REDIS_URL: redis://localhost:6379\n```\n\nUnit tests run on every PR for fast feedback; integration runs only on main or when a PR is labeled `run-integration`, saving CI minutes while keeping the option open.",
)

add(
    "You are a CI expert.",
    "How do I upload coverage to Codecov from GitHub Actions?",
    "Produce `coverage.xml` and use the official action.\n\n```yaml\n- name: Run tests with coverage\n  run: pytest --cov=myapp --cov-report=xml --cov-report=term-missing\n\n- name: Upload coverage to Codecov\n  uses: codecov/codecov-action@v5\n  with:\n    token: ${{ secrets.CODECOV_TOKEN }}\n    files: ./coverage.xml\n    flags: unittests\n    name: python-${{ matrix.python }}\n    fail_ci_if_error: false\n```\n\nSet a coverage target in `codecov.yml` at repo root:\n\n```yaml\ncoverage:\n  status:\n    project:\n      default:\n        target: 85%\n        threshold: 0.5%\n    patch:\n      default:\n        target: 90%\n        threshold: 0%\n\ncomment:\n  layout: \"reach, diff, flags, files\"\n  behavior: default\n\nignore:\n  - \"tests/**\"\n  - \"**/migrations/**\"\n```\n\n`project` is whole-repo; `patch` is new lines in the PR. `threshold` allows minor fluctuations before flagging a drop. The Codecov PR comment shows exactly which lines you added that are not covered, which is excellent review feedback.",
)

add(
    "You are a CI expert.",
    "Run pytest only on files changed in a PR.",
    "Use `pytest-testmon` or compute the list via git + `pytest --collect-only`.\n\n```bash\npip install pytest-testmon\npytest --testmon   # first run: full; subsequent: only tests affected by changes\n```\n\n`testmon` tracks which lines of code each test covers. When a file changes, only tests whose coverage trace hits that file are rerun. Combined with CI caching of the `.testmondata` DB, PR builds become much faster.\n\nGit-based alternative:\n\n```bash\n# collect changed files vs main\ncomm=$(git diff --name-only origin/main...HEAD -- '*.py')\n\n# crude: run tests whose path matches any changed module name\nfor f in $comm; do\n  mod=${f%.py}\n  pytest \"tests/\" -k \"$(basename $mod)\"\ndone\n```\n\nMore reliable with `pytest-picked`:\n\n```bash\npip install pytest-picked\npytest --picked   # only tests for files with unstaged or untracked changes\npytest --picked --mode=branch  # changed vs upstream branch\n```\n\nFor PRs, run full tests as a safety net on merge to main but speed up the development loop with one of the above during iteration.",
)

add(
    "You are a CI expert.",
    "Set up a GitHub Actions job that runs a security scan with bandit and pip-audit.",
    "Two short jobs that fail the build on findings.\n\n```yaml\nname: security\non: [push, pull_request]\n\njobs:\n  bandit:\n    runs-on: ubuntu-latest\n    steps:\n      - uses: actions/checkout@v5\n      - uses: actions/setup-python@v6\n        with: { python-version: \"3.12\", cache: pip }\n      - run: pip install bandit[toml]\n      - run: bandit -c pyproject.toml -r myapp/ -ll\n\n  deps:\n    runs-on: ubuntu-latest\n    steps:\n      - uses: actions/checkout@v5\n      - uses: actions/setup-python@v6\n        with: { python-version: \"3.12\", cache: pip }\n      - run: pip install pip-audit\n      - run: pip-audit -r requirements.txt --strict\n```\n\nBandit config in `pyproject.toml`:\n\n```toml\n[tool.bandit]\nexclude_dirs = [\"tests\", \"build\"]\nskips = [\"B101\"]  # allow assert in non-test code if we really need to\n```\n\n`-ll` sets the severity threshold to LOW and above. `pip-audit --strict` exits nonzero if any vulnerability is found. Add `uv pip audit` if you use uv, or `safety check` as an alternative. Schedule a weekly run (`on: schedule: - cron: '0 6 * * 1'`) so vulnerabilities published after your last commit are still caught.",
)

add(
    "You are a coverage expert.",
    "How do I diff coverage between a PR branch and main?",
    "`diff-cover` compares a coverage report against a git diff.\n\n```bash\npip install diff-cover\n\n# 1. generate coverage.xml on the PR branch\ncoverage run -m pytest\ncoverage xml\n\n# 2. diff-cover reads coverage.xml and git diff\ndiff-cover coverage.xml \\\n    --compare-branch=origin/main \\\n    --fail-under=90 \\\n    --html-report=diff-cover.html\n```\n\nSample output:\n\n```\n-------------\nDiff Coverage\nDiff: origin/main...HEAD\n-------------\nmyapp/auth.py (92.3%): Missing lines 45, 67\nmyapp/db.py (100%)\n-------------\nTotal:   42 lines\nMissing: 3 lines\nCoverage: 92.85%\n-------------\n```\n\nAdd to CI:\n\n```yaml\n- run: coverage run -m pytest\n- run: coverage xml\n- name: Diff coverage\n  run: diff-cover coverage.xml --compare-branch=origin/${{ github.base_ref }} --fail-under=90\n```\n\nThe `fetch-depth: 0` checkout option is required so the base branch is available locally. Diff coverage is far more useful than absolute coverage for PRs: legacy code is not your PR's problem, but new code without tests is.",
)

add(
    "You are a CI expert.",
    "How do I run pytest with GitHub Actions annotations for failures?",
    "Use `pytest --tb=short` and the built-in GitHub annotation plugin.\n\n```bash\npip install pytest-github-actions-annotate-failures\npytest   # emits ::error:: lines GitHub parses\n```\n\nResult: each failed test shows up as an inline annotation on the offending source line in the PR's Files Changed view. No config needed beyond installing the plugin.\n\nIf you prefer manual control:\n\n```yaml\n- name: Run tests\n  run: |\n    pytest --tb=short 2>&1 | tee pytest.log\n    exit ${PIPESTATUS[0]}\n- name: Annotate failures\n  if: failure()\n  run: |\n    awk '/^FAILED/ {print \"::error::\"$0}' pytest.log\n```\n\nYou can also upload a JUnit XML to use with test-reporting actions:\n\n```bash\npytest --junitxml=report.xml\n```\n\n```yaml\n- uses: EnricoMi/publish-unit-test-result-action@v2\n  if: always()\n  with:\n    files: report.xml\n```\n\nThe annotate-failures plugin is the lowest-friction option and works for nearly every project without config.",
)

add(
    "You are a coverage expert.",
    "Measure coverage of an application launched as a subprocess during tests.",
    "Set `COVERAGE_PROCESS_START` and hook it in `sitecustomize.py`.\n\n```ini\n# .coveragerc (or equivalent pyproject.toml block)\n[run]\nsource = myapp\nparallel = True\nconcurrency = multiprocessing\nsigterm = True\n```\n\n```python\n# tests/conftest.py or a sitecustomize.py on PYTHONPATH\nimport coverage\ncoverage.process_startup()\n```\n\n```bash\nexport COVERAGE_PROCESS_START=\"$PWD/.coveragerc\"\npytest tests/              # parent process uses coverage\n# subprocesses launched by tests will also record coverage thanks to sitecustomize\n\ncoverage combine           # merge parent + child .coverage.* files\ncoverage report\n```\n\nFor pytest-xdist: set `parallel = true` so each worker writes `.coverage.<host>.<pid>`, then combine. For Docker-containerized tests, mount the coverage directory as a volume so the child's data file survives the container exit. This technique is how you measure coverage of CLIs, worker processes, and test-spawned subprocesses without gaming the numbers.",
)

add(
    "You are a CI expert.",
    "Write a GitHub Actions workflow that runs on a schedule and on manual dispatch.",
    "Add `schedule` and `workflow_dispatch` triggers.\n\n```yaml\nname: nightly\non:\n  schedule:\n    - cron: \"17 3 * * *\"   # 03:17 UTC daily; avoid on-the-hour spikes\n  workflow_dispatch:\n    inputs:\n      python_version:\n        description: \"Python version to test\"\n        default: \"3.12\"\n        required: true\n      run_slow:\n        description: \"Include slow tests\"\n        type: boolean\n        default: true\n\njobs:\n  thorough:\n    runs-on: ubuntu-latest\n    steps:\n      - uses: actions/checkout@v5\n      - uses: actions/setup-python@v6\n        with:\n          python-version: ${{ github.event.inputs.python_version || '3.12' }}\n          cache: pip\n      - run: pip install -e \".[test]\"\n      - name: Run tests\n        env:\n          HYPOTHESIS_PROFILE: nightly\n        run: |\n          args=\"\"\n          if [ \"${{ github.event.inputs.run_slow }}\" = \"true\" ]; then\n            args=\"--runslow\"\n          fi\n          pytest $args -v\n```\n\nScheduled runs catch flaky tests, dependency drift, and vulnerabilities that surface without code changes. `workflow_dispatch` gives a button in the Actions tab for on-demand runs; inputs let you parameterize the run. Avoid `cron: '0 * * * *'` exact-hour scheduling because many repos share those slots and get queued.",
)

add(
    "You are a CI expert.",
    "How do I skip CI if only docs or markdown files changed?",
    "Use `paths-ignore` or `paths` filters at the workflow level.\n\n```yaml\nname: tests\non:\n  push:\n    paths-ignore:\n      - \"**.md\"\n      - \"docs/**\"\n      - \".github/ISSUE_TEMPLATE/**\"\n      - \"LICENSE\"\n  pull_request:\n    paths-ignore:\n      - \"**.md\"\n      - \"docs/**\"\n```\n\nFor branch protection that requires a check to pass, don't skip — required checks fail if they never run. Instead, use a stub job that always succeeds for ignored paths via dorny/paths-filter:\n\n```yaml\njobs:\n  changes:\n    runs-on: ubuntu-latest\n    outputs:\n      code: ${{ steps.filter.outputs.code }}\n    steps:\n      - uses: actions/checkout@v5\n      - uses: dorny/paths-filter@v3\n        id: filter\n        with:\n          filters: |\n            code:\n              - \"**/*.py\"\n              - \"pyproject.toml\"\n              - \"uv.lock\"\n\n  test:\n    needs: changes\n    if: needs.changes.outputs.code == 'true'\n    runs-on: ubuntu-latest\n    steps:\n      - run: pytest\n\n  test-pass:\n    needs: changes\n    if: needs.changes.outputs.code != 'true'\n    runs-on: ubuntu-latest\n    steps:\n      - run: echo \"no code changes\"\n```\n\nEither `test` or `test-pass` runs, so the required check is always satisfied.",
)

add(
    "You are a coverage expert.",
    "How do I see which tests are covering a specific line of source?",
    "Use `--cov-context=test` with pytest-cov to record per-test context.\n\n```ini\n# pyproject.toml\n[tool.coverage.run]\ndynamic_context = \"test_function\"\n\n[tool.coverage.html]\nshow_contexts = true\n```\n\n```bash\npytest --cov=myapp --cov-report=html\nopen htmlcov/index.html\n```\n\nIn the HTML report, click a line to expand the list of tests that executed it. Useful for:\n- Identifying overly-coupled code (one line covered by dozens of tests)\n- Finding under-tested branches (only one test covers this critical path)\n- Locating flaky failures (which test actually touches this rarely-run code)\n\nProgrammatic access:\n\n```python\nimport coverage\ncov = coverage.Coverage()\ncov.load()\ndata = cov.get_data()\nfor ctx in data.measured_contexts():\n    print(ctx)\n```\n\nCaveat: context tracking is slower and the coverage DB is bigger. Enable only when you need the insight, not on every CI run.",
)

add(
    "You are a CI expert.",
    "Write a matrix job that only tests one Python version on non-Linux OSes.",
    "Use `include` to add extra cells, and `matrix.os != 'ubuntu-latest'` to prune.\n\n```yaml\njobs:\n  test:\n    runs-on: ${{ matrix.os }}\n    strategy:\n      fail-fast: false\n      matrix:\n        os: [ubuntu-latest]\n        python: [\"3.10\", \"3.11\", \"3.12\", \"3.13\"]\n        include:\n          - os: macos-latest\n            python: \"3.12\"\n          - os: windows-latest\n            python: \"3.12\"\n    steps:\n      - uses: actions/checkout@v5\n      - uses: actions/setup-python@v6\n        with:\n          python-version: ${{ matrix.python }}\n          cache: pip\n      - run: pip install -e \".[test]\"\n      - run: pytest\n```\n\nThis runs 4 Linux jobs (one per Python version) + 1 macOS + 1 Windows = 6 jobs instead of the 12 you would get from a full cross product. `include` adds cells to a matrix; `exclude` removes them; use whichever is shorter. The pattern maximizes coverage of what actually breaks (Python version diffs) while still catching platform-specific issues with minimal cost.",
)

add(
    "You are a CI expert.",
    "How do I set up GitHub Actions to publish docs to GitHub Pages?",
    "Build with Sphinx or MkDocs, deploy via `actions/deploy-pages`.\n\n```yaml\nname: docs\non:\n  push:\n    branches: [main]\n\npermissions:\n  contents: read\n  pages: write\n  id-token: write\n\nconcurrency:\n  group: pages\n  cancel-in-progress: false\n\njobs:\n  build:\n    runs-on: ubuntu-latest\n    steps:\n      - uses: actions/checkout@v5\n      - uses: actions/setup-python@v6\n        with: { python-version: \"3.12\", cache: pip }\n      - run: pip install -e \".[docs]\"\n      - run: mkdocs build --site-dir _site\n      - uses: actions/upload-pages-artifact@v3\n        with:\n          path: _site\n\n  deploy:\n    needs: build\n    runs-on: ubuntu-latest\n    environment:\n      name: github-pages\n      url: ${{ steps.deployment.outputs.page_url }}\n    steps:\n      - uses: actions/deploy-pages@v4\n        id: deployment\n```\n\nEnable Pages with source \"GitHub Actions\" in repository settings. The workflow is triggered on main pushes and deploys automatically. The `id-token` permission is required for OIDC-based deploys. `concurrency` prevents two simultaneous deploys from racing.",
)

add(
    "You are a CI expert.",
    "Write a pytest config that enforces warnings as errors.",
    "Promote all warnings to errors so regressions surface as test failures.\n\n```toml\n# pyproject.toml\n[tool.pytest.ini_options]\nfilterwarnings = [\n    \"error\",                                         # all warnings are errors...\n    \"ignore::DeprecationWarning:dateutil\",           # ...except from third-party we do not control\n    \"ignore:.*coroutine.*was never awaited:RuntimeWarning\",\n    \"ignore::pytest.PytestUnraisableExceptionWarning\",\n]\naddopts = \"-ra --strict-markers --strict-config\"\n```\n\nPer-test override:\n\n```python\nimport pytest\nimport warnings\n\n@pytest.mark.filterwarnings(\"ignore::DeprecationWarning\")\ndef test_legacy():\n    import legacy_module\n    legacy_module.old_call()\n\ndef test_expect_warning():\n    with pytest.warns(DeprecationWarning, match=\"use .* instead\"):\n        import legacy_module\n        legacy_module.old_call()\n```\n\nRule of thumb: when a new Python or library version adds a deprecation warning for your own code, you want the test to fail immediately — not silently pass until the deprecation turns into an error in a future version. Whitelist specific warnings from dependencies you cannot fix. Review the whitelist quarterly.",
)

add(
    "You are a CI expert.",
    "Run a smoke test against a deployed staging environment from GitHub Actions.",
    "Trigger on deploy event or after release; pytest with an env-based URL.\n\n```yaml\nname: smoke\non:\n  deployment_status:\n  workflow_dispatch:\n    inputs:\n      target:\n        description: \"URL to smoke test\"\n        default: \"https://staging.example.com\"\n\njobs:\n  smoke:\n    if: github.event.deployment_status.state == 'success' || github.event_name == 'workflow_dispatch'\n    runs-on: ubuntu-latest\n    steps:\n      - uses: actions/checkout@v5\n      - uses: actions/setup-python@v6\n        with: { python-version: \"3.12\", cache: pip }\n      - run: pip install -e \".[test]\"\n      - name: Smoke\n        env:\n          TARGET_URL: ${{ github.event.deployment_status.target_url || github.event.inputs.target }}\n          API_KEY: ${{ secrets.STAGING_API_KEY }}\n        run: pytest tests/smoke -v\n      - name: Notify on failure\n        if: failure()\n        run: |\n          curl -X POST -H \"Content-Type: application/json\" \\\n               -d '{\"text\":\"Smoke tests failed on ${{ env.TARGET_URL }}\"}' \\\n               ${{ secrets.SLACK_WEBHOOK }}\n```\n\nSmoke tests should be minimal: hit the health endpoint, try one authenticated request, one read, one write-then-delete. Under 2 minutes total. Their job is to catch \"the deploy went out but the app is not actually serving\" within seconds of the deploy finishing.",
)


Path("/tmp/gen_block_4.json").write_text(json.dumps(samples))
print(f"Block 4: {len(samples)}")
