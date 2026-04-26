#!/usr/bin/env python3
"""Block 5: Packaging (items 86-115)."""
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
    "You are a Python packaging expert.",
    "Write a modern pyproject.toml for a small library.",
    "PEP 621 style with setuptools as the build backend.\n\n```toml\n[build-system]\nrequires = [\"setuptools>=68\", \"setuptools-scm>=8\"]\nbuild-backend = \"setuptools.build_meta\"\n\n[project]\nname = \"mylib\"\ndynamic = [\"version\"]\ndescription = \"A small utility library.\"\nreadme = \"README.md\"\nlicense = \"MIT\"\nrequires-python = \">=3.10\"\nauthors = [{ name = \"Alice\", email = \"a@x.com\" }]\nkeywords = [\"utility\", \"example\"]\nclassifiers = [\n    \"Development Status :: 4 - Beta\",\n    \"Programming Language :: Python :: 3\",\n    \"Programming Language :: Python :: 3 :: Only\",\n    \"Programming Language :: Python :: 3.10\",\n    \"Programming Language :: Python :: 3.11\",\n    \"Programming Language :: Python :: 3.12\",\n    \"Programming Language :: Python :: 3.13\",\n]\ndependencies = [\n    \"httpx>=0.28\",\n    \"pydantic>=2.5\",\n]\n\n[project.optional-dependencies]\ntest = [\"pytest>=8\", \"pytest-cov\", \"hypothesis\"]\ndev = [\"ruff\", \"mypy\", \"pre-commit\"]\n\n[project.scripts]\nmylib = \"mylib.cli:main\"\n\n[project.urls]\nHomepage = \"https://github.com/alice/mylib\"\nIssues = \"https://github.com/alice/mylib/issues\"\n\n[tool.setuptools.packages.find]\nwhere = [\"src\"]\n\n[tool.setuptools_scm]\n# version comes from git tags\n```\n\nLayout: `src/mylib/__init__.py`. Build with `python -m build`, upload with `twine` or trusted publishing.",
)

add(
    "You are a packaging expert.",
    "Explain src-layout vs flat-layout.",
    "src-layout keeps package code in `src/pkg/`, flat-layout puts it in `pkg/` next to tests.\n\n```\n# src-layout\nmyproject/\n  src/\n    mylib/\n      __init__.py\n      core.py\n  tests/\n    test_core.py\n  pyproject.toml\n\n# flat-layout\nmyproject/\n  mylib/\n    __init__.py\n    core.py\n  tests/\n    test_core.py\n  pyproject.toml\n```\n\nWhy src-layout wins for libraries:\n1. Running `python -c 'import mylib'` from the project root only works if `mylib` is installed, not if Python finds the source directory by accident. This catches missing `install` in CI before release.\n2. `pytest` from the project root must resolve imports through the installed package, not the working directory, ensuring you test what users will actually get.\n3. Clear separation: `src/` is what ships, `tests/` and `docs/` are not.\n\nSetuptools config:\n\n```toml\n[tool.setuptools.packages.find]\nwhere = [\"src\"]\n```\n\nFor applications (not libraries), flat-layout is sometimes simpler, but src-layout remains the safer default whenever you will publish.",
)

add(
    "You are a packaging expert.",
    "Show me a poetry-based pyproject.toml.",
    "Poetry uses its own `[tool.poetry]` table instead of the PEP 621 `[project]`.\n\n```toml\n[tool.poetry]\nname = \"mylib\"\nversion = \"0.1.0\"\ndescription = \"A small utility library.\"\nauthors = [\"Alice <a@x.com>\"]\nreadme = \"README.md\"\nlicense = \"MIT\"\npackages = [{ include = \"mylib\", from = \"src\" }]\n\n[tool.poetry.dependencies]\npython = \"^3.10\"\nhttpx = \"^0.28\"\npydantic = \"^2.5\"\n\n[tool.poetry.group.dev.dependencies]\npytest = \"^8\"\npytest-cov = \"^6\"\nruff = \"*\"\nmypy = \"*\"\n\n[tool.poetry.scripts]\nmylib = \"mylib.cli:main\"\n\n[build-system]\nrequires = [\"poetry-core>=1.9.0\"]\nbuild-backend = \"poetry.core.masonry.api\"\n```\n\nCommon commands:\n\n```bash\npoetry install              # create venv and install deps + dev deps\npoetry add httpx            # add runtime dep\npoetry add --group dev mypy # add dev dep\npoetry lock --no-update     # regenerate lock without bumping versions\npoetry run pytest           # run in the managed env\npoetry shell                # activate\npoetry build                # produce wheel + sdist\npoetry publish              # upload to PyPI\n```\n\nPoetry 2.0+ added PEP 621 support so the newer style works too; either is valid.",
)

add(
    "You are a packaging expert.",
    "Write a pyproject.toml using uv and hatchling.",
    "uv is the package manager; hatchling is the build backend. They complement nicely.\n\n```toml\n[build-system]\nrequires = [\"hatchling>=1.26\"]\nbuild-backend = \"hatchling.build\"\n\n[project]\nname = \"mylib\"\nversion = \"0.1.0\"\ndescription = \"Example library.\"\nreadme = \"README.md\"\nrequires-python = \">=3.10\"\nlicense = \"MIT\"\ndependencies = [\"httpx>=0.28\"]\n\n[project.optional-dependencies]\ntest = [\"pytest\", \"pytest-cov\"]\n\n[tool.hatch.build.targets.wheel]\npackages = [\"src/mylib\"]\n\n[tool.uv]\ndev-dependencies = [\"ruff\", \"mypy\", \"pytest\"]\n```\n\nWorkflow:\n\n```bash\n# install uv once\ncurl -LsSf https://astral.sh/uv/install.sh | sh\n\nuv sync                     # create .venv, install locked deps\nuv run pytest               # run a command in the env\nuv add httpx                # add a runtime dep (updates pyproject + lock)\nuv add --dev mypy           # add a dev dep\nuv lock                     # regenerate uv.lock\nuv build                    # build sdist + wheel via hatchling\n```\n\nuv is 10-100x faster than pip+venv, ships a single static binary, and reads standard `pyproject.toml`. `uv.lock` is a universal lockfile covering every platform.",
)

add(
    "You are a packaging expert.",
    "How do I include non-Python data files (templates, JSON) in my package?",
    "Two options: `package_data`/`[tool.setuptools.package-data]` or including via MANIFEST.in.\n\n```toml\n# pyproject.toml, setuptools backend\n[tool.setuptools.packages.find]\nwhere = [\"src\"]\n\n[tool.setuptools.package-data]\n\"mylib\" = [\"templates/*.html\", \"data/*.json\", \"py.typed\"]\n```\n\n```toml\n# hatchling backend\n[tool.hatch.build.targets.wheel]\npackages = [\"src/mylib\"]\n\n[tool.hatch.build.targets.wheel.force-include]\n\"src/mylib/templates\" = \"mylib/templates\"\n```\n\nAccess at runtime with `importlib.resources` (preferred over manual path building):\n\n```python\nfrom importlib import resources\n\ndef load_schema():\n    # works both from installed package and from source checkout\n    return resources.files(\"mylib.data\").joinpath(\"schema.json\").read_text()\n\ndef stream_template(name):\n    with resources.files(\"mylib.templates\").joinpath(name).open(\"r\") as f:\n        return f.read()\n```\n\nNever use `__file__` + os.path to locate data files; it breaks when your package is inside a zip, installed via a pip cache, or running in a frozen bundle.",
)

add(
    "You are a packaging expert.",
    "Configure entry points so my CLI tool is callable after pip install.",
    "Use `[project.scripts]` in pyproject.toml.\n\n```toml\n[project]\nname = \"mytool\"\nversion = \"0.1.0\"\ndependencies = [\"click>=8\"]\n\n[project.scripts]\nmytool = \"mytool.cli:main\"\nmytool-admin = \"mytool.admin:main\"\n```\n\n```python\n# src/mytool/cli.py\nimport click\n\n@click.command()\n@click.option(\"--verbose\", is_flag=True)\ndef main(verbose):\n    click.echo(\"hello\")\n\nif __name__ == \"__main__\":\n    main()\n```\n\nAfter `pip install -e .`, both `mytool` and `mytool-admin` are on PATH. The format is `<command-name> = \"<module>:<callable>\"`. The callable is invoked with no args; it should read `sys.argv` or use click/typer to parse.\n\nPlugin-style extension hooks use `[project.entry-points.\"my.plugins\"]`:\n\n```toml\n[project.entry-points.\"myapp.plugins\"]\ntext = \"myapp_text_plugin:Plugin\"\njson = \"myapp_json_plugin:Plugin\"\n```\n\nDiscover at runtime:\n\n```python\nfrom importlib.metadata import entry_points\nfor ep in entry_points(group=\"myapp.plugins\"):\n    plugin_cls = ep.load()\n```",
)

add(
    "You are a packaging expert.",
    "How do I build a wheel and sdist?",
    "The modern PyPA tool is `build`. It is backend-agnostic.\n\n```bash\npip install build\npython -m build\n# produces:\n#   dist/mylib-0.1.0.tar.gz    (sdist)\n#   dist/mylib-0.1.0-py3-none-any.whl  (wheel)\n```\n\nWhat is what:\n- **sdist** (source distribution): tarball of source files + pyproject.toml. pip can build it from scratch on install. Required on PyPI for users on unusual platforms.\n- **wheel**: pre-built binary distribution. `pip install` just unpacks it. Much faster; mandatory for C extensions and the norm for pure-Python too.\n\nBuild only one:\n\n```bash\npython -m build --wheel        # skip sdist\npython -m build --sdist        # skip wheel\n```\n\nInspect before uploading:\n\n```bash\npip install twine\ntwine check dist/*             # validate README rendering etc.\nunzip -l dist/*.whl            # see what ended up in the wheel\ntar -tzf dist/*.tar.gz         # see sdist contents\n```\n\nUpload (prefer trusted publishing in CI):\n\n```bash\ntwine upload dist/*\ntwine upload --repository testpypi dist/*   # dry run\n```",
)

add(
    "You are a packaging expert.",
    "Pin versions reproducibly with a lock file.",
    "`pip-tools`, `uv`, or `poetry` all produce a lockfile.\n\n```bash\n# pip-tools\npip install pip-tools\npip-compile pyproject.toml --extra=dev -o requirements-dev.txt\npip-sync requirements-dev.txt\n\n# uv (unified, fastest)\nuv lock\nuv sync\n\n# poetry\npoetry lock\npoetry install\n```\n\nExample output fragment from `requirements-dev.txt`:\n\n```\n# This file was generated by pip-compile\nhttpx==0.28.1 \\\n    --hash=sha256:abc... \\\n    --hash=sha256:def...\n    # via mylib\nrich==13.9.4 \\\n    --hash=sha256:ghi...\n    # via httpx\n```\n\nWith hashes, `pip install --require-hashes` aborts if a dep on PyPI was swapped for a malicious one with a different hash. For supply-chain safety in production deploys, lock + require-hashes is the minimum bar.\n\nCommit the lockfile. Regenerate on a cadence (monthly or via Dependabot). The `pyproject.toml` records your loose constraints; the lockfile records the exact resolved versions for reproducibility.",
)

add(
    "You are a packaging expert.",
    "Show me how to test a package across multiple Python versions locally with uv.",
    "`uv python install` manages interpreters and `uv run --python` targets them.\n\n```bash\n# install multiple interpreters\nuv python install 3.10 3.11 3.12 3.13\n\n# run tests on each\nfor v in 3.10 3.11 3.12 3.13; do\n  echo \"== Python $v ==\"\n  uv run --python $v --with pytest pytest\ndone\n\n# or with a pinned project env\nuv sync --python 3.11\nuv run pytest\nuv sync --python 3.12\nuv run pytest\n```\n\nFor libraries, `tox` or `nox` orchestrate this more declaratively, but `uv` is faster for ad-hoc testing. Example `noxfile.py`:\n\n```python\nimport nox\n\n@nox.session(python=[\"3.10\", \"3.11\", \"3.12\", \"3.13\"])\ndef tests(session):\n    session.install(\".[test]\")\n    session.run(\"pytest\", *session.posargs)\n\n@nox.session\ndef lint(session):\n    session.install(\"ruff\")\n    session.run(\"ruff\", \"check\", \".\")\n    session.run(\"ruff\", \"format\", \"--check\", \".\")\n```\n\n```bash\nnox           # all sessions\nnox -s tests  # only test sessions\nnox -p 3.12   # only 3.12\n```",
)

add(
    "You are a packaging expert.",
    "Explain editable installs (pip install -e).",
    "An editable install wires the source tree into the environment so import finds your files directly.\n\n```bash\npip install -e .\n# equivalent:\npip install --editable .\n# with dev extras:\npip install -e \".[dev]\"\n```\n\nUnder PEP 660, `pip install -e .` asks the build backend for an editable wheel. Setuptools and hatchling both support this; poetry-core does not yet. Result: changes to source files are visible immediately in the interpreter without reinstalling.\n\nInspect what was installed:\n\n```bash\npip show -f mylib     # lists files; for editable, shows the .pth link to src\ncat $(python -c 'import mylib, pathlib; print(pathlib.Path(mylib.__file__).parent)')\n```\n\nUse cases:\n- Local development: edit and run without reinstall\n- Monorepos: install sibling packages editably so cross-package changes work\n- Pre-commit debug: when a hook runs your package, an editable install reflects uncommitted changes\n\nAvoid shipping editable installs to production. Use them only during development and CI on local branches.",
)

add(
    "You are a packaging expert.",
    "How do I include type hints with py.typed so mypy users get checks?",
    "Create an empty `py.typed` marker and tell the build backend to ship it.\n\n```\nsrc/\n  mylib/\n    __init__.py\n    core.py\n    py.typed        # empty file per PEP 561\n```\n\n```toml\n# setuptools\n[tool.setuptools.package-data]\n\"mylib\" = [\"py.typed\"]\n\n# hatchling\n[tool.hatch.build.targets.wheel]\npackages = [\"src/mylib\"]\n# hatchling includes non-*.py files inside packages by default\n```\n\nVerify:\n\n```bash\nunzip -l dist/mylib-0.1.0-py3-none-any.whl | grep py.typed\n```\n\nWithout `py.typed`, mypy silently ignores your annotations (treats the package as untyped). With it, downstream users get full type checking on your API.\n\nBonus: write a quick smoke check for your annotations.\n\n```python\n# tests/test_types.py\nimport subprocess\n\ndef test_self_mypy():\n    result = subprocess.run(\n        [\"mypy\", \"src/mylib\"], capture_output=True, text=True\n    )\n    assert result.returncode == 0, result.stdout + result.stderr\n```",
)

add(
    "You are a packaging expert.",
    "Configure version from git tags with setuptools-scm.",
    "`setuptools-scm` reads the version from git tags and writes it into the built artifacts.\n\n```toml\n[build-system]\nrequires = [\"setuptools>=68\", \"setuptools-scm>=8\"]\nbuild-backend = \"setuptools.build_meta\"\n\n[project]\nname = \"mylib\"\ndynamic = [\"version\"]\n\n[tool.setuptools_scm]\nversion_file = \"src/mylib/_version.py\"\nversion_scheme = \"post-release\"\nlocal_scheme = \"no-local-version\"\n```\n\nExpose the version:\n\n```python\n# src/mylib/__init__.py\ntry:\n    from ._version import __version__\nexcept ImportError:\n    __version__ = \"0.0.0+unknown\"\n```\n\nTag and build:\n\n```bash\ngit tag v1.2.3\ngit push --tags\npython -m build\n# dist/mylib-1.2.3-py3-none-any.whl\n```\n\nBetween tags, `setuptools-scm` auto-generates a dev version like `1.2.4.dev3+gabc1234`. `local_scheme = \"no-local-version\"` strips the `+gabc1234` suffix because PyPI rejects local versions. The big win: you never edit a version number in source again. Releases are just `git tag`.",
)

add(
    "You are a packaging expert.",
    "Set up Dependabot for a Python project.",
    "Add `.github/dependabot.yml` at repo root.\n\n```yaml\nversion: 2\nupdates:\n  - package-ecosystem: \"pip\"\n    directory: \"/\"\n    schedule:\n      interval: \"weekly\"\n      day: \"monday\"\n    open-pull-requests-limit: 5\n    groups:\n      dev-dependencies:\n        dependency-type: \"development\"\n      production-minor-patch:\n        dependency-type: \"production\"\n        update-types: [\"minor\", \"patch\"]\n    ignore:\n      - dependency-name: \"numpy\"\n        update-types: [\"version-update:semver-major\"]\n    labels: [\"deps\"]\n\n  - package-ecosystem: \"github-actions\"\n    directory: \"/\"\n    schedule:\n      interval: \"monthly\"\n    groups:\n      actions:\n        patterns: [\"*\"]\n```\n\nDependabot reads `pyproject.toml`, `requirements*.txt`, `uv.lock`, or `poetry.lock`. `groups` bundles related updates into one PR to reduce noise. `ignore` pins risky majors. The `github-actions` ecosystem updates pinned action versions in `.github/workflows/`.\n\nCombine with the security scanner to auto-raise security PRs outside the weekly schedule:\n\n```yaml\n    security-updates:\n      open-pull-requests-limit: 10\n```\n\nAlternatives: Renovate (more configurable), `uv lock --upgrade` on a schedule, or `pip-audit --fix`.",
)

add(
    "You are a packaging expert.",
    "How do I specify optional dependencies for extras?",
    "`[project.optional-dependencies]` maps extra names to dep lists.\n\n```toml\n[project]\nname = \"mylib\"\nversion = \"0.1.0\"\ndependencies = [\"httpx>=0.28\"]\n\n[project.optional-dependencies]\npostgres = [\"psycopg[binary]>=3.2\"]\nredis = [\"redis>=5\"]\nfastapi = [\"fastapi>=0.115\", \"uvicorn[standard]>=0.30\"]\nall = [\"mylib[postgres,redis,fastapi]\"]\ndev = [\n    \"pytest>=8\",\n    \"pytest-cov\",\n    \"ruff\",\n    \"mypy\",\n    \"mylib[all]\",\n]\n```\n\nInstall with extras:\n\n```bash\npip install \"mylib[postgres]\"\npip install \"mylib[postgres,redis]\"\npip install \"mylib[all]\"        # every optional dep\npip install -e \".[dev]\"         # editable + dev extras\n```\n\nThe `all` extra re-references other extras, a common convention. Users who want everything get it in one flag. The `dev` extra typically includes testing/linting tools.\n\nAccess at runtime (soft-require):\n\n```python\ntry:\n    import psycopg\nexcept ImportError:\n    psycopg = None\n\ndef postgres_backend():\n    if psycopg is None:\n        raise RuntimeError(\"install with pip install mylib[postgres]\")\n    ...\n```",
)

add(
    "You are a virtualenv expert.",
    "What is the difference between venv, virtualenv, and conda?",
    "All three isolate Python environments; they differ in scope and speed.\n\n```bash\n# stdlib venv: Python 3.3+, zero deps, per-interpreter\npython3 -m venv .venv\nsource .venv/bin/activate\npip install -r requirements.txt\n\n# virtualenv: third-party, predates venv, faster, can target older pythons\npip install virtualenv\nvirtualenv --python=python3.12 .venv\nsource .venv/bin/activate\n\n# conda: full environment including non-Python deps (C libs, R, CUDA)\nconda create -n myenv python=3.12 numpy pytorch\nconda activate myenv\n```\n\nRule of thumb:\n- **venv**: your default choice for pure-Python projects. Ships with Python, no extra install.\n- **virtualenv**: use only if you need a feature venv lacks (rarely) or you target Python <3.3.\n- **conda**: use when you need non-Python deps bundled (CUDA, MKL, GDAL, binary scientific stack) or cross-platform binary compat.\n\nModern tools built on venv: `uv venv` (fast, caches), `poetry` (managed per-project), `pyenv` + `venv` (interpreter + env).",
)

add(
    "You are a virtualenv expert.",
    "How do I create a virtualenv with a specific Python version?",
    "Either install the interpreter and point venv at it, or let a tool manage both.\n\n```bash\n# with pyenv (manages interpreters)\npyenv install 3.12.7\npyenv install 3.11.11\npyenv local 3.12.7        # sets .python-version for this dir\npython -m venv .venv\nsource .venv/bin/activate\n\n# with uv (single tool handles everything)\nuv python install 3.12\nuv venv --python 3.12 .venv\nsource .venv/bin/activate\n\n# explicit interpreter path\n/opt/python-3.12/bin/python3.12 -m venv .venv\n\n# Windows\npy -3.12 -m venv .venv\n.venv\\Scripts\\activate\n```\n\nPin the version in your repo so collaborators and CI match:\n\n```\n# .python-version (pyenv reads this)\n3.12.7\n\n# pyproject.toml\n[project]\nrequires-python = \">=3.12\"\n```\n\nVerify:\n\n```bash\npython --version\npython -c \"import sys; print(sys.prefix)\"   # path to the venv\n```\n\n`uv` is the fastest way today: one binary handles interpreter install, venv creation, and dep resolution, and it does all of it in parallel.",
)

add(
    "You are a virtualenv expert.",
    "Write a script that bootstraps a dev environment.",
    "Shell script that creates venv, installs deps, and runs basic checks.\n\n```bash\n#!/usr/bin/env bash\nset -euo pipefail\n\nREQUIRED_PY=\"3.12\"\n\nif ! command -v python${REQUIRED_PY} >/dev/null; then\n    echo \"Python ${REQUIRED_PY} not found. Install via pyenv or uv python install.\" >&2\n    exit 1\nfi\n\nif [ ! -d .venv ]; then\n    echo \"Creating venv...\"\n    python${REQUIRED_PY} -m venv .venv\nfi\n\nsource .venv/bin/activate\n\npython -m pip install --upgrade pip\npip install -e \".[dev,test]\"\n\npre-commit install\npre-commit install --hook-type commit-msg\n\nprintf '\\n== environment ready ==\\n'\npython --version\npip --version\nprintf 'venv: %s\\n' \"$VIRTUAL_ENV\"\n\nif [ \"${RUN_TESTS:-1}\" = \"1\" ]; then\n    pytest -x --no-cov\nfi\n```\n\nSave as `scripts/bootstrap.sh`, `chmod +x`, commit. New contributors run `./scripts/bootstrap.sh` and are ready.\n\n`uv` version, one-liner:\n\n```bash\nuv sync && uv run pre-commit install && uv run pytest -x\n```\n\nKeep `Makefile` targets (`make setup`, `make test`) as a uniform interface regardless of which tool the script uses internally.",
)

add(
    "You are a virtualenv expert.",
    "How do I run a command in a venv without activating it?",
    "Invoke the venv's python or entry-point binaries directly.\n\n```bash\n# direct binary\n./.venv/bin/python script.py\n./.venv/bin/pytest tests/\n./.venv/bin/pip install requests\n\n# or the Scripts/ dir on Windows\n.venv\\Scripts\\python.exe script.py\n\n# uv makes this idiomatic\nuv run python script.py\nuv run pytest tests/\nuv run --with requests python -c \"import requests; print(requests.__version__)\"\n```\n\nUseful in shell scripts, cron jobs, Docker entry points, and Makefiles where activating a venv via `source` is awkward or unreliable. It also avoids confusion when scripts are called from another `$VIRTUAL_ENV` that is active.\n\nMakefile convention:\n\n```makefile\nPY := .venv/bin/python\nPYTEST := .venv/bin/pytest\n\n.PHONY: test\ntest:\n\t$(PYTEST) -n auto\n\n.PHONY: lint\nlint:\n\t$(PY) -m ruff check .\n\t$(PY) -m ruff format --check .\n```\n\nNothing depends on whether a shell has activated the venv. CI jobs and local terminals behave identically.",
)

add(
    "You are a logging expert.",
    "Show me a proper logging setup for an application.",
    "Use `logging.config.dictConfig` at startup and never the default root logger.\n\n```python\n# myapp/logging_config.py\nimport logging.config\n\nCONFIG = {\n    \"version\": 1,\n    \"disable_existing_loggers\": False,\n    \"formatters\": {\n        \"standard\": {\n            \"format\": \"%(asctime)s %(levelname)-8s %(name)s: %(message)s\",\n        },\n        \"json\": {\n            \"()\": \"pythonjsonlogger.jsonlogger.JsonFormatter\",\n            \"fmt\": \"%(asctime)s %(levelname)s %(name)s %(message)s\",\n        },\n    },\n    \"handlers\": {\n        \"console\": {\n            \"class\": \"logging.StreamHandler\",\n            \"stream\": \"ext://sys.stderr\",\n            \"formatter\": \"standard\",\n            \"level\": \"INFO\",\n        },\n        \"file\": {\n            \"class\": \"logging.handlers.RotatingFileHandler\",\n            \"filename\": \"app.log\",\n            \"maxBytes\": 10_000_000,\n            \"backupCount\": 5,\n            \"formatter\": \"json\",\n            \"level\": \"DEBUG\",\n        },\n    },\n    \"root\": {\"level\": \"DEBUG\", \"handlers\": [\"console\", \"file\"]},\n    \"loggers\": {\n        \"myapp\": {\"level\": \"DEBUG\"},\n        \"urllib3\": {\"level\": \"WARNING\"},\n    },\n}\n\ndef setup():\n    logging.config.dictConfig(CONFIG)\n```\n\nUsage:\n\n```python\nimport logging\nlog = logging.getLogger(__name__)\nlog.info(\"app started\")\n```\n\nEach module gets a logger named by its dotted path, so you can raise/lower noise per subsystem with one config change.",
)

add(
    "You are a logging expert.",
    "Explain DEBUG vs INFO vs WARNING vs ERROR vs CRITICAL levels.",
    "Five severity levels plus NOTSET; use them with intent.\n\n```python\nimport logging\nlog = logging.getLogger(__name__)\n\nlog.debug(\"loop iteration %d with batch size %d\", i, n)\n# noisy details only useful when tracing a bug\n\nlog.info(\"connected to %s in %.2fms\", url, ms)\n# normal operational events an operator wants to see\n\nlog.warning(\"retrying after timeout (%d/%d)\", attempt, max_attempts)\n# something unexpected but recoverable; deserves attention later\n\nlog.error(\"failed to send email for user_id=%s\", user_id, exc_info=True)\n# an operation failed; some user-visible action did not complete\n\nlog.critical(\"out of disk, refusing new writes\")\n# the process cannot continue normal operation\n```\n\nRules:\n1. Never log and re-raise the same exception unless you add context; the caller will log.\n2. Pass variadic args (`%s`), not f-strings, so formatting is deferred when the level is disabled.\n3. `exc_info=True` or `log.exception(...)` captures the traceback.\n4. Set production to INFO, CI and local dev to DEBUG, integration tests to WARNING to reduce noise.\n5. WARNING is for conditions someone should look at within a day. ERROR is for conditions someone should page on (depending on rate).",
)

add(
    "You are a logging expert.",
    "How do I add structured logging with context like request_id?",
    "Use a `contextvars`-based filter or `structlog`.\n\n```python\n# stdlib + contextvars\nimport logging\nimport contextvars\nimport uuid\n\nrequest_id_var = contextvars.ContextVar(\"request_id\", default=\"-\")\n\nclass RequestIdFilter(logging.Filter):\n    def filter(self, record):\n        record.request_id = request_id_var.get()\n        return True\n\nlogging.basicConfig(\n    format=\"%(asctime)s [%(request_id)s] %(levelname)s %(name)s: %(message)s\",\n    level=logging.INFO,\n)\nlogging.getLogger().addFilter(RequestIdFilter())\n\n# in a web middleware\ndef middleware(req, handler):\n    token = request_id_var.set(req.headers.get(\"X-Request-Id\", uuid.uuid4().hex))\n    try:\n        return handler(req)\n    finally:\n        request_id_var.reset(token)\n```\n\nWith structlog, the story is cleaner:\n\n```python\nimport structlog\n\nstructlog.configure(processors=[\n    structlog.contextvars.merge_contextvars,\n    structlog.processors.TimeStamper(fmt=\"iso\"),\n    structlog.processors.add_log_level,\n    structlog.processors.JSONRenderer(),\n])\nlog = structlog.get_logger()\n\nstructlog.contextvars.bind_contextvars(request_id=\"abc123\", user_id=42)\nlog.info(\"user action\", action=\"purchase\", amount=19.99)\n# {\"timestamp\":\"...\",\"level\":\"info\",\"event\":\"user action\",\"request_id\":\"abc123\",\"user_id\":42,...}\n```\n\nStructured JSON logs are query-friendly in ELK, Datadog, Loki.",
)

add(
    "You are a logging expert.",
    "How do I test that my code emits the right log messages?",
    "Use pytest's `caplog` fixture to inspect records.\n\n```python\nimport logging\nimport pytest\n\ndef process(payload):\n    log = logging.getLogger(\"myapp.processor\")\n    if not payload:\n        log.warning(\"empty payload ignored\")\n        return\n    log.info(\"processed %d items\", len(payload))\n\ndef test_warning_on_empty(caplog):\n    with caplog.at_level(logging.WARNING, logger=\"myapp.processor\"):\n        process([])\n    assert any(\n        r.levelno == logging.WARNING and \"empty payload\" in r.message\n        for r in caplog.records\n    )\n\ndef test_info_on_valid(caplog):\n    with caplog.at_level(logging.INFO, logger=\"myapp.processor\"):\n        process([1, 2, 3])\n    msgs = [r.getMessage() for r in caplog.records]\n    assert \"processed 3 items\" in msgs\n\ndef test_no_error(caplog):\n    process([1])\n    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]\n```\n\n`caplog.records` gives the list of `LogRecord` objects. `r.message` is the pre-formatted template; `r.getMessage()` is after `%` interpolation. `caplog.at_level` ensures the logger is temporarily set to capture the desired level.",
)

add(
    "You are a logging expert.",
    "Log exceptions properly with tracebacks.",
    "Use `log.exception` inside except blocks; it adds `exc_info=True` for you.\n\n```python\nimport logging\nlog = logging.getLogger(__name__)\n\ndef process_file(path):\n    try:\n        with open(path) as f:\n            return f.read()\n    except FileNotFoundError:\n        log.exception(\"file not found: %s\", path)\n        return None\n    except Exception:\n        log.exception(\"unexpected error reading %s\", path)\n        raise\n```\n\nOutput includes full traceback:\n\n```\n2026-04-24 10:00:00 ERROR myapp.io: file not found: /tmp/x.txt\nTraceback (most recent call last):\n  File \"myapp/io.py\", line 6, in process_file\n    with open(path) as f:\n         ^^^^^^^^^^\nFileNotFoundError: [Errno 2] No such file or directory: '/tmp/x.txt'\n```\n\nEquivalent forms:\n\n```python\nlog.error(\"msg\", exc_info=True)        # same as log.exception(\"msg\")\nlog.error(\"msg\", exc_info=e)           # specific exception instance\n```\n\nNever do `log.error(str(e))` — you lose the traceback and the event context. Always `log.exception(...)` so diagnostic info gets captured, and let the exception propagate if it is fatal for the caller.",
)

add(
    "You are a logging expert.",
    "Rotate log files by size or by time.",
    "Use `RotatingFileHandler` for size, `TimedRotatingFileHandler` for time.\n\n```python\nimport logging\nfrom logging.handlers import RotatingFileHandler, TimedRotatingFileHandler\n\n# by size: rotate when file hits 10MB, keep 5 backups\nsize_handler = RotatingFileHandler(\n    \"app.log\", maxBytes=10_000_000, backupCount=5, encoding=\"utf-8\"\n)\n\n# by time: rotate every midnight, keep 14 days\ntime_handler = TimedRotatingFileHandler(\n    \"app.log\", when=\"midnight\", backupCount=14, encoding=\"utf-8\"\n)\n\nlogger = logging.getLogger()\nlogger.setLevel(logging.INFO)\nlogger.addHandler(time_handler)\n```\n\n`when` values: `\"S\"`, `\"M\"`, `\"H\"`, `\"D\"`, `\"midnight\"`, `\"W0\"`-`\"W6\"` (weekly, day-of-week). Rotated files get suffixes like `app.log.2026-04-24`.\n\nFor containerized apps, prefer stdout/stderr and let the container runtime rotate:\n\n```python\nimport logging, sys\nlogging.basicConfig(stream=sys.stderr, level=logging.INFO)\n```\n\nThen Docker/Kubernetes/journald handles rotation. Writing to files from inside a container fights the platform; writing to stdout is the twelve-factor way.",
)

add(
    "You are a logging expert.",
    "How do I silence a noisy third-party library's logging?",
    "Lower its logger's level to WARNING or ERROR by name.\n\n```python\nimport logging\n\n# find the library's logger name: usually its top-level module\nlogging.getLogger(\"urllib3\").setLevel(logging.WARNING)\nlogging.getLogger(\"requests\").setLevel(logging.WARNING)\nlogging.getLogger(\"botocore\").setLevel(logging.ERROR)\nlogging.getLogger(\"asyncio\").setLevel(logging.WARNING)\nlogging.getLogger(\"pytest\").setLevel(logging.WARNING)\n```\n\nOr in a dictConfig `loggers:` block:\n\n```python\n\"loggers\": {\n    \"urllib3\": {\"level\": \"WARNING\"},\n    \"botocore\": {\"level\": \"ERROR\"},\n    \"sqlalchemy.engine\": {\"level\": \"WARNING\"},  # DEBUG to see all SQL\n}\n```\n\nHow to find the right logger name:\n\n```python\nimport logging\nlogging.basicConfig(level=logging.DEBUG)\n# run the noisy code, inspect\nfor name in sorted(logging.Logger.manager.loggerDict):\n    print(name)\n```\n\nThe noisy output will have the logger name in its header (`%(name)s`). Set that name's level to silence it. Avoid using `logging.disable()` globally — it breaks legitimate logs too.",
)

add(
    "You are a logging expert.",
    "Set up JSON logging for a production service.",
    "Use `python-json-logger` (or structlog) with a JSON formatter.\n\n```python\nimport logging\nimport sys\nfrom pythonjsonlogger import jsonlogger\n\ndef setup():\n    handler = logging.StreamHandler(sys.stdout)\n    fmt = jsonlogger.JsonFormatter(\n        fmt=\"%(asctime)s %(levelname)s %(name)s %(message)s\",\n        rename_fields={\"asctime\": \"timestamp\", \"levelname\": \"level\"},\n        timestamp=True,\n    )\n    handler.setFormatter(fmt)\n    root = logging.getLogger()\n    root.handlers = [handler]\n    root.setLevel(logging.INFO)\n\nsetup()\nlog = logging.getLogger(\"myapp\")\nlog.info(\"user login\", extra={\"user_id\": 42, \"ip\": \"10.0.0.1\"})\n```\n\nOutput:\n\n```json\n{\"timestamp\": \"2026-04-24T10:00:00\", \"level\": \"INFO\", \"name\": \"myapp\", \"message\": \"user login\", \"user_id\": 42, \"ip\": \"10.0.0.1\"}\n```\n\nEverything in `extra=` becomes a top-level JSON field, perfect for querying in Datadog/Elastic. One log line per event, parseable by any pipeline.\n\nOne gotcha: `extra` cannot use reserved field names (`message`, `levelname`, `name`). If your context key clashes, prefix it: `extra={\"ctx_name\": ...}`.\n\nstructlog gives a nicer API (`log.info(\"event\", user_id=42)`) plus native JSON rendering without the `extra=` boilerplate; consider it if you are starting fresh.",
)

add(
    "You are a packaging expert.",
    "Configure ruff as the linter and formatter.",
    "`[tool.ruff]` in pyproject.toml.\n\n```toml\n[tool.ruff]\nline-length = 100\ntarget-version = \"py312\"\nsrc = [\"src\", \"tests\"]\nextend-exclude = [\"migrations\"]\n\n[tool.ruff.lint]\nselect = [\n    \"E\", \"W\",  # pycodestyle\n    \"F\",       # pyflakes\n    \"I\",       # isort\n    \"B\",       # flake8-bugbear\n    \"C4\",      # flake8-comprehensions\n    \"UP\",      # pyupgrade\n    \"SIM\",     # flake8-simplify\n    \"RUF\",     # ruff-specific\n]\nignore = [\n    \"E501\",    # line too long (formatter handles this)\n    \"B008\",    # function call in default arg\n]\n\n[tool.ruff.lint.per-file-ignores]\n\"tests/**\" = [\"S101\"]       # allow asserts\n\"__init__.py\" = [\"F401\"]    # unused imports (re-exports)\n\n[tool.ruff.lint.isort]\nknown-first-party = [\"myapp\"]\ncombine-as-imports = true\n\n[tool.ruff.format]\nquote-style = \"double\"\nindent-style = \"space\"\nline-ending = \"lf\"\n```\n\nCommands:\n\n```bash\nruff check .            # lint\nruff check . --fix      # auto-fix what it can\nruff format .           # format\nruff format --check .   # CI check without writing\n```\n\nRuff replaces flake8, isort, pyupgrade, and (increasingly) black. It is 10-100x faster and configured in one place.",
)

add(
    "You are a packaging expert.",
    "Configure mypy strictly for a new project.",
    "Start strict and relax only what you must.\n\n```toml\n[tool.mypy]\npython_version = \"3.12\"\nfiles = [\"src\", \"tests\"]\nstrict = true\nwarn_unused_configs = true\nwarn_return_any = true\nwarn_unreachable = true\nno_implicit_reexport = true\ndisallow_untyped_defs = true\nshow_error_codes = true\npretty = true\n\n[[tool.mypy.overrides]]\nmodule = [\"tests.*\"]\ndisallow_untyped_defs = false\n\n[[tool.mypy.overrides]]\nmodule = [\"legacy.*\", \"third_party_without_stubs\"]\nignore_missing_imports = true\n```\n\n`strict = true` turns on a bundle of flags (`disallow_any_generics`, `disallow_untyped_calls`, etc.). Override per-module via `[[tool.mypy.overrides]]` when you cannot fix everything at once.\n\nFor gradual adoption on an existing codebase:\n\n```toml\n[tool.mypy]\npython_version = \"3.12\"\nfiles = [\"src\"]\ncheck_untyped_defs = false\ndisallow_untyped_defs = false\n\n# tighten per module as you type it\n[[tool.mypy.overrides]]\nmodule = [\"src.core\", \"src.auth\"]\ndisallow_untyped_defs = true\nstrict = true\n```\n\nRun in CI:\n\n```bash\nmypy src tests\n```\n\nStore a `mypy.cache_dir` in CI via `actions/cache` to speed up subsequent runs.",
)

add(
    "You are a packaging expert.",
    "Write a Dockerfile that uses uv for fast, reproducible builds.",
    "Multi-stage with bytecode compilation and an immutable lock.\n\n```dockerfile\nFROM python:3.12-slim AS builder\nCOPY --from=ghcr.io/astral-sh/uv:0.5.20 /uv /uvx /bin/\n\nENV UV_COMPILE_BYTECODE=1 \\\n    UV_LINK_MODE=copy \\\n    UV_PYTHON_DOWNLOADS=never\n\nWORKDIR /app\n\nCOPY pyproject.toml uv.lock ./\nRUN --mount=type=cache,target=/root/.cache/uv \\\n    uv sync --frozen --no-install-project --no-dev\n\nCOPY . .\nRUN --mount=type=cache,target=/root/.cache/uv \\\n    uv sync --frozen --no-dev\n\nFROM python:3.12-slim\nWORKDIR /app\n\nCOPY --from=builder /app /app\nENV PATH=\"/app/.venv/bin:$PATH\"\nENV PYTHONUNBUFFERED=1\n\nUSER nobody\nCMD [\"python\", \"-m\", \"myapp\"]\n```\n\nKey points:\n- Copy `pyproject.toml` + `uv.lock` first and install, so Docker caches dep install independent of source changes.\n- `--frozen` aborts if the lock is out of sync with pyproject (safety in CI).\n- `UV_COMPILE_BYTECODE=1` precompiles `.pyc` at build time for faster cold start.\n- BuildKit cache mount speeds up local builds; transparent in CI with buildx.\n- Final stage does not include uv binary, keeping the runtime image small (~60MB with slim base).",
)

add(
    "You are a packaging expert.",
    "What is vendored dependencies and when is it appropriate?",
    "Vendoring is copying a dep's source into your repo instead of installing it.\n\n```\nyourpkg/\n  _vendor/\n    __init__.py\n    urllib3/       # full copy of a specific urllib3 version\n  core.py          # imports from yourpkg._vendor.urllib3\n```\n\nUse cases (all rare):\n1. You ship a tool (pip, pipx, uv) that must not depend on the target Python's libraries, because it installs them.\n2. You distribute to an air-gapped environment with no package index.\n3. You need to patch a library in a way upstream rejects, and forking + renaming is impractical.\n\nWhy it is usually wrong:\n- You own security updates for every CVE in every vendored dep.\n- Diamond dependency problems multiply.\n- Bundle size grows; so does install friction if you publish to PyPI.\n\nIf you must vendor, use `vendoring`:\n\n```toml\n[tool.vendoring]\ndestination = \"src/myapp/_vendor/\"\nrequirements = \"src/myapp/_vendor/vendor.txt\"\npatches-dir = \"src/myapp/_vendor/patches\"\nprotected-files = [\"__init__.py\", \"README.md\"]\n```\n\nPin exact versions, strip tests, rewrite imports to point at `._vendor`. Keep a changelog.\n\nFor most projects: just `pip install` the dep. Vendoring is a last resort for special cases, not an optimization.",
)

add(
    "You are a packaging expert.",
    "Create a requirements.txt from pyproject.toml for deployment.",
    "Use `pip-compile`, `uv pip compile`, or `poetry export`.\n\n```bash\n# pip-tools (classic)\npip install pip-tools\npip-compile pyproject.toml -o requirements.txt\npip-compile pyproject.toml --extra=prod -o requirements-prod.txt\npip-compile pyproject.toml --extra=dev -o requirements-dev.txt\n\n# uv (fastest)\nuv pip compile pyproject.toml -o requirements.txt\nuv pip compile --extra=dev pyproject.toml -o requirements-dev.txt\n\n# poetry\npoetry export -f requirements.txt -o requirements.txt --without-hashes\npoetry export -f requirements.txt -o requirements-dev.txt --with dev\n```\n\nAdd `--generate-hashes` (pip-compile) or default (uv) to lock hashes, then install with:\n\n```bash\npip install --require-hashes -r requirements.txt\n```\n\n`--require-hashes` prevents a compromised mirror from swapping in a malicious wheel with the same version number. Essential for production.\n\nKeep `pyproject.toml` loose (compatible version ranges) and `requirements.txt` locked (exact pinned versions). Regenerate the lock on a cadence or via Dependabot. Commit both.",
)


Path("/tmp/gen_block_5.json").write_text(json.dumps(samples))
print(f"Block 5: {len(samples)}")
