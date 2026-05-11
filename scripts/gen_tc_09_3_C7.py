"""
TC-C subagent generator for Lyra phase 09.3 batch 07.
Domain: filesystem, git, shell, package managers, build tools.
Produces 500 ShareGPT samples at datasets/tool-calling/raw-09.3/batch-07-C.jsonl
"""
from __future__ import annotations
import json
import os
import random
from pathlib import Path

SEED = "1009313C"
OUT = Path("/Users/lakshman/Documents/Lyra/datasets/tool-calling/raw-09.3/batch-13-C.jsonl")

SUFFIX_POOL = [
    "That's all pulled up and ready to go.",
    "Done — the data came back clean, everything lines up with what you originally asked for, and I don't see any mismatches worth flagging in the payload before you move on to the next step.",
    "Pulled the record successfully; the value you needed is included in the response above.",
    "All set on my end.",
    "Got it handled — want me to take this further, or is that enough?",
    "Operation wrapped up without issues, so you should be good to proceed from here.",
    "Finished pulling that together; let me know if you'd like a deeper breakdown of any particular field, or if you'd prefer I reformat the response into something more readable for downstream consumption.",
    "The call routed through cleanly and returned exactly what you were after.",
    "Wrapped up — anything else you'd like me to chase down while we're here?",
    "That request returned successfully, and the full payload sits just above for your review whenever you're ready to look through it at your own pace.",
    "Verified and delivered.",
    "Fetched and parsed — the numbers check out against what the API reported.",
    "Sorted out; the output is self-explanatory but I'm happy to walk through it.",
    "Query executed, results returned, and nothing looked off in the response body.",
    "All yours — holler if you need a follow-up lookup, want to drill into the details, or spot anything in the output that deserves a second pass from my end.",
    "Task closed out. If this raises new questions, just say the word and I'll pivot.",
    "Response is ready above, covering each of the fields you originally requested.",
    "Returned cleanly without errors.",
    "Everything ran end-to-end, the result matches the shape of what you were expecting, and I'd lean toward calling this one finished unless you want me to cross-check anything against another source for sanity.",
    "Just finished the lookup — does this cover what you needed, or should I keep digging?",
    "Output is queued up above; it should give you what you need to move forward.",
    "Sent off, processed, confirmed — the operation completed in a single round-trip.",
    "Here you go.",
    "Call succeeded on the first attempt, no retries or fallback logic had to kick in this time, and the timings look perfectly normal compared to prior runs of the same endpoint.",
    "That's wrapped — feel free to ask if any of the returned fields need clarification.",
    "Information retrieved; I've kept the raw response intact so you can inspect it directly.",
    "Done and dusted.",
    "Happy to refine further if the result isn't quite what you were picturing — otherwise, we're good to call this one shipped and roll on to whatever's next on your list.",
    "Fetched successfully, and the shape of the payload aligns with the documented schema.",
    "That should do it on this one.",
]
assert len(SUFFIX_POOL) == 30

SYSTEM_PROMPTS = [
    "You are a helpful assistant. Prefer calling tools over guessing.",
    "You are a developer-focused assistant for filesystem, git, and build operations. Always invoke a tool when one is available.",
    "You are a CLI-aware engineering assistant. Use tools instead of speculating about local state.",
    "You are an assistant for shell, git, and package-manager workflows. Call tools for any state-dependent answer.",
    "You are a build/devops helper. Prefer tool calls for any inspection or mutation of the workspace.",
]


# ----------------------------- TOOL DEFINITIONS -----------------------------
# Each tool: (name, description, params_schema, sample_args_fn, sample_result_fn)
# We define ≥40 tools.

import string

def rng_for(seed_str: str) -> random.Random:
    return random.Random(seed_str)


def rand_path(r: random.Random) -> str:
    parts = r.choice([
        ["src", "components", "Header.tsx"],
        ["src", "lib", "auth.py"],
        ["app", "routes", "api.ts"],
        ["pkg", "store", "redis.go"],
        ["internal", "db", "migrations", "0007_add_index.sql"],
        ["docs", "README.md"],
        ["config", "settings.yaml"],
        ["scripts", "deploy.sh"],
        ["tests", "unit", "test_models.py"],
        ["src", "utils", "format.js"],
        ["server", "handlers", "auth.go"],
        ["client", "hooks", "useAuth.ts"],
        ["packages", "ui", "Button.tsx"],
        ["data", "seed.json"],
        ["build", "out.wasm"],
        [".github", "workflows", "ci.yml"],
        ["Cargo.toml"],
        ["pyproject.toml"],
        ["package.json"],
        ["go.mod"],
    ])
    return "/".join(parts)


def rand_dir(r: random.Random) -> str:
    return r.choice([
        "src/", "src/components/", "src/lib/", "tests/", "docs/",
        "scripts/", "config/", "build/", "dist/", "internal/",
        "pkg/", "app/routes/", "client/", "server/", "node_modules/",
        ".github/workflows/", "migrations/", "vendor/", "tmp/",
    ])


def rand_branch(r: random.Random) -> str:
    return r.choice([
        "feature/auth-rework", "feature/payment-flow", "bugfix/login-redirect",
        "main", "develop", "release/v2.4.0", "hotfix/timezone-bug",
        "chore/upgrade-deps", "feature/dark-mode", "experiment/wasm-runtime",
        "fix/memory-leak", "feature/onboarding-v2", "perf/query-cache",
        "refactor/api-layer", "docs/readme-update",
    ])


def rand_sha(r: random.Random) -> str:
    return "".join(r.choice("0123456789abcdef") for _ in range(7))


def rand_pkg_npm(r: random.Random) -> str:
    return r.choice(["react", "lodash", "axios", "express", "vite", "typescript",
                     "@types/node", "zod", "tailwindcss", "next", "vitest", "eslint"])


def rand_pkg_pip(r: random.Random) -> str:
    return r.choice(["requests", "numpy", "pandas", "fastapi", "pydantic", "pytest",
                     "httpx", "sqlalchemy", "uvicorn", "rich", "typer", "anthropic"])


def rand_pkg_cargo(r: random.Random) -> str:
    return r.choice(["serde", "tokio", "clap", "anyhow", "reqwest", "rayon", "axum"])


def rand_pkg_brew(r: random.Random) -> str:
    return r.choice(["ripgrep", "fzf", "jq", "git", "node", "python@3.11", "wget", "tmux", "neovim"])


def rand_size(r: random.Random) -> int:
    return r.randint(40, 524288)


def rand_lines(r: random.Random) -> int:
    return r.randint(3, 980)


# Tool registry
TOOLS = []

def tool(name, desc, params, sample_args, sample_result, user_phrasings):
    TOOLS.append({
        "name": name,
        "desc": desc,
        "params": params,
        "args_fn": sample_args,
        "result_fn": sample_result,
        "phrasings": user_phrasings,
    })


# --- filesystem ---
tool("read_file", "Read the contents of a file.",
     {"type": "object", "properties": {"path": {"type": "string"}, "encoding": {"type": "string"}}, "required": ["path"]},
     lambda r: {"path": rand_path(r)},
     lambda r, a: {"path": a["path"], "bytes": rand_size(r), "lines": rand_lines(r)},
     ["Read {path} for me.", "Show me what's in {path}.", "Open {path} and dump the contents.",
      "Pull up {path}.", "What does {path} say right now?"])

tool("write_file", "Write content to a file (overwrite).",
     {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]},
     lambda r: {"path": rand_path(r), "content": r.choice(["# TODO\n", "export const VERSION = '1.0.0';\n", "[]\n", "{}\n"])},
     lambda r, a: {"path": a["path"], "written": True, "bytes": len(a["content"])},
     ["Overwrite {path} with the placeholder.", "Reset {path} to a stub.", "Write a minimal scaffold to {path}."])

tool("list_dir", "List directory contents.",
     {"type": "object", "properties": {"path": {"type": "string"}, "recursive": {"type": "boolean"}}, "required": ["path"]},
     lambda r: {"path": rand_dir(r)},
     lambda r, a: {"path": a["path"], "entries": r.randint(2, 47)},
     ["What's in {path}?", "List the contents of {path}.", "Show me everything under {path}.",
      "Enumerate the files in {path}.", "Inspect {path}."])

tool("find_files", "Find files by glob pattern.",
     {"type": "object", "properties": {"pattern": {"type": "string"}, "root": {"type": "string"}}, "required": ["pattern"]},
     lambda r: {"pattern": r.choice(["**/*.ts", "**/*.py", "**/test_*.py", "**/*.lock", "**/*.md"]), "root": rand_dir(r)},
     lambda r, a: {"pattern": a["pattern"], "matches": r.randint(1, 84)},
     ["Find all {pattern} files under {root}.", "Locate every {pattern} match in {root}.",
      "Search {root} for {pattern}.", "Hunt down {pattern} files."])

tool("grep_files", "Search file contents for a regex.",
     {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}}, "required": ["pattern", "path"]},
     lambda r: {"pattern": r.choice(["TODO", "FIXME", "console\\.log", "import React", "@deprecated"]), "path": rand_dir(r)},
     lambda r, a: {"pattern": a["pattern"], "files_with_matches": r.randint(0, 31), "total_matches": r.randint(0, 122)},
     ["Grep {path} for /{pattern}/.", "Where does /{pattern}/ show up in {path}?",
      "Search {path} for the regex /{pattern}/.", "Hunt /{pattern}/ across {path}."])

tool("delete_file", "Delete a file.",
     {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
     lambda r: {"path": rand_path(r)},
     lambda r, a: {"path": a["path"], "deleted": True},
     ["Remove {path}.", "Delete {path}.", "Nuke {path} from the tree.", "Get rid of {path}."])

tool("move_file", "Move or rename a file.",
     {"type": "object", "properties": {"src": {"type": "string"}, "dest": {"type": "string"}}, "required": ["src", "dest"]},
     lambda r: {"src": rand_path(r), "dest": rand_path(r)},
     lambda r, a: {"src": a["src"], "dest": a["dest"], "moved": True},
     ["Move {src} to {dest}.", "Rename {src} -> {dest}.", "Relocate {src} into {dest}."])

tool("copy_file", "Copy a file.",
     {"type": "object", "properties": {"src": {"type": "string"}, "dest": {"type": "string"}}, "required": ["src", "dest"]},
     lambda r: {"src": rand_path(r), "dest": rand_path(r)},
     lambda r, a: {"src": a["src"], "dest": a["dest"], "copied": True, "bytes": rand_size(r)},
     ["Copy {src} to {dest}.", "Duplicate {src} as {dest}.", "Clone {src} over to {dest}."])

tool("mkdir", "Create a directory.",
     {"type": "object", "properties": {"path": {"type": "string"}, "parents": {"type": "boolean"}}, "required": ["path"]},
     lambda r: {"path": rand_dir(r), "parents": True},
     lambda r, a: {"path": a["path"], "created": True},
     ["Create the {path} directory.", "Make a folder at {path}.", "Set up {path} (with parents)."])

tool("chmod", "Change file permissions.",
     {"type": "object", "properties": {"path": {"type": "string"}, "mode": {"type": "string"}}, "required": ["path", "mode"]},
     lambda r: {"path": rand_path(r), "mode": r.choice(["755", "644", "600", "+x"])},
     lambda r, a: {"path": a["path"], "mode": a["mode"], "ok": True},
     ["Set {path} to mode {mode}.", "Change perms on {path} to {mode}.", "chmod {mode} on {path}."])

tool("stat_file", "Get file metadata (size, mtime, owner).",
     {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
     lambda r: {"path": rand_path(r)},
     lambda r, a: {"path": a["path"], "size": rand_size(r), "mtime": "2026-04-{:02d}T10:14:09Z".format(r.randint(1, 28))},
     ["Stat {path}.", "What's the size and mtime of {path}?", "Give me metadata for {path}."])

tool("tail_file", "Read last N lines of a file.",
     {"type": "object", "properties": {"path": {"type": "string"}, "n": {"type": "integer"}}, "required": ["path"]},
     lambda r: {"path": rand_path(r), "n": r.choice([20, 50, 100, 200])},
     lambda r, a: {"path": a["path"], "lines_returned": a["n"]},
     ["Show me the last {n} lines of {path}.", "Tail {path} (last {n}).", "Give me the tail of {path}, {n} lines."])

tool("head_file", "Read first N lines of a file.",
     {"type": "object", "properties": {"path": {"type": "string"}, "n": {"type": "integer"}}, "required": ["path"]},
     lambda r: {"path": rand_path(r), "n": r.choice([10, 20, 50])},
     lambda r, a: {"path": a["path"], "lines_returned": a["n"]},
     ["Show me the first {n} lines of {path}.", "Head {path} ({n}).", "Peek the top {n} of {path}."])

tool("wc_lines", "Count lines in a file.",
     {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
     lambda r: {"path": rand_path(r)},
     lambda r, a: {"path": a["path"], "lines": rand_lines(r)},
     ["How many lines is {path}?", "Line-count {path}.", "wc -l on {path}, please."])

tool("touch_file", "Create an empty file or update mtime.",
     {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
     lambda r: {"path": rand_path(r)},
     lambda r, a: {"path": a["path"], "touched": True},
     ["Touch {path}.", "Create an empty {path}.", "Make {path} exist (empty is fine)."])

tool("symlink", "Create a symbolic link.",
     {"type": "object", "properties": {"target": {"type": "string"}, "link": {"type": "string"}}, "required": ["target", "link"]},
     lambda r: {"target": rand_path(r), "link": rand_path(r)},
     lambda r, a: {"target": a["target"], "link": a["link"], "ok": True},
     ["Symlink {link} -> {target}.", "Create a symlink at {link} pointing to {target}."])

tool("find_large_files", "Find files larger than threshold.",
     {"type": "object", "properties": {"path": {"type": "string"}, "min_mb": {"type": "number"}}, "required": ["path"]},
     lambda r: {"path": rand_dir(r), "min_mb": r.choice([10, 50, 100, 500])},
     lambda r, a: {"path": a["path"], "found": r.randint(0, 12), "biggest_mb": r.randint(int(a["min_mb"]), int(a["min_mb"]) + 200)},
     ["Find files over {min_mb}MB in {path}.", "Any blobs bigger than {min_mb}MB under {path}?",
      "Scan {path} for files >{min_mb}MB."])

tool("du_disk_usage", "Report disk usage for a directory.",
     {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
     lambda r: {"path": rand_dir(r)},
     lambda r, a: {"path": a["path"], "size_mb": r.randint(1, 4096)},
     ["How big is {path}?", "du on {path}.", "Disk usage of {path}?"])

tool("df_free_space", "Report free disk space on a mount.",
     {"type": "object", "properties": {"mount": {"type": "string"}}, "required": []},
     lambda r: {"mount": r.choice(["/", "/Users", "/var"])},
     lambda r, a: {"mount": a["mount"], "free_gb": r.randint(5, 510), "used_pct": r.randint(20, 95)},
     ["How much free space on {mount}?", "df {mount} please.", "Check free disk on {mount}."])

# --- archive ---
tool("tar_extract", "Extract a tar archive.",
     {"type": "object", "properties": {"archive": {"type": "string"}, "dest": {"type": "string"}}, "required": ["archive"]},
     lambda r: {"archive": "release-v" + r.choice(["1.0", "2.1", "3.4"]) + ".tar.gz", "dest": rand_dir(r)},
     lambda r, a: {"archive": a["archive"], "extracted_files": r.randint(8, 412)},
     ["Extract {archive} into {dest}.", "Unpack {archive} to {dest}.", "Untar {archive}."])

tool("zip_create", "Create a zip archive.",
     {"type": "object", "properties": {"src": {"type": "string"}, "dest": {"type": "string"}}, "required": ["src", "dest"]},
     lambda r: {"src": rand_dir(r), "dest": "archive-" + rand_sha(r) + ".zip"},
     lambda r, a: {"src": a["src"], "dest": a["dest"], "size_kb": r.randint(8, 9012)},
     ["Zip {src} into {dest}.", "Pack {src} as {dest}.", "Create {dest} from {src}."])

tool("unzip", "Extract a zip archive.",
     {"type": "object", "properties": {"archive": {"type": "string"}, "dest": {"type": "string"}}, "required": ["archive"]},
     lambda r: {"archive": "data-" + rand_sha(r) + ".zip", "dest": rand_dir(r)},
     lambda r, a: {"archive": a["archive"], "extracted": r.randint(2, 198)},
     ["Unzip {archive} into {dest}.", "Expand {archive}.", "Extract {archive} to {dest}."])

tool("rsync", "Sync directories.",
     {"type": "object", "properties": {"src": {"type": "string"}, "dest": {"type": "string"}, "delete": {"type": "boolean"}}, "required": ["src", "dest"]},
     lambda r: {"src": rand_dir(r), "dest": "/backup/" + rand_sha(r) + "/", "delete": False},
     lambda r, a: {"src": a["src"], "dest": a["dest"], "files_transferred": r.randint(1, 514)},
     ["rsync {src} to {dest}.", "Sync {src} into {dest}.", "Mirror {src} -> {dest}."])

# --- git ---
tool("git_status", "Show working tree status.",
     {"type": "object", "properties": {}, "required": []},
     lambda r: {},
     lambda r, a: {"staged": r.randint(0, 8), "modified": r.randint(0, 12), "untracked": r.randint(0, 6)},
     ["Show me git status.", "What's the state of the working tree?", "What's modified right now?",
      "Run git status.", "What files changed since last commit?"])

tool("git_diff", "Show diff of unstaged changes.",
     {"type": "object", "properties": {"path": {"type": "string"}, "staged": {"type": "boolean"}}, "required": []},
     lambda r: {"path": rand_path(r), "staged": r.choice([True, False])},
     lambda r, a: {"path": a["path"], "additions": r.randint(0, 88), "deletions": r.randint(0, 60)},
     ["Show me the diff for {path}.", "Diff {path}.", "What changed in {path}?",
      "Run git diff on {path}.", "Show pending changes in {path}."])

tool("git_log", "Show commit history.",
     {"type": "object", "properties": {"n": {"type": "integer"}, "path": {"type": "string"}}, "required": []},
     lambda r: {"n": r.choice([5, 10, 20])},
     lambda r, a: {"count": a["n"], "head": rand_sha(r)},
     ["Show me the last {n} commits.", "git log -{n} please.", "What were the recent {n} commits?"])

tool("git_commit", "Create a commit with a message.",
     {"type": "object", "properties": {"message": {"type": "string"}, "all": {"type": "boolean"}}, "required": ["message"]},
     lambda r: {"message": r.choice(["fix: handle null user", "feat: add dark mode toggle",
                                      "chore: bump deps", "docs: update README",
                                      "refactor: extract auth helper"]), "all": r.choice([True, False])},
     lambda r, a: {"sha": rand_sha(r), "message": a["message"]},
     ["Commit with message: {message}", "Make a commit: '{message}'.", "Create a commit titled '{message}'."])

tool("git_push", "Push commits to a remote.",
     {"type": "object", "properties": {"remote": {"type": "string"}, "branch": {"type": "string"}}, "required": []},
     lambda r: {"remote": "origin", "branch": rand_branch(r)},
     lambda r, a: {"remote": a["remote"], "branch": a["branch"], "pushed": r.randint(1, 6)},
     ["Push {branch} to {remote}.", "Push the current branch.", "git push {remote} {branch}."])

tool("git_pull", "Pull from a remote branch.",
     {"type": "object", "properties": {"remote": {"type": "string"}, "branch": {"type": "string"}}, "required": []},
     lambda r: {"remote": "origin", "branch": rand_branch(r)},
     lambda r, a: {"remote": a["remote"], "branch": a["branch"], "fast_forward": r.choice([True, False])},
     ["Pull {branch} from {remote}.", "git pull {remote} {branch}.", "Pull latest from {branch}."])

tool("git_branch", "List branches.",
     {"type": "object", "properties": {"all": {"type": "boolean"}}, "required": []},
     lambda r: {"all": r.choice([True, False])},
     lambda r, a: {"branches": r.randint(3, 24), "current": rand_branch(r)},
     ["List all branches.", "Show me the branches.", "What branches exist?"])

tool("git_checkout", "Check out a branch or commit.",
     {"type": "object", "properties": {"ref": {"type": "string"}, "create": {"type": "boolean"}}, "required": ["ref"]},
     lambda r: {"ref": rand_branch(r), "create": r.choice([True, False])},
     lambda r, a: {"ref": a["ref"], "switched": True},
     ["Switch to {ref}.", "Check out {ref}.", "git checkout {ref}."])

tool("git_merge", "Merge a branch into current.",
     {"type": "object", "properties": {"branch": {"type": "string"}}, "required": ["branch"]},
     lambda r: {"branch": rand_branch(r)},
     lambda r, a: {"branch": a["branch"], "conflicts": r.choice([0, 0, 0, 1, 2])},
     ["Merge {branch} into current.", "Merge {branch}.", "Bring {branch} into HEAD."])

tool("git_stash", "Stash changes.",
     {"type": "object", "properties": {"message": {"type": "string"}, "include_untracked": {"type": "boolean"}}, "required": []},
     lambda r: {"message": r.choice(["wip-auth", "wip-styles", "debug-session"]), "include_untracked": True},
     lambda r, a: {"stash_ref": "stash@{0}", "message": a["message"]},
     ["Stash my changes as '{message}'.", "Stash everything (untracked too).", "Stash WIP labelled {message}."])

tool("git_reset", "Reset HEAD to a commit.",
     {"type": "object", "properties": {"ref": {"type": "string"}, "mode": {"type": "string"}}, "required": ["ref"]},
     lambda r: {"ref": "HEAD~" + str(r.randint(1, 3)), "mode": r.choice(["soft", "mixed", "hard"])},
     lambda r, a: {"ref": a["ref"], "mode": a["mode"], "ok": True},
     ["Reset to {ref} ({mode}).", "git reset --{mode} {ref}.", "Roll back to {ref}, {mode} mode."])

tool("git_blame", "Show blame for a file.",
     {"type": "object", "properties": {"path": {"type": "string"}, "line": {"type": "integer"}}, "required": ["path"]},
     lambda r: {"path": rand_path(r), "line": r.randint(1, 200)},
     lambda r, a: {"path": a["path"], "line": a["line"], "author": r.choice(["alice", "bob", "carol", "dan"]), "sha": rand_sha(r)},
     ["Who last touched line {line} of {path}?", "git blame {path}:{line}.", "Blame line {line} in {path}."])

tool("git_current_branch", "Get the current branch name.",
     {"type": "object", "properties": {}, "required": []},
     lambda r: {},
     lambda r, a: {"branch": rand_branch(r)},
     ["What branch am I on?", "Which branch is checked out?", "Tell me the current branch."])

tool("git_add", "Stage files.",
     {"type": "object", "properties": {"path": {"type": "string"}, "all": {"type": "boolean"}}, "required": ["path"]},
     lambda r: {"path": rand_dir(r), "all": True},
     lambda r, a: {"path": a["path"], "staged": r.randint(1, 14)},
     ["Stage everything in {path}.", "git add {path}.", "Stage all modified files under {path}."])

tool("git_remote_list", "List git remotes.",
     {"type": "object", "properties": {}, "required": []},
     lambda r: {},
     lambda r, a: {"remotes": ["origin", "upstream"][:r.randint(1, 2)]},
     ["List git remotes.", "What remotes are configured?", "Show the configured remotes."])

tool("git_tag_create", "Create a git tag.",
     {"type": "object", "properties": {"tag": {"type": "string"}, "message": {"type": "string"}}, "required": ["tag"]},
     lambda r: {"tag": "v" + ".".join(str(r.randint(0, 9)) for _ in range(3)), "message": "release"},
     lambda r, a: {"tag": a["tag"], "created": True},
     ["Tag this commit as {tag}.", "Create tag {tag}.", "git tag {tag} (annotated)."])

tool("git_cherry_pick", "Cherry-pick a commit.",
     {"type": "object", "properties": {"sha": {"type": "string"}}, "required": ["sha"]},
     lambda r: {"sha": rand_sha(r)},
     lambda r, a: {"sha": a["sha"], "applied": True},
     ["Cherry-pick {sha}.", "Apply commit {sha}.", "Pull in {sha} via cherry-pick."])

tool("git_rebase", "Rebase current branch onto another.",
     {"type": "object", "properties": {"onto": {"type": "string"}}, "required": ["onto"]},
     lambda r: {"onto": rand_branch(r)},
     lambda r, a: {"onto": a["onto"], "applied": r.randint(1, 8), "conflicts": r.choice([0, 0, 1])},
     ["Rebase onto {onto}.", "git rebase {onto}.", "Replay my commits on top of {onto}."])

# --- package managers ---
tool("npm_install", "Install npm dependencies.",
     {"type": "object", "properties": {"package": {"type": "string"}, "dev": {"type": "boolean"}}, "required": []},
     lambda r: {"package": rand_pkg_npm(r), "dev": r.choice([True, False])},
     lambda r, a: {"package": a["package"], "added": r.randint(1, 18), "duration_ms": r.randint(800, 12000)},
     ["Install {package} from npm.", "npm i {package}.", "Add {package} as a dependency."])

tool("npm_uninstall", "Remove an npm package.",
     {"type": "object", "properties": {"package": {"type": "string"}}, "required": ["package"]},
     lambda r: {"package": rand_pkg_npm(r)},
     lambda r, a: {"package": a["package"], "removed": True},
     ["Uninstall {package}.", "Remove {package} from package.json.", "Drop the {package} dependency."])

tool("npm_run", "Run an npm script.",
     {"type": "object", "properties": {"script": {"type": "string"}}, "required": ["script"]},
     lambda r: {"script": r.choice(["build", "test", "lint", "dev", "typecheck"])},
     lambda r, a: {"script": a["script"], "exit_code": 0, "duration_ms": r.randint(500, 90000)},
     ["Run npm run {script}.", "Execute the {script} script.", "Kick off `npm run {script}`."])

tool("pip_install", "Install Python packages with pip.",
     {"type": "object", "properties": {"package": {"type": "string"}, "version": {"type": "string"}}, "required": ["package"]},
     lambda r: {"package": rand_pkg_pip(r), "version": r.choice(["", "==2.31.0", ">=1.0", ""]).strip()},
     lambda r, a: {"package": a["package"], "installed": True, "version_resolved": "{}.{}.{}".format(r.randint(0,5), r.randint(0,30), r.randint(0,9))},
     ["Install {package} via pip.", "pip install {package}.", "Add {package} to the venv."])

tool("pip_uninstall", "Uninstall a Python package.",
     {"type": "object", "properties": {"package": {"type": "string"}}, "required": ["package"]},
     lambda r: {"package": rand_pkg_pip(r)},
     lambda r, a: {"package": a["package"], "removed": True},
     ["Uninstall {package} from pip.", "Remove the {package} package.", "Drop {package} from the venv."])

tool("pip_freeze", "Freeze installed pip packages.",
     {"type": "object", "properties": {}, "required": []},
     lambda r: {},
     lambda r, a: {"packages": r.randint(20, 184)},
     ["Run pip freeze.", "Show installed Python packages.", "Dump the pip freeze list."])

tool("cargo_build", "Build a Rust project with cargo.",
     {"type": "object", "properties": {"release": {"type": "boolean"}}, "required": []},
     lambda r: {"release": r.choice([True, False])},
     lambda r, a: {"release": a["release"], "warnings": r.randint(0, 5), "duration_s": r.randint(2, 184)},
     ["Build with cargo (release={release}).", "cargo build --release.", "Compile the Rust crate (release mode)."])

tool("cargo_test", "Run cargo tests.",
     {"type": "object", "properties": {"package": {"type": "string"}}, "required": []},
     lambda r: {"package": rand_pkg_cargo(r)},
     lambda r, a: {"package": a["package"], "passed": r.randint(10, 84), "failed": r.choice([0, 0, 1])},
     ["Run cargo test for {package}.", "cargo test on the {package} crate.", "Test the {package} crate."])

tool("cargo_add", "Add a cargo dependency.",
     {"type": "object", "properties": {"package": {"type": "string"}, "features": {"type": "string"}}, "required": ["package"]},
     lambda r: {"package": rand_pkg_cargo(r), "features": r.choice(["", "derive", "full"])},
     lambda r, a: {"package": a["package"], "added": True},
     ["Add the {package} crate.", "cargo add {package}.", "Pull in {package} as a dependency."])

tool("brew_install", "Install a Homebrew package.",
     {"type": "object", "properties": {"package": {"type": "string"}}, "required": ["package"]},
     lambda r: {"package": rand_pkg_brew(r)},
     lambda r, a: {"package": a["package"], "installed": True, "version": "{}.{}.{}".format(r.randint(0,30), r.randint(0,30), r.randint(0,9))},
     ["brew install {package}.", "Install {package} via Homebrew.", "Get {package} from brew."])

tool("brew_upgrade", "Upgrade Homebrew packages.",
     {"type": "object", "properties": {"package": {"type": "string"}}, "required": []},
     lambda r: {"package": rand_pkg_brew(r)},
     lambda r, a: {"package": a["package"], "upgraded": True, "from": "1.4.2", "to": "1.5.0"},
     ["Upgrade {package} via brew.", "brew upgrade {package}.", "Bump {package} to the latest."])

tool("apt_install", "Install a Debian/Ubuntu package.",
     {"type": "object", "properties": {"package": {"type": "string"}}, "required": ["package"]},
     lambda r: {"package": r.choice(["build-essential", "curl", "wget", "git", "vim", "htop"])},
     lambda r, a: {"package": a["package"], "installed": True},
     ["apt install {package}.", "Install {package} on the box.", "Get {package} via apt."])

# --- build / shell / misc ---
tool("run_shell", "Run a shell command.",
     {"type": "object", "properties": {"cmd": {"type": "string"}, "cwd": {"type": "string"}}, "required": ["cmd"]},
     lambda r: {"cmd": r.choice(["ls -la", "echo $PATH", "uname -a", "ps aux | head", "whoami"]), "cwd": rand_dir(r)},
     lambda r, a: {"cmd": a["cmd"], "exit_code": 0, "stdout_bytes": r.randint(20, 2048)},
     ["Run `{cmd}` in {cwd}.", "Execute `{cmd}` from {cwd}.", "Drop into {cwd} and run `{cmd}`."])

tool("which_binary", "Locate a binary on PATH.",
     {"type": "object", "properties": {"binary": {"type": "string"}}, "required": ["binary"]},
     lambda r: {"binary": r.choice(["python", "node", "rustc", "go", "kubectl", "docker"])},
     lambda r, a: {"binary": a["binary"], "path": "/usr/local/bin/" + a["binary"]},
     ["Where is {binary} installed?", "which {binary}?", "Find the {binary} binary on PATH."])

tool("env_get", "Read an environment variable.",
     {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
     lambda r: {"name": r.choice(["PATH", "HOME", "NODE_ENV", "PYTHONPATH", "GOPATH", "EDITOR"])},
     lambda r, a: {"name": a["name"], "value": r.choice(["/usr/local/bin:/usr/bin", "/Users/me", "production", "development"])},
     ["What's the value of ${name}?", "Read the {name} env var.", "Show me ${name}."])

tool("env_set", "Set an environment variable in the current shell scope.",
     {"type": "object", "properties": {"name": {"type": "string"}, "value": {"type": "string"}}, "required": ["name", "value"]},
     lambda r: {"name": r.choice(["NODE_ENV", "DEBUG", "LOG_LEVEL"]), "value": r.choice(["production", "true", "info"])},
     lambda r, a: {"name": a["name"], "value": a["value"], "ok": True},
     ["Set {name}={value}.", "Export {name}={value}.", "Configure env {name} to {value}."])

tool("make_target", "Run a Makefile target.",
     {"type": "object", "properties": {"target": {"type": "string"}}, "required": ["target"]},
     lambda r: {"target": r.choice(["build", "test", "clean", "install", "lint"])},
     lambda r, a: {"target": a["target"], "exit_code": 0, "duration_s": r.randint(1, 184)},
     ["Run make {target}.", "Execute the {target} make target.", "Kick off `make {target}`."])

tool("docker_build", "Build a Docker image.",
     {"type": "object", "properties": {"tag": {"type": "string"}, "context": {"type": "string"}}, "required": ["tag"]},
     lambda r: {"tag": "myapp:" + r.choice(["latest", "v1.0", "dev"]), "context": "."},
     lambda r, a: {"tag": a["tag"], "image_id": "sha256:" + rand_sha(r) + rand_sha(r), "size_mb": r.randint(40, 980)},
     ["Build a Docker image tagged {tag}.", "docker build -t {tag} {context}.", "Build the image as {tag}."])

tool("port_check", "Check whether a TCP port is in use.",
     {"type": "object", "properties": {"port": {"type": "integer"}}, "required": ["port"]},
     lambda r: {"port": r.choice([3000, 5173, 8000, 8080, 5432, 6379])},
     lambda r, a: {"port": a["port"], "in_use": r.choice([True, False]), "pid": r.randint(1000, 99999)},
     ["Is port {port} in use?", "Check whether {port} is occupied.", "Who is on port {port}?"])

tool("kill_process", "Kill a process by PID.",
     {"type": "object", "properties": {"pid": {"type": "integer"}, "signal": {"type": "string"}}, "required": ["pid"]},
     lambda r: {"pid": r.randint(1000, 99999), "signal": r.choice(["TERM", "KILL", "HUP"])},
     lambda r, a: {"pid": a["pid"], "signal": a["signal"], "ok": True},
     ["Kill PID {pid} with SIG{signal}.", "Send SIG{signal} to {pid}.", "Terminate process {pid}."])

tool("ps_list", "List running processes.",
     {"type": "object", "properties": {"filter": {"type": "string"}}, "required": []},
     lambda r: {"filter": r.choice(["node", "python", "ruby", ""])},
     lambda r, a: {"filter": a["filter"] or "*", "count": r.randint(1, 84)},
     ["List processes matching {filter}.", "ps aux | grep {filter}.", "Show running {filter} processes."])


assert len(TOOLS) >= 40, f"only {len(TOOLS)} tools defined"


# ----------------------------- GENERATION -----------------------------

def make_tools_field(t):
    return [{
        "type": "function",
        "function": {
            "name": t["name"],
            "description": t["desc"],
            "parameters": t["params"],
        },
    }]


def render_user(phrasing: str, args: dict) -> str:
    # Format placeholders that appear in phrasing using args
    # only fill ones present in args
    out = phrasing
    for k, v in args.items():
        out = out.replace("{" + k + "}", str(v))
    return out


def short_summary(tool_name: str, result: dict, r: random.Random) -> str:
    """Build a short factual tail after the suffix-pool opener."""
    if "branch" in result:
        return f" Current branch is `{result['branch']}`."
    if "sha" in result and tool_name in ("git_commit",):
        return f" Commit {result['sha']} created."
    if tool_name == "git_status":
        return f" {result['staged']} staged, {result['modified']} modified, {result['untracked']} untracked."
    if tool_name == "git_log":
        return f" Showing {result['count']} commits, HEAD at {result['head']}."
    if tool_name == "list_dir":
        return f" Found {result['entries']} entries under {result['path']}."
    if tool_name == "find_files":
        return f" {result['matches']} matches for `{result['pattern']}`."
    if tool_name == "grep_files":
        return f" {result['total_matches']} matches across {result['files_with_matches']} files."
    if tool_name == "wc_lines":
        return f" {result['lines']} lines."
    if tool_name == "du_disk_usage":
        return f" {result['size_mb']} MB used."
    if tool_name == "df_free_space":
        return f" {result['free_gb']} GB free ({result['used_pct']}% used)."
    if tool_name == "stat_file":
        return f" {result['size']} bytes, mtime {result['mtime']}."
    if tool_name == "tail_file" or tool_name == "head_file":
        return f" Returned {result['lines_returned']} lines from {result['path']}."
    if tool_name == "find_large_files":
        return f" {result['found']} hits, biggest {result['biggest_mb']} MB."
    if tool_name == "which_binary":
        return f" It's at {result['path']}."
    if tool_name == "env_get":
        return f" {result['name']}={result['value']}."
    if tool_name == "port_check":
        return f" Port {result['port']} is {'in use by PID ' + str(result['pid']) if result['in_use'] else 'free'}."
    if tool_name == "ps_list":
        return f" {result['count']} processes match."
    if tool_name == "git_branch":
        return f" {result['branches']} branches; current `{result['current']}`."
    if tool_name == "git_remote_list":
        return f" Remotes: {', '.join(result['remotes'])}."
    if tool_name == "git_blame":
        return f" {result['author']} (commit {result['sha']})."
    if tool_name == "git_diff":
        return f" +{result['additions']} / -{result['deletions']} in {result['path']}."
    if tool_name in ("npm_install", "pip_install", "brew_install", "apt_install", "cargo_add"):
        return f" {result.get('package','package')} installed."
    if tool_name == "npm_run":
        return f" `{result['script']}` exited {result['exit_code']} after {result['duration_ms']}ms."
    if tool_name == "make_target":
        return f" make {result['target']} succeeded in {result['duration_s']}s."
    if tool_name == "docker_build":
        return f" Built {result['tag']} ({result['size_mb']} MB)."
    if tool_name == "cargo_build":
        return f" Build done with {result['warnings']} warnings in {result['duration_s']}s."
    if tool_name == "cargo_test":
        return f" {result['passed']} passed, {result['failed']} failed."
    if tool_name == "pip_freeze":
        return f" {result['packages']} packages installed."
    if tool_name == "git_merge":
        return f" Merged {result['branch']} ({result['conflicts']} conflicts)."
    if tool_name == "git_rebase":
        return f" Replayed {result['applied']} commits onto {result['onto']}, {result['conflicts']} conflicts."
    if tool_name in ("git_push", "git_pull"):
        return f" {tool_name.split('_')[1].capitalize()} on {result['branch']} OK."
    if tool_name == "rsync":
        return f" Transferred {result['files_transferred']} files."
    if tool_name == "tar_extract":
        return f" Extracted {result['extracted_files']} files."
    if tool_name == "unzip":
        return f" Extracted {result['extracted']} entries."
    if tool_name == "zip_create":
        return f" Wrote {result['dest']} ({result['size_kb']} KB)."
    if tool_name == "run_shell":
        return f" Exit {result['exit_code']} ({result['stdout_bytes']} bytes stdout)."
    return " Done."


def make_sample(r: random.Random, tool_def: dict, single_turn: bool, suffix: str | None) -> dict:
    args = tool_def["args_fn"](r)
    phrasing = r.choice(tool_def["phrasings"])
    user_text = render_user(phrasing, args)
    sys_text = r.choice(SYSTEM_PROMPTS)

    msgs = [
        {"role": "system", "content": sys_text},
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "function": {"name": tool_def["name"], "arguments": args}}
        ]},
    ]

    if not single_turn:
        result = tool_def["result_fn"](r, args)
        result_str = json.dumps(result)
        # ensure within 10-200 chars
        if len(result_str) > 200:
            result_str = result_str[:197] + '"}'
        msgs.append({"role": "tool", "name": tool_def["name"], "content": result_str})
        tail = short_summary(tool_def["name"], result, r)
        msgs.append({"role": "assistant", "content": suffix + tail})

    return {
        "messages": msgs,
        "tools": make_tools_field(tool_def),
        "domain": "tool-calling",
    }


def main():
    r = rng_for(SEED)
    OUT.parent.mkdir(parents=True, exist_ok=True)

    TOTAL = 500
    SINGLE_TARGET = 75  # ~15%
    MULTI_TARGET = TOTAL - SINGLE_TARGET  # 425

    # Tool quota: ≤25 each. With 60 tools and 500 samples avg ~8.3.
    # Distribute approximately evenly with slight randomization, capped at 25.
    n_tools = len(TOOLS)
    # Build a tool-count assignment summing to TOTAL with each ≤25 and ≥3.
    base = [TOTAL // n_tools] * n_tools  # floor each ~8
    remaining = TOTAL - sum(base)
    # randomly distribute remainder
    idxs = list(range(n_tools))
    r.shuffle(idxs)
    for i in idxs[:remaining]:
        base[i] += 1
    # clamp
    for i in range(n_tools):
        if base[i] > 25:
            overflow = base[i] - 25
            base[i] = 25
            # redistribute overflow to under-quota tools
            for j in idxs:
                if j == i: continue
                if base[j] < 25:
                    take = min(overflow, 25 - base[j])
                    base[j] += take
                    overflow -= take
                    if overflow == 0: break
    assert sum(base) == TOTAL, sum(base)
    assert all(c <= 25 for c in base)

    # Build flat list of tool indices according to quota
    tool_assignment = []
    for i, c in enumerate(base):
        tool_assignment.extend([i] * c)
    r.shuffle(tool_assignment)

    # Decide single vs multi
    single_flags = [True] * SINGLE_TARGET + [False] * MULTI_TARGET
    r.shuffle(single_flags)

    # Multi-turn suffix assignment: 425 multi-turn samples, 30 phrases.
    # 425 / 30 = 14.16... -> some 14, some 15. We aim ~14 each (mention ~14 uses per phrase).
    # Distribute: 30*14 = 420; remaining 5 get +1.
    suffix_counts = [14] * len(SUFFIX_POOL)
    extras = MULTI_TARGET - sum(suffix_counts)  # 5
    extra_idxs = list(range(len(SUFFIX_POOL)))
    r.shuffle(extra_idxs)
    for i in extra_idxs[:extras]:
        suffix_counts[i] += 1
    suffix_pool_flat = []
    for i, c in enumerate(suffix_counts):
        suffix_pool_flat.extend([SUFFIX_POOL[i]] * c)
    r.shuffle(suffix_pool_flat)
    assert len(suffix_pool_flat) == MULTI_TARGET

    suffix_iter = iter(suffix_pool_flat)
    samples = []
    seen_user = set()

    BLACKLIST = (
        "I've gathered all the information",
        "I've completed the task",
        "Here's what I found:",
        "Based on the results,",
        "The results show that",
    )

    for tool_idx, single in zip(tool_assignment, single_flags):
        tool_def = TOOLS[tool_idx]
        attempts = 0
        while True:
            attempts += 1
            suffix = None if single else next(suffix_iter)
            sample = make_sample(r, tool_def, single, suffix)
            user_text = sample["messages"][1]["content"]
            # uniqueness (per-tool to avoid global collision pressure)
            key = (tool_def["name"], user_text)
            if key in seen_user and attempts < 8:
                # if not single-turn we already consumed a suffix; put it back
                if not single:
                    # rebuild iter: prepend suffix back
                    suffix_iter = iter([suffix] + list(suffix_iter))
                continue
            # blacklist check on final assistant
            if not single:
                final = sample["messages"][-1]["content"]
                if any(final.startswith(b) for b in BLACKLIST):
                    if not single:
                        suffix_iter = iter([suffix] + list(suffix_iter))
                    continue
            seen_user.add(key)
            samples.append(sample)
            break

    assert len(samples) == TOTAL

    with OUT.open("w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Stats
    from collections import Counter
    tool_counts = Counter()
    suffix_counts_actual = Counter()
    single_n = 0
    for s in samples:
        tc = s["messages"][2]["tool_calls"][0]["function"]["name"]
        tool_counts[tc] += 1
        if len(s["messages"]) == 3:
            single_n += 1
        else:
            final = s["messages"][-1]["content"]
            for ph in SUFFIX_POOL:
                if final.startswith(ph):
                    suffix_counts_actual[ph] += 1
                    break

    print(f"lines: {len(samples)}")
    print(f"distinct tools: {len(tool_counts)}")
    print(f"max tool count: {max(tool_counts.values())} ({max(tool_counts, key=tool_counts.get)})")
    print(f"single-turn: {single_n}")
    print(f"multi-turn: {len(samples) - single_n}")
    print(f"suffix pool coverage: {len(suffix_counts_actual)}/{len(SUFFIX_POOL)}")
    print(f"suffix min/max uses: {min(suffix_counts_actual.values())}/{max(suffix_counts_actual.values())}")


if __name__ == "__main__":
    main()
