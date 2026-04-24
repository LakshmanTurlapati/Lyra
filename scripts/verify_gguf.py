#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""verify_gguf.py -- Verify GGUF metadata meets Lyra release requirements (Phase 10 D-07).

Asserts that:
  1. `tokenizer.chat_template` key is present in GGUF metadata
  2. Template content contains the SmolLM2 `<|im_start|>` marker
  3. `general.architecture` and `general.name` keys are populated

Closes the same class of runtime template-drift bug that caused the Phase 09.1
D-03/D-04 regressions.

Usage:
  python3 -m scripts.verify_gguf build/gguf/lyra-v1.0-q4_k_m.gguf

Threat mitigations:
  T-10-02: gguf.GGUFReader validates GGUF file structure; we do NOT eval metadata content.
  T-10-01: input path validated against _GGUF_PATH_PATTERN before open.
"""
import argparse
import logging
import re
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_GGUF_PATH_PATTERN = re.compile(r"^[a-zA-Z0-9._/~\-]+$")
REQUIRED_KEYS = ["tokenizer.chat_template", "general.architecture", "general.name"]
SMOLLM2_MARKER = "<|im_start|>"


def _load_gguf_reader(gguf_path: str):
    """Lazy-import gguf so tests can monkeypatch via sys.modules without the package installed."""
    from gguf import GGUFReader
    return GGUFReader(gguf_path, mode="r")


def verify(gguf_path: Path) -> int:
    """Verify GGUF metadata has required keys and SmolLM2 chat template marker.

    Args:
        gguf_path: Path to GGUF file.

    Returns:
        0 if all required keys present and SmolLM2 marker found, 1 otherwise.
    """
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify GGUF metadata for Lyra release.")
    parser.add_argument("gguf_path", type=Path, help="Path to GGUF file to verify.")
    args = parser.parse_args()
    if not _GGUF_PATH_PATTERN.match(str(args.gguf_path)):
        print(f"Error: invalid path characters: {args.gguf_path}", file=sys.stderr)
        return 1
    if not args.gguf_path.exists():
        print(f"Error: file not found: {args.gguf_path}", file=sys.stderr)
        return 1
    try:
        return verify(args.gguf_path)
    except Exception as exc:
        print(f"Error verifying GGUF: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    sys.exit(main())
