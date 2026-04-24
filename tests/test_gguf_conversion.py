#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Unit tests for scripts/convert_gguf.sh + scripts/verify_gguf.py (Phase 10 REL-05)."""
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).parent.parent


def _mock_gguf_module(chat_template: str, include_keys=("tokenizer.chat_template", "general.architecture", "general.name")):
    mock_field = MagicMock()
    mock_field.contents.return_value = chat_template
    mock_reader = MagicMock()
    mock_reader.fields = {k: MagicMock() for k in include_keys}
    if "tokenizer.chat_template" in include_keys:
        mock_reader.fields["tokenizer.chat_template"] = mock_field
    mock_gguf = MagicMock()
    mock_gguf.GGUFReader.return_value = mock_reader
    return mock_gguf


def test_verify_gguf_accepts_good_metadata(tmp_path):
    """verify_gguf.py exits 0 when all required keys present + SmolLM2 marker found."""
    gguf_path = tmp_path / "fake.gguf"
    gguf_path.write_bytes(b"")
    mock_gguf = _mock_gguf_module("{% for m in messages %}<|im_start|>{{m.role}}")
    with patch.dict("sys.modules", {"gguf": mock_gguf}):
        from scripts.verify_gguf import verify
        assert verify(gguf_path) == 0


def test_verify_gguf_rejects_missing_template(tmp_path):
    """verify_gguf.py exits 1 when tokenizer.chat_template key is absent."""
    gguf_path = tmp_path / "fake.gguf"
    gguf_path.write_bytes(b"")
    mock_gguf = _mock_gguf_module("", include_keys=("general.architecture", "general.name"))
    with patch.dict("sys.modules", {"gguf": mock_gguf}):
        from scripts.verify_gguf import verify
        assert verify(gguf_path) == 1


def test_verify_gguf_rejects_wrong_marker(tmp_path):
    """verify_gguf.py exits 1 when chat_template present but missing <|im_start|> marker."""
    gguf_path = tmp_path / "fake.gguf"
    gguf_path.write_bytes(b"")
    mock_gguf = _mock_gguf_module("<|user|>...")
    with patch.dict("sys.modules", {"gguf": mock_gguf}):
        from scripts.verify_gguf import verify
        assert verify(gguf_path) == 1


def test_convert_gguf_rejects_bad_path():
    """convert_gguf.sh exits non-zero on invalid model dir arg (T-10-01 mitigation)."""
    script = REPO_ROOT / "scripts" / "convert_gguf.sh"
    result = subprocess.run(
        [str(script), "nonexistent_dir_!!"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode != 0
    out = (result.stdout + result.stderr).lower()
    assert "invalid" in out or "not found" in out


@pytest.mark.xfail(reason="Wave 0 RED stub -- GREEN in Plan 02 (actual GGUF produced by llama.cpp)")
def test_produced_gguf_has_chat_template():
    """Real GGUF produced by Plan 02 has tokenizer.chat_template metadata (REL-05, D-07)."""
    gguf = REPO_ROOT / "build" / "gguf" / "lyra-v1.0-q4_k_m.gguf"
    assert gguf.exists(), "Plan 02 must produce the Q4_K_M GGUF"
    from gguf import GGUFReader
    reader = GGUFReader(str(gguf), mode="r")
    assert "tokenizer.chat_template" in reader.fields
    assert "<|im_start|>" in reader.fields["tokenizer.chat_template"].contents()
