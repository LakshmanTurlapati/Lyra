#!/usr/bin/env python3
"""Smoke tests verifying release artifact presence -- REL-01 through REL-04."""
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).parent.parent


def test_license_file():
    """LICENSE file exists at repo root with MIT text (REL-04)."""
    license_path = REPO_ROOT / "LICENSE"
    assert license_path.exists(), "LICENSE file not found at repo root"
    text = license_path.read_text()
    assert "MIT License" in text, "LICENSE does not contain 'MIT License'"
    assert "Lakshman Turlapati" in text, "LICENSE missing copyright holder 'Lakshman Turlapati'"


def test_gitattributes_lfs():
    """gitattributes tracks *.safetensors with git-lfs filter (REL-03 / D-10)."""
    ga_path = REPO_ROOT / ".gitattributes"
    assert ga_path.exists(), ".gitattributes not found at repo root"
    text = ga_path.read_text()
    assert "safetensors" in text, ".gitattributes does not mention safetensors"
    assert "filter=lfs" in text, ".gitattributes does not set filter=lfs"


def test_model_card_frontmatter():
    """README.md contains YAML frontmatter with license: mit (REL-02 / D-08)."""
    readme_path = REPO_ROOT / "README.md"
    assert readme_path.exists(), "README.md not found"
    text = readme_path.read_text()
    assert text.startswith("---"), "README.md does not start with YAML frontmatter (---)"
    assert "license: mit" in text, "README.md frontmatter missing 'license: mit'"


def test_dataset_card():
    """datasets/README.md exists and contains required sections (REL-01 / D-08)."""
    card_path = REPO_ROOT / "datasets" / "README.md"
    assert card_path.exists(), "datasets/README.md not found"
    text = card_path.read_text()
    assert "license: mit" in text, "datasets/README.md frontmatter missing 'license: mit'"
    assert "## Dataset Description" in text or "## Description" in text, \
        "datasets/README.md missing description section"
    assert "## Statistics" in text or "## Dataset Statistics" in text, \
        "datasets/README.md missing statistics section"
    assert "## Limitations" in text, "datasets/README.md missing limitations section"
    assert "## Dataset Versions" in text or "### Dataset Versions" in text, \
        "datasets/README.md missing Dataset Versions section (REL-07)"


def test_gitattributes_gguf_lfs():
    """gitattributes tracks *.gguf under git-lfs (REL-05 / Phase 10 D-05)."""
    text = (REPO_ROOT / ".gitattributes").read_text()
    assert "*.gguf" in text, ".gitattributes does not track *.gguf"
    gguf_lines = [l for l in text.splitlines() if "*.gguf" in l]
    assert any("filter=lfs" in l for l in gguf_lines), "*.gguf line does not set filter=lfs"


def test_gitattributes_arrow_lfs():
    """gitattributes tracks *.arrow under git-lfs (REL-07 / Phase 10 D-10)."""
    text = (REPO_ROOT / ".gitattributes").read_text()
    assert "*.arrow" in text, ".gitattributes does not track *.arrow"


def test_changelog_exists_at_root():
    """CHANGELOG.md exists at repo root with Keep-a-Changelog header (REL-07 / D-10)."""
    path = REPO_ROOT / "CHANGELOG.md"
    assert path.exists(), "CHANGELOG.md not found"
    text = path.read_text()
    assert "Keep a Changelog" in text
    assert "Semantic Versioning" in text
    assert "## [Unreleased]" in text
