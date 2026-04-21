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
