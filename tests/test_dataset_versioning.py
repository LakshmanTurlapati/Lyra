#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Smoke tests for dataset versioning scheme (Phase 10 REL-07 / D-08)."""
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
DATASET_TAG_PATTERN = re.compile(r"^dataset-v\d+\.\d+\.\d+$")
MODEL_TAG_PATTERN = re.compile(r"^model-v\d+\.\d+\.\d+$")


def test_dataset_tag_format_accepts_valid():
    for tag in ["dataset-v1.0.0", "dataset-v1.2.10", "dataset-v10.0.0"]:
        assert DATASET_TAG_PATTERN.match(tag), f"{tag} should be valid"


def test_dataset_tag_format_rejects_invalid():
    for tag in ["v1.0.0", "dataset-1.0.0", "dataset-v1.0", "data-v1.0.0", "dataset-v1.0.0-rc1"]:
        assert not DATASET_TAG_PATTERN.match(tag), f"{tag} should be rejected"


def test_model_tag_format_accepts_valid():
    for tag in ["model-v1.0.0", "model-v2.1.3"]:
        assert MODEL_TAG_PATTERN.match(tag), f"{tag} should be valid"


def test_changelog_has_unreleased_and_v100():
    """CHANGELOG.md scaffold has [Unreleased], Dataset v1.0.0, Model v1.0.0 sections."""
    text = (REPO_ROOT / "CHANGELOG.md").read_text()
    assert "## [Unreleased]" in text
    assert "## [Dataset v1.0.0]" in text
    assert "## [Model v1.0.0]" in text


@pytest.mark.xfail(reason="Wave 0 RED stub -- GREEN in Plan 03 (dataset v1.0.0 release)")
def test_dataset_v100_release_exists():
    """gh release view dataset-v1.0.0 succeeds (Plan 03 cuts this release)."""
    import subprocess
    result = subprocess.run(
        ["gh", "release", "view", "dataset-v1.0.0", "--json", "name"],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert result.returncode == 0, f"gh release view failed: {result.stderr}"
