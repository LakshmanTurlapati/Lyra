"""Unit tests for prompt template library.

Validates structure, content, and cross-references across all 4 YAML template files.
Covers DATA-06 requirement: prompt template library organized by three categories.
"""

import pytest
import yaml
from pathlib import Path


TEMPLATES_DIR = Path("templates")

EXPECTED_CATEGORIES = {
    "tool-calling": ["single_call", "multi_turn", "parallel_calls", "mcp_patterns", "cli_commands"],
    "code": ["utility_functions", "file_operations", "debugging"],
    "knowledge": ["reasoning_chains", "factual_qa", "explanations"],
}

EXPECTED_DOMAINS = {
    "tool-calling": "tool-calling",
    "code": "code",
    "knowledge": "knowledge",
}

TEMPLATE_FILES = ["tool-calling", "code", "knowledge", "system-prompts"]


@pytest.fixture
def all_templates():
    """Load all 4 YAML template files, return dict keyed by filename stem."""
    templates = {}
    for name in TEMPLATE_FILES:
        filepath = TEMPLATES_DIR / f"{name}.yaml"
        with open(filepath) as f:
            templates[name] = yaml.safe_load(f)
    return templates


@pytest.fixture
def system_prompts():
    """Load system-prompts.yaml and return the parsed dict."""
    filepath = TEMPLATES_DIR / "system-prompts.yaml"
    with open(filepath) as f:
        return yaml.safe_load(f)


def test_all_templates_parseable():
    """All 4 YAML template files exist and parse without errors."""
    for name in TEMPLATE_FILES:
        filepath = TEMPLATES_DIR / f"{name}.yaml"
        assert filepath.exists(), f"Template file {filepath} does not exist"
        with open(filepath) as f:
            data = yaml.safe_load(f)
        assert data is not None, f"Template {name}.yaml parsed as None"
        assert isinstance(data, dict), f"Template {name}.yaml is not a dict"


def test_tool_calling_has_all_categories(all_templates):
    """tool-calling.yaml has exactly 5 categories matching all tool call patterns."""
    data = all_templates["tool-calling"]
    assert "categories" in data, "tool-calling.yaml missing 'categories' key"
    assert set(data["categories"].keys()) == set(EXPECTED_CATEGORIES["tool-calling"]), (
        f"Expected categories {EXPECTED_CATEGORIES['tool-calling']}, "
        f"got {list(data['categories'].keys())}"
    )


def test_code_has_all_categories(all_templates):
    """code.yaml has exactly 3 categories for code generation tasks."""
    data = all_templates["code"]
    assert "categories" in data, "code.yaml missing 'categories' key"
    assert set(data["categories"].keys()) == set(EXPECTED_CATEGORIES["code"]), (
        f"Expected categories {EXPECTED_CATEGORIES['code']}, "
        f"got {list(data['categories'].keys())}"
    )


def test_knowledge_has_all_categories(all_templates):
    """knowledge.yaml has exactly 3 categories for knowledge tasks."""
    data = all_templates["knowledge"]
    assert "categories" in data, "knowledge.yaml missing 'categories' key"
    assert set(data["categories"].keys()) == set(EXPECTED_CATEGORIES["knowledge"]), (
        f"Expected categories {EXPECTED_CATEGORIES['knowledge']}, "
        f"got {list(data['categories'].keys())}"
    )


def test_system_prompts_has_minimum_prompts(system_prompts):
    """system-prompts.yaml has at least 5 prompts, each with non-empty content."""
    prompts = system_prompts["system_prompts"]
    assert len(prompts) >= 5, f"Expected at least 5 system prompts, got {len(prompts)}"
    for prompt_id, prompt_data in prompts.items():
        assert "content" in prompt_data, f"Prompt {prompt_id} missing content"
        assert len(prompt_data["content"].strip()) > 0, f"Prompt {prompt_id} has empty content"


def test_categories_have_required_fields(all_templates):
    """Every category across all domains has description and system_prompt_ref fields."""
    for domain in EXPECTED_CATEGORIES:
        data = all_templates[domain]
        for cat_name, cat_data in data["categories"].items():
            assert "description" in cat_data, f"{domain}/{cat_name} missing description"
            assert "system_prompt_ref" in cat_data, f"{domain}/{cat_name} missing system_prompt_ref"


def test_system_prompt_refs_are_valid(all_templates, system_prompts):
    """Every system_prompt_ref in domain templates references a valid ID in system-prompts.yaml."""
    valid_ids = set(system_prompts["system_prompts"].keys())
    for domain in EXPECTED_CATEGORIES:
        data = all_templates[domain]
        for cat_name, cat_data in data["categories"].items():
            ref = cat_data["system_prompt_ref"]
            assert ref in valid_ids, (
                f"{domain}/{cat_name} references unknown system prompt '{ref}'. "
                f"Valid: {valid_ids}"
            )


def test_domain_fields_match(all_templates):
    """Each domain template has a top-level 'domain' field matching expected value."""
    for domain, expected_domain in EXPECTED_DOMAINS.items():
        data = all_templates[domain]
        assert "domain" in data, f"Template {domain}.yaml missing 'domain' field"
        assert data["domain"] == expected_domain, (
            f"Expected domain '{expected_domain}', got '{data['domain']}'"
        )
