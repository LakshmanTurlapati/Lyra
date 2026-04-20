#!/usr/bin/env python3
"""Tests for pipeline configuration loading and Pydantic validation."""
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from scripts.pipeline_config import (
    DomainConfig,
    PipelineConfig,
    StyleConfig,
    load_config,
)


# --- Config loading tests ---

class TestConfigLoading:
    """Test loading configs/pipeline.yaml."""

    def test_config_loads_valid_yaml(self):
        """Loading configs/pipeline.yaml produces valid PipelineConfig."""
        config_path = Path(__file__).parent.parent / "configs" / "pipeline.yaml"
        config = load_config(config_path)
        assert isinstance(config, PipelineConfig)
        assert config.version == 1
        assert config.dedup_threshold == 0.7
        assert config.ngram_size == 3

    def test_config_has_domains(self):
        """Config has tool-calling, code, and knowledge domain sections."""
        config_path = Path(__file__).parent.parent / "configs" / "pipeline.yaml"
        config = load_config(config_path)
        assert "tool-calling" in config.domains
        assert "code" in config.domains
        assert "knowledge" in config.domains

    def test_config_has_topic_distribution(self):
        """Config has topic_distribution section."""
        config_path = Path(__file__).parent.parent / "configs" / "pipeline.yaml"
        config = load_config(config_path)
        assert "tool-calling" in config.topic_distribution
        assert "code" in config.topic_distribution
        assert "knowledge" in config.topic_distribution


# --- Domain override tests ---

class TestDomainOverrides:
    """Test per-domain config merging."""

    def test_config_domain_override(self):
        """Code domain config merges global defaults with code-specific overrides."""
        config = PipelineConfig(
            defaults=DomainConfig(min_response_chars=10, max_turn_ratio=50.0),
            domains={
                "code": DomainConfig(
                    min_response_chars=20,
                    style=StyleConfig(max_tokens=600, require_code_blocks=True),
                ),
            },
        )
        domain_cfg = config.get_domain_config("code")
        # Overridden value
        assert domain_cfg.min_response_chars == 20
        # Style from code domain
        assert domain_cfg.style.max_tokens == 600
        assert domain_cfg.style.require_code_blocks is True
        # Inherited default
        assert domain_cfg.max_turn_ratio == 50.0

    def test_config_get_domain_config_unknown_domain(self):
        """Unknown domain falls back to global defaults."""
        config = PipelineConfig(
            defaults=DomainConfig(min_response_chars=10),
        )
        domain_cfg = config.get_domain_config("nonexistent")
        assert domain_cfg.min_response_chars == 10
        assert domain_cfg.style == StyleConfig()


# --- Validation tests ---

class TestConfigValidation:
    """Test Pydantic validation rejects invalid configs."""

    def test_config_validation_rejects_invalid_dedup_scope(self):
        """Invalid dedup_scope value raises ValidationError."""
        with pytest.raises(ValidationError):
            PipelineConfig(dedup_scope="invalid_scope")

    def test_config_validation_rejects_threshold_out_of_range(self):
        """dedup_threshold outside [0.0, 1.0] raises ValidationError."""
        with pytest.raises(ValidationError):
            PipelineConfig(dedup_threshold=1.5)

    def test_config_validation_rejects_ngram_size_out_of_range(self):
        """ngram_size outside [2, 5] raises ValidationError."""
        with pytest.raises(ValidationError):
            PipelineConfig(ngram_size=1)

    def test_config_valid_defaults(self):
        """Default PipelineConfig is valid."""
        config = PipelineConfig()
        assert config.version == 1
        assert config.dedup_scope == "response"
        assert config.style_validation is True
