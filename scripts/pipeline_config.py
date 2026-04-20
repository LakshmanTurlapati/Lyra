#!/usr/bin/env python3
"""pipeline_config.py -- Pydantic models and loader for pipeline.yaml.

Provides typed configuration for the curation pipeline with per-domain
overrides.  Global defaults are merged with domain-specific sections
so each module receives a flat config dict it can query with .get().

Separated from curate_pipeline.py to avoid circular imports -- the
style_validator and test modules both need config types without pulling
in the full pipeline orchestrator.
"""
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


class StyleConfig(BaseModel):
    """Style validation thresholds for a specific domain."""

    max_tokens: Optional[int] = None
    min_tokens: Optional[int] = None
    require_code_blocks: bool = False
    require_reasoning_markers: bool = False
    max_prose_ratio: Optional[float] = None
    min_prose_ratio: Optional[float] = None
    require_tool_calls: bool = False
    allow_terse: bool = False


class DomainConfig(BaseModel):
    """Per-domain configuration -- inherits from global defaults."""

    min_response_chars: int = 10
    max_turn_ratio: float = 50.0
    min_assistant_messages: int = 1
    dedup_scope: Optional[str] = None
    dedup_threshold: Optional[float] = None
    style: StyleConfig = StyleConfig()


class PipelineConfig(BaseModel):
    """Top-level pipeline configuration schema.

    Validated against configs/pipeline.yaml.  Provides get_domain_config()
    to merge global defaults with domain-specific overrides.
    """

    version: int = 1
    defaults: DomainConfig = DomainConfig()
    domains: dict[str, DomainConfig] = Field(default_factory=dict)
    ngram_size: int = Field(default=3, ge=2, le=5)
    dedup_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    dedup_scope: str = Field(default="response", pattern=r"^(response|prompt|full)$")
    style_validation: bool = True
    include_quality_scores: bool = True
    output_suffix: str = "_curated"
    topic_distribution: dict[str, float] = Field(default_factory=dict)
    template_paths: dict[str, str] = Field(default_factory=dict)

    def get_domain_config(self, domain: str) -> DomainConfig:
        """Merge global defaults with domain-specific overrides.

        Deep-merges the ``style`` sub-config so domain-level style keys
        overlay defaults rather than replacing the entire style block.

        Args:
            domain: Domain name (e.g. "code", "knowledge", "tool-calling").

        Returns:
            Merged DomainConfig for the requested domain.
        """
        base = self.defaults.model_dump()
        domain_model = self.domains.get(domain)
        if domain_model is None:
            return DomainConfig(**base)

        override = domain_model.model_dump(exclude_unset=True)

        # Deep-merge style sub-config
        if "style" in override:
            base_style = base.get("style", {})
            base_style.update(override.pop("style"))
            base["style"] = base_style

        base.update(override)
        return DomainConfig(**base)


def load_config(config_path: Path) -> PipelineConfig:
    """Load and validate pipeline configuration from a YAML file.

    Uses yaml.safe_load exclusively per T-02-04 (never yaml.load).

    Args:
        config_path: Path to the pipeline.yaml file.

    Returns:
        Validated PipelineConfig instance.

    Raises:
        FileNotFoundError: If config_path does not exist.
        pydantic.ValidationError: If YAML content fails schema validation.
    """
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return PipelineConfig.model_validate(raw)
