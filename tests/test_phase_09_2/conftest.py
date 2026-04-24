# SPDX-License-Identifier: MIT
"""Shared fixtures for Phase 09.2 tool-call regression diagnosis tests.

Provides three fixtures used by plans 02 and 04 to verify template parity
(D-04) and audit training-data distribution (D-06):

1. train_template_string       -- verbatim SFT chat template from scripts/train.py
                                   (includes TRL {% generation %} markers used
                                   for assistant_only_loss mask generation).
2. persisted_template_string   -- chat_template field loaded from
                                   models/lyra-merged/tokenizer_config.json;
                                   test is skipped if the merged model is
                                   absent (e.g. fresh worktree without LFS).
3. canonical_tool_call_prompt  -- minimal [system, user] message list for a
                                   weather-query tool-call probe.

All fixtures are pure Python values (no model loads) so collection is free of
heavyweight side effects.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


# Verbatim copy of the SFT chat template from scripts/train.py (lines 436-456).
# Downstream tests (plan 02) strip the {% generation %} / {% endgeneration %}
# markers and compare the result against the persisted template character-by-
# character and token-ID-by-token-ID on the canonical prompt.
_TRAIN_TEMPLATE = (
    "{% for message in messages %}"
    "{% if loop.first and messages[0]['role'] != 'system' %}"
    "{{ '<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n' }}"
    "{% endif %}"
    "{% set msg_content = message['content'] if message.get('content') else '' %}"
    "{% if message.get('tool_calls') %}"
    "{% set msg_content = '<tool_call>' + message['tool_calls']|tojson + '</tool_call>' %}"
    "{% endif %}"
    "{% if message['role'] == 'assistant' %}"
    "{% generation %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + msg_content + '<|im_end|>\n' }}"
    "{% endgeneration %}"
    "{% else %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + msg_content + '<|im_end|>\n' }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|im_start|>assistant\n' }}"
    "{% endif %}"
)


@pytest.fixture
def train_template_string() -> str:
    """Return the verbatim train-time chat template (with generation markers).

    Source of truth: scripts/train.py lines 436-456. If train.py's template
    changes, update this fixture in lock-step so the parity test catches
    train/persisted drift without masking itself.
    """
    return _TRAIN_TEMPLATE


@pytest.fixture
def persisted_template_string() -> str:
    """Load the chat_template field from models/lyra-merged/tokenizer_config.json.

    Skips the test (rather than failing) if the merged model directory or its
    tokenizer_config.json are absent -- this allows the test suite to collect
    cleanly in environments that do not have the 3.4GB merged safetensors
    pulled from Git LFS (e.g. fresh worktrees, CI runners without LFS).

    Uses json.load (per threat register T-09.2-03); the JSON parser rejects
    malformed input and no code execution is possible from tokenizer_config.
    """
    config_path = Path("models/lyra-merged/tokenizer_config.json")
    if not config_path.exists():
        pytest.skip(f"Persisted tokenizer_config not found at {config_path}")
    data = json.loads(config_path.read_text())
    template = data.get("chat_template")
    if not template:
        pytest.skip(
            f"chat_template field missing or empty in {config_path}; "
            "run scripts/train.py merge step to populate."
        )
    return template


@pytest.fixture
def canonical_tool_call_prompt() -> list[dict]:
    """Return the canonical [system, user] tool-call prompt used across parity tests.

    A weather-query is the reference because it is the single most common
    tool-call idiom in the Lyra training set and appears frequently in both
    the assembled test split and hand-written D-05 single-turn probes -- giving
    plan 02 the widest comparability baseline.
    """
    return [
        {"role": "system", "content": "You are a helpful assistant with access to tools."},
        {"role": "user", "content": "What's the weather in Paris?"},
    ]
