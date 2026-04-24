# SPDX-License-Identifier: MIT
"""D-04: Template parity — train-time (stripped) must equal persisted template
both at string level and at tokenization level. Phase 09.2 Plan 02.

These assertions lock the Phase 09.1 Plan 01 template-persistence fix into
regression-test infrastructure so future edits to ``scripts/train.py`` cannot
silently desynchronise the in-memory SFT chat template from the persisted
``models/lyra-merged/tokenizer_config.json`` ``chat_template`` string.

The two tests together conclusively rule out H5 (template persistence bug) from
the Phase 09.2 hypothesis ranking: identical string content AND identical token
IDs on the canonical tool-call probe prompt.
"""
from __future__ import annotations

import pytest
from transformers import AutoTokenizer


def test_stripped_train_template_matches_persisted_template(
    train_template_string, persisted_template_string
):
    """Stripping the TRL ``{% generation %}``/``{% endgeneration %}`` markers
    from the train-time template must yield the exact persisted template.

    If this fails, the on-disk ``tokenizer_config.json`` chat_template has
    drifted from the in-memory ``scripts/train.py`` template — a train/eval
    regression. The failure itself is diagnostic evidence for Plan 05's
    DIAGNOSIS.md; do not paper over it.
    """
    stripped = train_template_string
    for marker in ("{% generation %}", "{% endgeneration %}"):
        stripped = stripped.replace(marker, "")
    assert stripped == persisted_template_string, (
        "Train-time template (stripped) must match models/lyra-merged/tokenizer_config.json "
        "chat_template. Divergence indicates a train/eval template regression — see D-04 in "
        ".planning/phases/09.2-tool-call-regression-diagnosis/09.2-CONTEXT.md."
    )


def test_both_templates_produce_identical_token_ids(
    train_template_string, persisted_template_string, canonical_tool_call_prompt
):
    """Tokenizing the canonical tool-call prompt under both templates must
    yield identical ``list[int]`` token ID sequences.

    This is the stronger parity check: even if string equality were to hold,
    whitespace handling differences in Jinja or tokenizer normalisation could
    still produce divergent token streams. A mismatch here would be a bug in
    template persistence that a string-equality check alone would not catch.
    """
    tok = AutoTokenizer.from_pretrained("models/lyra-merged")

    tok.chat_template = train_template_string
    prompt_train = tok.apply_chat_template(
        canonical_tool_call_prompt, tokenize=False, add_generation_prompt=True
    )
    ids_train = tok(prompt_train)["input_ids"]

    tok.chat_template = persisted_template_string
    prompt_pers = tok.apply_chat_template(
        canonical_tool_call_prompt, tokenize=False, add_generation_prompt=True
    )
    ids_pers = tok(prompt_pers)["input_ids"]

    if ids_train != ids_pers:
        first_diff = next(
            (i for i, (a, b) in enumerate(zip(ids_train, ids_pers)) if a != b),
            min(len(ids_train), len(ids_pers)),
        )
        pytest.fail(
            f"Token-ID drift at index {first_diff}: "
            f"train={ids_train[first_diff:first_diff+5]!r} "
            f"pers={ids_pers[first_diff:first_diff+5]!r} "
            f"(train len={len(ids_train)}, pers len={len(ids_pers)})"
        )
