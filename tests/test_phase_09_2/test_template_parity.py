# SPDX-License-Identifier: MIT
"""RED stub for the D-04 template-parity check.

Plan 02 of Phase 09.2 will remove the @pytest.mark.skip decorator and
implement the real assertion: the train-time template (with {% generation %}
markers stripped) must produce bit-identical token IDs to the persisted
template when both render the canonical tool-call prompt through
AutoTokenizer.apply_chat_template.
"""
from __future__ import annotations

import pytest


@pytest.mark.skip(reason="RED stub: filled in by Plan 02")
def test_train_and_persisted_templates_produce_identical_tokens():
    """Placeholder -- plan 02 will implement the real assertion.

    Expected shape (for plan 02 implementation reference):
        stripped = train_template_string
        for marker in ("{% generation %}", "{% endgeneration %}"):
            stripped = stripped.replace(marker, "")
        assert stripped == persisted_template_string
        # Tokenize canonical_tool_call_prompt under both templates and diff IDs.
    """
    pass
