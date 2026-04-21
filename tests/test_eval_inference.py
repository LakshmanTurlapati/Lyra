#!/usr/bin/env python3
"""Unit tests for scripts/eval_inference.py -- custom inference eval helpers."""
import pytest


def test_check_tool_call_format_valid():
    """check_tool_call_format returns True for valid <tool_call> wrapped JSON."""
    from scripts.eval_inference import check_tool_call_format
    output = '<tool_call>{"name": "get_weather", "arguments": {"city": "Austin"}}</tool_call>'
    assert check_tool_call_format(output) is True


def test_check_tool_call_format_missing_wrapper():
    """check_tool_call_format returns False when <tool_call> wrapper is absent."""
    from scripts.eval_inference import check_tool_call_format
    output = '{"name": "get_weather", "arguments": {"city": "Austin"}}'
    assert check_tool_call_format(output) is False


def test_check_tool_call_format_missing_name():
    """check_tool_call_format returns False when JSON lacks 'name' key."""
    from scripts.eval_inference import check_tool_call_format
    output = '<tool_call>{"arguments": {"city": "Austin"}}</tool_call>'
    assert check_tool_call_format(output) is False


def test_check_tool_call_format_missing_arguments():
    """check_tool_call_format returns False when JSON lacks 'arguments' key."""
    from scripts.eval_inference import check_tool_call_format
    output = '<tool_call>{"name": "get_weather"}</tool_call>'
    assert check_tool_call_format(output) is False


def test_check_tool_call_format_invalid_json():
    """check_tool_call_format returns False for malformed JSON inside wrapper."""
    from scripts.eval_inference import check_tool_call_format
    output = '<tool_call>{not valid json}</tool_call>'
    assert check_tool_call_format(output) is False


def test_check_code_syntax_valid_python():
    """check_code_syntax returns True for syntactically valid Python code block."""
    from scripts.eval_inference import check_code_syntax
    output = '```python\ndef add(a, b):\n    return a + b\n```'
    assert check_code_syntax(output) is True


def test_check_code_syntax_invalid_python():
    """check_code_syntax returns False for Python block with syntax error."""
    from scripts.eval_inference import check_code_syntax
    output = '```python\ndef broken(\n    return\n```'
    assert check_code_syntax(output) is False


def test_check_code_syntax_no_code_block():
    """check_code_syntax returns False when no triple-backtick code block is present."""
    from scripts.eval_inference import check_code_syntax
    output = 'Here is the answer: x = 1 + 2'
    assert check_code_syntax(output) is False


def test_check_code_syntax_py_alias():
    """check_code_syntax accepts ```py shorthand as well as ```python."""
    from scripts.eval_inference import check_code_syntax
    output = '```py\nx = 42\n```'
    assert check_code_syntax(output) is True


def test_run_custom_eval_returns_category_result(tmp_path, monkeypatch):
    """run_custom_eval returns a CategoryResult with tool-call-format and code-syntax benchmarks."""
    import json, sys
    from pathlib import Path
    from scripts.eval_config import BenchmarkResult, CategoryResult, EvalResult

    # Create a minimal fake dataset at tmp_path/assembled
    assembled_dir = tmp_path / "assembled"
    assembled_dir.mkdir()
    # Minimal HuggingFace DatasetDict JSON structure
    dataset_info = {"splits": {"test": {"name": "test", "num_examples": 2}}}
    (assembled_dir / "dataset_info.json").write_text(json.dumps(dataset_info))

    # Monkeypatch load_from_disk to return a fake test split
    fake_sample_tool = {
        "messages": [
            {"role": "user", "content": "Call get_weather for Austin"},
            {"role": "assistant", "content": '<tool_call>{"name": "get_weather", "arguments": {"city": "Austin"}}</tool_call>'},
        ],
        "domain": "tool-calling",
    }
    fake_sample_code = {
        "messages": [
            {"role": "user", "content": "Write an add function"},
            {"role": "assistant", "content": "```python\ndef add(a, b):\n    return a + b\n```"},
        ],
        "domain": "code",
    }

    class FakeTestSplit:
        def __iter__(self):
            return iter([fake_sample_tool, fake_sample_code])

    class FakeDatasetDict:
        def __getitem__(self, key):
            return FakeTestSplit()

    import scripts.eval_inference as eval_inf_mod
    monkeypatch.setattr(eval_inf_mod, "load_from_disk", lambda path: FakeDatasetDict())

    # Monkeypatch run_inference_on_sample to return the last assistant message directly
    def fake_inference(model, tokenizer, sample, device, max_new_tokens=256):
        # Return the last assistant message content (the label)
        msgs = sample["messages"]
        for m in reversed(msgs):
            if m["role"] == "assistant":
                return m["content"]
        return ""

    monkeypatch.setattr(eval_inf_mod, "run_inference_on_sample", fake_inference)

    # Monkeypatch model/tokenizer loading to return None (fake_inference ignores them)
    import types
    fake_model = types.SimpleNamespace()
    fake_tokenizer = types.SimpleNamespace()
    monkeypatch.setattr(eval_inf_mod, "_load_model_and_tokenizer", lambda path, device: (fake_model, fake_tokenizer))

    result = eval_inf_mod.run_custom_eval("models/lyra-merged", "cpu", str(assembled_dir))
    assert isinstance(result, CategoryResult)
    assert result.category == "custom"
    bench_names = {b.benchmark for b in result.benchmarks}
    assert "tool-call-format" in bench_names
    assert "code-syntax" in bench_names
    # Both samples should pass (they have valid format in the fake data)
    scores = {b.benchmark: b.score for b in result.benchmarks}
    assert scores["tool-call-format"] == pytest.approx(1.0)
    assert scores["code-syntax"] == pytest.approx(1.0)
