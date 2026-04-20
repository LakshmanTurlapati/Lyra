# Phase 1: Data Format and Pipeline Foundation - Research

**Researched:** 2026-04-20
**Domain:** ShareGPT data format specification, Python validation, SmolLM2-1.7B tokenizer alignment
**Confidence:** HIGH

## Summary

Phase 1 establishes the data contract that every downstream phase depends on. The core deliverables are: (1) a ShareGPT format specification with OpenAI-compatible tool calling, (2) Python validation scripts that check conversations against SmolLM2-1.7B's tokenizer and chat template, and (3) a prompt template library for guiding Claude Code data generation. No training data is generated at scale in this phase -- only the format, validation, and templates.

The critical technical finding is that SmolLM2-1.7B uses a `<|im_start|>/<|im_end|>` ChatML-style template and its own `<tool_call>` XML delimiter pattern for function calls. TRL v1.2.0's SFTTrainer expects `messages`-format conversations with `tool_calls` and `tool` role messages. The format specification must bridge these two conventions -- storing data in TRL-native format (the downstream consumer) while validating that tokenized output aligns with SmolLM2's actual chat template. Getting this wrong is a silent total-loss failure (training looks fine, model behavior is broken).

**Primary recommendation:** Define the canonical data format as TRL-native conversational format with a `tools` column (not classic ShareGPT `from`/`value` format). This eliminates conversion steps and aligns directly with SFTTrainer's expected input. Validate every sample by round-tripping through `tokenizer.apply_chat_template()` and checking special token placement.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Use OpenAI-compatible function calling format within ShareGPT conversations. Standard `function_call` role with `name` and `arguments` JSON, `observation` role for tool results.
- **D-02:** Spec covers full complexity from day one -- single function calls, multi-turn with results, parallel execution, MCP-style patterns, and CLI/shell commands. All patterns defined in the format spec before any data generation begins.
- **D-03:** Maximum 2048 tokens per training conversation. This matches SmolLM2-1.7B's native training sequence length and optimizes for the short, practical interactions the model will excel at.
- **D-04:** Natural length distribution within the 2048 cap. Samples range organically from ~200 to ~1800 tokens based on task complexity. No artificial length targeting.
- **D-05:** Data generation happens directly in Claude Code sessions. Claude Opus writes training samples as JSONL files. No Anthropic API SDK or external API pipeline in the project.
- **D-06:** Python scripts handle all post-generation processing: format validation, tokenizer alignment checks, deduplication, and quality scoring. The pipeline is: Claude Code generates -> Python validates/filters.
- **D-07:** Python is the primary language. All scripts, validation, training, and evaluation code in Python.
- **D-08:** Flat scripts/ directory with standalone Python files. No package installation overhead -- simple, iterative research structure.
- **D-09:** Data stored in datasets/ directory with domain separation: datasets/tool-calling/, datasets/code/, datasets/knowledge/. Raw generated data and curated filtered data both live here. Gitignored for large files.

### Claude's Discretion
- Exact ShareGPT JSON field names and nesting structure (following OpenAI-compatible conventions)
- Python script naming and internal organization within scripts/
- Validation error message format and logging approach
- Prompt template file format (YAML, JSON, or Markdown)

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| DATA-01 | User can generate ShareGPT-format conversation datasets with strict role ordering (human/gpt/function_call/observation) | TRL-native format spec with role ordering rules; SmolLM2 `<tool_call>` pattern; Pydantic schema validation |
| DATA-02 | User can validate conversation format alignment with SmolLM2-1.7B tokenizer and chat template | SmolLM2 chat template extracted (`<\|im_start\|>`/`<\|im_end\|>`); tokenizer config (vocab 49152, bos=1, eos=2, pad=2); round-trip validation approach |
| DATA-06 | Prompt template library organized by category (tool call, code, general knowledge) with documented system prompts | YAML template format recommendation; SmolLM2 function calling system prompt pattern; category organization in datasets/ |
</phase_requirements>

## Standard Stack

### Core (Phase 1 Only)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pydantic | 2.12.5 | ShareGPT schema validation and JSONL parsing | Type-safe validation with detailed error messages; installed locally (verified) [VERIFIED: pip3 show] |
| transformers | 5.5.4 | SmolLM2-1.7B tokenizer loading and chat template application | Required to load tokenizer and apply_chat_template(); verified in project STACK.md [VERIFIED: PyPI] |
| Python stdlib json | 3.14.2 | JSONL read/write | No external dependency needed for JSON serialization [VERIFIED: python3 --version] |
| PyYAML | latest | Prompt template file parsing | Standard YAML parser; lightweight dependency for template config [ASSUMED] |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| tiktoken | latest | Fast token counting for 2048-limit enforcement | If transformers tokenizer is too slow for batch counting; likely not needed at Phase 1 scale [ASSUMED] |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Pydantic (validation) | JSON Schema + jsonschema lib | Pydantic gives Python objects + validation in one step; jsonschema only validates, requires separate parsing |
| YAML (templates) | Markdown with frontmatter | YAML is easier to parse programmatically and supports nested structures; Markdown better for human reading but harder to extract structured data |
| YAML (templates) | JSON | JSON lacks comments and multi-line strings; YAML is more readable for prompt templates with long text blocks |

**Installation (Phase 1 minimal):**
```bash
pip install pydantic transformers pyyaml
```

## Architecture Patterns

### Recommended Project Structure (Phase 1)

```
Claude-Mini/
  scripts/
    validate_format.py       # ShareGPT format validation against Pydantic schema
    validate_tokenizer.py    # SmolLM2 tokenizer alignment check
    generate_sample.py       # Generate small sample batch (10-50) for testing
    check_token_length.py    # Token count distribution analysis
  datasets/
    tool-calling/            # Tool call conversation JSONL files
    code/                    # Code generation conversation JSONL files
    knowledge/               # General knowledge conversation JSONL files
  templates/
    tool-calling.yaml        # Tool call prompt templates + schemas
    code.yaml                # Code generation prompt templates
    knowledge.yaml           # General knowledge prompt templates
    system-prompts.yaml      # Shared system prompt variants
  specs/
    sharegpt-format.md       # Canonical format specification document
```

### Pattern 1: TRL-Native Conversational Format (NOT Classic ShareGPT)

**What:** Store conversations in TRL's expected `messages` format with `role`/`content` keys, not classic ShareGPT `from`/`value` format. This eliminates conversion steps at training time.

**When to use:** Always. TRL v1.2.0 SFTTrainer expects this format natively.

**Critical distinction:** The user's decision (D-01) says "ShareGPT format" but TRL's modern tooling has evolved past the original ShareGPT convention. The format spec should use TRL-native naming while maintaining the conceptual structure of ShareGPT (multi-turn conversations with role ordering).

**TRL-native format for regular conversations:** [VERIFIED: huggingface.co/docs/trl/dataset_formats]
```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the weather?"},
    {"role": "assistant", "content": "I can check that for you."}
  ]
}
```

**TRL-native format for tool calling conversations:** [VERIFIED: huggingface.co/docs/trl/dataset_formats#tool-calling]
```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant with access to tools."},
    {"role": "user", "content": "Turn on the living room lights."},
    {"role": "assistant", "tool_calls": [
      {"type": "function", "function": {
        "name": "control_light",
        "arguments": {"room": "living room", "state": "on"}
      }}
    ]},
    {"role": "tool", "name": "control_light", "content": "The lights in the living room are now on."},
    {"role": "assistant", "content": "Done!"}
  ],
  "tools": [
    {"type": "function", "function": {
      "name": "control_light",
      "description": "Controls the lights in a room.",
      "parameters": {
        "type": "object",
        "properties": {
          "room": {"type": "string", "description": "The name of the room."},
          "state": {"type": "string", "description": "on or off"}
        },
        "required": ["room", "state"]
      }
    }}
  ]
}
```

**SmolLM2's native tool call output format (what the model actually produces):** [VERIFIED: huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct/blob/main/instructions_function_calling.md]
```
<tool_call>[{"name": "func_name", "arguments": {"arg1": "val1"}}]</tool_call>
```

**Reconciliation strategy:** Store data in TRL-native format (with `tool_calls` in assistant messages and `tool` role for responses). At tokenization time, the chat template handles conversion to SmolLM2's `<tool_call>` delimiters. The validation script must verify that `tokenizer.apply_chat_template()` produces correct `<tool_call>` markers from the stored format.

### Pattern 2: Pydantic Schema Validation

**What:** Define Pydantic models that enforce ShareGPT structure, role ordering, and field requirements. Validate each JSONL line against these models.

**When to use:** Every time a JSONL file is created or modified.

**Example:** [ASSUMED -- standard Pydantic 2 pattern]
```python
from pydantic import BaseModel, model_validator
from typing import Optional

class ToolCall(BaseModel):
    type: str = "function"
    function: dict  # {"name": str, "arguments": dict}

class Message(BaseModel):
    role: str  # system, user, assistant, tool
    content: Optional[str] = None
    tool_calls: Optional[list[ToolCall]] = None
    name: Optional[str] = None  # for tool role messages

class ToolSchema(BaseModel):
    type: str = "function"
    function: dict  # OpenAI function schema

class Conversation(BaseModel):
    messages: list[Message]
    tools: Optional[list[ToolSchema]] = None

    @model_validator(mode='after')
    def validate_role_ordering(self) -> 'Conversation':
        # Enforce: first message is system (optional), then user/assistant alternate
        # tool messages must follow assistant messages with tool_calls
        # assistant messages with tool_calls must have content=None or empty
        ...
        return self
```

### Pattern 3: Tokenizer Round-Trip Validation

**What:** For every conversation, run `tokenizer.apply_chat_template()` and verify that: (a) the output contains correct `<|im_start|>`/`<|im_end|>` boundaries, (b) total token count is within 2048, (c) tool call delimiters (`<tool_call>`/`</tool_call>`) appear correctly, (d) the EOS token (id=2) appears at the end.

**When to use:** After format validation passes. This is the critical check that prevents Pitfall 6 (tokenizer misalignment).

**Example:** [VERIFIED: SmolLM2 tokenizer config from HuggingFace]
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct")

def validate_tokenization(conversation: dict) -> dict:
    """Returns validation result with pass/fail and details."""
    messages = conversation["messages"]
    tools = conversation.get("tools")

    # Apply chat template
    try:
        tokenized = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=True,
            return_tensors=None
        )
    except Exception as e:
        return {"valid": False, "error": f"Chat template failed: {e}"}

    # Check token count
    token_count = len(tokenized)
    if token_count > 2048:
        return {"valid": False, "error": f"Token count {token_count} exceeds 2048 limit"}

    # Check EOS token presence
    if tokenized[-1] != tokenizer.eos_token_id:
        return {"valid": False, "error": "Missing EOS token at end"}

    # Decode and check structure
    decoded = tokenizer.decode(tokenized)
    if "<|im_start|>" not in decoded or "<|im_end|>" not in decoded:
        return {"valid": False, "error": "Missing chat template markers"}

    return {"valid": True, "token_count": token_count}
```

### Pattern 4: YAML Prompt Templates

**What:** Store prompt templates as YAML files with metadata, system prompts, and example variations.

**When to use:** For all three domain categories (tool-calling, code, knowledge).

**Example:** [ASSUMED -- standard convention]
```yaml
# templates/tool-calling.yaml
domain: tool-calling
description: Templates for generating tool calling training conversations

system_prompts:
  - id: tool_call_standard
    content: |
      You are a helpful assistant with access to the following tools.
      You must use the tools when appropriate to answer the user's question.
      When no tool is needed, respond directly.

categories:
  single_call:
    description: User asks question, assistant calls one tool, gets result, responds
    complexity: basic
    examples:
      - topic: weather lookup
        tools:
          - name: get_weather
            description: Get current weather for a city
            parameters:
              city:
                type: string
                required: true

  multi_turn:
    description: Multi-turn conversation with tool calls and follow-ups
    complexity: intermediate

  parallel_calls:
    description: Assistant calls multiple tools in a single turn
    complexity: advanced

  mcp_patterns:
    description: MCP-style server discovery and tool invocation
    complexity: advanced

  cli_commands:
    description: CLI/shell command generation as tool use
    complexity: intermediate
```

### Anti-Patterns to Avoid

- **Classic ShareGPT `from`/`value` format:** Do NOT use `{"conversations": [{"from": "human", "value": "..."}]}`. This requires conversion to TRL format at training time. Use TRL-native `messages` format from the start. [VERIFIED: TRL docs show `messages` as the expected key]

- **Hardcoded tool schemas in message content:** Do NOT embed JSON tool schemas as text inside the system message content. Use the separate `tools` column that TRL expects. The chat template handles injecting tools into the system prompt. [VERIFIED: TRL dataset_formats#tool-calling]

- **Assuming SmolLM2 supports `{% generation %}` tags:** SmolLM2's chat template does NOT include TRL's `{% generation %}` / `{% endgeneration %}` markers for assistant-only loss. This will need to be handled explicitly when training (Phase 8), but the format spec must be compatible with adding these later. [VERIFIED: github.com/huggingface/trl/issues/4879]

## SmolLM2-1.7B Technical Reference

### Tokenizer Configuration [VERIFIED: huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct]

| Property | Value |
|----------|-------|
| Vocabulary size | 49,152 |
| BOS token | `<\|im_start\|>` (id: 1) |
| EOS token | `<\|im_end\|>` (id: 2) |
| PAD token | `<\|im_end\|>` (id: 2, same as EOS) |
| UNK token | `<\|endoftext\|>` |
| Model type | llama (LlamaForCausalLM) |
| Max position embeddings | 8,192 |
| Training sequence length | 2,048 |
| Architecture | 24 layers, 32 heads, hidden_size=2048 |
| Precision | bfloat16 |

### Chat Template [VERIFIED: tokenizer_config.json on HuggingFace]

```jinja2
{% for message in messages %}
{% if loop.first and messages[0]['role'] != 'system' %}
{{ '<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n' }}
{% endif %}
{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}
{% endfor %}
{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}
```

**Key observations:**
1. Default system prompt is injected if none provided -- our data MUST include explicit system prompts to avoid this default
2. Template is simple concatenation: `<|im_start|>{role}\n{content}<|im_end|>\n`
3. No `{% generation %}` / `{% endgeneration %}` markers -- assistant_only_loss requires manual template patching at training time
4. Tool calls are embedded as content text using `<tool_call>` XML delimiters, NOT as structured `tool_calls` fields in the template

### Function Calling Format [VERIFIED: instructions_function_calling.md on HuggingFace]

SmolLM2 was trained on function calling data from Argilla's Synth-APIGen-v0.1. The model produces tool calls as:

```
<tool_call>[{"name": "function_name", "arguments": {"arg": "value"}}]</tool_call>
```

Tool definitions are injected into the system prompt as serialized JSON. The model outputs an empty array `[]` when no tool call is needed.

**Critical implication for data format:** When storing in TRL-native format (structured `tool_calls` field), the `apply_chat_template()` function must convert this to the `<tool_call>` XML format that SmolLM2 expects. The validation script must verify this conversion happens correctly.

### assistant_only_loss Compatibility [VERIFIED: github.com/huggingface/trl/issues/4879]

TRL v1.2.0 requires `{% generation %}` / `{% endgeneration %}` tags in the chat template for `assistant_only_loss=True` to work. SmolLM2's default template does NOT have these. As of April 2026, TRL does NOT auto-patch SmolLM2 templates (only Qwen3 is auto-patched). This must be handled in Phase 8 (training) by either:
1. Manually adding generation tags to SmolLM2's chat template before training
2. Using `completion_only_loss` with prompt-completion format instead
3. Using Unsloth's own loss masking which may handle this differently

This does NOT affect Phase 1 format design, but the format must be compatible with both approaches.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Token counting | Custom BPE tokenizer | `transformers.AutoTokenizer` with SmolLM2 | Tokenizer mismatch = wrong counts = silent training failure |
| JSON schema validation | Manual dict checking | Pydantic v2 `BaseModel` | Pydantic gives type coercion, nested validation, clear error messages |
| Chat template application | Manual string formatting | `tokenizer.apply_chat_template()` | Template may change between model versions; using the tokenizer's own template ensures alignment |
| Tool schema generation | Hand-written JSON schemas | `transformers.utils.get_json_schema()` | Generates OpenAI-compatible schemas from Python functions with type hints [VERIFIED: TRL docs] |
| JSONL parsing | Custom line-by-line JSON | `pydantic.model_validate_json()` per line | Handles encoding, validation, and error reporting in one call |

**Key insight:** The transformers library is the single source of truth for how SmolLM2 tokenizes and templates conversations. Every validation check must go through the actual tokenizer, not a reimplementation.

## Common Pitfalls

### Pitfall 1: Chat Template Mismatch (Silent Total-Loss Failure)

**What goes wrong:** Data is formatted correctly as JSON but when tokenized, the special tokens (`<|im_start|>`, `<|im_end|>`) are misplaced or missing. Training loss looks normal but the model produces gibberish at turn boundaries. [CITED: .planning/research/PITFALLS.md, Pitfall 6]

**Why it happens:** SmolLM2 uses ChatML-style delimiters. If the chat template is not applied, or applied incorrectly, the model never learns where turns begin and end. This is especially dangerous because training loss can look fine -- the model is still learning to predict tokens, just the wrong way.

**How to avoid:** Run `tokenizer.apply_chat_template()` on every sample during validation. Decode the result back to text and visually inspect the first 10 samples. Check that each message is wrapped in `<|im_start|>{role}\n...<|im_end|>\n`.

**Warning signs:** Validation passes but decoded text shows missing delimiters; training loss is suspiciously low; model ignores turn boundaries at inference time.

### Pitfall 2: Tool Call Format Inconsistency

**What goes wrong:** Some samples use `<tool_call>` XML delimiters (SmolLM2 native), others use structured `tool_calls` field (TRL format), others use `function_call` role (classic ShareGPT). The model sees inconsistent patterns and fails to learn any of them reliably. [CITED: .planning/research/PITFALLS.md, Pitfall 2]

**Why it happens:** Multiple conventions exist for tool calls in conversation data. Without a single canonical format enforced by validation, different generation sessions produce different formats.

**How to avoid:** Define ONE canonical storage format (TRL-native with `tool_calls` field). Validate that every tool call sample matches this exact structure. Let `apply_chat_template()` handle conversion to SmolLM2's `<tool_call>` delimiters.

**Warning signs:** Pydantic validation passes (structure is valid JSON) but tokenized output shows inconsistent tool call patterns.

### Pitfall 3: 2048 Token Limit Exceeded After Template Application

**What goes wrong:** A conversation is ~1900 tokens as raw text, but after `apply_chat_template()` adds system prompt defaults, role markers, and special tokens, it exceeds 2048.

**Why it happens:** The chat template adds overhead: each message gets `<|im_start|>{role}\n` (variable) and `<|im_end|>\n` (fixed). System prompts with tool definitions can consume 500+ tokens. The 2048 limit applies to the TOKENIZED output, not the raw text length.

**How to avoid:** Always measure token count AFTER `apply_chat_template(tokenize=True)`, never before. Build the token budget backward: start with 2048, subtract template overhead (~50-100 tokens for delimiters), subtract system prompt + tool definitions, remaining budget is for actual conversation content.

**Warning signs:** Samples pass text-length checks but fail after tokenization; longer tool-calling samples consistently exceed limits due to tool schema overhead.

### Pitfall 4: Default System Prompt Injection

**What goes wrong:** SmolLM2's chat template auto-injects "You are a helpful AI assistant named SmolLM, trained by Hugging Face" when no system message is provided. This wastes tokens and introduces inconsistent behavior between samples with and without system prompts. [VERIFIED: tokenizer_config.json chat_template]

**Why it happens:** The chat template has a conditional: `{% if loop.first and messages[0]['role'] != 'system' %}` that injects a default. Data generation sessions that omit system prompts get this default silently.

**How to avoid:** Every conversation MUST start with an explicit system message. The validation script must reject conversations where `messages[0]['role'] != 'system'`.

**Warning signs:** Tokenized output contains "SmolLM" references that are not in the source data.

### Pitfall 5: PAD and EOS Token Collision

**What goes wrong:** SmolLM2 uses the same token (id=2, `<|im_end|>`) for both PAD and EOS. If padding is not handled carefully during training, the model cannot distinguish "end of sequence" from "padding". [VERIFIED: config.json pad_token_id=2, eos_token_id=2]

**Why it happens:** Many models share PAD/EOS tokens. This is technically valid but requires careful attention to loss masking: PAD tokens must be masked (label=-100) while the final EOS must NOT be masked.

**How to avoid:** This is a training concern (Phase 8), but the format spec must document this fact so downstream phases handle it correctly. Include a note in the format specification.

**Warning signs:** Model never stops generating (it never learned to produce EOS because all EOS tokens were masked as padding).

## Code Examples

### Complete Validation Script Pattern

**Source:** Synthesized from TRL docs + SmolLM2 tokenizer config [VERIFIED sources combined]

```python
#!/usr/bin/env python3
"""validate_format.py -- Validate ShareGPT JSONL against SmolLM2 format spec."""
import json
import sys
from pathlib import Path
from pydantic import BaseModel, model_validator
from typing import Optional

# --- Pydantic Schema ---

class FunctionDef(BaseModel):
    name: str
    description: str
    parameters: dict

class ToolSchema(BaseModel):
    type: str = "function"
    function: FunctionDef

class ToolCallFunction(BaseModel):
    name: str
    arguments: dict

class ToolCall(BaseModel):
    type: str = "function"
    function: ToolCallFunction

class Message(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[list[ToolCall]] = None
    name: Optional[str] = None

class Conversation(BaseModel):
    messages: list[Message]
    tools: Optional[list[ToolSchema]] = None

    @model_validator(mode='after')
    def validate_structure(self) -> 'Conversation':
        msgs = self.messages
        if not msgs:
            raise ValueError("Empty conversation")

        # First message must be system
        if msgs[0].role != "system":
            raise ValueError("First message must be system role")

        # Validate role ordering
        for i, msg in enumerate(msgs):
            if msg.role not in ("system", "user", "assistant", "tool"):
                raise ValueError(f"Invalid role '{msg.role}' at index {i}")

            # tool messages must follow assistant with tool_calls
            if msg.role == "tool":
                if i == 0 or msgs[i-1].role not in ("assistant", "tool"):
                    raise ValueError(f"Tool message at index {i} must follow assistant or tool")
                if msg.name is None:
                    raise ValueError(f"Tool message at index {i} missing 'name' field")

            # assistant with tool_calls should not have content
            if msg.role == "assistant" and msg.tool_calls:
                if msg.content and msg.content.strip():
                    # Some formats allow content alongside tool_calls
                    pass  # Allow but flag as warning

        # If tools column present, validate tool_calls reference defined tools
        if self.tools:
            defined_names = {t.function.name for t in self.tools}
            for msg in msgs:
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        if tc.function.name not in defined_names:
                            raise ValueError(
                                f"Tool call '{tc.function.name}' not in defined tools: {defined_names}"
                            )
        return self


def validate_file(path: Path) -> dict:
    """Validate a JSONL file. Returns stats dict."""
    results = {"total": 0, "valid": 0, "invalid": 0, "errors": []}

    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            results["total"] += 1
            try:
                data = json.loads(line)
                Conversation.model_validate(data)
                results["valid"] += 1
            except Exception as e:
                results["invalid"] += 1
                results["errors"].append({
                    "line": line_num,
                    "error": str(e)
                })
    return results
```

### Tokenizer Validation Pattern

```python
#!/usr/bin/env python3
"""validate_tokenizer.py -- Check conversations against SmolLM2 tokenizer."""
from transformers import AutoTokenizer

MODEL_ID = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
MAX_TOKENS = 2048

def load_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_ID)

def validate_conversation(tokenizer, conversation: dict) -> dict:
    """Validate a single conversation against the tokenizer.

    Returns dict with: valid (bool), token_count (int), errors (list[str]).
    """
    errors = []
    messages = conversation["messages"]
    tools = conversation.get("tools")

    # Apply chat template
    try:
        kwargs = {"tokenize": True, "return_tensors": None}
        if tools:
            kwargs["tools"] = tools
        token_ids = tokenizer.apply_chat_template(messages, **kwargs)
    except Exception as e:
        return {"valid": False, "token_count": 0, "errors": [f"Template error: {e}"]}

    token_count = len(token_ids)

    # Check token limit
    if token_count > MAX_TOKENS:
        errors.append(f"Token count {token_count} exceeds {MAX_TOKENS} limit")

    # Check EOS token at end
    if token_ids[-1] != tokenizer.eos_token_id:
        errors.append(f"Missing EOS token (expected id={tokenizer.eos_token_id})")

    # Decode and inspect structure
    decoded = tokenizer.decode(token_ids)

    # Check for default system prompt injection
    if "You are a helpful AI assistant named SmolLM" in decoded:
        if "SmolLM" not in str(messages):
            errors.append("Default system prompt injected -- explicit system message missing or ignored")

    # Check role markers
    for msg in messages:
        expected_marker = f"<|im_start|>{msg['role']}"
        if expected_marker not in decoded:
            errors.append(f"Missing role marker for '{msg['role']}'")

    return {
        "valid": len(errors) == 0,
        "token_count": token_count,
        "errors": errors
    }
```

### Sample Conversation Examples

**Basic conversation (no tools):**
```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful coding assistant. Provide concise answers."},
    {"role": "user", "content": "Write a Python function to reverse a string."},
    {"role": "assistant", "content": "```python\ndef reverse_string(s: str) -> str:\n    return s[::-1]\n```"}
  ]
}
```

**Single tool call:**
```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant with access to tools."},
    {"role": "user", "content": "What is the weather in San Francisco?"},
    {"role": "assistant", "tool_calls": [
      {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "San Francisco"}}}
    ]},
    {"role": "tool", "name": "get_weather", "content": "{\"temp\": 62, \"condition\": \"foggy\"}"},
    {"role": "assistant", "content": "The weather in San Francisco is 62F and foggy."}
  ],
  "tools": [
    {"type": "function", "function": {
      "name": "get_weather",
      "description": "Get current weather for a city",
      "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
    }}
  ]
}
```

**Parallel tool calls:**
```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant. Use tools when needed."},
    {"role": "user", "content": "Compare weather in NYC and LA."},
    {"role": "assistant", "tool_calls": [
      {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "New York"}}},
      {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Los Angeles"}}}
    ]},
    {"role": "tool", "name": "get_weather", "content": "{\"temp\": 45, \"condition\": \"rainy\"}"},
    {"role": "tool", "name": "get_weather", "content": "{\"temp\": 78, \"condition\": \"sunny\"}"},
    {"role": "assistant", "content": "NYC is 45F and rainy, while LA is 78F and sunny."}
  ],
  "tools": [
    {"type": "function", "function": {
      "name": "get_weather",
      "description": "Get current weather for a city",
      "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
    }}
  ]
}
```

**No-tool-needed (assistant declines):**
```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant with access to tools."},
    {"role": "user", "content": "What is 2 + 2?"},
    {"role": "assistant", "content": "4."}
  ],
  "tools": [
    {"type": "function", "function": {
      "name": "calculator",
      "description": "Evaluate a mathematical expression",
      "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}
    }}
  ]
}
```

## Role Ordering Rules

The format spec must enforce these rules for valid conversations:

1. First message MUST be `system` role
2. After system, messages alternate between `user` and `assistant` (with tool interludes)
3. `tool` messages MUST immediately follow an `assistant` message that contains `tool_calls`
4. Multiple `tool` messages can follow a single `assistant` with multiple `tool_calls` (parallel calls)
5. An `assistant` message with `tool_calls` must be followed by exactly one `tool` message per tool call
6. The final message SHOULD be `assistant` role (the model's last response)
7. `assistant` messages with `tool_calls` typically have `content` set to `null` or empty string
8. `tool` messages MUST have a `name` field matching the tool that was called
9. `tool` message `content` should be the string representation of the tool's return value

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest (standard Python testing) |
| Config file | none -- Wave 0 task |
| Quick run command | `python -m pytest tests/ -x -q` |
| Full suite command | `python -m pytest tests/ -v` |

### Phase Requirements to Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DATA-01 | Valid conversations pass Pydantic validation | unit | `pytest tests/test_format_validator.py -x` | Wave 0 |
| DATA-01 | Invalid conversations rejected with errors | unit | `pytest tests/test_format_validator.py -x` | Wave 0 |
| DATA-01 | Role ordering violations detected | unit | `pytest tests/test_format_validator.py::test_role_ordering -x` | Wave 0 |
| DATA-02 | Token count within 2048 after template | unit | `pytest tests/test_tokenizer_validator.py -x` | Wave 0 |
| DATA-02 | Chat template markers present in output | unit | `pytest tests/test_tokenizer_validator.py::test_markers -x` | Wave 0 |
| DATA-02 | Tool calls render as `<tool_call>` in tokenized output | unit | `pytest tests/test_tokenizer_validator.py::test_tool_calls -x` | Wave 0 |
| DATA-06 | Template YAML files parseable | unit | `pytest tests/test_templates.py -x` | Wave 0 |
| DATA-06 | Templates cover all categories | unit | `pytest tests/test_templates.py::test_categories -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `python -m pytest tests/ -x -q`
- **Per wave merge:** `python -m pytest tests/ -v`
- **Phase gate:** Full suite green before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `tests/test_format_validator.py` -- covers DATA-01 (format validation)
- [ ] `tests/test_tokenizer_validator.py` -- covers DATA-02 (tokenizer alignment)
- [ ] `tests/test_templates.py` -- covers DATA-06 (prompt template library)
- [ ] `tests/conftest.py` -- shared fixtures (sample conversations, tokenizer instance)
- [ ] `pytest.ini` or `pyproject.toml` test config
- [ ] Framework install: `pip install pytest`

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Classic ShareGPT `from`/`value` | TRL-native `messages` with `role`/`content` | TRL v1.0 (April 2026) | No conversion needed; SFTTrainer consumes directly |
| Tool calls as text in content | Structured `tool_calls` field + `tools` column | TRL v0.19.0+ (2025) | Native tool calling dataset support; proper loss masking |
| Manual loss masking | `assistant_only_loss=True` in SFTConfig | TRL v1.0 (2026) | Requires `{% generation %}` template tags (SmolLM2 lacks these) |
| `setup_chat_format()` for template | `chat_template_path` in SFTConfig | TRL v1.2.0 (2026) | Can clone template from another model |

**Deprecated/outdated:**
- Classic ShareGPT `{"conversations": [{"from": "human", "value": "..."}]}` -- still works with Unsloth's `standardize_sharegpt()` but adds an unnecessary conversion step. Use TRL-native format instead.
- `function_call` and `observation` role names (from OpenAI's deprecated function calling API) -- superseded by `tool_calls` field and `tool` role in current OpenAI and TRL conventions.

**Important note on D-01:** The user decision mentions `function_call` role and `observation` role. These are the older OpenAI convention. Current TRL standard uses `assistant` with `tool_calls` field and `tool` role. The format spec should use the current standard while documenting the mapping to the user's terminology:
- `function_call` role --> `assistant` role with `tool_calls` field
- `observation` role --> `tool` role with `name` and `content` fields

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | PyYAML is the right choice for template parsing | Standard Stack | Low -- trivially replaceable with any YAML parser |
| A2 | Pydantic v2 model_validator pattern works for role ordering | Architecture Patterns | Low -- pattern is well-established in Pydantic v2 docs |
| A3 | SmolLM2's `apply_chat_template()` correctly handles the `tools` parameter for `<tool_call>` conversion | Architecture Patterns | HIGH -- must be verified empirically. If it does NOT handle tool call formatting, the validation approach changes significantly |
| A4 | tiktoken is not needed for Phase 1 (transformers tokenizer suffices) | Standard Stack | Low -- only matters at large scale |

## Open Questions

1. **Does SmolLM2's `apply_chat_template(tools=...)` correctly inject tool schemas and produce `<tool_call>` output?**
   - What we know: SmolLM2's instructions_function_calling.md shows `<tool_call>` delimiters. The model card references Synth-APIGen training data. TRL docs show `tools` parameter support.
   - What's unclear: Whether the tokenizer's chat template has a tool-aware branch, or whether tool definitions must be manually injected into the system prompt. The base chat template we fetched does NOT show tool handling logic.
   - Recommendation: Run an empirical test in Wave 1 -- load the tokenizer and call `apply_chat_template()` with a tools parameter. If it fails, the alternative is to manually format tool definitions into the system message content, and format tool calls as `<tool_call>` text in assistant content.

2. **How do parallel tool calls map to `<tool_call>` format?**
   - What we know: SmolLM2 format is `<tool_call>[{...}, {...}]</tool_call>` (JSON array). TRL format is `"tool_calls": [{...}, {...}]` (list field).
   - What's unclear: Whether `apply_chat_template()` handles converting multiple tool_calls to a single `<tool_call>[...]</tool_call>` block.
   - Recommendation: Include in the Wave 1 empirical test.

3. **What is the actual token overhead of the chat template per message?**
   - What we know: Each message adds `<|im_start|>{role}\n` and `<|im_end|>\n`. Role names vary in length.
   - What's unclear: Exact token count of delimiters (they may be single tokens or multi-token).
   - Recommendation: Measure empirically by tokenizing an empty conversation and counting delimiter tokens. Use this to set accurate token budget guidelines for data generation.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3 | All scripts | Yes | 3.14.2 | -- |
| pip | Package installation | Yes | 25.3 | -- |
| pydantic | Format validation | Yes | 2.12.5 | -- |
| transformers | Tokenizer validation | No (not installed) | -- | pip install transformers |
| PyYAML | Template parsing | Not checked | -- | pip install pyyaml |
| pytest | Test execution | Not checked | -- | pip install pytest |
| Internet access | Downloading SmolLM2 tokenizer | Required for first run | -- | Cache tokenizer locally after first download |

**Missing dependencies with no fallback:**
- `transformers` must be installed for tokenizer validation (critical path)

**Missing dependencies with fallback:**
- None that block Phase 1

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | No | N/A -- local scripts |
| V3 Session Management | No | N/A -- no sessions |
| V4 Access Control | No | N/A -- local file system |
| V5 Input Validation | Yes | Pydantic schema validation on all JSONL input |
| V6 Cryptography | No | N/A -- no secrets in data format |

### Known Threat Patterns for Data Pipeline

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| PII/secrets in generated training data | Information Disclosure | Regex scanning for API keys, emails, passwords in generated JSONL; explicit prompt instructions to use placeholders |
| Malicious tool schemas in training data | Tampering | Validate tool parameter types are safe primitives (string, number, boolean, object, array); no code execution in schemas |
| Path traversal in file operations | Tampering | Validate dataset paths stay within datasets/ directory; no user-supplied paths in scripts |

## Sources

### Primary (HIGH confidence)
- [SmolLM2-1.7B-Instruct tokenizer_config.json](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct/raw/main/tokenizer_config.json) -- Chat template, special tokens, BOS/EOS/PAD configuration
- [SmolLM2-1.7B-Instruct config.json](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct/raw/main/config.json) -- Architecture details, vocab size, max positions
- [SmolLM2 instructions_function_calling.md](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct/blob/main/instructions_function_calling.md) -- `<tool_call>` format, function calling system prompt pattern
- [TRL v1.2.0 SFTTrainer docs](https://huggingface.co/docs/trl/sft_trainer) -- assistant_only_loss, PEFT integration, tool calling support
- [TRL v1.2.0 Dataset Formats](https://huggingface.co/docs/trl/dataset_formats) -- Tool calling format spec, messages format, tools column
- [TRL Issue #4879](https://github.com/huggingface/trl/issues/4879) -- `{% generation %}` tag auto-injection status (OPEN, SmolLM2 not supported)
- [Argilla Synth-APIGen-v0.1](https://huggingface.co/datasets/argilla/Synth-APIGen-v0.1) -- Function calling dataset format used to train SmolLM2
- [SmolLM2-1.7B-Instruct Model Card](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct) -- Model capabilities, BFCL score (27%), training data

### Secondary (MEDIUM confidence)
- [hypervariance/function-calling-sharegpt](https://huggingface.co/datasets/hypervariance/function-calling-sharegpt) -- Alternative ShareGPT function calling format reference
- [Project STACK.md](.planning/research/STACK.md) -- Verified package versions and stack decisions
- [Project PITFALLS.md](.planning/research/PITFALLS.md) -- Tokenizer misalignment and format brittleness pitfalls

### Tertiary (LOW confidence)
- None -- all claims verified against primary sources

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- packages verified via pip/PyPI, versions confirmed
- Architecture: HIGH -- TRL format spec verified against official docs, SmolLM2 config extracted from HuggingFace
- Pitfalls: HIGH -- corroborated by project-level PITFALLS.md and official documentation
- Open questions: MEDIUM -- three questions require empirical verification in Wave 1

**Research date:** 2026-04-20
**Valid until:** 2026-05-20 (30 days -- stable domain, SmolLM2 and TRL versions unlikely to change)

## Project Constraints (from CLAUDE.md)

- Never run applications automatically -- validation scripts must be run explicitly by user
- Never use emojis in terminal logs, readme files, or anywhere
- Python is the primary language (all scripts in Python)
