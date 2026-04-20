# Lyra ShareGPT Format Specification

**Version:** 1.0
**Last updated:** 2026-04-20
**Status:** Canonical -- this is the single source of truth for all data generation

## Overview

Lyra uses a TRL-native conversational format with `messages`/`role`/`content` keys. This is NOT the classic ShareGPT `from`/`value` format. TRL v1.2.0 SFTTrainer expects this format natively, eliminating conversion steps at training time.

All training data generated in Phases 4-6 MUST conform to this specification. The `scripts/validate_format.py` script enforces these rules programmatically via Pydantic validation.

## Format Structure

Each line in a JSONL file is a JSON object representing one conversation:

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "tools": [...]
}
```

### Fields

#### `messages` (required)

A list of message objects. Each message has:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `role` | string | yes | One of: `system`, `user`, `assistant`, `tool` |
| `content` | string or null | no | The message text. Can be null for assistant messages with tool_calls. |
| `tool_calls` | list[ToolCall] | no | Tool calls made by the assistant. Only valid on assistant role. |
| `name` | string | conditional | Required for `tool` role messages. Must match the called tool name. |

#### `tools` (optional)

A list of tool schema objects defining available tools. Each tool has:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | string | yes | Always `"function"` |
| `function` | object | yes | Object with `name` (string), `description` (string), `parameters` (object -- JSON Schema format) |

#### ToolCall object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | string | yes | Always `"function"` |
| `function` | object | yes | Object with `name` (string) and `arguments` (object) |

## Role Ordering Rules

The following rules are enforced by the validation script. Violations produce specific error messages.

1. **First message MUST be system role.** Every conversation starts with an explicit system prompt to prevent SmolLM2's default system prompt injection. Error: `"First message must be system role"`

2. **After system, messages alternate user/assistant** (with tool interludes). The general pattern is system -> user -> assistant -> user -> assistant.

3. **`tool` messages MUST immediately follow an `assistant` message that contains `tool_calls`**, or follow another `tool` message (for parallel calls). Error: `"Tool message at index {i} must follow assistant or tool"`

4. **Multiple `tool` messages can follow a single `assistant`** with multiple `tool_calls` (parallel execution pattern).

5. **An `assistant` message with `tool_calls` must be followed by exactly one `tool` message per tool call.** If the assistant makes 2 tool calls, exactly 2 tool messages must follow. Error: `"Expected {n} tool messages after assistant at index {i}, got {actual}"`

6. **The final message SHOULD be assistant role** -- the model's last response.

7. **`assistant` messages with `tool_calls` typically have `content` set to `null` or empty string.** Content alongside tool_calls is permitted but unusual.

8. **`tool` messages MUST have a `name` field** matching the tool that was called. Error: `"Tool message at index {i} missing 'name' field"`

9. **`tool` message `content` should be the string representation** of the tool's return value (typically JSON-serialized).

Additional validation:
- Empty messages list is rejected. Error: `"Empty conversation"`
- Invalid role strings are rejected. Error: `"Invalid role '{role}' at index {i}"`
- If a `tools` column is present, all `tool_calls` must reference tool names defined in the `tools` list. Error: `"Tool call '{name}' not in defined tools: {defined_names}"`

## Tool Call Patterns

Lyra's training data covers five distinct tool calling patterns:

### Pattern 1: Single Function Call

A single tool invocation with result.

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

### Pattern 2: Multi-Turn with Results

Multiple rounds of user questions, tool calls, and follow-up responses.

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant with access to tools."},
    {"role": "user", "content": "Look up the stock price for AAPL."},
    {"role": "assistant", "tool_calls": [
      {"type": "function", "function": {"name": "get_stock_price", "arguments": {"symbol": "AAPL"}}}
    ]},
    {"role": "tool", "name": "get_stock_price", "content": "{\"price\": 185.42, \"currency\": \"USD\"}"},
    {"role": "assistant", "content": "AAPL is trading at $185.42 USD."},
    {"role": "user", "content": "What about GOOGL?"},
    {"role": "assistant", "tool_calls": [
      {"type": "function", "function": {"name": "get_stock_price", "arguments": {"symbol": "GOOGL"}}}
    ]},
    {"role": "tool", "name": "get_stock_price", "content": "{\"price\": 142.58, \"currency\": \"USD\"}"},
    {"role": "assistant", "content": "GOOGL is trading at $142.58 USD."}
  ],
  "tools": [
    {"type": "function", "function": {
      "name": "get_stock_price",
      "description": "Get the current stock price for a given ticker symbol",
      "parameters": {"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]}
    }}
  ]
}
```

### Pattern 3: Parallel Execution

Assistant makes multiple tool calls in a single turn, followed by matching tool responses.

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

### Pattern 4: MCP-Style (Server Discovery and Tool Invocation)

Simulates Model Context Protocol patterns where the assistant discovers available servers, lists their tools, then invokes them.

```json
{
  "messages": [
    {"role": "system", "content": "You are an assistant that can discover and use MCP tool servers."},
    {"role": "user", "content": "Connect to the database server and list all users."},
    {"role": "assistant", "tool_calls": [
      {"type": "function", "function": {"name": "mcp_list_tools", "arguments": {"server": "database"}}}
    ]},
    {"role": "tool", "name": "mcp_list_tools", "content": "[{\"name\": \"query_users\", \"description\": \"Query the users table\"}, {\"name\": \"query_orders\", \"description\": \"Query the orders table\"}]"},
    {"role": "assistant", "tool_calls": [
      {"type": "function", "function": {"name": "query_users", "arguments": {"limit": 10}}}
    ]},
    {"role": "tool", "name": "query_users", "content": "[{\"id\": 1, \"name\": \"Alice\"}, {\"id\": 2, \"name\": \"Bob\"}]"},
    {"role": "assistant", "content": "The database has 2 users: Alice and Bob."}
  ],
  "tools": [
    {"type": "function", "function": {
      "name": "mcp_list_tools",
      "description": "List available tools from an MCP server",
      "parameters": {"type": "object", "properties": {"server": {"type": "string"}}, "required": ["server"]}
    }},
    {"type": "function", "function": {
      "name": "query_users",
      "description": "Query the users table",
      "parameters": {"type": "object", "properties": {"limit": {"type": "integer"}}}
    }}
  ]
}
```

### Pattern 5: CLI/Shell Commands

Tool use for executing shell commands and processing output.

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant that can run shell commands."},
    {"role": "user", "content": "List all Python files in the current directory."},
    {"role": "assistant", "tool_calls": [
      {"type": "function", "function": {"name": "run_command", "arguments": {"command": "find . -name '*.py' -type f"}}}
    ]},
    {"role": "tool", "name": "run_command", "content": "./main.py\n./utils/helpers.py\n./tests/test_main.py"},
    {"role": "assistant", "content": "There are 3 Python files:\n- ./main.py\n- ./utils/helpers.py\n- ./tests/test_main.py"}
  ],
  "tools": [
    {"type": "function", "function": {
      "name": "run_command",
      "description": "Execute a shell command and return its output",
      "parameters": {"type": "object", "properties": {"command": {"type": "string", "description": "The shell command to execute"}}, "required": ["command"]}
    }}
  ]
}
```

## Token Budget

Maximum **2048 tokens** per conversation AFTER `apply_chat_template(tokenize=True)`. This matches SmolLM2-1.7B's native training sequence length.

Token budget breakdown:
- System prompt: ~200-500 tokens
- Tool definitions (for tool-calling samples): ~500-1500 tokens
- Conversation turns: ~1000-4000 tokens
- Reserve for generation buffer: ~1000 tokens
- Practical maximum per training sample: ~4000-6000 tokens total (raw), fitting within 2048 after tokenization overhead

Length distribution is natural -- samples range from ~200 to ~1800 tokens based on task complexity. No artificial length targeting.

## Storage

Training data is stored as JSONL files (one conversation per line) in the `datasets/` directory with domain separation:

```
datasets/
  tool-calling/    # Function calling, MCP, CLI samples
  code/            # Code generation, debugging, utilities
  knowledge/       # Reasoning, Q&A, explanations
```

## Important Notes

### PAD/EOS Token Collision

SmolLM2-1.7B uses the same token (id=2, `<|im_end|>`) for both PAD and EOS. This means:
- Padding tokens and end-of-sequence tokens are indistinguishable by token ID
- During training (Phase 8), PAD tokens must be masked (label=-100) while the final EOS must NOT be masked
- Failure to handle this correctly results in models that never stop generating

### Default System Prompt Injection

SmolLM2's chat template auto-injects "You are a helpful AI assistant named SmolLM, trained by Hugging Face" when no system message is provided. To avoid this:
- Every conversation MUST start with an explicit system role message
- The validation script rejects conversations without a system first message

### Legacy Terminology Mapping

| Legacy Term | Lyra Equivalent | Notes |
|-------------|-----------------|-------|
| `function_call` role | `assistant` with `tool_calls` field | Do NOT use `function_call` as a role |
| `observation` role | `tool` role | Do NOT use `observation` as a role |
| `from`/`value` keys | `role`/`content` keys | Classic ShareGPT format is NOT used |
| `conversations` key | `messages` key | TRL-native uses `messages` |

### Chat Template Format

When `apply_chat_template()` processes a conversation, each message becomes:
```
<|im_start|>{role}
{content}<|im_end|>
```

Tool calls are converted to SmolLM2's native format:
```
<tool_call>[{"name": "func_name", "arguments": {"arg": "value"}}]</tool_call>
```
