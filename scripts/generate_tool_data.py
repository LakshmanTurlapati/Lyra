#!/usr/bin/env python3
"""generate_tool_data.py -- Generate tool-calling training data batches.

Produces JSONL batches of tool-calling conversations across 5 categories:
single-call, CLI, multi-turn, parallel, and MCP patterns.

Per D-07: category batches with validation loops.
Per D-08: 50 samples per batch, one JSONL file per batch.
"""
import argparse
import json
import random
import sys
from pathlib import Path
from typing import Optional

import yaml

from scripts.validate_format import Conversation


# --- Constants ---

PROJECT_ROOT = Path(__file__).parent.parent
SCHEMA_POOL_PATH = PROJECT_ROOT / "datasets" / "tool-calling" / "tool_schemas.yaml"
SYSTEM_PROMPTS_PATH = PROJECT_ROOT / "templates" / "system-prompts.yaml"
OUTPUT_DIR = PROJECT_ROOT / "datasets" / "tool-calling"

VALID_CATEGORIES = ["single-call", "cli", "multi-turn", "parallel", "mcp"]
MAX_COUNT = 10000  # Sanity check per T-04-04


# --- Schema Pool Loading ---


def load_schemas(pool_path: Path = None) -> dict:
    """Load tool schemas from YAML pool file.

    Returns the 'schemas' dict from tool_schemas.yaml with structure:
    {domain: {subcategory: [schema_list]}} or {domain: [schema_list]}
    """
    path = pool_path or SCHEMA_POOL_PATH
    data = yaml.safe_load(path.read_text())
    return data["schemas"]


def load_system_prompts(prompts_path: Path = None) -> dict:
    """Load system prompts from YAML file.

    Returns dict mapping prompt_id -> content string.
    """
    path = prompts_path or SYSTEM_PROMPTS_PATH
    data = yaml.safe_load(path.read_text())
    prompts = {}
    for key, val in data["system_prompts"].items():
        prompts[key] = val["content"].strip()
    return prompts


def get_tools_for_category(schemas: dict, category: str, rng: random.Random,
                           count: int = 2) -> list[dict]:
    """Select tools appropriate for a category from the pool.

    Returns list of tool dicts in OpenAI format:
    [{"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}]
    """
    if category == "cli":
        cli_schemas = schemas.get("cli", [])
        return [_schema_to_tool(s) for s in cli_schemas]
    elif category == "mcp":
        mcp_schemas = schemas.get("mcp_meta", [])
        return [_schema_to_tool(s) for s in mcp_schemas]
    else:
        # Draw from developer and everyday pools
        all_schemas = []
        for domain_key in ["developer", "everyday"]:
            domain = schemas.get(domain_key, {})
            if isinstance(domain, dict):
                for subcat_list in domain.values():
                    all_schemas.extend(subcat_list)
            elif isinstance(domain, list):
                all_schemas.extend(domain)
        selected = rng.sample(all_schemas, min(count, len(all_schemas)))
        return [_schema_to_tool(s) for s in selected]


def _schema_to_tool(schema: dict) -> dict:
    """Convert a raw schema dict to OpenAI tool format."""
    return {
        "type": "function",
        "function": {
            "name": schema["name"],
            "description": schema["description"],
            "parameters": schema["parameters"],
        }
    }


# --- Topic/Query Templates ---

SINGLE_CALL_QUERIES = [
    "What is the weather in {city}?",
    "Look up the definition of {word}",
    "Search the web for {topic}",
    "Calculate {expression}",
    "Get the temperature in {city} in {unit}",
    "How far is it from {origin} to {destination}?",
    "Translate '{text}' to {language}",
    "What events do I have on {date}?",
    "Get a summary of {topic} from Wikipedia",
    "Check my inbox for recent emails",
    "Find {pattern} files in {directory}",
    "What are the current weather conditions in {city}?",
    "Read the contents of {path}",
    "Show me system health status",
    "List all tables in the {database} database",
    "Get the forecast for {city} for the next {days} days",
    "Send an email to {recipient} about {subject}",
    "What is {value} {from_unit} in {to_unit}?",
    "Get directions from {origin} to {destination}",
    "Search packages for {query} on {registry}",
    "What is the CPU usage on {host}?",
    "Describe the columns in the {table} table",
    "Get metrics for {service} on {metric}",
    "List active alerts at {severity} level",
    "Check if {url} is responding",
    "Download the file at {url}",
    "Get details about the {package} package",
    "Create a calendar event called {title} on {date}",
    "Delete the event with ID {event_id}",
    "Search my emails for {query}",
    "What is the stock of {package} in {registry}?",
    "Show me recently installed packages",
    "List files in {path}",
    "Get information about {path}",
    "What is the current time in {timezone}?",
    "Geocode the address {address}",
    "How is the weather looking in {city} today?",
    "Find the distance between {origin} and {destination}",
    "Convert {value} {from_unit} to {to_unit}",
    "Search for {query} online",
]

CLI_QUERIES = [
    "Show me the git status of this repository",
    "List all Python files in the current directory",
    "Find all files modified in the last {n} days",
    "Show me the last {n} git commits",
    "Check disk usage in the current directory",
    "Count the number of lines in {filename}",
    "Show me running processes that match {name}",
    "Find all TODO comments in the codebase",
    "Show the git diff for the last commit",
    "List environment variables containing {keyword}",
    "Check if port {port} is in use",
    "Show me the directory tree structure",
    "Find all files larger than {size}",
    "Show the contents of the {filename} file",
    "Get the current working directory",
    "Show git branches sorted by last commit date",
    "Count how many test files exist in the project",
    "Show the first {n} lines of {filename}",
    "Find all JSON files in {directory}",
    "Check the Node.js version installed",
    "Show the git log for {filename}",
    "List all Docker containers running",
    "Find duplicate files in {directory}",
    "Show permissions for {filename}",
    "Check which Python packages are installed",
    "Show the network interfaces",
    "Find all files containing {pattern}",
    "Compress {directory} into a tar archive",
    "Show memory usage on this system",
    "List all cron jobs configured",
]

MULTI_TURN_QUERIES = [
    ("Search for users named {name} in the database", "Get the details for the first result"),
    ("List all tables in the {database} database", "Describe the {table} table"),
    ("Search packages for {query}", "Install the {package} package"),
    ("Check the weather in {city}", "How about the forecast for the next {days} days?"),
    ("Search the web for {topic}", "Get the Wikipedia summary for that topic"),
    ("List my calendar events for today", "Create a meeting called {title} for tomorrow"),
    ("Show me the system health", "Get the CPU metrics for that service"),
    ("List available cloud instances", "Get the logs for the {service} service"),
    ("Search my emails for {query}", "Send a reply to {recipient} about {subject}"),
    ("Check the status of {url}", "Download the file from that URL"),
    ("Get the weather in {city}", "What about {city2}? Compare them"),
    ("List files in {directory}", "Read the contents of {filename}"),
    ("Show active monitoring alerts", "Get details on the {severity} alerts"),
    ("List installed packages", "Get info about {package}"),
    ("Search for {query} online", "Translate the summary to {language}"),
]

PARALLEL_QUERIES = [
    "Compare the weather in {city1} and {city2}",
    "Get the weather and temperature in {city}",
    "Look up directions and distance from {origin} to {destination}",
    "Check the status of {url1} and {url2}",
    "Get the forecast for {city1} and {city2} for tomorrow",
    "Search for {query1} and {query2} on the web",
    "Calculate {expr1} and {expr2}",
    "Get metrics for {service} on cpu and memory",
    "Check weather in {city1}, {city2}, and {city3}",
    "Translate '{text}' to {lang1} and {lang2}",
    "Get the definition of {word1} and {word2}",
    "Download files from {url1} and {url2}",
    "List events for today and tomorrow",
    "Search emails for {query1} and {query2}",
    "Get system health and active alerts",
]

MCP_QUERIES = [
    "I need to query the database for user records",
    "Help me find and read a configuration file",
    "I want to check the monitoring alerts",
    "Can you look up package information from the registry?",
    "I need to deploy the service to staging",
    "Help me search for documents in the knowledge base",
    "Can you check the cloud instances and their status?",
    "I need to send a notification via the messaging system",
    "Help me generate a report from the analytics service",
    "I want to run a health check on all services",
    "Can you look up weather data from the external API?",
    "I need to process some files using the data pipeline",
    "Help me check the build status",
    "I want to query logs from the logging service",
    "Can you update the cache using the cache manager?",
]

# --- Value Pools for Template Filling ---

CITIES = ["San Francisco", "New York", "London", "Tokyo", "Paris", "Berlin",
          "Sydney", "Toronto", "Mumbai", "Singapore", "Seoul", "Chicago"]
WORDS = ["ephemeral", "paradigm", "algorithm", "entropy", "recursive",
         "polymorphism", "idempotent", "orthogonal", "heuristic", "cognition"]
TOPICS = ["machine learning", "climate change", "quantum computing",
          "renewable energy", "artificial intelligence", "blockchain",
          "space exploration", "gene editing", "cybersecurity", "robotics"]
EXPRESSIONS = ["2 + 2", "sqrt(144)", "15 * 7", "1024 / 8", "2^10",
               "sin(pi/2)", "log(1000)", "45 + 67 - 12", "100 * 0.15", "3.14 * 4^2"]
PATHS = ["/home/user/project/main.py", "/etc/config.yaml", "src/utils.ts",
         "README.md", "package.json", "requirements.txt", "Makefile"]
DIRECTORIES = ["/home/user/project", "src/", "tests/", "docs/", "/var/log"]
DATABASES = ["users_db", "analytics", "products", "inventory", "logs"]
TABLES = ["users", "orders", "products", "sessions", "events"]
PACKAGES = ["requests", "flask", "numpy", "pandas", "fastapi", "pydantic"]
REGISTRIES = ["pypi", "npm", "cargo"]
SERVICES = ["api-gateway", "auth-service", "worker", "web-frontend", "cache"]
URLS = ["https://api.example.com/health", "https://cdn.example.com/data.json",
        "https://status.example.com", "https://docs.example.com/api"]
HOSTS = ["prod-server-1", "staging-01", "worker-node-3", "db-primary"]
LANGUAGES = ["Spanish", "French", "German", "Japanese", "Chinese"]
UNITS_FROM = ["kilometers", "pounds", "celsius", "meters", "gallons"]
UNITS_TO = ["miles", "kilograms", "fahrenheit", "feet", "liters"]
FILENAMES = ["app.py", "config.json", "server.log", "data.csv", "index.html"]
SEVERITIES = ["critical", "warning", "info"]
METRICS = ["cpu", "memory", "latency", "error_rate"]

# MCP server simulation data
MCP_SERVERS = [
    {"name": "database", "tools": ["query_database", "list_tables", "describe_table"]},
    {"name": "filesystem", "tools": ["read_file", "write_file", "list_directory"]},
    {"name": "monitoring", "tools": ["get_system_health", "list_alerts", "get_metrics"]},
    {"name": "cloud", "tools": ["list_instances", "deploy_service", "get_service_logs"]},
    {"name": "packages", "tools": ["search_packages", "install_package", "get_package_info"]},
    {"name": "search", "tools": ["search_web", "get_definition", "get_wikipedia_summary"]},
]


# --- Realistic Response Generation ---


def generate_tool_response(tool_name: str, arguments: dict, rng: random.Random,
                           error: bool = False) -> str:
    """Generate a realistic JSON string response for a tool call.

    When error=True, generates error responses for edge cases.
    Returns JSON-serialized string (tool content must be string per format spec).
    """
    if error:
        error_responses = [
            json.dumps({"error": "Not found", "code": 404}),
            json.dumps({"error": "Connection timeout", "code": 408}),
            json.dumps({"error": "Permission denied", "code": 403}),
            json.dumps({"error": "Rate limit exceeded", "code": 429}),
            json.dumps({"error": f"Invalid argument: {list(arguments.keys())[0] if arguments else 'unknown'}"}),
        ]
        return rng.choice(error_responses)

    # Generate varied responses based on tool name
    responses = {
        "get_weather": lambda: json.dumps({"temperature": rng.randint(20, 95),
                                           "condition": rng.choice(["sunny", "cloudy", "rainy", "foggy", "windy"]),
                                           "humidity": rng.randint(30, 90)}),
        "get_forecast": lambda: json.dumps({"days": [
            {"day": f"Day {i+1}", "high": rng.randint(50, 90), "low": rng.randint(30, 60),
             "condition": rng.choice(["sunny", "cloudy", "rainy"])}
            for i in range(rng.randint(2, 4))
        ]}),
        "get_temperature": lambda: json.dumps({"temperature": rng.randint(20, 95),
                                                "unit": arguments.get("unit", "fahrenheit")}),
        "calculate": lambda: json.dumps({"result": rng.uniform(1, 1000).__round__(2)}),
        "convert_units": lambda: json.dumps({"result": rng.uniform(0.1, 500).__round__(2),
                                              "from": arguments.get("from_unit", ""),
                                              "to": arguments.get("to_unit", "")}),
        "search_web": lambda: json.dumps({"results": [
            {"title": f"Result {i+1}: {arguments.get('query', 'topic')}",
             "url": f"https://example.com/result{i+1}",
             "snippet": f"Information about {arguments.get('query', 'topic')}..."}
            for i in range(3)
        ]}),
        "get_definition": lambda: json.dumps({"word": arguments.get("word", ""),
                                               "definition": f"A term used in formal contexts.",
                                               "part_of_speech": "noun"}),
        "translate_text": lambda: json.dumps({"translation": f"[Translated text to {arguments.get('target_language', 'unknown')}]",
                                               "language": arguments.get("target_language", "")}),
        "get_wikipedia_summary": lambda: json.dumps({"title": arguments.get("topic", ""),
                                                      "summary": f"Brief encyclopedia entry about {arguments.get('topic', 'the topic')}."}),
        "query_database": lambda: json.dumps({"rows": [{"id": i, "name": f"Record {i}"} for i in range(1, 4)],
                                              "count": 3}),
        "list_tables": lambda: json.dumps({"tables": ["users", "orders", "products", "sessions"]}),
        "describe_table": lambda: json.dumps({"columns": [
            {"name": "id", "type": "integer"}, {"name": "name", "type": "varchar"},
            {"name": "created_at", "type": "timestamp"}
        ]}),
        "read_file": lambda: f"# File contents\nLine 1 of the file\nLine 2 of the file\nLine 3 of the file",
        "write_file": lambda: json.dumps({"status": "success", "bytes_written": rng.randint(50, 5000)}),
        "list_directory": lambda: json.dumps({"entries": ["file1.py", "file2.js", "README.md", "config.yaml"]}),
        "get_file_info": lambda: json.dumps({"size": rng.randint(100, 50000), "modified": "2026-04-15T10:30:00Z",
                                              "type": "file"}),
        "search_files": lambda: json.dumps({"matches": [f"src/file{i}.py" for i in range(1, 4)]}),
        "http_get": lambda: json.dumps({"status": 200, "body": {"message": "OK"}}),
        "http_post": lambda: json.dumps({"status": 201, "body": {"id": rng.randint(1, 999)}}),
        "check_endpoint_status": lambda: json.dumps({"status": "up", "response_time_ms": rng.randint(10, 500)}),
        "download_file": lambda: json.dumps({"status": "success", "size_bytes": rng.randint(1000, 100000)}),
        "fetch_json": lambda: json.dumps({"data": {"key": "value", "count": rng.randint(1, 100)}}),
        "git_status": lambda: "On branch main\nChanges not staged for commit:\n  modified: src/app.py",
        "git_diff": lambda: "diff --git a/src/app.py\n+ added line\n- removed line",
        "git_log": lambda: json.dumps({"commits": [
            {"hash": f"abc{i}def", "message": f"Commit message {i}", "author": "dev"}
            for i in range(1, 4)
        ]}),
        "git_commit": lambda: json.dumps({"hash": "abc1234", "message": arguments.get("message", "")}),
        "git_branch": lambda: json.dumps({"branches": ["main", "develop", "feature/new-api"]}),
        "search_packages": lambda: json.dumps({"packages": [
            {"name": f"pkg-{i}", "version": f"1.{i}.0", "description": f"A useful package"}
            for i in range(1, 4)
        ]}),
        "install_package": lambda: json.dumps({"status": "installed",
                                                "name": arguments.get("name", ""),
                                                "version": arguments.get("version", "latest")}),
        "get_package_info": lambda: json.dumps({"name": arguments.get("name", ""),
                                                 "version": "2.1.0",
                                                 "dependencies": ["dep1", "dep2"]}),
        "list_installed_packages": lambda: json.dumps({"packages": [
            {"name": "pydantic", "version": "2.7.0"},
            {"name": "requests", "version": "2.31.0"},
            {"name": "pytest", "version": "9.0.2"},
        ]}),
        "list_instances": lambda: json.dumps({"instances": [
            {"id": f"i-{rng.randint(1000,9999)}", "state": "running", "type": "t3.medium"}
        ]}),
        "deploy_service": lambda: json.dumps({"status": "deployed", "url": f"https://{arguments.get('service', 'app')}.example.com"}),
        "get_service_logs": lambda: f"[INFO] Service started\n[INFO] Handling request\n[INFO] Response sent 200",
        "scale_service": lambda: json.dumps({"status": "scaled", "replicas": arguments.get("replicas", 2)}),
        "get_system_health": lambda: json.dumps({"status": "healthy", "uptime": "14d 3h 22m"}),
        "list_alerts": lambda: json.dumps({"alerts": [
            {"severity": "warning", "message": "High memory usage on worker-3"}
        ]}),
        "get_metrics": lambda: json.dumps({"service": arguments.get("service", ""),
                                            "metric": arguments.get("metric", "cpu"),
                                            "value": rng.uniform(10, 90).__round__(1),
                                            "unit": "%"}),
        "get_cpu_usage": lambda: json.dumps({"host": arguments.get("host", ""),
                                              "cpu_percent": rng.uniform(5, 95).__round__(1)}),
        "create_event": lambda: json.dumps({"id": f"evt-{rng.randint(100,999)}",
                                             "title": arguments.get("title", ""),
                                             "date": arguments.get("date", "")}),
        "list_events": lambda: json.dumps({"events": [
            {"id": "evt-101", "title": "Team standup", "time": "09:00"},
            {"id": "evt-102", "title": "Code review", "time": "14:00"},
        ]}),
        "delete_event": lambda: json.dumps({"status": "deleted", "id": arguments.get("event_id", "")}),
        "send_email": lambda: json.dumps({"status": "sent", "message_id": f"msg-{rng.randint(1000,9999)}"}),
        "search_emails": lambda: json.dumps({"results": [
            {"from": "alice@example.com", "subject": "Re: Project update", "date": "2026-04-19"}
        ]}),
        "get_inbox": lambda: json.dumps({"emails": [
            {"from": "bob@example.com", "subject": "Meeting tomorrow", "unread": True},
            {"from": "hr@company.com", "subject": "Policy update", "unread": False},
        ]}),
        "get_directions": lambda: json.dumps({"distance": f"{rng.randint(1,500)} km",
                                               "duration": f"{rng.randint(10, 600)} min",
                                               "route": "via Highway 101"}),
        "geocode": lambda: json.dumps({"lat": rng.uniform(-90, 90).__round__(4),
                                        "lng": rng.uniform(-180, 180).__round__(4)}),
        "get_distance": lambda: json.dumps({"distance_km": rng.uniform(1, 1000).__round__(1)}),
        "run_command": lambda: _generate_cli_output(arguments.get("command", ""), rng),
        "run_command_in_dir": lambda: _generate_cli_output(arguments.get("command", ""), rng),
        "mcp_list_servers": lambda: json.dumps({"servers": [
            {"name": s["name"], "status": "connected"} for s in MCP_SERVERS
        ]}),
        "mcp_list_tools": lambda: json.dumps({"tools": [
            {"name": t, "description": f"Tool for {t.replace('_', ' ')}"}
            for t in _get_mcp_server_tools(arguments.get("server", ""), rng)
        ]}),
        "mcp_invoke_tool": lambda: json.dumps({"result": "Operation completed successfully",
                                                "data": {"status": "ok"}}),
    }

    generator = responses.get(tool_name)
    if generator:
        return generator()
    # Fallback for unknown tools
    return json.dumps({"result": "success", "data": {}})


def _generate_cli_output(command: str, rng: random.Random) -> str:
    """Generate realistic CLI output based on the command."""
    if "ls" in command or "find" in command or "tree" in command:
        files = ["main.py", "utils.py", "config.yaml", "README.md", "tests/"]
        return "\n".join(rng.sample(files, min(rng.randint(2, 5), len(files))))
    elif "git status" in command:
        return "On branch main\nYour branch is up to date.\n\nnothing to commit, working tree clean"
    elif "git log" in command:
        return "\n".join([f"abc{i}def Fix: update module {i}" for i in range(1, 4)])
    elif "git diff" in command:
        return "diff --git a/file.py b/file.py\n-old line\n+new line"
    elif "wc" in command or "count" in command:
        return str(rng.randint(10, 5000))
    elif "ps" in command or "process" in command:
        return f"PID   CMD\n{rng.randint(1000,9999)}  python app.py\n{rng.randint(1000,9999)}  node server.js"
    elif "du" in command or "disk" in command:
        return f"4.2M\t./src\n1.1M\t./tests\n256K\t./docs\n5.6M\ttotal"
    elif "grep" in command:
        return f"src/app.py:42: # TODO: implement caching\nsrc/utils.py:17: # TODO: add tests"
    elif "cat" in command or "head" in command or "tail" in command:
        return "Line 1: import sys\nLine 2: from pathlib import Path\nLine 3:\nLine 4: def main():"
    elif "pwd" in command:
        return "/home/user/project"
    elif "docker" in command:
        return f"CONTAINER ID  IMAGE         STATUS\nabc123        web:latest    Up 2 hours"
    elif "pip" in command or "npm" in command:
        return "Package    Version\npydantic   2.7.0\nrequests   2.31.0"
    else:
        return f"Command executed successfully.\nOutput: {rng.randint(0, 100)} items processed."


def _get_mcp_server_tools(server_name: str, rng: random.Random) -> list[str]:
    """Get tools for an MCP server by name."""
    for server in MCP_SERVERS:
        if server["name"] == server_name:
            return server["tools"]
    # Fallback
    return rng.choice(MCP_SERVERS)["tools"]


def generate_assistant_summary(tool_name: str, tool_response: str, query: str,
                               rng: random.Random) -> str:
    """Generate a natural assistant message summarizing tool results."""
    # Parse tool response if JSON
    try:
        data = json.loads(tool_response)
    except (json.JSONDecodeError, TypeError):
        data = tool_response

    prefixes = [
        "Based on the results,",
        "Here's what I found:",
        "The results show that",
        "",
    ]
    prefix = rng.choice(prefixes)

    if isinstance(data, dict):
        if "error" in data:
            return f"I encountered an error: {data['error']}. Let me know if you'd like me to try a different approach."
        if "temperature" in data:
            return f"{prefix} the temperature is {data['temperature']} degrees and it's {data.get('condition', 'clear')}.".strip()
        if "result" in data:
            return f"{prefix} the result is {data['result']}.".strip()
        if "results" in data and isinstance(data["results"], list):
            count = len(data["results"])
            return f"{prefix} I found {count} results for your query.".strip()
        if "status" in data:
            return f"{prefix} the operation completed with status: {data['status']}.".strip()
    if isinstance(data, str):
        lines = data.strip().split("\n")
        if len(lines) > 3:
            return f"{prefix} here are the results:\n{data}".strip()
        return f"{prefix} {data}".strip()

    return f"I've completed the task. {prefix} the operation was successful.".strip()


# --- Category Batch Generators ---


def _fill_template(template: str, rng: random.Random) -> str:
    """Fill placeholder variables in a query template."""
    replacements = {
        "{city}": rng.choice(CITIES),
        "{city1}": rng.choice(CITIES),
        "{city2}": rng.choice(CITIES[:6]),
        "{city3}": rng.choice(CITIES[6:]),
        "{word}": rng.choice(WORDS),
        "{word1}": rng.choice(WORDS[:5]),
        "{word2}": rng.choice(WORDS[5:]),
        "{topic}": rng.choice(TOPICS),
        "{expression}": rng.choice(EXPRESSIONS),
        "{expr1}": rng.choice(EXPRESSIONS[:5]),
        "{expr2}": rng.choice(EXPRESSIONS[5:]),
        "{origin}": rng.choice(CITIES[:6]),
        "{destination}": rng.choice(CITIES[6:]),
        "{text}": rng.choice(["hello world", "good morning", "thank you", "how are you"]),
        "{language}": rng.choice(LANGUAGES),
        "{lang1}": rng.choice(LANGUAGES[:3]),
        "{lang2}": rng.choice(LANGUAGES[3:]),
        "{date}": f"2026-04-{rng.randint(1,30):02d}",
        "{days}": str(rng.randint(2, 5)),
        "{path}": rng.choice(PATHS),
        "{directory}": rng.choice(DIRECTORIES),
        "{database}": rng.choice(DATABASES),
        "{table}": rng.choice(TABLES),
        "{package}": rng.choice(PACKAGES),
        "{query}": rng.choice(TOPICS),
        "{query1}": rng.choice(TOPICS[:5]),
        "{query2}": rng.choice(TOPICS[5:]),
        "{service}": rng.choice(SERVICES),
        "{url}": rng.choice(URLS),
        "{url1}": rng.choice(URLS[:2]),
        "{url2}": rng.choice(URLS[2:]),
        "{host}": rng.choice(HOSTS),
        "{registry}": rng.choice(REGISTRIES),
        "{value}": str(rng.randint(1, 100)),
        "{from_unit}": rng.choice(UNITS_FROM),
        "{to_unit}": rng.choice(UNITS_TO),
        "{unit}": rng.choice(["celsius", "fahrenheit"]),
        "{recipient}": f"{rng.choice(['alice', 'bob', 'carol'])}@example.com",
        "{subject}": rng.choice(["Project update", "Meeting notes", "Quick question"]),
        "{event_id}": f"evt-{rng.randint(100,999)}",
        "{n}": str(rng.randint(3, 10)),
        "{filename}": rng.choice(FILENAMES),
        "{name}": rng.choice(["Alice", "Bob", "nginx", "python"]),
        "{keyword}": rng.choice(["PATH", "HOME", "API"]),
        "{port}": str(rng.choice([3000, 5000, 8080, 8443, 9090])),
        "{size}": rng.choice(["1MB", "10MB", "100KB"]),
        "{pattern}": rng.choice(["*.py", "*.js", "*.ts", "*.yaml"]),
        "{title}": rng.choice(["Team meeting", "Code review", "Sprint planning"]),
        "{severity}": rng.choice(SEVERITIES),
        "{metric}": rng.choice(METRICS),
        "{timezone}": rng.choice(["US/Pacific", "US/Eastern", "Europe/London", "Asia/Tokyo"]),
        "{address}": rng.choice(["123 Main St, NYC", "456 Oak Ave, SF", "789 Pine Rd, London"]),
        "{ext}": rng.choice([".py", ".js", ".go", ".rs"]),
        "{result_ref}": "the first result",
    }
    result = template
    for key, val in replacements.items():
        result = result.replace(key, val)
    return result


def _get_tool_arguments(tool_name: str, schemas: dict, rng: random.Random) -> dict:
    """Generate appropriate arguments for a tool based on its schema."""
    # Find the schema
    schema = None
    for domain_val in schemas.values():
        if isinstance(domain_val, list):
            for s in domain_val:
                if s["name"] == tool_name:
                    schema = s
                    break
        elif isinstance(domain_val, dict):
            for subcat_val in domain_val.values():
                for s in subcat_val:
                    if s["name"] == tool_name:
                        schema = s
                        break

    if not schema:
        return {}

    params = schema["parameters"]
    properties = params.get("properties", {})
    required = params.get("required", [])

    arguments = {}
    for prop_name, prop_def in properties.items():
        # Always fill required, sometimes fill optional
        if prop_name in required or rng.random() > 0.3:
            prop_type = prop_def.get("type", "string")
            if prop_type == "string":
                arguments[prop_name] = _get_string_value(prop_name, rng)
            elif prop_type == "integer":
                arguments[prop_name] = rng.randint(1, 20)
            elif prop_type == "number":
                arguments[prop_name] = rng.uniform(1, 100).__round__(2)
            elif prop_type == "object":
                arguments[prop_name] = {"key": "value"}
            elif prop_type == "boolean":
                arguments[prop_name] = rng.choice([True, False])

    return arguments


def _get_string_value(prop_name: str, rng: random.Random) -> str:
    """Generate a realistic string value based on property name."""
    mappings = {
        "city": CITIES,
        "query": TOPICS,
        "path": PATHS,
        "directory": DIRECTORIES,
        "database": DATABASES,
        "table": TABLES,
        "url": URLS,
        "host": HOSTS,
        "service": SERVICES,
        "name": ["users", "orders", "api-gateway", "config"],
        "word": WORDS,
        "topic": TOPICS,
        "command": ["ls -la", "git status", "pwd", "cat README.md"],
        "working_directory": DIRECTORIES,
        "server": [s["name"] for s in MCP_SERVERS],
        "tool": ["query_database", "read_file", "search_web"],
        "expression": EXPRESSIONS,
        "text": ["hello world", "good morning", "thank you"],
        "target_language": LANGUAGES,
        "from_unit": UNITS_FROM,
        "to_unit": UNITS_TO,
        "unit": ["celsius", "fahrenheit"],
        "format": ["table", "json"],
        "region": ["us-east-1", "eu-west-1", "ap-southeast-1"],
        "environment": ["staging", "production"],
        "severity": SEVERITIES,
        "metric": METRICS,
        "message": ["Update dependencies", "Fix typo in docs", "Add error handling"],
        "ref": ["main", "develop", "HEAD~1"],
        "files": ["src/app.py", "*.py", "README.md"],
        "pattern": ["*.py", "*.js", "TODO"],
        "registry": REGISTRIES,
        "version": ["1.0.0", "2.1.0", "latest"],
        "key": ["API_KEY", "DB_HOST", "LOG_LEVEL"],
        "value": ["production", "localhost", "info"],
        "instance_id": ["i-1234abcd", "i-5678efgh"],
        "to": ["alice@example.com", "bob@example.com"],
        "subject": ["Meeting update", "Quick question"],
        "body": ["Please review the attached document.", "Can we reschedule?"],
        "folder": ["inbox", "sent", "archive"],
        "title": ["Team standup", "Sprint review"],
        "date": ["2026-04-21", "2026-04-22"],
        "time": ["09:00", "14:00", "16:30"],
        "event_id": ["evt-101", "evt-202"],
        "address": ["123 Main St, NYC", "456 Oak Ave, SF"],
        "origin": CITIES[:6],
        "destination": CITIES[6:],
    }
    for key, values in mappings.items():
        if key in prop_name.lower():
            return rng.choice(values)
    return f"sample_{prop_name}"


def generate_single_call_batch(count: int = 50, schemas: dict = None,
                               system_prompts: dict = None, seed: int = None) -> list[dict]:
    """Generate single function call samples per TOOL-01.

    ~75% happy path (user asks, tool called, result summarized)
    ~25% edge cases: no-tool-needed, tool error handling, ambiguous, param edge cases.
    Uses tool_assistant system prompt.
    """
    rng = random.Random(seed)
    if schemas is None:
        schemas = load_schemas()
    if system_prompts is None:
        system_prompts = load_system_prompts()

    system_content = system_prompts["tool_assistant"]
    samples = []
    edge_case_count = max(1, count // 4)
    used_queries = set()

    for i in range(count):
        is_edge_case = i < edge_case_count
        tools = get_tools_for_category(schemas, "single-call", rng, count=rng.randint(1, 3))

        # Generate unique user query
        query = _get_unique_query(SINGLE_CALL_QUERIES, used_queries, rng)

        if is_edge_case and i % 4 == 0:
            # No-tool-needed edge case
            sample = _build_no_tool_sample(system_content, query, tools)
        elif is_edge_case and i % 4 == 1:
            # Error handling edge case
            tool = tools[0]
            tool_name = tool["function"]["name"]
            arguments = _get_tool_arguments(tool_name, schemas, rng)
            error_response = generate_tool_response(tool_name, arguments, rng, error=True)
            sample = _build_error_sample(system_content, query, tools, tool_name,
                                         arguments, error_response, rng)
        elif is_edge_case and i % 4 == 2:
            # Ambiguous - answer directly despite tools
            sample = _build_no_tool_sample(system_content, query, tools)
        elif is_edge_case and i % 4 == 3:
            # Parameter edge case - optional params omitted
            tool = tools[0]
            tool_name = tool["function"]["name"]
            # Use only required params
            schema_params = tool["function"]["parameters"]
            required = schema_params.get("required", [])
            arguments = {}
            for prop_name in required:
                arguments[prop_name] = _get_string_value(prop_name, rng)
            response = generate_tool_response(tool_name, arguments, rng)
            sample = _build_happy_path_sample(system_content, query, tools, tool_name,
                                              arguments, response, rng)
        else:
            # Happy path
            tool = rng.choice(tools)
            tool_name = tool["function"]["name"]
            arguments = _get_tool_arguments(tool_name, schemas, rng)
            response = generate_tool_response(tool_name, arguments, rng)
            sample = _build_happy_path_sample(system_content, query, tools, tool_name,
                                              arguments, response, rng)

        samples.append(sample)

    return samples


def generate_cli_batch(count: int = 50, schemas: dict = None,
                       system_prompts: dict = None, seed: int = None) -> list[dict]:
    """Generate CLI/shell command samples per TOOL-05.

    Uses run_command tool. Covers bash, git, file operations.
    Uses cli_assistant system prompt.
    ~75% happy path, ~25% edge cases.
    """
    rng = random.Random(seed)
    if schemas is None:
        schemas = load_schemas()
    if system_prompts is None:
        system_prompts = load_system_prompts()

    system_content = system_prompts["cli_assistant"]
    tools = get_tools_for_category(schemas, "cli", rng)
    samples = []
    edge_case_count = max(1, count // 4)
    used_queries = set()

    for i in range(count):
        is_edge_case = i < edge_case_count
        query = _get_unique_query(CLI_QUERIES, used_queries, rng)

        if is_edge_case and i % 2 == 0:
            # No-tool-needed: question about CLI concepts, answered directly
            sample = _build_no_tool_sample(system_content, query, tools)
        elif is_edge_case:
            # Error handling: command fails
            command = _generate_safe_command(query, rng)
            arguments = {"command": command}
            error_response = generate_tool_response("run_command", arguments, rng, error=True)
            sample = _build_error_sample(system_content, query, tools, "run_command",
                                         arguments, error_response, rng)
        else:
            # Happy path
            command = _generate_safe_command(query, rng)
            arguments = {"command": command}
            response = generate_tool_response("run_command", arguments, rng)
            sample = _build_happy_path_sample(system_content, query, tools, "run_command",
                                              arguments, response, rng)

        samples.append(sample)

    return samples


def generate_multi_turn_batch(count: int = 50, schemas: dict = None,
                              system_prompts: dict = None, seed: int = None) -> list[dict]:
    """Generate multi-turn tool conversations per TOOL-02.

    Each conversation has 2-3 tool call rounds.
    Follow-up questions build on previous tool results.
    Uses tool_assistant system prompt.
    ~75% happy path, ~25% edge cases.
    """
    rng = random.Random(seed)
    if schemas is None:
        schemas = load_schemas()
    if system_prompts is None:
        system_prompts = load_system_prompts()

    system_content = system_prompts["tool_assistant"]
    samples = []
    edge_case_count = max(1, count // 4)
    used_indices = set()

    for i in range(count):
        is_edge_case = i < edge_case_count
        tools = get_tools_for_category(schemas, "multi-turn", rng, count=rng.randint(2, 3))

        if is_edge_case:
            # No-tool-needed: assistant answers directly
            query_pair = MULTI_TURN_QUERIES[i % len(MULTI_TURN_QUERIES)]
            query = _fill_template(query_pair[0], rng)
            sample = _build_no_tool_sample(system_content, query, tools)
        else:
            # Happy path: 2-3 tool call rounds
            idx = i % len(MULTI_TURN_QUERIES)
            while idx in used_indices and len(used_indices) < len(MULTI_TURN_QUERIES):
                idx = (idx + 1) % len(MULTI_TURN_QUERIES)
            used_indices.add(idx)

            query_pair = MULTI_TURN_QUERIES[idx]
            initial_query = _fill_template(query_pair[0], rng)
            followup_query = _fill_template(query_pair[1], rng)

            # Build multi-turn conversation
            messages = [{"role": "system", "content": system_content}]

            # Round 1
            tool1 = rng.choice(tools)
            tool1_name = tool1["function"]["name"]
            args1 = _get_tool_arguments(tool1_name, schemas, rng)
            response1 = generate_tool_response(tool1_name, args1, rng)

            messages.append({"role": "user", "content": initial_query})
            messages.append({"role": "assistant", "tool_calls": [
                {"type": "function", "function": {"name": tool1_name, "arguments": args1}}
            ]})
            messages.append({"role": "tool", "name": tool1_name, "content": response1})
            summary1 = generate_assistant_summary(tool1_name, response1, initial_query, rng)
            messages.append({"role": "assistant", "content": summary1})

            # Round 2
            tool2 = rng.choice(tools)
            tool2_name = tool2["function"]["name"]
            args2 = _get_tool_arguments(tool2_name, schemas, rng)
            response2 = generate_tool_response(tool2_name, args2, rng)

            messages.append({"role": "user", "content": followup_query})
            messages.append({"role": "assistant", "tool_calls": [
                {"type": "function", "function": {"name": tool2_name, "arguments": args2}}
            ]})
            messages.append({"role": "tool", "name": tool2_name, "content": response2})
            summary2 = generate_assistant_summary(tool2_name, response2, followup_query, rng)
            messages.append({"role": "assistant", "content": summary2})

            sample = {"messages": messages, "tools": tools}

        samples.append(sample)

    return samples


def generate_parallel_batch(count: int = 50, schemas: dict = None,
                            system_prompts: dict = None, seed: int = None) -> list[dict]:
    """Generate parallel function execution samples per TOOL-03.

    2-3 tool_calls in a single assistant message.
    Each tool_call has a corresponding tool response.
    Uses tool_assistant system prompt.
    ~75% happy path, ~25% edge cases.
    """
    rng = random.Random(seed)
    if schemas is None:
        schemas = load_schemas()
    if system_prompts is None:
        system_prompts = load_system_prompts()

    system_content = system_prompts["tool_assistant"]
    samples = []
    edge_case_count = max(1, count // 4)
    used_queries = set()

    for i in range(count):
        is_edge_case = i < edge_case_count
        tools = get_tools_for_category(schemas, "parallel", rng, count=rng.randint(2, 3))

        query = _get_unique_query(PARALLEL_QUERIES, used_queries, rng)

        if is_edge_case:
            # No-tool-needed: answer directly
            sample = _build_no_tool_sample(system_content, query, tools)
        else:
            # Happy path: 2-3 parallel tool calls
            num_calls = rng.randint(2, min(3, len(tools)))
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": query},
            ]

            # Build parallel tool calls
            tool_calls = []
            tool_responses = []
            for j in range(num_calls):
                tool = tools[j % len(tools)]
                tool_name = tool["function"]["name"]
                arguments = _get_tool_arguments(tool_name, schemas, rng)
                tool_calls.append({
                    "type": "function",
                    "function": {"name": tool_name, "arguments": arguments}
                })
                response = generate_tool_response(tool_name, arguments, rng)
                tool_responses.append({"role": "tool", "name": tool_name, "content": response})

            messages.append({"role": "assistant", "tool_calls": tool_calls})
            messages.extend(tool_responses)

            # Final summary
            summary = f"I've gathered all the information. Here are the combined results for your query."
            messages.append({"role": "assistant", "content": summary})

            sample = {"messages": messages, "tools": tools}

        samples.append(sample)

    return samples


def generate_mcp_batch(count: int = 50, schemas: dict = None,
                       system_prompts: dict = None, seed: int = None) -> list[dict]:
    """Generate MCP-style discovery samples per TOOL-04.

    Pattern: mcp_list_servers -> mcp_list_tools -> invoke discovered tool.
    Uses mcp_assistant system prompt.
    ~75% happy path, ~25% edge cases.
    """
    rng = random.Random(seed)
    if schemas is None:
        schemas = load_schemas()
    if system_prompts is None:
        system_prompts = load_system_prompts()

    system_content = system_prompts["mcp_assistant"]
    mcp_tools = get_tools_for_category(schemas, "mcp", rng)
    samples = []
    edge_case_count = max(1, count // 4)
    used_queries = set()

    for i in range(count):
        is_edge_case = i < edge_case_count
        query = _get_unique_query(MCP_QUERIES, used_queries, rng)

        if is_edge_case:
            # No-tool-needed: answer about MCP concepts directly
            sample = _build_no_tool_sample(system_content, query, mcp_tools)
        else:
            # Happy path: discovery -> list tools -> invoke
            server = rng.choice(MCP_SERVERS)
            server_name = server["name"]
            discovered_tool = rng.choice(server["tools"])

            # Build the tools list: MCP meta-tools + discovered tool schema
            # Find discovered tool in schema pool
            discovered_schema = None
            for domain_val in schemas.values():
                if isinstance(domain_val, list):
                    for s in domain_val:
                        if s["name"] == discovered_tool:
                            discovered_schema = s
                            break
                elif isinstance(domain_val, dict):
                    for subcat_val in domain_val.values():
                        for s in subcat_val:
                            if s["name"] == discovered_tool:
                                discovered_schema = s
                                break

            all_tools = list(mcp_tools)
            if discovered_schema:
                all_tools.append(_schema_to_tool(discovered_schema))

            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": query},
            ]

            # Step 1: list servers
            messages.append({"role": "assistant", "tool_calls": [
                {"type": "function", "function": {"name": "mcp_list_servers", "arguments": {}}}
            ]})
            servers_response = generate_tool_response("mcp_list_servers", {}, rng)
            messages.append({"role": "tool", "name": "mcp_list_servers", "content": servers_response})

            # Intermediate: assistant decides which server
            messages.append({"role": "assistant", "content":
                f"I can see the {server_name} server is available. Let me check what tools it provides."})

            # Step 2: list tools on chosen server
            messages.append({"role": "user", "content": "Yes, please proceed."})
            messages.append({"role": "assistant", "tool_calls": [
                {"type": "function", "function": {"name": "mcp_list_tools",
                                                   "arguments": {"server": server_name}}}
            ]})
            tools_response = generate_tool_response("mcp_list_tools", {"server": server_name}, rng)
            messages.append({"role": "tool", "name": "mcp_list_tools", "content": tools_response})

            # Step 3: invoke discovered tool
            if discovered_schema:
                invoke_args = _get_tool_arguments(discovered_tool, schemas, rng)
                messages.append({"role": "assistant", "tool_calls": [
                    {"type": "function", "function": {"name": discovered_tool,
                                                      "arguments": invoke_args}}
                ]})
                invoke_response = generate_tool_response(discovered_tool, invoke_args, rng)
                messages.append({"role": "tool", "name": discovered_tool, "content": invoke_response})
                summary = generate_assistant_summary(discovered_tool, invoke_response, query, rng)
                messages.append({"role": "assistant", "content": summary})
            else:
                # Use mcp_invoke_tool as fallback
                invoke_args = {"server": server_name, "tool": discovered_tool, "arguments": {}}
                messages.append({"role": "assistant", "tool_calls": [
                    {"type": "function", "function": {"name": "mcp_invoke_tool",
                                                      "arguments": invoke_args}}
                ]})
                invoke_response = generate_tool_response("mcp_invoke_tool", invoke_args, rng)
                messages.append({"role": "tool", "name": "mcp_invoke_tool", "content": invoke_response})
                summary = generate_assistant_summary("mcp_invoke_tool", invoke_response, query, rng)
                messages.append({"role": "assistant", "content": summary})

            sample = {"messages": messages, "tools": all_tools}

        samples.append(sample)

    return samples


# --- Helper Builders ---


def _get_unique_query(templates: list, used: set, rng: random.Random) -> str:
    """Generate a unique query from templates, avoiding duplicates."""
    max_attempts = 50
    for _ in range(max_attempts):
        template = rng.choice(templates)
        query = _fill_template(template, rng)
        if query not in used:
            used.add(query)
            return query
    # Fallback: add index to make unique
    template = rng.choice(templates)
    query = _fill_template(template, rng)
    query = f"{query} (variant {len(used)})"
    used.add(query)
    return query


def _build_no_tool_sample(system_content: str, query: str, tools: list[dict]) -> dict:
    """Build a no-tool-needed edge case sample."""
    # Assistant answers directly without calling tools
    direct_responses = [
        f"I can answer that directly. Based on my knowledge, here is what I know about this topic.",
        f"That's a great question. I don't need to use any tools for this one.",
        f"I can help with that without needing to look anything up.",
        f"Based on what I know, I can provide a direct answer to your question.",
    ]
    return {
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": query},
            {"role": "assistant", "content": random.choice(direct_responses)},
        ],
        "tools": tools,
    }


def _build_error_sample(system_content: str, query: str, tools: list[dict],
                         tool_name: str, arguments: dict, error_response: str,
                         rng: random.Random) -> dict:
    """Build a tool error handling edge case sample."""
    recovery_messages = [
        "I encountered an error when trying to complete that request. The service returned an error. Would you like me to try a different approach?",
        "It seems there was an issue with that operation. Let me know if you'd like me to try again or use a different method.",
        "The tool returned an error. This might be a temporary issue. Would you like me to try again?",
    ]
    return {
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": query},
            {"role": "assistant", "tool_calls": [
                {"type": "function", "function": {"name": tool_name, "arguments": arguments}}
            ]},
            {"role": "tool", "name": tool_name, "content": error_response},
            {"role": "assistant", "content": rng.choice(recovery_messages)},
        ],
        "tools": tools,
    }


def _build_happy_path_sample(system_content: str, query: str, tools: list[dict],
                              tool_name: str, arguments: dict, response: str,
                              rng: random.Random) -> dict:
    """Build a standard happy-path tool call sample."""
    summary = generate_assistant_summary(tool_name, response, query, rng)
    return {
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": query},
            {"role": "assistant", "tool_calls": [
                {"type": "function", "function": {"name": tool_name, "arguments": arguments}}
            ]},
            {"role": "tool", "name": tool_name, "content": response},
            {"role": "assistant", "content": summary},
        ],
        "tools": tools,
    }


def _generate_safe_command(query: str, rng: random.Random) -> str:
    """Generate a safe, non-destructive shell command based on user query."""
    query_lower = query.lower()
    if "git status" in query_lower or "status" in query_lower:
        return "git status"
    elif "git log" in query_lower or "commits" in query_lower:
        return f"git log --oneline -n {rng.randint(3, 10)}"
    elif "git diff" in query_lower or "diff" in query_lower:
        return "git diff HEAD~1"
    elif "git branch" in query_lower or "branch" in query_lower:
        return "git branch --sort=-committerdate"
    elif "python file" in query_lower or ".py" in query_lower:
        return "find . -name '*.py' -type f"
    elif "json file" in query_lower or ".json" in query_lower:
        return "find . -name '*.json' -type f"
    elif "list" in query_lower or "files" in query_lower or "directory" in query_lower:
        return "ls -la"
    elif "disk" in query_lower or "usage" in query_lower:
        return "du -sh ."
    elif "process" in query_lower or "running" in query_lower:
        return "ps aux | head -20"
    elif "todo" in query_lower or "comment" in query_lower:
        return "grep -rn 'TODO' --include='*.py' ."
    elif "tree" in query_lower or "structure" in query_lower:
        return "find . -maxdepth 2 -type f | head -30"
    elif "count" in query_lower or "lines" in query_lower:
        return f"wc -l {rng.choice(FILENAMES)}"
    elif "modified" in query_lower or "recent" in query_lower:
        return f"find . -mtime -{rng.randint(1,7)} -type f"
    elif "head" in query_lower or "first" in query_lower:
        return f"head -n {rng.randint(5, 20)} {rng.choice(FILENAMES)}"
    elif "contents" in query_lower or "cat" in query_lower or "show" in query_lower:
        return f"cat {rng.choice(FILENAMES)}"
    elif "docker" in query_lower or "container" in query_lower:
        return "docker ps"
    elif "port" in query_lower:
        return "lsof -i -P | head -20"
    elif "pip" in query_lower or "package" in query_lower:
        return "pip list --format=columns | head -20"
    elif "working" in query_lower or "pwd" in query_lower:
        return "pwd"
    elif "permission" in query_lower:
        return f"ls -la {rng.choice(FILENAMES)}"
    elif "network" in query_lower:
        return "ifconfig | head -30"
    elif "memory" in query_lower:
        return "vm_stat"
    elif "cron" in query_lower:
        return "crontab -l"
    elif "env" in query_lower or "variable" in query_lower:
        return "env | grep -i PATH | head -5"
    elif "compress" in query_lower or "tar" in query_lower:
        return "tar -czf archive.tar.gz docs/"
    elif "larger" in query_lower or "size" in query_lower:
        return "find . -size +1M -type f"
    elif "duplicate" in query_lower:
        return "find . -type f -name '*.py' | sort"
    elif "node" in query_lower or "version" in query_lower:
        return "node --version"
    elif "contain" in query_lower or "pattern" in query_lower or "grep" in query_lower:
        return f"grep -rn '{rng.choice(['import', 'def ', 'class '])}' --include='*.py' . | head -10"
    else:
        return "ls -la"


# --- Validation and Writing ---


def validate_batch(samples: list[dict]) -> dict:
    """Validate all samples in a batch via Conversation.model_validate().

    Returns {"total": N, "valid": N, "invalid": N, "errors": [...]}
    """
    results = {"total": len(samples), "valid": 0, "invalid": 0, "errors": []}
    for i, sample in enumerate(samples):
        try:
            Conversation.model_validate(sample)
            results["valid"] += 1
        except Exception as e:
            results["invalid"] += 1
            results["errors"].append({"index": i, "error": str(e)})
    return results


def write_batch(samples: list[dict], output_path: Path) -> Path:
    """Write samples as JSONL to output_path. One JSON object per line."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    return output_path


def main():
    """CLI entry point.

    Usage: python -m scripts.generate_tool_data --category single-call --count 50 --batch 1
    Writes to: datasets/tool-calling/{category}-batch-{batch:02d}.jsonl
    Validates each sample before writing. Prints summary stats.
    """
    parser = argparse.ArgumentParser(
        description="Generate tool-calling training data batches for Lyra"
    )
    parser.add_argument(
        "--category",
        type=str,
        required=True,
        choices=VALID_CATEGORIES,
        help="Category of tool-calling data to generate",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=50,
        help="Number of samples to generate (default: 50, max: 10000)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        required=True,
        help="Batch number (positive integer)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory (default: datasets/tool-calling/)",
    )
    args = parser.parse_args()

    # Validate arguments per T-04-02
    if args.batch < 1:
        print("Error: --batch must be a positive integer", file=sys.stderr)
        sys.exit(1)
    if args.count < 1 or args.count > MAX_COUNT:
        print(f"Error: --count must be between 1 and {MAX_COUNT}", file=sys.stderr)
        sys.exit(1)

    # Load schemas and prompts
    schemas = load_schemas()
    system_prompts = load_system_prompts()

    # Generate batch
    generators = {
        "single-call": generate_single_call_batch,
        "cli": generate_cli_batch,
        "multi-turn": generate_multi_turn_batch,
        "parallel": generate_parallel_batch,
        "mcp": generate_mcp_batch,
    }

    generator = generators[args.category]
    print(f"Generating {args.count} {args.category} samples (batch {args.batch})...")
    samples = generator(
        count=args.count, schemas=schemas,
        system_prompts=system_prompts, seed=args.seed
    )

    # Validate
    results = validate_batch(samples)
    print(f"Validation: {results['valid']}/{results['total']} valid")
    if results["errors"]:
        for err in results["errors"][:5]:
            print(f"  Sample {err['index']}: {err['error']}")

    # Filter to only valid samples
    valid_samples = []
    for i, sample in enumerate(samples):
        try:
            Conversation.model_validate(sample)
            valid_samples.append(sample)
        except Exception:
            pass

    # Write output
    output_path = args.output_dir / f"{args.category}-batch-{args.batch:02d}.jsonl"
    write_batch(valid_samples, output_path)
    print(f"Wrote {len(valid_samples)} samples to {output_path}")

    if results["invalid"] > 0:
        print(f"Warning: {results['invalid']} samples failed validation and were excluded")
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
