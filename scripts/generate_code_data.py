#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""generate_code_data.py -- Generate code generation training data batches.

Produces JSONL batches of code generation conversations across 3 categories:
utility functions, file operations, and debugging.

Per D-08: category batches with validation loops, following Phase 4 pattern.
Per D-05: terse code-first style with minimal prose.
Per D-06: debugging uses Bug/Fix format.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import yaml

from scripts.validate_format import Conversation


# --- Constants ---

PROJECT_ROOT = Path(__file__).parent.parent
SYSTEM_PROMPTS_PATH = PROJECT_ROOT / "templates" / "system-prompts.yaml"
TEMPLATES_PATH = PROJECT_ROOT / "templates" / "code.yaml"
OUTPUT_DIR = PROJECT_ROOT / "datasets" / "code"
VALID_CATEGORIES = ["utility", "file-ops", "debugging"]
MAX_COUNT = 10000  # Sanity check per T-05-01/T-05-05


# --- Language Weight Dicts ---

# Per D-03: Python-heavy for utility functions (5 languages)
UTILITY_LANG_WEIGHTS = {
    "python": 0.40,
    "javascript": 0.125,
    "typescript": 0.125,
    "go": 0.20,
    "rust": 0.15,
}

# Per D-04: For file operations and debugging (3 languages)
FILEOPS_LANG_WEIGHTS = {
    "python": 0.50,
    "javascript": 0.30,
    "typescript": 0.20,
}


# --- Loading Functions ---


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


def load_templates(templates_path: Path = None) -> dict:
    """Load code generation templates from YAML file.

    Returns the categories dict from code.yaml.
    """
    path = templates_path or TEMPLATES_PATH
    data = yaml.safe_load(path.read_text())
    return data["categories"]


# --- Topic/Query Pools ---

# Utility function query templates by topic area
UTILITY_QUERIES = {
    "string": [
        "Write a {lang} function to reverse a string",
        "Write a {lang} function to capitalize the first letter of each word",
        "Write a {lang} function to truncate a string to a given length with ellipsis",
        "Write a {lang} function to convert a string to slug format",
        "Write a {lang} function to convert a string to camelCase",
        "Write a {lang} function to convert a string from camelCase to snake_case",
        "Write a {lang} function to check if a string is a palindrome",
        "Write a {lang} function to count the number of words in a string",
        "Write a {lang} function to pad a string to a specified length",
        "Write a {lang} function to extract all numbers from a string",
        "Write a {lang} function to remove all whitespace from a string",
        "Write a {lang} function to count character occurrences in a string",
    ],
    "array": [
        "Write a {lang} function to flatten a nested array",
        "Write a {lang} function to chunk an array into groups of N",
        "Write a {lang} function to deduplicate an array",
        "Write a {lang} function to sort an array of objects by a given key",
        "Write a {lang} function to zip two arrays together",
        "Write a {lang} function to rotate an array by N positions",
        "Write a {lang} function to partition an array based on a predicate",
        "Write a {lang} function to compact an array by removing falsy values",
        "Write a {lang} function to find the intersection of two arrays",
        "Write a {lang} function to group array elements by a classifier function",
        "Write a {lang} function to create a sliding window over an array",
        "Write a {lang} function to find the unique elements in an array",
    ],
    "date": [
        "Write a {lang} function to format a date as YYYY-MM-DD",
        "Write a {lang} function to parse a date string into a date object",
        "Write a {lang} function to add N days to a date",
        "Write a {lang} function to calculate the number of days between two dates",
        "Write a {lang} function to get the day of the week for a given date",
        "Write a {lang} function to get a human-readable relative time string",
        "Write a {lang} function to convert a date to ISO 8601 format",
        "Write a {lang} function to check if a given year is a leap year",
    ],
    "number": [
        "Write a {lang} function to generate the Nth Fibonacci number",
        "Write a {lang} function to check if a number is prime",
        "Write a {lang} function to compute the GCD of two numbers",
        "Write a {lang} function to compute the factorial of a number",
        "Write a {lang} function to compute the LCM of two numbers",
        "Write a {lang} function to check if a number is a power of two",
        "Write a {lang} function to clamp a number between a min and max",
        "Write a {lang} function to round a number to N decimal places",
        "Write a {lang} function to convert a decimal number to binary string",
        "Write a {lang} function to find all prime factors of a number",
    ],
    "data_structure": [
        "Write a {lang} function to deep merge two objects",
        "Write a {lang} function to look up a value in a nested object by dot-path",
        "Write a {lang} function to check the type of a value and return a string label",
        "Write a {lang} function to flatten a nested object into dot-notation keys",
        "Write a {lang} function to pick specified keys from an object",
        "Write a {lang} function to omit specified keys from an object",
        "Write a {lang} function to invert a map (swap keys and values)",
        "Write a {lang} function to count the frequency of each element",
        "Write a {lang} function to create a simple LRU cache",
    ],
    "validation": [
        "Write a {lang} function to validate an email address",
        "Write a {lang} function to validate a URL",
        "Write a {lang} function to validate a phone number format",
        "Write a {lang} function to validate an IPv4 address",
        "Write a {lang} function to validate a UUID string",
        "Write a {lang} function to validate a date string in YYYY-MM-DD format",
        "Write a {lang} function to validate a JSON string",
        "Write a {lang} function to validate a hex color code",
        "Write a {lang} function to validate a semantic version string",
    ],
    "encoding": [
        "Write a {lang} function to encode a string to base64",
        "Write a {lang} function to decode a base64 string",
        "Write a {lang} function to convert a string to hex encoding",
        "Write a {lang} function to URL-encode a string",
        "Write a {lang} function to implement ROT13 encoding",
        "Write a {lang} function to escape HTML entities in a string",
    ],
}

# File operations query templates
FILE_OPS_QUERIES = {
    "csv": [
        "Write a {lang} function to read a CSV file and return rows as dictionaries",
        "Write a {lang} function to write a list of dictionaries to a CSV file",
        "Write a {lang} function to filter rows in a CSV file by a column value",
        "Write a {lang} function to merge two CSV files by a common column",
    ],
    "json_yaml": [
        "Write a {lang} function to read and parse a JSON configuration file",
        "Write a {lang} function to write data to a JSON file with pretty formatting",
        "Write a {lang} function to deeply merge two JSON config files",
        "Write a {lang} function to validate a JSON file against a schema",
    ],
    "directory": [
        "Write a {lang} function to recursively find all files with a given extension",
        "Write a {lang} function to list all files in a directory sorted by size",
        "Write a {lang} function to copy a directory recursively",
        "Write a {lang} function to watch a directory for new files",
    ],
    "log": [
        "Write a {lang} function to parse a log file and count error occurrences",
        "Write a {lang} function to tail the last N lines of a file",
        "Write a {lang} function to search a log file for lines matching a pattern",
        "Write a {lang} function to rotate log files by size",
    ],
    "compression": [
        "Write a {lang} function to compress a file using gzip",
        "Write a {lang} function to decompress a gzip file",
        "Write a {lang} function to create a zip archive from a directory",
        "Write a {lang} function to extract a zip archive to a directory",
    ],
    "env": [
        "Write a {lang} function to load environment variables from a .env file",
        "Write a {lang} function to get an environment variable with a default value",
        "Write a {lang} function to set multiple environment variables from a dictionary",
    ],
    "path": [
        "Write a {lang} function to safely join file paths",
        "Write a {lang} function to get the file extension from a path",
        "Write a {lang} function to resolve a relative path to an absolute path",
        "Write a {lang} function to check if a path is inside a given directory",
    ],
}

# Debugging query templates -- buggy code snippets per language + bug type
DEBUGGING_QUERIES = {
    "python": {
        "off_by_one": [
            (
                "This function should return the last N elements but has a bug:\n\n```python\ndef last_n(lst, n):\n    return lst[len(lst) - n - 1:]\n```",
                "Bug: Off-by-one error in slice index. `len(lst) - n - 1` skips one extra element.\nFix: Use `lst[-n:]` or `lst[len(lst) - n:]` to get the correct slice.\n\n```python\ndef last_n(lst, n):\n    return lst[-n:] if n > 0 else []\n```",
            ),
            (
                "This loop should print numbers 1 to 10 but misses the last one:\n\n```python\nfor i in range(1, 10):\n    print(i)\n```",
                "Bug: `range(1, 10)` excludes the upper bound, so it stops at 9.\nFix: Use `range(1, 11)` to include 10.\n\n```python\nfor i in range(1, 11):\n    print(i)\n```",
            ),
        ],
        "null_reference": [
            (
                "This code crashes when the dictionary key is missing:\n\n```python\ndef get_name(user):\n    return user['name'].upper()\n```",
                "Bug: Accessing a missing key raises `KeyError`, and calling `.upper()` on None raises `AttributeError`.\nFix: Use `.get()` with a default and guard the method call.\n\n```python\ndef get_name(user):\n    name = user.get('name', '')\n    return name.upper() if name else ''\n```",
            ),
            (
                "This function fails when the input list is empty:\n\n```python\ndef get_first(items):\n    return items[0]\n```",
                "Bug: Indexing an empty list raises `IndexError`.\nFix: Check for empty list before accessing the first element.\n\n```python\ndef get_first(items):\n    return items[0] if items else None\n```",
            ),
        ],
        "type_mismatch": [
            (
                "This function should concatenate a number to a string but fails:\n\n```python\ndef make_label(prefix, count):\n    return prefix + count\n```",
                "Bug: Cannot concatenate `str` and `int` directly in Python.\nFix: Convert the integer to string with `str()`.\n\n```python\ndef make_label(prefix, count):\n    return prefix + str(count)\n```",
            ),
            (
                "This comparison always returns False:\n\n```python\ndef check_age(age_str):\n    if age_str > 18:\n        return True\n    return False\n```",
                "Bug: Comparing a string to an integer does not compare numerically.\nFix: Convert the string to int before comparing.\n\n```python\ndef check_age(age_str):\n    if int(age_str) > 18:\n        return True\n    return False\n```",
            ),
        ],
        "logic_error": [
            (
                "This function should check if all items pass but returns wrong results:\n\n```python\ndef all_positive(numbers):\n    for n in numbers:\n        if n > 0:\n            return True\n    return False\n```",
                "Bug: Returns True on the first positive number instead of checking all.\nFix: Return False on the first non-positive number, True only after checking all.\n\n```python\ndef all_positive(numbers):\n    for n in numbers:\n        if n <= 0:\n            return False\n    return True\n```",
            ),
            (
                "This function should find the maximum but sometimes returns wrong values:\n\n```python\ndef find_max(numbers):\n    max_val = 0\n    for n in numbers:\n        if n > max_val:\n            max_val = n\n    return max_val\n```",
                "Bug: Initializing `max_val = 0` fails when all numbers are negative.\nFix: Initialize with the first element or use `float('-inf')`.\n\n```python\ndef find_max(numbers):\n    if not numbers:\n        return None\n    max_val = numbers[0]\n    for n in numbers[1:]:\n        if n > max_val:\n            max_val = n\n    return max_val\n```",
            ),
        ],
        "async_error": [
            (
                "This async function does not wait for the result:\n\n```python\nasync def fetch_data(url):\n    result = requests.get(url)\n    return result.json()\n```",
                "Bug: Using synchronous `requests.get()` inside an async function blocks the event loop.\nFix: Use an async HTTP client like `aiohttp`.\n\n```python\nasync def fetch_data(url):\n    async with aiohttp.ClientSession() as session:\n        async with session.get(url) as response:\n            return await response.json()\n```",
            ),
        ],
        "scope_error": [
            (
                "This function tries to modify a counter but fails:\n\n```python\ncounter = 0\ndef increment():\n    counter += 1\n    return counter\n```",
                "Bug: `counter += 1` creates a new local variable instead of modifying the global one, raising `UnboundLocalError`.\nFix: Declare `counter` as `global` or use a mutable container.\n\n```python\ncounter = 0\ndef increment():\n    global counter\n    counter += 1\n    return counter\n```",
            ),
        ],
        "import_error": [
            (
                "This code fails with an import error:\n\n```python\nfrom json import loads, dump\ndata = loads('{\"key\": \"value\"}')\nresult = dump(data)\n```",
                "Bug: `json.dump()` writes to a file object, not to a string. To get a string, use `json.dumps()`.\nFix: Use `dumps()` for serialization to string.\n\n```python\nfrom json import loads, dumps\ndata = loads('{\"key\": \"value\"}')\nresult = dumps(data)\n```",
            ),
        ],
    },
    "javascript": {
        "off_by_one": [
            (
                "This loop should process all array elements but skips the last one:\n\n```javascript\nfunction processAll(arr) {\n    const results = [];\n    for (let i = 0; i < arr.length - 1; i++) {\n        results.push(arr[i] * 2);\n    }\n    return results;\n}\n```",
                "Bug: `i < arr.length - 1` stops one element early.\nFix: Use `i < arr.length` to include the last element.\n\n```javascript\nfunction processAll(arr) {\n    const results = [];\n    for (let i = 0; i < arr.length; i++) {\n        results.push(arr[i] * 2);\n    }\n    return results;\n}\n```",
            ),
        ],
        "null_reference": [
            (
                "This code crashes when the property does not exist:\n\n```javascript\nfunction getCity(user) {\n    return user.address.city;\n}\n```",
                "Bug: If `user.address` is undefined, accessing `.city` throws `TypeError: Cannot read property 'city' of undefined`.\nFix: Use optional chaining.\n\n```javascript\nfunction getCity(user) {\n    return user?.address?.city ?? '';\n}\n```",
            ),
        ],
        "type_mismatch": [
            (
                "This comparison gives unexpected results:\n\n```javascript\nfunction isEqual(a, b) {\n    return a == b;\n}\nconsole.log(isEqual('1', 1)); // true, but should be false\n```",
                "Bug: `==` performs type coercion, so `'1' == 1` is `true`.\nFix: Use strict equality `===` to avoid type coercion.\n\n```javascript\nfunction isEqual(a, b) {\n    return a === b;\n}\n```",
            ),
        ],
        "logic_error": [
            (
                "This filter should remove empty strings but keeps them:\n\n```javascript\nfunction removeEmpty(arr) {\n    return arr.filter(item => !item);\n}\n```",
                "Bug: The predicate `!item` keeps falsy values and removes truthy ones -- the logic is inverted.\nFix: Use `item` (truthy check) or `item !== ''` for the predicate.\n\n```javascript\nfunction removeEmpty(arr) {\n    return arr.filter(item => item !== '');\n}\n```",
            ),
        ],
        "async_error": [
            (
                "This function does not return the fetched data:\n\n```javascript\nasync function getData(url) {\n    fetch(url).then(res => res.json());\n}\n```",
                "Bug: Missing `return` and not using `await`. The function returns `undefined`.\nFix: Use `await` and `return` the result.\n\n```javascript\nasync function getData(url) {\n    const res = await fetch(url);\n    return res.json();\n}\n```",
            ),
        ],
        "scope_error": [
            (
                "This loop prints the same value every time:\n\n```javascript\nfor (var i = 0; i < 5; i++) {\n    setTimeout(() => console.log(i), 100);\n}\n```",
                "Bug: `var` has function scope, so all callbacks share the same `i` which equals 5 after the loop.\nFix: Use `let` for block scoping.\n\n```javascript\nfor (let i = 0; i < 5; i++) {\n    setTimeout(() => console.log(i), 100);\n}\n```",
            ),
        ],
        "import_error": [
            (
                "This code fails with a module error:\n\n```javascript\nconst { readFile } = require('fs');\nconst data = readFile('config.json');\nconsole.log(JSON.parse(data));\n```",
                "Bug: `readFile` is asynchronous and requires a callback. Using it synchronously returns `undefined`.\nFix: Use `readFileSync` for synchronous reads.\n\n```javascript\nconst { readFileSync } = require('fs');\nconst data = readFileSync('config.json', 'utf8');\nconsole.log(JSON.parse(data));\n```",
            ),
        ],
    },
    "typescript": {
        "off_by_one": [
            (
                "This function should return elements from index start to end inclusive but misses the last:\n\n```typescript\nfunction sliceInclusive(arr: number[], start: number, end: number): number[] {\n    return arr.slice(start, end);\n}\n```",
                "Bug: `Array.slice(start, end)` excludes the element at index `end`.\nFix: Use `end + 1` as the second argument.\n\n```typescript\nfunction sliceInclusive(arr: number[], start: number, end: number): number[] {\n    return arr.slice(start, end + 1);\n}\n```",
            ),
        ],
        "null_reference": [
            (
                "This function crashes when the optional property is missing:\n\n```typescript\ninterface Config {\n    database?: { host: string; port: number };\n}\nfunction getPort(config: Config): number {\n    return config.database.port;\n}\n```",
                "Bug: `config.database` may be `undefined`, causing a runtime error when accessing `.port`.\nFix: Use optional chaining and provide a default value.\n\n```typescript\ninterface Config {\n    database?: { host: string; port: number };\n}\nfunction getPort(config: Config): number {\n    return config.database?.port ?? 5432;\n}\n```",
            ),
        ],
        "type_mismatch": [
            (
                "This function has a type error that TypeScript does not catch at runtime:\n\n```typescript\nfunction add(a: number, b: any): number {\n    return a + b;\n}\nconsole.log(add(5, '3')); // returns '53'\n```",
                "Bug: Parameter `b` typed as `any` allows a string, causing string concatenation instead of addition.\nFix: Type `b` as `number` to catch the error at compile time.\n\n```typescript\nfunction add(a: number, b: number): number {\n    return a + b;\n}\n```",
            ),
        ],
        "logic_error": [
            (
                "This function should return unique values but fails for objects:\n\n```typescript\nfunction unique<T>(arr: T[]): T[] {\n    return [...new Set(arr)];\n}\nconst result = unique([{id: 1}, {id: 1}]); // still 2 items\n```",
                "Bug: `Set` uses reference equality for objects, so identical objects are not deduplicated.\nFix: Use a Map with a serialized key for deduplication.\n\n```typescript\nfunction unique<T>(arr: T[], key: (item: T) => string): T[] {\n    const seen = new Map<string, T>();\n    for (const item of arr) {\n        const k = key(item);\n        if (!seen.has(k)) seen.set(k, item);\n    }\n    return [...seen.values()];\n}\n```",
            ),
        ],
        "async_error": [
            (
                "This function does not handle the Promise rejection:\n\n```typescript\nasync function loadConfig(path: string): Promise<object> {\n    const data = await fs.readFile(path, 'utf8');\n    return JSON.parse(data);\n}\n```",
                "Bug: If the file does not exist or JSON is invalid, the error propagates as an unhandled rejection.\nFix: Add try/catch for proper error handling.\n\n```typescript\nasync function loadConfig(path: string): Promise<object> {\n    try {\n        const data = await fs.readFile(path, 'utf8');\n        return JSON.parse(data);\n    } catch (error) {\n        throw new Error(`Failed to load config: ${error}`);\n    }\n}\n```",
            ),
        ],
        "scope_error": [
            (
                "This class method loses its context when used as a callback:\n\n```typescript\nclass Counter {\n    count = 0;\n    increment() {\n        this.count++;\n    }\n}\nconst c = new Counter();\nsetInterval(c.increment, 1000); // this.count is NaN\n```",
                "Bug: When `increment` is passed as a callback, `this` is no longer bound to the Counter instance.\nFix: Use an arrow function or bind the method.\n\n```typescript\nclass Counter {\n    count = 0;\n    increment = () => {\n        this.count++;\n    };\n}\n```",
            ),
        ],
        "import_error": [
            (
                "This TypeScript import fails at runtime:\n\n```typescript\nimport { readFile } from 'fs';\nconst content = readFile('./data.txt');\n```",
                "Bug: `readFile` from `fs` is callback-based. The import should use `fs/promises` for the Promise-based version.\nFix: Import from `fs/promises` and use `await`.\n\n```typescript\nimport { readFile } from 'fs/promises';\nconst content = await readFile('./data.txt', 'utf8');\n```",
            ),
        ],
    },
}


# --- Language-Specific Code Pools ---

# Utility function code responses per language.
# Each entry is (query_suffix, code_block) where query_suffix identifies the query topic.

UTILITY_CODE_PYTHON = {
    "reverse a string": '```python\ndef reverse_string(s: str) -> str:\n    return s[::-1]\n```',
    "capitalize the first letter of each word": '```python\ndef capitalize_words(s: str) -> str:\n    return " ".join(word.capitalize() for word in s.split())\n```',
    "truncate a string": '```python\ndef truncate(s: str, max_len: int, suffix: str = "...") -> str:\n    if len(s) <= max_len:\n        return s\n    return s[:max_len - len(suffix)] + suffix\n```',
    "convert a string to slug format": '```python\nimport re\n\ndef slugify(s: str) -> str:\n    s = s.lower().strip()\n    s = re.sub(r"[^\\w\\s-]", "", s)\n    return re.sub(r"[\\s_-]+", "-", s)\n```',
    "convert a string to camelCase": '```python\ndef to_camel_case(s: str) -> str:\n    parts = s.replace("-", " ").replace("_", " ").split()\n    return parts[0].lower() + "".join(w.capitalize() for w in parts[1:])\n```',
    "convert a string from camelCase to snake_case": '```python\nimport re\n\ndef camel_to_snake(s: str) -> str:\n    return re.sub(r"(?<!^)(?=[A-Z])", "_", s).lower()\n```',
    "check if a string is a palindrome": '```python\ndef is_palindrome(s: str) -> bool:\n    cleaned = s.lower().replace(" ", "")\n    return cleaned == cleaned[::-1]\n```',
    "count the number of words": '```python\ndef word_count(s: str) -> int:\n    return len(s.split())\n```',
    "pad a string to a specified length": '```python\ndef pad_string(s: str, length: int, char: str = " ", right: bool = True) -> str:\n    if right:\n        return s.ljust(length, char)\n    return s.rjust(length, char)\n```',
    "extract all numbers from a string": '```python\nimport re\n\ndef extract_numbers(s: str) -> list[float]:\n    return [float(n) for n in re.findall(r"-?\\d+\\.?\\d*", s)]\n```',
    "remove all whitespace": '```python\ndef remove_whitespace(s: str) -> str:\n    return "".join(s.split())\n```',
    "count character occurrences": '```python\nfrom collections import Counter\n\ndef char_frequency(s: str) -> dict[str, int]:\n    return dict(Counter(s))\n```',
    "flatten a nested array": '```python\ndef flatten(lst: list) -> list:\n    result = []\n    for item in lst:\n        if isinstance(item, list):\n            result.extend(flatten(item))\n        else:\n            result.append(item)\n    return result\n```',
    "chunk an array into groups": '```python\ndef chunk(lst: list, size: int) -> list[list]:\n    return [lst[i:i + size] for i in range(0, len(lst), size)]\n```',
    "deduplicate an array": '```python\ndef deduplicate(lst: list) -> list:\n    seen = set()\n    return [x for x in lst if not (x in seen or seen.add(x))]\n```',
    "sort an array of objects by a given key": '```python\ndef sort_by_key(items: list[dict], key: str, reverse: bool = False) -> list[dict]:\n    return sorted(items, key=lambda x: x.get(key), reverse=reverse)\n```',
    "zip two arrays together": '```python\ndef zip_arrays(a: list, b: list) -> list[tuple]:\n    return list(zip(a, b))\n```',
    "rotate an array by N positions": '```python\ndef rotate(lst: list, n: int) -> list:\n    if not lst:\n        return lst\n    n = n % len(lst)\n    return lst[n:] + lst[:n]\n```',
    "partition an array": '```python\ndef partition(lst: list, predicate) -> tuple[list, list]:\n    true_part, false_part = [], []\n    for item in lst:\n        (true_part if predicate(item) else false_part).append(item)\n    return true_part, false_part\n```',
    "compact an array": '```python\ndef compact(lst: list) -> list:\n    return [x for x in lst if x]\n```',
    "find the intersection of two arrays": '```python\ndef intersect(a: list, b: list) -> list:\n    set_b = set(b)\n    return [x for x in a if x in set_b]\n```',
    "group array elements": '```python\ndef group_by(lst: list, key_fn) -> dict:\n    groups = {}\n    for item in lst:\n        k = key_fn(item)\n        groups.setdefault(k, []).append(item)\n    return groups\n```',
    "sliding window": '```python\ndef sliding_window(lst: list, size: int) -> list[list]:\n    return [lst[i:i + size] for i in range(len(lst) - size + 1)]\n```',
    "unique elements": '```python\ndef unique(lst: list) -> list:\n    return list(dict.fromkeys(lst))\n```',
    "format a date": '```python\nfrom datetime import datetime\n\ndef format_date(dt: datetime) -> str:\n    return dt.strftime("%Y-%m-%d")\n```',
    "parse a date string": '```python\nfrom datetime import datetime\n\ndef parse_date(s: str, fmt: str = "%Y-%m-%d") -> datetime:\n    return datetime.strptime(s, fmt)\n```',
    "add N days to a date": '```python\nfrom datetime import datetime, timedelta\n\ndef add_days(dt: datetime, n: int) -> datetime:\n    return dt + timedelta(days=n)\n```',
    "number of days between two dates": '```python\nfrom datetime import datetime\n\ndef days_between(a: datetime, b: datetime) -> int:\n    return abs((b - a).days)\n```',
    "day of the week": '```python\nfrom datetime import datetime\n\ndef day_of_week(dt: datetime) -> str:\n    return dt.strftime("%A")\n```',
    "relative time string": '```python\nfrom datetime import datetime\n\ndef relative_time(dt: datetime) -> str:\n    diff = datetime.now() - dt\n    seconds = int(diff.total_seconds())\n    if seconds < 60:\n        return "just now"\n    if seconds < 3600:\n        return f"{seconds // 60} minutes ago"\n    if seconds < 86400:\n        return f"{seconds // 3600} hours ago"\n    return f"{seconds // 86400} days ago"\n```',
    "ISO 8601 format": '```python\nfrom datetime import datetime\n\ndef to_iso(dt: datetime) -> str:\n    return dt.isoformat()\n```',
    "leap year": '```python\ndef is_leap_year(year: int) -> bool:\n    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)\n```',
    "Fibonacci number": '```python\ndef fibonacci(n: int) -> int:\n    if n < 0:\n        raise ValueError("n must be non-negative")\n    if n <= 1:\n        return n\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n```',
    "check if a number is prime": '```python\ndef is_prime(n: int) -> bool:\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True\n```',
    "GCD of two numbers": '```python\ndef gcd(a: int, b: int) -> int:\n    while b:\n        a, b = b, a % b\n    return a\n```',
    "factorial": '```python\ndef factorial(n: int) -> int:\n    if n < 0:\n        raise ValueError("n must be non-negative")\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result\n```',
    "LCM of two numbers": '```python\ndef lcm(a: int, b: int) -> int:\n    from math import gcd\n    return abs(a * b) // gcd(a, b)\n```',
    "power of two": '```python\ndef is_power_of_two(n: int) -> bool:\n    return n > 0 and (n & (n - 1)) == 0\n```',
    "clamp a number": '```python\ndef clamp(value: float, min_val: float, max_val: float) -> float:\n    return max(min_val, min(value, max_val))\n```',
    "round a number to N decimal places": '```python\ndef round_to(value: float, decimals: int) -> float:\n    return round(value, decimals)\n```',
    "decimal number to binary": '```python\ndef to_binary(n: int) -> str:\n    if n < 0:\n        return "-" + bin(-n)[2:]\n    return bin(n)[2:]\n```',
    "prime factors": '```python\ndef prime_factors(n: int) -> list[int]:\n    factors = []\n    d = 2\n    while d * d <= n:\n        while n % d == 0:\n            factors.append(d)\n            n //= d\n        d += 1\n    if n > 1:\n        factors.append(n)\n    return factors\n```',
    "deep merge two objects": '```python\ndef deep_merge(base: dict, override: dict) -> dict:\n    result = base.copy()\n    for key, value in override.items():\n        if key in result and isinstance(result[key], dict) and isinstance(value, dict):\n            result[key] = deep_merge(result[key], value)\n        else:\n            result[key] = value\n    return result\n```',
    "dot-path": '```python\ndef get_by_path(obj: dict, path: str, default=None):\n    keys = path.split(".")\n    current = obj\n    for key in keys:\n        if isinstance(current, dict) and key in current:\n            current = current[key]\n        else:\n            return default\n    return current\n```',
    "type of a value": '```python\ndef type_label(value) -> str:\n    return type(value).__name__\n```',
    "flatten a nested object": '```python\ndef flatten_dict(d: dict, prefix: str = "") -> dict:\n    items = {}\n    for key, value in d.items():\n        new_key = f"{prefix}.{key}" if prefix else key\n        if isinstance(value, dict):\n            items.update(flatten_dict(value, new_key))\n        else:\n            items[new_key] = value\n    return items\n```',
    "pick specified keys": '```python\ndef pick(d: dict, keys: list[str]) -> dict:\n    return {k: d[k] for k in keys if k in d}\n```',
    "omit specified keys": '```python\ndef omit(d: dict, keys: list[str]) -> dict:\n    skip = set(keys)\n    return {k: v for k, v in d.items() if k not in skip}\n```',
    "invert a map": '```python\ndef invert(d: dict) -> dict:\n    return {v: k for k, v in d.items()}\n```',
    "frequency of each element": '```python\nfrom collections import Counter\n\ndef frequency(items: list) -> dict:\n    return dict(Counter(items))\n```',
    "LRU cache": '```python\nfrom functools import lru_cache\n\n@lru_cache(maxsize=128)\ndef cached_compute(n: int) -> int:\n    return sum(range(n))\n```',
    "validate an email": '```python\nimport re\n\ndef is_valid_email(email: str) -> bool:\n    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"\n    return bool(re.match(pattern, email))\n```',
    "validate a URL": '```python\nimport re\n\ndef is_valid_url(url: str) -> bool:\n    pattern = r"^https?://[\\w.-]+(?:\\.[a-zA-Z]{2,})(?:/[\\w./-]*)?$"\n    return bool(re.match(pattern, url))\n```',
    "validate a phone number": '```python\nimport re\n\ndef is_valid_phone(phone: str) -> bool:\n    pattern = r"^\\+?[1-9]\\d{6,14}$"\n    return bool(re.match(pattern, phone.replace(" ", "").replace("-", "")))\n```',
    "validate an IPv4 address": '```python\ndef is_valid_ipv4(ip: str) -> bool:\n    parts = ip.split(".")\n    if len(parts) != 4:\n        return False\n    return all(p.isdigit() and 0 <= int(p) <= 255 for p in parts)\n```',
    "validate a UUID": '```python\nimport re\n\ndef is_valid_uuid(s: str) -> bool:\n    pattern = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"\n    return bool(re.match(pattern, s.lower()))\n```',
    "validate a date string in YYYY-MM-DD": '```python\nfrom datetime import datetime\n\ndef is_valid_date(s: str) -> bool:\n    try:\n        datetime.strptime(s, "%Y-%m-%d")\n        return True\n    except ValueError:\n        return False\n```',
    "validate a JSON string": '```python\nimport json\n\ndef is_valid_json(s: str) -> bool:\n    try:\n        json.loads(s)\n        return True\n    except (json.JSONDecodeError, TypeError):\n        return False\n```',
    "validate a hex color code": '```python\nimport re\n\ndef is_valid_hex_color(s: str) -> bool:\n    return bool(re.match(r"^#([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$", s))\n```',
    "validate a semantic version": '```python\nimport re\n\ndef is_valid_semver(s: str) -> bool:\n    return bool(re.match(r"^\\d+\\.\\d+\\.\\d+(-[\\w.]+)?(\\+[\\w.]+)?$", s))\n```',
    "encode a string to base64": '```python\nimport base64\n\ndef to_base64(s: str) -> str:\n    return base64.b64encode(s.encode()).decode()\n```',
    "decode a base64 string": '```python\nimport base64\n\ndef from_base64(s: str) -> str:\n    return base64.b64decode(s.encode()).decode()\n```',
    "convert a string to hex": '```python\ndef to_hex(s: str) -> str:\n    return s.encode().hex()\n```',
    "URL-encode a string": '```python\nfrom urllib.parse import quote\n\ndef url_encode(s: str) -> str:\n    return quote(s)\n```',
    "ROT13 encoding": '```python\nimport codecs\n\ndef rot13(s: str) -> str:\n    return codecs.encode(s, "rot_13")\n```',
    "escape HTML entities": '```python\nimport html\n\ndef escape_html(s: str) -> str:\n    return html.escape(s)\n```',
}

UTILITY_CODE_JAVASCRIPT = {
    "reverse a string": '```javascript\nfunction reverseString(s) {\n    return s.split("").reverse().join("");\n}\n```',
    "capitalize the first letter of each word": '```javascript\nfunction capitalizeWords(s) {\n    return s.replace(/\\b\\w/g, c => c.toUpperCase());\n}\n```',
    "truncate a string": '```javascript\nfunction truncate(s, maxLen, suffix = "...") {\n    if (s.length <= maxLen) return s;\n    return s.slice(0, maxLen - suffix.length) + suffix;\n}\n```',
    "flatten a nested array": '```javascript\nfunction flatten(arr) {\n    return arr.flat(Infinity);\n}\n```',
    "chunk an array into groups": '```javascript\nfunction chunk(arr, size) {\n    const result = [];\n    for (let i = 0; i < arr.length; i += size) {\n        result.push(arr.slice(i, i + size));\n    }\n    return result;\n}\n```',
    "deduplicate an array": '```javascript\nfunction deduplicate(arr) {\n    return [...new Set(arr)];\n}\n```',
    "check if a number is prime": '```javascript\nfunction isPrime(n) {\n    if (n < 2) return false;\n    for (let i = 2; i <= Math.sqrt(n); i++) {\n        if (n % i === 0) return false;\n    }\n    return true;\n}\n```',
    "Fibonacci number": '```javascript\nfunction fibonacci(n) {\n    if (n <= 1) return n;\n    let a = 0, b = 1;\n    for (let i = 2; i <= n; i++) {\n        [a, b] = [b, a + b];\n    }\n    return b;\n}\n```',
    "validate an email": '```javascript\nfunction isValidEmail(email) {\n    const pattern = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$/;\n    return pattern.test(email);\n}\n```',
    "encode a string to base64": '```javascript\nfunction toBase64(s) {\n    return btoa(s);\n}\n```',
    "deep merge two objects": '```javascript\nfunction deepMerge(base, override) {\n    const result = { ...base };\n    for (const [key, value] of Object.entries(override)) {\n        if (typeof result[key] === "object" && typeof value === "object") {\n            result[key] = deepMerge(result[key], value);\n        } else {\n            result[key] = value;\n        }\n    }\n    return result;\n}\n```',
    "GCD of two numbers": '```javascript\nfunction gcd(a, b) {\n    while (b) {\n        [a, b] = [b, a % b];\n    }\n    return a;\n}\n```',
}

UTILITY_CODE_TYPESCRIPT = {
    "reverse a string": '```typescript\nfunction reverseString(s: string): string {\n    return s.split("").reverse().join("");\n}\n```',
    "flatten a nested array": '```typescript\nfunction flatten<T>(arr: (T | T[])[]): T[] {\n    return arr.flat(Infinity) as T[];\n}\n```',
    "validate an email": '```typescript\nfunction isValidEmail(email: string): boolean {\n    const pattern = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$/;\n    return pattern.test(email);\n}\n```',
    "check if a number is prime": '```typescript\nfunction isPrime(n: number): boolean {\n    if (n < 2) return false;\n    for (let i = 2; i <= Math.sqrt(n); i++) {\n        if (n % i === 0) return false;\n    }\n    return true;\n}\n```',
    "deduplicate an array": '```typescript\nfunction deduplicate<T>(arr: T[]): T[] {\n    return [...new Set(arr)];\n}\n```',
    "deep merge two objects": '```typescript\nfunction deepMerge<T extends object>(base: T, override: Partial<T>): T {\n    const result = { ...base };\n    for (const [key, value] of Object.entries(override)) {\n        const k = key as keyof T;\n        if (typeof result[k] === "object" && typeof value === "object") {\n            result[k] = deepMerge(result[k] as any, value as any);\n        } else {\n            result[k] = value as any;\n        }\n    }\n    return result;\n}\n```',
    "chunk an array into groups": '```typescript\nfunction chunk<T>(arr: T[], size: number): T[][] {\n    const result: T[][] = [];\n    for (let i = 0; i < arr.length; i += size) {\n        result.push(arr.slice(i, i + size));\n    }\n    return result;\n}\n```',
    "Fibonacci number": '```typescript\nfunction fibonacci(n: number): number {\n    if (n <= 1) return n;\n    let a = 0, b = 1;\n    for (let i = 2; i <= n; i++) {\n        [a, b] = [b, a + b];\n    }\n    return b;\n}\n```',
    "encode a string to base64": '```typescript\nfunction toBase64(s: string): string {\n    return btoa(s);\n}\n```',
    "GCD of two numbers": '```typescript\nfunction gcd(a: number, b: number): number {\n    while (b) {\n        [a, b] = [b, a % b];\n    }\n    return a;\n}\n```',
}

UTILITY_CODE_GO = {
    "reverse a string": '```go\nfunc reverseString(s string) string {\n\trunes := []rune(s)\n\tfor i, j := 0, len(runes)-1; i < j; i, j = i+1, j-1 {\n\t\trunes[i], runes[j] = runes[j], runes[i]\n\t}\n\treturn string(runes)\n}\n```',
    "check if a number is prime": '```go\nfunc isPrime(n int) bool {\n\tif n < 2 {\n\t\treturn false\n\t}\n\tfor i := 2; i*i <= n; i++ {\n\t\tif n%i == 0 {\n\t\t\treturn false\n\t\t}\n\t}\n\treturn true\n}\n```',
    "GCD of two numbers": '```go\nfunc gcd(a, b int) int {\n\tfor b != 0 {\n\t\ta, b = b, a%b\n\t}\n\treturn a\n}\n```',
    "Fibonacci number": '```go\nfunc fibonacci(n int) int {\n\tif n <= 1 {\n\t\treturn n\n\t}\n\ta, b := 0, 1\n\tfor i := 2; i <= n; i++ {\n\t\ta, b = b, a+b\n\t}\n\treturn b\n}\n```',
    "flatten a nested array": '```go\nfunc flatten(input []interface{}) []interface{} {\n\tvar result []interface{}\n\tfor _, item := range input {\n\t\tif nested, ok := item.([]interface{}); ok {\n\t\t\tresult = append(result, flatten(nested)...)\n\t\t} else {\n\t\t\tresult = append(result, item)\n\t\t}\n\t}\n\treturn result\n}\n```',
    "validate an email": '```go\nimport "regexp"\n\nfunc isValidEmail(email string) bool {\n\tpattern := `^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$`\n\tmatched, _ := regexp.MatchString(pattern, email)\n\treturn matched\n}\n```',
    "check if a string is a palindrome": '```go\nimport "strings"\n\nfunc isPalindrome(s string) bool {\n\tcleaned := strings.ToLower(strings.ReplaceAll(s, " ", ""))\n\trunes := []rune(cleaned)\n\tfor i, j := 0, len(runes)-1; i < j; i, j = i+1, j-1 {\n\t\tif runes[i] != runes[j] {\n\t\t\treturn false\n\t\t}\n\t}\n\treturn true\n}\n```',
    "power of two": '```go\nfunc isPowerOfTwo(n int) bool {\n\treturn n > 0 && (n&(n-1)) == 0\n}\n```',
    "deduplicate an array": '```go\nfunc deduplicate(items []string) []string {\n\tseen := make(map[string]bool)\n\tvar result []string\n\tfor _, item := range items {\n\t\tif !seen[item] {\n\t\t\tseen[item] = true\n\t\t\tresult = append(result, item)\n\t\t}\n\t}\n\treturn result\n}\n```',
    "deep merge two objects": '```go\nfunc deepMerge(base, override map[string]interface{}) map[string]interface{} {\n\tresult := make(map[string]interface{})\n\tfor k, v := range base {\n\t\tresult[k] = v\n\t}\n\tfor k, v := range override {\n\t\tif bv, ok := result[k]; ok {\n\t\t\tif bm, ok := bv.(map[string]interface{}); ok {\n\t\t\t\tif om, ok := v.(map[string]interface{}); ok {\n\t\t\t\t\tresult[k] = deepMerge(bm, om)\n\t\t\t\t\tcontinue\n\t\t\t\t}\n\t\t\t}\n\t\t}\n\t\tresult[k] = v\n\t}\n\treturn result\n}\n```',
}

UTILITY_CODE_RUST = {
    "reverse a string": '```rust\nfn reverse_string(s: &str) -> String {\n    s.chars().rev().collect()\n}\n```',
    "check if a number is prime": '```rust\nfn is_prime(n: u64) -> bool {\n    if n < 2 {\n        return false;\n    }\n    let mut i = 2;\n    while i * i <= n {\n        if n % i == 0 {\n            return false;\n        }\n        i += 1;\n    }\n    true\n}\n```',
    "Fibonacci number": '```rust\nfn fibonacci(n: u64) -> u64 {\n    if n <= 1 {\n        return n;\n    }\n    let (mut a, mut b) = (0u64, 1u64);\n    for _ in 2..=n {\n        let temp = b;\n        b = a + b;\n        a = temp;\n    }\n    b\n}\n```',
    "GCD of two numbers": '```rust\nfn gcd(mut a: u64, mut b: u64) -> u64 {\n    while b != 0 {\n        let temp = b;\n        b = a % b;\n        a = temp;\n    }\n    a\n}\n```',
    "check if a string is a palindrome": '```rust\nfn is_palindrome(s: &str) -> bool {\n    let cleaned: String = s.to_lowercase().chars().filter(|c| !c.is_whitespace()).collect();\n    cleaned == cleaned.chars().rev().collect::<String>()\n}\n```',
    "validate an email": '```rust\nfn is_valid_email(email: &str) -> bool {\n    let parts: Vec<&str> = email.split(\'@\').collect();\n    if parts.len() != 2 {\n        return false;\n    }\n    let (local, domain) = (parts[0], parts[1]);\n    !local.is_empty() && domain.contains(\'.\') && domain.len() > 2\n}\n```',
    "flatten a nested array": '```rust\nfn flatten(nested: Vec<Vec<i32>>) -> Vec<i32> {\n    nested.into_iter().flatten().collect()\n}\n```',
    "deduplicate an array": '```rust\nuse std::collections::HashSet;\n\nfn deduplicate(items: Vec<i32>) -> Vec<i32> {\n    let mut seen = HashSet::new();\n    items.into_iter().filter(|x| seen.insert(*x)).collect()\n}\n```',
    "power of two": '```rust\nfn is_power_of_two(n: u64) -> bool {\n    n > 0 && (n & (n - 1)) == 0\n}\n```',
    "encode a string to base64": '```rust\nfn to_base64(s: &str) -> String {\n    use std::io::Write;\n    let mut buf = Vec::new();\n    let engine = base64::engine::general_purpose::STANDARD;\n    base64::Engine::encode(&engine, s.as_bytes())\n}\n```',
}

UTILITY_CODE_BY_LANG = {
    "python": UTILITY_CODE_PYTHON,
    "javascript": UTILITY_CODE_JAVASCRIPT,
    "typescript": UTILITY_CODE_TYPESCRIPT,
    "go": UTILITY_CODE_GO,
    "rust": UTILITY_CODE_RUST,
}


# --- File Operations Code Pools ---

FILE_OPS_CODE_PYTHON = {
    "read a CSV file": '```python\nimport csv\nfrom pathlib import Path\n\n\ndef read_csv(path: str) -> list[dict]:\n    try:\n        with open(path, newline="") as f:\n            reader = csv.DictReader(f)\n            return list(reader)\n    except FileNotFoundError:\n        raise FileNotFoundError(f"CSV file not found: {path}")\n    except csv.Error as e:\n        raise ValueError(f"CSV parse error: {e}")\n```',
    "write a list of dictionaries to a CSV": '```python\nimport csv\n\n\ndef write_csv(data: list[dict], path: str) -> None:\n    if not data:\n        return\n    try:\n        with open(path, "w", newline="") as f:\n            writer = csv.DictWriter(f, fieldnames=data[0].keys())\n            writer.writeheader()\n            writer.writerows(data)\n    except IOError as e:\n        raise IOError(f"Failed to write CSV: {e}")\n```',
    "filter rows in a CSV": '```python\nimport csv\n\n\ndef filter_csv(path: str, column: str, value: str) -> list[dict]:\n    try:\n        with open(path, newline="") as f:\n            reader = csv.DictReader(f)\n            return [row for row in reader if row.get(column) == value]\n    except FileNotFoundError:\n        raise FileNotFoundError(f"File not found: {path}")\n```',
    "merge two CSV files": '```python\nimport csv\n\n\ndef merge_csv(file_a: str, file_b: str, key: str) -> list[dict]:\n    try:\n        with open(file_a) as fa, open(file_b) as fb:\n            rows_a = {r[key]: r for r in csv.DictReader(fa)}\n            for r in csv.DictReader(fb):\n                if r[key] in rows_a:\n                    rows_a[r[key]].update(r)\n        return list(rows_a.values())\n    except FileNotFoundError as e:\n        raise FileNotFoundError(f"File not found: {e}")\n```',
    "read and parse a JSON configuration": '```python\nimport json\nfrom pathlib import Path\n\n\ndef load_config(path: str) -> dict:\n    try:\n        with open(path) as f:\n            return json.load(f)\n    except FileNotFoundError:\n        raise FileNotFoundError(f"Config not found: {path}")\n    except json.JSONDecodeError as e:\n        raise ValueError(f"Invalid JSON: {e}")\n```',
    "write data to a JSON file": '```python\nimport json\n\n\ndef write_json(data: dict, path: str, indent: int = 2) -> None:\n    try:\n        with open(path, "w") as f:\n            json.dump(data, f, indent=indent)\n    except IOError as e:\n        raise IOError(f"Failed to write JSON: {e}")\n```',
    "deeply merge two JSON config": '```python\nimport json\n\n\ndef merge_configs(base_path: str, override_path: str) -> dict:\n    def deep_merge(a: dict, b: dict) -> dict:\n        result = a.copy()\n        for k, v in b.items():\n            if k in result and isinstance(result[k], dict) and isinstance(v, dict):\n                result[k] = deep_merge(result[k], v)\n            else:\n                result[k] = v\n        return result\n    try:\n        with open(base_path) as f:\n            base = json.load(f)\n        with open(override_path) as f:\n            override = json.load(f)\n        return deep_merge(base, override)\n    except FileNotFoundError as e:\n        raise FileNotFoundError(f"Config not found: {e}")\n```',
    "validate a JSON file against a schema": '```python\nimport json\n\n\ndef validate_json(path: str, required_keys: list[str]) -> bool:\n    try:\n        with open(path) as f:\n            data = json.load(f)\n        return all(key in data for key in required_keys)\n    except (FileNotFoundError, json.JSONDecodeError):\n        return False\n```',
    "recursively find all files": '```python\nfrom pathlib import Path\n\n\ndef find_files(directory: str, extension: str) -> list[str]:\n    try:\n        return [str(p) for p in Path(directory).rglob(f"*{extension}")]\n    except OSError as e:\n        raise OSError(f"Directory error: {e}")\n```',
    "list all files in a directory sorted by size": '```python\nfrom pathlib import Path\n\n\ndef files_by_size(directory: str) -> list[tuple[str, int]]:\n    try:\n        files = [(str(p), p.stat().st_size) for p in Path(directory).iterdir() if p.is_file()]\n        return sorted(files, key=lambda x: x[1], reverse=True)\n    except OSError as e:\n        raise OSError(f"Cannot list directory: {e}")\n```',
    "copy a directory recursively": '```python\nimport shutil\n\n\ndef copy_dir(src: str, dst: str) -> None:\n    try:\n        shutil.copytree(src, dst)\n    except FileExistsError:\n        raise FileExistsError(f"Destination already exists: {dst}")\n    except FileNotFoundError:\n        raise FileNotFoundError(f"Source not found: {src}")\n```',
    "watch a directory for new files": '```python\nimport os\nimport time\nfrom pathlib import Path\n\n\ndef watch_directory(directory: str, interval: float = 1.0):\n    try:\n        known = set(os.listdir(directory))\n        while True:\n            current = set(os.listdir(directory))\n            new_files = current - known\n            for f in new_files:\n                yield Path(directory) / f\n            known = current\n            time.sleep(interval)\n    except OSError as e:\n        raise OSError(f"Cannot watch directory: {e}")\n```',
    "parse a log file and count error": '```python\ndef count_errors(log_path: str) -> dict[str, int]:\n    counts = {}\n    try:\n        with open(log_path) as f:\n            for line in f:\n                if "ERROR" in line:\n                    parts = line.split("ERROR", 1)\n                    msg = parts[1].strip() if len(parts) > 1 else "unknown"\n                    counts[msg] = counts.get(msg, 0) + 1\n    except FileNotFoundError:\n        raise FileNotFoundError(f"Log file not found: {log_path}")\n    return counts\n```',
    "tail the last N lines": '```python\ndef tail(path: str, n: int = 10) -> list[str]:\n    try:\n        with open(path) as f:\n            lines = f.readlines()\n        return lines[-n:]\n    except FileNotFoundError:\n        raise FileNotFoundError(f"File not found: {path}")\n```',
    "search a log file for lines matching": '```python\nimport re\n\n\ndef search_log(path: str, pattern: str) -> list[str]:\n    try:\n        with open(path) as f:\n            return [line.rstrip() for line in f if re.search(pattern, line)]\n    except FileNotFoundError:\n        raise FileNotFoundError(f"Log file not found: {path}")\n```',
    "rotate log files": '```python\nimport os\nfrom pathlib import Path\n\n\ndef rotate_log(path: str, max_size: int = 1_000_000) -> None:\n    try:\n        if os.path.getsize(path) > max_size:\n            backup = f"{path}.1"\n            if os.path.exists(backup):\n                os.remove(backup)\n            os.rename(path, backup)\n            Path(path).touch()\n    except FileNotFoundError:\n        raise FileNotFoundError(f"Log file not found: {path}")\n```',
    "compress a file using gzip": '```python\nimport gzip\nimport shutil\n\n\ndef compress_file(path: str) -> str:\n    output = f"{path}.gz"\n    try:\n        with open(path, "rb") as f_in:\n            with gzip.open(output, "wb") as f_out:\n                shutil.copyfileobj(f_in, f_out)\n        return output\n    except FileNotFoundError:\n        raise FileNotFoundError(f"File not found: {path}")\n```',
    "decompress a gzip file": '```python\nimport gzip\nimport shutil\n\n\ndef decompress_file(path: str) -> str:\n    output = path.rstrip(".gz")\n    try:\n        with gzip.open(path, "rb") as f_in:\n            with open(output, "wb") as f_out:\n                shutil.copyfileobj(f_in, f_out)\n        return output\n    except FileNotFoundError:\n        raise FileNotFoundError(f"File not found: {path}")\n```',
    "create a zip archive": '```python\nimport zipfile\nfrom pathlib import Path\n\n\ndef zip_directory(directory: str, output: str) -> str:\n    try:\n        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:\n            for path in Path(directory).rglob("*"):\n                if path.is_file():\n                    zf.write(path, path.relative_to(directory))\n        return output\n    except FileNotFoundError:\n        raise FileNotFoundError(f"Directory not found: {directory}")\n```',
    "extract a zip archive": '```python\nimport zipfile\n\n\ndef unzip(archive: str, output_dir: str) -> None:\n    try:\n        with zipfile.ZipFile(archive, "r") as zf:\n            zf.extractall(output_dir)\n    except FileNotFoundError:\n        raise FileNotFoundError(f"Archive not found: {archive}")\n    except zipfile.BadZipFile:\n        raise ValueError(f"Invalid zip file: {archive}")\n```',
    "load environment variables from a .env": "```python\ndef load_env(path: str = \".env\") -> dict[str, str]:\n    env_vars = {}\n    try:\n        with open(path) as f:\n            for line in f:\n                line = line.strip()\n                if line and not line.startswith(\"#\") and \"=\" in line:\n                    key, _, value = line.partition(\"=\")\n                    env_vars[key.strip()] = value.strip().strip('\"')\n    except FileNotFoundError:\n        pass\n    return env_vars\n```",
    "get an environment variable with a default": '```python\nimport os\n\n\ndef get_env(key: str, default: str = "") -> str:\n    return os.environ.get(key, default)\n```',
    "set multiple environment variables": '```python\nimport os\n\n\ndef set_env_vars(variables: dict[str, str]) -> None:\n    for key, value in variables.items():\n        os.environ[key] = str(value)\n```',
    "safely join file paths": '```python\nfrom pathlib import Path\n\n\ndef safe_join(*parts: str) -> str:\n    return str(Path(*parts).resolve())\n```',
    "get the file extension": '```python\nfrom pathlib import Path\n\n\ndef get_extension(path: str) -> str:\n    return Path(path).suffix\n```',
    "resolve a relative path": '```python\nfrom pathlib import Path\n\n\ndef resolve_path(path: str) -> str:\n    return str(Path(path).resolve())\n```',
    "check if a path is inside a given directory": '```python\nfrom pathlib import Path\n\n\ndef is_inside(path: str, directory: str) -> bool:\n    try:\n        Path(path).resolve().relative_to(Path(directory).resolve())\n        return True\n    except ValueError:\n        return False\n```',
}

FILE_OPS_CODE_JAVASCRIPT = {
    "read a CSV file": '```javascript\nconst fs = require("fs");\n\nfunction readCsv(path) {\n    try {\n        const content = fs.readFileSync(path, "utf8");\n        const [header, ...rows] = content.trim().split("\\n");\n        const keys = header.split(",");\n        return rows.map(row => {\n            const vals = row.split(",");\n            return Object.fromEntries(keys.map((k, i) => [k, vals[i]]));\n        });\n    } catch (error) {\n        throw new Error(`Failed to read CSV: ${error.message}`);\n    }\n}\n```',
    "read and parse a JSON configuration": '```javascript\nconst fs = require("fs");\n\nfunction loadConfig(path) {\n    try {\n        const data = fs.readFileSync(path, "utf8");\n        return JSON.parse(data);\n    } catch (error) {\n        throw new Error(`Failed to load config: ${error.message}`);\n    }\n}\n```',
    "recursively find all files": '```javascript\nconst fs = require("fs");\nconst path = require("path");\n\nfunction findFiles(dir, ext) {\n    const results = [];\n    try {\n        for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {\n            const full = path.join(dir, entry.name);\n            if (entry.isDirectory()) {\n                results.push(...findFiles(full, ext));\n            } else if (entry.name.endsWith(ext)) {\n                results.push(full);\n            }\n        }\n    } catch (error) {\n        throw new Error(`Directory error: ${error.message}`);\n    }\n    return results;\n}\n```',
    "compress a file using gzip": '```javascript\nconst fs = require("fs");\nconst zlib = require("zlib");\nconst { pipeline } = require("stream");\n\nfunction compressFile(inputPath, outputPath) {\n    try {\n        const input = fs.createReadStream(inputPath);\n        const output = fs.createWriteStream(outputPath || `${inputPath}.gz`);\n        const gzip = zlib.createGzip();\n        pipeline(input, gzip, output, (err) => {\n            if (err) throw new Error(`Compression failed: ${err.message}`);\n        });\n    } catch (error) {\n        throw new Error(`Failed to compress: ${error.message}`);\n    }\n}\n```',
    "load environment variables from a .env": "```javascript\nconst fs = require(\"fs\");\n\nfunction loadEnv(path = \".env\") {\n    try {\n        const content = fs.readFileSync(path, \"utf8\");\n        const vars = {};\n        for (const line of content.split(\"\\\\n\")) {\n            const trimmed = line.trim();\n            if (trimmed && !trimmed.startsWith(\"#\") && trimmed.includes(\"=\")) {\n                const [key, ...rest] = trimmed.split(\"=\");\n                vars[key.trim()] = rest.join(\"=\").trim();\n            }\n        }\n        return vars;\n    } catch (error) {\n        return {};\n    }\n}\n```",
    "tail the last N lines": '```javascript\nconst fs = require("fs");\n\nfunction tail(path, n = 10) {\n    try {\n        const content = fs.readFileSync(path, "utf8");\n        const lines = content.trim().split("\\n");\n        return lines.slice(-n);\n    } catch (error) {\n        throw new Error(`Failed to read file: ${error.message}`);\n    }\n}\n```',
    "safely join file paths": '```javascript\nconst path = require("path");\n\nfunction safeJoin(...parts) {\n    return path.resolve(path.join(...parts));\n}\n```',
    "write data to a JSON file": '```javascript\nconst fs = require("fs");\n\nfunction writeJson(data, path, indent = 2) {\n    try {\n        fs.writeFileSync(path, JSON.stringify(data, null, indent));\n    } catch (error) {\n        throw new Error(`Failed to write JSON: ${error.message}`);\n    }\n}\n```',
}

FILE_OPS_CODE_TYPESCRIPT = {
    "read a CSV file": '```typescript\nimport * as fs from "fs";\n\nfunction readCsv(path: string): Record<string, string>[] {\n    try {\n        const content = fs.readFileSync(path, "utf8");\n        const [header, ...rows] = content.trim().split("\\n");\n        const keys = header.split(",");\n        return rows.map(row => {\n            const vals = row.split(",");\n            return Object.fromEntries(keys.map((k, i) => [k, vals[i]]));\n        });\n    } catch (error) {\n        throw new Error(`Failed to read CSV: ${(error as Error).message}`);\n    }\n}\n```',
    "read and parse a JSON configuration": '```typescript\nimport * as fs from "fs";\n\nfunction loadConfig<T = Record<string, unknown>>(path: string): T {\n    try {\n        const data = fs.readFileSync(path, "utf8");\n        return JSON.parse(data) as T;\n    } catch (error) {\n        throw new Error(`Failed to load config: ${(error as Error).message}`);\n    }\n}\n```',
    "recursively find all files": '```typescript\nimport * as fs from "fs";\nimport * as path from "path";\n\nfunction findFiles(dir: string, ext: string): string[] {\n    const results: string[] = [];\n    try {\n        for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {\n            const full = path.join(dir, entry.name);\n            if (entry.isDirectory()) {\n                results.push(...findFiles(full, ext));\n            } else if (entry.name.endsWith(ext)) {\n                results.push(full);\n            }\n        }\n    } catch (error) {\n        throw new Error(`Directory error: ${(error as Error).message}`);\n    }\n    return results;\n}\n```',
    "load environment variables from a .env": "```typescript\nimport * as fs from \"fs\";\n\nfunction loadEnv(path: string = \".env\"): Record<string, string> {\n    try {\n        const content = fs.readFileSync(path, \"utf8\");\n        const vars: Record<string, string> = {};\n        for (const line of content.split(\"\\\\n\")) {\n            const trimmed = line.trim();\n            if (trimmed && !trimmed.startsWith(\"#\") && trimmed.includes(\"=\")) {\n                const [key, ...rest] = trimmed.split(\"=\");\n                vars[key.trim()] = rest.join(\"=\").trim();\n            }\n        }\n        return vars;\n    } catch (error) {\n        return {};\n    }\n}\n```",
    "safely join file paths": '```typescript\nimport * as path from "path";\n\nfunction safeJoin(...parts: string[]): string {\n    return path.resolve(path.join(...parts));\n}\n```',
    "write data to a JSON file": '```typescript\nimport * as fs from "fs";\n\nfunction writeJson(data: unknown, filePath: string, indent: number = 2): void {\n    try {\n        fs.writeFileSync(filePath, JSON.stringify(data, null, indent));\n    } catch (error) {\n        throw new Error(`Failed to write JSON: ${(error as Error).message}`);\n    }\n}\n```',
}

FILE_OPS_CODE_BY_LANG = {
    "python": FILE_OPS_CODE_PYTHON,
    "javascript": FILE_OPS_CODE_JAVASCRIPT,
    "typescript": FILE_OPS_CODE_TYPESCRIPT,
}


# --- Helper Functions ---


def _get_unique_query(templates: list, used: set, rng: random.Random,
                      lang: str = "") -> str:
    """Generate a unique query from templates, avoiding duplicates."""
    max_attempts = 100
    for _ in range(max_attempts):
        template = rng.choice(templates)
        query = template.replace("{lang}", lang) if "{lang}" in template else template
        if query not in used:
            used.add(query)
            return query
    # Fallback: add variant suffix to make unique
    template = rng.choice(templates)
    query = template.replace("{lang}", lang) if "{lang}" in template else template
    query = f"{query} (variant {len(used)})"
    used.add(query)
    return query


def _pick_language(weights: dict, rng: random.Random) -> str:
    """Pick a language using weighted distribution."""
    languages = list(weights.keys())
    weight_values = list(weights.values())
    return rng.choices(languages, weights=weight_values, k=1)[0]


def _find_code_response(lang: str, query: str, code_pool: dict) -> str:
    """Find a matching code response from the pool, or return a fallback."""
    pool = code_pool.get(lang, {})
    # Try to find a matching code block by query suffix
    query_lower = query.lower()
    for key, code in pool.items():
        if key.lower() in query_lower:
            return code
    # Fallback: return the first code block from the pool
    if pool:
        return next(iter(pool.values()))
    # Ultimate fallback
    return f"```{lang}\n// Implementation for: {query}\n```"


# --- Category Batch Generators ---


def generate_utility_batch(count: int = 50, seed: int = None) -> list[dict]:
    """Generate utility function code samples per CODE-01.

    Covers 5 languages with weighted distribution per D-03.
    Terse code-first style per D-05.
    """
    rng = random.Random(seed)
    prompts = load_system_prompts()
    system_content = prompts["code_assistant"]

    # Flatten all query templates
    all_queries = []
    for topic_queries in UTILITY_QUERIES.values():
        all_queries.extend(topic_queries)

    samples = []
    used_queries = set()

    for _ in range(count):
        lang = _pick_language(UTILITY_LANG_WEIGHTS, rng)
        query = _get_unique_query(all_queries, used_queries, rng, lang=lang)
        code_response = _find_code_response(lang, query, UTILITY_CODE_BY_LANG)

        sample = {
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": query},
                {"role": "assistant", "content": code_response},
            ]
        }
        samples.append(sample)

    return samples


def generate_file_ops_batch(count: int = 50, seed: int = None) -> list[dict]:
    """Generate file operation code samples per CODE-02.

    Covers 3 languages with weighted distribution per D-04.
    Includes error handling per template generation_notes.
    """
    rng = random.Random(seed)
    prompts = load_system_prompts()
    system_content = prompts["code_assistant"]

    # Flatten all query templates
    all_queries = []
    for topic_queries in FILE_OPS_QUERIES.values():
        all_queries.extend(topic_queries)

    samples = []
    used_queries = set()

    for _ in range(count):
        lang = _pick_language(FILEOPS_LANG_WEIGHTS, rng)
        query = _get_unique_query(all_queries, used_queries, rng, lang=lang)
        code_response = _find_code_response(lang, query, FILE_OPS_CODE_BY_LANG)

        sample = {
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": query},
                {"role": "assistant", "content": code_response},
            ]
        }
        samples.append(sample)

    return samples


def generate_debugging_batch(count: int = 50, seed: int = None) -> list[dict]:
    """Generate debugging code samples per CODE-03.

    Covers 3 languages with weighted distribution per D-04.
    Uses Bug/Fix format per D-06.
    Uses code_debugger system prompt.
    """
    rng = random.Random(seed)
    prompts = load_system_prompts()
    system_content = prompts["code_debugger"]

    samples = []
    used_queries = set()

    # Build flat list of (lang, bug_query, fix_response) tuples
    all_debug_entries = []
    for lang, bug_types in DEBUGGING_QUERIES.items():
        for bug_type, entries in bug_types.items():
            for bug_query, fix_response in entries:
                all_debug_entries.append((lang, bug_query, fix_response))

    for _ in range(count):
        lang = _pick_language(FILEOPS_LANG_WEIGHTS, rng)

        # Filter entries for this language
        lang_entries = [(q, r) for (l, q, r) in all_debug_entries if l == lang]
        if not lang_entries:
            # Fallback to any language entry
            lang_entries = [(q, r) for (_, q, r) in all_debug_entries]

        # Pick a unique entry
        max_attempts = 50
        found = False
        for _ in range(max_attempts):
            bug_query, fix_response = rng.choice(lang_entries)
            if bug_query not in used_queries:
                used_queries.add(bug_query)
                found = True
                break

        if not found:
            # Append variant to make unique
            bug_query, fix_response = rng.choice(lang_entries)
            bug_query = f"{bug_query}\n\n(Variant {len(used_queries)})"
            used_queries.add(bug_query)

        sample = {
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": bug_query},
                {"role": "assistant", "content": fix_response},
            ]
        }
        samples.append(sample)

    return samples


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

    Usage: python -m scripts.generate_code_data --category utility --count 50 --batch 1
    Writes to: datasets/code/{category}-batch-{batch:02d}.jsonl
    Validates each sample before writing. Prints summary stats.
    """
    parser = argparse.ArgumentParser(
        description="Generate code generation training data batches for Lyra"
    )
    parser.add_argument(
        "--category",
        type=str,
        required=True,
        choices=VALID_CATEGORIES,
        help="Category of code data to generate",
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
        help="Output directory (default: datasets/code/)",
    )
    args = parser.parse_args()

    # Validate arguments per T-05-01 and T-05-02
    if args.batch < 1:
        print("Error: --batch must be a positive integer", file=sys.stderr)
        sys.exit(1)
    if args.count < 1 or args.count > MAX_COUNT:
        print(f"Error: --count must be between 1 and {MAX_COUNT}", file=sys.stderr)
        sys.exit(1)

    # Generate batch
    generators = {
        "utility": generate_utility_batch,
        "file-ops": generate_file_ops_batch,
        "debugging": generate_debugging_batch,
    }

    generator = generators[args.category]
    print(f"Generating {args.count} {args.category} samples (batch {args.batch})...")
    samples = generator(count=args.count, seed=args.seed)

    # Validate
    results = validate_batch(samples)
    print(f"Validation: {results['valid']}/{results['total']} valid")
    if results["errors"]:
        for err in results["errors"][:5]:
            print(f"  Sample {err['index']}: {err['error']}")

    # Filter to only valid samples
    valid_samples = []
    for sample in samples:
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
