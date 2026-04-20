"""Tests for the tool schema pool YAML file.

Validates structure, constraints, and content of datasets/tool-calling/tool_schemas.yaml.
Per plan D-03/D-04: 50-100 unique schemas, developer/everyday mix, max 3 params each.
"""
from pathlib import Path

import pytest
import yaml


SCHEMA_PATH = Path("datasets/tool-calling/tool_schemas.yaml")


@pytest.fixture
def schema_data():
    """Load the tool schemas YAML file."""
    assert SCHEMA_PATH.exists(), f"Schema file not found: {SCHEMA_PATH}"
    return yaml.safe_load(SCHEMA_PATH.read_text())


@pytest.fixture
def all_schemas(schema_data):
    """Extract all schema items from the nested structure."""
    schemas = schema_data["schemas"]
    items = []
    for domain_key, domain_val in schemas.items():
        if isinstance(domain_val, list):
            items.extend(domain_val)
        elif isinstance(domain_val, dict):
            for subcat_key, subcat_val in domain_val.items():
                items.extend(subcat_val)
    return items


class TestSchemaFileLoads:
    """Test 1: YAML file loads without error via yaml.safe_load."""

    def test_file_exists(self):
        assert SCHEMA_PATH.exists(), f"Schema file not found: {SCHEMA_PATH}"

    def test_yaml_loads(self):
        data = yaml.safe_load(SCHEMA_PATH.read_text())
        assert data is not None


class TestSchemaStructure:
    """Test 2: File contains top-level schemas key with developer and everyday sub-keys."""

    def test_has_schemas_key(self, schema_data):
        assert "schemas" in schema_data

    def test_has_developer_key(self, schema_data):
        assert "developer" in schema_data["schemas"]

    def test_has_everyday_key(self, schema_data):
        assert "everyday" in schema_data["schemas"]


class TestSchemaCount:
    """Test 3: Total unique schema count is between 50 and 100 (inclusive)."""

    def test_count_in_range(self, all_schemas):
        count = len(all_schemas)
        assert 50 <= count <= 100, f"Schema count {count} not in 50-100 range"


class TestDeveloperWeight:
    """Test 4: Developer schemas outnumber everyday schemas (60/40 split per D-04)."""

    def test_developer_outnumbers_everyday(self, schema_data):
        schemas = schema_data["schemas"]
        developer_count = 0
        everyday_count = 0

        # Count developer schemas (nested dict with subcategories)
        dev = schemas.get("developer", {})
        if isinstance(dev, dict):
            for subcat in dev.values():
                developer_count += len(subcat)
        elif isinstance(dev, list):
            developer_count = len(dev)

        # Count everyday schemas
        every = schemas.get("everyday", {})
        if isinstance(every, dict):
            for subcat in every.values():
                everyday_count += len(subcat)
        elif isinstance(every, list):
            everyday_count = len(every)

        assert developer_count > everyday_count, (
            f"Developer ({developer_count}) should outnumber everyday ({everyday_count})"
        )


class TestSchemaFields:
    """Test 5: Every schema has name, description, parameters with type/properties/required."""

    def test_every_schema_has_required_fields(self, all_schemas):
        for schema in all_schemas:
            assert "name" in schema, f"Schema missing 'name': {schema}"
            assert isinstance(schema["name"], str), f"name must be string: {schema['name']}"
            assert "description" in schema, f"Schema missing 'description': {schema.get('name')}"
            assert isinstance(schema["description"], str)
            assert "parameters" in schema, f"Schema missing 'parameters': {schema['name']}"
            params = schema["parameters"]
            assert params.get("type") == "object", f"parameters.type must be 'object': {schema['name']}"
            assert "properties" in params, f"parameters missing 'properties': {schema['name']}"
            assert isinstance(params["properties"], dict)
            assert "required" in params, f"parameters missing 'required': {schema['name']}"
            assert isinstance(params["required"], list)


class TestMaxParameters:
    """Test 6: Every schema has at most 3 parameters (token budget constraint)."""

    def test_max_three_params(self, all_schemas):
        for schema in all_schemas:
            prop_count = len(schema["parameters"].get("properties", {}))
            assert prop_count <= 3, (
                f"Schema '{schema['name']}' has {prop_count} params (max 3)"
            )


class TestNoDuplicateNames:
    """Test 7: No duplicate schema names across all domains."""

    def test_unique_names(self, all_schemas):
        names = [s["name"] for s in all_schemas]
        duplicates = [n for n in names if names.count(n) > 1]
        assert len(duplicates) == 0, f"Duplicate schema names: {set(duplicates)}"


class TestMcpMetaTools:
    """Test 8: MCP meta-tools are present under mcp_meta key."""

    def test_mcp_meta_key_exists(self, schema_data):
        assert "mcp_meta" in schema_data["schemas"]

    def test_mcp_list_servers_exists(self, schema_data):
        mcp_schemas = schema_data["schemas"]["mcp_meta"]
        names = [s["name"] for s in mcp_schemas]
        assert "mcp_list_servers" in names

    def test_mcp_list_tools_exists(self, schema_data):
        mcp_schemas = schema_data["schemas"]["mcp_meta"]
        names = [s["name"] for s in mcp_schemas]
        assert "mcp_list_tools" in names

    def test_mcp_invoke_tool_exists(self, schema_data):
        mcp_schemas = schema_data["schemas"]["mcp_meta"]
        names = [s["name"] for s in mcp_schemas]
        assert "mcp_invoke_tool" in names


class TestCliTools:
    """Test 9: CLI tools (run_command) are present."""

    def test_cli_key_exists(self, schema_data):
        assert "cli" in schema_data["schemas"]

    def test_run_command_exists(self, schema_data):
        cli_schemas = schema_data["schemas"]["cli"]
        names = [s["name"] for s in cli_schemas]
        assert "run_command" in names
